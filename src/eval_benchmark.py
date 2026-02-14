import json
import torch
import re
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import time


class TISER_Evaluator:
    def __init__(self, base_model: str, adapter: str = None, prompt_mode: str = "standard"):
        assert prompt_mode in ("tiser", "standard")
        self.prompt_mode = prompt_mode

        self.base_model = base_model
        self.adapter = adapter
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.adapter}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                if self.adapter != None:
                    print("Loading LoRA adapter...")
                    self.model = PeftModel.from_pretrained(base_model, self.adapter)
            except Exception as e:
                print(f"Error occurs when loading model: {e}")
                raise e
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

        self.model.eval()
        print("hf_device_map:", getattr(self.model, "hf_device_map", None))
        print("Model loads successfully")


    @staticmethod
    def _extract_temporal_context_from_prompt(prompt: str) -> str:
        if not prompt:
            return ""
        m = re.search(r"Temporal context:\s*(.*)", prompt, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        tail = m.group(1).strip()

        stop = re.search(r"\n\s*###\s*(Answer|Question)\s*:", tail, re.IGNORECASE)
        if stop:
            tail = tail[: stop.start()].strip()

        return tail.strip()

    def build_prompt(self, sample: dict) -> str:
        if self.prompt_mode == "tiser":
            # return sample.get("prompt", "").strip()
            return sample.get("prompt", "").split("### Question:")[1].split("### Answer:")[0].strip()

        # standard
        question = (sample.get("question", "") or "").strip()
        context = (sample.get("context", "") or "").strip()
        if not context:
            context = self._extract_temporal_context_from_prompt(sample.get("prompt", ""))

        prompt = (
            "You are an AI assistant that has to respond to questions given a context.\n"
            "!!!Output ONLY the final answer with NO EXPLANATION OR ANY EXTRA TOKENS.!!!\n\n"
            f"Question: {question}\n"
            f"Temporal Context: {context}\n"
            "Answer:\n"
        )
        return prompt

    @staticmethod
    def normalize_answer(s: str) -> str:
        s = (s or "").lower().strip()

        time_units = [
            "years", "year", 
            "months", "month", 
            "weeks", "week", 
            "days", "day", 
            "hours", "hour", 
            "minutes", "minute", 
            "seconds", "second"
        ]
        
        pattern = r"\b(" + "|".join(time_units) + r")\b"
        s = re.sub(pattern, " ", s)

        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*,\s*", ", ", s)
        # Remove extra punctuations
        s = re.sub(r"[^\w\s,]", "", s)
        # Merge space again
        s = re.sub(r"\s+", " ", s).strip()
        
        return s
    

    def extract_answer(self, generated_text: str) -> str:
        if not generated_text:
            return ""

        # 1. Prioritize match <answer>...</answer> tags
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", generated_text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 2. Define all possible "cutting words"
        stop_tokens = [
            "<reasoning>", 
            "<explanation>", 
            "### Explanation", 
            "### Reasoning", 
            "Explanation:", 
            "Reasoning:",
            "\n\n"
        ]

        cut_pos = len(generated_text)
        for token in stop_tokens:
            pos = generated_text.lower().find(token.lower())
            if pos != -1:
                cut_pos = min(cut_pos, pos)

        text = generated_text[:cut_pos].strip()

        # 3. Clear the prefixes of the answer
        prefixes = [
            r"^the answer is\s*",
            r"^answer:\s*",
            r"^answer\s+is\s*",
            r"^prediction:\s*",
        ]
        for p in prefixes:
            text = re.sub(p, "", text, flags=re.IGNORECASE).strip()

        # 4. Only the first line is taken
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return ""
            
        final_ans = lines[0]

        # 5. Remove the period or comma at the end
        final_ans = final_ans.rstrip(".,;").strip()
            
        return final_ans
        

    def calculate_em_f1(self, predicted: str, ground_truth: str):
        pred_raw = (predicted or "").strip()
        truth_raw = (ground_truth or "").strip()

        em = 1 if self.normalize_answer(pred_raw) == self.normalize_answer(truth_raw) else 0

        pred = self.normalize_answer(pred_raw)
        truth = self.normalize_answer(truth_raw)

        pred_tokens = set(pred.split())
        truth_tokens = set(truth.split())

        if not pred_tokens or not truth_tokens:
            return em, 0.0

        common = pred_tokens.intersection(truth_tokens)
        if len(common) == 0:
            return em, 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return em, f1

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return generated

    def evaluate_dataset(self, dataset_samples, dataset_name: str, max_samples=None, verbose: bool = False):
        if max_samples and len(dataset_samples) > max_samples:
            samples = dataset_samples[:max_samples]
        else:
            samples = dataset_samples

        total_em, total_f1 = 0.0, 0.0
        processed = 0

        pbar = tqdm(samples, desc=f"  {dataset_name.ljust(20)}", unit="smpl")

        for i, sample in enumerate(pbar):
            ground_truth = sample.get("answer", "")
            if not ground_truth:
                continue

            prompt = self.build_prompt(sample)
            if not prompt:
                continue

            try:
                generated = self.generate_answer(prompt)
                predicted = self.extract_answer(generated)

                em, f1 = self.calculate_em_f1(predicted, ground_truth)

                # # =========== Testing start ===========
                # # if f1 > 0.7 and em < 0.6 and processed < 5:
                # if processed < 5:
                #     print(f"   Generated (Raw): {generated!r}")
                #     print(f"   Predicted (Cln): {predicted!r}")
                #     print(f"   Ground Truth:    {ground_truth!r}")
                #     print(f"   Norm Predicted:  {self.normalize_answer(predicted)!r}")
                #     print(f"   Norm Truth:      {self.normalize_answer(ground_truth)!r}")
                # # =========== Testing end ===========
                
                total_em += em
                total_f1 += f1
                processed += 1

                pbar.set_postfix({
                    "EM": f"{total_em/processed:.3f}" if processed > 0 else "0.000",
                    "F1": f"{total_f1/processed:.3f}" if processed > 0 else "0.000"
                })

                if verbose and i < 3:
                    print(f"\n   Sample {i+1}:")
                    print(f"     Mode: {self.prompt_mode}")
                    print(f"     Generation: {generated[:200]}...")
                    print(f"     Prediction: {predicted[:160]}...")
                    print(f"     Ground Truth: {ground_truth}")
                    print(f"     EM: {em}, F1: {f1:.4f}")

            except Exception as e:
                if verbose and i < 3:
                    print(f"\n   {i+1} samples failed: {e}")
                continue

        pbar.close()

        if processed == 0:
            return 0.0, 0.0, 0

        return total_em / processed, total_f1 / processed, processed


def load_test_data(json_path: str):
    print(f"Loading eval data: {json_path}")

    name_mapping = {
        "tgqa_test": "TGQA",
        "tempreason_l2_test": "TempReason (L2)",
        "tempreason_l3_test": "TempReason (L3)",
        "timeqa_easy_test": "TimeQA (easy)",
        "timeqa_hard_test": "TimeQA (hard)",
    }

    organized_data = defaultdict(list)
    total_samples = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Read JSON file"):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                ds_name = sample.get("dataset_name", "").strip().lower()
                standard_name = name_mapping.get(ds_name, ds_name)
                organized_data[standard_name].append(sample)
                total_samples += 1
            except Exception:
                continue

    print(f"Load complete: {total_samples} total samples")

    table1_datasets = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]
    print("\nDatset distribution:")
    for ds in table1_datasets:
        cnt = len(organized_data.get(ds, []))
        print(f"  {ds.ljust(25)}: {cnt:>6} 个样本")

    return organized_data


def print_table1_format(results):
    print("\n" + "=" * 90)
    print("Evaluation Results")
    print("=" * 90)

    header = f"{'Dataset':<25} {'Exact Match (EM)':<20} {'F1 Score':<20} {'Eval samples':<15}"
    print(header)
    print("-" * 85)

    table1_order = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]

    em_scores, f1_scores = [], []
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            em = res["EM"]
            f1 = res["F1"]
            samples = res["samples_processed"]
            total = res["total_samples"]
            print(f"{ds_name:<25} {em:<20.3f} {f1:<20.3f} {samples}/{total:<15}")
            if samples > 0:
                em_scores.append(em)
                f1_scores.append(f1)
        else:
            print(f"{ds_name:<25} {'-':<20} {'-':<20} {'0/0':<15}")

    if em_scores:
        macro_em = float(np.mean(em_scores))
        macro_f1 = float(np.mean(f1_scores))
        print("-" * 85)
        print(f"{'Macro Average':<25} {macro_em:<20.3f} {macro_f1:<20.3f}")
        return macro_em, macro_f1

    return 0.0, 0.0


def main():
    base_model_id = "Qwen/Qwen2.5-7B"
    adapter_path = "model/Qwen/Qwen2.5-7B-LoRA"
    # base_model_id = "Qwen/Qwen2.5-7B"
    # adapter_path = "model/Qwen/Qwen2.5-7B-LoRA"
    TEST_DATA_PATH = "data/TISER_test_random_3000.json"

    # PROMPT_MODE = "standard"
    PROMPT_MODE = "tiser"

    FULL_EVALUATION = True
    MAX_SAMPLES_PER_DATASET = None if FULL_EVALUATION else 100

    print(f"Model: {base_model_id}")
    print(f"Evaluation Data: {TEST_DATA_PATH}")
    print(f"Prompt Mode: {PROMPT_MODE}")
    print()

    start_time = time.time()
    test_data = load_test_data(TEST_DATA_PATH)
    data_load_time = time.time() - start_time

    evaluator = TISER_Evaluator(base_model_id, adapter_path, prompt_mode=PROMPT_MODE)
    # evaluator = TISER_Evaluator(base_model_id, prompt_mode=PROMPT_MODE)   # If use base model

    print("\n" + "=" * 90)
    print("Evaluation start")
    print("=" * 90)

    results = {}
    table1_datasets = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]
    total_evaluation_time = 0.0

    for ds_name in table1_datasets:
        if ds_name not in test_data or len(test_data[ds_name]) == 0:
            print(f"\nSkip: '{ds_name}' No data")
            results[ds_name] = {"EM": 0, "F1": 0, "samples_processed": 0, "total_samples": 0}
            continue

        samples = test_data[ds_name]
        total_samples = len(samples)

        print(f"\nEvaluation dataset: {ds_name}")
        print(f"Total samples: {total_samples}")

        eval_start = time.time()
        em_score, f1_score, processed = evaluator.evaluate_dataset(
            samples, ds_name, MAX_SAMPLES_PER_DATASET, verbose=False
        )
        eval_time = time.time() - eval_start

        results[ds_name] = {
            "EM": em_score,
            "F1": f1_score,
            "samples_processed": processed,
            "total_samples": total_samples,
            "eval_time": eval_time
        }
        print(f"  Complete: EM={em_score:.4f}, F1={f1_score:.4f}, Time={eval_time:.1f}秒")
        total_evaluation_time += eval_time

    print("\n" + "=" * 90)
    macro_em, macro_f1 = print_table1_format(results)

    output_file = f"table1_results_{PROMPT_MODE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_mode": PROMPT_MODE,
            "results": results,
            "macro_average": {"EM": macro_em, "F1": macro_f1},
            "evaluation_settings": {
                "adapter_path": adapter_path,
                "test_data_path": TEST_DATA_PATH,
                "full_evaluation": FULL_EVALUATION,
                "max_samples_per_dataset": MAX_SAMPLES_PER_DATASET
            },
            "timing": {
                "data_loading": data_load_time,
                "total_evaluation": total_evaluation_time,
                "total": time.time() - start_time
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetails saved to: {output_file}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
