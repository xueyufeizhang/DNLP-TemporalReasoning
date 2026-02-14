import json
import torch
import re
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from collections import Counter
import time


class TISER_Evaluator:
    def __init__(self, base_model: str, adapter: str, prompt_mode: str = "standard"):
        assert prompt_mode in ("tiser", "standard")
        self.prompt_mode = prompt_mode
        self.base_model = base_model
        self.adapter = adapter
        self.model = None
        self.tokenizer = None
        self.device = None

        self.stop_text = "</answer>"
        self.stop_ids = None

        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.adapter}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stop_ids = torch.tensor(
            self.tokenizer.encode(self.stop_text, add_special_tokens=False),
            dtype=torch.long
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter)
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
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
            return sample.get("prompt", "").strip()

        question = (sample.get("question", "") or "").strip()
        context = (sample.get("context", "") or "").strip()
        if not context:
            context = self._extract_temporal_context_from_prompt(sample.get("prompt", ""))

        prompt = (
            "You are a helpful assistant for temporal reasoning.\n"
            "Answer the question using ONLY the temporal context.\n"
            "Output ONLY the final answer (entity/event) with no explanation.\n\n"
            f"Question: {question}\n"
            f"Temporal Context: {context}\n"
        )
        return prompt

    # -------------------- normalize / extract / metric --------------------
    @staticmethod
    def normalize_answer(s: str) -> str:
        import string

        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text): return text.lower()

        s = re.sub(r'\b(\d+)\s*(year|years)\b', r'\1', s, flags=re.IGNORECASE)
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def extract_answer(self, generated_text: str) -> str:
        if not generated_text:
            return ""

        ms = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", generated_text, re.DOTALL | re.IGNORECASE))
        if ms:
            return ms[-1].group(1).strip().rstrip(".,;").strip()

        m = re.search(r"\b(true|false)\b", generated_text, re.IGNORECASE)
        if m:
            return "True" if m.group(1).lower() == "true" else "False"

        lines = [l.strip() for l in generated_text.splitlines() if l.strip()]
        return lines[-1].rstrip(".,;").strip() if lines else ""

    def calculate_em_f1(self, predicted: str, ground_truth: str):
        pred_raw = (predicted or "").strip()
        truth_raw = (ground_truth or "").strip()

        em = 1 if self.normalize_answer(pred_raw) == self.normalize_answer(truth_raw) else 0

        pred = pred_raw.lower().strip().split()
        truth = truth_raw.lower().strip().split()

        if not pred and not truth:
            return em, 1.0
        if not pred or not truth:
            return em, 0.0

        pred_set, truth_set = set(pred), set(truth)
        common = pred_set & truth_set
        if not common:
            return em, 0.0

        precision = len(common) / len(pred_set)
        recall = len(common) / len(truth_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return em, f1

    # =========================
    # Greedy batch generation
    def generate_answers_batch(self, prompts, max_new_tokens: int):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
            )

        sequences = outputs.sequences
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        decoded = []
        for i in range(sequences.size(0)):
            gen_ids = sequences[i, prompt_lens[i]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            pos = text.lower().find(self.stop_text.lower())
            if pos != -1:
                text = text[:pos + len(self.stop_text)]
            decoded.append(text)

        return decoded

    # =========================
    def _vote_with_fallback(
        self,
        candidates,              # List[str]  (Include greedy + sampled)
        greedy_answer: str,      # str
        fallback_threshold: float = 0.6,
        max_answer_len: int = 200
    ):
        cleaned = []
        for a in candidates:
            a = (a or "").strip()
            if not a:
                continue
            if len(a) > max_answer_len:
                continue
            cleaned.append(a)

        greedy_answer = (greedy_answer or "").strip()

        if not cleaned:
            return greedy_answer

        normed = [self.normalize_answer(a) for a in cleaned]
        cnt = Counter(normed)
        best_norm, best_count = cnt.most_common(1)[0]
        conf = best_count / len(normed)

        if conf < fallback_threshold:
            return greedy_answer

        best_originals = [cleaned[i] for i in range(len(cleaned)) if normed[i] == best_norm]
        return min(best_originals, key=len) if best_originals else greedy_answer

    # =========================
    # Self-Consistency
    def generate_answers_batch_self_consistency(
        self,
        prompts,
        max_new_tokens: int,
        k: int = 5,
        temperature: float = 0.6,
        top_p: float = 0.9,
        fallback_threshold: float = 0.6
    ):

        # 1) greedy
        greedy_texts = self.generate_answers_batch(prompts, max_new_tokens=max_new_tokens)
        greedy_answers = [self.extract_answer(t) for t in greedy_texts]

        # 2) Sampling k times
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.model.device)

        all_k_answers = []  # [k][bs]

        with torch.inference_mode():
            for _ in range(k):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

                sequences = outputs.sequences
                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

                batch_answers = []
                for i in range(sequences.size(0)):
                    gen_ids = sequences[i, prompt_lens[i]:]
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    pos = text.lower().find(self.stop_text.lower())
                    if pos != -1:
                        text = text[:pos + len(self.stop_text)]

                    ans = self.extract_answer(text)
                    batch_answers.append(ans)

                all_k_answers.append(batch_answers)

        # 3) Aggregation：greedy + sampled
        final_preds = []
        bsz = len(prompts)
        for i in range(bsz):
            sampled_i = [all_k_answers[t][i] for t in range(k)]
            greedy_i = greedy_answers[i]
            candidates = [greedy_i] + sampled_i

            voted = self._vote_with_fallback(
                candidates=candidates,
                greedy_answer=greedy_i,
                fallback_threshold=fallback_threshold,
                max_answer_len=200
            )
            final_preds.append(voted)

        return final_preds

    # =========================
    def evaluate_dataset(
        self,
        dataset_samples,
        dataset_name: str,
        max_samples=None,
        verbose: bool = False,
        batch_size: int = 4,
        use_self_consistency: bool = True,
        sc_k: int = 5,
        sc_temperature: float = 0.6,
        sc_top_p: float = 0.9,
        sc_fallback_threshold: float = 0.6
    ):
        if max_samples and len(dataset_samples) > max_samples:
            samples = dataset_samples[:max_samples]
        else:
            samples = dataset_samples

        total_em, total_f1, processed = 0.0, 0.0, 0

        pbar = tqdm(range(0, len(samples), batch_size), desc=f"  {dataset_name.ljust(20)}", unit="batch")
        for base in pbar:
            batch = samples[base: base + batch_size]

            prompts, golds = [], []
            for s in batch:
                gt = s.get("answer", "")
                if not gt:
                    continue
                pr = self.build_prompt(s)
                if not pr:
                    continue
                prompts.append(pr)
                golds.append(gt)

            if not prompts:
                continue

            try:
                max_new = 1024 if self.prompt_mode == "tiser" else 128

                if use_self_consistency:
                    preds = self.generate_answers_batch_self_consistency(
                        prompts,
                        max_new_tokens=max_new,
                        k=sc_k,
                        temperature=sc_temperature,
                        top_p=sc_top_p,
                        fallback_threshold=sc_fallback_threshold
                    )
                else:
                    gens = self.generate_answers_batch(prompts, max_new_tokens=max_new)
                    preds = [self.extract_answer(g) for g in gens]

                for pred, gt in zip(preds, golds):
                    em, f1 = self.calculate_em_f1(pred, gt)

                    if processed < 5:
                        print(f"   Predicted:      {pred!r}")
                        print(f"   Ground Truth:   {gt!r}")
                        print(f"   Norm Predicted: {self.normalize_answer(pred)!r}")
                        print(f"   Norm Truth:     {self.normalize_answer(gt)!r}")

                    total_em += em
                    total_f1 += f1
                    processed += 1

                pbar.set_postfix({
                    "EM": f"{total_em/processed:.3f}" if processed else "0.000",
                    "F1": f"{total_f1/processed:.3f}" if processed else "0.000",
                    "bsz": len(prompts),
                    "SC": "on" if use_self_consistency else "off",
                    "k": sc_k if use_self_consistency else 1,
                    "thr": sc_fallback_threshold if use_self_consistency else "-"
                })

            except Exception as e:
                if verbose:
                    print(f"\n   batch {base} failed: {e}")
                continue

        if processed == 0:
            return 0.0, 0.0, 0
        return total_em / processed, total_f1 / processed, processed


def load_test_data(json_path: str):
    print(f"Loading eval data: {json_path}")

    name_mapping = {
        "tgqa_test": "TGQA",
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

    print(f"Load complete: {total_samples} 个总样本")

    table1_datasets = ["TGQA", "TimeQA (easy)", "TimeQA (hard)"]
    print("\nDatset distribution:")
    for ds in table1_datasets:
        cnt = len(organized_data.get(ds, []))
        print(f"  {ds.ljust(25)}: {cnt:>6} 个样本")

    return organized_data


def print_table1_format(results):
    print("\n" + "=" * 90)
    print("Evaluation results")
    print("=" * 90)

    header = f"{'Dataset':<25} {'Exact Match (EM)':<20} {'F1 Score':<20} {'Evaluation samples':<15}"
    print(header)
    print("-" * 85)

    table1_order = ["TGQA", "TimeQA (easy)", "TimeQA (hard)"]

    em_scores, f1_scores = [], []
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            em, f1 = res["EM"], res["F1"]
            samples, total = res["samples_processed"], res["total_samples"]
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
    base_model_id = r"D:\projects\nlp\Mistral\Mistral-7B-v0.3"
    adapter_path = r"C:\Users\admin\Desktop\Mistral-7b\lora_adapter"
    TEST_DATA_PATH = r"D:\projects\TISER\data\TISER_test_random_3000.json"

    PROMPT_MODE = "tiser"  # standard / tiser
    FULL_EVALUATION = True
    MAX_SAMPLES_PER_DATASET = None if FULL_EVALUATION else 100

    BATCH_SIZE = 32

    # ===== Self-Consistency Configuration =====
    USE_SELF_CONSISTENCY = True
    SC_K = 5
    SC_TEMPERATURE = 0.6
    SC_TOP_P = 0.9
    SC_FALLBACK_THRESHOLD = 0.6
    # =============================================

    print(f"Model: {base_model_id}")
    print(f"Evaluation Data: {TEST_DATA_PATH}")
    print(f"Prompt Mode: {PROMPT_MODE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Self-Consistency: {USE_SELF_CONSISTENCY} "
        f"(k={SC_K}, temp={SC_TEMPERATURE}, top_p={SC_TOP_P}, thr={SC_FALLBACK_THRESHOLD})"
    )
    print()

    start_time = time.time()
    test_data = load_test_data(TEST_DATA_PATH)
    data_load_time = time.time() - start_time

    evaluator = TISER_Evaluator(base_model_id, adapter_path, prompt_mode=PROMPT_MODE)

    print("\n" + "=" * 90)
    print("Evaluation start")
    print("=" * 90)

    results = {}
    table1_datasets = ["TGQA", "TimeQA (easy)", "TimeQA (hard)"]
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
            samples,
            ds_name,
            MAX_SAMPLES_PER_DATASET,
            verbose=False,
            batch_size=BATCH_SIZE,
            use_self_consistency=USE_SELF_CONSISTENCY,
            sc_k=SC_K,
            sc_temperature=SC_TEMPERATURE,
            sc_top_p=SC_TOP_P,
            sc_fallback_threshold=SC_FALLBACK_THRESHOLD
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

    tag = (
        f"sc{SC_K}_t{SC_TEMPERATURE}_p{SC_TOP_P}_thr{SC_FALLBACK_THRESHOLD}"
        if USE_SELF_CONSISTENCY else "greedy"
    )
    output_file = f"table1_results_{PROMPT_MODE}_{tag}_batch{BATCH_SIZE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_mode": PROMPT_MODE,
            "batch_size": BATCH_SIZE,
            "self_consistency": {
                "enabled": USE_SELF_CONSISTENCY,
                "k": SC_K,
                "temperature": SC_TEMPERATURE,
                "top_p": SC_TOP_P,
                "fallback_threshold": SC_FALLBACK_THRESHOLD
            },
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
