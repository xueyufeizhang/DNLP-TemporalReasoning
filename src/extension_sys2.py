import json
import torch
import re
import os
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import time
import networkx as nx

class TISER_Evaluator:
    def __init__(self, base_model: str, adapter: str, prompt_mode: str = "standard"):
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
                print("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(base_model, self.adapter)
                print("Model Loaded")
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

    # ==========================================
    # [Core] Logic Validation
    # ==========================================
    def verify_timeline_logic(self, response):
        # --- Checkpoint 1: Structure completeness ---
        timeline_match = re.search(r'<timeline>(.*?)</timeline>', response, re.DOTALL | re.IGNORECASE)
        if not timeline_match:
            return False, "Missing <timeline> tags. You must generate a structured timeline first."
        
        timeline_text = timeline_match.group(1).strip()
        if not timeline_text:
            return False, "Empty <timeline> content."

        # --- Checkpoint 2: Parsing the validity ---
        events = defaultdict(dict)
        # Parsing the format: (Event Name) starts/ends at Year
        pattern = r'\((.*?)\)\s*(starts|ends)\s*at\s*(\d+)'
        matches = re.findall(pattern, timeline_text, re.IGNORECASE)
        
        if not matches:
            # Tags available but wrong format
            return False, "Timeline format is incorrect. Expected format: (Event) starts/ends at Year."

        all_mentioned_years = set()

        for event_name, type_, year_str in matches:
            event_name = event_name.strip()
            try:
                year = int(year_str)
                all_mentioned_years.add(year)
            except ValueError:
                continue 
                
            if type_.lower() == 'starts':
                events[event_name]['start'] = year
            elif type_.lower() == 'ends':
                events[event_name]['end'] = year

        # --- Checkpoint 3: Physical logic contradiction ---
        for event, times in events.items():
            start = times.get('start')
            end = times.get('end')
            if start is not None and end is not None:
                if end < start:
                    return False, f"Logical Error: Event '{event}' ends in {end} but starts in {start}."

        
        # Extracting Year data from answer
        answer_years = re.findall(r'\b(1\d{3}|20\d{2})\b', response.split('</timeline>')[-1])
        if answer_years:
            ans_year = int(answer_years[0])
            if all_mentioned_years:
                min_year = min(all_mentioned_years)
                max_year = max(all_mentioned_years)
                # if ans_year not in all_mentioned_years: 
                #     return False, f"Reasoning Mismatch: The answer year {ans_year} is not derived from the timeline events {sorted(list(all_mentioned_years))}."
                pass 

        return True, None

    
    def _raw_generate(self, prompt: str, max_new_tokens: int = 512) -> str:
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


    def generate_answer_with_system2(self, prompt, max_new_tokens=512, max_retries=1):
        """
        Return: (final_response, retry_count, trace_log)
        """
        # 1. Initializing Generation
        initial_response = self._raw_generate(prompt, max_new_tokens)
        
        # Checking validity
        is_valid, error_msg = self.verify_timeline_logic(initial_response)
        
        trace_log = {
            'triggered': False,
            'initial_output': initial_response,
            'error_detected': None,
            'final_output': initial_response
        }

        retry_count = 0
        current_response = initial_response
        
        # If empty at first, pass
        if not initial_response.strip():
            return "", 0, trace_log

        while not is_valid and retry_count < max_retries:
            retry_count += 1
            trace_log['triggered'] = True
            trace_log['error_detected'] = error_msg
            
            # 2. Structure correction prompt
            intervention_prompt = (
                f"\n\n[System Alert]: I detected a logical error or missing timeline in your response: {error_msg}.\n"
                "Please regenerate the response correctly.\n"
                "Step 1: Extract the events into <timeline> tags.\n"
                "Step 2: Provide the final <answer>.\n\n"
                "Corrected Response:\n"
            )
            
            new_input_prompt = prompt + "\n" + current_response + intervention_prompt
            
            # 3. Regeneration
            # print(f"[DEBUG] System 2 Retry {retry_count} Prompt Tail:\n{new_input_prompt[-300:]}") # 调试用
            new_response = self._raw_generate(new_input_prompt, max_new_tokens)
            
            # Avoid output empty answer
            if not new_response or not new_response.strip():
                print(f"System 2 generate an empty answer (Error: {error_msg})")
                break
            
            # Update current_response
            current_response = new_response
        
        trace_log['final_output'] = current_response
        return current_response, retry_count, trace_log
        
    
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
            "seconds", "second",
            "starts", "ends"
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
    
    
    
    def evaluate_dataset(self, dataset_samples, dataset_name, max_samples=None):
        if max_samples: samples = dataset_samples[:max_samples]
        else: samples = dataset_samples
        
        total_em, total_f1 = 0.0, 0.0
        processed = 0
        system2_triggered_count = 0 
        
        intervention_logs = []
        
        pbar = tqdm(samples, desc=f"  {dataset_name.ljust(20)}", unit="smpl")
        
        for i, sample in enumerate(pbar):
            prompt = self.build_prompt(sample)
            ground_truth = sample.get('answer', '')
            if not prompt or not ground_truth: continue
            
            try:
                response, retries, trace_log = self.generate_answer_with_system2(prompt, max_retries=1)
                
                if retries > 0:
                    system2_triggered_count += 1
                    intervention_logs.append({
                        'id': i,
                        'dataset': dataset_name,
                        'prompt': prompt,
                        'ground_truth': ground_truth,
                        'initial_wrong_output': trace_log['initial_output'],
                        'error_message': trace_log['error_detected'],
                        'corrected_output': trace_log['final_output']
                    })
                
                predicted = self.extract_answer(response)
                em, f1 = self.calculate_em_f1(predicted, ground_truth)

                # # =========== Testing start ===========
                # if f1 > 0.7 and em < 0.5 and processed < 10:
                # # if processed < 10:
                #     print(f"   Generated (Raw): {response!r}")
                #     print(f"   Predicted (Cln): {predicted!r}")
                #     print(f"   Ground Truth:    {ground_truth!r}")
                #     print(f"   Norm Predicted:  {self.normalize_answer(predicted)!r}")
                #     print(f"   Norm Truth:      {self.normalize_answer(ground_truth)!r}")
                # # =========== Testing end ===========

                total_em += em
                total_f1 += f1
                processed += 1

                if processed > 0:
                    pbar.set_postfix(
                        EM=f'{total_em/processed:.3f}',
                        F1=f'{total_f1/processed:.3f}',
                        Sys2=f'{system2_triggered_count}',
                        refresh=True
                    )
                    
            except Exception as e:
                continue
        
        pbar.close()
        
        if processed == 0: return 0.0, 0.0, 0, []
        
        avg_em = total_em / processed
        avg_f1 = total_f1 / processed
        
        if system2_triggered_count > 0:
            print(f"  [Stats] Triggered corrections: {system2_triggered_count}/{processed} ({system2_triggered_count/processed:.1%})")
        
        return avg_em, avg_f1, processed, intervention_logs, system2_triggered_count

def load_test_data(json_path):
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

    table1_datasets = ["TGQA", "TempReason (L2)"]
    print("\nDatset distribution:")
    for ds in table1_datasets:
        cnt = len(organized_data.get(ds, []))
        print(f"  {ds.ljust(25)}: {cnt:>6} samples")

    return organized_data

def print_table1_format(results):
    print("\n" + "="*90 + "\nEvaluation results\n" + "="*90)
    print(f"{'Dataset':<25} {'EM':<20} {'F1':<20} {'#Sample':<15}")
    print("-" * 85)
    table1_order = ['TGQA', 'TempReason (L2)']
    em_scores, f1_scores = [], []
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            print(f"{ds_name:<25} {res['EM']:<20.3f} {res['F1']:<20.3f} {res['samples_processed']}/{res['total_samples']:<15}")
            if res['samples_processed'] > 0:
                em_scores.append(res['EM'])
                f1_scores.append(res['F1'])
        else:
            print(f"{ds_name:<25} {'-':<20} {'-':<20} 0/0")
    if em_scores:
        print("-" * 85)
        print(f"{'Macro Average':<25} {np.mean(em_scores):<20.3f} {np.mean(f1_scores):<20.3f}")

def main():
    base_model_id = "mistralai/Mistral-7B-v0.3"
    adapter_path = "model/Mistral/Mistral-7B-v0.3-LoRA"
    # base_model_id = "Qwen/Qwen2.5-7B"
    # adapter_path = "model/Qwen/Qwen2.5-7B-LoRA"
    TEST_DATA_PATH = "data/TISER_test_random_3000.json"

    # PROMPT_MODE = "standard"
    PROMPT_MODE = "tiser"

    FULL_EVALUATION = True
    MAX_SAMPLES_PER_DATASET = 20 if not FULL_EVALUATION else None

    print(f"Model: {base_model_id}")
    print(f"Evaluation Data: {TEST_DATA_PATH}")
    print(f"Prompt Mode: {PROMPT_MODE}")
    print()

    evaluator = TISER_Evaluator(base_model_id, adapter_path, prompt_mode=PROMPT_MODE)
    test_data = load_test_data(TEST_DATA_PATH)
    
    results = {}
    all_intervention_cases = []
    total_evaluation_time = 0.0

    table1_datasets = ['TGQA', 'TempReason (L2)']
    
    print("\nEvaluation start (Intervention Mode: ON)")
    
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
        em_score, f1_score, processed, current_logs, sys2_count = evaluator.evaluate_dataset(
            samples, ds_name, MAX_SAMPLES_PER_DATASET, verbose=False
        )
        eval_time = time.time() - eval_start

        results[ds_name] = {
            "EM": em_score,
            "F1": f1_score,
            "Sys2": sys2_count,
            "samples_processed": processed,
            "total_samples": total_samples,
            "eval_time": eval_time
        }
        all_intervention_cases.extend(current_logs)
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
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetails saved to: {output_file}")
    
    # Save Bad Case Log
    log_file = "system2_intervention_logs.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(all_intervention_cases, f, indent=2, ensure_ascii=False)
    
    print(f"\nCorrection cases saved to: {log_file}")
    print(f"   Capture {len(all_intervention_cases)} correction cases")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()