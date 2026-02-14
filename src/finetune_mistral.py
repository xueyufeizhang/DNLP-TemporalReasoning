import os
import json
import argparse
from typing import Dict, List, Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def load_json_or_jsonl(path: str) -> Dataset:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            ds = Dataset.from_list(data)
            if "prompt" not in ds.column_names or "output" not in ds.column_names:
                raise ValueError(f"JSON lacks of prompt/output field. Current field：{ds.column_names}")
            return ds
    except json.JSONDecodeError:
        pass

    # 再尝试 JSONL
    ds = load_dataset("json", data_files={"train": path})["train"]
    if "prompt" not in ds.column_names or "output" not in ds.column_names:
        raise ValueError(f"JSONL lacks of prompt/output field. Current field：{ds.column_names}")
    return ds


class PromptMaskedDataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [(f.get("prompt") or "").rstrip() for f in features]
        outputs = [(f.get("output") or "").lstrip() for f in features]

        full_texts = [p + "\n" + o for p, o in zip(prompts, outputs)]

        enc = self.tok(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()

        prompt_enc = self.tok(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        for i in range(len(features)):
            prompt_len = len(prompt_enc["input_ids"][i])
            seq_len = int(attention_mask[i].sum().item())
            prompt_len = min(prompt_len, seq_len)

            labels[i, :prompt_len] = -100
            labels[i, attention_mask[i] == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)   # e.g. mistralai/Mistral-7B-v0.3
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="mistral7b_v0_3_lora_out")
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_output_dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and (not use_bf16)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dataset
    ds = load_json_or_jsonl(args.train_path)

    # model
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
    )

    model.config.use_cache = False

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_collator = PromptMaskedDataCollator(tokenizer, max_length=args.max_len)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()

    # 1) Save LoRA adapter first
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("Saved LoRA adapter to:", adapter_dir)

    # 2) Then export complete model：merge LoRA -> base weights
    merge_dir = args.merge_output_dir or os.path.join(args.output_dir, "merged_full_model")
    os.makedirs(merge_dir, exist_ok=True)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merge_dir, safe_serialization=True)
    tokenizer.save_pretrained(merge_dir)

    print("Done! Saved merged fine-tuned model to:", merge_dir)


if __name__ == "__main__":
    main()
