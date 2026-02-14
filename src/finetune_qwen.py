import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-7B"
output_dir = "model/Qwen/Qwen2.5-7B-LoRA"
data_path = "data/TISER_train.json"

# Hyperparameters
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

def train():
    print(f"Loading model: {model_id}...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model (Bfloat16 + Flash Attention 2)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    )


    def formatting_prompts_func(data):
        text = {
            "prompt": data['prompt'].strip(),
            "completion": " " + data['output'].strip()
        }
        return text
    

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,                       # LoRA Rank
        lora_alpha=32,              # Usally twice the LoRA Rank
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear' # Finetune all linear
    )

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    dataset = dataset.map(formatting_prompts_func)
    dataset_split = dataset.train_test_split(test_size=1000, seed=114)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total evaluation samples: {len(eval_dataset)}")

    # Set training parameters
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=2000,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        report_to="wandb",
        run_name="Qwen2.5-7B-TISER",
        
        # Sequence length
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
        completion_only_loss=True,
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train()