# DNLP-TemporalReasoning
This project aims to reproduce the core methods of the TISER (Temporal Information and Reasoning) paper and explores extensions to further enhance the model's temporal reasoning capabilities.

The project consists of three core components:
1. **TISER Baseline Reproduction**: Performed LoRA fine-tuning and standard evaluation on _Qwen2.5-7B_ and _Mistral-7B_ models based on TISER data formats.
2. **Extension 1 - Self-Consistency**: Introduces a Self-Consistency mechanism to improve reasoning stability through multiple sampling and majority voting.
3. **Extension 2 - System 2 Logic Calibration**: Introduces a System 2 based logic calibration mechanism, enabling the model to generate timelines and perform self-correction based on logical checks.

## Directory Structure

```text
DNLP-TemporalReasoning/
│
├── data/               # Dataset files (SemEval 2026)
│   ├── TISER_train.json          # Raw training data
│   ├── TISER_test.json           # Raw test data
│   └── TISER_test_random_3000.json  # Sampled test data
├── model/              # Model weights and LoRA adapters
├── report/             # LaTex code of final report
├── src/
│   ├── data_format.ipynb         # Data preprocessing and sampling Notebook
│   ├── finetune_qwen.py          # Qwen2.5 LoRA fine-tuning script (Baseline)
│   ├── finetune_mistral.py       # Mistral LoRA fine-tuning script (Baseline)
│   ├── eval_benchmark.py         # Standard evaluation script (Single inference - Baseline)
│   ├── eval_benchmark_bathch.py  # Batch evaluation script (Batch inference - Baseline)
│   ├── extension_sc.py           # Extension 1: Self-Consistency (Majority Voting)
└── └── extension_sys2.py         # Extension 2: System 2 (Logic-based Iterative Correction)
```

## Quick Start
1. Environment Preparation

Ensure you have Python 3.8+ and PyTorch installed. It is recommended to install the following dependencies:
```bash
> pip install torch transformers peft datasets trl bitsandbytes tqdm numpy
```

2. Data Preparation

Before training or evaluation, ensure the corresponding data files exist in the `data/` directory. You can use `src/data_format.ipynb` to sample and format the raw data:
```bash
> jupyter notebook src/data_format.ipynb
```

3. Model Fine-tuning (TISER Reproduction)

This project supports fine-tuning models using LoRA (Low-Rank Adaptation) to reproduce the training process of the TISER paper.

#### Fine-tuning Qwen2.5
`finetune_qwen.py` uses hardcoded paths. Please check `model_id` and `data_path` inside the script before running.
```bash
> python src/finetune_qwen.py
```

#### Fine-tuning Mistral
`finetune_mistral.py` supports command-line arguments for more flexibility:
```bash
> python src/finetune_mistral.py \
    --model_path "mistralai/Mistral-7B-v0.3" \
    --train_path "data/TISER_train.json" \
    --output_dir "model/Mistral/output" \
    --batch_size 1 \
    --epochs 3
```

4. Evaluation & Extensions

Multiple evaluation modes are provided to verify the reproduction results and the effectiveness of the extension methods.

Note: Before running, please modify `base_model_id` and `adapter_path` in the scripts to match your local paths.

**A. Standard Evaluation (Baseline Reproduction)**

The most basic evaluation script, corresponding to the standard reasoning mode in the paper. It calculates Exact Match (EM) and F1 scores.
```bash
> python src/eval_benchmark.py
```

**B. Batch Evaluation (Baseline Reproduction)**

Uses batch generation to accelerate the inference process, also used for baseline metric testing.

```bash
> python src/eval_benchmark_bathch.py
```

**C. Extension 1: Self-Consistency**

Introduces a self-consistency mechanism on top of TISER. It improves answer stability by sampling multiple times and performing majority voting.
- Default configuration: `k=5` samples.
- Fallback Mechanism: If the voting confidence is low, it falls back to the greedy search result.
```bash
> python src/extension_sc.py
```

**D. Extension 2: System 2 Logic Calibration**

Simulates the "System 2" slow-thinking process, aiming to solve hallucination issues in complex temporal reasoning:
1. Generates a preliminary answer containing a `<timeline>`.
2. Code parses and verifies the timeline logic (e.g., checks if an end time is earlier than the start time).
3. If a logical contradiction is found, it automatically constructs a prompt to ask the model for self-correction.
```bash
> python src/extension_sys2.py
```

## Supported Datasets
The evaluation scripts have built-in logic to load and evaluate the following datasets:
- TGQA
- TempReason (L2, L3)
- TimeQA (Easy, Hard)

## Key Features
- **LoRA Efficient Fine-tuning**: Reproduces the paper's training settings, allowing fine-tuning of 7B models with low VRAM usage.
- **Prompt Mode**: Supports `standard` (QA format) and `tiser` (Specific format) prompt modes.
- **Flash Attention 2**: Enable Flash Attention 2 acceleration by default (requires hardware support).
- **Advanced Reasoning Extensions**: Integrates Self-Correction and Self-Consistency strategies.

## License
This project is licensed under the Apache License 2.0.
