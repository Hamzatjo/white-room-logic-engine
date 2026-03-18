# White-Room Logic Engine

**Teaching Deductive Reasoning to a Sub-Billion Parameter Language Model Through Synthetic Data Fine-Tuning**

*Hamza Zaraoui — Amsterdam University of Applied Sciences (HvA)*

[Read the Paper](paper.md)

---

## Overview

The White-Room Logic Engine is a fine-tuned 0.8B parameter language model specialized for deductive reasoning. Starting from Qwen 3.5 0.8B (4-bit quantized), we apply LoRA fine-tuning using ~3,200 synthetically generated logic puzzles to teach the model structured, multi-step deductive reasoning.

### Key Results

| Metric | Score |
|---|---|
| ProntoQA Benchmark (1–5 hop) | **60%** |
| Real-world entity puzzles | **87%** |
| Format compliance | **100%** |
| Rule-order independence | **0% drop** |
| Catastrophic forgetting | **None** |
| Base model (no fine-tuning) | **0%** |

## Quick Start

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (for MLX) or any system with `pip`
- ~4 GB disk space

### Setup

```bash
python3 -m venv mlx_env
source mlx_env/bin/activate
pip install mlx-lm
```

### Interactive Chat

```bash
python3 chat_engine.py
```

This loads the fine-tuned model and lets you test it with custom logic puzzles.

### Run Evaluations

```bash
# In-distribution test set (318 puzzles)
python3 eval_model.py

# Real-world names + format variations
python3 eval_variants.py

# Shuffled rule order test
python3 eval_shuffled.py

# ProntoQA benchmark (requires cloning ProntoQA repo)
git clone https://github.com/asaparov/prontoqa.git /tmp/prontoqa
cd /tmp/prontoqa && unzip generated_ood_data.zip -d generated_data
cd -
python3 eval_prontoqa.py

# Catastrophic forgetting test
python3 eval_forgetting.py
```

## Repository Structure

```
white-room-logic-engine/
│
├── paper.md                        # Full research paper
│
├── adapters/                       # Trained LoRA weights (~7 MB)
│   ├── adapters.safetensors        # Final adapter weights
│   └── adapter_config.json         # LoRA configuration
│
├── data/                           # MLX-formatted training splits
│   ├── train.jsonl                 # Training set
│   └── valid.jsonl                 # Validation set
│
├── train_dataset.jsonl             # Full training dataset (3,179 examples)
├── test_dataset.jsonl              # Held-out test set (318 examples)
├── test_realworld.jsonl            # Real-world entity test puzzles (15)
├── test_format_variations.jsonl    # Format variation test puzzles (10)
├── test_ordered.jsonl              # Ordered rule test puzzles (8)
├── test_shuffled.jsonl             # Shuffled rule test puzzles (8)
│
├── generate_gemini.py              # Dataset generation (Gemini 3 Flash)
├── generate_seeds.py               # Puzzle parameter pre-generation
├── generate_variant_tests.py       # Variant test set generation
├── prepare_dataset.py              # ShareGPT format converter
│
├── eval_model.py                   # Main evaluation (in-distribution + base model)
├── eval_variants.py                # Real-world + format variation eval
├── eval_shuffled.py                # Rule-order independence eval
├── eval_prontoqa.py                # ProntoQA benchmark eval
├── eval_forgetting.py              # Catastrophic forgetting eval
│
├── chat_engine.py                  # Interactive chat interface
├── compare_models.py               # Side-by-side model comparison
├── prontoqa_results.json           # ProntoQA benchmark results
│
└── README.md
```

## Training

The model was trained using MLX on an Apple Mac Mini M4 (16 GB RAM):

- **Base model:** `mlx-community/Qwen3.5-0.8B-4bit`
- **Method:** LoRA (1.8M trainable parameters, 0.24% of total)
- **Iterations:** 1,000
- **Training time:** ~4 hours
- **Final loss:** 0.921 (train) / 0.930 (validation)

To reproduce training:

```bash
export HF_HOME=$(pwd)/hf_cache

python3 -m mlx_lm.lora \
  --model mlx-community/Qwen3.5-0.8B-4bit \
  --data ./data \
  --train \
  --iters 1000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --num-layers 16 \
  --adapter-path ./adapters
```

## Dataset

The dataset contains 3,497 synthetic logic puzzles across five categories:

| Category | Count |
|---|---|
| Conditional Logic | 707 |
| Relational Mapping | 656 |
| Temporal Ordering | 600 |
| Set Theory | 542 |
| Elimination Grid | 356 |

All puzzles use fictional entity names (e.g., "Lanvkaen", "Fethjeixks") to prevent the model from relying on pre-trained knowledge — the **White-Room principle**.

## Citation

If you use this work, please cite:

```bibtex
@article{zaraoui2026whiteroom,
  title={White-Room Logic Engine: Teaching Deductive Reasoning to a Sub-Billion Parameter Language Model Through Synthetic Data Fine-Tuning},
  author={Zaraoui, Hamza},
  year={2026},
  note={Independent Research, Amsterdam University of Applied Sciences}
}
```

## License

MIT License
