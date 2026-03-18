#!/usr/bin/env python3
"""Side-by-side comparison: Base Qwen 3.5 0.8B vs Fine-tuned Logic Engine"""

import json
import subprocess
import os

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

from mlx_lm import load, generate

# Load the fine-tuned model
print("Loading fine-tuned Logic Engine...")
model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)
print("Fine-tuned model loaded!\n")

# Grab the first test puzzle
with open("test_dataset.jsonl", "r") as f:
    item = json.loads(f.readline())

puzzle_prompt = item["conversations"][0]["value"]
expected_answer = item["conversations"][1]["value"]

print("=" * 80)
print("PUZZLE:")
print("=" * 80)
print(puzzle_prompt)
print("=" * 80)

# --- Fine-tuned model (use chat template) ---
print("\n--- FINE-TUNED LOGIC ENGINE RESPONSE:")
print("-" * 40)

messages = [{"role": "user", "content": puzzle_prompt}]
chat_prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

response = generate(
    model, tokenizer,
    prompt=chat_prompt,
    max_tokens=512,
    verbose=False
)
print(response)

# --- Base model via Ollama ---
print("\n\n🤖 BASE QWEN 3.5 0.8B RESPONSE:")
print("-" * 40)
result = subprocess.run(
    ["ollama", "run", "qwen3.5:0.8b", puzzle_prompt],
    capture_output=True, text=True, timeout=120
)
print(result.stdout[:2000])

print("\n\n📋 EXPECTED ANSWER:")
print("-" * 40)
print(expected_answer[:500])
