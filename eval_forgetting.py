#!/usr/bin/env python3
"""Test for catastrophic forgetting: compare base vs fine-tuned on general tasks."""

import os, time
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

from mlx_lm import load, generate

# General knowledge / basic tasks the base model should handle
tests = [
    {"q": "What is the capital of France?", "expect": "paris"},
    {"q": "What is 12 + 15?", "expect": "27"},
    {"q": "What is 7 * 8?", "expect": "56"},
    {"q": "Name 3 colors of the rainbow.", "expect": "red"},
    {"q": "What planet is closest to the Sun?", "expect": "mercury"},
    {"q": "Translate 'hello' to Spanish.", "expect": "hola"},
    {"q": "What is the boiling point of water in Celsius?", "expect": "100"},
    {"q": "Who wrote Romeo and Juliet?", "expect": "shakespeare"},
    {"q": "What is the largest ocean on Earth?", "expect": "pacific"},
    {"q": "How many legs does a spider have?", "expect": "8"},
    {"q": "What is the square root of 144?", "expect": "12"},
    {"q": "What gas do plants absorb from the atmosphere?", "expect": "carbon dioxide"},
    {"q": "Finish this sentence: The quick brown fox jumps over the lazy", "expect": "dog"},
    {"q": "What year did World War II end?", "expect": "1945"},
    {"q": "Is water wet? Answer yes or no.", "expect": "yes"},
]

def run_model(model, tokenizer, label):
    correct = 0
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    for i, t in enumerate(tests):
        msgs = [{"role": "user", "content": t["q"]}]
        cp = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        resp = generate(model, tokenizer, prompt=cp, max_tokens=100, verbose=False)
        
        got = resp.strip().lower()
        match = t["expect"].lower() in got
        if match:
            correct += 1
        
        status = "[MATCH]" if match else "[MISMATCH]"
        print(f"  {status} Q: {t['q']}")
        print(f"     A: {resp.strip()[:80]}")
        print()
    
    pct = correct / len(tests) * 100
    print(f"  SCORE: {correct}/{len(tests)} ({pct:.0f}%)")
    return correct

# Load BASE model (no adapters)
print("Loading BASE model (no adapters)...")
base_model, base_tok = load("mlx-community/Qwen3.5-0.8B-4bit")
base_score = run_model(base_model, base_tok, "BASE MODEL (no fine-tuning)")

# Free memory
del base_model
import gc; gc.collect()

# Load FINE-TUNED model (with adapters)  
print("\n\nLoading FINE-TUNED model (with adapters)...")
ft_model, ft_tok = load("mlx-community/Qwen3.5-0.8B-4bit", adapter_path="./adapters")
ft_score = run_model(ft_model, ft_tok, "FINE-TUNED MODEL (Logic Engine)")

# Comparison
print(f"\n{'='*60}")
print(f"  CATASTROPHIC FORGETTING TEST")
print(f"{'='*60}")
print(f"  Base model:      {base_score}/{len(tests)} ({base_score/len(tests)*100:.0f}%)")
print(f"  Fine-tuned model: {ft_score}/{len(tests)} ({ft_score/len(tests)*100:.0f}%)")
diff = ft_score - base_score
if diff > 0:
    print(f"  Change:          +{diff} (IMPROVED)")
elif diff < 0:
    print(f"  Change:          {diff} (DEGRADED)")
else:
    print(f"  Change:          0 (NO CHANGE)")
print(f"{'='*60}")
