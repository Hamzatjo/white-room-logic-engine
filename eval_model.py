#!/usr/bin/env python3
"""Improved evaluation with better fuzzy matching + base model comparison."""

import json
import re
import os
import time
import subprocess

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def answers_match(generated, expected):
    gen_raw = extract_answer(generated)
    exp_raw = extract_answer(expected)
    gen = normalize(gen_raw)
    exp = normalize(exp_raw)
    
    # Exact match
    if gen == exp:
        return True
    
    # One contains the other
    if gen in exp or exp in gen:
        return True
    
    # Both say "cannot be determined"
    if "cannot be determined" in gen and "cannot be determined" in exp:
        return True
    
    # Synonyms for state (deactivated/inactive/off, activated/active/on)
    inactive_words = {"deactivated", "inactive", "off", "disabled", "shut down"}
    active_words = {"activated", "active", "on", "enabled"}
    gen_inactive = any(w in gen for w in inactive_words)
    gen_active = any(w in gen for w in active_words)
    exp_inactive = any(w in exp for w in inactive_words)
    exp_active = any(w in exp for w in active_words)
    if gen_inactive and exp_inactive:
        return True
    if gen_active and exp_active:
        return True
    
    # Both say yes or both say no
    if gen.strip() in ("yes", "yes it is", "yes it does") and exp.strip() in ("yes", "yes it is", "yes it does"):
        return True
    if gen.strip() in ("no", "no it is not", "no it does not") and exp.strip() in ("no", "no it is not", "no it does not"):
        return True
    
    # Number-only answers
    gen_nums = re.findall(r"\d+", gen)
    exp_nums = re.findall(r"\d+", exp)
    if gen_nums and exp_nums and gen_nums[0] == exp_nums[0]:
        # Check if the units are similar (allow typos with 80% char overlap)
        gen_unit = re.sub(r"\d+\s*", "", gen).strip()
        exp_unit = re.sub(r"\d+\s*", "", exp).strip()
        if not gen_unit or not exp_unit:
            return gen_nums[0] == exp_nums[0] and not gen_unit and not exp_unit
        overlap = sum(1 for a, b in zip(gen_unit, exp_unit) if a == b)
        max_len = max(len(gen_unit), len(exp_unit))
        if max_len > 0 and overlap / max_len >= 0.8:
            return True
    
    # Key word overlap (60%+ of expected words found in generated)
    gen_words = set(w for w in gen.split() if len(w) > 2)
    exp_words = [w for w in exp.split() if len(w) > 2]
    if exp_words:
        overlap = sum(1 for w in exp_words if w in gen_words)
        if overlap / len(exp_words) >= 0.7:
            return True
    
    return False

def eval_model(name, generate_fn, test_data):
    """Run evaluation using the provided generate function."""
    results = {
        "correct": 0, "incorrect": 0,
        "has_reasoning_tag": 0, "has_answer_tag": 0,
        "loop_detected": 0, "timeout": 0,
        "total": len(test_data), "failures": []
    }
    
    start_time = time.time()
    
    for i, item in enumerate(test_data):
        puzzle_prompt = item["conversations"][0]["value"]
        expected = item["conversations"][1]["value"]
        
        response = generate_fn(puzzle_prompt)
        
        if response is None:
            results["timeout"] += 1
            results["incorrect"] += 1
            results["failures"].append({
                "index": i,
                "expected_answer": extract_answer(expected)[:100],
                "got_answer": "TIMEOUT"
            })
        else:
            has_r = "<reasoning>" in response
            has_a = "<answer>" in response
            if has_r: results["has_reasoning_tag"] += 1
            if has_a: results["has_answer_tag"] += 1
            
            words = response.split()
            if len(words) > 20 and len(set(words)) / len(words) < 0.2:
                results["loop_detected"] += 1
            
            if answers_match(response, expected):
                results["correct"] += 1
            else:
                results["incorrect"] += 1
                results["failures"].append({
                    "index": i,
                    "expected_answer": extract_answer(expected)[:100],
                    "got_answer": extract_answer(response)[:100]
                })
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(test_data) - i - 1) / rate
            pct = results["correct"] / (i + 1) * 100
            print(f"  [{name}] [{i+1}/{len(test_data)}] Accuracy: {pct:.1f}% | ~{remaining:.0f}s left")
    
    elapsed = time.time() - start_time
    return results, elapsed

# Load test data
with open("test_dataset.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

print(f"Loaded {len(test_data)} test puzzles.\n")

# ===== FINE-TUNED MODEL =====
print("=" * 60)
print("EVALUATING FINE-TUNED LOGIC ENGINE")
print("=" * 60)

from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)

def finetuned_generate(prompt):
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=False
    )
    return generate(model, tokenizer, prompt=chat_prompt, max_tokens=512, verbose=False)

ft_results, ft_time = eval_model("FT", finetuned_generate, test_data)

# ===== BASE MODEL (via Ollama, with 30s timeout) =====
print("\n" + "=" * 60)
print("EVALUATING BASE QWEN 3.5 0.8B (Ollama)")
print("=" * 60)

def base_generate(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen3.5:0.8b", prompt],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return None

base_results, base_time = eval_model("BASE", base_generate, test_data)

# ===== FINAL COMPARISON =====
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"{'Metric':<25} {'Fine-tuned':<15} {'Base Model':<15}")
print("-" * 55)
print(f"{'Correct answers':<25} {ft_results['correct']}/{ft_results['total']} ({ft_results['correct']/ft_results['total']*100:.1f}%)    {base_results['correct']}/{base_results['total']} ({base_results['correct']/base_results['total']*100:.1f}%)")
print(f"{'<reasoning> tag':<25} {ft_results['has_reasoning_tag']/ft_results['total']*100:.0f}%            {base_results['has_reasoning_tag']/base_results['total']*100:.0f}%")
print(f"{'<answer> tag':<25} {ft_results['has_answer_tag']/ft_results['total']*100:.0f}%            {base_results['has_answer_tag']/base_results['total']*100:.0f}%")
print(f"{'Loops detected':<25} {ft_results['loop_detected']}              {base_results['loop_detected']}")
print(f"{'Timeouts':<25} {ft_results['timeout']}              {base_results['timeout']}")
print(f"{'Time':<25} {ft_time:.0f}s            {base_time:.0f}s")
print("=" * 60)

with open("eval_comparison.json", "w") as f:
    json.dump({"finetuned": ft_results, "base": base_results}, f, indent=2)
print("\nDetailed results saved to eval_comparison.json")
