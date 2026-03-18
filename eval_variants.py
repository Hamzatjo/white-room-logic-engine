#!/usr/bin/env python3
"""Evaluate on variant test sets: real-world names + format variations."""

import json
import re
import os
import time

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

from mlx_lm import load, generate

def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def has_format(text):
    return "<reasoning>" in text, "<answer>" in text

# Load model
print("Loading Logic Engine...")
model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)
print("Model loaded!\n")

def run_eval(test_file, label):
    with open(test_file) as f:
        data = [json.loads(line) for line in f]
    
    print(f"\n{'='*60}")
    print(f"  {label} ({len(data)} puzzles)")
    print(f"{'='*60}\n")
    
    correct = 0
    format_ok = 0
    loops = 0
    
    for i, item in enumerate(data):
        prompt = item["conversations"][0]["value"]
        expected = extract_answer(item["conversations"][1]["value"])
        
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False
        )
        
        response = generate(model, tokenizer, prompt=chat_prompt, max_tokens=512, verbose=False)
        got = extract_answer(response)
        has_r, has_a = has_format(response)
        
        # Loop detection
        words = response.split()
        is_loop = len(words) > 20 and len(set(words)) / len(words) < 0.2
        if is_loop:
            loops += 1
        
        if has_r and has_a:
            format_ok += 1
        
        # Print each result for manual review
        print(f"  Puzzle #{i+1}:")
        print(f"    Format: {'[OK]' if has_r and has_a else '[FAIL]'} {'(LOOP)' if is_loop else ''}")
        print(f"    Expected: {expected[:80]}")
        print(f"    Got:      {got[:80]}")
        
        # Simple match check
        exp_norm = re.sub(r'[^a-z0-9\s]', '', expected.lower()).strip()
        got_norm = re.sub(r'[^a-z0-9\s]', '', got.lower()).strip()
        
        match = False
        if exp_norm in got_norm or got_norm in exp_norm:
            match = True
        elif "cannot be determined" in got_norm and "cannot be determined" in exp_norm:
            match = True
        elif "no" == got_norm.split()[0] and "no" == exp_norm.split()[0]:
            match = True
        elif "yes" == got_norm.split()[0] and "yes" == exp_norm.split()[0]:
            match = True
        else:
            # Key word overlap
            exp_words = [w for w in exp_norm.split() if len(w) > 3]
            got_words = set(got_norm.split())
            if exp_words:
                overlap = sum(1 for w in exp_words if w in got_words) / len(exp_words)
                if overlap >= 0.6:
                    match = True
        
        if match:
            correct += 1
            print(f"    Result:   [MATCH]")
        else:
            print(f"    Result:   [MISMATCH]")
        print()
    
    print(f"  {'─'*50}")
    print(f"  SUMMARY: {label}")
    print(f"  Accuracy:     {correct}/{len(data)} ({correct/len(data)*100:.0f}%)")
    print(f"  Format OK:    {format_ok}/{len(data)} ({format_ok/len(data)*100:.0f}%)")
    print(f"  Loops:        {loops}/{len(data)}")
    print(f"  {'─'*50}")
    
    return {"correct": correct, "total": len(data), "format_ok": format_ok, "loops": loops}

# Run both test sets
r1 = run_eval("test_realworld.jsonl", "TEST 1: Real-World Names")
r2 = run_eval("test_format_variations.jsonl", "TEST 2: Format Variations")

# Final summary
print(f"\n{'='*60}")
print(f"  FINAL COMPARISON")
print(f"{'='*60}")
print("  Test Set                     Accuracy        Format")
print("  " + "─"*55)
print("  Original (fictional names)   55.2%           100%")
rw_acc = f"{r1['correct']}/{r1['total']} ({r1['correct']/r1['total']*100:.0f}%)"
rw_fmt = f"{r1['format_ok']}/{r1['total']}"
fv_acc = f"{r2['correct']}/{r2['total']} ({r2['correct']/r2['total']*100:.0f}%)"
fv_fmt = f"{r2['format_ok']}/{r2['total']}"
print(f"  Real-world names             {rw_acc:<16}{rw_fmt}")
print(f"  Format variations            {fv_acc:<16}{fv_fmt}")
print("=" * 60)
