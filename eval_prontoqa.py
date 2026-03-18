#!/usr/bin/env python3
"""ProntoQA Benchmark: evaluate fine-tuned Logic Engine on standard deductive reasoning benchmark."""

import json
import re
import os
import time
import glob

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

from mlx_lm import load, generate

# Load model
print("Loading Logic Engine...")
model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)
print("Model loaded!\n")

# Load ProntoQA data — use ProofsOnly across different hop counts
test_files = {
    "1-hop": "/tmp/prontoqa/generated_data/1hop_ProofsOnly_random_noadj.json",
    "2-hop": "/tmp/prontoqa/generated_data/2hop_ProofsOnly_random_noadj.json",
    "3-hop": "/tmp/prontoqa/generated_data/3hop_ProofsOnly_random_noadj.json",
    "4-hop": "/tmp/prontoqa/generated_data/4hop_ProofsOnly_random_noadj.json",
    "5-hop": "/tmp/prontoqa/generated_data/1hop_ProofsOnly_5testhops_random_noadj.json",
}

all_results = {}

for hop_label, filepath in test_files.items():
    with open(filepath) as f:
        data = json.load(f)
    
    examples = list(data.values())
    # Extract test examples
    puzzles = []
    for ex in examples:
        test = ex["test_example"]
        # The query is "Prove: X" or "Disprove: X"
        query = test["query"]
        question = test["question"]
        chain = test["chain_of_thought"]
        
        # The final step in chain_of_thought is the conclusion
        conclusion = chain[-1] if chain else ""
        
        # Determine if it's True or False based on query
        if query.startswith("Prove:"):
            expected_bool = "true"
            claim = query.replace("Prove: ", "")
        elif query.startswith("Disprove:"):
            expected_bool = "false"
            claim = query.replace("Disprove: ", "")
        else:
            claim = query
            expected_bool = "true"
        
        puzzles.append({
            "question": question,
            "query": query,
            "claim": claim,
            "expected_bool": expected_bool,
            "conclusion": conclusion,
            "chain": chain
        })
    
    # Run evaluation (use first 50 per hop to keep it manageable)
    sample = puzzles[:50]
    correct = 0
    format_ok = 0
    
    print(f"\n{'='*60}")
    print(f"  ProntoQA {hop_label} ({len(sample)} examples)")
    print(f"{'='*60}")
    
    start = time.time()
    
    for i, p in enumerate(sample):
        # Format as our puzzle style
        prompt = f"""Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{p['question']}
</document>

<question>
{p['query']} Is this true or false?
</question>"""

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False
        )
        
        response = generate(model, tokenizer, prompt=chat_prompt, max_tokens=512, verbose=False)
        
        # Check format
        if "<reasoning>" in response and "<answer>" in response:
            format_ok += 1
        
        # Extract answer and check
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        got = answer_match.group(1).strip().lower() if answer_match else response.strip().lower()
        
        # Determine if model said true/false
        got_bool = None
        if "true" in got and "false" not in got:
            got_bool = "true"
        elif "false" in got and "true" not in got:
            got_bool = "false"
        elif "not" in got or "disprove" in got:
            got_bool = "false"
        else:
            # Check if conclusion matches
            conclusion_norm = p["conclusion"].lower().strip().rstrip(".")
            if conclusion_norm in got:
                got_bool = p["expected_bool"]
        
        is_correct = got_bool == p["expected_bool"]
        if is_correct:
            correct += 1
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            pct = correct / (i + 1) * 100
            print(f"  [{i+1}/{len(sample)}] Accuracy: {pct:.0f}%")
    
    elapsed = time.time() - start
    pct = correct / len(sample) * 100
    fmt_pct = format_ok / len(sample) * 100
    
    print(f"\n  {hop_label} RESULT: {correct}/{len(sample)} ({pct:.0f}%)")
    print(f"  Format compliance: {format_ok}/{len(sample)} ({fmt_pct:.0f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/len(sample):.1f}s/example)")
    
    all_results[hop_label] = {
        "correct": correct,
        "total": len(sample),
        "accuracy": pct,
        "format_ok": format_ok
    }

# Final summary
print(f"\n{'='*60}")
print(f"  ProntoQA BENCHMARK SUMMARY")
print(f"{'='*60}")
print(f"  {'Hops':<10} {'Accuracy':<15} {'Format':<15}")
print(f"  {'─'*40}")
for hop, r in all_results.items():
    print(f"  {hop:<10} {r['correct']}/{r['total']} ({r['accuracy']:.0f}%)      {r['format_ok']}/{r['total']}")

total_correct = sum(r["correct"] for r in all_results.values())
total_n = sum(r["total"] for r in all_results.values())
print(f"  {'─'*40}")
print(f"  {'OVERALL':<10} {total_correct}/{total_n} ({total_correct/total_n*100:.0f}%)")
print(f"{'='*60}")

with open("prontoqa_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nResults saved to prontoqa_results.json")
