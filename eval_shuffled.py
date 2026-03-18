#!/usr/bin/env python3
"""Test with shuffled rule order to see if model traces logic or just reads top-to-bottom."""

import json
import re
import os
import random

os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

random.seed(42)

puzzles = [
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

A Threvnok is measured at 80 Dralvens.
</document>

<question>
What is the final state of the Keltharn array?
</question>""",
        "rules": [
            "- Any Threvnok measured above 50 Dralvens will activate all connected Broskel units.",
            "- All Broskel units are classified as a type of Grendar device.",
            "- Every Grendar device is part of the Keltharn array.",
            "- The activation of any Grendar device triggers a Velmox pulse.",
            "- A Velmox pulse deactivates the entire Keltharn array.",
            "- Threvnok frequencies below 200 Jolvens produce a slight vibration, which has no effect on any system.",
        ],
        "answer": "The Keltharn array is deactivated."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

Patient Thompson has a temperature of 39.5°C. Patient Thompson weighs 82 kg, which is not relevant to fever protocols.
</document>

<question>
What is Patient Thompson's current status?
</question>""",
        "rules": [
            "- If a patient's temperature exceeds 38.5°C, the nurse must administer antipyretics.",
            "- Administering antipyretics requires approval from the attending physician.",
            "- Physician approval automatically triggers a chart update.",
            "- A chart update flags the patient for a 4-hour observation window.",
        ],
        "answer": "Patient Thompson is flagged for a 4-hour observation window."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

The current Draxil reading is 150 Kelthons. The facility was built in 2018, which has no effect on protocols.
</document>

<question>
What happens to the Frenvik containment field?
</question>""",
        "rules": [
            "- A Draxil reading above 100 Kelthons triggers a Brenvol cascade.",
            "- A Brenvol cascade forces the Spelvik regulators offline.",
            "- When Spelvik regulators go offline, the Frenvik containment field collapses.",
            "- A collapsed Frenvik containment field releases all stored Garthex particles.",
            "- Garthex particles are harmless in concentrations below 500 units.",
        ],
        "answer": "The Frenvik containment field collapses."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

Server Bravo's CPU usage is at 45%. Server Bravo handles 10,000 requests per second, which does not affect scaling rules.
</document>

<question>
Does Server Bravo get additional resources?
</question>""",
        "rules": [
            "- If a server's CPU usage exceeds 80%, the load balancer adds a replica.",
            "- Adding a replica requires approval from the infrastructure team.",
            "- If infrastructure team does not respond within 10 minutes, auto-scaling activates.",
            "- Auto-scaling provisions a new instance from the cloud pool.",
        ],
        "answer": "No. Server Bravo's CPU usage of 45% does not exceed the 80% threshold, so no scaling is triggered."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

The Vorskil emission level is 300 Prethals. Channel 4 has unlimited capacity. The weather is stormy, which has no effect on the system.
</document>

<question>
Through which channel are Drelvon transmissions routed?
</question>""",
        "rules": [
            "- Vorskil emissions above 200 Prethals cause Thrannek nodes to go offline.",
            "- Thrannek nodes going offline causes the Gelvix relay to lose signal.",
            "- Gelvix relay signal loss activates the Brenvox backup system.",
            "- The Brenvox backup system reroutes all Drelvon transmissions through Channel 4.",
        ],
        "answer": "Channel 4."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

The oak tree on Main Street is 20 meters tall and stands next to a power line. The tree was planted in 1965, which does not affect any regulations.
</document>

<question>
What immediate action is required?
</question>""",
        "rules": [
            "- Trees taller than 15 meters near power lines must be trimmed.",
            "- Trimming requires a permit from the city council.",
            "- Permits take 2 weeks to process.",
            "- During permit processing, a warning sign must be placed near the tree.",
        ],
        "answer": "A permit must be requested and a warning sign must be placed near the tree."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

The Brenthar crystal is vibrating at 75 Kernols. The crystal was mined from Sector 9, which has no bearing on resonance protocols.
</document>

<question>
What is the state of the Threlvak shield?
</question>""",
        "rules": [
            "- Any Brenthar crystal vibrating above 60 Kernols enters resonance mode.",
            "- A crystal in resonance mode emits Prelvax waves.",
            "- Prelvax waves destabilize all nearby Gormund barriers.",
            "- Destabilized Gormund barriers cause the Threlvak shield to engage.",
            "- Threlvak shield engagement consumes 100 Kelvax of energy per hour.",
        ],
        "answer": "The Threlvak shield is engaged."
    },
    {
        "prompt_template": """Solve the following logic puzzle. You must provide step-by-step reasoning in a <reasoning> tag before providing your final answer in an <answer> tag.

<document>
{rules}

Invoice #7712 is for $25,000 from Delta Systems. The Compliance team completed their review. Legal has been on holiday for 2 weeks and has not responded.
</document>

<question>
What happens to Invoice #7712?
</question>""",
        "rules": [
            "- Any invoice over $10,000 must be reviewed by the Compliance department.",
            "- Compliance-approved invoices require sign-off from the Legal team.",
            "- If Legal does not respond within 7 business days, the invoice is escalated to the CFO.",
            "- CFO escalation adds a 5% surcharge to the invoice total.",
        ],
        "answer": "Invoice #7712 is escalated to the CFO with a 5% surcharge."
    },
]

# Generate two versions of each: ordered and shuffled
ordered_data = []
shuffled_data = []

for p in puzzles:
    # Ordered version
    ordered_rules = "\n".join(p["rules"])
    ordered_prompt = p["prompt_template"].format(rules=ordered_rules)
    ordered_data.append({
        "conversations": [
            {"from": "human", "value": ordered_prompt},
            {"from": "gpt", "value": f"<answer>\n{p['answer']}\n</answer>"}
        ]
    })
    
    # Shuffled version
    shuffled_rules_list = p["rules"].copy()
    random.shuffle(shuffled_rules_list)
    shuffled_rules = "\n".join(shuffled_rules_list)
    shuffled_prompt = p["prompt_template"].format(rules=shuffled_rules)
    shuffled_data.append({
        "conversations": [
            {"from": "human", "value": shuffled_prompt},
            {"from": "gpt", "value": f"<answer>\n{p['answer']}\n</answer>"}
        ]
    })
    
    print(f"Puzzle: {p['answer'][:50]}")
    print(f"  Ordered:  {[r[:30] for r in p['rules']]}")
    print(f"  Shuffled: {[r[:30] for r in shuffled_rules_list]}")
    print()

with open("test_ordered.jsonl", "w") as f:
    for item in ordered_data:
        f.write(json.dumps(item) + "\n")

with open("test_shuffled.jsonl", "w") as f:
    for item in shuffled_data:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(ordered_data)} ordered + {len(shuffled_data)} shuffled puzzles")

# Now evaluate both
from mlx_lm import load, generate

print("\nLoading Logic Engine...")
model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)

def eval_set(data, label):
    correct = 0
    for i, item in enumerate(data):
        prompt = item["conversations"][0]["value"]
        expected = item["conversations"][1]["value"]
        exp_answer = re.search(r"<answer>(.*?)</answer>", expected, re.DOTALL).group(1).strip().lower()
        
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False
        )
        response = generate(model, tokenizer, prompt=chat_prompt, max_tokens=512, verbose=False)
        
        got_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        got = got_match.group(1).strip().lower() if got_match else response.strip().lower()
        
        exp_norm = re.sub(r'[^a-z0-9\s]', '', exp_answer)
        got_norm = re.sub(r'[^a-z0-9\s]', '', got)
        
        match = (exp_norm in got_norm or got_norm in exp_norm or
                 (len(set(exp_norm.split()) & set(got_norm.split())) / max(len(exp_norm.split()), 1) >= 0.5))
        
        if match:
            correct += 1
        
        status = "[MATCH]" if match else "[MISMATCH]"
        print(f"  {label} #{i+1}: {status} | Expected: {exp_answer[:50]} | Got: {got[:50]}")
    
    print(f"\n  {label} SCORE: {correct}/{len(data)} ({correct/len(data)*100:.0f}%)\n")
    return correct, len(data)

print("\n" + "="*60)
o_c, o_t = eval_set(ordered_data, "ORDERED")
print("="*60)
s_c, s_t = eval_set(shuffled_data, "SHUFFLED")
print("="*60)
print(f"\n  ORDERED:  {o_c}/{o_t} ({o_c/o_t*100:.0f}%)")
print(f"  SHUFFLED: {s_c}/{s_t} ({s_c/s_t*100:.0f}%)")
drop = (o_c - s_c) / max(o_c, 1) * 100
print(f"  DROP:     {drop:.0f}%")
print("="*60)
