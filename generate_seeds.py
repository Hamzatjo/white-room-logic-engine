import argparse
import json
import os
import random
import re

BASE_DIR = "/Users/talon/.openclaw/workspace/reasoning-data"
NAMES_PATH = os.path.join(BASE_DIR, "used-names.txt")
QUEUE_PATH = os.path.join(BASE_DIR, "generation_queue.jsonl")
TRAIN_JSONL_PATH = os.environ.get("JSONL_PATH", os.path.join(BASE_DIR, "reasoning-train.jsonl"))

# ============================================================
# NAME GENERATOR
# ============================================================

ONSETS = [
    'b', 'br', 'd', 'dr', 'f', 'fl', 'fr', 'g', 'gl', 'gr',
    'h', 'j', 'k', 'kr', 'l', 'm', 'n', 'p', 'pr',
    'qu', 'r', 's', 'sh', 'sk', 'sl', 'sp', 'st',
    't', 'th', 'tr', 'v', 'w', 'z'
]
NUCLEI = ['a', 'e', 'i', 'o', 'u', 'ae', 'ei', 'ou', 'al', 'el', 'ol', 'ar', 'or', 'en', 'an']
CODAS = ['', 'n', 'x', 'k', 'th', 'r', 's', 'l', 'v']

def generate_syllable():
    return random.choice(ONSETS) + random.choice(NUCLEI) + random.choice(CODAS)

def generate_name(used_names, min_syllables=2, max_syllables=3):
    for _ in range(500):
        num_syllables = random.randint(min_syllables, max_syllables)
        name = ''.join(generate_syllable() for _ in range(num_syllables))
        name = name.capitalize()
        if 4 <= len(name) <= 10 and name not in used_names and name.lower() not in {'the', 'and', 'for', 'not', 'but', 'are', 'was', 'has', 'had'}:
            return name
    raise RuntimeError("Could not generate unique name after 500 attempts")

def generate_unit(used_names):
    suffixes = ['s', 'les', 'ns', 'ks', 'ms', 'rs', 'ts']
    for _ in range(200):
        base = generate_syllable() + generate_syllable()
        unit = base.capitalize() + random.choice(suffixes)
        if unit not in used_names:
            return unit
    raise RuntimeError("Could not generate unique unit")

def generate_name_set(used_names, count):
    names = []
    for _ in range(count):
        name = generate_name(used_names)
        names.append(name)
        used_names.add(name)
    return names

# ============================================================
# STRUCTURAL JITTER
# ============================================================

def get_structural_jitter(puzzle_type):
    sub_types = {
        'conditional_logic': [
            'Forward Cascading Chain (A triggers B triggers C)',
            'Branching System States (one input leads to multiple possible outputs based on conditions)',
            'Mutually Exclusive Triggers (if A happens, B cannot; if B happens, C must)',
            'Cyclic/Looping Rules (output feeds back as input under certain conditions)',
            'Threshold-Based Activation (effects only trigger when a value exceeds a limit)',
        ],
        'relational_mapping': [
            'Linear Ranking (1st to Nth, strict ordering)',
            'Spatial/Seating Arrangement (Left/Right/Opposite in a row)',
            'Circular Seating (around a round table)',
            'Hierarchical Tree (superior/subordinate relationships)',
            'Network Connections (who is connected to whom, with constraints)',
        ],
        'temporal_ordering': [
            'Strict Consecutive Sequence (no gaps, events fill all slots)',
            'Relative Ordering with Unknown Gaps (before/after but not necessarily adjacent)',
            'Anchored Events with Relative Placements (some events have fixed positions)',
            'Immediately-Before/After Constraints (consecutive pairs within a larger sequence)',
            'Parallel Timelines Merging (two sequences that share common events)',
        ],
        'set_theory': [
            'Two Overlapping Groups (inclusion-exclusion with two sets)',
            'Three Overlapping Groups (full inclusion-exclusion)',
            'Two Groups with a Large "Neither" Condition',
            'Subset/Superset Containment (one group is entirely within another)',
            'Exactly-One / Exactly-Two Counting (how many belong to specific overlap regions)',
        ],
        'elimination_grid': [
            '3x3 Attribute Matching (3 entities, 3 attributes each from 3 categories)',
            '4x4 Attribute Matching with Negative Clues (4 entities, clues say what is NOT true)',
            '3x3 with Mixed Positive and Negative Clues',
            '4x4 with Linked Attributes (the entity with attribute X must also have attribute Y)',
            '3x3 with Conditional Assignments (if entity has X, then it must have Y)',
        ],
    }

    verbs = [
        "orbits / is orbited by",
        "powers / drains",
        "outranks / submits to",
        "neutralizes / amplifies",
        "trades with / boycotts",
        "unlocks / seals",
        "transmits to / blocks",
        "fuels / extinguishes",
        "commands / obeys",
        "precedes / follows",
        "contains / is contained by",
        "activates / deactivates",
    ]

    operators = [
        "You MUST use the operator 'unless' at least once in a clue.",
        "You MUST use the construction 'neither... nor' in one of your clues.",
        "You MUST use the phrase 'if and only if' to establish a strict bidirectional rule.",
        "You MUST use a mutually exclusive constraint (e.g., 'exactly one of X or Y').",
        "You MUST use a conditional exception (e.g., 'all X are Y, except when Z').",
        "You MUST use an 'immediately before/after' or 'directly adjacent' constraint.",
        "Use standard logical phrasing (no forced operators).",
    ]

    return (
        random.choice(sub_types[puzzle_type]),
        random.choice(verbs),
        random.choice(operators),
    )

# ============================================================
# FILE I/O
# ============================================================

def load_used_names():
    names = set()
    if os.path.exists(NAMES_PATH):
        with open(NAMES_PATH, 'r') as f:
            names = set(line.strip() for line in f if line.strip())
    return names

def save_used_names(names):
    with open(NAMES_PATH, 'w') as f:
        f.write('\n'.join(sorted(names)))

def get_highest_id():
    """Find the highest H# ID across both the train set and the generation queue."""
    max_num = 0
    # Check the actual training data
    if os.path.exists(TRAIN_JSONL_PATH):
        with open(TRAIN_JSONL_PATH, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    sid = sample.get('id', '')
                    match = re.search(r'H(\d+)', sid)
                    if match:
                        max_num = max(max_num, int(match.group(1)))
                except json.JSONDecodeError:
                    continue
                    
    # Check the queue we're about to add to
    if os.path.exists(QUEUE_PATH):
        with open(QUEUE_PATH, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    sid = sample.get('id', '')
                    match = re.search(r'H(\d+)', sid)
                    if match:
                        max_num = max(max_num, int(match.group(1)))
                except json.JSONDecodeError:
                    continue
                    
    return max_num

# ============================================================
# MAIN
# ============================================================

def generate_queue(count):
    used_names = load_used_names()
    print(f"Loaded {len(used_names)} banned names")
    
    start_id_num = get_highest_id() + 1
    next_id = start_id_num
    
    types = ['conditional_logic', 'relational_mapping', 'temporal_ordering', 'set_theory', 'elimination_grid']
    step_configs = [3, 4, 3, 4, 5]
    distractor_configs = [True, False, True, False, True]

    entries_to_write = []
    
    print(f"Generating {count} puzzle configurations starting at ID H{next_id}...")

    while len(entries_to_write) < count:
        round_steps = step_configs.copy()
        round_distractors = distractor_configs.copy()
        random.shuffle(round_steps)
        random.shuffle(round_distractors)
        
        # Determine the unsolvable index for this "round" of 5
        # We roughly mock the old `round_num % 5` behavior to keep 20% unsolvable
        unsolvable_idx = (len(entries_to_write) // 5) % 5

        for i, puzzle_type in enumerate(types):
            if len(entries_to_write) >= count:
                break
                
            steps = round_steps[i]
            has_distractor = round_distractors[i]
            solvable = (i != unsolvable_idx)

            if steps == 3:
                num_entities = random.randint(3, 4)
            elif steps == 4:
                num_entities = random.randint(3, 5)
            else:
                num_entities = random.randint(4, 6)
            
            num_extras = random.randint(1, 3)
            entity_names = generate_name_set(used_names, num_entities + num_extras)
            
            unit_names = [generate_unit(used_names) for _ in range(2)]
            for u in unit_names:
                used_names.add(u)

            jitter = get_structural_jitter(puzzle_type)

            config = {
                "id": f"H{next_id}",
                "puzzle_type": puzzle_type,
                "steps": steps,
                "has_distractor": has_distractor,
                "solvable": solvable,
                "entity_names": entity_names,
                "unit_names": unit_names,
                "jitter": {
                    "subtype": jitter[0],
                    "verb_theme": jitter[1],
                    "operator_constraint": jitter[2]
                }
            }
            
            entries_to_write.append(config)
            next_id += 1

    # Append to queue
    with open(QUEUE_PATH, 'a') as f:
        for entry in entries_to_write:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    # Save the updated banned names list
    save_used_names(used_names)
    
    print(f"Appended {len(entries_to_write)} configurations to {QUEUE_PATH}")
    print(f"Total banned names saved: {len(used_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seed configurations for reasoning puzzles.")
    parser.add_argument("--count", type=int, default=50, help="Number of puzzle configurations to generate and append to the queue.")
    args = parser.parse_args()
    
    generate_queue(args.count)
