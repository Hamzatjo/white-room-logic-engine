import requests
import json
import re
import os
import sys
import time

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("MODEL", "gemini-3-flash-preview")
API_URL = os.environ.get("API_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")

BASE_DIR = "/Users/talon/.openclaw/workspace/reasoning-data"
JSONL_PATH = os.environ.get("JSONL_PATH", os.path.join(BASE_DIR, "reasoning-train-gemini.jsonl"))
QUEUE_PATH = os.path.join(BASE_DIR, "generation_queue.jsonl")

EXAMPLE = """<document>
In the region of Quelvara, there are three substances: Drennix, Fossik, and Yulthari. The following rules govern their interactions:

- If Drennix is combined with Fossik, the result is always Molvane.
- If Molvane is present in a container, the container becomes unstable.
- Unstable containers must be stored in a Grelloch vault.
- Yulthari glows faintly when exposed to temperatures above 400 Prells, but this has no effect on any other substance or container.

A researcher places Drennix and Fossik into Container 7.
</document>
<question>
Where must Container 7 be stored?
</question>
<reasoning>
Step 1: The researcher places Drennix and Fossik into Container 7. According to the first rule, combining Drennix with Fossik always produces Molvane. Therefore, Molvane is now present in Container 7.

Step 2: The second rule states that if Molvane is present in a container, that container becomes unstable. Since Container 7 now contains Molvane, Container 7 is unstable.

Step 3: The third rule states that unstable containers must be stored in a Grelloch vault. Since Container 7 is unstable, it must be stored in a Grelloch vault.

Note: The information about Yulthari glowing at temperatures above 400 Prells is irrelevant to this question.
</reasoning>
<answer>
A Grelloch vault.
</answer>"""

# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(config):
    puzzle_type = config['puzzle_type']
    steps = config['steps']
    has_distractor = config['has_distractor']
    solvable = config['solvable']
    entity_names = config['entity_names']
    unit_names = config['unit_names']
    subtype = config['jitter']['subtype']
    verb_theme = config['jitter']['verb_theme']
    operator_constraint = config['jitter']['operator_constraint']

    type_descriptions = {
        'conditional_logic': 'Conditional Logic puzzle (If X then Y chains, rule-based deduction)',
        'relational_mapping': 'Relational Mapping puzzle (rankings, seating arrangements, who is next to whom)',
        'temporal_ordering': 'Temporal Ordering puzzle (events in chronological sequence)',
        'set_theory': 'Set Theory puzzle (overlapping groups, inclusion-exclusion)',
        'elimination_grid': 'Elimination Grid puzzle (match entities to attributes using clues)',
    }

    type_desc = type_descriptions[puzzle_type]

    distractor_instruction = (
        "Include one piece of irrelevant red-herring information that does NOT affect the solution."
        if has_distractor else
        "Do NOT include any distractors or irrelevant information."
    )

    if not solvable:
        solvable_instruction = (
            "This puzzle must be UNSOLVABLE. Deliberately omit a key premise so the question "
            "cannot be answered. The <reasoning> tag must show the solver recognizing the missing "
            "information. The <answer> must be: \"Cannot be determined.\" followed by a brief "
            "explanation of what's missing."
        )
    else:
        solvable_instruction = (
            "This puzzle MUST have exactly ONE unique solution. Build the answer first, "
            "then construct constraints backward to guarantee uniqueness."
        )

    names_str = ", ".join(entity_names)
    units_str = ", ".join(unit_names)

    return f"""Generate exactly ONE {type_desc}.

CRITICAL STRUCTURAL CONSTRAINTS:
- Puzzle Sub-Type: {subtype}
- Relational Theme: The puzzle logic should involve entities that act upon each other using concepts similar to: "{verb_theme}".
- Operator Constraint: {operator_constraint}

RULES:
1. YOU MUST USE THESE EXACT ENTITY NAMES: [{names_str}]. Do not invent any other nouns for people, places, or objects. Use these names for all entities in the puzzle.
2. USE THESE FICTIONAL UNITS where measurements are needed: [{units_str}]. NO real-world units.
3. The puzzle requires exactly {steps} steps of deduction to solve.
4. {distractor_instruction}
5. {solvable_instruction}
6. The <reasoning> tag must use structured natural language: "Step 1: The document states... Step 2: Since..."
7. The <answer> must be extremely concise: single entity, short list, or Yes/No.
8. Do NOT include any meta-commentary, design notes, or self-reflection in the <reasoning> tag. Only include the step-by-step deduction a solver would perform.

Here is an example of the EXACT format required:

{EXAMPLE}

Now generate exactly ONE puzzle using the provided entity names and structural constraints. Output ONLY the XML tags (<document>, <question>, <reasoning>, <answer>). No commentary, no numbering, no markdown."""

# ============================================================
# API CALL
# ============================================================

def call_llm(prompt):
    """Hits the primary API with the appropriate formatting."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    system_instruction = "You are a synthetic data generator. You output ONLY XML-formatted logic puzzles. You never use real-world names or concepts. You follow structural constraints exactly.\n\n"
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": system_instruction + prompt}
        ],
        "temperature": 0.95,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=(10, 45))
    except requests.exceptions.Timeout:
        print(f"  Timeout on primary API")
        time.sleep(5) # Cooldown after timeout
        return None
    except Exception as e:
        print(f"  Connection error: {e}")
        time.sleep(5)
        return None

    if resp.status_code != 200:
        print(f"  API error {resp.status_code}: {resp.text[:300]}")
        # ALWAYS force a severe backoff on 422 or 429 to prevent spamming
        if resp.status_code in (422, 429, 503):
            print("  Forcing 15 second backoff to respect rate limits...")
            time.sleep(15)
        else:
            time.sleep(5)
        return None

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  Malformed response JSON: {e}")
        print(f"  Raw response keys: {list(data.keys())}")
        if "error" in data:
            print(f"  Error details: {data['error']}")
        return None

# ============================================================
# PARSER + VALIDATOR
# ============================================================

def parse_single_sample(text, sample_id, puzzle_type, steps, has_distractor, solvable):
    text = re.sub(r'```xml\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    doc_match = re.search(r'<document>\n?(.*?)\n?</document>', text, re.DOTALL)
    q_match = re.search(r'<question>\n?(.*?)\n?</question>', text, re.DOTALL)
    t_match = re.search(r'<reasoning>\n?(.*?)\n?</reasoning>', text, re.DOTALL)
    a_match = re.search(r'<answer>\n?(.*?)(?:\n?</answer>|$)', text, re.DOTALL)

    if not all([doc_match, q_match, t_match, a_match]):
        print(f"  PARSE FAIL: missing tags")
        return None

    document = doc_match.group(1).strip()
    question = q_match.group(1).strip()
    think = t_match.group(1).strip()
    answer = a_match.group(1).strip()

    # Reject empty content
    if not document or document == '...' or not question or not answer or answer == '...':
        print(f"  QUALITY FAIL: empty content")
        return None

    # Reject real-world terms
    real_world = re.findall(
        r'\b(Alice|Bob|Charlie|David|Eve|red|blue|green|yellow|purple|Monday|Tuesday|'
        r'Wednesday|January|February|March|cat|dog|apple|orange|New York|London|Paris|'
        r'meter|kilogram|second|hour|minute|dollar|euro|pound)\b',
        document, re.IGNORECASE
    )
    if real_world:
        print(f"  QUALITY FAIL: real-world terms: {real_world[:5]}")
        return None

    # Reject meta-commentary in think tag
    meta_patterns = [
        r'I think it', r'we should ensure', r'output only', r'it might be',
        r'Now,? output', r'we have \d+ steps', r'let me (fix|revise|redesign)',
        r'this puzzle (is|has|needs)', r'acceptable', r'straightforward',
        r'the (only )?way for the puzzle', r'unique solution', r'for the puzzle to',
        r'the puzzle (requires|asks|states|implies|assumes)',
        r'as (a |the )?puzzle', r'puzzle design',
    ]
    for pattern in meta_patterns:
        if re.search(pattern, think, re.IGNORECASE):
            print(f"  QUALITY FAIL: meta-commentary in think: '{pattern}'")
            return None

    # Reject self-doubt in solvable puzzles
    if solvable:
        doubt_patterns = [
            r'wait\s*[-—]', r'let me re(?!ad)', r'contradiction',
            r'no valid solution', r'puzzle is ambiguous',
            r'both .* (are|seem) valid', r'I need to fix', r'hmm',
        ]
        for pattern in doubt_patterns:
            if re.search(pattern, think, re.IGNORECASE):
                print(f"  QUALITY FAIL: self-doubt in think: '{pattern}'")
                return None

    full_text = (
        f"<document>\n{document}\n</document>\n"
        f"<question>\n{question}\n</question>\n"
        f"<reasoning>\n{think}\n</reasoning>\n"
        f"<answer>\n{answer}\n</answer>"
    )

    return {
        'id': sample_id,
        'type': puzzle_type,
        'steps': steps,
        'has_distractor': has_distractor,
        'solvable': solvable,
        'document': document,
        'question': question,
        'think': think,
        'answer': answer,
    }

import fcntl

def consume_next_from_queue():
    """Safely reads the first item from the queue and immediately removes it using file locks."""
    if not os.path.exists(QUEUE_PATH):
        return None
        
    with open(QUEUE_PATH, 'r+') as f:
        # Acquire an exclusive lock
        fcntl.flock(f, fcntl.LOCK_EX)
        
        try:
            lines = f.readlines()
            if not lines:
                return None
                
            # Pop the first line
            target_line = lines[0]
            config = json.loads(target_line)
            
            # Rewrite the file without the first line
            f.seek(0)
            f.writelines(lines[1:])
            f.truncate()
            
            return config
        finally:
            # Always release the lock
            fcntl.flock(f, fcntl.LOCK_UN)

def return_to_queue(config):
    """Safely appends a failed config back to the end of the queue."""
    with open(QUEUE_PATH, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(config, ensure_ascii=False) + '\n')
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

# ============================================================
# MAIN
# ============================================================

def main():
    if not API_KEY:
        print("Error: API_KEY not set")
        sys.exit(1)

    print(f"Model: {MODEL}")

    total_generated = 0
    total_failed = 0
    
    # Keep consuming until the queue runs dry
    while True:
        config = consume_next_from_queue()
        
        if not config:
            print("Queue is empty. Exiting.")
            break
        
        sid = config['id']
        puzzle_type = config['puzzle_type']
        steps = config['steps']
        has_distractor = config['has_distractor']
        solvable = config['solvable']

        label = (
            f"[{sid}] {puzzle_type} ({steps}-step, "
            f"{'distractor' if has_distractor else 'clean'}, "
            f"{'solvable' if solvable else 'UNSOLVABLE'})"
        )
        print(f"\n  {label}")

        prompt = build_prompt(config)

        # Retry up to 3 times
        sample = None
        for attempt in range(3):
            if attempt > 0:
                print(f"    Retry {attempt}...")

            raw = call_llm(prompt)
            if not raw:
                continue

            sample = parse_single_sample(raw, sid, puzzle_type, steps, has_distractor, solvable)
            if sample:
                break

            debug_path = os.path.join(BASE_DIR, f"debug-{sid}-attempt{attempt}.txt")
            with open(debug_path, 'w') as f:
                f.write(raw)

        if not sample:
            print(f"    FAILED after retries (Moving config to end of queue)")
            total_failed += 1
            # Push failed config back to queue
            return_to_queue(config)
            continue

        # Append to JSONL safely
        with open(JSONL_PATH, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

        total_generated += 1
        print(f"    ✓ OK (Saved to {os.path.basename(JSONL_PATH)})")

        import random
        time.sleep(random.randint(2, 5))

    print(f"\n{'='*50}")
    print(f"Session Complete.")
    print(f"Successfully Generated: {total_generated}")
    print(f"Failed Generations: {total_failed}")
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, 'r') as f:
            print(f"Total in Dataset: {sum(1 for _ in f)}")

if __name__ == "__main__":
    main()
