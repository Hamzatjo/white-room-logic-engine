#!/usr/bin/env python3
"""Interactive chat with the fine-tuned Logic Engine (stateless, no thinking)"""

import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")

from mlx_lm import load, generate

print("Loading Logic Engine...")
model, tokenizer = load(
    "mlx-community/Qwen3.5-0.8B-4bit",
    adapter_path="./adapters"
)
print("Logic Engine ready! Type your puzzle (or 'quit' to exit)\n")

while True:
    try:
        lines = []
        prompt = input("You> ")
        if prompt.lower() in ("quit", "exit"):
            break

        print("  (multi-line: keep typing, empty line to send)")
        while True:
            line = input("...  ")
            if line == "":
                break
            lines.append(line)
        if lines:
            prompt = prompt + "\n" + "\n".join(lines)

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False
        )

        print("\nLogic Engine>")
        response = generate(
            model, tokenizer,
            prompt=chat_prompt,
            max_tokens=512,
            verbose=False
        )
        print(response)
        print()

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break
