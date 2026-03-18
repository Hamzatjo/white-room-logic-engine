import json
import os
import glob
import random

def main():
    # Find all reasoning data files
    files = glob.glob("*.jsonl")
    data_files = [f for f in files if "generation_queue" not in f and "dataset" not in f]
    
    # Include the locked file if it exists
    if os.path.exists("dataset_500_locked.jsonl"):
        data_files.append("dataset_500_locked.jsonl")
        
    print(f"Found {len(data_files)} data files: {data_files}")
    
    formatted_data = []
    
    for file in data_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                # Format into standard conversational format (ShareGPT / ChatML style)
                # User turn: The document and the question
                user_content = (
                    "Solve the following logic puzzle. You must provide step-by-step reasoning "
                    "in a <reasoning> tag before providing your final answer in an <answer> tag.\n\n"
                    f"<document>\n{data['document']}\n</document>\n\n"
                    f"<question>\n{data['question']}\n</question>"
                )
                
                # Assistant turn: The reasoning and the answer
                assistant_content = (
                    f"<reasoning>\n{data['think']}\n</reasoning>\n"
                    f"<answer>\n{data['answer']}\n</answer>"
                )
                
                formatted_item = {
                    "conversations": [
                        {"from": "user", "value": user_content},
                        {"from": "assistant", "value": assistant_content}
                    ],
                    "metadata": {
                        "id": data.get("id"),
                        "type": data.get("type"),
                        "steps": data.get("steps"),
                        "solvable": data.get("solvable")
                    }
                }
                
                formatted_data.append(formatted_item)
                
    # Shuffle the dataset to mix puzzle types
    random.shuffle(formatted_data)
    
    # Calculate splits (90% Train, 10% Test)
    total = len(formatted_data)
    split_idx = int(total * 0.9)
    train_data = formatted_data[:split_idx]
    test_data = formatted_data[split_idx:]
    
    print(f"\nExtracted {total} valid puzzles.")
    print(f"Train split: {len(train_data)} examples")
    print(f"Test split: {len(test_data)} examples")
    
    # Save to disk
    with open("train_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("test_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
            
    print("\nSaved `train_dataset.jsonl` and `test_dataset.jsonl` in ShareGPT conversational format.")
    print("These files are perfectly formatted for HuggingFace SFTTrainer or Unsloth QLoRA!")

if __name__ == "__main__":
    main()
