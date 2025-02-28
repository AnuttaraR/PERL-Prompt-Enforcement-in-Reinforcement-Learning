import json
import os
import torch
from transformers import AutoTokenizer

# Define paths
INPUT_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/merged_dataset.json"
OUTPUT_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/tokenized_dataset.json"

# Load tokenizer
TOKENIZER_MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


def tokenize_dataset():
    """Loads the dataset, tokenizes text, and saves it with tokenized fields."""
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Dataset file not found: {INPUT_DATA_PATH}")
        return

    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenized_data = []

    for entry in dataset:
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        question_type = entry["question_type"]

        # Tokenize question
        encoded = tokenizer(question, truncation=True, padding="max_length", max_length=512)

        # Tokenize ground truth
        encoded_gt = tokenizer(ground_truth, truncation=True, padding="max_length", max_length=512)

        tokenized_entry = {
            "question": question,
            "ground_truth": ground_truth,  # Keep original text for evaluation
            "question_type": question_type,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "gt_input_ids": encoded_gt["input_ids"],  # Tokenized ground truth
            "gt_attention_mask": encoded_gt["attention_mask"],
        }

        tokenized_data.append(tokenized_entry)

    # Save tokenized dataset
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenized_data, f, indent=4)

    print(f"Tokenized dataset saved: {OUTPUT_DATA_PATH} ({len(tokenized_data)} entries)")


if __name__ == "__main__":
    tokenize_dataset()
