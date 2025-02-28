import json
import torch
from transformers import AutoTokenizer

DATASET_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/merged_dataset.json"
TOKENIZED_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/tokenized_dataset.json"
TOKENIZER_MODEL = "bert-base-uncased"


def preprocess_data():
    """Tokenize and preprocess dataset."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed_data = []
    for entry in dataset:
        tokenized = tokenizer(entry["question"], entry["ground_truth"], truncation=True, padding="max_length",
                              max_length=512)
        processed_data.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "question_type": entry["question_type"],
            "ground_truth": entry["ground_truth"]
        })

    with open(TOKENIZED_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    print(f"Tokenized dataset saved as: {TOKENIZED_PATH} ({len(processed_data)} samples)")


if __name__ == "__main__":
    preprocess_data()
