import os
import json

# Define paths to the augmented datasets
DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data"
FILES = [
    os.path.join(DATA_PATH, "what_questions_augmented.json"),
    os.path.join(DATA_PATH, "how_questions_augmented.json"),
    os.path.join(DATA_PATH, "if_can_questions_augmented.json")
]

OUTPUT_FILE = os.path.join(DATA_PATH, "merged_dataset.json")


def merge_datasets():
    """Merge all augmented datasets into one JSON file."""
    merged_data = []

    for file in FILES:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged_data.extend(data)  # Append data to merged dataset
        else:
            print(f"Warning: {file} not found!")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged dataset saved as: {OUTPUT_FILE} ({len(merged_data)} samples)")


if __name__ == "__main__":
    merge_datasets()
