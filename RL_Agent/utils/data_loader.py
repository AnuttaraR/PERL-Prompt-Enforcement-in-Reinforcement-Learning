import json
import os


def load_dataset():
    """
    Load and merge datasets from separate JSON files into a single dataset.
    If a merged dataset already exists, load that instead of merging.
    """
    merged_dataset_path = "data/merged_dataset.json"

    # Check if merged dataset exists
    if os.path.exists(merged_dataset_path):
        print(f"Using existing merged dataset: {merged_dataset_path}")
        with open(merged_dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Otherwise, merge individual question-type datasets
    dataset = []
    json_files = [
        "data/what_questions.json",
        "data/how_questions.json",
        "data/if_can_questions.json"
    ]

    for file in json_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                dataset.extend(json.load(f))  # Merge all question types into one list
        else:
            print(f"arning: {file} not found.")

    # Save the merged dataset for future use
    with open(merged_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"Merged dataset saved with {len(dataset)} entries.")

    return dataset


if __name__ == "__main__":
    dataset = load_dataset()
    print("Sample Data:", dataset[:3])  # Print first 3 entries for verification
