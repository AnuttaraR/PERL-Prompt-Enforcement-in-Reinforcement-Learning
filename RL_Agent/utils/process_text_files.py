import os
import json
import re
import contractions

# Define the base folder for data
QUESTION_FOLDERS = {
    "what": "D:/Text Files - The Empoyer_s Question Answer Guide to Employee Benefits 2020 - Copy/What-Questions",
    "how": "D:/Text Files - The Empoyer_s Question Answer Guide to Employee Benefits 2020 - Copy/How-Questions",
    "if_can": "D:/Text Files - The Empoyer_s Question Answer Guide to Employee Benefits 2020 - Copy/If-Can-Questions"
}

# Define output data folder
OUTPUT_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data"


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Created missing directory: {directory}")


def clean_text(text):
    """Preprocess and clean text: remove extra spaces, special characters, expand contractions, and normalize."""
    text = text.replace('\n', ' ').strip()  # Remove line breaks and extra spaces
    text = contractions.fix(text)  # Expand contractions (e.g., "can't" → "cannot")
    text = re.sub(r'[^a-zA-Z0-9.,!?\-\s]', '', text)  # Keep only relevant characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.lower()  # Normalize case


def extract_question_and_answer(file_path):
    """Extract the question and ground truth from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        # Extract the question from the first line after the second '-'
        first_line = lines[0].strip()
        question_match = re.split(r' - ', first_line, maxsplit=2)
        question = question_match[-1].strip() if len(question_match) > 1 else first_line

        ground_truth = ' '.join(lines[1:]).strip()  # Remaining lines are ground truth

        return clean_text(question), clean_text(ground_truth)


def save_json(file_path, data):
    """Save data to a JSON file, preventing overwrites if data exists."""
    ensure_directory_exists(os.path.dirname(file_path))  # ✅ Ensure directory exists before saving

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        data = existing_data + data  # Append new data instead of overwriting

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved {len(data)} entries to {file_path}")


def process_text_files():
    """Read all text files from the three folders and convert them into JSON format."""
    categorized_data = {"what": [], "how": [], "if_can": []}
    unique_questions = set()

    ensure_directory_exists(OUTPUT_DATA_PATH)  # ✅ Ensure output folder exists

    for q_type, folder in QUESTION_FOLDERS.items():
        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}, skipping...")
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                question, ground_truth = extract_question_and_answer(file_path)

                # Ensure uniqueness
                if question not in unique_questions:
                    unique_questions.add(question)
                    categorized_data[q_type].append({
                        "question": question,
                        "ground_truth": ground_truth,
                        "question_type": q_type
                    })

    # Save data as separate JSON files without overwriting existing data
    for q_type, data in categorized_data.items():
        json_file = os.path.normpath(os.path.join(OUTPUT_DATA_PATH, f"{q_type}_questions.json"))
        save_json(json_file, data)


if __name__ == "__main__":
    process_text_files()
