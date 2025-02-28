import os
import json
import logging
from sklearn.model_selection import train_test_split

# Define paths
TOKENIZED_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/tokenized_dataset.json"
TRAIN_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json"
TEST_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json"

os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def prepare_datasets():
    """Load, split, and save train-test datasets only once."""
    if os.path.exists(TRAIN_DATA_PATH) and os.path.exists(TEST_DATA_PATH):
        logging.info("✅ Train-Test datasets already exist. Skipping dataset split.")
        return  # Skip re-processing

    # Load full dataset
    logging.info("🔄 Loading tokenized dataset...")
    with open(TOKENIZED_DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Ensure dataset is shuffled and split
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    # Save train dataset
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    # Save test dataset
    with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    logging.info(f"✅ Dataset split complete: {len(train_data)} train, {len(test_data)} test")


if __name__ == "__main__":
    prepare_datasets()
