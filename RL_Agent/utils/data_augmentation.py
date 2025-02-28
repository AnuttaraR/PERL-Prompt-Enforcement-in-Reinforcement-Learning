import os
import json
import random
import openai
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from deep_translator import GoogleTranslator
import os
import nltk

# Set correct path with normalized backslashes
NLTK_DATA_PATH = os.path.normpath("C:/Users/USER/nltk_data")

# Set the environment variable
os.environ["NLTK_DATA"] = NLTK_DATA_PATH

# Download the resource to the specified directory
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_PATH)

# Load OpenAI API key from environment (if using GPT-4)
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Define augmentation pipelines
synonym_aug = naw.SynonymAug(aug_src='wordnet')
random_deletion_aug = naw.RandomWordAug(action="delete", aug_p=0.1)
random_swap_aug = naw.RandomWordAug(action="swap", aug_p=0.2)
random_insert_aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased", action="insert")
typo_aug = nac.RandomCharAug(action="insert", aug_char_p=0.1)


# Function to apply **back translation**
def back_translate(text, source_lang="en", target_lang="fr"):
    """Translates text to target language and back to English for augmentation."""
    translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    back_translated = GoogleTranslator(source=target_lang, target=source_lang).translate(translated)
    return back_translated


# Function to apply **EDA Augmentations**
def augment_text(text):
    """Apply multiple augmentations (synonym replacement, deletion, swapping, insertion, back translation)."""
    augmented_texts = set()

    # List of augmenters to apply
    augmenters = [
        synonym_aug,
        random_deletion_aug,
        random_swap_aug,
        random_insert_aug,
        typo_aug
    ]

    for augmenter in augmenters:
        augmented = augmenter.augment(text)
        if isinstance(augmented, list):
            augmented_texts.update(augmented)  # Add each element from the list
        else:
            augmented_texts.add(augmented)

    # Apply back translation separately
    augmented_texts.add(back_translate(text))

    return list(augmented_texts)




# Function to augment dataset
def augment_dataset(input_file, target_count):
    """Augment dataset using multiple augmentation strategies."""
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    original_size = len(dataset)
    print(f"Original size of {input_file}: {original_size}")

    while len(dataset) < target_count:
        sample = random.choice(dataset)
        question = sample["question"]
        answer = sample["ground_truth"]

        # Generate augmentations for both question and answer
        augmented_questions = augment_text(question)
        augmented_answers = augment_text(answer)

        # Ensure uniqueness
        for new_question, new_answer in zip(augmented_questions, augmented_answers):
            if len(dataset) >= target_count:
                break
            if not any(entry["question"] == new_question for entry in dataset):
                dataset.append({
                    "question": new_question,
                    "ground_truth": new_answer,
                    "question_type": sample["question_type"]
                })

    # Save augmented dataset
    output_file = input_file.replace(".json", "_augmented.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"Augmented dataset saved: {output_file} ({len(dataset)} total entries)")


if __name__ == "__main__":
    # Set target dataset size
    TARGET_SIZE = 200

    # Augment each category
    augment_dataset("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/what_questions.json", TARGET_SIZE)
    augment_dataset("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/how_questions.json", TARGET_SIZE)
    augment_dataset("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/if_can_questions.json", TARGET_SIZE)
