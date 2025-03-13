import json
import os
import argparse
import logging
import random
import time
import numpy as np
from tqdm import tqdm
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import utility functions
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from RL_Agent.utils.retrieval import retrieve_context
from dpo_model import calculate_answer_scores, load_config, load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("preference_pairs_generator")


def sample_actions_weighted(action_space, question_type):
    """
    Sample actions with weights that favor specific actions over general ones.

    Args:
        action_space: Action space configuration
        question_type: Type of question ('what', 'how', or 'if_can')

    Returns:
        tuple: Two sampled actions (each a tuple of (id, description))
    """
    general_actions = action_space["general_actions"]
    specific_actions = action_space[f"{question_type}_question_actions"]

    # Create weighted distribution that favors specific actions
    general_weight = 0.2  # Lower weight for general/keep unchanged actions
    specific_weight = 0.8  # Higher weight for specific actions

    action_weights = []
    action_list = []

    # Add general actions with lower weight
    for k, v in general_actions.items():
        action_list.append((int(k), v))
        # Even lower weight for "keep unchanged" (action 0)
        if int(k) == 0:
            action_weights.append(general_weight * 0.5 / len(general_actions))
        else:
            action_weights.append(general_weight / len(general_actions))

    # Add specific actions with higher weight
    general_count = len(general_actions)
    for k, v in specific_actions.items():
        # Calculate the actual action ID by adding original ID to general_count
        action_id = int(k) + general_count - 1  # -1 because specific actions start at 1
        action_list.append((action_id, v))
        action_weights.append(specific_weight / len(specific_actions))

    # Normalize weights
    total_weight = sum(action_weights)
    action_weights = [w / total_weight for w in action_weights]

    # Sample two different actions based on weights
    indices = np.random.choice(len(action_list), 2, replace=False, p=action_weights)

    logger.debug(f"Sampled actions: {action_list[indices[0]]}, {action_list[indices[1]]}")

    return action_list[indices[0]], action_list[indices[1]]


def create_preference_pairs(dataset, action_space, num_pairs=1000, batch_size=16, output_path=None,
                            min_score_diff=0.1, max_unchanged_ratio=0.3):
    """
    Create preference pairs for DPO training by sampling different prompt actions
    and comparing their generated answers.

    Args:
        dataset: Training dataset
        action_space: Action space configuration
        num_pairs: Number of preference pairs to generate
        batch_size: Batch size for processing
        output_path: Path to save generated pairs (optional)
        min_score_diff: Minimum score difference to include a pair
        max_unchanged_ratio: Maximum ratio of pairs where "keep unchanged" is chosen

    Returns:
        preference_pairs: List of preference pair dictionaries
    """
    logger.info(f"Creating {num_pairs} preference pairs from dataset with improved sampling")
    logger.info(f"Min score difference: {min_score_diff}, Max 'unchanged' ratio: {max_unchanged_ratio}")
    start_time = time.time()
    preference_pairs = []

    question_types = ["what", "how", "if_can"]

    # Group questions by type
    questions_by_type = {qt: [q for q in dataset if q["question_type"] == qt] for qt in question_types}

    # Track chosen action counts
    action_counts = {"unchanged": 0, "specific": 0}
    max_unchanged_pairs = int(num_pairs * max_unchanged_ratio)

    # Create preference pairs for each question type
    pairs_per_type = num_pairs // len(question_types)
    for question_type in question_types:
        questions = questions_by_type[question_type]
        if not questions:
            logger.warning(f"No questions found for type: {question_type}")
            continue

        # Process in batches to improve efficiency
        pairs_created = 0
        attempts = 0
        max_attempts = pairs_per_type * 3  # Allow for some pairs to be skipped
        progress_bar = tqdm(total=pairs_per_type, desc=f"Creating {question_type} pairs")

        logger.info(f"Generating {pairs_per_type} pairs for {question_type} questions")

        while pairs_created < pairs_per_type and attempts < max_attempts:
            current_batch_size = min(batch_size, pairs_per_type - pairs_created)
            batch_pairs = []

            for _ in range(current_batch_size):
                # Select a random question
                question_data = random.choice(questions)
                question = question_data["question"]
                ground_truth = question_data["ground_truth"]

                # Sample two different actions using weighted sampling
                action1, action2 = sample_actions_weighted(action_space, question_type)

                # Apply both actions to get two different prompts
                prompt1_mod = f"{action1[1]}: {question}"
                prompt2_mod = f"{action2[1]}: {question}"

                batch_pairs.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "question_type": question_type,
                    "action1": action1,
                    "action2": action2,
                    "prompt1_mod": prompt1_mod,
                    "prompt2_mod": prompt2_mod
                })

            # Get modified prompts from LLM
            logger.info(f"Generating modified prompts for batch of {len(batch_pairs)} pairs")
            for pair in batch_pairs:
                modified_prompt1 = get_llm_response(pair["prompt1_mod"], model="gpt-3.5-turbo")
                modified_prompt2 = get_llm_response(pair["prompt2_mod"], model="gpt-3.5-turbo")

                # Handle invalid responses
                if modified_prompt1 in ["INVALID", "Invalid input", "Invalid response"]:
                    modified_prompt1 = pair["question"]
                if modified_prompt2 in ["INVALID", "Invalid input", "Invalid response"]:
                    modified_prompt2 = pair["question"]

                pair["modified_prompt1"] = modified_prompt1
                pair["modified_prompt2"] = modified_prompt2

            # Retrieve context for each question (once per question to save API calls)
            logger.info("Retrieving context for questions")
            questions_list = [pair["question"] for pair in batch_pairs]
            contexts = {}
            for q in set(questions_list):
                contexts[q] = retrieve_context(q, top_k=3)

            # Generate answers using the modified prompts
            logger.info("Generating answers for modified prompts")
            valid_pairs = []
            for pair in batch_pairs:
                attempts += 1
                context = contexts[pair["question"]]

                # Generate answers using both prompts
                final_prompt1 = f"Question: {pair['modified_prompt1']}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."
                final_prompt2 = f"Question: {pair['modified_prompt2']}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."

                answer1 = generate_answer_from_llm(final_prompt1, model="gpt-3.5-turbo")
                answer2 = generate_answer_from_llm(final_prompt2, model="gpt-3.5-turbo")

                # Calculate scores for each answer
                scores1 = calculate_answer_scores(answer1, pair["ground_truth"])
                scores2 = calculate_answer_scores(answer2, pair["ground_truth"])

                # Determine which answer is better (higher average score)
                avg_score1 = sum(scores1.values()) / len(scores1)
                avg_score2 = sum(scores2.values()) / len(scores2)
                score_diff = abs(avg_score1 - avg_score2)

                # Skip pairs with too small a difference - not informative for training
                if score_diff < min_score_diff:
                    logger.debug(f"Skipping pair with small score difference: {score_diff:.4f}")
                    continue

                # Create preference pair based on scores
                if avg_score1 >= avg_score2:
                    chosen_action = pair["action1"]
                    chosen_prompt = pair["modified_prompt1"]
                    chosen_answer = answer1
                    chosen_scores = scores1

                    rejected_action = pair["action2"]
                    rejected_prompt = pair["modified_prompt2"]
                    rejected_answer = answer2
                    rejected_scores = scores2
                else:
                    chosen_action = pair["action2"]
                    chosen_prompt = pair["modified_prompt2"]
                    chosen_answer = answer2
                    chosen_scores = scores2

                    rejected_action = pair["action1"]
                    rejected_prompt = pair["modified_prompt1"]
                    rejected_answer = answer1
                    rejected_scores = scores1

                # Check if we've reached the limit for "keep unchanged" actions
                is_unchanged_action = chosen_action[0] == 0
                if is_unchanged_action and action_counts["unchanged"] >= max_unchanged_pairs:
                    logger.debug(f"Skipping 'keep unchanged' pair (reached limit of {max_unchanged_pairs})")
                    continue

                # Update action counts
                if is_unchanged_action:
                    action_counts["unchanged"] += 1
                else:
                    action_counts["specific"] += 1

                # Store the preference pair
                valid_pairs.append({
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "question_type": pair["question_type"],
                    "chosen_action_id": chosen_action[0],
                    "chosen_action_desc": chosen_action[1],
                    "chosen_prompt": chosen_prompt,
                    "chosen_answer": chosen_answer,
                    "chosen_scores": chosen_scores,
                    "rejected_action_id": rejected_action[0],
                    "rejected_action_desc": rejected_action[1],
                    "rejected_prompt": rejected_prompt,
                    "rejected_answer": rejected_answer,
                    "rejected_scores": rejected_scores,
                    "score_diff": score_diff
                })

            # Add valid pairs to the preference_pairs list
            preference_pairs.extend(valid_pairs)
            pairs_created += len(valid_pairs)
            progress_bar.update(len(valid_pairs))

            # Save intermediate results
            if output_path and preference_pairs and pairs_created % 50 == 0:
                temp_output = f"{output_path}.temp"
                with open(temp_output, 'w', encoding='utf-8') as f:
                    json.dump(preference_pairs, f, indent=2)
                logger.info(f"Saved {len(preference_pairs)} pairs so far to {temp_output}")

                # Log action distribution
                unchanged_ratio = action_counts["unchanged"] / len(preference_pairs)
                logger.info(
                    f"Current action distribution - Unchanged: {action_counts['unchanged']} ({unchanged_ratio:.1%}), "
                    f"Specific: {action_counts['specific']} ({1 - unchanged_ratio:.1%})")

        progress_bar.close()

        logger.info(f"Completed generating {pairs_created} pairs for {question_type} questions "
                    f"(Target: {pairs_per_type}, Attempts: {attempts})")

    # Log final action distribution
    total_pairs = len(preference_pairs)
    if total_pairs > 0:
        unchanged_ratio = action_counts["unchanged"] / total_pairs
        logger.info(f"Final action distribution:")
        logger.info(f"- Unchanged actions: {action_counts['unchanged']} ({unchanged_ratio:.1%})")
        logger.info(f"- Specific actions: {action_counts['specific']} ({1 - unchanged_ratio:.1%})")

    # Save final results if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preference_pairs, f, indent=2)
        logger.info(f"Saved {len(preference_pairs)} preference pairs to {output_path}")

    logger.info(f"Created {len(preference_pairs)} preference pairs in {time.time() - start_time:.2f}s")
    return preference_pairs


def analyze_preference_pairs(preference_pairs, action_space):
    """Analyze the distribution of actions in the preference pairs"""
    logger.info("Analyzing preference pairs distribution")

    # Count chosen actions by question type
    counts = {}
    for qt in ["what", "how", "if_can"]:
        counts[qt] = {}

    # Count total occurrences of each action
    for pair in preference_pairs:
        qt = pair["question_type"]
        action_id = pair["chosen_action_id"]

        if action_id not in counts[qt]:
            counts[qt][action_id] = 0
        counts[qt][action_id] += 1

    # Calculate percentages and print summary
    for qt in counts:
        total = sum(counts[qt].values())
        if total == 0:
            continue

        logger.info(f"Action distribution for {qt} questions:")

        # Get action descriptions
        general_actions = action_space["general_actions"]
        specific_actions = action_space[f"{qt}_question_actions"]

        for action_id, count in sorted(counts[qt].items()):
            percentage = (count / total) * 100

            # Get action description
            if action_id == 0:  # Special case for "keep unchanged"
                description = general_actions["0"]
            elif action_id < len(general_actions):
                description = general_actions[str(action_id)]
            else:
                # For specific actions, we need to adjust the index
                specific_id = action_id - len(general_actions) + 1
                description = specific_actions.get(str(specific_id), "Unknown action")

            logger.info(f"  Action {action_id} ({description}): {count} occurrences ({percentage:.1f}%)")

    return counts


def save_preference_pairs(preference_pairs, output_path):
    """Save preference pairs to a JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, indent=2)

    logger.info(f"Saved {len(preference_pairs)} preference pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate preference pairs for DPO training')
    parser.add_argument("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json", type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json", type=str, required=True,
                        help='Path to action space configuration')
    parser.add_argument("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/dpo_pairs_balanced.json", type=str, required=True,
                        help='Path to save generated preference pairs')
    parser.add_argument('--num_pairs', type=int, default=1200,
                        help='Number of preference pairs to generate (400 per question type)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--min_score_diff', type=float, default=0.1,
                        help='Minimum score difference to include a pair')
    parser.add_argument('--max_unchanged_ratio', type=float, default=0.3,
                        help='Maximum ratio of pairs where "keep unchanged" is chosen')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Configure file logging
    log_file = f"{args.output_path}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info(f"Starting preference pair generation with improved sampling")
    logger.info(f"Arguments: {args}")

    # Load configurations
    action_space = load_config(args.action_space)
    train_dataset = load_dataset(args.train_data)

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Create preference pairs
    start_time = time.time()
    preference_pairs = create_preference_pairs(
        train_dataset,
        action_space,
        num_pairs=args.num_pairs,
        batch_size=args.batch_size,
        output_path=args.output_path,
        min_score_diff=args.min_score_diff,
        max_unchanged_ratio=args.max_unchanged_ratio
    )

    # Analyze the preference pairs
    analyze_preference_pairs(preference_pairs, action_space)

    total_time = time.time() - start_time
    logger.info(f"Process completed in {total_time:.2f}s ({total_time / 60:.2f}m)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()