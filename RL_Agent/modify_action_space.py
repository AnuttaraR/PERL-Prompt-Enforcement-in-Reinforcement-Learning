import json
import os


def create_modified_action_space(original_action_space):
    """Create a new action space without the 'keep unchanged' action (action 0)"""
    modified_action_space = {
        "general_actions": {},  # Empty - no general actions
        "what_question_actions": {},
        "how_question_actions": {},
        "if_can_question_actions": {}
    }

    # Copy only specific actions, reindexing them to start from 0
    for qt in ["what", "how", "if_can"]:
        qt_key = f"{qt}_question_actions"
        for idx, (action_id, desc) in enumerate(original_action_space[qt_key].items()):
            modified_action_space[qt_key][str(idx)] = desc

    return modified_action_space


def filter_preference_pairs_no_unchanged(preference_pairs):
    """Filter out preference pairs where chosen action is 'keep unchanged'"""
    filtered_pairs = []

    for pair in preference_pairs:
        if pair["chosen_action_id"] != 0:  # Not "keep unchanged"
            # Re-index action IDs (subtract 1 from all IDs > 0)
            modified_pair = pair.copy()

            if modified_pair["chosen_action_id"] > 0:
                modified_pair["chosen_action_id"] -= 1

            if modified_pair["rejected_action_id"] > 0:
                modified_pair["rejected_action_id"] -= 1

            filtered_pairs.append(modified_pair)

    return filtered_pairs


def main():
    # Paths to your files
    action_space_path = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json"
    preference_pairs_path = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/dpo_run_20250306_165600/preference_pairs.json"

    output_dir = "modified_data"
    os.makedirs(output_dir, exist_ok=True)

    # Load original action space
    with open(action_space_path, "r") as f:
        original_action_space = json.load(f)

    # Create modified action space
    modified_action_space = create_modified_action_space(original_action_space)

    # Save modified action space
    modified_action_space_path = os.path.join(output_dir, "modified_action_space.json")
    with open(modified_action_space_path, "w") as f:
        json.dump(modified_action_space, f, indent=2)
    print(f"Modified action space saved to {modified_action_space_path}")

    # Load preference pairs
    with open(preference_pairs_path, "r") as f:
        preference_pairs = json.load(f)

    # Filter preference pairs
    filtered_pairs = filter_preference_pairs_no_unchanged(preference_pairs)

    # Save filtered pairs
    filtered_pairs_path = os.path.join(output_dir, "filtered_preference_pairs.json")
    with open(filtered_pairs_path, "w") as f:
        json.dump(filtered_pairs, f, indent=2)
    print(f"Filtered preference pairs saved to {filtered_pairs_path}")
    print(f"Original pairs: {len(preference_pairs)}, Filtered pairs: {len(filtered_pairs)}")


if __name__ == "__main__":
    main()