import json
import logging

logger = logging.getLogger("ablation_utils")


def apply_action_space_ablation(action_space, ablation_type):
    """Apply different types of action space ablations."""

    # Deep copy to avoid modifying the original
    original_action_space = json.loads(json.dumps(action_space))

    if ablation_type == "unified":
        # Create a unified action space with all actions
        all_actions = {}
        action_id = len(original_action_space["general_actions"])

        # Collect all question-specific actions
        for qt_key in ["what_question_actions", "how_question_actions", "if_can_question_actions"]:
            for action_id_str, action_desc in original_action_space.get(qt_key, {}).items():
                all_actions[str(action_id)] = action_desc
                action_id += 1

        # Apply to all question types
        new_action_space = {
            "general_actions": original_action_space["general_actions"],
            "what_question_actions": all_actions,
            "how_question_actions": all_actions,
            "if_can_question_actions": all_actions
        }

        logger.info(f"Created unified action space with {len(all_actions)} specific actions")
        return new_action_space

    elif ablation_type == "what_only":
        # Use only what_question_actions for all types
        what_actions = original_action_space.get("what_question_actions", {})
        new_action_space = {
            "general_actions": original_action_space["general_actions"],
            "what_question_actions": what_actions,
            "how_question_actions": what_actions,
            "if_can_question_actions": what_actions
        }
        logger.info(f"Using only 'what' question actions for all question types")
        return new_action_space

    elif ablation_type == "how_only":
        # Use only how_question_actions for all types
        how_actions = original_action_space.get("how_question_actions", {})
        new_action_space = {
            "general_actions": original_action_space["general_actions"],
            "what_question_actions": how_actions,
            "how_question_actions": how_actions,
            "if_can_question_actions": how_actions
        }
        logger.info(f"Using only 'how' question actions for all question types")
        return new_action_space

    elif ablation_type == "if_can_only":
        # Use only if_can_question_actions for all types
        if_can_actions = original_action_space.get("if_can_question_actions", {})
        new_action_space = {
            "general_actions": original_action_space["general_actions"],
            "what_question_actions": if_can_actions,
            "how_question_actions": if_can_actions,
            "if_can_question_actions": if_can_actions
        }
        logger.info(f"Using only 'if_can' question actions for all question types")
        return new_action_space

    elif ablation_type == "minimal":
        # Use only general actions
        new_action_space = {
            "general_actions": original_action_space["general_actions"],
            "what_question_actions": {},
            "how_question_actions": {},
            "if_can_question_actions": {}
        }
        logger.info(f"Using only general actions for all question types")
        return new_action_space

    # Return original if no matching ablation type
    logger.info(f"No matching ablation type '{ablation_type}', using original action space")
    print(f"🟥 No matching ablation type '{ablation_type}', using original action space")
    return original_action_space