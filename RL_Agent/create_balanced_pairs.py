import json
import os
import random
from collections import defaultdict

# Paths
BASE_DIR = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent"
input_path = os.path.join(BASE_DIR, "dpo_run_20250306_165600/preference_pairs.json")
output_path = os.path.join(BASE_DIR, "dpo_pairs_balanced.json")

# Load original preference pairs
with open(input_path, 'r', encoding='utf-8') as f:
    original_pairs = json.load(f)

print(f"Loaded {len(original_pairs)} original preference pairs")

# Group by question type
pairs_by_type = defaultdict(list)
for pair in original_pairs:
    qt = pair["question_type"]
    pairs_by_type[qt].append(pair)

print(f"Distribution by question type:")
for qt, pairs in pairs_by_type.items():
    print(f"  {qt}: {len(pairs)} pairs")

# Count action distributions
action_counts = defaultdict(lambda: defaultdict(int))
for pair in original_pairs:
    qt = pair["question_type"]
    action_counts[qt][pair["chosen_action_id"]] += 1

print("\nChosen action distribution:")
for qt, counts in action_counts.items():
    print(f"  {qt}:")
    total = sum(counts.values())
    for action_id, count in sorted(counts.items()):
        print(f"    Action {action_id}: {count} ({count / total:.1%})")

# Create balanced pairs
balanced_pairs = []
pairs_per_type = 200  # Aim for 600 total (200 per type)
max_unchanged_percent = 0.4  # Max percentage of "keep unchanged" actions

for qt, pairs in pairs_by_type.items():
    # Group pairs by chosen action
    pairs_by_action = defaultdict(list)
    for pair in pairs:
        pairs_by_action[pair["chosen_action_id"]].append(pair)

    # Calculate how many "keep unchanged" (action 0) pairs to include
    unchanged_count = int(pairs_per_type * max_unchanged_percent)

    # Sample pairs with constraints
    qt_balanced_pairs = []

    # First, add limited "keep unchanged" pairs
    if 0 in pairs_by_action and len(pairs_by_action[0]) > 0:
        unchanged_pairs = random.sample(pairs_by_action[0],
                                        min(unchanged_count, len(pairs_by_action[0])))
        qt_balanced_pairs.extend(unchanged_pairs)

    # Then fill the rest with other actions
    other_actions = [a for a in pairs_by_action.keys() if a != 0]
    remaining_count = pairs_per_type - len(qt_balanced_pairs)

    # Distribute remaining count as evenly as possible
    if other_actions:
        pairs_per_action = remaining_count // len(other_actions)

        for action_id in other_actions:
            action_pairs = pairs_by_action[action_id]
            if len(action_pairs) > 0:
                sampled = random.sample(action_pairs,
                                        min(pairs_per_action, len(action_pairs)))
                qt_balanced_pairs.extend(sampled)

    # If we still need more pairs, sample randomly from all pairs
    if len(qt_balanced_pairs) < pairs_per_type:
        remaining = pairs_per_type - len(qt_balanced_pairs)

        # Create array of indices for all pairs and chosen pairs
        all_indices = list(range(len(pairs)))
        chosen_indices = []

        # Track which indices we've already used
        for chosen_pair in qt_balanced_pairs:
            # Find the index of this pair in the original list
            for i, pair in enumerate(pairs):
                if pair is chosen_pair:  # Identity comparison
                    chosen_indices.append(i)
                    break

        # Get remaining indices
        remaining_indices = [i for i in all_indices if i not in chosen_indices]

        # Sample from remaining indices
        if remaining_indices:
            sample_indices = random.sample(remaining_indices,
                                           min(remaining, len(remaining_indices)))
            additional = [pairs[i] for i in sample_indices]
            qt_balanced_pairs.extend(additional)

    balanced_pairs.extend(qt_balanced_pairs)
    print(f"Added {len(qt_balanced_pairs)} balanced pairs for {qt}")

# Shuffle final list
random.shuffle(balanced_pairs)

# Save balanced pairs
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(balanced_pairs, f, indent=2)

print(f"\nSaved {len(balanced_pairs)} balanced preference pairs to {output_path}")

# Analyze distribution in balanced pairs
balanced_action_counts = defaultdict(lambda: defaultdict(int))
for pair in balanced_pairs:
    qt = pair["question_type"]
    balanced_action_counts[qt][pair["chosen_action_id"]] += 1

print("\nBalanced pairs action distribution:")
for qt, counts in balanced_action_counts.items():
    print(f"  {qt}:")
    total = sum(counts.values())
    for action_id, count in sorted(counts.items()):
        print(f"    Action {action_id}: {count} ({count / total:.1%})")