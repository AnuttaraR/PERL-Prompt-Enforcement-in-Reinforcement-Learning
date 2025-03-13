import os
import sys
import subprocess
import json
import shutil
from datetime import datetime

# Set up paths
BASE_DIR = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(BASE_DIR, f"dpo_ablation_study_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# Python executable
python_executable = sys.executable

# Mapping file to track output directories
mapping = {}

# Ablation types to run
ablation_types = [
    "baseline",  # Standard configuration
    "balanced_pairs",  # Equal distribution across action types
    "higher_temperature",  # Higher temperature for exploration
    "action_diversity_reward",  # Explicitly reward diverse actions
    "action_minimal",  # Only general actions
    "beta_low",  # Lower regularization (beta=0.01)
    "beta_high",  # Higher regularization (beta=1.0)
    "no_keep_unchanged",  # Remove the "keep unchanged" action
    "question_weighted_loss"  # Weight loss by question type performance
]

# Common parameters
preference_pairs_path = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/dpo_run_20250306_165600/preference_pairs.json"
common_params = [
    "--train_data", os.path.join(BASE_DIR, "data/train_data.json"),
    "--test_data", os.path.join(BASE_DIR, "data/test_data.json"),
    "--action_space", os.path.join(BASE_DIR, "config/action_space_config.json"),
    "--epochs", "10",
    "--batch_size", "32",
    "--load_pairs", preference_pairs_path
]


# Special parameter handling for each ablation type
def get_special_params(ablation_type):
    if ablation_type == "beta_low":
        return ["--beta", "0.01"]
    elif ablation_type == "beta_high":
        return ["--beta", "1.0"]
    elif ablation_type == "higher_temperature":
        return ["--temperature", "10.0"]
    elif ablation_type == "no_keep_unchanged":
        # Use modified action space without "keep unchanged"
        modified_action_space = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/modified_data/modified_action_space.json"
        # Use filtered preference pairs
        filtered_pairs_path = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/modified_data/filtered_preference_pairs.json"
        return ["--action_space", modified_action_space, "--load_pairs", filtered_pairs_path]
    elif ablation_type == "action_diversity_reward":
        return ["--diversity_weight", "0.5"]  # Weight for diversity reward
    elif ablation_type == "question_weighted_loss":
        return ["--weighted_loss"]  # Use question-type weighted loss
    elif ablation_type == "action_minimal":
        return ["--ablation", "action_minimal"]
    elif ablation_type == "balanced_pairs":
        balanced_pairs_path = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/dpo_pairs_balanced.json"
        return ["--load_pairs", balanced_pairs_path]
    else:
        return []


# Run each ablation type
for ablation_type in ablation_types:
    print(f"\n{'=' * 50}")
    print(f"⭐ STARTING DPO ABLATION: {ablation_type}")
    print(f"{'=' * 50}\n")

    # Create directories for this ablation
    ablation_base_dir = os.path.join(experiment_dir, ablation_type)
    os.makedirs(ablation_base_dir, exist_ok=True)

    # Run the experiment with a timestamped output dir
    timestamp_dir = f"{ablation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(experiment_dir, timestamp_dir)

    # Store in mapping
    mapping[ablation_type] = output_dir

    # Get special parameters for this ablation
    special_params = get_special_params(ablation_type)

    cmd = [
        python_executable,
        os.path.join(BASE_DIR, "dpo_model.py"),
        *common_params,
        "--output_dir", output_dir,
        *special_params
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Find and copy the evaluation results
    eval_results_path = os.path.join(output_dir, "evaluation_results.json")

    if os.path.exists(eval_results_path):
        # Copy to standard location for analysis
        results_file = os.path.join(ablation_base_dir, "evaluation_results.json")
        shutil.copy2(eval_results_path, results_file)
        print(f"✓ Copied evaluation results to {results_file}")

        # Run analysis on the copied results
        analysis_cmd = [
            python_executable,
            os.path.join(BASE_DIR, "analyze_dpo.py"),
            "--results", results_file,
            "--model", os.path.join(output_dir, "final_model"),
            "--output_dir", os.path.join(ablation_base_dir, "analysis")
        ]
        print(f"Running analysis for: {ablation_type}")
        subprocess.run(analysis_cmd)
    else:
        print(f"⚠️ No evaluation results found at {eval_results_path}")

# Save mapping file for future reference
mapping_file = os.path.join(experiment_dir, "ablation_mapping.json")
with open(mapping_file, 'w') as f:
    json.dump(mapping, f, indent=2)

# After all ablations complete, run the comparison
print("\nAll ablations complete. Generating comparison visualizations...")
comparison_cmd = [
    python_executable,
    os.path.join(BASE_DIR, "compare_dpo_results.py"),
    "--experiment_dir", experiment_dir,
    "--mapping_file", mapping_file
]
subprocess.run(comparison_cmd)

print(f"\nDPO Ablation study completed. Results saved to {experiment_dir}")