import io
import os
import sys
import subprocess
import shutil
from datetime import datetime

# Set up paths
BASE_DIR = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/experiments"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(BASE_DIR, f"action_ablation_study_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# Python executable
python_executable = sys.executable

# Ablation types to run
ablation_types = [
    "baseline",  # No ablation
    "action_unified",  # All action types available to all question types
    "action_what_only",  # Only what actions for all types
    "action_how_only",  # Only how actions for all types
    "action_if_can_only",  # Only if_can actions for all types
    "action_minimal"  # Only general actions
]

# Common parameters
common_params = [
    "--train_data", "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json",
    "--test_data", "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json",
    "--action_space", "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json",
    "--reward_config", "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json",
    "--episodes", "100",  # Reduced for testing
    "--batch_size", "64"
]

# Run each ablation type
for ablation_type in ablation_types:
    print(f"\n{'=' * 50}")
    print(f"⭐ STARTING ABLATION: {ablation_type}")
    print(f"{'=' * 50}\n")

    # Create directories for this ablation
    ablation_base_dir = os.path.join(experiment_dir, ablation_type)
    os.makedirs(ablation_base_dir, exist_ok=True)

    # Run the experiment with a timestamped output dir
    timestamp_dir = f"{ablation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(experiment_dir, timestamp_dir)

    cmd = [
        python_executable,
        "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/train_script.py",
        *common_params,
        "--output_dir", output_dir
    ]

    # Add ablation parameter if not baseline
    if ablation_type != "baseline":
        cmd.extend(["--ablation", ablation_type])

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Find and copy the evaluation metrics
    eval_metrics_path = None
    for root, dirs, files in os.walk(experiment_dir):
        if "evaluation_metrics.json" in files and ablation_type in os.path.basename(root):
            eval_metrics_path = os.path.join(root, "evaluation_metrics.json")
            print(f"Found metrics at: {eval_metrics_path}")
            break

    if eval_metrics_path and os.path.exists(eval_metrics_path):
        # Copy to standard location for analysis
        metrics_file = os.path.join(ablation_base_dir, "metrics.json")
        shutil.copy2(eval_metrics_path, metrics_file)
        print(f"✓ Copied evaluation metrics to {metrics_file}")

        # Run analysis on the copied metrics
        analysis_cmd = [
            python_executable,
            "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/ppo_analysis.py",
            "--metrics_file", metrics_file,
            "--output_dir", os.path.join(ablation_base_dir, "analysis")
        ]
        print(f"Running analysis for: {ablation_type}")
        subprocess.run(analysis_cmd)
    else:
        print(f"⚠️ No evaluation metrics found at {eval_metrics_path}")

# After all ablations complete, run the comparison
print("\nAll ablations complete. Generating comparison visualizations...")
comparison_cmd = [
    python_executable,
    "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/compare_ablation_results.py",
    "--experiment_dir", experiment_dir
]
subprocess.run(comparison_cmd)

print(f"\nAblation study completed. Results saved to {experiment_dir}")