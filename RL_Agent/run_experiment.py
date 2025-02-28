import os
import argparse
import json
import subprocess
import logging
import shutil
import time
import sys
from datetime import datetime

# Run normal training script: python train_script.py --train_data C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json --test_data C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json --action_space C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json --reward_config C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json --episodes 10 --batch_size 64

# Create logs directory if it doesn't exist
os.makedirs("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", exist_ok=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", "run_experiment.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_experiment")


def setup_experiment_directory(base_dir="experiments"):
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up experiment-specific log file
    experiment_log = os.path.join(experiment_dir, "experiment.log")
    file_handler = logging.FileHandler(experiment_log)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir


def copy_configs(experiment_dir, action_space_path, reward_config_path):
    """Copy existing configuration files to the experiment directory."""
    target_action_space = os.path.join(experiment_dir, "action_space.json")
    target_reward_config = os.path.join(experiment_dir, "reward_config.json")

    # Copy the configuration files
    shutil.copy2(action_space_path, target_action_space)
    shutil.copy2(reward_config_path, target_reward_config)

    logger.info(f"Copied configuration files to {experiment_dir}")
    return target_action_space, target_reward_config


def run_training(experiment_dir, train_data, test_data, action_space, reward_config, episodes, batch_size, lr):
    """Run the training process."""
    logger.info("Starting model training...")
    train_start_time = time.time()
    python_executable = sys.executable

    train_script = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/train_script.py"
    output_dir = os.path.join(experiment_dir, "training_output")

    cmd = [
        python_executable,
        train_script,
        "--train_data", train_data,
        "--test_data", test_data,
        "--action_space", action_space,
        "--reward_config", reward_config,
        "--episodes", str(episodes),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--output_dir", output_dir
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Capture and log both stdout and stderr
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Log stdout
        if result.stdout:
            print("STDOUT:", result.stdout)
            logger.info("SUBPROCESS STDOUT:\n%s", result.stdout)

        # Log stderr
        if result.stderr:
            print("STDERR:", result.stderr)
            logger.error("SUBPROCESS STDERR:\n%s", result.stderr)

    except subprocess.CalledProcessError as e:
        logger.error(f"Training command failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise

    train_time = time.time() - train_start_time
    logger.info(f"Training completed in {train_time:.2f}s ({train_time / 60:.2f} minutes)")
    logger.info(f"Results saved to {output_dir}")

    return output_dir


def run_ablation_studies(experiment_dir, train_data, test_data, action_space, reward_config, episodes, batch_size, lr):
    """Run ablation studies."""
    logger.info("Starting ablation studies...")
    ablation_start_time = time.time()
    python_executable = sys.executable

    train_script = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/train_script.py"

    ablation_types = ["no_bert", "no_rouge", "no_question_specific"]
    ablation_results = {}

    for ablation in ablation_types:
        logger.info(f"Running ablation study: {ablation}")
        ablation_type_start = time.time()

        output_dir = os.path.join(experiment_dir, f"ablation_{ablation}")

        cmd = [
            python_executable,
            train_script,
            "--train_data", train_data,
            "--test_data", test_data,
            "--action_space", action_space,
            "--reward_config", reward_config,
            "--episodes", str(episodes),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--output_dir", output_dir,
            "--ablation", ablation
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.info(f"Ablation {ablation} stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Ablation {ablation} stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ablation {ablation} command failed with exit code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            logger.warning(f"Continuing with next ablation study despite error")
            continue

        ablation_type_time = time.time() - ablation_type_start
        logger.info(
            f"Ablation study {ablation} completed in {ablation_type_time:.2f}s ({ablation_type_time / 60:.2f} minutes)")
        logger.info(f"Results saved to {output_dir}")

        ablation_results[ablation] = output_dir

    total_ablation_time = time.time() - ablation_start_time
    logger.info(
        f"All ablation studies completed in {total_ablation_time:.2f}s ({total_ablation_time / 60:.2f} minutes)")

    return ablation_results


def run_analysis(results_dir):
    """Run analysis on the results."""
    logger.info("Running analysis and visualization...")
    analysis_start_time = time.time()

    analysis_script = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/ppo_analysis.py"

    metrics_file = os.path.join(results_dir, "metrics.json")
    output_dir = os.path.join(results_dir, "analysis")

    cmd = [
        "python", analysis_script,
        "--metrics_file", metrics_file,
        "--output_dir", output_dir
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Analysis stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Analysis stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis command failed with exit code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        logger.warning(f"Continuing despite analysis error")

    analysis_time = time.time() - analysis_start_time
    logger.info(f"Analysis completed in {analysis_time:.2f}s ({analysis_time / 60:.2f} minutes)")
    logger.info(f"Results saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run a complete experiment')
    parser.add_argument('--train_data', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json',
                        help='Path to training dataset')
    parser.add_argument('--test_data', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration file')
    parser.add_argument('--reward_config', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json',
                        help='Path to reward configuration file')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='PPO batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--run_ablation', action='store_true',
                        help='Whether to run ablation studies')
    parser.add_argument('--base_dir', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/experiments',
                        help='Base directory for experiments')

    args = parser.parse_args()

    # Create experiment directory
    experiment_dir = setup_experiment_directory(args.base_dir)

    # Copy existing configurations
    action_space_path, reward_config_path = copy_configs(
        experiment_dir,
        args.action_space,
        args.reward_config
    )

    # Run training
    training_output = run_training(
        experiment_dir,
        args.train_data,
        args.test_data,
        action_space_path,
        reward_config_path,
        args.episodes,
        args.batch_size,
        args.lr
    )

    # Run analysis
    analysis_output = run_analysis(training_output)

    # Run ablation studies if requested
    if args.run_ablation:
        ablation_results = run_ablation_studies(
            experiment_dir,
            args.train_data,
            args.test_data,
            action_space_path,
            reward_config_path,
            args.episodes,
            args.batch_size,
            args.lr
        )

        # Run analysis on each ablation study
        for ablation, result_dir in ablation_results.items():
            run_analysis(result_dir)

    # Create experiment summary
    summary = {
        "experiment_dir": experiment_dir,
        "training_output": training_output,
        "analysis_output": analysis_output,
        "parameters": {
            "train_data": args.train_data,
            "test_data": args.test_data,
            "action_space": args.action_space,
            "reward_config": args.reward_config,
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "learning_rate": args.lr
        }
    }

    if args.run_ablation:
        summary["ablation_studies"] = ablation_results

    # Save summary
    summary_path = os.path.join(experiment_dir, "experiment_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()