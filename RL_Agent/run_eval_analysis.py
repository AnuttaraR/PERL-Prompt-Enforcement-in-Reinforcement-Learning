import os
import sys
import json
import torch
import numpy as np
import argparse
import logging
import time
import subprocess
from datetime import datetime

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import necessary modules
from ppo_model import PPOAgent, load_dataset, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("/RL_Agent/ppo_results/logs", "evaluate_model.log"),
                            encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("evaluate_model")


def save_metrics(metrics, output_dir):
    """Save metrics to JSON file with proper serialization"""
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = f"{output_dir}/evaluation_metrics.json"
    logger.info(f"Saving metrics to {metrics_file}")

    # Helper function to convert non-serializable objects
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj

    # Special handling for nested dictionaries and arrays
    def process_json_structure(obj):
        if isinstance(obj, dict):
            return {str(k): process_json_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [process_json_structure(item) for item in obj]
        else:
            return convert_for_json(obj)

    # Process the entire metrics structure
    processed_metrics = process_json_structure(metrics)

    with open(metrics_file, 'w') as f:
        json.dump(processed_metrics, f, indent=2)

    logger.info(f"Metrics saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Evaluate best PPO model on test dataset')
    parser.add_argument('--model_path', type=str, default='best_ppo_model',
                        help='Path to the saved model')
    parser.add_argument('--test_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration')
    parser.add_argument('--reward_config', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json',
                        help='Path to reward configuration')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on (cuda or cpu)')

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging for this run
    file_handler = logging.FileHandler(os.path.join(output_dir, "evaluation.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 50)
    logger.info(f"Starting evaluation of model at {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Device: {args.device}")
    start_time = time.time()

    # Load configurations
    action_space = load_config(args.action_space)
    reward_config = load_config(args.reward_config)

    # Load test dataset
    test_dataset = load_dataset(args.test_data)
    logger.info(f"Loaded test dataset with {len(test_dataset)} examples")

    # Calculate input dimension based on first example
    input_dim = len(test_dataset[0]["tokens"])
    logger.info(f"Input dimension: {input_dim}")

    # Initialize agent with the same parameters used during training
    agent = PPOAgent(
        input_dim=input_dim,
        action_space=action_space,
        reward_config=reward_config,
        device=args.device
    )

    # Load the trained model
    if agent.load_model(args.model_path):
        logger.info(f"Successfully loaded model from {args.model_path}")
    else:
        logger.error(f"Failed to load model from {args.model_path}")
        return

    # Run evaluation
    logger.info("Starting evaluation on test dataset...")
    eval_start_time = time.time()
    test_metrics = agent.evaluate(test_dataset)
    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f}s ({eval_time / 60:.2f} minutes)")

    # Save metrics
    save_metrics(test_metrics, output_dir)

    # Calculate and print summary statistics
    all_rewards = []
    logger.info("\nEvaluation Summary:")

    for qt, metrics in test_metrics.items():
        if qt == "action_selections":
            continue

        if "rewards" in metrics and metrics["rewards"]:
            avg_reward = np.mean(metrics["rewards"])
            all_rewards.extend(metrics["rewards"])

            logger.info(f"Question Type: {qt}")
            logger.info(f"  Samples: {len(metrics['rewards'])}")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")

            if "bert_scores" in metrics:
                logger.info(f"  Avg BERTScore: {np.mean(metrics['bert_scores']):.3f}")
            if "rouge_scores" in metrics:
                logger.info(f"  Avg ROUGE-L: {np.mean(metrics['rouge_scores']):.3f}")
            if "meteor_scores" in metrics:
                logger.info(f"  Avg METEOR: {np.mean(metrics['meteor_scores']):.3f}")

    if all_rewards:
        logger.info(f"\nOverall Average Reward: {np.mean(all_rewards):.3f}")

    # Log top actions
    logger.info("\nTop Actions Used by Question Type:")
    if "action_selections" in test_metrics:
        action_selections = test_metrics["action_selections"]
        for qt, actions in action_selections.items():
            if actions:
                total = sum(actions.values())
                percentages = {k: (v / total) * 100 for k, v in actions.items()}
                top_actions = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]

                logger.info(f"{qt} questions:")
                for action_id, percentage in top_actions:
                    action_desc = agent.get_action_description(int(action_id), qt)
                    logger.info(f"  - {action_desc}: {percentage:.1f}%")

    total_time = time.time() - start_time
    logger.info(f"\nTotal evaluation time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)

    print(f"\nEvaluation complete! Results saved to {output_dir}")

    # Run PPO analysis on the saved metrics
    logger.info("Running PPO analysis on evaluation metrics...")
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")

    # Prepare the analysis command
    analysis_script = os.path.join(current_dir, "ppo_analysis.py")
    analysis_cmd = [
        sys.executable,
        analysis_script,
        "--metrics_file", metrics_file,
        "--output_dir", analysis_dir
    ]

    try:
        logger.info(f"Running command: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd, check=True)
        logger.info(f"Analysis completed successfully. Results saved to {analysis_dir}")
        print(f"Analysis completed successfully. Results saved to {analysis_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running analysis: {e}")
        print(f"Error running analysis: {e}")

    return output_dir  # Return the output directory for potential further use


if __name__ == "__main__":
    main()