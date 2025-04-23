import os
import sys
import argparse
import json
import logging
import time
import torch
import random
import numpy as np
from datetime import datetime
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the DPO model and utilities
from dpo_model import (
    DPOTrainer, load_config, load_dataset, load_preference_pairs,
    calculate_answer_scores
)

from create_preference_pairs import create_preference_pairs

# Import utility functions for LLM interaction
try:
    from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
    from RL_Agent.utils.retrieval import retrieve_context
except ImportError:
    print("Warning: Could not import from RL_Agent.utils. Make sure the path is correct.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("integrated_dpo")


def compare_models(ppo_metrics_path, dpo_results_path, output_dir):
    """Compare PPO and DPO model performance"""
    logger.info("Comparing PPO and DPO model performance")

    # Load metrics files
    with open(ppo_metrics_path, 'r') as f:
        ppo_metrics = json.load(f)

    with open(dpo_results_path, 'r') as f:
        dpo_results = json.load(f)

    # Extract metrics
    question_types = ["what", "how", "if_can"]

    ppo_rewards = {}
    dpo_accuracy = {}

    # Extract PPO metrics
    if "test" in ppo_metrics:
        for qt in question_types:
            if qt in ppo_metrics["test"] and "rewards" in ppo_metrics["test"][qt]:
                ppo_rewards[qt] = np.mean(ppo_metrics["test"][qt]["rewards"])

    # Extract DPO metrics
    for qt in question_types:
        if qt in dpo_results:
            dpo_accuracy[qt] = dpo_results[qt]["accuracy"]

    # Overall metrics
    ppo_overall = np.mean([ppo_rewards.get(qt, 0) for qt in question_types])
    dpo_overall = dpo_results.get("overall", {}).get("accuracy", 0)

    # Create comparison plot
    plt.figure(figsize=(12, 8))

    # Bar positions
    bar_width = 0.35
    index = np.arange(len(question_types) + 1)

    # Add the question type bars
    ppo_vals = [ppo_rewards.get(qt, 0) for qt in question_types] + [ppo_overall]
    dpo_vals = [dpo_accuracy.get(qt, 0) / 100 for qt in question_types] + [dpo_overall / 100]

    # Convert DPO accuracy to 0-1 scale for comparison
    plt.bar(index - bar_width / 2, ppo_vals, bar_width, label='PPO (Avg Reward)')
    plt.bar(index + bar_width / 2, dpo_vals, bar_width, label='DPO (Accuracy as 0-1)')

    plt.xlabel('Question Type')
    plt.ylabel('Performance')
    plt.title('PPO vs DPO Performance Comparison')
    plt.xticks(index, question_types + ['Overall'])
    plt.legend()

    # Add value labels on top of bars
    for i, v in enumerate(ppo_vals):
        plt.text(i - bar_width / 2, v + 0.02, f'{v:.3f}', ha='center')

    for i, v in enumerate(dpo_vals):
        plt.text(i + bar_width / 2, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ppo_vs_dpo_comparison.png"))
    plt.close()

    # Save comparison data
    comparison = {
        "ppo": {
            "question_types": ppo_rewards,
            "overall": ppo_overall
        },
        "dpo": {
            "question_types": dpo_accuracy,
            "overall": dpo_overall
        }
    }

    with open(os.path.join(output_dir, "model_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Model comparison saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run DPO training with preference pair generation')
    # Data paths
    parser.add_argument('--train_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json',
                        help='Path to training dataset')
    parser.add_argument('--test_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration')

    # Preference pair options - Updated with new defaults
    parser.add_argument('--skip_pair_gen', action='store_true', default=True,
                        help='Skip preference pair generation (use existing pairs)')
    parser.add_argument('--existing_pairs', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/dpo_run_20250306_165600/preference_pairs.json',
                        help='Path to existing preference pairs JSON file')
    parser.add_argument('--num_pairs', type=int, default=1200,
                        help='Number of preference pairs to generate (only used if skip_pair_gen is False)')
    parser.add_argument('--pair_batch_size', type=int, default=8,
                        help='Batch size for preference pair generation (only used if skip_pair_gen is False)')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter')

    # Comparison and evaluation
    parser.add_argument('--ppo_metrics', type=str, default=None,
                        help='Path to PPO metrics JSON for comparison (optional)')

    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: timestamped directory)')

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"dpo_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure file logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "dpo_run.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Start timer
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"Starting integrated DPO training pipeline")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Handle preference pairs (generate or load)
    if args.skip_pair_gen:
        if not args.existing_pairs:
            logger.error("Must provide --existing_pairs when using --skip_pair_gen")
            return 1
        logger.info(f"Skipping preference pair generation, loading from {args.existing_pairs}")
        preference_pairs = load_preference_pairs(args.existing_pairs)
    else:
        # Step 1a: Load configurations and datasets
        logger.info("Loading action space and training dataset")
        action_space = load_config(args.action_space)
        train_dataset = load_dataset(args.train_data)

        # Step 1b: Generate preference pairs
        logger.info(f"Generating {args.num_pairs} preference pairs")
        pairs_path = os.path.join(output_dir, "preference_pairs.json")
        preference_pairs = create_preference_pairs(
            train_dataset,
            action_space,
            num_pairs=args.num_pairs,
            batch_size=args.pair_batch_size,
            output_path=pairs_path
        )
        logger.info(f"Preference pairs saved to {pairs_path}")

    # Step 2: Load test dataset and action space
    test_dataset = load_dataset(args.test_data)
    action_space = load_config(args.action_space)

    # Step 3: Initialize tokenizer and DPO trainer
    logger.info("Initializing tokenizer and DPO trainer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Calculate input dimension based on tokenizer
    input_dim = 768

    # Initialize DPO trainer
    trainer = DPOTrainer(
        input_dim=input_dim,
        action_space=action_space,
        learning_rate=args.lr,
        beta=args.beta
    )

    # Step 4: Train the model
    logger.info(f"Starting DPO training for {args.epochs} epochs")
    train_start = time.time()
    metrics = trainer.train(
        preference_pairs,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1
    )
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f}s ({train_time / 60:.2f}m)")

    # After training is completed
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    # Step 5: Save trained model
    model_path = os.path.join(output_dir, "dpo_model")
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Step 6: Evaluate model
    logger.info("Evaluating model on test dataset")
    eval_start = time.time()
    results = trainer.evaluate(test_dataset, tokenizer)
    eval_time = time.time() - eval_start
    logger.info(f"Evaluation completed in {eval_time:.2f}s ({eval_time / 60:.2f}m)")

    # Step 7: Save results
    results_path = os.path.join(output_dir, "action_minimal_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    # Step 8: Compare with PPO if metrics provided
    if args.ppo_metrics:
        logger.info(f"Comparing with PPO metrics from {args.ppo_metrics}")
        compare_models(args.ppo_metrics, results_path, output_dir)

    # Log overall performance
    overall_accuracy = results.get("overall", {}).get("accuracy", 0)
    logger.info(f"Overall accuracy: {overall_accuracy:.2f}%")

    # Log performance by question type
    logger.info("Performance by question type:")
    for qt in ["what", "how", "if_can"]:
        if qt in results:
            logger.info(f"  {qt}: {results[qt]['accuracy']:.2f}% ({results[qt]['correct']}/{results[qt]['total']})")

    # Log top actions by question type
    logger.info("Top actions by question type:")
    for qt in ["what", "how", "if_can"]:
        if qt in results and "top_actions" in results[qt]:
            logger.info(f"  {qt}:")
            for action in results[qt]["top_actions"]:
                logger.info(f"    - {action['description']}: {action['percentage']:.1f}%")

    # Log configuration
    config = {
        "train_data": args.train_data,
        "test_data": args.test_data,
        "action_space": args.action_space,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta": args.beta,
        "seed": args.seed,
        "input_dim": input_dim,
        "num_preference_pairs": len(preference_pairs),
        "training_time": train_time,
        "evaluation_time": eval_time
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    total_time = time.time() - start_time
    logger.info(f"Total run time: {total_time:.2f}s ({total_time / 60:.2f}m)")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
