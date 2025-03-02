import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging
import torch
import sys
from RL_Agent.utils.ablation_utils import apply_action_space_ablation
from ppo_model import PPOAgent, load_config, load_dataset

# Create logs directory if it doesn't exist
os.makedirs("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", "train_script.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_script")


def plot_training_curves(rewards, actor_losses, critic_losses, qt_rewards, output_dir):
    """Plot and save training curves"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating training plots in {output_dir}")

    # Plot overall reward
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')

    # Add smoothed curve if we have enough data
    if len(rewards) > 10:
        window_size = min(50, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, window_size - 1 + len(smoothed)), smoothed,
                 linewidth=2, label=f'Smoothed (window={window_size})')

    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/rewards.png")
    plt.close()

    # Plot actor and critic losses
    if actor_losses and critic_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(actor_losses, label='Actor Loss', alpha=0.3)
        plt.plot(critic_losses, label='Critic Loss', alpha=0.3)

        # Add smoothed curves if we have enough data
        if len(actor_losses) > 10:
            window_size = min(20, len(actor_losses) // 5)
            smoothed_actor = np.convolve(actor_losses, np.ones(window_size) / window_size, mode='valid')
            smoothed_critic = np.convolve(critic_losses, np.ones(window_size) / window_size, mode='valid')

            plt.plot(range(window_size - 1, window_size - 1 + len(smoothed_actor)), smoothed_actor,
                     linewidth=2, label='Actor Loss (Smoothed)')
            plt.plot(range(window_size - 1, window_size - 1 + len(smoothed_critic)), smoothed_critic,
                     linewidth=2, label='Critic Loss (Smoothed)')

        plt.title('PPO Losses')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/losses.png")
        plt.close()

    # Plot question type rewards
    plt.figure(figsize=(10, 6))
    for qt, rewards in qt_rewards.items():
        if rewards:  # Only plot if we have data
            plt.plot(rewards, label=f'{qt} Questions', alpha=0.3)

            # Add smoothed curves if we have enough data
            if len(rewards) > 10:
                window_size = min(20, len(rewards) // 5)
                smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size - 1, window_size - 1 + len(smoothed)), smoothed,
                         linewidth=2, label=f'{qt} Smoothed')

    plt.title('Rewards by Question Type')
    plt.xlabel('Question Count')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/question_type_rewards.png")
    plt.close()

    logger.info(f"Training plots saved to {output_dir}")


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

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(processed_metrics, f, indent=2)

    logger.info(f"Metrics saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Train PPO for prompt optimization')
    parser.add_argument('--train_data', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json',
                        help='Path to training dataset')
    parser.add_argument('--test_data', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration')
    parser.add_argument('--reward_config', type=str, default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json',
                        help='Path to reward configuration')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='PPO batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for models and plots')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Run ablation study (e.g., "no_bert", "no_rouge")')
    parser.add_argument('--checkpoint_freq', type=int, default=50,
                        help='Save checkpoints every N episodes')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Run evaluation every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Set up a file handler for this specific run
    run_log_file = os.path.join(output_dir, "run.log")
    file_handler = logging.FileHandler(run_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 50)
    logger.info(f"Starting training run at {timestamp}")
    logger.info(f"Arguments: {args}")

    # Log environment information
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    start_time = time.time()

    # Load configurations
    logger.info(f"Loading action space from {args.action_space}")
    action_space = load_config(args.action_space)
    logger.info(f"Action space contains {sum(len(actions) for actions in action_space.values())} total actions")

    logger.info(f"Loading reward config from {args.reward_config}")
    reward_config = load_config(args.reward_config)

    # Apply ablation if specified
    if args.ablation:
        logger.info(f"Applying ablation study: {args.ablation}")
        if args.ablation == "no_bert":
            # Remove BERTScore from all question types
            for qt in ["what_question_rewards", "how_question_rewards", "if_can_question_rewards"]:
                if qt in reward_config and "semantic_similarity" in reward_config[qt]:
                    if "bert_score_weight" in reward_config[qt]["semantic_similarity"]:
                        logger.info(f"Removing BERTScore from {qt}")
                        del reward_config[qt]["semantic_similarity"]["bert_score_weight"]

        elif args.ablation == "no_rouge":
            # Remove ROUGE from all question types
            for qt in ["what_question_rewards", "how_question_rewards", "if_can_question_rewards"]:
                if qt in reward_config and "lexical_overlap" in reward_config[qt]:
                    if "rouge_l_weight" in reward_config[qt]["lexical_overlap"]:
                        logger.info(f"Removing ROUGE-L from {qt}")
                        del reward_config[qt]["lexical_overlap"]["rouge_l_weight"]

        elif args.ablation == "no_question_specific":
            # Use the same reward structure for all question types
            what_rewards = reward_config.get("what_question_rewards", {})
            # Apply what_question rewards to all types
            reward_config["how_question_rewards"] = what_rewards
            reward_config["if_can_question_rewards"] = what_rewards
            logger.info("Using the same reward structure for all question types")

            # Use the same action space for all question types
            what_actions = action_space.get("what_question_actions", {})
            action_space["how_question_actions"] = what_actions
            action_space["if_can_question_actions"] = what_actions
            logger.info("Using the same action space for all question types")

        elif args.ablation.startswith("action_"):
            ablation_type = args.ablation.replace("action_", "")
            action_space = apply_action_space_ablation(action_space, ablation_type)
            logger.info(f"Applied action space ablation: {ablation_type}")
            print(f"🟦 Applied action space ablation: {ablation_type}")

        # Save the modified configurations
        with open(f"{output_dir}/ablated_reward_config.json", 'w', encoding='utf-8') as f:
            json.dump(reward_config, f, indent=2)
        with open(f"{output_dir}/ablated_action_space.json", 'w', encoding='utf-8') as f:
            json.dump(action_space, f, indent=2)

    # Load datasets
    dataset_start_time = time.time()
    logger.info(f"Loading training dataset from {args.train_data}")
    train_dataset = load_dataset(args.train_data)
    logger.info(f"Loading test dataset from {args.test_data}")
    test_dataset = load_dataset(args.test_data)
    logger.info(f"Datasets loaded in {time.time() - dataset_start_time:.2f}s")

    # Calculate input dimension based on tokenized state
    input_dim = len(train_dataset[0]["tokens"])
    logger.info(f"Input dimension: {input_dim}")

    # Initialize PPO agent
    agent = PPOAgent(
        input_dim=input_dim,
        action_space=action_space,
        reward_config=reward_config,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Train the agent
    logger.info(f"Starting training for {args.episodes} episodes...")
    training_start_time = time.time()
    episode_rewards = agent.train(train_dataset, num_episodes=args.episodes)
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f}s ({training_time / 60:.2f} minutes)")

    # Save the final model
    agent.save_model(path=f"{output_dir}/final_model")
    logger.info(f"Final model saved to {output_dir}/final_model")

    # Plot training curves
    logger.info("Generating training visualizations...")
    plot_training_curves(
        agent.episode_rewards,
        agent.actor_losses,
        agent.critic_losses,
        agent.question_type_rewards,
        output_dir
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    eval_start_time = time.time()
    test_metrics = agent.evaluate(test_dataset)
    eval_time = time.time() - eval_start_time
    logger.info(f"Evaluation completed in {eval_time:.2f}s ({eval_time / 60:.2f} minutes)")

    # Save metrics
    metrics = {
        "training": {
            "episode_rewards": agent.episode_rewards,
            "actor_losses": agent.actor_losses,
            "critic_losses": agent.critic_losses,
            "question_type_rewards": agent.question_type_rewards
        },
        "test": test_metrics,
        "config": {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "ablation": args.ablation,
            "seed": args.seed
        },
        "timing": {
            "training_time": training_time,
            "evaluation_time": eval_time,
            "total_time": time.time() - start_time
        }
    }
    save_metrics(metrics, output_dir)

    total_time = time.time() - start_time
    logger.info(f"Total run time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
