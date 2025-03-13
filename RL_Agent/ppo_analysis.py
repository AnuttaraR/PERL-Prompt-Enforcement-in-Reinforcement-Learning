import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import logging
import time
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("/RL_Agent/ppo_results/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("/RL_Agent/ppo_results/logs", "ppo_analysis.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ppo_analysis")


def load_metrics(metrics_file):
    """Load metrics from JSON file"""
    logger.info(f"Loading metrics from {metrics_file}")
    start_time = time.time()

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        load_time = time.time() - start_time
        logger.info(f"Metrics loaded successfully in {load_time:.2f}s")

        # Check if this is an evaluation metrics file (without training data)
        if "training" not in metrics and any(qt in metrics for qt in ["what", "how", "if_can"]):
            logger.info("Detected evaluation metrics format - converting to standard format")

            # Restructure evaluation metrics to match expected format
            question_types = [key for key in metrics.keys() if key != "action_selections"]

            converted_metrics = {
                "test": {}
            }

            # Move data from question type keys to test structure
            for qt in question_types:
                converted_metrics["test"][qt] = metrics[qt]

            # Add action selections if available
            if "action_selections" in metrics:
                converted_metrics["test"]["action_selections"] = metrics["action_selections"]

            metrics = converted_metrics
            logger.info(f"Converted metrics structure with {len(question_types)} question types")

        # Log some basic stats about the metrics
        if "training" in metrics:
            episodes = len(metrics["training"].get("episode_rewards", []))
            logger.info(f"Training data contains {episodes} episodes")

        if "test" in metrics:
            qt_counts = {qt: len(data.get("rewards", [])) for qt, data in metrics["test"].items()
                         if qt != "action_selections" and "rewards" in data}
            logger.info(f"Test data contains question counts: {qt_counts}")

        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics file: {e}")
        raise


def plot_reward_comparison(metrics, output_dir):
    """Plot comparison of rewards across question types"""
    logger.info("Generating reward comparison plot")
    start_time = time.time()

    # Extract test metrics
    test_metrics = metrics["test"]

    # Filter out non-question type keys like action_selections
    question_types = [qt for qt in test_metrics.keys() if qt != "action_selections" and "rewards" in test_metrics[qt]]
    avg_rewards = [np.mean(test_metrics[qt]["rewards"]) for qt in question_types]

    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(question_types, avg_rewards, color=sns.color_palette("muted", len(question_types)))

    # Add value labels on top of bars
    for bar, reward in zip(bar_plot, avg_rewards):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{reward:.3f}', ha='center', fontsize=10)

    plt.title('Average Reward by Question Type')
    plt.xlabel('Question Type')
    plt.ylabel('Average Reward')
    plt.ylim(0, max(avg_rewards) * 1.2)  # Add some space for the labels
    plt.grid(True, alpha=0.3, axis='y')

    output_path = f"{output_dir}/reward_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Reward comparison plot saved to {output_path}")
    logger.info(f"Plot generation completed in {time.time() - start_time:.2f}s")


def plot_score_comparison(metrics, output_dir):
    """Plot comparison of different scores across question types"""
    logger.info("Generating score comparison plot")
    start_time = time.time()

    # Extract test metrics
    test_metrics = metrics["test"]

    # Filter out non-question type keys like action_selections
    question_types = [qt for qt in test_metrics.keys() if qt != "action_selections" and "rewards" in test_metrics[qt]]
    score_types = ["bert_scores", "rouge_scores", "meteor_scores"]
    score_labels = ["BERTScore", "ROUGE-L", "METEOR"]

    # Prepare data
    data = []
    for qt in question_types:
        for score_type, label in zip(score_types, score_labels):
            if score_type in test_metrics[qt]:
                avg_score = np.mean(test_metrics[qt][score_type])
                data.append({
                    "Question Type": qt,
                    "Metric": label,
                    "Score": avg_score
                })

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Question Type", y="Score", hue="Metric", data=df, palette="viridis")
    plt.title('Evaluation Metrics by Question Type')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars
    for i, metric in enumerate(score_labels):
        for j, qt in enumerate(question_types):
            filtered_data = df[(df["Metric"] == metric) & (df["Question Type"] == qt)]
            if not filtered_data.empty:
                score = filtered_data.iloc[0]["Score"]
                x_pos = j + (i - 1) * 0.3  # Adjust position based on bar group
                plt.text(j, score + 0.02, f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    output_path = f"{output_dir}/score_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Score comparison plot saved to {output_path}")
    logger.info(f"Plot generation completed in {time.time() - start_time:.2f}s")


def plot_learning_curves(metrics, output_dir):
    """Plot smoothed learning curves"""
    logger.info("Generating learning curves")
    start_time = time.time()

    # Extract training metrics
    train_metrics = metrics["training"]

    # Define window size for smoothing
    window_size = 20

    # Smooth rewards
    rewards = train_metrics["episode_rewards"]
    smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    plt.plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed Rewards (window={window_size})')
    plt.title('Training Rewards (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = f"{output_dir}/smoothed_rewards.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Smoothed rewards plot saved to {output_path}")

    # Smooth losses
    if "actor_losses" in train_metrics and train_metrics["actor_losses"]:
        actor_losses = train_metrics["actor_losses"]
        critic_losses = train_metrics["critic_losses"]

        smoothed_actor = pd.Series(actor_losses).rolling(window=window_size, min_periods=1).mean().values
        smoothed_critic = pd.Series(critic_losses).rolling(window=window_size, min_periods=1).mean().values

        plt.figure(figsize=(12, 6))
        plt.plot(actor_losses, alpha=0.2, color='red', label='Raw Actor Loss')
        plt.plot(critic_losses, alpha=0.2, color='blue', label='Raw Critic Loss')
        plt.plot(smoothed_actor, color='red', linewidth=2, label=f'Smoothed Actor Loss')
        plt.plot(smoothed_critic, color='blue', linewidth=2, label=f'Smoothed Critic Loss')
        plt.title('Training Losses (Smoothed)')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = f"{output_dir}/smoothed_losses.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Smoothed losses plot saved to {output_path}")

    logger.info(f"Learning curves generation completed in {time.time() - start_time:.2f}s")


def plot_question_type_progression(metrics, output_dir):
    """Plot how each question type's performance progresses"""
    logger.info("Generating question type progression plot")
    start_time = time.time()

    # Extract training metrics
    train_metrics = metrics["training"]
    question_type_rewards = train_metrics["question_type_rewards"]

    # Define window size for smoothing
    window_size = 20

    plt.figure(figsize=(12, 6))

    # Define colors for each question type
    colors = {'what': 'blue', 'how': 'green', 'if_can': 'orange'}

    for qt, rewards in question_type_rewards.items():
        if rewards:  # Only plot if we have data
            # Create x-axis points (episode numbers)
            episodes = range(len(rewards))

            # Smooth rewards
            smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values

            # Plot
            plt.plot(episodes, rewards, alpha=0.2, color=colors.get(qt, 'gray'), label=f'{qt} (Raw)')
            plt.plot(episodes, smoothed_rewards, linewidth=2, color=colors.get(qt, 'gray'), label=f'{qt} (Smoothed)')

    plt.title('Question Type Reward Progression')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = f"{output_dir}/question_type_progression.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.info(f"Question type progression plot saved to {output_path}")
    logger.info(f"Plot generation completed in {time.time() - start_time:.2f}s")


def plot_action_distribution(metrics, output_dir):
    """Plot distribution of actions selected for each question type"""
    logger.info("Generating action distribution plot")
    start_time = time.time()

    # Check if we have action selection data in the metrics
    if "test" not in metrics or "action_selections" not in metrics["test"]:
        logger.warning("No action selection data available, skipping action distribution plot")
        return

    action_selections = metrics["test"]["action_selections"]
    question_types = list(action_selections.keys())

    # Create a plot for each question type
    for qt in question_types:
        if not action_selections[qt]:
            continue

        # Calculate percentages
        total_actions = sum(action_selections[qt].values())
        action_percentages = {int(action): count / total_actions * 100 for action, count in
                              action_selections[qt].items()}

        # Sort by action number
        sorted_actions = sorted(action_percentages.items())
        actions = [str(a[0]) for a in sorted_actions]
        percentages = [a[1] for a in sorted_actions]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(actions, percentages, color=sns.color_palette("viridis", len(actions)))

        # Add value labels on top of bars
        for bar, percentage in zip(bars, percentages):
            if percentage > 3:  # Only label bars with significant percentages
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{percentage:.1f}%', ha='center', fontsize=8)

        plt.title(f'Action Distribution for {qt.upper()} Questions')
        plt.xlabel('Action ID')
        plt.ylabel('Selection Percentage (%)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)

        output_path = f"{output_dir}/action_distribution_{qt}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Action distribution plot for {qt} questions saved to {output_path}")

    logger.info(f"Action distribution plots completed in {time.time() - start_time:.2f}s")


def generate_summary_report(metrics, output_dir):
    """Generate a text summary report of the analysis"""
    logger.info("Generating summary report")
    start_time = time.time()

    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("PPO MODEL PERFORMANCE SUMMARY")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Training summary
    if "training" in metrics:
        train_metrics = metrics["training"]
        episodes = len(train_metrics.get("episode_rewards", []))

        report_lines.append(f"TRAINING SUMMARY")
        report_lines.append(f"- Episodes completed: {episodes}")

        if episodes > 0:
            last_rewards = train_metrics["episode_rewards"][-min(100, episodes):]
            avg_reward = np.mean(last_rewards)
            report_lines.append(f"- Average reward (last 100 episodes): {avg_reward:.4f}")

        # Question type statistics
        qt_rewards = train_metrics.get("question_type_rewards", {})
        report_lines.append("- Question type counts:")
        for qt, rewards in qt_rewards.items():
            if rewards:
                report_lines.append(f"  * {qt}: {len(rewards)} questions")
                report_lines.append(f"    Average reward: {np.mean(rewards):.4f}")

        report_lines.append("")

    # Test summary
    if "test" in metrics:
        test_metrics = metrics["test"]
        report_lines.append(f"TEST SUMMARY")

        # Overall metrics
        all_rewards = []
        for qt, qt_data in test_metrics.items():
            if "rewards" in qt_data:
                all_rewards.extend(qt_data["rewards"])

        if all_rewards:
            report_lines.append(f"- Overall average reward: {np.mean(all_rewards):.4f}")

        # Per question type metrics
        report_lines.append("- Performance by question type:")
        for qt, qt_data in test_metrics.items():
            if "rewards" not in qt_data:
                continue

            rewards = qt_data["rewards"]
            report_lines.append(f"  * {qt} questions ({len(rewards)} examples):")
            report_lines.append(f"    - Average reward: {np.mean(rewards):.4f}")

            if "bert_scores" in qt_data:
                report_lines.append(f"    - Average BERTScore: {np.mean(qt_data['bert_scores']):.4f}")

            if "rouge_scores" in qt_data:
                report_lines.append(f"    - Average ROUGE-L: {np.mean(qt_data['rouge_scores']):.4f}")

            if "meteor_scores" in qt_data:
                report_lines.append(f"    - Average METEOR: {np.mean(qt_data['meteor_scores']):.4f}")

        report_lines.append("")

    # Configuration summary
    if "config" in metrics:
        config = metrics["config"]
        report_lines.append(f"CONFIGURATION")
        for key, value in config.items():
            report_lines.append(f"- {key}: {value}")

        report_lines.append("")

    # Timing information
    if "timing" in metrics:
        timing = metrics["timing"]
        report_lines.append(f"TIMING")
        for key, value in timing.items():
            report_lines.append(f"- {key}: {value:.2f}s ({value / 60:.2f} minutes)")

        report_lines.append("")

    # Write the report
    report_path = f"{output_dir}/summary_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Summary report saved to {report_path}")
    logger.info(f"Report generation completed in {time.time() - start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Analyze PPO model performance')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to metrics JSON file')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Output directory for analysis plots')
    parser.add_argument('--interactive', action='store_true',
                        help='Show plots interactively instead of saving to files')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up a file handler for this specific analysis
    analysis_log_file = os.path.join(args.output_dir, "analysis.log")
    file_handler = logging.FileHandler(analysis_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 50)
    logger.info(f"Starting PPO model analysis")
    logger.info(f"Metrics file: {args.metrics_file}")
    logger.info(f"Output directory: {args.output_dir}")
    total_start_time = time.time()

    # Load metrics
    metrics = load_metrics(args.metrics_file)

    # Generate plots
    plot_reward_comparison(metrics, args.output_dir)
    plot_score_comparison(metrics, args.output_dir)
    plot_learning_curves(metrics, args.output_dir)
    plot_question_type_progression(metrics, args.output_dir)
    plot_action_distribution(metrics, args.output_dir)

    # Generate summary report
    generate_summary_report(metrics, args.output_dir)

    total_time = time.time() - total_start_time
    logger.info(f"Analysis completed in {total_time:.2f}s ({total_time / 60:.2f} minutes)")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()