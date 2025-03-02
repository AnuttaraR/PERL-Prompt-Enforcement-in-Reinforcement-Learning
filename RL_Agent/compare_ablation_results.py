import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compare_ablation_results(experiment_dir,  mapping_file=None):
    """
    Generate comparison visualizations across all ablation types in the experiment directory.
    """
    # Use mapping file if provided
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            output_directories = json.load(f)

        # Collect metrics from each ablation using the mapping
        ablation_metrics = {}
        for ablation_type, output_dir in output_directories.items():
            metrics_file = os.path.join(output_dir, "metrics.json")
            if not os.path.exists(metrics_file):
                metrics_file = os.path.join(output_dir, "evaluation_metrics.json")

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    ablation_metrics[ablation_type] = metrics
    else:
        # Original behavior - get all ablation directories
        ablation_dirs = [d for d in os.listdir(experiment_dir)
                        if os.path.isdir(os.path.join(experiment_dir, d))]

        # Collect metrics from each ablation
        ablation_metrics = {}
        for ablation_type in ablation_dirs:
            metrics_file = os.path.join(experiment_dir, ablation_type, "metrics.json")
            if not os.path.exists(metrics_file):
                metrics_file = os.path.join(experiment_dir, ablation_type, "evaluation_metrics.json")

            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    ablation_metrics[ablation_type] = metrics

    # Ensure we have data to compare
    if not ablation_metrics:
        print("ERROR: No metrics found for any ablation type")
        return

    # Create comparison visualizations directory
    comparison_dir = os.path.join(experiment_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Compare Average Rewards by Question Type
    compare_rewards_by_question_type(ablation_metrics, comparison_dir)

    # 2. Compare Action Selections
    compare_action_selections(ablation_metrics, comparison_dir)

    # 3. Compare Evaluation Metrics (BERT, ROUGE, METEOR)
    compare_evaluation_metrics(ablation_metrics, comparison_dir)

    # 4. Generate Summary Report
    generate_summary_report(ablation_metrics, comparison_dir)

    print(f"Comparison visualizations saved to {comparison_dir}")


def compare_rewards_by_question_type(ablation_metrics, output_dir):
    """Compare average rewards by question type across ablations"""
    # Prepare data for plotting
    data = []

    for ablation_type, metrics in ablation_metrics.items():
        if "test" in metrics:
            for qt in ["what", "how", "if_can"]:
                if qt in metrics["test"] and "rewards" in metrics["test"][qt]:
                    rewards = metrics["test"][qt]["rewards"]
                    avg_reward = np.mean(rewards) if rewards else 0

                    data.append({
                        "Ablation Type": ablation_type,
                        "Question Type": qt,
                        "Average Reward": avg_reward
                    })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Ablation Type", y="Average Reward", hue="Question Type", data=df)
    plt.title('Average Rewards by Question Type Across Ablations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=300)
    plt.close()


def compare_action_selections(ablation_metrics, output_dir):
    """Compare action selections across ablations"""
    # For each question type, create a plot showing action distribution
    for qt in ["what", "how", "if_can"]:
        data = []

        for ablation_type, metrics in ablation_metrics.items():
            if "test" in metrics and qt in metrics["test"]:
                if "action_selections" in metrics["test"][qt]:
                    action_selections = metrics["test"][qt]["action_selections"]

                    # Calculate percentages
                    total_actions = sum(int(count) for count in action_selections.values())

                    if total_actions > 0:
                        for action_id, count in action_selections.items():
                            percentage = (int(count) / total_actions) * 100
                            data.append({
                                "Ablation Type": ablation_type,
                                "Action ID": action_id,
                                "Percentage": percentage
                            })

        if data:
            # Create DataFrame
            df = pd.DataFrame(data)

            # Plot
            plt.figure(figsize=(14, 8))
            sns.barplot(x="Ablation Type", y="Percentage", hue="Action ID", data=df)
            plt.title(f'Action Selection Distribution for {qt.upper()} Questions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"action_distribution_{qt}.png"), dpi=300)
            plt.close()


def compare_evaluation_metrics(ablation_metrics, output_dir):
    """Compare evaluation metrics across ablations"""
    # Metrics to compare
    metrics_list = ["bert_scores", "rouge_scores", "meteor_scores"]
    metric_names = ["BERTScore", "ROUGE-L", "METEOR"]

    # Prepare data for plotting
    data = []

    for ablation_type, metrics in ablation_metrics.items():
        if "test" in metrics:
            for qt in ["what", "how", "if_can"]:
                if qt in metrics["test"]:
                    for metric_key, metric_name in zip(metrics_list, metric_names):
                        if metric_key in metrics["test"][qt]:
                            scores = metrics["test"][qt][metric_key]
                            avg_score = np.mean(scores) if scores else 0

                            data.append({
                                "Ablation Type": ablation_type,
                                "Question Type": qt,
                                "Metric": metric_name,
                                "Score": avg_score
                            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot for each metric
    for metric_name in metric_names:
        metric_df = df[df["Metric"] == metric_name]

        plt.figure(figsize=(14, 8))
        sns.barplot(x="Ablation Type", y="Score", hue="Question Type", data=metric_df)
        plt.title(f'Average {metric_name} Across Ablations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name.lower()}_comparison.png"), dpi=300)
        plt.close()


def generate_summary_report(ablation_metrics, output_dir):
    """Generate a summary report of comparisons"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ACTION SPACE ABLATION STUDY COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summarize average rewards by question type
    report_lines.append("AVERAGE REWARDS BY QUESTION TYPE")
    report_lines.append("-" * 40)

    # Table header
    header = f"{'Ablation Type':<20} | {'What':<10} | {'How':<10} | {'If/Can':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for ablation_type, metrics in ablation_metrics.items():
        if "test" in metrics:
            what_avg = np.mean(metrics["test"]["what"]["rewards"]) if "what" in metrics["test"] and "rewards" in \
                                                                      metrics["test"]["what"] else 0
            how_avg = np.mean(metrics["test"]["how"]["rewards"]) if "how" in metrics["test"] and "rewards" in \
                                                                    metrics["test"]["how"] else 0
            ifcan_avg = np.mean(metrics["test"]["if_can"]["rewards"]) if "if_can" in metrics["test"] and "rewards" in \
                                                                         metrics["test"]["if_can"] else 0

            row = f"{ablation_type:<20} | {what_avg:<10.4f} | {how_avg:<10.4f} | {ifcan_avg:<10.4f}"
            report_lines.append(row)

    report_lines.append("")

    # Summarize primary action selections
    report_lines.append("PRIMARY ACTION SELECTIONS")
    report_lines.append("-" * 40)

    # Table header
    header = f"{'Ablation Type':<20} | {'What':<10} | {'How':<10} | {'If/Can':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for ablation_type, metrics in ablation_metrics.items():
        if "test" in metrics:
            what_action = "N/A"
            how_action = "N/A"
            ifcan_action = "N/A"

            if "what" in metrics["test"] and "action_selections" in metrics["test"]["what"]:
                actions = metrics["test"]["what"]["action_selections"]
                if actions:
                    what_action = max(actions.items(), key=lambda x: int(x[1]))[0]

            if "how" in metrics["test"] and "action_selections" in metrics["test"]["how"]:
                actions = metrics["test"]["how"]["action_selections"]
                if actions:
                    how_action = max(actions.items(), key=lambda x: int(x[1]))[0]

            if "if_can" in metrics["test"] and "action_selections" in metrics["test"]["if_can"]:
                actions = metrics["test"]["if_can"]["action_selections"]
                if actions:
                    ifcan_action = max(actions.items(), key=lambda x: int(x[1]))[0]

            row = f"{ablation_type:<20} | {what_action:<10} | {how_action:<10} | {ifcan_action:<10}"
            report_lines.append(row)

    # Write the report
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare ablation study results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to ablation experiment directory')
    parser.add_argument('--mapping_file', type=str, default=None,
                        help='Path to output directory mapping file')

    args = parser.parse_args()
    compare_ablation_results(args.experiment_dir, args.mapping_file)