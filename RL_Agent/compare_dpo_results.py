import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compare_dpo_results(experiment_dir, mapping_file=None):
    """
    Generate comparison visualizations across all ablation types in the experiment directory.
    """
    # Use mapping file if provided
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            output_directories = json.load(f)

        # Collect results from each ablation using the mapping
        ablation_results = {}
        for ablation_type, output_dir in output_directories.items():
            results_file = os.path.join(output_dir, "action_minimal_evaluation_results.json")

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    ablation_results[ablation_type] = results
    else:
        # Original behavior - get all ablation directories
        ablation_dirs = [d for d in os.listdir(experiment_dir)
                         if os.path.isdir(os.path.join(experiment_dir, d))]

        # Collect results from each ablation
        ablation_results = {}
        for ablation_type in ablation_dirs:
            results_file = os.path.join(experiment_dir, ablation_type, "action_minimal_evaluation_results.json")

            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    ablation_results[ablation_type] = results

    # Ensure we have data to compare
    if not ablation_results:
        print("ERROR: No results found for any ablation type")
        return

    # Create comparison visualizations directory
    comparison_dir = os.path.join(experiment_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Compare Accuracy by Question Type
    compare_accuracy_by_question_type(ablation_results, comparison_dir)

    # 2. Compare Top Actions Selected
    compare_top_actions(ablation_results, comparison_dir)

    # 3. Generate Summary Report
    generate_summary_report(ablation_results, comparison_dir)

    print(f"Comparison visualizations saved to {comparison_dir}")


def compare_accuracy_by_question_type(ablation_results, output_dir):
    """Compare accuracy by question type across ablations"""
    # Prepare data for plotting
    data = []

    for ablation_type, results in ablation_results.items():
        for qt in ["what", "how", "if_can", "overall"]:
            if qt in results:
                accuracy = results[qt].get("accuracy", 0)

                data.append({
                    "Ablation Type": ablation_type,
                    "Question Type": qt,
                    "Accuracy (%)": accuracy
                })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Ablation Type", y="Accuracy (%)", hue="Question Type", data=df)
    plt.title('DPO Accuracy by Question Type Across Ablations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()


def compare_top_actions(ablation_results, output_dir):
    """Compare top actions selected across ablations"""
    # For each question type, create a plot showing top action distribution
    for qt in ["what", "how", "if_can"]:
        data = []

        for ablation_type, results in ablation_results.items():
            if qt in results and "top_actions" in results[qt]:
                top_actions = results[qt]["top_actions"]

                for action_data in top_actions:
                    data.append({
                        "Ablation Type": ablation_type,
                        "Action": f"{action_data['action_id']}: {action_data['description'][:20]}...",
                        "Percentage (%)": action_data["percentage"]
                    })

        if data:
            # Create DataFrame
            df = pd.DataFrame(data)

            # Plot
            plt.figure(figsize=(16, 10))
            sns.barplot(x="Ablation Type", y="Percentage (%)", hue="Action", data=df)
            plt.title(f'Top Actions Selected for {qt.upper()} Questions Across Ablations')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"top_actions_{qt}.png"), dpi=300)
            plt.close()


def generate_summary_report(ablation_results, output_dir):
    """Generate a summary report of comparisons"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DPO ABLATION STUDY COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summarize accuracy by question type
    report_lines.append("ACCURACY BY QUESTION TYPE")
    report_lines.append("-" * 40)

    # Table header
    header = f"{'Ablation Type':<20} | {'What':<10} | {'How':<10} | {'If/Can':<10} | {'Overall':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for ablation_type, results in ablation_results.items():
        what_acc = results.get("what", {}).get("accuracy", 0)
        how_acc = results.get("how", {}).get("accuracy", 0)
        ifcan_acc = results.get("if_can", {}).get("accuracy", 0)
        overall_acc = results.get("overall", {}).get("accuracy", 0)

        row = f"{ablation_type:<20} | {what_acc:<10.2f} | {how_acc:<10.2f} | {ifcan_acc:<10.2f} | {overall_acc:<10.2f}"
        report_lines.append(row)

    report_lines.append("")

    # Summarize top actions for each question type
    for qt in ["what", "how", "if_can"]:
        report_lines.append(f"TOP ACTIONS FOR {qt.upper()} QUESTIONS")
        report_lines.append("-" * 40)

        for ablation_type, results in ablation_results.items():
            report_lines.append(f"Ablation: {ablation_type}")

            if qt in results and "top_actions" in results[qt]:
                top_actions = results[qt]["top_actions"]

                for i, action_data in enumerate(top_actions, 1):
                    action_id = action_data["action_id"]
                    description = action_data["description"]
                    percentage = action_data["percentage"]

                    report_lines.append(f"  {i}. Action {action_id}: {description} ({percentage:.1f}%)")
            else:
                report_lines.append("  No data available")

            report_lines.append("")

    # Write the report
    with open(os.path.join(output_dir, "comparison_summary.txt"), "w") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare DPO ablation study results')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to ablation experiment directory')
    parser.add_argument('--mapping_file', type=str, default=None,
                        help='Path to output directory mapping file')

    args = parser.parse_args()
    compare_dpo_results(args.experiment_dir, args.mapping_file)