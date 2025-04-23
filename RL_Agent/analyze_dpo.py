import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dpo_analysis")


def load_results(results_path):
    """Load evaluation results from JSON file"""
    logger.info(f"Loading results from {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def load_metadata(model_path):
    """Load model metadata from JSON file"""
    metadata_path = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning(f"No metadata found at {metadata_path}")
        return None

    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def plot_training_curves(metadata, output_dir):
    """Plot training curves from metadata"""
    if not metadata or "train_losses" not in metadata:
        logger.warning("No training data found in metadata")
        return

    logger.info("Plotting training curves")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(metadata["train_losses"], marker='o')
    plt.title('DPO Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "action_minimal_dpo_loss_curve.png"))
    plt.close()

    # Plot validation accuracy curve
    if "validation_accuracies" in metadata:
        plt.figure(figsize=(10, 6))
        plt.plot(metadata["validation_accuracies"], marker='o', color='green')
        plt.title('DPO Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "action_minimal_dpo_accuracy_curve.png"))
        plt.close()


def plot_question_type_performance(results, output_dir):
    """Plot performance by question type"""
    logger.info("Plotting performance by question type")

    question_types = ["what", "how", "if_can"]
    accuracies = []

    for qt in question_types:
        if qt in results:
            accuracies.append(results[qt]["accuracy"])
        else:
            accuracies.append(0)

    # Add overall accuracy
    question_types.append("overall")
    accuracies.append(results.get("overall", {}).get("accuracy", 0))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(question_types, accuracies, color=sns.color_palette("muted", len(question_types)))

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', fontsize=10)

    plt.title('DPO Model Accuracy by Question Type')
    plt.xlabel('Question Type')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')

    plt.savefig(os.path.join(output_dir, "question_type_accuracy.png"))
    plt.close()


def plot_action_distribution(results, output_dir):
    """Plot distribution of selected actions for each question type"""
    logger.info("Plotting action distribution")

    for qt in ["what", "how", "if_can"]:
        if qt not in results or "top_actions" not in results[qt]:
            continue

        actions = results[qt]["top_actions"]
        if not actions:
            continue

        # Sort by percentage
        actions = sorted(actions, key=lambda x: x["percentage"], reverse=True)

        labels = [action["description"][:30] + "..." if len(action["description"]) > 30
                  else action["description"] for action in actions]
        percentages = [action["percentage"] for action in actions]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, percentages, color=sns.color_palette("viridis", len(actions)))

        # Add value labels
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{pct:.1f}%', ha='center', fontsize=8)

        plt.title(f'Action Distribution for {qt.upper()} Questions')
        plt.xlabel('Action')
        plt.ylabel('Selection Percentage (%)')
        plt.ylim(0, max(percentages) * 1.2)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"action_distribution_{qt}.png"))
        plt.close()


def generate_summary_report(results, metadata, config, output_dir):
    """Generate a text summary report of the analysis"""
    logger.info("Generating summary report")

    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("DPO MODEL PERFORMANCE SUMMARY")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Overall performance
    overall = results.get("overall", {})
    report_lines.append("OVERALL PERFORMANCE")
    report_lines.append(f"- Accuracy: {overall.get('accuracy', 0):.2f}%")
    report_lines.append(f"- Total examples: {overall.get('total', 0)}")
    report_lines.append(f"- Correct predictions: {overall.get('correct', 0)}")
    report_lines.append("")

    # Performance by question type
    report_lines.append("PERFORMANCE BY QUESTION TYPE")
    for qt in ["what", "how", "if_can"]:
        if qt in results:
            qt_results = results[qt]
            report_lines.append(f"- {qt.upper()} Questions")
            report_lines.append(f"  * Accuracy: {qt_results.get('accuracy', 0):.2f}%")
            report_lines.append(f"  * Total examples: {qt_results.get('total', 0)}")
            report_lines.append(f"  * Correct predictions: {qt_results.get('correct', 0)}")

            # Top actions
            if "top_actions" in qt_results and qt_results["top_actions"]:
                report_lines.append("  * Top actions:")
                for action in qt_results["top_actions"]:
                    report_lines.append(f"    - {action['description']}: {action['percentage']:.1f}%")

            report_lines.append("")

    # Training information
    if metadata:
        report_lines.append("TRAINING INFORMATION")
        if "train_losses" in metadata:
            final_loss = metadata["train_losses"][-1] if metadata["train_losses"] else None
            report_lines.append(
                f"- Final training loss: {final_loss:.4f}" if final_loss else "- No loss data available")

        if "validation_accuracies" in metadata:
            final_acc = metadata["validation_accuracies"][-1] if metadata["validation_accuracies"] else None
            report_lines.append(
                f"- Final validation accuracy: {final_acc:.2f}%" if final_acc else "- No validation data available")

        report_lines.append("")

    # Configuration
    if config:
        report_lines.append("CONFIGURATION")
        report_lines.append(f"- Number of preference pairs: {config.get('num_preference_pairs', 'unknown')}")
        report_lines.append(f"- Training epochs: {config.get('epochs', 'unknown')}")
        report_lines.append(f"- Batch size: {config.get('batch_size', 'unknown')}")
        report_lines.append(f"- Learning rate: {config.get('learning_rate', 'unknown')}")
        report_lines.append(f"- Beta (regularization): {config.get('beta', 'unknown')}")
        report_lines.append("")

        if "training_time" in config:
            report_lines.append("TIMING")
            training_time = config.get("training_time", 0)
            evaluation_time = config.get("evaluation_time", 0)
            report_lines.append(f"- Training time: {training_time:.2f}s ({training_time / 60:.2f} minutes)")
            report_lines.append(f"- Evaluation time: {evaluation_time:.2f}s ({evaluation_time / 60:.2f} minutes)")
            report_lines.append("")

    # Write the report
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze DPO model performance')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to evaluation results JSON file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file (optional)')
    parser.add_argument('--output_dir', type=str, default='dpo_analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure file logging
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "analysis.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 50)
    logger.info("Starting DPO model analysis")
    logger.info(f"Results file: {args.results}")
    logger.info(f"Model directory: {args.model}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    results = load_results(args.results)
    metadata = load_metadata(args.model)

    # Load config if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    # Generate plots
    plot_training_curves(metadata, args.output_dir)
    plot_question_type_performance(results, args.output_dir)
    plot_action_distribution(results, args.output_dir)

    # Generate summary report
    generate_summary_report(results, metadata, config, args.output_dir)

    logger.info(f"Analysis completed. Results saved to {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()