import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
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
logger = logging.getLogger("ablation_analysis")


class AblationAnalyzer:
    """Class for analyzing ablation study results"""

    def __init__(self, ppo_experiment_dir=None, dpo_experiment_dir=None, output_dir="ablation_analysis"):
        """
        Initialize the ablation analyzer.

        Args:
            ppo_experiment_dir: Directory containing PPO ablation experiment results
            dpo_experiment_dir: Directory containing DPO ablation experiment results
            output_dir: Directory to save analysis results
        """
        self.ppo_experiment_dir = ppo_experiment_dir
        self.dpo_experiment_dir = dpo_experiment_dir
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results containers
        self.ppo_results = {}
        self.dpo_results = {}

        print("Ablation analyzer initialized")

    def load_ppo_ablations(self, mapping_file=None):
        """
        Load PPO ablation results from experiment directory.

        Args:
            mapping_file: Optional path to mapping file for ablation directories
        """
        if not self.ppo_experiment_dir:
            logger.warning("PPO experiment directory not specified")
            return

        print(f"Loading PPO ablation results from {self.ppo_experiment_dir}")

        base_dir = os.path.join("run_results", "ppo")

        # First check if we're looking at the right directory
        if not os.path.exists(base_dir):
            base_dir = self.ppo_experiment_dir
            if not os.path.exists(base_dir):
                logger.warning(f"PPO directory not found at {base_dir}")
                return

        # Add main model
        main_metrics_file = os.path.join(base_dir, "main", "evaluation", "metrics.json")
        if os.path.exists(main_metrics_file):
            try:
                with open(main_metrics_file, 'r') as f:
                    metrics = json.load(f)
                    self.ppo_results["main"] = metrics
                    print(f"Loaded metrics for main PPO model from {main_metrics_file}")
            except Exception as e:
                logger.error(f"Error loading main PPO metrics: {e}")

        # Load ablation results
        ablations_dir = os.path.join(base_dir, "ablations")
        if os.path.exists(ablations_dir):
            for ablation_type in os.listdir(ablations_dir):
                ablation_path = os.path.join(ablations_dir, ablation_type)
                if os.path.isdir(ablation_path):
                    # Try to load metrics from evaluation directory
                    metrics_file = os.path.join(ablation_path, "evaluation", "metrics.json")

                    # If not found, try looking in analysis subdirectory
                    if not os.path.exists(metrics_file):
                        metrics_file = os.path.join(ablation_path, "analysis", "evaluation_metrics.json")

                    # If still not found, try without the evaluation subdirectory
                    if not os.path.exists(metrics_file):
                        metrics_file = os.path.join(ablation_path, "metrics.json")

                    # Load metrics if found
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                                self.ppo_results[ablation_type] = metrics
                                print(f"Loaded metrics for PPO ablation {ablation_type} from {metrics_file}")
                        except Exception as e:
                            logger.error(f"Error loading metrics from {metrics_file}: {e}")
                    else:
                        # For action_minimal, there's a special directory structure
                        if ablation_type == "action_minimal":
                            # Check both subdirectories
                            for subdir in ["one_general_action", "more_general_actions"]:
                                sub_path = os.path.join(ablation_path, subdir)
                                if os.path.exists(sub_path):
                                    metrics_file = os.path.join(sub_path, "evaluation", "metrics.json")
                                    if os.path.exists(metrics_file):
                                        try:
                                            with open(metrics_file, 'r') as f:
                                                metrics = json.load(f)
                                                self.ppo_results[f"{ablation_type}_{subdir}"] = metrics
                                                print(
                                                    f"Loaded metrics for PPO ablation {ablation_type}_{subdir} from {metrics_file}")
                                        except Exception as e:
                                            logger.error(f"Error loading metrics from {metrics_file}: {e}")

        print(f"Loaded {len(self.ppo_results)} PPO ablation results")

    def load_dpo_ablations(self, mapping_file=None):
        """
        Load DPO ablation results from experiment directory.

        Args:
            mapping_file: Optional path to mapping file for ablation directories
        """
        if not self.dpo_experiment_dir:
            logger.warning("DPO experiment directory not specified")
            return

        print(f"Loading DPO ablation results from {self.dpo_experiment_dir}")

        base_dir = os.path.join("run_results", "dpo")

        # First check if we're looking at the right directory
        if not os.path.exists(base_dir):
            base_dir = self.dpo_experiment_dir
            if not os.path.exists(base_dir):
                logger.warning(f"DPO directory not found at {base_dir}")
                return

        # Add main model
        main_results_file = os.path.join(base_dir, "main", "evaluation", "evaluation_results.json")
        if os.path.exists(main_results_file):
            try:
                with open(main_results_file, 'r') as f:
                    results = json.load(f)
                    self.dpo_results["main"] = results
                    print(f"Loaded results for main DPO model from {main_results_file}")
            except Exception as e:
                logger.error(f"Error loading main DPO results: {e}")

        # Load ablation results
        ablations_dir = os.path.join(base_dir, "ablations")
        if os.path.exists(ablations_dir):
            for ablation_type in os.listdir(ablations_dir):
                ablation_path = os.path.join(ablations_dir, ablation_type)
                if os.path.isdir(ablation_path):
                    # Try to find the evaluation results file
                    eval_dir = os.path.join(ablation_path, "evaluation")

                    if os.path.exists(eval_dir):
                        # Look for files matching the expected pattern
                        results_files = [
                            f"action_{ablation_type}_evaluation_results.json",
                            "action_diversity_reward_evaluation_results.json",
                            "evaluation_results.json"
                        ]

                        results_file = None
                        for filename in os.listdir(eval_dir):
                            if filename.endswith("_evaluation_results.json"):
                                results_file = os.path.join(eval_dir, filename)
                                break

                        # If not found, try the specific patterns
                        if not results_file:
                            for pattern in results_files:
                                file_path = os.path.join(eval_dir, pattern)
                                if os.path.exists(file_path):
                                    results_file = file_path
                                    break

                        # Load results if found
                        if results_file and os.path.exists(results_file):
                            try:
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                    self.dpo_results[ablation_type] = results
                                    print(f"Loaded results for DPO ablation {ablation_type} from {results_file}")
                            except Exception as e:
                                logger.error(f"Error loading results from {results_file}: {e}")
                        else:
                            logger.warning(f"No results file found for DPO ablation {ablation_type}")

        print(f"Loaded {len(self.dpo_results)} DPO ablation results")

    def analyze_ppo_ablations(self):
        """Analyze PPO ablation results and generate visualizations"""
        if not self.ppo_results:
            logger.warning("No PPO ablation results to analyze")
            return

        print("Analyzing PPO ablation results")

        # Create PPO analysis directory
        ppo_dir = os.path.join(self.output_dir, "ppo_analysis")
        os.makedirs(ppo_dir, exist_ok=True)

        # 1. Compare Average Rewards by Question Type
        self._compare_ppo_rewards_by_question_type(ppo_dir)

        # 2. Compare Action Selections
        self._compare_ppo_action_selections(ppo_dir)

        # 3. Compare Evaluation Metrics (BERT, ROUGE, METEOR)
        self._compare_ppo_evaluation_metrics(ppo_dir)

        # 4. Generate Summary Report
        self._generate_ppo_summary_report(ppo_dir)

        print(f"PPO ablation analysis completed. Results saved to {ppo_dir}")

    def analyze_dpo_ablations(self):
        """Analyze DPO ablation results and generate visualizations"""
        if not self.dpo_results:
            logger.warning("No DPO ablation results to analyze")
            return

        print("Analyzing DPO ablation results")

        # Create DPO analysis directory
        dpo_dir = os.path.join(self.output_dir, "dpo_analysis")
        os.makedirs(dpo_dir, exist_ok=True)

        # 1. Compare Accuracy by Question Type
        self._compare_dpo_accuracy_by_question_type(dpo_dir)

        # 2. Compare Top Actions Selected
        self._compare_dpo_top_actions(dpo_dir)

        # 3. Generate Summary Report
        self._generate_dpo_summary_report(dpo_dir)

        print(f"DPO ablation analysis completed. Results saved to {dpo_dir}")

    def _compare_ppo_rewards_by_question_type(self, output_dir):
        """Compare average rewards by question type across PPO ablations"""
        print("Generating PPO reward comparison plot")

        # Prepare data for plotting
        data = []

        for ablation_type, metrics in self.ppo_results.items():
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
        plt.title('Average Rewards by Question Type Across PPO Ablations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_comparison.png"), dpi=300)
        plt.close()

    def _compare_ppo_action_selections(self, output_dir):
        """Compare action selections across PPO ablations"""
        print("Generating PPO action selection comparison plots")

        # For each question type, create a plot showing action distribution
        for qt in ["what", "how", "if_can"]:
            data = []

            for ablation_type, metrics in self.ppo_results.items():
                if "test" in metrics and "action_selections" in metrics["test"]:
                    if qt in metrics["test"]["action_selections"]:
                        action_selections = metrics["test"]["action_selections"][qt]

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

    def _compare_ppo_evaluation_metrics(self, output_dir):
        """Compare evaluation metrics across PPO ablations"""
        print("Generating PPO evaluation metrics comparison plots")

        # Metrics to compare
        metrics_list = ["bert_scores", "rouge_scores", "meteor_scores"]
        metric_names = ["BERTScore", "ROUGE-L", "METEOR"]

        # Prepare data for plotting
        data = []

        for ablation_type, metrics in self.ppo_results.items():
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
            plt.title(f'Average {metric_name} Across PPO Ablations')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric_name.lower()}_comparison.png"), dpi=300)
            plt.close()

    def _generate_ppo_summary_report(self, output_dir):
        """Generate a summary report of PPO ablation comparisons"""
        print("Generating PPO ablation summary report")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PPO ABLATION STUDY COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summarize average rewards by question type
        report_lines.append("AVERAGE REWARDS BY QUESTION TYPE")
        report_lines.append("-" * 40)

        # Table header
        header = f"{'Ablation Type':<20} | {'What':<10} | {'How':<10} | {'If/Can':<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        for ablation_type, metrics in self.ppo_results.items():
            if "test" in metrics:
                what_avg = np.mean(metrics["test"]["what"]["rewards"]) if "what" in metrics["test"] and "rewards" in \
                                                                          metrics["test"]["what"] else 0
                how_avg = np.mean(metrics["test"]["how"]["rewards"]) if "how" in metrics["test"] and "rewards" in \
                                                                        metrics["test"]["how"] else 0
                ifcan_avg = np.mean(metrics["test"]["if_can"]["rewards"]) if "if_can" in metrics[
                    "test"] and "rewards" in metrics["test"]["if_can"] else 0

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

        for ablation_type, metrics in self.ppo_results.items():
            if "test" in metrics and "action_selections" in metrics["test"]:
                what_action = "N/A"
                how_action = "N/A"
                ifcan_action = "N/A"

                if "what" in metrics["test"]["action_selections"]:
                    actions = metrics["test"]["action_selections"]["what"]
                    if actions:
                        what_action = max(actions.items(), key=lambda x: int(x[1]))[0]

                if "how" in metrics["test"]["action_selections"]:
                    actions = metrics["test"]["action_selections"]["how"]
                    if actions:
                        how_action = max(actions.items(), key=lambda x: int(x[1]))[0]

                if "if_can" in metrics["test"]["action_selections"]:
                    actions = metrics["test"]["action_selections"]["if_can"]
                    if actions:
                        ifcan_action = max(actions.items(), key=lambda x: int(x[1]))[0]

                row = f"{ablation_type:<20} | {what_action:<10} | {how_action:<10} | {ifcan_action:<10}"
                report_lines.append(row)

        # Write the report
        with open(os.path.join(output_dir, "ppo_comparison_summary.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    def _compare_dpo_accuracy_by_question_type(self, output_dir):
        """Compare accuracy by question type across DPO ablations"""
        print("Generating DPO accuracy comparison plot")

        # Prepare data for plotting
        data = []

        for ablation_type, results in self.dpo_results.items():
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

    def _compare_dpo_top_actions(self, output_dir):
        """Compare top actions selected across DPO ablations"""
        print("Generating DPO top actions comparison plots")

        # For each question type, create a plot showing top action distribution
        for qt in ["what", "how", "if_can"]:
            data = []

            for ablation_type, results in self.dpo_results.items():
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
                plt.title(f'Top Actions Selected for {qt.upper()} Questions Across DPO Ablations')
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"top_actions_{qt}.png"), dpi=300)
                plt.close()

    def _generate_dpo_summary_report(self, output_dir):
        """Generate a summary report of DPO ablation comparisons"""
        print("Generating DPO ablation summary report")

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

        for ablation_type, results in self.dpo_results.items():
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

            for ablation_type, results in self.dpo_results.items():
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
        with open(os.path.join(output_dir, "dpo_comparison_summary.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    def run_comprehensive_analysis(self):
        """Run both PPO and DPO analyses and generate cross-model comparison"""
        print("Running comprehensive ablation analysis")

        # Analyze PPO ablations
        if self.ppo_results:
            self.analyze_ppo_ablations()

        # Analyze DPO ablations
        if self.dpo_results:
            self.analyze_dpo_ablations()

        # Generate cross-model comparison if both have results
        if self.ppo_results and self.dpo_results:
            self._generate_cross_model_comparison()

    def _generate_cross_model_comparison(self):
        """Generate comparison between PPO and DPO approaches"""
        print("Generating cross-model comparison")

        # Create cross-model comparison directory
        cross_dir = os.path.join(self.output_dir, "cross_model_comparison")
        os.makedirs(cross_dir, exist_ok=True)

        # 1. Compare best performers from each approach
        self._compare_best_performers(cross_dir)

        # 2. Generate summary report
        self._generate_cross_model_report(cross_dir)

    def _compare_best_performers(self, output_dir):
        """Compare the best performing configurations from PPO and DPO"""
        # Find best PPO configuration
        best_ppo = None
        best_ppo_reward = -float("inf")

        for ablation_type, metrics in self.ppo_results.items():
            if "test" in metrics:
                # Calculate average reward across all question types
                avg_reward = 0
                count = 0

                for qt in ["what", "how", "if_can"]:
                    if qt in metrics["test"] and "rewards" in metrics["test"][qt]:
                        avg_reward += np.mean(metrics["test"][qt]["rewards"])
                        count += 1

                if count > 0:
                    avg_reward /= count

                    if avg_reward > best_ppo_reward:
                        best_ppo_reward = avg_reward
                        best_ppo = ablation_type

        # Find best DPO configuration
        best_dpo = None
        best_dpo_accuracy = -float("inf")

        for ablation_type, results in self.dpo_results.items():
            if "overall" in results:
                accuracy = results["overall"].get("accuracy", 0)

                if accuracy > best_dpo_accuracy:
                    best_dpo_accuracy = accuracy
                    best_dpo = ablation_type

        # Create comparison visualizations
        if best_ppo and best_dpo:
            # Compare by question type
            data = []

            # Add PPO data
            ppo_metrics = self.ppo_results[best_ppo]
            for qt in ["what", "how", "if_can"]:
                if "test" in ppo_metrics and qt in ppo_metrics["test"] and "rewards" in ppo_metrics["test"][qt]:
                    reward = np.mean(ppo_metrics["test"][qt]["rewards"])

                    data.append({
                        "Approach": f"PPO ({best_ppo})",
                        "Question Type": qt,
                        "Performance": reward,
                        "Metric": "Reward"
                    })

            # Add DPO data
            dpo_results = self.dpo_results[best_dpo]
            for qt in ["what", "how", "if_can"]:
                if qt in dpo_results:
                    accuracy = dpo_results[qt].get("accuracy", 0)

                    data.append({
                        "Approach": f"DPO ({best_dpo})",
                        "Question Type": qt,
                        "Performance": accuracy / 100,  # Scale to 0-1 for comparison
                        "Metric": "Accuracy"
                    })

            # Create DataFrame
            df = pd.DataFrame(data)

            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x="Question Type", y="Performance", hue="Approach", data=df)
            plt.title('Comparison of Best PPO and DPO Configurations')
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "best_performers_comparison.png"), dpi=300)
            plt.close()

    def _generate_cross_model_report(self, output_dir):
        """Generate a summary report comparing PPO and DPO approaches"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CROSS-MODEL COMPARISON: PPO vs DPO")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summarize PPO findings
        report_lines.append("PPO APPROACH SUMMARY")
        report_lines.append("-" * 40)

        # Find best and worst PPO configurations
        best_ppo = None
        best_ppo_reward = -float("inf")
        worst_ppo = None
        worst_ppo_reward = float("inf")

        for ablation_type, metrics in self.ppo_results.items():
            if "test" in metrics:
                # Calculate average reward across all question types
                avg_reward = 0
                count = 0

                for qt in ["what", "how", "if_can"]:
                    if qt in metrics["test"] and "rewards" in metrics["test"][qt]:
                        avg_reward += np.mean(metrics["test"][qt]["rewards"])
                        count += 1

                if count > 0:
                    avg_reward /= count

                    if avg_reward > best_ppo_reward:
                        best_ppo_reward = avg_reward
                        best_ppo = ablation_type

                    if avg_reward < worst_ppo_reward:
                        worst_ppo_reward = avg_reward
                        worst_ppo = ablation_type

        if best_ppo:
            report_lines.append(f"Best configuration: {best_ppo} (Avg reward: {best_ppo_reward:.4f})")
            report_lines.append(f"Worst configuration: {worst_ppo} (Avg reward: {worst_ppo_reward:.4f})")
            report_lines.append(f"Performance range: {best_ppo_reward - worst_ppo_reward:.4f}")
        else:
            report_lines.append("No PPO configurations to compare")

        report_lines.append("")

        # Summarize DPO findings
        report_lines.append("DPO APPROACH SUMMARY")
        report_lines.append("-" * 40)

        # Find best and worst DPO configurations
        best_dpo = None
        best_dpo_accuracy = -float("inf")
        worst_dpo = None
        worst_dpo_accuracy = float("inf")

        for ablation_type, results in self.dpo_results.items():
            if "overall" in results:
                accuracy = results["overall"].get("accuracy", 0)

                if accuracy > best_dpo_accuracy:
                    best_dpo_accuracy = accuracy
                    best_dpo = ablation_type

                if accuracy < worst_dpo_accuracy:
                    worst_dpo_accuracy = accuracy
                    worst_dpo = ablation_type

        if best_dpo:
            report_lines.append(f"Best configuration: {best_dpo} (Accuracy: {best_dpo_accuracy:.2f}%)")
            report_lines.append(f"Worst configuration: {worst_dpo} (Accuracy: {worst_dpo_accuracy:.2f}%)")
            report_lines.append(f"Performance range: {best_dpo_accuracy - worst_dpo_accuracy:.2f}%")
        else:
            report_lines.append("No DPO configurations to compare")

        report_lines.append("")

        # Compare approaches by question type
        report_lines.append("QUESTION TYPE PERFORMANCE COMPARISON")
        report_lines.append("-" * 40)

        if best_ppo and best_dpo:
            # Table header
            header = f"{'Question Type':<10} | {'PPO (reward)':<15} | {'DPO (accuracy)':<20}"
            report_lines.append(header)
            report_lines.append("-" * len(header))

            ppo_metrics = self.ppo_results[best_ppo]
            dpo_results = self.dpo_results[best_dpo]

            for qt in ["what", "how", "if_can"]:
                ppo_reward = np.mean(ppo_metrics["test"][qt]["rewards"]) if "test" in ppo_metrics and qt in ppo_metrics[
                    "test"] and "rewards" in ppo_metrics["test"][qt] else 0
                dpo_accuracy = dpo_results[qt].get("accuracy", 0) if qt in dpo_results else 0

                row = f"{qt:<10} | {ppo_reward:<15.4f} | {dpo_accuracy:<20.2f}%"
                report_lines.append(row)

        # Write the report
        with open(os.path.join(output_dir, "cross_model_comparison.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze PPO and DPO ablation study results')
    parser.add_argument('--ppo_experiment_dir', type=str, default="run_results/ppo",
                        help='Directory containing PPO ablation experiment results')
    parser.add_argument('--ppo_mapping_file', type=str, default=None,
                        help='Path to PPO ablation mapping file')
    parser.add_argument('--dpo_experiment_dir', type=str, default="run_results/dpo",
                        help='Directory containing DPO ablation experiment results')
    parser.add_argument('--dpo_mapping_file', type=str, default=None,
                        help='Path to DPO ablation mapping file')
    parser.add_argument('--output_dir', type=str, default='ablation_analysis_results',
                        help='Directory to save analysis results')

    args = parser.parse_args()

    # Set default paths if not specified
    ppo_experiment_dir = args.ppo_experiment_dir
    if not ppo_experiment_dir and args.ppo_results_parent:
        ppo_experiment_dir = args.ppo_results_parent

    dpo_experiment_dir = args.dpo_experiment_dir
    if not dpo_experiment_dir and args.dpo_results_parent:
        dpo_experiment_dir = args.dpo_results_parent

    # Create analyzer
    analyzer = AblationAnalyzer(
        ppo_experiment_dir=ppo_experiment_dir,
        dpo_experiment_dir=dpo_experiment_dir,
        output_dir=args.output_dir
    )

    # Load ablation results
    if args.ppo_experiment_dir:
        analyzer.load_ppo_ablations(args.ppo_mapping_file)

    if args.dpo_experiment_dir:
        analyzer.load_dpo_ablations(args.dpo_mapping_file)

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()

    print(f"Ablation analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()