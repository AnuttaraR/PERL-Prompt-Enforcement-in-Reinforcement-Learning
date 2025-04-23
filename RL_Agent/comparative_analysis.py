import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import necessary modules from your codebase
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from RL_Agent.utils.retrieval import retrieve_context
from ppo_model import PPOAgent, load_config, load_dataset, ActorNetwork, CriticNetwork
from dpo_model import DPOTrainer, calculate_answer_scores

# Configure logging
os.makedirs("evaluation_logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("evaluation_logs", "comparative_analysis.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("comparative_analysis")

# Download NLTK resources if needed
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


class ModelEvaluator:
    """Class for evaluating and comparing different RAG approaches"""

    def __init__(self, test_data_path, action_space_path, reward_config_path, output_dir,
                 llm_models=None, embedding_model="e5-base", models_dir=None,
                 ppo_dir=None, dpo_dir=None):
        """
        Initialize the evaluator with paths to data and models.

        Args:
            test_data_path: Path to test dataset
            action_space_path: Path to action space configuration
            reward_config_path: Path to reward configuration
            output_dir: Directory to save evaluation results
            llm_models: List of LLM models to use (defaults to GPT-3.5 and GPT-4)
            embedding_model: Embedding model to use for retrieval
            models_dir: Legacy parameter, kept for compatibility
            ppo_dir: Path to PPO model directory
            dpo_dir: Path to DPO model directory
        """
        self.test_data_path = test_data_path
        self.action_space_path = action_space_path
        self.reward_config_path = reward_config_path
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.llm_models = llm_models or ["gpt-3.5-turbo", "gpt-4"]
        self.embedding_model = embedding_model
        self.ppo_dir = ppo_dir
        self.dpo_dir = dpo_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load configurations
        print("Loading configurations and test data...")
        self.action_space = load_config(action_space_path)
        self.reward_config = load_config(reward_config_path)
        self.test_dataset = load_dataset(test_data_path)

        # Calculate input dimension based on first example
        self.input_dim = len(self.test_dataset[0]["tokens"])

        # Initialize metrics trackers
        self.results = {
            "baseline": {llm: {} for llm in self.llm_models},
            "ppo": {},
            "dpo": {}
        }

        # Initialize evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        print(f"Evaluator initialized with {len(self.test_dataset)} test examples")
        print(f"Using LLM models: {self.llm_models}")
        print(f"Using embedding model: {self.embedding_model}")

    def evaluate_baseline_rag(self, subset_size=None):
        """
        Evaluate baseline RAG pipeline without any optimization.

        Args:
            subset_size: Optional size of test subset to use (for quicker testing)
        """
        print("Starting baseline RAG evaluation...")

        # Use subset of test data if specified
        test_data = self.test_dataset
        if subset_size is not None and subset_size < len(self.test_dataset):
            test_data = self.test_dataset[:subset_size]

        # Track metrics by question type for each LLM
        for llm_model in self.llm_models:
            print(f"Evaluating baseline RAG with {llm_model}...")

            # Initialize results structure
            question_type_metrics = {
                "what": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []},
                "how": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []},
                "if_can": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []}
            }

            # Process each test example
            for i, question_data in enumerate(tqdm(test_data, desc=f"Baseline RAG {llm_model}")):
                # Extract question data
                question = question_data["question"]
                ground_truth = question_data["ground_truth"]
                question_type = question_data["question_type"]

                try:
                    # Retrieve context
                    context = retrieve_context(question, top_k=3)

                    # Generate prompt and answer (baseline prompt)
                    prompt = f"Question: {question}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."
                    answer = generate_answer_from_llm(prompt, model=llm_model)

                    # Calculate scores
                    scores = calculate_answer_scores(answer, ground_truth)
                    reward = sum(scores.values()) / len(scores)

                    # Store metrics
                    question_type_metrics[question_type]["rewards"].append(reward)
                    question_type_metrics[question_type]["bert_scores"].append(scores["bert_score"])
                    question_type_metrics[question_type]["rouge_scores"].append(scores["rouge_score"])
                    question_type_metrics[question_type]["meteor_scores"].append(scores["meteor_score"])

                    # Log progress periodically
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(test_data)} examples")
                except Exception as e:
                    print(f"Error evaluating example {i}: {e}")

            # Store results
            self.results["baseline"][llm_model] = question_type_metrics

            # Calculate and log average metrics
            all_rewards = []
            for qt, metrics in question_type_metrics.items():
                avg_reward = np.mean(metrics["rewards"])
                avg_bert = np.mean(metrics["bert_scores"])
                avg_rouge = np.mean(metrics["rouge_scores"])
                avg_meteor = np.mean(metrics["meteor_scores"])

                all_rewards.extend(metrics["rewards"])

                print(f"{llm_model} Baseline - {qt} questions:")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Avg BERTScore: {avg_bert:.4f}")
                print(f"  Avg ROUGE-L: {avg_rouge:.4f}")
                print(f"  Avg METEOR: {avg_meteor:.4f}")

            print(f"{llm_model} Baseline - Overall average reward: {np.mean(all_rewards):.4f}")

            # Save intermediate results
            self.save_results("baseline")

    def evaluate_ppo_model(self, model_path, subset_size=None):
        """
        Evaluate a trained PPO model.

        Args:
            model_path: Path to the saved PPO model
            subset_size: Optional size of test subset to use
        """
        print(f"Evaluating PPO model from {model_path}...")

        # Use subset of test data if specified
        test_data = self.test_dataset
        if subset_size is not None and subset_size < len(self.test_dataset):
            test_data = self.test_dataset[:subset_size]

        # Load metadata to get the original action counts
        metadata_path = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                original_action_counts = metadata.get("action_counts", {'what': 5, 'how': 5, 'if_can': 5})
                print(f"Using original action counts from metadata: {original_action_counts}")
        else:
            # If metadata doesn't exist, use this fallback
            original_action_counts = {'what': 5, 'how': 5, 'if_can': 5}
            print(f"No metadata found, using fallback action counts: {original_action_counts}")

        # Initialize PPO agent with the correct action counts
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = PPOAgent(
            input_dim=self.input_dim,
            action_space=self.action_space,
            reward_config=self.reward_config,
            device=device
        )

        # Override the action counts to match what was used during training
        agent.action_counts = original_action_counts

        # Re-initialize the actor and critic with the correct output dimensions
        agent.actor = ActorNetwork(agent.input_dim, agent.action_counts).to(agent.device)
        agent.critic = CriticNetwork(agent.input_dim).to(agent.device)

        # Load the trained model
        if agent.load_model(model_path):
            print(f"Successfully loaded PPO model from {model_path}")
        else:
            print(f"Failed to load PPO model from {model_path}")
            return

        # Check if evaluation metrics already exist for this model
        # First check if this is the main PPO model
        model_name = os.path.basename(os.path.dirname(model_path))
        if model_name == "final_ppo_model":
            model_name = "main"

        # For ablation models, use the ablation type
        ablation_dir = os.path.dirname(os.path.dirname(model_path))
        if os.path.basename(ablation_dir) == "model":
            # If the structure is ablations/ablation_type/model/final_model
            ablation_type = os.path.basename(os.path.dirname(ablation_dir))
            model_name = ablation_type

        # Look for existing metrics
        metrics_file = None

        # Check run_results structure
        if model_name == "main":
            metrics_file = os.path.join("run_results", "ppo", "main", "evaluation", "metrics.json")
        else:
            metrics_file = os.path.join("run_results", "ppo", "ablations", model_name, "evaluation", "metrics.json")

        # Load metrics if they exist, otherwise evaluate
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    print(f"Loaded existing metrics for {model_name} from {metrics_file}")

                    # Store in results
                    self.results["ppo"][model_name] = metrics["test"] if "test" in metrics else metrics

                    # Log some metrics
                    self._log_ppo_metrics(model_name, metrics)

                    return metrics
            except Exception as e:
                print(f"Error loading metrics from {metrics_file}: {e}")

        # Evaluate the model
        metrics = agent.evaluate(test_data)

        # Store results
        self.results["ppo"][model_name] = metrics

        # Log metrics
        self._log_ppo_metrics(model_name, {"test": metrics})

        # Save results
        self.save_results("ppo")

        return metrics

    def _log_ppo_metrics(self, model_name, metrics):
        """Helper to log PPO metrics in a consistent way"""
        if "test" in metrics:
            test_metrics = metrics["test"]
        else:
            test_metrics = metrics  # Assume it's already the test metrics

        for qt, qt_metrics in test_metrics.items():
            if qt == "action_selections":
                continue

            if "rewards" in qt_metrics:
                avg_reward = np.mean(qt_metrics["rewards"])
                avg_bert = np.mean(qt_metrics["bert_scores"]) if "bert_scores" in qt_metrics else 0
                avg_rouge = np.mean(qt_metrics["rouge_scores"]) if "rouge_scores" in qt_metrics else 0
                avg_meteor = np.mean(qt_metrics["meteor_scores"]) if "meteor_scores" in qt_metrics else 0

                print(f"PPO {model_name} - {qt} questions:")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  Avg BERTScore: {avg_bert:.4f}")
                print(f"  Avg ROUGE-L: {avg_rouge:.4f}")
                print(f"  Avg METEOR: {avg_meteor:.4f}")

    def evaluate_dpo_model(self, model_path, subset_size=None):
        """
        Evaluate a trained DPO model.

        Args:
            model_path: Path to the saved DPO model
            subset_size: Optional size of test subset to use
        """
        print(f"Evaluating DPO model from {model_path}...")

        # Use subset of test data if specified
        test_data = self.test_dataset
        if subset_size is not None and subset_size < len(self.test_dataset):
            test_data = self.test_dataset[:subset_size]

        # Check if evaluation results already exist for this model
        model_name = os.path.basename(model_path)
        if model_name == "dpo_model":
            model_name = "main"

        # For ablation models, use the ablation type
        ablation_dir = os.path.dirname(os.path.dirname(model_path))
        if os.path.basename(ablation_dir) == "model":
            # If the structure is ablations/ablation_type/model/final_model
            ablation_type = os.path.basename(os.path.dirname(ablation_dir))
            model_name = ablation_type

        # Look for existing results
        results_file = None

        # Check run_results structure
        if model_name == "main":
            results_file = os.path.join("run_results", "dpo", "main", "evaluation", "evaluation_results.json")
        else:
            eval_dir = os.path.join("run_results", "dpo", "ablations", model_name, "evaluation")
            if os.path.exists(eval_dir):
                for file in os.listdir(eval_dir):
                    if file.endswith("_evaluation_results.json"):
                        results_file = os.path.join(eval_dir, file)
                        break

        # Load results if they exist, otherwise evaluate
        if results_file and os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    print(f"Loaded existing results for {model_name} from {results_file}")

                    # Store in results
                    self.results["dpo"][model_name] = results

                    # Log some metrics
                    print(f"DPO {model_name} evaluation results:")
                    for qt, metrics in results.items():
                        if qt == "overall":
                            print(f"  Overall accuracy: {metrics.get('accuracy', 0):.2f}%")
                        else:
                            print(f"  {qt} questions - Accuracy: {metrics.get('accuracy', 0):.2f}%")

                    return results
            except Exception as e:
                print(f"Error loading results from {results_file}: {e}")

        try:
            # Initialize tokenizer
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            # Initialize DPO trainer
            input_dim = 768  # BERT's default embedding dimension
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = DPOTrainer(
                input_dim=input_dim,
                action_space=self.action_space,
                device=device
            )

            # Load the trained model
            if trainer.load_model(model_path):
                print(f"Successfully loaded DPO model from {model_path}")
            else:
                print(f"Failed to load DPO model from {model_path}")
                return

            # Evaluate the model
            results = trainer.evaluate(test_data, tokenizer)

            # Store results
            self.results["dpo"][model_name] = results

            # Log metrics
            print(f"DPO {model_name} evaluation results:")
            for qt, metrics in results.items():
                if qt == "overall":
                    print(f"  Overall accuracy: {metrics.get('accuracy', 0):.2f}%")
                else:
                    print(f"  {qt} questions - Accuracy: {metrics.get('accuracy', 0):.2f}%")

            # Save results
            self.save_results("dpo")

            return results
        except Exception as e:
            print(f"Error evaluating DPO model: {e}")
            return None

    def evaluate_all_models(self, subset_size=None):
        """
        Evaluate baseline RAG, PPO models, and DPO models.

        Args:
            subset_size: Optional size of test subset to use
        """
        print("Starting comprehensive evaluation of all models...")

        # Evaluate baseline RAG
        self.evaluate_baseline_rag(subset_size)

        # This method is now just a wrapper for individual evaluation methods
        # The actual model evaluation logic is in the main function to handle
        # your specific directory structure

        # The actual evaluation of PPO and DPO models is done in the main function
        # by finding and calling evaluate_ppo_model and evaluate_dpo_model directly

        # Generate comparison visualizations
        self.generate_comparisons()

    def save_results(self, model_type):
        """
        Save evaluation results to a JSON file.

        Args:
            model_type: Type of model (baseline, ppo, dpo)
        """

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

        # Process the results structure
        processed_results = process_json_structure(self.results)

        # Save to file
        output_file = os.path.join(self.output_dir, f"{model_type}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results[model_type], f, indent=2)

        print(f"Saved {model_type} results to {output_file}")

    def generate_comparisons(self):
        """Generate comparison visualizations for all evaluated models"""
        print("Generating comparison visualizations...")

        # Create visualizations directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Aggregate metrics for comparison
        metrics = ["bert_scores", "rouge_scores", "meteor_scores", "rewards"]
        question_types = ["what", "how", "if_can"]  # Define question types here for all methods to use

        # Create dataframe for visualization
        comparison_data = []

        # Add baseline RAG data
        for llm_model in self.llm_models:
            if llm_model in self.results["baseline"]:
                baseline_results = self.results["baseline"][llm_model]

                for qt in question_types:
                    if qt in baseline_results:
                        for metric in metrics:
                            if metric in baseline_results[qt]:
                                metric_value = np.mean(baseline_results[qt][metric])

                                comparison_data.append({
                                    "Model": f"Baseline RAG ({llm_model})",
                                    "Question Type": qt,
                                    "Metric": metric,
                                    "Value": metric_value
                                })

        # Add PPO model data
        for model_name, ppo_results in self.results["ppo"].items():
            for qt in question_types:
                if qt in ppo_results and qt != "action_selections":
                    for metric in metrics:
                        if metric in ppo_results[qt]:
                            metric_value = np.mean(ppo_results[qt][metric])

                            comparison_data.append({
                                "Model": f"PPO ({model_name})",
                                "Question Type": qt,
                                "Metric": metric,
                                "Value": metric_value
                            })

        # Add DPO model data (note: DPO uses different metrics structure)
        for model_name, dpo_results in self.results["dpo"].items():
            for qt in question_types:
                if qt in dpo_results:
                    # DPO reports accuracy
                    accuracy = dpo_results[qt].get("accuracy", 0)

                    comparison_data.append({
                        "Model": f"DPO ({model_name})",
                        "Question Type": qt,
                        "Metric": "accuracy",
                        "Value": accuracy
                    })

        # Create DataFrame for visualization
        df = pd.DataFrame(comparison_data)

        # Generate comparison plots
        self._plot_metric_comparison(df, vis_dir)
        self._plot_question_type_comparison(df, vis_dir)

        # Generate summary report
        self._generate_summary_report(df, vis_dir)

        print(f"Comparison visualizations and reports saved to {vis_dir}")

    def _plot_metric_comparison(self, df, output_dir):
        """
        Plot comparison of metrics across models.

        Args:
            df: DataFrame with comparison data
            output_dir: Directory to save plots
        """
        # Get unique metrics and question types from the data
        metrics = df["Metric"].unique()

        # Plot each metric separately
        for metric in metrics:
            metric_df = df[df["Metric"] == metric]

            plt.figure(figsize=(14, 8))
            sns.barplot(x="Model", y="Value", hue="Question Type", data=metric_df)
            plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Models')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_comparison.png"), dpi=300)
            plt.close()

    def _plot_question_type_comparison(self, df, output_dir):
        """
        Plot comparison by question type across models.

        Args:
            df: DataFrame with comparison data
            output_dir: Directory to save plots
        """
        # Get unique question types from the data
        question_types = df["Question Type"].unique()

        # Plot each question type separately
        for qt in question_types:
            qt_df = df[df["Question Type"] == qt]

            plt.figure(figsize=(14, 8))
            sns.barplot(x="Model", y="Value", hue="Metric", data=qt_df)
            plt.title(f'Model Performance for {qt.upper()} Questions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{qt}_comparison.png"), dpi=300)
            plt.close()

    def _generate_summary_report(self, df, output_dir):
        """
        Generate a text summary report of the comparative analysis.

        Args:
            df: DataFrame with comparison data
            output_dir: Directory to save report
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("COMPARATIVE ANALYSIS SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Define question types locally
        question_types = ["what", "how", "if_can"]

        # Overall performance comparison
        report_lines.append("OVERALL PERFORMANCE COMPARISON")
        report_lines.append("-" * 40)

        # Get average rewards by model
        rewards_df = df[df["Metric"] == "rewards"]
        pivot_df = rewards_df.pivot_table(
            values="Value",
            index="Model",
            columns="Question Type",
            aggfunc="mean"
        ).reset_index()

        # Add overall column - properly handling missing question types
        available_question_types = [qt for qt in question_types if qt in pivot_df.columns]
        if available_question_types:
            pivot_df["Overall"] = pivot_df[available_question_types].mean(axis=1)
        else:
            pivot_df["Overall"] = 0  # Default if no question types are available

        # Convert to string table
        table_lines = []
        header = f"{'Model':<30} | {'What':<10} | {'How':<10} | {'If/Can':<10} | {'Overall':<10}"
        table_lines.append(header)
        table_lines.append("-" * len(header))

        for _, row in pivot_df.iterrows():
            model = row["Model"]
            what_val = row.get("what", 0)
            how_val = row.get("how", 0)
            if_can_val = row.get("if_can", 0)
            overall_val = row["Overall"]

            line = f"{model:<30} | {what_val:<10.4f} | {how_val:<10.4f} | {if_can_val:<10.4f} | {overall_val:<10.4f}"
            table_lines.append(line)

        report_lines.extend(table_lines)
        report_lines.append("")

        # Add analysis by question type
        for qt in question_types:
            report_lines.append(f"ANALYSIS FOR {qt.upper()} QUESTIONS")
            report_lines.append("-" * 40)

            qt_df = df[df["Question Type"] == qt]

            # Group by model and metric
            summary = qt_df.groupby(["Model", "Metric"])["Value"].mean().reset_index()

            # For each model, list metrics
            for model in summary["Model"].unique():
                report_lines.append(f"Model: {model}")
                model_metrics = summary[summary["Model"] == model]

                for _, row in model_metrics.iterrows():
                    metric = row["Metric"]
                    value = row["Value"]
                    report_lines.append(f"  - {metric}: {value:.4f}")

                report_lines.append("")

        # Write to file
        report_path = os.path.join(output_dir, "comparative_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparative Analysis of RAG, PPO, and DPO Models')
    parser.add_argument('--test_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/action_space_config.json',
                        help='Path to action space configuration')
    parser.add_argument('--reward_config', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/config/reward_config.json',
                        help='Path to reward configuration')
    parser.add_argument('--models_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--ppo_dir', type=str,
                        default='run_results/ppo/main/model/final_ppo_model',
                        help='Directory with PPO model to evaluate')
    parser.add_argument('--dpo_dir', type=str,
                        default='run_results/dpo/main/model/dpo_model',
                        help='Directory with DPO model to evaluate')
    parser.add_argument('--ppo_ablation_dir', type=str,
                        default='run_results/ppo/ablations',
                        help='Directory containing PPO ablation models')
    parser.add_argument('--dpo_ablation_dir', type=str,
                        default='run_results/dpo/ablations',
                        help='Directory containing DPO ablation models')
    parser.add_argument('--output_dir', type=str, default='comparative_analysis_results',
                        help='Directory to save results')
    parser.add_argument('--llm_models', type=str, nargs='+', default=['gpt-3.5-turbo', 'gpt-4'],
                        help='LLM models to use for evaluation')
    parser.add_argument('--embedding_model', type=str, default='e5-base',
                        help='Embedding model to use for retrieval')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Size of test subset to use (for quicker testing)')
    parser.add_argument('--eval_baseline', action='store_true',
                        help='Evaluate baseline RAG')
    parser.add_argument('--eval_ppo', action='store_true', default=True,
                        help='Evaluate PPO models')
    parser.add_argument('--eval_dpo', action='store_true', default=True,
                        help='Evaluate DPO models')
    parser.add_argument('--eval_all', action='store_true', default=True,
                        help='Evaluate all models')
    parser.add_argument('--ppo_model_path', type=str, default=None,
                        help='Path to specific PPO model to evaluate')
    parser.add_argument('--dpo_model_path', type=str, default=None,
                        help='Path to specific DPO model to evaluate')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        test_data_path=args.test_data,
        action_space_path=args.action_space,
        reward_config_path=args.reward_config,
        models_dir=None,  # We'll use specific paths instead
        output_dir=output_dir,
        llm_models=args.llm_models,
        embedding_model=args.embedding_model,
        ppo_dir=args.ppo_dir,
        dpo_dir=args.dpo_dir
    )

    if args.ppo_ablation_dir and os.path.exists(args.ppo_ablation_dir):
        # Find and evaluate key ablation models
        ablation_types = ["action_what_only", "action_how_only", "action_if_can_only",
                          "action_minimal", "action_unified"]

        for ablation_type in ablation_types:
            # Look for the model in the ablation directory
            ablation_model_path = None

            # Special handling for action_minimal
            if ablation_type == "action_minimal":
                # Look for subdirectories under action_minimal
                minimal_types = ["one_general_action", "more_general_actions"]
                for minimal_type in minimal_types:
                    potential_path = os.path.join(args.ppo_ablation_dir, ablation_type, minimal_type)
                    final_model_path = os.path.join(potential_path, "final_model")

                    if os.path.exists(final_model_path):
                        ablation_model_path = final_model_path
                        print(f"Found action minimal model: {minimal_type}")
                        break

            # Standard path for other ablation types
            if not ablation_model_path:
                # Try direct path
                direct_path = os.path.join(args.ppo_ablation_dir, ablation_type)
                if os.path.exists(direct_path):
                    # Look for final_model subdirectory
                    final_model_path = os.path.join(direct_path, "final_model")
                    if os.path.exists(final_model_path):
                        ablation_model_path = final_model_path

                # If not found, try the timestamped directories
                if not ablation_model_path:
                    # Find directories matching the pattern ablation_type_timestamp
                    matching_dirs = [d for d in os.listdir(args.ppo_ablation_dir)
                                     if os.path.isdir(os.path.join(args.ppo_ablation_dir, d))
                                     and d.startswith(ablation_type + "_")]

                    for dir_name in matching_dirs:
                        final_model_path = os.path.join(args.ppo_ablation_dir, dir_name, "final_model")
                        if os.path.exists(final_model_path):
                            ablation_model_path = final_model_path
                            break

            # If we found a model, evaluate it
            if ablation_model_path:
                print(f"Evaluating PPO ablation model: {ablation_type}")
                evaluator.evaluate_ppo_model(ablation_model_path, args.subset_size)
            else:
                logger.warning(f"Could not find PPO ablation model for: {ablation_type}")

    # Evaluate top DPO ablation models
    if args.dpo_ablation_dir and os.path.exists(args.dpo_ablation_dir):
        # Find and evaluate key DPO ablation models
        ablation_types = [
            "action_diversity_reward",
            "action_minimal",
            "balanced_pairs",
            "baseline",
            "beta_high",
            "beta_low",
            "higher_temperature",
            "no_keep_unchanged",
            "question_weighted_loss"
        ]

        for ablation_type in ablation_types:
            # Look for the model in the ablation directory
            ablation_model_path = None

            # Try direct path
            direct_path = os.path.join(args.dpo_ablation_dir, ablation_type)
            if os.path.exists(direct_path):
                # Look for final_model subdirectory
                final_model_path = os.path.join(direct_path, "final_model")
                if os.path.exists(final_model_path):
                    ablation_model_path = final_model_path

            # If not found, try the timestamped directories
            if not ablation_model_path:
                # Find directories matching the pattern ablation_type_timestamp
                matching_dirs = [d for d in os.listdir(args.dpo_ablation_dir)
                                 if os.path.isdir(os.path.join(args.dpo_ablation_dir, d))
                                 and d.startswith(ablation_type + "_")]

                for dir_name in matching_dirs:
                    final_model_path = os.path.join(args.dpo_ablation_dir, dir_name, "final_model")
                    if os.path.exists(final_model_path):
                        ablation_model_path = final_model_path
                        break

            # If we found a model, evaluate it
            if ablation_model_path:
                print(f"Evaluating DPO ablation model: {ablation_type}")
                evaluator.evaluate_dpo_model(ablation_model_path, args.subset_size)
            else:
                logger.warning(f"Could not find DPO ablation model for: {ablation_type}")

        if args.dpo_ablation_dir and os.path.exists(args.dpo_ablation_dir):
            # Find and evaluate key DPO ablation models
            ablation_types = [
                "action_diversity_reward",
                "action_minimal",
                "balanced_pairs",
                "baseline",
                "beta_high",
                "beta_low",
                "higher_temperature",
                "no_keep_unchanged",
                "question_weighted_loss"
            ]

            for ablation_type in ablation_types:
                # Look for the model in the ablation directory
                ablation_model_path = None

                # Check the specific path with evaluation and model folders
                full_ablation_path = os.path.join(args.dpo_ablation_dir, ablation_type)

                # Check final_model path
                potential_final_model_paths = [
                    os.path.join(full_ablation_path, "model", "final_model"),
                    os.path.join(full_ablation_path, "final_model")
                ]

                for potential_path in potential_final_model_paths:
                    if os.path.exists(os.path.join(potential_path, "actor.pth")):
                        ablation_model_path = potential_path
                        break

                # If not found, try best_model path
                if not ablation_model_path:
                    potential_best_model_paths = [
                        os.path.join(full_ablation_path, "model", "best_model"),
                        os.path.join(full_ablation_path, "best_model")
                    ]

                    for potential_path in potential_best_model_paths:
                        if os.path.exists(os.path.join(potential_path, "actor.pth")):
                            ablation_model_path = potential_path
                            break

                # If still not found, try timestamped directories
                if not ablation_model_path:
                    # Find directories matching the pattern ablation_type_timestamp
                    matching_dirs = [d for d in os.listdir(args.dpo_ablation_dir)
                                     if os.path.isdir(os.path.join(args.dpo_ablation_dir, d))
                                     and d.startswith(ablation_type + "_")]

                    for dir_name in matching_dirs:
                        # Check potential paths in timestamped directory
                        potential_paths = [
                            os.path.join(args.dpo_ablation_dir, dir_name, "model", "final_model"),
                            os.path.join(args.dpo_ablation_dir, dir_name, "final_model"),
                            os.path.join(args.dpo_ablation_dir, dir_name, "model", "best_model"),
                            os.path.join(args.dpo_ablation_dir, dir_name, "best_model")
                        ]

                        for potential_path in potential_paths:
                            if os.path.exists(os.path.join(potential_path, "actor.pth")):
                                ablation_model_path = potential_path
                                break

                        if ablation_model_path:
                            break

                # If we found a model, evaluate it
                if ablation_model_path:
                    print(f"Evaluating DPO ablation model: {ablation_type}")
                    print(f"Model path: {ablation_model_path}")
                    evaluator.evaluate_dpo_model(ablation_model_path, args.subset_size)
                else:
                    logger.warning(f"Could not find DPO ablation model for: {ablation_type}")
    else:
        if args.eval_baseline:
            evaluator.evaluate_baseline_rag(args.subset_size)

        if args.eval_ppo:
            if args.ppo_model_path:
                evaluator.evaluate_ppo_model(args.ppo_model_path, args.subset_size)
            elif args.ppo_dir and os.path.exists(args.ppo_dir):
                evaluator.evaluate_ppo_model(args.ppo_dir, args.subset_size)
            else:
                logger.warning("No PPO model path specified for evaluation")

        if args.eval_dpo:
            if args.dpo_model_path:
                evaluator.evaluate_dpo_model(args.dpo_model_path, args.subset_size)
            elif args.dpo_dir and os.path.exists(args.dpo_dir):
                evaluator.evaluate_dpo_model(args.dpo_dir, args.subset_size)
            else:
                logger.warning("No DPO model path specified for evaluation")

    # Generate comparison visualizations
    evaluator.generate_comparisons()

    print(f"Comparative analysis completed. Results saved to {output_dir}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")