import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import nltk
from transformers import BertTokenizer, BertModel
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import torch

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import utility functions
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from RL_Agent.utils.retrieval import retrieve_context

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=print,
)

# Download NLTK resources if needed
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


def load_dataset(dataset_path):
    """Load dataset from a JSON file"""
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded dataset with {len(data)} items")
    return data


def calculate_metrics(answer, ground_truth):
    """Calculate evaluation metrics between the generated answer and ground truth"""
    # BERT Score
    P, R, F1 = bert_score([answer], [ground_truth], lang="en", return_hash=False)
    bert_score_val = F1.item()

    # ROUGE Score
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = rouge_scorer_obj.score(ground_truth, answer)
    rouge_score_val = rouge['rougeL'].fmeasure

    # METEOR Score
    answer_tokens = nltk.word_tokenize(answer.lower())
    ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
    meteor_score_val = meteor_score([ground_truth_tokens], answer_tokens)

    # F1 Score based on token overlap
    answer_tokens_set = set(answer_tokens)
    ground_truth_tokens_set = set(ground_truth_tokens)

    common = len(answer_tokens_set.intersection(ground_truth_tokens_set))
    precision = common / len(answer_tokens_set) if answer_tokens_set else 0
    recall = common / len(ground_truth_tokens_set) if ground_truth_tokens_set else 0

    f1_score_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "bert_score": bert_score_val,
        "rouge_score": rouge_score_val,
        "meteor_score": meteor_score_val,
        "f1_score": f1_score_val,
        "avg_score": (bert_score_val + rouge_score_val + meteor_score_val + f1_score_val) / 4
    }


def run_baseline_rag(test_dataset, llm_model, top_k=3, output_dir="baseline_results"):
    """
    Run baseline RAG pipeline on test dataset.

    Args:
        test_dataset: List of test examples
        llm_model: LLM model to use (e.g., gpt-3.5-turbo, gpt-4)
        top_k: Number of context chunks to retrieve
        output_dir: Directory to save results

    Returns:
        Dictionary of results by question type
    """
    print(f"Running baseline RAG with {llm_model}, retrieving top {top_k} chunks")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    results = {
        "what": {"rewards": [], "answers": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": [],
                 "f1_scores": []},
        "how": {"rewards": [], "answers": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": [],
                "f1_scores": []},
        "if_can": {"rewards": [], "answers": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": [],
                   "f1_scores": []}
    }

    # Track overall performance by question
    question_results = []

    # Process each test example
    for i, question_data in enumerate(tqdm(test_dataset, desc=f"Baseline RAG ({llm_model})")):
        # Extract question data
        question = question_data["question"]
        ground_truth = question_data["ground_truth"]
        question_type = question_data["question_type"]
        print("ITERATION ", i+1)

        start_time = time.time()

        try:
            # Retrieve context
            context = retrieve_context(question, top_k=top_k)

            # Generate prompt and answer
            prompt = f"Question: {question}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."
            print(f"Prompt: {prompt[:50].encode('utf-8', 'ignore').decode('utf-8')}...")
            answer = generate_answer_from_llm(prompt, model=llm_model)
            print(f"Answer: {answer[:50].encode('utf-8', 'ignore').decode('utf-8')}...")

            # Calculate metrics
            metrics = calculate_metrics(answer, ground_truth)
            reward = metrics["avg_score"]

            # Store results
            results[question_type]["rewards"].append(reward)
            results[question_type]["answers"].append(answer)
            results[question_type]["bert_scores"].append(metrics["bert_score"])
            results[question_type]["rouge_scores"].append(metrics["rouge_score"])
            results[question_type]["meteor_scores"].append(metrics["meteor_score"])
            results[question_type]["f1_scores"].append(metrics["f1_score"])

            # Store individual question result
            question_results.append({
                "question": question,
                "question_type": question_type,
                "ground_truth": ground_truth,
                "answer": answer,
                "metrics": metrics,
                "processing_time": time.time() - start_time
            })

            # Log progress periodically
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} questions")

                # Calculate average metrics so far
                for qt in ["what", "how", "if_can"]:
                    if results[qt]["rewards"]:
                        avg_reward = np.mean(results[qt]["rewards"])
                        avg_bert = np.mean(results[qt]["bert_scores"])
                        avg_rouge = np.mean(results[qt]["rouge_scores"])

                        print(
                            f"{qt} questions - Avg Reward: {avg_reward:.4f}, BERTScore: {avg_bert:.4f}, ROUGE: {avg_rouge:.4f}")

        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")

    # Calculate overall metrics
    overall_results = {"overall": {}}

    all_rewards = []
    all_bert_scores = []
    all_rouge_scores = []
    all_meteor_scores = []
    all_f1_scores = []

    for qt in ["what", "how", "if_can"]:
        if results[qt]["rewards"]:
            avg_reward = np.mean(results[qt]["rewards"])
            avg_bert = np.mean(results[qt]["bert_scores"])
            avg_rouge = np.mean(results[qt]["rouge_scores"])
            avg_meteor = np.mean(results[qt]["meteor_scores"])
            avg_f1 = np.mean(results[qt]["f1_scores"])

            print(f"{qt} questions - Final Metrics:")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg BERTScore: {avg_bert:.4f}")
            print(f"  Avg ROUGE-L: {avg_rouge:.4f}")
            print(f"  Avg METEOR: {avg_meteor:.4f}")
            print(f"  Avg F1 Score: {avg_f1:.4f}")

            all_rewards.extend(results[qt]["rewards"])
            all_bert_scores.extend(results[qt]["bert_scores"])
            all_rouge_scores.extend(results[qt]["rouge_scores"])
            all_meteor_scores.extend(results[qt]["meteor_scores"])
            all_f1_scores.extend(results[qt]["f1_scores"])

    # Calculate overall averages
    overall_results["overall"]["avg_reward"] = np.mean(all_rewards)
    overall_results["overall"]["avg_bert_score"] = np.mean(all_bert_scores)
    overall_results["overall"]["avg_rouge_score"] = np.mean(all_rouge_scores)
    overall_results["overall"]["avg_meteor_score"] = np.mean(all_meteor_scores)
    overall_results["overall"]["avg_f1_score"] = np.mean(all_f1_scores)

    print(f"Overall metrics:")
    print(f"  Avg Reward: {overall_results['overall']['avg_reward']:.4f}")
    print(f"  Avg BERTScore: {overall_results['overall']['avg_bert_score']:.4f}")
    print(f"  Avg ROUGE-L: {overall_results['overall']['avg_rouge_score']:.4f}")
    print(f"  Avg METEOR: {overall_results['overall']['avg_meteor_score']:.4f}")
    print(f"  Avg F1 Score: {overall_results['overall']['avg_f1_score']:.4f}")

    # Save results
    combined_results = {**results, **overall_results}

    results_file = os.path.join(output_dir, f"baseline_rag_{llm_model.replace('-', '_')}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)

    questions_file = os.path.join(output_dir, f"baseline_rag_{llm_model.replace('-', '_')}_questions.json")
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(question_results, f, indent=2)

    print(f"Results saved to {results_file}")
    print(f"Detailed question results saved to {questions_file}")

    return combined_results


def main():
    print("Starting baseline run...")
    # Create timestamped output directory for all results
    base_output_dir = "comparative_analysis_DIR"
    os.makedirs(base_output_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description='Run baseline RAG pipeline on test dataset')
    parser.add_argument('--test_data', type=str,
                        default='C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json',
                        help='Path to test dataset')
    parser.add_argument('--llm_models', type=str, nargs='+', default=['gpt-3.5-turbo', 'gpt-4'],
                        help='LLM models to use')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of context chunks to retrieve')
    parser.add_argument('--output_dir', type=str, default='baseline_results',
                        help='Directory to save results')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of examples to use (for quicker testing)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test dataset
    test_dataset = load_dataset(args.test_data)

    # Use subset if specified
    if args.subset_size and args.subset_size < len(test_dataset):
        test_dataset = test_dataset[:args.subset_size]
        print(f"Using subset of {args.subset_size} examples")

    # Run baseline RAG for each specified LLM model
    all_results = {}
    for llm_model in args.llm_models:
        print(f"Starting baseline RAG evaluation with {llm_model}")
        results = run_baseline_rag(
            test_dataset=test_dataset,
            llm_model=llm_model,
            top_k=args.top_k,
            output_dir=args.output_dir
        )
        all_results[llm_model] = results

    # Save combined results
    combined_file = os.path.join(args.output_dir, f"{base_output_dir}/baseline_rag_all_models.json")
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"All results saved to {combined_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")