import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from RL_Agent.bart_scorer import BARTScorer

# Logging Configuration
logging.basicConfig(
    level=logging.INFO
)


def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score with smoothing."""
    try:
        reference = [reference.split()]
        hypothesis = hypothesis.split()
        score = sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method1)
        logging.info(f"BLEU Score: {score:.4f}")
        return score
    except Exception as e:
        logging.error(f"Error calculating BLEU: {e}")
        return 0.0


def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores."""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        logging.info(f"ROUGE Scores: {scores}")
        return scores
    except Exception as e:
        logging.error(f"Error calculating ROUGE: {e}")
        return {}


def calculate_meteor(reference, hypothesis):
    """Calculate METEOR score."""
    try:
        score = meteor_score([reference.split()], hypothesis.split())
        logging.info(f"METEOR Score: {score:.4f}")
        return score
    except Exception as e:
        logging.error(f"Error calculating METEOR: {e}")
        return 0.0


def calculate_bertscore(reference, hypothesis):
    """Calculate BERTScore."""
    try:
        P, R, F1 = bert_score([hypothesis], [reference], lang="en")
        scores = {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}
        logging.info(f"BERTScore: {scores}")
        return scores
    except Exception as e:
        logging.error(f"Error calculating BERTScore: {e}")
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}


def calculate_bart_score(bart_scorer, reference, hypothesis):
    """Ensure BARTScorer gets reference and hypothesis in the correct order."""
    try:
        score = bart_scorer.score([hypothesis], [reference])  # ✅ Wrap in list
        if isinstance(score, list):
            return score[0]  # Now safe!
        return score  # Handle float case safely
    except Exception as e:
        logging.error(f"Error calculating BARTScore: {e}")
        return 0.0


def evaluate_metrics(reference, hypothesis, bart_scorer, tokenizer=None):
    """Compute all evaluation metrics and log the results."""
    logging.info("Starting evaluation for a new sample.")

    # Decode tokenized ground truth if necessary
    if isinstance(reference, list) and tokenizer is not None:
        reference = tokenizer.decode(reference, skip_special_tokens=True)

    scores = {
        "BLEU": calculate_bleu(reference, hypothesis),
        "ROUGE": calculate_rouge(reference, hypothesis),
        "METEOR": calculate_meteor(reference, hypothesis),
        "BERTScore": calculate_bertscore(reference, hypothesis),
        "BARTScore": calculate_bart_score(bart_scorer, reference, hypothesis)
    }

    logging.info(f"Final Evaluation Scores: {scores}")
    return scores



if __name__ == "__main__":
    reference_text = "A benefits broker helps employers determine benefits plans."
    hypothesis_text = "A benefits broker assists companies in choosing employee benefits."
    bart_scorer = BARTScorer(device='cpu')

    logging.info("Running evaluation script with test samples.")
    scores = evaluate_metrics(reference_text, hypothesis_text, bart_scorer)
    logging.info("Evaluation script execution completed.")

    print("Evaluation Scores:", scores)
