import os

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

current_dir = os.path.dirname(os.path.abspath(__file__))


class BARTScorer:
    def __init__(self, engine='facebook/bart-large-cnn', device='cpu'):
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(engine)
        self.model = BartForConditionalGeneration.from_pretrained(engine).to(self.device)

    def score(self, candidate, reference):
        candidate_tokens = self.tokenizer(candidate, return_tensors='pt', truncation=True, padding=True)
        reference_tokens = self.tokenizer(reference, return_tensors='pt', truncation=True, padding=True)

        candidate_tokens = {key: val.to(self.device) for key, val in candidate_tokens.items()}
        reference_tokens = {key: val.to(self.device) for key, val in reference_tokens.items()}

        with torch.no_grad():
            candidate_logits = self.model(**candidate_tokens).logits
            reference_logits = self.model(**reference_tokens).logits

        candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)

        # Calculate the average log-probability for the candidate given the reference.
        candidate_log_probs = candidate_log_probs.gather(2, candidate_tokens['input_ids'].unsqueeze(-1)).squeeze(-1)
        reference_log_probs = reference_log_probs.gather(2, reference_tokens['input_ids'].unsqueeze(-1)).squeeze(-1)

        candidate_score = candidate_log_probs.mean()
        reference_score = reference_log_probs.mean()

        bart_score = candidate_score + reference_score
        return bart_score.item()
