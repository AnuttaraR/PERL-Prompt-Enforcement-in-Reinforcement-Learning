import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random
from tqdm import tqdm
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import utility functions from existing code
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from RL_Agent.utils.retrieval import retrieve_context

# Create logs directory if it doesn't exist
os.makedirs("ppo_results/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("ppo_results/logs", "dpo_model.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DPO_model")

# Download NLTK resources if needed
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


# Load configurations
def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Load the dataset
def load_dataset(dataset_path):
    """Load dataset from JSON file"""
    start_time = time.time()
    logger.info(f"Loading dataset from {dataset_path}")
    print(f"Loading dataset from {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded dataset with {len(data)} items in {time.time() - start_time:.2f}s")
    print(f"Loaded dataset with {len(data)} items in {time.time() - start_time:.2f}s")
    return data


# Create preference pairs from dataset
def create_preference_pairs(dataset, action_space, num_pairs=1000, batch_size=16):
    """
    Create preference pairs for DPO training by sampling different prompt actions
    and comparing their generated answers.
    """
    logger.info(f"Creating {num_pairs} preference pairs from dataset")
    print(f"Creating {num_pairs} preference pairs from dataset")
    start_time = time.time()
    preference_pairs = []

    question_types = ["what", "how", "if_can"]

    # Group questions by type
    questions_by_type = {qt: [q for q in dataset if q["question_type"] == qt] for qt in question_types}

    # Create preference pairs for each question type
    pairs_per_type = num_pairs // len(question_types)
    for question_type in question_types:
        questions = questions_by_type[question_type]
        if not questions:
            logger.warning(f"No questions found for type: {question_type}")
            continue

        # Determine available actions for this question type
        general_actions = action_space["general_actions"]
        specific_actions = action_space[f"{question_type}_question_actions"]

        # Combine action lists and convert keys to integers
        action_list = [(int(k), v) for k, v in general_actions.items()] + \
                      [(int(k) + len(general_actions), v) for k, v in specific_actions.items()]

        # Process in batches to improve efficiency
        pairs_created = 0
        progress_bar = tqdm(total=pairs_per_type, desc=f"Creating {question_type} pairs")

        while pairs_created < pairs_per_type:
            batch_size = min(batch_size, pairs_per_type - pairs_created)
            batch_pairs = []

            for _ in range(batch_size):
                # Select a random question
                question_data = random.choice(questions)
                question = question_data["question"]
                ground_truth = question_data["ground_truth"]

                # Sample two different actions
                action1_id, action2_id = random.sample(range(len(action_list)), 2)
                action1 = action_list[action1_id]
                action2 = action_list[action2_id]

                # Apply both actions to get two different prompts
                prompt1_mod = f"{action1[1]}: {question}"
                prompt2_mod = f"{action2[1]}: {question}"

                batch_pairs.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "question_type": question_type,
                    "action1": action1,
                    "action2": action2,
                    "prompt1_mod": prompt1_mod,
                    "prompt2_mod": prompt2_mod
                })

            # Get modified prompts from LLM in parallel (if available)
            for pair in batch_pairs:
                modified_prompt1 = get_llm_response(pair["prompt1_mod"], model="gpt-3.5-turbo")
                modified_prompt2 = get_llm_response(pair["prompt2_mod"], model="gpt-3.5-turbo")

                # Handle invalid responses
                if modified_prompt1 in ["INVALID", "Invalid input", "Invalid response"]:
                    modified_prompt1 = pair["question"]
                if modified_prompt2 in ["INVALID", "Invalid input", "Invalid response"]:
                    modified_prompt2 = pair["question"]

                pair["modified_prompt1"] = modified_prompt1
                pair["modified_prompt2"] = modified_prompt2

            # Retrieve context for each question (once per question to save API calls)
            questions_list = [pair["question"] for pair in batch_pairs]
            contexts = {}
            for q in set(questions_list):
                contexts[q] = retrieve_context(q, top_k=3)

            # Generate answers using the modified prompts
            for pair in batch_pairs:
                context = contexts[pair["question"]]

                # Generate answers using both prompts
                final_prompt1 = f"Question: {pair['modified_prompt1']}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."
                final_prompt2 = f"Question: {pair['modified_prompt2']}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."

                answer1 = generate_answer_from_llm(final_prompt1, model="gpt-3.5-turbo")
                answer2 = generate_answer_from_llm(final_prompt2, model="gpt-3.5-turbo")

                # Calculate scores for each answer
                scores1 = calculate_answer_scores(answer1, pair["ground_truth"])
                scores2 = calculate_answer_scores(answer2, pair["ground_truth"])

                # Determine which answer is better (higher average score)
                avg_score1 = sum(scores1.values()) / len(scores1)
                avg_score2 = sum(scores2.values()) / len(scores2)

                # Ensure chosen_answer is always better than rejected_answer
                if avg_score1 >= avg_score2:
                    chosen_action = pair["action1"]
                    chosen_prompt = pair["modified_prompt1"]
                    chosen_answer = answer1
                    chosen_scores = scores1

                    rejected_action = pair["action2"]
                    rejected_prompt = pair["modified_prompt2"]
                    rejected_answer = answer2
                    rejected_scores = scores2
                else:
                    chosen_action = pair["action2"]
                    chosen_prompt = pair["modified_prompt2"]
                    chosen_answer = answer2
                    chosen_scores = scores2

                    rejected_action = pair["action1"]
                    rejected_prompt = pair["modified_prompt1"]
                    rejected_answer = answer1
                    rejected_scores = scores1

                # Store the preference pair
                preference_pairs.append({
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "question_type": pair["question_type"],
                    "chosen_action_id": chosen_action[0],
                    "chosen_action_desc": chosen_action[1],
                    "chosen_prompt": chosen_prompt,
                    "chosen_answer": chosen_answer,
                    "chosen_scores": chosen_scores,
                    "rejected_action_id": rejected_action[0],
                    "rejected_action_desc": rejected_action[1],
                    "rejected_prompt": rejected_prompt,
                    "rejected_answer": rejected_answer,
                    "rejected_scores": rejected_scores,
                })

            pairs_created += len(batch_pairs)
            progress_bar.update(len(batch_pairs))

        progress_bar.close()

    logger.info(f"Created {len(preference_pairs)} preference pairs in {time.time() - start_time:.2f}s")
    print(f"Created {len(preference_pairs)} preference pairs in {time.time() - start_time:.2f}s")
    return preference_pairs


def calculate_answer_scores(answer, ground_truth):
    """Calculate various scores for an answer compared to ground truth"""
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
        "f1_score": f1_score_val
    }


def save_preference_pairs(preference_pairs, output_path):
    """Save preference pairs to a JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, indent=2)

    logger.info(f"Saved {len(preference_pairs)} preference pairs to {output_path}")
    print(f"Saved {len(preference_pairs)} preference pairs to {output_path}")


def load_preference_pairs(input_path):
    """Load preference pairs from a JSON file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        preference_pairs = json.load(f)

    logger.info(f"Loaded {len(preference_pairs)} preference pairs from {input_path}")
    print(f"Loaded {len(preference_pairs)} preference pairs from {input_path}")
    return preference_pairs


# DPO Actor Network
class DPOActorNetwork(nn.Module):
    def __init__(self, input_dim, action_counts, variant=None):
        super(DPOActorNetwork, self).__init__()

        # Model architecture varies based on variant
        if variant == "deeper_network":
            # Deeper network with more layers
            self.shared_layer = nn.Linear(input_dim, 512)
            self.shared_layer2 = nn.Linear(512, 256)
            self.shared_layer3 = nn.Linear(256, 128)  # Additional layer

            if variant == "shared_heads":
                # Single head for all question types
                self.output_head = nn.Linear(128, max(action_counts.values()))
            else:
                # Question-type specific heads
                self.what_head = nn.Linear(128, action_counts["what"])
                self.how_head = nn.Linear(128, action_counts["how"])
                self.if_can_head = nn.Linear(128, action_counts["if_can"])
        else:
            # Standard architecture
            self.shared_layer = nn.Linear(input_dim, 256)
            self.shared_layer2 = nn.Linear(256, 128)

            if variant == "shared_heads":
                # Single head for all question types
                self.output_head = nn.Linear(128, max(action_counts.values()))
            else:
                # Question-type specific heads
                self.what_head = nn.Linear(128, action_counts["what"])
                self.how_head = nn.Linear(128, action_counts["how"])
                self.if_can_head = nn.Linear(128, action_counts["if_can"])

        # Store the variant for forward pass
        self.variant = variant

        # Beta parameter for DPO training
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, question_type):
        x = F.relu(self.shared_layer(state))
        x = F.relu(self.shared_layer2(x))

        # Apply additional layer if using deeper network
        if self.variant == "deeper_network":
            x = F.relu(self.shared_layer3(x))

        # Select the appropriate head based on variant and question type
        if self.variant == "shared_heads":
            logits = self.output_head(x)
        else:
            if question_type == "what":
                logits = self.what_head(x)
            elif question_type == "how":
                logits = self.how_head(x)
            else:  # if_can
                logits = self.if_can_head(x)

        return logits

    def get_action_probs(self, state, question_type):
        """Get action probabilities (used during inference)"""
        logits = self.forward(state, question_type)
        return F.softmax(logits, dim=-1)

# Reference model (initialized with same weights but frozen during training)
class ReferenceModel(nn.Module):
    def __init__(self, input_dim, action_counts, device):
        super(ReferenceModel, self).__init__()
        # Create a new instance with the same architecture
        self.model = DPOActorNetwork(input_dim, action_counts).to(device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, state, question_type):
        with torch.no_grad():
            return self.model(state, question_type)


# DPO Trainer
class DPOTrainer:
    def __init__(self, input_dim, action_space, learning_rate=1e-4, beta=0.1,
                 ablation=None, model_variant=None, temperature=1.0, diversity_weight=0.0,
                 weighted_loss=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        logger.info(f"Initializing DPO Trainer with input_dim={input_dim}, lr={learning_rate}, beta={beta}")
        logger.info(f"Using device: {device}, ablation={ablation}, model_variant={model_variant}")
        logger.info(f"Temperature={temperature}, diversity_weight={diversity_weight}, weighted_loss={weighted_loss}")

        # Check if we're using modified action space (without action 0)
        self.has_keep_unchanged = "0" in action_space["general_actions"]
        self.action_id_offset = 0 if self.has_keep_unchanged else 1
        logger.info(f"Action space has 'keep unchanged': {self.has_keep_unchanged}")

        # Apply ablations to action space if specified
        if ablation:
            action_space = self.apply_action_space_ablation(action_space, ablation)
            logger.info(f"Applied action space ablation: {ablation}")

        # Count actions per question type
        self.action_counts = {}

        # General actions count
        general_count = len(action_space["general_actions"])
        logger.info(f"Found {general_count} general actions")

        # Count specific actions for each question type
        for qt in ["what", "how", "if_can"]:
            specific_key = f"{qt}_question_actions"
            if specific_key in action_space:
                specific_count = len(action_space[specific_key])
                # Total count is general + specific actions
                self.action_counts[qt] = general_count + specific_count
                logger.info(
                    f"Found {specific_count} specific actions for {qt} questions (total: {self.action_counts[qt]})")
            else:
                # Fallback if no specific actions are defined
                self.action_counts[qt] = general_count
                logger.warning(f"No specific actions found for {qt} questions, using only general actions")

        self.action_space = action_space
        self.device = device
        self.beta = beta
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.weighted_loss = weighted_loss

        # Set question-type weights for weighted loss
        if weighted_loss:
            # These weights are inverses of baseline performance
            # Lower performance = higher weight
            self.question_type_weights = {
                "what": 1.0 / 0.593,  # 59.3% in baseline
                "how": 1.0 / 0.308,  # 30.8% in baseline (highest weight)
                "if_can": 1.0 / 0.9  # 90% in baseline (lowest weight)
            }
            logger.info(f"Using question-type weighted loss: {self.question_type_weights}")
        else:
            self.question_type_weights = {"what": 1.0, "how": 1.0, "if_can": 1.0}

        # Initialize actor network with variant
        self.actor = DPOActorNetwork(input_dim, self.action_counts, variant=model_variant).to(device)

        # Create reference model with same architecture and weights
        self.ref_model = ReferenceModel(input_dim, self.action_counts, device)
        # Copy weights from actor to reference model
        self.ref_model.model.load_state_dict(self.actor.state_dict())

        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Metrics tracking
        self.train_losses = []
        self.validation_accuracies = []
        logger.info("DPO Trainer initialized successfully")
        print("DPO Trainer initialized successfully")

    def apply_action_space_ablation(self, action_space, ablation):
        """Apply ablation to action space configuration"""
        logger.info(f"Applying action space ablation: {ablation}")
        if ablation == "action_unified":
            # Make all actions available to all question types
            all_actions = {}

            # Add general actions
            for k, v in action_space["general_actions"].items():
                all_actions[k] = v

            # Add all specific actions from all question types
            offset = len(all_actions)
            for qt in ["what", "how", "if_can"]:
                qt_key = f"{qt}_question_actions"
                for i, (k, v) in enumerate(action_space[qt_key].items()):
                    all_actions[str(offset + i)] = v

            # Create new action space with all actions for each question type
            ablated_space = {
                "general_actions": action_space["general_actions"],
                "what_question_actions": all_actions,
                "how_question_actions": all_actions,
                "if_can_question_actions": all_actions
            }
            return ablated_space

        elif ablation == "action_what_only":
            # Only what actions for all question types
            ablated_space = action_space.copy()
            ablated_space["how_question_actions"] = action_space["what_question_actions"]
            ablated_space["if_can_question_actions"] = action_space["what_question_actions"]
            return ablated_space

        elif ablation == "action_how_only":
            # Only how actions for all question types
            ablated_space = action_space.copy()
            ablated_space["what_question_actions"] = action_space["how_question_actions"]
            ablated_space["if_can_question_actions"] = action_space["how_question_actions"]
            return ablated_space

        elif ablation == "action_if_can_only":
            # Only if_can actions for all question types
            ablated_space = action_space.copy()
            ablated_space["what_question_actions"] = action_space["if_can_question_actions"]
            ablated_space["how_question_actions"] = action_space["if_can_question_actions"]
            return ablated_space

        elif ablation == "action_minimal":
            # Only general actions, no specific actions
            ablated_space = {
                "general_actions": action_space["general_actions"],
                "what_question_actions": {},
                "how_question_actions": {},
                "if_can_question_actions": {}
            }
            return ablated_space

        # Default: return original action space
        return action_space

    def get_action_description(self, action, question_type):
        """Get action description with fixed mapping"""
        general_actions = self.action_space["general_actions"]
        general_count = len(general_actions)

        # Print debug info for this call
        print(f"DEBUG - get_action_description called with action={action}, question_type={question_type}")
        print(f"DEBUG - general_count={general_count}")

        # Check if it's a general action
        if action < general_count:
            action_key = str(action)
            print(f"DEBUG - Returning general action: {general_actions[action_key]}")
            return general_actions[action_key]

        # It's a question-specific action
        qt_key = f"{question_type}_question_actions"
        specific_actions = self.action_space[qt_key]

        # Correctly calculate the key in the specific actions dict
        specific_idx = action - general_count
        specific_key = str(specific_idx + 1)  # +1 because specific actions typically start at 1

        if specific_key in specific_actions:
            print(f"DEBUG - Returning specific action: {specific_actions[specific_key]}")
            return specific_actions[specific_key]
        else:
            print(f"DEBUG - No matching action found! Available specific keys: {list(specific_actions.keys())}")
            return "Unknown action"

    def dpo_loss(self, chosen_logits, rejected_logits, chosen_actions, rejected_actions, beta=None):
        """
        Calculate the DPO loss based on logits for chosen and rejected actions.

        Args:
            chosen_logits: Logits from policy model for states with chosen actions
            rejected_logits: Logits from policy model for states with rejected actions
            chosen_actions: Indices of the chosen actions
            rejected_actions: Indices of the rejected actions
            beta: Temperature parameter (if None, use self.beta)

        Returns:
            The DPO loss (scalar)
        """
        if beta is None:
            beta = self.beta

        # Calculate log probabilities
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)

        # Extract the log probs for the specific actions
        chosen_action_log_probs = chosen_log_probs.gather(1, chosen_actions.unsqueeze(1)).squeeze()
        rejected_action_log_probs = rejected_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze()

        # Calculate the DPO loss
        loss = -torch.mean(
            torch.log(
                torch.sigmoid(
                    beta * (chosen_action_log_probs - rejected_action_log_probs)
                )
            )
        )

        return loss

    def train_step(self, batch, diversity_weight=None):
        """
        Perform a single training step on a batch of preference pairs.
        """
        if diversity_weight is None:
            diversity_weight = self.diversity_weight

        self.actor.train()
        self.optimizer.zero_grad()

        states = batch["states"].to(self.device)
        question_types = batch["question_types"]
        chosen_actions = batch["chosen_actions"].to(self.device)
        rejected_actions = batch["rejected_actions"].to(self.device)

        batch_size = len(states)
        batch_policy_logits = []
        batch_ref_logits = []

        # Fix action indices to ensure they're within bounds
        valid_chosen_actions = []
        valid_rejected_actions = []
        valid_indices = []
        valid_question_types = []

        # Process each example in the batch
        for i in range(batch_size):
            state = states[i].unsqueeze(0)
            qt = question_types[i]

            # Get the maximum valid action index for this question type
            max_action_idx = self.action_counts[qt] - 1

            # Check if actions are within valid range
            chosen_action = chosen_actions[i].item()
            rejected_action = rejected_actions[i].item()

            if chosen_action > max_action_idx or rejected_action > max_action_idx:
                # Skip this example as it has invalid action indices
                logger.warning(f"Skipping example with out-of-range actions: chosen={chosen_action}, "
                               f"rejected={rejected_action}, max_valid={max_action_idx}, type={qt}")
                continue

            # Get logits from both models
            policy_logits = self.actor(state, qt)
            ref_logits = self.ref_model(state, qt)

            # Store logits for both models
            batch_policy_logits.append(policy_logits)
            batch_ref_logits.append(ref_logits)

            # Store valid actions and indices
            valid_chosen_actions.append(chosen_action)
            valid_rejected_actions.append(rejected_action)
            valid_indices.append(i)
            valid_question_types.append(qt)

        # If no valid examples remain, return zero loss
        if not valid_indices:
            logger.warning("No valid examples in batch after filtering out-of-range actions")
            return 0.0

        # Update with valid examples only
        batch_policy_logits = torch.cat(batch_policy_logits, dim=0)
        batch_ref_logits = torch.cat(batch_ref_logits, dim=0)
        chosen_actions = torch.tensor(valid_chosen_actions, device=self.device)
        rejected_actions = torch.tensor(valid_rejected_actions, device=self.device)

        # Calculate log probabilities for policy model
        policy_log_probs = F.log_softmax(batch_policy_logits, dim=-1)

        # Extract log probs for the chosen and rejected actions
        chosen_log_probs = policy_log_probs.gather(1, chosen_actions.unsqueeze(1)).squeeze(1)
        rejected_log_probs = policy_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

        # Calculate log probabilities for reference model
        with torch.no_grad():
            ref_log_probs = F.log_softmax(batch_ref_logits, dim=-1)
            chosen_ref_log_probs = ref_log_probs.gather(1, chosen_actions.unsqueeze(1)).squeeze(1)
            rejected_ref_log_probs = ref_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

        # Calculate the DPO loss
        logits_diff = (chosen_log_probs - chosen_ref_log_probs) - (rejected_log_probs - rejected_ref_log_probs)

        # Apply question-type weighting if enabled
        if self.weighted_loss:
            # Get weights for each example
            weights = torch.tensor([self.question_type_weights[qt] for qt in valid_question_types],
                                   device=self.device)
            # Normalize weights
            weights = weights / weights.sum()
            # Apply weights to the losses
            losses = -torch.log(torch.sigmoid(self.beta * logits_diff))
            loss = (losses * weights).sum()
        else:
            # Standard unweighted loss
            loss = -torch.mean(torch.log(torch.sigmoid(self.beta * logits_diff)))

        # Add diversity reward if enabled
        if diversity_weight > 0:
            # Calculate action entropy (higher = more diverse actions)
            probs = F.softmax(batch_policy_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            # Specifically discourage "keep unchanged" action if it exists
            if self.has_keep_unchanged:
                # Extract probability of "keep unchanged" (action 0)
                keep_unchanged_probs = probs[:, 0]
                # Additional penalty based on keep_unchanged probability (higher prob = higher penalty)
                keep_unchanged_penalty = torch.mean(keep_unchanged_probs) * 0.5

                # Combined diversity score (higher is better)
                diversity_score = entropy.mean() - keep_unchanged_penalty
            else:
                diversity_score = entropy.mean()

            # Add entropy bonus to the loss (negative because we're minimizing)
            diversity_bonus = -diversity_weight * diversity_score
            loss += diversity_bonus
            logger.debug(f"Added diversity bonus: {diversity_bonus.item():.6f}")

        # Backpropagate and update weights
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, preference_pairs, num_epochs=10, batch_size=32, validation_split=0.1, output_dir="dpo_output"):
        """
        Train the DPO model on preference pairs.

        Args:
            preference_pairs: List of preference pair dictionaries
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation

        Returns:
            training_metrics: Dictionary of training metrics
        """
        logger.info(f"Starting DPO training for {num_epochs} epochs with batch_size={batch_size}")
        print(f"Starting DPO training for {num_epochs} epochs with batch_size={batch_size}")
        start_time = time.time()

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Prepare dataset
        total_pairs = len(preference_pairs)
        val_size = int(total_pairs * validation_split)
        train_size = total_pairs - val_size

        # Add to train method
        best_val_accuracy = 0
        patience = 3
        patience_counter = 0

        # Shuffle and split into train/val
        random.shuffle(preference_pairs)
        train_pairs = preference_pairs[:train_size]
        val_pairs = preference_pairs[train_size:]

        logger.info(f"Training on {train_size} pairs, validating on {val_size} pairs")
        print(f"Training on {train_size} pairs, validating on {val_size} pairs")

        # Create tokenizer if needed
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Preprocess data - make sure embeddings dimensions are handled correctly
        train_data = self.preprocess_preference_pairs(train_pairs, tokenizer)
        val_data = self.preprocess_preference_pairs(val_pairs, tokenizer)

        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_losses = []

            # Create batches and shuffle
            batch_indices = list(range(0, train_size, batch_size))
            random.shuffle(batch_indices)

            # Progress bar
            progress_bar = tqdm(batch_indices, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for start_idx in progress_bar:
                end_idx = min(start_idx + batch_size, train_size)
                batch_indices_list = list(range(start_idx, end_idx))

                # Create batch
                batch = {
                    "states": torch.stack([train_data["states"][i] for i in batch_indices_list]),
                    "question_types": [train_data["question_types"][i] for i in batch_indices_list],
                    "chosen_actions": torch.tensor([train_data["chosen_actions"][i] for i in batch_indices_list]),
                    "rejected_actions": torch.tensor([train_data["rejected_actions"][i] for i in batch_indices_list])
                }

                # Train on batch
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})

            # Calculate epoch metrics
            epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(epoch_loss)

            # Validate
            val_accuracy = self.validate(val_data)
            self.validation_accuracies.append(val_accuracy)

            # Inside the training loop of the train method
            best_val_accuracy = 0.0
            best_model_path = os.path.join(os.path.dirname(f"dpo_model_epoch_{epoch + 1}"), "best_model")

            # After validation
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(os.path.join(output_dir, "best_model"))
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)")
                    break

            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - "
                        f"Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s - "
                        f"Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_model(f"dpo_model_epoch_{epoch + 1}")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f}m)")
        print(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f}m)")

        self.plot_training_curves(output_dir=output_dir)

        # Return training metrics
        return {
            "train_losses": self.train_losses,
            "validation_accuracies": self.validation_accuracies,
            "best_validation_accuracy": max(self.validation_accuracies) if self.validation_accuracies else 0,
            "training_time": total_time
        }

    def preprocess_preference_pairs(self, preference_pairs, tokenizer):
        """
        Preprocess preference pairs for training with additional debugging.
        """
        logger.info(f"Preprocessing {len(preference_pairs)} preference pairs")

        # First, analyze the action indices in the data
        action_id_counts = {}
        for qt in self.action_counts.keys():
            action_id_counts[qt] = {"chosen": {}, "rejected": {}}

        # Count action frequencies and find the max action ID
        max_action_id = 0
        for pair in preference_pairs:
            qt = pair["question_type"]
            chosen_id = pair["chosen_action_id"]
            rejected_id = pair["rejected_action_id"]

            # Track max action ID to understand range
            max_action_id = max(max_action_id, chosen_id, rejected_id)

            # Count occurrences
            if chosen_id not in action_id_counts[qt]["chosen"]:
                action_id_counts[qt]["chosen"][chosen_id] = 0
            action_id_counts[qt]["chosen"][chosen_id] += 1

            if rejected_id not in action_id_counts[qt]["rejected"]:
                action_id_counts[qt]["rejected"][rejected_id] = 0
            action_id_counts[qt]["rejected"][rejected_id] += 1

        # Log action ID distributions
        logger.info(f"Maximum action ID in preference pairs: {max_action_id}")
        logger.info(f"Model expected action counts: {self.action_counts}")

        for qt in action_id_counts:
            logger.info(f"Action ID distribution for {qt} questions:")
            logger.info(f"  Chosen actions: {sorted(action_id_counts[qt]['chosen'].items())}")

        # IMPORTANT - Analyze the issue:
        logger.info(f"Action ID analysis:")
        for pair in preference_pairs[:5]:  # Look at just a few examples
            logger.info(f"Question type: {pair['question_type']}")
            logger.info(f"  Chosen action ID: {pair['chosen_action_id']}, desc: {pair['chosen_action_desc']}")
            logger.info(f"  Rejected action ID: {pair['rejected_action_id']}, desc: {pair['rejected_action_desc']}")

        # Option 1: Remap action IDs based on descriptions
        # This assumes action descriptions match between preference pairs and action space
        if max_action_id >= max(self.action_counts.values()):
            logger.warning("Action IDs in preference pairs exceed model's expected range. Attempting to remap...")
            # Build description-to-id mapping from action space
            # Build a more robust description-to-id mapping from action space
            desc_to_id = {}
            for qt in ["what", "how", "if_can"]:
                # Add general actions
                for action_id, desc in self.action_space["general_actions"].items():
                    # Store with exact description match
                    desc_to_id[(desc, qt)] = int(action_id)
                    # Also store with case-insensitive match
                    desc_to_id[(desc.lower(), qt)] = int(action_id)
                    # Store with normalized whitespace
                    desc_to_id[(desc.strip().lower(), qt)] = int(action_id)

                # Add specific actions
                qt_key = f"{qt}_question_actions"
                if qt_key in self.action_space:
                    general_count = len(self.action_space["general_actions"])
                    for action_id, desc in self.action_space[qt_key].items():
                        # Special indexing for specific actions
                        remapped_id = general_count + int(action_id) - 1  # -1 because specific actions start at 1
                        # Store with exact description match
                        desc_to_id[(desc, qt)] = remapped_id
                        # Also store with case-insensitive match
                        desc_to_id[(desc.lower(), qt)] = remapped_id
                        # Store with normalized whitespace
                        desc_to_id[(desc.strip().lower(), qt)] = remapped_id

            # Now process with more flexible matching
            states = []
            question_types = []
            chosen_actions = []
            rejected_actions = []
            skipped_count = 0
            remapped_count = 0

            # Process preference pairs with direct ID fallback
            for i, pair in enumerate(preference_pairs):
                qt = pair["question_type"]
                chosen_desc = pair["chosen_action_desc"]
                rejected_desc = pair["rejected_action_desc"]

                # Try multiple matching strategies
                chosen_key = (chosen_desc, qt)
                chosen_key_lower = (chosen_desc.lower(), qt)
                chosen_key_normalized = (chosen_desc.strip().lower(), qt)

                rejected_key = (rejected_desc, qt)
                rejected_key_lower = (rejected_desc.lower(), qt)
                rejected_key_normalized = (rejected_desc.strip().lower(), qt)

                # Try to find matches with progressively more lenient matching
                chosen_id = None
                if chosen_key in desc_to_id:
                    chosen_id = desc_to_id[chosen_key]
                elif chosen_key_lower in desc_to_id:
                    chosen_id = desc_to_id[chosen_key_lower]
                elif chosen_key_normalized in desc_to_id:
                    chosen_id = desc_to_id[chosen_key_normalized]
                else:
                    # Direct use of the ID from the pair as fallback
                    chosen_id = pair["chosen_action_id"]
                    if chosen_id >= self.action_counts[qt]:
                        # If ID is still too large, cap it to the maximum valid ID
                        chosen_id = self.action_counts[qt] - 1

                rejected_id = None
                if rejected_key in desc_to_id:
                    rejected_id = desc_to_id[rejected_key]
                elif rejected_key_lower in desc_to_id:
                    rejected_id = desc_to_id[rejected_key_lower]
                elif rejected_key_normalized in desc_to_id:
                    rejected_id = desc_to_id[rejected_key_normalized]
                else:
                    # Direct use of the ID from the pair as fallback
                    rejected_id = pair["rejected_action_id"]
                    if rejected_id >= self.action_counts[qt]:
                        # If ID is still too large, cap it to the maximum valid ID
                        rejected_id = self.action_counts[qt] - 1

                # Process this pair (we'll now always have a valid ID)
                question = pair["question"]

                # Process in small batches to avoid memory issues
                encoded = tokenizer(question, truncation=True, padding="max_length",
                                    max_length=128, return_tensors="pt").to(self.device)

                # Generate embeddings
                bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
                bert_model.eval()

                with torch.no_grad():
                    outputs = bert_model(**encoded)
                    state = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

                states.append(state.cpu().squeeze(0))
                question_types.append(qt)
                chosen_actions.append(chosen_id)
                rejected_actions.append(rejected_id)
                remapped_count += 1

                # Log progress occasionally
                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Processed {i + 1}/{len(preference_pairs)} pairs - valid: {remapped_count}, skipped: {skipped_count}")

        else:
            # Regular processing if IDs seem to be in range
            states = []
            question_types = []
            chosen_actions = []
            rejected_actions = []
            skipped_count = 0

            # Initialize BERT model for embeddings
            bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            bert_model.eval()  # Set to evaluation mode

            # Process in batches
            batch_size = 32
            for i in range(0, len(preference_pairs), batch_size):
                batch_pairs = preference_pairs[i:i + batch_size]
                logger.info(f"preprocessing... {i}  of  {len(preference_pairs)}")

                batch_questions = [pair["question"] for pair in batch_pairs]

                # Tokenize questions
                encoded = tokenizer(batch_questions, truncation=True, padding="max_length",
                                    max_length=128, return_tensors="pt").to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    outputs = bert_model(**encoded)
                    batch_states = outputs.last_hidden_state.mean(dim=1)

                # Store states and metadata
                for j, pair in enumerate(batch_pairs):
                    qt = pair["question_type"]
                    chosen_action_id = pair["chosen_action_id"]
                    rejected_action_id = pair["rejected_action_id"]

                    # Check if actions are within valid range
                    max_action_idx = self.action_counts[qt] - 1

                    if chosen_action_id > max_action_idx or rejected_action_id > max_action_idx:
                        skipped_count += 1
                        continue

                    states.append(batch_states[j].cpu())
                    question_types.append(qt)
                    chosen_actions.append(chosen_action_id)
                    rejected_actions.append(rejected_action_id)

                # Log progress periodically
                if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= len(preference_pairs):
                    logger.info(
                        f"Preprocessed {len(states)} valid preference pairs (skipped {skipped_count} with out-of-range actions)")

        if len(states) == 0:
            raise ValueError("No valid preference pairs found after filtering out-of-range actions!")

        # Convert to appropriate formats
        processed_data = {
            "states": torch.stack(states),
            "question_types": question_types,
            "chosen_actions": torch.tensor(chosen_actions, dtype=torch.long),
            "rejected_actions": torch.tensor(rejected_actions, dtype=torch.long)
        }

        return processed_data

    def validate(self, val_data, batch_size=32):
        """
        Validate the model on validation data.

        Args:
            val_data: Validation data dictionary
            batch_size: Batch size for validation

        Returns:
            accuracy: Preference matching accuracy (%)
        """
        self.actor.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i in range(0, len(val_data["states"]), batch_size):
                end_idx = min(i + batch_size, len(val_data["states"]))
                batch_indices = list(range(i, end_idx))

                batch_states = val_data["states"][i:end_idx].to(self.device)
                batch_question_types = [val_data["question_types"][j] for j in batch_indices]
                batch_chosen_actions = [val_data["chosen_actions"][j] for j in batch_indices]
                batch_rejected_actions = [val_data["rejected_actions"][j] for j in batch_indices]

                for j in range(len(batch_states)):
                    state = batch_states[j].unsqueeze(0)
                    qt = batch_question_types[j]
                    chosen_action = batch_chosen_actions[j]
                    rejected_action = batch_rejected_actions[j]

                    # Get action probabilities
                    action_probs = self.actor.get_action_probs(state, qt).cpu().numpy()[0]

                    # Check if model assigns higher probability to chosen action
                    if action_probs[chosen_action] > action_probs[rejected_action]:
                        correct_predictions += 1

                    total_predictions += 1

        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy

    def get_best_action(self, question, question_type, tokenizer, temperature=None):
        """
        Get the best action for a given question with enhanced debugging.
        """
        if temperature is None:
            temperature = self.temperature

        self.actor.eval()

        # Initialize BERT model for embeddings (or reuse from class)
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        bert_model.eval()

        # Tokenize question
        encoded = tokenizer(question, truncation=True, padding="max_length",
                            max_length=128, return_tensors="pt").to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = bert_model(**encoded)
            state = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            # Get raw logits and check their values
            logits = self.actor(state, question_type)

            # Check if logits are all very similar (possible issue)
            logits_np = logits.cpu().numpy()[0]
            logit_range = np.max(logits_np) - np.min(logits_np)
            if logit_range < 0.1:  # Small range could indicate a problem
                logger.warning(f"WARNING - Logits have very small range: {logit_range:.6f}")
                logger.warning(f"WARNING - Logits: {logits_np}")

            # Get action probabilities with improved temperature scaling
            scaled_logits = logits / max(0.1, temperature)  # Prevent division by very small values
            probs = F.softmax(scaled_logits, dim=-1)

            # Dynamically adjust exploration rate based on question type performance
            if question_type == "how":  # This type performs worst, explore more
                explore_chance = 0.8
            elif question_type == "what":  # Medium performance, moderate exploration
                explore_chance = 0.6
            else:  # "if_can" performs well, less exploration needed
                explore_chance = 0.4

            # Log exploration strategy for debugging
            logger.debug(f"Using explore_chance={explore_chance} for {question_type} question")

            # Decide whether to explore or exploit
            if random.random() < explore_chance:
                # Sample from distribution for exploration
                action_distribution = torch.distributions.Categorical(probs)
                best_action_id = action_distribution.sample().item()
                logger.debug(f"Exploring: sampled action {best_action_id}")
            else:
                # Choose the highest probability action for exploitation
                best_action_id = torch.argmax(probs).item()
                logger.debug(f"Exploiting: chose highest probability action {best_action_id}")

            # More aggressive attempt to avoid "keep unchanged" for "how" questions
            if best_action_id == 0 and self.has_keep_unchanged:
                avoid_chance = 0.7 if question_type == "how" else 0.5
                if random.random() < avoid_chance:
                    # Create a list of non-zero actions and randomly select one
                    non_zero_actions = list(range(1, self.action_counts[question_type]))
                    if non_zero_actions:  # If there are any non-zero actions
                        old_action = best_action_id
                        best_action_id = random.choice(non_zero_actions)
                        logger.debug(f"Avoiding 'keep unchanged': changed action from {old_action} to {best_action_id}")

            # Get action description and validate it exists
            best_action_desc = self.get_action_description(best_action_id, question_type)
            if best_action_desc == "Unknown action":
                logger.warning(f"Invalid action ID {best_action_id} for {question_type} question")
                # Fallback to a known valid action
                best_action_id = 0 if self.has_keep_unchanged else 1
                best_action_desc = self.get_action_description(best_action_id, question_type)
                logger.info(f"Falling back to safe action: {best_action_id} ({best_action_desc})")

            action_probs = probs.cpu().numpy()[0]

        return best_action_id, best_action_desc, action_probs

    def save_model(self, path="dpo_model"):
        """Save the model weights"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")

        # Save metadata about the model
        metadata = {
            "action_counts": self.action_counts,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_losses": self.train_losses,
            "validation_accuracies": self.validation_accuracies
        }

        with open(f"{path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model saved to {path}")
        print(f"Model saved to {path}")

    def load_model(self, path="dpo_model"):
        """Load the model weights"""
        if not os.path.exists(f"{path}/actor.pth"):
            logger.error(f"Model file not found at {path}/actor.pth")
            return False

        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))

        # Load metadata if available
        metadata_path = f"{path}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            if "train_losses" in metadata:
                self.train_losses = metadata["train_losses"]
            if "validation_accuracies" in metadata:
                self.validation_accuracies = metadata["validation_accuracies"]

            logger.info(f"Loaded model saved at {metadata.get('saved_at', 'unknown time')}")
        else:
            logger.info(f"Model loaded from {path} (no metadata available)")

        return True

    def plot_training_curves(self, output_dir="plots"):
        """Plot training curves and save to the specified output directory"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, marker='o')
        plt.title('DPO Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "action_minimal_dpo_loss_curve.png"))
        plt.close()

        # Plot validation accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.validation_accuracies, marker='o', color='green')
        plt.title('DPO Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "action_minimal_dpo_accuracy_curve.png"))
        plt.close()

        logger.info(f"Training curves plotted and saved to '{output_dir}' directory")

    def evaluate(self, test_dataset, tokenizer):
        """
        Evaluate the model on test dataset with additional debugging.

        Args:
            test_dataset: Test dataset
            tokenizer: Tokenizer to use for encoding questions

        Returns:
            results: Dictionary with evaluation results
        """
        logger.info(f"Evaluating DPO model on {len(test_dataset)} test examples")
        self.actor.eval()

        question_type_metrics = {
            "what": {"correct": 0, "total": 0, "actions": {}},
            "how": {"correct": 0, "total": 0, "actions": {}},
            "if_can": {"correct": 0, "total": 0, "actions": {}}
        }

        # Add debug flag to inspect logits and probabilities
        debug_flag = True
        debug_samples = min(5, len(test_dataset))  # Number of examples to debug

        with torch.no_grad():
            for i, question_data in enumerate(tqdm(test_dataset, desc="Evaluating")):
                question = question_data["question"]
                question_type = question_data["question_type"]
                ground_truth = question_data["ground_truth"]

                # Get best action with detailed debugging for a few samples
                if debug_flag and i < debug_samples:
                    logger.info(f"DEBUG - Question {i + 1}: {question}")
                    logger.info(f"DEBUG - Question type: {question_type}")

                    # Get embeddings
                    encoded = tokenizer(question, truncation=True, padding="max_length",
                                        max_length=128, return_tensors="pt").to(self.device)

                    # Initialize BERT model for embeddings
                    bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
                    bert_model.eval()

                    with torch.no_grad():
                        outputs = bert_model(**encoded)
                        state = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

                    # Get raw logits
                    logits = self.actor(state, question_type)
                    logger.info(f"DEBUG - Raw logits: {logits.cpu().numpy()}")

                    # Get probabilities
                    probs = F.softmax(logits, dim=-1)
                    logger.info(f"DEBUG - Action probabilities: {probs.cpu().numpy()[0]}")

                    # Get action with highest probability
                    best_action_id = torch.argmax(probs).item()
                    best_action_desc = self.get_action_description(best_action_id, question_type)

                    # Check all available actions
                    logger.info(f"DEBUG - Available actions for {question_type} questions:")
                    for action_id in range(self.action_counts[question_type]):
                        action_desc = self.get_action_description(action_id, question_type)
                        prob = probs[0][action_id].item()
                        logger.info(f"DEBUG -   Action {action_id}: {action_desc} (Prob: {prob:.4f})")

                    logger.info(f"DEBUG - Selected action {best_action_id}: {best_action_desc}")

                # Regular evaluation code
                best_action_id, best_action_desc, _ = self.get_best_action(question, question_type, tokenizer)

                # Track action selection
                if best_action_id not in question_type_metrics[question_type]["actions"]:
                    question_type_metrics[question_type]["actions"][best_action_id] = 0
                question_type_metrics[question_type]["actions"][best_action_id] += 1

                # Apply action to get modified prompt
                prompt_mod = f"{best_action_desc}: {question}"
                modified_prompt = get_llm_response(prompt_mod, model="gpt-3.5-turbo")

                if modified_prompt in ["INVALID", "Invalid input", "Invalid response"]:
                    modified_prompt = question

                # Get context from retrieval
                context = retrieve_context(question, top_k=3)

                # Generate answer
                final_prompt = f"Question: {modified_prompt}\n\nContext Information: {context}\n\nPlease answer the question based on the provided context."
                answer = generate_answer_from_llm(final_prompt, model="gpt-3.5-turbo")

                # Calculate scores
                scores = calculate_answer_scores(answer, ground_truth)

                # Consider evaluation successful if average score is above threshold
                avg_score = sum(scores.values()) / len(scores)
                success_threshold = 0.5

                if avg_score > success_threshold:
                    question_type_metrics[question_type]["correct"] += 1

                question_type_metrics[question_type]["total"] += 1

        # Calculate metrics
        results = {}
        for qt, metrics in question_type_metrics.items():
            if metrics["total"] > 0:
                accuracy = (metrics["correct"] / metrics["total"]) * 100
            else:
                accuracy = 0

            # Get top actions
            action_counts = metrics["actions"]
            total_actions = sum(action_counts.values())
            action_percentages = {
                action_id: (count / total_actions) * 100
                for action_id, count in action_counts.items()
            }
            top_actions = sorted(
                [(action_id, percentage) for action_id, percentage in action_percentages.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]

            results[qt] = {
                "accuracy": accuracy,
                "total": metrics["total"],
                "correct": metrics["correct"],
                "top_actions": [
                    {
                        "action_id": int(action_id),
                        "description": self.get_action_description(int(action_id), qt),
                        "percentage": percentage
                    }
                    for action_id, percentage in top_actions
                ]
            }

        # Add overall results
        total_correct = sum(metrics["correct"] for metrics in question_type_metrics.values())
        total_examples = sum(metrics["total"] for metrics in question_type_metrics.values())

        if total_examples > 0:
            overall_accuracy = (total_correct / total_examples) * 100
        else:
            overall_accuracy = 0

        results["overall"] = {
            "accuracy": overall_accuracy,
            "total": total_examples,
            "correct": total_correct
        }

        logger.info(f"Evaluation completed with overall accuracy: {overall_accuracy:.2f}%")
        return results


# Create the training script for DPO
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train DPO for prompt optimization')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--action_space', type=str, required=True,
                        help='Path to action space configuration')
    parser.add_argument('--output_dir', type=str, default='dpo_output',
                        help='Output directory for models and plots')
    parser.add_argument('--num_pairs', type=int, default=1000,
                        help='Number of preference pairs to generate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter (regularization strength)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--load_pairs', type=str, default=None,
                        help='Path to load preference pairs from instead of generating new ones')

    # Add ablation-related arguments
    parser.add_argument('--ablation', type=str, default=None,
                        help='Action space ablation type (e.g., "action_unified", "action_minimal")')
    parser.add_argument('--model_variant', type=str, default=None,
                        help='Model architecture variant (e.g., "shared_heads", "deeper_network")')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for action selection (higher = more exploration)')
    parser.add_argument('--diversity_weight', type=float, default=0.0,
                        help='Weight for action diversity reward (0.0 = disabled)')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use question-type weighted loss')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    log_file = os.path.join(args.output_dir, "dpo_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Starting DPO Training")
    logger.info(f"Arguments: {args}")

    # Save config for later reference
    config = vars(args)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # Load configurations
    action_space = load_config(args.action_space)

    # Load datasets
    train_dataset = load_dataset(args.train_data)
    test_dataset = load_dataset(args.test_data)

    # Create preference pairs or load existing ones
    if args.load_pairs:
        preference_pairs = load_preference_pairs(args.load_pairs)
    else:
        preference_pairs = create_preference_pairs(
            train_dataset,
            action_space,
            num_pairs=args.num_pairs,
            batch_size=args.batch_size
        )

        # Save generated pairs
        pairs_path = os.path.join(args.output_dir, "preference_pairs.json")
        save_preference_pairs(preference_pairs, pairs_path)

    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Calculate input dimension
    input_dim = 768  # BERT's default embedding dimension

    # Initialize DPO Trainer with ablation parameters
    trainer = DPOTrainer(
        input_dim=input_dim,
        action_space=action_space,
        learning_rate=args.lr,
        beta=args.beta,
        ablation=args.ablation,
        model_variant=args.model_variant,
        temperature=args.temperature,
        diversity_weight=args.diversity_weight,
        weighted_loss=args.weighted_loss
    )

    # Train the model
    training_metrics = trainer.train(
        preference_pairs,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    # Save the final model
    trainer.save_model(path=os.path.join(args.output_dir, "final_model"))

    # Evaluate on test set
    results = trainer.evaluate(test_dataset, tokenizer)

    # Save results
    results_path = os.path.join(args.output_dir, "action_minimal_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training and evaluation completed. Results saved to {args.output_dir}")
    logger.info("=" * 60)

    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()