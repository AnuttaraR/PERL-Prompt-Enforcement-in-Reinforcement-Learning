import sys
from bart_scorer import BARTScorer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import json
import random
from tqdm import tqdm
import logging
import os
import time
from datetime import datetime
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now use direct imports
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from RL_Agent.utils.retrieval import retrieve_context

# Create logs directory if it doesn't exist
os.makedirs("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/logs", "ppo_model.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PPO_model")

nltk.download('punkt_tab', download_dir="C:\\Users\\USER\\PycharmProjects\\fyp-rnd\\venv\\nltk_data")

# Download NLTK resources if needed
nltk.download('wordnet')
nltk.download('punkt')

bart_scorer = BARTScorer(device="cuda" if torch.cuda.is_available() else "cpu")

# Load action spaces and reward configuration
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


ACTION_SPACE = "RL_Agent/config/action_space_config.json"
REWARD_CONFIG = "RL_Agent/config/reward_config.json"


# Load the tokenized dataset
def load_dataset(dataset_path):
    start_time = time.time()
    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded raw dataset with {len(data)} items")

    # Convert token lists to numpy arrays for easier processing
    token_count = 0
    for item in data:
        if 'input_ids' in item:
            item['tokens'] = np.array(item['input_ids'])
            token_count += len(item['input_ids'])
        else:
            # If for some reason input_ids is missing, use empty array
            item['tokens'] = np.array([])
            logger.warning(f"Missing 'input_ids' in dataset item: {item.get('question', 'unknown')}")

    # Log dataset statistics
    question_types = {}
    for item in data:
        qt = item.get('question_type', 'unknown')
        question_types[qt] = question_types.get(qt, 0) + 1

    logger.info(f"Dataset statistics:")
    logger.info(f"Total items: {len(data)}")
    logger.info(f"Average tokens per item: {token_count / len(data):.2f}")
    logger.info(f"Question types distribution: {question_types}")
    logger.info(f"Dataset loading completed in {time.time() - start_time:.2f} seconds")

    return data


# PPO Actor Network for selecting actions
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_counts):
        super(ActorNetwork, self).__init__()
        # Question-type specific networks
        self.shared_layer = nn.Linear(input_dim, 256)

        # Create separate output layers for each question type
        self.what_head = nn.Linear(256, action_counts["what"])
        self.how_head = nn.Linear(256, action_counts["how"])
        self.if_can_head = nn.Linear(256, action_counts["if_can"])

    def forward(self, state, question_type):
        x = F.relu(self.shared_layer(state))

        # Select the appropriate head based on question type
        if question_type == "what":
            action_probs = F.softmax(self.what_head(x), dim=-1)
        elif question_type == "how":
            action_probs = F.softmax(self.how_head(x), dim=-1)
        else:  # if_can
            action_probs = F.softmax(self.if_can_head(x), dim=-1)

        return action_probs


# PPO Critic Network for estimating state values
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.shared_layer = nn.Linear(input_dim, 256)

        # Separate value heads for each question type
        self.what_value = nn.Linear(256, 1)
        self.how_value = nn.Linear(256, 1)
        self.if_can_value = nn.Linear(256, 1)

    def forward(self, state, question_type):
        x = F.relu(self.shared_layer(state))

        # Select the appropriate value head
        if question_type == "what":
            value = self.what_value(x)
        elif question_type == "how":
            value = self.how_value(x)
        else:  # if_can
            value = self.if_can_value(x)

        return value


# Memory for storing experiences
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.old_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.question_types = []
        self.batch_size = batch_size

    def store(self, state, action, action_prob, value, reward, done, question_type):
        self.states.append(state)
        self.actions.append(action)
        self.old_probs.append(action_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.question_types.append(question_type)

    def clear(self):
        self.states = []
        self.actions = []
        self.old_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.question_types = []

    def sample_batch(self):
        batch_step = min(self.batch_size, len(self.states))
        indices = np.random.choice(len(self.states), batch_step, replace=False)

        return (
            torch.tensor(np.array(self.states)[indices], dtype=torch.float32),
            torch.tensor(np.array(self.actions)[indices], dtype=torch.long),
            torch.tensor(np.array(self.old_probs)[indices], dtype=torch.float32),
            torch.tensor(np.array(self.values)[indices], dtype=torch.float32),
            torch.tensor(np.array(self.rewards)[indices], dtype=torch.float32),
            torch.tensor(np.array(self.dones)[indices], dtype=torch.bool),
            [self.question_types[i] for i in indices]
        )

    def __len__(self):
        return len(self.states)


# Reward calculation functions
class RewardCalculator:
    def __init__(self, config):
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_bert_score(self, candidate, reference):
        """Calculate BERTScore between candidate and reference"""
        P, R, F1 = bert_score([candidate], [reference], lang="en", return_hash=False)
        return F1.item()

    def calculate_rouge_score(self, candidate, reference):
        """Calculate ROUGE-L score"""
        rouge = self.rouge_scorer.score(reference, candidate)
        return rouge['rougeL'].fmeasure

    def calculate_meteor_score(self, candidate, reference):
        """Calculate METEOR score"""
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())
        return meteor_score([reference_tokens], candidate_tokens)

    def calculate_bleu_score(self, candidate, reference):
        """Calculate BLEU score"""
        smoothing = SmoothingFunction().method1
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)

    def calculate_f1_score(self, candidate, reference):
        """Calculate F1 score based on token overlap"""
        candidate_tokens = set(nltk.word_tokenize(candidate.lower()))
        reference_tokens = set(nltk.word_tokenize(reference.lower()))

        common = len(candidate_tokens.intersection(reference_tokens))
        precision = common / len(candidate_tokens) if candidate_tokens else 0
        recall = common / len(reference_tokens) if reference_tokens else 0

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)

    def calculate_semantic_similarity(self, candidate, reference, question_type):
        """Calculate semantic similarity based on configured weights"""
        qt_key = f"{question_type}_question_rewards"

        if qt_key not in self.config:
            return 0

        if "semantic_similarity" not in self.config[qt_key]:
            return 0

        sim_config = self.config[qt_key]["semantic_similarity"]
        total_score = 0
        total_weight = 0

        if "bert_score_weight" in sim_config:
            bert_score_val = self.calculate_bert_score(candidate, reference)
            weight = sim_config["bert_score_weight"]
            total_score += bert_score_val * weight
            total_weight += weight

        if "meteor_weight" in sim_config:
            meteor_score_val = self.calculate_meteor_score(candidate, reference)
            weight = sim_config["meteor_weight"]
            total_score += meteor_score_val * weight
            total_weight += weight

        if "bart_score_weight" in sim_config:
            bart_score_val = bart_scorer.score([candidate], [reference])
            if isinstance(bart_score_val, list):
                bart_score_val = bart_score_val[0]
            weight = sim_config["bart_score_weight"]
            total_score += bart_score_val * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        return total_score / total_weight

    def calculate_lexical_overlap(self, candidate, reference, question_type):
        """Calculate lexical overlap based on configured weights"""
        qt_key = f"{question_type}_question_rewards"

        if qt_key not in self.config:
            return 0

        if "lexical_overlap" not in self.config[qt_key]:
            return 0

        overlap_config = self.config[qt_key]["lexical_overlap"]
        total_score = 0
        total_weight = 0

        if "rouge_l_weight" in overlap_config:
            rouge_score_val = self.calculate_rouge_score(candidate, reference)
            weight = overlap_config["rouge_l_weight"]
            total_score += rouge_score_val * weight
            total_weight += weight

        if "bleu_weight" in overlap_config:
            bleu_score_val = self.calculate_bleu_score(candidate, reference)
            weight = overlap_config["bleu_weight"]
            total_score += bleu_score_val * weight
            total_weight += weight

        if "f1_score_weight" in overlap_config:
            f1_score_val = self.calculate_f1_score(candidate, reference)
            weight = overlap_config["f1_score_weight"]
            total_score += f1_score_val * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        return total_score / total_weight

    def calculate_penalty(self, candidate, reference, question_type):
        """Calculate penalties based on configured weights"""
        penalties = 0

        # Get question-type specific hallucination penalty
        if "hallucination_penalty" in self.config["penalties"]:
            hallucination_config = self.config["penalties"]["hallucination_penalty"]
            if question_type in hallucination_config:
                # A simple hallucination detection: if semantic similarity is very low
                bert_score_val = self.calculate_bert_score(candidate, reference)
                if bert_score_val < 0.4:  # Threshold for hallucination detection
                    penalties += hallucination_config[f"{question_type}_questions"]

        # Get question-type specific ambiguity penalty
        if "ambiguity_penalty" in self.config["penalties"]:
            ambiguity_config = self.config["penalties"]["ambiguity_penalty"]
            if question_type in ambiguity_config:
                # A simple ambiguity detection: if the answer is too short
                if len(candidate.split()) < 10:
                    penalties += ambiguity_config[f"{question_type}_questions"]

        # Apply length penalty if answer is too long
        if "length_penalty" in self.config["penalties"]:
            if len(candidate.split()) > 150:  # Threshold for excessive length
                penalties += self.config["penalties"]["length_penalty"]

        # Apply token efficiency penalty
        if "token_efficiency_penalty" in self.config["penalties"]:
            # Calculate information density (semantic similarity / token count)
            token_count = len(candidate.split())
            if token_count > 0:
                bert_score_val = self.calculate_bert_score(candidate, reference)
                info_density = bert_score_val / token_count
                if info_density < 0.005:  # Threshold for inefficient responses
                    penalties += self.config["penalties"]["token_efficiency_penalty"]

        return penalties

    def calculate_reward(self, candidate, reference, question_type):
        """Calculate the total reward for a response"""
        qt_key = f"{question_type}_question_rewards"

        # Calculate base rewards
        base_reward = 0
        if "base_rewards" in self.config:
            if "relevance" in self.config["base_rewards"]:
                # Use BERTScore as a measure of relevance
                relevance = self.calculate_bert_score(candidate, reference)
                base_reward += relevance * self.config["base_rewards"]["relevance"]

            if "coherence" in self.config["base_rewards"]:
                # Use ROUGE-L as a proxy for coherence
                coherence = self.calculate_rouge_score(candidate, reference)
                base_reward += coherence * self.config["base_rewards"]["coherence"]

        # Calculate question-type specific rewards
        type_specific_reward = 0
        if qt_key in self.config:
            question_config = self.config[qt_key]

            # Calculate semantic similarity
            semantic_sim = self.calculate_semantic_similarity(candidate, reference, question_type)

            # Calculate lexical overlap
            lexical_overlap = self.calculate_lexical_overlap(candidate, reference, question_type)

            # Combine with question-type specific weights
            if "definitional_clarity" in question_config and question_type == "what":
                type_specific_reward += semantic_sim * question_config["definitional_clarity"]
                type_specific_reward += lexical_overlap * question_config["definitional_clarity"]

            if "procedural_completeness" in question_config and question_type == "how":
                type_specific_reward += semantic_sim * question_config["procedural_completeness"]
                type_specific_reward += lexical_overlap * question_config["procedural_completeness"]

            if "conditional_analysis" in question_config and question_type == "if_can":
                type_specific_reward += semantic_sim * question_config["conditional_analysis"]
                type_specific_reward += lexical_overlap * question_config["conditional_analysis"]

        # Calculate penalties
        penalties = self.calculate_penalty(candidate, reference, question_type)

        # Combine all components
        total_reward = base_reward + type_specific_reward + penalties

        # Log the reward components for debugging
        logger.debug(
            f"Reward components: base={base_reward}, type_specific={type_specific_reward}, penalties={penalties}")

        return total_reward


# PPO Agent
class PPOAgent:
    def __init__(self, input_dim, action_space, reward_config, batch_size=64,
                 learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):

        # Set global variables for easier access
        global ACTION_SPACE, REWARD_CONFIG
        ACTION_SPACE = action_space
        REWARD_CONFIG = reward_config

        logger.info(f"Initializing PPO Agent with:")
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Discount factor (gamma): {gamma}")
        logger.info(f"GAE lambda: {gae_lambda}")
        logger.info(f"Policy clip: {policy_clip}")
        logger.info(f"Update epochs: {epochs}")

        # Count actions per question type
        self.action_counts = {
            "what": len(action_space["what_question_actions"]),
            "how": len(action_space["how_question_actions"]),
            "if_can": len(action_space["if_can_question_actions"])
        }

        # Add general actions to each count
        for qt in self.action_counts:
            self.action_counts[qt] += len(action_space["general_actions"])

        logger.info(f"Action counts per question type: {self.action_counts}")

        # Initialize networks
        logger.info("Initializing actor and critic networks")
        self.actor = ActorNetwork(input_dim, self.action_counts).to(device)
        self.critic = CriticNetwork(input_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = PPOMemory(batch_size)
        self.reward_calculator = RewardCalculator(reward_config)

        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.device = device

        # Track metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.question_type_rewards = {"what": [], "how": [], "if_can": []}

        logger.info("PPO Agent initialization complete")

    def get_action(self, state, question_type):
        """Select an action based on the current policy"""
        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
        action_probs = self.actor(state_tensor, question_type).cpu().detach().numpy()[0]

        # Sample action from the probability distribution
        action_distribution = Categorical(torch.tensor(action_probs))
        action = action_distribution.sample().item()
        action_prob = action_probs[action]

        # Get the value estimate
        value = self.critic(state_tensor, question_type).cpu().detach().numpy()[0]

        logger.debug(f"Selected action {action} with probability {action_prob:.4f} for {question_type} question")

        return action, action_prob, value

    def get_action_description(self, action, question_type):
        general_actions = self.action_space["general_actions"]
        general_count = len(general_actions)

        # Check if it's a general action
        if action < general_count:
            action_key = str(action)
            return general_actions[action_key]

        # Adjust index for question-specific actions
        qt_actions = self.action_space[f"{question_type}_question_actions"]

        # For the first iteration, use the first action of the question type
        mapped_key = str(action - general_count + 1)

        return qt_actions.get(mapped_key, "No change, return exactly as it is.")

    def calculate_advantages(self, values, rewards, dones):
        """Calculate advantages using GAE"""
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage

        return advantages

    def update_policy(self):
        """Update the policy using PPO"""
        if len(self.memory) < self.batch_size:
            logger.debug(f"Not enough samples in memory ({len(self.memory)}/{self.batch_size}), skipping update")
            return

        update_start_time = time.time()
        logger.info(f"Starting policy update with {len(self.memory)} samples")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            states, actions, old_probs, values, rewards, dones, question_types = self.memory.sample_batch()

            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_probs = old_probs.to(self.device)
            values = values.to(self.device)
            rewards = rewards.to(self.device)

            # Calculate advantages
            advantages = torch.tensor(
                self.calculate_advantages(values.cpu().numpy(),
                                          rewards.cpu().numpy(),
                                          dones.cpu().numpy()),
                dtype=torch.float32
            ).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute returns
            returns = advantages + values

            # Update actor
            actor_loss = 0
            critic_loss = 0

            for i in range(len(states)):
                state = states[i].unsqueeze(0)
                action = actions[i]  # Keep as tensor
                old_prob = old_probs[i]  # Keep as tensor
                advantage = advantages[i]  # Keep as tensor

                if returns.dim() > 1:
                    # If returns is 2D with shape [batch_size, 1]
                    return_val = returns[i, 0] if returns.size(1) == 1 else returns[i].mean()
                else:
                    # If returns is 1D
                    return_val = returns[i]

                qt = question_types[i]

                # Get new action probability
                action_probs = self.actor(state, qt)
                dist = Categorical(action_probs)
                new_prob = dist.probs[0, action]  # Keep as tensor

                # Compute probability ratio
                ratio = new_prob / (old_prob + 1e-8)
                weighted_probs = advantage * ratio
                weighted_clipped_probs = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage

                # Actor loss
                actor_loss -= torch.min(weighted_probs, weighted_clipped_probs)

                # Critic loss
                critic_value = self.critic(state, qt)
                critic_loss += F.mse_loss(critic_value, return_val.reshape(-1, 1))

            # Average losses
            actor_loss = actor_loss / len(states)
            critic_loss = critic_loss / len(states)

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss

            # Log losses
            self.actor_losses.append(actor_loss.item())  # Now convert to scalar for logging
            self.critic_losses.append(critic_loss.item())  # Now convert to scalar for logging

            # Perform optimization
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            epoch_time = time.time() - epoch_start_time
            logger.debug(f"Epoch {epoch + 1}/{self.epochs} completed in {epoch_time:.2f}s. "
                         f"Actor loss: {actor_loss.item():.5f}, Critic loss: {critic_loss.item():.5f}")

        # Clear memory after updates
        self.memory.clear()

        update_time = time.time() - update_start_time
        logger.info(f"Policy update completed in {update_time:.2f}s. Final actor loss: {actor_loss.item():.5f}, Final critic loss: {critic_loss.item():.5f}")
        print(f"Policy update completed in {update_time:.2f}s. Final actor loss: {actor_loss.item():.5f}, Final critic loss: {critic_loss.item():.5f}")

    def save_model(self, path="ppo_model"):
        """Save the model weights"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

        # Save metadata about the model
        metadata = {
            "action_counts": self.action_counts,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode_rewards": self.episode_rewards[-10:] if len(self.episode_rewards) > 10 else self.episode_rewards,
            "actor_losses": self.actor_losses[-10:] if len(self.actor_losses) > 10 else self.actor_losses,
            "critic_losses": self.critic_losses[-10:] if len(self.critic_losses) > 10 else self.critic_losses,
            "question_type_rewards": {
                qt: rewards[-10:] if len(rewards) > 10 else rewards
                for qt, rewards in self.question_type_rewards.items() if rewards
            }
        }

        with open(f"{path}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Model saved to {path}")

    def load_model(self, path="ppo_model"):
        """Load the model weights"""
        if not os.path.exists(f"{path}/actor.pth") or not os.path.exists(f"{path}/critic.pth"):
            logger.error(f"Model files not found at {path}")
            return False

        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))

        # Load metadata if available
        metadata_path = f"{path}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model saved at {metadata.get('saved_at', 'unknown time')}")
            logger.info(f"Model action counts: {metadata.get('action_counts', 'unknown')}")
        else:
            logger.info(f"Model loaded from {path} (no metadata available)")

        return True

    def train(self, train_dataset, num_episodes=1000, max_steps=5):
        """Train the PPO agent"""
        total_start_time = time.time()
        logger.info(f"Starting training for {num_episodes} episodes with max {max_steps} steps per episode")
        logger.info(f"Training dataset contains {len(train_dataset)} examples")

        episode_rewards = []
        best_avg_reward = float('-inf')

        # Track question type distribution in training
        qt_counts = {"what": 0, "how": 0, "if_can": 0}
        qt_action_counts = {
            "what": {i: 0 for i in range(self.action_counts["what"])},
            "how": {i: 0 for i in range(self.action_counts["how"])},
            "if_can": {i: 0 for i in range(self.action_counts["if_can"])}
        }

        # Create progress bar
        progress_bar = tqdm(range(num_episodes), desc="Training Episodes")

        for episode in progress_bar:
            episode_start_time = time.time()

            # Randomly select a question from the training dataset
            question_data = random.choice(train_dataset)
            question = question_data["question"]
            ground_truth = question_data["ground_truth"]
            question_type = question_data["question_type"]

            episode_context = retrieve_context(question, top_k=3)

            # Update question type counter
            qt_counts[question_type] = qt_counts.get(question_type, 0) + 1

            # Get the tokens - these will be used as our state representation
            state = question_data["tokens"]

            # Check if tokens are available
            if len(state) == 0:
                logger.warning(f"Skipping question with empty tokens: {question}")
                continue

            episode_reward = 0

            logger.debug(f"Episode {episode + 1}: Processing {question_type} question: '{question}...'")

            for step in range(max_steps):
                step_start_time = time.time()

                # Select action
                action, action_prob, value = self.get_action(state, question_type)

                # Update action counter
                qt_action_counts[question_type][action] = qt_action_counts[question_type].get(action, 0) + 1

                # Get action description
                action_desc = self.get_action_description(action, question_type)

                logger.debug(f"Episode {episode + 1}, Step {step + 1}: Selected action '{action_desc}'")

                # Modify the prompt using the selected action
                prompt_modification_instruction = f"{action_desc}: {question}"
                logger.debug(
                    f"Episode {episode + 1}, Step {step + 1}: Sending to LLM: '{prompt_modification_instruction}'")

                llm_start_time = time.time()
                modified_prompt = get_llm_response(prompt_modification_instruction, model="gpt-3.5-turbo")
                llm_time = time.time() - llm_start_time

                # If we get an invalid response, use the original prompt
                invalid_responses = ["INVALID", "Invalid input", "Invalid response", "Error"]
                invalid_response = False  # Add this flag
                if modified_prompt == "INVALID" or any(
                        modified_prompt.strip().lower() == resp.lower() for resp in invalid_responses):
                    logger.warning(f"Invalid response from LLM: '{modified_prompt}', using original prompt")
                    modified_prompt = question
                    invalid_response = True  # Apply penalty for invalid responses

                logger.debug(f"Episode {episode + 1}, Step {step + 1}: Modified prompt: '{modified_prompt[:50]}...'")

                # Get top chunks from Pinecone
                retrieval_start_time = time.time()
                # context_chunks = retrieve_context(question, top_k=3)
                retrieval_time = time.time() - retrieval_start_time

                # Generate answer using the modified prompt and context chunks
                final_prompt = f"Question: {modified_prompt}\n\nContext Information: {episode_context}\n\nPlease answer the question based on the provided context."

                answer_start_time = time.time()
                answer = generate_answer_from_llm(final_prompt, model="gpt-3.5-turbo")
                answer_time = time.time() - answer_start_time

                logger.debug(f"Episode {episode + 1}, Step {step + 1}: Generated answer of length {len(answer)}")

                # Calculate reward
                reward_start_time = time.time()
                reward = self.reward_calculator.calculate_reward(answer, ground_truth, question_type)
                if invalid_response:
                    reward -= 0.5
                reward_time = time.time() - reward_start_time

                logger.debug(f"☑️ Episode {episode + 1}, Step {step + 1}: Reward: {reward:.4f}")
                print(f"☑️ Episode {episode + 1}, Step {step + 1}: Reward: {reward:.4f}")

                # Store the experience in memory
                done = (step == max_steps - 1)
                self.memory.store(state, action, action_prob, value, reward, done, question_type)

                # Accumulate episode reward
                episode_reward += reward

                # Log step timing information
                step_time = time.time() - step_start_time
                logger.debug(f"Episode {episode + 1}, Step {step + 1} timing: "
                             f"Total: {step_time:.2f}s, LLM: {llm_time:.2f}s, "
                             f"Retrieval: {retrieval_time:.2f}s, Answer: {answer_time:.2f}s, "
                             f"Reward: {reward_time:.2f}s")

                # Update policy if we have enough samples
                if len(self.memory) >= self.batch_size:
                    self.update_policy()

                # Early stopping condition - if we found a good action
                if reward > 0.8:
                    logger.info(f"🥇 Found excellent action (reward {reward:.4f}), ending episode early")
                    print(f"🥇 Found excellent action (reward {reward:.4f}), ending episode early")
                    break

            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.question_type_rewards[question_type].append(episode_reward)

            # Calculate episode time
            episode_time = time.time() - episode_start_time

            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
                self.episode_rewards)
            progress_bar.set_postfix({
                'avg_reward': f'{avg_reward:.3f}',
                f'{question_type}_reward': f'{episode_reward:.3f}',
                'time': f'{episode_time:.1f}s'
            })

            # Log episode information
            logger.info(f" ✅ Episode {episode + 1}/{num_episodes} completed in {episode_time:.2f}s. "
                        f"Reward: {episode_reward:.4f}, Type: {question_type}, "
                        f"Running avg reward: {avg_reward:.4f}")
            print(f" ✅ Episode {episode + 1}/{num_episodes} completed in {episode_time:.2f}s. "
                        f"Reward: {episode_reward:.4f}, Type: {question_type}, "
                        f"Running avg reward: {avg_reward:.4f}")

            # Save the best model
            if avg_reward > best_avg_reward and episode > 100:
                best_avg_reward = avg_reward
                self.save_model(path="best_ppo_model")
                logger.info(f"🌟 New best model saved with average reward: {best_avg_reward:.3f}")

            # Save periodically
            if episode % 50 == 0 and episode > 0:
                self.save_model(path=f"checkpoint_episode_{episode}")
                logger.info(f"Checkpoint saved at episode {episode}")
                self.log_metrics()

        # Log final statistics
        total_time = time.time() - total_start_time
        logger.info(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f} minutes)")
        logger.info(f"Question type distribution: {qt_counts}")

        # Log action selection distribution
        for qt, action_counts in qt_action_counts.items():
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                percentages = {action: count / total_actions * 100 for action, count in action_counts.items()}
                top_actions = sorted(percentages.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"Top actions for {qt} questions: {top_actions}")

        # Save the final model
        self.save_model(path="final_ppo_model")
        logger.info(f"Final model saved to 'final_ppo_model'")

        return self.episode_rewards

    def evaluate(self, test_dataset):
        """Evaluate the agent on test dataset"""
        eval_start_time = time.time()
        logger.info(f"Starting evaluation on {len(test_dataset)} test examples")

        self.actor.eval()
        self.critic.eval()

        question_type_metrics = {
            "what": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []},
            "how": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []},
            "if_can": {"rewards": [], "bert_scores": [], "rouge_scores": [], "meteor_scores": []}
        }

        # Track action selection
        action_selections = {
            "what": {},
            "how": {},
            "if_can": {}
        }

        progress_bar = tqdm(test_dataset, desc="Evaluating")

        for idx, question_data in enumerate(progress_bar):
            eval_item_start = time.time()

            question = question_data["question"]
            ground_truth = question_data["ground_truth"]
            question_type = question_data["question_type"]

            # Get the tokens from the dataset
            state = question_data["tokens"]

            # Skip questions with empty tokens
            if len(state) == 0:
                logger.warning(f"Skipping evaluation for question with empty tokens: {question}")
                continue

            logger.debug(f"Evaluating test item {idx + 1}: {question_type} question: '{question[:50]}...'")

            # Select best action (deterministic policy)
            with torch.no_grad():
                state_tensor = torch.tensor([state], dtype=torch.float32).to(self.device)
                action_probs = self.actor(state_tensor, question_type).cpu().numpy()[0]
                action = np.argmax(action_probs)

                # Track action selection
                action_selections[question_type][action] = action_selections[question_type].get(action, 0) + 1

            # Get action description
            action_desc = self.get_action_description(action, question_type)
            logger.debug(f"Selected action: {action_desc}")

            # Modify the prompt using the selected action
            prompt_modification_instruction = f"{action_desc}: {question}"

            llm_start_time = time.time()
            modified_prompt = get_llm_response(prompt_modification_instruction, model="gpt-3.5-turbo")
            llm_time = time.time() - llm_start_time

            invalid_responses = ["INVALID", "Invalid input", "Invalid response", "Error"]
            invalid_response = False  # Add flag
            if modified_prompt == "INVALID" or any(
                    modified_prompt.strip().lower() == resp.lower() for resp in invalid_responses):
                logger.warning(f"Invalid response from LLM: '{modified_prompt}', using original prompt")
                modified_prompt = question
                invalid_response = True  # Set flag

            # Get top chunks from Pinecone
            retrieval_start_time = time.time()
            context_chunks = retrieve_context(question, top_k=3)
            retrieval_time = time.time() - retrieval_start_time

            # Generate answer using the modified prompt and context chunks
            final_prompt = f"Question: {modified_prompt}\n\nContext Information: {context_chunks}\n\nPlease answer the question based on the provided context."

            answer_start_time = time.time()
            answer = generate_answer_from_llm(final_prompt, model="gpt-3.5-turbo")
            answer_time = time.time() - answer_start_time

            # Calculate metrics
            metrics_start_time = time.time()
            reward = self.reward_calculator.calculate_reward(answer, ground_truth, question_type)

            if invalid_response:
                reward -= 0.5

            bert_score_val = self.reward_calculator.calculate_bert_score(answer, ground_truth)
            rouge_score_val = self.reward_calculator.calculate_rouge_score(answer, ground_truth)
            meteor_score_val = self.reward_calculator.calculate_meteor_score(answer, ground_truth)
            metrics_time = time.time() - metrics_start_time

            # Store metrics
            question_type_metrics[question_type]["rewards"].append(reward)
            question_type_metrics[question_type]["bert_scores"].append(bert_score_val)
            question_type_metrics[question_type]["rouge_scores"].append(rouge_score_val)
            question_type_metrics[question_type]["meteor_scores"].append(meteor_score_val)

            # Log metrics for this item
            logger.debug(f"Test item {idx + 1} metrics: "
                         f"Reward: {reward:.4f}, BERTScore: {bert_score_val:.4f}, "
                         f"ROUGE-L: {rouge_score_val:.4f}, METEOR: {meteor_score_val:.4f}")

            # Log timing information
            eval_item_time = time.time() - eval_item_start
            logger.debug(f"Test item {idx + 1} timing: "
                         f"Total: {eval_item_time:.2f}s, LLM: {llm_time:.2f}s, "
                         f"Retrieval: {retrieval_time:.2f}s, Answer: {answer_time:.2f}s, "
                         f"Metrics: {metrics_time:.2f}s")

            # Update progress bar with reward information
            progress_bar.set_postfix({
                'reward': f'{reward:.3f}',
                'bertscore': f'{bert_score_val:.3f}'
            })

        # Calculate and log average metrics
        logger.info(f"Evaluation results:")

        all_rewards = []
        for qt, metrics in question_type_metrics.items():
            if not metrics["rewards"]:
                logger.warning(f"No evaluation data for question type: {qt}")
                continue

            avg_reward = np.mean(metrics["rewards"])
            avg_bert = np.mean(metrics["bert_scores"])
            avg_rouge = np.mean(metrics["rouge_scores"])
            avg_meteor = np.mean(metrics["meteor_scores"])

            all_rewards.extend(metrics["rewards"])

            logger.info(f"Question Type: {qt}")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")
            logger.info(f"  Avg BERTScore: {avg_bert:.3f}")
            logger.info(f"  Avg ROUGE-L: {avg_rouge:.3f}")
            logger.info(f"  Avg METEOR: {avg_meteor:.3f}")

            # Log action distribution
            if action_selections[qt]:
                total_actions = sum(action_selections[qt].values())
                action_percentages = {action: count / total_actions * 100 for action, count in
                                      action_selections[qt].items()}
                top_actions = sorted(action_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"  Top actions: {top_actions}")

        # Include action selections in the metrics
        for qt in question_type_metrics:
            question_type_metrics[qt]["action_selections"] = action_selections[qt]

        # Log overall average
        if all_rewards:
            logger.info(f"Overall average reward: {np.mean(all_rewards):.3f}")

        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_time:.2f}s ({eval_time / 60:.2f} minutes)")

        self.actor.train()
        self.critic.train()

        return {"action_selections": action_selections, **question_type_metrics}

    def log_metrics(self):
        """Log training metrics"""
        logger.info("===== Current Training Metrics =====")

        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
            self.episode_rewards)

        avg_what_reward = np.mean(self.question_type_rewards["what"][-100:]) if len(
            self.question_type_rewards["what"]) >= 100 else np.mean(self.question_type_rewards["what"] or [0])
        avg_how_reward = np.mean(self.question_type_rewards["how"][-100:]) if len(
            self.question_type_rewards["how"]) >= 100 else np.mean(self.question_type_rewards["how"] or [0])
        avg_if_can_reward = np.mean(self.question_type_rewards["if_can"][-100:]) if len(
            self.question_type_rewards["if_can"]) >= 100 else np.mean(self.question_type_rewards["if_can"] or [0])

        avg_actor_loss = np.mean(self.actor_losses[-100:]) if len(self.actor_losses) >= 100 else np.mean(
            self.actor_losses or [0])
        avg_critic_loss = np.mean(self.critic_losses[-100:]) if len(self.critic_losses) >= 100 else np.mean(
            self.critic_losses or [0])

        # Count questions per type
        qt_counts = {
            "what": len(self.question_type_rewards["what"]),
            "how": len(self.question_type_rewards["how"]),
            "if_can": len(self.question_type_rewards["if_can"])
        }

        logger.info(f"Episodes completed: {len(self.episode_rewards)}")
        logger.info(f"Question distribution: {qt_counts}")
        logger.info(f"Average Reward (last 100): {avg_reward:.3f}")
        logger.info(
            f"Question Type Rewards - What: {avg_what_reward:.3f}, How: {avg_how_reward:.3f}, If/Can: {avg_if_can_reward:.3f}")
        logger.info(f"Average Losses - Actor: {avg_actor_loss:.3f}, Critic: {avg_critic_loss:.3f}")

        # Create and save learning curves
        self.plot_learning_curves()

    def plot_learning_curves(self):
        """Create and save plots of learning curves"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs("plots", exist_ok=True)

            # Plot overall reward curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.episode_rewards, alpha=0.3, label='Raw')

            # Add smoothed curve if we have enough data
            if len(self.episode_rewards) > 10:
                window_size = min(50, len(self.episode_rewards) // 5)
                smoothed = np.convolve(self.episode_rewards, np.ones(window_size) / window_size, mode='valid')
                plt.plot(range(window_size - 1, window_size - 1 + len(smoothed)), smoothed, linewidth=2,
                         label=f'Smoothed (window={window_size})')

            plt.title('Training Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("plots/reward_curve.png")
            plt.close()

            # Plot rewards by question type
            plt.figure(figsize=(10, 6))
            for qt, rewards in self.question_type_rewards.items():
                if len(rewards) > 0:
                    plt.plot(rewards, label=f'{qt} Questions', alpha=0.3)

                    # Add smoothed curves if we have enough data
                    if len(rewards) > 10:
                        window_size = min(20, len(rewards) // 5)
                        smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                        plt.plot(range(window_size - 1, window_size - 1 + len(smoothed)), smoothed, linewidth=2,
                                 label=f'{qt} Smoothed')

            plt.title('Rewards by Question Type')
            plt.xlabel('Question #')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("plots/question_type_rewards.png")
            plt.close()

            # Plot actor and critic losses
            if self.actor_losses and self.critic_losses:
                plt.figure(figsize=(10, 6))
                plt.plot(self.actor_losses, label='Actor Loss', alpha=0.3)
                plt.plot(self.critic_losses, label='Critic Loss', alpha=0.3)

                # Add smoothed curves if we have enough data
                if len(self.actor_losses) > 10:
                    window_size = min(20, len(self.actor_losses) // 5)
                    smoothed_actor = np.convolve(self.actor_losses, np.ones(window_size) / window_size, mode='valid')
                    smoothed_critic = np.convolve(self.critic_losses, np.ones(window_size) / window_size, mode='valid')

                    plt.plot(range(window_size - 1, window_size - 1 + len(smoothed_actor)), smoothed_actor,
                             linewidth=2, label='Actor Loss (Smoothed)')
                    plt.plot(range(window_size - 1, window_size - 1 + len(smoothed_critic)), smoothed_critic,
                             linewidth=2, label='Critic Loss (Smoothed)')

                plt.title('Policy Network Losses')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig("plots/network_losses.png")
                plt.close()

            logger.info(f"Learning curves plotted and saved to 'plots/' directory")
        except Exception as e:
            logger.error(f"Error plotting learning curves: {e}")
