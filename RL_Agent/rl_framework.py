import os
import logging
import sys
import json
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from RL_Agent.utils.retrieval import retrieve_context
from RL_Agent.utils.evaluation import evaluate_metrics
from RL_Agent.utils.query_llm import get_llm_response, generate_answer_from_llm
from transformers import AutoTokenizer
from RL_Agent.bart_scorer import BARTScorer

# Check if TRL (for DPO) is available
try:
    from trl import DPOTrainer

    DPO_AVAILABLE = True
except ImportError:
    DPO_AVAILABLE = False

TOKENIZED_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/tokenized_dataset.json"
TRAIN_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/train_data.json"
TEST_DATA_PATH = "C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/data/test_data.json"
TOKENIZER_MODEL = "bert-base-uncased"
INVALID_RESPONSE = "INVALID"

# Define log file path
LOG_DIR = "/RL_Agent/ppo_results/results"
LOG_FILE = os.path.join(LOG_DIR, "ppo_training_log.txt")

# Create the directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to write to a file + show in console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Overwrites log file on each run
        logging.StreamHandler(sys.stdout)  # Keeps showing logs in console
    ]
)

# Redirect print() and errors to the same log file
sys.stdout = sys.__stdout__  # Keep console output
sys.stderr = sys.__stderr__  # Keep error output
logging.info(f"Logging setup complete. Logs also saved to {LOG_FILE}")


def compute_final_metrics(evaluation_data, bart_scorer, tokenizer):
    """Compute final evaluation metrics and log them after training is fully done."""
    logging.info("\n✅ Training complete! Computing final evaluation metrics...")

    if not evaluation_data["generated_answers"]:
        logging.warning("No evaluation data collected. Skipping metric computation.")
        return

    # Compute metrics
    metrics = evaluate_metrics(
        evaluation_data["ground_truths"],
        evaluation_data["generated_answers"],
        bart_scorer,
        tokenizer
    )

    # Log and print metrics
    logging.info(f"\n📊 FINAL EVALUATION METRICS:\n{json.dumps(metrics, indent=4)}")
    print("\n📊 FINAL EVALUATION METRICS:")
    print(json.dumps(metrics, indent=4))

    # Save metrics to a JSON file
    metrics_path = "/RL_Agent/ppo_results/results/ppo_final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"📄 Final performance metrics saved at {metrics_path}")


class CustomPromptEnv(gym.Env):
    def __init__(self, action_config, reward_config, dataset, model_type="gpt-4", remove_bert_score=False,
                 remove_retrieval=False):
        self.dataset = dataset
        super(CustomPromptEnv, self).__init__()

        logging.info("Initializing CustomPromptEnv")

        self.action_space_config = self.load_action_space(action_config)
        self.reward_config = self.load_reward_config(reward_config, remove_bert_score, remove_retrieval)
        self.dataset = self.load_tokenized_dataset()
        self.current_index = 0
        self.total_samples = len(self.dataset)  # Total dataset size
        self.epoch = 0  # Epoch counter
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        self.action_tokens = [f"<ACTION_{i}>" for i in range(len(self.action_space_config))]
        self.tokenizer.add_tokens(self.action_tokens)
        self.evaluation_data = {
            "generated_answers": [],
            "ground_truths": [],
            "rewards": [],
        }
        # Add episode tracking
        self.max_attempts_per_prompt = 5  # Max attempts before moving to next prompt
        self.current_attempt = 0  # Track attempts for current prompt
        self.reward_threshold = 1.5  # Threshold for "good enough" reward
        self.best_reward_for_current = -float('inf')  # Track best reward for current prompt

        # ✅ Resize model embeddings if a transformer-based model is being used
        if hasattr(self, "model") and hasattr(self.model, "resize_token_embeddings"):
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info(f"Resized model embeddings to match new tokenizer size: {len(self.tokenizer)}")

        self.bart_scorer = BARTScorer(device="cuda" if torch.cuda.is_available() else "cpu")

        self.action_embedding = nn.Embedding(len(self.action_space_config),
                                             768)  # 768-dim for transformer compatibility
        self.action_embedding.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available

        # Define state and action spaces for tokenized inputs
        self.question_type_embedding = nn.Embedding(3, 768)  # 3 types: what, how, if/can
        self.question_type_map = {"what": 0, "how": 1, "if_can": 2}

        self.observation_space = gym.spaces.Dict({
            "input_ids": gym.spaces.Box(low=0, high=self.tokenizer.vocab_size, shape=(512,), dtype=np.int32),
            "attention_mask": gym.spaces.Box(low=0, high=1, shape=(512,), dtype=np.int32),
            "question_type": gym.spaces.Discrete(3)  # Add question type to state
        })

        self.action_space = gym.spaces.Discrete(len(self.action_space_config))

    def load_tokenized_dataset(self):
        """Load the pre-tokenized dataset and verify its structure."""
        logging.info("Loading tokenized dataset")
        with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if not isinstance(dataset, list):
            raise ValueError("Tokenized dataset should be a list of dictionaries.")

        for i, entry in enumerate(dataset):
            if not isinstance(entry, dict):
                raise ValueError(f"Dataset entry {i} is not a dictionary: {entry}")
            if "input_ids" not in entry or not isinstance(entry["input_ids"], list):
                raise ValueError(f"Dataset entry {i} is missing 'input_ids' or it's not a list.")
            if "attention_mask" not in entry or not isinstance(entry["attention_mask"], list):
                raise ValueError(f"Dataset entry {i} is missing 'attention_mask' or it's not a list.")
            if "ground_truth" not in entry or not isinstance(entry["ground_truth"], str):
                raise ValueError(f"Dataset entry {i} is missing 'ground_truth' or it's not a string.")

        return dataset

    def reset(self, seed=None, options=None):
        """Reset environment for a new episode while keeping track of prompt attempts."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Move to next prompt only if max attempts reached
        if self.current_attempt >= self.max_attempts_per_prompt or self.best_reward_for_current >= self.reward_threshold:
            self.current_index = (self.current_index + 1) % self.total_samples
            self.current_attempt = 0  # Reset attempt counter
            self.best_reward_for_current = -float('inf')  # Reset best reward tracker

        sample = self.dataset[self.current_index]
        logging.info(
            f"Reset for prompt {self.current_index}, attempt {self.current_attempt + 1}/{self.max_attempts_per_prompt}")

        # Retrieve context for the question
        self.current_question = sample["question"]
        self.current_context = retrieve_context(self.current_question, top_k=3)
        logging.info(f"🔍 Retrieved Context for '{self.current_question}': {self.current_context[:500]}")

        # Get question type for observation
        question_type = self.question_type_map[sample["question_type"]]

        obs = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "question_type": question_type,
            "gt_input_ids": torch.tensor(sample["gt_input_ids"], dtype=torch.long),
            "gt_attention_mask": torch.tensor(sample["gt_attention_mask"], dtype=torch.long)
        }

        return obs, {}

    def step(self, action):
        """Modify prompt, get LLM response, compute reward, and return observation."""
        self.current_attempt += 1
        logging.info(
            f"Step: Prompt {self.current_index}, Attempt {self.current_attempt}/{self.max_attempts_per_prompt}")

        # ✅ Only wrap around when actually moving to next prompt
        if self.current_index >= self.total_samples:
            logging.info("🚨 Dataset index exceeded total samples. Wrapping around.")
            self.current_index = 0

        if not isinstance(action, (int, np.integer)):
            action = int(action)
        action_instruction = self.action_space_config[action]

        # Track visit counts per attempt
        if "visit_count" not in self.dataset[self.current_index]:
            self.dataset[self.current_index]["visit_count"] = 0
        self.dataset[self.current_index]["visit_count"] += 1
        visit_count = self.dataset[self.current_index]["visit_count"]

        # Get current prompt info
        original_plain_prompt = self.dataset[self.current_index]["question"]
        question_type = self.dataset[self.current_index]["question_type"]
        original_prompt = f"""Instruction: {action_instruction}\nQuestion Type: {question_type}\nOriginal Question: {original_plain_prompt}\n"""
        logging.info(f"📝 Current Prompt [{self.current_index}, Attempt {self.current_attempt}]: {original_prompt}")

        # First LLM call - modify prompt
        modified_prompt = get_llm_response(original_prompt, self.model_type, tokenizer=self.tokenizer)
        logging.info(f"🛠️ Modified Prompt: {modified_prompt}")

        if modified_prompt.lower() == "INVALID":
            logging.warning(f"⚠️ LLM returned 'INVALID' response. Ending attempt with penalty.")
            reward = -2.0
            # Don't move to next prompt, just mark attempt as done
            done = True
            encoded_obs = self.tokenizer(original_prompt, padding="max_length", truncation=True, max_length=512,
                                         return_tensors="pt")
            return {
                "input_ids": encoded_obs["input_ids"].squeeze(0).numpy(),
                "attention_mask": encoded_obs["attention_mask"].squeeze(0).numpy(),
                "question_type": self.question_type_map[question_type]
            }, reward, done, False, {}

        # Second LLM call - generate answer
        final_prompt = f"Question: {modified_prompt}\n\nContext: {self.current_context}\nAnswer:"
        generated_answer = generate_answer_from_llm(final_prompt, self.model_type)
        ground_truth = self.dataset[self.current_index]["ground_truth"]

        # Calculate reward and update best
        reward = self.calculate_reward(generated_answer, ground_truth)
        self.best_reward_for_current = max(self.best_reward_for_current, reward)
        logging.info(f"🏆 Attempt {self.current_attempt} Reward = {reward}, Best = {self.best_reward_for_current}")

        # Store evaluation data
        self.evaluation_data["generated_answers"].append(generated_answer)
        self.evaluation_data["ground_truths"].append(ground_truth)
        self.evaluation_data["rewards"].append(reward)

        # Check if episode should end
        done = (self.current_attempt >= self.max_attempts_per_prompt) or (reward >= self.reward_threshold)

        # Prepare next observation
        encoded_obs = self.tokenizer(modified_prompt, padding="max_length", truncation=True, max_length=512,
                                     return_tensors="pt")
        next_obs = {
            "input_ids": encoded_obs["input_ids"].squeeze(0).numpy(),
            "attention_mask": encoded_obs["attention_mask"].squeeze(0).numpy(),
            "question_type": self.question_type_map[question_type]
        }

        if done:
            logging.info(f"Episode finished: Prompt {self.current_index}, Best Reward: {self.best_reward_for_current}")
            # Reset will handle moving to next prompt
            next_obs = self.reset()[0]

        return next_obs, reward, done, False, {}


    def load_action_space(self, config_file):
        """Load action space from a config file or dictionary, ensuring integer keys."""
        logging.info("Loading action space configuration")

        if isinstance(config_file, dict):
            logging.info("✅ Action space already loaded as dictionary, using directly.")
            return {int(k): v for k, v in config_file.items()}  # Convert keys to integers

        # ✅ Ensure the file exists before attempting to open it
        if not os.path.exists(config_file):
            logging.error(f"🚨 Action space config file not found: {config_file}")
            raise FileNotFoundError(f"Action space config file '{config_file}' does not exist.")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                action_config = json.load(f)

            # ✅ Convert string keys to integers
            action_config = {int(k): v for k, v in action_config.items()}

            logging.info(f"✅ Action space configuration successfully loaded from '{config_file}'.")
            logging.info(f"📝 Loaded Actions: {json.dumps(action_config, indent=4)}")
            return action_config
        except json.JSONDecodeError:
            logging.error(f"🚨 Failed to parse JSON in '{config_file}'. Check for syntax errors.")
            raise ValueError(f"Invalid JSON format in '{config_file}'.")
        except Exception as e:
            logging.error(f"🚨 Unexpected error loading action space config: {e}")
            raise

    def load_reward_config(self, config_file, remove_bert_score, remove_retrieval):
        """Load reward function configuration and apply ablation settings."""
        logging.info("Loading reward configuration")

        # Load configuration
        if isinstance(config_file, dict):
            logging.info("Reward configuration already loaded as dictionary, using directly.")
            reward_config = config_file  # Use it directly if already loaded
        else:
            try:
                with open(config_file, "r") as f:
                    reward_config = json.load(f)
                logging.info("Reward configuration successfully loaded from file.")
            except Exception as e:
                logging.error(f"Failed to load reward config: {e}")
                raise

        # Log the full reward configuration BEFORE any modification
        logging.info(f"Loaded Reward Configuration: {json.dumps(reward_config, indent=4)}")

        # Apply ablation settings
        if remove_bert_score:
            if "bert_score_weight" in reward_config.get("content_quality", {}):
                del reward_config["content_quality"]["bert_score_weight"]
                logging.warning("BERTScore removed for ablation study")

        if remove_retrieval:
            if "precision_weight" in reward_config.get("accuracy", {}):
                del reward_config["accuracy"]["precision_weight"]
            if "recall_weight" in reward_config.get("accuracy", {}):
                del reward_config["accuracy"]["recall_weight"]
            logging.warning("Retrieval accuracy removed for ablation study")

        # Log the final reward configuration AFTER modifications
        logging.info(f"Final Reward Configuration After Ablation: {json.dumps(reward_config, indent=4)}")

        return reward_config

    def modify_prompt(self, action):
        """Modify tokenized prompt by applying the selected action and return tokenized observations."""
        original_prompt = self.dataset[self.current_index]["question"]

        logging.info(f"🛠 Action Received: {action} | Type: {type(action)}")

        if not isinstance(action, int):
            raise TypeError(f"🚨 Expected `action` to be an integer but got {type(action)}")

        action_instruction = self.action_space_config[action]
        modified_prompt = f"{action_instruction}: {original_prompt}"

        logging.info(f"🔍 Before: {original_prompt}")
        logging.info(f"📝 After: {modified_prompt}")

        # ✅ Tokenize the modified prompt correctly
        encoded_prompt = self.tokenizer(modified_prompt, padding="max_length", truncation=True, max_length=512,
                                        return_tensors="pt")

        # ✅ Return tokenized observation dictionary
        return {
            "input_ids": encoded_prompt["input_ids"].squeeze(0).numpy(),  # Convert to numpy array
            "attention_mask": encoded_prompt["attention_mask"].squeeze(0).numpy()
        }

    def check_hallucination(self, generated_answer, ground_truth):
        """
        Checks if the generated answer contains hallucinated content
        (i.e., information not supported by the ground truth).
        """
        if not isinstance(generated_answer, str) or not isinstance(ground_truth, str):
            logging.error(
                f"❌ Invalid types: generated_answer={type(generated_answer)}, ground_truth={type(ground_truth)}")
            return 1.5  # Assign the maximum hallucination penalty

        generated_tokens = set(generated_answer.lower().split())  # Tokenize generated response
        truth_tokens = set(ground_truth.lower().split())  # Tokenize ground truth

        if not truth_tokens:
            logging.warning("⚠️ Ground truth is empty! Treating as fully hallucinated.")
            return 1.5  # Max penalty since there's no valid ground truth

        # ✅ Identify missing tokens (words in generated answer but not in ground truth)
        missing_tokens = generated_tokens - truth_tokens
        hallucination_ratio = len(missing_tokens) / max(1, len(generated_tokens))  # Prevent div by zero

        # ✅ Apply penalties based on hallucination severity
        if hallucination_ratio >= 0.7:
            penalty = 1.5  # 🚨 Severe hallucination
            logging.warning(f"🚨 Severe Hallucination! Ratio: {hallucination_ratio:.2f} | Penalty: {penalty}")
        elif hallucination_ratio >= 0.5:
            penalty = 1.0  # ⚠️ Moderate hallucination
            logging.warning(f"⚠️ Moderate Hallucination. Ratio: {hallucination_ratio:.2f} | Penalty: {penalty}")
        elif hallucination_ratio >= 0.3:
            penalty = 0.5  # 🔍 Light hallucination
            logging.info(f"🔍 Slight Hallucination. Ratio: {hallucination_ratio:.2f} | Penalty: {penalty}")
        else:
            penalty = 0.0  # ✅ No significant hallucination
            logging.info(f"✅ Minimal Hallucination. Ratio: {hallucination_ratio:.2f} | No penalty applied.")

        return penalty

    def calculate_token_efficiency(self, response):
        """
        Calculates token efficiency by determining how much of the response consists of essential content.

        """
        if not isinstance(response, str):
            logging.error(f"Expected a string for response, but got {type(response)}")
            return 0.0  # Return default value

        total_tokens = len(response.split())

        # Define a basic rule: efficient responses have < 30 words, excessive > 100 words
        if total_tokens <= 30:
            return 1.0  # Fully efficient
        elif total_tokens > 100:
            return 0.2  # Strong penalty for verbosity
        else:
            return 1.0 - ((total_tokens - 30) / 70)  # Linearly decrease efficiency

    def calculate_reward(self, generated_answer, ground_truth):
        """Calculate reward with question-type specific weighting while maintaining existing penalties."""
        logging.info(f"🔍 Evaluating Generated Answer vs Ground Truth...")

        # Get question type for current sample
        question_type = self.dataset[self.current_index]["question_type"]

        # Get base metrics
        reward_dict = evaluate_metrics(ground_truth, generated_answer, self.bart_scorer, self.tokenizer)
        logging.info(f" Raw Reward Scores: {reward_dict}")

        # Question type-specific weighting
        if question_type == "what":
            # Factual accuracy is more important for "what" questions
            content_weights = {
                "bert_score_weight": self.reward_config["content_quality"]["bert_score_weight"] * 0.7,
                "rouge_l_weight": self.reward_config["content_quality"]["rouge_l_weight"] * 0.7,
                "meteor_weight": self.reward_config["content_quality"]["meteor_weight"] * 0.7
            }
            accuracy_weights = {
                "precision_weight": self.reward_config["accuracy"]["precision_weight"] * 1.2,  # Boost precision
                "recall_weight": self.reward_config["accuracy"]["recall_weight"] * 1.2
            }
        elif question_type == "how":
            # Step-by-step clarity important for "how" questions
            content_weights = {
                "bert_score_weight": self.reward_config["content_quality"]["bert_score_weight"] * 1.0,
                "rouge_l_weight": self.reward_config["content_quality"]["rouge_l_weight"] * 1.2,
                # Boost ROUGE for sequence
                "meteor_weight": self.reward_config["content_quality"]["meteor_weight"] * 1.0
            }
            accuracy_weights = {
                "precision_weight": self.reward_config["accuracy"]["precision_weight"] * 1.0,
                "recall_weight": self.reward_config["accuracy"]["recall_weight"] * 1.0
            }
        else:  # if/can
            # Logical consistency key for yes/no questions
            content_weights = {
                "bert_score_weight": self.reward_config["content_quality"]["bert_score_weight"] * 0.8,
                "rouge_l_weight": self.reward_config["content_quality"]["rouge_l_weight"] * 0.8,
                "meteor_weight": self.reward_config["content_quality"]["meteor_weight"] * 0.8
            }
            accuracy_weights = {
                "precision_weight": self.reward_config["accuracy"]["precision_weight"] * 1.3,
                "recall_weight": self.reward_config["accuracy"]["recall_weight"] * 0.9
            }

        # Normalize weights
        total_content_weight = sum(content_weights.values())
        total_accuracy_weight = sum(accuracy_weights.values())

        # Keep your existing normalization checks
        if total_content_weight == 0:
            logging.warning("⚠️ Content quality weights sum to zero! Adjusting to default equal distribution.")
            total_content_weight = 1.0

        if total_accuracy_weight == 0:
            logging.warning("⚠️ Accuracy weights sum to zero! Adjusting to default equal distribution.")
            total_accuracy_weight = 1.0

        # Calculate weighted scores using question-type specific weights
        content_quality_score = ((reward_dict["BERTScore"]["F1"] * content_weights["bert_score_weight"]) +
                                 (reward_dict["ROUGE"]["rougeL"].fmeasure * content_weights["rouge_l_weight"]) +
                                 (reward_dict["METEOR"] * content_weights["meteor_weight"])
                                 ) / total_content_weight

        accuracy_score = ((reward_dict["BERTScore"]["Precision"] * accuracy_weights["precision_weight"]) +
                          (reward_dict["BERTScore"]["Recall"] * accuracy_weights["recall_weight"])
                          ) / total_accuracy_weight

        # Keep your existing penalty calculations
        hallucination_penalty = self.reward_config["penalties"]["hallucination_penalty"] * max(
            0.0, float(self.check_hallucination(generated_answer, ground_truth)) - 0.1
        )

        response_length = len(generated_answer.split())
        max_length = 100
        length_penalty = self.reward_config["penalties"]["length_penalty"] * min(
            1.0, response_length / max_length
        )

        token_efficiency_penalty = self.reward_config["penalties"]["token_efficiency_penalty"] * max(
            0.0, float(self.calculate_token_efficiency(generated_answer)) - 0.5
        )

        # Handle INVALID responses
        if generated_answer.lower() in ["invalid input", "i need more information", "can you clarify?"]:
            final_reward = -2.0
            logging.warning(f"⚠️ INVALID Response Detected. Applying Max Penalty: {final_reward}")
            return final_reward

        # Final reward calculation with all components
        final_reward = (
                content_quality_score + accuracy_score -
                hallucination_penalty - length_penalty - token_efficiency_penalty
        )

        final_reward = float(np.clip(final_reward, -2, 2))
        logging.info(f"🏆 Final Reward Calculated for {question_type} question: {final_reward}")

        return final_reward


def train_ppo(env):
    """Train PPO agent with proper episodic learning."""
    logging.info("Starting PPO training")

    # Detect and log if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Store original env before wrapping
    base_env = env

    # Proper env wrapping
    from gymnasium.wrappers import FilterObservation
    env = FilterObservation(env, ['input_ids', 'attention_mask', 'question_type'])
    env = FlattenObservation(env)

    # Initialize PPO with proper hyperparameters
    agent = PPO(
        "MlpPolicy",
        env,
        n_steps=64,  # Collect 64 steps before updating
        batch_size=32,  # Process 32 samples per mini-batch
        n_epochs=10,  # Number of passes over the data
        learning_rate=3e-4,  # Standard PPO learning rate
        ent_coef=0.01,  # Encourage exploration
        verbose=1,
        tensorboard_log="./ppo_logs/"
    )

    # Training parameters
    total_episodes = 10  # Total number of episodes to train
    evaluation_interval = 5  # Evaluate every 5 episodes
    episodes_completed = 0

    # Store metrics for each question type
    metrics_by_type = {
        "what": {"rewards": [], "best_rewards": []},
        "how": {"rewards": [], "best_rewards": []},
        "if_can": {"rewards": [], "best_rewards": []}  # Changed from "if/can" to "if_can"
    }

    while episodes_completed < total_episodes:
        # Reset environment and get initial observation
        obs = env.reset()[0]
        episode_reward = 0
        done = False

        # Access question type from base env
        current_question_type = base_env.dataset[base_env.current_index]["question_type"]

        # Episode loop
        while not done:
            # Use stochastic actions during training
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        # Track metrics by question type
        metrics_by_type[current_question_type]["rewards"].append(episode_reward)
        metrics_by_type[current_question_type]["best_rewards"].append(base_env.best_reward_for_current)

        episodes_completed += 1
        logging.info(f"Completed episode {episodes_completed}/{total_episodes}")
        logging.info(f"Episode Reward: {episode_reward}, Question Type: {current_question_type}")

        # Train the agent
        if episodes_completed % 5 == 0:  # Update every 5 episodes
            agent.learn(
                total_timesteps=base_env.max_attempts_per_prompt * 5,
                reset_num_timesteps=False,
                progress_bar=True
            )

        # Periodic evaluation and checkpointing
        if episodes_completed % evaluation_interval == 0:
            # Log metrics for each question type
            for q_type in metrics_by_type:
                avg_reward = np.mean(metrics_by_type[q_type]["rewards"][-evaluation_interval:])
                avg_best = np.mean(metrics_by_type[q_type]["best_rewards"][-evaluation_interval:])
                logging.info(f"Question Type {q_type}:")
                logging.info(f"Average Reward: {avg_reward:.3f}")
                logging.info(f"Average Best Reward: {avg_best:.3f}")

            # Save checkpoint
            checkpoint_path = f"ppo_results/results/ppo_model_checkpoint_{episodes_completed}"
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint at episode {episodes_completed}")

    # Save final model and metrics
    agent.save("results/ppo_final_model")

    # Save final metrics by question type
    metrics_path = "ppo_results/results/ppo_metrics_by_type.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_by_type, f, indent=4)

    logging.info("Training complete!")
    return agent, metrics_by_type


def train_dpo(env):
    """Train DPO agent."""
    logging.info("Starting DPO training")

    if not DPO_AVAILABLE:
        logging.error("DPOTrainer is not installed. Exiting.")
        raise ImportError("DPOTrainer is not installed. Install `trl` for DPO support.")

    trainer = DPOTrainer(model="facebook/opt-1.3b", reward_model=None, dataset=env.dataset)
    trainer.train()
    trainer.save("C:/Users/USER/PycharmProjects/fyp-rnd/RL_Agent/results/dpo_model")

    logging.info("DPO Training Complete")
    print("DPO Training Complete")


if __name__ == "__main__":
    logging.info("Initializing RL Training Script")

    # CHOOSE MODEL TYPE (PPO or DPO)
    MODEL_TYPE = "PPO"  # Change to "DPO" if needed
    logging.info(f"Selected Model Type: {MODEL_TYPE}")

    # CHOOSE LLM MODEL
    LLM_MODEL = "gpt-4"  # Change to "gpt-3.5-turbo", "llama-2", or "mistral"
    logging.info(f"Selected LLM Model: {LLM_MODEL}")

    # ENABLE ABLATION STUDIES (Set to True if you want to remove a feature)
    REMOVE_BERT_SCORE = False  # Set True to remove BERTScore
    REMOVE_RETRIEVAL = False  # Set True to remove retrieval-based accuracy

    # Load configurations
    with open("config/reward_config.json") as f:
        reward_config = json.load(f)
    with open("config/action_space_config.json") as f:
        action_config = json.load(f)

    if os.path.exists(TOKENIZED_DATA_PATH):
        with open(TOKENIZED_DATA_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded tokenized dataset from {TOKENIZED_DATA_PATH}")
    else:
        raise FileNotFoundError(
            f"Tokenized dataset not found at {TOKENIZED_DATA_PATH}. Run tokenize_dataset.py first.")

    # Ensure dataset integrity
    required_keys = ["input_ids", "attention_mask", "question_type", "gt_input_ids", "gt_attention_mask"]
    if not all(all(key in entry for key in required_keys) for entry in dataset):
        logging.error("Dataset is missing required tokenized fields.")
        raise ValueError(f"Dataset must include: {required_keys}")

    env = CustomPromptEnv(
        action_config="config/action_space_config.json",
        reward_config="config/reward_config.json",
        dataset=dataset,
        model_type=LLM_MODEL,
        remove_bert_score=REMOVE_BERT_SCORE,
        remove_retrieval=REMOVE_RETRIEVAL
    )

    # Train selected model
    if MODEL_TYPE == "PPO":
        train_ppo(env)
    elif MODEL_TYPE == "DPO":
        train_dpo(env)
    else:
        logging.error("Invalid MODEL_TYPE selected.")
        raise ValueError("Invalid MODEL_TYPE. Choose 'PPO' or 'DPO'.")
