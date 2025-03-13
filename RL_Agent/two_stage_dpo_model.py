import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel


class BinaryDecisionNetwork(nn.Module):
    """Network to decide whether to modify the prompt or keep it unchanged"""

    def __init__(self, input_dim):
        super(BinaryDecisionNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 2)  # 0 = keep unchanged, 1 = modify

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ActionSelectorNetwork(nn.Module):
    """Network to select which specific action to apply (if modifying)"""

    def __init__(self, input_dim, action_counts):
        super(ActionSelectorNetwork, self).__init__()
        self.shared_layer = nn.Linear(input_dim, 256)
        self.shared_layer2 = nn.Linear(256, 128)

        # Create separate output layers for each question type
        self.what_head = nn.Linear(128, action_counts["what"] - 1)  # -1 for excluding action 0
        self.how_head = nn.Linear(128, action_counts["how"] - 1)
        self.if_can_head = nn.Linear(128, action_counts["if_can"] - 1)

    def forward(self, x, question_type):
        x = F.relu(self.shared_layer(x))
        x = F.relu(self.shared_layer2(x))

        # Select the appropriate head based on question type
        if question_type == "what":
            return self.what_head(x)
        elif question_type == "how":
            return self.how_head(x)
        else:  # if_can
            return self.if_can_head(x)


class TwoStageDPOTrainer:
    def __init__(self, input_dim, action_space, learning_rate=1e-4,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Store action space
        self.action_space = action_space
        self.device = device

        # Count actions per question type
        self.action_counts = {}
        for qt in ["what", "how", "if_can"]:
            specific_key = f"{qt}_question_actions"
            if specific_key in action_space:
                # Add 1 for the "keep unchanged" action
                self.action_counts[qt] = 1 + len(action_space[specific_key])
            else:
                self.action_counts[qt] = 1

        # Initialize the two networks
        self.decision_network = BinaryDecisionNetwork(input_dim).to(device)
        self.action_network = ActionSelectorNetwork(input_dim, self.action_counts).to(device)

        # Initialize optimizers
        self.decision_optimizer = optim.Adam(self.decision_network.parameters(), lr=learning_rate)
        self.action_optimizer = optim.Adam(self.action_network.parameters(), lr=learning_rate)

        # Metrics tracking
        self.decision_losses = []
        self.action_losses = []
        self.validation_accuracies = []

    def get_action_description(self, action_id, question_type):
        """Get description for an action ID"""
        if action_id == 0:
            return "Keep the following prompt unchanged."

        # Adjust index for specific actions
        specific_id = action_id - 1 + 1  # -1 to convert to 0-index, +1 because specific actions start at 1
        specific_key = f"{question_type}_question_actions"
        return self.action_space[specific_key].get(str(specific_id), "Unknown action")

    def get_best_action(self, question, question_type, tokenizer):
        """Decide whether to modify and which action to use"""
        # Get embeddings
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        bert_model.eval()

        encoded = tokenizer(question, truncation=True, padding="max_length",
                            max_length=128, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Generate embeddings
            outputs = bert_model(**encoded)
            state = outputs.last_hidden_state.mean(dim=1)

            # First decision: modify or not?
            decision_logits = self.decision_network(state)
            decision_probs = F.softmax(decision_logits, dim=1)
            should_modify = decision_probs[0, 1] > 0.5  # If prob of modifying > 0.5

            if not should_modify:
                # Keep unchanged
                return 0, "Keep the following prompt unchanged.", decision_probs.cpu().numpy()[0]

            # Second decision: which specific action?
            action_logits = self.action_network(state, question_type)
            action_probs = F.softmax(action_logits, dim=1)

            # Get best specific action (add 1 to convert to actual action ID)
            best_specific_id = torch.argmax(action_probs, dim=1).item() + 1

            best_action_desc = self.get_action_description(best_specific_id, question_type)

            # Combine probabilities for full action space
            full_probs = torch.zeros(self.action_counts[question_type])
            full_probs[0] = decision_probs[0, 0]  # Prob of keeping unchanged
            full_probs[1:] = decision_probs[0, 1] * action_probs[0]  # Prob of modifying * specific action probs

            return best_specific_id, best_action_desc, full_probs.cpu().numpy()

    # Add methods for training, evaluation, etc. (similar to DPOTrainer)
    # ...