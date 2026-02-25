import torch.nn as nn

class RewardModel(nn.Module):
    """Module [11]: Alignment & Reward Model for preference learning."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reward_head = nn.Linear(config.d_model, 1)

    def forward(self, hidden_states):
        return self.reward_head(hidden_states)
