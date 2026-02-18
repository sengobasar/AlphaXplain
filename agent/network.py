import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    """
    (p, v) = f_theta(s)

    p = policy (move probabilities)
    v = value  (position evaluation)
    """

    def __init__(self, board_size=8, channels=12, action_dim=20480):
        super().__init__()

        self.input_dim = board_size * board_size * channels

        # -------- Shared trunk --------
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # -------- Policy head --------
        self.policy_head = nn.Linear(256, action_dim)

        # -------- Value head --------
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        """
        Expected input shape:
        [batch, 8, 8, 12]  OR  [batch, 768]
        """

        # Flatten safely
        x = x.view(x.size(0), -1)

        # Shared trunk
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Policy
        p = self.policy_head(x)
        p = F.softmax(p, dim=1)

        # Value
        v = self.value_head(x)
        v = torch.tanh(v)

        return p, v
