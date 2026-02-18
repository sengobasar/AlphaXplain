import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    """
    f_theta(s) -> (policy, value)
    """

    def __init__(self, board_size=8, channels=12, action_dim=4672):
        super().__init__()

        # ---------- CNN Backbone ----------

        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        # ---------- Policy Head ----------

        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * board_size * board_size,
                                   action_dim)

        # ---------- Value Head ----------

        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # x: [B,12,8,8]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # ---------- Policy ----------

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)

        logits = self.policy_fc(p)

        policy = F.softmax(logits, dim=1)

        # ---------- Value ----------

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)

        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value
