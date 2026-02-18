import torch
import torch.nn.functional as F
import torch.optim as optim


class Trainer:
    """
    Actor-Critic Trainer

    Policy: π(a|s;θ)
    Value:  V(s;ϕ)
    """

    def __init__(self, policy, value, lr=1e-3, gamma=0.99):

        self.policy = policy
        self.value = value

        self.gamma = gamma

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value.parameters(), lr=lr)

    def train(self, buffer, batch_size=32):

        # Wait until enough data
        if len(buffer) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states = buffer.sample(batch_size)

        # Convert to tensors
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()

        # -----------------------
        # 1️⃣ VALUE UPDATE
        # -----------------------

        v = self.value(states).squeeze()
        v_next = self.value(next_states).squeeze().detach()

        target = rewards + self.gamma * v_next

        value_loss = F.mse_loss(v, target)

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        # -----------------------
        # 2️⃣ POLICY UPDATE
        # -----------------------

        probs = self.policy(states)

        # Numerical safety
        probs = torch.clamp(probs, 1e-8, 1.0)

        log_probs = torch.log(probs)

        selected_log_probs = log_probs.gather(
            1, actions.unsqueeze(1)
        ).squeeze()

        advantage = (target - v).detach()

        policy_loss = -(selected_log_probs * advantage).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
