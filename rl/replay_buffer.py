import random
import torch


class ReplayBuffer:
    """
    Stores self-play data:
    (state, policy_target, outcome)

    state  : board tensor
    policy : MCTS visit distribution
    value  : final game result (z)
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, policy, value):
        """
        Add one training sample
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        """
        Sample random mini-batch
        """
        batch = random.sample(self.buffer, batch_size)

        states, policies, values = zip(*batch)

        states = torch.tensor(states).float()
        policies = torch.tensor(policies).float()
        values = torch.tensor(values).float()

        return states, policies, values

    def __len__(self):
        return len(self.buffer)
