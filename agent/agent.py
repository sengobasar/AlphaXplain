import torch
import numpy as np


class Agent:

    def __init__(self, policy, value):

        self.policy = policy
        self.value = value

    def act(self, state):

        s = torch.tensor(state).float().unsqueeze(0)

        probs = self.policy(s).detach().numpy()[0]

        action = np.random.choice(len(probs), p=probs)

        return action
