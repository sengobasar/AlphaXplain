import random
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_next):
        self.buffer.append((s, a, r, s_next))

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states = zip(*batch)

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)
