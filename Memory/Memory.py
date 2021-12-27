import random
from collections import deque

class Memory(object):

    def __init__(self, capacity=8000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n):
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)
        return sample_batch