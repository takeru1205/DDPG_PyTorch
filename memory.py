# Replay Memory
import torch
import numpy as np


class ReplayMemory(object):
    """
    Replay Memory for Experience Replay.

    Data sampling randomly from buffer.

    """

    def __init__(self, state_dim, act_dim, max_size=int(10e+6)):
        self.max_size = max_size
        self.state_memory = torch.zeros((self.max_size, state_dim), dtype=torch.float)
        self.new_state_memory = torch.zeros((self.max_size, state_dim), dtype=torch.float)
        self.action_memory = torch.zeros((self.max_size, act_dim), dtype=torch.float)
        self.reward_memory = torch.zeros(self.max_size, dtype=torch.float)
        self.terminal_memory = torch.zeros(self.max_size, dtype=torch.uint8)
        self.mem_ctrl = 0

    def store_transition(self, state, action, state_, reward, done):
        index = self.mem_ctrl % self.max_size
        self.state_memory[index] = torch.from_numpy(state)
        self.new_state_memory[index] = torch.from_numpy(state_)
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.from_numpy(np.array([reward]).astype(np.float))
        self.terminal_memory[index] = torch.from_numpy(np.array([1 - done]).astype(np.uint8))
        self.mem_ctrl += 1

    def sample(self, batch_size=64):
        mem_size = min(self.mem_ctrl, self.max_size)
        batch = np.random.choice(mem_size, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states.to('cuda'), actions.to('cuda'), states_.to('cuda'), rewards.to('cuda'), terminal

    def __len__(self):
        return self.mem_ctrl
