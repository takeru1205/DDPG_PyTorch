import torch.nn as nn
import torch.optim as optim
from model import Actor, Critic
from memory import ReplayMemory

from const import *


class DDPG(object):
    def __init__(self, env):
        self.env = env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Randomly initialize network parameter
        self.actor = Actor(state_dim, action_dim, env.action_space.high[0]).to('cuda')
        self.critic = Critic(state_dim, action_dim).to('cuda')

        # Initialize target network parameter
        self.target_actor = Actor(state_dim, action_dim, env.action_space.high[0]).to('cuda')
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Replay memory
        self.memory = ReplayMemory(state_dim, action_dim)

        self.gamma = gamma
        self.criterion = nn.MSELoss()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
