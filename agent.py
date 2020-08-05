import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Actor, Critic
from memory import ReplayMemory

from const import *


class DDPG(object):
    """
    Deep Deterministic Policy Gradient Algorithm
    """

    def __init__(self, env, writer=None):
        self.env = env
        self.writer = writer

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # Randomly initialize network parameter
        self.actor = Actor(state_dim, action_dim).to('cuda')
        self.critic = Critic(state_dim, action_dim).to('cuda')

        # Initialize target network parameter
        self.target_actor = Actor(state_dim, action_dim).to('cuda')
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(state_dim, action_dim).to('cuda')
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Replay memory
        self.memory = ReplayMemory(state_dim, action_dim)

        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.tau = tau

        # network parameter optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)

    def get_action(self, state, ou_noise):
        action = self.actor(torch.from_numpy(state).to('cuda', torch.float))
        noise = ou_noise()
        return np.clip(action.to('cpu').detach().numpy().copy() + noise, -1, 1)

    def store_transition(self, state, action, state_, reward, done):
        self.memory.store_transition(state, action, state_, reward, done)

    def soft_update(self, target_net, net):
        """Target parameters soft update"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def update(self, time_step, batch_size=64):
        """Network parameter update"""
        if len(self.memory) < batch_size:
            return

        states, actions, states_, rewards, terminals = self.memory.sample(batch_size)
        with torch.no_grad():
            y = rewards.unsqueeze(1) + self.gamma * \
                self.target_critic(states_, self.target_actor(states_).clamp(-1, 1))

        # Update Critic
        q = self.critic(states, actions)
        critic_loss = self.criterion(y, q)
        if self.writer:
            self.writer.add_scalar("loss/critic", critic_loss.item(), time_step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor (Policy Gradient)
        j = torch.mean(self.critic(states, actions) * self.actor(states))
        if self.writer:
            self.writer.add_scalar("loss/actor", j, time_step)
        self.actor_optimizer.zero_grad()
        j.backward()
        self.actor_optimizer.step()

        # target parameter soft update
        self.soft_update(self.target_actor, self.actor)  # update target actor network
        self.soft_update(self.target_critic, self.critic)  # update target critic network

    def save_model(self, path='models/'):
        torch.save(self.actor.state_dict(), path + 'actor')
        torch.save(self.critic.state_dict(), path + 'critic')
        torch.save(self.target_actor.state_dict(), path + 'target_actor')
        torch.save(self.target_critic.state_dict(), path + 'target_critic')

    def load_model(self, path='models/'):
        self.actor.load_state_dict(torch.load(path + 'actor'))
        self.critic.load_state_dict(torch.load(path + 'critic'))
        self.target_actor.load_state_dict(torch.load(path + 'target_actor'))
        self.target_critic.load_state_dict(torch.load(path + 'target_critic'))
