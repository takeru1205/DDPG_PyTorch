# Actor and Critic models
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_init, actor_last_layer_init, critic_last_layer_init


class Actor(nn.Module):
    """
    Actor Network
    """

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        # Initialize actor weights
        layer_init(self.fc1)
        layer_init(self.fc2)
        actor_last_layer_init(self.fc3)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Initialize critic weights
        layer_init(self.fc1)
        layer_init(self.fc2)
        critic_last_layer_init(self.fc3)

    def forward(self, x, action):
        x = F.softplus(self.fc1(torch.cat([x, action], dim=1)))
        x = F.softplus(self.fc2(x))
        x = self.fc3(x)
        return x
