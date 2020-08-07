# Actor and Critic models
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_init, actor_last_layer_init, critic_last_layer_init


class Actor(nn.Module):
    """
    Actor Network
    """

    def __init__(self, action_dim, width=40, height=40):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_width * conv_height * 32

        self.fc1 = nn.Linear(linear_input_size, 200)
        self.fc2 = nn.Linear(200, action_dim)

        # Initialize actor weights
        layer_init(self.fc1)
        actor_last_layer_init(self.fc2)

    def forward(self, x):
        x = F.softplus(self.conv1(x))
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.softplus(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class Critic(nn.Module):
    """
    Critic Network
    """

    def __init__(self, action_dim, width=40, height=40):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = conv_width * conv_height * 32
        self.fc1 = nn.Linear(linear_input_size + action_dim, 200)
        self.fc2 = nn.Linear(200, 1)

        # Initialize critic weights
        layer_init(self.fc1)
        critic_last_layer_init(self.fc2)

    def forward(self, x, action):
        x = F.softplus(self.conv1(x))
        x = F.softplus(self.conv2(x))
        x = F.softplus(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.softplus(self.fc1(torch.cat([x, action], dim=1)))
        x = self.fc2(x)
        return x
