# Ornstein-Uhlenbeck process
import numpy as np


class OUActionNoise(object):
    """
    Orstein-Uhleneck Process.

    This stochastic process is exploration for agent.

    """

    def __init__(self, action_dim, mu=0.0, sigma=0.2,
                 max_sigma=0.3, min_sigma=-0.3, theta=0.15, decay_period=100000, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.theta = theta
        self.x0 = x0
        self.action_dim = action_dim
        self.decay_period = decay_period
        self.reset()

    def evolve_state(self):
        x = self.x0
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.x0 = x + dx
        return self.x0

    def __call__(self, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return ou_state

    def reset(self):
        self.x0 = np.ones(self.action_dim) * self.mu
