# Ornstein-Uhlenbeck process
import numpy as np


class OUActionNoise(object):
    def __init__(self, action_dim, mu=0.0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.x_prev = self.reset()
        self.action_dim = action_dim

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x

    def reset(self):
        return self.x0 if self.x0 is not None else np.ones(self.action_dim) * self.mu
