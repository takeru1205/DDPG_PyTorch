from collections import deque
import gym
import numpy as np
from agent import DDPG
from utils import get_screen

env = gym.make('Pendulum-v0')

agent = DDPG(env, memory=False)
agent.load_model()

env.reset()
pixel = env.render(mode='rgb_array')
state = deque([get_screen(pixel) for _ in range(3)], maxlen=3)
cumulative_reward = 0
for timestep in range(200):
    action = agent.get_action(np.array(state)[np.newaxis])
    _, reward, _, _ = env.step(action * 2)
    pixel = env.render(mode='rgb_array')
    state_ = state.copy()
    state_.append(get_screen(pixel))
    state = state_
    cumulative_reward += reward
print('Cumulative Reward: {}'.format(cumulative_reward))
