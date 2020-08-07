import random
from collections import deque
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import DDPG
from exploration import OUActionNoise
from utils import get_screen

epoch = 5000
env = gym.make('Pendulum-v0')

# seed
np.random.seed(42)
env.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

writer = SummaryWriter(log_dir='logs/')
agent = DDPG(env, writer)

all_timesteps = 0

for e in range(epoch):
    noise = OUActionNoise(env.action_space.shape[0])
    env.reset()
    pixel = env.render(mode='rgb_array')
    state = deque([get_screen(pixel) for _ in range(3)], maxlen=3)
    cumulative_reward = 0
    for timestep in range(200):
        action = agent.get_action(np.array(state)[np.newaxis], noise, timestep)
        _, reward, done, _ = env.step(action * env.action_space.high[0])
        pixel = env.render(mode='rgb_array')
        state_ = state.copy()
        state_.append(get_screen(pixel))
        agent.store_transition(np.array(state), action, np.array(state_), reward, done)

        state = state_
        cumulative_reward += reward

        agent.update(all_timesteps, batch_size=16)
        all_timesteps += 1
    print('Epoch : {} / {}, Cumulative Reward : {}'.format(e, epoch, cumulative_reward))
    writer.add_scalar("reward", cumulative_reward, e)
    if e % 500 == 0:
        agent.save_model('models/' + str(e) + '_')
agent.save_model()
