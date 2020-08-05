import random
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import DDPG
from exploration import OUActionNoise


epoch = 500
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
    state = env.reset()
    cumulative_reward = 0
    for timestep in range(200):
        action = agent.get_action(state, noise, timestep)
        state_, reward, done, _ = env.step(action * env.action_space.high[0])
        # env.render()
        agent.store_transition(state, action, state_, reward, done)

        cumulative_reward += reward

        agent.update(all_timesteps)
        all_timesteps += 1
    print('Epoch : {} / {}, Cumulative Reward : {}'.format(e, epoch, cumulative_reward))
    writer.add_scalar("reward", cumulative_reward, e)

agent.save_model()


