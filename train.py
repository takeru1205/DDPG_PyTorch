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

agent = DDPG(env)

writer = SummaryWriter(log_dir='logs/')
all_timesteps = 0

for e in range(epoch):
    noise = OUActionNoise(env.action_space.shape[0])
    state = env.reset()
    cumulative_reward = 0
    for timestep in range(200):
        action = agent.get_action(state, noise)
        state_, reward, done, _ = env.step(action)
        agent.store_transition(state, action, state_, reward, done)

        cumulative_reward += reward

        critic_loss, actor_loss = agent.update()
        writer.add_scalar("loss/critic", critic_loss, all_timesteps)
        writer.add_scalar("loss/actor", actor_loss, all_timesteps)
        all_timesteps += 1
    print('Epoch : {} / {}, Cumulative Reward : {}'.format(e, epoch, cumulative_reward))
    writer.add_scalar("reward", cumulative_reward, epoch)


