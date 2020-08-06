import gym
from agent import DDPG

epoch = 500
env = gym.make('Pendulum-v0')

agent = DDPG(env)
agent.load_model()

state = env.reset()

cumulative_reward = 0
for i in range(200):
    action = agent.get_action(state)
    env.render()
    state, reward, _, _ = env.step(action * 2)
    cumulative_reward += reward
print('Cumulative Reward: {}'.format(cumulative_reward))
