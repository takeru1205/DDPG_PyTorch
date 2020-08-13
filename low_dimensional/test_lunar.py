import gym
from agent import DDPG

env = gym.make('LunarLanderContinuous-v2')

env.seed(36)

agent = DDPG(env)
agent.load_model()

state = env.reset()

cumulative_reward = 0
for i in range(env.spec.max_episode_steps):
    action = agent.get_action(state)
    env.render()
    state, reward, done, _ = env.step(action)
    cumulative_reward += reward
    if done:
        break
print('Cumulative Reward: {}'.format(cumulative_reward))

env.close()
