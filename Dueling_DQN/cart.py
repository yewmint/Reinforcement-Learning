import numpy as np
import pandas as pd
import gym
from DuelingAgent import DuelingAgent

UPDATE_FREQUENCY = 5
EPISODES = 10000
EPISODES_TEST = 10
STEPS = 200

def train(path):
  env = gym.make('CartPole-v0')

  state_dims = 4
  action_size = 2

  agent = DuelingAgent(state_dims, action_size)

  scores = []

  for episode in range(EPISODES):
    state = env.reset()
    score = 0

    for _ in range(STEPS):
      # env.render()

      action = agent.act(state)
      next_state, reward, done, _ = env.step(action)

      score += reward
      reward = -10 if done else reward

      agent.remember(state, action, reward, next_state)
      loss = agent.replay()
      state = next_state

      if done:
        break

    print('Episodes %d/%d, score %d' % (episode + 1, EPISODES, score))
    scores.append(score)

    if episode % 50 == 0:
      agent.save(path)
      pd.DataFrame(scores).to_csv('scores.csv')

def play(path):
  env = gym.make('CartPole-v0')

  state_dims = 4
  action_size = 2

  agent = DuelingAgent(state_dims, action_size)
  agent.load(path)

  for episode in range(EPISODES_TEST):
    state = env.reset()
    score = 0

    for _ in range(STEPS):
      env.render()

      action = agent.act(state, learning=False)
      state, reward, done, _ = env.step(action)
      score += reward

      if done:
        break

    print('Episodes %d/%d, score %d' % (episode + 1, EPISODES_TEST, score))

if __name__ == "__main__":
  train('saves/cart.ckpt')
  # play('saves/cart.ckpt')