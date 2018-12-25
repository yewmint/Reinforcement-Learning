import numpy as np
import gym
from ActorCriticAgent import ActorCriticAgent

UPDATE_FREQUENCY = 5
EPISODES = 10000
EPISODES_TEST = 10
STEPS = 200

def train(path):
  env = gym.make('CartPole-v0')
  env = env.unwrapped

  # to reproduce result, adjust seeds accroding to environment
  env.seed(1)
  np.random.seed(1)

  action_size = env.action_space.n

  agent = ActorCriticAgent((4,), action_size)
  scores = []

  for episode in range(EPISODES):
    state = env.reset()
    score = 0

    for _ in range(STEPS):
      # env.render()

      action_id = agent.act(state)
      action = np.argmax(action_id)
      next_state, reward, done, _ = env.step(action)
      score += reward
      action = np.identity(action_size)[action]

      if not done:
        x, _, theta, _ = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
      else:
        reward = -10

      agent.learn(state, action_id, reward, next_state)
      state = next_state

      if done:
        break

    scores.append(score)
    print('Episodes %d/%d, score %d' % (episode + 1, EPISODES, score))

    if episode % 50 == 0 and episode != 0:
      from_ = episode - 50
      to_ = episode
      print(
        'Episodes %d - %d, mean %d' % (from_, to_, np.mean(scores[from_:to_]))
      )
      agent.save(path)


if __name__ == "__main__":
  train('./saves/cart.ckpt')
