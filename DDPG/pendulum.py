import gym
from DDPGAgent import DDPGAgent

EPISODES = 1000
EPISODES_TEST = 10
STEPS = 200

def train(path):
  env = gym.make('Pendulum-v0')
  env = env.unwrapped
  env.seed(1)

  state_dims = env.observation_space.shape[0]
  action_dims = env.action_space.shape[0]
  action_bound = env.action_space.high

  agent = DDPGAgent(state_dims, action_dims, action_bound)

  for episode in range(EPISODES):
    state = env.reset()
    score = 0

    for _ in range(STEPS):
      env.render()

      action = agent.act(state)
      next_state, reward, _, _ = env.step(action)

      agent.remember(state, action, reward, next_state)
      agent.replay()

      state = next_state
      score += reward
    
    print('Episode %d/%d, score %d' % (episode + 1, EPISODES, score))

    if episode % 50 == 0:
      agent.save(path)

def play(path):
  env = gym.make('Pendulum-v0')
  env = env.unwrapped
  env.seed(1)

  state_dims = env.observation_space.shape[0]
  action_dims = env.action_space.shape[0]
  action_bound = env.action_space.high

  agent = DDPGAgent(state_dims, action_dims, action_bound)
  agent.load(path)

  for episode in range(EPISODES_TEST):
    state = env.reset()
    score = 0

    for _ in range(STEPS):
      env.render()

      action = agent.act(state, learning=False)
      state, reward, _, _ = env.step(action)
      score += reward
    
    print('Episode %d/%d, score %d' % (episode + 1, EPISODES_TEST, score))

if __name__ == "__main__":
  # train('./saves/pendulum.ckpt')
  play('./saves/pendulum.ckpt')

