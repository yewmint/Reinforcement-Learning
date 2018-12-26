import tensorflow as tf
import numpy as np
from collections import deque

# hypers for 50 steps to stand upright
MEMORY_LEN = 5000
BATCH_SIZE = 256
HIDDEN_UNITS = 30
GAMMA = 0.9
LR_ACTOR = 0.01
LR_CRITIC = 0.02
VARIETY_DECAY = 0.9995
VARIETY_MIN = 0.5
SYNC_FREQUENCY = 20

class DDPGAgent:
  def __init__(self, state_dims, action_dims, action_bound):
    self.state_dims = state_dims
    self.action_dims = action_dims
    self.action_bound = action_bound

    self.variety = action_bound * 2
    self.replay_num = 0

    self.states = deque(maxlen=MEMORY_LEN)
    self.actions = deque(maxlen=MEMORY_LEN)
    self.rewards = deque(maxlen=MEMORY_LEN)
    self.next_actions = deque(maxlen=MEMORY_LEN)

    self._build_training()
    self.saver = tf.train.Saver()

  def _build_actor(self, states, name, trainable):
    with tf.variable_scope(name):
      hidden = tf.layers.dense(
        states, 
        HIDDEN_UNITS, 
        activation=tf.nn.relu,
        name=name + '_hidden',
        trainable=trainable
      )
      output = tf.layers.dense(
        hidden, 
        self.action_dims, 
        activation=tf.nn.tanh,
        name=name + '_output',
        trainable=trainable
      )
      action = tf.multiply(
        output, 
        self.action_bound, 
        name=name + '_action_out'
      )
      return action

  def _build_critic(self, states, actions, name, trainable):
    with tf.variable_scope(name):
      w_states = tf.get_variable(
        'w_states',
        (self.state_dims, HIDDEN_UNITS),
        trainable=trainable
      )
      w_actions = tf.get_variable(
        'w_actions',
        (self.action_dims, HIDDEN_UNITS),
        trainable=trainable
      )
      w_bias = tf.get_variable(
        'w_bias',
        (1, HIDDEN_UNITS),
        trainable=trainable
      )
      matS = tf.matmul(states, w_states)
      matA = tf.matmul(actions, w_actions)
      hidden = tf.nn.relu(matS + matA + w_bias)
      Q = tf.layers.dense(hidden, 1, name=name + '_Qa')
      Q = tf.squeeze(Q, axis=1)
      return Q

  def _build_training(self):
    self.sess = tf.Session()

    # placeholders
    self.states_in = tf.placeholder(
      tf.float32,
      (None, self.state_dims),
      name='states_in'
    )
    self.rewards_in = tf.placeholder(
      tf.float32,
      (None,),
      name='rewards_in'
    )
    self.next_states_in = tf.placeholder(
      tf.float32,
      (None, self.state_dims),
      name='next_states_in'
    )

    # 4 models
    with tf.variable_scope('actor'):
      self.eval_actions = self._build_actor(
        self.states_in,
        'eval',
        True
      )
      self.target_next_actions = self._build_actor(
        self.next_states_in,
        'target',
        False
      )

    with tf.variable_scope('critic'):
      self.eval_Q = self._build_critic(
        self.states_in,
        self.eval_actions,
        'eval',
        True
      )
      self.target_next_Q = self._build_critic(
        self.next_states_in,
        self.target_next_actions,
        'target',
        False
      )

    # copy variables
    aes = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='actor/eval'
    )
    ats = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='actor/target'
    )
    ces = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='critic/eval'
    )
    cts = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='critic/target'
    )

    self.sync_nets = []
    for ae, at, ce, ct in zip(aes, ats, ces, cts):
      self.sync_nets.append(at.assign(ae))
      self.sync_nets.append(ct.assign(ce))

    # losses
    self.actor_loss = -tf.reduce_mean(self.eval_Q)
    optimizer = tf.train.AdamOptimizer(LR_ACTOR)
    self.train_actor = optimizer.minimize(self.actor_loss, var_list=aes)

    self.real_Q = self.rewards_in + GAMMA * self.target_next_Q
    self.critic_loss = tf.losses.mean_squared_error(self.real_Q, self.eval_Q)
    optimizer = tf.train.AdamOptimizer(LR_CRITIC)
    self.train_critic = optimizer.minimize(self.critic_loss, var_list=ces)

    # init
    self.sess.run(tf.global_variables_initializer())

  def act(self, state, learning=True):
    states = np.expand_dims(state, 0)
    action = self.sess.run(self.eval_actions, {self.states_in: states})[0]
    if learning:
      action = np.random.normal(action, self.variety)
      action = np.clip(action, -self.action_bound, self.action_bound)
    return action

  def remember(self, state, action, reward, next_action):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.next_actions.append(next_action)

  def replay(self):
    batch_size = min(BATCH_SIZE, len(self.states))
    batch_indexes = np.random.choice(len(self.states), batch_size)

    states = np.array(self.states)[batch_indexes]
    actions = np.array(self.actions)[batch_indexes]
    rewards = np.array(self.rewards)[batch_indexes]
    next_actions = np.array(self.next_actions)[batch_indexes]

    actor_loss, _ = self.sess.run(
      (self.actor_loss, self.train_actor), 
      {self.states_in: states}
    )

    critic_loss, _ = self.sess.run(
      (self.critic_loss, self.train_critic), 
      {
        self.states_in: states,
        self.eval_actions: actions,
        self.rewards_in: rewards,
        self.next_states_in: next_actions
      }
    )

    if self.variety > VARIETY_MIN:
      self.variety *= VARIETY_DECAY

    self.replay_num += 1
    if self.replay_num % SYNC_FREQUENCY == 0:
      self.sess.run(self.sync_nets)

    return (actor_loss, critic_loss)

  def save(self, path):
    self.saver.save(self.sess, path)

  def load(self, path):
    self.saver.restore(self.sess, path)
