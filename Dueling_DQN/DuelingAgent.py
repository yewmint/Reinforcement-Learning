import numpy as np
import tensorflow as tf

from Memory import Memory

HIDDEN_UNITS = 20
GAMMA = 0.9
LR = 0.001
BATCH_SIZE = 64
SYNC_FREQUENCY = 20
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

class DuelingAgent:
  """
  Agent to play cartpole-v0 by Dueling DQN
  """
  def __init__(self, state_dims, action_size):
    """
    :param state_dims: dims of state
    :param action_size: size of action
    """
    self.state_dims = state_dims
    self.action_size = action_size

    self.memory = Memory(['states', 'actions', 'rewards', 'next_states'], 5000)
    self.replay_num = 0
    self.epsilon = 1

    self._build_training()
    self.saver = tf.train.Saver()

  def _build_dueling(self, name, states_in, trainable):
    """
    build dueling model

    :param name: name of dueling model
    :param states_in: input states
    :param trainable: if model is trainable
    """
    with tf.variable_scope(name):
      hidden = tf.layers.dense(
        states_in, 
        HIDDEN_UNITS,
        activation=tf.nn.relu,
        name=name + '_hidden'
      )
      values = tf.layers.dense(
        hidden,
        1,
        activation=None,
        name=name + '_value'
      )
      advantages = tf.layers.dense(
        hidden,
        self.action_size,
        activation=None,
        name=name + '_advantage'
      )
      Qs = tf.add(
        values,
        advantages - tf.reduce_mean(advantages, axis=1, keepdims=True),
        name=name + '_Qs'
      )
    return Qs

  def _build_training(self):
    """
    build training network
    """
    # placeholders
    self.states_in = tf.placeholder(
      tf.float32, 
      (None, self.state_dims), 
      name='states_in'
    )
    self.actions_in = tf.placeholder(
      tf.int32, 
      (None,), 
      name='actions_in'
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

    # models
    self.eval_Qs = self._build_dueling(
      'eval', 
      self.states_in, 
      trainable=True
    )
    self.next_target_Qs = self._build_dueling(
      'target', 
      self.next_states_in, 
      trainable=False
    )

    # copy
    eps = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='eval'
    )
    tps = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, 
      scope='target'
    )

    self.sync_nets = []
    for ep, tp in zip(eps, tps):
      self.sync_nets.append(tp.assign(ep))

    # losses
    next_target_Qas = tf.reduce_max(self.next_target_Qs, axis=1)
    target_Qas = self.rewards_in + GAMMA * next_target_Qas

    actions = tf.one_hot(self.actions_in, self.action_size)
    eval_Qas = tf.reduce_sum(self.eval_Qs * actions, axis=1)

    self.loss = tf.losses.mean_squared_error(target_Qas, eval_Qas)
    optimizer = tf.train.AdamOptimizer(LR)
    self.train_op = optimizer.minimize(self.loss)

    # session
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def act(self, state, learning=True):
    """
    get action of current state

    :param state: current state
    :param learning: if exploration is enabled
    """
    if learning and np.random.rand() <= self.epsilon:
      return np.random.randint(0, self.action_size)
    else:
      Qs = self.sess.run(
        self.eval_Qs, 
        {self.states_in: np.expand_dims(state, 0)}
      )
      return np.argmax(Qs[0])
    
  def remember(self, state, action, reward, next_state):
    """
    save memory slice

    :param state:
    :param action:
    :param reward:
    :param next_state:
    """
    self.memory.remember({
      'states': state,
      'actions': action,
      'rewards': reward,
      'next_states': next_state,
    })

  def replay(self):
    """
    replay memory and train model
    """
    batch = self.memory.random_batch(BATCH_SIZE)

    loss, _ = self.sess.run(
      (self.loss, self.train_op),
      {
        self.states_in: batch['states'],
        self.actions_in: batch['actions'],
        self.rewards_in: batch['rewards'],
        self.next_states_in: batch['next_states'],
      }
    )

    self.replay_num += 1
    if self.replay_num % SYNC_FREQUENCY == 0:
      self.sess.run(self.sync_nets)

    if self.epsilon > EPSILON_MIN:
      self.epsilon *= EPSILON_DECAY

    return loss

  def save(self, path):
    """
    save weights
    """
    self.saver.save(self.sess, path)

  def load(self, path):
    """
    load weights
    """
    self.saver.restore(self.sess, path)
