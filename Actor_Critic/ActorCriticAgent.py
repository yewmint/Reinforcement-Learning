import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.001
GAMMA = 0.9
HIDDEN_UNITS = 20

class ActorCriticAgent:
  def __init__(self, state_shape, action_size):
    self.state_shape = state_shape
    self.action_size = action_size
    
    self._build_actor()
    self._build_critic()

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

    self.saver = tf.train.Saver()

    self.writer = tf.summary.FileWriter("logs", self.sess.graph)
    self.write_op = tf.summary.merge_all()

    self.learn_num = 0

  def _build_actor(self):
    with tf.variable_scope('actor'):
      # input placeholders
      self.actor_state_in = tf.placeholder(
        tf.float32, 
        shape=(None, *self.state_shape),
        name='state_in'
        )

      self.actor_action_in = tf.placeholder(
        tf.float32,
        shape=(None, self.action_size),
        name='action_in'
        )

      self.actor_qa_in = tf.placeholder(
        tf.float32, 
        shape=(None,),
        name='tderror_in'
        )

      # model
      hidden = tf.layers.dense(
        self.actor_state_in,
        HIDDEN_UNITS,
        activation=tf.nn.relu
        )

      self.actor_dist_out = tf.layers.dense(
        hidden,
        self.action_size,
        activation=tf.nn.softmax
        )

      # trainning
      log_dist = tf.multiply(self.actor_dist_out, self.actor_action_in)
      log_dist = tf.reduce_sum(log_dist, axis=1)
      log_dist = tf.log(log_dist)
      self.actor_loss = tf.reduce_mean(-log_dist * self.actor_qa_in)
      tf.summary.scalar('actor_loss', self.actor_loss)
  
      optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)
      self.actor_train_op = optimizer.minimize(self.actor_loss)

  def _build_critic(self):
    with tf.variable_scope('critic'):
      # input placeholders
      self.critic_state_in = tf.placeholder(
        tf.float32, 
        shape=(None, *self.state_shape),
        name='state_in'
        )

      self.critic_target_qa_in = tf.placeholder(
        tf.float32, 
        shape=(None,),
        name='next_v_in'
        )

      self.critic_action_in = tf.placeholder(
        tf.float32, 
        shape=(None, self.action_size),
        name='reward_in'
        )
      
      # model
      hidden = tf.layers.dense(
        self.critic_state_in,
        HIDDEN_UNITS,
        activation=tf.nn.relu
        )

      self.critic_q_out = tf.layers.dense(
        hidden,
        self.action_size,
        activation=None
        )

      # trainning
      qa = tf.reduce_sum(self.critic_q_out * self.critic_action_in, axis=1)
      tderror = self.critic_target_qa_in - qa
      self.critic_loss = tf.reduce_sum(tf.square(tderror))
      tf.summary.scalar('critic_loss', self.critic_loss)

      optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)
      self.critic_train_op = optimizer.minimize(self.critic_loss)

  def learn(self, state, action, reward, next_state):
    states = np.expand_dims(state, 0)
    actions = np.expand_dims(action, 0)
    rewards = np.expand_dims(reward, 0)
    next_states = np.expand_dims(next_state, 0)

    current_qs = self.sess.run(self.critic_q_out, {
      self.critic_state_in: states
      })
    current_qas = np.sum(current_qs * actions, axis=1)

    next_qs = self.sess.run(self.critic_q_out, {
      self.critic_state_in: next_states
      })
    next_qas = rewards + GAMMA * np.amax(next_qs, axis=1)
    
    self.sess.run(self.critic_train_op, {
      self.critic_state_in: states,
      self.critic_target_qa_in: next_qas,
      self.critic_action_in: actions,
      })

    self.sess.run(self.actor_train_op, {
      self.actor_state_in: states,
      self.actor_action_in: actions,
      self.actor_qa_in: current_qas
      })

    self.learn_num += 1
    summary = self.sess.run(self.write_op, {
      self.critic_state_in: states,
      self.critic_target_qa_in: next_qas,
      self.critic_action_in: actions,
      self.actor_state_in: states,
      self.actor_action_in: actions,
      self.actor_qa_in: current_qas
      })
    self.writer.add_summary(summary, self.learn_num)
    self.writer.flush()

  def act(self, state):
    dists = self.sess.run(self.actor_dist_out, {
      self.actor_state_in: np.expand_dims(state, 0)
      })

    index = np.random.choice(self.action_size, p=dists[0])
    return np.identity(self.action_size)[index]

    # if np.random.rand() <= np.power(0.9995, self.learn_num):
    #   index = np.random.choice(self.action_size)
    #   return np.identity(self.action_size)[index]
    # else:
    #   states = np.expand_dims(state, 0)
    #   qs = self.sess.run(self.critic_q_out, {self.critic_state_in: states})
    #   index = np.argmax(qs[0])
    #   return np.identity(self.action_size)[index]

  def save(self, path):
    self.saver.save(self.sess, path)

  def load(self, path):
    self.saver.restore(self.sess, path)
