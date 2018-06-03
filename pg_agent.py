from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import features
from pysc2.lib import actions
import tensorflow as tf

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

class NNAgent(object):
  def __init__(self, hparams, sess):
    # initialization
    self._s = sess

    # build the graph
    self._input = tf.placeholder(tf.float32,
            shape=[None, hparams['input_size']])

    hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal)

    logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams['num_actions'],
            activation_fn=None)

    # op to sample an action
    self._sample = tf.reshape(tf.multinomial(logits, 1), [])

    # get log probabilities
    log_prob = tf.log(tf.nn.softmax(logits))

    # training part of graph
    self._acts = tf.placeholder(tf.int32)
    self._advantages = tf.placeholder(tf.float32)

    # get log probs of actions from episode
    indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self._acts
    act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

    # surrogate loss
    loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))

    # update
    optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
    self._train = optimizer.minimize(loss)

  def act(self, observation):
    # get one action, by sampling
    return self._s.run(self._sample, feed_dict={self._input: [observation]})

  def train_step(self, obs, acts, advantages):
    batch_feed = { self._input: obs, \
            self._acts: acts, \
            self._advantages: advantages }
    self._s.run(self._train, feed_dict=batch_feed)

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class PgAgent(BaseAgent):
  """A Policy Gradient implementation of an SC2 agent."""

  def __init__(self, input_size, num_actions):
    super(PgAgent, self).__init__()
    # hyper parameters
    hparams = {
            'input_size': input_size,
            'hidden_size': 36,
            'num_actions': num_actions,
            'learning_rate': 0.1
    }

    # environment params
    eparams = {
            'num_batches': 40,
            'ep_per_batch': 10
    }

  def setup(self, obs_spec, action_spec):
    super(PgAgent, self).setup(obs_spec, action_spec)
    self.

  def reset(self):
    super(PgAgent, self).reset()

  def step(self, obs):
    super(PgAgent, self).step(obs)



