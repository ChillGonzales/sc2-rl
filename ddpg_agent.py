from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import features
from pysc2.lib import actions
import tensorflow as tf
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.noise import *

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class DDPGAgent(Object):
  """A Deep Deterministic Policy Gradient implementation of an SC2 agent."""

  def setup(self, obs_shape, action_shape, nb_actions, noise_type, gamma, tau, layer_norm=True):
    action_noise = None
    param_noise = None

    # Parse noise_type
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    self.memory = Memory(limit=int(1e6), action_shape=action_shape, observation_shape=obs_shape)
    self.critic = Critic(layer_norm=layer_norm)
    self.actor = Actor(nb_actions, layer_norm=layer_norm)

    tf.reset_default_graph()

    max_action = env.action_space.high
    self.agent = DDPG(actor=self.actor, critic=self.critic, memory=self.memory, observation_shape=env.observation_space.shape, 
        action_shape=env.action_space.shape, gamma=gamma, tau=tau, action_noise=action_noise, param_noise=param_noise)
