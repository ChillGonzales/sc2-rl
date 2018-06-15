from base_agent import BaseAgent
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

class DDPGAgent(BaseAgent):
    """A Deep Deterministic Policy Gradient implementation of an SC2 agent."""

    def __init__(self):
        super(DDPGAgent, self).__init__()
        return

    def setup(self, obs_shape, nb_actions, action_spec, noise_type, gamma=1., tau=0.01, layer_norm=True):
        super(DDPGAgent, self).setup(obs_shape, nb_actions, action_spec, noise_type, gamma, tau, layer_norm)

        self.action_spec = action_spec
        action_noise = None
        param_noise = None

        # Parse noise_type
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(
                    stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(
                    nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(
                    nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError(
                    'unknown noise type "{}"'.format(current_noise_type))

        # Configure components.
        self.memory = Memory(limit=int(50), action_shape=(
            nb_actions, ), observation_shape=obs_shape)
        self.critic = Critic(layer_norm=layer_norm)
        self.actor = Actor(nb_actions, layer_norm=layer_norm)

        tf.reset_default_graph()

        # max_action = env.action_space.high
        self.ddpg = DDPG(actor=self.actor, critic=self.critic, memory=self.memory, observation_shape=obs_shape,
                         action_shape=(nb_actions, ), gamma=gamma, tau=tau, action_noise=action_noise, param_noise=param_noise)

    def step(self, obs):
        super(DDPGAgent, self).step(obs)
        acts, q = self.ddpg.p(obs, apply_noise=True, compute_Q=True)
        # Move distribution from [-1, 1] to [0, 2] and convert to z-score
        actions_z = (2 - (acts + 1)) / 2
        return actions_z, q

    def reset(self):
        super(DDPGAgent, self).reset()
        self.ddpg.reset()

    def initialize(self, sess):
        super(DDPGAgent, self).initialize(sess)
        self.ddpg.initialize(sess)

    def store_transition(self, obs, action, r, new_obs, done):
        super(DDPGAgent, self).store_transition(obs, action, r, new_obs, done)
        self.ddpg.store_transition(obs, action, r, new_obs, done)
    
    def train(self):
        super(DDPGAgent, self).train()
        return self.ddpg.train()
    
    def adapt_param_noise(self):
        super(DDPGAgent, self).adapt_param_noise()
        return self.ddpg.adapt_param_noise()
    
    def backprop(self):
        super(DDPGAgent, self).backprop()
        self.ddpg.update_target_network()
    
    def get_memory_size(self):
        super(DDPGAgent, self).get_memory_size()
        return self.memory.nb_entries
    
    @property
    def action_spec(self):
        super(DDPGAgent, self).action_spec()
        return self.action_spec
