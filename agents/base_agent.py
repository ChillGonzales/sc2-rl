class BaseAgent(object):
  """ A base class to define the sc2 agent interface"""

  def __init__(self):
    return

  def setup(self, obs_shape, nb_actions, action_spec, noise_type, gamma, tau, layer_norm):
    return
  
  def step(self, obs):
    return
  
  def reset(self):
    return

  def initialize(self, sess):
    return
  
  def store_transition(self, obs, action, r, new_obs, done):
    return
  
  def train(self):
    return
  
  def adapt_param_noise(self):
    return
  
  def backprop(self):
    return

  def get_memory_size(self):
    return

  @property
  def action_spec(self):
    return