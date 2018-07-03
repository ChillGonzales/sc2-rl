from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from agents.ddpg_agent import DDPGAgent


""" Factory for providing agents from a name """
def get_agent_from_name(name="ddpg"):
  if name == "ddpg":
    return DDPGAgent()

""" Factory for providing game instance """
def get_game_env(use_rgb=False, 
                screen_dims=(150, 150), 
                minimap_dims=(50, 50), 
                map_name="MoveToBeacon", 
                players=[], 
                step_mul=3,
                visualize=False):

    Dimensions = features.Dimensions
    AgentInterfaceFormat = features.AgentInterfaceFormat

    dims = Dimensions(screen=screen_dims, minimap=minimap_dims)
    format = AgentInterfaceFormat(feature_dimensions=dims, action_space=actions.ActionSpace.FEATURES)
    game = SC2Env(map_name=map_name,
                    players=players,
                    step_mul=step_mul,
                    agent_interface_format=format,
                    visualize=visualize)
    return game