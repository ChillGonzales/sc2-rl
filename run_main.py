from pysc2.agents.random_agent import RandomAgent
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env
from pysc2.env.run_loop import run_loop

Dimensions = features.Dimensions
AgentInterfaceFormat = features.AgentInterfaceFormat

def main():
  dims = Dimensions(screen=(200, 200), minimap=(50, 50))
  format = AgentInterfaceFormat(feature_dimensions=dims)
  game = SC2Env(map_name="Simple64",
                agent_interface_format=format,
                visualize=True)
  run_loop([RandomAgent()], env=game)

if __name__ == "__main__":
  import sys
  from absl import flags
  FLAGS = flags.FLAGS
  FLAGS(sys.argv)
  main()