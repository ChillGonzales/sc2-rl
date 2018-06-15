from ddpg_agent import DDPGAgent

""" Factory for providing agents from a name """
def get_agent_from_name(name="ddpg"):
  if name == "ddpg":
    return DDPGAgent()