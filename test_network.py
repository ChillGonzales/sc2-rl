import tensorflow as tf
import numpy as np
from run_main import flatten_features
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from factories import get_agent_from_name, get_game_env

Dimensions = features.Dimensions
AgentInterfaceFormat = features.AgentInterfaceFormat

def main():
    game = get_game_env(map_name="CollectMineralShards",
                        players=[],
                        step_mul=1,
                        visualize=True)
    agent = get_agent_from_name("ddpg")

    obs = game.reset()[0]  # Only care about 1 agent right now
    features, available_actions = flatten_features(obs.observation), obs.observation.available_actions

    # Setup agent
    obs_shape = (len(features), )
    nb_actions = 7
    agent.setup(obs_shape=obs_shape, 
                nb_actions=nb_actions, 
                action_spec=game.action_spec(),
                noise_type="adaptive-param_0.01,ou_0.01")
    
    with tf.Session() as sess:
        agent.initialize(sess)
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/model_final")
        sess.graph.finalize()

        for epoch in range(0, 200):
            for step in range(0, 10000):
                action_values, q = agent.step(features)
                # First index of actions is the function id, rest are argument values
                fun_id = available_actions[int(action_values[0] * len(available_actions)) - 1]

                required_args = agent.action_spec[0].functions[fun_id].args
                args = [[int(action_values[i] * size) for size in required_args[i].sizes]
                                for i in range(len(required_args))]
                
                try:
                    action = actions.FunctionCall(fun_id, args)
                    new_obs = game.step([action])[0]
                except ValueError:
                    new_obs = game.step([actions.FunctionCall(0, [])])[0]

                new_features, r, available_actions = new_obs.observation, new_obs.reward, new_obs.observation.available_actions
                new_features = flatten_features(new_features)[:agent.obs_shape[0]] # Trim off feature set in case it changes size to fit network
                diff = agent.obs_shape[0] - len(new_features)
                if diff > 0:
                    new_features = np.append(new_features, [0 for i in range(0, diff)])
                    new_features.flatten()
                obs = new_obs
                features = new_features
                
                if obs.last():
                    # Episode done.
                    # print("Epoch", epoch, "complete. Total reward:", episode_reward, ". Final reward:", r, ". Chosen percent:",
                    #       (chosen_count / (t + 1)) * 100, ". Explore Prob: ", explore_prob, ". Steps taken:", t + 1, ". Win:", obs.reward != -1)
                    print("Epoch done! Result:", obs.reward != -1)
                    break


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()