from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from agent_factory import get_agent_from_name
from collections import deque
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import argparse
import random
import time

# Global variables/aliases
Dimensions = features.Dimensions
AgentInterfaceFormat = features.AgentInterfaceFormat
OBS_DIM = 0
ACT_DIM = 7
TOTAL_FN = 541

def flatten_features(obs):
    flat_list = np.array([])
    feature_keys = ['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen',
      'feature_minimap', 'last_actions', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'player', 'control_groups', 
      'available_actions']
    for feature in feature_keys:
        for lst in obs[feature]:
            flat_list = np.append(flat_list, lst)
    flat_list.flatten()
    return flat_list

def run_agent(agent, game, nb_epochs, nb_rollout_steps):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        agent.initialize(sess)
        sess.graph.finalize()

        # Prepare everything.
        agent.reset()
        obs = game.reset()[0]  # Only care about 1 agent right now
        features, available_actions = flatten_features(obs.observation), obs.observation.available_actions
        done, episode_reward, episode_step = False, 0., 0
        step, episode = 0, 0
        episode_rewards_history = deque(maxlen=100)

        # TODO: Implement epoch timing
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        chosen_count = 0
        explore_prob = 0.9 

        for epoch in range(nb_epochs):
            print("Starting epoch", epoch)
            # Exponentially decay our explore rate over time
            explore_prob = explore_prob * np.power(0.999, epoch)

            # Perform rollouts.
            for t in range(nb_rollout_steps):
                # Predict next action.
                action_values, q = agent.step(features)

                # First index of actions is the function id, rest are argument values
                fun_id = available_actions[int(action_values[0] * len(available_actions))]

                # Choose the network's output with a probability of (1-explore_prob)
                if random.random() > explore_prob:
                    required_args = agent.action_spec[0].functions[fun_id].args
                    args = [[int(action_values[i] * size) for size in required_args[i].sizes]
                                    for i in range(len(required_args))]
                    chosen_count += 1
                else:
                    # "Explore" with random action and args
                    fun_id = np.random.choice(available_actions)
                    args = [[np.random.randint(0, size) for size in arg.sizes]
                        for arg in agent.action_spec[0].functions[fun_id].args]
                
                try:
                    action = actions.FunctionCall(fun_id, args)
                    new_obs = game.step([action])[0]
                except ValueError:
                    new_obs = game.step([actions.FunctionCall(0, [])])[0]

                # TODO: Do we use game score for rewards or win/loss? 
                new_features, r, available_actions = new_obs.observation, int(np.sum(new_obs.observation.score_cumulative[3:7])), new_obs.observation.available_actions
                new_features = flatten_features(new_features)[:OBS_DIM] # Trim off feature set in case it changes size to fit network

                # Book-keeping
                episode_reward += r
                episode_step += 1
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(features, fun_id, r, new_features, done)
                obs = new_obs
                features = new_features

                if obs.last():
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    print("Epoch", epoch, "complete. Total reward:", episode_reward, ". Final reward:", r, ". Chosen percent:",
                          (chosen_count / (t + 1)) * 100, ". Explore Prob: ", explore_prob, ". Steps taken:", t + 1, ". Win:", obs.reward != -1)
                    episode_reward = 0.
                    episode_step = 0
                    chosen_count = 0
                    epoch_episodes += 1
                    break

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            nb_train_steps = 25
            batch_size = 25
            param_noise_adaptation_interval = 25
            print("Training network...")
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if agent.memory.nb_entries >= batch_size and t_train % param_noise_adaptation_interval == 0:
                    distance = agent.ddpg.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.ddpg.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.ddpg.update_target_net()
            saver.save(sess, './checkpoints/model_epoch', global_step=1)

            # Reset game after training is complete
            agent.reset()
            obs = game.reset()

            #TODO: Implement evaluation

        saver.save(sess, './checkpoints/model_final')

def main(nb_epochs, max_rollouts, agent_type_name, map_name, step_mul):
    dims = Dimensions(screen=(200, 200), minimap=(50, 50))
    format = AgentInterfaceFormat(feature_dimensions=dims)
    game = SC2Env(map_name=map_name,
                  players=[Agent(Race.zerg), Bot(Race.terran, Difficulty.easy)],
                  step_mul=step_mul,
                  agent_interface_format=format,
                  visualize=False)

    # Set size of network by resetting the game to get observation space
    init_obs = game.reset()
    OBS_DIM = len(flatten_features(init_obs.observation))

    agent = get_agent_from_name(agent_type_name)

    # Setup agent
    obs_shape = (OBS_DIM, )
    nb_actions = ACT_DIM
    agent.setup(obs_shape=obs_shape, 
                nb_actions=nb_actions, 
                action_spec=game.action_spec(),
                noise_type="adaptive-param_0.01,ou_0.01")
    
    # Run the training
    run_agent(agent, game, nb_epochs, max_rollouts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=200, help="Number of epochs to train the agent.")
    parser.add_argument("--rollout", default=10000, help="Number of rollouts to limit the agent to per epoch.")
    parser.add_argument("--agent", default="ddpg", help="Name of agent type to train with. Available: 'ddpg'")
    parser.add_argument("--map", default="Simple64", help="Name of map to train agent on.")
    parser.add_argument("--stepmul", default=2, help="Action step to game step multiplier."
        "The higher the number the more steps the game will take between agent actions.")
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    args = parser.parse_args()
    main(nb_epochs=args.epoch, 
         max_rollouts=args.rollout, 
         agent_type_name=args.agent, 
         map_name=args.map,
         step_mul=args.stepmul)