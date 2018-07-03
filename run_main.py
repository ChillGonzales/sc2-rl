from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from factories import get_agent_from_name, get_game_env
from collections import deque
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import argparse
import random
import time
import utils

# Global variables
ACT_DIM = 7
TOTAL_FN = 541

def flatten_features(obs):
    flat_list = np.array([])
    feature_keys = ['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen',
      'last_actions', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'player', 'control_groups', 
      'available_actions']
    # feature_keys = ['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'rgb_screen',
    #   'rgb_minimap', 'last_actions', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'player', 'control_groups', 
    #   'available_actions']
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
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        explore_prob = 0.9
        chosen_count = 0
        override_r = False

        for epoch in range(nb_epochs):
            print("Starting epoch", epoch)
            # Exponentially decay our explore rate over time 
            explore_prob = explore_prob * np.power(0.999, epoch) 
            if explore_prob < 0.01: 
                explore_prob = 0.01 

            # Perform rollouts.
            for t in range(nb_rollout_steps):
                # Predict next action.
                action_values, q = agent.step(features)
                z_action_values = utils.convert_to_zscore(action_values)

                # First index of actions is the function id, rest are argument values
                available_actions = np.sort(available_actions, kind='mergesort')
                fun_id = available_actions[int(z_action_values[0] * len(available_actions)) - 1]
                required_args = actions.FUNCTIONS[fun_id].args

                # Choose the network's output with a probability of (1-explore_prob) 
                if random.random() > explore_prob: 
                    # required_args = agent.action_spec[0].functions[fun_id].args
                    args = []
                    i = 1
                    for arg in required_args:
                        sizes = []
                        for size in arg.sizes:
                            sizes.append(int(z_action_values[i] * (size - 1)))
                            i += 1
                        args.append(sizes)

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
                    override_r = True
                    print("Invalid action/args chosen. Function id:", fun_id, "Args:", args, "Available:", available_actions, "Available args:", required_args)

                # TODO: Do we use game score for rewards or win/loss? 
                #int(np.sum(new_obs.observation.score_cumulative[3:7])),
                new_features, r, available_actions = new_obs.observation, new_obs.reward, new_obs.observation.available_actions
                new_features = flatten_features(new_features)[:agent.obs_shape[0]] # Trim off feature set in case it changes size to fit network
                diff = agent.obs_shape[0] - len(new_features)
                if diff > 0:
                    new_features = np.append(new_features, [0 for i in range(0, diff)])
                    new_features.flatten()

                # Book-keeping
                if override_r:
                    r = -1
                episode_reward += r
                episode_step += 1
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(features, action_values, r, new_features, done)
                obs = new_obs
                features = new_features
                override_r = False

                if obs.last():
                    break

            # Epoch done.
            epoch_episode_rewards.append(episode_reward)
            episode_rewards_history.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            print("Epoch", epoch, "complete. Total reward:", episode_reward, ". Final reward:", r, ". Chosen percent:", 
                          (chosen_count / (t + 1)) * 100, ". Explore Prob: ", explore_prob, ". Steps taken:", t + 1, ". Win:", obs.reward != -1) 
            episode_reward = 0.
            episode_step = 0
            epoch_episodes += 1
            chosen_count = 0

            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            nb_train_steps = 100
            batch_size = int(agent.memory.nb_entries / 2)
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
            saver.save(sess, './checkpoints/model_epoch', global_step=epoch)

            # Reset game after training is complete
            agent.reset()
            obs = game.reset()

            #TODO: Implement evaluation

        saver.save(sess, './checkpoints/model_final')

def main(nb_epochs, max_rollouts, agent_type_name, map_name, step_mul):
    game = get_game_env(step_mul=step_mul,
                        map_name=map_name,
                        players=[])

    # Set size of network by resetting the game to get observation space
    init_obs = game.reset()[0]
    print(np.shape(init_obs.observation.feature_screen))
    obs_dimension = len(flatten_features(init_obs.observation))

    agent = get_agent_from_name(agent_type_name)

    # Setup agent
    obs_shape = (obs_dimension, )
    nb_actions = ACT_DIM
    agent.setup(obs_shape=obs_shape, 
                nb_actions=nb_actions, 
                action_spec=game.action_spec(),
                noise_type="adaptive-param_0.01,ou_0.01")
    
    # Run the training
    run_agent(agent, game, nb_epochs, max_rollouts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=2500, help="Number of epochs to train the agent.")
    parser.add_argument("--rollout", default=512, help="Number of rollouts to limit the agent to per epoch.")
    parser.add_argument("--agent", default="ddpg", help="Name of agent type to train with. Available: 'ddpg'")
    parser.add_argument("--map", default="MoveToBeacon", help="Name of map to train agent on.")
    parser.add_argument("--stepmul", default=3, help="Action step to game step multiplier."
        "The higher the number the more steps the game will take between agent actions.")
    args = parser.parse_args()
    # TODO: Figure out how to remove this flags stuff without breaking sc2 environment
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main(nb_epochs=args.epoch, 
         max_rollouts=args.rollout, 
         agent_type_name=args.agent, 
         map_name=args.map,
         step_mul=args.stepmul)