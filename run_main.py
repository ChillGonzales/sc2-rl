from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env.sc2_env import SC2Env, Agent, Race, Bot, Difficulty
from pysc2.env.run_loop import run_loop
from ddpg_agent import DDPGAgent
from collections import deque
import time
import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
import random

# Global variables/aliases
Dimensions = features.Dimensions
AgentInterfaceFormat = features.AgentInterfaceFormat
OBS_DIM = 680012
ACT_DIM = 7
TOTAL_FN = 541


def flattenFeatures(obs):
    flat_list = np.array([])
    for feature in ['feature_screen', 'available_actions', 'action_result', 'single_select', 'multi_select']:
        for lst in obs[feature]:
            flat_list = np.append(flat_list, lst)
    flat_list.flatten()
    return flat_list

def runAgent(agent, game, nb_epochs, nb_rollout_steps):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        agent.initialize(sess)
        sess.graph.finalize()
        step = 0
        episode = 0
        eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        # Prepare everything.
        agent.reset()
        obs = game.reset()[0]  # Only care about 1 agent right now
        features, available_actions = flattenFeatures(
            obs.observation), obs.observation.available_actions
        done = False
        episode_reward = 0.
        episode_step = 0

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
            print("Starting epoch ", epoch)
            # Perform rollouts.
            for t in range(nb_rollout_steps):
                explore_prob = explore_prob * np.power(0.999, epoch)
                # Predict next action.
                action_values, q = agent.step(features)

                # First index of actions is the function id, rest are argument values
                fun_id = available_actions[int(action_values[0] * len(available_actions))]
                # If our choice isn't available then just take a random action.
                # TODO: Should we be masking the unavailable actions and then selecting based on that distribution?
                # valid = fun_id in available_actions
                if random.random() > explore_prob:
                    required_args = agent.action_spec[0].functions[fun_id].args
                    args = [[int(action_values[i] * size) for size in required_args[i].sizes]
                                    for i in range(len(required_args))]
                    chosen_count += 1
                else:
                    fun_id = np.random.choice(available_actions)
                    args = [[np.random.randint(0, size) for size in arg.sizes]
                        for arg in agent.action_spec[0].functions[fun_id].args]
                
                action = actions.FunctionCall(fun_id, args)

                try:
                    new_obs = game.step([action])[0]
                except ValueError:
                    new_obs = game.step([actions.FunctionCall(0, [])])[0]

                new_features, r, available_actions = new_obs.observation, int(np.sum(new_obs.observation.score_cumulative[3:7])), new_obs.observation.available_actions
                new_features = flattenFeatures(new_features)[:OBS_DIM]

                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                # if r != 0 or chosen_count <= 5:
                agent.store_transition(features, fun_id, r, new_features, done)
                    # last_nonzero_r = r
                obs = new_obs
                features = new_features

                if obs.last():
                    # Episode done.
                    epoch_episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    epoch_episode_steps.append(episode_step)
                    print("Epoch", epoch, "complete. Total reward:", r, ". Final reward:", r, ". Chosen percent:",
                          (chosen_count / (t + 1)) * 100, ". Explore Prob: ", explore_prob, ". Steps taken:", t + 1, ". Win:", obs.reward != -1)
                    episode_reward = 0.
                    episode_step = 0
                    chosen_count = 0
                    epoch_episodes += 1
                    agent.reset()
                    obs = game.reset()
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
            #saver.save(sess, 'model_epoch', global_step=1)

            # Evaluate.
            # eval_episode_rewards = []
            # eval_qs = []
            # if eval_env is not None:
            #   eval_episode_reward = 0.
            #   for t_rollout in range(nb_eval_steps):
            #     eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
            #     eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            #     if render_eval:
            #       eval_env.render()
            #     eval_episode_reward += eval_r

            #     eval_qs.append(eval_q)
            #     if eval_done:
            #       eval_obs = eval_env.reset()
            #       eval_episode_rewards.append(eval_episode_reward)
            #       eval_episode_rewards_history.append(eval_episode_reward)
            #       eval_episode_reward = 0.
        # saver.save(sess, 'model_final')

def main():
    dims = Dimensions(screen=(200, 200), minimap=(50, 50))
    format = AgentInterfaceFormat(feature_dimensions=dims)
    game = SC2Env(map_name="Simple64",
                  players=[Agent(Race.zerg), Bot(Race.terran, Difficulty.easy)],
                  step_mul=16,
                  agent_interface_format=format,
                  visualize=False)

    agent = DDPGAgent(game.action_spec())
    obs_shape = (OBS_DIM, )
    nb_actions = ACT_DIM
    agent.setup(obs_shape, nb_actions, TOTAL_FN,
                noise_type="adaptive-param_0.01,ou_0.01")
    runAgent(agent, game, 100, 10000)
    # feature_keys = ['single_select', 'multi_select', 'build_queue', 'cargo', 'cargo_slots_available', 'feature_screen',
    #   'feature_minimap', 'last_actions', 'action_result', 'alerts', 'game_loop', 'score_cumulative', 'player', 'control_groups', 'available_actions']


if __name__ == "__main__":
    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    main()
