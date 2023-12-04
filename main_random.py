import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from itertools import count
import pandas as pd

import gym
from gym.envs.registration import register
from gym.wrappers import FlattenObservation

from utils import log_params

import time

#register(
#        id='SimpleRecSimEnv-v0',
#        entry_point='envs.simple_recsim:SimpleRecSimEnv'
#    )
#register(
#        id='MovieLensRecSimEnv-v0',
#        entry_point='envs.movielens_recsim:MovieLensRecSimEnv'
#    )


def main_random(args):
    np.random.seed(0)
    num_episodes = args.num_episodes

    name_env = args.name_env
    env = gym.make(name_env,
                   user_dropout_prob=args.user_dropout_prob,
                   num_items=args.num_candidate_items,
                   num_clusters=args.num_clusters,
                   max_episode_length=args.max_episode_length,
                   null_utility=args.null_utility,
                   fairness_target=args.fairness_target)
    env = FlattenObservation(env)

    results_df_list = []
    for id_trial in range(args.num_trials):
        print(f"Trial nb. {id_trial}")
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()

        global steps_done
        steps_done = 0

        episode_rewards = []
        episode_fairness = []
        episode_quality = []
        episode_accuracy = []
        episode_regret = []

        for i_episode in range(num_episodes):
            start = time.time()
            # Initialize the environment and get it's state
            print(f"Trial{id_trial}/{args.num_trials} Episode {i_episode}")
            state, _ = env.reset()
            cum_reward = 0
            cum_quality = 0
            correct_pred = 0
            cum_regret = 0
            for t in count():
                action = np.random.choice(n_actions)
                observation, reward, terminated, truncated, info = env.step(action)
                user_choice, regret = info["user_choice"], info["regret"]
                correct_pred += (user_choice == action)
                cum_regret += regret
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation

                # Store the transition in memory and increment reward
                cum_reward += reward
                cum_quality += info["quality"]

                # Move to the next state
                state = next_state

                if done:
                    end = time.time()
                    print(f'Episode duration: {end - start}s')
                    episode_rewards.append(cum_reward / (t + 1))
                    episode_quality.append(cum_quality / (t + 1))
                    episode_accuracy.append(correct_pred / (t + 1))
                    episode_regret.append(cum_regret / (t + 1))
                    fairness_metric = info["fairness_metric"]
                    episode_fairness.append(fairness_metric)
                    break

        results_df_trial = pd.DataFrame(
            [episode_rewards, episode_regret, episode_accuracy, episode_quality, episode_fairness]).T
        results_df_trial.columns = ['reward', 'regret', 'accuracy', 'quality', 'fairness']
        results_df_trial['trial'] = id_trial
        results_df_list.append(results_df_trial)

    t = 1000 * time.time()
    np.random.seed(int(t) % 2 ** 32)  # Set new random for file id
    xpid = np.random.randint(1e6)
    file_name = f'{args.method}_{xpid}'
    results_df = pd.concat(results_df_list)
    results_df.set_index(['trial', results_df.index])
    results_df = results_df[['reward', 'regret', 'accuracy', 'quality', 'fairness']]
    results_df_mean = results_df.mean(level=0)
    results_df_mean.columns = ['reward_mean', 'regret_mean', 'accuracy_mean', 'quality_mean', 'fairness_mean']
    results_df_std = results_df.std(level=0)
    results_df_std.columns = ['reward_std', 'regret_std', 'accuracy_std', 'quality_std', 'fairness_std']
    results_df = pd.concat([results_df_mean, results_df_std], axis=1)
    results_df.to_csv(f'./results/results_' + file_name + '.csv')

    params_dict = {
        'id': xpid,
        'method': args.method,
        'name_env': args.name_env,
        'num_episodes': args.num_episodes,
        'cluster_target': env.cluster_target,
        'user_dropout_prob': env.user_dropout_prob,
        'null_utility': env.null_utility,
        'max_episode_length': env.max_episode_length,
        'num_candidate_items:': env.num_items,
        'num_clusters': env.num_clusters,
        'num_trials': args.num_trials
    }
    log_params(file_name, params_dict)
