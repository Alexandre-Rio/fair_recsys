import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import gym
from gym.envs.registration import register
from gym.wrappers import FlattenObservation

from utils import log_params, DQN

import time

#register(
#        id='SimpleRecSimEnv-v0',
#        entry_point='envs.simple_recsim:SimpleRecSimEnv'
#    )
#register(
#        id='MovieLensRecSimEnv-v0',
#        entry_point='envs.movielens_recsim:MovieLensRecSimEnv'
#    )
register(
        id='HiringRecSimEnv-v0',
        entry_point='envs.hiring_recsim:HiringRecSimEnv'
    )

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


def select_action(env, policy_net, state, fairness_weight_algo):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # Compute bonuses
    cluster_tensor = state[:, :env.action_space.n].long()
    cluster_prop = state[:, env.action_space.n: env.action_space.n + 2]
    cluster_target = torch.tensor(env.unwrapped.cluster_target, device=device).unsqueeze(dim=0)
    div = cluster_target - cluster_prop
    bonuses = torch.take(div, cluster_tensor)

    if sample > eps_threshold:
        with torch.no_grad():
            return (policy_net(state) + fairness_weight_algo * bonuses).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer, env, fairness_weight_algo):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_bonuses = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values = target_net(non_final_next_states)
        # Next state values
        next_state_values[non_final_mask] = next_q_values.max(1)[0]
        cluster_tensor = non_final_next_states[:, :env.action_space.n].long()
        cluster_prop = non_final_next_states[:, env.action_space.n: env.action_space.n + 2]
        cluster_target = torch.tensor(env.unwrapped.cluster_target, device=device).unsqueeze(dim=0)
        div = cluster_target - cluster_prop
        torch.take(div, cluster_tensor)
        bonuses = torch.take(div, cluster_tensor)
        next_composed_q = next_q_values + fairness_weight_algo * bonuses
        next_state_values[non_final_mask] = next_composed_q.max(1)[0].type(torch.float32)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values + fairness_weight_algo * next_bonuses) * GAMMA + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main_q_bonus(args):
    np.random.seed(0)

    fairness_weight_algo_start = args.fairness_weight_algo_start
    fairness_adaptive = args.fairness_adaptive
    fairness_weight_window = args.fairness_weight_window
    num_episodes = args.num_episodes
    fairness_weight_algo = fairness_weight_algo_start
    discrimination_target = args.discrimination_target

    name_env = args.name_env
    bonus_type = 'P' if args.method in ['q_bonus', 'q_bonus_adaptive'] else 'D'
    env = gym.make(name_env,
                   fairness_weight=args.fairness_weight_env,
                   user_dropout_prob=args.user_dropout_prob,
                   num_items=args.num_candidate_items,
                   num_clusters=args.num_clusters,
                   max_episode_length=args.max_episode_length,
                   null_utility=args.null_utility,
                   bonus_type=bonus_type,
                   fairness_target=args.fairness_target)
    env = FlattenObservation(env)

    results_df_list = []
    for id_trial in range(args.num_trials):
        print(f"Trial nb. {id_trial}")
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        global steps_done
        steps_done = 0

        episode_rewards = []
        episode_fairness = []
        episode_quality = []
        episode_accuracy = []
        episode_regret = []
        episode_fairness_weight = [fairness_weight_algo_start]

        for i_episode in range(num_episodes):
            print(f"Trial{id_trial}/{args.num_trials} Episode {i_episode}")
            start = time.time()
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            cum_reward = 0
            cum_quality = 0
            correct_pred = 0
            cum_regret = 0
            for t in count():
                action = select_action(env, policy_net, state, fairness_weight_algo)
                observation, reward, terminated, truncated, info = env.step(action.item())
                user_choice, regret = info["user_choice"], info["regret"]
                correct_pred += (user_choice == action.item())
                cum_regret += regret
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory and increment reward
                memory.push(state, action, next_state, reward)
                cum_reward += reward.item()
                cum_quality += info["quality"]

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, policy_net, target_net, optimizer, env, fairness_weight_algo)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    end = time.time()
                    print(f'Episode duration: {end - start}s')
                    episode_rewards.append(cum_reward / (t + 1))
                    episode_quality.append(cum_quality / (t + 1))
                    episode_accuracy.append(correct_pred / (t + 1))
                    episode_regret.append(cum_regret / (t + 1))
                    fairness_metric = info["fairness_metric"]
                    episode_fairness.append(fairness_metric)
                    if fairness_adaptive and len(episode_fairness) >= fairness_weight_window:
                        if np.mean(episode_fairness[-fairness_weight_window:]) > discrimination_target:
                            fairness_weight_algo *= 2
                        else:
                            fairness_weight_algo /= 2
                        episode_fairness_weight.append(fairness_weight_algo)
                        print(f"Episode: {i_episode}; Fairness Weight: {fairness_weight_algo}")
                    break

        print('Complete')

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
        'fairness_weight_algo_start': args.fairness_weight_algo_start,
        'fairness_target': args.fairness_target,
        'fairness_tolerance': args.fairness_tolerance,
        'fairness_adaptive': args.fairness_adaptive,
        'fairness_weight_window': args.fairness_weight_window,
        'fairness_weight_env': args.fairness_weight_env,
        'cluster_target': env.cluster_target,
        'user_dropout_prob': env.user_dropout_prob,
        'null_utility': env.null_utility,
        'max_episode_length': env.max_episode_length,
        'num_candidate_items:': env.num_items,
        'num_clusters': env.num_clusters,
        'num_trials': args.num_trials,
        'discrimination_target': args.discrimination_target
    }
    log_params(file_name, params_dict)
