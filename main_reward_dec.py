import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import gym
from gym.envs.registration import register
from gym.wrappers import FlattenObservation

from utils import DQN, log_params

#register(
#        id='SimpleRecSimEnv-v0',
#        entry_point='envs.simple_recsim:SimpleRecSimEnv'
#    )
#register(
#        id='MovieLensRecSimEnv-v0',
#        entry_point='envs.movielens_recsim:MovieLensRecSimEnv'
#    )

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward_q', 'reward_f'))


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


def select_action(env, policy_net_q, policy_net_f, state, fairness_weight_algo, train=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if train:
        fairness_weight_algo = 1.0

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return (policy_net_q(state) + fairness_weight_algo * policy_net_f(state)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net_q, policy_net_f, target_net_q, target_net_f, optimizer_q, optimizer_f, env, fairness_weight_algo):
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
    reward_q_batch = torch.cat(batch.reward_q)
    reward_f_batch = torch.cat(batch.reward_f)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values_q = policy_net_q(state_batch).gather(1, action_batch)
    state_action_values_f = policy_net_f(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_q = torch.zeros(BATCH_SIZE, device=device)
    next_state_values_f = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_q_values_tot = target_net_q(non_final_next_states) + target_net_f(non_final_next_states)
        next_q_values_q = target_net_q(non_final_next_states)
        next_q_values_f = target_net_f(non_final_next_states)
        next_actions = next_q_values_tot.argmax(1)
        next_q_values_q = torch.take(next_q_values_q, next_actions)
        next_q_values_f = torch.take(next_q_values_f, next_actions)
        next_state_values_q[non_final_mask] = next_q_values_q
        next_state_values_f[non_final_mask] = next_q_values_f
    # Compute the expected Q values
    expected_state_action_values_q = (next_state_values_q * GAMMA) + reward_q_batch
    expected_state_action_values_f = (next_state_values_f * GAMMA) + reward_f_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss_q = criterion(state_action_values_q, expected_state_action_values_q.unsqueeze(1))
    loss_f = criterion(state_action_values_f, expected_state_action_values_f.unsqueeze(1))

    # Optimize the quality model
    optimizer_q.zero_grad()
    loss_q.backward()
    # Optimize the fairness model
    optimizer_f.zero_grad()
    loss_f.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_q.parameters(), 100)
    optimizer_q.step()
    torch.nn.utils.clip_grad_value_(policy_net_f.parameters(), 100)
    optimizer_f.step()


def main_reward_dec(args):
    #np.random.seed(0)

    name_env = args.name_env
    env = gym.make(name_env,
                   dec=True,
                   user_dropout_prob=args.user_dropout_prob,
                   num_items=args.num_candidate_items,
                   num_clusters=args.num_clusters,
                   max_episode_length=args.max_episode_length,
                   null_utility=args.null_utility,
                   fairness_target=args.fairness_target)
    env = FlattenObservation(env)

    results_df_list = []
    for id_trial in range(args.num_trials):
        fairness_weight_algo_start = args.fairness_weight_algo_start
        fairness_target = args.fairness_target
        fairness_adaptive = args.fairness_adaptive
        fairness_weight_window = args.fairness_weight_window
        num_episodes_train = args.num_episodes
        num_episodes_eval = args.num_episodes_eval
        fairness_weight_algo = fairness_weight_algo_start
        print(f"Nb. Episodes: {num_episodes_train} / Nb. Episodes Eval.: {num_episodes_eval}")
        print(f"Trial nb. {id_trial}")

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        n_observations = len(state)

        policy_net_q = DQN(n_observations, n_actions).to(device)
        target_net_q = DQN(n_observations, n_actions).to(device)
        target_net_q.load_state_dict(policy_net_q.state_dict())
        policy_net_f = DQN(n_observations, n_actions).to(device)
        target_net_f = DQN(n_observations, n_actions).to(device)
        target_net_f.load_state_dict(policy_net_q.state_dict())

        optimizer_q = optim.AdamW(policy_net_q.parameters(), lr=LR, amsgrad=True)
        optimizer_f = optim.AdamW(policy_net_f.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        global steps_done
        steps_done = 0

        episode_rewards = []
        episode_fairness = []
        episode_quality = []
        episode_accuracy = []
        episode_regret = []
        episode_fairness_weight = [fairness_weight_algo_start]

        for i_episode in range(num_episodes_train):
            start = time.time()
            print(f"Trial{id_trial}/{args.num_trials} Episode {i_episode}")
            # Initialize the environment and get it's state
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            cum_reward = 0
            cum_quality = 0
            correct_pred = 0
            cum_regret = 0
            for t in count():
                action = select_action(env, policy_net_q, policy_net_f, state, fairness_weight_algo)
                observation, reward, terminated, truncated, info = env.step(action.item())
                user_choice, regret = info["user_choice"], info["regret"]
                correct_pred += (user_choice == action.item())
                cum_regret += regret
                reward_q = torch.tensor([reward[0]], device=device)
                reward_f = fairness_weight_algo * torch.tensor([reward[1]], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory and increment reward
                memory.push(state, action, next_state, reward_q, reward_f)
                cum_reward += reward_q.item() + reward_f.item()
                cum_quality += reward_q.item()

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, policy_net_q, policy_net_f, target_net_q, target_net_f, optimizer_q, optimizer_f, env, fairness_weight_algo)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict_q = target_net_q.state_dict()
                policy_net_state_dict_q = policy_net_q.state_dict()
                for key in policy_net_state_dict_q:
                    target_net_state_dict_q[key] = policy_net_state_dict_q[key] * TAU + target_net_state_dict_q[key] * (1 - TAU)
                target_net_q.load_state_dict(target_net_state_dict_q)
                target_net_state_dict_f = target_net_f.state_dict()
                policy_net_state_dict_f = policy_net_f.state_dict()
                for key in policy_net_state_dict_f:
                    target_net_state_dict_f[key] = policy_net_state_dict_f[key] * TAU + target_net_state_dict_f[key] * (1 - TAU)
                target_net_f.load_state_dict(target_net_state_dict_f)

                if done:
                    end = time.time()
                    print(f'Episode duration: {end - start}s')
                    episode_rewards.append(cum_reward / (t + 1))
                    episode_quality.append(cum_quality / (t + 1))
                    fairness_metric = info["fairness_metric"]
                    episode_fairness.append(fairness_metric)
                    episode_accuracy.append(correct_pred / (t + 1))
                    episode_regret.append(cum_regret / (t + 1))
                    if fairness_adaptive and i_episode % fairness_weight_window == 0 and len(
                            episode_fairness[num_episodes_train:]) >= fairness_weight_window:
                        if np.mean(episode_fairness[-fairness_weight_window:]) > fairness_target:
                            fairness_weight_algo = np.min([fairness_weight_algo * 2, 5])
                            #fairness_weight_algo *= 2
                        else:
                            fairness_weight_algo /= 2
                        episode_fairness_weight.append(fairness_weight_algo)
                        print(f"Episode: {i_episode}; Fairness Weight: {fairness_weight_algo}")
                    break

        print("Training Complete")
        print("Evaluating...")

        fairness_weight_algo_start_eval = args.fairness_weight_algo_start_eval
        fairness_target_eval = args.fairness_target_eval
        fairness_adaptive_eval = args.fairness_adaptive_eval
        fairness_weight_window_eval = args.fairness_weight_window_eval
        fairness_weight_algo = fairness_weight_algo_start_eval

        for i_episode in range(num_episodes_eval):
            start = time.time()
            print(f"Trial{id_trial}/{args.num_trials} Episode Eval. {i_episode}")
            # Initialize the environment and get it's state
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            cum_reward = 0
            cum_quality = 0
            correct_pred = 0
            cum_regret = 0
            for t in count():
                action = select_action(env, policy_net_q, policy_net_f, state, fairness_weight_algo, train=False)
                observation, reward, terminated, truncated, info = env.step(action.item())
                user_choice, regret = info["user_choice"], info["regret"]
                correct_pred += (user_choice == action.item())
                cum_regret += regret
                reward_q = torch.tensor([reward[0]], device=device)
                reward_f = torch.tensor([reward[1]], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory and increment reward
                cum_reward += reward_q.item() + reward_f.item()
                cum_quality += reward_q.item()

                # Move to the next state
                state = next_state

                if done:
                    end = time.time()
                    print(f'Episode duration: {end - start}s')
                    episode_rewards.append(cum_reward / (t+1))
                    episode_quality.append(cum_quality / (t+1))
                    fairness_metric = info["fairness_metric"]
                    episode_fairness.append(fairness_metric)
                    episode_accuracy.append(correct_pred / (t + 1))
                    episode_regret.append(cum_regret / (t + 1))
                    if fairness_adaptive_eval and i_episode % fairness_weight_window_eval == 0 and len(
                            episode_fairness[num_episodes_train:]) >= fairness_weight_window_eval:
                        if np.mean(episode_fairness[-fairness_weight_window_eval:]) > fairness_target_eval:
                            fairness_weight_algo = np.min([fairness_weight_algo * 2, 10])
                            #fairness_weight_algo *= 2
                        else:
                            fairness_weight_algo /= 2
                        episode_fairness_weight.append(fairness_weight_algo)
                        print(f"Episode: {i_episode}; Fairness Weight: {fairness_weight_algo}")
                    break

        print('Evaluation Complete')

        results_df_trial = pd.DataFrame(
            [episode_rewards, episode_regret, episode_accuracy, episode_quality, episode_fairness]).T
        results_df_trial.columns = ['reward', 'regret', 'accuracy', 'quality', 'fairness']
        results_df_trial['trial'] = id_trial
        results_df_list.append(results_df_trial)

    t = 1000 * time.time()
    np.random.seed(int(t) % 2 ** 32) # Set new random for file id
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
        'num_episodes_eval': args.num_episodes_eval,
        'fairness_weight_algo_start': args.fairness_weight_algo_start,
        'fairness_target': args.fairness_target,
        'fairness_tolerance': args.fairness_tolerance,
        'fairness_adaptive': args.fairness_adaptive,
        'fairness_weight_window': args.fairness_weight_window,
        'fairness_weight_env': args.fairness_weight_env,
        'fairness_weight_algo_start_eval': args.fairness_weight_algo_start_eval,
        'fairness_target_eval': args.fairness_target_eval,
        'fairness_adaptive_eval': args.fairness_adaptive_eval,
        'fairness_weight_window_eval': args.fairness_weight_window_eval,
        'cluster_target': env.cluster_target,
        'user_dropout_prob': env.user_dropout_prob,
        'null_utility': env.null_utility,
        'max_episode_length': env.max_episode_length,
        'num_candidate_items:': env.num_items,
        'num_clusters': env.num_clusters,
        'num_trials': args.num_trials
    }
    log_params(file_name, params_dict)
