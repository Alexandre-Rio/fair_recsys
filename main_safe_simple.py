import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import pandas as pd
from utils import log_params, DQN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from gym.envs.registration import register
from gym.wrappers import FlattenObservation

#register(
#        id='SimpleRecSimEnv-v0',
#        entry_point='envs.simple_recsim_dec:SimpleRecSimEnvDec'
#    )
#register(
#        id='MovieLensRecSimEnv-v0',
#        entry_point='envs.movielens_recsim:MovieLensRecSimEnv'
#    )

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward_q'))


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


def select_action(policy_net_q, state, safe_actions):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    safe_actions = safe_actions.to(device)
    if sample > eps_threshold:
        with torch.no_grad():
            return safe_actions[policy_net_q(state)[:, safe_actions].max(1)[1].cpu()].view(1, 1)
    else:
        return torch.tensor([[np.random.choice(safe_actions.cpu())]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net_q, target_net_q, optimizer_q, env, ubs):
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

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values_q = policy_net_q(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_q = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        clusters_id = non_final_next_states[:, :env.num_items].squeeze(0) #.reshape(-1, env.num_items, env.num_clusters).argmax(-1)
        future_counts = torch.tensor(env.cluster_counts, device=device).repeat(non_final_next_states.shape[0], env.num_items, 1) + F.one_hot(non_final_next_states[:, :env.num_items].squeeze(0).long())
        slack = torch.tensor(ubs, device=device) - future_counts
        non_safe_mask = torch.any(slack < 0, axis=-1)
        safe_mask = ~non_safe_mask
        empty_safe_mask = safe_mask.sum(axis=1) == 0
        trunc_safe_mask = (clusters_id == (torch.tensor(ubs, device=device) - torch.tensor(env.cluster_counts, device=device)).argmax())
        new_safe_mask = torch.zeros_like(safe_mask)
        new_safe_mask[empty_safe_mask] = trunc_safe_mask[empty_safe_mask, :]
        new_safe_mask[~empty_safe_mask] = trunc_safe_mask[~empty_safe_mask, :]

        next_q_values_q = target_net_q(non_final_next_states)
        next_q_values_q.view(1, -1)[~new_safe_mask.reshape(1, -1)] = -1
        next_q_values_q.view(1, -1)[~new_safe_mask.reshape(1, -1)] = -1
        next_state_values_q[non_final_mask] = next_q_values_q.max(dim=1)[0]
    # Compute the expected Q values
    expected_state_action_values_q = (next_state_values_q * GAMMA) + reward_q_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss_q = criterion(state_action_values_q, expected_state_action_values_q.unsqueeze(1))

    # Optimize the quality model
    optimizer_q.zero_grad()
    loss_q.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_q.parameters(), 100)
    optimizer_q.step()


def main_safe(args):
    np.random.seed(0)

    target = args.fairness_target
    tol = args.fairness_tolerance
    num_episodes = args.num_episodes

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
        print(f"Trial nb. {id_trial}")

        ubs = np.floor([(target + tol) * env.max_episode_length, (1 - target + tol) * env.max_episode_length])

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, _ = env.reset()
        n_observations = len(state)

        policy_net_q = DQN(n_observations, n_actions).to(device)
        target_net_q = DQN(n_observations, n_actions).to(device)
        target_net_q.load_state_dict(policy_net_q.state_dict())

        optimizer_q = optim.AdamW(policy_net_q.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        global steps_done
        steps_done = 0

        episode_rewards = []
        episode_fairness = []
        episode_quality = []
        episode_accuracy = []
        episode_regret = []

        for i_episode in range(num_episodes):
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

                clusters_id = state[:, :env.num_items].squeeze(0)
                future_counts = torch.tensor(env.cluster_counts, device=device).repeat(env.num_items, 1) + F.one_hot(state[0, :env.num_items].long())
                slack = torch.tensor(ubs, device=device) - future_counts
                non_safe_mask = torch.any(slack < 0, axis=-1)
                safe_actions = torch.where(~non_safe_mask)[0]
                if len(safe_actions) == 0:  # If no safe action, it means we have exceeded the constraint for one of the cluster. Take the other cluster instead (but should not happen in practice)
                    end = True
                    safe_actions = \
                    torch.where(clusters_id == (torch.tensor(ubs, device=device) - torch.tensor(env.cluster_counts, device=device)).argmax())[
                        0]  # Choose the "safest" action when no safe actions available
                if len(safe_actions) == 0:  # If still no safe action, all actions are deemed safe...
                    safe_actions = torch.arange(0, env.num_items, device=device)

                action = select_action(policy_net_q, state, safe_actions)
                observation, reward, terminated, truncated, info = env.step(action.item())
                reward_q = torch.tensor([reward[0]], device=device)
                done = terminated or truncated
                user_choice, regret = info["user_choice"], info["regret"]

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory and increment reward
                memory.push(state, action, next_state, reward_q)
                cum_reward += reward_q.item()
                cum_quality += reward_q.item()
                cum_regret += regret
                correct_pred += (user_choice == action.item())

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, policy_net_q, target_net_q, optimizer_q, env, ubs)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict_q = target_net_q.state_dict()
                policy_net_state_dict_q = policy_net_q.state_dict()
                for key in policy_net_state_dict_q:
                    target_net_state_dict_q[key] = policy_net_state_dict_q[key] * TAU + target_net_state_dict_q[key] * (1 - TAU)
                target_net_q.load_state_dict(target_net_state_dict_q)

                if done:
                    end = time.time()
                    print(f'Episode duration: {end - start}s')
                    episode_rewards.append(cum_reward / (t + 1))
                    episode_quality.append(cum_quality / (t + 1))
                    fairness_metric = info["fairness_metric"]
                    episode_fairness.append(fairness_metric)
                    episode_accuracy.append(correct_pred / (t + 1))
                    episode_regret.append(cum_regret / (t + 1))
                    break
        results_df_trial = pd.DataFrame(
            [episode_rewards, episode_regret, episode_accuracy, episode_quality, episode_fairness]).T
        results_df_trial.columns = ['reward', 'regret', 'accuracy', 'quality', 'fairness']
        results_df_trial['trial'] = id_trial
        results_df_list.append(results_df_trial)

    print("Training Complete")

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
        'num_trials': args.num_trials
    }
    log_params(file_name, params_dict)
