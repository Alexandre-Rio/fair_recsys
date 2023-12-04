import gym
from gym import spaces
import numpy as np
import pandas as pd
import os

from gym.envs.registration import register
from gym.wrappers import FlattenObservation

data_root = 'envs/data/hiring'
DISC_VARIABLE = "gender"  # "gender", "age" or "nationality"

USER_ID_VARIABLE = "company"
ITEM_ID_VARIABLE = "applicantId"
QUALITY_VARIABLE = "score"
GROUND_TRUTH_VARIABLE = "decision"

class Item:
    """
    Item parent class.
    """
    def __init__(self, values, user_data):
        self.user_type = user_data.iloc[0]['company']
        self.item_id = int(values[ITEM_ID_VARIABLE])
        self.quality = float(values[f'{QUALITY_VARIABLE}_{self.user_type}'])
        self.cluster_id = int(values["disc_" + DISC_VARIABLE])
        self.decision = int(values[GROUND_TRUTH_VARIABLE])
        self.features = values[['ind-university_grade', 'ind-debateclub', 'ind-programming_exp', 'ind-international_exp', 'ind-entrepeneur_exp', 'ind-languages', 'ind-exact_study', 'ind-degree_bachelor', 'ind-degree_master', 'ind-degree_phd']]
        self.features['ind-university_grade'] /= 100
        self.features['ind-languages'] /= 3


class ItemSampler:
    """
    Item sampler.
    """
    def __init__(self, data, num_items=10, num_clusters=2):
        self.data = data
        self.data_A = self.data[self.data['company'] == 'A']
        self.data_B = self.data[self.data['company'] == 'B']
        self.data_C = self.data[self.data['company'] == 'C']
        self.data_D = self.data[self.data['company'] == 'D']
        self.item_ids_A = self.data_A[ITEM_ID_VARIABLE].unique()
        self.item_ids_B = self.data_B[ITEM_ID_VARIABLE].unique()
        self.item_ids_C = self.data_C[ITEM_ID_VARIABLE].unique()
        self.item_ids_D = self.data_D[ITEM_ID_VARIABLE].unique()
        self.item_ids = {'A': self.item_ids_A, 'B': self.item_ids_B, 'C': self.item_ids_C, 'D': self.item_ids_D}
        self.num_items = num_items
        self.num_clusters = num_clusters
        self.current_user = None

    def sample_items(self, user_data, user_step):
        items_list = []
        for id in range(self.num_items):
            item_id = np.random.choice(self.item_ids[self.current_user.user_id], replace=False)
            item = Item(self.data[self.data[ITEM_ID_VARIABLE] == item_id].drop_duplicates(ITEM_ID_VARIABLE), user_data)
            items_list.append(item)
        return items_list


class ChoiceModel:
    def __init__(self):
        self.probs = None

    def compute_probs(self, item_scores):
        pass

    def sample_item(self):
        return np.random.choice(np.arange(len(self.probs)), p=self.probs)


class ConditionalLogit(ChoiceModel):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def compute_probs(self, item_scores):
        item_scores *= self.temperature
        C = item_scores.max()
        probs = np.exp(item_scores - C) / np.exp(item_scores - C).sum()
        self.probs = probs
        return probs


class User:
    def __init__(self, user_id, user_data, choice_model='cond_logit'):
        self.user_id = user_id
        self.user_id_num = np.zeros(4)
        alpha_num_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.user_id_num[alpha_num_dict[self.user_id]] = 1

        self.data = user_data
        self.session_length = self.data.shape[0]
        self.choice_model = ConditionalLogit(temperature=10) if choice_model == 'cond_logit' else None


class UserSampler:
    """
    Item sampler.
    """
    def __init__(self, data, num_clusters=2):
        self.data = data
        self.user_ids = self.data[USER_ID_VARIABLE].unique()
        self.num_clusters = num_clusters

    def sample_user(self):
        user_id = np.random.choice(self.user_ids)
        user_data = self.data[self.data[USER_ID_VARIABLE] == user_id]
        #aff_scores = np.random.rand(self.num_clusters)
        user = User(user_id, user_data)
        return user


class PersoHiringRecSimEnv(gym.Env):
    """
    Simplified RecSim environment with one-item slates.
    """
    def __init__(self,
                 dec=False,
                 num_items=10,
                 num_clusters=2,
                 max_episode_length=30,
                 fairness_weight=0.0,
                 user_dropout_prob=0.01,
                 null_utility=1.25,
                 fairness_target=0.7,
                 bonus_type='P',
                 reward_type='continuous'):
        self.dec = dec  # True if reward decomposition
        self.data = pd.read_csv(os.path.join(data_root, 'hiring_proc.csv'))

        # Problem parameters
        self.num_items = num_items
        self.num_clusters = num_clusters
        self.quality_ranges = np.array([5] * self.num_items)
        self.cluster_id_ranges = np.array([self.num_clusters] * self.num_items)
        self.max_episode_length = max_episode_length
        self.cluster_target = np.array([fairness_target, 1 - fairness_target])  # o modify for mroe clusters
        self.fairness_weight = fairness_weight
        self.user_dropout_prob = user_dropout_prob
        self.null_utility = null_utility
        self.bonus_type = bonus_type
        self.reward_type = reward_type

        # Problem variables
        self.timestep = 0
        self.cluster_counts = np.zeros(self.num_clusters)
        self.cluster_prop = np.zeros(self.num_clusters)
        self.current_obs = None
        self.current_info = None
        self.current_items = None

        # Instances
        self.item_sampler = ItemSampler(self.data, self.num_items, self.num_clusters)
        self.user_sampler = UserSampler(self.data, self.num_clusters)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "timestep": spaces.Box(0, self.max_episode_length, shape=(self.max_episode_length,), dtype=int),
                "user_id": spaces.Box(0, 1, shape=(4,), dtype=int),
                #"quality": spaces.Box(0, 5, shape=(self.num_items,), dtype=np.float32),
                "features": spaces.Box(0, 1, shape=(self.num_items * 10,), dtype=np.float32),
                "cluster_id": spaces.Box(0, self.cluster_id_ranges, shape=(self.num_items,), dtype=int),
                "cluster_prop": spaces.Box(0, 1, shape=(self.num_clusters,), dtype=np.float32),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(self.num_items)

    def flatten_obs(self, obs_dict):
        objects = [feature for feature in obs_dict.values()]
        objects[0] = np.array([objects[0]])
        return np.concatenate(objects)

    def _get_obs_old(self, terminated):
        if not terminated:
            items = self.item_sampler.sample_items(self.current_user.data, self.timestep)
            quality_vec = np.array([item.quality for item in items])
            cluster_id_vec = np.array([item.cluster_id for item in items])
            obs = {"timestep": self.timestep, "quality": quality_vec, "cluster_id": cluster_id_vec,
                "cluster_prop": self.cluster_prop}
            self.current_obs = obs
        else:
            obs = self.observation_space.sample()
        return obs

    def _get_obs(self, terminated):
        if not terminated:
            items = self.item_sampler.sample_items(self.current_user.data, self.timestep)
            quality_vec = np.array([item.quality for item in items])
            features_vec = np.concatenate([item.features for item in items])
            cluster_id_vec = np.array([item.cluster_id for item in items])
            user_id = self.current_user.user_id_num
            #obs = {"timestep": self.timestep, "quality": quality_vec, "features": features_vec, "cluster_id": cluster_id_vec,
            #    "cluster_prop": self.cluster_prop}
            obs = {"timestep": self.timestep,
                   "user_id": user_id,
                   #"quality": quality_vec,
                   "features": features_vec,
                   "cluster_id": cluster_id_vec,
                   "cluster_prop": self.cluster_prop}
            self.current_items = items
            self.current_obs = obs
        else:
            obs = self.observation_space.sample()
        return obs

    def _get_info(self, quality, user_choice, regret):
        info = {"fairness_metric": np.abs(self.cluster_target - self.cluster_prop).sum(), "quality": quality, "user_choice": user_choice, "regret": regret}
        self.current_info = info
        return info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset problem variables
        self.timestep = 0
        self.cluster_counts = np.zeros(self.num_clusters)
        self.cluster_prop = np.zeros(self.num_clusters)

        # Sample new user
        self.current_user = self.user_sampler.sample_user()
        self.item_sampler.current_user = self.current_user

        # Get obs and info
        observation = self._get_obs(False)
        info = self._get_info(0, 0, 0)

        return observation, info

    def step(self, action):
        # Retrieve recommended item
        qualities = np.array([item.quality for item in self.current_items])
        quality = qualities[action]
        cluster_ids = self.current_obs["cluster_id"]
        cluster_id = cluster_ids[action]
        decisions = np.array([item.decision for item in self.current_items])

        # Retrieve selected items
        #selected_item_data = self.current_user.data.iloc[self.timestep]
        #ground_truth_item = selected_item_data[ITEM_ID_VARIABLE]
        decision = decisions[action]  # or decision?
        #regret = (ground_truth_rating - quality) / ground_truth_rating
        regret = decision / decisions.mean() if decisions.sum() > 0 else 1.0  # Not really a regret, increasing with perf

        # Update time step and cluster proportions
        self.timestep += 1
        self.cluster_counts[cluster_id] += 1
        self.cluster_prop = self.cluster_counts / self.cluster_counts.sum()

        # An episode is done if the maximum number of steps has been reached or if user has dropped out
        dropped_out = (self.timestep >= self.current_user.session_length)
        terminated = (self.timestep == self.max_episode_length) or dropped_out
        reward = self._get_reward(quality, decision, cluster_id)

        # Get new state
        observation = self._get_obs(terminated)
        info = self._get_info(quality, decision, regret)

        return observation, reward, terminated, False, info

    def _get_reward(self, quality, decision, cluster_id):
        if self.reward_type == 'continuous':
            reward = quality / 5
        elif self.reward_type == 'sparse':
            reward = decision
        if self.dec:
            fairness_bonus = (self.cluster_target - self.current_obs["cluster_prop"])[cluster_id]
            return reward, fairness_bonus
        else:
            if self.bonus_type == 'P':
                fairness_bonus = (self.cluster_target - self.current_obs["cluster_prop"])[cluster_id]
                return reward + self.fairness_weight * fairness_bonus
            elif self.bonus_type == 'D':
                fairness_bonus = 0.0
                if self.fairness_weight > 0 and self.timestep > 1:
                    old_cluster_counts = self.cluster_counts.copy()
                    old_cluster_counts[cluster_id] -= 1
                    old_cluster_prop = old_cluster_counts / old_cluster_counts.sum()
                    old_div = np.abs(old_cluster_prop - self.cluster_target).sum()
                    new_div = np.abs(self.cluster_prop - self.cluster_target).sum()
                    fairness_bonus = (old_div - new_div) * self.cluster_counts.sum()
                return reward + self.fairness_weight * fairness_bonus
