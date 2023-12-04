import gym
from gym import spaces
import numpy as np
import pandas as pd
import os

from gym.envs.registration import register
from gym.wrappers import FlattenObservation

data_root = 'envs/data/ml-100k'


class Item:
    """
    Item parent class.
    """
    def __init__(self, values):
        self.item_id = int(values["movieId"])
        self.quality = float(values["movie_mean"])
        self.cluster_id = int(values["disc"])


class ItemSampler:
    """
    Item sampler.
    """
    def __init__(self, data, num_items=10, num_clusters=2):
        self.data = data
        self.item_ids = self.data['movieId'].unique()
        self.num_items = num_items
        self.num_clusters = num_clusters

    def sample_items(self, user_data, user_step):
        true_item_id = user_data.iloc[user_step]["movieId"]
        true_item = Item(self.data[self.data["movieId"] == true_item_id][["movieId", "movie_mean", "disc"]].drop_duplicates("movieId"))
        items_list = [true_item]
        for id in range(self.num_items - 1):
            item_id = np.random.choice(self.item_ids, replace=False)
            item = Item(self.data[self.data["movieId"] == item_id][["movieId", "movie_mean", "disc"]].drop_duplicates("movieId"))
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
        self.data = user_data
        self.session_length = self.data.shape[0]
        self.choice_model = ConditionalLogit(temperature=10) if choice_model == 'cond_logit' else None


class UserSampler:
    """
    Item sampler.
    """
    def __init__(self, data, num_clusters=2):
        self.data = data
        self.user_ids = self.data['userId'].unique()
        self.num_clusters = num_clusters

    def sample_user(self):
        user_id = np.random.choice(self.user_ids)
        user_data = self.data[self.data['userId'] == user_id]
        #aff_scores = np.random.rand(self.num_clusters)
        user = User(user_id, user_data)
        return user


class MovieLensRecSimEnv(gym.Env):
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
                 bonus_type='P'):
        self.dec = dec  # True if reward decomposition
        self.data = pd.read_csv(os.path.join(data_root, 'ratings_proc.csv'))

        # Problem parameters
        self.num_items = num_items  # The size of the square grid
        self.num_clusters = num_clusters
        self.quality_ranges = np.array([5] * self.num_items)
        self.cluster_id_ranges = np.array([self.num_clusters] * self.num_items)
        self.max_episode_length = max_episode_length
        self.cluster_target = np.array([fairness_target, 1 - fairness_target])  # o modify for mroe clusters
        self.fairness_weight = fairness_weight
        self.user_dropout_prob = user_dropout_prob
        self.null_utility = null_utility
        self.bonus_type = bonus_type

        # Problem variables
        self.timestep = 0
        self.cluster_counts = np.zeros(self.num_clusters)
        self.cluster_prop = np.zeros(self.num_clusters)
        self.current_obs = None

        # Instances
        self.item_sampler = ItemSampler(self.data, self.num_items, self.num_clusters)
        self.user_sampler = UserSampler(self.data, self.num_clusters)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "timestep": spaces.Box(0, self.max_episode_length, shape=(self.max_episode_length,), dtype=int),
                "quality": spaces.Box(0, 5, shape=(self.num_items,), dtype=np.float32),
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

    def _get_obs(self, terminated):
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

    def _get_info(self, quality, user_choice, regret):
        return {"fairness_metric": np.abs(self.cluster_target - self.cluster_prop).sum(), "quality": quality, "user_choice": user_choice, "regret": regret}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset problem variables
        self.timestep = 0
        self.cluster_counts = np.zeros(self.num_clusters)
        self.cluster_prop = np.zeros(self.num_clusters)

        # Sample new user
        self.current_user = self.user_sampler.sample_user()

        # Get obs and info
        observation = self._get_obs(False)
        info = self._get_info(0, 0, 0)

        return observation, info

    def step(self, action):
        # Retrieve recommended item
        qualities = self.current_obs["quality"]
        cluster_ids = self.current_obs["cluster_id"]
        quality = qualities[action]  # expected quality
        cluster_id = cluster_ids[action]

        # Retrieve selected items
        selected_item_data = self.current_user.data.iloc[self.timestep]
        ground_truth_item = selected_item_data["movieId"]
        ground_truth_rating = selected_item_data["rating"]
        regret = (ground_truth_rating - quality) / ground_truth_rating

        # Update time step and cluster proportions
        self.timestep += 1
        self.cluster_counts[cluster_id] += 1
        self.cluster_prop = self.cluster_counts / self.cluster_counts.sum()

        # An episode is done if the maximum number of steps has been reached or if user has dropped out
        dropped_out = (self.timestep >= self.current_user.session_length)
        terminated = (self.timestep == self.max_episode_length) or dropped_out
        reward = self._get_reward(quality, cluster_id)

        # Get new state
        observation = self._get_obs(terminated)
        info = self._get_info(quality, ground_truth_item, regret)

        return observation, reward, terminated, False, info

    def _get_reward(self, quality, cluster_id):
        reward = quality
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
