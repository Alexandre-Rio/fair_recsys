import gym
from gym import spaces
import numpy as np

from gym.envs.registration import register
from gym.wrappers import FlattenObservation


class Item:
    """
    Item parent class.
    """
    def __init__(self, **values):
        self.item_id = values["item_id"]
        self.quality = values["quality"]
        self.cluster_id = values["cluster_id"]


class ItemSampler:
    """
    Item sampler.
    """
    def __init__(self, num_items=10, num_clusters=2):
        self.num_items = num_items
        self.num_clusters = num_clusters

    def sample_items(self):
        items_list = []
        for id in range(self.num_items):
            quality = 5 * np.random.rand()
            cluster_id = np.random.choice(np.arange(0, self.num_clusters))
            values = {"item_id": id, "quality": quality, "cluster_id": cluster_id}
            item = Item(**values)
            items_list.append(item)
        return items_list


class SimpleRecSimEnv(gym.Env):
    """
    Simplified RecSim environment with one-item slates.
    """
    def __init__(self,
                 dec=False,
                 num_items=10,
                 num_clusters=2,
                 max_episode_length=50,
                 fairness_weight=0.0,
                 user_dropout_prob=0.01,
                 null_utility=1.25,
                 fairness_target=0.7,
                 bonus_type='P'):
        self.dec = dec
        # Problem parameters
        self.num_items = num_items  # The size of the square grid
        self.num_clusters = num_clusters
        self.quality_ranges = np.array([5] * self.num_items)
        self.cluster_id_ranges = np.array([self.num_clusters] * self.num_items)
        self.max_episode_length = max_episode_length
        self.fairness_target = fairness_target
        self.cluster_target = np.array([self.fairness_target, 1 - self.fairness_target])  # o modify for mroe clusters
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
        self.item_sampler = ItemSampler(self.num_items, self.num_clusters)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "timestep": spaces.Box(0, self.max_episode_length, shape=(self.max_episode_length,), dtype=int),
                "quality": spaces.Box(0, 5, shape=(self.num_items,), dtype=np.float32),
                "cluster_id": spaces.Box(0, self.cluster_id_ranges, shape=(self.num_items,), dtype=int),
                "cluster_prop": spaces.Box(0, 1, shape=(self.num_clusters,), dtype=np.float32)
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(self.num_items)

    def flatten_obs(self, obs_dict):
        objects = [feature for feature in obs_dict.values()]
        objects[0] = np.array([objects[0]])
        return np.concatenate(objects)

    def _get_obs(self):
        items = self.item_sampler.sample_items()
        quality_vec = np.array([item.quality for item in items])
        cluster_id_vec = np.array([item.cluster_id for item in items])
        obs = {"timestep": self.timestep, "quality": quality_vec, "cluster_id": cluster_id_vec,
               "cluster_prop": self.cluster_prop}
        self.current_obs = obs
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

        observation = self._get_obs()
        info = self._get_info(0, 0, 0)

        return observation, info

    def step(self, action):
        # Retrieve selected item
        qualities = self.current_obs["quality"]
        cluster_ids = self.current_obs["cluster_id"]
        quality = qualities[action]
        cluster_id = cluster_ids[action]

        # Update time step and cluster proportions
        self.timestep += 1
        self.cluster_counts[cluster_id] += 1
        self.cluster_prop = self.cluster_counts / self.cluster_counts.sum()

        # An episode is done if the maximum number of steps has been reached or if user has dropped out
        dropped_out = False
        z = np.random.rand()
        if z < self.user_dropout_prob:
            dropped_out = True
        terminated = (self.timestep == self.max_episode_length) or dropped_out
        reward = self._get_reward(quality, cluster_id)
        best_item = qualities.argmax()
        regret = (qualities.max() - quality) / qualities.max()

        # Get new state
        observation = self._get_obs()
        info = self._get_info(quality, best_item, regret)

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
