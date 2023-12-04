import argparse

from main_random import main_random
from main_q_bonus import main_q_bonus
from main_safe_simple import main_safe
from main_reward_dec import main_reward_dec

import warnings
warnings.filterwarnings("ignore")
from gym.envs.registration import register

register(
        id='PersoHiringRecSimEnv-v0',
        entry_point='envs.perso_hiring_recsim:PersoHiringRecSimEnv'
    )
register(
        id='SimpleRecSimEnv-v0',

        entry_point='envs.simple_recsim:SimpleRecSimEnv'
    )
register(
        id='MovieLensRecSimEnv-v0',

        entry_point='envs.movielens_recsim:MovieLensRecSimEnv'
    )

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--name_env", type=str, default='PersoHiringRecSimEnv-v0')
parser.add_argument("--method", type=str, default='q_bonus')
parser.add_argument("--num_episodes", type=int, default=2500)
parser.add_argument("--num_episodes_eval", type=int, default=1000)
parser.add_argument("--fairness_weight_algo_start", type=float, default=0.0)
parser.add_argument("--fairness_weight_env", type=float, default=0.0)
parser.add_argument("--fairness_target", type=float, default=0.5)
parser.add_argument("--fairness_tolerance", type=float, default=0.0)
parser.add_argument("--fairness_adaptive", action='store_true')
parser.add_argument("--fairness_weight_window", type=int, default=5)
parser.add_argument("--fairness_weight_algo_start_eval", type=float, default=5.0)
parser.add_argument("--fairness_target_eval", type=float, default=0.4)
parser.add_argument("--fairness_adaptive_eval", action='store_true')
parser.add_argument("--fairness_weight_window_eval", type=int, default=5)
parser.add_argument("--user_dropout_prob", type=float, default=0.0)
parser.add_argument("--null_utility", type=float, default=1.25)
parser.add_argument("--max_episode_length", type=int, default=20)
parser.add_argument("--num_candidate_items", type=int, default=10)
parser.add_argument("--num_clusters", type=int, default=2)
parser.add_argument("--num_trials", type=int, default=25)
parser.add_argument("--discrimination_target", type=float)
args = parser.parse_args()

if __name__ == '__main__':
    if args.method == 'random':
        main_random(args)
    elif args.method in ['reward_shaping', 'q_bonus']:
        # For reward shaping, set args.fairness_weight_algo=0, and args.fairness_weight_algo >= 0
        main_q_bonus(args)
    elif args.method == 'reward_dec':
        main_reward_dec(args)
    elif args.method == 'safe':
        main_safe(args)
    else:
        raise Exception("Method not in list.")
    end = True
