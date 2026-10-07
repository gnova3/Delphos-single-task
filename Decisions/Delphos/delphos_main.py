"""NEW entry point replacing analysis.py; originals are not imported or modified."""
import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from delphos import DQNLearner

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(101, 111))
N_OBSERVATIONS = 10_019
K_MAX = 80
ATTRIBUTES = {1: {1,3}, 2: {1, 2, 3}, 3: {1, 2,3}, 4:{1,3}, 5:{1}, 6:{1}}
COVARIATES = {  'purpose':[1, 2, 3, 4, 5, 6, 7, 8],
                'city':[1,2,3],
                'income_hh':[1,2,3,4,5,6,7,8,9,10,11,12],
                'female':[0,1],
                'age':[2,3,4,5,6,7,8,9],
                'education':[1,2,3,4,5,6],
                'hh_size':[1,2,3,4,5],
                'british_origin':[0,1]}

STATE_SPACE = {'num_vars': 4, 'transformations': ['linear', 'log', 'box-cox'],
               'taste': ['generic', 'specific'], 'covariates': list(COVARIATES)}

reward_weights = {'LLout': 0.7, 'numParams':0.3}
behavioural_expectation = {1: '-', 3: '-'}
cfg = {
        "depth": 2,
        "hidden_units": 256,
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "total_episodes": 10_000,
        "buffer_size": 5_000,
        "target_update_freq": 5,
        "config_id": 1,
        "early_stop_window": 500,
        "early_stop_tolerance": 0.001,
        "patience": 100,
        "min_percentage": 0.2,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'dataset' / 'decisions_data.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--max-episodes', type=int, default=cfg["total_episodes"], help='Safety cap if reward convergence is not reached')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.max_episodes <= 0:
        parser.error('max-episodes must be positive')
    if not args.dataset.is_file():
        parser.error(f'Dataset does not exist: {args.dataset}')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for seed in args.seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
  
        agent = DQNLearner(
                dataset=args.dataset,
                output_dir=args.output / f"seed_{seed}",
                state_space_params=STATE_SPACE,
                attributes=ATTRIBUTES,
                covariates=COVARIATES,
                max_episodes=args.max_episodes,
                learning_rate=cfg["learning_rate"],
                discount_factor=cfg["discount_factor"],
                buffer_size=cfg["buffer_size"],
                target_update_freq=cfg["target_update_freq"],
                hidden_layers=[cfg["hidden_units"]] * cfg["depth"],
                reward_weights=reward_weights,
                behavioural_expectation=behavioural_expectation,
                n_observations=N_OBSERVATIONS,
                n_parameters_max=K_MAX,
                early_stop_window=cfg["early_stop_window"],
                early_stop_tolerance=cfg["early_stop_tolerance"],
                patience=cfg["patience"],
                min_percentage=cfg["min_percentage"],
            )

        agent.config['seed'] = seed
        (agent.subfolder / 'config.json').write_text(json.dumps(agent.config, indent=2))
        summary = agent.train()
        if args.plots:
            from reporting import plot_training
            plot_training(agent)
        if summary['stop_reason'] == 'episode_limit':
            logging.warning('Seed %d reached the episode limit before reward convergence.', seed)


if __name__ == '__main__':
    main()
