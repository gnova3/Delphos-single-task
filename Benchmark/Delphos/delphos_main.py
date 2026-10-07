"""NEW entry point replacing analysis.py; originals are not imported or modified."""
import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from delphos_new import DQNLearner

ROOT = Path(__file__).resolve().parent
B = 500
SEEDS = list(range(101, 111))
N_OBSERVATIONS = 5409
ATTRIBUTES = {1: {1, 2, 3}, 2: {1, 2, 3, 4}, 3: {1, 2}}
COVARIATES = {'age': [1, 2, 3, 4, 5], 'income': [1, 2, 3, 4],
              'class': [0, 1], 'ga': [0, 1], 'gender': [0, 1]}
STATE_SPACE = {'num_vars': 5, 'transformations': ['linear', 'log', 'box-cox'],
               'taste': ['generic', 'specific'], 'covariates': list(COVARIATES)}

cfg = {
    "depth": 2,
    "hidden_units": 256,
    "learning_rate": 1e-4,
    "discount_factor": 0.99,
    "total_episodes": 10_000,
    "buffer_size": 5_000,
    "target_update_freq": 5,
    "config_id": 1,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=B)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--dataset', type=Path, default=ROOT.parent / 'dataset' / 'swissmetro_process.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments_new' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--max-episodes', type=int, default=10_000, help='Safety cap; reaching it before the budget is reported as incomplete')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    if args.budget < 0 or args.max_episodes <= 0:
        parser.error('budget must be non-negative and max-episodes must be positive')
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
                estimation_budget=args.budget,
                max_episodes=cfg["total_episodes"],
                learning_rate=cfg["learning_rate"],
                discount_factor=cfg["discount_factor"],
                buffer_size=cfg["buffer_size"],
                target_update_freq=cfg["target_update_freq"],
                hidden_layers=[cfg["hidden_units"]] * cfg["depth"],
                reward_weights={'AIC': 1.0}, 
                n_observations=N_OBSERVATIONS
            )

        agent.config['seed'] = seed
        (agent.subfolder / 'config_new.json').write_text(json.dumps(agent.config, indent=2))
        summary = agent.train()
        if args.plots:
            from reporting_new import plot_training
            plot_training(agent)
        if summary['stop_reason'] != 'estimation_budget_reached':
            logging.warning('Seed %d stopped before its fit budget: %s', seed, summary['stop_reason'])


if __name__ == '__main__':
    main()
