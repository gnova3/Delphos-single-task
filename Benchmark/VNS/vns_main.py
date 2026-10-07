"""
VNS Benchmark: This file separates data, catalogue, search, stopping, and reporting. 
"""

from vns_benchmark_controlled import run_seed
from utils import write_json
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent



def main() -> None:
    """Default to one 50-fit test; launch the full batch only by explicit CLI choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=50, help='Actual fits per seed, including initial fit')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101])
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--search-attempts', type=int, default=100000,
                        help='Secondary native VNS safety cap, NOT estimation budget')
    parser.add_argument('--train', type=Path, default=SCRIPT_DIR.parent / 'dataset/IS_swissmetro.csv')
    parser.add_argument('--test', type=Path, default=SCRIPT_DIR.parent / 'dataset/OOS_swissmetro.csv')
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'test_50')
    args = parser.parse_args()
    if args.budget < 1 or args.search_attempts < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Budgets must be positive and seeds unique')
    args.train, args.test, args.output = (p.resolve() for p in (args.train, args.test, args.output))
    if args.seed is not None:
        run_seed(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    for seed in args.seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(seed), MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR', str(args.output / 'mpl_cache')))
        command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(seed),
                   '--budget', str(args.budget), '--search-attempts', str(args.search_attempts),
                   '--train', str(args.train), '--test', str(args.test), '--output', str(args.output)]
        with (args.output / f'seed_{seed}.log').open('w') as log_file:
            completed = subprocess.run(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        if completed.returncode:
            failures.append(seed)
        print(f'Seed {seed}: {"FAILED" if completed.returncode else "finished"}', flush=True)
        summaries = [json.loads(p.read_text()) for p in sorted(args.output.glob('seed_*/summary.json'))]
        pd.DataFrame(summaries).to_csv(args.output / 'summary.csv', index=False)
    write_json(args.output / 'batch_status.json', {'requested_seeds': args.seeds, 'failed_seeds': failures})
    if failures:
        raise SystemExit(f'Failed seeds: {failures}; inspect individual logs')

if __name__ == '__main__':
    main()
