"""Launch sequential SearchLibrium runs; default: one seed, 50 actual fits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101])
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--train', type=Path, default=SCRIPT_DIR.parent / 'dataset/IS_swissmetro.csv')
    parser.add_argument('--test', type=Path, default=SCRIPT_DIR.parent / 'dataset/OOS_swissmetro.csv')
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'test_50')
    parser.add_argument('--initial-temperature', type=float, default=1000.)
    parser.add_argument('--final-temperature', type=float, default=.001)
    parser.add_argument('--temperature-steps', type=int, default=100)
    parser.add_argument('--neighbours', type=int, default=20,
                        help='Proposals per temperature, not number of estimations')
    parser.add_argument('--dry-run', action='store_true', help='Validate and archive inputs, with zero fits')
    args = parser.parse_args()
    if args.budget < 1 or args.temperature_steps < 2 or args.neighbours < 1:
        parser.error('Budget/neighbours must be positive; temperature steps must be >= 2')
    if not (0 < args.final_temperature < args.initial_temperature < float('inf')):
        parser.error('Require finite initial temperature > final temperature > 0')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be distinct integers between 0 and 2**32-1')
    args.train, args.test, args.output = (p.expanduser().resolve() for p in (args.train, args.test, args.output))
    if args.seed is not None:
        from sa_benchmark_controlled import run_seed
        run_seed(args)
        return
    # Each seed gets its own interpreter, random state, package cache, and log.
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    for seed in args.seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(seed),
                   MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR', str(args.output / 'mpl_cache')))
        command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(seed),
                   '--budget', str(args.budget), '--train', str(args.train), '--test', str(args.test),
                   '--output', str(args.output), '--initial-temperature', str(args.initial_temperature),
                   '--final-temperature', str(args.final_temperature),
                   '--temperature-steps', str(args.temperature_steps), '--neighbours', str(args.neighbours)]
        if args.dry_run:
            command.append('--dry-run')
        with (args.output / f'seed_{seed}.log').open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(seed)
        print(f'Seed {seed}: {"FAILED" if result.returncode else "finished"}', flush=True)
        # Summarise completed seeds only; failed seeds remain visible in batch status.
        import pandas as pd
        summaries = [json.loads(p.read_text()) for p in sorted(args.output.glob('seed_*/summary.json'))]
        pd.DataFrame(summaries).to_csv(args.output / 'summary.csv', index=False)
    from utils import write_json
    write_json(args.output / 'batch_status.json', {'requested_seeds': args.seeds, 'failed_seeds': failures})
    if failures:
        raise SystemExit(f'Failed seeds: {failures}; inspect their logs')


if __name__ == '__main__':
    main()
