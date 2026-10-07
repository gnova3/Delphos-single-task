"""Per-run result cache and durable ledger of actual Apollo fitting attempts."""
import json
import math
import sqlite3
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


def json_value(value):
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def successful(row):
    value = row.get('successfulEstimation', False)
    return str(value).strip().lower() in ('true', '1', '1.0')


class ResultStore:
    """A single writer per run; no cross-seed or implicit legacy cache reuse.

    `attempt` is assigned only by the callback immediately before Apollo fitting.
    Preparation failures are cached with no attempt. Started rows survive a crash.
    Existing stores are refused: this class does not resume DQN training.
    """
    def __init__(self, path, budget):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('estimation_budget must be a non-negative integer')
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'Use a fresh run directory: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE models (specification TEXT PRIMARY KEY, attempt INTEGER UNIQUE, status TEXT NOT NULL, results TEXT NOT NULL)')
        self.db.commit()
        self.budget = budget
        self.attempts = 0
        self.cache = {}

    def _write(self, name, attempt, status, row):
        row = dict(row, specification=name, attempt=attempt, status=status)
        self.db.execute('INSERT OR REPLACE INTO models VALUES (?, ?, ?, ?)',
                        (name, attempt, status, json.dumps(row, allow_nan=False)))
        self.db.commit()

    def evaluate(self, name, estimator):
        if name in self.cache:
            return dict(self.cache[name]), True
        if self.attempts >= self.budget:
            raise BudgetExhausted('Estimation budget reached')
        attempt = None

        def before_estimate():
            nonlocal attempt
            if attempt is not None:
                raise RuntimeError('One fitting call is allowed per specification')
            if self.attempts >= self.budget:
                raise BudgetExhausted('Estimation budget reached')
            self.attempts += 1
            attempt = self.attempts
            self._write(name, attempt, 'started', {'specification': name})

        try:
            row = {k: json_value(v) for k, v in estimator(name, before_estimate).items()}
            if attempt is None:
                raise RuntimeError('Estimator returned without calling before_estimate')
            status = 'success' if successful(row) else 'failed'
        except BudgetExhausted:
            raise
        except Exception as error:
            row = {'successfulEstimation': False, 'error': f'{type(error).__name__}: {error}'}
            status = 'error' if attempt is not None else 'preparation_error'
        row.update(specification=name, attempt=attempt, status=status)
        self._write(name, attempt, status, row)
        self.cache[name] = row
        return dict(row), False

    def export_csv(self, path):
        import pandas as pd
        rows = [json.loads(row[0]) for row in self.db.execute('SELECT results FROM models ORDER BY rowid')]
        pd.DataFrame(rows, columns=None if rows else ['specification', 'attempt', 'status']).to_csv(path, index=False)

    def close(self):
        self.db.close()
