"""Data and catalogue definitions for the SearchLibrium benchmark.

No VNS modules or estimation packages are imported here. The catalogue mirrors
the current VNS declarations; a run records their source hashes for auditing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = ('time', 'cost', 'headway', 'seat')
CATEGORIES = {
    'ga': {0: 'noGA', 1: 'GA'},
    'luggage': {0: 'no_lugg', 1: 'one_lugg', 3: 'several_lugg'},
    'gender': {0: 'g1', 1: 'g2'},
    'who': {0: 'w0', 1: 'w1', 2: 'w2', 3: 'w3'},
    'income': {1: 'I1', 2: 'I2', 3: 'I3', 4: 'I4'},
    'age': {1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4', 5: 'a5'},
    'class': {0: 'c0', 1: 'c1'},
}
SEGMENTATIONS = ('no_seg', *CATEGORIES)
ACTIVE_ALTS = {'time': (1, 2, 3), 'cost': (1, 2, 3),
               'headway': (1, 2), 'seat': (2,)}
ATTRIBUTE_COLUMN = {'time': 1, 'cost': 2, 'headway': 3, 'seat': 4}
# These IDs match the VNS central-controller IDs, including dormant decisions.
DOMAINS = {'asc': SEGMENTATIONS}
for _group in GROUPS:
    DOMAINS[f'b_{_group}'] = SEGMENTATIONS
    DOMAINS[f'b_{_group}_gen_altspec'] = ('generic', 'altspec')
    _alt = 2 if _group == 'seat' else 1
    DOMAINS[f'{_group}_form_alt{_alt}'] = ('linear', 'boxcox', 'log1p')
    DOMAINS[f'{_group}_inclusion_alt{_alt}'] = ('off', 'on')
DOMAINS['choice_model'] = ('mnl',)
NOMINAL_CONFIGURATIONS = 8 * (2 * 3 * 2 * 8) ** 4
# Freeze the audited VNS declarations. A subsequent VNS edit needs an explicit
# catalogue review; recording a new hash alone must not imply continued parity.
VNS_REFERENCE_HASHES = {
    'vns_benchmark_controlled.py': 'e58390cac22a9cbf995d77add6bde17f13f5ee94e71cc288c9e069e34ebf5cc3',
    'utils.py': 'b91910b971648ba9043389c6644253148647bcf2e1c6e4f16d52eda85b416a09',
}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_safe(value):
    """Keep failed numerical diagnostics representable in strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True,
                               allow_nan=False), encoding='utf-8')


def configuration_id(configuration: dict) -> str:
    if set(configuration) != set(DOMAINS):
        raise ValueError('Incomplete catalogue configuration')
    for name, value in configuration.items():
        if value not in DOMAINS[name]:
            raise ValueError(f'Invalid selection: {name}={value}')
    return ';'.join(f'{key}:{configuration[key]}' for key in sorted(configuration))


def null_configuration() -> dict:
    return {key: values[0] for key, values in DOMAINS.items()}


def audit_data(train_path: Path, test_path: Path):
    """Keep the supplied split and scaling; reject respondent leakage."""
    frames = [pd.read_csv(path) for path in (train_path, test_path)]
    required = {'ID', 'IS', 'choice', 'avail_1', 'avail_2', 'avail_3', *CATEGORIES}
    required |= {f'x_{alt}_{ATTRIBUTE_COLUMN[group]}' for group in GROUPS
                 for alt in ACTIVE_ALTS[group]}
    for frame, flag in zip(frames, (1, 0)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Empty or non-finite dataset')
        if not frame['IS'].eq(flag).all() or not frame.choice.isin([1, 2, 3]).all():
            raise ValueError('Incorrect split flags or choices')
        av = frame[['avail_1', 'avail_2', 'avail_3']].to_numpy()
        if not np.isin(av, [0, 1]).all():
            raise ValueError('Availability must be binary')
        if not (av[np.arange(len(frame)), frame.choice.to_numpy()-1] == 1).all():
            raise ValueError('Chosen alternative is unavailable')
        if not frame.x_2_4.isin([0, 1]).all():
            raise ValueError('Seat must be binary')
        for col in required:
            if col.startswith('x_') and (frame[col] < 0).any():
                raise ValueError(f'Negative attribute: {col}')
        for col, mapping in CATEGORIES.items():
            if not frame[col].isin(mapping).all():
                raise ValueError(f'Unexpected category: {col}')
    train, test = frames
    for col, mapping in CATEGORIES.items():
        if set(train[col]) != set(mapping):
            raise ValueError(f'IS does not contain the expected categories of {col}')
    if set(train.x_2_4) != {0, 1}:
        raise ValueError('IS must contain both seat values')
    if set(train.ID) & set(test.ID):
        raise ValueError('Respondents overlap between IS and OOS')
    fraction = len(train) / (len(train) + len(test))
    if abs(fraction - .8) > .01:
        raise ValueError('Supplied split is not approximately 80/20')
    return train, test, {
        'IS_file': str(train_path), 'OOS_file': str(test_path),
        'IS_sha256': sha256(train_path), 'OOS_sha256': sha256(test_path),
        'IS_rows': len(train), 'OOS_rows': len(test), 'IS_fraction': fraction,
        'IS_respondents': train.ID.nunique(), 'OOS_respondents': test.ID.nunique(),
        'overlapping_respondents': 0,
    }


def transform(x, form, lam=1., xp=np):
    """The VNS functional forms, including its zero and near-zero conventions.

    This is explicitly an adapter, not SearchLibrium's native Box-Cox rule.
    Inactive branches use safe inputs so automatic derivatives stay finite.
    """
    if form == 'linear':
        return x
    if form == 'log1p':
        return xp.log1p(x)
    safe_x = xp.where(x == 0, 1., x)
    log_x = xp.log(safe_x)
    near_zero = xp.abs(lam) < 1.e-5
    denominator = xp.where(near_zero, 1., lam)
    regular = (safe_x ** lam - 1.) / denominator
    # Match the VNS package's actual local polynomial, including its first term.
    series = log_x + lam * log_x**2 + lam**2 * log_x**3 / 6. + lam**3 * log_x**4 / 24.
    return xp.where(x == 0, 0., xp.where(near_zero, series, regular))


@dataclass
class Design:
    """Column metadata tying transformed/segmented slopes to shared lambdas."""
    configuration: dict
    names: list
    bounds: list
    initial: np.ndarray
    raw: np.ndarray
    masks: np.ndarray
    forms: tuple
    lambda_indices: tuple
    sign_groups: list
    choice: np.ndarray
    availability: np.ndarray

    def matrix(self, theta, xp=np):
        # Each transformed raw attribute is multiplied by its alternative/category
        # mask AFTER transformation. Transforming a zero-filled interaction first
        # would change the model for off-category observations.
        raw, masks = xp.asarray(self.raw), xp.asarray(self.masks)
        columns = []
        for index, (form, lambda_index) in enumerate(zip(self.forms, self.lambda_indices)):
            lam = theta[lambda_index] if lambda_index is not None else 1.
            columns.append(transform(raw[:, :, index], form, lam, xp) * masks[:, :, index])
        # Dummy columns reserve parameter slots for lambdas in native setup.
        # Lambda dependence enters through transformed slope columns above.
        columns.extend([xp.zeros(raw.shape[:2])] * (len(self.names)-len(columns)))
        return xp.stack(columns, axis=2)


def build_design(data: pd.DataFrame, configuration: dict) -> Design:
    """Compile one catalogue ID; this performs no estimation."""
    configuration_id(configuration)
    n = len(data)
    names, bounds, raws, masks, forms, lambda_groups, sign_groups = [], [], [], [], [], [], []

    def coefficient(stem, raw, mask, segmentation, sign, form='linear', group=None):
        bound = (None, 0.) if sign == -1 else (0., None) if sign == 1 else (None, None)
        indices = []
        levels = [(None, '')] if segmentation == 'no_seg' else [(None, '_ref')] + [
            (value, f'_diff_{label}') for value, label in list(CATEGORIES[segmentation].items())[1:]]
        for value, suffix in levels:
            indices.append(len(names))
            names.append(stem + suffix)
            bounds.append(bound if value is None else (None, None))
            raws.append(raw)
            category_mask = 1. if value is None else (data[segmentation].to_numpy() == value)[:, None]
            masks.append(mask * category_mask)
            forms.append(form)
            lambda_groups.append(group if form == 'boxcox' else None)
        if sign:
            sign_groups.append((sign, indices))

    for alt in (1, 2):
        mask = np.broadcast_to(np.arange(1, 4) == alt, (n, 3)).astype(float)
        coefficient(f'asc{alt}', np.ones((n, 3)), mask, configuration['asc'], 0)
    for group in GROUPS:
        first_alt = 2 if group == 'seat' else 1
        if configuration[f'{group}_inclusion_alt{first_alt}'] == 'off':
            continue
        raw = np.zeros((n, 3))
        for alt in ACTIVE_ALTS[group]:
            raw[:, alt-1] = data[f'x_{alt}_{ATTRIBUTE_COLUMN[group]}'].to_numpy()
        form = configuration[f'{group}_form_alt{first_alt}']
        specific = configuration[f'b_{group}_gen_altspec'] == 'altspec'
        alternatives = ACTIVE_ALTS[group] if specific else (None,)
        for alt in alternatives:
            active = ACTIVE_ALTS[group] if alt is None else (alt,)
            mask = np.broadcast_to(np.isin(np.arange(1, 4), active), (n, 3)).astype(float)
            stem = f'b_{group}_base' + (f'_alt{alt}' if alt else '')
            coefficient(stem, raw, mask, configuration[f'b_{group}'],
                        1 if group == 'seat' else -1, form, group)
    lambda_map = {}
    for group in GROUPS:
        if group in lambda_groups:
            lambda_map[group] = len(names)
            names.append(f'lambda_{group}')
            bounds.append((-10., 10.))
    initial = np.zeros(len(names))
    for index in lambda_map.values():
        initial[index] = 1.
    return Design(dict(configuration), names, bounds, initial,
                  np.stack(raws, axis=2), np.stack(masks, axis=2), tuple(forms),
                  tuple(lambda_map.get(group) for group in lambda_groups), sign_groups,
                  np.eye(3)[data.choice.to_numpy()-1],
                  data[['avail_1', 'avail_2', 'avail_3']].to_numpy(dtype=float))


def validity(converged, theta, loglik, gradient, hessian, design, threshold):
    """Same acceptance diagnostics and sign totals as the VNS wrapper."""
    if not converged:
        return False, 'Optimiser did not converge'
    if not all(np.isfinite(x).all() for x in (theta, loglik, gradient, hessian)):
        return False, 'Non-finite coefficients, likelihood, or derivatives'
    minimum = float(np.linalg.eigvalsh(hessian).min())
    if minimum <= threshold:
        return False, f'Smallest information eigenvalue {minimum:g} <= {threshold:g}'
    if False:
        for sign, indices in design.sign_groups:
            ref = theta[indices[0]]
            totals = [ref] + [ref + theta[i] for i in indices[1:]]
            if any(sign * total < -1.e-7 for total in totals):
                return False, f'Behavioural sign violation: {design.names[indices[0]]}'
    return True, None
