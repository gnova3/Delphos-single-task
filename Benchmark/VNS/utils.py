import hashlib
from typing import Any
from pathlib import Path
import tomllib
import pandas as pd
import numpy as np
import json

# Use the same declared threshold as the benchmark's Biogeme settings.
with (Path(__file__).resolve().parent / 'biogeme.toml').open('rb') as stream:
    IDENTIFICATION_THRESHOLD = float(tomllib.load(stream)['Output']['identification_threshold'])

def sha256(path: Path) -> str:
    """Return a stable fingerprint of an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_toml(TOML_FILE : Path) -> dict[str, Any]:
    with TOML_FILE.open("rb") as stream:
        return tomllib.load(stream)

def write_json(path: Path, value: Any) -> None:
    """Write strict JSON so missing diagnostics cannot silently become NaN."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def audit_data(train_path: Path, test_path: Path) -> tuple:
    """Validate fixed respondent-disjoint files; never use holdout to fit/select."""
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    required = {'ID', 'IS', 'choice', 'avail_1', 'avail_2', 'avail_3',
                'x_1_1', 'x_1_2', 'x_1_3', 'x_2_1', 'x_2_2', 'x_2_3',
                'x_2_4', 'x_3_1', 'x_3_2', 'ga', 'luggage', 'gender',
                'who', 'income', 'age', 'class'}
    for label, frame in [('IS', train), ('OOS', test)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f'{label}: missing columns {sorted(missing)}')
    if set(train.columns) != set(test.columns):
        raise ValueError('IS/OOS columns differ')
    for label, frame, flag in [('IS', train, 1), ('OOS', test, 0)]:
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: empty or non-finite data')
        if not frame['IS'].eq(flag).all():
            raise ValueError(f'{label}: incorrect split flag')
        if not frame['x_2_4'].isin([0, 1]).all():
            raise ValueError(f'{label}: seat must be binary')
        continuous = ['x_1_1', 'x_1_2', 'x_1_3', 'x_2_1', 'x_2_2',
                      'x_2_3', 'x_3_1', 'x_3_2']
        if (frame[continuous] < 0).any().any():
            raise ValueError(f'{label}: negative attribute outside transformation domain')
        if not frame.choice.isin([1, 2, 3]).all():
            raise ValueError(f'{label}: invalid choice')
        for alt in (1, 2, 3):
            if not frame[f'avail_{alt}'].isin([0, 1]).all():
                raise ValueError(f'{label}: invalid availability')
            if not frame.loc[frame.choice.eq(alt), f'avail_{alt}'].eq(1).all():
                raise ValueError(f'{label}: chosen alternative unavailable')
        # Apply identical domain checks to OOS without deriving its catalogue.
        for col in ['ga', 'luggage', 'gender', 'who', 'income', 'age', 'class']:
            if not set(frame[col]).issubset(set(train[col])):
                raise ValueError(f'{label}: unseen category in {col}')
    if set(train.ID) & set(test.ID):
        raise ValueError('Respondents overlap between IS and OOS')
    ratio = len(train) / (len(train) + len(test))
    if abs(ratio - .8) > .01:
        raise ValueError(f'Expected approximately 80% IS, found {ratio}')
    return train, test, {
        'IS_rows': len(train), 'OOS_rows': len(test), 'IS_fraction': ratio,
        'IS_respondents': int(train.ID.nunique()),
        'OOS_respondents': int(test.ID.nunique()), 'overlapping_respondents': 0,
        'IS_sha256': sha256(train_path), 'OOS_sha256': sha256(test_path),
        'IS_file': str(train_path), 'OOS_file': str(test_path),
    }

#######
# Sign and validty check
##

def _sign_check(beta_values: dict[str, float], prefix: str, expected_sign: int, tolerance: float = 1.0e-7,) -> tuple[bool, str | None]:
    
    relevant = {name: float(value) for name, value in beta_values.items() if name.startswith(prefix)}

    def wrong_sign(value: float) -> bool:
        if expected_sign < 0:
            return value > tolerance
        return value < -tolerance

    reference_names = [name for name in relevant if name.endswith("_ref")]
    handled: set[str] = set()

    for reference_name in reference_names:
        reference_value = relevant[reference_name]
        stem = reference_name[: -len("_ref")]
        handled.add(reference_name)
        if wrong_sign(reference_value):
            return False, f"Behavioural sign violation: {reference_name}={reference_value:g}."

        difference_prefix = f"{stem}_diff_"
        for difference_name, difference_value in relevant.items():
            if difference_name.startswith(difference_prefix):
                handled.add(difference_name)
                category_total = reference_value + difference_value
                if wrong_sign(category_total):
                    return False, f"Behavioural sign violation: {reference_name} + {difference_name} = {category_total:g}."

    for name, value in relevant.items():
        if name in handled or "_diff_" in name:
            continue
        if wrong_sign(value):
            return False, f"Behavioural sign violation: {name}={value:g}."

    return True, None

def validity(results: Any) -> tuple[bool, str | None]:
    """Reject non-converged, unidentified, non-finite, or implausible models."""
    if not np.isfinite(float(results.final_loglikelihood)):
        return False, "Non-finite final log likelihood."
    if not getattr(results, "algorithm_has_converged", False):
        return False, "The optimisation algorithm did not converge."

    try:
        gradient_norm = float(results.gradient_norm)
    except Exception as exc:
        return False, f"The final gradient is unavailable: {exc}"
    if not np.isfinite(gradient_norm):
        return False, "The final gradient norm is not finite."

    try:
        eigenvalues, _ = results.eigen_structure()
        eigenvalues = np.asarray(eigenvalues, dtype=float)
    except Exception as exc:
        return False, f"The Hessian identification diagnostic is unavailable: {exc}"

    if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
        return False, "The Hessian eigenvalues are missing or non-finite."
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue <= IDENTIFICATION_THRESHOLD:
        return False, (
            f"Smallest eigenvalue {minimum_eigenvalue:g} is not above "
            f"{IDENTIFICATION_THRESHOLD:g}."
        )

    try:
        beta_values = results.get_beta_values()
    except Exception as exc:
        return False, f"Estimated coefficients are unavailable: {exc}"

    if not all(np.isfinite(float(value)) for value in beta_values.values()):
        return False, "At least one estimated coefficient is non-finite."

    if False:
        for prefix in ("b_time", "b_cost", "b_headway"):
            valid, reason = _sign_check(beta_values, prefix, expected_sign=-1)
            if not valid:
                return valid, reason

        # A value of one denotes airline-style seating in the Swissmetro data.
        valid, reason = _sign_check(beta_values, "b_seat", expected_sign=1)
        if not valid:
            return valid, reason

    return True, None
