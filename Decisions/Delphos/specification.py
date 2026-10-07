"""Model parameter selection and collision-free RL state encoding.

Variable 0 is the ASC; attributes occupy variables 1..N. Covariate codes
follow the configured dictionary order. The last alternative is the ASC base.
"""


def parameter_names(attributes, covariates):
    names = []
    for alt in sorted(attributes):
        names.append(f'asc_alt{alt}')
        names.extend(f'asc_alt{alt}_{cov}_{k}' for cov, cats in covariates.items() for k in cats)
    for attr in range(1, max(max(v) for v in attributes.values()) + 1):
        for prefix in [f'b_{attr}_generic'] + [f'b_{alt}_{attr}' for alt in sorted(attributes)]:
            for suffix in ['', '_log', '_box_cox']:
                names.append(prefix + suffix)
                names.extend(f'{prefix}{suffix}_{cov}_{k}' for cov, cats in covariates.items() for k in cats)
        names.append(f'L_{attr}')
    return names


def create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates, info=False):
    """Return the complement of the selected coefficients in the parameter universe."""
    n = max(max(v) for v in attributes.values()) + 1
    if any(len(v) != n for v in (state_0, specific_0, covariates_0)):
        raise ValueError(f'Specification vectors must have length {n}')
    if sorted(attributes) != list(range(1, max(attributes) + 1)):
        raise ValueError('Alternative IDs must be contiguous and start at 1')
    cov_names = list(covariates)
    selected = set()
    for var, (trans, specific, cov) in enumerate(zip(state_0, specific_0, covariates_0)):
        if trans not in (0, 1, 2, 3) or specific not in (0, 1) or not 0 <= cov <= len(cov_names):
            raise ValueError('Invalid transformation, taste or covariate code')
        if var == 0 and (trans not in (0, 1) or specific != 0):
            raise ValueError('ASC supports linear transformation and taste code 0')
        if trans == 0:
            continue
        if var == 0:
            prefixes = [f'asc_alt{alt}' for alt in sorted(attributes)[:-1]]
        else:
            prefixes = ([f'b_{alt}_{var}' for alt in sorted(attributes) if var in attributes[alt]]
                        if specific else [f'b_{var}_generic'])
            suffix = ['', '', '_log', '_box_cox'][trans]
            prefixes = [p + suffix for p in prefixes]
            if trans == 3:
                selected.add(f'L_{var}')
        if cov:
            cov_name = cov_names[cov - 1]
            selected.update(f'{p}_{cov_name}_{k}' for p in prefixes for k in covariates[cov_name])
        else:
            selected.update(prefixes)
    fixed = [p for p in parameter_names(attributes, covariates) if p not in selected]
    if info:
        print('Estimated parameters:', sorted(selected))
    return fixed


def create_apollo_non_fixed(beta_fixed, attributes, covariates, info=False):
    fixed = set(beta_fixed)
    return [p for p in parameter_names(attributes, covariates) if p not in fixed]


class StateManager:
    """Use one (variable, transformation, taste, covariate) tuple everywhere."""
    transformation_codes = {0: 'none', 1: 'linear', 2: 'log', 3: 'box-cox'}
    inverse_mapping = {v: k for k, v in transformation_codes.items()}

    def __init__(self, state_space_params):
        self.state_space_params = dict(state_space_params)
        self.num_vars = state_space_params['num_vars']
        self.cov_count = len(state_space_params.get('covariates', [])) + 1
        self.tastes = [0, 1] if 'specific' in state_space_params.get('taste', ['generic']) else [0]
        self.transformations = state_space_params.get('transformations', ['linear', 'log', 'box-cox'])
        if self.num_vars < 2 or not self.transformations or any(t not in ('linear', 'log', 'box-cox') for t in self.transformations):
            raise ValueError('Provide at least one attribute and valid transformations')

    def get_state_length(self):
        return self.cov_count + (self.num_vars - 1) * 3 * len(self.tastes) * self.cov_count

    def encode_state_to_vector(self, state):
        import torch
        vector = torch.zeros(self.get_state_length())
        for var, trans, taste, cov in state:
            if var == 0:
                index = cov
            else:
                index = (self.cov_count + (var - 1) * 3 * len(self.tastes) * self.cov_count
                         + ((self.inverse_mapping[trans] - 1) * len(self.tastes) + taste) * self.cov_count + cov)
            vector[index] = 1
        return vector

    def encode_state_to_string(self, state):
        representation = ['000'] * self.num_vars
        for var, trans, taste, cov in state:
            representation[var] = f'{self.inverse_mapping[trans]}{taste}{cov}'
        return '_'.join(representation)

    def decode_string_to_specification(self, representation):
        codes = representation.split('_')
        if len(codes) != self.num_vars or any(len(c) < 3 or not c.isdigit() for c in codes):
            raise ValueError(f'Invalid specification: {representation}')
        return ([int(c[0]) for c in codes], [int(c[1]) for c in codes], [int(c[2:]) for c in codes])

    def decode_string_to_state(self, representation):
        trans, tastes, covs = self.decode_string_to_specification(representation)
        return [(i, self.transformation_codes[t], s, c)
                for i, (t, s, c) in enumerate(zip(trans, tastes, covs)) if t]

    def define_action_space(self):
        actions = [('terminate',)]
        for var in range(self.num_vars):
            for trans in (['linear'] if var == 0 else self.transformations):
                for taste in ([0] if var == 0 else self.tastes):
                    for cov in range(self.cov_count):
                        actions.extend((kind, var, trans, taste, cov) for kind in ('add', 'change'))
        return actions, {a: i for i, a in enumerate(actions)}

    def mask_invalid_actions(self, state, action_space):
        current = {entry[0]: tuple(entry) for entry in state}
        return [a for a in action_space if
                (a[0] == 'terminate' and bool(state)) or
                (a[0] == 'add' and a[1] not in current) or
                (a[0] == 'change' and a[1] in current and tuple(a[1:]) != current[a[1]])]
