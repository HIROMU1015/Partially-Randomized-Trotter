"""R-aware metadata and saved-only C routing; no price model evaluation.

Reuse B identity/schedules and fixed H4/H6 providers. A new R is unacquired;
fixed-R coefficients are never rescaled. B owns menu selection and G.
"""
from copy import deepcopy
import hashlib
import json


def describe_request(task, row, axis, *, pipeline):
    """Describe a validated row; allocation/event counts are recipe metadata.

    join_key binds the full B identity plus axis, not a trajectory cache key.
    Actual trajectory reuse still requires the existing wrapper fingerprint.
    """
    p = pipeline
    p.validate_task(task)
    if axis not in p.AXES or not any(p.same(row, r) for r in task['candidates']):
        raise ValueError('PRICE_ROW_OR_AXIS_OUTSIDE_TASK')
    c = row['candidate']; system = c['system']; arm = p.arm(c)
    if system == 'H4':
        recipe = c['formula'] == '2nd' and c['delta'] == .1 and (
            (arm == 'PR' and c['prefix'] == 1 and c['K'] == 2) or
            (c['method'] == 'B0' and c['prefix'] == 7))
        anchor, minimum = 4, 1
    elif system == 'H6':
        recipe = c['formula'] == '4th(m5_best)' and c['delta'] == 1.6 and (
            (arm == 'PR' and c['prefix'] == 7 and c['K'] == 2) or
            (c['method'] == 'FULL' and c['prefix'] == 19))
        anchor, minimum = 16, 11
    else:
        raise ValueError('H4_DF10_OR_H6_DF19_PRICE_DESIGN_ONLY')
    if not recipe:
        raise ValueError('OUTSIDE_R_VARIABLE_PRICE_DESIGN_RECIPE')
    R = c['R_outer']
    if arm == 'PR':
        if type(R) is not int or R < minimum:
            raise ValueError('R_OUTER_BELOW_SIGNED_PIECE_MINIMUM')
    elif R is not None or c['K'] is not None:
        raise ValueError('PF_HAS_NO_R_OUTER_OR_K')
    identity = p.identity(task, row)
    join_payload = dict(identity=identity, axis=axis)
    join_key = hashlib.sha256(json.dumps(join_payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(identity=identity, axis=axis, join_key=join_key,
        T=c['T'], q_PF=c['q'], delta=c['delta'], R_outer=R,
        q_RPE=row['q_RPE'], allocation=deepcopy(row['structure']['allocation']),
        event_count=row['structure']['event_count'],
        allocation_policy=row['structure']['allocation_policy'],
        minimum_legal_R_outer=minimum if arm == 'PR' else None,
        fixed_R_model_anchor=anchor if arm == 'PR' else None,
        R_evidence_status=('PF_R_NOT_APPLICABLE' if arm == 'PF' else
            'FIXED_R_SAVED_LOOKUP_ONLY' if R == anchor else 'NEW_R_CALIBRATION_REQUIRED'),
        tau_definition='lambda_tail * delta * signed_weight[j] / allocation[j]',
        price_law='declared_finite_K2_law_plus_explicit_sampler_binding_required' if arm == 'PR' else 'deterministic',
        law_probabilities_evaluated=False, numerical_model_evaluated=False,
        R_systematic_uncertainty=None, full_RPE_price_adopted=False)


class RVariableSavedPriceProvider:
    """c_provider(task, requested_rows), using B's existing fixed providers.

    No new R price, coefficient interpolation, fallback or G entry. The delegate
    retains its domain, cohorts, nulls, covariance and qualification. A future
    calibrated R model is a separate scientific implementation, not installed.
    """
    def __init__(self, read_json, *, pipeline, selections=None):
        self.read_json = read_json
        self.pipeline = pipeline
        self.selections = deepcopy(selections or {})
        self._providers = {}

    def __call__(self, task, requested_rows):
        p = self.pipeline
        descriptions = {(r['candidate']['id'], a): describe_request(task, r, a, pipeline=p)
            for r in requested_rows for a in p.AXES}
        if not descriptions:
            return []
        system = task['binding']['system']
        if system not in self._providers:
            factory = p.h4_saved_price_provider if system == 'H4' else p.h6_saved_price_provider
            self._providers[system] = factory(self.read_json, selections=self.selections)
        records = deepcopy(self._providers[system](task, requested_rows))
        for record in records:
            key = record['identity']['candidate']['id'], record['axis']
            description = descriptions[key]
            if not p.same(record['identity'], description['identity']):
                raise ValueError('FIXED_PROVIDER_RETURNED_FOREIGN_R_OR_BINDING')
            # Fixed providers reject new R. Never let a different R price slip in.
            if (description['R_evidence_status'] == 'NEW_R_CALIBRATION_REQUIRED'
                    and record.get('mean', {}).get('rz_count') is not None):
                raise ValueError('FIXED_R_PRICE_CANNOT_BE_TRANSPLANTED')
            record['R_price_scope'] = description
        return records
