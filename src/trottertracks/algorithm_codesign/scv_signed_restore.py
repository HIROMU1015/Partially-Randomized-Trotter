"""Typed restoration of a hash-pinned saved SignedControl; no moment sums.

The caller supplies immutable bytes and a fixed provenance binding. Import is
passive. Only the known fields of the frozen kernel class are restored.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import json
import re

from . import scv_signed_support as signed
from . import scv_scalar as scalar

SCHEMA = 'scv_signed_support_h4_metadata_result_v1'


@dataclass(frozen=True)
class Binding:
    result_commit: str
    result_path: str
    result_sha256: str
    metadata_identity: str
    control_identity: str
    law_sha256: str
    catalog_sha256: str
    hamiltonian_hash: str
    preparation_hash: str
    a: str
    b: str
    system_qubits: int


def fraction(value):
    if not isinstance(value, str) or not re.fullmatch(r'-?\d+(?:/[1-9]\d*)?', value):
        raise ValueError('canonical exact Fraction string required')
    result = F(value)
    if str(result) != value:
        raise ValueError('noncanonical Fraction')
    return result


def restore_saved_control(result_bytes, law_bytes, catalog_bytes, binding, *, commit, path):
    if not isinstance(binding, Binding) or (commit, path) != (binding.result_commit, binding.result_path):
        raise ValueError('saved result commit/path mismatch')
    if type(result_bytes) is not bytes or scalar.digest(result_bytes) != binding.result_sha256:
        raise ValueError('saved result byte hash mismatch')
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:raise ValueError('duplicate JSON field')
            d[k] = v
        return d
    result = json.loads(result_bytes, object_pairs_hook=unique)
    if (result.get('schema') != SCHEMA or result.get('feature_version') != signed.FEATURE
        or result.get('metadata_identity') != binding.metadata_identity
        or result.get('control_identity') != binding.control_identity
        or result.get('metadata_structural_eligibility') != 'ELIGIBLE'
        or result.get('h1_nonconstant') is not True
        or result.get('feature_bounds_exact') != ['-1', '1']
        or any(result.get(k) is not False for k in
               ('engine_loaded', 'state_loaded', 'old_measurement_data_used', 'next_stage_started', 'science_ready'))):
        raise ValueError('saved schema/feature/identity/eligibility mismatch')
    md = signed.metadata_from_bytes(law_bytes, catalog_bytes,
        system_qubits=binding.system_qubits, law_sha256=binding.law_sha256,
        catalog_sha256=binding.catalog_sha256, hamiltonian_hash=binding.hamiltonian_hash,
        preparation_hash=binding.preparation_hash, a=binding.a, b=binding.b)
    if md.identity != binding.metadata_identity:raise ValueError('saved metadata identity mismatch')
    cells = result.get('cells')
    if not isinstance(cells, list) or len(cells) != 5:raise ValueError('five saved cells required')
    means, second, variances, masses = [], [], [], []
    for i, row in enumerate(cells):
        if (set(row) != {'cell','pattern','mass_exact','mean_exact','second_moment_exact','variance_exact'}
            or row['cell'] != str(i) or row['pattern'] != list(md.sampler.patterns[i])):
            raise ValueError('saved cell schema/order/pattern mismatch')
        p, m, s, v = (fraction(row[k]) for k in
            ('mass_exact', 'mean_exact', 'second_moment_exact', 'variance_exact'))
        if p <= 0 or not -1 <= m <= 1 or not 0 <= s <= 1 or v < 0 or v != s-m*m:
            raise ValueError('inconsistent saved moment algebra')
        masses.append(p);means.append(m);second.append(s);variances.append(v)
    # Existing stratum binding only. Never sum signed label features/moments.
    if tuple(masses) != tuple(p for _,p in md.sampler.cells('B1-OL')) or sum(masses) != 1 or variances[1] <= 0:
        raise ValueError('saved masses/positive h1 variance mismatch')
    identity = scalar.fingerprint(dict(metadata=md.identity, feature=signed.FEATURE,
        means=tuple(means), second_moments=tuple(second), feature_bounds=(-1,1)))
    if identity != binding.control_identity:raise ValueError('saved control fingerprint mismatch')
    control = object.__new__(signed.SignedControl)
    for name, value in (('metadata',md), ('means',tuple(means)), ('second_moments',tuple(second)),
                        ('variances',tuple(variances)), ('identity',identity)):
        object.__setattr__(control, name, value)
    # Compare the entire known snapshot, not a permissive subset.
    normalized = json.loads(json.dumps(control.snapshot(), default=str))
    if normalized != result.get('snapshot'):raise ValueError('saved snapshot mismatch')
    return control, tuple(masses)
