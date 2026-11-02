"""Saved-only F2 descriptive attribution; no circuit, law or sampling calls.

Reuse aggregate_metrics from the frozen F2 pipeline for independent total
reconciliation. Added work is classification and exact grouped accounting.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

from .f2_execution import aggregate_metrics
from .f2_plan import ARMS, LENGTHS, SAVED_INPUT_HASHES, case_id

METRICS = ('rz_count', 'cx_count', 'total_depth', 'rz_depth', 'cx_depth',
           'quantum_only_depth', 'circuit_size', 'measurement_count')
MECHANISMS = ('II', 'IF', 'IP', 'FF_same', 'FF_different', 'FP', 'PP')
PUBLIC_HASHES = {
    'CASE_RESOURCES.csv.gz': 'eb104a86ceef0b22ef7adf9d912d4201df760eba4810bff4415b62c48dc5240d',
    'F2_laws.json': '45a270811323975942c39ea7178864e7e33de6f75e504e4c0f09c47d31be8fe8',
    'RESULT_SUMMARY.json': '00a49c90972bc9089bc444af49e6be474677d4f8c2ab28247d1dbdf0e8255aec',
}


def checked_bytes(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Saved input hash mismatch: ' + Path(path).name)
    return data


def atom_features(atom):
    """Copy structural metadata only; coefficients/operators are not evaluated."""
    kind = atom['kind']
    word = atom['word']
    support = [i for i, ch in enumerate(word) if ch != 'I']
    if kind not in ('IDENTITY', 'FRAME_Z', 'PHYSICAL_PAULI') or len(word) != 4:
        raise ValueError('Unknown atom kind or mode count')
    if any(ch not in 'IXYZ' for ch in word) or (kind == 'IDENTITY') != (not support):
        raise ValueError('Atom kind/word mismatch')
    if kind == 'FRAME_Z' and (any(ch not in 'IZ' for ch in word)
            or len(support) not in (1, 2) or atom['frame_id'] not in ('F0', 'F1')):
        raise ValueError('Invalid frame metadata')
    origins = sorted(atom['origins'])
    if not origins or len(set(origins)) != len(origins) or set(origins) - {'F0', 'F1'}:
        raise ValueError('Invalid fragment origins')
    basis = atom['basis_key']
    if kind == 'IDENTITY' and basis is not None or kind != 'IDENTITY' and not basis:
        raise ValueError('Missing or inconsistent basis key')
    return dict(atom_id=atom['atom_id'], kind=kind, support=support,
                basis_key=basis, frame_id=atom['frame_id'], origins=origins)


def features_from_saved(arms_root):
    arms = {}
    for arm in ARMS:
        key = 'arms/' + arm + '.json'
        payload = json.loads(checked_bytes(Path(arms_root)/key, SAVED_INPUT_HASHES[key]))['payload']
        atoms = payload['atoms']
        if payload['arm'] != arm or payload['L'] != LENGTHS[arm] or len(atoms) != LENGTHS[arm]:
            raise ValueError('Saved atom inventory mismatch')
        arms[arm] = [atom_features(a) for a in atoms]
    return dict(schema='f2-saved-atom-features-v1',
                saved_atom_sha256={k: SAVED_INPUT_HASHES[k] for k in SAVED_INPUT_HASHES if k.startswith('arms/')},
                arms=arms)


def classify(left, right):
    """Each named view is a separate exhaustive, disjoint partition."""
    atoms = (left, right)
    frames = [a for a in atoms if a['kind'] == 'FRAME_Z']
    paulis = [a for a in atoms if a['kind'] == 'PHYSICAL_PAULI']
    live = [a for a in atoms if a['kind'] != 'IDENTITY']
    same = len(live) == 2 and left['basis_key'] == right['basis_key']
    if not live:
        mechanism, basis = 'II', 'identity_only'
    elif len(live) == 1:
        mechanism = 'IF' if frames else 'IP'
        basis = 'frame_singleton' if frames else 'pauli_singleton_' + (
            'Z' if paulis[0]['basis_key'] == 'local:ZZZZ' else 'XY')
    elif len(frames) == 2:
        mechanism = 'FF_same' if same else 'FF_different'
        basis = 'frame_shared_full' if same else 'separate_frames'
    elif frames:
        mechanism, basis = 'FP', 'frame_pauli_boundary'
    else:
        mechanism = 'PP'
        basis = 'pauli_shared_' + ('Z' if left['basis_key'] == 'local:ZZZZ' else 'XY') if same else 'pauli_separate_local'
    if not frames:
        fragment, recipe = 'no_frame', 'no_frame'
    elif len(frames) == 1:
        fragment = frames[0]['frame_id'] + '_singleton'
        recipe = fragment + '_s' + str(len(frames[0]['support']))
    elif same:
        fragment = frames[0]['frame_id'] + '_shared_full'
        recipe = fragment
    else:
        fragment = 'F0_F1_separate'
        recipe = '>'.join(a['frame_id'] + '_s' + str(len(a['support'])) for a in atoms)
    def support_label(a):
        return 'I' if a['kind'] == 'IDENTITY' else ('F' if a['kind'] == 'FRAME_Z' else 'P') + str(len(a['support']))
    return dict(mechanism=mechanism, any_frame='frame_present' if frames else 'no_frame',
                basis_relation=basis, support_pair='>'.join(support_label(a) for a in atoms),
                origin_pair='>'.join('+'.join(a['origins']) for a in atoms),
                frame_fragment=fragment, frame_recipe=recipe,
                pauli_basis='none' if not paulis else 'all_Z' if all(a['basis_key'] == 'local:ZZZZ' for a in paulis) else 'has_XY')


def read_records(path):
    with gzip.open(path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            metrics = {m: int(row[m]) for m in METRICS}
            if any(v < 0 or str(v) != row[m] for m, v in metrics.items()):
                raise ValueError('Resource metric must be a saved nonnegative integer')
            yield dict(id=row['id'], arm=row['arm'], axis=row['axis'],
                       i=int(row['i']), j=int(row['j']), metrics=metrics)


def analyze(records, weights, features):
    """Exact integer numerators per arm, then exact Fraction conversion.

    Pair labels and weights are cached across X/Y. All logical cases, including
    compile aliases, retain their original probability. Diagnostics are excluded
    from the probability law, without subtracting any wrapper costs.
    """
    denominators = {a: math.lcm(*(p.denominator for p in weights[a])) for a in ARMS}
    numerators = {a: [int(p*denominators[a]) for p in weights[a]] for a in ARMS}
    groups = {}; pairs = {}; seen = set(); diagnostics = set(); cases = defaultdict(int)
    for arm in ARMS:
        if sum(weights[arm]) != 1 or any(p <= 0 for p in weights[arm]):
            raise ValueError('Saved probability law is not normalized and positive')
        if len(features[arm]) != len(weights[arm]):
            raise ValueError('Atom/probability length mismatch')

    def visit():
        for row in records:
            arm, axis, i, j = row['arm'], row['axis'], row['i'], row['j']
            if row['id'] in seen or axis not in ('X', 'Y') or row['id'] != case_id(arm, i, j, axis):
                raise ValueError('Duplicate or inconsistent saved case ID')
            seen.add(row['id'])
            if set(row['metrics']) != set(METRICS):
                raise ValueError('Missing resource metric')
            if arm == 'diagnostic':
                if (i, j) != (0, 0): raise ValueError('Unexpected diagnostic')
                diagnostics.add(axis)
                yield row
                continue
            if arm not in ARMS or not (0 <= i < len(weights[arm]) and 0 <= j < len(weights[arm])):
                raise ValueError('Out-of-scope saved case')
            key = (arm, i, j)
            if key not in pairs:
                pairs[key] = (numerators[arm][i]*numerators[arm][j], classify(features[arm][i], features[arm][j]))
            weight, labels = pairs[key]
            cases[arm, axis] += 1
            for view, label in labels.items():
                acc = groups.setdefault((view, arm, axis, label), [0, 0, {m: 0 for m in METRICS}])
                acc[0] += 1; acc[1] += weight
                for m, value in row['metrics'].items(): acc[2][m] += weight*value
            yield row

    totals = aggregate_metrics(visit(), weights)
    if diagnostics != {'X', 'Y'}:
        raise ValueError('Missing common diagnostic records')
    for arm in ARMS:
        for axis in ('X', 'Y'):
            if cases[arm, axis] != len(weights[arm])**2:
                raise ValueError('Missing ordered trajectory cases')
            for label in MECHANISMS:
                groups.setdefault(('mechanism', arm, axis, label), [0, 0, {m: 0 for m in METRICS}])
    result = []
    for (view, arm, axis, label), (count, mass, costs) in sorted(groups.items()):
        denominator = denominators[arm]**2
        result.append(dict(view=view, arm=arm, axis=axis, group=label, cases=count,
                           probability=Fraction(mass, denominator),
                           conditional={m: Fraction(value, mass) if mass else None for m, value in costs.items()},
                           contribution={m: Fraction(value, denominator) for m, value in costs.items()}))
    # An independent partition conservation check for every view, arm and axis.
    conservation = {}
    for row in result:
        key = row['view'], row['arm'], row['axis']
        acc = conservation.setdefault(key, [0, Fraction(0), {m: Fraction(0) for m in METRICS}])
        acc[0] += row['cases']; acc[1] += row['probability']
        for m in METRICS: acc[2][m] += row['contribution'][m]
    for (_, arm, axis), (count, mass, costs) in conservation.items():
        if count != len(weights[arm])**2 or mass != 1 or costs != totals[arm][axis]:
            raise ValueError('Grouped probability/cost conservation failure')
    return dict(groups=result, totals=totals, physical_records=sum(cases.values()),
                diagnostic_records=len(diagnostics), partition_checks=len(conservation))


def differences(groups, totals):
    lookup = {(r['view'], r['arm'], r['axis'], r['group']): r for r in groups}
    deltas = []
    for arm, baseline in (('SS', 'PP'), ('SP', 'PP'), ('PS', 'PP'), ('PS', 'SP')):
        for axis in ('X', 'Y'):
            pieces = {}
            for label in MECHANISMS:
                left = lookup['mechanism', arm, axis, label]['contribution']
                right = lookup['mechanism', baseline, axis, label]['contribution']
                pieces[label] = {m: left[m]-right[m] for m in METRICS}
            gap = {m: totals[arm][axis][m]-totals[baseline][axis][m] for m in METRICS}
            if any(sum(v[m] for v in pieces.values()) != gap[m] for m in METRICS):
                raise ValueError('Descriptive gap decomposition failure')
            deltas.append(dict(arm=arm, baseline=baseline, axis=axis, gap=gap, by_mechanism=pieces))
    # Symmetric algebraic decomposition, descriptive only; neither term is causal.
    symmetric = []
    for axis in ('X', 'Y'):
        composition = {m: Fraction(0) for m in METRICS}
        conditional = composition.copy(); pieces = {}
        for label in ('frame_present', 'no_frame'):
            a = lookup['any_frame', 'PS', axis, label]
            b = lookup['any_frame', 'SP', axis, label]
            if not a['probability'] or not b['probability']:
                raise ValueError('Common positive groups required for symmetric decomposition')
            p, q = a['probability'], b['probability']
            mix = {m: (p-q)*(a['conditional'][m]+b['conditional'][m])/2 for m in METRICS}
            cost = {m: (p+q)*(a['conditional'][m]-b['conditional'][m])/2 for m in METRICS}
            pieces[label] = dict(probability_mix=mix, conditional_cost=cost)
            for m in METRICS: composition[m] += mix[m]; conditional[m] += cost[m]
        for m in METRICS:
            if composition[m]+conditional[m] != totals['PS'][axis][m]-totals['SP'][axis][m]:
                raise ValueError('Symmetric decomposition does not reconstruct PS-SP')
        symmetric.append(dict(arm='PS', baseline='SP', axis=axis, probability_mix=composition,
                              conditional_cost=conditional, by_group=pieces))
    return deltas, symmetric


def write_groups(path, groups):
    fields = ['view', 'arm', 'axis', 'group', 'cases', 'probability_exact', 'probability']
    fields += [k for m in METRICS for k in ('conditional_'+m, 'contribution_'+m, 'contribution_'+m+'_exact')]
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for row in groups:
            data = {k: row[k] for k in ('view', 'arm', 'axis', 'group', 'cases')}
            data.update(probability_exact=str(row['probability']), probability=format(float(row['probability']), '.17g'))
            for m in METRICS:
                mean = row['conditional'][m]; value = row['contribution'][m]
                data['conditional_'+m] = '' if mean is None else format(float(mean), '.17g')
                data['contribution_'+m] = format(float(value), '.17g')
                data['contribution_'+m+'_exact'] = str(value)
            writer.writerow(data)


def exact_json(value):
    if isinstance(value, Fraction): return str(value)
    if isinstance(value, dict): return {k: exact_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [exact_json(v) for v in value]
    return value


def run_saved(public_root, atoms_root=None, feature_file=None):
    public_root = Path(public_root)
    saved = {name: checked_bytes(public_root/name, sha) for name, sha in PUBLIC_HASHES.items()}
    summary = json.loads(saved['RESULT_SUMMARY.json']); laws = json.loads(saved['F2_laws.json'])
    if (atoms_root is None) == (feature_file is None):
        raise ValueError('Exactly one saved atom root or derived feature file is required')
    if atoms_root is not None:
        features = features_from_saved(atoms_root)
    else:
        features = json.loads(Path(feature_file).read_text())
        expected = {k: SAVED_INPUT_HASHES[k] for k in SAVED_INPUT_HASHES if k.startswith('arms/')}
        if features['schema'] != 'f2-saved-atom-features-v1' or features['saved_atom_sha256'] != expected:
            raise ValueError('Derived feature provenance mismatch')
    if summary['source_commit'] != '9468c7089727f5648d9f59cddbaebe0a42f5f0a0':
        raise ValueError('Different measurement SOURCE')
    if summary['source_input_provenance']['context'] != dict(K=0, T='1/5', delta='1/5', h='1/10', q=1, r=2):
        raise ValueError('Different F2 context')
    weights = {a: [Fraction(p) for p in laws[a]['finite_bit_probabilities']] for a in ARMS}
    if any(len(weights[a]) != LENGTHS[a] for a in ARMS):
        raise ValueError('Different saved F2 law lengths')
    result = analyze(read_records(public_root/'CASE_RESOURCES.csv.gz'), weights, features['arms'])
    published = {a: {b: {m: Fraction(v) for m, v in summary['Phi'][a][b].items()} for b in ('X', 'Y')} for a in ARMS}
    if result['totals'] != published:
        raise ValueError('Exact public F2 expected cost mismatch')
    delta, symmetric = differences(result['groups'], result['totals'])
    index = {(g['view'], g['arm'], g['axis'], g['group']): g for g in result['groups']}
    axes_equal = all(index[v, a, 'X', label]['probability'] == index[v, a, 'Y', label]['probability']
                     and index[v, a, 'X', label]['contribution'] == index[v, a, 'Y', label]['contribution']
                     for v, a, axis, label in index if axis == 'X')
    return result, features, dict(
        schema='f2-saved-descriptive-attribution-v1', status='EXACT_SAVED_ATTRIBUTION_PASS_MANDATORY_STOP',
        scientific_scope='Saved-only descriptive regrouping of the completed limited F2 run',
        measurement_source_commit=summary['source_commit'], measurement_result_commit='dc7378ab34015287215bcf4ce52147e46d94ecd9',
        static_investigation_commit='3a782bd79967d23c47f27e75907027e2c2e87bf4',
        context=summary['source_input_provenance']['context'], compiler=summary['source_input_provenance']['compiler'],
        public_input_sha256=PUBLIC_HASHES, saved_atom_sha256=features['saved_atom_sha256'],
        counts=dict(physical_rows=result['physical_records'], common_diagnostics=result['diagnostic_records'],
                    group_rows=len(result['groups']), partition_reconciliations=result['partition_checks']),
        total_expectations=result['totals'], public_exact_reconciliation=True, all_group_axes_equal=axes_equal,
        differences=delta, symmetric_PS_minus_SP=symmetric,
        symmetric_identity='sum_g (p_PS-p_SP)*(mu_PS+mu_SP)/2 + (p_PS+p_SP)*(mu_PS-mu_SP)/2',
        classification_rule='Each view separately partitions all ordered trajectories; views must not be added together.',
        zero_probability_rule='Conditional expectation undefined/blank; unconditional contribution exactly zero.',
        origin_rule='F0+F1 provenance retained as one joint origin set; no fractional cost assignment.',
        statistical_certification='UNQUALIFIED_UNCHANGED', N=None, N_weighted_cost=None,
        causal_effect_identified=False, new_builds=0, new_compiles=0, new_simulations=0,
        new_sampling=0, new_laws=0, grants_used=0, mandatory_stop=True, next_owner='USER_GPT')
