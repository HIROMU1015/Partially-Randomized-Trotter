"""N3: saved-output integer checks only; no scientific loader/primitive calls."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f2638a7845685a119619382b850441f5d8d33713'
PLAN = 'ba6169028d6d71105aa1f474dfcd285461339070'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ratio(literal):
    if literal['type'] == 'builtins.int':
        return int(literal['integer']), 1
    require(literal['type'] in ('builtins.float', 'numpy.float64'), 'Literal type')
    value = float.fromhex(literal['hex'])
    require(math.isfinite(value), 'Nonfinite saved literal')
    return value.as_integer_ratio()


def check_grid(records, ell):
    require(records and len({r['key'] for r in records}) == len(records), 'Generator identity coverage')
    maxima = []
    for row in records:
        n, d = ratio(row['literal'])
        require(row['role'] in ('angle', 'phase'), 'Generator role')
        require(d > 0 and d & (d-1) == 0, 'Generator denominator')
        k = d.bit_length()-1
        expected = k+int(row['role'] == 'angle')
        require(row['numerator'] == str(n) and row['denominator_power'] == k
                and row['required_ell'] == expected, 'Saved exact ratio / half exponent')
        required_d = d*(2 if row['role'] == 'angle' else 1)
        require((1 << ell) % required_d == 0, 'Common integer grid inclusion')
        # Check both signs, without evaluating any physical F/G sum.
        for sign in (-1, 1):
            scaled = Fraction(sign*n, required_d)*(1 << ell)
            require(scaled.denominator == 1, 'Signed exact half/grid test')
        maxima.append(expected)
    require(ell == max(maxima), 'Maximum exponent mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    require(not (out/'CHECKS.json').exists(), 'Preserve existing N3 checks')
    names = ['SOURCE.json', 'N1_CATALOG.json', 'N2_GENERATORS.json', 'RESULTS.json', 'STOP.json']
    raw = {name:(out/name).read_bytes() for name in names}
    source, n1, n2, result, stop = [json.loads(raw[name]) for name in names]
    require(source['scope_commit'] == PLAN and source['scientific_input_commit'] == BASE, 'Scope identity')
    git_bytes = lambda ref,path: subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    for path,key in [('scripts/run_pr_reference_v6q3_n.py','driver_sha256'),
                     ('scripts/check_pr_reference_v6q3_n.py','checker_sha256')]:
        require((ROOT/path).read_bytes() == git_bytes(source['execution_commit'], path)
                and sha((ROOT/path).read_bytes()) == source[key], 'Audit SOURCE preservation')
    for path, binding in source['source_bindings'].items():
        require(sha(git_bytes(binding['ref'], path)) == binding['sha256']
                == sha((ROOT/path).read_bytes()), 'Frozen SOURCE preservation')
    require(sha((out/'common.pkl').read_bytes()) == source['cache_pickle_sha256'], 'Cache preservation')
    a1raw = git_bytes(BASE, 'artifacts/pr_reference_interference_pilot/2026-11-06/A1_REFERENCE_PHASE.json')
    v2raw = git_bytes(BASE, 'artifacts/pr_reference_bprime/2026-11-07/V2_INPUTS_CONSTANTS.json')
    require(sha(a1raw) == source['saved_input_hashes']['A1'] and
            sha(v2raw) == source['saved_input_hashes']['V2'], 'Saved witness preservation')
    a1, v2 = json.loads(a1raw), json.loads(v2raw)
    bases = n1['bases']
    require([b['index'] for b in bases if b['kind'] == 'D_basis'] == list(range(4)), 'D catalog coverage')
    require({b['basis_id'] for b in bases if b['kind'] == 'R_basis'} ==
            {b['basis_id'] for b in a1['checks'] if b['kind'] == 'R_basis'}, 'R catalog coverage')
    basis_by_id = {b['basis_id']:b for b in bases if b['kind'] == 'R_basis'}
    types = Counter()
    for b in bases:
        for index, op in enumerate(b['operations']):
            width, arity = (1, 1) if op['name'] == 'rz' else (2, 2)
            require(op['name'] in ('rz', 'xx_plus_yy') and op['index'] == index
                and op['width'] == len(op['wires']) == width and len(set(op['wires'])) == width
                and all(type(w) is int and 0 <= w < 8 for w in op['wires'])
                and op['num_clbits'] == 0 and op['condition'] is None
                and len(op['parameters']) == arity and op['saved_metadata_matches'], 'N1 gate metadata')
            for p in op['parameters']:
                ratio(p)
                types[p['type']] += 1
        require(b['parameter_count'] == sum(len(op['parameters']) for op in b['operations']), 'N1 parameter count')
    for c in n1['components']:
        require(len(c['support']) in (1,2) and c['basis_id'] in basis_by_id
                and c['basis_hash'] == basis_by_id[c['basis_id']]['basis_hash'], 'Tail support and registry binding')
    require(not n1['basis_exponents_evaluated'] and n1['all_supported_finite']
            and n1['all_saved_metadata_matches'], 'N1 scope and pass')
    require([(c['block'], c['time_hex']) for c in n2['contexts']] ==
            [(i,(.01).hex()) for i in range(4)]+[(0,(.02).hex())], 'Five primitive contexts')
    records = n2['records']
    for ctx in n2['contexts']:
        prefix = 'D'+str(ctx['block'])+'@'+ctx['time_hex']+'/'
        subset = [r for r in records if r['key'].startswith(prefix)]
        require(sum(r['key'] == prefix+'global' for r in subset) == 1, 'D global phase coverage')
        require(sum('/rz/' in r['key'] for r in subset) == 8, 'D RZ coverage')
        require(sum('/rzz/' in r['key'] for r in subset) == (0 if ctx['block'] == 0 else 28), 'D RZZ coverage')
    for order in (0,2):
        r = [r for r in records if r['key'] == 'RTE/r32/order'+str(order)]
        w = [w for w in v2['angles'] if w['r'] == 32 and w['order'] == order]
        require(len(r) == len(w) == 1 and r[0]['saved_witness'] == w[0]
                and r[0]['literal']['hex'] == w[0]['actual_hex'] and r[0]['signs_covered'] == [-1,1], 'Central saved angle coverage')
    for key, expected in [('zero',0.), ('pi',math.pi), ('tau',math.tau)]:
        row = [r for r in records if r['key'] == key]
        require(len(row) == 1 and row[0]['literal']['hex'] == expected.hex(), 'Period generator')
    scalar = [r for r in records if r['key'] == 'scalar/n=65536']
    require(len(scalar) == 1 and scalar[0]['expression'] == '-.02*n*(constant_coefficient+extracted_identity_coefficient)'
            and scalar[0]['n'] == 65536, 'Fixed scalar SOURCE expression')
    ell = n2['lattice']['ell']
    check_grid(records, ell)
    require(n2['lattice'] == result['lattice'] and result['lattice']['sufficient'] == (ell <= 1022), 'Lattice summary')
    require(result['N1_basis_count'] == len(bases) and result['N1_component_count'] == len(n1['components'])
        and result['N1_operation_count'] == sum(len(b['operations']) for b in bases)
        and result['N1_parameter_types'] == dict(types) and result['N2_generator_count'] == len(records), 'Summary counts')
    require(result['calls'] == stop['calls'] == dict(cache_loads=1, diagonal_primitive_calls=5,
        scalar_evaluations=1, saved_central_angle_reads=2, provider_phase_calls=0, new_atan_calls=0,
        trajectory_circuit_compile_simulation_sampling_RPE_RZ_CT_price_calls=0), 'Scientific call scope')
    status = 'PASS_IN_FIXED_ARITHMETIC_MODEL' if ell <= 1022 else 'INCONCLUSIVE_LATTICE'
    require(result['status'] == stop['status'] == status and stop['scientific_campaigns'] == 1
        and stop['scientific_retries'] == 0 and stop['next_action'] == 'MANDATORY_STOP', 'Campaign STOP')
    require(not result['actual_m17_caller_connected'] and not result['V6P_cost_promoted']
        and result['old_M_and_Bnorm_preserved'], 'Qualification boundary')
    for name in names:
        require((out/name).read_bytes() == raw[name], 'Saved result mutated')
    checks = dict(status='N3_SAVED_OUTPUT_CHECKS_PASS', scientific_campaigns=1, scientific_retries=0,
        scientific_cache_loads_or_primitive_calls_in_checker=0, grid_integer_checks=True,
        catalog_and_generator_coverage=True, ell=ell, campaign_status=status,
        preserved_sha256={name:sha(value) for name,value in raw.items()}, next_action='MANDATORY_STOP')
    with (out/'CHECKS.json').open('x') as stream:
        json.dump(checks, stream, indent=2); stream.write('\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
