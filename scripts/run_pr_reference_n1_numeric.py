"""N1 saved-input integer/Fraction accounting only; no quantum/price providers.

Scope: bf0a7b1b5d0afb6ead3cc83ce909bbe53514b24a:
docs/research/pr_reference_n1_preparation_20261108/NUMERIC_EXECUTION_SCOPE.md
"""
import ast
from collections import Counter
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as F
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = '0dbf1fe265176677e026ea982c0bd435987f4338'
PREPARATION = 'bf0a7b1b5d0afb6ead3cc83ce909bbe53514b24a'
REFS = 'artifacts/pr_reference_n1_preparation/2026-11-08/INPUT_REFERENCES.json'
OUT = ROOT / 'artifacts/pr_reference_n1_numeric/2026-11-08'
CT4 = 'artifacts/pr_reference_ct4/2026-11-07/'
GUARD = 'exact cyclotomic word matrix + outward interval strict Frobenius'


def saved(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def unique(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), f'duplicate {key}'
    return result


def evaluate():
    start = time.perf_counter()
    refs_bytes = saved(PREPARATION, REFS)
    refs = json.loads(refs_bytes)
    assert refs['base'] == BASE
    data = {}
    for entry in refs['files']:
        value = saved(BASE, entry['path'])
        assert len(value) == entry['bytes']
        assert hashlib.sha256(value).hexdigest() == entry['sha256'], entry['path']
        blob = hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest()
        assert blob == entry['git_blob'], entry['path']
        data[entry['path']] = value
    read = lambda path: json.loads(data[path])
    role_rows = unique(read(CT4 + 'ROLE_T_COUNTS.json')['rows'], 'name')
    pilots = unique(read(CT4 + 'PILOT_T_COUNTS.json')['rows'], 'name')
    assert set(role_rows) == set(pilots) == set(refs['prices']['names'])
    roles = {}
    role_checks = []
    for name, row in role_rows.items():
        assert row['projection_matches_literal_histogram'] is True
        assert row['status'] == 'QUALIFIED_SIGNED_ROLE_ACCOUNTING'
        rr = unique(row['roles'], 'role')
        for role, item in rr.items():
            assert item['keys_available'] is True and item['account'] is not None
            assert item['signed_accounting_correction'] == (role == 'D_boundary_fusion_difference')
            assert type(item['account']['T']) is int
        roles[name] = {role: item['account']['T'] for role, item in rr.items()}
        pilot = pilots[name]
        assert pilot['q'] == 128 and pilot['r'] == 1 and pilot['axis'] in 'XY'
        total = sum(roles[name].values())
        assert total == pilot['account']['T'], name
        role_checks.append(dict(name=name,role_T_sum=total,saved_native_T=pilot['account']['T'],passed=True))

    def anchor(arm, role):
        values = {roles[f'rare{rare}_{arm}_{axis}'][role] for rare in (0, 1) for axis in 'XY'}
        assert len(values) == 1, (arm, role)
        return values.pop()

    anchors = {arm: {role: anchor(arm, role) for role in ('D_unfused','D_boundary_fusion_difference')}
               for arm in ('Bprime', 'reference')}
    db = sum(anchors['Bprime'].values())
    dr = sum(anchors['reference'].values())
    cswap = anchor('reference', 'CSWAP')
    scalar = anchor('Bprime', 'scalar_or_vacuum_correction')
    assert anchor('Bprime', 'CSWAP') == 0

    # Reuse the existing pure metadata validator without importing its runners.
    validator_path = 'src/trottertracks/pr_reference_ct0.py'
    parsed = ast.parse(data[validator_path], filename=validator_path)
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ('positive','validate_record')]
    assert {n.name for n in functions} == {'positive','validate_record'}
    namespace = dict(math=math, F=F, hashlib=hashlib)
    exec(compile(ast.Module(body=functions, type_ignores=[]), validator_path, 'exec'), namespace)
    validate_record = namespace['validate_record']
    inventory = read(CT4 + 'KEY_REUSE_AND_SYNTHESIS.json')
    assert inventory['exponent'] == -11 and F(inventory['epsilon']) == F(1,10**11)
    word_checks = []
    prices = {}
    with tarfile.open(fileobj=io.BytesIO(data[CT4 + 'strict_records.tar.xz']),mode='r:xz') as archive:
        members = unique([dict(key=Path(m.name).stem, member=m) for m in archive.getmembers() if m.isfile()], 'key')
        for angle, symbol in refs['central_keys']['angle_to_symbol'].items():
            candidates = [r for r in inventory['records'] if r['angle'] == angle and r['exponent'] == -11]
            assert len(candidates) == 1
            row = candidates[0]
            raw = json.load(archive.extractfile(members[row['key']]['member']))
            assert validate_record(raw) == (angle, -11)
            assert raw['guard_method'] == row['guard_method'] == GUARD
            assert row['positive_word_negative_actual_adjoint'] is True
            for key in ('key','spec','sequence_sha256','global_W','strict_error_upper','route','T','Tdagger'):
                assert raw[key] == row[key], (symbol, key)
            prices[symbol] = raw['T']
            word_checks.append(dict(symbol=symbol,angle=angle,key=raw['key'],T=raw['T'],Tdagger=raw['Tdagger'],
                                    global_W=raw['global_W'],sequence_sha256=raw['sequence_sha256'],
                                    guard_metadata=raw['guard_method'],passed=True))

    def m7(rows):
        found = [r for r in rows if r['m'] == 7]
        assert len(found) == 1
        row = found[0]
        assert (row['q'],row['r'],row['bits']) == (128,1,73)
        return row

    row = m7(read('artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json')['rows'])
    assert row['shots_per_axis'] == 199
    probability_text = row['contributions']['p_order2']
    assert probability_text == refs['probability']['saved_fraction']
    p2 = F(probability_text)
    assert 0 <= p2 <= 1 and p2.denominator == 1 << 73
    precision = read('artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json')
    m7(precision['rows'])
    assert precision['alpha_per_axis'] == '1/720'
    ledger = json.loads(lzma.decompress(data['artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz']))
    ledger_rows = [r for r in ledger['rows'] if r['m'] == 7 and r['arm'] in ('Bprime','reference')]
    assert {(r['arm'],r['axis']) for r in ledger_rows} == {(a,x) for a in ('Bprime','reference') for x in 'XY'}
    assert len(ledger_rows) == 4
    assert all((r['q'],r['r'],r['bits'],r['shots']) == (128,1,73,199) for r in ledger_rows)
    error = read(CT4 + 'ERROR_AND_SHOTS.json')
    assert (str(error['energy_accuracy_Ha']),error['phase_cap'],error['total_failure'],error['alpha']) == ('0.00015936','.4','.05','1/720')
    assert len(error['rows']) == 16 and all(r['saved_N'] == 199 and r['saved_N_sufficient'] for r in error['rows'])

    # The six requested boundary quantities are evaluated here, once.
    k = db - dr - cswap + scalar
    d0 = 2 * prices['h0'] - prices['f0']
    d2 = 2 * prices['h2'] - prices['f2']
    dmin = min(d0, d2)
    v_threshold = k + 128 * dmin
    u_threshold = k + 128 * ((1 - p2) * d0 + p2 * d2)
    assert u_threshold == k + 128 * d0 + 128 * p2 * (d2 - d0)

    inputs = read(CT4 + 'FIXED_INPUTS.json')
    assert (inputs['q'],inputs['r'],inputs['exponent']) == (128,1,-11)
    assert F(inputs['epsilon']) == F(1,10**11)
    trajectories = unique(inputs['trajectories'], 'rare')
    assert set(trajectories) == {False, True}
    pair_checks = []
    for rare, trajectory in trajectories.items():
        assert (trajectory['q'],trajectory['r'],trajectory['j'],trajectory['source_CSWAPs']) == (128,1,0,16)
        events = trajectory['events']
        orders = Counter(e['order'] for e in events)
        assert len(events) == 128 and set(orders) <= {0,2}
        n0,n2 = orders[0],orders[2]
        for axis in 'XY':
            bn,rn = f'rare{int(rare)}_Bprime_{axis}',f'rare{int(rare)}_reference_{axis}'
            b,r = roles[bn],roles[rn]
            assert b['RTE_basis_frames'] == r['RTE_basis_frames']
            assert b['RTE_products'] == r['RTE_products'] == b['XY_wrapper'] == r['XY_wrapper'] == 0
            assert r['RTE_event_sign_phase'] == 0 and b['RTE_event_sign_phase'] >= 0
            assert b['RTE_central_rotation'] == 2 * (n0*prices['h0'] + n2*prices['h2'])
            assert r['RTE_central_rotation'] == n0*prices['f0'] + n2*prices['f2']
            predicted = k + n0*d0 + n2*d2 + b['RTE_event_sign_phase'] - r['scalar_or_vacuum_correction']
            direct = pilots[bn]['account']['T'] - pilots[rn]['account']['T']
            assert predicted == direct
            pair_checks.append(dict(rare=rare,axis=axis,order0_count=n0,order2_count=n2,
                                    saved_event_sha256=trajectory['event_sha256'],
                                    B_event_T=b['RTE_event_sign_phase'],saved_reference_vacuum_T=r['scalar_or_vacuum_correction'],
                                    predicted_T_difference=predicted,saved_native_T_difference=direct,passed=True))

    def outward(value):
        bounds = []
        for rounding in (ROUND_FLOOR,ROUND_CEILING):
            with localcontext() as context:
                context.prec=40
                context.rounding=rounding
                bounds.append(Decimal(value.numerator)/Decimal(value.denominator))
        assert F(bounds[0]) <= value <= F(bounds[1])
        return dict(lower=str(bounds[0]),upper=str(bounds[1]),significant_digits=40,rounding=['ROUND_FLOOR','ROUND_CEILING'])

    result = dict(status='CONDITIONAL_THRESHOLD_EVALUATED',base=BASE,preparation_commit=PREPARATION,
                  scope='docs/research/pr_reference_n1_preparation_20261108/NUMERIC_EXECUTION_SCOPE.md',
                  unit='adopted physical T+Tdagger per axis per shot, q128/r1/m7; not minimum T',
                  input_references=dict(commit=PREPARATION,path=REFS,sha256=hashlib.sha256(refs_bytes).hexdigest(),verified_files=len(data)),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime=dict(host=os.uname().nodename,python=platform.python_version(),available_CPUs=len(os.sched_getaffinity(0)),wall_seconds=time.perf_counter()-start),
                  anchors=dict(roles=anchors,D_B_star=db,D_R_star=dr,Csw_star=cswap,S_B_star=scalar),
                  saved_p2=probability_text,central_prices=prices,
                  exact=dict(K=k,d0=d0,d2=d2,dmin=dmin,V_threshold=v_threshold,U_threshold=str(u_threshold),
                             U_numerator=u_threshold.numerator,U_denominator=u_threshold.denominator),
                  U_decimal_outward=outward(u_threshold),
                  conditions=dict(pointwise='V_max < V_threshold',expected='U_V < U_threshold',compiler='CT-5B condition C',
                                  policy='same adopted e=-11 positive words; negative actual adjoint; completion for all reached keys remains unqualified',
                                  joint_law='canonical 73-bit law; no independence assumption between vacuum/order/product signs',
                                  E_B='nonnegative physical event price dropped for conservative sufficient conditions'),
                  checks=dict(input_hashes_passed=True,record_metadata_and_word_counts=word_checks,saved_role_sums=role_checks,
                              saved_pair_identities=pair_checks,alternative_expectation_formula=True,outward_interval_contains_exact=True,
                              stored_m7_bits_shots_and_error_conditions=True,new_error_or_shot_accounting=False),
                  unqualified=['vacuum full-support words/termination/strict qualification','V_max','U_V','missing common frame words',
                               'Bprime b2 word if reachable','full-support CT RPE success','all18 total T','T-depth'],
                  not_performed=['new synthesis/guard/word generation','circuit build/compile','new price acquisition','probability regeneration',
                                 'simulation/sampling','precision/shot/alpha/fusion optimization','SK/CT-3B/N2/N3','shot-multiplied round T/all18 T'])
    save('RESULTS.json', result)
    return result


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        result = evaluate()
    except Exception as error:
        save('STOP.json', dict(status='MANDATORY_STOP',completed=False,error=f'{type(error).__name__}: {error}',automatic_continuation=False))
        raise
    save('STOP.json',dict(status='MANDATORY_STOP',completed=True,qualification=result['status'],base=BASE,preparation_commit=PREPARATION,
                          old_SOURCE_results_STOPs_modified=False,new_synthesis_guard_compile_sampling_simulation=0,
                          all18_T_or_Tdepth_computed=False,automatic_continuation=False,next_owner='User / GPT'))
    print(json.dumps(dict(status=result['status'],exact=result['exact'],U_decimal_outward=result['U_decimal_outward'],
                          role_sum_checks=len(result['checks']['saved_role_sums']),pair_checks=len(result['checks']['saved_pair_identities']),
                          STOP='MANDATORY_STOP'),ensure_ascii=False,indent=2))
