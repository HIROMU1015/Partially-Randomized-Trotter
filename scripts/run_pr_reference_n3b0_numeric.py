"""Fixed N3-B0 parameter/key/old-record accounting, not a word policy.

No synthesis, strict guard, circuit, sampling, T sums/maxima, or backend calls.
"""
import ast
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as F
import hashlib
from importlib.metadata import version
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = '385c24a00fb9b47d6639fea1e7aecb0c37d55b23'
BASE = '79dfa01e6796d1ecab50e30694f5b3f73b95f4ab'
DOCS = 'docs/research/pr_reference_n3b0_preflight_20261108/'
REFS = 'artifacts/pr_reference_n3b0_preflight/2026-11-08/INPUT_REFERENCES.json'
OUT = ROOT / 'artifacts/pr_reference_n3b0_numeric/2026-11-08'
EPS = F(1, 10**11)
GUARD = 'exact cyclotomic word matrix + outward interval strict Frobenius'
PYTHON = '/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python'


def blob(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def exact_endpoint(raw):
    # Same mantissa/exponent -> Fraction convention as the saved strict guard
    # and track_c_h4_finite_bits.weight_interval; do not call those runners.
    sign, mantissa, exponent, bits = raw
    assert bits >= 0
    return F((-1 if sign else 1) * mantissa) * F(2)**exponent


def decimal_enclosure(value):
    bounds = []
    for rounding in (ROUND_FLOOR, ROUND_CEILING):
        with localcontext() as context:
            context.prec = 50
            context.rounding = rounding
            bounds.append(Decimal(value.numerator) / Decimal(value.denominator))
    assert F(bounds[0]) <= value <= F(bounds[1])
    return dict(lower=str(bounds[0]), upper=str(bounds[1]), significant_digits=50)


def evaluate():
    start = time.perf_counter()
    assert sys.version_info[:3] == (3, 12, 3)
    assert version('mpmath') == '1.3.0' and sys.executable == PYTHON
    refs_bytes = blob(PREFLIGHT, REFS)
    refs = json.loads(refs_bytes)
    assert refs['base'] == BASE
    data = {}
    for entry in refs['files']:
        value = blob(BASE, entry['path'])
        assert len(value) == entry['bytes']
        assert hashlib.sha256(value).hexdigest() == entry['sha256'], entry['path']
        assert hashlib.sha1(b'blob ' + str(len(value)).encode() + b'\0' + value).hexdigest() == entry['git_blob']
        data[entry['path']] = value
    read = lambda path: json.loads(data[path])
    rule_files = {name: hashlib.sha256(blob(PREFLIGHT, DOCS + name)).hexdigest()
                  for name in ('PREFLIGHT.md','PARAMETER_SELECTION_RULES.md','NUMERIC_EXECUTION_SCOPE.md')}
    save('INPUTS.json', dict(preflight_commit=PREFLIGHT, scientific_base=BASE,
                            references_path=REFS, references_sha256=hashlib.sha256(refs_bytes).hexdigest(),
                            files=refs['files'], selection_documents_sha256=rule_files,
                            record_pool_order=refs['record_pool_order']))
    save('SOURCE.json', dict(path='scripts/run_pr_reference_n3b0_numeric.py',
                            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            reused_metadata_validator='src/trottertracks/pr_reference_ct0.py: positive, validate_record',
                            interval_endpoint_convention='saved strict_guard and finite_bits mantissa/exponent -> Fraction',
                            quantum_modules_imported=False, old_runners_executed=False))

    phase = read('artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json')
    phase_rows = [r for r in phase['vacuum_phase'] if r['m'] == 7]
    assert len(phase_rows) == 1 and (phase_rows[0]['q'], phase_rows[0]['r']) == (128, 1)
    saved_l = phase_rows[0]['absolute_actual_binary_angle_upper']
    assert math.isfinite(saved_l)
    length = F(*saved_l.as_integer_ratio())
    assert length > 1
    precision = read('artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json')
    m7 = [r for r in precision['rows'] if r['m'] == 7]
    assert len(m7) == 1 and (m7[0]['q'],m7[0]['r'],m7[0]['bits']) == (128,1,73)
    assert precision['alpha_per_axis'] == '1/720'
    n1 = read('artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json')
    assert n1['status'] == 'CONDITIONAL_THRESHOLD_EVALUATED'
    assert n1['exact']['V_threshold'] == 173790
    assert n1['exact']['U_threshold'] == '1602930468830332421754729/9223372036854775808'

    # One fixed-precision interval evaluation, followed only by exact arithmetic.
    mp.iv.prec = 192
    sqrt2_interval = mp.iv.sqrt(mp.iv.mpf(2))
    clo, chi = map(exact_endpoint, sqrt2_interval._mpi_)
    assert 0 < clo <= chi and clo*clo <= 2 <= chi*chi
    rows = []
    for b in range(1075):
        floor_input = length * (1 << b) + F(1, 2)
        n = floor_input.numerator // floor_input.denominator
        assert n <= floor_input < n + 1 and n > 0
        m = n.bit_length()
        j = m - 1 - b
        assert (1 << (m - 1)) <= n < (1 << m)
        assert m == j + b + 1 and -1074 <= -b <= j <= 1023
        qlo, qup = clo * F(2)**(-b-2), chi * F(2)**(-b-2)
        room = EPS - qup
        row = dict(b=b,N_b=str(n),J_b=j,M_b=m,Q_lower=str(qlo),Q_upper=str(qup),G_b=str(room))
        if room <= 0:
            row.update(status='REJECTED_NO_POSITIVE_REMAINDER',p_b=None)
        else:
            p = 1
            allocation = F(1, 10)
            while m * allocation >= room:
                p += 1
                allocation /= 10
            assert allocation == F(10)**(-p)
            assert m*allocation < room
            assert p == 1 or m*allocation*10 >= room
            row.update(status='CONDITIONAL_PARAMETER_CANDIDATE',p_b=p,
                       uniform_basic_allocation=str(allocation),emitter_required_strict_cap=str(room-m*allocation))
        rows.append(row)
    candidates = [r for r in rows if r['p_b'] is not None]
    assert candidates, 'mandatory STOP: no admissible parameter tuple'
    chosen = min(candidates, key=lambda r:(r['p_b'],r['M_b'],r['b']))
    chosen_tuple = (chosen['p_b'],chosen['M_b'],chosen['b'])
    assert all(chosen_tuple <= (r['p_b'],r['M_b'],r['b']) for r in candidates)
    assert sum((r['p_b'],r['M_b'],r['b']) == chosen_tuple for r in candidates) == 1
    b,jmax,m,p = chosen['b'],chosen['J_b'],chosen['M_b'],chosen['p_b']
    allocation = F(chosen['uniform_basic_allocation'])
    qlo,qup = F(chosen['Q_lower']),F(chosen['Q_upper'])
    rlo,rhi = EPS-qup-m*allocation,EPS-qlo-m*allocation
    assert 0 < rlo <= rhi and qup + m*allocation + rlo == EPS
    decisions = dict(rule_commit=PREFLIGHT,rule='lexicographic minimum (p_b,M_b,b), before coverage',
                     sqrt2_lower=str(clo),sqrt2_upper=str(chi),rows=rows)
    decision_bytes = (json.dumps(decisions,ensure_ascii=False,indent=2)+'\n').encode()
    (OUT/'CANDIDATE_DECISIONS.json.xz').write_bytes(lzma.compress(decision_bytes))

    basic_keys = []
    for j in range(-b,jmax+1):
        value = math.ldexp(1.0,j)
        assert math.isfinite(value) and value > 0
        literal = value.hex()
        assert F(*value.as_integer_ratio()) == F(2)**j
        assert F(*float.fromhex(literal).as_integer_ratio()) == F(2)**j
        basic_keys.append(dict(j=j,positive_binary64_key=literal,exact_angle=str(F(2)**j),
                               allocated_Frobenius_upper=str(allocation),prospective_precision_exponent=-p))
    assert len(basic_keys) == m and len({k['positive_binary64_key'] for k in basic_keys}) == m
    parameters = dict(status='PARAMETER_SCOPE_FIXED',preflight_commit=PREFLIGHT,scientific_base=BASE,
                      scientific_conditions=dict(model='Linear H4',distance_A='1.00',basis='STO-3G',DF_rank=12,
                          input='exact original-H ground supplied',state_preparation_cost_excluded=True,
                          PF='Partial S2',LD=3,h='0.02',finite_RTE_K=2,controlled_convention='ordinary diag(I,V)',
                          m=7,q=128,r=1,canonical_bits=73,compiler='A',qiskit='1.3.0',native_basis=['rz','sx','x','cx'],
                          shots_per_axis=199,energy_accuracy_Ha='1.5936e-4',total_failure='0.05',phase_cap='0.4',alpha='1/720'),
                      L_saved_binary64=saved_l,L_hex=saved_l.hex(),L_exact=str(length),epsilon=str(EPS),
                      b=b,j_min=-b,j_max=jmax,N_b=chosen['N_b'],M_b=m,internal_p=p,
                      prospective_precision_exponent=-p,uniform_basic_Frobenius_allocation=str(allocation),
                      maximum_basic_word_uses=m,selected_tuple=list(chosen_tuple),
                      candidate_b_count=len(rows),conditional_candidate_count=len(candidates),
                      negative_angles='actual adjoint of the entire positive concatenated word; not generated here',
                      existing_native_C0_words='preserved unchanged',basic_keys=basic_keys)
    save('FIXED_PARAMETERS.json',parameters)
    save('ERROR_BUDGET.json',dict(epsilon=str(EPS),interval_precision_bits=192,
        sqrt2=dict(lower=str(clo),upper=str(chi),exact_square_enclosure_passed=True),
        quantization_Frobenius=dict(lower=str(qlo),upper=str(qup),lower_decimal=decimal_enclosure(qlo),upper_decimal=decimal_enclosure(qup)),
        uniform_basic_Frobenius_allocation=str(allocation),basic_allocation_sum=str(m*allocation),
        emitter_remaining=dict(lower=str(rlo),upper=str(rhi),lower_decimal=decimal_enclosure(rlo),upper_decimal=decimal_enclosure(rhi),
                               required_condition='e_emit < lower',actual_e_emit_upper=None,status='UNQUALIFIED'),
        exact_budget_identity='Q_upper + M*g_alloc + R_emit_lower = epsilon; strict inequality needs e_emit < R_emit_lower',
        whole_RPE_error_or_shots_recomputed=False))

    # Pure saved-record metadata checks, reusing existing code without imports
    # of Qiskit, pygridsynth, acquire, strict_guard, or any scientific runner.
    source_path = 'src/trottertracks/pr_reference_ct0.py'
    parsed = ast.parse(data[source_path],filename=source_path)
    funcs = [r for r in parsed.body if isinstance(r,ast.FunctionDef) and r.name in ('positive','validate_record')]
    assert {r.name for r in funcs} == {'positive','validate_record'}
    context = dict(F=F,math=math,hashlib=hashlib)
    exec(compile(ast.Module(body=funcs,type_ignores=[]),source_path,'exec'),context)
    validate = context['validate_record']
    by_angle = defaultdict(list)
    signatures = {}
    archive_stats = []
    core = ('theta_hex','precision_exponent','key','spec','status','sequence','sequence_sha256',
            'global_W','T','Tdagger','route','guard_method','strict_error_upper')
    c0 = read('artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json')
    assert c0['exponent'] == -11 and F(c0['epsilon']) == EPS
    c0_rows = {r['key']:r for r in c0['records']}
    assert len(c0_rows) == len(c0['records'])
    c0_seen = set()
    duplicate_count = 0
    for rank,path in enumerate(refs['record_pool_order']):
        bundle_path = str(Path(path).parent/'BUNDLES.json')
        bundle = next(x for x in read(bundle_path) if x['path'] == Path(path).name)
        assert hashlib.sha256(data[path]).hexdigest() == bundle['sha256']
        raw_count = qualified_count = legacy_count = duplicates = 0
        with tarfile.open(fileobj=io.BytesIO(data[path]),mode='r:xz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                assert member.name.endswith('.json')
                raw = json.load(archive.extractfile(member))
                raw_count += 1
                validate(raw)
                assert Path(member.name).stem == raw['key']
                signature = tuple(F(raw[k]) if k == 'strict_error_upper' else raw.get(k) for k in core)
                if raw['key'] in signatures:
                    assert signatures[raw['key']] == signature, ('saved record scientific disagreement',raw['key'])
                    duplicates += 1
                    duplicate_count += 1
                else:
                    signatures[raw['key']] = signature
                if rank == 0:
                    assert raw['key'] in c0_rows
                    old = c0_rows[raw['key']]
                    assert raw['theta_hex'] == old['angle'] and raw['precision_exponent'] == old['exponent']
                    for key in ('spec','sequence_sha256','global_W','guard_method','strict_error_upper','route','T','Tdagger'):
                        assert raw[key] == old[key], ('existing C0 record mismatch',raw['key'],key)
                    assert old['positive_word_negative_actual_adjoint'] is True
                    c0_seen.add(raw['key'])
                method_ok = raw.get('guard_method') == GUARD
                qualified_count += int(method_ok)
                legacy_count += int(not method_ok)
                by_angle[raw['theta_hex']].append(dict(pool_rank=rank,archive=path,member=member.name,
                    key=raw['key'],precision_exponent=raw['precision_exponent'],sequence_sha256=raw['sequence_sha256'],
                    global_W=raw['global_W'],guard_method=raw.get('guard_method'),strict_error_upper=raw['strict_error_upper'],
                    method_qualified=method_ok))
        archive_stats.append(dict(path=path,records_checked=raw_count,method_qualified=qualified_count,
                                  method_unqualified=legacy_count,duplicate_keys_with_earlier_pools=duplicates,
                                  metadata_checks_passed=True))
    assert c0_seen == set(c0_rows)
    coverage_rows = []
    missing = []
    for key in basic_keys:
        matches = sorted(by_angle.get(key['positive_binary64_key'],[]),key=lambda r:(r['pool_rank'],r['key']))
        checked = []
        for row in matches:
            row = dict(row)
            row['budget_eligible'] = row['method_qualified'] and F(row['strict_error_upper']) <= allocation
            row['classification'] = ('ELIGIBLE_SAVED_CANDIDATE' if row['budget_eligible'] else
                                     'GUARD_METHOD_UNQUALIFIED' if not row['method_qualified'] else 'SAVED_BOUND_EXCEEDS_ALLOCATION')
            checked.append(row)
        eligible = [r for r in checked if r['budget_eligible']]
        if not eligible:
            missing.append(key)
        coverage_rows.append(dict(**key,status='EXISTING_WORD_AVAILABLE' if eligible else 'ACQUISITION_REQUIRED',
                                  matches=checked,preferred_candidate=eligible[0] if eligible else None,
                                  adopted_word=None))
    covered = m - len(missing)
    special_angle = '0x1.0000000000000p-61'
    special = dict(positive_binary64_key=special_angle,
                   within_selected_key_set=special_angle in {k['positive_binary64_key'] for k in basic_keys},
                   records=by_angle.get(special_angle,[]),
                   note='Existing N3-A candidate only; never extend the selected key set to include it.')
    coverage = dict(status='EXISTING_WORD_COVERAGE_PARTIAL' if 0 < covered < m else
                           'NO_ELIGIBLE_EXISTING_BASIC_KEYS' if covered == 0 else 'ALL_KEYS_HAVE_SAVED_CANDIDATES',
                    selected_b=b,internal_p=p,required_basic_keys=m,covered_basic_keys=covered,
                    acquisition_required_basic_keys=len(missing),archive_checks=archive_stats,
                    unique_record_keys_checked=len(signatures),duplicate_records_checked=duplicate_count,
                    rows=coverage_rows,missing_keys=missing,special_existing_2minus61=special,
                    no_new_word_or_guard=True,existing_C0_preserved=True,T_sums_or_maxima_computed=False)
    save('BASIC_KEY_COVERAGE.json',coverage)

    # Arithmetic-only boundary checks; this is not an online word adapter.
    tie_cases = [(F(0),0),(F(1,2),0),(F(-1,2),0),(F(3,2),2),(F(-3,2),-2),
                 (F(5,2),2),(F(-5,2),-2),(F(7,2),4),(F(-7,2),-4)]
    for value,expected in tie_cases:
        assert round(value) == expected and round(-value) == -expected
    endpoint_checks = []
    for theta in (F(0),length,-length):
        index = round(theta * (1 << b))
        delta = abs(theta - F(index,1 << b))
        assert delta <= F(1,1 << (b+1)) and abs(index) <= int(chosen['N_b'])
        assert round(-theta * (1 << b)) == -index
        endpoint_checks.append(dict(theta=str(theta),index=str(index),absolute_angle_error=str(delta),passed=True,
                                    interpretation='arithmetic envelope check; no claim endpoint is reachable'))
    save('CHECKS.json',dict(status='PASS',input_hashes=len(data),sqrt2_exact_square_enclosure=True,
        candidate_integer_floor_bitlength_checks=len(rows),coarseness_checked_for_every_candidate=True,
        lexicographic_selection_unique=True,selected_basic_key_exact_ratio_checks=m,
        exact_budget_identity=True,actual_e_emit_not_assumed_zero=True,
        metadata_only_record_checks=archive_stats,duplicate_scientific_content_matched=True,
        existing_C0_unchanged=True,arithmetic_tie_cases=len(tie_cases),endpoint_checks=endpoint_checks,
        no_synthesis_guard_policy_compile_sampling_or_T_aggregation=True))
    save('ENVIRONMENT.json',dict(host=platform.node(),python=platform.python_version(),executable=sys.executable,
        mpmath=version('mpmath'),interval_precision_bits=192,available_CPU_affinity=len(os.sched_getaffinity(0)),
        disk_free_bytes=shutil.disk_usage(ROOT).free,wall_seconds=time.perf_counter()-start,
        workload='small exact arithmetic and read-only saved-record metadata; one process without a resource cap',
        shared_settings_or_packages_changed=False))
    labels = ['PARAMETER_SCOPE_FIXED','PRICE_BOUND_CONDITIONAL','UNQUALIFIED_EMITTER_AND_FULL_SUPPORT_POLICY']
    if 0 < covered < m:
        labels.append('EXISTING_WORD_COVERAGE_PARTIAL')
    if missing:
        labels.append('DICTIONARY_ACQUISITION_REQUIRED')
    return dict(labels=labels,b=b,j_min=-b,j_max=jmax,M_b=m,internal_p=p,
                covered=covered,missing=len(missing),emitter_strict_cap_exact=str(rlo),
                emitter_strict_cap_decimal=decimal_enclosure(rlo),coverage_status=coverage['status'])


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        summary = evaluate()
    except Exception as error:
        save('STOP.json',dict(status='MANDATORY_STOP',completed=False,error=f'{type(error).__name__}: {error}',automatic_continuation=False))
        raise
    save('STOP.json',dict(status='MANDATORY_STOP',completed=True,preflight_commit=PREFLIGHT,scientific_base=BASE,
                          qualification=summary['labels'],new_synthesis_guard_policy_compile_sampling=0,
                          T_price_aggregation_or_all18=False,old_SOURCE_results_STOPs_modified=False,
                          actual_emitter='UNQUALIFIED',next_owner='GPT / User',automatic_continuation=False))
    print(json.dumps(dict(summary,STOP='MANDATORY_STOP'),ensure_ascii=False,indent=2))
