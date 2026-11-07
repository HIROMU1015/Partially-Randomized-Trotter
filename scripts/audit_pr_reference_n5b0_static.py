"""N5-B0 saved metadata/SOURCE only. No guard, emitter or numeric error calls.

W residues are integer word metadata, not angle reduction. Fractions below
only compare existing guard/cap fields; no new error sum or bound is formed.
"""
from __future__ import annotations
import ast
from collections import Counter
from fractions import Fraction as F
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "d1d5992baa6b818fb17fb0c05ba0c100bab3806d"
OUT = ROOT/"artifacts/pr_reference_n5b0_static/2026-11-08"
N5 = "artifacts/pr_reference_n5a_static/2026-11-08/"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
B4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/"
C4 = "artifacts/pr_reference_n4c_accounting/2026-11-08/"
C0 = "artifacts/pr_reference_ct4/2026-11-07/"
CT0 = "artifacts/pr_reference_ct0/2026-11-07/"
REFERENCES = {}


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    REFERENCES[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    raw = blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith('.json.xz') else raw)


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode='r:xz') as t:
        return {m.name:t.extractfile(m).read() for m in t if m.isfile()}


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False)+'\n')


def main():
    raw = blob('src/trottertracks/pr_reference_ct0.py')
    selected = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
                and n.name in ('positive','validate_record')]
    ns = dict(math=math, F=F, hashlib=hashlib)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'fixed CT0 metadata only', 'exec'), ns)
    m17 = next(r for r in read(N5+'ROUND_METADATA.json')['rows'] if r['saved_condition']['m']==17)
    condition = m17['saved_condition']
    coverage = read(CT0+'KEY_COVERAGE.json.xz')
    caps = {arm:{axis:F(a['uniform_gate_error_strict_upper']) for axis,a in old['axis'].items()}
            for arm,old in condition['arms'].items()}
    headroom = {r['key']:r for r in read(D3+'C0_STRICT_HEADROOM.json')['rows']}
    parameters = read(D3+'INPUTS.json')['fixed_parameters']
    budget = read(D3+'STRICT_ERROR_BUDGET.json')
    c0, words = [], {}
    for member, data in archive(C0+'strict_records.tar.xz').items():
        r = json.loads(data);ns['validate_record'](r)
        assert r['guard_method'] is not None
        h = headroom[r['theta_hex']]
        assert h['strict_error_upper']==r['strict_error_upper'] and h['sequence_sha256']==r['sequence_sha256']
        row = dict(theta_hex=r['theta_hex'], record_key=r['key'], archive=C0+'strict_records.tar.xz',
            member=member, sequence_sha256=r['sequence_sha256'], global_W=r['global_W'],
            W_residue=r['global_W']%8, saved_logical_guard=r['strict_error_upper'],
            saved_emitter=h['emitter_upper'], saved_total_certificate=h['total_upper'],
            precision_exponent=r['precision_exponent'], empty_word=(r['sequence']==''),
            metadata_only_comparisons={arm:{axis:dict(
                logical_guard_alone_fits_saved_cap=F(r['strict_error_upper'])<=cap,
                saved_guard_plus_emitter_fits_saved_cap=F(h['total_upper'])<=cap)
                for axis,cap in armcaps.items()} for arm,armcaps in caps.items()})
        c0.append(row);words[r['theta_hex']]=row
    assert len(c0)==299
    basic, basic_archive = [], archive(D3+'basic_records.tar.xz')
    for m in read(D3+'BASIC_WORDS_MANIFEST.json')['rows']:
        raw = basic_archive[m['record']]
        assert hashlib.sha256(raw).hexdigest()==m['record_sha256']
        r=json.loads(raw);ns['validate_record'](r)
        assert r['strict_error_upper']==m['strict_error_upper']
        basic.append(dict(j=m['j'], theta_hex=r['theta_hex'], record_key=r['key'],
            saved_logical_guard=r['strict_error_upper'], global_W=r['global_W'], W_residue=r['global_W']%8,
            precision_exponent=r['precision_exponent'], archive=D3+'basic_records.tar.xz',
            member=m['record'], sequence_sha256=r['sequence_sha256'],
            logical_guard_alone_fits_Y_cap={arm:F(r['strict_error_upper'])<=armcaps['Y']
                                          for arm,armcaps in caps.items()}))
    assert [r['j'] for r in basic]==list(range(-36,7))
    role_certs = read(B4+'ROLE_WORD_CERTIFICATES.json')['rows']
    cert_summary = [dict(theta_hex=r['positive_literal_hex'],
        quantization=r['positive']['quantization_upper'], basic_guard_sum=r['positive']['word_error_upper'],
        emitter=r['positive']['emitter_upper'], saved_total=r['positive']['strict_operator_Frobenius_upper'],
        sequence_sha256=r['positive']['positive_sequence_sha256'],
        global_W=r['positive']['global_W_signed'], W_residue=r['positive']['global_W_signed']%8)
        for r in role_certs]
    phase = read(CT0+'PHASE_SUPPORT.json')
    event = next(r for r in phase['event_phase'] if r['r']==32)
    role_rows = [r for r in coverage['role_rows'] if r['m']==17 and r['arm'] in ('Bprime','reference')]
    literals = {}
    for arm in ('Bprime','reference'):
        for role in ('scalar_or_vacuum_correction','RTE_central_rotation'):
            matched=[r for r in role_rows if r['arm']==arm and r['role']==role]
            literals[arm+':'+role]=dict(signed_numeric_keys=sorted({k for r in matched for k in r['numeric_keys']}),
                symbolic_keys=sorted({k for r in matched for k in r['symbolic_keys']}))
    # Freeze representative EXISTING words only; no new word, angle or policy.
    representatives = []
    for residue in sorted({r['W_residue'] for r in c0}):
        representatives.append(dict(min((r for r in c0 if r['W_residue']==residue), key=lambda r:r['theta_hex']),
                                    selection='lexicographic smallest positive literal within this existing W residue'))
    if not any(r['theta_hex']=='0x0.0p+0' for r in representatives):
        representatives.append(dict(words['0x0.0p+0'],selection='fixed empty C0 identity'))
    prior = read('artifacts/pr_reference_n3b1b0_runtime/2026-11-08/PHASE_TESTS.json')
    tested = {r['word_sha256'] for r in prior['written_word_tests']}
    for row in representatives:
        row['existing_N3_written_word_test_same_sequence'] = row['sequence_sha256'] in tested
    extras = []
    for key in ['0x1.08ba32d420f2cp+2','0x1.921fb54442d18p+2']:
        r=next(r for r in role_certs if r['positive_literal_hex']==key)
        extras.append(dict(theta_hex=key, source=B4+'ROLE_WORD_CERTIFICATES.json',
            sequence_sha256=r['positive']['positive_sequence_sha256'], global_W=r['positive']['global_W_signed'],
            W_residue=r['positive']['global_W_signed']%8, already_N4B_local_tested=True))
    # No quantum module is imported: inspect source control/phase boundaries.
    source_functions=[]
    for path,names in {
        'src/trottertracks/track_c2_b_verify.py':['lower_native'],
        'src/trottertracks/pr_reference_common_word_consumer.py':['consume_native'],
        'src/trottertracks/track_c_h4_synthesis.py':['strict_guard','exact_word_matrix'],
        'src/trottertracks/pr_reference_dyadic_dictionary.py':['round_index'],
    }.items():
        raw=blob(path)
        for n in ast.parse(raw).body:
            if isinstance(n,ast.FunctionDef) and n.name in names:
                calls=[x.func.attr for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
                if n.name in ('lower_native','consume_native'):assert 'control' not in calls
                source_functions.append(dict(path=path,function=n.name,
                    source_sha256=hashlib.sha256(ast.get_source_segment(raw.decode(),n).encode()).hexdigest(),
                    new_control_calls=calls.count('control')))
    primary=read('artifacts/pr_reference_n3b1a_emitter_static/2026-11-08/SOURCE_REFERENCES.json')['qiskit']
    py=next(r for r in primary['python_source_byte_matches'] if r['path']=='qiskit/circuit/quantumcircuit.py')
    installed=Path(primary['installed_python_root'])/'qiskit/circuit/quantumcircuit.py'
    current=installed.read_bytes()
    assert hashlib.sha256(current).hexdigest()==py['sha256']
    assert hashlib.sha1(b'blob '+str(len(current)).encode()+b'\0'+current).hexdigest()==py['sha']
    for path in ['scripts/account_pr_reference_n4c.py','scripts/verify_pr_reference_n3b1b0_runtime.py',
        'src/trottertracks/pr_reference_ct4.py','src/trottertracks/pr_reference_ct1.py',
        'src/trottertracks/pr_reference_bprime.py','src/trottertracks/pr_reference_interference_pilot.py',
        'docs/research/pr_reference_n3b1a_emitter_static_20261108/PHASE_ERROR_BOUND.md',
        'docs/research/pr_reference_n3b1b0_runtime_20261108/EMITTER_BOUND_ASSESSMENT.md',
        'docs/research/pr_reference_n4b_consumer_20261108/PHASE_ACCOUNTING.md',
        'docs/research/pr_reference_n5a_static_20261108/REPORT.md']:
        blob(path)
    summary=dict(C0_W_residue_histogram=dict(sorted(Counter(r['W_residue'] for r in c0).items())),
        C0_logical_guard_alone_Y_cap_matches={arm:sum(r['metadata_only_comparisons'][arm]['Y']['logical_guard_alone_fits_saved_cap'] for r in c0) for arm in caps},
        C0_saved_guard_plus_emitter_Y_cap_matches={arm:sum(r['metadata_only_comparisons'][arm]['Y']['saved_guard_plus_emitter_fits_saved_cap'] for r in c0) for arm in caps},
        basic_guard_alone_Y_cap_matches={arm:sum(r['logical_guard_alone_fits_Y_cap'][arm] for r in basic) for arm in caps})
    save('WORD_EVIDENCE.json',dict(status='SAVED_METADATA_ONLY',summary=summary,
        C0=c0,basic43=basic,saved_role85_components=cert_summary,
        subset_count_is_not_physical_multiplicity=True, no_new_error_sum_or_guard=True))
    save('M17_INPUTS.json',dict(condition=condition,saved_vacuum=m17['saved_vacuum'],
        fixed_dictionary_parameters=parameters,saved_dictionary_budget=budget,
        actual_saved_literals=literals,event32_saved_superset=event,
        original_shell=read(C4+'SHELL_ERROR_ACCOUNTING.json'),
        scope='saved fields copied; no m17 RPE or synthesis error evaluated'))
    save('NEXT_DIAGNOSTIC_INPUTS.json',dict(status='PROPOSED_NOT_EXECUTED', C0_representatives=representatives,
        existing_composites=extras, signs=[1,-1],
        prefix_phase_labels_from_existing_N3=['zero','one_tenth','prev_tau','tau_via_negative_subnormal'],
        local_positions=['before_CX','after_CX'],
        no_new_synthesis_guard_precision_or_policy=True,
        reuse_old_identical_tests_before_any_new_local_call=True))
    save('CHECKS.json',dict(status='PASS_STATIC_TECHNICAL_CHECKS',C0=299,basic=43,role_certificates=85,
        source_functions=source_functions, guard_metadata_only=True,
        installed_primary_source_byte_match=True, no_control_after_word_in_fixed_emitter_or_consumer=True,
        new_error_bounds=0,new_guard_synthesis_provider_emitter_circuit_compile_RPE_calls=0,
        new_precision_policy_or_emitter_adopted=False, saved_errors_not_lower_bounds=True))
    save('SOURCE_REFERENCES.json',dict(base_commit=BASE,files=REFERENCES,
        primary_Qiskit=dict(commit=primary['commit'],path=str(installed),sha256=py['sha256']),
        helper='scripts/audit_pr_reference_n5b0_static.py',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_functions='CT-0 positive / validate_record metadata AST only; N5-A extraction pattern',
        techniques='hash/schema, W integer residue, saved guard/cap comparisons; no new error, price, shot or angle generation'))
    save('STOP.json',dict(status='MANDATORY_STOP',reason='m17 static feasibility and proof obligations documented',
        m17_CT_RPE='UNQUALIFIED', next_owner='GPT/user',automatic_N5B1_execution=False))
    print(json.dumps(dict(status='STATIC_COMPLETE',summary=summary)))


if __name__=='__main__':
    main()
