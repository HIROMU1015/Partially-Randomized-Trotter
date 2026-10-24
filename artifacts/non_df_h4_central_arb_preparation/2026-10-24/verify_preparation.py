"""Read-only metadata verification; never imports Qiskit, Arb or a runner."""
import ast
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
KEY = 'h4_central_arb_preparation_20261024'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify():
    checks = []
    def check(value, name):
        if not value: raise AssertionError(name)
        checks.append(name)
    old = read(BASE/'preservation_before.json')
    check(all(sha(ROOT/p)==s for p,s in old.items()),'OLD_EVIDENCE_SHA_UNCHANGED')
    inputs = read(BASE/'inputs.json')['files']
    check(all(sha(ROOT/p)==s for p,s in inputs.items()),'SAVED_INPUT_SHA_UNCHANGED')
    prior = read(BASE/'prior_manifest_rows.json')
    manifest = read(ROOT/'artifacts/validation_manifest.json')
    check(set(manifest)==set(prior)|{KEY},'ONLY_ONE_NEW_MANIFEST_ROW')
    check(all(digest(manifest[k])==s for k,s in prior.items()),'OLD_MANIFEST_ROWS_UNCHANGED')
    contract = read(BASE/'pilot_contract_DRAFT_DISABLED.json')
    check(all(contract[k] is False for k in ('execution_authorized','dispatch_enabled','grant_valid')),'CONTRACT_DISABLED')
    check(all(contract[k] is None for k in ('grant','scientific_runner_SOURCE','launch_command')),'NO_GRANT_OR_LAUNCH_SOURCE')
    precision = contract['precision']
    check(precision['schedule_bits']==[64,96,128] and precision['max_compiled_state_walk_attempts']==3 and precision['logical_cases']==1,'FINITE_MINIMAL_FUTURE_SCOPE')
    check(precision['new_C_target_state_actions']==precision['new_input_QPY_state_actions']==0,'NO_EXTRA_TARGET_OR_INPUT_ACTIONS')
    check(not precision['width_is_mandatory'] and not precision['width_is_new_allowance'],'WIDTH_NOT_ALLOWANCE_CHANGE')
    p = read(BASE/'PF_cosine_program_NO_EXECUTION.json')
    inspection = read(BASE/'structural_inspection.json')
    check(sha(BASE/'PF_cosine_program_NO_EXECUTION.json')==inspection['descriptor_sha256'],'DESCRIPTOR_SHA')
    check(p['domain']=='SAVED_QPY_NO_EXECUTION' and p['n_system']==8 and p['ancilla']==8,'SAVED_PF_LAYOUT_DOMAIN')
    check(p['axis']=='cosine' and p['origin']['state_actions']==0,'STRUCTURAL_DECODE_ONLY')
    check(p['origin']['qpy_sha256']==contract['candidate']['compiled_sha256'] and p['origin']['fingerprint']==contract['candidate']['fingerprint'],'QPY_CONTRACT_BINDING')
    check(inspection['histogram']==dict(rz=15028,sx=7170,cx=26768) and len(p['instructions'])==48966,'NATIVE_INSTRUCTION_COUNTS')
    for g in p['instructions']:
        checkable = g['name'] in ('rz','sx','x','cx') and len(set(g['qubits']))==len(g['qubits']) and all(0<=q<=8 for q in g['qubits'])
        if not checkable: raise AssertionError('DESCRIPTOR_NATIVE_WIRES')
        for theta in g['params_exact']:
            if Q(float(Q(theta)))!=Q(theta): raise AssertionError('EXACT_BINARY64_DESCRIPTOR')
    checks.append('ALL_NATIVE_WIRES_AND_EXACT_BINARY64_PARAMETERS')
    for name,original in [('PF_C_target_interval_QUOTED.json','PF_C_target_saved.json'),('state_identity_QUOTED.json','state_identity.json')]:
        check(read(BASE/name)==read(ROOT/'artifacts/non_df_h4_central_r0_empirical/2026-10-24/run01'/original),'UNCHANGED_QUOTE_'+name)
    tests = read(BASE/'artificial_tests.json')
    check(tests['unittest_total']==25 and all(r['exit_code']==0 for r in tests['records']),'ARTIFICIAL_25_TESTS_PASS')
    stress = read(BASE/'artificial_long_sequence.json')
    check(stress['native_instructions']==49000 and stress['total_qubits']==3 and not stress['generated_from_H4'],'ARTIFICIAL_LONG_SEQUENCE_SCOPE')
    check(all(r['contains_analytic_identity'] and Q(r['interval_width_exact'])<=Q(1,10**8) for r in stress['records']),'ARTIFICIAL_LONG_SEQUENCE_CONTAINS_KNOWN_RESULT')
    check(not (BASE/'run01').exists() and not list(BASE.glob('*grant*')),'NO_SCIENCE_RUN_OR_GRANT_FILES')
    stop = read(BASE/'STOP.json')
    check(stop['mandatory_STOP'] and not stop['automatic_next_stage'] and stop['real_H4_state_actions']==0,'PREPARATION_STOP')
    # Guard and disabled runner are part of new SOURCE; neither calls an old issuer.
    tree = ast.parse((ROOT/'src/trotterlib/h4_arb_bridge.py').read_text())
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    check(isinstance(functions['execution_request'].body[0],ast.Raise),'NUMERICAL_EXECUTION_REQUEST_REJECTS')
    changed = subprocess.check_output(['git','diff','--name-only','-z','36e993e698953f24415f4e90c4cac199a693ee16'],cwd=ROOT).decode().split('\0')
    allowed = {'PROJECT_MAP.md','VALIDATION_STATUS.md','artifacts/validation_manifest.json','docs/research/研究概要・現状.md',
               'scripts/README.md','src/trotterlib/README.md','docs/README.md'}
    check(all(not f or f in allowed or f.startswith(('artifacts/non_df_h4_central_arb_preparation/','docs/non_df_h4_central_arb_preparation_',
              'docs/research/研究ノート/2026-10-24_H4_Arb_preparation','src/trotterlib/h4_arb_bridge.py','scripts/inspect_h4_arb_qpy.py',
              'scripts/check_h4_arb_artificial.py','tests/test_h4_arb_')) for f in changed),'NO_OLD_SOURCE_OR_EVIDENCE_MODIFICATIONS')
    return dict(status='PASS_PREPARATION_METADATA_ONLY',checks=checks,check_count=len(checks),preserved_files=len(old),
        saved_inputs=len(inputs),prior_manifest_rows=len(prior),real_H4_Arb_evaluations=0,real_H4_state_actions=0,
        science_grants_issued=0,science_grants_consumed=0,new_compile=0,sampling=0,N_G_reanalysis=0,
        local_H4_rho_upper=None,formal_certificate_accepted=False,mandatory_STOP=True)


if __name__ == '__main__': print(json.dumps(verify(),indent=2,sort_keys=True))
