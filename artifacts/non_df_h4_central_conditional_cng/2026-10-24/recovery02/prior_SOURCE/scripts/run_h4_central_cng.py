#!/usr/bin/env python3
"""One-shot authority for the saved central Pauli C/N/G package only."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Reuse the old scalar runner's private namespace, runtime discovery and C IO.
spec = importlib.util.spec_from_file_location('central_cng_legacy_runner', ROOT/'scripts/run_h4_conditional_cng.py')
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
C = legacy.C
from h4_cng_private import h4_central_cng as A
BASE = A.BASE
os.environ['MPLCONFIGDIR'] = str(BASE/'mplconfig')
LIMITS = dict(input_binding_batch=1, pauli_10_conditions_20_axes_with_guard_faces=1,
              paired_G_statistics_batch=1, comparisons_15_with_guard_boxes=1)
CODE = legacy.CODE + ['scripts/run_h4_central_cng.py', 'src/trotterlib/h4_central_cng.py',
                      'tests/test_h4_central_cng.py', 'src/trotterlib/h2_non_df_pilot.py',
                      'artifacts/non_df_h4_central_conditional_cng/2026-10-24/prepare_inputs.py']
SCHEMA = dict(version='H4_CENTRAL_CONDITIONAL_CNG_V1', conditions=10, primary_axes=20,
              analytic_guard_faces=20, comparisons=15, paired_prices=257,
              primary_scenario='rho_new_cos=rho_new_sin=0 HYPOTHETICAL_ONLY',
              required_result_fields=['case','epsilon_sig','N_cos','N_sin','N_total','prices_exact',
                                      'G','G_exact','G_SD','G_SE','G_nominal95','actual_N','actual_G','actual_rho_new'],
              actual_rho_default=None, actual_N_G_default=None, formal_certificate_accepted=False)
FORBIDDEN = dict(compile=0, circuit_build=0, QPY_operator_load=0, sampling=0, RNG_replay=0,
                 SCF_FCI_DF=0, state_signal=0, C_L_recompute=0, numerical_rho=0,
                 new_order4_diagnosis=0, quantum_shots=0, D3=0, DF_N_G_recompute=0,
                 certificate_acceptance=0, old_evidence_rewrite=0)


def verify(seal):
    C.require(seal['kind']=='H4_CENTRAL_CNG_SEAL_V1', 'SEAL_KIND')
    C.require(C.sha(sys.executable)==seal['python_sha256'], 'PYTHON')
    C.hashes({str(ROOT/k):v for k,v in seal['code_files'].items()})
    for p, h in seal['code_files'].items():
        b = subprocess.check_output(['git','show',seal['SOURCE']+':'+p], cwd=ROOT)
        C.require(hashlib.sha256(b).hexdigest()==h, 'COMMITTED_SOURCE:'+p)
    for p, h in seal['bound_manifests'].items():
        C.require(C.sha(BASE/p)==h, 'BOUND_MANIFEST:'+p)
    C.hashes(C.read(BASE/'runtime.json')['files'])
    C.hashes({str(ROOT/k):v for k,v in C.read(BASE/'inputs.json')['files'].items()})
    C.hashes({str(ROOT/k):v for k,v in C.read(BASE/'preservation_before.json').items()})
    C.require(C.read(BASE/'output_schema.json')==SCHEMA, 'OUTPUT_SCHEMA')


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(), 'ALREADY_PREPARED')
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    code = {p:C.sha(ROOT/p) for p in CODE}
    for p in code:
        C.require(subprocess.check_output(['git','show',source+':'+p],cwd=ROOT)==(ROOT/p).read_bytes(), 'UNCOMMITTED_CODE')
    C.save(BASE/'runtime.json', legacy.runtime())
    C.save(BASE/'output_schema.json', SCHEMA)
    bound = ['runtime.json','output_schema.json','inputs.json','preservation_before.json',
             'approval.txt','github_verification.json']
    seal = dict(kind='H4_CENTRAL_CNG_SEAL_V1',SOURCE=source,code_files=code,
                python_sha256=C.sha(sys.executable),bound_manifests={p:C.sha(BASE/p) for p in bound},
                real_H4_scalar_calls_before_seal=0,UTC=C.utc())
    C.save(BASE/'seal.json',seal)
    old = C.read(ROOT/A.OLD/'active_contract.json')
    contract = dict(kind='H4_CENTRAL_CNG_ACTIVE_V1',version=SCHEMA['version'],
        base_commit='564cd71936fa27225624b00a7d5d1f12dc4cc6dc', SOURCE=source,
        scientific_SOURCE=old['scientific_SOURCE'],
        central_SOURCE='74515d8eb4f073c40fc783681a3775dbca89e9b1',
        price_SOURCE='a37232f85812688c3b9ab957c5073c5ce143b17e', fixed_model=old['fixed_model'],
        accounting=old['accounting'],epsilon_sig=list(A.EPS),cases=list(A.CASES),axes=list(A.AXES),
        comparisons=[list(p) for p in A.PAIRS],scope=SCHEMA, authority_batch_limits=LIMITS,
        forbidden_calls=FORBIDDEN,seal_sha256=C.sha(BASE/'seal.json'),
        approval_sha256=C.sha(BASE/'approval.txt'),execution_authorized=True,
        real_rho_new=None, rho_zero_hypothetical_only=True, DF_rows='saved only; no evaluator',
        statistical_plan='one fixed batch; paired within each Pauli trajectory; no Pauli/DF index pairing',
        guard='independent method/axis residuals; monotone ceil endpoints only; no extra scenario grid',
        technical_recovery='diagnosed technical only; preserve successes/failures; new SOURCE/seal/namespace/grant if code changes',
        formal_certificate_accepted=False,automatic_retry=False,mandatory_STOP=True,
        resources='light saved scalar workload: one worker, no CPU/RAM/affinity ceilings or shared setting changes')
    C.save(BASE/'active_contract.json',contract)
    verify(seal)
    ns = BASE/'run01'; ns.mkdir()
    grant = dict(kind='H4_CENTRAL_CNG_ONE_SHOT',namespace=str(ns),contract_sha256=C.sha(BASE/'active_contract.json'),
                 seal_sha256=C.sha(BASE/'seal.json'),approval_sha256=C.sha(BASE/'approval.txt'),one_shot=True)
    grant['grant_id'] = C.digest(grant)
    C.save(ns/'grant.json',grant)
    print(json.dumps(dict(status='SEALED_NOT_EXECUTED',SOURCE=source,grant_id=grant['grant_id'])))


def check(consume=False):
    ns = BASE/'run01'; c=C.read(BASE/'active_contract.json'); s=C.read(BASE/'seal.json'); g=C.read(ns/'grant.json')
    C.require(c['kind']=='H4_CENTRAL_CNG_ACTIVE_V1' and c['execution_authorized'], 'AUTHORITY')
    C.require(c['SOURCE']==s['SOURCE'] and c['scope']==SCHEMA and c['authority_batch_limits']==LIMITS and c['forbidden_calls']==FORBIDDEN, 'SCOPE')
    old = C.read(ROOT/A.OLD/'active_contract.json')
    C.require(all(c[k]==old[k] for k in ('scientific_SOURCE','fixed_model','accounting')), 'FIXED_SCIENCE')
    C.require(c['epsilon_sig']==list(A.EPS) and c['cases']==list(A.CASES) and c['axes']==list(A.AXES) and c['comparisons']==[list(p) for p in A.PAIRS], 'FIXED_TARGETS')
    C.require(c['mandatory_STOP'] and not c['automatic_retry'] and not c['formal_certificate_accepted'] and c['real_rho_new'] is None, 'STOP_RESIDUAL_CERTIFICATE')
    C.require(g['kind']=='H4_CENTRAL_CNG_ONE_SHOT' and g['namespace']==str(ns) and g['one_shot'], 'GRANT_NAMESPACE')
    C.require(g['grant_id']==C.digest({k:v for k,v in g.items() if k!='grant_id'}), 'GRANT_DIGEST')
    C.require(g['contract_sha256']==C.sha(BASE/'active_contract.json') and g['seal_sha256']==C.sha(BASE/'seal.json')==c['seal_sha256'], 'GRANT_BINDING')
    C.require(g['approval_sha256']==c['approval_sha256']==s['bound_manifests']['approval.txt'], 'APPROVAL_BINDING')
    verify(s)
    C.require(not (ns/'STOP.json').exists(), 'ALREADY_STOPPED')
    if consume:
        C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id'],UTC=C.utc(),nonrefundable=True))
    else:
        C.require(C.read(ns/'CONSUMED.json')['grant_id']==g['grant_id'], 'NOT_CONSUMED')
    return c,s


def worker():
    c,s=check(); ns=BASE/'run01'
    C.require((ns/'LAUNCH_CLAIM.json').is_file(), 'NO_LAUNCH_CLAIM')
    C.save(ns/'WORKER_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    A.run(ns,c,C.CallBudget(ns,LIMITS))
    verify(s)


def launch():
    c,s=check(consume=True); ns=BASE/'run01'
    C.save(ns/'LAUNCH_CLAIM.json',dict(pid=os.getpid(),UTC=C.utc()))
    # Same native witness; only the new-receipt scope string is replaced.
    tree=ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='witness')
    for n in ast.walk(node):
        if isinstance(n,ast.Constant) and n.value=='NEW_C_WORKER_ONLY':
            n.value='NEW_CENTRAL_CONDITIONAL_CNG_ONLY'
    env=dict(vars(C)); exec(compile(ast.Module(body=[node],type_ignores=[]),'<reused scoped witness>','exec'),env)
    status=env['witness'](ns/'native',[sys.executable,'-B',str(Path(__file__).resolve()),'--worker'])
    validation_error=None
    try:
        if status==0:
            verify(s)
            C.require(C.read(ns/'call_ledger.json')['used']==LIMITS,'INCOMPLETE_BATCHES')
            C.require(C.read(ns/'result_summary.json')['status']=='CENTRAL_CONDITIONAL_CNG_COMPLETE_MANDATORY_STOP','INCOMPLETE_OUTPUT')
    except Exception as e:
        status=1; validation_error=str(e)
    C.save(ns/'STOP.json',dict(status='COMPLETE_MANDATORY_STOP' if status==0 else 'FAILED_NO_AUTOMATIC_RETRY',
        native_exit=C.read(ns/'native/worker_native_exit.json'),all_owned_reaped=C.read(ns/'native/STOP.json')['all_owned_reaped'],
        validation_error=validation_error,original_H4_native_verified=False,automatic_retry=False,UTC=C.utc()))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','launch','worker'): g.add_argument('--'+mode, action='store_true')
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.launch: sys.exit(launch())
    else: worker()
