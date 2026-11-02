"""Build dormant technical-preparation metadata. No science or production launch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pf_direct_compile_contract as c
from check_pf_direct_compile_preparation import ROOT, PLAN, OUT

BASE = c.PLAN_COMMIT
CODE = ('scripts/pf_direct_compile_contract.py','scripts/pf_direct_compile_adapter.py',
        'scripts/check_pf_direct_compile_preparation.py','scripts/prepare_pf_direct_compile_technical.py',
        'tests/test_pf_direct_compile_preparation.py')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    preserved={str(p.relative_to(ROOT)):c.sha(p) for p in PLAN.iterdir() if p.is_file()}
    # Compare current bytes with the accepted plan's immutable Git object.
    for path,digest in preserved.items():
        raw=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==digest,'PLAN_BYTES_CHANGED:'+path)
    save('preservation.json',dict(base_commit=BASE,files=preserved,
        old_SOURCE_results_grants_ledgers_markers_STOP_modified=False))
    proposal=PLAN/'proposed_compile_axes.csv'
    rows=c.parse_proposals(proposal.read_bytes(),c.sha(proposal))
    save('scope_acceptance.json',dict(date_JST='2026-10-20',plan_commit=BASE,
        user_instruction='計画は今回の研究目的に適合しています。まず技術準備へ進み、実際の788軸の取得範囲と科学実行は別途承認する方針でよいです。',
        plan_suitable_for_research=True,technical_preparation_authorized=True,
        acquisition_scope_approved=False,scientific_execution_authorized=False,
        proposed_axes=788,selected_axes=0,conditional_48_axes_automatically_included=False))
    save('execution_contract_DRAFT.json',c.draft(rows,c.sha(proposal),c.sha(PLAN/'system_bindings.json')))
    specs=json.loads((PLAN/'source_inventory.json').read_text())
    by_id={r['id']:r for r in specs}
    backend_rows=[]
    for system,label,functions,excluded in [
        ('H2_DF3','H2_COMPILE_CODE',['context'],['execute: fixed D recovery scope, old grant and accounting']),
        ('H4_DF10','H4_COMPILE_CODE',['h_compile','setup'],['setup: scientific primitives/actions; not invoked','execute: signal/acquisition/accounting pipeline']),
        ('H4_LEGACY_DF12','H4_LEGACY_COMPILE_CODE',['compile_axes','deterministic_task'],['confirmation_task: PR sampling','deterministic_account: shot accounting and PF selection']),
        ('H6_DF19','H6_COMPILE_CODE',['load','control','compile_pair'],['direct: hard-coded p11/q4/S4','compile_pair predictions/2-percent STOP: do not call with predictions']),
        ('H8_DF28','H8_COMPILE_CODE',['load','measured_worker'],['evaluate: old selected cases/grant/caps/shared-model STOP'])]:
        original=by_id[label]
        raw=subprocess.check_output(['git','show',original['commit']+':'+original['path']],cwd=ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==original['sha256'],'ORIGINAL_SOURCE_BYTES_CHANGED')
        tree=ast.parse(raw);defs={node.name for node in tree.body if isinstance(node,ast.FunctionDef)}
        c.require(set(functions)<=defs,'REFERENCED_FUNCTION_NOT_FOUND')
        backend_rows.append(dict(system=system,origin=original,functions_inspected=functions,
            excluded_old_driver_paths=excluded,
            new_adapter='scripts/pf_direct_compile_adapter.py:NativeKernel',
            reusable_provider_APIs=['build_deterministic_native','build_native_hadamard_wrapper',
                'canonical_qiskit_circuit_fingerprint','transpile_and_measure_cost'],
            molecular_loader_bound=False,profile_runtime_live_verified=False,
            independent_control_oracle_for_selected_scope_bound=False,
            worker_spawner_and_wait4_witness_bound=False,
            provider_infinite_instruction_allowance_compatibility_exercised=False,
            production_ready=False))
    save('backend_binding_DRAFT.json',dict(status='STATIC_API_MAPPING_AND_FAKE_PROVIDER_VALIDATION_ONLY',backends=backend_rows))
    save('parallel_dispatch_DRAFT.json',dict(status='ADMISSION_POLICY_IMPLEMENTED_PRODUCTION_SPAWNER_NOT_BOUND',
        CPU_limit=None,RAM_limit=None,worker_limit=None,arbitrary_q_limit=None,
        admission_function='pf_direct_compile_contract.admissible_pairs',
        resource_source='fresh permitted affinity/CPU, effective memory and FS; contract-bound per-task estimates',
        no_estimate_fabricated=True,within_pair='share evolution, cosine then sine',
        across_pairs='future owned processes, admission-driven; no existing job control changes',
        transient_PSI_or_CPU_competition='observe; do not kill healthy workers',
        scientific_or_integrity_failure='STOP affected task/dependency only',
        real_worker_processes_started=0))
    save('readiness_boundaries.json',dict(complete=['plan suitability recorded','immutable 788-axis proposal byte binding',
        'exact pair-scope selection and explicit execution rejection','before-call reservation and exclusive output prototype',
        'provider API adapter with all compiled metrics','owned-result acceptance rejects missing exit/reap',
        'dynamic admission policy','synthetic tests and metadata preflight'],
        not_started=['select actual acquisition scope','issue valid grant','bind actual molecular loaders and independent control oracles',
            'seal per-series input/provider/runtime/profile closure','bind owned worker/wait4 implementation and live resource gate',
            'exercise any molecular input, circuit, control probe or compile'],
        next_action='User decides actual acquisition scope; Codex completes runtime-specific binding/seal; scientific execution requires separate explicit approval.',
        no_final_evidence_freeze=True,no_detailed_scientific_review=True))
    save('MANDATORY_STOP.json',dict(status='MANDATORY_STOP_TECHNICAL_PREPARATION_ONLY',
        proposed_axes=788,selected_axes=0,scientific_execution_authorized=False,grant_issued=False,
        real_input_decodes=0,real_circuit_builds=0,real_control_probes=0,real_compiles=0,
        new_sampling=0,new_fit=0,new_cost_reaccounting=0,next_stage_authorized=False))
    save('technical_source_closure.json',dict(kind='TECHNICAL_CODE_BYTES_NOT_A_SCIENTIFIC_EXECUTION_SEAL',
        files={path:c.sha(ROOT/path) for path in CODE},scientific_SOURCE=None,grant_issued=False))
    print('DORMANT_TECHNICAL_PACKET_CREATED; proposed=788 selected=0 production dispatch disabled')


if __name__=='__main__':prepare()
