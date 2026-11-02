"""Static evidence/recipe builder. Never imports a scientific SOURCE module."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import h2_common_production_contract as c

OUT=c.ROOT/'artifacts/resource_applicability/track_a_h2_common_production/2026-10-23'
BASE='b18a9ce5cfa824d68a88f2d8050795692ed32ca4'
PREP='artifacts/resource_applicability/track_a_h2_common_native_allowance_preparation/2026-10-23/'
CC='cc7404ed189f8c768ecc50a2066485975c9c68f4'
manifest=[]


def fixed(commit,path):
    b=subprocess.check_output(['git','-C',str(c.ROOT),'show',commit+':'+path])
    manifest.append(dict(commit=commit,path=path,sha256=hashlib.sha256(b).hexdigest(),
        blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),
        url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+commit+'/'+path))
    return b


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for n in ['REPORT.md','NATIVE_DESIGN.md','ALLOWANCE_SCHEMA.md','EXECUTION_PLAN.md',
        'execution_contract_DRAFT.json','deferred_native_recipes_4.json','kernel_bindings.json',
        'diagnostic_coverage_4.json','technical_verification.json','STOP.json']:
        fixed(BASE,PREP+n)
    for p in ['src/trotterlib/h2_common_native_policy.py','scripts/prepare_h2_common_native.py','tests/test_h2_common_native_policy.py']:
        fixed('bcc05ef0a028c6bbd18ccf9c7a260734078ca5bf',p)
    for commit,path in [('366b2fd24f7e86aa82b5b518b6e3774520f65596','artifacts/resource_applicability/track_a_pf_order_reversal_existing_evidence_audit/2026-10-23/REPORT.md'),
        (CC,'artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/REPORT.md'),
        (CC,'artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/REPORT.md'),
        ('e3531956af6a6d359e2f1fb320e9a5179f41eb98','artifacts/resource_applicability/track_a_h2_outer32_metadata_export/2026-10-23/README.md')]:fixed(commit,path)
    bindings=json.loads(fixed(BASE,PREP+'kernel_bindings.json'))
    for row in bindings.values():
        b=fixed(row['commit'],row['path']);assert hashlib.sha256(b).hexdigest()==row['sha256'];ast.parse(b)
    for p in ['scripts/h46_m5_contract.py','scripts/h46_new2_core.py','scripts/h46_pr_pilot_repair.py','scripts/h2_outer32_runtime.py']:
        ast.parse(fixed(c.ACQUISITION,p))
    accounting='src/trottertracks/resource_applicability/h6_matched_accounting_v1.py'
    assert fixed('217c971dd537b1cc7fbc13742bc0f6c263ff7f45',accounting)==fixed(c.PROVIDER,accounting)
    for p in ['src/trottertracks/resource_applicability/ax2b_numerical_accounting.py',
        'src/trotterlib/df_partial_s2.py','src/trotterlib/df_hamiltonian.py','src/trotterlib/pr2_s0_s1_validation.py',
        'src/trotterlib/rte.py','src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py']:
        ast.parse(fixed(c.PROVIDER,p))
    source_paths=['scripts/h2_common_production_contract.py','scripts/h2_common_production_backend.py',
        'scripts/h2_common_production_diagnostics.py','scripts/run_h2_common_production.py',
        'scripts/prepare_h2_common_production.py','tests/test_h2_common_production.py']
    for p in source_paths:ast.parse((c.ROOT/p).read_bytes())
    # Snapshot and receipt bytes only: never np.load or parse a numerical array.
    private=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h2-pr-resource-preparation-20261019/.h2_pr_resource_private/2026-10-19/D_activation01/D_spec_EXECUTION.json')
    inventory=[]
    if private.is_file():
        spec=c.read(private)
        for row in spec['inputs']:
            if row['role'] not in ('F_snapshot','F_receipt'):continue
            p=Path(row['path']);entry=dict(role=row['role'],exists=p.is_file(),locator='KNOWN_ORIGINAL_SPEC_INPUTS; PRIVATE_PATH_NOT_REPUBLISHED')
            if p.is_file():
                before=c.sha(p);entry.update(sha256=before,matches_historical=before==row['sha256'],bytes=p.stat().st_size,post_read_unchanged=c.sha(p)==before)
                assert entry['matches_historical'] and entry['post_read_unchanged']
            inventory.append(entry)
    write('input_byte_checks.json',dict(files=inventory,arrays_loaded=False,private_bodies_exported=False))
    excerpt_path='artifacts/resource_applicability/track_a_pf_order_reversal_h2_common_policy_preparation/2026-10-23/saved_input_excerpts.json'
    old=json.loads(fixed(CC,excerpt_path))['old_reference']
    reuse=[]
    task_path='artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received/task_acceptance_67.json'
    saved_tasks=json.loads(fixed(CC,task_path))['tasks']
    for candidate in c.policy.CANDIDATES[2:]:
        selected=[t for t in saved_tasks if t['task']['candidate']['id']==candidate.id]
        assert len(selected)==(33 if candidate.q==4 else 34)
        rows=[t['result']['saved'] for t in selected]
        assert all(r['input_binding']['hamiltonian']==c.BINDING['hamiltonian'] and r['input_binding']['state']==c.BINDING['state'] for r in rows)
        reuse.append(dict(candidate=candidate.id,saved_tasks=33 if candidate.q==4 else 34,sampling_required_for_diagnostics=False,
            status='BLOCKED_NOT_ACCEPTED_FOR_NEW_SERIES',known=['saved candidate/H/state/signed schedule/cohort receipt mapping'],
            saved_evidence=dict(locator=CC+':'+task_path,
                scientific_SOURCE=sorted({r['SOURCE'] for r in rows}),
                preparation_hashes=sorted({r['preparation_hash'] for r in rows}),
                compiler_settings_hashes=sorted({a['compiler_settings_hash'] for r in rows for a in r['axis_costs']}),
                native_policy_labels=sorted({a['native_policy'] for r in rows for a in r['axis_costs']}),
                event_digests_present=all(bool(r['event_digest']) for r in rows),
                event_body_scope='N2/N4/stratum summary; not full event operation stream',
                conditional_sampling=sorted({r['conditional_sampling'] for r in rows}),
                cohort_pooling=sorted({r['cohort_pooling'] for r in rows})),
            missing=['new operation-stream hash and same-rotation witness','exact event manifest and task receipts',
                'same scalar/wrapper/lowering/compiler closure witness','fresh complete runtime closure and correctness scope acceptance'],
            automatic_resample=False,automatic_recompile=False,
            minimal_remedy='retrieve exact missing receipt/event metadata; if identity differs enumerate affected existing event/axis pairs in a new contract; no automatic 67-task job'))
    write('PR_C_reuse_assessment.json',reuse)
    recipes={branch:dict(budget=c.counts(branch),status='BLOCKED_PENDING_BINDINGS_AND_USER_APPROVAL',
        reference_evaluation=0,new_q_search=0,new_Hamiltonian_generation=0,
        counted_unit='entered API boundaries and registered RTE events; nested entries are not independent experiments') for branch in c.BRANCHES}
    write('call_recipes.json',recipes)
    template=dict(schema='h2_common_production_contract_v1',series=c.SERIES,branch='COMPARE_REUSE_PR',
        source_commit=None,physical=c.policy.PHYSICAL,compiler=c.policy.COMPILER,
        native_policy=c.policy.NATIVE_POLICY,allowance_policy=c.policy.ALLOWANCE_POLICY,
        reference_policy='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC',
        candidates=[asdict(x) for x in c.policy.CANDIDATES],budget=c.counts('COMPARE_REUSE_PR'),
        files=[],source_files=[],acquisition_root=None,provider_root=None,inputs={},
        binding={k:old['binding'][k] for k in ('snapshot','hamiltonian','state')},runtime={},
        run_root=None,reuse_witnesses=[],retry=False,mandatory_STOP=True)
    # Pin a committed implementation only when every file matches that Git tree.
    rev=subprocess.check_output(['git','-C',str(c.ROOT),'log','-1','--format=%H','--',*source_paths]).decode().strip()
    if rev and all(subprocess.run(['git','-C',str(c.ROOT),'cat-file','-e',rev+':'+p],capture_output=True).returncode==0 for p in source_paths):
        assert all(subprocess.check_output(['git','-C',str(c.ROOT),'show',rev+':'+p])==(c.ROOT/p).read_bytes() for p in source_paths)
        template['source_commit']=rev
    write('contract_TEMPLATE_BLOCKED.json',template)
    write('blockers.json',dict(execution_ready=False,items=[
        'user adoption of native/allowance/reference and instrumentation definitions',
        'complete used runtime/import closure, fresh byte seals and canonical source/input paths',
        'PR C full identity proof for COMPARE_REUSE_PR; no substitution with proposed labels',
        'reference meaning/old empirical uncertainty acceptance; new reference is a separately blocked policy branch',
        'optional PR control seeds and explicit budget approval if that branch is selected'],
        precise_budget_available_for=list(c.BRANCHES),live_grants_created=0))
    unique={(r['commit'],r['path']):r for r in manifest};write('source_manifest.json',list(unique.values()))
    (OUT/'SOURCE_INDEX.md').write_text('# 固定SOURCE索引\n\n'+ '\n'.join('- ['+r['path']+']('+r['url']+') commit `'+r['commit']+'` SHA256 `'+r['sha256']+'`' for r in unique.values())+'\n')
    print(json.dumps(dict(fixed_source_records=len(unique),source_commit=template['source_commit'],branches=len(recipes),science_calls=0,execution_ready=False)))


if __name__=='__main__':run()
