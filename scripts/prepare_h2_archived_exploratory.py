"""Seal a saved-value comparison without evaluating allowance, shots, G or ranks."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import h2_archived_exploratory as c

ROOT=c.ROOT
OUT=ROOT/c.OUT_DIR
ORIGIN=ROOT.parent/'h2-diagnostics-binding-assets-20261023'


def build(source):
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    source_rows=[];files={}
    def add(path,role):
        p=Path(path).resolve();files[str(p)]=dict(path=str(p),sha256=c.sha(p),role=role)
    for path,(_,raw) in zip(c.NEW+c.REUSED,c.seal.git_batch(source,c.NEW+c.REUSED)):
        p=ROOT/path;c.require(p.read_bytes()==raw,'COMMITTED_NEW_AND_REUSED_SOURCE')
        if path in c.REUSED:c.require(raw==c.recovery.git_bytes(c.BASE,path),'UNCHANGED_REUSED_IMPLEMENTATION')
        source_rows.append(dict(root='runner',path=path,local_path=str(p.resolve()),sha256=c.sha(p)))
        add(p,'SOURCE:runner')
    c.require({p.name for p in OUT.iterdir()}<={'cache','technical_tests.txt','test_results.json','attempts'},'FRESH_OUTPUT_NAMESPACE')
    c.ASSETS.mkdir(exist_ok=False)
    for path,(_,raw) in zip(c.PROVIDER_PATHS,c.seal.git_batch(c.PROVIDER,c.PROVIDER_PATHS)):
        p=c.ASSETS/'provider'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        source_rows.append(dict(root='provider',path=path,local_path=str(p.resolve()),sha256=c.sha(p)))
        add(p,'SOURCE:original_accounting_provider')
    def publish(name,value):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
        c.require('/home/' not in raw and '/usr/' not in raw,'NO_PRIVATE_ABSOLUTE_PATH_IN_PUBLIC_PROJECTION')
        with (OUT/name).open('x') as f:f.write(raw)
    original_cfg=c.read(ORIGIN/'DIAGNOSTICS_contract.json')
    c.require(c.digest(original_cfg)==c.accepted.ORIGIN_CONTRACT,'PINNED_DIAGNOSTICS')
    origin_run=Path(original_cfg['run_root'])
    receipt=c.seal.accepted_diagnostics_receipt(origin_run,original_cfg,completion_sha256=c.accepted.ORIGIN_COMPLETION)
    add(ORIGIN/'DIAGNOSTICS_contract.json','INPUT:original_diagnostics_contract')
    for p in [origin_run/'completion.json',origin_run/'CONSUMED.json',*sorted((origin_run/'ledger').glob('*.json')),
              *[origin_run/rel for rel in receipt['files']]]:add(p,'INPUT:accepted_diagnostics')
    cost_files={}
    for key,name in [('PR_costs','saved_PR_cost_objects_134.json'),('Full_costs','saved_common_PF_cost_objects_4.json')]:
        raw=c.recovery.git_bytes(c.B,c.B_DIR+name)
        p=c.ASSETS/'inputs'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(raw);add(p,'INPUT:fixed_saved_costs')
        cost_files[key]=dict(commit=c.B,path=c.B_DIR+name,sha256=c.sha(p),bytes=len(raw),local_path=str(p.resolve()))
    b2_files={}
    for name in ('REPORT.md','integrated_axis_identity_138.json','policy2a_conditions.json','correctness_scope.json','scientific_native_compiler_join.json'):
        raw=c.recovery.git_bytes(c.BASE,c.B2_DIR+name)
        b2_files[name]=dict(commit=c.BASE,path=c.B2_DIR+name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    manifest=dict(schema=c.SERIES+'_INPUTS',B2_identity_commit=c.BASE,physical=c.policy.PHYSICAL,
        candidates=[asdict(x) for x in c.policy.CANDIDATES],cost_files=cost_files,
        origin_completion=c.accepted.ORIGIN_COMPLETION,origin_contract=c.accepted.ORIGIN_CONTRACT,
        origin_SOURCE=c.accepted.ORIGIN_SOURCE,
        diagnostics={cid:dict(locator='${ACCEPTED_DIAGNOSTICS}/results/'+cid+'_diagnostics.json',
            sha256=receipt['files']['results/'+cid+'_diagnostics.json']) for cid in [x.id for x in c.policy.CANDIDATES]},
        identity_evidence=b2_files,identity_audit_repeated=False,signal_reference_recomputed=False)
    manifest_path=c.ASSETS/'input_manifest_PRIVATE.json';c.save(manifest_path,manifest);add(manifest_path,'INPUT:manifest')
    # Reuse the existing runtime index as a bounded stdlib file list; do not import its scientific packages.
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    probe='import sys,sysconfig,json;print(json.dumps(dict(stdlib=sysconfig.get_path("stdlib"),sys_path=sys.path,executable=sys.executable,modules=list(sys.modules))))'
    info=json.loads(subprocess.check_output([str(c.seal.PYTHON),'-I','-S','-B','-c',probe],env=env))
    c.require(not any(n.split('.')[0] in ('numpy','qiskit','scipy') for n in info['modules']),'STDLIB_ONLY_RUNTIME_PROBE')
    old_runtime=c.read(original_cfg['runtime']['closure_manifest']);stdlib=Path(info['stdlib']).resolve()
    runtime_paths={Path(r['path']).resolve() for r in old_runtime['files'] if
        Path(r['path']).resolve().is_relative_to(stdlib) and
        not Path(r['path']).resolve().is_relative_to(stdlib/'site-packages')}
    runtime_paths.add(c.seal.PYTHON.resolve())
    elf,elf_rows,missing=c.seal.elf_closure(runtime_paths);c.require(not missing,'NUMERICAL_RUNTIME_ELF_CLOSURE')
    runtime_paths.update(elf)
    def runtime_row(p):return dict(path=str(p),sha256=c.sha(p),role='runtime:stdlib_or_ELF')
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for row in pool.map(runtime_row,sorted(runtime_paths)):files[row['path']]=row
    runtime=dict(python_target=str(c.seal.PYTHON.resolve()),python_sha256=c.sha(c.seal.PYTHON.resolve()),
        sys_path=info['sys_path']+[str((c.ASSETS/'provider/src').resolve())],
        cache_prefix=str(c.ASSETS/'NO_BYTECODE_CACHE'),
        inventory_sha256=c.digest(sorted((str(p),files[str(p)]['sha256']) for p in runtime_paths)),
        historical_compile_runtime='ARCHIVED_PYTHON_3_12_COSTS',analysis_runtime='Python3.11 stdlib-only arithmetic; same original COMPARE numeric SOURCE',
        full_historical_load_claim=False)
    cfg=dict(schema=c.SERIES,source_commit=source,physical=c.policy.PHYSICAL,
        candidates=[asdict(x) for x in c.policy.CANDIDATES],budget=c.counts(),limitations=list(c.LIMITS),
        files=sorted(files.values(),key=lambda r:r['path']),source_files=source_rows,
        manifest=str(manifest_path.resolve()),origin_contract=str((ORIGIN/'DIAGNOSTICS_contract.json').resolve()),
        origin_run_root=str(origin_run.resolve()),runtime=runtime,run_root=str((c.ASSETS/'EXPLORATORY_run01').resolve()),
        retry=False,mandatory_STOP=True)
    contract=c.ASSETS/'EXPLORATORY_contract_PRIVATE.json';c.save(contract,cfg)
    cmd=[str(c.seal.PYTHON),'-I','-S','-B',str(ROOT/'scripts/launch_h2_archived_exploratory.py'),'--contract',str(contract)]
    run=subprocess.run(cmd,env=env,capture_output=True,text=True)
    with (OUT/'cache/preflight_process_PRIVATE.json').open('x') as f:json.dump(dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr),f)
    c.require(run.returncode==0,'METADATA_PREFLIGHT_FAILED:'+run.stderr[-1500:])
    pf=json.loads(run.stdout);c.require(pf['technical_ready'] and not pf['execution_authorized'],'NO_SCIENCE_BEFORE_APPROVAL')
    publish('metadata_preflight.json',pf)
    public_manifest=json.loads(json.dumps(manifest))
    for row in public_manifest['cost_files'].values():row.pop('local_path')
    publish('fixed_input_manifest.json',public_manifest)
    sources=[{k:v for k,v in x.items() if k!='local_path'} for x in source_rows]
    publish('SOURCE.json',dict(technical_SOURCE=source,base_commit=c.BASE,files=sources,
        source_files_sha256=c.digest(sources),scientific_source_changed=False,strict_COMPARE_source='46a3b33e3717c4fadf11846e922caf17d10156d9'))
    publish('execution_contract_DRAFT.json',dict(schema=c.SERIES,status='SEALED_AWAITING_SEPARATE_SCIENTIFIC_APPROVAL',
        contract_sha256=c.digest(cfg),input_manifest_sha256=c.sha(manifest_path),source_commit=source,
        physical=cfg['physical'],candidates=cfg['candidates'],exact_calls=cfg['budget'],
        run_root='${NEW_EXPLORATORY_ASSETS}/EXPLORATORY_run01',retry=False,mandatory_STOP=True,
        science_authorized=False,grant_issued=False,strict_COMPARE='BLOCKED',limitations=list(c.LIMITS)))
    publish('runtime_seal.json',dict(python_sha256=runtime['python_sha256'],analysis_runtime=runtime['analysis_runtime'],
        historical_compile_runtime=runtime['historical_compile_runtime'],runtime_file_count=len(runtime_paths),
        runtime_inventory_sha256=runtime['inventory_sha256'],source_file_count=len(source_rows),
        sealed_files=len(files),static_ELF_dependencies_resolved=True,mapped_executables_preflight_verified=True,
        package_science_imports=0,arbitrary_resource_caps=False))
    publish('authority_templates.json',dict(approval=dict(issuer='USER_EXPLICIT',execution_authorized=False,
        contract_sha256=c.digest(cfg),accepted_limitations=list(c.LIMITS)),
        grant=dict(id=None,purpose=c.SERIES,state='INVALID_AWAITING_USER_APPROVAL',contract_sha256=c.digest(cfg),
        approval_sha256=None,run_root='${NEW_EXPLORATORY_ASSETS}/EXPLORATORY_run01',budget=c.counts(),retry=False),
        actual_grants_issued=0,old_grants_reusable=False))
    publish('output_schema_DRAFT.json',dict(schema=c.SERIES+'_RESULT',methods=4,
        per_method=['compiled_RZ_statistics: n/mean/min/max/sample_sd/standard_error',
            'analytic_shots: real/imag/N_total','allowance: inherited empirical definition',
            'G_axis: cosine/sine or null','G_total: original COMPARE arithmetic or null',
            'paired_resource_statistics: existing G standard error, not population guarantee',
            'nominal_interval: null (no new interval rule)', 'status: accounting or C-only'],
        ranking='Original Backend.rank Full S2/Y4 and PR S2/Y4 contrasts; incomplete if any method ineligible',
        four_method_totals_comparable=True,science_GO=False,strict_COMPARE='BLOCKED',limitations=list(c.LIMITS)))
    publish('resources.json',dict(host=socket.gethostname(),CPU_affinity=len(os.sched_getaffinity(0)),
        MemAvailable=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
        arbitrary_CPU_RAM_worker_caps=False,shared_environment_changes=False))
    publish('STOP.json',dict(status='MANDATORY_STOP_PHASE3_SCIENTIFIC_APPROVAL_GATE',actual_calls=dict.fromkeys(c.KINDS,0),
        science_authorized=False,live_grants=0,new_u_N_G_evaluated=False,ranking_evaluated=False,
        strict_COMPARE='BLOCKED',private_transfer=0,push=0,mandatory_STOP=True))
    print(json.dumps(dict(status=pf['status'],source=source,contract_sha256=c.digest(cfg),
        sealed_files=len(files),runtime_files=len(runtime_paths),science_calls=0,grant_issued=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);args=parser.parse_args();build(args.source)
