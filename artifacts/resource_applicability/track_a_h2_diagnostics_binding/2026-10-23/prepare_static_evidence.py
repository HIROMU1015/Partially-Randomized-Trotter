"""Read/copy/hash/AST audit only. Does not import science or evaluate saved values."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import h2_diagnostics_binding as b
import h2_common_production_contract as c
BASE='d405b880a679ee4a95d233f5535c4e3bf2db53f4'
RECEIVED=ROOT/'artifacts/resource_applicability/track_a_pf_order_reversal_h2_metadata_receipt/2026-10-23/received'


def main():
    cfg=b.read(OUT/'DIAGNOSTICS_contract.json')
    runtime=b.read(cfg['runtime']['closure_manifest'])
    # Preserve the complete static graph in Git as well as the canonical local seal.
    shutil.copyfile(cfg['runtime']['closure_manifest'],OUT/'runtime_closure.json')
    b.write(OUT/'call_budgets.json',{branch:c.counts(branch) for branch in c.BRANCHES})
    syntax_paths=sorted({r['path'] for r in cfg['files'] if r['path'].endswith('.py')})
    program='''import ast,json,sys
errors=[]
paths=json.load(sys.stdin)
for p in paths:
 try: ast.parse(open(p,"rb").read(),filename=p)
 except (SyntaxError,ValueError) as e: errors.append(dict(path=p,error=str(e)))
print(json.dumps(dict(interpreter=sys.version,files=len(paths),errors=errors,scientific_imports=False)))
'''
    result=subprocess.run([str(b.PYTHON),'-I','-S','-B','-c',program],input=json.dumps(syntax_paths),
                          capture_output=True,text=True)
    b.require(result.returncode==0,'STDLIB_SYNTAX_AUDIT_PROCESS')
    syntax=json.loads(result.stdout);b.write(OUT/'python_syntax_audit.json',syntax)
    b.require(not syntax['errors'],'PYTHON_SYNTAX_INCOMPATIBILITY')
    # Original Git exporter only, never execute it or follow guessed private roots.
    export_commit='4f1e4ebd46a5e03876971a5374d390bd4ead921a'
    export_path='scripts/export_h2_outer32_metadata.py'
    blob,raw=b.git_batch(export_commit,[export_path])[0]
    (OUT/'evidence_sources').mkdir(exist_ok=True)
    (OUT/'evidence_sources/export_h2_outer32_metadata.py').write_bytes(raw)
    b.write(OUT/'retrieval_audit.json',dict(exporter=dict(commit=export_commit,path=export_path,
        git_blob_sha1=blob,sha256=hashlib.sha256(raw).hexdigest(),executed=False),
        bounded_sources=['received origin_inventory/acquisition_context/task/schedule/runtime metadata',
          'fixed exporter Git object; previously recovered 366b2fd2 static audit',
          'known old H2 private D/FP/F locators only'],
        missing_canonical_locator='${NEW_EXECUTION} mapping or canonical original exploration config O00002',
        required_originals={r['id']:{k:r[k] for k in ('locator','original_sha256','bytes')}
            for r in b.read(RECEIVED/'origin_inventory.json')['originals'] if r['id'] in {'O00001','O00002'}},
        broad_private_search=False,raw_events_recovered=False,
        retrieval_method='Owner supplies private token map or exact original config locator; follow original sealed files/storage manifest, verify raw/archive hashes before any metadata-only decoding. No resampling/recompile.'))
    schedules={r['task_id']:r for r in b.read(RECEIVED/'rte_schedule_metadata.json')['tasks']}
    tasks=b.read(RECEIVED/'task_acceptance_67.json')['tasks'];pairs=[]
    for row in tasks:
        task=row['task'];saved=row['result']['saved'];axes=saved['axis_costs'];cid=task['candidate']['id']
        b.require(cid in {x.id for x in c.policy.CANDIDATES[2:]},'FIXED_PR_METADATA_ONLY')
        b.require({a['axis'] for a in axes}=={'cosine','sine'} and len(axes)==2,'PAIRED_METADATA')
        b.require(all(a['event_digest']==saved['event_digest'] and a['preparation_hash']==saved['preparation_hash'] for a in axes),'AXIS_METADATA_JOIN')
        pairs.append(dict(task=task,candidate=cid,projection_sha256=c.policy.digest(saved),
            SOURCE=saved['SOURCE'],input_binding=saved['input_binding'],preparation_hash=saved['preparation_hash'],
            scalar=saved['scalar'],event_digest=saved['event_digest'],events_summary=saved['events'],
            event_operations_present=False,operation_stream_witness_present=False,
            schedule=schedules[task['id']],axes=axes,checks=saved['checks'],
            native_scientific_PASS=saved['native_scientific_PASS'],
            paired_trajectory=saved['paired_trajectory'],seed_cohort=saved['seed_cohort'],
            conditional_sampling=saved['conditional_sampling'],cohort_pooling=saved['cohort_pooling']))
    counts=Counter(r['candidate'] for r in pairs)
    b.require(counts=={'H2O32_c7ec590a32a859bc':33,'H2O32_8a2ecdd0294ffb74':34},'FIXED_COHORT_COUNTS')
    b.write(OUT/'PR_C_task_identity.json',dict(status='PARTIAL_METADATA_NOT_IDENTITY_ACCEPTANCE',tasks=67,axes=134,
        candidates=dict(counts),rows=pairs,numerical_reaccounting=False,compare_execution_ready=False))
    sources=[]
    for f in ['task_acceptance_67.json','preparation_metadata.json','rte_schedule_metadata.json',
              'runtime_provider_manifest.json','missing_evidence.json','origin_inventory.json','acquisition_context.json']:
        rel=str((RECEIVED/f).relative_to(ROOT));blob,raw=b.git_batch(BASE,[rel])[0]
        b.require((ROOT/rel).read_bytes()==raw,'OLD_METADATA_PRESERVED')
        sources.append(dict(path=rel,git_blob_sha1=blob,sha256=hashlib.sha256(raw).hexdigest(),base_commit=BASE))
    b.write(OUT/'metadata_sources.json',sources)
    # Hash known originals and sealed original sources, without decoding vectors.
    old=b.OLD_PRIVATE;spec=b.read(old/'D_activation01/D_spec_EXECUTION.json')
    originals=[dict(path=str(old/'D_activation01/D_spec_EXECUTION.json'),
                    historical_sha256='5c037ec2b45ee5be364064b86b7b6d81c049367f739e93eb35963fa54c40d8f8')]
    originals += [dict(path=r['path'],historical_sha256=r['sha256']) for r in spec['inputs']]
    dep=spec['dependencies']['FP']
    for key in ('result','acceptance','completion_audit'):
        originals.append(dict(path=dep[key],historical_sha256=dep['sha256' if key=='result' else key+'_sha256']))
    for r in originals:
        r['current_sha256']=b.sha(r['path']);r['matches_historical']=r['historical_sha256']==r['current_sha256']
    b.require(all(r['matches_historical'] for r in originals),'ORIGINAL_PRESERVATION')
    preserved=b.read(old/'D_recovery_activation01/D_completed_preservation_manifest_PRIVATE.json')
    historical_files=[]
    for row in preserved:
        p=Path(row['path']).resolve()
        b.require(p.is_relative_to(old.resolve()),'BOUNDED_HISTORICAL_PRESERVATION_PATH')
        current=b.sha(p)
        historical_files.append(dict(path=str(p),historical_sha256=row['sha256'],current_sha256=current,
            bytes=p.stat().st_size,matches_historical=current==row['sha256']))
    b.require(all(r['matches_historical'] for r in historical_files),'COMPLETED_OLD_RUN_PRESERVATION')
    b.write(OUT/'historical_completed_preservation.json',dict(files=historical_files,
        manifest_locator=str((old/'D_recovery_activation01/D_completed_preservation_manifest_PRIVATE.json').resolve()),
        manifest_sha256=b.sha(old/'D_recovery_activation01/D_completed_preservation_manifest_PRIVATE.json'),
        numeric_data_decoded=False,originals_written=False))
    snapshots=[]
    for directory in [old/'D_activation01',old/'D_recovery_activation01',old/'attempts/D_run01',old/'attempts/FP_run01']:
        # Bounded control-plane files only; raw output subdirectories are not walked.
        files=list(directory.glob('*.json'))+list((directory/'ledger').glob('*.json'))
        for p in sorted(files):snapshots.append(dict(path=str(p.resolve()),sha256=b.sha(p),bytes=p.stat().st_size))
    b.write(OUT/'preservation_observation.json',dict(originals=originals,
        known_control_plane_files=snapshots,grant_contents_published=False,
        original_files_written=False,new_run_root_exists=Path(cfg['run_root']).exists(),
        new_cache_prefix_exists=Path(cfg['runtime']['cache_prefix']).exists(),
        new_live_grants=0,new_scientific_ledgers=0,new_scientific_markers=0,
        scope='Known originals verified against saved hashes; bounded control files hashed for final recheck. Other worktrees are not frozen.'))
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    disk=shutil.disk_usage(ROOT)
    b.write(OUT/'resources.json',dict(affinity_cpus=len(os.sched_getaffinity(0)),
        available_ram_bytes=mem['MemAvailable'],free_disk_bytes=disk.free,
        technical_hash_workers=len(os.sched_getaffinity(0)),arbitrary_resource_caps_added=False,
        shared_environment_modified=False,scientific_workers=0))
    print(json.dumps(dict(status='STATIC_EVIDENCE_SAVED',syntax_files=len(syntax_paths),tasks=67,axes=134)))


if __name__=='__main__':main()
