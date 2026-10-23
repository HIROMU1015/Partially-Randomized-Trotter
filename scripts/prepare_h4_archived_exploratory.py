"""Metadata-only collection and sealing; never call a molecular/numerical kernel."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import h4_archived_exploratory as c


def gitfile(commit,path):
    return c.h2.recovery.git_bytes(commit,path)


def collect(analysis_pack):
    """Recover only the two registered groups and original saved endpoint fields."""
    files={};catalog=[]
    def protect(path,role):
        p=Path(path).resolve();files[str(p)]=dict(path=str(p),sha256=c.sha(p),role=role)
    def fixed(commit,path):
        raw=gitfile(commit,path)
        p=c.ASSETS/'fixed'/commit/path;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():c.require(p.read_bytes()==raw,'FIXED_REFERENCE_CHANGED')
        else:p.write_bytes(raw)
        protect(p,'INPUT:fixed_Git_result')
        catalog.append(dict(commit=commit,path=path,sha256=c.sha(p),bytes=len(raw)))
        return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)
    dataset=fixed(c.ADOPTED,c.ADOPTED_DIR+'dataset.json.gz')
    fullcat=fixed(c.ADOPTED,c.ADOPTED_DIR+'source_catalog.json')
    native_meta=fullcat['H4_native_inventory']
    natives=fixed(native_meta['commit'],native_meta['path'])
    for key in ('H4_identity','H4_contract','H4_audit','H4_STOP'):
        r=fullcat[key];fixed(r['commit'],r['path'])
        c.require(catalog[-1]['sha256']==r['sha256'],'ADOPTED_FIXED_CATALOG_SHA')
    fixed(c.SAVED,c.SAVED_DIR+'input_provenance.json')
    fixed(c.SAVED,c.SAVED_DIR+'end_audit.json')
    evaluations=fixed(c.SAVED,c.SAVED_DIR+'evaluation.json')
    for name in ('REPORT.md','comparison.json','execution_ledger_receipt.json','native_completion.json','final_verification.json'):
        path='artifacts/resource_applicability/track_a_h2_archived_exploratory_run01/2026-10-24/'+name
        raw=gitfile(c.H2_RESULT,path)
        catalog.append(dict(commit=c.H2_RESULT,path=path,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    analysis=Path(analysis_pack).resolve()
    cfgp=analysis/'config_APPROVED_PRIVATE.json';cfg=c.read(cfgp);plan=cfg['plan']
    c.require(c.sha(cfg['plan_path'])==cfg['plan_sha256']==
        'c2a9cce59b75abbc1669d60600e7f9fa5e8e016f2efa9a722e64e7f340057a71','ACCEPTED_SAVED_ANALYSIS_PLAN')
    root=Path(cfg['run_root'])
    c.require(c.sha(root/'evaluation.json')=='b6d1309fc615c6df2c3a65c81f0a62dfff327435e98fde3bc2d74f3b80504f5e',
        'ACCEPTED_SAVED_ANALYSIS_RESULT')
    for p in (cfgp,Path(cfg['plan_path']),root/'evaluation.json',root/'completion.json',root/'CONSUMED.json',
              analysis/'grant_APPROVED_PRIVATE.json',analysis/'external_exit_PRIVATE.json'):
        protect(p,'PROTECTED:original_analysis_authority_and_results')
    data=dict(schema=c.SERIES+'_INPUTS',physical=c.PHYSICAL,binding=c.BINDING,candidates=c.candidates(),
        compiler=c.COMPILER,native=c.NATIVE,adopted_commit=c.ADOPTED,saved_commit=c.SAVED,
        reference_policy='CONDITIONAL_SAVED_H4_REFERENCE_NOT_ZERO_ERROR_BOUND',methods={})
    for candidate in c.CANDIDATES:
        cid=candidate.id;costs=[]
        if candidate.method=='FULL':
            item=next(x for x in dataset['candidates'] if x['candidate_status_saved']['id']==cid)
            target=item['candidate_status_saved'];saved=item['signal_record_saved'];signal=saved['signal']
            endpoint=next(e for e in item['endpoints_saved'] if e['epsilon']==.0005)
            c.require(target['system']=='H4' and target['formula']==candidate.formula and
                target['prefix']==candidate.prefix and target['q']==candidate.q,'FULL_CANDIDATE_JOIN')
            identity=dataset['systems']['H4']['scientific_identity_saved']['binding']
            c.require(all(identity[k]==v for k,v in c.BINDING.items()) and identity['rank']==10,'FULL_DF10_BINDING')
            actual=next(x for x in natives if x['candidate']==endpoint['candidate'])
            c.require(actual['result_sha256']==endpoint['price']['result_sha256'],'FULL_MEASURED_RECEIPT')
            costs=[dict(task=cid,cosine=endpoint['price']['cosine'],sine=endpoint['price']['sine'],
                receipt_sha256=actual['result_sha256'])]
            prep=actual['preparation_hash'];u=endpoint['u'];margin=endpoint['log_B_margin']
            shots={a:endpoint['axes'][a]['shots'] for a in ('real','imag')}
            source=dict(commit=c.ADOPTED,pointer='/candidates/'+str(dataset['candidates'].index(item)),
                receipt_commit=native_meta['commit'],receipt_path=native_meta['path'],
                acquisition_SOURCE=dataset['systems']['H4']['SOURCE'])
        else:
            group=next(g for g in plan['groups'] if g['record']['candidate']['id']==cid)
            record=group['record'];target=record['candidate'];signal=record['signal']
            c.require(all(target[k]==getattr(candidate,k) for k in ('id','method','formula','prefix','q','K','r_policy','r_value')),
                'ORIGINAL_REGISTERED_PR_CANDIDATE')
            c.require(all(group['binding'][k]==v for k,v in c.BINDING.items()),'PR_INPUT_BINDING')
            prep=record['preparation_hash'];empirical=record['empirical_eligibility']
            u=empirical['u'];margin=empirical['log_B_margin']
            saved=next(g for g in evaluations['groups'] if g['candidate']['id']==cid)
            endpoint=next(e for e in saved['rows'] if e['mode']=='USUAL_SIGNED' and e['epsilon']==.0005)
            c.require(endpoint['shot']['u']==u and endpoint['shot']['log_B_margin']==margin and
                endpoint['shot']['saved_signed_bias']==signal['total_signed_bias'],'SAVED_PR_SHOT_INPUT_JOIN')
            shots={a:endpoint['shot']['axes'][a]['shots'] for a in ('real','imag')}
            c.require(len(group['tasks'])==candidate.n,'EXACT_ORIGINAL_COHORT')
            for entry in group['tasks']:
                p=Path(entry['result_path']);c.require(c.sha(p)==entry['sha256'],'ORIGINAL_TRAJECTORY_SHA')
                protect(p,'INPUT:PR_original_trajectory_receipt');raw=c.read(p)
                c.require(raw['task']==entry['task'] and raw['SOURCE']==group['acquisition_SOURCE'] and
                    raw['plan_sha256']==group['acquisition_plan_sha256'] and raw['input_binding']==group['binding'] and
                    raw['preparation_hash']==prep and raw['log_B']==signal['log_B'],'ORIGINAL_PR_BINDING')
                c.require(raw['seed_cohort']=='EXPLORATION' and raw['paired_trajectory'] is True and
                    raw['native_scientific_PASS'] is True and raw['byte_restoration_PASS'] is True and
                    raw['conditional_sampling'] is False and raw['cohort_pooling'] is False,'ORIGINAL_PR_ACCEPTANCE')
                axes=raw['axis_costs']
                c.require([a['axis'] for a in axes]==['cosine','sine'] and len(raw['checks'])==8 and
                    all(x['PASS'] and not x['phase_alignment'] for x in raw['checks']),'ORIGINAL_PR_AXIS_CORRECTNESS')
                c.require(all(a['compiler_settings_hash']==c.COMPILER and a['native_policy']==c.NATIVE and
                    a['preparation_hash']==prep and a['event_digest']==raw['event_digest'] for a in axes),
                    'ORIGINAL_PR_NATIVE_COMPILER')
                costs.append(dict(task=entry['task']['id'],receipt_sha256=entry['sha256'],
                    **{a['axis']:a['metrics']['rz_count'] for a in axes}))
            # Reuse the previously recorded paired price file; do not recompute statistics here.
            pricefile=root/'trajectory_prices'/(cid+'.json');protect(pricefile,'INPUT:accepted_paired_prices')
            c.require([list((r['cosine'],r['sine'])) for r in costs]==c.read(pricefile)['paired_RZ'],
                    'SAVED_PAIRED_PRICE_FILE')
            source=dict(commit=c.SAVED,acquisition_SOURCE=group['acquisition_SOURCE'],
                acquisition_plan_sha256=group['acquisition_plan_sha256'],saved_analysis_SOURCE=evaluations['SOURCE'])
        data['methods'][cid]=dict(candidate=asdict(candidate),binding=c.BINDING,unit=c.PHYSICAL['unit'],
            cohort='EXPLORATION' if candidate.method=='PR' else 'COMMON_PF_DETERMINISTIC',
            compiler=c.COMPILER,native=c.NATIVE,preparation_hash=prep,costs=costs,
            bias=signal['total_signed_bias'],saved_reference=signal['reference'],log_B=signal['log_B'],
            u=u,log_B_margin=margin,saved_shots=shots,allowance_recomputed=False,reference_error_bound=None,
            source=source)
    return c.validate_bundle(data),files,catalog


def build(source,analysis_pack):
    source=subprocess.check_output(['git','rev-parse',source],cwd=c.ROOT,text=True).strip()
    c.ASSETS.mkdir(exist_ok=False);c.OUT.mkdir(parents=True,exist_ok=True);(c.OUT/'cache').mkdir(exist_ok=True)
    data,files,catalog=collect(analysis_pack)
    def add(p,role):
        p=Path(p).resolve();files[str(p)]=dict(path=str(p),sha256=c.sha(p),role=role)
    sources=[]
    for rel,(_,raw) in zip(c.NEW+c.REUSED,c.h2.seal.git_batch(source,c.NEW+c.REUSED)):
        p=c.ROOT/rel;c.require(p.read_bytes()==raw,'COMMITTED_RUNNER_SOURCE')
        if rel in c.REUSED:c.require(raw==gitfile(c.H2_SOURCE,rel),'H2_NUMERIC_SOURCE_UNCHANGED')
        add(p,'SOURCE:runner');sources.append(dict(root='runner',path=rel,local_path=str(p.resolve()),sha256=c.sha(p)))
    for rel,(_,raw) in zip(c.h2.PROVIDER_PATHS,c.h2.seal.git_batch(c.h2.PROVIDER,c.h2.PROVIDER_PATHS)):
        p=c.ASSETS/'provider'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        add(p,'SOURCE:original_numeric_provider');sources.append(dict(root='provider',path=rel,local_path=str(p.resolve()),sha256=c.sha(p)))
    manifest=c.ASSETS/'input_manifest_PRIVATE.json';c.save(manifest,data);add(manifest,'INPUT:fixed_H4_manifest')
    # Reuse the successful H2 stdlib/ELF inventory; no packages are imported or modified.
    oldcfg=c.read(c.ROOT.parent/'h2-archived-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json')
    runtime_files=[r for r in oldcfg['files'] if r['role'].startswith('runtime:')]
    for r in runtime_files:
        c.require(c.sha(r['path'])==r['sha256'],'ANALYSIS_RUNTIME_FILE_UNCHANGED');files[r['path']]=r
    rt=dict(oldcfg['runtime']);rt['sys_path']=[p for p in rt['sys_path'] if not p.endswith('/provider/src')]+[str(c.ASSETS/'provider/src')]
    rt['cache_prefix']=str(c.ASSETS/'NO_BYTECODE_CACHE')
    cfg=dict(schema=c.SERIES,source_commit=source,physical=c.PHYSICAL,candidates=c.candidates(),
        budget=c.counts(),limitations=list(c.LIMITS),files=sorted(files.values(),key=lambda x:x['path']),
        source_files=sources,manifest=str(manifest),manifest_sha256=c.sha(manifest),runtime=rt,
        run_root=str(c.ASSETS/'EXPLORATORY_run01'),retry=False,mandatory_STOP=True)
    cp=c.ASSETS/'EXPLORATORY_contract_PRIVATE.json';c.save(cp,cfg)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    command=[str(c.h2.seal.PYTHON),'-I','-S','-B',str(c.ROOT/'scripts/launch_h4_archived_exploratory.py'),'--contract',str(cp)]
    run=subprocess.run(command,env=env,capture_output=True,text=True)
    c.save(c.OUT/'cache/preflight_process_PRIVATE.json',dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    c.require(run.returncode==0,'H4_METADATA_PREFLIGHT_FAILED:'+run.stderr[-1500:])
    pf=json.loads(run.stdout)
    def public(name,value):
        text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
        c.require('/home/' not in text and '/usr/' not in text,'PUBLIC_PROJECTION_NO_INTERNAL_PATHS')
        (c.OUT/name).write_text(text)
    public('metadata_preflight.json',pf)
    public('SOURCE.json',dict(source_commit=source,H2_numeric_SOURCE=c.H2_SOURCE,provider=c.h2.PROVIDER,
        files=[{k:v for k,v in r.items() if k!='local_path'} for r in sources],original_sources_modified=False))
    public('fixed_input_manifest.json',dict(schema=data['schema'],input_manifest_sha256=c.sha(manifest),
        physical=c.PHYSICAL,binding=c.BINDING,candidates=c.candidates(),saved_axes=166,
        input_files=catalog,per_method={k:dict(candidate=v['candidate'],source=v['source'],
            preparation_hash=v['preparation_hash'],paired_tasks=len(v['costs']),u=v['u'],log_B_margin=v['log_B_margin'],
            allowance_recomputed=False,saved_shot_available=True) for k,v in data['methods'].items()}))
    public('execution_contract_DRAFT.json',dict(schema=c.SERIES,status='SEALED_AWAITING_H4_SCIENTIFIC_APPROVAL',
        source_commit=source,contract_sha256=c.digest(cfg),input_manifest_sha256=c.sha(manifest),
        physical=c.PHYSICAL,candidates=c.candidates(),exact_calls=c.counts(),kernel_ledger_events=sum(c.counts().values()),
        allowances='Reuse H4 saved u and log_B_margin unchanged; no new empirical_allowance evaluation',
        shots='Existing u_aware_shots; integers must reproduce saved H4 records',
        run_root='${H4_EXPLORATORY_ASSETS}/EXPLORATORY_run01',limitations=list(c.LIMITS),
        science_authorized=False,retry=False,mandatory_STOP=True,strict_COMPARE='BLOCKED'))
    public('authority_templates.json',dict(actual_grants_issued=0,
        approval=dict(issuer='USER_EXPLICIT',execution_authorized=False,contract_sha256=c.digest(cfg),accepted_limitations=list(c.LIMITS)),
        grant=dict(id=None,purpose=c.SERIES,state='INVALID_AWAITING_H4_APPROVAL',contract_sha256=c.digest(cfg),
            approval_sha256=None,run_root='${H4_EXPLORATORY_ASSETS}/EXPLORATORY_run01',budget=c.counts(),retry=False),
        H2_or_old_grants_reusable=False))
    public('runtime_seal.json',dict(analysis_runtime='Python3.11 isolated stdlib arithmetic',
        python_sha256=rt['python_sha256'],runtime_files=len(runtime_files),sealed_files=len(files),
        inventory_sha256=rt['inventory_sha256'],mapped_executables_preflight_verified=True,
        historical_compiler_runtime='Preserved saved PR/common PF compiler identities; no new compile',
        arbitrary_resource_caps=False,scientific_packages_imported=0))
    public('resources.json',dict(host=socket.gethostname(),CPU_affinity=len(os.sched_getaffinity(0)),
        MemAvailable=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')),
        free_disk_bytes=__import__('shutil').disk_usage(c.ASSETS).free,artificial_limits=False,shared_changes=False))
    public('STOP.json',dict(status='MANDATORY_STOP_H4_SCIENTIFIC_APPROVAL_GATE',science_authorized=False,
        actual_calls=dict.fromkeys(c.KINDS,0),new_allowance_N_G_ranking=False,live_grants=0,mandatory_STOP=True))
    c.save(c.OUT/'cache/protected_paths_PRIVATE.json',dict(files=list(files.values())))
    print(json.dumps(dict(status=pf['status'],SOURCE=source,contract_sha256=c.digest(cfg),
        input_manifest_sha256=c.sha(manifest),sealed_files=len(files),science_calls=0,grants_issued=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--saved-analysis-pack',required=True)
    a=p.parse_args();build(a.source,a.saved_analysis_pack)
