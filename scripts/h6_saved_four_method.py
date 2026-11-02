"""Quote complete H6 saved results; only the optional joint ranking needs science approval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace
import h4_archived_exploratory as h4

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT.parent/'h6-four-method-saved-results-assets-20261024'
OUT=ROOT/'artifacts/resource_applicability/track_a_h6_saved_four_method/2026-10-24'
SERIES='H6_FOUR_METHOD_SAVED_RESULTS_RANK_ONLY_V1'
BASE='868ee7ced6818b2531a3aa9c12498f02bce55dc3'
ADOPTED=h4.ADOPTED
SAVED=h4.SAVED
PHYSICAL=dict(h4.PHYSICAL,system='H6',rank=19,system_qubits=12,ancilla=12)
BINDING=dict(hamiltonian='223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9',
 state='664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3',
 snapshot='99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005')
CANDIDATES=[dict(id='H6_FULL_19_8_2nd',method='FULL',formula='2nd',prefix=19,q=8,n=1),
 dict(id='H6_FULL_19_4_4th',method='FULL',formula='4th',prefix=19,q=4,n=1),
 dict(id='M5_MIN_a08e8bfb6642eb93',method='PR',formula='2nd',prefix=5,q=8,n=33),
 dict(id='M5_MIN_37ff3428bb58f5cd',method='PR',formula='4th',prefix=5,q=4,n=33)]
LIMITS=[x.replace('H4','H6') for x in h4.LIMITS]+['H6_original_parent_exit_reap_unknown',
 'H6_inherited_complete_compiled_pairs_audited_not_original_stage_completion']
NEW=('scripts/h6_saved_four_method.py','scripts/launch_h6_saved_four_method.py')
REUSED=tuple(dict.fromkeys(h4.NEW+h4.REUSED))
KINDS=h4.KINDS
read,save,sha,digest,require,Stop=h4.read,h4.save,h4.sha,h4.digest,h4.require,h4.Stop
policy=SimpleNamespace(CANDIDATES=[SimpleNamespace(**x) for x in CANDIDATES],number=h4.policy.number)


def counts():
    d=dict.fromkeys(KINDS,0);d.update(bootstrap=1,bundle_load=1,ranking=1);return d


def validate(data):
    require(data['schema']==SERIES+'_INPUTS' and data['physical']==PHYSICAL and
        data['binding']==BINDING and data['candidates']==CANDIDATES,'FIXED_H6_DF19_TARGET')
    require(data['cohort_pooling'] is False and data['new_accounting'] is False,'SAVED_VALUES_ONLY')
    require(data['limitations']==LIMITS and len(data['methods'])==4,'SAVED_SCOPE_LIMITS')
    for row,can in zip(data['methods'],CANDIDATES):
        require(row['candidate']==can and row['unit']==PHYSICAL['unit'] and
            row['binding']==BINDING and row['status']=='ELIGIBLE','SAVED_CANDIDATE_IDENTITY')
        require(row['cohort']==('EXPLORATION' if can['method']=='PR' else 'COMMON_PF_DETERMINISTIC'),
            'ORIGINAL_H6_COHORT')
        require(row['compiler']==h4.COMPILER and row['native']==h4.NATIVE,'SAVED_NATIVE_COMPILER')
        for k in ('N_cos','N_sin'):require(type(row[k]) is int and row[k]>0,'SAVED_SHOT_INTEGER')
        for k in ('C_cos','C_sin'):policy.number(row[k],k)
        policy.number(row['G_saved'],'saved_G') if not isinstance(row['G_saved'],str) else require(
            row['G_saved'].isdigit(),'SAVED_FULL_EXACT_INTEGER_STRING')
        require(row['u']==1e-9 and row['reference_error_bound'] is None,'H6_SAVED_EMPIRICAL_POLICY')
        if can['method']=='PR':
            require(row['n']==33 and row['paired'] is True and row['G_SE_saved'] is not None,'SAVED_PAIRED_SE')
            policy.number(row['G_SE_saved'],'saved_paired_SE')
        else:require(row['G_SE_saved'] is None,'NO_FABRICATED_FULL_SE')
    return data


def collect():
    catalog=[];files=[]
    def fixed(commit,path):
        raw=h4.h2.recovery.git_bytes(commit,path)
        p=ASSETS/'fixed'/commit/path;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():require(p.read_bytes()==raw,'PINNED_GIT_BYTES')
        else:p.write_bytes(raw)
        files.append(dict(path=str(p),sha256=sha(p),role='INPUT:fixed_saved_results'))
        catalog.append(dict(commit=commit,path=path,sha256=sha(p),bytes=len(raw)))
        return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)
    dataset=fixed(ADOPTED,h4.ADOPTED_DIR+'dataset.json.gz')
    cat=fixed(ADOPTED,h4.ADOPTED_DIR+'source_catalog.json')
    native=cat['H6_native_inventory'];natives=fixed(native['commit'],native['path'])
    require(catalog[-1]['sha256']==native['sha256'],'FIXED_NATIVE_CATALOG')
    for key in ('H6_identity','H6_audit','H6_STOP'):
        r=cat[key];fixed(r['commit'],r['path']);require(catalog[-1]['sha256']==r['sha256'],'FIXED_H6_CATALOG')
    evaluations=fixed(SAVED,h4.SAVED_DIR+'evaluation.json')
    fixed(SAVED,h4.SAVED_DIR+'input_provenance.json')
    fixed(SAVED,h4.SAVED_DIR+'end_audit.json')
    identity=dataset['systems']['H6']['scientific_identity_saved']['binding']
    require(all(identity[k]==v for k,v in BINDING.items()) and identity['rank']==19,'COMMON_H6_INPUT_BINDING')
    methods=[]
    for can in CANDIDATES:
        cid=can['id']
        if can['method']=='FULL':
            item=next(x for x in dataset['candidates'] if x['candidate_status_saved']['id']==cid)
            e=next(e for e in item['endpoints_saved'] if e['epsilon']==.0005)
            require(e['candidate']==dict(system='H6',**{k:can[k] for k in ('method','formula','prefix','q')}),
                'SAVED_FULL_TARGET')
            receipt=next(x for x in natives if x['candidate']==e['candidate'])
            require(e['price']['result_sha256']==receipt['result_sha256'],'AUDITED_COMPLETE_FULL_PAIR')
            s=item['signal_record_saved']['signal']
            row=dict(N_cos=e['N_cos'],N_sin=e['N_sin'],N_total_saved=None,
                C_cos=e['price']['cosine'],C_sin=e['price']['sine'],G_saved=e['total_RZ_exact'],
                G_cos_saved=None,G_sin_saved=None,G_SE_saved=None,C_cos_SE_saved=None,C_sin_SE_saved=None,
                n=1,paired=True,u=e['u'],log_B=e['log_B'],log_B_margin=e['log_B_margin'],
                reference=s['reference'],signed_bias=s['total_signed_bias'],status=e['status'],
                source=dict(commit=ADOPTED,pointer='/candidates/'+str(dataset['candidates'].index(item)),
                    receipt_sha256=receipt['result_sha256'],receipt_SOURCE=receipt['SOURCE'],
                    inherited_from_interrupted_stage=receipt['inherited_from_interrupted_stage']),
                omitted_derived_fields=['N_total','G_cos','G_sin'])
        else:
            g=next(x for x in evaluations['groups'] if x['candidate']['id']==cid)
            require(all(g['candidate'][k]==can[k] for k in ('id','method','formula','prefix','q')) and
                g['candidate']['K']==4 and g['candidate']['r_policy']=='ABS_TIME_PER_OUTER' and
                g['candidate']['r_value']==32 and g['candidate']['input_series']=='H6_TOL_ONLY_DF19','SAVED_PR_LAW')
            e=next(e for e in g['rows'] if e['mode']=='USUAL_SIGNED' and e['epsilon']==.0005);s=e['shot']
            require(e['n']==can['n'] and g['unit_statistics']['cosine']['n']==33 and
                g['unit_statistics']['sine']['n']==33,'SAVED_SAMPLE_COUNT')
            row=dict(N_cos=s['N_cos'],N_sin=s['N_sin'],N_total_saved=s['N_total'],
                C_cos=e['mean_cos_RZ'],C_sin=e['mean_sin_RZ'],G_saved=e['total_RZ'],
                G_cos_saved=e['cosine_RZ_contribution'],G_sin_saved=e['sine_RZ_contribution'],
                G_SE_saved=e['total_RZ_SE'],C_cos_SE_saved=g['unit_statistics']['cosine']['SE'],
                C_sin_SE_saved=g['unit_statistics']['sine']['SE'],n=e['n'],paired=True,u=s['u'],
                log_B=s['log_B'],log_B_margin=s['log_B_margin'],signed_bias=s['saved_signed_bias'],
                reference=None,status=e['status'],source=dict(commit=SAVED,candidate_id=cid,
                    analysis_SOURCE=evaluations['SOURCE'],exact_G=e['total_RZ_exact']),omitted_derived_fields=[])
        row.update(candidate=can,binding=BINDING,unit=PHYSICAL['unit'],compiler=h4.COMPILER,native=h4.NATIVE,
            cohort='EXPLORATION' if can['method']=='PR' else 'COMMON_PF_DETERMINISTIC',reference_error_bound=None)
        methods.append(row)
    data=dict(schema=SERIES+'_INPUTS',physical=PHYSICAL,binding=BINDING,candidates=CANDIDATES,
        methods=methods,catalog=catalog,cohort_pooling=False,new_accounting=False,limitations=LIMITS)
    return validate(data),files


def verify(cfg, *, unused=True,runtime=True):
    require(cfg['schema']==SERIES and cfg['physical']==PHYSICAL and cfg['candidates']==CANDIDATES and
        h4.h2.policy.strict_equal(cfg['budget'],counts()) and cfg['limitations']==LIMITS and cfg['retry'] is False and
        cfg['mandatory_STOP'] is True,'H6_RANK_ONLY_CONTRACT')
    require(Path(cfg['run_root'])==ASSETS/'RANK_run01' and (not unused or not Path(cfg['run_root']).exists()),'UNUSED_H6_RANK_ROOT')
    indexed=h4.h2.old.seal_check(cfg)
    require({r['path'] for r in cfg['source_files']}==set(NEW+REUSED),'SOURCE_SCOPE')
    for row,(_,raw) in zip(cfg['source_files'],h4.h2.seal.git_batch(cfg['source_commit'],[r['path'] for r in cfg['source_files']])):
        require(Path(row['local_path']) in indexed and Path(row['local_path']).read_bytes()==raw,'COMMITTED_H6_RANK_SOURCE')
    require(sha(cfg['manifest'])==cfg['manifest_sha256'],'FIXED_SAVED_INPUTS')
    if runtime:
        require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,'ISOLATED_I_S_B')
        require(str(Path(sys.executable).resolve())==cfg['runtime']['python_target'] and
            sha(Path(sys.executable).resolve())==cfg['runtime']['python_sha256'],'SEALED_ANALYSIS_PYTHON')
        require(sys.path==[str(ROOT/'scripts')]+cfg['runtime']['sys_path'],'SEALED_IMPORT_PATH')
        require(not os.environ.get('LD_LIBRARY_PATH') and not os.environ.get('LD_PRELOAD'),'CLEAN_LOADER')
        h4.h2.seal.verify_mapped_runtime(cfg)
    return validate(read(cfg['manifest']))


def preflight(cfg):
    data=verify(cfg)
    return dict(status='H6_SAVED_RESULTS_PREFLIGHT_PASS',technical_ready=True,execution_authorized=False,
        contract_sha256=digest(cfg),source_commit=cfg['source_commit'],manifest_sha256=cfg['manifest_sha256'],
        saved_methods=len(data['methods']),reaccounting_required=False,scientific_calls=0,budget=counts(),mandatory_STOP=True)


def authorize(cfg,approval,grant):
    require(approval.get('issuer')=='USER_EXPLICIT' and approval.get('execution_authorized') is True and
        approval.get('contract_sha256')==digest(cfg) and approval.get('accepted_limitations')==LIMITS,
        'EXPLICIT_FIXED_H6_RANK_APPROVAL_REQUIRED')
    require(grant.get('purpose')==SERIES and grant.get('state')=='UNUSED' and
        grant.get('id','').startswith(SERIES+':') and grant.get('contract_sha256')==digest(cfg) and
        grant.get('approval_sha256')==digest(approval) and grant.get('budget')==counts() and
        grant.get('run_root')==cfg['run_root'] and grant.get('retry') is False,'UNUSED_H6_RANK_GRANT')
    verify(cfg)


def rebound(fn):
    return FunctionType(fn.__code__,dict(fn.__globals__,c=sys.modules[__name__],rebound=rebound),
        fn.__name__,fn.__defaults__,fn.__closure__)


def rank_rows(data):
    # The only new scientific operation: reuse the H4 rank body on already computed G.
    from h4_archived_exploratory_backend import Backend
    rows=[dict(candidate=r['candidate']['id'],eligible=r['status']=='ELIGIBLE',
        G=int(r['G_saved']) if isinstance(r['G_saved'],str) else r['G_saved']) for r in data['methods']]
    return rebound(Backend.rank)(SimpleNamespace(legacy=None),rows)


class Session(h4.h2.Session):
    def __init__(self,cfg,approval,grant):
        authorize(cfg,approval,grant);self.cfg=cfg;self.root=Path(cfg['run_root']);self.root.mkdir(exist_ok=False)
        self.used=dict.fromkeys(KINDS,0);self.sequence=0;self.active=None
        save(self.root/'CONSUMED.json',dict(grant=grant,state='CONSUMED_NO_RETRY',approval_sha256=digest(approval)))
        self.marker_sha=sha(self.root/'CONSUMED.json')
        for name in ('results','ledger'):(self.root/name).mkdir()


def execute(cfg,approval,grant):
    s=Session(cfg,approval,grant);guard=h4.h2.old.ImportSeal(cfg)
    try:
        guard.install();s.take('bootstrap');data=s.call('bundle_load',lambda:validate(read(cfg['manifest'])))
        result=s.call('ranking',rank_rows,data);h4.h2.audit_end(s);verify(cfg,unused=False)
        save(s.root/'results/comparison.json',dict(series=SERIES,saved_methods=data['methods'],ranking=result,
            new_allowance_N_G_statistics=False,limitations=LIMITS,science_GO=False,strict_COMPARE='BLOCKED'))
        save(s.root/'completion.json',dict(status='RANK_FROM_SAVED_COMPLETE_NOT_REACCOUNTED',
            source_commit=cfg['source_commit'],contract_sha256=digest(cfg),used=s.used,
            comparison_sha256=sha(s.root/'results/comparison.json'),marker_sha256=s.marker_sha,mandatory_STOP=True))
        save(s.root/'STOP.json',dict(status='MANDATORY_STOP',used=s.used,retry=False,mandatory_STOP=True))
        return result
    except BaseException as exc:
        save(s.root/'STOP.json',dict(status='STOP_PARTIAL_NOT_PASS',reason=str(exc),used=s.used,retry=False,mandatory_STOP=True));raise
    finally:guard.remove()


def prepare(source):
    source=subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()
    ASSETS.mkdir(exist_ok=False);OUT.mkdir(parents=True,exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    data,files=collect();sources=[]
    for rel,(_,raw) in zip(NEW+REUSED,h4.h2.seal.git_batch(source,NEW+REUSED)):
        p=ROOT/rel;require(p.read_bytes()==raw,'COMMITTED_SOURCE')
        if rel in REUSED:require(raw==h4.h2.recovery.git_bytes(BASE,rel),'UNCHANGED_REUSED_CODE')
        files.append(dict(path=str(p),sha256=sha(p),role='SOURCE:runner'))
        sources.append(dict(path=rel,local_path=str(p),sha256=sha(p)))
    old=read(ROOT.parent/'h4-four-method-exploratory-assets-20261024/EXPLORATORY_contract_PRIVATE.json')
    runtime_rows=[r for r in old['files'] if r['role'].startswith('runtime:')]
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        require(all(pool.map(lambda r:sha(r['path'])==r['sha256'],runtime_rows)),'UNCHANGED_STDLIB_RUNTIME')
    files+=runtime_rows
    manifest=ASSETS/'saved_input_manifest_PRIVATE.json';save(manifest,data)
    files.append(dict(path=str(manifest),sha256=sha(manifest),role='INPUT:saved_projection'))
    rt=dict(old['runtime']);rt['sys_path']=[p for p in rt['sys_path'] if not p.endswith('/provider/src')]
    cfg=dict(schema=SERIES,source_commit=source,physical=PHYSICAL,candidates=CANDIDATES,budget=counts(),
        limitations=LIMITS,files=files,source_files=sources,manifest=str(manifest),manifest_sha256=sha(manifest),
        runtime=rt,run_root=str(ASSETS/'RANK_run01'),retry=False,mandatory_STOP=True)
    cp=ASSETS/'RANK_contract_PRIVATE.json';save(cp,cfg)
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    r=subprocess.run([str(h4.h2.seal.PYTHON),'-I','-S','-B',str(ROOT/'scripts/launch_h6_saved_four_method.py'),
        '--contract',str(cp)],env=env,capture_output=True,text=True)
    save(OUT/'cache/preflight_PRIVATE.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    require(r.returncode==0,'METADATA_PREFLIGHT:'+r.stderr[-1000:])
    def public(name,value):
        encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        require('/home/' not in encoded and '/usr/' not in encoded,'PUBLIC_PATH_PRIVACY')
        (OUT/name).write_text(encoded)
    public('saved_results_projection.json',data)
    public('metadata_preflight.json',json.loads(r.stdout))
    public('SOURCE.json',dict(source_commit=source,base_commit=BASE,
        files=[{k:v for k,v in x.items() if k!='local_path'} for x in sources],new_numeric_algorithm=False))
    public('execution_contract_DRAFT.json',dict(schema=SERIES,source_commit=source,contract_sha256=digest(cfg),
        input_manifest_sha256=sha(manifest),physical=PHYSICAL,candidates=CANDIDATES,exact_calls=counts(),
        kernel_events=3,new_scientific_operation='ranking1 only; saved row load1 and bootstrap1',
        reaccounting=False,new_science_started=False,grant_issued=False,limitations=LIMITS,retry=False,mandatory_STOP=True))
    public('authority_templates.json',dict(actual_grants_issued=0,
        approval=dict(issuer='USER_EXPLICIT',execution_authorized=False,contract_sha256=digest(cfg),accepted_limitations=LIMITS),
        grant=dict(id=None,purpose=SERIES,state='INVALID_AWAITING_FIXED_H6_APPROVAL',retry=False,
            contract_sha256=digest(cfg),approval_sha256=None,run_root='${H6_SAVED_ASSETS}/RANK_run01',budget=counts())))
    columns=['candidate','method','prefix','q','n','C_cos','C_sin','N_cos','N_sin','N_total_saved','G_cos_saved',
             'G_sin_saved','G_saved','G_SE_saved','C_cos_SE_saved','C_sin_SE_saved']
    with (OUT/'four_method_saved_values.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(columns)
        for row in data['methods']:
            can=row['candidate'];w.writerow([can['id'],can['method'],can['prefix'],can['q'],row['n']]+
                [row[k] for k in columns[5:]])
    public('STOP.json',dict(status='SAVED_RESULTS_RECOVERED_OPTIONAL_JOINT_RANK_APPROVAL_GATE',
        actual_calls=dict.fromkeys(KINDS,0),new_allowance_N_G_statistics=False,live_grants=0,mandatory_STOP=True))
    print(json.dumps(dict(status='SAVED_FOUR_METHOD_RESULTS_RECOVERED',source=source,contract_sha256=digest(cfg),
        manifest_sha256=sha(manifest),sealed_files=len(files),real_science_calls=0,grants_issued=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-source',required=True)
    prepare(p.parse_args().prepare_source)
