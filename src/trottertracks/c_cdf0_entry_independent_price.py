"""Fresh fixed-N price-only comparison; import/preflight are metadata only.

Reuse the existing sampler/compiler/full512 evaluator and Welch engine.
No selection, PF/RTE/RPE/DF solve or old-price restoration is implemented.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io

FAMILY='artifacts/research_c_independent_price_precision/2026-11-08'
PROPOSAL='docs/research/research_c_cdf0_entry_independent_price_proposal_20261108.json'
VERSION='CDF0_ENTRY_FRESH_TWO_ARM_FIXED128_PRICE_ONLY_V1'
N=128
RELOCATION='tail_linear_Z_to_existing_kappa_keep_frame_identity_v1'
OLD_CONTRACTS={
    'docs/research/research_c_cdf0_relocated_contract_rerun02_20261108.json':'045b15ea10ecfdc7ee86cd781e9601b923092951e10df8bfa2431c790be7e02c',
    'docs/research/research_c_h4_handoff_contract_20261106.json':'5454c539bc52cbf51fad6a8af6802d5acbe48ff1e341e1edf12ae60f19a66bbb'}
ENTRY_GROUND_SHA='6843e56b955a131583950c585287a41f61c08321b48618c1525b83d2361471b4'
OLD_STOPS=['artifacts/research_c_cdf0_relocated/2026-11-08/science/run02/STOP.json',
           'artifacts/research_c_h4_handoff/2026-11-06/science/run01/STOP.json']
PR_POLICY=dict(kind='partial ordered symmetric S2',s=1,h='1/10',delta='1/10',K=0,
               rounds=list(range(13)),state='own_DF_ground_state_oracle',state_preparation='EXCLUDED')
ARMS=dict(CDF0=dict(id='CDF0_STAGE_A_X0_V2_GAMMA0_FINAL_R8',sha256='f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02',r=64),
          ENTRY=dict(id='H4_100_C0_ENTRY_g003_budget64',sha256='0e1119dfb6be9b75fb5e592b7a65c76cda61e348fd7cad65d7f104bf997929f5',r=16))
STATISTICS=dict(samples=N,confidence=.95,primary_metric='rz_count',primary_lane='ideal_exact_native',
    method='scipy_Welch_fixed_N_independent_cohorts_affine_total_difference',
    precision_halfwidth_fraction_of_ENTRY=.05,skewness_abs_threshold=2.,
    maximum_delete_one_shift_fraction_of_halfwidth=.25,
    family='one prespecified new-CDF0-minus-new-ENTRY difference',old_samples_pooled=False,
    adaptive=False,interim_inference=False,secondary_inference=False)


def candidate_identity(arm):
    return ARMS[arm]['sha256']+':'+RELOCATION+':LD2'


def sampling(arm,batch):
    if arm not in ARMS or type(batch) is not int or not 0<=batch<N//32:
        raise ValueError('FIXED_ARM_BATCH')
    return dict(master=2026110803,cohort=f'cdf0_entry_independent_price_v1:{arm}:r{ARMS[arm]["r"]}:final:batch{batch:02d}',
        axis='INDEPENDENT_PRICE_FINAL_ONLY',samples=32,events_per_sample=ARMS[arm]['r'],
        PRNG='numpy.random.PCG64',paired=False,adaptive=False)


def seeds_for(policy,digest,candidate,*,role='baseline'):
    from .c1_stage3_connection import trajectory_seed
    return [trajectory_seed((policy['master'],policy['cohort'],digest),role,candidate,-1,policy['axis'],i,0) for i in range(32)]


def seed_manifest(digest,old_specs):
    old=[seed for s in old_specs for seed in seeds_for(s['policy'],s['SOURCE'],s['candidate'],role=s['role'])]
    rows=[]
    for arm in ARMS:
        for b in range(N//32):
            rows.extend(dict(id=f'{arm}/batch{b:02d}/sample{i:02d}',arm=arm,batch=b,sample=i,seed=seed)
                for i,seed in enumerate(seeds_for(sampling(arm,b),digest,candidate_identity(arm))))
    values=[x['seed'] for x in rows]
    if len(set(values))!=2*N or set(values)&set(old):raise ValueError('SEED_COLLISION_OR_OLD_OVERLAP')
    return dict(rows=rows,old_keys_checked=len(old),PRNG_draws=0,
        qualification='distinct hash-derived seeds only;IID or PCG64 joint-law independence not proved')


def call_limits():
    out=dict(kappa_load=1,shell_load=6,shell_gate=3,car_context=1,primary_comparison=1,result_summary=1)
    for arm in ARMS:
        limits=dict(candidate_load=1,reference_load=1,relocated_model=1,sector_context=1,relocated_sector=1,
            original_factor_gate=9,Hamiltonian_gate=1,ground_restore=1,ground_gate=1,full_CAR=1,
            law_prepare=1,law_gate=1,normalization_gate=1,full_sector_gate=1,reference_context=1,
            full_law_gate=1,gaussian_build=9,relative_gaussian=4,calibration_summary=N//32,
            final_calibration=1,new_rows_validation=1,arm_summary=1)
        limits.update({k:N for k in ('trajectory_sample','kernel_build','trajectory_reference','raw_operator_gate',
            'compile','compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics')})
        out.update({arm+'.'+k:v for k,v in limits.items()})
    return out


def old_seed_specs(cdf,entry):
    """Conservative old registered namespaces, including unexecuted CDF grid.

    Uses contracts/identities only; no price/event rows or PRNG draws.
    """
    def specs(candidate,source,role,prefix,pairs):
        key=prefix+'relocated:'+RELOCATION+':'+candidate
        return [dict(candidate=candidate,SOURCE=source,role=role,policy=dict(master=2026110501,
            cohort=f'integrated:{key}:{split}:r{r}:batch{batch:02d}',
            axis=f'INTEGRATED_{split.upper()}_PRICE_ONLY',samples=32,events_per_sample=r,
            PRNG='numpy.random.PCG64',paired=False,adaptive=False)) for split,r,batch in pairs]
    result=specs(candidate_identity('CDF0'),cdf['SOURCE_sha256'],'baseline','cdf0-20261108:',
        [('selection',r,0) for r in (8,16,32,64,128)]+[('final',64,b) for b in (0,1)])
    for label in ('P','E1','E2'):
        identity=entry['candidates'][label]['file_sha256']+':'+RELOCATION+':LD2'
        rs=(16,) if label=='E2' else (16,32)
        result+=specs(identity,entry['sampling_SOURCE_sha256'],'proposed' if label=='P' else 'baseline','',
            [('selection',r,0) for r in (8,16,32,64,128)]+[('final',r,b) for r in rs for b in (0,1)])
    return result


def saved_bindings(root,c):
    """Anchor common science and old namespaces to immutable old contracts."""
    saved=[]
    for name,want in OLD_CONTRACTS.items():
        if c['inputs'].get(name)!=want or io.sha(_local(root,name))!=want:raise ValueError('OLD_CONTRACT_PIN')
        old=io.read(root/name);saved.append(old)
        if c['inputs'].get(old['SOURCE'])!=old['SOURCE_sha256']:raise ValueError('OLD_SOURCE_PIN')
        for p,digest in old['inputs'].items():
            if c['inputs'].get(p)!=digest:raise ValueError('OLD_INPUT_SET:'+p)
        for key in ('input_identity','kappa_export','kappa_array_identity','shells','shell_reports','compiler','packages','target','correctness'):
            if c[key]!=old[key]:raise ValueError('COMMON_POLICY_DRIFT:'+key)
    cdf,entry=saved
    if c['physical_conditions']!=entry['physical_conditions'] or c['PR_policy']!=PR_POLICY:raise ValueError('PHYSICAL_PR_POLICY')
    if c['old_STOP']!=OLD_STOPS or any(p not in c['inputs'] for p in OLD_STOPS):raise ValueError('OLD_STOP_SET')
    if c['old_seed_specs']!=old_seed_specs(cdf,entry):raise ValueError('OLD_SEED_DOMAIN_DRIFT')
    for arm in ARMS:
        new=c['native_inputs'][arm]
        summary=(cdf['fresh_output']+'/final_summary.json') if arm=='CDF0' else cdf['ENTRY_summary']
        if new['summary']!=summary or summary not in c['inputs']:raise ValueError('OLD_SUMMARY_PIN')
        if arm=='CDF0':
            for k,old_key in (('candidate','path'),('reference','reference'),('ground','ground'),('physical_record','physical_record')):
                if new[k]!=cdf['candidate'][old_key]:raise ValueError('CDF0_SAVED_PATH')
        else:
            for k,old_key in (('candidate','path'),('reference','reference_path'),('physical','physical')):
                if new[k]!=entry['candidates']['E2'][old_key]:raise ValueError('ENTRY_SAVED_PATH')
            if c['inputs'][new['ground']]!=ENTRY_GROUND_SHA:raise ValueError('ENTRY_SAVED_GROUND_PIN')
    return saved


def planning_table(scalars,counts=((64,64),(96,96),(128,128),(128,64))):
    """Design proxy from ALREADY PUBLISHED variances; not a CI/test/reanalysis.

    The factor2 is a planning approximation, not a guaranteed t critical value.
    New outcomes, raw rows and recomputed historical moments are not accepted.
    """
    a,b=(scalars[k] for k in ARMS)
    if any(x.get('origin')!='published_scalar_summary' for x in (a,b)):
        raise ValueError('PUBLISHED_SUMMARY_ONLY')
    v=[x['W']**2*x['kernel_variance'] for x in (a,b)]
    result=[]
    for na,nb in counts:
        if any(type(n) is not int or n<2 for n in (na,nb)):raise ValueError('PLANNED_FIXED_N')
        result.append(dict(CDF0_n=na,ENTRY_n=nb,planning_factor=2.,
            planning_halfwidth_fraction=2*math.sqrt(v[0]/na+v[1]/nb)/b['saved_total_point'],
            inference_performed=False,precision_achievement_guaranteed=False))
    return dict(rows=result,allocation_equal_cost_proxy=math.sqrt(v[0]/v[1]),
        unequal_cost_allocation='sqrt(v_CDF0*cost_ENTRY/(v_ENTRY*cost_CDF0));runtime costs not measured here',
        actual_interval=None,actual_test=None,old_verdict_changed=False)


def _local(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root):raise ValueError('LOCAL_PATH')
    return p


def preflight(root,path=PROPOSAL):
    root=Path(root).resolve();c=io.read(_local(root,path))
    if (c['scope']!=VERSION or c['arms']!=ARMS or c['relocation']!=RELOCATION or c['statistics']!=STATISTICS
            or c['call_limits']!=call_limits() or c['maximum_scientific_calls']!=sum(call_limits().values())
            or c['retry'] or c['resume'] or not c['mandatory_stop']
            or c['execution_permission'] is not False or c['selection_repeated'] or c['old_samples_pooled']
            or c['fresh_output']!=FAMILY+'/science/run01' or c['sampling']!={a:[sampling(a,b) for b in range(N//32)] for a in ARMS}):
        raise ValueError('FIXED_INDEPENDENT_PRICE_DESIGN')
    if io.sha(_local(root,c['SOURCE']))!=c['SOURCE_sha256']:raise ValueError('NEW_SOURCE_MANIFEST')
    for name,want in io.read(root/c['SOURCE'])['files'].items():
        if io.sha(_local(root,name))!=want:raise ValueError('SOURCE_DRIFT:'+name)
    saved_bindings(root,c)
    missing=[]
    for name,want in c['inputs'].items():
        p=_local(root,name)
        if not p.exists():missing.append(dict(path=name,expected_sha256=want));continue
        if io.sha(p)!=want:raise ValueError('INPUT_DRIFT:'+name)
    # Missing raw E2 ground is an input gap, never a fallback eigensolve.
    if any(x['path']!=c['native_inputs']['ENTRY']['ground'] for x in missing):
        raise ValueError('UNEXPECTED_MISSING_INPUT')
    for arm in ARMS:
        r=c['native_inputs'][arm]
        if (r['candidate_sha256']!=ARMS[arm]['sha256'] or r['candidate'] not in c['inputs']
                or c['inputs'][r['candidate']]!=r['candidate_sha256'] or r['ground'] not in c['inputs']):
            raise ValueError('CANDIDATE_GROUND_BINDING')
        saved=io.read(root/r['summary'])
        result=saved['result'];binding=saved['bindings']['ideal_exact_native']
        if (result['candidate']!=candidate_identity(arm) or result['r']!=ARMS[arm]['r']
                or binding!=c['affine'][arm] or not result['native_admitted']
                or result['split']!='final' or result['samples']!=64
                or saved['bindings']['evidence'] is not None):raise ValueError('SAVED_RPE_BINDING')
    for stop in c['old_STOP']:
        s=io.read(root/stop)
        if s['failed'] or s['retry'] or s['resume'] or not s['mandatory_stop']:raise ValueError('OLD_STOP_IDENTITY')
    seeds=seed_manifest(c['SOURCE_sha256'],c['old_seed_specs'])
    if _local(root,c['fresh_output']).exists():raise FileExistsError('FRESH_NO_RETRY_RESUME')
    return c,dict(status='PASSIVE_CHECKS_INPUT_GAP' if missing else 'PASSIVE_CHECKS_PASS',
        missing_inputs=missing,new_sample_keys=len(seeds['rows']),old_seed_keys_checked=seeds['old_keys_checked'],
        execution_ready=not missing,execution_permission=False,scientific_calls=0,arrays_decoded=0,
        actual_CA_achieved=None,mandatory_stop=True)


def validate_new_rows(rows,arm,digest):
    """Rows include their actual event metadata, never a legacy row pool."""
    if len(rows)!=N:raise ValueError('FIXED_N_NO_DROPPING_OR_EXTENSION')
    expected={(b,i) for b in range(N//32) for i in range(32)};seen=set()
    for x in rows:
        key=(x['batch'],x['sample']);meta=x['event_metadata']
        if key in seen or key not in expected:raise ValueError('DUPLICATE_OR_OLD_SAMPLE_ID')
        seen.add(key)
        if (x['arm']!=arm or x['correctness']!='PASS' or meta['SOURCE']!=digest
                or meta['JU']!=candidate_identity(arm) or meta['sample']!=x['sample']
                or meta['sampling']!=sampling(arm,x['batch']) or meta['old_cohort_pooled']
                or meta['quantum_shots']!=0 or not meta['coherent_occurrences_independent']
                or len(meta['events'])!=ARMS[arm]['r']):
            raise ValueError('NEW_COHORT_SOURCE_CORRECTNESS')
    from .c_r32_price_statistics import validate_rows
    validate_rows([x['metrics'] for x in rows],N)
    return [x['metrics'] for x in sorted(rows,key=lambda x:(x['batch'],x['sample']))]


def collect_arm(root,c,out,call,arm,a,prepared,*,native_fn=None):
    from . import c_pr_integrated_performance as perf
    from . import c_near_r32_statistics as stats
    native_fn=native_fn or perf.native
    if (a['arm'].model.source_identity!=candidate_identity(arm) or a['arm'].model.input_identity!=c['input_identity']
            or a['arm'].plan.ld!=2 or a['config'].rte_steps!=ARMS[arm]['r']):
        raise ValueError('NATIVE_CANDIDATE_R_RELOCATION')
    metadata=[];runtime=dict(source_manifest_sha256=c['SOURCE_sha256'],shells=c['shells'])
    for batch in range(N//32):
        folder=out/f'batch{batch:02d}';folder.mkdir()
        rows,_=native_fn(root,runtime,folder,call,a,sampling=sampling(arm,batch),prepared=prepared,backend='process')
        if len(rows)!=32:raise ValueError('INCOMPLETE_BATCH_NO_EXCLUSIONS')
        for i,row in enumerate(rows):
            meta=io.read(folder/'samples'/f'sample{i:02d}_events.json')
            if io.read(folder/'samples'/f'sample{i:02d}_XY.json')['status']!='PASS':raise ValueError('CORRECTNESS_FAILURE')
            metadata.append(dict(arm=arm,batch=batch,sample=i,metrics=row,event_metadata=meta,correctness='PASS'))
    values=call('new_rows_validation',arm,lambda:validate_new_rows(metadata,arm,c['SOURCE_sha256']))
    cal=call('final_calibration',arm,lambda:perf.b2.calibration(values))
    summary=call('arm_summary',arm,lambda:stats.arm_summary(values,{'ideal_exact_native':c['affine'][arm]},
        None,calibration=cal,expected_n=N,statistics=STATISTICS))
    io.save(out/'summary.json',dict(candidate=candidate_identity(arm),r=ARMS[arm]['r'],
        rows=values,metadata=metadata,bindings=c['affine'][arm],statistics=summary,calibration=cal))
    return values,cal,summary


def _prepare_arm(root,c,out,call,arm,kappa,constant,shells,car):
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_h4_relocated_downstream as bridge
    from . import c_cdf0_relocated as cdf0
    from . import c_pr_aware_df_stage_b as b
    from . import c_rte_r_retuning as ret
    from .c_rc2_pr_stage_b import distribution_gate
    r=c['native_inputs'][arm]
    if not (root/r['ground']).exists():raise FileNotFoundError('SAVED_GROUND_REQUIRED_NO_SOLVE')
    data=call('candidate_load',arm,lambda:loader.load(root/r['candidate'],('J','U')))
    keys=('generators','H') if arm=='CDF0' else ('J','U','generators','H')
    sector=call('reference_load',arm,lambda:loader.load(root/r['reference'],keys))
    if arm=='CDF0':
        rel,a,gate=cdf0.connect(kappa,constant,data,sector,io.read(root/r['physical_record']),
            row=dict(sha256=r['candidate_sha256']),input_identity=c['input_identity'],
            output=out/'A',call=call,shells=io.read(root/c['shell_reports']),ground=root/r['ground'])
    else:
        rel,a,gate=bridge.prepare_candidate(kappa,constant,**data,saved_sector=sector,physical=r['physical'],
            candidate=r['candidate_sha256'],input_identity=c['input_identity'],role='baseline',
            out=out/'A',call=call,shell_reports=io.read(root/c['shell_reports']),saved_ground=root/r['ground'])
    if a is None:raise ValueError('SAVED_GROUND_NOT_QUALIFIED')
    tail,config,dist=call('law_prepare',arm,lambda:b.configure(a['arm'],Fraction(1,10),ARMS[arm]['r'],0))
    def law():
        distribution_gate(tail,config,dist,expected_steps=ARMS[arm]['r'])
        return bridge.tail_gate(a['ctx'],a['arm'],a['sector']['generators'],tail)
    io.save(out/'law.json',call('law_gate',arm,law))
    io.save(out/'normalization.json',call('normalization_gate',arm,lambda:ret.normalization(tail,config,ARMS[arm]['r'])))
    a.update(tail=tail,config=config,dist=dist)
    _,G=call('full_CAR',arm,lambda:bridge.raw_generators(rel,car=car))
    return a,bridge.native_context(a,car,G,shells,call=call)


def execute(root,path=PROPOSAL,*,execute_science=False):
    if not execute_science:raise PermissionError('NEW_SCIENTIFIC_EXECUTION_PERMISSION_REQUIRED')
    root=Path(root).resolve();c,status=preflight(root,path)
    if not status['execution_ready']:raise FileNotFoundError('MISSING_SAVED_INPUTS')
    # Only after explicit execution request + complete bytes/fresh checks.
    from . import c_h4_relocated_downstream as bridge
    from . import c_pr_aware_df_stage_b_runner as loader
    from . import c_pr_aware_df_stage_b2 as b2
    from . import c_near_r32_statistics as stats
    from .c1_stage1 import car_bilinears
    from .c1_stage3_integrated import Calls
    from .c1_h4_native_fair import profile_dict
    from qiskit import qpy
    if profile_dict()!=c['compiler']:raise ValueError('P0_COMPILER_DRIFT')
    for p,want in c['packages'].items():
        if importlib.metadata.version(p)!=want:raise ValueError('NUMERICAL_PACKAGE:'+p)
    contract_hash=io.sha(root/path)
    def work(task,out):
        calls=Calls(out,call_limits(),sum(call_limits().values()))
        try:
            io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['SOURCE']))
            exported=calls('kappa_load','saved kappa/scalar',lambda:loader.load(root/c['kappa_export'],('kappa','constant')))
            for key,value in exported.items():
                if io.array_record(value)!=c['kappa_array_identity'][key]:raise ValueError('KAPPA_SCALAR_IDENTITY')
            shells={}
            for label,paths in c['shells'].items():
                qs={}
                for kind,name in paths.items():
                    def load(name=name):
                        with (root/name).open('rb') as stream:items=qpy.load(stream)
                        if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                        return items[0]
                    qs[kind]=calls('shell_load',label+kind,load)
                shells[label],_=calls('shell_gate',label,lambda:b2.shell_assessment(qs['raw'],qs['compiled'],label))
            car=calls('car_context','shared H4 CAR',lambda:car_bilinears(8));values={}
            for arm in ARMS:
                folder=out/arm;folder.mkdir()
                def call(kind,label,fn,arm=arm):return calls(arm+'.'+kind,label,fn)
                call.stop=calls.stop
                a,prepared=_prepare_arm(root,c,folder,call,arm,exported['kappa'],float(exported['constant']),shells,car)
                values[arm]=collect_arm(root,c,folder,call,arm,a,prepared)
            comparison=calls('primary_comparison','new CDF0 minus new ENTRY',lambda:stats.primary(
                values['CDF0'][0],values['ENTRY'][0],c['affine']['CDF0'],c['affine']['ENTRY'],
                calibrations=[values['CDF0'][1],values['ENTRY'][1]],expected_n=N,statistics=STATISTICS))
            return calls('result_summary','one registered comparison',lambda:dict(comparison=comparison,
                comparison_label='new_CDF0_minus_new_ENTRY;legacy_PR_key_means_CDF0',old_results_changed=False,
                quantum_shots_drawn=0,actual_CA_achieved=None,mandatory_stop=True))
        finally:
            io.save(out/'call_summary.json',dict(counts=dict(calls.counts),first_failure=calls.first_failure))
            # The fixed output now exists; postflight verifies bytes without accepting a resume.
            if io.sha(root/path)!=contract_hash:raise ValueError('CONTRACT_CHANGED')
            if io.sha(root/c['SOURCE'])!=c['SOURCE_sha256']:raise ValueError('SOURCE_MANIFEST_CHANGED')
            for name,want in io.read(root/c['SOURCE'])['files'].items():
                if io.sha(root/name)!=want:raise ValueError('SOURCE_CHANGED')
            for name,want in c['inputs'].items():
                if io.sha(root/name)!=want:raise ValueError('INPUT_CHANGED')
    return bridge.run_fresh(root/c['fresh_output'],[dict(id='independent_comparison')],work)
