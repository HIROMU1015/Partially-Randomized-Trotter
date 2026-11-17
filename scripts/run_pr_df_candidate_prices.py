"""Candidate-owned PR-DF price acquisition/model path, using fixed D2 workers.

Preparation preserves the supplied mapper's q/R/step. No prefix, kappa, step or
shot exploration is performed here. Root coordinator owns publication sequence.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from fractions import Fraction
import hashlib
import fcntl
import importlib.util
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CALLER='scripts/run_pr_df_candidate_prices.py'
FIXED_VIEW=ROOT.parent/'d2-pr-df-ld1-m3-acquisition-20261115'
VIEW_COMMIT='326feadd093920bbea29b498108aa4eae6f67a75'
REUSED='ea9e66092465d39dfcd4aeb6fea8e9f09808cabb'
ADAPTER_SOURCE='e827367629ca6c2920d9d908fc1631314399db86'
SOLVER=('d83737b6d5241962ce39c81d39c56af3506c0cdd','src/trottertracks/price_model_validation.py')
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
BASE_INPUTS=dict(commit='e0b3cad2a024070389f757160f3a20d7da02c5ff',path='docs/research/d_pr_df_ld_split_results_20261115/ACQUISITION_INPUTS.json')
SCALAR_CONSTANT=dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/SCALAR.json',selector=['hamiltonian_constant'])
RUNTIME=None
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)


def blob(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=ROOT)
def read(loc):
    value=json.loads(blob(loc['commit'],loc['path']))
    for k in loc.get('selector',[]):value=value[k]
    return value

def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)
def dec(x):return Fraction(x['numerator'],x['denominator'])if isinstance(x,dict)else Fraction(x)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def fixed(source):
    if blob(source,CALLER)!=Path(__file__).read_bytes():raise ValueError('Unfixed DF price runner SOURCE')
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def runtime():
    global RUNTIME
    if RUNTIME is not None:return RUNTIME
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=FIXED_VIEW,text=True).strip()!=VIEW_COMMIT:raise ValueError('DF worker view differs')
    path='scripts/run_pr_df_ld_split.py'
    if (FIXED_VIEW/path).read_bytes()!=blob(REUSED,path):raise ValueError('Reused DF path worker bytes differ')
    # The normal controller has already imported its trottertracks package.
    # sys.path alone cannot expose legacy-only helpers to that existing package.
    # Add the verified provider as a temporary fallback, preserving all loaded
    # controller modules and restoring both search paths after the import.
    import trottertracks
    old_path=list(sys.path);old_package_path=list(trottertracks.__path__)
    try:
        sys.path[:0]=[str(FIXED_VIEW/'src'),str(FIXED_VIEW/'scripts')]
        legacy_package=str(FIXED_VIEW/'src/trottertracks')
        if legacy_package not in trottertracks.__path__:trottertracks.__path__.append(legacy_package)
        for helper in ('p1_efficient_zz','efficient_controlled_zz'):
            relative='src/trottertracks/'+helper+'.py'
            if (FIXED_VIEW/relative).read_bytes()!=blob(VIEW_COMMIT,relative):
                raise ValueError('Fixed DF helper SOURCE differs: '+helper)
            loaded=sys.modules.get('trottertracks.'+helper)
            if loaded is not None and Path(loaded.__file__).read_bytes()!=blob(VIEW_COMMIT,relative):
                raise ValueError('Already loaded DF helper SOURCE differs: '+helper)
        spec=importlib.util.spec_from_file_location('_candidate_df_price_worker',FIXED_VIEW/path)
        R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
    finally:
        sys.path[:]=old_path;trottertracks.__path__[:]=old_package_path
    R.ROOT=ROOT;RUNTIME=R;return R


def solve(matrix,rhs):
    tree=ast.parse(blob(*SOLVER));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_solve')
    namespace=dict(Fraction=Fraction)
    exec(compile(ast.Module(body=[node],type_ignores=[]),SOLVER[0]+':_solve','exec'),namespace)
    return namespace['_solve'](matrix,rhs)


def design(config,covered_rounds=None):
    all_rows=config['rounds'];s=config['spec'];bad=[]
    covered=list(range(13))if covered_rounds is None else sorted(covered_rounds)
    if not covered or len(set(covered))!=len(covered)or any(type(m)is not int or m not in range(13)for m in covered):
        raise ValueError('INVALID_DF_PRICE_ROUND_COVERAGE')
    rows=[all_rows[m]for m in covered]
    if s['method']!='PR'or s['split'].get('tail_dictionary'):bad.append('PR_DF_REQUIRED')
    if s['formula']!='2nd'or s['finite_K']!=2 or any(r['structure']['signed_weights']!=[1.]or len(r['structure']['allocation'])!=1 for r in rows):
        bad.append('PRICE_COEFFICIENTS_FOR_CHANGED_FORMULA_PIECES_OR_K_NOT_ADOPTED')
    if len({dec(r['actual_delta_pf'])for r in rows})!=1:bad.append('MULTIPLE_ACTUAL_STEPS_REQUIRE_SEPARATE_CALIBRATION_GROUPS')
    if len(all_rows)!=13 or [r['m']for r in all_rows]!=list(range(13)):bad.append('CURRENT_H4_13_ROUND_SCOPE_REQUIRED')
    if len(rows)==1 and covered_rounds is not None:
        return dict(status='CALIBRATION_DESIGN_READY'if not bad else'NEEDS_PRICE_RULE_OR_DESIGN_EXTENSION',reasons=bad,
                    mode='DIRECT_FINITE_COHORT',calibration_rounds=covered,holdout_round=None,matrix=[[Fraction(1)]],covered_rounds=covered)
    mode='FULL_AFFINE_Q_R1'if all(r['R_outer']==1 for r in rows)else'FULL_AFFINE_QR'
    dimension=2 if mode=='FULL_AFFINE_Q_R1'else 3
    offsets=[0,6]if dimension==2 else[0,3,6]
    calibration=[covered[min(i,len(covered)-1)]for i in offsets]
    def features(m):
        r=all_rows[m]
        return [Fraction(1),Fraction(r['q_PF'])]+([]if dimension==2 else[Fraction(r['q_PF']*r['R_outer'])])
    def full_rank(ms):
        if len(set(ms))!=dimension:return False
        try:solve([features(m)for m in ms],[Fraction(0)]*dimension);return True
        except ValueError:return False
    if covered_rounds is not None and not full_rank(calibration):
        calibration=next((list(ms)for ms in itertools.combinations(covered,dimension)if full_rank(ms)),calibration)
    preferred=covered[min(3 if dimension==2 else 4,len(covered)-1)]
    holdout=preferred if preferred not in calibration else next((m for m in covered if m not in calibration),None)
    if holdout is None:bad.append('INDEPENDENT_SAME_STEP_HOLDOUT_ROUND_UNAVAILABLE')
    matrix=[features(m)for m in calibration]
    try:solve(matrix,[Fraction(0)]*len(matrix))
    except ValueError:bad.append('RANK_DEFICIENT_HEAD_Q_AND_TAIL_QR_COEFFICIENTS')
    return dict(status='CALIBRATION_DESIGN_READY'if not bad else'NEEDS_PRICE_RULE_OR_DESIGN_EXTENSION',reasons=bad,
                mode=mode,calibration_rounds=calibration,holdout_round=holdout,matrix=matrix,
                **({}if covered_rounds is None else dict(covered_rounds=covered)))


def design_groups(config,quote=None):
    """Keep every supplied mapper row; identify only missing actual-step inputs."""
    seen=set();missing=set();actions=[]
    for row in (quote or{}).get('rows',[]):
        key=row['m'],row['axis']
        if key in seen:raise ValueError('DUPLICATE_DF_PRICE_AXIS')
        seen.add(key);r=config['rounds'][row['m']]
        if (row.get('q')!=r['q_PF']or row.get('R')!=r['R_outer']or
                row.get('physical_id')not in(r['candidate']['id'],r.get('physical_candidate',{}).get('id'))):
            raise ValueError('DF_QUOTE_PHYSICAL_SCHEDULE_DIFFERS')
        if row.get('unit_price_RZ')is None:missing.add(key)
        elif dec(row['unit_price_RZ'])<=0 or not row.get('source'):raise ValueError('UNSOURCED_OR_NONPOSITIVE_DF_PRICE')
    if quote is not None and (quote.get('configuration_id')!=config['id']or seen!={(m,a)for m in range(13)for a in('X','Y')}):
        raise ValueError('DF_QUOTE_CONFIGURATION_OR_COVERAGE_DIFFERS')
    if quote is None:missing={(m,a)for m in range(13)for a in('X','Y')}
    grouped={}
    for row in config['rounds']:grouped.setdefault(dec(row['actual_delta_pf']),[]).append(row['m'])
    groups=[]
    for step,ms in grouped.items():
        keys=[dict(m=m,axis=a)for m in ms for a in('X','Y')if(m,a)in missing]
        if not keys:continue
        selected=design(config,ms)
        payload=dict(configuration_family=dict(binding=config['spec']['binding'],split=config['spec']['split'],
            formula=config['spec']['formula'],K=config['spec']['finite_K'],law=config['spec']['law'],profile=config['spec']['profile']),
            actual_step=enc(step),cells=[dict(m=m,q=config['rounds'][m]['q_PF'],R=config['rounds'][m]['R_outer'],
                T=config['rounds'][m]['T'],structure=config['rounds'][m]['structure'])
                for m in selected['calibration_rounds']+([]if selected['holdout_round']is None else[selected['holdout_round']])])
        gid=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        groups.append(dict(group_id=gid,actual_step=enc(step),covered_rounds=ms,axis_keys=keys,
                           design={k:v for k,v in selected.items()if k!='matrix'},status=selected['status']))
    for m in range(13):
        for axis in('X','Y'):
            g=next((v for v in groups if m in v['covered_rounds']),None)
            actions.append(dict(m=m,axis=axis,action='REUSE_PRICE'if(m,axis)not in missing else
                'CALIBRATION_REQUEST'if g and g['status']=='CALIBRATION_DESIGN_READY'else'UNSUPPORTED',
                group_id=None if g is None else g['group_id'],reasons=[]if g is None else g['design']['reasons']))
    return dict(status='REUSE_SUPPLIED_PRICES_NO_CALIBRATION'if not groups else'ACTUAL_STEP_GROUPS_PLANNED',
                groups=groups,round_actions=actions,execution_performed=False)



GATE='INPUT_PREPARATION_REQUEST_LOWERING_COMPILER_FINGERPRINT_CHECKED'
RECEIPT_ERRORS=(ValueError,KeyError,TypeError,StopIteration,OSError,subprocess.CalledProcessError)
PHYSICAL_FIELDS=('physical_candidate','structure','binding','split','one_body_policy','law','price_profile','implementation','expected_scalar')


def request_physical(request):
    """Keep physical + implementation contracts; only SOURCE epochs/axes may differ.

    The original saved execution/input locators remain in every reused witness.
    We do not require a new configuration ID when only declaration metadata changed.
    """
    context=request['physical_context']
    return {k:context[k]for k in PHYSICAL_FIELDS}


def validate_axis(cell,index,axis,witness,reader=None,source_reader=None):
    """Verify an individual original path AND its separately stored axis receipt."""
    reader=reader or read;source_reader=source_reader or blob
    path_loc=witness['path_receipt'];axis_loc=witness['receipt']
    for loc in (path_loc,axis_loc):
        if not loc.get('commit')or not loc.get('path'):raise ValueError('UNFIXED_RECEIPT_LOCATOR')
    if Path(axis_loc['path']).name!={'X':'cosine.json','Y':'sine.json'}[axis]:raise ValueError('AXIS_RECEIPT_PATH_MISMATCH')
    value=reader(path_loc);record=value['record'];request=record['request'];pc=request['physical_context']
    canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    copy=deepcopy(request);request_id=copy.pop('request_id')
    if request_id!=canonical(copy)or request['context_key']!=canonical(pc):raise ValueError('REQUEST_OR_CONTEXT_HASH_MISMATCH')
    if value.get('status')!='VALID'or value.get('gate_status')!=GATE:raise ValueError('INVALID_OR_UNCHECKED_PATH_RECEIPT')
    if value.get('m')!=cell['m']or value.get('index')!=index or request.get('m')!=cell['m']:raise ValueError('CELL_OR_PATH_MISMATCH')
    if record.get('cohort')!=cell['cohort']or record.get('index')!=index or record.get('seeds')!=cell['seeds'][index]:raise ValueError('COHORT_INDEX_OR_SEED_MISMATCH')
    if request_physical(request)!=request_physical(cell['request']):raise ValueError('PHYSICAL_REQUEST_OR_IMPLEMENTATION_MISMATCH')
    axes=[v for v in record['records']if v.get('axis')==axis]
    if len(axes)!=1 or axis not in request['axes']:raise ValueError('AXIS_MISSING_OR_DUPLICATE')
    a=axes[0];raw=reader(axis_loc)
    if a['C']!=a['metrics']['rz_count']or a['D']!=a['C']-a['S']or type(a['C'])is not int or a['C']<0:raise ValueError('METRIC_IDENTITY_MISMATCH')
    if any(type(a['metrics'].get(k))is not int or a['metrics'][k]<0 for k in ('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')):raise ValueError('INVALID_SIX_COMPILER_METRICS')
    if raw.get('metrics')!=a['metrics']or raw.get('fingerprint')!=a['fingerprint']or raw.get('compiler')!=pc['price_profile']['compiler']:raise ValueError('AXIS_RECEIPT_FINGERPRINT_OR_COMPILER_MISMATCH')
    if not record.get('event_digest')or not a.get('fingerprint'):raise ValueError('MISSING_EVENT_OR_WRAPPER_FINGERPRINT')
    execution=pc['execution_source']
    if record.get('source',{}).get('execution')!=execution or record.get('source',{}).get('input')!=pc['input_source']:raise ValueError('SOURCE_BINDING_MISMATCH')
    if not execution.get('commit')or not execution.get('path')or not source_reader(execution['commit'],execution['path']):raise ValueError('MISSING_EXECUTION_SOURCE')
    resolved=record.get('resolved_context',{})
    if record.get('resolved_context_key')!=canonical(dict(context=pc,resolved=resolved)):raise ValueError('RESOLVED_CONTEXT_HASH_MISMATCH')
    candidate=pc['physical_candidate'];prefix=candidate['prefix']
    if resolved.get('deterministic_fragments')!=list(range(prefix))or resolved.get('randomized_fragments')!=list(range(prefix,pc['binding']['rank'])):raise ValueError('RESOLVED_HEAD_TAIL_ORDER_MISMATCH')
    if float(dec(pc['split']['lambda_R'])).hex()!=float(resolved['lambda_R']).hex():raise ValueError('RESOLVED_TAIL_LAMBDA_MISMATCH')
    if resolved.get('scalar')!=pc['expected_scalar']or not resolved.get('preparation_hash')or not resolved.get('tail_hash'):raise ValueError('RESOLVED_PREPARATION_OR_SCALAR_MISMATCH')
    expected=pc.get('preparation_hash')
    if expected is not None and expected!=resolved['preparation_hash']:raise ValueError('PREPARATION_HASH_MISMATCH')
    for key,want in (('axis',axis),('index',index),('cohort',cell['cohort']),('seeds',record['seeds']),('fingerprint',a['fingerprint']),('event_digest',record['event_digest'])):
        if key in witness and witness[key]!=want:raise ValueError('WITNESS_'+key.upper()+'_MISMATCH')
    if 'physical_id'in witness and witness['physical_id']!=pc['physical_candidate']['id']:raise ValueError('WITNESS_PHYSICAL_ID_MISMATCH')
    return dict(status='REUSE',receipt=axis_loc,path_receipt=path_loc,execution_source=execution,
                axis=axis,index=index,cohort=cell['cohort'],seeds=record['seeds'],event_digest=record['event_digest'],
                fingerprint=a['fingerprint'],S=a['S'],C=a['C'],D=a['D'])


def direct_witnesses(alternative,reader=None):
    """Registry schemas: saved mean.paths OR explicit receipt_source rows."""
    reader=reader or read
    if alternative.get('receipt_source'):
        rows=reader(alternative['receipt_source'])
    else:
        row=reader(alternative['source']);rows=row.get('paths',[])
    if not isinstance(rows,list):raise ValueError('UNSUPPORTED_INDIVIDUAL_RECEIPT_SCHEMA')
    return rows


def resolve_reuse(cells,quote,reader=None,source_reader=None):
    """Resolve only exact frozen cohort paths. Never mix a different cohort.

    An unresolved alternative holds its own axis, not the whole candidate. Other
    missing axes/cells remain runnable. Original finite prices are retained.
    """
    reader=reader or read;source_reader=source_reader or blob
    summary=[]
    for cell in cells:
        cell['axis_evidence']={str(i):{a:dict(status='MISSING')for a in ('X','Y')}for i in cell['indices']}
        for axis in ('X','Y'):
            alternatives=[d for row in (quote or {}).get('rows',[])if(row['m'],row['axis'])==(cell['m'],axis)for d in row.get('direct_alternatives',[])]
            if not alternatives:continue
            errors=[];accepted={};conflicts=set()
            for alternative in alternatives:
                try:
                    # Holdout must remain independent; a calibration role is not holdout.
                    role=alternative.get('role')or''
                    if cell['role']=='HOLDOUT'and role not in ('HOLDOUT','INDEPENDENT_HOLDOUT'):raise ValueError('SAVED_ROLE_IS_NOT_INDEPENDENT_HOLDOUT')
                    if cell['role']=='CALIBRATION'and'HOLDOUT'in role:raise ValueError('DO_NOT_REUSE_HOLDOUT_FOR_CALIBRATION')
                    if alternative.get('cohort')!=cell['cohort']:raise ValueError('INDEPENDENT_COHORT_REQUIRES_EXPLICIT_INPUT_ADOPTION_NO_POOLING')
                    witnesses=[w for w in direct_witnesses(alternative,reader)if w.get('axis',axis)==axis]
                    if not witnesses:raise ValueError('MEAN_WITHOUT_INDIVIDUAL_PATH_RECEIPTS')
                    for witness in witnesses:
                        i=witness.get('index')
                        if i not in cell['indices']:continue
                        try:
                            value=validate_axis(cell,i,axis,witness,reader,source_reader)
                            if i in accepted and accepted[i]!=value:
                                conflicts.add(i);raise ValueError('AMBIGUOUS_MULTIPLE_PATH_RECEIPTS')
                            accepted[i]=value
                        except RECEIPT_ERRORS as exc:errors.append('path'+str(i)+': '+str(exc))
                    if any(i not in accepted for i in cell['indices']):errors.append('EXISTING_FINITE_PRICE_HAS_UNRESOLVED_INDIVIDUAL_PATHS_NO_DUPLICATE_ACQUISITION')
                except RECEIPT_ERRORS as exc:errors.append(str(exc))
            for i in conflicts:accepted.pop(i,None)
            for i in cell['indices']:
                if i in accepted:cell['axis_evidence'][str(i)][axis]=accepted[i]
                elif errors:cell['axis_evidence'][str(i)][axis]=dict(status='HELD_UNRESOLVED_SAVED_AXIS',reasons=errors)
            summary.append(dict(m=cell['m'],axis=axis,alternatives=deepcopy(alternatives),accepted_indices=sorted(accepted),unresolved_reasons=errors))
    return summary


def adopt_saved_calibration_cohorts(cells,quote,reader=None,source_reader=None):
    """Adopt complete verifiable saved cohorts before the new INPUTS are fixed.

    At least one valid X/Y receipt for each of the four path indices is enough
    to recover the original seed matrix. Missing axes can then be acquired from
    those exact paths. We select by metadata/SOURCE order, never observed price.
    Independent holdout evidence is never reclassified as calibration.
    """
    reader=reader or read;source_reader=source_reader or blob;decisions=[]
    for cell in cells:
        if cell['role']!='CALIBRATION':continue
        groups={};rejected=[]
        for row in (quote or {}).get('rows',[]):
            if row['m']!=cell['m']:continue
            for alternative in row.get('direct_alternatives',[]):
                role=alternative.get('role')or'';cohort=alternative.get('cohort')
                if role in ('HOLDOUT','INDEPENDENT_HOLDOUT')or'HOLDOUT'in role:
                    rejected.append(dict(cohort=cohort,source=alternative.get('source'),reason='PRESERVE_SAVED_HOLDOUT_ROLE'));continue
                if not cohort or not role:
                    rejected.append(dict(cohort=cohort,source=alternative.get('source'),reason='SAVED_COHORT_OR_ROLE_UNIDENTIFIED'));continue
                groups.setdefault(cohort,[]).append((row['axis'],alternative))
        candidates=[]
        for cohort,alternatives in sorted(groups.items()):
            trial=deepcopy(cell);trial['cohort']=cohort;seeds={};valid={};evidence=[];reasons=[];conflicts=set()
            alternatives.sort(key=lambda pair:(json.dumps(pair[1].get('source'),sort_keys=True,separators=(',',':')),pair[0]))
            for axis,alternative in alternatives:
                try:witnesses=direct_witnesses(alternative,reader)
                except RECEIPT_ERRORS as exc:reasons.append(str(exc));continue
                for witness in witnesses:
                    if witness.get('axis',axis)!=axis:continue
                    index=witness.get('index')
                    if index not in cell['indices']:continue
                    try:
                        record=reader(witness['path_receipt'])['record'];saved_seeds=record['seeds']
                        q=cell['request']['physical_context']['physical_candidate']['q']
                        if len(saved_seeds)!=q or any(type(v)is not int or v<0 for v in saved_seeds):raise ValueError('INVALID_SAVED_OUTER_SEEDS')
                        trial['seeds'][index]=saved_seeds
                        accepted=validate_axis(trial,index,axis,witness,reader,source_reader)
                        if index in seeds and seeds[index]!=saved_seeds:
                            conflicts.add(index);raise ValueError('SAME_COHORT_INDEX_HAS_CONFLICTING_SEEDS')
                        if any(i==index and v['event_digest']!=accepted['event_digest']for(i,a),v in valid.items()):
                            conflicts.add(index);raise ValueError('SAME_COHORT_INDEX_HAS_CONFLICTING_EVENTS')
                        if(index,axis)in valid and valid[index,axis]['fingerprint']!=accepted['fingerprint']:
                            conflicts.add(index);raise ValueError('SAME_COHORT_INDEX_AXIS_HAS_CONFLICTING_FINGERPRINT')
                        seeds[index]=saved_seeds;valid[index,axis]=accepted
                        evidence.append(dict(axis=axis,index=index,source=alternative['source'],role=alternative['role'],path_receipt=witness['path_receipt'],receipt=witness['receipt']))
                    except RECEIPT_ERRORS as exc:reasons.append('path'+str(index)+'/'+axis+': '+str(exc))
            if set(seeds)==set(cell['indices'])and not conflicts:
                # Each seed row came from a valid source-bound axis receipt.
                key=(cohort,json.dumps([v['source']for v in evidence],sort_keys=True,separators=(',',':')))
                candidates.append((key,[seeds[i]for i in cell['indices']],evidence))
            else:rejected.append(dict(cohort=cohort,reason='SAVED_COHORT_NOT_FULLY_VERIFIABLE_FOR_FIXED_PATH_INDICES',verified_indices=sorted(seeds),conflicting_indices=sorted(conflicts),details=reasons))
        if candidates:
            key,seeds,evidence=min(candidates,key=lambda v:v[0]);old=cell['cohort'];cell['cohort']=key[0];cell['seeds']=deepcopy(seeds)
            decisions.append(dict(m=cell['m'],status='ADOPTED_SAVED_CALIBRATION_COHORT',replaced_planned_cohort=old,cohort=key[0],indices=cell['indices'],seeds=seeds,evidence=evidence,
                                  selection='LEXICOGRAPHIC_COHORT_AND_SOURCE_LOCATORS_NOT_PRICE',other_verified_cohorts=[v[0][0]for v in candidates if v[0]!=key],rejected=rejected))
        else:decisions.append(dict(m=cell['m'],status='KEEP_NEW_FIXED_COHORT',cohort=cell['cohort'],rejected=rejected))
    return decisions

def scalar_metadata(config,tail_descriptor=None):
    """Resolve the original saved tail/scalar contract; no native preparation.

    Both calibration and optional validation use this same fixed input mapping.
    Historical canonical configurations intentionally need not embed scalar.
    """
    index=read(INDEX);descriptor=index['T20_reference_files']['H4'];data=Path(descriptor['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:raise ValueError('Saved tail descriptor bytes differ')
    context=json.loads(data);ld=config['spec']['L_D']
    if tail_descriptor is None:
        tail=context['prefixes'][str(ld)]
        tail_source=dict(index=INDEX,context_sha256=descriptor['sha256'],selector=['prefixes',str(ld)])
    else:
        extension_spec=importlib.util.spec_from_file_location('_df_price_descriptor_extension',ROOT/'src/trottertracks/pr_df_descriptor_extension.py')
        extension=importlib.util.module_from_spec(extension_spec);extension_spec.loader.exec_module(extension)
        tail=extension.validate(extension.read(ROOT,tail_descriptor),tail_descriptor,config);tail_source=tail_descriptor
    if config['spec']['split']['lambda_source']!=tail_source or float(dec(config['spec']['split']['lambda_R'])).hex()!=float(tail['lambda_R']).hex():raise ValueError('Candidate tail binding differs')
    constant=read(SCALAR_CONSTANT)
    if constant['source']['hamiltonian_hash']!=config['spec']['binding']['hamiltonian']:raise ValueError('Hamiltonian constant binding differs')
    scalar=constant['value']+tail['scalar'];scalar_record=dict(hamiltonian_constant=constant,tail_identity=dict(value=tail['scalar'],source=tail_source),total=scalar,total_hex=scalar.hex(),application='ONCE_AT_TOTAL_T')
    return dict(tail_source=tail_source,scalar=scalar,record=scalar_record)


def prepare(config_source,output,source,cohort,price_quote=None,tail_descriptor=None,covered_rounds=None):
    fixed(source);config=read(config_source);output=ROOT/output;output.mkdir(parents=True,exist_ok=True)
    selected=design(config,covered_rounds)
    if price_quote is not None:
        if price_quote.get('configuration_id')!=config['id']:raise ValueError('Price quote belongs to another configuration')
        write(output/'PRICE_QUOTE.json',price_quote)
    if selected['reasons']:
        write(output/'NEEDS.json',{k:v for k,v in selected.items()if k!='matrix'});return
    R=runtime();resolved=scalar_metadata(config,tail_descriptor)
    scalar=resolved['scalar'];tail_source=resolved['tail_source'];scalar_record=resolved['record']
    write(output/'SCALAR.json',scalar_record)
    seed=R.P.trial_seed_function(ROOT);cells=[]
    for m,n,role in [(m,4,'CALIBRATION')for m in selected['calibration_rounds']]+([]if selected['holdout_round']is None else[(selected['holdout_round'],2,'HOLDOUT')]):
        request=R.D.from_b(config,m,config['rounds'][m]['physical_candidate'],
            input_source=dict(configuration=config_source,tail=tail_source,scalar=dict(hamiltonian_constant=SCALAR_CONSTANT,tail=tail_source)),
            execution_source=dict(commit=source,path=CALLER),expected_scalar=scalar)
        c=request['physical_context']['physical_candidate'];name=cohort+'_'+role
        seeds=[[seed(c,name,i,j,0)for j in range(c['q'])]for i in range(n)]
        cells.append(dict(m=m,role=role,request=request,cohort=name,indices=list(range(n)),seeds=seeds))
    adoption_summary=adopt_saved_calibration_cohorts(cells,price_quote)
    reuse_summary=resolve_reuse(cells,price_quote)
    all_seeds=[v for c in cells for row in c['seeds']for v in row]
    if len(set(all_seeds))!=len(all_seeds):raise ValueError('Calibration/holdout seed overlap')
    old=read(BASE_INPUTS)
    write(output/'INPUTS.json',dict(schema='PR_DF_CANDIDATE_PRICE_INPUTS_V1',source=source,configuration=config_source,configuration_id=config['id'],output_directory=rel(output),
        **({}if covered_rounds is None else dict(covered_rounds=selected['covered_rounds'])),
        cohort_adoption=adoption_summary,price_reuse=reuse_summary,tail_descriptor=tail_descriptor,cells=cells,model_design={k:v for k,v in selected.items()if k!='matrix'},runtime=old['runtime'],saved_input_files=old['saved_input_files'],
        local_support=old['local_support'],reused_worker=REUSED,adapter_source=ADAPTER_SOURCE,scalar=scalar_record,
        max_new_paths=sum(any(v['status']=='MISSING'for v in path.values())for c in cells for path in c['axis_evidence'].values()),max_wrapper_compiles=sum(v['status']=='MISSING'for c in cells for path in c['axis_evidence'].values()for v in path.values()),
        full_wrapper_statevector_validation=False,numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',
        model_qualification='FINITE_COHORT_EXPLORATORY_REFERENCE_NOT_EXPECTATION'))


def load_inputs(path,commit,source):
    fixed(source);path=Path(path).resolve();raw=blob(commit,rel(path))
    if raw!=path.read_bytes():raise ValueError('Input bytes differ from fixed input commit')
    inputs=json.loads(raw)
    if inputs['source']!=source:raise ValueError('Execution SOURCE mismatch')
    return inputs


def selected_acquire_path(job):
    """Fixed worker with only output suffix, selected-axis cap and zip changed.

    Adapter already accepts selected axes. All original sampling, wrapper,
    scalar, metric/fingerprint, compiler, cache and failure checks are retained.
    """
    R=runtime();tree=ast.parse(blob(REUSED,'scripts/run_pr_df_ld_split.py'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='acquire_path')
    count={'directory':0,'limit':0,'zip':0}
    class Select(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if any(isinstance(t,ast.Name)and t.id=='directory'for t in node.targets):
                count['directory']+=1
                node.value=ast.BinOp(left=node.value,op=ast.Div(),right=ast.parse("'axes_'+''.join(req['axes'])",mode='eval').body)
            return node
        def visit_Compare(self,node):
            self.generic_visit(node)
            if ast.unparse(node)=='len(compiles) >= 2':
                count['limit']+=1;node.comparators[0]=ast.parse("len(req['axes'])",mode='eval').body
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name)and node.func.id=='zip'and len(node.args)==2 and ast.unparse(node.args[1])=="('X', 'Y')":
                count['zip']+=1;node.args[1]=ast.parse("req['axes']",mode='eval').body
            return node
    fn=Select().visit(fn)
    if count!={'directory':1,'limit':1,'zip':1}:raise ValueError('Fixed worker axis specialization does not match SOURCE')
    scope=dict(R.__dict__);module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(module,REUSED+':selected_acquire_path','exec'),scope)
    return scope['acquire_path'](job)


def selected_request(request,axes):
    selected=deepcopy(request);selected['axes']=list(axes);selected.pop('request_id')
    selected['request_id']=hashlib.sha256(json.dumps(selected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return selected


def local_witness_reader(locator):
    if locator['commit']=='WORKING':
        value=json.loads((ROOT/locator['path']).read_text())
        for key in locator.get('selector',[]):value=value[key]
        return value
    return read(locator)


def scan_path_axes(cell,index,out):
    """Read only this path's known output files and published reuse witnesses."""
    evidence={a:deepcopy(cell.get('axis_evidence',{}).get(str(index),{}).get(a,dict(status='MISSING')))for a in ('X','Y')}
    for axis,v in list(evidence.items()):
        if v['status']=='REUSE':
            try:evidence[axis]=validate_axis(cell,index,axis,v,reader=local_witness_reader)
            except RECEIPT_ERRORS as exc:evidence[axis]=dict(status='HELD_INVALID_SAVED_AXIS',reasons=[str(exc)])
    pathdir=out/f'm{cell["m"]}'/f'p{index}'
    for directory in [pathdir]+[pathdir/('axes_'+a)for a in ('XY','X','Y')]:
        path=directory/'PATH_RECEIPT.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        for row in value.get('record',{}).get('records',[]):
            axis=row.get('axis')
            if axis not in evidence:continue
            witness=dict(path_receipt=dict(commit='WORKING',path=rel(path)),receipt=dict(commit='WORKING',path=rel(directory/({'X':'cosine','Y':'sine'}[axis]+'.json'))))
            try:valid=validate_axis(cell,index,axis,witness,reader=local_witness_reader)
            except RECEIPT_ERRORS as exc:
                evidence[axis]=dict(status='HELD_INVALID_LOCAL_AXIS',reasons=[str(exc)]);continue
            if evidence[axis]['status']=='REUSE'and evidence[axis]['fingerprint']!=valid['fingerprint']:
                evidence[axis]=dict(status='HELD_CONFLICTING_RECEIPTS',reasons=['Same path/axis has inconsistent fingerprints'])
            else:evidence[axis]=valid
    valid=[v for v in evidence.values()if v['status']=='REUSE']
    if len(valid)==2 and valid[0]['event_digest']!=valid[1]['event_digest']:
        evidence={a:dict(status='HELD_EVENT_DIGEST_MISMATCH',reasons=['X/Y must be same path'])for a in evidence}
    return evidence


def _acquire(inputs,input_commit,stage,source,model_commit=None):
    R=runtime();base=inputs['output_directory'];R.PUBLIC=base+'/'+stage
    for path,want in inputs.get('local_support',{}).items():
        if hashlib.sha256((FIXED_VIEW/path).read_bytes()).hexdigest()!=want:raise ValueError('Reused support SOURCE differs: '+path)
    cells=[c for c in inputs['cells']if c['role']==('CALIBRATION'if stage=='calibration'else'HOLDOUT')]
    if stage=='holdout':
        if not model_commit:raise ValueError('Independent holdout requires published frozen model')
        model=read(dict(commit=model_commit,path=base+'/FROZEN_MODELS.json'));prices=read(dict(commit=model_commit,path=base+'/AXIS_PRICES.json'))
        covered=inputs.get('covered_rounds',list(range(13)))
        if model['status']!='FROZEN_2_AXIS_MODELS'or len(prices['rows'])!=2*len(covered) or any(dec(r['C_ref'])<=0 for r in prices['rows']):raise ValueError('Positive frozen model prerequisite')
    out=ROOT/R.PUBLIC/'acquisition';out.mkdir(parents=True,exist_ok=True);pending=[];processes=[];start=time.perf_counter();initial={};new_results=[]
    for cell in cells:
        for index in cell['indices']:
            evidence=scan_path_axes(cell,index,out);initial[cell['m'],index]=evidence
            axes=[a for a in ('X','Y')if evidence[a]['status']=='MISSING']
            marker=out/f'STARTED_m{cell["m"]}_p{index}_axes{"".join(axes)}.json'
            old_marker=out/f'STARTED_m{cell["m"]}_p{index}.json';stop=out/f'STOP_m{cell["m"]}_p{index}.json'
            if axes and (marker.exists()or old_marker.exists()or stop.exists()):
                for a in axes:evidence[a]=dict(status='HELD_PRIOR_START_OR_FAILURE_NO_RETRY')
                axes=[]
            if axes:
                selected=deepcopy(cell);selected['request']=selected_request(cell['request'],axes);pending.append((selected,index))
    if pending:
        ham,metadata=R.restore(pending[0][0]['request'],inputs['saved_input_files'],inputs['runtime']);contexts={};runtimes={}
        for cell in cells:
            if not any(pair[0]['m']==cell['m']for pair in pending):continue
            pc=cell['request']['physical_context'];r=R.D.NativeRuntime(ROOT,ham,binding=pc['binding'],input_source=pc['input_source'],execution_source=pc['execution_source'])
            runtimes[cell['m']]=r;contexts[cell['m']]=r.context(cell['request'])
        preparation=dict(metadata=metadata,contexts={str(m):dict(preparation_hash=c.prep.preparation_hash,scalar=c.scalar,head=list(c.prep.deterministic_fragment_indices),tail=list(c.prep.randomized_block_indices))for m,c in contexts.items()})
        if (out/'PREPARATION.json').exists():
            old=json.loads((out/'PREPARATION.json').read_text())
            if old['metadata']!=metadata or any(old['contexts'].get(m)!=value for m,value in preparation['contexts'].items()if m in old['contexts']):raise ValueError('Preparation identity differs on resume')
            preparation['contexts']={**old['contexts'],**preparation['contexts']};atomic(out/'PREPARATION.json',preparation)
        else:write(out/'PREPARATION.json',preparation)
        R.STATE.update(inputs=dict(inputs,cache_root=str(out/'isolated_cache')),runtimes=runtimes,contexts=contexts)
        ctx=mp.get_context('fork');flag=ctx.Event()
        with ProcessPoolExecutor(max_workers=len(pending),mp_context=ctx,initializer=R.init_price,initargs=(flag,))as pool:
            futures={}
            for pair in pending:
                cell,index=pair;axes=cell['request']['axes'];write(out/f'STARTED_m{cell["m"]}_p{index}_axes{"".join(axes)}.json',dict(source=source,inputs_commit=input_commit,axes=axes,epoch=time.time(),automatic_retry=False))
                futures[pool.submit(selected_acquire_path,pair)]=(cell['m'],index,axes)
            processes=list(pool._processes.values())
            while futures:
                done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    m,i,axes=futures.pop(future)
                    try:value=future.result()
                    except BaseException:value=dict(m=m,index=i,status='WORKER_ERROR',error=traceback.format_exc(),compiles=[])
                    new_results.append(value)
                print('PRICE_PROGRESS',stage,'finished',len(new_results),'pending',list(futures.values()),flush=True)
    rows=[]
    for cell in cells:
        for index in cell['indices']:
            evidence=scan_path_axes(cell,index,out)
            # Preserve the no-retry state detected before workers launched.
            for axis,v in initial[cell['m'],index].items():
                if v['status'].startswith('HELD_PRIOR')and evidence[axis]['status']=='MISSING':evidence[axis]=v
            good=all(v['status']=='REUSE'for v in evidence.values())
            row=dict(m=cell['m'],index=index,status='VALID'if good else'PARTIAL_OR_HELD',axes=evidence)
            atomic(out/f'm{cell["m"]}'/f'p{index}'/'PATH_INDEX.json',row);rows.append(row)
    good=all(r['status']=='VALID'for r in rows)
    reused=sum(v['status']=='REUSE'for evidence in initial.values()for v in evidence.values())
    atomic(out/'RESULT.json',dict(status='ACQUIRED'if good else'STOP_PARTIAL',rows=rows,
        new_paths=len(pending),new_independent_paths=sum(all(v['status']=='MISSING'for v in initial[c['m'],i].values())for c,i in pending),resumed_partial_paths=sum(any(v['status']=='REUSE'for v in initial[c['m'],i].values())for c,i in pending),reused_axes=reused,new_wrapper_axes=sum(len(c['request']['axes'])for c,i in pending),
        held_axes=sum(v['status']!='REUSE'for r in rows for v in r['axes'].values()),
        compile_attempts=sum(len(r['compiles'])for r in new_results),completed_compiles=sum(c['status']=='COMPLETE'for r in new_results for c in r['compiles']),
        source=source,inputs_commit=input_commit,model_commit=model_commit,elapsed_seconds=time.perf_counter()-start,
        process_exits=[dict(pid=p.pid,exitcode=p.exitcode)for p in processes],automatic_retry=False))
    return good

def acquire(inputs,input_commit,stage,source,model_commit=None):
    directory=ROOT/inputs['output_directory']/stage/'acquisition';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'DRIVER.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise ValueError('Another acquisition owns this candidate/stage')
    try:return _acquire(inputs,input_commit,stage,source,model_commit)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def means(inputs,raw_commit,stage):
    base=inputs['output_directory'];role='CALIBRATION'if stage=='calibration'else'HOLDOUT';out=[]
    for cell in [c for c in inputs['cells']if c['role']==role]:
        m=cell['m'];n=len(cell['indices'])
        for axis in ('X','Y'):
            paths=[]
            for index in cell['indices']:
                index_path=base+f'/{stage}/acquisition/m{m}/p{index}/PATH_INDEX.json'
                carrier=read(dict(commit=raw_commit,path=index_path));witness=deepcopy(carrier['axes'][axis])
                if witness.get('status')!='REUSE':raise ValueError(f'Incomplete frozen {m}/{axis}/{index}: '+witness.get('status','MISSING'))
                for key in ('receipt','path_receipt'):
                    if witness[key]['commit']=='WORKING':witness[key]['commit']=raw_commit
                valid=validate_axis(cell,index,axis,witness)
                paths.append({k:v for k,v in valid.items()if k!='status'})
            c=cell['request']['physical_context']['physical_candidate'];out.append(dict(m=m,axis=axis,q=c['q'],R=c['R_outer'],physical_id=c['id'],n_paths=n,cohort=cell['cohort'],mean_C=enc(sum(Fraction(p['C'])for p in paths)/n),paths=paths))
    return out

def freeze(inputs,raw_commit,source):
    cfg=read(inputs['configuration']);mode=inputs['model_design']['mode'];cells=means(inputs,raw_commit,'calibration');models={};rows=[]
    covered=inputs.get('covered_rounds',list(range(13)))
    if len({dec(cfg['rounds'][m]['actual_delta_pf'])for m in covered})!=1:
        raise ValueError('DF_MODEL_MIXES_DIFFERENT_ACTUAL_STEPS')
    for axis in ('X','Y'):
        own=[c for c in cells if c['axis']==axis]
        if mode=='DIRECT_FINITE_COHORT':
            if len(own)!=1 or len(covered)!=1 or own[0]['m']!=covered[0]:raise ValueError('DIRECT_DF_POINT_SCOPE_DIFFERS')
            co=[];model=dict(features=[],offset=0,calibration=own,coefficients={},direct=True,
                independent_validation='FINITE_COHORT_DIRECT_POINT_ONLY_NOT_FIT')
        else:
            matrix=[[Fraction(1),Fraction(c['q'])]+([]if mode=='FULL_AFFINE_Q_R1'else[Fraction(c['q']*c['R'])])for c in own]
            co=solve(matrix,[dec(c['mean_C'])for c in own]);model=dict(features=['one','q']if len(co)==2 else['one','q','qR'],offset=0,
                calibration=own,calibration_residuals=[enc(0)]*len(own),independent_validation='NOT_YET_OPENED',coefficients={k:enc(v)for k,v in zip(('one','q','qR'),co)})
            if len(co)==2:model.update(alpha=enc(co[0]),beta=enc(co[1]))
        models[axis]=model
        for r in [cfg['rounds'][m]for m in covered]:
            q,Rr=r['q_PF'],r['R_outer'];p=dec(own[0]['mean_C'])if mode=='DIRECT_FINITE_COHORT'else co[0]+co[1]*q+(co[2]*q*Rr if len(co)==3 else 0)
            if p<=0:raise ValueError('Nonpositive candidate model price; no clipping/refit')
            rows.append(dict(L_D=cfg['spec']['L_D'],configuration_id=cfg['id'],m=r['m'],axis=axis,physical_id=r['physical_candidate']['id'],q=q,R=Rr,C_ref=enc(p),
                application='DIRECT_FINITE_COHORT'if mode=='DIRECT_FINITE_COHORT'else'CALIBRATION'if r['m']in inputs['model_design']['calibration_rounds']else'LONG_ROUND_EXTRAPOLATION'if r['m']>max(inputs['model_design']['calibration_rounds'])else'UNCHECKED_INTERPOLATION',law=r['law_binding']))
    output=ROOT/inputs['output_directory'];write(output/'FROZEN_MODELS.json',dict(status='FROZEN_2_AXIS_MODELS',source=source,raw_commit=raw_commit,mode=mode,models=models,
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',finite_cohort_is_law_expectation=False,holdout_not_opened=True,formula='FULL_WRAPPER_C_OFFSET_ZERO_NO_S_ADDITION',
        **({}if 'covered_rounds'not in inputs else dict(covered_rounds=covered,actual_step=cfg['rounds'][covered[0]]['actual_delta_pf']))))
    write(output/'AXIS_PRICES.json',dict(source=source,rows=sorted(rows,key=lambda r:(r['m'],r['axis'])),positive_finite_axes=2*len(covered)))


def compare(inputs,raw_commit,model_commit,source):
    base=inputs['output_directory'];cells=means(inputs,raw_commit,'holdout');prices=read(dict(commit=model_commit,path=base+'/AXIS_PRICES.json'));rows=[]
    for c in cells:
        p=next(v for v in prices['rows']if(v['m'],v['axis'])==(c['m'],c['axis']));P=dec(p['C_ref']);O=dec(c['mean_C']);d=P-O
        rows.append(dict(**c,prediction=enc(P),observed=enc(O),signed_error=enc(d),absolute_error=enc(abs(d)),signed_relative_error=None if O==0 else enc(d/O),absolute_relative_error=None if O==0 else enc(abs(d)/O),relative_reason='ZERO_OBSERVED_PRICE'if O==0 else None))
    write(ROOT/base/'HOLDOUT_COMPARISON.json',dict(source=source,raw_commit=raw_commit,model_commit=model_commit,rows=rows,status='DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED'if any(dec(r['observed'])==0 for r in rows)else'DESCRIPTIVE_COMPLETE',model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',refit=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','acquire','freeze','compare-holdout'));ap.add_argument('--source',required=True)
    ap.add_argument('--tail-descriptor',type=Path);ap.add_argument('--price-quote',type=Path);ap.add_argument('--configuration');ap.add_argument('--selector',default='[]');ap.add_argument('--output');ap.add_argument('--cohort');ap.add_argument('--input',type=Path);ap.add_argument('--inputs-commit');ap.add_argument('--stage',choices=('calibration','holdout'));ap.add_argument('--raw-commit');ap.add_argument('--model-commit');ap.add_argument('--covered-rounds',type=json.loads);a=ap.parse_args()
    if a.mode=='prepare':
        commit,path=a.configuration.split(':',1);prepare(dict(commit=commit,path=path,selector=json.loads(a.selector)),a.output,a.source,a.cohort,None if a.price_quote is None else json.loads(a.price_quote.read_text()),None if a.tail_descriptor is None else json.loads(a.tail_descriptor.read_text()),a.covered_rounds)
    else:
        inputs=load_inputs(a.input,a.inputs_commit,a.source)
        if a.mode=='acquire':return 0 if acquire(inputs,a.inputs_commit,a.stage,a.source,a.model_commit)else 2
        if a.mode=='freeze':freeze(inputs,a.raw_commit,a.source)
        else:compare(inputs,a.raw_commit,a.model_commit,a.source)
    return 0


if __name__=='__main__':raise SystemExit(main())
