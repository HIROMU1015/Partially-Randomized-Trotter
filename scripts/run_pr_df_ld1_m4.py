"""Fixed PR-DF LD1 m4 holdout; reuse m3 worker/backend, never fit or regenerate predictions."""
import argparse
from copy import deepcopy
from fractions import Fraction
import faulthandler
import hashlib
import json
from pathlib import Path
import time
import traceback
import run_pr_df_ld1_m3 as M

ROOT=M.ROOT
PUBLIC='docs/research/d_pr_df_ld1_m4_holdout_results_20261115'
PRIVATE='/home/AbeHiromu/projects/d2-pr-df-ld1-m4-holdout-20261115-private/run01'
PRIOR={'commit':'2f423befbba5075bad4ede3057f89edd788d2661','path':'docs/research/d_pr_df_ld1_m3_results_20261115/INPUTS.json'}
FROZEN={'commit':'3cac53da1a4a8b2228245593c74c1ff9ec1f5212','path':'docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json'}
C_HANDOFF={'commit':'890755f0aeea3abfcdd5c849422bb198666a477e','path':'docs/research/pr_df_ld1_residual_freeze_20261115/HANDOFF_D2.md'}
VERDICT='NOT_ASSESSED_NO_ADOPTED_TOLERANCE'

def read_ref(ref):return json.loads(M.blob(ref['commit'],ref['path']))
def encode(value):return dict(numerator=value.numerator,denominator=value.denominator)
def rational(value):
    if set(value)!={'numerator','denominator'} or any(type(value[k]) is not int for k in value) or value['denominator']<=0:
        raise ValueError('Exact integer rational required')
    return Fraction(value['numerator'],value['denominator'])

def make_inputs():
    prior=read_ref(PRIOR);frozen=read_ref(FROZEN);fixed=prior['future_holdout']
    if frozen['holdout_metadata']['row']!=fixed['row'] or frozen['holdout_metadata']['cell']!=fixed['cell']:
        raise ValueError('Frozen holdout metadata differs from pre-acquisition plan')
    if frozen['holdout_source']!=dict(PRIOR,selector=['future_holdout']) or frozen['comparison_rule']!=prior['model_plan']['holdout_rule']:
        raise ValueError('Frozen holdout SOURCE/rule differs')
    c=fixed['row']['candidate'];cell=fixed['cell']
    if (c['id'],c['q'],c['R_outer'],c['T'],cell['cohort'],cell['path_indices'])!=('3df0cfd3356c358aeda8',16,32,1.6,'D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1',[0,1,2,3]):
        raise ValueError('m4 scope differs')
    if len(cell['seeds'])!=4 or any(len(row)!=16 for row in cell['seeds']):raise ValueError('Fixed 4 x 16 seeds required')
    cal=[*prior['calibration_saved_cells'],prior['cells'][0]]
    if {s for r in cell['seeds'] for s in r}&{s for v in cal for r in v['seeds'] for s in r}:raise ValueError('Holdout/calibration seed overlap')
    predictions={}
    for axis in ('X','Y'):
        model=frozen['models'][axis];p=model['m4_prediction']
        if (model['axis'],p['axis'],p['m'],p['physical_id'],p['q'],p['R_outer'],p['T'])!=(axis,axis,4,c['id'],16,32,1.6):raise ValueError('Frozen prediction identity differs')
        if p['ideal_structural']['law_binding']!=fixed['row']['law_binding']:raise ValueError('Frozen law/exact kappa differs')
        if p['observed_RZ'] is not None or p['errors'] is not None or p['model_verdict']!=VERDICT:raise ValueError('Frozen prediction already observed/judged')
        if model['qualified_for_candidate'] is not False:raise ValueError('Unqualified frozen model required')
        if rational(p['predicted_RZ'])!=Fraction(217164876899444503,1683627180032):raise ValueError('Frozen exact prediction differs')
        predictions[axis]=dict(source=dict(FROZEN,selector=['models',axis,'m4_prediction']),predicted_RZ=deepcopy(p['predicted_RZ']),
                               identity=deepcopy(p['identity']),semantics=p['semantics'])
    plan=deepcopy(prior);active=deepcopy(cell);active['role']='HOLDOUT'
    plan.update(schema='PR_DF_LD1_M4_FIXED_HOLDOUT_V1',cells=[active],menus={'H4':dict(prior['menus']['H4'],candidates=[deepcopy(fixed['row'])])},
        public=PUBLIC,private=PRIVATE,prior_input_source=PRIOR,frozen_model_source=FROZEN,C_handoff=C_HANDOFF,
        predictions=predictions,comparison_rule=deepcopy(frozen['comparison_rule']),seed_input='READ_FIXED_FUTURE_HOLDOUT_TABLE',
        not_in_scope=['m3 reexecution','other paths/rounds','refit','prediction regeneration','CI','qualified E[C]','N/G/winner','retry'])
    return plan


def install_fixed_seed_sampler(ctx,plan,private=None):
    """Original sample_requests body with literal saved seed lookup; no trial_seed regeneration."""
    cell=plan['cells'][0];candidate=plan['menus']['H4']['candidates'][0]['candidate']
    original=ctx.h['sample_requests'];calls=[]
    if original.__globals__ is not ctx.h:raise ValueError('Sampler globals are not the active context')
    def lookup(row,cohort,index,outer,piece):
        if (row!=candidate or cohort!=cell['cohort'] or type(index) is not int or not 0<=index<4
                or type(outer) is not int or not 0<=outer<candidate['q'] or piece!=0):raise ValueError('Fixed seed lookup identity/order differs')
        seed=cell['seeds'][index][outer];calls.append((index,outer,piece,seed));return seed
    def sample(row,prep,cohort,index):
        begin=len(calls);result=original(row,prep,cohort,index)
        expected=[(index,outer,0,seed) for outer,seed in enumerate(cell['seeds'][index])]
        if calls[begin:]!=expected or result[1]!=cell['seeds'][index]:raise ValueError('Fixed seeds were not used by the original sampler')
        if private is not None:
            M.progress(Path(private)/f'PROGRESS_p{index}.jsonl','fixed_seed_lookup','CHECKED',index=index,seeds=result[1],
                       source=dict(PRIOR,selector=['future_holdout','cell','seeds',index]))
        return result
    ctx.h['trial_seed']=lookup;ctx.h['sample_requests']=sample
    return calls


def compare(predictions,rows):
    """Only exact P versus admitted C mean; never re-add ideal structural/residual terms."""
    if set(predictions)!={'X','Y'}:raise ValueError('Separate X/Y frozen predictions required')
    units=[]
    if any(r.get('axis') not in ('X','Y') for r in rows):raise ValueError('Unknown observation axis')
    for axis in ('X','Y'):
        selected=[r for r in rows if r['axis']==axis];reasons=[]
        if sorted(r['index'] for r in selected)!=[0,1,2,3]:reasons.append('INCOMPLETE_OR_DUPLICATE_FIXED_FOUR_PATHS')
        for r in selected:
            if r.get('valid') is not True:reasons.append(r.get('reason') or 'RECEIPT_INVALID')
            if type(r.get('C')) is not int or r['C']<0:reasons.append('INVALID_INTEGER_COMPILED_PRICE')
        p=rational(predictions[axis]['predicted_RZ'])
        entry=dict(m=4,axis=axis,n_paths=4,P=encode(p),O=None,signed_error=None,absolute_error=None,
            signed_relative_error=None,absolute_relative_error=None,valid=not reasons,reasons=sorted(set(reasons)),
            prediction_source=predictions[axis]['source'],receipts=selected,model_verdict=VERDICT)
        if not reasons:
            observed=Fraction(sum(r['C'] for r in selected),4);d=p-observed
            entry.update(O=encode(observed),signed_error=encode(d),absolute_error=encode(abs(d)))
            if observed:entry.update(signed_relative_error=encode(d/observed),absolute_relative_error=encode(abs(d)/observed))
            else:entry['reasons']=['ZERO_OBSERVED_PRICE']
        units.append(entry)
    status=('INCOMPLETE_OR_INVALID' if not all(r['valid'] for r in units) else
            'DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED' if any(r['O']['numerator']==0 for r in units) else 'DESCRIPTIVE_COMPLETE')
    return dict(campaign=status,model_verdict=VERDICT,tolerance=None,units=units,refit=False,prediction_regeneration=False,
                qualified_E_C=False,law_I_Q_distance=None,CI=None,N=None,G=None,winner=None,mandatory_stop=True)


def execute(source,inputs):
    if M.git('rev-parse','HEAD')!=source or M.git('status','--porcelain','--untracked-files=no'):raise ValueError('Fixed committed execution SOURCE required')
    plan=json.loads(inputs.read_text())
    if plan!=make_inputs():raise ValueError('Fixed m4 INPUTS differ')
    for path in ('src/trottertracks/p1_efficient_zz.py','src/trottertracks/compiled_price_fast.py','src/trottertracks/efficient_controlled_zz.py','scripts/h46_b3_fast_fingerprint.py'):
        if (ROOT/path).read_bytes()!=M.blob(M.BASE,path):raise ValueError('Fixed P2 pipeline changed:'+path)
    for item in plan['saved_input_files'].values():
        if M.R.file_sha(item['path'])!=item['sha256']:raise ValueError('Saved scientific input differs')
    out=ROOT/PUBLIC/'run01';private=Path(PRIVATE)
    out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();records=[];log=private/'PARENT_PROGRESS.jsonl'
    base=dict(execution_source=source,pipeline_source=M.BASE,inputs_sha256=M.R.file_sha(inputs),compile_budget=8,resources_before=M.R.resources(),mandatory_stop=True)
    stack=(private/'PARENT_STACK.txt').open('a');faulthandler.dump_traceback_later(60,repeat=True,file=stack)
    try:
        M.progress(log,'context','START')
        science=M.P.load_c1_backend(ROOT)['r_price_phase1_science'];cls=M.F.make_backend_class(science.Backend,science)
        backend=cls(ROOT,M.P.execution_plan(plan,source,ROOT));ctx=backend.context('H4')
        if ctx.prep.preparation_hash!=plan['expected_preparation_hash'] or float(ctx.scalar).hex()!=plan['scalar_saved']['hex']:raise ValueError('Preparation/scalar differs')
        install_fixed_seed_sampler(ctx,plan,private)
        import trotterlib.df_rte_qiskit as dfq
        M.progress(log,'context','END');faulthandler.cancel_dump_traceback_later();stack.close()
        M.STATE.update(plan=plan,ctx=ctx,backend=backend,science=science,dfq=dfq,private=str(private))
        M.R.write(out/'PREPARATION_CHECK.json',dict(source=source,preparation_hash=ctx.prep.preparation_hash,scalar=ctx.scalar,
            scalar_hex=float(ctx.scalar).hex(),fragment_order=list(ctx.prep.deterministic_fragment_indices),status='SAVED_CONTEXT_CHECKED',
            fixed_seed_lookup_installed=True,full_wrapper_statevector_validation=False))
        records=M.acquire_four(out)
        success=len(records)==4 and all(r['ok'] and r['exitcode']==0 for r in records)
        logs=[]
        for r in records:
            entry=(r.get('value') or {}).get('compile_log')
            if entry is None:
                stop=private/f"STOP_p{r['index']}.json";entry=json.loads(stop.read_text())['compile_log'] if stop.exists() else []
            logs.extend(entry)
        M.R.write(out/'RESULT.json',dict(base,status='M4_SC_PAIRS_ACQUIRED' if success else 'STOP_PARTIAL',records=records,
            new_compiles=sum(r['status']=='COMPLETE' for r in logs),compile_attempts=len(logs),total_wall_seconds=time.perf_counter()-start,
            resources_after=M.R.resources(),fit=False,prediction_regeneration=False,full_wrapper_statevector_validation=False,all_law_E_C=None,CI=None,N=None,G=None))
        return 0 if success else 2
    except BaseException:
        M.R.write(out/'STOP.json',dict(base,status='STOP',records=records,error=traceback.format_exc(),total_wall_seconds=time.perf_counter()-start));raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if not stack.closed:stack.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--source');parser.add_argument('--inputs',type=Path)
    parser.add_argument('--compare-admitted',type=Path);parser.add_argument('--comparison-out',type=Path)
    args=parser.parse_args()
    if args.prepare:M.R.write(args.prepare,make_inputs());return 0
    if args.compare_admitted:
        plan=json.loads(args.inputs.read_text());frozen=read_ref(FROZEN)
        predictions={axis:dict(predicted_RZ=frozen['models'][axis]['m4_prediction']['predicted_RZ'],
            source=dict(FROZEN,selector=['models',axis,'m4_prediction'])) for axis in ('X','Y')}
        if predictions!={a:{k:plan['predictions'][a][k] for k in ('predicted_RZ','source')} for a in ('X','Y')}:raise ValueError('Frozen P differs from preopen input')
        M.R.write(args.comparison_out,compare(predictions,json.loads(args.compare_admitted.read_text())['rows']));return 0
    if not(args.execute and args.source and args.inputs):parser.error('Use --prepare, --execute or --compare-admitted')
    return execute(args.source,args.inputs)

if __name__=='__main__':raise SystemExit(main())
