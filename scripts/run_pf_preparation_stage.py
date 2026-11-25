"""PF leaf phases using the existing D numerical compiler and C receipt guard.

The native compile body is reused from D d8780090f acquire(). Receipt admission
is a separate phase; neither phase chooses a candidate, shots or a new scope.
This adapter runs only in the task-owned provider workspace.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import runpy
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import FunctionType


def bind_saved_signal_paths():
    """Relocate only frozen H4 loader I/O; retain its code and hash checks.

    The provider's receipt and context retain their original absolute paths.
    The worker supplies an explicit map to byte-verified attempt-local copies.
    Numerical functions and the original provider files are never rewritten.
    """
    from trottertracks import pf_evidence_preparation as E
    helper = Path(__file__).resolve().parents[1] / 'src/trottertracks/rpe_preparation_paths.py'
    resolve = runpy.run_path(str(helper))['resolve']
    original = E.N.load_shared_h4
    scope = dict(original.__globals__)
    original_path = scope['Path']

    def mapped_path(value, *parts):
        # Hash/read helpers in the frozen saved-snapshot loader all receive
        # this same constructor, including paths stored inside the receipt.
        return original_path(resolve(str(value)), *parts)

    scope['Path'] = mapped_path
    relocated = FunctionType(original.__code__, scope, original.__name__,
                             original.__defaults__, original.__closure__)
    relocated.__kwdefaults__ = original.__kwdefaults__
    E.N.load_shared_h4 = relocated
    return relocated


def cache_mean_pool(root, directory):
    """Reuse the fixed PF provider's saved-round locator index across planners."""
    from trottertracks import pf_evidence_preparation as E
    if subprocess.run(['git','diff','--quiet','HEAD','--','src','scripts'],cwd=root).returncode != 0:
        return
    trees=subprocess.check_output(['git','rev-parse','HEAD:src','HEAD:scripts'],cwd=root,text=True).splitlines()
    helper=Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_saved_signal_index.py'
    functions=runpy.run_path(str(helper))
    load_or_build,fixed_locator=functions['load_or_build'],functions['fixed_locator']
    original=E.mean_pool

    def pool(root):
        def build():
            # make_plan reads the locator. The fixed saved_record and bind_mean
            # still validate the selected original numerical row.
            rows=[[key,[dict(source=hit['source']) for hit in hits]]
                  for key,hits in original(root).items()]
            return rows,all(fixed_locator(hit['source']) for _,hits in rows for hit in hits)

        return {tuple(key):hits for key,hits in load_or_build(directory,['pf-mean-pool-v1',trees],build)}

    E.mean_pool=pool


def saved_price_helpers():
    path = Path(__file__).resolve().parents[1]/'src/trottertracks/pf_saved_prices.py'
    return runpy.run_path(str(path))


def bind_preflight_snapshot():
    """Resolve only the frozen inputs read by the native duplicate check."""
    directory = os.environ.get('RPE_PF_PREFLIGHT_SNAPSHOT')
    if not directory:
        return
    import run_pf_m0_prices as native
    original = native.duplicate_check
    baseline = native.read(dict(commit=native.BASELINE,
        path='docs/research/d_pf_stage2_holdout_20261115/DUPLICATE_AND_INPUT_PREFLIGHT.json'))
    mapping = {item['path']: str(Path(directory)/Path(item['path']).name)
               for item in baseline['input_bytes_rechecked'].values()}
    scope = dict(original.__globals__)
    original_path = scope['Path']

    def mapped_path(value, *parts):
        path = original_path(value, *parts)
        return original_path(mapping.get(str(path), path))

    scope['Path'] = mapped_path
    native.duplicate_check = FunctionType(original.__code__, scope, original.__name__,
                                         original.__defaults__, original.__closure__)
    native.duplicate_check.__kwdefaults__ = original.__kwdefaults__


PRICE_PLAN_HOLDS = {
    'At least two distinct-q calibration cells and one later holdout required',
    'One saved axis only: resolve per-axis native execution before compiling',
}


def exploratory_direct_cells(requests, direct, *, allow_single_axis):
    """Only when a fixed model cannot be designed: explicit missing direct axes.

    The selected-axis provider still runs its ordinary duplicate preflight and
    C admission.  No coefficient or price is transported across actual widths.
    """
    existing = {(row['m'], row['axis']) for row in direct}
    cells = []
    for row in requests:
        axes = [axis for axis in ('X', 'Y') if (row['m'], axis) not in existing]
        if axes and (allow_single_axis or len(axes) == 2):
            cells.append(dict(m=row['m'], axes=axes, role='EARLY_DIRECT'))
    return cells


def use_exploratory_plan(*, recover_partial=False):
    from trottertracks import pf_evidence_preparation as evidence
    original = evidence.make_plan

    def plan(root, cfg, source, **options):
        try:
            value = original(root, cfg, source, **options)
        except ValueError as error:
            if str(error) not in PRICE_PLAN_HOLDS:
                raise
            prepared = evidence.plan(root, cfg, source)
            direct = evidence.accepted_direct(root, prepared)
            pool = evidence.mean_pool(str(root))
            means = []
            for item in prepared['pending']['requests']:
                physical = item['physical_candidate']
                hits = pool.get(evidence.physical_key(physical), [])
                means.append(dict(m=item['m'], physical_id=physical['id'],
                    acquisition='REUSE_SAVED_MEAN' if hits else 'NEW_MEAN_REQUIRED',
                    saved_source=hits[0]['source'] if hits else None))
            value = dict(schema='D_PF_SELECTED_EVIDENCE_PLAN_V1', configuration=cfg, prepared=prepared,
                         direct_prices=direct, signal_rounds=means, new_signal_means=sum(not r['saved_source'] for r in means),
                         calibration_rounds=[], holdout_round=None, new_compile_cells=[], actual_width=cfg['spec']['delta_pf'],
                         new_Q3=0, price_plan_issue=str(error))
            # Initial broad signal screening may leave price unknown.  The
            # axis-capable shortlist recheck supplies only missing wrappers,
            # even when the strict model+holdout design cannot exist.
            if options.get('allow_single_axis'):
                value.update(selected_axis_compile=True,
                             new_compile_cells=exploratory_direct_cells(
                                 prepared['pending']['requests'], direct, allow_single_axis=True))
        if recover_partial and options.get('allow_single_axis'):
            # The previous exploratory publication did not price the full
            # requested vector. Do not reserve its same incomplete model
            # cohort again: give every missing axis to the normal duplicate
            # and original-owner preflight. Only clear axes can compile.
            value.update(selected_axis_compile=True,
                         new_compile_cells=exploratory_direct_cells(
                             value['prepared']['pending']['requests'], value['direct_prices'],
                             allow_single_axis=True))
        value['new_compile_cells'] = [r for r in value['new_compile_cells'] if r['role'] != 'HOLDOUT']
        value['maximum_new_compile'] = sum(len(r['axes']) for r in value['new_compile_cells'])
        value['price_scope'] = 'EXPLORATORY_CALIBRATION_ONLY'
        return value

    evidence.make_plan = plan


def load_provider(root):
    root=Path(root).resolve()
    sys.path[:0]=[str(root/'scripts'),str(root/'src')]
    spec=importlib.util.spec_from_file_location('fixed_pf_provider',root/'scripts/prepare_pf_candidate.py')
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    # Function globals resolve to the same fixed provider modules and helpers.
    globals().update(ROOT=root,native_row=provider.native_row,comparison_method=provider.comparison_method,
        write=provider.write,load=provider.load,publish=provider.publish,head=provider.head)
    return provider


def bind_native_compile_inputs():
    """Resolve the saved native config before the pinned provider installs."""
    import h46_long_time_signal as signal
    helper = Path(__file__).resolve().parents[1] / 'src/trottertracks/rpe_preparation_paths.py'
    paths = runpy.run_path(str(helper))
    resolve = paths['resolve']
    original = paths['PF_NATIVE_ORIGIN_CONFIG']
    config_path = Path(resolve(original))
    config_raw = config_path.read_bytes()
    if hashlib.sha256(config_raw).hexdigest() != paths['PF_NATIVE_ORIGIN_SHA256']:
        raise ValueError('PF native origin configuration differs from its fixed SHA-256')
    config = json.loads(config_raw)
    config['provider_root'] = resolve(config['provider_root'])
    for name in paths['PF_NATIVE_H4_INPUTS']:
        item = config['systems']['H4']['inputs'][name]
        item['path'] = resolve(item['path'])
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('PF native H4 input differs from its fixed SHA-256: '+name)
    # The original install() and native_context() see the same mapped input.
    # They retain their provider byte checks and scientific guards.
    signal.origin_config = lambda: config


def compile_receipts(value,out,phase,ms,source,*,axes_by_round=None):
    from trottertracks import pf_evidence_preparation as E
    import run_pf_m0_prices as M
    import run_pf_p7_h_prices as P
    cfg=value['configuration'];p=cfg['spec']['L_D'];route=comparison_method(cfg)
    cells=[dict(m=r['m'],formula=r['physical_candidate']['formula'],row=native_row(cfg,r['physical_candidate']),role=phase.upper()) for r in value['prepared']['pending']['requests'] if r['m'] in ms]
    if (value.get('selected_axis_compile') is True) != (axes_by_round is not None):
        raise ValueError('Selected-axis plan requires explicit compiled axes')
    if axes_by_round is not None:
        if (set(axes_by_round) != {str(m) for m in ms}
                or any(tuple(axes) not in (('X',), ('Y',), ('X','Y')) for axes in axes_by_round.values())):
            raise ValueError('Selected compile axes differ from the fixed PF request')
        for cell in cells:
            cell['axes'] = axes_by_round[str(cell['m'])]
    if [cell['m'] for cell in cells] != ms or not cells:
        raise ValueError('Selected compile rounds differ from the fixed PF plan')
    expected = {(cell['m'], axis) for cell in cells for axis in cell.get('axes', ('X','Y'))}
    folder=out/phase/'run01';private=ROOT/'.pf_evidence_private'/cfg['id']/phase/'run01'
    folder.mkdir(parents=True);private.mkdir(parents=True,exist_ok=False)
    saved=M.read(dict(commit=M.BASELINE,path=(M.F if route=='FULL_PF' else M.T).OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    constant=saved['facts']['scalar'] if route=='FULL_PF' else saved['facts']['scalar']-saved['facts']['tail_identity']
    # Use the original saved H constant, not subtraction, for binary64 order.
    full=M.read(dict(commit=M.BASELINE,path=M.F.OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    constant=full['facts']['scalar'];tail=0. if p==10 else E.read(ROOT,E.scalar_source(p))
    from pf_common_run_contract import resources,workers
    obs=resources();n=workers(obs,len(cells),2*2**30,100*2**20);E.check(n>0,'No available worker slot')
    job=dict(route=route,prefix=p,rounds=ms,cells=cells,cohort='D_PF_SELECTED_'+cfg['id'][:12]+'_'+phase,public=str(folder/'child_public'),private=str(private),workers=n,expected_scalar=constant+tail,tail_scalar=tail,
      preparation_hash=saved['trial']['preparation_hash'] if p in (7,10) else None,execution_SOURCE=source['commit'],driver_source=source,fixed_plan=str(out/'PLAN.json'))
    # Non-p7 preparations get checked against exact prefix order/scalar; the
    # native preparation hash is recorded for their own request, never borrowed.
    if p not in (7,10):job['prefix_specific_preparation']=True
    write(folder/'JOB.json',job);started=time.perf_counter()
    child = [sys.executable, '-B', str(Path(__file__).resolve()), '--provider-root', str(ROOT),
             '--native-child', str(folder/'JOB.json')]
    # Match the historical kernel before importing NumPy. For p8 the
    # saved tail identity itself is bit-sensitive even when the total
    # scalar rounds to the same binary64 value.
    env = dict(os.environ)
    if p in (7, 8, 10):
        env.update(OPENBLAS_CORETYPE='HASWELL', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (private/'child.log').open('x') as log:
        r = subprocess.run(child, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    result=load(private/'child.json') if (private/'child.json').exists() else dict(results=[],error='Native child result absent')
    trace=folder/'TRACE';trace.mkdir()
    for name in ['child.log','context_progress.jsonl']+[x.name for x in private.glob('progress_*.jsonl')]:
        if (private/name).exists():shutil.copyfile(private/name,trace/name)
    prices=[];sides=[];proofs={};module=M.F if route=='FULL_PF' else M.T
    for row in result['results']:
        if not row['ok']:continue
        kwargs={} if route=='FULL_PF' else dict(scalar_source=E.scalar_source(p))
        a,b,c=module.price_bindings(ROOT,value['prepared'],row['cell'],row['trial'],price_source=source,evidence_commit=M.S.PLACEHOLDER,evidence_path=str((folder/'IMPLEMENTATION_EVIDENCE.json').relative_to(ROOT)),**kwargs)
        for price,side in zip(a,b):
            price['cohort']=side['cohort']=job['cohort'];price['diagnostic']['compiled_wrapper']=row['wrappers']['cosine' if price['axis']=='X' else 'sine']
            key=side['implementation_evidence']['selector'][0];c[key]={k:deepcopy(side[k]) for k in ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
        prices+=a;sides+=b;proofs.update(c)
    for name,data in [('PRICES.json',prices),('SIDECARS_UNBOUND.json',sides),('IMPLEMENTATION_EVIDENCE.json',proofs)]:write(folder/name,data)
    result.update(source=source,child_exit_code=r.returncode,child_signal=-r.returncode if r.returncode<0 else None,wall_seconds=time.perf_counter()-started)
    write(folder/'RESULT.json',result)
    E.check(r.returncode==0 and len(prices)==len(expected)
            and {(row['m'], row['axis']) for row in prices}==expected
            and all(x['ok'] for x in result['results']),'Native STOP; preserve partial results, no retry')
    pub=publish([out],'Preserve unadmitted PF compile receipts for separate audit')
    receipt = dict(status='COMPILE_RECEIPTS_READY',phase=phase,rounds=ms,raw_commit=pub,relative=str(folder.relative_to(ROOT)),route=route)
    if axes_by_round is not None:
        receipt['axes_by_round'] = axes_by_round
    return receipt


def audit_receipts(provider,value,out,compiled):
    import run_pf_m0_prices as M
    import run_pf_p7_h_prices as P
    from trottertracks import pf_evidence_preparation as E
    folder=ROOT/compiled['relative']
    prices=load(folder/'PRICES.json')
    axes = compiled.get('axes_by_round')
    if axes is not None or value.get('selected_axis_compile'):
        if axes is None:  # A retained pair predates explicit axis metadata.
            axes = {str(m): ['X','Y'] for m in compiled['rounds']}
        expected = {(int(m), axis) for m, selected in axes.items() for axis in selected}
        planned = {(cell['m'], axis) for cell in value['new_compile_cells'] for axis in cell['axes']}
        if not expected <= planned:
            raise ValueError('Raw receipt axes do not belong to the selected PF plan')
        if len(prices) != len(expected) or {(price['m'], price['axis']) for price in prices} != expected:
            raise ValueError('Raw receipts do not match the original selected axes')
    # The original C implementation guard reads committed raw receipts and
    # matching implementation evidence. It never recompiles a circuit.
    P.bind_records(value['prepared'],compiled['relative'],compiled['raw_commit'],names=M.RAW,
        policy_id='PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1',route=compiled['route'])
    bound=publish([out],'Admit transferred PF receipts through original C implementation guard')
    return [dict(m=x['m'],axis=x['axis'],physical_id=x['physical_context']['physical_candidate']['id'],
        unit_price_RZ=E.enc(x['mean']['rz_count']),kind='DIRECT_COMPILED_DETERMINISTIC',
        price_evidence=dict(source=E.ref(bound,str((folder/'PRICES.json').relative_to(ROOT)),i)),
        gate='C_IMPLEMENTATION_GUARD_PASS')for i,x in enumerate(prices)]


def bind_total(view,configuration,source):
    path=Path(view['view'])
    commit=view['view_commit']
    try:
        relative='src/trottertracks/rpe_total_binding.py'
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
        original=subprocess.check_output(['git','show',commit+':'+relative],cwd=path)
        if actual!=commit or (path/relative).read_bytes()!=original:
            raise ValueError('Pinned B binding view changed')
    except OSError as error:
        raise ValueError('Pinned B binding view is unavailable: '+str(path)) from error
    code="""import sys,json
sys.path.insert(0,'src')
from trottertracks import rpe_total_binding as T
d=json.load(sys.stdin)
b=T.create(d['root'],d['configuration'],original_source=d['source'])
print(json.dumps({'id':b['id'],'total_configuration':{'id':b['total_configuration']['id']}}))
"""
    payload=json.dumps(dict(root=str(ROOT),configuration=configuration,source=source)).encode()
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=path,input=payload))


def run(provider,request):
    value=request['value'];out=ROOT/request['relative_output'];phase=request['phase']
    source=request['source']
    if phase=='preflight':
        provider.source_gate()
        bind_preflight_snapshot()
        if request.get('deferred_price_rounds'):
            helpers = saved_price_helpers()
            value = helpers['independent_plan'](value, request['deferred_price_rounds'])
        provider.preflight(value,out)
        publication=publish([out],'Fix distributed PF input plan and duplicate preflight')
        return dict(status='PREFLIGHT_PASS',source=dict(commit=publication,caller='scripts/prepare_pf_candidate.py',
            input_plan=dict(commit=publication,path=str((out/'PLAN.json').relative_to(ROOT)))))
    if phase=='signal':
        bind_saved_signal_paths()
        provider.signals(value,out,source)
        publication=publish([out],'Preserve PF signal task for central adoption')
        return dict(status='SIGNALS_READY',publication=publication)
    if phase=='compile':
        axes = request.get('axes_by_round')
        if axes is not None and not getattr(provider, 'SELECTED_AXIS_COMPILE', False):
            raise ValueError('Fixed PF provider does not support selected axes')
        return compile_receipts(value,out,request['cohort'],request['rounds'],source,axes_by_round=axes)
    if phase=='audit':return dict(status='AUDIT_PASS',prices=audit_receipts(provider,value,out,request['compiled']))
    if phase=='freeze':return dict(status='MODEL_FROZEN',model=provider.freeze(value,request['prices'],out,source))
    if phase=='finish':return dict(status='PRICES_READY',prices=provider.finish(value,request['prices'],out,request['model'],source))
    if phase == 'exploratory-prices':
        from fractions import Fraction
        from trottertracks import price_model_validation as model
        from trottertracks.saved_rpe_cost_catalog import enc, frac

        cfg = value['configuration']
        fixed_model = request.get('model')
        fitted = provider.load(ROOT/fixed_model['path']) if fixed_model else None
        known = {(r['m'], r['axis']): r for r in request['prices']}
        calibration = [cfg['rounds'][m] for m in value['calibration_rounds']]
        structures = [r['structure'] for r in calibration]
        q_max = max((r['q_PF'] for r in calibration), default=0)
        rows = []
        for item in value['prepared']['pending']['requests']:
            m = item['m']
            target = cfg['rounds'][m]
            q = item['q_PF']
            for axis in ('X', 'Y'):
                if (m, axis) in known:
                    rows.append(known[m, axis])
                    continue
                if (fitted is None or not structures or any(target['structure'] != s for s in structures)
                        or frac(target['actual_delta_pf']) != frac(value['actual_width'])
                        or frac(target['T']) != Fraction(q) * frac(value['actual_width'])):
                    continue
                price = model.predict(fitted['models'][axis], dict(one=1, q=q))
                if price > 0:
                    rows.append(dict(m=m, axis=axis, physical_id=item['physical_candidate']['id'],
                                     unit_price_RZ=enc(price), kind='EXPLORATORY_AFFINE_PRICE_REFERENCE',
                                     price_evidence=dict(source=fixed_model, axis=axis),
                                     gate='CALIBRATION_ONLY', q=q, T=target['T'],
                                     actual_delta_pf=target['actual_delta_pf'],
                                     unvalidated_extrapolation=q > q_max, model_error_bound_RZ=None))
        path = out/'EXPLORATORY_AXIS_PRICES.json'
        if not rows:
            return dict(status='EXPLORATORY_PRICE_UNAVAILABLE', prices=None)
        write(path, dict(configuration_id=cfg['id'], axes=rows, model=fixed_model,
                         coverage='PARTIAL_DIRECT_PRICE_EVIDENCE' if len(rows) != 2*len(cfg['rounds']) else 'COMPLETE',
                         scope='EXPLORATORY_CALIBRATION_ONLY', qualified_for_candidate=False, holdout=[]))
        commit = publish([path], 'Preserve exploratory PF prices and calibration references')
        return dict(status='EXPLORATORY_PRICES', prices=dict(commit=commit, path=str(path.relative_to(ROOT))))
    if phase=='register':
        # The central owned clone imports remote objects without switching its
        # HEAD. Bind its newly received signal files before gitref() is used.
        publish([out],'Bind received PF evidence before central registry publication')
        if request.get('binding_view'):
            fixed=load(ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json')['execution_routes']['FULL_PF'][0]
            view=request['binding_view']
            if view['view_commit']!=fixed['view_commit']:
                raise ValueError('Requested PF binding does not match the fixed provider SOURCE')
            provider.total_binding=lambda cfg,src:bind_total(view,cfg,src)
        admitted=provider.register(value,out,request['prices'],ROOT/request['registry'])
        publish([out],'Publish distributed PF completed evidence')
        return dict(status='READY_FOR_SAME_OPTIMIZER',configuration_id=value['configuration']['id'],
            entry=admitted,output=str(out))
    raise ValueError('Unknown PF preparation phase')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider-root',type=Path,required=True);p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--plan-signal-index',type=Path)
    p.add_argument('--saved-price-index',type=Path)
    p.add_argument('--native-child',type=Path);a=p.parse_args()
    provider=load_provider(a.provider_root)
    if a.native_child is not None:
        if any((a.input, a.output, a.plan_signal_index, a.saved_price_index)):
            p.error('--native-child cannot be combined with planning options')
        bind_native_compile_inputs()
        return provider.native_child(a.native_child)
    if a.input is None or a.output is None:
        p.error('--input and --output are required')
    request=load(a.input)
    if a.plan_signal_index is not None:
        if request.get('exploratory_cost') is True:
            use_exploratory_plan(recover_partial=request.get('partial_price_recovery') is True)
        if a.saved_price_index is not None:
            saved_price_helpers()['use_saved_prices'](a.saved_price_index)
        cache_mean_pool(a.provider_root,a.plan_signal_index)
        sys.argv=[str(a.provider_root/'scripts/prepare_pf_candidate.py'),'--request',str(a.input),
                  '--output',str(a.output),'--mode','plan']
        return provider.main()
    result=run(provider,request);write(a.output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
