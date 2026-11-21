"""Selected deterministic-PF evidence plans. Metadata is separate from science.

Uses the existing C signal provider, saved direct receipts and D's native child.
It neither chooses shot allocations nor evaluates Q3.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess

from . import rpe_optimization_pipeline as I
from . import saved_pf_mean_staging as S
from . import baseline_signal_numerics as N
from . import pf_shortlist_mean_completion as F
from . import pf_candidate_space as PS
C=I.C
MODULE='src/trottertracks/pf_evidence_preparation.py'
INDEX=dict(commit='7f27e18f06dde12127c85be898777b685d29bacc',path='docs/research/pf_joint_width_shot_plan_20261115/WIDTH_SIGNAL_COVERAGE.json')
COSTS=dict(commit='1002f1dfea34de439ca1b704a0192bca473c5387',path='docs/research/pf6_allround_cost_20261115/INPUTS.json')
CATALOG=dict(commit='1d44d07397031547e402171faae268a640a2807b',path='docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json')
T1='5e6f9fd214631f7ce92fad0e9c0d273dfb479533'
D='015d04a2a3db1c7d7140f36edd86cc700191808b'


def check(ok,why):
    if not ok:raise ValueError(why)
def ref(commit,path,*selector):return dict(commit=commit,path=path,selector=list(selector))
def read(root,loc):
    raw=C.blob(root,loc['commit'],loc['path']);v=json.loads(gzip.decompress(raw) if loc['path'].endswith('.gz') else raw)
    for k in loc.get('selector',[]):v=v[k]
    return v
def enc(v):
    v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)
def frac(v):return Fraction(v['numerator'],v['denominator']) if isinstance(v,dict) else Fraction(v)
def scalar_source(p):return ref(T1,'docs/research/truncated_pf_t1_results_20261114/run01/prefix_%d.json'%p,'tail_coefficients','used_c_tail')
def physical_key(c):return tuple(c.get(k) for k in ('system','method','prefix','formula','T','q','delta','R_outer','K'))


def plan(root,cfg,source):
    manifest=read(root,S.MANIFEST)
    return I.plan(root,cfg,target=manifest['shared']['target'],signal_source=source,references=S.references(root,cfg,manifest['shared']['binding']))


def published_ready(root,cfg):
    """Return an already complete fixed-C input row, with no refit or science.

    A previous Q3 failure does not invalidate complete inputs. Conditional
    prices retain their saved scientific status; this does not upgrade them.
    """
    catalog=read(root,CATALOG)
    for index,row in enumerate(catalog['candidates']):
        if row['configuration_id']!=cfg['id'] or len(row.get('price_axes',[]))!=26:continue
        if not row.get('signal_source') or not row.get('configuration_source') or not row.get('case_source'):continue
        original=read(root,row['configuration_source']);packet=read(root,row['signal_source'])
        check(original['id']==cfg['id'] and C.p.same(original['spec'],cfg['spec']),'Saved catalog configuration differs')
        check(packet['configuration_id']==cfg['id'] and len(packet['means'])==13,'Saved catalog signal packet incomplete')
        prepared=plan(root,cfg,packet['source'])
        means={m['m']:m for m in packet['means']}
        check(set(means)==set(range(13)),'Saved catalog mean round identity differs')
        for m,mean in means.items():
            check(mean.get('raw') is not None and mean['normalization']['numerical'].get('gates_passed') is True,'Saved catalog numerical mean unresolved')
            check(mean['identity']==cfg['rounds'][m]['identity'] and mean['law_binding']==cfg['rounds'][m]['law_binding'],'Saved catalog signal identity differs')
            I.provider_binding_check(prepared,mean)
        axes=row['price_axes']
        check(len({(a['m'],a['axis']) for a in axes})==26 and {(a['m'],a['axis']) for a in axes}=={(m,a) for m in range(13) for a in ('X','Y')},'Saved catalog price axes incomplete')
        for axis in axes:
            m=axis['m'];expected=prepared['pending']['requests'][m]
            check(axis['physical_id']==expected['physical_candidate']['id'],'Saved catalog price physical identity differs')
            check(axis.get('canonical_id',cfg['rounds'][m]['candidate']['id'])==cfg['rounds'][m]['candidate']['id'],'Saved catalog canonical price identity differs')
            check(all(axis['compiler_profile'].get(k)==v for k,v in cfg['spec']['profile'].items()),'Saved catalog price profile differs')
        return dict(row=deepcopy(row),index=index,catalog_source=dict(CATALOG,selector=['candidates',index]))
    return None


@lru_cache(maxsize=8)
def mean_pool(root):
    """Current gated packets/rounds only; legacy-only means remain candidates."""
    index=read(root,INDEX);origins=[]
    for cand in index['candidates']:
        if cand.get('source_bound_13round_packet'):
            origins.append(cand['source_bound_13round_packet'])
        for hits in cand.get('round_source_locators',{}).values():origins+=hits.get('current',[])
    origins += [ref('f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53','docs/research/d_p7_y4_coarse_signal_q3_20261116/BOUND_SIGNAL_INPUT.json')]
    # Prefix8/9 are absent from the older Full/p7 width index. Their T2
    # first/m12 means are already saved and use the same unchanged gates.
    for cell in PS._t2_cells(str(root)):
        if int(cell['prefix']) not in (8,9):continue
        directory=(PS.T2_DIR+'run01/cell_p'+cell['prefix']+'_'
            +cell['formula'].replace('(','_').replace(')','')+'_'
            +str(frac(cell['actual_delta_over_h'])).replace('/','o'))
        origins += [ref(PS.T2_SOURCE,directory+'/'+name)
                    for name in ('round_first_check.json','round_m12_check.json')]
    by_key={};seen=set()
    def add(value,origin):
        c=value.get('physical_candidate') or (value.get('provider_binding') or {}).get('physical_context',{}).get('physical_candidate')
        if c is None:return
        if value.get('raw') is not None:
            numerical=(value.get('normalization') or {}).get('numerical') or {}
            if numerical.get('gates_passed') is not True:return
        elif not (value.get('gate_pass') is True and value.get('numerical',{}).get('gate_pass') is True):return
        by_key.setdefault(physical_key(c),[]).append(dict(value=value,source=origin))
    for origin in origins:
        tag=json.dumps(origin,sort_keys=True)
        if tag in seen:continue
        seen.add(tag);value=read(root,origin)
        if 'means' in value:
            for j,mean in enumerate(value['means']):add(mean,dict(origin,selector=origin.get('selector',[])+['means',j]))
        else:add(value,origin)
    return by_key


@lru_cache(maxsize=8)
def price_pool(root):
    table=read(root,COSTS);out={}
    for row in table['axes']:
        loc=row['source'].get('receipt')
        if not loc:continue  # conditional closure is not a direct observation
        value=read(root,loc);c=value['physical_context']['physical_candidate']
        out.setdefault((physical_key(c),value['axis']),dict(value=value,source=loc,admission=ref(COSTS['commit'],COSTS['path'],'axes',table['axes'].index(row))))
    # Later width-specific direct prices retain their own original producers.
    for name,commit,field in [('d_pf_p7_s2_h_direct_prices_20261115','e04a14797a56a586dcb72065a55f4d97bfa1479b','axes'),
                              ('d_p7_y4_coarse_price_calibration_20261116','c7c8c1edd0996ad9ef97629961f67ca508a182f0','axes')]:
        file='run01/AXIS_PRICES.json' if name.startswith('d_pf_p7_s2') else 'PRICE_AND_COST.json'
        data=read(root,ref(commit,'docs/research/'+name+'/'+file))
        is_h=name.startswith('d_pf_p7_s2')
        rows=data.get('rows' if is_h else field,[])
        for i,row in enumerate(rows):
            if is_h:
                if row.get('status')!='ACCEPTED_DIRECT':continue
                check(row['gate_result']['C_guard']=='PASS' and row['gate_result']['original_action_gates']=='PASS','Saved h-price admission unresolved')
                origin=row['receipt']
            else:
                if row['kind']!='DIRECT_COMPILED_DETERMINISTIC':continue
                origin=row['price_evidence']['source']
            v=read(root,origin);c=v['physical_context']['physical_candidate']
            if is_h:
                check(c==row['physical_candidate'] and v['axis']==row['axis'],'Saved h-price row/receipt binding differs')
                check(v['mean']['rz_count']==row['compiled_RZ'],'Saved h-price row/receipt count differs')
            out.setdefault((physical_key(c),v['axis']),dict(value=v,source=origin,admission=ref(commit,'docs/research/'+name+'/'+file,'rows' if is_h else field,i)))
    return out


def accepted_direct(root,prepared):
    output=[]
    for request in I.price_requests(root,prepared):
        hit=price_pool(str(root)).get((physical_key(request['physical_context']['physical_candidate']),request['axis']))
        if hit is None:continue
        old=hit['value']
        check(old['physical_context']==request['physical_context'],'Saved price physical/input/signed domain differs')
        for k in ('axis','measurement_axis','observable','outcome_convention','quantity','units','P'):
            check(old[k]==request[k],'Saved direct quantity/axis differs: '+k)
        check(old['price_profile']['qiskit_version']=='1.3.0' and old['price_profile']['basis_gates']==['rz','sx','x','cx'] and old['price_profile']['optimization_level']==1 and old['price_profile']['seed_transpiler']==17,'Saved compiler differs')
        check(old['price_profile'].get('lowering_id')=='H46_LONG_TIME_NATIVE_COMPOSE','Saved deterministic native lowering differs')
        for k,v in request['physical_context']['profile'].items():
            check(old['price_profile'].get(k)==v,'Saved wrapper/control/price profile differs: '+k)
        check(old['classification']=='DIRECT_COMPILED_DETERMINISTIC' and old['expectation']=='DETERMINISTIC','Not a deterministic direct receipt')
        output.append(dict(m=request['m'],axis=request['axis'],physical_id=request['physical_context']['physical_candidate']['id'],unit_price_RZ=enc(old['mean']['rz_count']),kind='DIRECT_COMPILED_DETERMINISTIC',price_evidence=dict(source=hit['source'],admission=hit['admission'],original_SOURCE=old['source'],cohort=old['cohort']),gate='EXISTING_C_ADMITTED_PHYSICAL_MATCH',metrics=old['metrics'],profile=old['price_profile']))
    return output


def make_plan(root,cfg,source,*,allow_single_axis=False):
    prepared=plan(root,cfg,source);direct=accepted_direct(root,prepared);requests=prepared['pending']['requests']
    means=[]
    for req in requests:
        hits=mean_pool(str(root)).get(physical_key(req['physical_candidate']),[])
        means.append(dict(m=req['m'],physical_id=req['physical_candidate']['id'],acquisition='REUSE_SAVED_MEAN' if hits else 'NEW_MEAN_REQUIRED',saved_source=hits[0]['source'] if hits else None))
    width=frac(cfg['spec']['delta_pf']);saturated=[r for r in requests if frac(r['T'])/r['q_PF']==width]
    eligible=[r for r in saturated if r['q_PF']>=2]
    check(len(eligible)>=3,'At least two distinct-q calibration cells and one later holdout required')
    calibration=eligible[:2]
    existing={(r['m'],r['axis']) for r in direct}
    later=[r for r in eligible[2:] if r['q_PF']>=4*calibration[-1]['q_PF']]
    # Choose from receipt availability, never observed price values. A saved
    # later-width check avoids rebuilding an earlier unsaved wrapper solely
    # because its round index is smaller. It remains outside calibration.
    saved_later=[r for r in later if all((r['m'],a) in existing for a in ('X','Y'))]
    hold=(saved_later or later or eligible[-1:])[0]
    calibration_ms=[r['m'] for r in calibration];hm=hold['m']
    # Every early actual-width exception needs direct prices; same-width later
    # cells may use only the explicitly conditional candidate-specific model.
    needed=sorted({r['m'] for r in requests if frac(r['T'])/r['q_PF']!=width or r['q_PF']==1}|set(calibration_ms+[hm]))
    new=[]
    for m in needed:
        axes=[a for a in ('X','Y') if (m,a) not in existing]
        if axes:
            check(allow_single_axis or axes==['X','Y'],'One saved axis only: resolve per-axis native execution before compiling')
            new.append(dict(m=m,axes=axes,role='CALIBRATION' if m in calibration_ms else 'HOLDOUT' if m==hm else 'EARLY_DIRECT'))
    result = dict(schema='D_PF_SELECTED_EVIDENCE_PLAN_V1',configuration=cfg,prepared=prepared,signal_rounds=means,direct_prices=direct,
        calibration_rounds=calibration_ms,holdout_round=hm,actual_width=enc(width),new_compile_cells=new,
        maximum_new_compile=sum(len(cell['axes']) for cell in new),new_signal_means=sum(r['saved_source'] is None for r in means),
        model_formula='C_axis(q)=alpha_axis+beta_axis*q',model_features=['one','q'],tolerance=None,
        extrapolation_error_bound=None,refit_holdout=False,price_scope='CONDITIONAL_COMPARISON_MODEL_WITH_EXPLICIT_LONG_Q_UNKNOWN',new_Q3=0)
    if allow_single_axis:
        result['selected_axis_compile'] = True
    return result


def saved_record(root,prepared,req,origin,source):
    old=read(root,origin);c=req['physical_candidate']
    oldc=old.get('physical_candidate') or old.get('provider_binding',{}).get('physical_context',{}).get('physical_candidate')
    check(oldc==c,'Saved mean physical schedule differs')
    if old.get('raw') is not None:
        context=(old.get('provider_binding') or {}).get('physical_context')
        if context:check(context==I.price_context(prepared,req['m']),'Saved mean input/profile/split differs')
        num=old['normalization']['numerical'];check(num['gates_passed'] is True and old['normalization']['B']==1,'Missing old mean numerical gates')
        check(num.get('scalar_once') is True and num.get('repaired') is False,'Saved scalar/power application unresolved')
        check(old['normalization']['log_B']==0 and C.p.same(old['raw'],old['corrected']),'Saved deterministic B=1 signal differs')
        check(C.p.same(old['target'],req['domain']['target']),'Saved mean target differs')
        pair=read(root,req['domain']['reference']['source'])
        check(0<=pair['signal_difference']<=1e-10 and C.p.same(old['reference'],pair['reference']),'Saved mean reference binding differs')
        signal=dict(raw=old['raw'],corrected=old['corrected'],reference=old['reference'],log_B=old['normalization']['log_B'],B=1.,attenuation=1.,total_signed_bias=old['bias']['total_signed_bias'])
    else:
        num=old['numerical'];check(old['gate_pass'] and num['gate_pass'] and num['norm_gate_pass'] and num['unitarity_gate_pass'] and not num['repaired'],'Saved scalar/power gates unresolved')
        check(old.get('scalar_once_at_total_T_in_direct_route',old.get('scalars_once_at_total_T_in_direct_route')) is True,'Saved scalar application differs')
        pair=read(root,req['domain']['reference']['source']);check(0<=pair['signal_difference']<=1e-10,'Saved reference pair gate')
        signal=dict(raw=deepcopy(old['direct_PF_mean']),corrected=deepcopy(old['direct_PF_mean']),reference=deepcopy(pair['reference']),log_B=0.,B=1.,attenuation=1.,total_signed_bias=None)
        num=dict(num,gates_passed=True,scalar_once=True,backend='SAVED_CURRENT_DIRECT_MEAN_NO_REEVALUATION')
    return dict(domain=deepcopy(req['domain']),source=source,status='NUMERICAL_MEAN_AVAILABLE',signal=deepcopy(signal),numerical=deepcopy(num),input_source=dict(original_record=origin,original_numerical_SOURCE=old.get('source'),binding_only_no_new_evaluation=True))


def bind_mean(prepared,req,record):
    mean=C.b_mean(req,record)
    mean['provider_binding']=dict(physical_context=I.price_context(prepared,req['m']),numerical_source=deepcopy(record['source']),input_source=deepcopy(record.get('input_source',record.get('input_provenance'))),reference_source=req['domain']['reference'])
    I.provider_binding_check(prepared,mean);return mean


def step_key(c,weights):
    """Same T*w/q binary64 operation order as the existing numerical Context."""
    return c['prefix'],c['formula'],tuple((c['T']*w/c['q']).hex() for w in weights)


def saved_step_metadata(root,commit,directory,field,scope_filename='CELL_SCOPE.json'):
    """Normalize fixed producer metadata; never load an operator here.

    The original S2/h pilot stores SCOPE.json and its recipe under
    operator.identity.step_key. Scan/T2 cells use CELL_SCOPE.json. The NPZ field
    explicitly selects the step without global scalars in both formats.
    """
    scope_source=ref(commit,directory+'/'+scope_filename)
    summary_source=ref(commit,directory+'/result.json')
    scope=read(root,scope_source);summary=read(root,summary_source)
    if scope_filename=='SCOPE.json':
        check(field=='PF_step_without_global_scalar','Saved pilot global scalar field differs')
        key=scope['operator']['identity']['step_key']
        check(key['kind']=='FULL_PF' and key['prefix']==10,'Saved pilot is not Full PF')
        formula=key['formula'];width=frac(key['actual_delta']);weights=key['signed_weights']
        # Pilot first check is T=width,q=1. Its exported NPZ field is the
        # scalar-free Context.pf_step, as saved by the original pilot caller.
        c=dict(method='FULL',prefix=10,formula=formula,T=float(width),q=1,delta=float(width))
        binding=deepcopy(key['binding']);constant=tail=None
        producer='FULL_PF_INITIAL_PILOT_SCOPE'
    else:
        c=scope['first_check']['physical_candidate'];formula=scope['formula']
        width=frac(scope['actual_delta_pf']);weights=scope['signed_weights'];binding=None
        check(c['formula']==formula and c['delta']==float(width),'Saved step width/formula differs')
        check(summary['formula']==formula and frac(summary['actual_delta_pf'])==width,'Saved step summary differs')
        constant=summary.get('scalar_constant_ha');tail=summary.get('c_tail_ha')
        producer='SAVED_SCAN_OR_T2_CELL'
    check(c['method'] in ('FULL','B0I') and c['prefix'] in range(11),'Saved step prefix/method differs')
    check((c['method']=='FULL')==(c['prefix']==10),'Saved step Full/B0I prefix differs')
    return dict(operator_source=dict(commit=commit,path=directory+'/OPERATORS.npz',field=field),
        scope_source=scope_source,summary_source=summary_source,producer_format=producer,
        physical_recipe=deepcopy(c),prefix=c['prefix'],formula=formula,actual_delta_pf=enc(width),
        signed_weights=deepcopy(weights),binding=binding,scalar_constant=constant,tail_scalar=tail)


def saved_step_descriptors(root,prepared):
    """Resolve saved operators for the requested prefix/formula/actual widths."""
    cfg=prepared['configuration'];p=cfg['spec']['L_D'];formula=cfg['spec']['formula']
    needed={frac(r['T'])/r['q_PF'] for r in prepared['pending']['requests']}
    locations={}
    if p==10:
        scan=read(root,ref(F.FULL,'docs/research/h4_full_pf_heff_scan_preparation_20261114/PLAN.json'))
        for i,cell in enumerate(scan['step_cells']):
            width=frac(cell['actual_delta_pf'])
            if cell['formula']!=formula or width not in needed:continue
            original=cell.get('saved_source')
            if original:
                commit=original['commit'];directory=original['path'].rsplit('/',1)[0]
                scope='SCOPE.json'
            else:
                commit=F.FULL;directory='docs/research/h4_full_pf_heff_scan_results_20261114/run01/cell_%02d'%i
                scope='CELL_SCOPE.json'
            locations[width]=(commit,directory,'PF_step_without_global_scalar',scope)
        follow=read(root,ref(F.FULL,F.FULLDIR+'C_HANDOFF.json'))
        for candidate in follow['configurations']:
            if candidate['formula']!=formula:continue
            for row in candidate['rounds']:
                width=frac(row['actual_delta_pf'])
                if width not in needed or width in locations:continue
                original=row['effective_spectrum_source'];commit=original.get('commit',F.FULL)
                directory=original['path'].rsplit('/',1)[0]
                if 'commit' not in original:directory=F.FULLDIR+directory
                locations[width]=(commit,directory,'PF_step_without_global_scalar','CELL_SCOPE.json')
    elif p in (7,8,9):
        for cell in PS.priority_evidence(root,p,formula,cfg)['T2']:
            operator=cell['operator_source'];width=frac(cell['actual_delta_over_h'])*S.H
            if width in needed:
                locations[width]=(operator['commit'],operator['path'].rsplit('/',1)[0],operator['field'],'CELL_SCOPE.json')
    return [saved_step_metadata(root,*locations[w]) for w in sorted(locations)]


class SavedOrNativeContext(N.Context):
    """Same numerical kernel; cache correctly keyed saved operators or build missing steps."""
    def __init__(self,root,prepared):
        super().__init__(root,*N.load_shared_h4(root,prepared['configuration']['spec']['binding']))
        p=prepared['configuration']['spec']['L_D']
        self.prefix=p;self.formula=prepared['configuration']['spec']['formula']
        if p<10:self.tail_scalars[p]=float(read(root,scalar_source(p)))
        self.saved_steps={};self.new_step_keys=[];self.saved_step_sources=[]
        for descriptor in saved_step_descriptors(root,prepared):self._load_saved(descriptor)
        self.metadata.update(step_reuse='SAVED_OPERATORS_WHERE_EXACT_SCOPE_MATCHES_ELSE_ORIGINAL_CONTEXT',prefix_scalar_source=None if p==10 else scalar_source(p))

    def _load_saved(self,descriptor):
        import numpy as np
        d=descriptor;c=d['physical_recipe'];op=d['operator_source']
        if d['binding'] is not None:
            check(all(self.binding.get(k)==v for k,v in d['binding'].items()),'Saved pilot H/state/snapshot binding differs')
        check(c['prefix']==self.prefix and c['formula']==self.formula,'Saved step selected prefix/formula differs')
        raw=C.blob(self.root,op['commit'],op['path'])
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:matrix=np.array(arrays[op['field']],copy=True)
        check(matrix.shape==(36,36) and np.isfinite(matrix).all(),'Saved step layout differs')
        if d['scalar_constant'] is not None:check(d['scalar_constant']==self.constant,'Saved H constant differs')
        if c['method']=='B0I':check(d['tail_scalar']==self.tail_scalar(c['prefix']),'Saved prefix scalar differs')
        key=step_key(c,d['signed_weights'])
        matrix.setflags(write=False);self.saved_steps[key]=matrix
        self.saved_step_sources.append(dict(d,sha256=hashlib.sha256(raw).hexdigest()))

    def pf_step(self,c,weights):
        key=step_key(c,weights)
        if key in self.saved_steps:return self.saved_steps[key]
        native_key=key
        if native_key not in self.steps:self.new_step_keys.append(native_key)
        return super().pf_step(c,weights)
