"""Approved 48-round Full PF/B0I completion using saved steps and C.execute.

No new PF step, spectrum, reference, risk/N or price generation. Metadata
preparation reuses A's saved staging; numerics delegate to the existing kernel.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import saved_pf_mean_staging as S
from . import rpe_optimization_pipeline as I
from . import baseline_signal_numerics as N
C=I.C
STAGING='217eab647b540a34ca98c285fbc00063cf1d4923'
STAGEDIR='docs/research/a_saved_pf_mean_staging_20261115/'
FULL='714247619afb4714fd663eb15c0470ef32bdc7af'
FULLDIR='docs/research/h4_full_pf_followup_results_20261114/'
TRUNC='5fb2c32e39e63267ea1748861269c362a93a1b86'
TRUNCDIR='docs/research/truncated_pf_t2_results_20261114/run01/'
B_SOURCE='9c0e12d87bf83429e3411d295b16c47d913283d2'
MODULE='src/trottertracks/pf_shortlist_mean_completion.py'
OUT='docs/research/pf_shortlist_mean_completion_20261115/'
EXPECTED=(tuple([2,3,*range(5,12)]),tuple([3,*range(5,12)]),tuple(range(5,12)))*2


def read(root,commit,path):
    return S.git_json(root,dict(commit=commit,path=path))


def exact(value):
    return Fraction(value['numerator'],value['denominator'])


def step_descriptor(root,candidate,request,full_handoff):
    """Resolve the already saved operator, exact actual width and signed order."""
    m=request['m'];c=request['physical_candidate']
    if candidate['method']=='FULL':
        cfg=next(v for v in full_handoff['configurations'] if v['label']==candidate['label'].removeprefix('FULL_PF_'))
        row=cfg['rounds'][m]
        if row['physical_candidate']!=c:raise ValueError('FULL_SAVED_STEP_PHYSICAL_RECIPE')
        src=row['effective_spectrum_source']
        if 'commit' in src:
            commit=src['commit'];directory=src['path'].rsplit('/',1)[0]
        else:
            commit=FULL;directory=FULLDIR+src['path'].rsplit('/',1)[0]
        key='PF_step_without_global_scalar'
    else:
        delta=exact(request['T'])/request['q_PF']
        # Preserve the exact h-relative rational (including /3 or /7), rather
        # than reinterpreting the display binary64 delta as that exact value.
        width=str(delta/S.H).replace('/','o')
        directory=TRUNCDIR+'cell_p7_'+c['formula'].replace('(','_').replace(')','')+'_'+width
        commit=TRUNC;key='PF_step_without_global_scalars'
    cell=read(root,commit,directory+'/CELL_SCOPE.json')
    summary=read(root,commit,directory+'/result.json')
    first=cell['first_check']['physical_candidate']
    if (first['method']!=c['method'] or first['prefix']!=c['prefix'] or first['formula']!=c['formula']
            or exact(cell['actual_delta_pf'])!=exact(request['T'])/request['q_PF']
            or cell['signed_weights']!=request['structure']['signed_weights']
            or summary['formula']!=c['formula'] or summary['actual_delta_pf']!=cell['actual_delta_pf']):
        raise ValueError('SAVED_STEP_FORMULA_WIDTH_ORDER_OR_PREFIX')
    raw=C.blob(root,commit,directory+'/OPERATORS.npz')
    return dict(commit=commit,path=directory+'/OPERATORS.npz',field=key,
        sha256=hashlib.sha256(raw).hexdigest(),scope_source=dict(commit=commit,path=directory+'/CELL_SCOPE.json'),
        cell_result_source=dict(commit=commit,path=directory+'/result.json'),
        physical_recipe=deepcopy(first),actual_delta_pf=cell['actual_delta_pf'],
        signed_weights=cell['signed_weights'],scalar_constant=summary.get('scalar_constant_ha'),
        tail_scalar=summary.get('c_tail_ha'))


def prepare(root,source):
    """No array/power/mean calls. Reuse existing 30 saved records and coverage."""
    api=I.providers(str(root));manifest=S.git_json(root,S.MANIFEST)
    stage=read(root,STAGING,STAGEDIR+'STAGING.json')
    full=read(root,FULL,FULLDIR+'C_HANDOFF.json')
    references=read(root,S.PHYSICS_REF['commit'],S.PHYSICS_REF['path'])
    schedules=[];steps={};new=[];saved_sources=[]
    for candidate,expected in zip(S.CANDIDATES,EXPECTED):
        label=candidate['label'];request=S.plan_request(root,candidate,I=I,staging_source=source)
        config=request['configuration'];coverage=S.coverage_rounds(root,candidate)
        staged=read(root,STAGING,STAGEDIR+'STAGED_SIGNAL_INPUT_'+label+'.json')
        state=next(v for v in stage['candidates'] if v['candidate']==label)
        if staged['configuration_id']!=config['id'] or state['configuration_id']!=config['id']:
            raise ValueError('STAGING_CONFIG_CHANGED')
        saved=[];missing=[]
        for req,old,state_row in zip(request['pending']['requests'],staged['means'],state['rounds']):
            m=req['m'];c=req['physical_candidate']
            if (old['identity']!=req['identity'] or old['law_binding']!=req['law_binding']
                    or state_row['physical_id']!=c['id'] or state_row['actual_delta_pf']!=config['rounds'][m]['actual_delta_pf']):
                raise ValueError('CANONICAL_OR_PHYSICAL_ID_CHANGED')
            if old['raw'] is not None:
                origin=old['provider_binding']['input_source']['saved_mean']
                hit=dict(source=origin['commit']+':'+origin['path'],note=old['provider_binding']['input_source']['origin_note'])
                pair=references['rounds'][m]['diagnostics']['reference_pair']
                record=S.stage_record(root,candidate,None,req,hit,pair,source)
                rebound=C.b_mean(req,record)
                for key in ('raw','corrected','reference','normalization'):
                    if rebound[key]!=old[key]:raise ValueError('SAVED_S2_VALUE_OR_GATE_MUTATED: '+label)
                saved.append(record);saved_sources.append(dict(label=label,m=m,physical_id=c['id'],source=origin))
            else:
                missing.append(m)
                desc=step_descriptor(root,candidate,req,full)
                sk=(desc['commit'],desc['path'])
                steps[sk]=desc
                new.append(dict(label=label,m=m,physical_id=c['id'],configuration_id=config['id'],
                    canonical_id=req['identity']['candidate']['id'],T=c['T'],q=c['q'],actual_delta_pf=config['rounds'][m]['actual_delta_pf'],
                    structure=req['structure'],original_status=state_row['state'],step_source=desc))
        if tuple(missing)!=expected:raise ValueError('FIXED_48_ROUND_MENU_CHANGED: '+label)
        schedules.append(dict(label=label,candidate=deepcopy(candidate),request=request,saved=saved))
    if len(new)!=48 or len(saved_sources)!=30 or len(steps)!=10:
        raise ValueError('EXPECTED_48_NEW_30_SAVED_10_STEPS')
    return dict(schedules=schedules,new=new,saved_sources=saved_sources,steps=list(steps.values()),
        shared=manifest['shared'],source=source)


def load_inputs(root,binding):
    """Unchanged save-only loader; no matrices, PF steps or eigensystems built."""
    from .r_price_phase1_science import functions
    idx=read(root,N.INDEX['commit'],N.INDEX['path']);inputs=idx['saved_inputs']['H4']['inputs']
    desc=idx['T20_reference_files']['H4']
    for spec in [*inputs.values(),desc]:
        if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('SAVED_INPUT_HASH')
    ctx=json.loads(Path(desc['path']).read_bytes())
    if any(binding.get(k)!=v for k,v in ctx['binding'].items()):raise ValueError('SAVED_INPUT_BINDING')
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    scope=dict(Path=Path,c=SimpleNamespace(sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        read=lambda path:json.loads(Path(path).read_bytes()),require=require))
    functions(root,N.SNAPSHOT,'scripts/h46_m5_tol_saved_input.py',('array_record','load'),scope)
    ham,sector,full,state,observed=scope['load'](dict(mapping=dict(tol_snapshot=inputs['H4_tol_snapshot.npz']['path'],tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    if len(state)!=36 or sector.dimension!=36 or ham.n_blocks!=10:raise ValueError('H4_DF10_SECTOR_LAYOUT')
    return dict(state=state,constant=ham.constant,tail_scalar=ctx['prefixes']['7']['scalar'],
        source=dict(index=N.INDEX,inputs=inputs,context=desc,loader=N.SNAPSHOT),observed=observed)


def load_steps(root,descriptors):
    import numpy as np
    result={}
    for desc in descriptors:
        raw=C.blob(root,desc['commit'],desc['path'])
        if hashlib.sha256(raw).hexdigest()!=desc['sha256']:raise ValueError('SAVED_OPERATOR_BYTES')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:matrix=np.array(z[desc['field']],copy=True)
        if matrix.shape!=(36,36) or not np.isfinite(matrix).all():raise ValueError('SAVED_OPERATOR_LAYOUT_OR_NONFINITE')
        matrix.setflags(write=False);result[desc['commit'],desc['path']]=matrix
    return result


class SavedStepContext:
    """Existing numerical evaluate() interface backed ONLY by saved operators."""
    def __init__(self,root,request,inputs,entries,steps):
        self.root=str(root);self.binding=deepcopy(request['configuration']['spec']['binding'])
        self.state=inputs['state'];self.constant=inputs['constant'];self.tail=inputs['tail_scalar']
        self.metadata=dict(source=deepcopy(inputs['source']),step_reuse='SAVED_DIRECT_EXPM_OPERATOR_NO_NEW_BUILD')
        self.entries={r['physical_id']:r for r in entries};self.steps=steps

    def pf_step(self,c,weights):
        if c['id'] not in self.entries:raise ValueError('UNAPPROVED_OR_DUPLICATE_PHYSICAL_ROUND')
        entry=self.entries[c['id']];d=entry['step_source']
        if (c['T']!=entry['T'] or c['q']!=entry['q'] or weights!=d['signed_weights']
                or c['delta']!=float(exact(d['actual_delta_pf']))):
            raise ValueError('SAVED_STEP_REQUEST_DIFFERENT')
        if d['scalar_constant'] is not None and d['scalar_constant']!=self.constant:
            raise ValueError('FULL_SCALAR_DIFFERENT')
        if d['tail_scalar'] is not None and d['tail_scalar']!=self.tail:
            raise ValueError('B0I_SCALAR_DIFFERENT')
        return self.steps[d['commit'],d['path']]

    def tail_scalar(self,prefix):
        if prefix!=7:raise ValueError('ONLY_B0I_PREFIX7')
        return self.tail
