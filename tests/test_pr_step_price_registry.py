"""Scoped price intake: artificial products and already-published JSON receipts.

No path, Hamiltonian, compiler, fitting or Q3 worker is invoked.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.pr_step_price_registry import StepPriceRegistry, exact, frac, digest
from trottertracks.pr_price_rules import GitReader


class Documents:
    def __init__(self):self.root=ROOT;self.docs={}
    def add(self,path,value,commit='ARTIFICIAL'):
        self.docs[commit,path]=deepcopy(value);return dict(commit=commit,path=path)
    def read(self,loc):
        value=deepcopy(self.docs[loc['commit'],loc['path']])
        for key in loc.get('selector',[]):value=value[key]
        return value
    def blob(self,*args):return b'ARTIFICIAL_FIXED_SOURCE'


def config(delta=.2,pauli=True):
    h=Fraction(.1);delta=Fraction(delta)
    law=dict(model='ideal_finite_RTE_law',source=dict(commit='ARTIFICIAL',path='rte.py'))
    split=dict(L_D=3 if pauli else 8,lambda_R=.25,scalar=.01)
    if pauli:split['tail_dictionary']=dict(prefix=3,source=dict(commit='ARTIFICIAL',path='dict.json'),dictionary_sha256='TOY')
    spec=dict(method='PR',L_D=split['L_D'],split=split,formula='2nd',finite_K=2,h=exact(h),M=12,
        delta_pf=exact(delta),binding=dict(hamiltonian='H',state='S',snapshot='SNAP',rank=10),
        profile=dict(compiler='COMPILER'),law=law)
    out=dict(id='ARTIFICIAL_CONFIG_'+str(delta),spec=spec,rounds=[])
    for m in range(13):
        T=h*2**m;q=math.ceil(T/delta);R=1;step=T/q
        physical=dict(id='P'+str(m),q=q,R_outer=R,prefix=spec['L_D'],T=exact(T),delta=exact(step))
        candidate=dict(physical,id='RMSE:'+str(m),event_count=q*R,formula='2nd',K=2)
        out['rounds'].append(dict(m=m,T=exact(T),q_PF=q,R_outer=R,actual_delta_pf=exact(step),
            candidate=candidate,physical_candidate=physical,
            structure=dict(signed_weights=[1.],allocation=[R],event_count=q*R),
            law_binding=dict(definition=law)))
    return out


def implementation(pauli=True):
    return dict(lowering_id='PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF'if pauli else'LOW',
        scalar='HAMILTONIAN_CONSTANT_PLUS_SAVED_sR_ONCE_AT_TOTAL_T',fragment_order='SAVED_SNAPSHOT_ORDER_ONE_BODY_SEPARATE_HEAD',
        sampler='ORIGINAL_NORMAL_PCG64_NUMPY_CHOICE')


def missing_quote(cfg,method='PR_PAULI'):
    return dict(schema='PR_PRICE_APPLICABILITY_V1',configuration_id=cfg['id'],policy='MODEL',method=method,
        L_D=cfg['spec']['L_D'],rows=[dict(m=r['m'],axis=a,physical_id=r['candidate']['id'],q=r['q_PF'],R=r['R_outer'],
            unit_price_RZ=None,classification='NEEDS_PRICE_EVIDENCE')for r in cfg['rounds']for a in('X','Y')],complete=False)


def pauli_fixture():
    reader=Documents();cfg=config();imp=implementation();cref=reader.add('CONFIG.json',cfg)
    step=Fraction(.2);covered=list(range(1,13));q=8;R=1;req=dict(prefix=3,lambda_P=.25,q=q,R=R,T=float(step*q),step_time=float(step),K=2,
        candidate_id='PAULI_PRICE_CELL:ARTIFICIAL',physical_cell=dict(binding=cfg['spec']['binding'],split=cfg['spec']['split'],
            profile=cfg['spec']['profile'],q=q,R=R,K=2,T=float(step*q),lowering=imp['lowering_id'],route='PR_PAULI_TAIL'))
    binding=dict(status='BINDING_FIXED_BEFORE_SAMPLING',configuration=dict(cref,id=cfg['id']),
        dictionary_identity=cfg['spec']['split']['tail_dictionary'],request=req,cohort='TOY_CAL',scalar=dict(value=2.5,application='ONCE_AT_TOTAL_T'))
    bref=reader.add('BINDING.json',binding)
    raw=dict(status='FINITE_LD3_CAL_PATHS_COMPLETE',compiles=4,binding=bref)
    rawref=reader.add('RESULT.json',raw)
    receipt_rows=[];paths=[];decomp={}
    for index in range(2):
        pid='Q8R1/p'+str(index);axes={}
        for a in('X','Y'):
            C=11+5*q+2*q*R
            ref=reader.add('receipt_%s_%d.json'%(a,index),dict(request=req,cohort='TOY_CAL',record=dict(metrics=dict(rz_count=C))))
            axes[a]=dict(receipt=ref);decomp[pid+':'+a]=dict(exact=True)
            receipt_rows.append(dict(path_id=pid,axis=a,physical_id=req['candidate_id'],q=q,R=R,compiled_RZ=C,receipt=ref))
        paths.append(dict(path_id=pid,axes=axes))
    audit=dict(stops=[],rule_supported=True,paths_ok=[p['path_id']for p in paths],checks=dict(P1_all=True,P2_all=True),
        P3=dict(status='PASS'),P5=dict(status='PASS',margins_all_ok=True,final_run_rz=[2]),P6=dict(status='PASS',decomposition=decomp),
        edges=dict(constants=dict(later_prefix_rz=3,suffix_rz=2,first_prefix_rz=7,end_rz=5)),
        segments={p['path_id']:[dict(context='LATER',body_rz=2*R)for _ in range(q-1)]for p in paths})
    price=dict(status='PRICE_RULE_SUPPORTED_MODEL_REFERENCE',configuration_id=cfg['id'],configuration_source=cref,
        coefficients=dict(A3=11,H3=5,E3=exact(2)),calibration_cell=dict(decomposition_supported=True,rows=receipt_rows),rows=[
            dict(m=r['m'],axis=a,C_ref=exact(11+5*r['q_PF']+2*r['q_PF']*r['R_outer']))for r in cfg['rounds']for a in('X','Y')])
    inputs=dict(configuration=cref,raw=rawref['commit'],scalar=dict(value_hex=2.5.hex()),paths=paths,
        covered_rounds=covered,round_T={str(r['m']):r['T']for r in cfg['rounds']if r['m']in covered})
    group=dict(covered_rounds=covered,driver_request=dict(driver_spec=dict(configuration=cfg,config_id=cfg['id'],implementation=imp)),driver_metadata=dict(request=req))
    package=dict(kind='PR_PAULI_STEP_AUDIT',configuration_source=cref,implementation=imp,preparation_group=group,
        price_source=reader.add('PRICE.json',price),audit_source=reader.add('AUDIT.json',audit),audit_inputs_source=reader.add('INPUTS_AUDIT.json',inputs),binding_source=bref,raw_source=rawref)
    return reader,cfg,imp,package


def df_direct_fixture():
    helper_spec=importlib.util.spec_from_file_location('_step_existing_df_receipt_fixture',ROOT/'tests/test_partial_df_price_reuse.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    reader=Documents();cfg=config(pauli=False);imp=implementation(False);cref=reader.add('CONFIG.json',cfg)
    target=cfg['rounds'][0];pc=dict(physical_candidate=target['physical_candidate'],structure=target['structure'],binding=cfg['spec']['binding'],
        split=cfg['spec']['split'],one_body_policy='SEPARATE_HEAD',law=target['law_binding'],
        price_profile=dict(cfg['spec']['profile'],lowering_id=imp['lowering_id']),implementation=dict(adapter='PR_CANDIDATE_PRICE_CONTEXT_V1'),
        expected_scalar=2.5,preparation_hash=None,input_source=dict(configuration=cref),execution_source=dict(commit='ARTIFICIAL',path='CALLER.py'))
    request=dict(m=0,axes=['X','Y'],physical_context=pc,context_key=digest(pc));request['request_id']=digest(request)
    cell=dict(m=0,role='CALIBRATION',cohort='TOY_CAL',indices=[0],seeds=[[17]],request=request)
    path=helper.fixture_path(cell);pref=reader.add('PATH.json',path)
    calibration=[];prices=[]
    for a in path['record']['records']:
        ref=reader.add({'X':'cosine.json','Y':'sine.json'}[a['axis']],dict(metrics=a['metrics'],fingerprint=a['fingerprint'],compiler='COMPILER'))
        witness=dict(index=0,axis=a['axis'],cohort=cell['cohort'],seeds=[17],receipt=ref,path_receipt=pref)
        calibration.append(dict(m=0,axis=a['axis'],q=target['q_PF'],R=target['R_outer'],n_paths=1,cohort=cell['cohort'],mean_C=exact(a['C']),paths=[witness]))
        prices.append(dict(configuration_id=cfg['id'],m=0,axis=a['axis'],C_ref=exact(a['C'])))
    model=dict(status='FROZEN_2_AXIS_MODELS',holdout_not_opened=True,formula='FULL_WRAPPER_C_OFFSET_ZERO_NO_S_ADDITION',mode='DIRECT_FINITE_COHORT',
        models={a:dict(direct=True,coefficients={},calibration=[c for c in calibration if c['axis']==a])for a in('X','Y')})
    inputs=dict(configuration_id=cfg['id'],configuration=cref,covered_rounds=[0],cells=[cell])
    package=dict(kind='PR_DF_STEP_MODEL',configuration_source=cref,implementation=imp,inputs_source=reader.add('INPUTS.json',inputs),
        model_source=reader.add('FROZEN_MODELS.json',model),price_source=reader.add('AXIS_PRICES.json',dict(rows=prices)))
    return reader,cfg,imp,package


class ScopedPriceTests(unittest.TestCase):
    def test_partial_actual_step_only_registers_missing_rounds_and_is_idempotent(self):
        reader,cfg,imp,p=pauli_fixture();reg=StepPriceRegistry(reader)
        self.assertEqual(reg.ingest(p)['status'],'REGISTERED');self.assertEqual(reg.ingest(p)['status'],'REUSED')
        base=missing_quote(cfg)
        for row in base['rows']:
            if row['m']==0:row.update(unit_price_RZ=exact(3),classification='MODEL_REFERENCE')
        q=reg.quote(cfg,imp,base);self.assertTrue(q['complete'])
        self.assertTrue(all(r['unit_price_RZ']==exact(3)for r in q['rows']if r['m']==0))
        self.assertTrue(all(r['model_alternative']['covered_rounds']==list(range(1,13))for r in q['rows']if r['m']>0))
        reloaded=StepPriceRegistry(reader,state=reg.state);self.assertEqual(q,reloaded.quote(cfg,imp,base))

    def test_exact_one_fifth_does_not_borrow_binary64_step_models(self):
        reader,cfg,imp,p=pauli_fixture();reg=StepPriceRegistry(reader);reg.ingest(p)
        exact_cfg=config('1/5');q=reg.quote(exact_cfg,imp,missing_quote(exact_cfg))
        self.assertFalse(q['complete']);self.assertTrue(all(r['unit_price_RZ']is None for r in q['rows']))

    def test_wrong_audit_dictionary_scalar_and_receipt_do_not_register(self):
        changes=[('AUDIT.json',lambda x:x['P5'].update(status='FAIL')),
                 ('BINDING.json',lambda x:x['dictionary_identity'].update(dictionary_sha256='OTHER')),
                 ('INPUTS_AUDIT.json',lambda x:x['scalar'].update(value_hex=3.5.hex())),
                 ('receipt_X_0.json',lambda x:x['record']['metrics'].update(rz_count=999))]
        for path,change in changes:
            reader,cfg,imp,p=pauli_fixture();change(reader.docs['ARTIFICIAL',path]);reg=StepPriceRegistry(reader)
            with self.assertRaises(ValueError):reg.ingest(p)
            self.assertEqual(reg.state['entries'],[])

    def test_shared_cell_price_transfers_only_same_family_step_and_wrapper_times(self):
        reader,cfg,imp,p=pauli_fixture();reg=StepPriceRegistry(reader);reg.ingest(p)
        changed=deepcopy(cfg);changed['id']='ANOTHER_KAPPA'
        for row in changed['rounds']:
            row['R_outer']=2;row['structure'].update(allocation=[2],event_count=2*row['q_PF'])
            for k in('candidate','physical_candidate'):row[k]['R_outer']=2
            row['candidate']['event_count']=2*row['q_PF']
        q=reg.quote(changed,imp,missing_quote(changed))
        self.assertIsNone(q['rows'][0]['unit_price_RZ'])
        self.assertTrue(all(r['unit_price_RZ']is not None for r in q['rows']if r['m']>0))
        self.assertTrue(all('R'in r['model_alternative']['extrapolation']for r in q['rows']if r['m']>0))
        wrong=deepcopy(changed);wrong['spec']['split']['tail_dictionary']['dictionary_sha256']='DIFFERENT'
        self.assertFalse(any(r['unit_price_RZ']is not None for r in reg.quote(wrong,imp,missing_quote(wrong))['rows']))

    def test_df_direct_receipts_scope_only_identical_physical_point(self):
        reader,cfg,imp,p=df_direct_fixture();reg=StepPriceRegistry(reader);reg.ingest(p)
        q=reg.quote(cfg,imp,missing_quote(cfg,'PR_DF'))
        self.assertFalse(q['complete']);self.assertEqual(q['rows'][0]['classification'],'DIRECT_FINITE_COHORT')
        self.assertEqual(q['rows'][0]['unit_price_RZ'],exact(31));self.assertEqual(q['rows'][1]['unit_price_RZ'],exact(32))
        self.assertTrue(all(r['unit_price_RZ']is None for r in q['rows'][2:]))
        reader.docs['ARTIFICIAL','cosine.json']['fingerprint']='WRONG'
        with self.assertRaises(ValueError):StepPriceRegistry(reader).ingest(p)

    def test_real_saved_pauli_and_df_adapters_read_only_metadata(self):
        r=GitReader(ROOT);loc=lambda path,commit='97413fc1218beb20c70744ca183bfd4f3dfa2437':dict(commit=commit,path=path)
        b='docs/research/d2_automatic_price_20261116/8bf6eb88c735daf2499c/calibration/'
        imp=dict(implementation(),contract_source=dict(commit='13ab1acfaebe5f33958e1db67862f5c45725ab91',path='scripts/run_pauli_b1_prices.py'))
        p=dict(kind='PR_PAULI_STEP_AUDIT',configuration_source=loc('docs/research/d2_automatic_price_20261116/8bf6eb88c735daf2499c/CONFIGURATION.json'),implementation=imp,
            price_source=loc(b+'PRICE_26AXIS.json'),audit_source=loc(b+'audit_run01/AUDIT_RESULT.json'),audit_inputs_source=loc(b+'INPUTS_AUDIT.json'),
            binding_source=loc(b+'BINDING.json'),raw_source=loc(b+'run01/RESULT.json','5aa04f5784c92317ac3e5bb26ebbfb921f790390'))
        reg=StepPriceRegistry(r);self.assertEqual(reg.ingest(p)['status'],'REGISTERED')
        cfg=r.read(p['configuration_source']);q=reg.quote(cfg,imp,missing_quote(cfg));self.assertTrue(q['complete'])
        price=r.read(p['price_source']);self.assertEqual([frac(v['unit_price_RZ'])for v in q['rows']],[frac(v.get('C_ref',v.get('C_ref_exact')))for v in price['rows']])
        c='0ef9668e997b57214eeab2f944e4bcff967b7b4d';b='docs/research/d_pr_candidates_auto_20261116/run01/completion/candidates/20950c2caf36faaf2eb9/df_prices/'
        loc=lambda path,commit=c:dict(commit=commit,path=path)
        inputs=r.read(loc(b+'INPUTS.json'));hold=r.read(loc(b+'HOLDOUT_COMPARISON.json'))
        imp=next(e['implementation']for e in r.read(dict(commit='c1f95e080',path='configs/pr_price_rules/h4_saved_rules.json'))['entries']if e['name']=='DF_LD7')
        p=dict(kind='PR_DF_STEP_MODEL',configuration_source=inputs['configuration'],implementation=imp,inputs_source=loc(b+'INPUTS.json'),
            model_source=loc(b+'FROZEN_MODELS.json',hold['model_commit']),price_source=loc(b+'AXIS_PRICES.json',hold['model_commit']),holdout_source=loc(b+'HOLDOUT_COMPARISON.json'))
        reg=StepPriceRegistry(r);self.assertEqual(reg.ingest(p)['status'],'REGISTERED')
        cfg=r.read(p['configuration_source']);q=reg.quote(cfg,imp,missing_quote(cfg,'PR_DF'));self.assertTrue(q['complete'])
        self.assertEqual([frac(v['unit_price_RZ'])for v in q['rows']],[frac(v['C_ref'])for v in r.read(p['price_source'])['rows']])


class CommonPauliFloorTests(unittest.TestCase):
    def test_only_source_bound_complete_one_piece_monotone_pauli_model(self):
        reader=Documents()
        cfg=config();imp=implementation()
        model=dict(coefficients={'A3':11,'H3':5,'E3':2})
        entry=dict(name='PAULI_LD3',mode='PAULI_AUDIT',configuration_source=reader.add('CONFIG',cfg),
                   model_source=reader.add('MODEL',model),implementation=imp)
        registry=StepPriceRegistry(reader,dict(entries=[entry]))
        terms=registry.common_pauli_model_terms(cfg,imp)
        self.assertEqual(len(terms),26)
        self.assertEqual(terms[0],(18,2))
        self.assertIsNone(registry.common_pauli_model_terms(config(pauli=False),implementation(pauli=False)))
        self.assertIsNone(registry.common_pauli_model_terms(config(delta=.05),imp))
        registry.state['entries'].append(dict(domain=dict(family=registry.api['family'](cfg,imp))))
        self.assertIsNone(registry.common_pauli_model_terms(cfg,imp))
        registry.state['entries'].clear()
        rules=registry.api['PriceRules'](reader,registry.registry)
        rules.entries[0]['saved'][1,'X']=dict(value=1,source=entry['model_source'])
        self.assertIsNone(registry.common_pauli_model_terms(cfg,imp,rules=rules))
        entry['model_source']=reader.add('NEGATIVE_MODEL',dict(coefficients={'A3':11,'H3':5,'E3':-2}))
        self.assertIsNone(StepPriceRegistry(reader,dict(entries=[entry])).common_pauli_model_terms(cfg,imp))


if __name__=='__main__':unittest.main(verbosity=2)
