"""Saved replay and invented metadata/fake-provider records, no H4 evaluation."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_df_signal_entry import (DFSignalEntry,FixedRuntime,IMPLEMENTATION,
    INDEX,SAVED,LOCAL,RECIPE,_package,mean_from_record)


class MemoryReader:
    def __init__(self):self.docs={}
    def put(self,path,data):
        ref=dict(commit='ARTIFICIAL',path=path);self.docs[path]=deepcopy(data);return ref
    def read(self,ref):return deepcopy(self.docs[ref['path']])


def fixture_info():
    binding=dict(system='H4',basis='ARTIFICIAL',geometry='ARTIFICIAL',rank=10,
        rank_policy='ARTIFICIAL',hamiltonian='ARTIFICIAL_H',state='ARTIFICIAL_PSI',snapshot='ARTIFICIAL_SNAPSHOT')
    return dict(binding=binding,descriptor_sha256='ARTIFICIAL_DESCRIPTOR',
        prefixes={'7':dict(lambda_R=.25,scalar=.125),'9':dict(lambda_R=.125,scalar=.0625)},
        hamiltonian_constant=dict(value=.375,source=dict(fixture='ARTIFICIAL_C')))


def refresh(cfg,common):
    """IDs/identities for an invented fixture, not B's H4 candidate generator."""
    spec=cfg['spec']
    for row in cfg['rounds']:
        c=row['physical_candidate'];c['id']=common.candidate_id(c)
        row.update(T=dict(numerator=c['T'].as_integer_ratio()[0],denominator=c['T'].as_integer_ratio()[1]),
                   actual_delta_pf=dict(numerator=c['delta'].as_integer_ratio()[0],denominator=c['delta'].as_integer_ratio()[1]),
                   q_PF=c['q'],R_outer=c['R_outer'],structure=common.structure(c))
        row['candidate']=dict(c,id='RMSE:ARTIFICIAL_'+cfg['id']+'_'+str(row['m']))
        row['identity']=dict(m=row['m'],candidate=row['candidate'],configuration=dict(configuration_id=cfg['id'],spec=deepcopy(spec)))
        row['law_binding']=dict(configuration_id=cfg['id'],candidate_id=row['candidate']['id'],kappa=spec['kappa'],
            definition=dict(model='ideal_finite_RTE_law',source=dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',path='src/trotterlib/rte.py'),
                A_declaration=dict(model='ideal_finite_RTE_law',candidate_id=c['id'],q_PF=c['q'],R_outer=c['R_outer'],K=c['K'],
                    source=dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',path='src/trotterlib/rte.py'))))
    return cfg


def invented(common,ld=7):
    info=fixture_info();tail=info['prefixes'][str(ld)]
    spec=dict(method='PR',L_D=ld,M=12,formula='2nd',finite_K=2,kappa='ARTIFICIAL_KAPPA',delta_pf=.125,
        binding=info['binding'],profile=dict(fixture='ARTIFICIAL_PROFILE'),
        law=dict(model='ideal_finite_RTE_law',source=dict(composition_recipe=IMPLEMENTATION)),
        split=dict(L_D=ld,deterministic_scope='PREFIX',randomized_scope='TAIL_AFTER_PREFIX',original_PF_method='B0',
            lambda_R=tail['lambda_R'],lambda_source=dict(index=INDEX,context_sha256=info['descriptor_sha256'],selector=['prefixes',str(ld)])))
    cfg=dict(schema='PR_RPE_RMSE_V2',id='ARTIFICIAL_LD'+str(ld),spec=spec,rounds=[])
    for m in range(13):
        cfg['rounds'].append(dict(m=m,q_RPE=1<<m,physical_candidate=dict(system='H4',method='PR',prefix=ld,
            formula='2nd',K=2,T=.125*(1<<m),delta=.125,q=1<<m,R_outer=1)))
    return refresh(cfg,common)


def fake_record(job):
    record={k:deepcopy(job[k])for k in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context')}
    signal={k:dict(real=.125,imag=.25)for k in ('raw','corrected','reference','exact_tail','total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')}
    signal.update(log_B=.125,B=1.125,attenuation=.875,
        allowance_evidence=dict(normalizer_one_ulp_log_sensitivity=.001,maximum_norm_domain='ARTIFICIAL'))
    record.update(status=LOCAL,candidate=deepcopy(job['physical_context']['physical_candidate']),source=IMPLEMENTATION,
        numerical=dict(gate_pass=True),scalar_once=True,input_source=dict(fixture='ARTIFICIAL_INPUT'),
        reference_source=dict(fixture='ARTIFICIAL_REFERENCE'),state_evidence=dict(fixture='ARTIFICIAL_PSI'),
        signal=signal,allowance=dict(log_B_margin=.001,rigorous_bound=False))
    return record


class FixtureRuntime:
    def __init__(self,actual):
        self.metadata=actual.metadata;self.common=actual.common;self.calls=[]
        self.provider=SimpleNamespace(saved_h4_reference=self.reference)
    def reference(self,root,job):
        if job['physical_context']['physical_candidate']['T'] != .125*(1<<job['m']):
            raise ValueError('ARTIFICIAL reference time missing')
        return dict(source=dict(fixture='ARTIFICIAL_REFERENCE'),T=.125*(1<<job['m']))
    def jobs(self,cfg):
        def validate(c):
            assert c['id'].startswith('ARTIFICIAL') and c['spec']['binding']['hamiltonian']=='ARTIFICIAL_H'
        return self.metadata.plan(cfg,{r['m']:r['physical_candidate']for r in cfg['rounds']},
            b_validate=validate,b_exact_scalar=lambda v:v['numerator']/v['denominator']if isinstance(v,dict)else v,
            numerical_recipe=dict(source=IMPLEMENTATION,deterministic_PF='DIRECT_EXPM',finite_RTE=RECIPE,sampling=False))['jobs']


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actual=FixedRuntime(ROOT)
        cls.saved=DFSignalEntry(ROOT,runtime=cls.actual)

    def fixture(self,ld=7):
        rt=FixtureRuntime(self.actual);reader=MemoryReader();cfg=invented(rt.common,ld)
        e=DFSignalEntry(ROOT,runtime=rt,reader=reader,metadata=fixture_info(),saved=[])
        cs=reader.put('configuration.json',cfg);target=dict(binding=cfg['spec']['binding'],fixture='ARTIFICIAL_TARGET')
        means=[]
        for job in rt.jobs(cfg):
            rec=fake_record(job);src=reader.put('saved/m'+str(job['m'])+'.json',rec)
            means.append(mean_from_record(cfg['rounds'][job['m']],rec,src,target))
        bundle=dict(status='ALL13_PHYSICAL_MEANS_NUMERICALLY_CHECKED_RMSE_UNQUALIFIED',configuration_source=cs,
            configuration_id=cfg['id'],numerical_source=IMPLEMENTATION,raw_result_carrier='ARTIFICIAL',target=target,
            shots=None,sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True,means=means)
        packet=reader.put('packet.json',bundle);e.saved=[packet]
        reader.docs[SAVED[1]['path']]=deepcopy(bundle)  # existing field-layout template, invented payload
        return e,cfg,reader,rt

    def test_saved_four_configurations_and_LD7_LD9_exact_bundle(self):
        with patch.object(self.actual.provider,'evaluate',side_effect=AssertionError('NO_REAL_SIGNAL')),patch.object(self.actual.provider,'load_h4_context',side_effect=AssertionError('NO_H4_CONTEXT')):
            for ref in SAVED:
                packet=self.saved.reader.read(ref);cfg=self.saved.reader.read(packet['configuration_source'])
                self.assertEqual(self.saved.plan(cfg)['counts'],dict(REUSE=13,REQUEST=0,UNSUPPORTED=0))
                self.assertEqual(self.saved.assemble(packet['configuration_source'],{m['m']:m['source']for m in packet['means']}),packet)

    def test_saved_bundles_through_original_Q3_intake(self):
        view=ROOT.parent/'d2-pr-df-ld-split-q3-source-20261115'
        name=_package(view,'27e70efe533862f38f4080aabb9cdd3ad1749fe9','_pr_df_test_q3_intake')
        K=importlib.import_module(name+'.rpe_signal_statistics_caller')
        C=importlib.import_module(name+'.rpe_c_interfaces');I=C.interfaces(str(view)).pipeline
        for ref in SAVED[2:]:
            old=self.saved.reader.read(ref);cfg=self.saved.reader.read(old['configuration_source'])
            bundle=self.saved.assemble(old['configuration_source'],{r['m']:r['source']for r in old['means']})
            request=I.plan(view,cfg,target=bundle['target'],signal_source=bundle['numerical_source'],pr_physical={r['m']:r['physical_candidate']for r in cfg['rounds']})
            packet=K.bind_bundle(cfg,bundle,{m['m']:self.saved.reader.read(m['source'])for m in bundle['means']},
                bundle_source=ref,request=request,integration=I)
            self.assertEqual(len(packet['means']),13)
            self.assertEqual(packet['binding_status'],'SOURCE_BOUND_VIEW_NOT_PHYSICAL_QUALIFICATION')

    def test_LD9_never_uses_LD7_on_same_R_and_event_count(self):
        e,cfg,reader,rt=self.fixture(7)
        other=invented(rt.common,9)
        self.assertEqual(e.plan(other)['counts'],dict(REUSE=0,REQUEST=13,UNSUPPORTED=0))

    def test_kappa_declaration_only_reuses_every_round(self):
        e,cfg,reader,rt=self.fixture();cfg['id']='ARTIFICIAL_KAPPA_ALIAS';cfg['spec']['kappa']='ARTIFICIAL_OTHER_KAPPA'
        refresh(cfg,rt.common)
        self.assertEqual(e.plan(cfg)['counts'],dict(REUSE=13,REQUEST=0,UNSUPPORTED=0))

    def test_step_change_requests_only_changed_rounds(self):
        e,cfg,reader,rt=self.fixture();cfg['id']='ARTIFICIAL_DELTA';cfg['spec']['delta_pf']=.0625
        for m in (3,8):
            cfg['rounds'][m]['physical_candidate']['q']*=2
            cfg['rounds'][m]['physical_candidate']['delta']*=.5
        refresh(cfg,rt.common);plan=e.plan(cfg)
        self.assertEqual([r['m']for r in plan['rounds']if r['decision']=='REQUEST'],[3,8])
        self.assertEqual(plan['counts'],dict(REUSE=11,REQUEST=2,UNSUPPORTED=0))

    def test_kappa_schedule_change_requests_only_changed_R(self):
        e,cfg,reader,rt=self.fixture();cfg['id']='ARTIFICIAL_KAPPA_R';cfg['spec']['kappa']='ARTIFICIAL_CHANGED'
        for m in (10,11,12):cfg['rounds'][m]['physical_candidate']['R_outer']=2
        refresh(cfg,rt.common);plan=e.plan(cfg)
        self.assertEqual([r['m']for r in plan['rounds']if r['decision']=='REQUEST'],[10,11,12])

    def test_head_tail_scalar_law_and_H_mismatches(self):
        mutations=[lambda c:c['spec']['split'].__setitem__('scalar',9),
            lambda c:c['spec']['binding'].__setitem__('hamiltonian','OTHER'),
            lambda c:c['spec']['split'].__setitem__('one_body_policy','RANDOM'),
            lambda c:c['spec']['split'].__setitem__('tail_dictionary',dict(kind='PAULI')),
            lambda c:c['spec']['law'].__setitem__('model','OTHER'),
            lambda c:c['spec']['law']['source'].__setitem__('composition_recipe',dict(commit='OTHER')),
            lambda c:c['spec']['split'].__setitem__('fragment_order',[9,8,7])]
        for mutate in mutations:
            e,cfg,reader,rt=self.fixture();mutate(cfg)
            self.assertEqual(e.plan(cfg)['counts']['UNSUPPORTED'],13)

    def test_round_law_source_unsupported_and_no_reference_fallback(self):
        e,cfg,reader,rt=self.fixture();cfg['rounds'][4]['law_binding']['definition']['source']['commit']='OTHER'
        plan=e.plan(cfg)
        self.assertEqual(plan['rounds'][4]['decision'],'UNSUPPORTED')
        self.assertEqual(plan['counts']['REUSE'],12)
        e,cfg,reader,rt=self.fixture();cfg['rounds'][5]['physical_candidate']['T']*=.5
        cfg['rounds'][5]['physical_candidate']['delta']*=.5
        refresh(cfg,rt.common);self.assertEqual(e.plan(cfg)['rounds'][5]['decision'],'UNSUPPORTED')

    def test_supported_m5_and_unsupported_formula(self):
        e,cfg,reader,rt=self.fixture();cfg['spec']['formula']='4th(m5_best)'
        for row in cfg['rounds']:row['physical_candidate'].update(formula='4th(m5_best)',R_outer=128)
        refresh(cfg,rt.common);self.assertEqual(e.plan(cfg)['counts']['REQUEST'],13)
        cfg['spec']['formula']='4th(yoshida)'
        self.assertEqual(e.plan(cfg)['counts']['UNSUPPORTED'],13)

    def test_stage_rebound_metadata_only_and_bundle(self):
        e,cfg,reader,rt=self.fixture();cfg['id']='ARTIFICIAL_ALIAS';cfg['spec']['kappa']='ARTIFICIAL_NEW_KAPPA';refresh(cfg,rt.common)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'job';result=e.stage(cfg,out);self.assertEqual(result['status'],'CARRIERS_READY')
            csrc=reader.put('new/CONFIGURATION.json',json.loads((out/'CONFIGURATION.json').read_text()));sources={}
            for m in range(13):
                rec=json.loads((out/'raw'/('m'+str(m)+'.json')).read_text())
                old=reader.docs['saved/m'+str(m)+'.json']
                for key in ('signal','numerical','allowance','source','reference_source','state_evidence'):
                    self.assertEqual(rec[key],old[key])
                self.assertIn('reuse',rec);sources[m]=reader.put('new/raw/m'+str(m)+'.json',rec)
            bundle=e.assemble(csrc,sources)
            self.assertEqual(bundle['configuration_id'],cfg['id']);self.assertEqual(len(bundle['means']),13)
            with self.assertRaises(FileExistsError):e.stage(cfg,out)

    def test_missing_acquisition_and_failed_gate_preserved(self):
        e,cfg,reader,rt=self.fixture();cfg['rounds'][3]['physical_candidate']['R_outer']=2;refresh(cfg,rt.common)
        with tempfile.TemporaryDirectory() as tmp:
            missing=e.stage(cfg,Path(tmp)/'missing')
            self.assertEqual(missing['status'],'PARTIAL_OR_INVALID_NO_BUNDLE');self.assertEqual(missing['missing'],[3])
            job=e.plan(cfg)['rounds'][3]['job'];rec=fake_record(job)
            rec.update(status='NUMERICAL_FAILURE',signal=None,numerical=dict(gate_pass=False,normalizer=1.0))
            src=reader.put('bad.json',rec);res=e.stage(cfg,Path(tmp)/'bad',obtained={3:src})
            self.assertEqual(res['status'],'PARTIAL_OR_INVALID_NO_BUNDLE')
            self.assertEqual(json.loads((Path(tmp)/'bad/raw/invalid_m3.json').read_text()),rec)

    def test_fake_provider_output_only_for_requested_round(self):
        e,cfg,reader,rt=self.fixture();cfg['rounds'][6]['physical_candidate']['R_outer']=2;refresh(cfg,rt.common)
        plan=e.plan(cfg);calls=[]
        def fake_evaluate(job):calls.append(job['m']);return fake_record(job)
        obtained={r['m']:reader.put('obtained/m'+str(r['m'])+'.json',fake_evaluate(r['job']))for r in plan['rounds']if r['decision']=='REQUEST'}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(e.stage(cfg,Path(tmp)/'out',obtained=obtained)['status'],'CARRIERS_READY')
        self.assertEqual(calls,[6])

    def test_no_partial_mixed_carrier_or_foreign_record_bundle(self):
        e,cfg,reader,rt=self.fixture();cs=dict(commit='ARTIFICIAL',path='configuration.json')
        src={m:dict(commit='ARTIFICIAL',path='saved/m'+str(m)+'.json')for m in range(13)}
        bad=deepcopy(src);bad.pop(3)
        with self.assertRaisesRegex(ValueError,'Complete'):e.assemble(cs,bad)
        bad=deepcopy(src);bad[3]['commit']='OTHER'
        with self.assertRaisesRegex(ValueError,'one raw'):e.assemble(cs,bad)
        reader.docs['saved/m3.json']['B_candidate_id']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'mismatch'):e.assemble(cs,src)

    def test_cli_plan_bundle_and_staging_use_same_saved_route(self):
        old=self.saved.reader.read(SAVED[3])
        req=dict(configuration_source=old['configuration_source'],raw_sources={str(m['m']):m['source']for m in old['means']})
        cli=ROOT/'scripts/prepare_pr_df_signals.py'
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);inp=base/'input.json';inp.write_text(json.dumps(req))
            for mode,name in [('plan','plan.json'),('bundle','bundle.json'),('stage','carrier')]:
                subprocess.run([sys.executable,str(cli),mode,'--input',str(inp),'--output',str(base/name)],check=True)
            self.assertEqual(json.loads((base/'plan.json').read_text())['counts'],dict(REUSE=13,REQUEST=0,UNSUPPORTED=0))
            self.assertEqual(json.loads((base/'bundle.json').read_text()),old)
            for mean in old['means']:
                self.assertEqual(json.loads((base/'carrier/raw'/('m'+str(mean['m'])+'.json')).read_text()),self.saved.reader.read(mean['source']))

    def test_existing_price_rule_entry_is_unchanged_and_usable(self):
        from trottertracks.pr_price_rules import PriceRules
        prices=PriceRules.from_repo(ROOT)
        result=prices.quote(prices.template('DF_LD7'))
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['rows']),26)


if __name__=='__main__':unittest.main(verbosity=2)
