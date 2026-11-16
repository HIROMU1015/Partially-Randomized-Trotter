"""Artificial receipt/worker continuation only; no scientific arrays/compile."""
from copy import deepcopy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

DRAFT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_partial_df_prices',DRAFT/'scripts/run_pr_df_candidate_prices.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
SOURCE_FILE=E.FIXED_VIEW/'scripts/run_pr_df_ld_split.py'
FIXED_SOURCE=SOURCE_FILE.read_bytes()


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def request(axes=('X','Y')):
    pc=dict(physical_candidate=dict(id='PHYSICAL0',q=1,R_outer=1,prefix=8),structure={'allocation':[1]},binding={'hamiltonian':'H','state':'S','snapshot':'SNAP','rank':10},split={'L_D':8,'lambda_R':0.25},one_body_policy='SEPARATE_HEAD',law={'K':2,'source':'FIXED_LAW'},price_profile={'compiler':'COMPILER'},implementation={'lowering':'FIXED_ZZ'},expected_scalar=2.5,preparation_hash=None,input_source={'configuration':{'commit':'CONFIG_COMMIT','path':'CONFIG.json'}},execution_source={'commit':'CODE_COMMIT','path':'CALLER.py'})
    r=dict(m=0,axes=list(axes),physical_context=pc,context_key=digest(pc));r['request_id']=digest(r);return r


def cell():return dict(m=0,role='CALIBRATION',cohort='CAL',indices=[0],seeds=[[17]],request=request())


def fixture_path(c,axes=('X','Y'),index=0):
    req=E.selected_request(c['request'],axes)
    rows=[dict(axis=a,C=31 if a=='X'else 32,S=20,D=11 if a=='X'else 12,metrics=dict(rz_count=31 if a=='X'else 32,rz_depth=15,cx_count=11,cx_depth=8,total_depth=40,circuit_size=75),fingerprint='FP_'+a)for a in axes]
    record=dict(request=req,records=rows,cohort=c['cohort'],index=index,seeds=c['seeds'][index],event_digest='EVENT',new_compiles=len(axes),cache_hits=0,resolved_context=dict(scalar=2.5,preparation_hash='PREP',tail_hash='TAIL',lambda_R=0.25,deterministic_fragments=list(range(8)),randomized_fragments=[8,9]),source=dict(execution=req['physical_context']['execution_source'],input=req['physical_context']['input_source']))
    record['resolved_context_key']=digest(dict(context=req['physical_context'],resolved=record['resolved_context']))
    return dict(status='VALID',m=c['m'],index=index,record=record,gate_status=E.GATE,compiles=[])


def documents(c,axes=('X','Y'),index=0):
    value=fixture_path(c,axes,index);docs={'PATH.json':value};witnesses=[]
    for a in value['record']['records']:
        docs[{'X':'cosine.json','Y':'sine.json'}[a['axis']]]=dict(metrics=a['metrics'],fingerprint=a['fingerprint'],compiler='COMPILER')
        witnesses.append(dict(axis=a['axis'],index=index,cohort=c['cohort'],seeds=c['seeds'][index],receipt=dict(commit='RAW',path={'X':'cosine.json','Y':'sine.json'}[a['axis']]),path_receipt=dict(commit='RAW',path='PATH.json')))
    return docs,witnesses


class InlinePool:
    def __init__(self,*args,initializer=None,initargs=(),**kw):
        self._processes={}
        if initializer:initializer(*initargs)
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def submit(self,fn,arg):
        future=Future()
        try:future.set_result(fn(arg))
        except BaseException as e:future.set_exception(e)
        return future


class FakeAdapter:
    def __init__(self,root,runtime,**kwargs):self.root=root;self.runtime=runtime;self.observer=kwargs['observer']
    def acquire(self,req,requests,cohort,index,directory):
        c=dict(m=req['m'],request=req,cohort=cohort,seeds={index:[17+index]})
        value=fixture_path(c,req['axes'],index);directory.mkdir(parents=True,exist_ok=True)
        for row in value['record']['records']:
            a=row['axis'];self.observer({'X':'cosine','Y':'sine'}[a]+'_compile','STARTED')
            self.runtime.ctx.compile(NS(num_qubits=9,num_clbits=1,count_ops=lambda:{'measure':1}),{})
            E.write(directory/({'X':'cosine','Y':'sine'}[a]+'.json'),dict(metrics=row['metrics'],fingerprint=row['fingerprint'],compiler='COMPILER'))
        return value['record']


def fake_runtime(root):
    rr=NS(science=NS(m=NS(digest=lambda value:'EVENT')))
    prep=NS(preparation_hash='PREP',deterministic_fragment_indices=list(range(8)),randomized_block_indices=[8,9])
    ev=NS(to_dict=lambda:{'artificial':True})
    ctx=NS(prep=prep,scalar=2.5,h={'sample_requests':lambda physical,prep,cohort,index:([NS(rte_occurrence=NS(events=[ev]))],[17+index],[])},compile=lambda *a,**k:None)
    rr.ctx=ctx;rr.context=lambda req:ctx
    R=NS(ROOT=root,PUBLIC='OUT/calibration',STATE={},D=NS(NativeRuntime=lambda *a,**k:rr,Adapter=FakeAdapter,AXES={'X':'cosine','Y':'sine'},p=NS(PROFILE={'compiler':'COMPILER'})),P=NS(event_metrics=lambda events:{'artificial':len(events)}),restore=lambda *a:('ARTIFICIAL_H',{'fixture':'INPUT'}),time=time,json=json,resource=resource,traceback=traceback,write=E.write)
    R.init_price=lambda flag:setattr(R,'FAILURE',flag)
    return R


def cohort_documents(c,axes=('X',)):
    docs={};witnesses=[]
    for index in c['indices']:
        d,ws=documents(c,axes,index)
        for key,value in d.items():docs[c['cohort']+'/p'+str(index)+'/'+key]=value
        for w in ws:
            w=deepcopy(w)
            for key in ('receipt','path_receipt'):w[key]['path']=c['cohort']+'/p'+str(index)+'/'+w[key]['path']
            witnesses.append(w)
    mean=c['cohort']+'/MEAN.json';docs[mean]={'paths':witnesses}
    quote=dict(m=c['m'],axis=axes[0],direct_alternatives=[dict(source=dict(commit='SAVED_MEAN',path=mean),role='CALIBRATION',cohort=c['cohort'],n_paths=len(c['indices']))])
    return docs,quote


class PartialReuseTests(unittest.TestCase):
    def source(self,c,p):return FIXED_SOURCE if p=='scripts/run_pr_df_ld_split.py'else b'artificial fixed source'

    def test_only_verified_axis_reused_mean_only_or_independent_held_per_axis(self):
        c=cell();docs,witnesses=documents(c,('X',));docs['MEAN.json']={'paths':witnesses}
        q=dict(rows=[dict(m=0,axis='X',direct_alternatives=[dict(source=dict(commit='MEAN',path='MEAN.json'),role='CALIBRATION',cohort='CAL')])])
        E.resolve_reuse([c],q,reader=lambda loc:deepcopy(docs[loc['path']]),source_reader=self.source)
        self.assertEqual(c['axis_evidence']['0']['X']['status'],'REUSE');self.assertEqual(c['axis_evidence']['0']['Y']['status'],'MISSING')
        q['rows'][0]['direct_alternatives'][0]['cohort']='INDEPENDENT'
        E.resolve_reuse([c],q,reader=lambda loc:deepcopy(docs[loc['path']]),source_reader=self.source)
        self.assertEqual(c['axis_evidence']['0']['X']['status'],'HELD_UNRESOLVED_SAVED_AXIS');self.assertEqual(c['axis_evidence']['0']['Y']['status'],'MISSING')
        q['rows'][0]['direct_alternatives'][0]['cohort']='CAL';docs['MEAN.json']={'mean_C':31}
        E.resolve_reuse([c],q,reader=lambda loc:deepcopy(docs[loc['path']]),source_reader=self.source)
        self.assertIn('MEAN_WITHOUT_INDIVIDUAL_PATH_RECEIPTS',c['axis_evidence']['0']['X']['reasons'])

    def test_identity_source_axis_metrics_fingerprint_and_cohort_rejected(self):
        c=cell();docs,w=documents(c)
        cases=[('fingerprint',lambda d:d['cosine.json'].update(fingerprint='WRONG')),
               ('compiler',lambda d:d['cosine.json'].update(compiler='WRONG')),
               ('seed',lambda d:d['PATH.json']['record'].update(seeds=[99])),
               ('cohort',lambda d:d['PATH.json']['record'].update(cohort='OTHER')),
               ('source',lambda d:d['PATH.json']['record']['source'].update(execution={'commit':'OTHER','path':'CALLER.py'})),
               ('missing',lambda d:d.pop('cosine.json')),
               ('axis',lambda d:d['PATH.json']['record']['records'].pop(0))]
        for name,corrupt in cases:
            with self.subTest(name=name):
                damaged=deepcopy(docs);corrupt(damaged)
                with self.assertRaises((ValueError,KeyError)):E.validate_axis(c,0,'X',w[0],reader=lambda loc:damaged[loc['path']],source_reader=self.source)

    def test_axis_receipt_swap_rejected_even_if_metrics_and_fingerprint_equal(self):
        c=cell();docs,w=documents(c)
        wrong=deepcopy(w[0]);wrong['receipt']=w[1]['receipt']
        docs['sine.json']=deepcopy(docs['cosine.json'])
        with self.assertRaisesRegex(ValueError,'AXIS_RECEIPT_PATH_MISMATCH'):
            E.validate_axis(c,0,'X',wrong,reader=lambda loc:docs[loc['path']],source_reader=self.source)

    def test_partial_axis_acquire_then_resume_has_no_repeat_and_means_trace_original(self):
        c=cell();docs,w=documents(c,('X',))
        valid=E.validate_axis(c,0,'X',w[0],reader=lambda loc:docs[loc['path']],source_reader=self.source)
        c['axis_evidence']={'0':{'X':valid,'Y':{'status':'MISSING'}}}
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=fake_runtime(root);inputs=dict(output_directory='OUT',cells=[c],saved_input_files={},runtime={})
            def reader(loc):
                if loc['commit']=='RAW':return deepcopy(docs[loc['path']])
                value=json.loads((root/loc['path']).read_text())
                for key in loc.get('selector',[]):value=value[key]
                return value
            with patch.object(E,'ROOT',root),patch.object(E,'read',side_effect=reader),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R),patch.object(E,'ProcessPoolExecutor',InlinePool):
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['reused_axes'],1);self.assertEqual(result['new_wrapper_axes'],1);self.assertEqual(result['compile_attempts'],1)
                directory=root/'OUT/calibration/acquisition/m0/p0/axes_Y';self.assertTrue((directory/'sine.json').exists());self.assertFalse((directory/'cosine.json').exists())
                original=(directory/'PATH_RECEIPT.json').read_bytes()
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['reused_axes'],2);self.assertEqual(original,(directory/'PATH_RECEIPT.json').read_bytes())
                means=E.means(inputs,'NEW_RAW','calibration');self.assertEqual(means[0]['paths'][0]['receipt']['commit'],'RAW');self.assertEqual(means[1]['paths'][0]['receipt']['commit'],'NEW_RAW');self.assertEqual(means[0]['mean_C'],E.enc(31));self.assertEqual(means[1]['mean_C'],E.enc(32))

    def test_held_X_does_not_prevent_missing_Y_and_partial_is_preserved(self):
        c=cell();c['axis_evidence']={'0':{'X':{'status':'HELD_UNRESOLVED_SAVED_AXIS','reasons':['MEAN_ONLY']},'Y':{'status':'MISSING'}}}
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=fake_runtime(root);inputs=dict(output_directory='OUT',cells=[c],saved_input_files={},runtime={})
            with patch.object(E,'ROOT',root),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R),patch.object(E,'ProcessPoolExecutor',InlinePool):
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_wrapper_axes'],1);self.assertEqual(result['held_axes'],1);self.assertEqual(result['rows'][0]['axes']['Y']['status'],'REUSE')
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0)

    def test_missing_old_receipt_holds_that_index_no_blind_duplicate(self):
        c=cell();c['indices']=[0,1];c['seeds']=[[17],[19]];docs,witnesses=documents(c,('X',))
        docs['MEAN.json']={'paths':witnesses};q=dict(rows=[dict(m=0,axis='X',direct_alternatives=[dict(source=dict(commit='MEAN',path='MEAN.json'),n_paths=2,role='CALIBRATION',cohort='CAL')])])
        E.resolve_reuse([c],q,reader=lambda loc:deepcopy(docs[loc['path']]),source_reader=self.source)
        self.assertEqual(c['axis_evidence']['0']['X']['status'],'REUSE');self.assertEqual(c['axis_evidence']['1']['X']['status'],'HELD_UNRESOLVED_SAVED_AXIS');self.assertEqual(c['axis_evidence']['1']['Y']['status'],'MISSING')

    def test_all_saved_axes_skip_acquisition_and_conflicting_same_path_rejected(self):
        c=cell();docs,w=documents(c)
        c['axis_evidence']={'0':{a:E.validate_axis(c,0,a,w[i],reader=lambda loc:docs[loc['path']],source_reader=self.source)for i,a in enumerate(('X','Y'))}}
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=NS();inputs=dict(output_directory='OUT',cells=[c])
            with patch.object(E,'ROOT',root),patch.object(E,'read',side_effect=lambda loc:deepcopy(docs[loc['path']])),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R):
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['reused_axes'],2)
                c['axis_evidence']['0']['Y']['event_digest']='OTHER_EVENT'
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['held_axes'],1)

    def test_fixed_worker_rejects_bad_full_wrapper_before_compile(self):
        c=cell();c['request']=E.selected_request(c['request'],['Y'])
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=fake_runtime(root);rr=R.D.NativeRuntime();R.STATE.update(contexts={0:rr.ctx},runtimes={0:rr},inputs={'cache_root':'ARTIFICIAL_CACHE'});R.init_price(threading.Event());(root/'OUT/calibration/acquisition').mkdir(parents=True)
            class BadWrapper:
                def __init__(self,*a,**kw):pass
                def acquire(self,*a,**kw):rr.ctx.compile(NS(num_qubits=8,num_clbits=1,count_ops=lambda:{'measure':1}),{})
            R.D.Adapter=BadWrapper
            with patch.object(E,'ROOT',root),patch.object(E,'runtime',return_value=R),patch.object(E,'blob',side_effect=self.source):
                result=E.selected_acquire_path((c,0))
                self.assertEqual(result['status'],'STOP');self.assertIn('Full measured wrapper',result['error']);self.assertEqual(result['compiles'],[])

    def test_adopt_saved_four_path_cohort_then_acquire_only_missing_Y_without_seed_input(self):
        saved=cell();saved.update(cohort='A_SAVED_CAL',indices=list(range(4)),seeds=[[17+i]for i in range(4)])
        docs,row=cohort_documents(saved);c=deepcopy(saved);c.update(cohort='NEW_AUTOMATIC_CAL',seeds=[[100+i]for i in range(4)])
        quote={'rows':[row]};reader=lambda loc:deepcopy(docs[loc['path']])
        adopted=E.adopt_saved_calibration_cohorts([c],quote,reader=reader,source_reader=self.source)
        self.assertEqual(c['cohort'],'A_SAVED_CAL');self.assertEqual(c['seeds'],saved['seeds']);self.assertEqual(adopted[0]['status'],'ADOPTED_SAVED_CALIBRATION_COHORT')
        E.resolve_reuse([c],quote,reader=reader,source_reader=self.source)
        self.assertTrue(all(v['X']['status']=='REUSE'and v['Y']['status']=='MISSING'for v in c['axis_evidence'].values()))
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=fake_runtime(root);inputs=dict(output_directory='OUT',cells=[c],saved_input_files={},runtime={})
            def read_all(loc):
                return reader(loc)if loc['path']in docs else json.loads((root/loc['path']).read_text())
            with patch.object(E,'ROOT',root),patch.object(E,'read',side_effect=read_all),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R),patch.object(E,'ProcessPoolExecutor',InlinePool):
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['compile_attempts'],4);self.assertEqual(result['resumed_partial_paths'],4);self.assertEqual(result['new_independent_paths'],0)
                self.assertTrue(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((root/'OUT/calibration/acquisition/RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['reused_axes'],8)
                # A stored path with a different digest cannot be combined with saved X.
                p=root/'OUT/calibration/acquisition/m0/p0/axes_Y/PATH_RECEIPT.json';v=json.loads(p.read_text());v['record']['event_digest']='OTHER_EVENT';p.write_text(json.dumps(v))
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                index=json.loads((root/'OUT/calibration/acquisition/m0/p0/PATH_INDEX.json').read_text());self.assertTrue(all(a['status']=='HELD_EVENT_DIGEST_MISMATCH'for a in index['axes'].values()))

    def test_adoption_uses_source_order_not_price_preserves_holdout_and_rejects_incomplete(self):
        docs={};rows=[]
        for name,price in [('B_SAVED',1),('A_SAVED',999)]:
            saved=cell();saved.update(cohort=name,indices=list(range(4)),seeds=[[17+i]for i in range(4)]);d,r=cohort_documents(saved);docs.update(d);r['direct_alternatives'][0]['value']=E.enc(price);rows.append(r)
        c=cell();c.update(cohort='NEW',indices=list(range(4)),seeds=[[100+i]for i in range(4)])
        reader=lambda loc:deepcopy(docs[loc['path']]);quote={'rows':rows}
        E.adopt_saved_calibration_cohorts([c],quote,reader=reader,source_reader=self.source);self.assertEqual(c['cohort'],'A_SAVED')
        for row in rows:row['direct_alternatives'][0]['role']='INDEPENDENT_HOLDOUT'
        c['cohort']='NEW';report=E.adopt_saved_calibration_cohorts([c],quote,reader=reader,source_reader=self.source);self.assertEqual(c['cohort'],'NEW');self.assertTrue(all(r['reason']=='PRESERVE_SAVED_HOLDOUT_ROLE'for r in report[0]['rejected']))
        # A missing path cannot supply a whole cohort's seed matrix.
        rows=rows[:1];rows[0]['direct_alternatives'][0]['role']='CALIBRATION';docs['B_SAVED/MEAN.json']['paths']=docs['B_SAVED/MEAN.json']['paths'][:3]
        report=E.adopt_saved_calibration_cohorts([c],{'rows':rows},reader=reader,source_reader=self.source);self.assertEqual(c['cohort'],'NEW');self.assertEqual(report[0]['status'],'KEEP_NEW_FIXED_COHORT')
        hold=deepcopy(c);hold['role']='HOLDOUT';self.assertEqual(E.adopt_saved_calibration_cohorts([hold],quote,reader=reader,source_reader=self.source),[])

    def test_prepare_persists_adopted_cohort_and_missing_axis_plan_without_manual_seeds(self):
        import hashlib
        saved=cell();saved.update(cohort='SAVED_FOUR_CAL',indices=list(range(4)),seeds=[[17+i]for i in range(4)])
        docs,row=cohort_documents(saved);quote=dict(configuration_id='CFG',rows=[row]);config_source=dict(commit='CFG_COMMIT',path='CFG.json')
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);context=root/'CONTEXT.json';context.write_text(json.dumps({'prefixes':{'8':dict(lambda_R=.25,scalar=.25)}}));h=hashlib.sha256(context.read_bytes()).hexdigest()
            tail_source=dict(index=E.INDEX,context_sha256=h,selector=['prefixes','8'])
            cfg=dict(id='CFG',spec=dict(L_D=8,binding=dict(hamiltonian='H'),split=dict(lambda_source=tail_source,lambda_R=E.enc(.25))),rounds=[dict(physical_candidate=dict(id='PHYSICAL'+str(m)))for m in range(13)])
            def reader(loc):
                if loc==config_source:return deepcopy(cfg)
                if loc==E.INDEX:return dict(T20_reference_files=dict(H4=dict(path=str(context),sha256=h)))
                if loc==E.SCALAR_CONSTANT:return dict(source=dict(hamiltonian_hash='H'),value=2.25)
                if loc==E.BASE_INPUTS:return dict(runtime={},saved_input_files={},local_support={})
                return deepcopy(docs[loc['path']])
            def from_b(config,m,physical,**kw):
                req=request();req['m']=m;req['physical_context']['physical_candidate']['id']='PHYSICAL'+str(m)if m else'PHYSICAL0';req['physical_context']['input_source']=kw['input_source'];req['physical_context']['execution_source']=kw['execution_source'];req['context_key']=digest(req['physical_context']);req.pop('request_id');req['request_id']=digest(req);return req
            R=NS(P=NS(trial_seed_function=lambda root:lambda c,name,i,j,k:1000+(0 if c['id']=='PHYSICAL0'else 100)+i*10+j),D=NS(from_b=from_b))
            selected=dict(status='CALIBRATION_DESIGN_READY',reasons=[],mode='FULL_AFFINE_Q_R1',calibration_rounds=[0],holdout_round=3)
            with patch.object(E,'ROOT',root),patch.object(E,'fixed'),patch.object(E,'read',side_effect=reader),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R),patch.object(E,'design',return_value=selected):
                E.prepare(config_source,'OUTPUT','NEW_SOURCE','AUTOMATIC',quote)
            inputs=json.loads((root/'OUTPUT/INPUTS.json').read_text());cal=inputs['cells'][0]
            self.assertEqual(cal['cohort'],'SAVED_FOUR_CAL');self.assertEqual(cal['seeds'],saved['seeds']);self.assertTrue(all(v['X']['status']=='REUSE'and v['Y']['status']=='MISSING'for v in cal['axis_evidence'].values()));self.assertEqual(inputs['cohort_adoption'][0]['status'],'ADOPTED_SAVED_CALIBRATION_COHORT');self.assertEqual(inputs['max_wrapper_compiles'],8)

    def test_started_axis_no_retry(self):
        c=cell();c['axis_evidence']={'0':{'X':{'status':'MISSING'},'Y':{'status':'MISSING'}}}
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);R=fake_runtime(root);inputs=dict(output_directory='OUT',cells=[c]);out=root/'OUT/calibration/acquisition';out.mkdir(parents=True);(out/'STARTED_m0_p0_axesXY.json').write_text('{}')
            with patch.object(E,'ROOT',root),patch.object(E,'blob',side_effect=self.source),patch.object(E,'runtime',return_value=R):
                self.assertFalse(E.acquire(inputs,'INPUTS','calibration','SOURCE'))
                result=json.loads((out/'RESULT.json').read_text());self.assertEqual(result['new_paths'],0);self.assertEqual(result['held_axes'],2)

if __name__=='__main__':unittest.main(verbosity=2)
