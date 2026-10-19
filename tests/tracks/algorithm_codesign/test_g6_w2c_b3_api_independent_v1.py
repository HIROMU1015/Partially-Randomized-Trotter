"""Independent expectations: synthetic replay, immutable contracts and saved evidence."""
import copy,json,os,pathlib,tempfile,shutil,hashlib,unittest,sys
from trottertracks.algorithm_codesign.g6_w2c_b3_api_independent_v1 import static as S,contract as K,bits as B,adapter as A
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V
W=pathlib.Path(__file__).resolve().parents[3]
P=W/'docs/research/track_b_g6_w2c_b3_api_binding_independent_acceptance_local_20261019'
H='82896d53fa97235fa848d6c0833eb193af8b99d6';SC='76ca3c4089b201d55da0fede08e1f77733ac38a5'
Q=P/'fixed_reference'/H/'docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019'
RAW=pathlib.Path(os.environ['B3_API_RAW_ROOT']);OUT=pathlib.Path(os.environ['B3_API_TEST_OUTPUT']);OUT.mkdir(parents=True,exist_ok=True)
C=K.read(Q/'technical_contract_candidate.json');TREE=P/'fixed_reference'/H
RESULTS=[]

def loc(name):return K.read(Q/'acceptance'/(name+'_location.json'))['root']
def normal_trial(arm='ordinary',layer='B',axis='X',index=0,phase='correctness'):
    counter=0
    for _ in range(index):_,_,_,counter=B.replay(arm,axis,counter)
    kind,event,tr,_=B.replay(arm,axis,counter)
    task=dict(arm=arm,layer=layer,condition='S',phase=phase,pass_name='baseline')
    local=f'{phase}/baseline/{arm}/{layer}/{axis}/{index}'
    record=dict(local_trial_id=local,global_trial_id='ARTIFICIAL_TEST/S/'+local,domain=tr['domain'],consumption=tr,event_digest=V.event_hash(dict(kind=kind,event=event)))
    angle=dict(artificial=True,not_a_mathematical_enclosure=True,actual_angle=None,event_requery_surrogate=hashlib.sha256(V.canonical(event).encode()).hexdigest()) if kind=='EVENT' and layer=='B' else None
    payload=dict(trial_id=local,axis=axis,kind=kind,event=event,evidence=dict(artificial=True,consumption=tr),angle=angle)
    if phase=='profile' and index>=16:
        payload=dict(trial_id=local,axis=axis,kind=kind,event_sha256=record['event_digest'],evidence_sha256=hashlib.sha256(V.canonical(payload['evidence']).encode()).hexdigest(),angle_sha256=hashlib.sha256(V.canonical(angle).encode()).hexdigest(),retention='HASHED_FULL_DOCUMENTS_NOT_RETAINED')
    return payload,record,task,{'X':counter if axis=='X' else 0,'Y':counter if axis=='Y' else 0},{'X':index if axis=='X' else 0,'Y':index if axis=='Y' else 0}

def write_json(p,obj):
    if p.exists():p.unlink()
    p.write_text(json.dumps(obj,separators=(',',':'))+'\n')

class Replay(unittest.TestCase):
    def test_modes_AB_passes_share_seed_but_global_ids_distinct(self):
        p,r,t,c,i=normal_trial();ids=set()
        for mode in ('S','P1'):
            for layer in ('A','B'):
                for pa in ('baseline','instrumented'):
                    q,s,u,cs,ns=normal_trial(layer=layer);u.update(condition=mode,pass_name=pa)
                    local='/'.join((u['phase'],pa,u['arm'],layer,'X','0'));q['trial_id']=s['local_trial_id']=local;s['global_trial_id']='ARTIFICIAL_TEST/'+mode+'/'+local
                    B.trial(q,s,u,'ARTIFICIAL_TEST',cs,ns);ids.add(s['global_trial_id'])
        self.assertEqual(len(ids),8)
    def test_eight_separate_artificial_domains(self):
        self.assertEqual(len({B.replay(a,x,0)[2]['draws'][0]['value_hex'] for a in V.ARMS for x in ('X','Y')}),8)
    def test_independent_codec_boundary(self):
        source=(P/'fixed_reference'/SC/S.SOURCE/'g6_f3_contract.py').read_text()
        self.assertIn("'$rat'",source);self.assertNotIn('"$rat"',V.canonical(__import__('fractions').Fraction(1,3)))

def replay_case(arm,layer,axis):
    def test(self):
        for phase,index in [('correctness',0),('profile',16)]:
            p,r,t,c,i=normal_trial(arm,layer,axis,index,phase);B.trial(p,r,t,'ARTIFICIAL_TEST',c,i)
    return test
for a in V.ARMS:
    for l in ('A','B'):
        for x in ('X','Y'):setattr(Replay,'test_replay_'+a+'_'+l+'_'+x,replay_case(a,l,x))

MUTATORS={
'bool_width':lambda p,r:r['consumption']['draws'][0].__setitem__('width',True),
'bool_counter':lambda p,r:r['consumption']['draws'][0].__setitem__('counter_before',False),
'bool_total':lambda p,r:r['consumption'].__setitem__('counter_after',True),
'wrong_width':lambda p,r:r['consumption']['draws'][0].__setitem__('width',159),
'wrong_counter':lambda p,r:r['consumption']['draws'][0].__setitem__('counter_before',1),
'wrong_value':lambda p,r:r['consumption']['draws'][0].__setitem__('value_hex','0x0'),
'wrong_discard':lambda p,r:r['consumption']['draws'][0].__setitem__('discarded_bits',0),
'missing_draw':lambda p,r:r['consumption']['draws'].pop(),
'extra_draw':lambda p,r:r['consumption']['draws'].append(copy.deepcopy(r['consumption']['draws'][0])),
'wrong_consumed':lambda p,r:r['consumption'].__setitem__('bits_consumed',0),
'mode_in_seed':lambda p,r:r.__setitem__('domain','ARTIFICIAL_API_BINDING_V1/S/ordinary/X'),
'science_namespace':lambda p,r:r.__setitem__('domain','PROPOSED_B3_LIMITED_G7_PILOT_20261019_V1/ordinary/X'),
'real_equivalence':lambda p,r:r['consumption'].__setitem__('real_consumption_equivalence',True),
'artificial_false':lambda p,r:r['consumption'].__setitem__('artificial_only',False),
'wrong_global_mode':lambda p,r:r.__setitem__('global_trial_id',r['global_trial_id'].replace('/S/','/P1/')),
'wrong_run':lambda p,r:r.__setitem__('global_trial_id',r['global_trial_id'].replace('ARTIFICIAL_TEST','other')),
'wrong_layer':lambda p,r:p.__setitem__('trial_id',p['trial_id'].replace('/B/','/A/')),
'wrong_axis':lambda p,r:p.__setitem__('axis','Y'),
'wrong_trial':lambda p,r:r.__setitem__('local_trial_id',r['local_trial_id'][:-1]+'1'),
'wrong_kind':lambda p,r:p.__setitem__('kind','ZERO_C'),
'missing_weight':lambda p,r:p['event'].pop('weight'),
'float_ratio':lambda p,r:p['event'].__setitem__('ratio',float(p['event']['ratio'])),
'wrong_fraction':lambda p,r:p['event'].__setitem__('ratio',p['event']['ratio']+1),
'missing_child':lambda p,r:p['event'].pop('child'),
'bool_child':lambda p,r:p['event'].__setitem__('child',True),
'wrong_raw':lambda p,r:p['event'].__setitem__('raw_word',[1,0]),
'wrong_word':lambda p,r:p['event'].__setitem__('word',[0,0]),
'wrong_proposal':lambda p,r:p['event'].__setitem__('proposal',p['event']['proposal']*2),
'wrong_coefficient':lambda p,r:p['event'].__setitem__('coefficient',p['event']['coefficient']*2),
'wrong_event_digest':lambda p,r:r.__setitem__('event_digest','0'*64),
'wrong_requery':lambda p,r:p['angle'].__setitem__('event_requery_surrogate','0'*64),
'angle_as_real':lambda p,r:p['angle'].__setitem__('not_a_mathematical_enclosure',False),
'angle_value':lambda p,r:p['angle'].__setitem__('actual_angle',0),
'missing_angle':lambda p,r:p.pop('angle'),
'missing_evidence':lambda p,r:p.pop('evidence'),
'wrong_schema':lambda p,r:r['consumption'].__setitem__('schema','unsupported'),
}
class ReplayNegative(unittest.TestCase):pass
for name,mut in MUTATORS.items():
    def test(self,name=name,mut=mut):
        p,r,t,c,i=normal_trial();mut(p,r)
        with self.assertRaises(V.Invalid):B.trial(p,r,t,'ARTIFICIAL_TEST',c,i)
    setattr(ReplayNegative,'test_'+name,test)
for field in ('event_sha256','evidence_sha256','angle_sha256','retention'):
    def test(self,field=field):
        p,r,t,c,i=normal_trial(index=16,phase='profile');p[field]='MUTATED'
        with self.assertRaises(V.Invalid):B.trial(p,r,t,'ARTIFICIAL_TEST',c,i)
    setattr(ReplayNegative,'test_profile_'+field,test)

class Static(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.inv=S.inventory(P/'fixed_reference'/SC);cls.pub=K.read(Q/'scientific_api_inventory.json');cls.mapping=K.read(Q/'api_binding_manifest_candidate.json')
    def test_complete_independent_inventory(self):self.assertEqual(S.compare(self.inv,self.pub)['entries'],201)
    def test_static_mapping(self):self.assertEqual(S.verify_mapping(P/'fixed_reference'/SC,self.mapping)['status'],'PASS_STATIC_MAPPING_ONLY')
    def test_owned_return_excludes_nested(self):
        rows=self.inv['modules'][S.SOURCE+'g6_w2c_b3_driver.py']['entries'];r=next(x for x in rows if x['name']=='run_pair');self.assertLess(len(r['owned_returns']),len(r['return_syntax']))
    def test_no_producer_science_import_in_checker(self):
        forbidden={'g7_generator','g7_provider','g6_f3_generator','g6_f3_precision','g6_f3_contract','return_aggregation','g6_w2c_b3_api_binding_v1','g6_w2c_b3_parallel_scheduler','g6_w2c_b3_driver'}
        self.assertFalse([n for n in sys.modules if n.rsplit('.',1)[-1] in forbidden or '.g6_w2c_b3_api_binding_v1' in n])
for field,value in [('signature','wrong'),('name','nonexistent'),('source_sha256','0'*64),('kind','ClassDef')]:
    def test(self,field=field,value=value):
        x=copy.deepcopy(self.pub);row=next(r for r in x['modules'][S.SOURCE+'g7_generator.py']['entries'] if 'signature' in r);row[field]=value
        with self.assertRaises((V.Invalid,KeyError)):S.compare(self.inv,x)
    setattr(Static,'test_reject_'+field,test)
for arm in V.ARMS:
    def test(self,arm=arm):
        m=copy.deepcopy(self.mapping);m['arms'][arm]['layer_B']='g6_f3_generator.prepare -> CanonicalFacade.sample'
        with self.assertRaises(V.Invalid):S.verify_mapping(P/'fixed_reference'/SC,m)
    setattr(Static,'test_reject_mapping_'+arm,test)

class Contracts(unittest.TestCase):
    def test_pinned_candidate(self):K.candidate(C,C,TREE)
    def test_authority_snapshot_bound_but_not_atomic_history(self):
        r=K.authority(RAW/'port_cases/sealed_small',C,loc('sealed_small'));self.assertFalse(r['complete_atomic_history_proven'])
for key in ['schema','run_ready','content_adopted','contract_grants_execution','science_release','scientific_adoption','retry','resume','python','source_pins','scientific_reference','source_release_candidate','static_api_inventory_digest','artificial_seed_namespace','production_dispatch','authority_policy','api_binding','monitoring']:
    def test(self,key=key):
        v=copy.deepcopy(C);v[key]='MUTATED'
        with self.assertRaises(V.Invalid):K.candidate(v,C,TREE)
    setattr(Contracts,'test_reject_'+key,test)

class Cases(unittest.TestCase):pass
for name,calls,status in [('sealed_small',1536,'VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE'),('sealed_n256',16896,'VERIFIED_ARTIFICIAL_PORT_GRID_COMPLETE'),('S_generator',224,'VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE'),('P1_angle',992,'VERIFIED_ARTIFICIAL_PORT_TECHNICAL_FAILURE')]:
    def test(self,name=name,calls=calls,status=status):
        r=A.verify(RAW/'port_cases'/name,C,TREE,trusted=C,execution_root=loc(name));self.assertEqual(r['status'],status,r['issues']);self.assertEqual(r['calls'],calls);RESULTS.append(dict(case=name,verification=r))
    setattr(Cases,'test_'+name,test)

SAVED={
'report_wrong_source':('report',lambda x:x.__setitem__('measured_SOURCE','0'*40)),
'report_wrong_seed_digest':('report',lambda x:x.__setitem__('candidate_digest','0'*64)),
'report_false_STOP':('report',lambda x:x.__setitem__('mandatory_STOP',False)),
'report_science_GO':('report',lambda x:x.__setitem__('scientific_GO',True)),
'report_false_integration':('report',lambda x:x.__setitem__('global_bundle_equivalence',True)),
'report_wrong_budget':('report',lambda x:x.__setitem__('planned',1537)),
'report_bool_n':('report',lambda x:x.__setitem__('n',True)),
'report_duplicate_pair':('report',lambda x:x['rows'].__setitem__(1,copy.deepcopy(x['rows'][0]))),
'report_bool_cpu':('report',lambda x:x['outer'].__setitem__('parent_cpu_ns',True)),
'wrong_source_release':('authority',lambda x:x.__setitem__('source_release_candidate','0'*64)),
'wrong_runtime':('authority',lambda x:x.__setitem__('runtime_hash','0'*64)),
'wrong_contract':('authority',lambda x:x.__setitem__('contract_hash','0'*64)),
'wrong_scope':('authority',lambda x:x.__setitem__('global_scope','REAL_SCOPE')),
'wrong_claim':('authority',lambda x:x.__setitem__('global_claim','0'*64)),
'wrong_budget':('authority',lambda x:x.__setitem__('call_budget',1536)),
'bool_budget':('authority',lambda x:x.__setitem__('call_budget',True)),
'retry':('authority',lambda x:x.__setitem__('retry',1)),
'resume':('authority',lambda x:x.__setitem__('resume',True)),
'fake_to_real':('authority',lambda x:x.__setitem__('real_authority',True)),
'fake_real_grant':('authority',lambda x:x.__setitem__('real_grant','fake')),
'wrong_port_domain':('side',lambda x:x['records'][0].__setitem__('domain','other')),
'bool_consume':('side',lambda x:x['records'][0]['consumption'].__setitem__('bits_consumed',True)),
'duplicate_trial':('side',lambda x:x['records'].append(copy.deepcopy(x['records'][0]))),
'wrong_event_hash':('side',lambda x:x['records'][0].__setitem__('event_digest','0'*64)),
'missing_record':('side',lambda x:x['records'].pop(0)),
'wrong_global_mode':('side',lambda x:x['records'][0].__setitem__('global_trial_id','wrong')),
'snapshot_refund':('snapshot',lambda x:x['claim'].__setitem__(-1,0)),
'snapshot_unstop':('snapshot',lambda x:x['claim'].__setitem__(-2,'SPENT')),
'caller_bad_hash':('caller',lambda x:x.__setitem__('report_sha256','0'*64)),
'caller_not_closed':('caller',lambda x:x.__setitem__('report_write_flush_close_observed',False)),
}
class SavedNegative(unittest.TestCase):pass
for name,(target,mut) in SAVED.items():
    def test(self,name=name,target=target,mut=mut):
        base=RAW/'port_cases/sealed_small';d=OUT/name;shutil.copytree(base,d,copy_function=os.link)
        paths=dict(report='port_report.json',authority='authority_binding.json',side='ports/S__correctness_baseline_ordinary_A.port.json',snapshot='fake_snapshot.json',caller='port_caller.json')
        path=d/paths[target];x=K.read(path);mut(x);write_json(path,x)
        # Report mutation gets a valid external report hash; this is not just checksum rejection.
        if target=='report':
            caller=K.read(d/'port_caller.json');caller['report_sha256']=V.file_digest(path);write_json(d/'port_caller.json',caller)
        r=A.verify(d,C,TREE,trusted=C,execution_root=loc('sealed_small'));self.assertEqual(r['status'],'INTEGRITY_MISMATCH',r['issues']);RESULTS.append(dict(mutation=name,target=paths[target],status=r['status'],issues=r['issues'],base_case='sealed_small'))
    setattr(SavedNegative,'test_'+name,test)
for name,path in [('missing_caller','port_caller.json'),('missing_report','port_report.json'),('missing_raw','ports/S__correctness_baseline_ordinary_A.jsonl'),('missing_claim','FAKE_ONLY.sqlite3'),('missing_sidecar','ports/S__correctness_baseline_ordinary_A.port.json')]:
    def test(self,name=name,path=path):
        d=OUT/name;shutil.copytree(RAW/'port_cases/sealed_small',d,copy_function=os.link);(d/path).unlink()
        r=A.verify(d,C,TREE,trusted=C,execution_root=loc('sealed_small'));self.assertIn(r['status'],('INCOMPLETE_EVIDENCE','INTEGRITY_MISMATCH'));RESULTS.append(dict(mutation=name,target=path,status=r['status'],issues=r['issues'],base_case='sealed_small'))
    setattr(SavedNegative,'test_'+name,test)
