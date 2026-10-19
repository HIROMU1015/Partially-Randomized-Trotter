"""Independent saved-byte mutations; never imports producer or science modules."""
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,os,shutil,sqlite3,tempfile,unittest,ast
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_adapter_v1 as A
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_identity_v1 as I
from trottertracks.algorithm_codesign import g6_w2c_b3_bundle_independent_trace_v1 as T
ROOT=Path(__file__).resolve().parents[3]
PACKET=ROOT/'docs/research/track_b_g6_w2c_b3_global_bundle_independent_acceptance_local_20261019'
TREE=PACKET/'fixed_reference/bridge'
RAW=Path(os.environ['B3_BUNDLE_RAW_ROOT'])
FRESH=Path(os.environ['B3_BUNDLE_FRESH_ROOT'])
RESULTS=[]

def encoded(v):return json.dumps(v,separators=(',',':'),ensure_ascii=True,default=lambda x:{'exact':A.V.canonical(x)}).encode()+b'\n'
def inventory(p):return {str(x.relative_to(p)):A.V.file_digest(x)for x in p.rglob('*')if x.is_file()}

def write(path,data):
 if path.exists():path.unlink()
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)

class PublicCases(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ref=I.Reference(TREE)

def public_case(name,expected):
 def test(self):
  r=A.verify(RAW/'cases'/name,self.ref);self.assertEqual(r['status'],expected,(r['issues'],r['evidence_gaps']))
  if 'normal_bundle' in name:
   total=16896 if name.endswith('n256')else 1536
   self.assertEqual([r['calls'][k]for k in ('planned','reserved','started','completed')],[total]*4)
   self.assertTrue(r['documents']['full_success_join']);self.assertEqual(r['history']['phase_history'],list(range(6)))
  RESULTS.append(dict(test=self.id(),kind='public_case',case=name,status=r['status'],issues=r['issues']))
 return test
matrix=json.loads((TREE/I.DOC/'artificial_acceptance_matrix.json').read_text())['cases']
for item in matrix:setattr(PublicCases,'test_public_'+item['case'],public_case(item['case'],item['verification_status']))

class PublicOverlays(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.ref=I.Reference(TREE)
def public_overlay(name,expected):
 def test(self):
  r=A.verify(RAW/'restored_mutations'/name,self.ref);self.assertEqual(r['status'],expected,(name,r['issues']));self.assertFalse(r['comparison_eligible'])
  RESULTS.append(dict(test=self.id(),kind='public_overlay_reread',case=name,status=r['status'],issues=r['issues']))
 return test
for item in json.loads((TREE/I.DOC/'negative_mutation_results.json').read_text())['observations']:
 name=item['test_id'].rsplit('.',1)[-1];setattr(PublicOverlays,'test_public_overlay_'+name,public_overlay(name,item['status']))

class FreshCases(unittest.TestCase):pass
def fresh_case(name,expected):
 def test(self):
  case=FRESH/name;candidate=I.read(case/'candidate_input.json');ref=I.Reference(TREE,candidate,allow_runtime_projection=True,verify_local_runtime=os.environ.get('B3_BUNDLE_VERIFY_LIVE_RUNTIME','1')=='1')
  r=A.verify(case,ref);self.assertEqual(r['status'],expected,(r['issues'],r['evidence_gaps']))
  self.assertFalse(r['scientific_GO']);self.assertTrue(r['identity']['runtime_projection'])
  RESULTS.append(dict(test=self.id(),kind='fresh_local_artificial',case=name,status=r['status'],issues=r['issues'],runtime_projection=True))
 return test
for name in ['local_normal_bundle_small','local_normal_bundle_n256','local_S_condition_receipt_close','local_global_receipt_after_write','local_final_write','local_delivery_write','local_caller_delivery_close','local_state_complete_persistence']:
 setattr(FreshCases,'test_'+name,fresh_case(name,'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE'if 'normal_bundle'in name else'INCOMPLETE_EVIDENCE'))

class Mutations(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.ref=I.Reference(TREE);cls.base=RAW/'cases/normal_bundle_small';cls.base_inventory=inventory(cls.base)
 @classmethod
 def tearDownClass(cls):
  if inventory(cls.base)!=cls.base_inventory:raise AssertionError('Original saved raw changed by a mutation')
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='b3-local-bundle-mutation-');self.addCleanup(self.tmp.cleanup)
  self.case=Path(self.tmp.name)/'case';shutil.copytree(self.base,self.case,copy_function=os.link);self.out=self.case/'output'
  self.c=I.read(self.out/'candidate.json');self.m=I.read(self.out/'manifest.json');self.r=I.read(self.out/'final.json')
 def put(self,path,value,*,size=None):
  data=encoded(value)
  if size is not None:
   self.assertLessEqual(len(data),size);data+=b' '*(size-len(data))
  write(path,data)
 def final(self):
  self.put(self.out/'final.json',self.r,size=4*1024*1024)
  d=I.read(self.out/'delivery.json');d['final_sha256']=A.V.file_digest(self.out/'final.json');self.put(self.out/'delivery.json',d,size=4096)
  if (self.case/'external_observation.json').exists():
   o=I.read(self.case/'external_observation.json');o['delivery_sha256']=A.V.file_digest(self.out/'delivery.json');self.put(self.case/'external_observation.json',o,size=4096)
 def document(self,name,change):
  path=self.out/name;v=I.read(path);change(v);size=path.stat().st_size;data=encoded(v)
  self.put(path,v,size=size if len(data)<=size else None)
  if name not in ('final.json','delivery.json'):self.r['physical_files_before_final'][name]=dict(bytes=path.stat().st_size,sha256=A.V.file_digest(path))
  self.final()
 def trace(self,change):
  path=self.out/'operations.jsonl';rows=[json.loads(json.loads(line)['payload'])for line in path.read_bytes().splitlines()];change(rows);previous='0'*64;lines=[]
  for seq,r in enumerate(rows):
   r['seq']=seq;p=json.dumps(r,separators=(',',':'),ensure_ascii=True);h=sha256(previous.encode()+p.encode()).hexdigest();lines.append(encoded(dict(payload=p,previous=previous,sha256=h)));previous=h
  data=b''.join(lines);write(path,data);self.r['evidence_v2']['trace_prefix']=dict(bytes=len(data),rows=len(lines),chain=previous,sha256=sha256(data).hexdigest());self.r['physical_files_before_final'][path.name]=dict(bytes=len(data),sha256=sha256(data).hexdigest());self.final()
 def check(self,expected='INTEGRITY_MISMATCH',code=None):
  self.final();r=A.verify(self.case,self.ref);self.assertEqual(r['status'],expected,(r['issues'],r['evidence_gaps']));self.assertFalse(r['comparison_eligible']);self.assertFalse(r['scientific_GO'])
  if code:self.assertIn(code,','.join(r['issues']))
  item=dict(test=self.id(),kind='new_independent_mutation',base_case=self.base.name,expected=expected,status=r['status'],issues=r['issues'],gaps=r['evidence_gaps'],accepted=False)
  if target:=os.environ.get('B3_BUNDLE_MUTATION_EVIDENCE'):
   dest=Path(target)/self._testMethodName;dest.mkdir(parents=True,exist_ok=False);actual=inventory(self.case);changed=[];deleted=[]
   for name,h in self.base_inventory.items():
    if name not in actual:deleted.append(name)
    elif h!=actual[name]:
     q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.case/name,q);changed.append(dict(path=name,sha256=actual[name],bytes=q.stat().st_size))
   for name,h in actual.items():
    if name not in self.base_inventory:
     q=dest/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.case/name,q);changed.append(dict(path=name,sha256=h,bytes=q.stat().st_size))
   (dest/'overlay.json').write_text(json.dumps(dict(base_case=self.base.name,changed_files=changed,deleted_files=deleted),indent=2)+'\n');item['overlay']=str(dest.name)
  RESULTS.append(item);return r
 def test_pristine(self):self.assertEqual(A.verify(self.case,self.ref)['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE')
 def test_input_read_only(self):before=inventory(self.case);A.verify(self.case,self.ref);self.assertEqual(inventory(self.case),before)
 def test_wrong_SOURCE(self):self.c['source_pins'][next(iter(self.c['source_pins']))]='0'*64;self.put(self.out/'candidate.json',self.c);self.check(code='TRUSTED_CANDIDATE')
 def test_wrong_runtime(self):self.c['python']['executable_sha256']='0'*64;self.put(self.out/'candidate.json',self.c);self.check(code='TRUSTED_CANDIDATE')
 def test_wrong_scientific_reference(self):self.c['scientific_reference']['fixed']['m']=4;self.put(self.out/'candidate.json',self.c);self.check(code='TRUSTED_CANDIDATE')
 def test_wrong_schema(self):self.r['schema']='b3_pilot_storage_result_v2';self.check(code='GLOBAL_RESULT_SCHEMA')
 def test_wrong_claim(self):self.m['global_claim']='0'*64;self.put(self.out/'manifest.json',self.m);self.check(code='GLOBAL_CLAIM')
 def test_wrong_scope(self):self.m['global_scope']='SCIENCE';self.put(self.out/'manifest.json',self.m);self.check(code='ARTIFICIAL_SCOPE')
 def test_mode_seed_domain(self):self.c['proposal']['mode_is_seed_domain']=True;self.put(self.out/'candidate.json',self.c);self.check()
 def test_wrong_prefix_namespace(self):self.m['seed_namespace']='SCIENCE';self.put(self.out/'manifest.json',self.m);self.check(code='COMPARISON_BINDING:seed_namespace')
 def test_retry(self):self.m['retry']=1;self.put(self.out/'manifest.json',self.m);self.check(code='CLOSED_FLAG:retry')
 def test_resume(self):self.m['resume']=True;self.put(self.out/'manifest.json',self.m);self.check(code='CLOSED_FLAG:resume')
 def test_fake_approval(self):self.m['approval']='REAL';self.put(self.out/'manifest.json',self.m);self.check(code='MANIFEST_FIELDS')
 def test_adoption_flip(self):self.c['content_adopted']=True;self.put(self.out/'candidate.json',self.c);self.check()
 def test_call_cap(self):self.m['call_limit']=self.m['planned']+1;self.put(self.out/'manifest.json',self.m);self.check(code='GLOBAL_CALL_CAP')
 def test_boolean_call_cap(self):self.m['call_limit']=True;self.put(self.out/'manifest.json',self.m);self.check(code='GLOBAL_CALL_CAP')
 def test_double_global_budget(self):self.m['byte_limits']['log']*=2;self.put(self.out/'manifest.json',self.m);self.check(code='GLOBAL_BYTE_LIMIT:log')
 def test_trial_id_collision(self):self.trace(lambda rows:next(r for r in rows if r['op']=='RESERVE').update(global_trial_id='duplicate'));self.check(code='GLOBAL_TRIAL_ID')
 def test_global_offset(self):self.trace(lambda rows:next(r for r in rows if r['op']=='RESERVE').update(global_offset=999));self.check(code='GLOBAL_OFFSET')
 def test_boolean_offset(self):self.trace(lambda rows:next(r for r in rows if r['op']=='RESERVE').update(global_offset=False));self.check(code='GLOBAL_OFFSET')
 def test_duplicate_reserve(self):
  def mutate(rows):i=next(i for i,r in enumerate(rows)if r['op']=='RESERVE');rows.insert(i+1,deepcopy(rows[i]))
  self.trace(mutate);self.check(code='RESERVE_PHASE_OR_DUPLICATE')
 def test_duplicate_complete(self):
  def mutate(rows):i=next(i for i,r in enumerate(rows)if r['op']=='COMPLETE');rows.insert(i+1,deepcopy(rows[i]))
  self.trace(mutate);self.check(code='MISSING_OR_DUPLICATE_TERMINAL')
 def test_enter_after_stop(self):
  def mutate(rows):i=next(i for i,r in enumerate(rows)if r['op']=='ENTER');rows.insert(i,dict(seq=0,run_id=self.m['run_id'],op='STOP',condition='S',parent_arrival=1))
  self.trace(mutate);self.check(code='ENTER_AFTER_STOP')
 def test_reserve_after_stop(self):
  def mutate(rows):i=next(i for i,r in enumerate(rows)if r['op']=='RESERVE');rows.insert(i,dict(seq=0,run_id=self.m['run_id'],op='STOP',condition='S',parent_arrival=1))
  self.trace(mutate);self.check(code='RESERVE_AFTER_STOP')
 def test_phase_barrier(self):self.trace(lambda rows:next(r for r in rows if r['op']=='PHASE_BEGIN').update(phase=1));self.check(code='SIX_PHASE_BARRIER')
 def test_S_failure_then_P1(self):self.trace(lambda rows:next(r for r in rows if r['op']=='CONDITION_END').update(complete=False));self.check(code='P1_AFTER_S_FAILURE')
 def test_condition_mode_collision(self):self.trace(lambda rows:next(r for r in rows if r['op']=='RESERVE').update(condition='P1'));self.check(code='TASK_MODE')
 def test_global_counter_refund(self):self.r['reserved']-=1;self.check(code='RAM_GLOBAL_RESERVED')
 def test_started_false(self):self.r['started_observed']-=1;self.check()
 def test_cursor_refund(self):self.r['evidence_v2']['next_indices'][self.r['contract_order'][0]]-=1;self.check(code='CURSOR_RESERVED_SUM')
 def test_current_ticket(self):self.r['evidence_v2']['current_tickets']=[dict(index=0)];self.check(code='FINAL_UNDRAINED_TICKETS')
 def test_RAM_physical_disagreement(self):self.r['evidence_v2']['shadow_sha256']='0'*64;self.check(code='RAM_PHYSICAL_SHA')
 def test_false_COMPLETE_state(self):p=self.out/'trial_states.bin';b=bytearray(p.read_bytes());b[0]=7;write(p,b);self.check(code='COMPLETION_AFTER_UNFINISHED_TRIAL')
 def test_duplicate_pair(self):self.r['pairs'][1]=deepcopy(self.r['pairs'][0]);self.check(code='JOURNAL_EXTRA_PAIR')
 def test_pair_offset(self):self.r['pairs'][0]['task']['offset']+=1;self.check()
 def test_P1_failure_loses_S_raw(self):
  source=RAW/'cases/P1_failure_preserves_S';self.base=source;self.base_inventory=inventory(source);shutil.rmtree(self.case);shutil.copytree(source,self.case,copy_function=os.link);self.r=I.read(self.out/'final.json');(self.out/'S__correctness_baseline_ordinary_A.jsonl').unlink();self.check('INCOMPLETE_EVIDENCE')
 def test_trace_chain(self):p=self.out/'operations.jsonl';b=bytearray(p.read_bytes());b[30]^=1;write(p,b);self.check()
 def test_trusted_prefix_mid_row(self):self.r['evidence_v2']['trace_prefix']['bytes']-=1;self.check(code='TRUSTED_PREFIX_ROW_BOUNDARY')
 def test_trusted_prefix_rows(self):self.r['evidence_v2']['trace_prefix']['rows']-=1;self.check(code='TRUSTED_PREFIX_ROWS')
 def test_trusted_prefix_chain(self):self.r['evidence_v2']['trace_prefix']['chain']='0'*64;self.check(code='TRUSTED_PREFIX_CHAIN')
 def test_trusted_prefix_SHA(self):self.r['evidence_v2']['trace_prefix']['sha256']='0'*64;self.check(code='TRUSTED_PREFIX_SHA')
 def test_partial_trace_tail(self):p=self.out/'operations.jsonl';write(p,p.read_bytes()[:-1]);self.check()
 def test_unknown_credit_refund(self):self.r['output']['unknown_credit']['result']=10;self.check(code='UNKNOWN_CREDIT_NONREFUND')
 def test_ACK_exceeds_reserved(self):self.r['output']['observed']['log']+=1;self.check(code='GLOBAL_PHYSICAL_RESERVED_CAP')
 def test_global_cap_doubled_in_report(self):self.r['output']['limits']['log']*=2;self.check(code='ONE_GLOBAL_BYTE_LIMIT')
 def test_trace_self_double_charge(self):self.r['output']['reserved']['log']+=self.r['evidence_v2']['trace_prefix']['bytes'];self.r['output']['unknown_credit']['log']=self.r['output']['reserved']['log']-self.r['output']['observed']['log'];self.check(code='TRACE_POST_CLOSE_RESERVATION')
 def test_marker_wrong_receipt(self):self.document('marker.json',lambda d:d.update(receipt_sha256='0'*64));self.check(code='MARKER_RECEIPT_HASH')
 def test_marker_wrong_run(self):self.document('marker.json',lambda d:d.update(run_id='0'*64));self.check(code='POST_CLOSE_RUN:marker')
 def test_receipt_preclose(self):self.document('receipt.json',lambda d:d.update(storage_closed_before_receipt=0));self.check(code='RECEIPT_POST_CLOSE')
 def test_diagnostic_false_success(self):self.document('diagnostic.json',lambda d:d.update(final_acceptance=0));self.check(code='DIAGNOSTIC_IS_NOT_ACCEPTANCE')
 def test_receipt_counter(self):self.document('receipt.json',lambda d:d.update(completed=1535));self.check(code='GLOBAL_RECEIPT_COUNTER')
 def test_summary_prefix(self):self.document('summary.json',lambda d:d['trace_prefix'].update(chain='0'*64));self.check(code='TRUSTED_PREFIX_CHAIN')
 def test_summary_condition(self):self.document('summary.json',lambda d:d['conditions'][0].update(reserved=767));self.check(code='SUMMARY_CONDITIONS')
 def test_missing_caller(self):(self.case/'external_observation.json').unlink();self.check('INCOMPLETE_EVIDENCE')
 def test_missing_receipt(self):(self.out/'receipt.json').unlink();self.check('INCOMPLETE_EVIDENCE')
 def test_delivery_hash(self):self.final();d=I.read(self.out/'delivery.json');d['final_sha256']='0'*64;self.put(self.out/'delivery.json',d,size=4096);r=A.verify(self.case,self.ref);self.assertEqual(r['status'],'INTEGRITY_MISMATCH');self.assertIn('DELIVERY_FINAL_HASH',r['issues'])
 def test_caller_return_failure(self):
  self.final();o=I.read(self.case/'external_observation.json');o['bridge_returned_normally']=False;o['exception']='ARTIFICIAL_CALLER_FAILURE';self.put(self.case/'external_observation.json',o,size=4096);r=A.verify(self.case,self.ref);self.assertEqual(r['status'],'INCOMPLETE_EVIDENCE');self.assertFalse(r['comparison_eligible'])
 def test_CPU_double_add(self):self.r['total_cpu_ns']+=self.r['worker_cpu_ns'];self.check(code='CPU_DOUBLE_ADD')
 def test_parent_CPU_zero(self):self.r['host_outer']['parent_cpu_ns']=0;self.check(code='HOST_OUTER_BOUNDARY')
 def test_wall_sum_false(self):self.r['host_outer']['wall_ns']=0;self.check(code='HOST_OUTER_BOUNDARY')
 def test_condition_CPU_double_add(self):self.r['condition_worker_cpu_ns']+=1;self.check(code='CONDITION_WORKER_CPU_SUM')
 def test_RPC_count_is_not_trial_count(self):self.r['parent_messages']['ENTER']=1;self.check(code='COALESCED_ENTER_TRIAL_COUNT')
 def test_SQLite_binding_mutation(self):
  p=self.case/'FAKE_ONLY.sqlite3';b=p.read_bytes();write(p,b)
  with sqlite3.connect(p)as db:db.execute("UPDATE claims SET binding=?",('0'*64,))
  self.check(code='GLOBAL_CLAIM_BINDING')
 def test_SQLite_double_claim(self):
  p=self.case/'FAKE_ONLY.sqlite3';write(p,p.read_bytes())
  with sqlite3.connect(p)as db:db.execute("INSERT INTO claims VALUES('ARTIFICIAL_ONLY/EXTRA','EXTRA','EXTRA','SPENT',1)")
  self.check(code='ONE_GLOBAL_CLAIM')
 def test_source_independence(self):
  for name in ['identity','trace','storage','cost','adapter']:
   p=ROOT/'src/trottertracks/algorithm_codesign'/('g6_w2c_b3_bundle_independent_'+name+'_v1.py');tree=ast.parse(p.read_text())
   for n in ast.walk(tree):
    if isinstance(n,(ast.Import,ast.ImportFrom)):
     names=([x.name for x in n.names]if isinstance(n,ast.Import)else[n.module or'']+[x.name for x in n.names])
     self.assertFalse(any(x in names for x in ['g6_w2c_b3_bundle_bridge_v1','g6_w2c_b3_bundle_offline_v1','g6_w2c_b3_bundle_contract_v1','g6_w2c_b3_parallel_scheduler','g7_generator','g6_f3_generator']))

class StaticContract(unittest.TestCase):
 def test_16896(self):tasks,total=I.plan(256);self.assertEqual(total,16896);self.assertEqual(len(tasks),48);self.assertEqual(sum(2*t['n']for t in tasks if t['condition']=='S'),8448)
 def test_modes_have_disjoint_offsets(self):tasks,total=I.plan(256);self.assertEqual(tasks[24]['offset'],8448);self.assertEqual(tasks[-1]['offset']+2*tasks[-1]['n'],total)
 def test_bool_n_rejected(self):self.assertRaises(A.V.Invalid,I.plan,True)
 def test_seed_domain_static(self):
  text=(TREE/'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_fixture_v1.py').read_text();tree=ast.parse(text);init=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='__init__');names={n.id for n in ast.walk(init)if isinstance(n,ast.Name)};self.assertFalse({'mode','layer','phase','pass_name'}&names);self.assertIn('arm',names);self.assertIn('axis',names)
 def test_production_entry_is_closed(self):
  p=TREE/'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_bridge_v1.py';nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in ('production_entry','ProductionFactory','ProductionBits')]
  class Denied(RuntimeError):pass
  scope=dict(Denied=Denied);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
  for call in (scope['production_entry'],scope['ProductionFactory'](),scope['ProductionBits']()):self.assertRaises(Denied,call,adopted=True,run_ready=True)
 def test_artificial_API_accepts_no_external_factory(self):
  p=TREE/'src/trottertracks/algorithm_codesign/g6_w2c_b3_bundle_bridge_v1.py';f=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute_artificial');names={a.arg for a in f.args.args+f.args.kwonlyargs};self.assertEqual(names,{'c','m','root','store','storage_fault'});self.assertIsNone(f.args.kwarg)


 def test_archived_runtime_declaration_is_not_current_byte_proof(self):
  case=FRESH/'local_normal_bundle_small';c=I.read(case/'candidate_input.json');m=I.read(case/'manifest_input.json')
  ref=I.Reference(TREE,c,allow_runtime_projection=True,verify_local_runtime=False);r=ref.identity(c,m)
  self.assertFalse(r['local_runtime_bytes_proven']);self.assertFalse(r['historical_runtime_bytes_proven'])
 def test_live_runtime_bytes_are_required_when_requested(self):
  c=I.read(FRESH/'local_normal_bundle_small/candidate_input.json');c['python']['executable_sha256']='0'*64
  self.assertRaises(A.V.Invalid,I.Reference,TREE,c,allow_runtime_projection=True,verify_local_runtime=True)

class CLI(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='b3-global-cli-test-');self.addCleanup(self.temp.cleanup);self.tmp=Path(self.temp.name)
 def test_complete_export(self):
  from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import main
  out=self.tmp/'report';rc=main(['--case',str(RAW/'cases/normal_bundle_small'),'--fixed-tree',str(TREE),'--new-output',str(out)])
  self.assertEqual(rc,0);self.assertEqual(len((out/'comparison.csv').read_text().splitlines()),3);self.assertTrue(I.read(out/'acceptance.json')['mandatory_STOP'])
 def test_reject_existing_output(self):
  from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import main
  self.assertRaises(A.V.Invalid,main,['--case',str(RAW/'cases/normal_bundle_small'),'--fixed-tree',str(TREE),'--new-output',str(self.tmp)])
 def test_reject_input_overlap(self):
  from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import main
  self.assertRaises(A.V.Invalid,main,['--case',str(RAW/'cases/normal_bundle_small'),'--fixed-tree',str(TREE),'--new-output',str(RAW/'cases/normal_bundle_small/unsafe')])
 def test_failure_not_in_comparison(self):
  from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import main
  out=self.tmp/'report';rc=main(['--case',str(RAW/'cases/P1_failure_preserves_S'),'--fixed-tree',str(TREE),'--new-output',str(out)])
  self.assertEqual(rc,0);self.assertEqual(len((out/'comparison.csv').read_text().splitlines()),1)
 def test_incomplete_not_in_comparison(self):
  from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_cli_v1 import main
  out=self.tmp/'report';rc=main(['--case',str(RAW/'cases/parent_crash'),'--fixed-tree',str(TREE),'--new-output',str(out)])
  self.assertEqual(rc,2);self.assertEqual(len((out/'comparison.csv').read_text().splitlines()),1);checks=I.read(out/'acceptance.json')['checks'];self.assertTrue(all(c['formal_adoption_required']for c in checks))

class Memory(unittest.TestCase):
 def test_streaming_peak_across_n16_n256(self):
  import subprocess,sys
  code="""import sys,json,resource,pathlib
sys.dont_write_bytecode=True;sys.path.insert(0,sys.argv[1])
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_adapter_v1 import verify
from trottertracks.algorithm_codesign.g6_w2c_b3_bundle_independent_identity_v1 import Reference
r=verify(pathlib.Path(sys.argv[3]),Reference(sys.argv[2]));print(json.dumps(dict(status=r['status'],HWM_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,trace_bytes=r['trace']['bytes'])))
"""
  results=[]
  for name in ('normal_bundle_small','normal_bundle_n256'):
   p=subprocess.run([sys.executable,'-c',code,str(ROOT/'src'),str(TREE),str(RAW/'cases'/name)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
   r=json.loads(p.stdout);self.assertEqual(r['status'],'VERIFIED_ARTIFICIAL_BUNDLE_COMPLETE');results.append(r)
  self.assertGreater(results[1]['trace_bytes'],10*results[0]['trace_bytes'])
  self.assertLess(results[1]['HWM_bytes'],results[0]['HWM_bytes']+32*1024*1024)
  if target:=os.environ.get('B3_BUNDLE_MEMORY_REPORT'):Path(target).write_text(json.dumps(dict(results=results,method='Two fresh processes; measured HWM, no execution memory cap',O48_cursor_state=True,all_trial_ID_set=False),indent=2)+'\n')

if __name__=='__main__':unittest.main()
