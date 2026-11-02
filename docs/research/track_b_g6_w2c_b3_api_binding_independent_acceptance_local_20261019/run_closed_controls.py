import sys,pathlib,json,tempfile,copy,concurrent.futures,multiprocessing as mp,importlib
import argparse
p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--new-output',required=True);a=p.parse_args()
ROOT=pathlib.Path(a.runtime).resolve();T=pathlib.Path(a.new_output).resolve();sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S,ports as P,authority as A

def guard_probe(name):
 P.install_guard();before=set(sys.modules)
 try:importlib.import_module('trottertracks.algorithm_codesign.'+name)
 except S.Rejected as e:
  assert str(e).startswith('SCIENTIFIC_IMPORT_BLOCKED_BEFORE_LOAD:') and not P.forbidden_imports()
  return dict(name=name,blocked_before_load=True,science_modules_added=sorted(n for n in set(sys.modules)-before if n.rsplit('.',1)[-1] in S.SCIENCE))
 raise AssertionError('science import not blocked')

def main():
 T.mkdir(exist_ok=False)
 P.install_guard();c=S.candidate(ROOT);preflight=S.preflight(c,ROOT);observations=[]
 for fn in (S.production_entry,S.real_bits,S.real_authority):
  for request in ({},{'run_ready':True,'grant':'FAKE_APPROVAL','real_authority':True},{'science_release':'fake','content_adopted':True}):
   try:fn(**request)
   except S.Rejected as e:observations.append(dict(entry=fn.__name__,request=request,rejected=str(e)))
   else:raise AssertionError('gate opened')
 roots=[]
 def authority(label):
  root=pathlib.Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-independent-'+label+'-'));roots.append(str(root));return A.ArtificialAuthority(c,root)
 au=authority('race');b=au.expected;approval=A.approval(b)
 def attempt(i):
  try:au.claim(b,approval);return dict(contender=i,winner=True)
  except RuntimeError as e:return dict(contender=i,winner=False,reason=str(e))
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:race=list(pool.map(attempt,range(32)))
 assert sum(r['winner'] for r in race)==1
 after_claim=au.snapshot();au.stop(b);after_stop=au.snapshot()
 try:au.claim(b,approval)
 except RuntimeError as e:double_rejection=str(e)
 else:raise AssertionError('claim reused')
 ack=authority('ack');bb=ack.expected
 def fault(point):assert point=='admission.commit_ack';raise OSError('OWN_FAKE_COMMIT_ACK_UNKNOWN')
 try:ack.claim(bb,A.approval(bb),fault=fault)
 except OSError:pass
 ack_spent=ack.snapshot();assert ack_spent['claim'][-2:]==['SPENT',1]
 try:ack.claim(bb,A.approval(bb))
 except RuntimeError as e:ack_rejection=str(e)
 else:raise AssertionError('ACK spent refunded')
 ack.stop(bb);ack_stopped=ack.snapshot()
 binding=authority('binding');binding_reject=[]
 for key in binding.expected:
  if key=='schema':v='unsupported'
  elif type(binding.expected[key])is bool:v=not binding.expected[key]
  elif type(binding.expected[key])is int:v=binding.expected[key]+1
  else:v='MUTATED'
  d=copy.deepcopy(binding.expected);d[key]=v
  try:binding.claim(d,A.approval(d))
  except S.Rejected as e:binding_reject.append(dict(field=key,rejected=str(e)))
  else:raise AssertionError('binding splice accepted:'+key)
 assert binding.snapshot()['claim'] is None
 parents=[guard_probe(n) for n in S.SCIENCE]
 with mp.get_context('spawn').Pool(len(S.SCIENCE)) as pool:workers=pool.map(guard_probe,S.SCIENCE)
 out=dict(schema='b3_local_independent_direct_closed_controls_v1',candidate=c,preflight=preflight,gate=observations,race=race,after_claim=after_claim,after_stop=after_stop,double_claim_rejection=double_rejection,ack_spent=ack_spent,ack_rejection=ack_rejection,ack_stopped=ack_stopped,binding_rejections=binding_reject,parent_guard=parents,spawn_guard=workers,forbidden_imports=P.forbidden_imports(),fresh_roots=roots,snapshot_not_complete_atomic_history_proof=True,production_dispatches=0,science_imports=0)
 (T/'direct_closed_controls.json').write_text(json.dumps(out,indent=2))
 for i,root in enumerate(roots):
  import shutil
  dest=T/'direct_authority_raw'/str(i);shutil.copytree(root,dest)
 print('closed9, race32 winner1, binding',len(binding_reject),'parent/spawn guard12, ACK spent confirmed')
if __name__=='__main__':main()
