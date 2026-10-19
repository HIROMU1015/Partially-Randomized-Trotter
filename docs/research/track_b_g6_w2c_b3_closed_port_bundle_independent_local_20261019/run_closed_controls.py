import pathlib,json,sys,tempfile,concurrent.futures,multiprocessing as mp,importlib,copy,shutil,os
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--runtime',required=True);ap.add_argument('--new-output',required=True);a=ap.parse_args();T=pathlib.Path(a.new_output).resolve();ROOT=pathlib.Path(a.runtime).resolve();sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_port_bundle_v1 import contract as K,bridge as B,adapter as F
from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import ports as P

def guard(name):
 P.install_guard()
 try:importlib.import_module('trottertracks.algorithm_codesign.'+name)
 except P.Rejected as e:
  assert str(e).startswith('SCIENTIFIC_IMPORT_BLOCKED_BEFORE_LOAD:') and not P.forbidden_imports();return dict(name=name,blocked_before_load=True)
 raise AssertionError('import guard bypassed')

def main():
 T.mkdir(exist_ok=False)
 P.install_guard();c=K.candidate(ROOT);gates=[]
 for fn in [B.production_entry,B.ProductionFactory(),B.ProductionBits()]:
  for kw in [{},{'run_ready':True,'content_adopted':True,'authority':'FAKE'},{'grant':'FAKE','bits':'ARTIFICIAL_API_BINDING_V1'}]:
   try:fn(**kw)
   except B.Denied as e:gates.append(str(e))
   else:raise AssertionError('science gate opened')
 roots=[]
 def fresh(label):
  root=pathlib.Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-integration-independent-'+label+'-'));roots.append(root);return root,B.FakeStore(root),K.manifest(c,'ARTIFICIAL_ONLY/'+root.name)
 root,store,m=fresh('race')
 def claim(i):
  try:store.claim(m);return True
  except B.Denied:return False
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:claims=list(pool.map(claim,range(32)))
 assert sum(claims)==1;spent=store.snapshot(m);store.stop(m);stopped=store.snapshot(m)
 try:store.claim(m)
 except B.Denied:pass
 else:raise AssertionError('spent reused')
 root2,st2,m2=fresh('ack')
 def unknown(where):assert where=='admission.commit_ack';raise OSError('OWN_FAKE_ACK_UNKNOWN')
 try:st2.claim(m2,unknown)
 except OSError:pass
 ack=st2.snapshot(m2);assert ack['claim'][-2:]==['SPENT',1]
 try:st2.claim(m2)
 except B.Denied:pass
 else:raise AssertionError('unknown refunded')
 st2.stop(m2)
 root3,st3,m3=fresh('preclaim');rejected=[]
 for key,value in [('run_ready',True),('content_adopted',True),('science_release','fake'),('scientific_adoption','fake'),('contract_grants_execution',True),('retry',1),('resume',True),('candidate_binding','0'*64),('seed_namespace','ARTIFICIAL_API_BINDING_V1/S'),('port_variant','REAL'),('approval','FAKE')]:
  bad=copy.deepcopy(m3);bad[key]=value
  try:B.validate(c,bad)
  except (B.Denied,ValueError):rejected.append(key)
  else:raise AssertionError('manifest not rejected:'+key)
 assert st3.snapshot(m3)['claim'] is None
 parent=[guard(n) for n in P.SCIENCE]
 with mp.get_context('spawn').Pool(len(P.SCIENCE)) as pool:workers=pool.map(guard,P.SCIENCE)
 out=dict(gate_rejections=gates,race_attempts=32,winner=sum(claims),losers=32-sum(claims),spent=spent,stopped=stopped,ACK_unknown_spent=ack,reuse_rejected=True,unknown_credit_refunded=False,manifest_rejections=rejected,parent_guard=parent,spawn_guard=workers,forbidden_imports=P.forbidden_imports(),real_authority=False,scientific_dispatches=0,full_atomic_durability_proof=False,roots=[str(p) for p in roots])
 (T/'direct_controls.json').write_text(json.dumps(out,indent=2));D=T/'direct_authority_raw';D.mkdir()
 for i,p in enumerate(roots):shutil.copytree(p,D/str(i))
 print('9 gate rejections; fresh32 race winner1; ACK unknown spent; 11 preclaim rejections; parent/spawn guards12')
if __name__=='__main__':main()
