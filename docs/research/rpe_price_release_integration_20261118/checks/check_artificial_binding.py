import json,pathlib,sys,subprocess
B=pathlib.Path('/home/abe/Project/Partially Randomized Trotter/.rpe-central');ROOT=B/'releases/e917165ba-price-fleet';D=B/'acceptance/price-fleet-v2-e917165ba';F=B/'fleet/romutail-price-fleet-e917165ba'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_rpe_q3_relay import Relay
r=Relay(D/'RELAY.json')
code='''import json,pathlib,sys,subprocess,copy
v=json.load(sys.stdin);sys.path[:0]=[v['root']+'/src',v['root']+'/scripts']
from trottertracks.rpe_release_deployment import read,validate_assignment_release
m=read(v['manifest']);l=read(v['layout']);p=read(v['profile']);out=[]
for f in sorted(pathlib.Path(v['spool']).glob('*/*/ASSIGNMENT.json')):
 a=read(f);good=validate_assignment_release(m,l,a,p);bad=copy.deepcopy(a);bad['required_release']['manifest_sha256']='WRONG'
 try:validate_assignment_release(m,l,bad,p)
 except ValueError:rejected=True
 else:raise RuntimeError('Wrong release was accepted')
 out.append(dict(job_id=a['job_id'],stage=good['stage'],source=good['scientific_source'],required_release=a['required_release'],wrong_release_rejected=rejected))
assert out
print(json.dumps(dict(status='PASS',host=v['host'],optimizer_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=v['root'],text=True).strip(),assignments=out,scientific_calls=0)))'''
rows=[]
for h in ['local','server2','local0-wsl']:
 l=json.loads((F/'hosts'/f'{h}.json').read_text());dep=json.loads((D/'DEPLOYMENT.json').read_text())['hosts'][h]
 v=dict(host=h,root=l['component_roots']['optimizer'],manifest=l['manifest_path'],layout=l['layout_path'],profile=str(D/'PROFILE_local.json')if h=='local'else dep['directory']+'/PROFILE.json',spool=dep['directory']+'/spool')
 if h=='local':
  p=subprocess.run([sys.executable,'-B','-c',code],input=json.dumps(v),text=True,capture_output=True,check=True);out=json.loads(p.stdout)
 else:
  w=next(w for w in r.config['workers']if w['host_id']==h);out,_=r.remote_command(w,[w['python'],'-B','-c',code],'binding-proof',v)
 rows.append(out)
proof=dict(status='PASS',release='romutail-price-fleet-e917165ba',hosts=rows,scientific_calls=0)
(D/'RELEASE_BINDING.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
