import pathlib,json,sys,copy,hashlib
B=pathlib.Path('/home/abe/Project/Partially Randomized Trotter/.rpe-central');S='95a74e4486131a97bd5235221bf2d8ab9f99f591';RID='romutail-pbs-current-95a74e448'
ROOT=B/'releases/95a74e448-pbs-current';PARENT=B/'acceptance/pbs-current-entry-20261118';D=PARENT/'fixture-v2';FLEET=B/'fleet'/RID
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import check_rpe_pbs as P
from trottertracks.rpe_q3_coordinator import source_manifest
from trottertracks.rpe_release_registry import register

def read(p):return json.loads(pathlib.Path(p).read_text())
def put(p,v):p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
settings=read(B/'deployment'/RID/'PBS_SETTINGS.json')
settings['remote_root']='/home/AbeHiromu/projects/rpe-pbs-20261118/acceptance/pbs-current-entry-20261118'
profile=settings['workers'][0]['profile'];profile['prefix_cache_directory']=settings['remote_root']+'/fixture-cache/pbs-0/prefix';profile['artifact_cache_directory']=settings['remote_root']+'/fixture-cache/pbs-0/artifacts'
settings['resources']=dict(queue='workq',ncpus=2,mem='4gb',walltime='00:04:00')
P.prepare(D,ROOT,settings)
spec=read(D/'ROUND.json');production=read(B/'deployment'/RID/'ROUND.default.json')['executor']['distributed']
profile=copy.deepcopy(production['local_worker']);profile.update(work_dir=str(D/'local/spool'),artifact_cache_directory=str(D/'local/artifacts'),prefix_cache_directory=str(D/'local/prefix'),slots=1,slot_policy=dict(mode='auto',initial_slots=1,max_slots=1))
dist=spec['executor']['distributed'];dist.update(computation_source='28d976b99cdf8329adcc1b79327e329e947ef774',computation_root=profile['root'],worker_script=production['worker_script'],local_worker=profile,local_deployment_layout=str(FLEET/'hosts/local.json'))
dist['host_stage_capabilities']={'local':['signal','audit'],'pbs-0':['q3','compile']}
dist['preparation_profiles']={'local':profile,'pbs-0':settings['workers'][0]['profile']}
spec['artificial_preparation_source']=source_manifest(B/'releases/d887abdca','d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2')
spec['price_validation']=dict(store=str(PARENT/'price-validation/models.sqlite3'),source='a18d5a04df5ec8c789f274730bff87a2c0a7cc89',publish_shared_models=False)
put(D/'ROUND.json',spec)
request=read(PARENT/'SCOPE.json');request['scope']['stop']['max_cycles']=3;put(PARENT/'SCOPE.json',request)
service=dict(code_root=str(ROOT),code_source=S,computation_source=dist['computation_source'],problem=str(D/'PROBLEM.json'),round=str(D/'ROUND.json'),state_directory=str(PARENT/'gateway-v2/state'),entry='schedule',python='/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python',artificial_controller_directory=str(D),release_manifest=str(FLEET/'RELEASE_COMPONENTS.json'),host_layouts={h:str(FLEET/'hosts'/f'{h}.json')for h in ['local','server2','server1','local0-wsl','pbs-0']})
service['fixed_settings']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [D/'PROBLEM.json',D/'ROUND.json',FLEET/'RELEASE_COMPONENTS.json',*(FLEET/'hosts').glob('*.json')]}
put(PARENT/'gateway-v2/SERVICE.release.json',service)
register(PARENT/'gateway-v2/RELEASES.json',RID,PARENT/'gateway-v2/SERVICE.release.json',activate=True,evidence='ISOLATED_ARTIFICIAL_NORMAL_ENTRY_ONLY')
put(PARENT/'gateway-v2/SERVICE.json',dict(release_registry=str(PARENT/'gateway-v2/RELEASES.json'),state_directory=service['state_directory']))
put(PARENT/'CLIENT.local.json',dict(transport='local',python=service['python'],remote_entry=str(ROOT/'tools/local0-search/remote.py'),server_config=str(PARENT/'gateway-v2/SERVICE.json'),host_id='local',require_components=True))
print(json.dumps(dict(status='PREPARED_NOT_STARTED',directory=str(D),source=S,release=RID,artificial_price_store=spec['price_validation']['store'],resources=settings['resources'],gateway_python=settings['gateway_python'])))
