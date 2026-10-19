"""Stdlib-only pilot scope and immutable authorization. No scientific imports."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('m5_plan', ROOT/'src/trotterlib/h46_m5_plan.py')
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
POLICY = 'M5_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'
LANES = ('H4_P2_DET', 'H4_P2_PR', 'H6_P2_DET', 'H6_P2_PR')
COMPILER = dict(basis_gates=['rz','sx','x','cx'], optimization_level=1,
                seed_transpiler=17, backend=None, target=None, coupling_map=None,
                qiskit_version='1.3.0')
COMPILER_HASH = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
SOURCE_PATHS = ('scripts/h46_m5_contract.py','scripts/h46_m5_native.py',
    'scripts/launch_h46_m5_pilot.py','scripts/prepare_h46_m5_pilot.py',
    'scripts/h46_new2_native.py','scripts/h46_new2_core.py',
    'scripts/prepare_h46_confirmation_native.py','scripts/h46_confirmation_contract.py',
    'src/trotterlib/h46_m5_plan.py','references/h46_m5_best_formula_v1.json')


def require(test, reason):
    if not test: raise ValueError(reason)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def scope(lane):
    require(lane in LANES,'FIXED_FOUR_LANES')
    system,_,kind=lane.split('_');p=core.pilot(system)
    groups=p['deterministic' if kind=='DET' else 'PR']
    return dict(lane=lane,system=system,kind=kind,groups=groups,
        tasks=[dict(id=g['id']+'__'+str(i),group=g,sample=i)
               for g in groups for i in range(1 if kind=='DET' else 2)],
        compile_axes=18 if kind=='DET' else 12,reference_jobs=1 if kind=='DET' else 0,
        T=.8,control='symmetric_directional',native_policy=POLICY,
        compiler=COMPILER,compiler_hash=COMPILER_HASH,numeric_tolerance=1e-9,
        coefficient_optimization=False,shot_accounting=False,ranking=False,
        retry=False,phase3_authorized=False,mandatory_STOP=True)


def seed_value(source,lane,task,outer,piece):
    return int(core.digest(['M5_PILOT_ONLY_V1',source,lane,task,outer,piece])[:16],16)


def registry(source,excluded):
    occupied=set(excluded);result={}
    for lane in LANES:
        result[lane]={}
        for t in scope(lane)['tasks']:
            if t['group']['method']!='PR':continue
            g=t['group'];values={}
            for s in core.schedule(g['formula'],g['q'],g['policy'],g['r_value']):
                key=f"{s['outer']}:{s['piece']}"
                v=seed_value(source,lane,t['id'],s['outer'],s['piece'])
                require(v not in occupied,'KNOWN_OR_NEW_SEED_COLLISION')
                occupied.add(v);values[key]=v
            result[lane][t['id']]=values
    return result


def verify_seals(config):
    require(config['SOURCE']==config['plan']['SOURCE'],'SOURCE_BINDING')
    require(read(config['plan_path'])==config['plan'] and sha(config['plan_path'])==config['plan_sha256'],'PLAN_BYTES')
    require(config['plan']['scope']==scope(config['lane']),'FIXED_SCOPE')
    for e in config['sealed_files']:
        require(Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256'],'SEALED_BYTES:'+e['role'])
    require(core.digest(config['sealed_files'])==config['plan']['closure_digest'],'CLOSURE_DIGEST')
    require(core.digest(config['inputs'])==config['plan']['input_digest'],'INPUT_DIGEST')
    require(core.digest(config['binding'])==config['plan']['binding_digest'],'BINDING_DIGEST')
    require(core.digest(config['runtime'])==config['plan']['runtime_digest'],'RUNTIME_DIGEST')
    require(config['binding']['control']=='symmetric_directional' and config['binding']['compiler']==COMPILER_HASH,'MAIN_CONTROL_COMPILER')
    require(config['native_policy']==POLICY,'NATIVE_POLICY')
    require(sha(config['registry_path'])==config['plan']['seed_registry_sha256'],'SEED_REGISTRY_BYTES')
    excluded=read(config['excluded_seed_path'])['values']
    require(len(excluded)==len(set(excluded)) and all(type(x) is int for x in excluded),'SEED_EXCLUSION_SET')
    expected=registry(config['SOURCE'],excluded)
    require(read(config['registry_path'])['lanes']==expected,'SEED_DERIVATION')
    # Check only the files actually used. HEAD/status of other worktrees are irrelevant.
    for relative in SOURCE_PATHS:
        original=subprocess.check_output(['git','show',config['SOURCE']+':'+relative],cwd=ROOT)
        require(original==(ROOT/relative).read_bytes(),'GIT_SOURCE_BYTES:'+relative)


def authorize(config):
    # Fail before file hashing, chemistry imports, output or markers for invalid drafts.
    grant=config.get('grant',{})
    require(config.get('authorized') is True and grant.get('valid') is True
            and isinstance(grant.get('approval_reference'),str) and grant['approval_reference'].strip(),
            'EXPLICIT_PILOT_SAMPLING_COMPILE_APPROVAL_REQUIRED')
    expected=dict(lane=config['lane'],SOURCE=config['SOURCE'],plan_sha256=config['plan_sha256'],
        maximum_axes=scope(config['lane'])['compile_axes'],
        maximum_trajectories=6 if config['lane'].endswith('PR') else 0,
        maximum_reference_jobs=scope(config['lane'])['reference_jobs'],
        retry=False,phase3_authorized=False)
    require(all(grant.get(k)==v for k,v in expected.items()),'GRANT_FIXED_SCOPE')
    require(isinstance(grant.get('grant_id'),str) and grant['grant_id'],'NEW_GRANT_ID_REQUIRED')
    require(config['system']==scope(config['lane'])['system'],'LANE_SYSTEM_IDENTITY')
    for key in ('output','ledger','marker'):
        require(config[key]==grant.get(key),'GRANT_PATH_BINDING')
        p=Path(config[key]).resolve()
        require(p.is_relative_to(Path(config['run_root']).resolve()),'OUTPUT_MUST_BE_NEW_RUN_SCOPE')
        require(not p.is_relative_to(ROOT) and not ROOT.is_relative_to(p),'SOURCE_OUTPUT_OVERLAP')
    ref=Path(config['reference_root']).resolve()
    require(ref==Path(config['run_root']).resolve()/(config['system']+'_reference'),'REFERENCE_PATH_BINDING')
    verify_seals(config)


def consumed(config):
    receipt=read(config['marker'])
    require(receipt==dict(state='CONSUMED_NO_REFUND_NO_RETRY',grant=config['grant']),'CLAIMED_GRANT_MARKER')


def reserve(config,task,axis,fingerprint):
    require(task in config['plan']['scope']['tasks'] and axis in ('cosine','sine'),'REGISTERED_COMPILE')
    path=Path(config['ledger'])/(task['id']+'__'+axis+'.json')
    save(path,dict(state='ENTERED_NONREFUNDABLE',task=task['id'],axis=axis,
        SOURCE=config['SOURCE'],plan_sha256=config['plan_sha256'],fingerprint=fingerprint))
    return path
