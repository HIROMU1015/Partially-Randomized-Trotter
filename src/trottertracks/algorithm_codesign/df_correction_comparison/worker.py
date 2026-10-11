"""Authorized children only. Never launch this during preparation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
from .launch import ART, RESULT, verify_child, sha
from .guard import exclusive_json


def claim_job(root,job):
    out=root/RESULT
    if (out/'terminal_status.json').exists():raise PermissionError('Terminal STOP already saved')
    allowed={'stage','output','marker_sha256'}
    stage=job['stage']
    if stage=='CD':label='CD'
    elif stage=='E_inputs':label='E-inputs'
    elif stage=='native':
        allowed|={'r','frames'}
        if job['r'] not in (3,6):raise PermissionError('Unregistered native fixture')
        label='E-r'+str(job['r'])
        inputs=json.loads((out/'E-inputs.json').read_text())
        if job['frames']!=inputs[str(job['r'])]:raise PermissionError('Native input identity mismatch')
    elif stage=='synthesis':
        allowed.add('angle')
        inventory=json.loads((out/'synthesis_inventory.json').read_text())
        matches=[i for i,a in enumerate(inventory['unique_keys']) if a==job['angle']]
        if len(matches)!=1:raise PermissionError('Synthesis key not in frozen inventory')
        label='T-'+str(matches[0])
    else:raise PermissionError('Unregistered worker stage')
    if set(job)!=allowed or Path(job['output']).resolve()!=(out/(label+'.json')).resolve():
        raise PermissionError('Worker payload/output path override')
    exclusive_json(out/(label+'.started.json'),{'label':label,'job_sha256':hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),'retries':0})


def dump(path, data):
    # New-run prefix only. Old evidence/source/authorization paths never passed.
    p = Path(path)
    temporary=p.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(data,stream,default=str,indent=2);stream.write('\n')
    temporary.replace(p)


def runtime_check(contract, stage):
    section=contract['runtimes']['synthesis' if stage=='synthesis' else 'numeric']
    if sha(Path(sys.executable).resolve())!=section['executable_sha256']:
        raise PermissionError('Runtime executable hash changed')
    for name,version in section['packages'].items():
        if importlib.metadata.version(name)!=version:raise PermissionError('Package version changed: '+name)
    # Read-only code/binary identity, with a frozen per-runtime selection policy.
    for name,expected in section['source_trees'].items():
        dist=importlib.metadata.distribution(name);h=hashlib.sha256()
        shared=section['tree_policy']=='python_and_shared_libraries'
        for rel in sorted(str(p) for p in dist.files if str(p).endswith('.py') or (shared and '.so' in str(p))):
            h.update(rel.encode()+b'\0');h.update(Path(dist.locate_file(rel)).read_bytes()+b'\0')
        if h.hexdigest()!=expected:raise PermissionError('Package source tree changed: '+name)


def execute(root, job, contract, capability):
    stage=job['stage'];runtime_check(contract,stage)
    if stage=='synthesis':
        from .angles import Angle,synthesize
        return synthesize(Angle.load(job['angle']),contract['T']['gate_precision'],contract['T']['options'])
    from .intervals import configure
    configure()
    if stage=='CD':
        from fractions import Fraction as F
        from .comparison import cd_recipe,certified_primitives,tuples,cd_task
        from .native_evaluation import serialize_frames
        frames=cd_recipe(capability)
        qs,certificates=certified_primitives(frames,contract['CD']['primitive_signs'])
        terms=tuples(qs,contract['CD']['coefficients'])
        rows=[]
        for delta,n in zip(contract['CD']['deltas'],contract['CD']['steps']):
            for arm in ('A','B'):
                rows.append(cd_task(qs,terms,F(delta),n,arm,contract))
                dump(job['output']+'.prefix.json',{'completed_tasks':rows,'partial_not_scientific_outcome':True})
        return {'domain':'CD_CONDITIONAL_ENVELOPE_SYMBOLIC_COST','frames':serialize_frames(frames),
                'frame_certificates':certificates,'rows':rows,'whole_PR_resource_winner':None}
    if stage=='E_inputs':
        from .comparison import e_recipe
        from .native_evaluation import serialize_frames
        return {str(r):serialize_frames(frames) for r,frames in e_recipe(capability).items()}
    if stage=='native':
        from .native_evaluation import native_fixture,deserialize_frames
        prior=json.loads((root/'artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json').read_text())
        def on_compile(prefix):
            dump(job['output']+'.prefix.json',{'prefix':prefix,'partial_not_scientific_outcome':True})
        return native_fixture(job['r'],deserialize_frames(job['frames']),prior['central_compilations'],on_compile)
    raise ValueError('Unknown fixed stage')


def main():
    start,cpu=time.monotonic(),time.process_time()
    root=Path(__file__).resolve().parents[4]
    job=json.loads(Path(sys.argv[1]).read_text())
    capability=verify_child(root,job)  # BEFORE numpy/scipy/qiskit or registered recipes.
    claim_job(root,job)  # Exclusive per-job receipt: no same-key retry within a consumed marker.
    contract=json.loads((root/ART/'contract_v2.json').read_text())
    result=execute(root,job,contract,capability)
    result['_process_receipt']={'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'retries':0}
    dump(job['output'],result)


if __name__=='__main__':main()
