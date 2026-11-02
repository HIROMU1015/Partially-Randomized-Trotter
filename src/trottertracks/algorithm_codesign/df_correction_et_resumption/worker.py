"""Authorized children only. Never launch this during preparation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
from .launch import ART, RESULT, verify_child, sha
from ..df_correction_comparison.guard import exclusive_json
from ..df_correction_comparison.storage import dump_json, load_json, artifact_exists
from ..df_correction_comparison.inventory_integrity import verify_inventory


def claim_job(root,job):
    out=root/RESULT
    if artifact_exists(out/'terminal_status.json'):raise PermissionError('Terminal STOP already saved')
    allowed={'stage','output','marker_sha256'}
    stage=job['stage']
    if stage=='native':
        allowed|={'r','frames'}
        if type(job['r']) is not int or job['r'] not in (3,6):raise PermissionError('Unregistered native fixture')
        label='E-r'+str(job['r'])
        from .saved_data import load_saved_inputs
        contract=json.loads((root/ART/'contract_v4.json').read_text())
        inputs=load_saved_inputs(root,contract)
        if job['frames']!=inputs[str(job['r'])]:raise PermissionError('Native saved input identity mismatch')
    elif stage=='synthesis':
        allowed|={'angle','inventory_sha256','inventory_receipt_sha256'}
        binding={k:job[k] for k in ('inventory_sha256','inventory_receipt_sha256')}
        inventory=verify_inventory(out,binding,marker_sha256=job['marker_sha256'])
        from .acceptance import verify_native_bindings
        verify_native_bindings(out,inventory)
        matches=[i for i,a in enumerate(inventory['unique_keys']) if a==job['angle']]
        if len(matches)!=1:raise PermissionError('Synthesis key not in frozen inventory')
        label='T-'+str(matches[0])
    else:raise PermissionError('Unregistered worker stage')
    if set(job)!=allowed or Path(job['output']).resolve()!=(out/(label+'.json')).resolve():
        raise PermissionError('Worker payload/output path override')
    exclusive_json(out/(label+'.started.json'),{'label':label,'job_sha256':hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),'retries':0,
        **(binding if stage=='synthesis' else {})})


def dump(path, data):
    # Only repeated prefix paths replace a prior NEW-run generation.
    return dump_json(path,data,replace_existing=str(path).endswith('.prefix.json'))


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
        inventory=verify_inventory(root/RESULT,{k:job[k] for k in ('inventory_sha256','inventory_receipt_sha256')},
            marker_sha256=job['marker_sha256'],provenance={k:capability[k] for k in ('source','authorization','authorization_sha256','contract_sha256','source_manifest_sha256')})
        from .acceptance import verify_native_bindings
        verify_native_bindings(root/RESULT,inventory)
        from ..df_correction_comparison.angles import Angle,synthesize
        return synthesize(Angle.load(job['angle']),contract['T']['gate_precision'],contract['T']['options'])
    from ..df_correction_comparison.intervals import configure
    configure()
    if stage=='native':
        from .native_evaluation import native_fixture,deserialize_frames
        from .saved_data import verify_file
        prior=json.loads(verify_file(root,contract['resumption']['historical_phase_E']).read_text())
        def on_compile(prefix):
            if prefix['compile_calls']>2+2*job['r']:raise RuntimeError('Compile cap per rank')
            dump(job['output']+'.prefix.json',{'prefix':prefix,'partial_not_scientific_outcome':True})
        saved_diagnostics=set()
        def on_diagnostic(label,record):
            record['provenance']={k:capability[k] for k in ('source','authorization',
                'authorization_sha256','contract_sha256','source_manifest_sha256','marker_sha256')}
            path=Path(job['output']+'.raw-provenance-'+label+'.json')
            receipt=dump_json(path,record,replace_existing=label in saved_diagnostics)
            saved_diagnostics.add(label)
            return receipt
        return native_fixture(job['r'],deserialize_frames(job['frames']),prior['central_compilations'],on_compile,
                              on_diagnostic=on_diagnostic)
    raise ValueError('Unknown fixed stage')


def main():
    start,cpu=time.monotonic(),time.process_time()
    root=Path(__file__).resolve().parents[4]
    job=json.loads(Path(sys.argv[1]).read_text())
    capability=verify_child(root,job)  # BEFORE numpy/scipy/qiskit or registered recipes.
    claim_job(root,job)  # Exclusive per-job receipt: no same-key retry within a consumed marker.
    contract=json.loads((root/ART/'contract_v4.json').read_text())
    result=execute(root,job,contract,capability)
    if job['stage']=='synthesis':
        binding={k:job[k] for k in ('inventory_sha256','inventory_receipt_sha256')}
        verify_inventory(root/RESULT,binding,marker_sha256=job['marker_sha256'])
        result.update(binding)
    result['_process_receipt']={'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
        'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'retries':0}
    dump(job['output'],result)


if __name__=='__main__':main()
