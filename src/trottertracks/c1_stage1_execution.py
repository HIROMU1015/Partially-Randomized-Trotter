"""Limited Stage 1 dispatcher. Approval/identity gates precede numeric imports.

No grant is generated here. Current contract remains unapproved for science.
The user must separately approve its hash, SOURCE, input, output and one fit.
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

SCOPE="C1_STAGE1_P_H4_ONE_FIT_COMPAT_V2"
P_H4_SHA256="8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50"
ENV_DIR="artifacts/research_c_c1_stage1_compatibility/2026-10-25/environment"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,payload):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runtime_identity(root,role):
    root=Path(root)
    if sys.version_info[:3]!=(3,11,1):
        raise ValueError('PYTHON_VERSION_EXPECTED_3_11_1')
    pins=(root/ENV_DIR/f'{role}-pins.txt').read_text().splitlines()
    packages={}
    for line in pins:
        name,expected=line.split('==')
        actual=importlib.metadata.version(name)
        if actual!=expected:
            raise ValueError('DEPENDENCY_VERSION: '+name)
        packages[name]=actual
    return dict(python=sys.version,executable=sys.executable,packages=packages)


def preflight(root,contract_path,source_path,input_path):
    """Bytes/JSON only. No npz decode, dependency import, output or grant."""
    root=Path(root).resolve(); contract_path=Path(contract_path).resolve()
    source_path=Path(source_path).resolve(); input_path=Path(input_path).resolve()
    contract=read(contract_path); source=read(source_path)
    from .c1_stage1 import FIT_SETTINGS, ALLOWANCE_VERSION, COMPAT_VERSION, expected_evaluation_calls
    if (contract['scope']!=SCOPE or contract['conditions_adopted'] is not True
            or contract['execution_authorized'] is not False
            or contract['grant_issued'] is not False):
        raise ValueError('CONTRACT_SCOPE_OR_STATE')
    if (contract['compatibility']['version']!=COMPAT_VERSION
            or contract['fit_settings']!=FIT_SETTINGS or contract['allowance']['version']!=ALLOWANCE_VERSION
            or contract['call_limits']['evaluation_max_R8']!=expected_evaluation_calls(8,36)):
        raise ValueError('CONTRACT_NUMERICAL_POLICY')
    for name,expected in source['files'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or sha(target)!=expected:
            raise ValueError('SOURCE_HASH: '+name)
        frozen=subprocess.check_output(['git','show',source['implementation_commit']+':'+name],cwd=root)
        if hashlib.sha256(frozen).hexdigest()!=expected:
            raise ValueError('IMPLEMENTATION_COMMIT_FILE: '+name)
    if contract['input']['sha256']!=P_H4_SHA256 or sha(input_path)!=P_H4_SHA256:
        raise ValueError('INPUT_HASH')
    if contract['source_manifest_sha256']!=sha(source_path):
        raise ValueError('SOURCE_MANIFEST_HASH')
    if source['implementation_commit']!=contract['implementation_source_commit']:
        raise ValueError('SOURCE_COMMIT')
    return dict(root=root,contract_path=contract_path,source_path=source_path,
                input=input_path,contract=contract,source=source)


def verify_approval(verified,approval_path,output,*,worker=False):
    from .c1_stage1 import ALLOWANCE_VERSION
    approval=read(approval_path); output=Path(output).resolve();root=verified['root']
    actual_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if (approval.get('scope')!=SCOPE or approval.get('approved') is not True
            or approval.get('authority')!='USER_EXPLICIT_C1_STAGE1_SCIENCE'
            or approval.get('contract_sha256')!=sha(verified['contract_path'])
            or approval.get('source_manifest_sha256')!=sha(verified['source_path'])
            or approval.get('source_commit')!=actual_head
            or approval.get('input_sha256')!=sha(verified['input'])
            or approval.get('output')!=str(output)
            or approval.get('numerical_allowance_version')!=ALLOWANCE_VERSION
            or approval.get('numerical_allowance_accepted') is not True
            or approval.get('one_fit') is not True):
        raise PermissionError('NEW_EXPLICIT_STAGE1_SCIENCE_APPROVAL_REQUIRED')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=root):
        raise ValueError('TRACKED_SOURCE_DRIFT')
    if not worker and output.exists():
        raise PermissionError('FRESH_OUTPUT_REQUIRED_NO_RESUME')
    if worker and (not output.is_dir() or (output/'STOP.json').exists()
                   or read(output/'approval.json')!=approval):
        raise PermissionError('WORKER_ATTEMPT_IDENTITY')
    return approval


def journal(output,phase,limits):
    counts=Counter()
    def append(kind,label='',count=1):
        if count<1 or kind not in limits or counts[kind]+count>limits[kind]:
            raise ValueError('CALL_LIMIT: '+kind)
        counts[kind]+=count
        row=dict(phase=phase,kind=kind,label=label,count=count,event='attempt',time=time.time())
        with (Path(output)/f'{phase}_calls.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    return append


def array_record(array):
    import numpy as np
    a=np.asarray(array)
    return dict(shape=list(a.shape),dtype=a.dtype.str,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def load_input(path,*,evaluation=False):
    """Explicit whitelist; do not read saved states, matrices or energies."""
    import numpy as np
    from .c1_stage1 import finite_arrays
    names=['one_mo','eri_chemist','nuclear_repulsion','mo_coeff']
    if evaluation:
        names+=['df_one_body','df_lambdas','df_g_matrices']
    with np.load(path,allow_pickle=False) as packet:
        arrays={name:packet[name] for name in names}
    shapes=dict(one_mo=(4,4),eri_chemist=(4,)*4,nuclear_repulsion=(),mo_coeff=(4,4),
                df_one_body=(8,8),df_lambdas=(10,),df_g_matrices=(10,8,8))
    for name,a in arrays.items():
        if a.shape!=shapes[name] or a.dtype!=np.dtype('complex128' if name in ('df_one_body','df_g_matrices') else 'float64'):
            raise ValueError('INPUT_MEMBER_SCHEMA: '+name)
    finite_arrays(*arrays.values())
    return arrays


def worker_fit(v,output):
    import numpy as np
    from . import c1_stage1 as s
    output=Path(output)
    # Exclusive claim prevents repeated subprocess invocation from spending fit2.
    with (output/'fit.claim').open('x') as f:
        f.write(SCOPE+'\n')
    runtime=runtime_identity(v['root'],'fit')
    import ffsim
    import jax
    installed=Path(ffsim.__file__).parent
    for name,expected in v['source']['ffsim_python_files'].items():
        if sha(installed/name)!=expected:
            raise ValueError('FFSIM_SOURCE_DRIFT: '+name)
    if not jax.config.jax_enable_x64:
        raise ValueError('JAX_X64_REQUIRED')
    runtime.update(x64=True,devices=list(map(str,jax.devices())),compatibility_version=s.COMPAT_VERSION)
    save(output/'fit_runtime.json',runtime)
    calls=s.Calls(journal(output,'fit',v['contract']['call_limits']['fit']))
    calls.mark('input_load');a=load_input(v['input'])
    def raw(J,U,result):
        with (output/'raw_fit.npz').open('xb') as f:
            np.savez(f,J=J,U=U,x=result.x,jac=result.jac,fun=result.fun)
        save(output/'raw_status.json',dict(success=bool(result.success),status=int(result.status),
             message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev)))
    arrays,metadata=s.fit_once(a['one_mo'],a['eri_chemist'],a['nuclear_repulsion'],calls,raw_callback=raw)
    write_export(output,arrays,metadata,a,v,calls)
    if not metadata['optimizer']['success']:
        raise RuntimeError('SCIENTIFIC_STOP_FIT_UNSUCCESSFUL')


def write_export(output,arrays,metadata,a,v,calls):
    import numpy as np
    output=Path(output)
    metadata.update(input_sha256=sha(v['input']),source_manifest_sha256=sha(v['source_path']),
                    input_members={k:array_record(x) for k,x in a.items()},
                    frame_ids=[f'F{i}' for i in range(metadata['actual_rank'])],
                    group_ids=[f'G{i}' for i in range(metadata['actual_rank'])],
                    arrays={k:array_record(x) for k,x in arrays.items()},calls=dict(calls.counts))
    with (output/'fit_export.npz').open('xb') as f:
        np.savez(f,**{k:np.asarray(x,dtype='<f8',order='C') for k,x in arrays.items()})
    metadata['export_sha256']=sha(output/'fit_export.npz')
    save(output/'fit_export.json',metadata)


def load_export(output,verified):
    import numpy as np
    from . import c1_stage1 as s
    metadata=read(Path(output)/'fit_export.json')
    if (metadata['schema']!=s.EXPORT_SCHEMA or metadata['ffsim_commit']!=s.FFSIM_COMMIT
            or metadata.get('compatibility_version')!=s.COMPAT_VERSION
            or metadata['z_representation'] is not False
            or metadata['one_body_role']!='ffsim_corrected_kappa'
            or metadata['settings']!=s.FIT_SETTINGS or not metadata['optimizer']['success']
            or metadata['input_sha256']!=sha(verified['input'])
            or metadata['source_manifest_sha256']!=sha(verified['source_path'])
            or metadata['export_sha256']!=sha(Path(output)/'fit_export.npz')):
        raise ValueError('EXPORT_PROVENANCE_OR_POLICY')
    with np.load(Path(output)/'fit_export.npz',allow_pickle=False) as p:
        arrays={k:p[k] for k in p.files}
    if {k:array_record(x) for k,x in arrays.items()}!=metadata['arrays']:
        raise ValueError('EXPORT_MEMBER_HASH')
    R=metadata['actual_rank']
    shapes=dict(kappa=(4,4),diag_coulomb_mats=(R,4,4),orbital_rotations=(R,4,4),constant=())
    if type(R) is not int or not 0<R<=8:
        raise ValueError('EXPORT_RANK')
    for k,shape in shapes.items():
        if arrays[k].shape!=shape or arrays[k].dtype!=np.dtype('<f8'):
            raise ValueError('EXPORT_ARRAY_SCHEMA')
    s.finite_arrays(*arrays.values())
    return arrays,metadata


def execute(v,approval_path,output,fit_python):
    """Only called after fresh explicit approval, never during preparation."""
    from . import c1_stage1 as s
    output=Path(output).resolve()
    approval=verify_approval(v,approval_path,output)
    runtime=runtime_identity(v['root'],'eval')
    output.mkdir(parents=True)
    save(output/'approval.json',approval);save(output/'SOURCE.json',v['source'])
    save(output/'contract.json',v['contract']);save(output/'eval_runtime.json',runtime)
    cache=output/'.cache';(cache/'tmp').mkdir(parents=True)
    env=dict(os.environ,XDG_CACHE_HOME=str(cache),TMPDIR=str(cache/'tmp'),
             MPLCONFIGDIR=str(cache/'matplotlib'),PYTHONPYCACHEPREFIX=str(cache/'pycache'),
             NUMBA_CACHE_DIR=str(cache/'numba'))
    # This CLI owns this process. Keep evaluation imports/caches isolated too.
    for name in ('XDG_CACHE_HOME','TMPDIR','MPLCONFIGDIR','PYTHONPYCACHEPREFIX','NUMBA_CACHE_DIR'):
        os.environ[name]=env[name]
    try:
        command=[str(fit_python),str(v['root']/'scripts/run_research_c_c1_stage1.py'),
                 '--contract',str(v['contract_path']),'--source',str(v['source_path']),
                 '--input',str(v['input']),'--approval',str(Path(approval_path).resolve()),
                 '--output',str(output),'--worker-fit']
        with (output/'fit_process.log').open('x') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                returncode=process.wait()
            except BaseException:
                # Only this task's newly created process group is affected.
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                raise
        if returncode:
            raise RuntimeError('FIT_WORKER_STOP: '+str(returncode))
        trace=journal(output,'load',dict(input_load=1,export_load=1))
        trace('input_load');a=load_input(v['input'],evaluation=True)
        trace('export_load');e,metadata=load_export(output,v)
        if array_record(e['constant'])!=array_record(a['nuclear_repulsion']):
            raise ValueError('CONSTANT_OWNERSHIP')
        calls=s.Calls(journal(output,'evaluation',v['contract']['call_limits']['evaluation_max_R8']))
        (output/'operators').mkdir()
        def keep(name,matrix):
            import numpy as np
            path=output/'operators'/f'{name}.npz'
            with path.open('xb') as stream:
                np.savez(stream,matrix=matrix)
            row=dict(name=name,file=path.name,sha256=sha(path),member=array_record(matrix))
            with (output/'operator_inventory.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        result=s.evaluate(a['one_mo'],a['eri_chemist'],float(a['nuclear_repulsion']),
             e['kappa'],e['diag_coulomb_mats'],e['orbital_rotations'],a['df_one_body'],
             a['df_lambdas'],a['df_g_matrices'],nalpha=2,nbeta=2,
             source_identity=sha(v['source_path']),input_identity=sha(v['input']),calls=calls,operator_sink=keep)
        result['fit_calls']=metadata['calls'];save(output/'result.json',result)
        save(output/'STOP.json',dict(status=result['status'],mandatory_stop=True,Stage2_authorized=False))
        return result
    except BaseException as error:
        save(output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        if not (output/'STOP.json').exists():
            save(output/'STOP.json',dict(status='STOP_PARTIAL_EVIDENCE',mandatory_stop=True,automatic_retry=False))
        raise
