"""Dormant B1 saved-circuit pilot. Chat permission required before execution.

No fit,sampling,circuit build/compile,or molecular reference evaluator. It
compares only the exact stored gate semantics of a frozen finite QPY list.
"""
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import time

from . import c1_stage1_execution as io
from . import c_pr_aware_df_guarantee_strategy as strategy
from . import c_pr_aware_df_fixed_certificate as fixed

LIMITS=dict(qpy_decode=10,fixed_pair_certificate=9)
MAX_CALLS=19


def preflight(root,path):
    from .c_pr_aware_df_c2a import P0
    from .c_pr_aware_df_zsxx_enclosure import runtime_model,PRECISION
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=strategy.SCOPE or c['implementation']!=strategy.VERSION
            or c['verifier']!=fixed.VERSION or c['status']!='UNAPPROVED_FIXED_CIRCUIT_ONLY'
            or c['executable'] is not True or c['maximum_logical_units']!=MAX_CALLS
            or c['call_limits']!=LIMITS or c['compiler']!=P0 or c['precision_bits']!=PRECISION
            or c['retry'] or c['resume']):raise ValueError('B1_FIXED_CONTRACT_SCOPE')
    manifest=(root/c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest)!=c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'],c['saved_inputs']):
        for name,h in group.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+name)
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT:'+name)
    from importlib.util import find_spec
    qroot=Path(find_spec('qiskit').origin).parent.resolve()
    for name,h in c['external_python_source_hashes'].items():
        target=(qroot/name).resolve()
        if not target.is_relative_to(qroot) or io.sha(target)!=h:raise ValueError('COMPILER_SOURCE:'+name)
    if (set(c['circuits'])!={'kernel_raw','kernel_compiled','kernel_raw_adjoint','kernel_compiled_adjoint',
                           'head_raw','head_compiled','X_raw','X_compiled','Y_raw','Y_compiled'}
            or any(name not in c['saved_inputs'] for name in c['circuits'].values())):
        raise ValueError('FIXED_INPUT_LIST')
    return dict(status='IDENTITIES_PASS_SCIENCE_NOT_STARTED',scientific_calls=0,
        executable_after_separate_chat_permission=True,runtime_model=runtime_model())


def _read_qpy(path,*,shell,expected_qubits):
    from qiskit import qpy
    with Path(path).open('rb') as stream:circuits=qpy.load(stream)
    if len(circuits)!=1 or circuits[0].num_qubits!=expected_qubits:
        raise ValueError('FIXED_QPY_DIMENSION')
    return fixed.record(circuits[0],reduce_ancilla=shell,strip_terminal_measurement=shell)


def _shell(label):
    r=fixed.identity(1)
    if label=='Y':r['gates'].append(dict(name='sdg',qubits=[0],params_hex=[]))
    r['gates'].append(dict(name='h',qubits=[0],params_hex=[]));return r


def execute(root,c,output,call,*,loader=_read_qpy,compare=fixed.compare):
    """Shared with artificial/mock E2E;production list is frozen by preflight."""
    records={}
    for label,path in c['circuits'].items():
        records[label]=call('qpy_decode',label,lambda label=label,path=path:
            loader(Path(root)/path,shell=not label.startswith('kernel'),expected_qubits=c['expected_qubits']))
        io.save(Path(output)/(label+'_decoded.json'),records[label])
    pairs=[('kernel_compiled_raw',records['kernel_raw'],records['kernel_compiled']),
           ('kernel_raw_actual_adjoint',fixed.adjoint(records['kernel_raw']),records['kernel_raw_adjoint']),
           ('kernel_compiled_actual_adjoint',fixed.adjoint(records['kernel_compiled']),records['kernel_compiled_adjoint'])]
    for label in ('head','X','Y'):
        pairs.append((label+'_compiled_raw',records[label+'_raw'],records[label+'_compiled']))
        pairs.append((label+'_raw_exact_reference',records[label+'_raw'],_shell(label)))
    rows={};tolerance=Fraction(c['allowance_exact'])
    for label,left,right in pairs:
        def assess():
            r=compare(left,right);lower=Fraction(r['lower_exact']);upper=Fraction(r['upper_exact'])
            if lower>tolerance:status='CORRECTNESS_FAIL_FIXED_ALLOWANCE'
            elif upper<=tolerance:status='VALIDATED_WITHIN_FIXED_ALLOWANCE'
            else:status='UNRESOLVED_ENCLOSURE_WIDTH'
            r.update(gate_status=status,allowance_exact=str(tolerance));io.save(Path(output)/(label+'.json'),r)
            if status.startswith('CORRECTNESS_FAIL'):raise ValueError(label+':FIXED_ALLOWANCE_FAIL')
            return r
        rows[label]=call('fixed_pair_certificate',label,assess)
        if rows[label]['gate_status']=='UNRESOLVED_ENCLOSURE_WIDTH':break
    result=dict(status='FIXED_B1_CIRCUIT_PILOT_COMPLETE' if len(rows)==9 and all(r['gate_status']=='VALIDATED_WITHIN_FIXED_ALLOWANCE' for r in rows.values()) else 'UNRESOLVED_FIXED_PILOT',
        gates={k:v['gate_status'] for k,v in rows.items()},whole_law_bound=None,
        raw_Hamiltonian_reference_bound=None,CA_achieved=None,shots=None,total_resources=None,
        finite_instance='baseline_00 only;no arm comparison,population claim,or unseen trajectory coverage',mandatory_stop=True)
    io.save(Path(output)/'result.json',result);return result


def run(root,contract):
    """Only call after user permission for this new19-unit finite scope."""
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();preflight(root,contract);c=io.read(contract)
    output=root/c['fresh_output']
    if output.exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RESUME')
    output.mkdir(parents=True);call=Calls(output,LIMITS,MAX_CALLS)
    start=time.monotonic();cpu=time.process_time()
    io.save(output/'SOURCE.json',io.read(root/c['source_manifest']))
    io.save(output/'contract.json',c)
    try:return execute(root,c,output,call)
    except BaseException as error:
        message=str(error)
        classification=('UNRESOLVED_UNREGISTERED_GATE_PATH' if any(x in message for x in
            ('UNSUPPORTED_FIXED_GATE','UNKNOWN_CRZZ','UNREVIEWED_LOCAL_GATE')) else 'FAIL_STOP')
        io.save(output/'failure.json',dict(type=type(error).__name__,error=message,
            qualification=classification,first_failure=call.first_failure))
        raise
    finally:
        io.save(output/'STOP.json',dict(mandatory_stop=True,retry=0,resume=0,
            call_counts=dict(call.counts),logical_units=sum(call.counts.values()),first_failure=call.first_failure,
            whole_law_bound=None,raw_Hamiltonian_reference_bound=None,CA_achieved=None,shots=None,total_resources=None))
        io.save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,parent_CPU_seconds=time.process_time()-cpu,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            children_CPU_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            children_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
