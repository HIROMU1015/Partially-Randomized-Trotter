"""Dormant F1/F2/F3 adapters; separate chat science permission is required.

Passive preflight reads bytes/JSON only. Historical B1 runner/contract is intact.
No sampler, fitter, native emitter or compiler is called by this module.
"""
from fractions import Fraction as F
import importlib.metadata
from pathlib import Path
import resource
import time

from . import c1_stage1_execution as io
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_fixed_car as car
from . import c_pr_aware_df_guarantee_execution as old

VERSION='fixed_CAR_and_proposed_saved_adapter_v1'
LIMITS={'F1':dict(qpy_decode=4,fixed_pair_certificate=3),
        'F2':dict(input_arrays=1,arm_replay=1,trajectory_load=1,CAR_reference=1,
                  qpy_decode=1,raw_CAR_certificate=1,triangle_composition=1),
        'F3':dict(input_arrays=1,arm_replay=1,trajectory_load=1,CAR_reference=1,
                  qpy_decode=1,raw_CAR_certificate=1,triangle_composition=1)}


def _pinned(root,c,path):
    target=(Path(root)/path).resolve()
    if not target.is_relative_to(Path(root).resolve()) or c['saved_inputs'].get(path)!=io.sha(target):
        raise ValueError('SOURCE_OR_INPUT:'+path)
    return target


def trajectory_gate(payload, *, role, source_digest, export_digest, sample=0):
    """Metadata only; role/source/law separation, never index-based pairing."""
    if (payload['arm']!=role or payload['sample']!=sample or payload['SOURCE_sha256']!=source_digest
            or payload['final_JU_sha256']!=export_digest
            or payload['cohort']!='B2_fixed32_per_arm_v1' or payload['microsteps']!=16
            or payload['quantum_shots'] or payload['reused_as_coherent_repetitions']):
        raise ValueError('ARM_TRAJECTORY_SOURCE_IDENTITY')
    expected=[]
    for e in payload['events']:
        if (e['taylor_order']!=0 or e['product_component_ids'] or e['phase']!={'real':1.,'imag':0.}
                or not e['rotation_component_id'].startswith('CDF_'+role+':')):
            raise ValueError('SINGLE_TRAJECTORY_K0_NOT_LAW_MEAN')
        expected.append([0,e['rotation_component_id'],[]])
    if expected!=payload['trajectory_key'] or len(expected)!=16:
        raise ValueError('EVENT_ORDER_OR_COUNT')
    return payload


def restore_events(payload,arm):
    """Existing dataclasses enforce stored event semantics; no new sampling."""
    from trotterlib.rte import RTEEvent,RTEEventApplication
    catalog={a.atom_id:a for a in arm.plan.atoms};events=[]
    for original in payload['events']:
        e=dict(original);apps=[]
        for row in e['application_sequence']:
            row=dict(row)
            if row['basis_change_operations']:raise ValueError('UNREGISTERED_EVENT_BASIS_OPERATIONS')
            row['basis_change_operations']=()
            row['diagonal_pauli_support']=tuple(row['diagonal_pauli_support'])
            apps.append(RTEEventApplication(**row))
        e['application_sequence']=tuple(apps);e['basis_reuse_intervals']=tuple(e['basis_reuse_intervals'])
        for key in ('product_component_ids','selected_component_ids','df_fragment_ids'):e[key]=tuple(e[key])
        e['phase']=complex(e['phase']['real'],e['phase']['imag'])
        event=RTEEvent(**e);atom=catalog[event.rotation_component_id];app=apps[-1]
        if (app.coefficient_sign!=(1 if atom.coefficient>0 else -1)
                or app.diagonal_pauli_support!=tuple(atom.support)
                or F.from_float(app.coefficient_abs)!=abs(atom.coefficient)
                or event.basis_hash!=atom.basis_key):
            raise ValueError('SIGNED_ATOM_SOURCE_MISMATCH')
        events.append(event)
    return tuple(events)


def reuse_gate(root,c):
    """Same shells, certificates and exact stored hashes; no recomputation."""
    for path in c['reuse']['files']: _pinned(root,c,path)
    for label,path in c['reuse']['certificates'].items():
        row=io.read(Path(root)/path)
        if (row['gate_status']!='VALIDATED_WITHIN_FIXED_ALLOWANCE'
                or row['phase_fit'] is not False or row['precision_bits']!=192
                or F(row['upper_exact'])>F(c['allowance_exact'])):
            raise ValueError('REUSED_B1_CERTIFICATE:'+label)
    return dict(status='PINNED_B1_CERTIFICATES_REUSED',recomputed=0,
        shell_scope='same ancilla-only QPY and terminal-measurement rule',whole_law_bound=None)


def preflight(root,path):
    """Bytes+metadata+versions. Never QPY/NPZ numeric decode or certificate."""
    root=Path(root).resolve();c=io.read(path);experiment=c['experiment']
    if (experiment not in LIMITS or c['implementation']!=VERSION or c['verifier']!=fixed.VERSION
            or c['car_adapter']!=car.VERSION or c['status']!='UNAPPROVED'
            or c['call_limits']!=LIMITS[experiment]
            or c['maximum_logical_units']!=sum(LIMITS[experiment].values())
            or c['retry'] or c['resume'] or c['precision_bits']!=192
            or c['allowance_exact']!='1/100000000' or c['global_phase_fit'] is not False):
        raise ValueError('FIXED_NEW_SCOPE')
    manifest=root/c['source_manifest']
    if io.sha(manifest)!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest:raise ValueError('SOURCE_OR_INPUT:'+name)
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT:'+name)
    from .c_pr_aware_df_zsxx_enclosure import runtime_model
    runtime_model()
    if c['arm']!=('baseline' if experiment=='F2' else 'proposed'):raise ValueError('EXPERIMENT_ARM')
    trajectory_gate(io.read(_pinned(root,c,c['trajectory'])),role=c['arm'],
        source_digest=c['sampling_SOURCE_sha256'],export_digest=c['saved_inputs'][c['export']])
    reuse_gate(root,c)
    return dict(status='PASSIVE_IDENTITIES_PASS_NO_SCIENCE',scientific_calls=0,
        executable=c['executable'],maximum_logical_units=c['maximum_logical_units'])


def assess(result,allowance,output,label):
    tolerance=F(allowance);lo=F(result['lower_exact']);hi=F(result['upper_exact'])
    status=('CORRECTNESS_FAIL_FIXED_ALLOWANCE' if lo>tolerance else
            'VALIDATED_WITHIN_FIXED_ALLOWANCE' if hi<=tolerance else 'UNRESOLVED_ENCLOSURE_WIDTH')
    result.update(gate_status=status,allowance_exact=str(tolerance))
    io.save(Path(output)/(label+'.json'),result)
    if status.startswith('CORRECTNESS_FAIL'):raise ValueError(label+':FIXED_ALLOWANCE_FAIL')
    return result


def execute_F1(root,c,out,call,*,loader=old._read_qpy,compare=fixed.compare):
    records={}
    for label,path in c['circuits'].items():
        records[label]=call('qpy_decode',label,lambda path=path:
            loader(Path(root)/path,shell=False,expected_qubits=9))
        io.save(Path(out)/(label+'_decoded.json'),records[label])
    pairs=[('compiled_raw',records['raw'],records['compiled']),
        ('raw_actual_adjoint',fixed.adjoint(records['raw']),records['raw_adjoint']),
        ('compiled_actual_adjoint',fixed.adjoint(records['compiled']),records['compiled_adjoint'])]
    rows={}
    for label,left,right in pairs:
        row=call('fixed_pair_certificate',label,lambda:assess(compare(left,right),c['allowance_exact'],out,label))
        rows[label]=row['gate_status']
        if row['gate_status']=='UNRESOLVED_ENCLOSURE_WIDTH':break
    result=dict(status='COMPLETE' if len(rows)==3 and all(x=='VALIDATED_WITHIN_FIXED_ALLOWANCE' for x in rows.values()) else 'UNRESOLVED',
        experiment='F1',arm=c['arm'],gates=rows,whole_law_bound=None,raw_CAR_bound=None,
        CA_achieved=None,shots=None,total_resources=None,mandatory_stop=True)
    io.save(Path(out)/'result.json',result);return result


def prepare_reference(root,c,call):
    """Only invoked in separately permitted F2/F3; no build/compile/fit."""
    from . import c_pr_aware_df_stage_b as b
    from .c_pr_aware_df_stage_b_runner import load
    arrays=call('input_arrays',c['arm'],lambda:(
        load(Path(root)/c['input'],('h','g','constant')),
        load(Path(root)/c['export'],('J','U'))))
    p,data=arrays
    arm=call('arm_replay',c['arm'],lambda:b.prepare_arm(**p,**data,role=c['arm'],
        source_identity=c['saved_inputs'][c['export']],input_identity=c['original_input_sha256'],ld=2))
    payload=io.read(Path(root)/c['trajectory'])
    events=call('trajectory_load',c['arm'],lambda:restore_events(trajectory_gate(payload,role=c['arm'],
        source_digest=c['sampling_SOURCE_sha256'],export_digest=c['saved_inputs'][c['export']]),arm))
    return call('CAR_reference',c['arm'],lambda:car.build_reference(arm,
        b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],events))


def execute_CAR(root,c,out,call,*,reference_loader=prepare_reference,loader=old._read_qpy,compare=car.compare):
    reference=reference_loader(root,c,call);io.save(Path(out)/'CAR_reference.json',reference)
    record=call('qpy_decode','raw',lambda:loader(Path(root)/c['circuits']['raw'],shell=False,expected_qubits=9))
    io.save(Path(out)/'raw_decoded.json',record)
    row=call('raw_CAR_certificate',c['arm'],lambda:assess(compare(record,reference),c['allowance_exact'],out,'raw_CAR'))
    triangle=None
    if row['gate_status']=='VALIDATED_WITHIN_FIXED_ALLOWANCE':
        prior=io.read(Path(root)/c['reuse']['compiled_raw'])
        triangle=call('triangle_composition',c['arm'],lambda:car.compose(prior,row,
            raw_identity=c['saved_inputs'][c['circuits']['raw']],expected_raw_identity=c['reuse']['raw_sha256']))
        io.save(Path(out)/'compiled_CAR.json',triangle)
    result=dict(status='COMPLETE' if triangle is not None else 'UNRESOLVED',experiment=c['experiment'],
        arm=c['arm'],raw_CAR=row['gate_status'],compiled_CAR=triangle,
        whole_law_bound=None,physical_Hamiltonian_bound=None,CA_achieved=None,shots=None,total_resources=None,mandatory_stop=True)
    io.save(Path(out)/'result.json',result);return result


def run(root,path):
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();preflight(root,path);c=io.read(path)
    if c['executable'] is not True:raise ValueError('CONTRACT_NOT_EXECUTABLE')
    out=(root/c['fresh_output']).resolve()
    if not out.is_relative_to(root) or out.exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY')
    out.mkdir(parents=True);call=Calls(out,c['call_limits'],c['maximum_logical_units'])
    start=time.monotonic()
    io.save(out/'SOURCE.json',io.read(root/c['source_manifest']));io.save(out/'contract.json',c)
    try:
        if c['experiment']=='F1':return execute_F1(root,c,out,call)
        return execute_CAR(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(mandatory_stop=True,call_counts=dict(call.counts),
            scientific_units=sum(call.counts.values()),first_failure=call.first_failure,retry=0,resume=0,
            new_fit=0,new_sampling=0,build_compile=0))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
