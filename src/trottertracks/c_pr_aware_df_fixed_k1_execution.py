"""K1-only saved-baseline runner; passive by default, chat science permission.

Reuses existing load/arm/event/CAR/column/Calls plumbing. No fitter, sampler,
native emitter, compiler or approval-proof mechanism is invoked here.
"""
import importlib.metadata
from pathlib import Path
import resource
import sys
import time

from . import c1_stage1_execution as io
from . import c_pr_aware_df_fixed_car_execution as old
from . import c_pr_aware_df_fixed_f2 as f2
from . import c_pr_aware_df_fixed_k1 as k1

VERSION = 'fixed_F2_K1_saved_runner_v1'
SCOPE = 'C_PR_AWARE_DF_FIXED_F2_K1_V1'
BASE = '8e53df35e4f2e78fb07cd5b8b409ff69dedb868d'
FRESH = 'artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-K1-run01'
OLD_CONTRACT = 'docs/research/research_c_pr_aware_df_fixed_F2_polar_execution_contract_20261027.json'
OLD_SHA = 'c310f95ad6f25500284c46edbd82fb31dad1a1b82c7cd5c4980a8c9991716da3'
LIMITS = old.LIMITS['F2'].copy()
POLICY = dict(target=k1.TARGET,precision_bits=192,allowance_exact='1/100000000',
    h_exact='1/40',half_head_occurrences=2,kappa='literal returned corrected_kappa',
    kappa_H='exact (kappa+kappa.H)/2 without binary64 rerounding',
    kappa_A='exact (kappa-kappa.H)/2; entry real/imag L1 spectral upper',
    lift='full-Fock maximum8 particles',
    kappa_bridge='two-half-head telescoping;middle norm <= 1+old polar product bound',
    U_bridge='V1(kappa_H,U) to V2(kappa_H,right_polar_U)',
    evaluated_reference='V0(kappa,U)',phase_fit=False,retry=0,resume=0)


def preflight(root, path):
    root=Path(root).resolve(); c=io.read(path)
    if io.sha(root/OLD_CONTRACT)!=OLD_SHA: raise ValueError('OLD_F2_CONTRACT_CHANGED')
    prior=io.read(root/OLD_CONTRACT)
    if (c['scope']!=SCOPE or c['execution_adapter']!=VERSION or c['car_adapter']!=k1.VERSION
            or c['base_commit']!=BASE or c['status']!='UNAPPROVED' or not c['executable']
            or c['policy']!=POLICY or c['call_limits']!=LIMITS or c['maximum_logical_units']!=7
            or c['fresh_output']!=FRESH or c['arm']!='baseline' or c['trajectory_index']!=0
            or c['target']!=k1.TARGET):
        raise ValueError('K1_FIXED_POLICY')
    for key in ('verifier','compiler','packages','circuits','input','export','trajectory',
                'sampling_SOURCE_sha256','original_input_sha256','reuse',
                'B2_execution_SOURCE_commit','B2_source_record','B2_stop_record','B1_contract_record',
                'precision_bits','allowance_exact','global_phase_fit','retry','resume'):
        if c[key]!=prior[key]: raise ValueError('K1_OLD_BASELINE_BINDING:'+key)
    if list(sys.version_info[:3])!=c['python']:raise ValueError('K1_PYTHON_VERSION')
    if any(c['saved_inputs'].get(n)!=v for n,v in prior['saved_inputs'].items()):
        raise ValueError('K1_OLD_INPUT_BINDING')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:
        raise ValueError('K1_SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest: raise ValueError('SOURCE_OR_INPUT:'+name)
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT:'+name)
    k1.z.runtime_model()
    old.trajectory_gate(io.read(root/c['trajectory']),role='baseline',
        source_digest=c['sampling_SOURCE_sha256'],export_digest=c['saved_inputs'][c['export']])
    old.reuse_gate(root,c)
    f2.prior_gate(c,io.read(root/c['B2_source_record']),io.read(root/c['B2_stop_record']),
                  io.read(root/c['B1_contract_record']),io.read(root/c['reuse']['compiled_raw']))
    d=io.read(root/c['diagnostic_result']); stop=io.read(root/c['diagnostic_STOP'])
    if (d['status']!='DIAGNOSTIC_COMPLETE' or stop['scientific_units']!=6
            or stop['first_failure'] is not None or stop['retry'] or stop['resume']
            or d['target_adopted'] or not stop['mandatory_stop']):
        raise ValueError('K1_DIAGNOSTIC_PROVENANCE')
    if (root/FRESH).exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY')
    return dict(status='PASSIVE_IDENTITIES_PASS_NO_SCIENCE',scientific_calls=0,
                target=k1.TARGET,executable=True,maximum_logical_units=7,
                source_manifest_sha256=c['source_manifest_sha256'])


def load_arrays(root,c):
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load
    p=load(root/c['input'],('h','g','constant')); data=load(root/c['export'],('J','U'))
    k1.diagnostic.validate(p['h'],p['g'],p['constant'])
    if (p['h'].shape!=(4,4) or p['g'].dtype!=np.float64 or data['J'].shape!=(8,4,4)
            or data['U'].shape!=(8,4,4) or data['J'].dtype!=np.float64
            or data['U'].dtype!=np.float64 or not np.isfinite(data['U']).all()):
        raise ValueError('K1_SAVED_ARRAY_SCHEMA')
    for J in data['J']: k1.car._hermitian_input(J)
    return p,data


def prepare(root,c,out,call):
    from . import c_pr_aware_df_stage_b as b
    p,data=call('input_arrays','fixed original h/g/c and baseline J/U',lambda:load_arrays(root,c))
    arm=call('arm_replay','unchanged original-kappa native metadata',lambda:b.prepare_arm(**p,**data,
        role='baseline',source_identity=c['saved_inputs'][c['export']],
        input_identity=c['original_input_sha256'],ld=2))
    payload=io.read(root/c['trajectory'])
    events=call('trajectory_load','saved signed K0 events',lambda:old.restore_events(
        old.trajectory_gate(payload,role='baseline',source_digest=c['sampling_SOURCE_sha256'],
                            export_digest=c['saved_inputs'][c['export']]),arm))
    return call('CAR_reference','B0 blocks,exact H/A,kappa and U bridges',lambda:k1.build_reference(
        arm,b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],events,
        diagnostic_record=io.read(root/c['diagnostic_exact']),
        progress=lambda name,row:io.save(out/(name+'.json'),row)))


def execute(root,c,out,call,*,reference_loader=prepare,loader=old.old._read_qpy,compare=k1.compare):
    reference=reference_loader(root,c,out,call);io.save(out/'CAR_reference.json',reference)
    raw=c['circuits']['raw'];identity=c['saved_inputs'][raw]
    record=call('qpy_decode','same baseline_00 raw',lambda:loader(root/raw,shell=False,expected_qubits=9))
    io.save(out/'raw_decoded.json',record)
    def certificate():
        result=compare(record,reference,raw_identity=identity)
        if result['upper_exact'] is None:
            io.save(out/'raw_CAR.json',dict(result,gate_status='UNRESOLVED_MODEL_BRIDGE'))
            return dict(result,gate_status='UNRESOLVED_MODEL_BRIDGE')
        return old.assess(result,POLICY['allowance_exact'],out,'raw_CAR')
    row=call('raw_CAR_certificate','full512 phase-sensitive K1',certificate)
    triangle=None
    if row['gate_status']=='VALIDATED_WITHIN_FIXED_ALLOWANCE':
        triangle=call('triangle_composition','same raw and existing compiled/raw certificate',lambda:k1.compose(
            io.read(root/c['reuse']['compiled_raw']),row,raw_identity=identity,
            expected_raw_identity=c['reuse']['raw_sha256']))
        io.save(out/'compiled_CAR.json',triangle)
    result=dict(status='COMPLETE' if triangle is not None else 'UNRESOLVED',target=k1.TARGET,
        raw_CAR=row['gate_status'],compiled_CAR=triangle,whole_law_bound=None,
        CA_achieved=None,shots=None,total_resources=None,mandatory_stop=True)
    io.save(out/'result.json',result);return result


def run(root,path):
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();preflight(root,path);c=io.read(path);out=root/FRESH
    out.mkdir(parents=True,exist_ok=False);call=Calls(out,LIMITS,7);tick=time.monotonic()
    try:
        io.save(out/'SOURCE.json',io.read(root/c['source_manifest']));io.save(out/'contract.json',c)
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure,
            target=k1.TARGET,raw_CAR_bound=None,model_bridge=None,CA_achieved=None))
        raise
    finally:
        io.save(out/'STOP.json',dict(mandatory_stop=True,scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts),first_failure=call.first_failure,retry=0,resume=0,
            new_fit=0,new_sampling=0,build_compile=0,old_F2_restarted=False))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
