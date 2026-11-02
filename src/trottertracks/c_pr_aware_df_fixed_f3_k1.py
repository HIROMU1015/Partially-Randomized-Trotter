"""Proposed-only thin binding for unchanged K1 CAR/Arb mathematics.

No molecular arithmetic occurs on import/preflight. F2 remains fixed to its
Baseline. Chat permission, not an approval-proof artifact, authorizes science.
"""
from copy import deepcopy
from fractions import Fraction as F
import importlib.metadata
from pathlib import Path
import resource
import sys
import time

from . import c_pr_aware_df_fixed_k1_execution as f2

io, k1, old = f2.io, f2.k1, f2.old
VERSION = 'fixed_F3_K1_proposed_saved_binding_v1'
SCOPE = 'C_PR_AWARE_DF_FIXED_F3_K1_V1'
BASE = '8af9c2748bf2fd9b9b8ce691f7f1df0bee7a364a'
FAMILY = 'artifacts/research_c_pr_aware_df_fixed_car/2026-10-28'
FRESH = FAMILY+'/F3-K1-run01'
F1 = 'artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F1-run01'
F1_CONTRACT_SHA = '07d4dde2260179e84d828fa89f82bac783df5052c7c9dfa4b512e4010e073e07'
F1_SOURCE_SHA = '0b10324fd59835dc371a65290dbd52e8909d9d85fbb44120421907f107f77701'
F2_CONTRACT = 'docs/research/research_c_pr_aware_df_F2_K1_contract_draft_20261027.json'
F2_CONTRACT_SHA = 'bc7a63d6ba0dd3355c9ea8b4aa3231d6187a1cd5ee97f1c8e76d44d3185c4b63'
DIAGNOSTIC = 'artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01'
SEED = 12704117278500360798
LIMITS = f2.LIMITS.copy()
POLICY = f2.POLICY.copy()


def expected_reuse(f1):
    reuse = deepcopy(f1['reuse'])
    reuse.update(compiled_raw=F1+'/compiled_raw.json',
                 raw_sha256=f1['saved_inputs'][f1['circuits']['raw']])
    reuse['files'] += [F1+'/'+name for name in
                      ('SOURCE.json','contract.json','STOP.json','result.json','compiled_raw.json')]
    return reuse


def f1_gate(c, prior, stop, result, pair):
    """Pin role and the exact pair, never infer it from index0 or a bound value."""
    if (prior['experiment']!='F1' or prior['arm']!='proposed' or prior['trajectory_index']!=0
            or prior['source_manifest_sha256']!=F1_SOURCE_SHA
            or prior['precision_bits']!=192 or prior['global_phase_fit'] is not False
            or prior['allowance_exact']!=POLICY['allowance_exact']
            or c['reuse']!=expected_reuse(prior)
            or c['circuits']!=prior['circuits']
            or any(c['saved_inputs'].get(path)!=prior['saved_inputs'][path]
                   for path in prior['circuits'].values())):
        raise ValueError('F3_F1_PROPOSED_RAW_IDENTITY')
    if (stop['scientific_units']!=7 or stop['first_failure'] is not None
            or stop['call_counts']!=old.LIMITS['F1'] or stop['retry'] or stop['resume']
            or not stop['mandatory_stop'] or result['status']!='COMPLETE'
            or result['arm']!='proposed' or result['experiment']!='F1'
            or set(result['gates'])!={'compiled_raw','raw_actual_adjoint','compiled_actual_adjoint'}
            or any(x!='VALIDATED_WITHIN_FIXED_ALLOWANCE' for x in result['gates'].values())):
        raise ValueError('F3_F1_COMPLETION')
    if (pair['status']!='VALIDATED_FIXED_CIRCUIT_PAIR'
            or pair['gate_status']!='VALIDATED_WITHIN_FIXED_ALLOWANCE'
            or pair['norm']!='full_operator_2' or pair['phase_fit'] is not False
            or pair['precision_bits']!=192 or pair['dimension']!=512
            or pair['columns_evaluated']!=512 or pair['allowance_exact']!=POLICY['allowance_exact']
            or not F(0)<=F(pair['lower_exact'])<=F(pair['upper_exact'])<=F(POLICY['allowance_exact'])):
        raise ValueError('F3_F1_CERTIFICATE_QUALIFICATION')


def trajectory_gate(payload, c):
    old.trajectory_gate(payload,role='proposed',sample=0,
        source_digest=c['sampling_SOURCE_sha256'],export_digest=c['saved_inputs'][c['export']])
    if payload['seed']!=SEED:
        raise ValueError('F3_PROPOSED_SEED')
    return payload


def diagnostic_gate(root,c):
    """Metadata/SOURCE only: do not decode or recalculate saved kappa values."""
    prior=io.read(root/(DIAGNOSTIC+'/contract.json'))
    source=io.read(root/(DIAGNOSTIC+'/SOURCE.json'))
    stop=io.read(root/(DIAGNOSTIC+'/STOP.json'))
    result=io.read(root/(DIAGNOSTIC+'/result.json'))
    name='src/trottertracks/c1_stage1.py'
    if (c['input']!=prior['input'] or c['saved_inputs'][c['input']]!=prior['inputs'][prior['input']]
            or source['files'][name]!=io.sha(root/name)
            or prior['python']!=c['python'] or prior['packages']['numpy']!=c['packages']['numpy']
            or c['diagnostic_exact']!=DIAGNOSTIC+'/exact_dyadic.json'
            or prior['policy']!=k1.diagnostic.POLICY
            or stop['scientific_units']!=6 or stop['counts']!=k1.diagnostic.LIMITS
            or stop['first_failure'] is not None or stop['retry'] or stop['resume']
            or not stop['mandatory_stop'] or result['status']!='DIAGNOSTIC_COMPLETE'):
        raise ValueError('F3_COMMON_KAPPA_DIAGNOSTIC_IDENTITY')


def preflight(root,path):
    """Byte/JSON/version checks only. Never NPZ/QPY scientific decode."""
    root=Path(root).resolve();c=io.read(path)
    if io.sha(root/(F1+'/contract.json'))!=F1_CONTRACT_SHA:
        raise ValueError('F3_PINNED_F1_CONTRACT')
    if io.sha(root/F2_CONTRACT)!=F2_CONTRACT_SHA:
        raise ValueError('F3_PINNED_F2_CONTRACT')
    prior=io.read(root/(F1+'/contract.json'));baseline=io.read(root/F2_CONTRACT)
    if (c['scope']!=SCOPE or c['execution_adapter']!=VERSION or c['base_commit']!=BASE
            or c['car_adapter']!=k1.VERSION or c['target']!=k1.TARGET
            or c['policy']!=POLICY or c['call_limits']!=LIMITS or c['maximum_logical_units']!=7
            or c['fresh_output']!=FRESH or c['arm']!='proposed' or c['trajectory_index']!=0
            or c['seed']!=SEED or c['status']!='UNAPPROVED' or c['executable'] is not True
            or c['retry'] or c['resume'] or c['paired_trajectory'] is not False):
        raise ValueError('F3_FIXED_SCOPE')
    for name in ('input','export','trajectory','circuits','sampling_SOURCE_sha256',
                 'original_input_sha256','B2_execution_SOURCE_commit'):
        if c[name]!=prior[name]:raise ValueError('F3_PROPOSED_BINDING:'+name)
    if c['reuse']!=expected_reuse(prior):raise ValueError('F3_F1_REUSE_BINDING')
    for name in ('compiler','packages','precision_bits','allowance_exact','global_phase_fit',
                 'scientific_context','python'):
        if c[name]!=baseline[name]:raise ValueError('F3_COMMON_K1_POLICY:'+name)
    if list(sys.version_info[:3])!=c['python']:raise ValueError('F3_PYTHON_VERSION')
    if any(c['saved_inputs'].get(n)!=v for n,v in prior['saved_inputs'].items()):
        raise ValueError('F3_PROPOSED_SAVED_INPUT_BINDING')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:
        raise ValueError('F3_SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for n,h in group.items():
            p=(root/n).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_OR_INPUT:'+n)
    if io.sha(root/(F1+'/SOURCE.json'))!=F1_SOURCE_SHA:raise ValueError('F3_F1_SOURCE')
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('ENVIRONMENT:'+name)
    k1.z.runtime_model()
    f1_gate(c,prior,io.read(root/(F1+'/STOP.json')),
            io.read(root/(F1+'/result.json')),io.read(root/c['reuse']['compiled_raw']))
    old.reuse_gate(root,c)
    bsource=io.read(root/c['B2_source_record']);bstop=io.read(root/c['B2_stop_record'])
    if (bsource['execution_commit']!=c['B2_execution_SOURCE_commit']
            or bsource['sampling_SOURCE_sha256']!=c['sampling_SOURCE_sha256']
            or bstop['scientific_units']!=652 or bstop['failed'] or bstop['retry_within_run']
            or bstop['resume'] or not bstop['mandatory_stop']):
        raise ValueError('F3_B2_PROVENANCE')
    trajectory_gate(io.read(root/c['trajectory']),c)
    diagnostic_gate(root,c)
    if (root/FRESH).exists():raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY')
    return dict(status='PASSIVE_IDENTITIES_PASS_NO_SCIENCE',arm='proposed',
        target=k1.TARGET,scientific_calls=0,executable=True,maximum_logical_units=7,
        source_manifest_sha256=c['source_manifest_sha256'],F1_certificate_recomputed=0,
        kappa_diagnostic_recomputed=0,paired_trajectory=False)


def prepare(root,c,out,call):
    from . import c_pr_aware_df_stage_b as b
    p,data=call('input_arrays','common h/g/c and Proposed J/U only',lambda:f2.load_arrays(root,c))
    arm=call('arm_replay','Proposed metadata with original kappa',lambda:b.prepare_arm(**p,**data,
        role='proposed',source_identity=c['saved_inputs'][c['export']],
        input_identity=c['original_input_sha256'],ld=2))
    events=call('trajectory_load','Proposed saved16 events,seed and source',lambda:old.restore_events(
        trajectory_gate(io.read(root/c['trajectory']),c),arm))
    return call('CAR_reference','Proposed literal B0,exact H/A,and two arm-local bridges',
        lambda:k1.build_reference(arm,b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],events,
            diagnostic_record=io.read(root/c['diagnostic_exact']),
            progress=lambda name,row:io.save(out/(name+'.json'),row)))


def execute(root,c,out,call,**kwargs):
    # The unchanged F2 executor is arm-neutral once given this reference loader
    # and the pinned F1 reuse binding. Only its inherited journal label changes.
    def dispatch(kind,label,function):
        return call(kind,label.replace('baseline_00','proposed_00'),function)
    kwargs.setdefault('reference_loader',prepare)
    return f2.execute(root,c,out,dispatch,**kwargs)


def run(root,path):
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();preflight(root,path);c=io.read(path);out=root/FRESH
    out.mkdir(parents=True,exist_ok=False);call=Calls(out,LIMITS,7);tick=time.monotonic()
    try:
        io.save(out/'SOURCE.json',io.read(root/c['source_manifest']));io.save(out/'contract.json',c)
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure,
            arm='proposed',target=k1.TARGET,raw_CAR_bound=None,model_bridge=None))
        raise
    finally:
        io.save(out/'STOP.json',dict(mandatory_stop=True,arm='proposed',
            scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),
            first_failure=call.first_failure,retry=0,resume=0,new_fit=0,new_sampling=0,
            build_compile=0,F1_reexecuted=False,F2_reexecuted=False))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
