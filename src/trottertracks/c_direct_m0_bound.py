"""Producer-bound saved numerical own-DF oracle, fixed m0 execution proposal.

The old runner/contract remain unchanged. Source lineage establishes the same
saved eigensolver input; exact real symmetry and ground isolation remain
scientific runtime gates. No approval record or substitute state is created.
"""
from pathlib import Path
import importlib.metadata
import sys
import time
import resource
import subprocess

from . import c_direct_m0_runner as prior

io=prior.io
BASE='e409111ccd1427c343c2c44a4e87a3230f70a824'
SCOPE='C_PR_PRODUCER_BOUND_SAVED_OWN_DF_M0_V1'
FAMILY='artifacts/research_c_direct_m0_oracle_binding/2026-11-05'
CONTRACT='docs/research/research_c_direct_m0_bound_contract_20261105.json'
ORIGIN_CONTRACT='artifacts/research_c_near_practical_r32/2026-11-03/run01/contract.json'
ORIGIN_SOURCE='artifacts/research_c_near_practical_r32/2026-11-03/run01/SOURCE.json'
PRODUCERS=(
    'src/trottertracks/c_near_practical_r32.py',
    'src/trottertracks/c_pr_aware_df_stage_b_runner.py',
    'src/trottertracks/c1_stage3_molecular.py',
)


def oracle_binding(root,c):
    """Bytes/JSON lineage, NOT a numerical Hermiticity or eigenstate result.

    The reviewed frozen chain is prepare_arm -> ground(sector['H']) ->
    loader.ground -> MolecularBackend._ground -> np.linalg.eigh(matrix).
    Its default triangle is immaterial only after exact-symmetry runtime PASS.
    """
    root=Path(root);origin=io.read(root/ORIGIN_CONTRACT);source=io.read(root/ORIGIN_SOURCE)
    if (io.sha(root/ORIGIN_SOURCE)!=origin['source_manifest_sha256']
            or origin['scope']!='C_NEAR_PRACTICAL_TWO_ENDPOINT_COMMON_R32_DOWNSTREAM_V1'
            or origin['policy']['oracle']!='own_DF_ground_state_oracle'
            or origin['candidate_sha256']['PR']!=c['identity']['candidate']):
        raise ValueError('SAVED_ORACLE_PRODUCER_OR_CANDIDATE')
    roles=c['roles'];paths=origin['endpoints']['PR']
    if (roles['sector']!=paths['reference'] or roles['candidate']!=paths['candidate']
            or roles['ground']!=origin['fresh_output']+'/PR/A/ground_candidate.npz'):
        raise ValueError('SAVED_GROUND_AND_H_PRODUCER_PATHS')
    for role in ('sector','candidate'):
        p=roles[role]
        if c['saved_inputs'][p]!=origin['saved_inputs'][p]:
            raise ValueError('SAME_ORACLE_INPUT_BYTES')
    for name in PRODUCERS:
        if io.sha(root/name)!=source['files'][name]: raise ValueError('FROZEN_ORACLE_PRODUCER_SOURCE:'+name)
    return dict(kind='SAME_SAVED_NUMERICAL_OWN_DF_EIGENSOLVER_INPUT_WITH_RUNTIME_SYMMETRY_GATE',
        candidate=c['identity']['candidate'],H_file=roles['sector'],H_array='H',
        H_file_sha256=c['saved_inputs'][roles['sector']],
        ground_file=roles['ground'],ground_sha256=c['saved_inputs'][roles['ground']],
        origin_contract=ORIGIN_CONTRACT,origin_source=ORIGIN_SOURCE,
        numerical_producer='prepare_arm -> ground(sector H) -> loader.ground -> MolecularBackend._ground -> numpy.linalg.eigh',
        runtime_conditions=['exact real symmetric H without projection','exact inertia isolates one ground eigenvalue','certified original-centre distance'],
        qualification='same existing numerical oracle reference only;not an exact integral/CAR-construction bridge',
        exact_symmetry_evaluated=False,state_distance_evaluated=False)


def preflight(root,path=None):
    """Read-only fixed-scope checks. Readiness never grants science permission."""
    root=Path(root).resolve();c=io.read(path or root/CONTRACT)
    from .c_b1_b2_scope_integration import identity
    expected=dict(scope=SCOPE,base_commit=BASE,m=0,occurrences=1,precision_bits=192,
        call_limits=prior.LIMITS,maximum_logical_units=prior.MAXIMUM,identity=identity(),
        fresh_sampling=0,new_build_compile=0,reference_regeneration=0,
        new_ground_vector_eigensolve=0,new_eigenvalue_proposal=1,new_PF_RTE=0,
        expectation_calibration=0,Stage_C_executable=False,actual_CA_achieved=None,
        retry=0,resume=0,mandatory_stop=True)
    if any(c.get(k)!=v for k,v in expected.items()): raise ValueError('FIXED_PRODUCER_BOUND_M0_SCOPE')
    manifest=(root/c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest)!=c['source_manifest_sha256']:
        raise ValueError('BOUND_SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest: raise ValueError('SOURCE_OR_INPUT:'+name)
    if not set(c['roles'].values())<=set(c['saved_inputs']): raise ValueError('UNPINNED_ROLE')
    if not {ORIGIN_CONTRACT,ORIGIN_SOURCE}<=set(c['saved_inputs']): raise ValueError('UNPINNED_ORACLE_LINEAGE')
    if prior.CONTRACT not in c['saved_inputs']: raise ValueError('UNPINNED_PARENT_PREPARATION')
    parent=io.read(root/prior.CONTRACT)
    if any(c[k]!=parent[k] for k in ('roles','word','literal_target_id')):
        raise ValueError('SAME_FIXED_WORD_AND_INPUT_ROLES')
    if any(c['saved_inputs'].get(name)!=digest for name,digest in parent['saved_inputs'].items()):
        raise ValueError('PRESERVED_PARENT_INPUT_BYTES')
    binding=oracle_binding(root,c)
    if c['oracle_target_identity']!=binding: raise ValueError('PRODUCER_TARGET_BINDING')
    if sys.version_info[:3]!=(3,11,1): raise ValueError('PINNED_PYTHON')
    for name,version in c['packages'].items():
        if importlib.metadata.version(name)!=version: raise ValueError('PINNED_PACKAGE:'+name)
    output=(root/c['fresh_output']).resolve()
    if not output.is_relative_to(root) or output.exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    missing=[]
    if c.get('executable') is not True: missing.append('nonexecutable_source_preparation')
    sha=c.get('scientific_SOURCE')
    if not isinstance(sha,str) or len(sha)!=40 or any(x not in '0123456789abcdef' for x in sha):
        missing.append('scientific_SOURCE_commit_to_be_frozen')
    else:
        saved=subprocess.run(['git','show',sha+':'+c['source_manifest']],cwd=root,
            capture_output=True,check=False)
        if saved.returncode or saved.stdout!=manifest.read_bytes():
            raise ValueError('FROZEN_SCIENTIFIC_SOURCE_COMMIT')
    return dict(status='NOT_READY' if missing else 'FIXED_SCOPE_READY_AWAITING_EXPLICIT_EXECUTION_PERMISSION',
        execution_ready=not missing,missing=missing,scientific_calls=0,arrays_decoded=0,QPY_decoded=0,
        maximum_logical_units=prior.MAXIMUM,oracle_binding=binding,
        numerical_oracle_qualified=False,Stage_C_ready=False,actual_CA_achieved=None,mandatory_stop=True)


def run(root,path=None,*,execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_EXPLICIT_FIXED_15_CALL_SCIENTIFIC_PERMISSION_REQUIRED')
    root=Path(root).resolve();path=Path(path or root/CONTRACT);ready=preflight(root,path)
    if not ready['execution_ready']: raise ValueError('FIXED_SCIENTIFIC_SOURCE_NOT_READY')
    prior.z.runtime_model()
    from .c1_stage3_integrated import Calls
    c=io.read(path);out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    call=Calls(out,prior.LIMITS,prior.MAXIMUM);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        io.save(out/'oracle_producer_binding.json',ready['oracle_binding'])
        result=prior.execute(root,c,out,call)
        if dict(call.counts)!=prior.LIMITS: raise ValueError('COMPLETE_FIXED_15_CALL_COVERAGE')
        result.update(scope=SCOPE,oracle_qualification='same saved numerical own-DF reference;runtime exact-symmetry and isolated-ground PASS',
            oracle_producer_binding=ready['oracle_binding'],ideal_integral_or_exact_CAR_ground_claimed=False)
        io.save(out/'result.json',result);return result
    except BaseException as exc:
        call.stop.set()
        if call.first_failure is None:call.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(call_counts=dict(call.counts),scientific_units=sum(call.counts.values()),
            maximum=prior.MAXIMUM,failed=call.first_failure,retry=0,resume=0,mandatory_stop=True,
            Stage_C_executed=False,actual_CA_achieved=None))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
