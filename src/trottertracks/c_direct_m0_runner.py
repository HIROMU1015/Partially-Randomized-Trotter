"""Prepared fixed saved-word m0 runner; default entry is bytes-only preflight.

No fresh/IID sample is invented for a saved word. The committed draft is
nonexecutable pending scientific Hamiltonian-target identity and frozen SOURCE.
"""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import json
import time
import resource

from . import c_direct_round_signal as direct
from . import c_direct_m0_oracle as oracle
from . import c_fresh_iid_calibration as fresh
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z
from . import c1_stage1_execution as io

BASE='db6313156f97a37c2710f56ed7a93a34967d58f4'
SCOPE='C_PR_FIXED_SAVED_WORD_M0_ORACLE_SIGNAL_V1'
FAMILY='artifacts/research_c_direct_m0_runner/2026-11-05'
CONTRACT='docs/research/research_c_direct_m0_contract_unapproved_20261105.json'
LIMITS={k:1 for k in ('metadata_load','arrays_load','target_identity','shell_decode',
    'oracle_proposal','oracle_certificate','sector_embedding','state_binding',
    'literal_prepare','head_action','forward_action','inverse_witness','readout_X','readout_Y','summary')}
MAXIMUM=sum(LIMITS.values())


def preflight(root, path=None):
    """Hash bytes and read JSON only; never decode arrays/QPY/Arb numbers."""
    root=Path(root).resolve();c=io.read(path or root/CONTRACT)
    from .c_b1_b2_scope_integration import identity as fixed_identity
    if (c['scope']!=SCOPE or c['base_commit']!=BASE or c['call_limits']!=LIMITS
            or c['maximum_logical_units']!=MAXIMUM or c['precision_bits']!=192
            or c['m']!=0 or c['occurrences']!=1 or c['retry']!=0 or c['resume']!=0
            or c['mandatory_stop'] is not True or c['fresh_sampling']!=0
            or c['new_build_compile']!=0 or c['Stage_C_executable'] is not False
            or c['identity']!=fixed_identity()):
        raise ValueError('FIXED_M0_PREPARATION_SCOPE')
    manifest=(root/c['source_manifest']).resolve()
    if not manifest.is_relative_to(root) or io.sha(manifest)!=c['source_manifest_sha256']:
        raise ValueError('TECHNICAL_SOURCE_MANIFEST')
    for group in (io.read(manifest)['files'],c['saved_inputs']):
        for name,digest in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=digest: raise ValueError('SOURCE_OR_INPUT_BYTES:'+name)
    if not set(c['roles'].values()) <= set(c['saved_inputs']): raise ValueError('UNPINNED_INPUT_ROLE')
    output=(root/c['fresh_output']).resolve()
    if not output.is_relative_to(root) or output.exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
    missing=[]
    if c.get('executable') is not True: missing.append('nonexecutable_draft')
    if c.get('scientific_SOURCE') is None: missing.append('scientific_SOURCE_to_be_frozen')
    target=c.get('oracle_target_identity')
    if target is None: missing.append('saved_H_vs_existing_own_DF_oracle_target_identity')
    else:
        want=dict(kind='EXACT_SAVED_SECTOR_H_IS_EXISTING_OWN_DF_ORACLE_TARGET',
            candidate=c['identity']['candidate'],H_file=c['roles']['sector'],
            H_file_sha256=c['saved_inputs'][c['roles']['sector']],H_array='H')
        if target!=want: raise ValueError('ORACLE_TARGET_IDENTITY_SCOPE')
    return dict(status='NOT_READY' if missing else 'PREPARED_FIXED_SCOPE',missing=missing,
        execution_ready=not missing,scientific_calls=0,arrays_decoded=0,QPY_decoded=0,
        maximum_logical_units=MAXIMUM,Stage_C_ready=False,mandatory_stop=True)


def metadata(root,c):
    keys=('raw','compiled','actual_inverse','inverse_binding','literal','per_word')
    return {k:io.read(root/c['roles'][k]) for k in keys}


def arrays(root,c):
    from .c_pr_aware_df_stage_b_runner import load
    return dict(candidate=load(root/c['roles']['candidate'],('J','U')),
        sector=load(root/c['roles']['sector'],('J','U','H')),
        basis=load(root/c['roles']['basis'],('sector_basis_indices',)),
        ground=load(root/c['roles']['ground'],('energy','sector_state')))


def identity(bundle,data,c):
    """Exact candidate/word/phase provenance; no signal or ground evaluation."""
    import numpy as np
    R,C,I=[bundle[k] for k in ('raw','compiled','actual_inverse')]
    for r in (R,C,I):
        fixed._validate(r)
        if r['qubits']!=9: raise ValueError('FULL_CONTROLLED_REGISTER')
    if any(g['name'] not in ('rz','sx','x','cx') for g in C['gates']): raise ValueError('P0_NATIVE_GRAMMAR')
    ref=bundle['literal'];p=bundle['per_word'];b=ref['binding'];target=b['target']
    if (ref['status']!='VALIDATED_LITERAL_REFERENCE_BLOCKS' or ref['precision_bits']!=192
            or ref['system_dimension']!=256 or b['target_id']!=c['literal_target_id']
            or b['target_id']!=fresh.digest(target) or target['candidate']!=c['identity']['candidate']
            or b['word']!=c['word'] or b['word_id']!=fresh.digest(c['word'])
            or p['binding']!=b or p['compiled_id']!=fresh.digest(C) or p['raw_id']!=fresh.digest(R)
            or bundle['inverse_binding']['same_realized_forward_record']!=fresh.digest(C)):
        raise ValueError('SAME_SOURCE_WORD_PHASE_REFERENCE_AND_INVERSE')
    for key in ('J','U'):
        a=data['candidate'][key];saved=data['sector'][key]
        if not np.array_equal(a,saved): raise ValueError('EXACT_CANDIDATE_J_U_IDENTITY')
        want=target[key]
        if (list(a.shape)!=want['shape'] or a.dtype.str!=want['dtype']
                or __import__('hashlib').sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=want['SHA256']):
            raise ValueError('LITERAL_CANDIDATE_ARRAY_IDENTITY')
    H=data['sector']['H'];v=data['ground']['sector_state']
    if H.shape!=(36,36) or v.shape!=(36,) or not np.isfinite(H).all() or not np.isfinite(v).all():
        raise ValueError('SAVED_SECTOR_ORACLE_SCHEMA')
    # This exact gate is deliberately stronger than the old eigh triangle
    # convention. It must STOP rather than silently choosing a different H.
    exact_H=oracle.symmetric_matrix([[complex(x) for x in row] for row in H])
    exact_v=[[str(F(float(x.real))),str(F(float(x.imag)))] for x in v]
    return dict(status='FIXED_SAVED_WORD_IDENTITY_PASS',literal_target_id=b['target_id'],
        word_id=b['word_id'],compiled_id=fresh.digest(C),raw_id=fresh.digest(R),
        inverse_id=fresh.digest(I),H_id=oracle.matrix_id(exact_H)),exact_H,exact_v


def embedding(indices, amplitudes, *, modes=8, nalpha=2, nbeta=2):
    ix=[]
    for value in indices:
        j=int(value)
        if j!=value or not 0<=j<1<<modes: raise ValueError('EXACT_INTEGER_SECTOR_INDICES')
        ix.append(int(f'{j:0{modes}b}'[::-1],2))
    allowed=[i for i in range(1<<modes)
        if sum((i>>q)&1 for q in range(0,modes,2))==nalpha
        and sum((i>>q)&1 for q in range(1,modes,2))==nbeta]
    if sorted(ix)!=allowed or len(amplitudes)!=len(ix): raise ValueError('COMPLETE_PHYSICAL_SECTOR_EMBEDDING')
    v=[['0','0'] for _ in range(1<<modes)]
    for j,pair in zip(ix,amplitudes):v[j]=list(pair)
    return v,dict(system_dimension=1<<modes,sector_dimension=len(ix),indices=ix,
        basis='bit_reverse(saved_interleaved_sector_indices)',intermediate_sector_projection=False)


def shell_record(circuit):
    r=fixed.record(circuit,reduce_ancilla=True,strip_terminal_measurement=True)
    if r['embedded_qubits']!=9: raise ValueError('SHELL_FULL_REGISTER_IDENTITY')
    return dict(r,qubits=9,gates=[dict(g,qubits=[8]) for g in r['gates']])


def decode_shells(root,c):
    from qiskit import qpy
    result={}
    for label in ('head','X','Y'):
        with (root/c['roles']['shell_'+label]).open('rb') as stream:items=qpy.load(stream)
        if len(items)!=1: raise ValueError('ONE_REGISTERED_SHELL')
        result[label]=shell_record(items[0])
    return result


def inverse_witness(actual,forward,state):
    """Reuse public exact unitary-factor reduction for this one state norm."""
    from .c_arb_column_batch import reduce_pair
    a,b,proof=reduce_pair(actual,fixed.adjoint(forward))
    prefix=fixed.identity(forward['qubits']);prefix['gates']=actual['gates'][:proof['prefix_gates']]
    v,e=direct.native_action(prefix,*state)
    left,el=direct.native_action(a,v,e);right,er=direct.native_action(b,v,e)
    return dict(upper_exact=str(F(z._norm(left-right)['upper_exact'])+el+er),
        qualification='propagated_input_state_only_not_operator_or_literal_inverse',
        same_realized_forward_id=fresh.digest(forward),actual_inverse_id=fresh.digest(actual),
        exact_unitary_factor_reduction=proof,rounding_enclosed=True,precision_bits=192)


def save_state(path,state):
    v,e=state
    io.save(path,dict(centre=[[str(z._fraction(v[i,0].real)),str(z._fraction(v[i,0].imag))]
        for i in range(v.nrows())],vector2_radius_exact=str(e),precision_bits=192))


def evaluate(bundle,data,c,out,call,*,shell_loader=decode_shells,root=None):
    """Numerical implementation; readiness tests inject invented inputs only."""
    from flint import ctx
    id_record,H,v=call('target_identity','exact candidate/target/register/no projection',lambda:identity(bundle,data,c))
    io.save(out/'identity.json',id_record)
    shells=call('shell_decode','compiled head/X/Y, no new build/compile',lambda:shell_loader(root,c))
    proposal=call('oracle_proposal','one numerical isolation proposal;not a certificate',lambda:oracle.propose_isolation(H))
    io.save(out/'oracle_proposal.json',proposal)
    cert=call('oracle_certificate','exact inertia and outward state distance',lambda:oracle.qualify_ground(H,v,proposal))
    io.save(out/'oracle_certificate.json',cert)
    embedded,indices=call('sector_embedding','same bit-reversed physical basis',lambda:embedding(
        data['basis']['sector_basis_indices'],v))
    io.save(out/'embedding.json',indices)
    state=call('state_binding','original unnormalized saved centre plus certified distance',lambda:direct.bind_state(
        embedded,source_id=c['source_manifest_sha256'],target_id=c['literal_target_id'],
        uncertainty=cert['vector2_distance_upper_exact'],provenance='exact input-H ground distance;research-target identity separately fixed by contract'))
    io.save(out/'initial_state.json',state)
    with z._BALL_LOCK,ctx.workprec(192):
        initial=direct._state(state,c['source_manifest_sha256'],c['literal_target_id'])
        ref=call('literal_prepare','saved literal blocks,no regeneration',lambda:direct._literal(bundle['literal'],256))
        def head():
            ideal=direct.ideal_shells(9)
            return direct.native_action(shells['head'],*initial),direct.native_action(ideal['head'],*initial)
        a,b=call('head_action','compiled and ideal ancilla head once',head)
        save_state(out/'compiled_head_state.json',a);save_state(out/'literal_head_state.json',b)
        def forward():return direct.native_action(bundle['compiled'],*a),direct.literal_action(ref,*b)
        compiled,literal=call('forward_action','one same-word native/literal occurrence',forward)
        save_state(out/'compiled_forward_state.json',compiled);save_state(out/'literal_forward_state.json',literal)
        inverse=call('inverse_witness','same compiled forward actual inverse',lambda:inverse_witness(
            bundle['actual_inverse'],bundle['compiled'],compiled))
        io.save(out/'inverse_state_witness.json',inverse)
        axes={};ideal=direct.ideal_shells(9)
        for axis in ('X','Y'):
            def read(axis=axis):
                C=direct.observable_z(*direct.native_action(shells[axis],*compiled))
                D=direct.observable_z(*direct.native_action(ideal[axis],*literal))
                return dict(compiled=C,literal_atom=D,difference=[str(F(C['signal'][0])-F(D['signal'][1])),
                    str(F(C['signal'][1])-F(D['signal'][0]))])
            axes[axis]=call('readout_'+axis,'same forward state,axis-specific shell',read)
            io.save(out/(axis+'.json'),axes[axis])
    return call('summary','finite m0 only;no expectation calibration',lambda:dict(
        status='FIXED_SAVED_WORD_M0_STATE_SIGNAL_COMPLETE',scope=SCOPE,m=0,occurrences=1,
        word_id=id_record['word_id'],state_id=state['state_id'],literal_target_id=c['literal_target_id'],
        oracle_H_id=cert['H_id'],oracle_qualification='conditional on fixed input-H/research-target identity',
        X=axes['X']['difference'],Y=axes['Y']['difference'],axes=axes,
        raw_signals={kind:dict(real=axes['X'][kind]['signal'],imag=axes['Y'][kind]['signal'])
            for kind in ('compiled','literal_atom')},
        raw_signal_schema='complex_rectangle_exact_rational_endpoints_v1',
        qualification=direct.QUALIFICATION,precision_bits=192,rounding_enclosed=True,
        full_operator_error_certified=False,fresh_IID_sample=False,calibration_eligible=False,
        statistical_confidence=None,Stage_C_ready=False,actual_CA_achieved=None,mandatory_stop=True))


def execute(root,c,out,call):
    b=call('metadata_load','saved gate/reference/inverse metadata',lambda:metadata(root,c))
    d=call('arrays_load','saved candidate/H/ground/basis only',lambda:arrays(root,c))
    return evaluate(b,d,c,out,call,root=root)


def run(root,path=None,*,execute_science=False):
    if not execute_science: raise PermissionError('SEPARATE_EXPLICIT_FIXED_M0_SCIENTIFIC_PERMISSION_REQUIRED')
    root=Path(root).resolve();path=Path(path or root/CONTRACT);check=preflight(root,path)
    if not check['execution_ready']: raise ValueError('NOT_READY:'+','.join(check['missing']))
    z.runtime_model()
    from .c1_stage3_integrated import Calls
    c=io.read(path);out=root/c['fresh_output'];out.mkdir(parents=True,exist_ok=False)
    calls=Calls(out,LIMITS,MAXIMUM);tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        value=execute(root,c,out,calls);io.save(out/'result.json',value);return value
    except BaseException as exc:
        calls.stop.set()
        if calls.first_failure is None:
            calls.first_failure=dict(error_type=type(exc).__name__,error=str(exc))
        io.save(out/'partial_failure.json',dict(error_type=type(exc).__name__,error=str(exc)));raise
    finally:
        io.save(out/'STOP.json',dict(counts=dict(calls.counts),attempted=sum(calls.counts.values()),
            maximum=MAXIMUM,failed=calls.first_failure,retry=0,resume=0,mandatory_stop=True,
            Stage_C_executed=False,actual_CA_achieved=None))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
