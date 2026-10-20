"""Closed reference binding for the 106 stopped axes; no input I/O/imports."""
import cmath
import math
from pf_direct_compile_contract import require

OVERLAY='ROTATE_B0_SIGNAL_ADD_UNROTATED_FULL_H_REFERENCE_V1'


def policy_for(task):
    method=task['candidate_id'].split('_')[0]
    single=(task['system']=='H2_DF3' and int(task['prefix'])==0 and task['PF']=='4th')
    require(method=='B0I' or single,'NOT_REPAIR106_REFERENCE_SCOPE')
    return dict(schema='PF106_REFERENCE_BINDING_V1',system=task['system'],candidate_id=task['candidate_id'],
        prefix=int(task['prefix']),q=int(task['q']),formula=task['PF'],method=method,
        field='exact_truncated' if single else 'corrected',
        forward_rotation='SAVED_C_ONCE' if method=='B0I' else 'NONE',
        preparation='B0_DISCARD_PLUS_SAVED_C' if method=='B0I' else 'B0_DISCARD',
        single_term_S4_saved_signal_inconsistent=single,
        reason='ONE_TERM_YOSHIDA_SUM_IS_ONE; OLD_MERGED_ITERATOR_SUM_IS_2_MINUS_W' if single
            else 'B0I_OVERLAY_ROTATES_SIGNALS_BUT_PRESERVES_UNROTATED_INDEPENDENT_FORWARD')


def zrecord(value):
    require(isinstance(value,dict),'REFERENCE_RECORD_REQUIRED')
    result=complex(value['real'],value['imag'])
    require(math.isfinite(result.real) and math.isfinite(result.imag),'FINITE_SAVED_REFERENCE')
    return result


def bind_references(task,row,policy,*,T=.8):
    require(policy==policy_for(task) and T==.8,'FIXED_REFERENCE_POLICY')
    cell=row['cell']
    require((cell['id'],cell['method'],cell['prefix'],cell['q'],cell['formula'])==
        (task['candidate_id'],policy['method'],int(task['prefix']),int(task['q']),task['PF']),
        'OWN_REFERENCE_IDENTITY')
    field=policy['field'];primary=zrecord(row['signals'][field])
    forward=zrecord(row['independent_forward_signals'][field])
    angle=0.
    if policy['method']=='B0I':
        require(row['allowance']['policy']==OVERLAY and row['normalized_deterministic'] is True,
            'EXPLICIT_SAVED_B0_OVERLAY_REQUIRED')
        receipt=row['scalar_receipt']
        require(receipt['policy']=='EXACT_STORED_FULL_FOCK_TAIL_IDENTITY'
            and not isinstance(receipt['C'],bool),'EXACT_SAVED_C_REQUIRED')
        C=float(receipt['C']);require(math.isfinite(C),'FINITE_SAVED_C')
        angle=-T*C;forward=forward*cmath.exp(1j*angle)
    receipt=dict(policy,primary=dict(real=primary.real,imag=primary.imag),
        independent_forward_after_binding=dict(real=forward.real,imag=forward.imag),
        forward_rotation_angle=angle,original_saved_corrected=row['signals']['corrected'],
        original_independent_forward=row['independent_forward_signals'][field],
        original_saved_record_modified=False,new_classical_signal_or_solver=False)
    return (primary,forward),receipt
