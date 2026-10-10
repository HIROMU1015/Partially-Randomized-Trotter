"""Policy-bound snapshot loader; does not loosen or call the frozen v3 loader."""
from __future__ import annotations
import json
from decimal import Decimal
import numpy as np
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import npz_bytes
from .ax2b_h6_weighted_projection_v1 import policy, structure
from .ax2b_h6_df_diagnostic_port_v1 import array_record
from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.df_partial_s2 import df_hamiltonian_hash
from trotterlib.pr2_s0_s1_validation import _array_hash, _sector_hash, _state_hash


def load_h6_snapshot(path, snapshot_receipt, df_receipt):
    members=npz_bytes(path,snapshot_receipt,cap=16*2**20)
    metadata=snapshot_receipt['metadata'];hm=metadata['hamiltonian_metadata']
    projection=df_receipt['projection_receipt']
    if (hm!=df_receipt['hamiltonian_metadata'] or projection.get('policy')!=policy()
            or projection.get('status')!='PASS_ENGINEERING'
            or hm.get('projection_receipt_digest')!=digest(projection)
            or hm.get('input_policy')!=policy()['name']
            or hm.get('parents')!=projection.get('parents')
            or not projection.get('independent_coefficients_checked')
            or not projection.get('saved_summary_checked')
            or projection.get('representation_error_certified') is not False):
        raise ValueError('SAVED_COMPLETION_POLICY_BINDING')
    eta=Decimal(projection['eta_N_hartree_decimal']['12'])
    if not eta.is_finite() or not Decimal(0)<=eta<=Decimal('9.9e-11'):
        raise ValueError('SAVED_COMPLETION_WEIGHTED_RECEIPT')
    rank=hm['df_rank_actual']
    if type(rank) is not int or rank!=19 or projection['actual_rank']!=rank:
        raise ValueError('SAVED_COMPLETION_RANK')
    layouts={'constant':((),'<f8'),'one_body':((12,12),'<c16'),'lambdas':((rank,),'<f8'),
        'g_matrices':((rank,12,12),'<c16'),'sector_basis_indices':((400,),'<i8'),
        'state_vector':((4096,),'<c16'),'sector_state_vector':((400,),'<c16')}
    if set(members)!=set(layouts)|{'metadata_json'}:
        raise ValueError('SAVED_COMPLETION_KEYS')
    for key,(shape,dtype) in layouts.items():
        if members[key][0]!={'shape':shape,'descr':dtype,'fortran_order':False}:
            raise ValueError('SAVED_COMPLETION_LAYOUT:'+key)
    header,raw=members['metadata_json']
    if header['shape']!=() or not header['descr'].startswith('<U') or json.loads(raw.decode('utf-32-le').rstrip('\0'))!=metadata:
        raise ValueError('SAVED_COMPLETION_METADATA')
    with np.load(path,allow_pickle=False) as z:
        a={key:np.array(z[key],copy=True) for key in layouts}
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('SAVED_COMPLETION_NONFINITE')
    if (array_record(a['lambdas'])!=projection['lambdas_raw']
            or array_record(a['one_body'])!=projection['corrected_one_body_after']
            or float(a['constant'])!=projection['constant']
            or len(projection['fragments'])!=rank):
        raise ValueError('SAVED_COMPLETION_TARGET_BYTES')
    for i,g in enumerate(a['g_matrices']):
        if projection['fragments'][i]['index']!=i or array_record(g)!=projection['fragments'][i]['after']:
            raise ValueError('SAVED_COMPLETION_FRAGMENT_BYTES')
        structure(g,post=True)
    structure(a['one_body'],post=True)
    ham=DFHamiltonian(float(a['constant']),a['one_body'],a['lambdas'],tuple(a['g_matrices']),hm)
    sector=PhysicalSector.spin_sector(n_qubits=12,nelec_alpha=3,nelec_beta=3)
    if not np.array_equal(a['sector_basis_indices'],sector.basis_indices):
        raise ValueError('SAVED_COMPLETION_SECTOR_ORDER')
    expected={'hamiltonian_hash':df_hamiltonian_hash(ham),'sector_hash':_sector_hash(sector),
        'state_vector_hash':_array_hash(a['state_vector']),
        'sector_state_vector_hash':_array_hash(a['sector_state_vector']),
        'state_hash':_state_hash(a['state_vector'],a['sector_state_vector'])}
    if any(metadata.get(k)!=v for k,v in expected.items()) or expected['hamiltonian_hash']!=df_receipt['hamiltonian_hash']:
        raise ValueError('SAVED_COMPLETION_INTERNAL_HASH')
    lifted=np.zeros(4096,dtype='<c16');lifted[sector.basis_indices]=a['sector_state_vector']
    if not np.array_equal(lifted,a['state_vector']) or abs(np.linalg.norm(lifted)-1)>1e-12:
        raise ValueError('SAVED_COMPLETION_STATE_BRIDGE')
    if metadata.get('ground_state_certified') is not False or metadata.get('numerical_allowance_certified') is not False:
        raise ValueError('SAVED_COMPLETION_UNSUPPORTED_CERTIFICATE')
    return ham,sector,a['state_vector'],a['sector_state_vector'],metadata,layouts
