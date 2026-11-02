"""Repair exact basis-key reconstruction; preserve the v1 SOURCE unchanged."""
from .c_df_basis_surrogate import (
    MODELS, atoms_from_records, law_features, overlap, trajectory_features,
    describe_models, qualification,
)

VERSION='basis_run_atom_descriptors_existing_spin_lift_v2'


def geometry(atoms,U):
    """Use the exact existing producer's tuple, including its 0j convention."""
    import numpy as np,hashlib
    from .c1_general_k_adapter import _matrix,_spin_lift
    U=np.asarray(U);pairs={};frame_basis={};matrices={}
    if U.shape!=(8,4,4) or U.dtype!=np.float64:raise ValueError('FIXED_CDF8_U_SCHEMA')
    for a in atoms:
        if a.frame_id not in frame_basis:
            leaf=a.frame_id.rsplit(':',1)[-1]
            if leaf not in ('F2','F3','F4','F5','F6','F7'):raise ValueError('ORIGINAL_TAIL_FRAME')
            mat=U[int(leaf[1:])];overlap(mat,mat)
            rotation=_spin_lift(_matrix(mat,'saved_U'))
            key=hashlib.sha256(repr(rotation).encode()).hexdigest()
            if key!=a.basis_key:raise ValueError('EXACT_SAVED_BASIS_KEY')
            frame_basis[a.frame_id]=key;matrices[a.frame_id]=mat
        if frame_basis[a.frame_id]!=a.basis_key:raise ValueError('FRAME_BASIS_CONFLICT')
    for l,u in matrices.items():
        for k,v in matrices.items():pairs[l,k]=overlap(u,v)
    return pairs
