"""Frozen save-only H4 loader from F/FP; no generation entry point.

Original: 97d25cc431516d720ce91c7cb4f148e061d8ea34:scripts/h4_tol_only_input.py
Function bodies are reused unchanged. Imports are bound to the new contract.
"""
from pathlib import Path
import h46_m5_tol_contract as c


def array_record(a):
    import numpy as np
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest())


def load(config):
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
    from trotterlib.df_partial_s2 import df_hamiltonian_hash
    from trotterlib.pr2_s0_s1_validation import _state_hash
    path=Path(config['mapping']['tol_snapshot']);receipt=c.read(config['mapping']['tol_receipt'])
    c.require(c.sha(path)==receipt['sha256'] and receipt['status']=='COMPLETE_MANDATORY_STOP','TOL_INPUT_COMPLETION')
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    c.require({k:array_record(v) for k,v in arrays.items()}==receipt['arrays'],'TOL_ARRAY_IDENTITY')
    ham=DFHamiltonian(float(arrays['constant']),arrays['one_body'],arrays['lambdas'],tuple(arrays['g_matrices']),receipt['hamiltonian_metadata'])
    sector=PhysicalSector.spin_sector(n_qubits=8,nelec_alpha=2,nelec_beta=2)
    full,state=arrays['state_vector'],arrays['sector_state_vector']
    c.require(np.array_equal(arrays['sector_basis_indices'],sector.basis_indices) and np.array_equal(full[sector.basis_indices],state),'TOL_STATE_LIFT')
    c.require(abs(np.linalg.norm(state)-1)<1e-12 and df_hamiltonian_hash(ham)==receipt['hamiltonian_hash'] and
              _state_hash(full,state)==receipt['state_hash'],'TOL_HAMILTONIAN_STATE_IDENTITY')
    return ham,sector,full,state,dict(hamiltonian_hash=receipt['hamiltonian_hash'],state_hash=receipt['state_hash'],rank=receipt['rank'],projection=receipt['projection'])
