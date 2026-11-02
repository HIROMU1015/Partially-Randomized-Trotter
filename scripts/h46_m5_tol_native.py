"""Tol-only saved-input bridge; original signed/native science code is reused."""
from pathlib import Path
from types import SimpleNamespace
import h46_m5_tol_contract as c
import h46_m5_native as original

CACHE = {}


def restore(config):
    """No SCF, DF decomposition or state solve; called only after authorization."""
    if config['system'] == 'H4':
        from h46_m5_tol_saved_input import load
        return load(dict(mapping=dict(
            tol_snapshot=config['inputs']['H4_tol_snapshot.npz']['path'],
            tol_receipt=config['inputs']['H4_tol_receipt.json']['path'])))
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
    result = load_h6_snapshot(Path(config['inputs']['H6_snapshot.npz']['path']),
        c.read(config['inputs']['H6_snapshot_receipt.json']['path']),
        c.read(config['inputs']['H6_df_receipt.json']['path']))
    return result[:5]


def context(config, *, reference_job=False):
    key = config['SOURCE'], config['system'], c.core.digest(config['inputs']), reference_job
    if key in CACHE:
        return CACHE[key]
    install(config)
    import numpy as np
    from h46_new2_native import OccupationOracle, prepare
    from h46_new2_core import spectrum, matrix_only_tail_coefficients
    ham, sector, full, state, md = restore(config)
    binding, system = config['binding'], c.core.SYSTEMS[config['system']]
    c.require(md['hamiltonian_hash'] == binding['hamiltonian']
        and md['state_hash'] == binding['state'], 'SAVED_HAMILTONIAN_STATE')
    c.require(ham.n_blocks == binding['rank'] == system['rank']
        and ham.n_qubits == (8 if config['system'] == 'H4' else 12), 'TOL_SAVED_LAYOUT')
    # The following numerical operations and order match the original context.
    fullprep = prepare(ham, ham.n_blocks)
    cut = prepare(ham, system['anchor'])
    oracle = OccupationOracle(ham, full, prefix=ham.n_blocks)
    qstate = np.zeros(1 << ham.n_qubits, complex)
    qstate[oracle.qindices] = oracle.initial
    c.require(abs(np.linalg.norm(qstate)-1) < 1e-12, 'SAVED_STATE_NORMALIZATION')
    matrices = [(spec[1]*spec[0]) @ spec[2] for spec in oracle.specs]
    reference_spec = None
    tail_identity, lam = matrix_only_tail_coefficients(ham, system['anchor'])
    c.require(abs(cut.extracted_identity_coefficient-tail_identity) < 1e-9
        and abs(cut.exact_rte_lambda_r-lam) < 1e-9, 'FULL_FOCK_SCALAR_LAMBDA')
    c.require(tuple(cut.deterministic_fragment_indices) == tuple(range(system['anchor']))
        and tuple(fullprep.deterministic_fragment_indices) == tuple(range(ham.n_blocks)), 'GENERATION_ORDER')
    tail = sum(matrices[system['anchor']+1:])-tail_identity*np.eye(len(oracle.initial))
    tail_spec = spectrum(tail)
    if reference_job:
        reference_spec = spectrum(sum(matrices))
        reference = reference_spec[1] @ (np.exp(-.8j*reference_spec[0]) * (reference_spec[2] @ oracle.initial))
        reference *= np.exp(-.8j*ham.constant)
    else:
        path = Path(config['reference_root']) / 'reference.npz'
        receipt = c.read(path.parent/'receipt.json')
        c.require(c.sha(path) == receipt['reference_sha256'], 'IMMUTABLE_FULL_H_REFERENCE_BYTES')
        with np.load(path, allow_pickle=False) as saved:
            reference = saved['reference'].copy()
            c.require(np.array_equal(saved['qstate'], qstate), 'SHARED_REFERENCE_STATE')
    CACHE[key] = SimpleNamespace(np=np, ham=ham, oracle=oracle, qstate=qstate,
        fullprep=fullprep, cut=cut, reference=reference, reference_spec=reference_spec,
        tail_spec=tail_spec, scalar=ham.constant, tail_identity=tail_identity, lambda_R=lam)
    return CACHE[key]


_shared = c.rebind(original, dict(c=c, CACHE=CACHE, context=context))
for _name in ('install', 'make_requests', 'compose', 'preflight', 'signal', 'run_task', '_run_task'):
    globals()[_name] = _shared[_name]
