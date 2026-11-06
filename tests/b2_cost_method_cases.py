"""Artificial two-mode B2 fixtures and independent CAR oracle; no file inputs."""
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator

from test_b2_rpe_connection import toy, context
from test_o1_m1_matched_split import annihilators, lifted_one
from trotterlib.b2_rpe_connection import B2RPERound, prepare_b2
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
from trotterlib.rpe_resource_accounting import RPERoundSpecification
from trotterlib.rte import enumerate_rte_events, event_unitary, make_rte_config
from trotterlib.rte_compiled_cost import canonical_qiskit_circuit_fingerprint
from trottertracks.resource_applicability.ax2a_native_df import NativeEvolution


def artificial_round(index, ld):
    old = toy()
    h = DFHamiltonian(constant=old.constant, one_body=old.one_body,
        lambdas=np.r_[old.lambdas, -.11],
        g_matrices=(*old.g_matrices, np.array([[.8,.17j],[-.17j,-.3]],complex)),
        metadata={'name':'ARTIFICIAL_TWO_MODE_B2_RANK4'})
    p = prepare_b2(h, L_D=ld)
    specification = RPERoundSpecification(index,.02)
    config, distribution = make_rte_config(p.rte_preparation.symbolic_tail,
        evolution_time=.02, rte_steps=4, finite_taylor_order=2, truncation_tolerance=1.)
    return h, B2RPERound(context(h),p,specification,ld,4,2,.02,1<<index,config,distribution)


def event_pattern(round_, pattern):
    pool = enumerate_rte_events(round_.tail.components,round_.distribution,max_events=10000)
    zero = [e for e in pool if e.taylor_order == 0]
    negative = next(e for e in pool if e.taylor_order == 2
                    and e.product_component_ids[0] != e.product_component_ids[1])
    template = (zero[0], negative, zero[-1], negative)
    for k in range(round_.q*round_.r):
        yield template[(k if pattern == 'repeat' else k+k//round_.r) % len(template)]


def expected_controlled(h, round_, events):
    a = annihilators(2)
    p = round_.preparation
    blocks = [lifted_one(h.one_body,a)] + [
        h.lambdas[i]*lifted_one(h.g_matrices[i],a)@lifted_one(h.g_matrices[i],a)
        for i in range(round_.L_D)]
    halves = [expm(-.5j*round_.delta*b) for b in blocks]
    tail = extraction_to_normalized_rte_tail(p.tail_extraction)
    component_ops = {c.component_id:o for c,o in zip(tail.components,tail.operators)}
    expected = np.eye(4,dtype=complex)
    for offset in range(0,len(events),round_.r):
        for half in halves:
            expected = half@expected
        for event in events[offset:offset+round_.r]:
            expected = event_unitary(event,component_ops)@expected
        for half in reversed(halves):
            expected = half@expected
    expected *= np.exp(-1j*round_.delta*round_.q*(h.constant+p.extracted_identity_coefficient))
    controlled = np.zeros((8,8),complex)
    controlled[:4,:4] = np.eye(4)
    controlled[4:,4:] = expected
    return controlled


def native_record(circuit, round_):
    return NativeEvolution(circuit,2,round_.delta*round_.q,round_.q,'partial_2nd',
        'ordinary',len(circuit.data),canonical_qiskit_circuit_fingerprint(circuit))


def measurement_error(reference, candidate):
    u = np.asarray(Operator(reference.remove_final_measurements(inplace=False)).data)
    v = np.asarray(Operator(candidate.remove_final_measurements(inplace=False)).data)
    z = np.diag([1.]*4+[-1.]*4)
    return dict(channel_max_abs=float(np.max(abs(np.kron(u.conj(),u)-np.kron(v.conj(),v)))),
        measured_Z_effect_max_abs=float(np.max(abs(u.conj().T@z@u-v.conj().T@z@v))))
