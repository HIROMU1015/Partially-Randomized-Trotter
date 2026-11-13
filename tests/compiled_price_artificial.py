"""Literal synthetic C1 context: no molecular files, sampler, or real H4 data.

Eight system wires let the unchanged H4 Accounting use its control wire 8.
The arbitrary matrices below are test inputs, not a hydrogen Hamiltonian.
"""
import ast
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = Path('/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/provider')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import trottertracks
if str(PROVIDER / 'src/trottertracks') not in trottertracks.__path__:
    trottertracks.__path__.insert(0, str(PROVIDER / 'src/trottertracks'))
sys.path.append(str(PROVIDER / 'src'))

import numpy as np
import qiskit
from trottertracks import p1_efficient_zz as P
from trottertracks import compiled_price_fast as F
from trottertracks import efficient_controlled_zz as E
from trottertracks.h4_df10_price_v1 import Accounting
from trottertracks.resource_applicability import ax2b_native_df_v5 as native
from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
from trotterlib import df_rte_qiskit as dfq
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2, DFPartialS2StepRequest
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
from trotterlib.rte import CompilerSettings, make_rte_config, _make_event
from trotterlib import rte_compiled_cost as costs


def fixture(q=2, R=4, *, scalar=.173, negative=False, mixed=True):
    mods = P.load_c1_backend(ROOT)
    science = mods['r_price_phase1_science']
    h = dict(np=np, math=math)
    tree = ast.parse(P._blob(ROOT, P.CARRIER, 'src/trotterlib/h46_m5_plan.py'))
    tail = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'TAIL' for t in n.targets))
    core = dict(math=math, TAIL=tail)
    science.functions(ROOT, P.CARRIER, 'scripts/h46_new2_core.py', ['allocation'], core)
    science.functions(ROOT, P.CARRIER, 'src/trotterlib/h46_m5_plan.py', ['weights', 'schedule'], core)
    h['core'] = SimpleNamespace(**{k: core[k] for k in ('weights', 'allocation', 'schedule')})
    science.functions(ROOT, P.CARRIER, 'scripts/h46_long_time_native.py', ['compose'], h, ('h46_m5_contract',))
    one = np.diag([.3, -.17, .09, -.23, .11, .07, -.13, .19]).astype(complex)
    one[0, 1], one[1, 0] = .037j, -.037j
    one[2, 3] = one[3, 2] = -.021
    g1 = np.diag([.7, -.4, .13, .8, -.2, .31, .17, -.43])
    g2 = np.diag([.2, .9, -.7, .4, .23, -.16, .37, -.51])
    g2[0, 1] = g2[1, 0] = .043
    ham = DFHamiltonian(constant=scalar, one_body=one, lambdas=np.array([.23, -.37]),
                        g_matrices=(g1, g2), metadata={'synthetic': True, 'not_molecular': True})
    science.functions(ROOT, P.CARRIER, 'scripts/h46_new2_native.py', ['gaussian_operations'], h)
    from trotterlib.df_trotter import ops
    previous = ops.U_to_qiskit_ops_jw
    try:
        ops.U_to_qiskit_ops_jw = h['gaussian_operations']
        prep = prepare_df_partial_s2(ham, split_df_hamiltonian_by_ld(ham, 1))
    finally:
        ops.U_to_qiskit_ops_jw = previous
    dt = -.125 if negative else .125
    config, dist = make_rte_config(prep.rte_preparation.symbolic_tail, evolution_time=dt,
        rte_steps=R, truncation_tolerance=1., finite_taylor_order=2)
    comps = prep.rte_preparation.symbolic_tail.components
    # Explicit event schedule exercises Z/ZZ, product signs, phase and basis reuse.
    chosen = [i for support in (1, 2) for i, c in enumerate(comps)
              if c.diagonal_pauli_support is not None and len(c.diagonal_pauli_support) == support]
    requests = []
    for outer in range(q):
        events = []
        for j in range(R):
            rot = chosen[(j + 3 * outer) % len(chosen)]
            indices = (rot, chosen[-1], chosen[0]) if mixed and (j + outer) % 5 == 2 else (rot,)
            events.append(_make_event(indices, comps, dist, (len(indices) - 1) // 2))
        occurrence = DFRTEEventSequenceCircuitRequest(events=tuple(events),
            component_specs=prep.rte_preparation.component_specs, controlled=True, ancilla_qubit=8,
            tail_id=config.tail_id, tail_hash=config.tail_hash, occurrence_rte_steps=R)
        requests.append(DFPartialS2StepRequest(prep, dt, config, dist, occurrence,
            controlled=True, ancilla_qubit=8, seed=0))  # required metadata; never used to sample
    blocks = []
    for b in prep.deterministic_blocks:
        diag = (one_body_diagonal_primitives(np.array(b.diagonal_eigenvalues), dt / 2)
                if b.block_kind == 'one_body' else
                df_squared_diagonal_primitives(np.array(b.diagonal_eta), b.lam, dt / 2))
        basis = [dict(name=o.name, qubits=list(qs), parameters=[str(t) for t in o.params])
                 for o, qs in b.runtime_basis_operations]
        blocks.append(dict(basis=basis, diagonal=diag))
    ctx = SimpleNamespace(prep=prep, h=h, scalar=scalar, blocks=[blocks], Accounting=Accounting,
        native=native, fingerprint_context=use_fingerprint_reuse, CompilerSettings=CompilerSettings,
        compile=costs.transpile_and_measure_cost)
    c = dict(system='H4', id='SYNTHETIC_q%d_R%d' % (q, R), q=q, R_outer=R,
             T=dt * q, formula='2nd', method='PR', prefix=1, K=2, delta=dt)
    cell = dict(id=c['id'], candidate_id=c['id'], system='H4', cohort='ARTIFICIAL_ONLY', n=1)
    plan = dict(source='ARTIFICIAL_TECHNICAL_TEST', menus=dict(H4=dict(candidates=[
        dict(candidate=c, structure=dict(allocation=[R], signed_weights=[1.]))])),
        runtime=dict(old_cache='SET_BY_TEST'))
    return science, ctx, c, cell, plan, requests


def backend(science, ctx, plan, fast=False):
    cls = F.make_backend_class(science.Backend, science) if fast else science.Backend
    value = cls(ROOT, plan)
    value.contexts['H4'] = ctx  # avoids all historical context/input-loading code
    return value


@contextmanager
def compile_local():
    original = costs.transpile
    costs.transpile = partial(qiskit.transpile, num_processes=1)
    try:
        yield
    finally:
        costs.transpile = original
