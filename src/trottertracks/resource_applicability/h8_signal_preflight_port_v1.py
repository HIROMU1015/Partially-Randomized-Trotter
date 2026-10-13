"""Authorized H8 reference/signal and eight cost groups; frozen science intact."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import numpy as np

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector, df_linear_operator
from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, df_hamiltonian_hash
from trotterlib.pr2_s0_s1_validation import _array_hash, _sector_hash, _state_hash
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import npz_bytes
from .h8_saved_projection_v1 import policy, structure, array_record
from .h8_signal_preflight_contract_v1 import INPUT_SHA, INPUT_BYTES
from .h4_common_panel_port_v1 import bound_function
from .h48_preflight_contract_v1 import task_caps
from .h6_matched_port_v1 import MatchedPort
from .ax2b_molecular_ports_v3 import MolecularPort, actual_bounds, check_bounds, complex_record
from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_h6_contract import primitive_time_schedule
from .ax2b_h6_controller import finite_scale_guard
from .h8_sparse_reference_v1 import (OccupationCSR, counted_operator, df_operator,
    SpinPrimitive, reference_pair, exponential_action, recurrence_paths)


def private_path(root, name):
    p = Path(name)
    if not p.is_absolute() or not p.is_relative_to('/home/AbeHiromu') or p.is_symlink():
        raise ValueError('H8_INPUT_PRIVATE_PATH')
    return p


def load_h8_snapshot(path, receipt, df):
    if receipt['sha256'] != INPUT_SHA or receipt['bytes'] != INPUT_BYTES:
        raise ValueError('H8_SNAPSHOT_IDENTITY')
    members = npz_bytes(Path(path), receipt, cap=16*2**20)
    md = receipt['metadata']; hm = md['hamiltonian_metadata']; proj = df['projection_receipt']
    if (hm != df['hamiltonian_metadata'] or proj['policy'] != policy() or proj['status'] != 'PASS_ENGINEERING'
        or hm['projection_receipt_digest'] != digest(proj)
        or hm['input_policy'] != policy()['name'] or proj['actual_rank'] != 28
        or proj['representation_error_certified'] is not False):
        raise ValueError('H8_PROJECTION_BINDING')
    eta = Decimal(proj['eta_N_hartree_decimal']['16'])
    if not eta.is_finite() or not Decimal(0) <= eta <= Decimal('9.9e-11'):
        raise ValueError('H8_PROJECTION_GATE')
    shapes = dict(constant=((), '<f8'), one_body=((16,16), '<c16'),
        lambdas=((28,), '<f8'), g_matrices=((28,16,16), '<c16'),
        sector_basis_indices=((4900,), '<i8'), state_vector=((65536,), '<c16'),
        sector_state_vector=((4900,), '<c16'))
    if set(members) != set(shapes)|{'metadata_json'}:
        raise ValueError('H8_SNAPSHOT_KEYS')
    for name, (shape, dtype) in shapes.items():
        if members[name][0] != dict(shape=shape, descr=dtype, fortran_order=False):
            raise ValueError('H8_SNAPSHOT_LAYOUT:'+name)
    header, raw = members['metadata_json']
    if header['shape'] != () or not header['descr'].startswith('<U') or json.loads(raw.decode('utf-32-le').rstrip('\0')) != md:
        raise ValueError('H8_SNAPSHOT_METADATA')
    with np.load(path, allow_pickle=False) as z:
        a = {name: z[name].copy() for name in shapes}
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('H8_SNAPSHOT_NONFINITE')
    if (array_record(a['lambdas']) != proj['lambdas_raw']
        or array_record(a['one_body']) != proj['corrected_one_body_after']
        or float(a['constant']) != proj['constant']):
        raise ValueError('H8_COEFFICIENT_RECEIPT')
    for i, g in enumerate(a['g_matrices']):
        if proj['fragments'][i]['index'] != i or array_record(g) != proj['fragments'][i]['after']:
            raise ValueError('H8_FRAGMENT_RECEIPT')
        structure(g, post=True)
    structure(a['one_body'], post=True)
    ham = DFHamiltonian(float(a['constant']), a['one_body'], a['lambdas'], tuple(a['g_matrices']), hm)
    sector = PhysicalSector.spin_sector(n_qubits=16, nelec_alpha=4, nelec_beta=4)
    hashes = dict(hamiltonian_hash=df_hamiltonian_hash(ham), sector_hash=_sector_hash(sector),
        state_vector_hash=_array_hash(a['state_vector']), sector_state_vector_hash=_array_hash(a['sector_state_vector']),
        state_hash=_state_hash(a['state_vector'], a['sector_state_vector']))
    if any(md[k] != v for k, v in hashes.items()) or hashes['hamiltonian_hash'] != df['hamiltonian_hash']:
        raise ValueError('H8_INTERNAL_HASH')
    if not np.array_equal(a['sector_basis_indices'], sector.basis_indices):
        raise ValueError('H8_SECTOR_ORDER')
    lifted = np.zeros(65536, dtype=complex); lifted[sector.basis_indices] = a['sector_state_vector']
    if not np.array_equal(lifted, a['state_vector']) or abs(np.linalg.norm(lifted)-1) > 1e-12:
        raise ValueError('H8_SAVED_STATE_BRIDGE')
    if md['ground_state_certified'] is not False or md['numerical_allowance_certified'] is not False:
        raise ValueError('H8_CERTIFICATE_FLAGS')
    return ham, sector, a['state_vector'], a['sector_state_vector'], md, shapes


class H8SignalPort(MatchedPort):
    def setup(self):
        self.phase('input_reference')
        i = self.manifest['input_identity']
        self.ham, self.sector, full, state, md, _ = load_h8_snapshot(
            i['snapshot_path'], json.loads(Path(i['snapshot_receipt_path']).read_text()),
            json.loads(Path(i['df_receipt_path']).read_text()))
        cert = primitive_sector_certificate(self.ham, self.sector)
        self.qindices, self.qstate = checked_basis_bridge(self.ham, self.sector, full, state, 1e-12)
        self.state = state.copy(); self.saved_state = state.copy()
        self.tails = {}; self.oracle_tails = {}; self.spin = {}; self.primitive_ops = {}
        cache = {}
        for c in self.cells:
            if c['prefix'] not in cache:
                cache[c['prefix']] = _prepare(self.ham, c['prefix'])
            self.preps[c['id']] = cache[c['prefix']]
        bounds = actual_bounds(self.cells, self.preps, T=self.plan['T']); check_bounds(bounds, self.caps)
        if digest([dict(cell_id=r['cell_id'], schedule=r['schedule']) for r in bounds['cells']]) != digest(self.plan['coverage']['cells']):
            raise ValueError('H8_ACTUAL_PRIMITIVE_COVERAGE')
        self.writer.write('prepared_representation.json', dict(bounds=bounds, unique_preparations=len(cache),
            preparations=[dict(cell_id=c['id'], preparation_hash=self.preps[c['id']].preparation_hash,
                lambda_r=self.preps[c['id']].exact_rte_lambda_r, constant=self.preps[c['id']].constant_coefficient,
                extracted_identity=self.preps[c['id']].extracted_identity_coefficient) for c in self.cells]))
        self.progress.update(point='independent_occupation_CSR_build')
        topology = OccupationCSR(self.sector.basis_indices, 16)
        self.one = topology.one(self.ham.one_body)
        self.blocks = tuple(topology.one(g) for g in self.ham.g_matrices)
        del topology
        op, trace = df_operator(self.ham.constant, self.one, self.ham.lambdas, self.blocks, self.calls)
        production, _ = df_linear_operator(self.ham, self.sector, **self.plan['matrix_free'])
        probes = [self.state, np.eye(1, len(state), 0, dtype=complex)[0],
                  np.eye(1, len(state), len(state)-1, dtype=complex)[0]]
        errors = []
        for x in probes:
            self.calls.take('reference_matvec'); errors.append(float(np.linalg.norm(production@x-op@x)))
        if max(errors) > 1e-10:
            raise ValueError('H8_PRODUCTION_CSR_AGREEMENT')
        self.reference, pair = reference_pair(op, trace, state, self.plan['T'])
        self.reference_error = max(errors)+pair['signal_difference']
        self.writer.write('input_reference.json', dict(metadata=md, sector_certificate=cert,
            reference=complex_record(self.reference), reference_agreement=pair,
            production_CSR_probe_errors=errors, reference_matvecs=self.calls.used['reference_matvec'],
            sparse_entries=sum(g.nnz for g in self.blocks)+self.one.nnz,
            state_renormalized=False, ground_state_certified=False, numerical_allowance_certified=False,
            H8_full_campaign_authorized=False))
        del op, production
        self.primitives()

    def key(self, block):
        return 'one' if isinstance(block, DFDeterministicOneBodySpec) else block.original_fragment_index

    def primitive(self, key):
        if key not in self.spin:
            a = self.ham.one_body if key == 'one' else self.ham.g_matrices[key]
            w = None if key == 'one' else self.ham.lambdas[key]
            self.spin[key] = SpinPrimitive(a, self.sector, weight=w)
        return self.spin[key]

    def oracle_primitive(self, key):
        if key not in self.primitive_ops:
            a = self.one if key == 'one' else self.blocks[key]
            w = None if key == 'one' else float(self.ham.lambdas[key])
            action = (lambda v: a@v) if w is None else (lambda v: w*(a@(a@v)))
            trace = float(np.real(a.diagonal().sum())) if w is None else w*float(np.sum(np.abs(a.data)**2))
            self.primitive_ops[key] = counted_operator(self.sector.dimension, action, trace, self.calls)
        return self.primitive_ops[key]

    def actions(self, prep, *, oracle=False):
        if not oracle:
            return [self.primitive(self.key(b)) for b in prep.deterministic_blocks]
        rows = []
        for b in prep.deterministic_blocks:
            op, trace = self.oracle_primitive(self.key(b))
            rows.append(lambda x, t, op=op, trace=trace: exponential_action(op, trace, x, t))
        return rows

    def primitives(self):
        self.phase('primitive_validation')
        visited = set(); records = []
        probes = [self.state, np.eye(1, len(self.state), 0, dtype=complex)[0],
                  np.eye(1, len(self.state), len(self.state)-1, dtype=complex)[0]]
        for c in self.cells:
            prep = self.preps[c['id']]
            native = MolecularPort.actions(self, prep); fast = self.actions(prep); oracle = self.actions(prep, oracle=True)
            for i, t in primitive_time_schedule(c, T=self.plan['T'])['unique_primitive_times']:
                key = self.key(prep.deterministic_blocks[i])
                if (str(key), t) in visited:
                    continue
                visited.add((str(key), t)); ne = []; se = []
                for x in probes:
                    self.calls.take('primitive'); a = fast[i](x, t); o = oracle[i](x, t); n = native[i](x, t)
                    ne.append(float(np.linalg.norm(n-a))); se.append(float(np.linalg.norm(a-o)))
                    finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=max(ne[-1], se[-1]))
                records.append(dict(block=str(key), time=t, native_spin_errors=ne, spin_CSR_errors=se))
                if len(records) % 20 == 0:
                    self.progress.update(point='native_16_mode_primitive', validated_times=len(records))
        if 3*len(visited) != self.plan['coverage']['primitive_actions']:
            raise ValueError('H8_NATIVE_PRIMITIVE_COVERAGE')
        self.writer.write('primitive_validation.json', dict(actual_time_count=len(visited), probe_count=3,
            records=records, max_native_spin_error=max(max(r['native_spin_errors']) for r in records),
            max_spin_CSR_error=max(max(r['spin_CSR_errors']) for r in records),
            includes_negative_yoshida_times=True, all_registered_times=True, certified=False))

    def oracle_signal(self, c):
        p = self.preps[c['id']]; tail = None
        if c['R']:
            if c['prefix'] not in self.oracle_tails:
                ids = p.randomized_block_indices; lam = p.exact_rte_lambda_r
                zero = self.one*0
                self.oracle_tails[c['prefix']] = df_operator(-p.extracted_identity_coefficient/lam, zero,
                    self.ham.lambdas[list(ids)]/lam, tuple(self.blocks[i] for i in ids), self.calls)
            tail = self.oracle_tails[c['prefix']]
        return recurrence_paths(self.state, self.actions(p, oracle=True), tail, cell=c, T=self.plan['T'],
            scalar=p.constant_coefficient+p.extracted_identity_coefficient,
            lambda_r=p.exact_rte_lambda_r if c['R'] else 0.,
            maximum_tail=self.caps['tail_matvecs_corrected_and_raw'].get(c['method'], 0))


def signal_port(root, manifest, writer, progress, *, port=None):
    port = H8SignalPort(root, manifest, writer, progress) if port is None else port
    port.setup(); port.correctness()
    writer.write('signal_result.json', dict(signal_count=port.completed, calls_attempted=port.calls.used,
        snapshot_sha256=INPUT_SHA, state_or_hamiltonian_regenerated=False,
        H8_full_campaign_authorized=False, numerical_allowance_certified=False))
    return port


def cost_port(root, manifest, task, writer, progress, trace):
    from . import h6_matched_cost_port_v3 as frozen
    from .h6_hash_reuse_v1 import use_fingerprint_reuse
    local = deepcopy(manifest)
    local['plan']['caps_proposed'] = task_caps(task, manifest['plan']['caps_proposed'])
    local['reuse_registry'] = {'missing': []}
    f = bound_function(frozen.cost_task, dict(load_h6_snapshot=load_h8_snapshot, safe_path=private_path))
    with use_fingerprint_reuse() as cache:
        f(Path(root), local, task, writer, progress, trace)
        writer.write('hash_reuse_receipt.json', cache.snapshot())
