"""H8 B0/B1/B2 signal adapter; saved state, CSR and spin primitive unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from trotterlib.df_hamiltonian import df_linear_operator
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from .h8_signal_preflight_port_v1 import H8SignalPort, load_h8_snapshot
from .h8_sparse_reference_v1 import OccupationCSR, df_operator, reference_pair
from .ax2b_h4_science_v5 import primitive_sector_certificate, checked_basis_bridge
from .ax2b_molecular_ports_v3 import actual_bounds, check_bounds, complex_record
from .ax2a_preparation import digest
from .h8_resource_map_contract_v1 import coverage, cells

class H8CoreSignalPort(H8SignalPort):
    def setup(self):
        self.phase('input_reference')
        i = self.manifest['input_identity']
        self.ham, self.sector, full, state, md, _ = load_h8_snapshot(
            i['snapshot_path'], json.loads(Path(i['snapshot_receipt_path']).read_text()),
            json.loads(Path(i['df_receipt_path']).read_text()))
        cert = primitive_sector_certificate(self.ham, self.sector)
        self.qindices, self.qstate = checked_basis_bridge(self.ham, self.sector, full, state, 1e-12)
        self.state = state.copy(); self.saved_state = state.copy()
        self.tails = {}; self.oracle_tails = {}; self.spin = {}; self.primitive_ops = {}; self.truncated_references = {}
        cache = {}
        for c in self.cells:
            key = (c['method'] == 'B0', c['prefix'])
            if key not in cache:
                cache[key] = (_prepare_discard if key[0] else _prepare)(self.ham, c['prefix'])
            self.preps[c['id']] = cache[key]
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
            H8_full_campaign_authorized=True))
        del op, production
        self.primitives()

    def oracle_signal(self, c):
        value = super().oracle_signal(c)
        if c['method'] == 'B0':
            p = c['prefix']
            if p not in self.truncated_references:
                op, trace = df_operator(self.ham.constant, self.one,
                    self.ham.lambdas[:p], self.blocks[:p], self.calls)
                self.truncated_references[p] = reference_pair(op, trace, self.state, self.plan['T'])
            value['signals']['exact_truncated'], value['truncated_reference'] = self.truncated_references[p]
        return value


def signal_task(root, manifest, task, writer, progress):
    local = deepcopy(manifest)
    local['plan']['cells'] = [c for c in cells() if c['id'] in task['cell_ids']]
    local['plan']['coverage'] = coverage(local['plan']['cells'])
    port = H8CoreSignalPort(root, local, writer, progress)
    # Counted CSR operators enforce global calls. The per-exponential guard
    # checks actual deltas as well; source changes remain process-local.
    from . import h8_sparse_reference_v1 as sparse
    old = sparse.exponential_action
    original_take = port.calls.take
    action_start = [None]
    def checked_take(name, count=1):
        if (name == 'reference_matvec' and action_start[0] is not None
            and port.calls.used[name]+count-action_start[0] > local['plan']['caps_proposed']['reference_matvec_per_action']):
            raise ValueError('H8_REFERENCE_ACTION_CALL_CAP')
        original_take(name,count)
    port.calls.take = checked_take
    def bounded(op, trace, vector, t):
        before = port.calls.used['reference_matvec']
        previous=action_start[0];action_start[0]=before
        try:return old(op, trace, vector, t)
        finally:action_start[0]=previous
    sparse.exponential_action = bounded
    try:
        port.setup()
        port.correctness()
        writer.write('signal_result.json', dict(signal_count=port.completed,
            calls_attempted=port.calls.used, normalization_evaluations=sum(bool(c['R']) for c in port.cells),
            state_or_hamiltonian_regenerated=False, H8_full_campaign_authorized=True,
            numerical_allowance_certified=False))
    finally:
        sparse.exponential_action = old
