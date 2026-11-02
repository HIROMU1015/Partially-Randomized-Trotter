"""Lazy connection to the fixed H6 provider; never imported science at startup."""
from copy import deepcopy
import importlib
import importlib.util
import csv
import io
import math
from pathlib import Path
import time

_spec = importlib.util.spec_from_file_location('h6_pilot_bridge',
    Path(__file__).with_name('h6_b0i_family_bridge_v1.py'))
b = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(b)
g = b.g
_identity_spec=importlib.util.spec_from_file_location('h6_preparation_identity',
    Path(__file__).with_name('h6_b0i_family_preparation_binding_v1.py'))
identity_binding=importlib.util.module_from_spec(_identity_spec);_identity_spec.loader.exec_module(identity_binding)
METRICS = dict(RZ='rz_count', CX='cx_count', RZ_depth='rz_depth', CX_depth='cx_depth',
    depth='total_depth', size='circuit_size')


def metrics_checked(row):
    if set(row) != set(METRICS) or any(type(row[m]) is not int or row[m] < 0 for m in METRICS):
        raise g.Stop('SIX_NONNEGATIVE_INTEGER_METRICS_REQUIRED')
    return row


def stored_independent(record):
    expected = record.get('independent_forward_signals', {}).get('corrected')
    if not isinstance(expected, dict) or set(expected) != {'real','imag'}:
        raise g.Stop('SAVED_INDEPENDENT_REFERENCE_REQUIRED')
    return complex(b.p.finite(expected['real']), b.p.finite(expected['imag']))


class NativeAdapter:
    def __init__(self, grant, verified, mapping, calls, guard):
        grant.require(b.SCOPE)
        if grant.synthetic: raise g.Stop('ARTIFICIAL_RUN_REQUIRES_MOCK_ADAPTER')
        self.grant, self.verified, self.mapping, self.calls, self.guard = grant, verified, mapping, calls, guard
        owner=getattr(guard,'__self__',None);previous_phase=getattr(owner,'phase',None)
        def phase(value):
            if owner is not None and previous_phase is not None:owner.phase=value
            guard()
        phase('A_IMPORT_NATIVE_PROVIDER')
        self.provider = b.native_provider(grant, grant.contract['native_provider']['checkout'])
        phase('A_IMPORT_NUMERICAL_PORT')
        self.np = importlib.import_module('numpy')
        self.port = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
        self.cost_api = importlib.import_module('trotterlib.rte_compiled_cost')
        self.fp_api = importlib.import_module('trottertracks.resource_applicability.ax2b_stream_fingerprint_v5')
        loader = importlib.import_module('trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1')
        self.prep = importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')
        snapshot = g.d0.strict_json(verified['snapshot_receipt.json'])
        for key in ('hamiltonian', 'state'):
            if snapshot['metadata'][key+'_hash'] != grant.contract['binding'][key]:
                raise g.Stop('SNAPSHOT_INTERNAL_IDENTITY')
        phase('A_SNAPSHOT_LOAD')
        calls.take('snapshot_load')
        self.ham, sector, full, state, _, _ = loader.load_h6_snapshot(
            g.home_path(mapping['h6_input_snapshot.npz'], existing=True), snapshot,
            g.d0.strict_json(verified['df_receipt.json']))
        if self.ham.n_qubits != 12 or len(self.ham.lambdas) != 19 or sector.dimension != 400:
            raise g.Stop('H6_DF19_SECTOR_BINDING')
        _, self.qstate = self.port.checked_basis_bridge(self.ham, sector, full, state, 1e-12)
        self.records = g.d0.strict_json(verified['signal_summary.json'])['signals']
        qiskit = importlib.import_module('qiskit')
        cp = grant.contract['compiler_options']
        self.compiler = self.port.CompilerSettings(tuple(cp['basis_gates']), None, None,
            cp['optimization_level'], None, None, cp['seed_transpiler'], qiskit.__version__)
        if self.cost_api.compiler_settings_hash(self.compiler) != grant.contract['binding']['compiler']:
            raise g.Stop('COMPILER_SETTINGS_HASH')
        if owner is not None and previous_phase is not None:owner.phase=previous_phase
        guard()

    def fingerprint(self, circuit):
        self.guard()
        value = self.fp_api.canonical_qiskit_circuit_fingerprint(circuit)
        if not isinstance(value, str) or len(value) != 64: raise g.Stop('NUMERIC_FINGERPRINT_REQUIRED')
        return value

    def prepare(self, prefix, q):
        self.guard()
        cid = f'B0_p{prefix}_q{q}_2nd'
        found = [r for r in self.records if r['cell']['id'] == cid]
        if len(found) != 1: raise g.Stop('SAVED_B0_RECORD_UNIQUE')
        record = found[0]
        # Require a separately saved forward oracle, never recompute an exp/action.
        z = stored_independent(record) if record.get('independent_forward_signals',{}).get('corrected') else None
        fragments = [dict(id=str(i), G=tuple(tuple(complex(z) for z in row) for row in matrix),
            **{'lambda':float(self.ham.lambdas[i])}) for i, matrix in enumerate(self.ham.g_matrices)]
        policy = self.grant.contract['scalar_error_policy'][f'{prefix}/{q}']
        preparation_audit={}
        def load_prepared(_):
            prepared=self.prep._prepare_discard(self.ham,prefix)
            if 'preparation_identity' in self.grant.contract:
                prepared,audit=identity_binding.checked_prepared(self.grant,prepared,prefix=prefix,q=q,bridge=b)
                preparation_audit.update(audit)
            return prepared
        request = b.prepare(self.grant, prefix=prefix, q=q,
            binding=b.binding_for(self.grant, prefix, q), load_fragments=lambda _:fragments,
            load_prepared=load_prepared, saved_record=record,
            scalar_input_error_upper=policy['stored_model_coefficient_error_upper'],
            rotation_roundoff_upper=policy['rotation_roundoff_upper'],
            input_mapping_loader=lambda:self.mapping, guard=self.guard)
        independent = b.p.rotate_saved_signal(z, total_time=.8,
            tail_identity=request['scalar_receipt']['C']) if z is not None else None
        return dict(request=request, independent=dict(real=independent.real, imag=independent.imag) if independent is not None else None,
            preparation_binding=preparation_audit)

    def build(self, case):
        return b.build(self.grant, case['request'], self.provider, guard=self.guard)

    def validate(self, case, built):
        np = self.np; dim = len(self.qstate); low = np.concatenate((self.qstate, np.zeros_like(self.qstate)))
        high = np.concatenate((np.zeros_like(self.qstate), self.qstate))
        self.calls.take('control_probe', 2); self.guard()
        left = self.port.simulate_statevector(built['evolution'].circuit, low)
        right = self.port.simulate_statevector(built['evolution'].circuit, high)
        if not np.isfinite(left).all() or not np.isfinite(right).all(): raise g.Stop('NONFINITE_CONTROL_PROBE')
        z = complex(np.vdot(self.qstate, right[dim:])); r = case['request']['signal']['signals']['corrected']
        independent = complex(case['independent']['real'], case['independent']['imag'])
        errors = [float(np.linalg.norm(left-low)), float(np.linalg.norm(right[:dim])),
            abs(float(np.linalg.norm(right))-1), abs(z-complex(r['real'], r['imag'])), abs(z-independent)]
        fingerprints = {}
        for axis, target in (('cosine', z.real), ('sine', z.imag)):
            wrapper = built['wrappers'][axis]
            fingerprints[axis] = self.fingerprint(wrapper)
            self.calls.take('probe_wrapper_copy'); probe = wrapper.remove_final_measurements(inplace=False)
            self.calls.take('control_probe'); self.guard()
            state = self.port.simulate_statevector(probe, low)
            if not np.isfinite(state).all(): raise g.Stop('NONFINITE_WRAPPER_PROBE')
            errors.append(abs(self.port.hadamard_expectation(state, 12)-target))
            del probe, state
        maximum = max(errors)
        self.port.finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=maximum)
        if not math.isfinite(maximum) or maximum > self.grant.contract['control_tolerance']:
            raise g.Stop('NATIVE_CONTROL_REFERENCE_MISMATCH')
        return dict(status='PASS', max_error=maximum, native_signal=dict(real=z.real, imag=z.imag),
            independent_saved_signal=case['independent'], fingerprints=fingerprints,
            reference_policy='SAVED_INDEPENDENT_FORWARD_OVERLAP_NOT_NEW_FULL_VECTOR_ORACLE', certified=False)

    def compile(self, wrapper, expected_fingerprint):
        before = self.fingerprint(wrapper)
        if before != expected_fingerprint: raise g.Stop('PRECOMPILE_FINGERPRINT_CHANGED')
        started = time.monotonic()
        cost = self.cost_api.transpile_and_measure_cost(wrapper, self.compiler,
            actual_circuit_fingerprint=expected_fingerprint)
        after = self.fingerprint(wrapper)
        if (cost.actual_circuit_fingerprint != expected_fingerprint or
                after != expected_fingerprint or
                cost.compiler_settings_hash != self.grant.contract['binding']['compiler']):
            raise g.Stop('COMPILED_INPUT_OR_COMPILER_BINDING')
        if len(cost.transpiled_circuit.data) > self.grant.contract['transpiled_instruction_cap']:
            raise g.Stop('TRANSPILED_INSTRUCTION_CAP')
        metrics = metrics_checked({m:getattr(cost, attr) for m,attr in METRICS.items()})
        self.guard()
        return dict(metrics=metrics, fingerprint=expected_fingerprint,
            input_fingerprint_check=dict(before=before,after=after),
            compiler_hash=cost.compiler_settings_hash, wall_seconds=time.monotonic()-started)

    def legacy_b0(self, prefix, q):
        rows = list(csv.DictReader(io.StringIO(self.verified['legacy_wrapper_metrics.csv'].decode())))
        selected = [r for r in rows if r['cell_id']==f'B0_p{prefix}_q{q}_2nd' and
            r['control']=='symmetric_directional' and r['compiler_hash']==self.grant.contract['binding']['compiler']]
        result = {}
        for axis in ('cosine','sine'):
            candidates = [r for r in selected if r['axis']==axis]
            if len(candidates) != 1:
                result[axis] = dict(status='MISSING_OR_NONUNIQUE',metrics=None)
            else:
                row = candidates[0]
                result[axis] = dict(status='SAVED_METADATA_REFERENCE',
                    metrics=metrics_checked({m:int(row[m]) for m in METRICS}),
                    source=row['source_commit'],fingerprint=row['fingerprint'],
                    note='Per-row profile/full numeric B0 match not acquired by this pilot; delta remains null')
        return result

    def verify_bindings(self):
        """End-of-run disk/Git check without importing or rebuilding science."""
        import subprocess
        provider = self.grant.contract['native_provider']; checkout = g.home_path(provider['checkout'])
        for name, expected in provider['source_hashes'].items():
            raw = g.read_bytes(checkout/name)
            fixed = subprocess.check_output(['git','-C',str(checkout),'show',provider['SOURCE']+':'+name])
            if raw != fixed or g.sha(raw) != expected: raise g.Stop('FINAL_NATIVE_PROVIDER_BINDING')
            self.guard()

    def close(self):
        self.records = []; self.qstate = None; self.ham = None
