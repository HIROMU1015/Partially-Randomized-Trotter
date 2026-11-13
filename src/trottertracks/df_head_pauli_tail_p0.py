"""P0 operator/dictionary diagnostics only: no propagation or time normalizer.

The scientific leaf is explicit. Saved factors are never Hermitized/refitted;
the full-Fock CAR action and M1's square-factor/JW construction are reused.
Qiskit little-endian matrices are permuted to the saved OpenFermion basis.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

PLAN = '75fb0ba114ac518bfe5d65d0164cee07d7469bb4'
MENU = 'a9abf8de75bf3432fcd714e868e5c14c8884b8f8'
M1 = '5e4f00487720dced86d53637464c48245a35a557'
SAVED = '155529e20a8fac39e00786c40b57830b71203c4f'
INDEX = dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
    path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
DIRECTORY = 'docs/research/pr_rpe_first_signal_menu_20261113/'
TOL = 1e-9
STATE_TOL = 1e-12


def blob(root, commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=root)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(root):
    """Only fixed JSON and file-byte binding; no molecular array load."""
    read = lambda commit, path: json.loads(blob(root, commit, path))
    configs = [read(MENU, DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json') for p in (1, 2)]
    binding = configs[0]['spec']['binding']
    for p, config in zip((1, 2), configs):
        s = config['spec']
        if (s['binding'] != binding or s['L_D'] != p or s['M'] != 12
                or s['method'] != 'PR' or s['formula'] != '2nd'
                or s['A_input']['kappa_paper'] != configs[0]['spec']['A_input']['kappa_paper']):
            raise ValueError('FIXED_LD1_LD2_METADATA')
    index = read(INDEX['commit'], INDEX['path'])
    inputs = index['saved_inputs']['H4']['inputs']
    context_spec = index['T20_reference_files']['H4']
    specs = [*inputs.values(), context_spec]
    if any(sha(v['path']) != v['sha256'] for v in specs):
        raise ValueError('SAVED_INPUT_OR_CONTEXT_BYTES_CHANGED')
    context = json.loads(Path(context_spec['path']).read_bytes())
    # The B configuration additionally records the later direct-expm
    # composition recipe. Compare every original context field, retaining the
    # extra recipe unchanged rather than relabeling the historical context.
    if (any(binding.get(k) != v for k, v in context['binding'].items())
            or context['sector_dimension'] != 36
            or binding['decomposition']['contract']['completion'] != 'saved_weighted_Hermitian_completion'
            or binding['decomposition']['snapshot'] != binding['snapshot']):
        raise ValueError('SAVED_CONTEXT_BINDING')
    tails = {str(p): deepcopy(context['prefixes'][str(p)]) for p in (1, 2)}
    if any(not math.isfinite(t[k]) for t in tails.values() for k in ('scalar', 'lambda_R')):
        raise ValueError('SAVED_TAIL_DESCRIPTOR')
    return dict(plan_source=PLAN, binding=binding, inputs=inputs,
        context_source=context_spec, original_context_binding=context['binding'], saved_tail_descriptors=tails,
        configuration_sources=[dict(commit=MENU, path=DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json') for p in (1, 2)],
        original_DF_configuration_ids=[c['id'] for c in configs],
        original_exact_kappa=configs[0]['spec']['A_input']['kappa_paper'],
        pauli_canonical_configuration=None, operator_gate=TOL, state_gate=STATE_TOL)


def fixed_function(root, commit, path, name, scope):
    # Reuse the established SOURCE-bound function extractor, without a runner.
    from .h4_pr_m1_rmse import function
    return function(root, commit, path, name, scope)


def load_saved(root, meta):
    """Unchanged save-only loader: receipt/array/H/state hashes and state lift."""
    def require(value, message):
        if not value:
            raise ValueError(message)
    scope = dict(Path=Path, c=SimpleNamespace(sha=sha,
        read=lambda path: json.loads(Path(path).read_bytes()), require=require))
    path = 'scripts/h46_m5_tol_saved_input.py'
    fixed_function(root, SAVED, path, 'array_record', scope)
    load = fixed_function(root, SAVED, path, 'load', scope)
    inputs = meta['inputs']
    values = load(dict(mapping=dict(tol_snapshot=inputs['H4_tol_snapshot.npz']['path'],
        tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    ham, sector, full_state, state, observed = values
    if (ham.n_qubits != 8 or ham.n_blocks != 10 or sector.dimension != 36
            or observed['hamiltonian_hash'] != meta['binding']['hamiltonian']
            or observed['state_hash'] != meta['binding']['state']):
        raise ValueError('SAVED_H4_LAYOUT_OR_IDENTITY')
    return values


def car_api(root):
    one = fixed_function(root, SAVED, 'scripts/h46_new2_core.py',
        'occupation_one_body', dict(np=np))
    # make_context imports the unchanged occupation action by its established
    # script name. Bind that module to the same frozen function, with no runner.
    module = SimpleNamespace(occupation_one_body=one)
    old = sys.modules.get('h46_new2_core')
    def context(ham, sector, state, tails):
        sys.modules['h46_new2_core'] = module
        try:
            make = fixed_function(root, PLAN, 'src/trottertracks/h6_df19_rpe_d2.py',
                'make_context', dict(np=np, SimpleNamespace=SimpleNamespace))
            return make(ham, sector, state, tails)
        finally:
            if old is None:
                sys.modules.pop('h46_new2_core', None)
            else:
                sys.modules['h46_new2_core'] = old
    return context


def pauli_api():
    # Avoid trotterlib's eager public API; this imports primitives, never builds
    # a circuit. The loader may already have imported the normal package.
    return importlib.import_module('trotterlib.pauli_signal_adapter')


def square_factor_dictionary(root, ham, prefix):
    """M1's existing one-body FermionOperator helper, restricted to DF tail."""
    from openfermion import FermionOperator, jordan_wigner
    path = 'src/trotterlib/o1_m1_matched_split.py'
    parent = next(n for n in ast.parse(blob(root, M1, path)).body
        if isinstance(n, ast.FunctionDef) and n.name == 'prepare_matched_a')
    one = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'one_body')
    scope = dict(np=np, Any=object, FermionOperator=FermionOperator)
    exec(compile(ast.Module(body=[one], type_ignores=[]), M1+':'+path, 'exec'), scope)
    residual = FermionOperator()
    for weight, g in zip(ham.lambdas[prefix:], ham.g_matrices[prefix:], strict=True):
        a = scope['one_body'](g)
        residual += float(weight)*(a*a)
    return jordan_wigner(residual)


def reverse_bits(n):
    return np.array([int(format(j, '0'+str(n)+'b')[::-1], 2) for j in range(2**n)])


def leakage(label, basis):
    """Exact bit action in the SAVED big-endian mode basis (no event sampling).

    Input label uses the existing Qiskit string convention. Z/Y phases cannot
    affect whether the image basis state leaves the declared sector.
    """
    n = len(label)
    mask = sum(1 << (n-1-j) for j, axis in enumerate(reversed(label)) if axis in 'XY')
    allowed = set(int(x) for x in basis)
    escaped = [(int(x), int(x)^mask) for x in basis if int(x)^mask not in allowed]
    return dict(saved_basis_flip_mask=mask, escapes_from_sector=len(escaped),
        preserves_sector=not escaped, escaping_basis_transitions=escaped)


def operator_metrics(matrix, basis):
    outside = np.setdiff1d(np.arange(len(matrix)), basis)
    return dict(hermiticity_fro=float(np.linalg.norm(matrix-matrix.conj().T)),
        sector_escape_fro=float(np.linalg.norm(matrix[np.ix_(outside, basis)])),
        sector_return_fro=float(np.linalg.norm(matrix[np.ix_(basis, outside)])))


def diagnose(root, ham, sector, full_state, state, meta, result, save):
    """Authorized P0 leaf, also exercised on nonmolecular artificial fixtures."""
    checks = result.setdefault('checks', [])
    def gate(name, value, tol=TOL):
        ok = math.isfinite(float(value)) and float(value) <= tol
        checks.append(dict(name=name, value=float(value), threshold=tol, PASS=ok))
        save('CHECKPOINT.json', result)
        if not ok:
            raise ValueError('P0_NUMERICAL_GATE: '+name)
    basis = np.asarray(sector.basis_indices, dtype=int)
    n = ham.n_qubits
    gate('saved_sector_state_norm', abs(np.linalg.norm(state)-1), STATE_TOL)
    gate('saved_full_state_norm', abs(np.linalg.norm(full_state)-1), STATE_TOL)
    lifted = np.zeros(2**n, complex); lifted[basis] = state
    gate('saved_state_full_lift', np.linalg.norm(full_state-lifted), STATE_TOL)
    context = car_api(root)
    reduced = context(ham, sector, state, meta['saved_tail_descriptors'])
    full_sector = SimpleNamespace(basis_indices=np.arange(2**n))
    full = context(ham, full_sector, full_state, meta['saved_tail_descriptors'])
    if 'scope' in result:
        result['scope']['full_Fock_extensions'] = 1
    # Preserve the actual extension even if a subsequent Hermiticity/sector
    # gate fails. No projection or symmetrization is used to repair it.
    save('COMMON_OPERATORS.npz', dict(blocks=np.asarray(full.matrices), full_H=full.full,
        sector_basis_indices=basis))
    result['common_operators'] = dict(dimension_full=2**n, dimension_sector=len(basis),
        scalar_constant=float(ham.constant), blocks=[])
    for j, (block, reference) in enumerate(zip(full.matrices, reduced.matrices, strict=True)):
        m = operator_metrics(block, basis)
        m['projection_fro'] = float(np.linalg.norm(block[np.ix_(basis, basis)]-reference))
        result['common_operators']['blocks'].append(dict(index=j, **m))
        for name, value in m.items():
            gate('block_'+str(j)+'_'+name, value)
    for name, value in operator_metrics(full.full, basis).items():
        gate('full_H_'+name, value)
    gate('full_H_projection_fro', np.linalg.norm(full.full[np.ix_(basis, basis)]-reduced.full))
    pauli = pauli_api()
    permutation = reverse_bits(n)
    for prefix in (1, 2):
        record = dict(prefix=prefix, status='STARTED', saved_DF_descriptor=deepcopy(meta['saved_tail_descriptors'][str(prefix)]))
        result.setdefault('prefixes', []).append(record)
        eye = np.eye(2**n, dtype=complex)
        head = sum(full.matrices[:prefix+1], np.zeros_like(eye))
        tail = sum(full.matrices[prefix+1:], np.zeros_like(eye))
        op = square_factor_dictionary(root, ham, prefix)
        raw = [dict(term=[[int(q), axis] for q, axis in term], real=float(complex(c).real),
                    imag=float(complex(c).imag)) for term, c in op.terms.items()]
        save('LD'+str(prefix)+'_JW_COEFFICIENTS.json', raw)
        record['JW_coefficient_imaginary_max'] = max((abs(v['imag']) for v in raw), default=0.)
        record['JW_coefficient_imaginary_L1'] = math.fsum(abs(v['imag']) for v in raw)
        model = pauli.PauliHamiltonian.from_openfermion(op, num_qubits=n,
            model_id='saved_DF10_tail_LD'+str(prefix), metadata=dict(
                source=M1, dictionary='GLOBAL_JW_COLLECTED_SAVED_DF_TAIL', prefix=prefix))
        little_matrix = model.sparse_operator(constant=0.).toarray()
        A = little_matrix[np.ix_(permutation, permutation)]
        lam = math.fsum(abs(c) for label, c in model.terms)
        if not math.isfinite(lam) or lam <= 0:
            raise ValueError('INVALID_PAULI_LAMBDA')
        rows = [dict(label_qiskit=label, label_in_saved_tensor_order=label[::-1], coefficient=c,
            coefficient_hex=c.hex(), sign=1 if c > 0 else -1, probability=abs(c)/lam,
            **leakage(label, basis)) for label, c in model.terms]
        payload = dict(prefix=prefix, binding=meta['binding'], lambda_P=lam,
            pauli_identity=model.constant, saved_DF_scalar=record['saved_DF_descriptor']['scalar'],
            mapping='OpenFermion mode q -> Qiskit qubit q; reverse_bits to saved mode-0-MSB basis',
            terms=rows, collector_metadata=json.loads(model.metadata_json),
            mapper_source=dict(commit=M1, path='src/trotterlib/o1_m1_matched_split.py',
                function='prepare_matched_a.one_body; prefix-restricted sum(weight*(a*a)); jordan_wigner'))
        dictionary_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        record.update(lambda_P=lam, pauli_identity=model.constant, terms=len(rows),
            individual_labels_leaving_sector=sum(not v['preserves_sector'] for v in rows),
            probability_sum=math.fsum(v['probability'] for v in rows),
            dictionary_sha256=hashlib.sha256(dictionary_bytes).hexdigest(),
            scalar_difference=float(model.constant-record['saved_DF_descriptor']['scalar']),
            head=operator_metrics(head, basis), tail=operator_metrics(tail, basis),
            Pauli_nonidentity=operator_metrics(A, basis))
        save('LD'+str(prefix)+'_DICTIONARY.json', payload)
        save('LD'+str(prefix)+'_OPERATORS.npz', dict(head=head, tail=tail, Pauli_nonidentity=A))
        for kind in ('head', 'tail', 'Pauli_nonidentity'):
            for name, value in record[kind].items():
                gate('LD'+str(prefix)+'_'+kind+'_'+name, value)
        metrics = dict(
            full_H_split_fro=np.linalg.norm(full.full-(head+tail+ham.constant*eye)),
            Pauli_tail_full_fro=np.linalg.norm(A+model.constant*eye-tail),
            Pauli_tail_saved_scalar_fro=np.linalg.norm(A+record['saved_DF_descriptor']['scalar']*eye-tail),
            Pauli_tail_sector_fro=np.linalg.norm((A+model.constant*eye)[np.ix_(basis, basis)]-
                (reduced.tails[prefix]['matrix']+record['saved_DF_descriptor']['scalar']*np.eye(len(basis)))),
            identity_full_trace_difference=abs(np.trace(tail)/(2**n)-model.constant),
            probability_sum_error=abs(record['probability_sum']-1))
        record['identity_metrics'] = {k:float(v) for k,v in metrics.items()}
        for name, value in metrics.items():
            gate('LD'+str(prefix)+'_'+name, value)
        record['status'] = 'P0_OPERATOR_DICTIONARY_GATES_PASS_NOT_P1_OR_RPE_QUALIFIED'
        save('CHECKPOINT.json', result)
    return result
