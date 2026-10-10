"""Future one-shot H6 input port. Never called by preparation or synthetic CLI."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

from .ax2b_h6_input import tol_only_df_from_integrals
from .ax2b_h6_controller import bounded_solver
from .ax2b_limits import CallBudget, output_size
from .ax2b_h4_science_v5 import primitive_sector_certificate
from .ax2b_h6_input_generation_contract_v1 import plan
from trotterlib.df_hamiltonian import PhysicalSector, df_linear_operator, _hartree_fock_initial_vector
from trotterlib.df_partial_s2 import df_hamiltonian_hash
from trotterlib.pr2_s0_s1_validation import _array_hash, _sector_hash, _state_hash, _canonicalize_state_phase


def obtain_integrals(output, policy):
    """Reuse installed provider steps without its implicit MolecularData.save."""
    from openfermion import MolecularData
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf._run_pyscf import prepare_pyscf_molecule, compute_scf, compute_integrals
    from pyscf import lib
    lib.num_threads(1)
    target = plan()['target']
    molecule = MolecularData([(atom, tuple(xyz)) for atom, xyz in target['geometry']],
                            target['basis'], target['multiplicity'], target['charge'],
                            filename=str(Path(output)/'scratch'/'molecule_not_saved'))
    mol = prepare_pyscf_molecule(molecule)
    if str(mol.unit).lower() != 'angstrom' or mol.symmetry or mol.nao_nr() != 6 or mol.nelectron != 6:
        raise ValueError('MOLECULAR_INPUT_SCOPE')
    mean_field = compute_scf(mol)
    mean_field.verbose = 0
    mean_field.conv_tol, mean_field.max_cycle = policy['conv_tol'], policy['max_cycle']
    mean_field.chkfile = str(Path(output)/'scratch'/'scf.chk')
    mean_field.kernel()
    if not mean_field.converged or not math.isfinite(float(mean_field.e_tot)):
        raise ValueError('SCF_NOT_CONVERGED')
    one, two = compute_integrals(mol, mean_field)
    h1, h2 = spinorb_from_spatial(one, two)
    return {'constant': float(mol.energy_nuc()), 'one_body': np.asarray(h1, dtype=np.complex128),
            'two_body': np.asarray(.5*h2, dtype=np.complex128), 'spatial_one_body': one,
            'spatial_two_body': two, 'canonical_orbitals': np.asarray(mean_field.mo_coeff),
            'hf_energy': float(mean_field.e_tot), 'scf_converged': True,
            'scf_conv_tol': float(mean_field.conv_tol), 'scf_max_cycle': int(mean_field.max_cycle),
            'scf_cycles': int(mean_field.cycles) if getattr(mean_field, 'cycles', None) is not None else None}


def atomic_npz(writer, name, arrays, *, expanded_cap):
    """Bounded uncompressed NPZ, exclusive publication; keep failed bytes for audit."""
    if Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('SNAPSHOT_NAME')
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError('SNAPSHOT_OBJECT_ARRAY')
    expanded = sum(np.asarray(a).nbytes+65536 for a in arrays.values())
    if expanded > expanded_cap:
        raise RuntimeError('SNAPSHOT_EXPANDED_CAP')
    if output_size(writer.output)+expanded > writer.byte_cap-writer.reserve:
        raise RuntimeError('SNAPSHOT_OUTPUT_CAP')
    pending = writer.output/('.pending_'+name)
    with pending.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if pending.stat().st_size > expanded or output_size(writer.output) > writer.byte_cap-writer.reserve:
        raise RuntimeError('SNAPSHOT_WRITTEN_CAP')
    os.link(pending, writer.output/name)
    pending.unlink()
    return {'file': name, 'sha256': hashlib.sha256((writer.output/name).read_bytes()).hexdigest(),
            'bytes': (writer.output/name).stat().st_size,
            'arrays': {k: {'shape': list(np.asarray(a).shape), 'dtype': np.asarray(a).dtype.str,
                          'sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
                       for k, a in arrays.items()}}


class InputGenerationPort:
    def __init__(self, manifest, writer, progress, *, grant_sha256,
                 integrals=obtain_integrals, decomposer=None, solver=None, operator_factory=None):
        self.manifest, self.writer, self.progress = manifest, writer, progress
        self.plan, self.caps = manifest['plan'], manifest['plan']['caps_proposed']
        self.grant_sha256 = grant_sha256
        self.integrals, self.decomposer, self.solver = integrals, decomposer, solver
        self.operator_factory = operator_factory
        self.calls = CallBudget(**{k:self.caps[k] for k in
                                  ('integral_build','df_decomposition','solver_matvec','trajectory','occurrence','compile')})
        self.completed = dict.fromkeys(('integral_build','df_decomposition','solver_matvec'), 0)

    def phase(self, name):
        self.writer.write('phase_'+name+'.json', {'phase': name, 'elapsed': time.monotonic()-self.progress.started})
        self.progress.update(phase=name, point='phase_start', calls_attempted=dict(self.calls.used),
                             calls_completed=dict(self.completed))

    def execute(self):
        self.phase('integrals')
        self.calls.take('integral_build')
        self.progress.update(point='integrals_started', calls_attempted=dict(self.calls.used))
        payload = self.integrals(self.writer.output, self.plan['integrals'])
        self.completed['integral_build'] += 1
        one, two = np.asarray(payload['one_body']), np.asarray(payload['two_body'])
        if (one.shape != (12,12) or two.shape != (12,)*4 or not np.isfinite(one).all()
                or not np.isfinite(two).all() or not math.isfinite(payload['constant'])
                or not math.isfinite(payload['hf_energy'])
                or payload.get('scf_converged') is not True
                or payload.get('scf_conv_tol') != 1e-9 or payload.get('scf_max_cycle') != 50):
            raise ValueError('INTEGRAL_PAYLOAD_SCOPE_OR_CONVERGENCE')
        integral_arrays = {k: np.ascontiguousarray(payload[k], dtype='<c16') for k in
                           ('one_body','two_body','spatial_one_body','spatial_two_body','canonical_orbitals')}
        integral_arrays['constant'] = np.array(payload['constant'], dtype='<f8')
        receipt = atomic_npz(self.writer, 'integrals.npz', integral_arrays,
                             expanded_cap=self.caps['snapshot_expanded_bytes'])
        self.writer.write('integral_receipt.json', {**receipt, 'hf_energy': payload['hf_energy'],
                          'scf_converged': True, 'scf_conv_tol': payload['scf_conv_tol'],
                          'scf_max_cycle': payload['scf_max_cycle'], 'scf_cycles': payload.get('scf_cycles'),
                          'convention': self.plan['integrals']})
        self.progress.update(point='integrals_saved', last_completed_record='integral_receipt.json',
                             calls_completed=dict(self.completed))
        self.phase('df_decomposition')
        self.calls.take('df_decomposition')
        self.progress.update(point='df_started', calls_attempted=dict(self.calls.used))
        ham = tol_only_df_from_integrals(constant=payload['constant'], one_body=one, two_body=two,
                    df_tol=1e-8, hermitization_tolerance=1e-10, decomposer=self.decomposer)
        self.completed['df_decomposition'] += 1
        if not 2 <= ham.n_blocks <= 144:
            raise ValueError('ACTUAL_RANK_OUTSIDE_REGISTERED_INPUT_SCOPE')
        sector = PhysicalSector.spin_sector(n_qubits=12, nelec_alpha=3, nelec_beta=3)
        if sector.dimension != 400:
            raise ValueError('H6_SECTOR_DIMENSION')
        certificate = primitive_sector_certificate(ham, sector)
        self.writer.write('df_receipt.json', {'hamiltonian_metadata':ham.metadata,
                          'actual_rank': ham.n_blocks, 'hamiltonian_hash': df_hamiltonian_hash(ham),
                          'sector_certificate': certificate, 'sector_hash': _sector_hash(sector)})
        self.progress.update(point='df_saved', last_completed_record='df_receipt.json',
                             calls_completed=dict(self.completed))
        self.phase('state_snapshot')
        initial = _hartree_fock_initial_vector(sector)
        if initial is None:
            raise ValueError('FIXED_HF_INITIAL_VECTOR_MISSING')
        if self.operator_factory is None:
            operator, _counter = df_linear_operator(ham, sector, backend='numba', num_threads=1, block_chunk_size=1)
        else:
            operator = self.operator_factory(ham, sector)
        def action(vector):
            # bounded_solver increments the shared before-call budget first.
            number = self.calls.used['solver_matvec']
            if number == 1 or number % 100 == 0:
                self.progress.update(point='solver_matvec_started', calls_attempted=dict(self.calls.used),
                                     calls_completed=dict(self.completed))
            result = np.asarray(operator @ vector, dtype=np.complex128)
            if result.shape != (400,) or not np.isfinite(result).all():
                raise ValueError('STATE_MATVEC_RESULT')
            self.completed['solver_matvec'] += 1
            if number == 1 or number % 100 == 0:
                self.progress.update(point='solver_matvec_completed', calls_attempted=dict(self.calls.used),
                                     calls_completed=dict(self.completed))
            return result
        if self.solver is None:
            from scipy.sparse.linalg import eigsh
            self.solver = eigsh
        (values, vectors), bounded, counter = bounded_solver(action, 400, initial, self.calls, solver=self.solver)
        values, vectors = np.asarray(values), np.asarray(vectors)
        if (values.shape != (1,) or vectors.shape != (400,1) or not np.isfinite(values).all()
                or not np.isfinite(vectors).all() or np.any(np.imag(values) != 0)):
            raise ValueError('STATE_SOLVER_RESULT')
        energy = float(np.real(values[0]))
        state = np.array(vectors[:,0], dtype='<c16', copy=True)
        norm_before = float(np.linalg.norm(state))
        if not math.isfinite(norm_before) or norm_before <= 0:
            raise ValueError('STATE_SOLVER_NORM')
        state /= norm_before
        full = np.zeros(4096, dtype='<c16')
        full[sector.basis_indices] = state
        full, state = _canonicalize_state_phase(full, state)
        residual = float(np.linalg.norm(bounded @ state-energy*state))
        threshold = 1e-9+1e-10*max(1., abs(energy))
        if not math.isfinite(residual) or residual > threshold:
            raise ValueError('STATE_RESIDUAL_GATE')
        if abs(float(np.linalg.norm(state))-1.) > 1e-12 or abs(float(np.linalg.norm(full))-1.) > 1e-12:
            raise ValueError('SAVED_STATE_NORM')
        metadata = {'model':'linear_H6', 'geometry_angstrom':1., 'basis':'sto-3g',
                    'hamiltonian_metadata':ham.metadata,
                    'sector':{'n_qubits':12,'nelec_alpha':3,'nelec_beta':3,'dimension':400},
                    'sector_hash':_sector_hash(sector), 'hamiltonian_hash':df_hamiltonian_hash(ham),
                    'state_vector_hash':_array_hash(full),'sector_state_vector_hash':_array_hash(state),
                    'state_hash':_state_hash(full,state), 'sector_certificate':certificate,
                    'state_policy':self.plan['state_policy'], 'solver_energy':energy,
                    'global_phase_policy':'largest_sector_amplitude_real_positive_v1',
                    'solver_residual':residual, 'solver_residual_threshold':threshold,
                    'solver_matvec_calls_including_residual':counter.used['calls'],
                    'state_norm_before_final_normalization':norm_before,
                    'ground_state_certified':False, 'numerical_allowance_certified':False,
                    'accuracy_eligibility':'UNDETERMINED', 'N':None, 'G':None,
                    'input_generation':{'source_commit':self.manifest['source_commit'],
                        'authorization_sha256':self.grant_sha256, 'retry':False, 'resume':False},
                    'H6_status':'H6_NOT_AUTHORIZED', 'contract_status':'DRAFT_NOT_AUTHORIZATION'}
        arrays = {'constant':np.array(ham.constant,dtype='<f8'),
                  'one_body':np.ascontiguousarray(ham.one_body,dtype='<c16'),
                  'lambdas':np.ascontiguousarray(ham.lambdas,dtype='<f8'),
                  'g_matrices':np.ascontiguousarray(ham.g_matrices,dtype='<c16'),
                  'sector_basis_indices':np.ascontiguousarray(sector.basis_indices,dtype='<i8'),
                  'state_vector':full,'sector_state_vector':state,
                  'metadata_json':np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))}
        saved = atomic_npz(self.writer,'h6_input_snapshot.npz',arrays,
                           expanded_cap=self.caps['snapshot_expanded_bytes'])
        # Schema/hash compatibility check only; no signal/circuit computation.
        from .ax2b_molecular_ports_v3 import load_h6_snapshot
        load_h6_snapshot(self.writer.output/'h6_input_snapshot.npz', metadata)
        self.writer.write('snapshot_receipt.json', {**saved,'metadata':metadata,
                          'loader_roundtrip_checked':True,'calls_attempted':dict(self.calls.used),
                          'calls_completed':dict(self.completed),'mandatory_stop':True,
                          'next_stage_authorized':False,'H6_status':'H6_NOT_AUTHORIZED'})
        self.progress.update(point='snapshot_saved', last_completed_record='snapshot_receipt.json',
                             calls_attempted=dict(self.calls.used), calls_completed=dict(self.completed))
