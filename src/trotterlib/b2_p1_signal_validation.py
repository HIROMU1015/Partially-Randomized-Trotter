"""Fixed P1 two-round pilot. Explicit inputs only; no shots, prices or solves.

All molecular I/O and circuit/mean evaluation occurs inside run(), never import.
Private arrays stay in memory. Results contain scalar signals and diagnostics.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from . import b2_rpe_connection as connection
from . import pr2_matched_accuracy_m1_execution as legacy
from .df_hamiltonian import PhysicalSector
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_rte_tail import basis_change_unitary, diagonal_pauli_matrix
from .h2_non_df_pilot import reverse_bits
from .h4_molecular_inputs import spin_lift_unthresholded
from .o1_m1_circuit_signal_validation import controlled_reference, representative_events
from .o1_m1_resource_bridge import radius_phase_diagnostics
from .pauli_signal_adapter import paired_polynomial_action
from .pr2_s0_s1_validation import _array_hash, load_snapshot
from .rpe_resource_accounting import RPEErrorAllocation, RPERoundSpecification
from .rte import (CompilerSettings, event_unitary, finite_rte_operator_moments,
                  finite_taylor_operator, make_rte_config)
from trottertracks.resource_applicability import ax2a_native_df as native

ROOT = Path(__file__).resolve().parents[2]
STATE_RESULT = '89165cf70958e7e4676d0580e3acb7faf588772c'
OPERATOR_RESULT = '275813d0a12a4bca62a8c972312c80386829de22'
DF_COMMIT = 'bf5d3b2405b2bec4f88dd0589b75dd737b13e052'
DF_PATH = 'artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz'
STATE_SHA = 'deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c'
HDF_SHA = 'a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958'
DF_SHA = '3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a'
DELTA, R, K, LD = .02, 4, 2, 3
U = 1e-8
MATCH = U/10
CAPS = dict(model=.02, PF=.06, RTE=.04, statistical=.24, implementation=.02, reserve=.02)
ALLOCATION = RPEErrorAllocation(.06, .04, .24, .05/(2*18), .05/(2*18))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def public_json(commit, path):
    return json.loads(git('show', commit+':'+path))


def car_reference():
    path = ROOT/'scripts/run_o1_m1_h4_operator_correctness.py'
    spec = importlib.util.spec_from_file_location('b2_p1_existing_car', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, name):
    if not condition:
        raise ValueError(name)


def phase_distance(left, right):
    if abs(left) == 0 or abs(right) == 0:
        return None
    z = left*right.conjugate()
    return abs(math.atan2(z.imag, z.real))


def signal_record(z):
    return dict(real=float(z.real), imag=float(z.imag), radius=float(abs(z)),
                phase_rad=float(np.angle(z)))


def signal_difference(left, right):
    return dict(complex_absolute=float(abs(left-right)),
                radius_absolute=float(abs(abs(left)-abs(right))),
                phase_rad=phase_distance(left, right))


def load_private_inputs(private_dir):
    """Read existing files, verify exact provenance; never substitute a DF state."""
    from openfermion import MolecularData
    private_dir = Path(private_dir)
    files = {name: private_dir/name for name in
             ('state.npz', 'provenance.json', 'H4_sto-3g_singlet_df_d100.hdf5')}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    require(hashes['state.npz'] == STATE_SHA, 'STATE_FILE_SHA256')
    require(hashes['H4_sto-3g_singlet_df_d100.hdf5'] == HDF_SHA, 'HDF5_FILE_SHA256')
    prior = public_json(STATE_RESULT, 'artifacts/b2_hphys_ground_state/2026-11-06/run01/result.json')
    require(json.loads(files['provenance.json'].read_text()) == prior, 'STATE_PROVENANCE')
    spec = public_json(STATE_RESULT, 'artifacts/b2_hphys_ground_state/2026-11-06/run01/agent_b_input_spec.json')
    operator_prior = public_json(OPERATOR_RESULT, 'artifacts/b2_original_df12_correspondence/2026-11-06/run03/result.json')
    with np.load(files['state.npz'], allow_pickle=False) as payload:
        state = payload['state_vector'].copy()
        sector_state = payload['sector_state_vector'].copy()
        of_indices = payload['sector_basis_indices'].copy()
        indices = payload['qiskit_full_vector_indices'].copy()
        energy = float(payload['energy'])
    require(state.shape == (256,) and state.dtype.str == '<c16', 'FULL_STATE_SHAPE_DTYPE')
    require(sector_state.shape == (36,) and sector_state.dtype.str == '<c16', 'SECTOR_STATE_SHAPE_DTYPE')
    require(of_indices.shape == indices.shape == (36,) and
            of_indices.dtype.str == indices.dtype.str == '<i8', 'BASIS_SHAPE_DTYPE')
    require('S0_array_hash:'+_array_hash(state) == spec['original_state'], 'STATE_ARRAY_HASH')
    sector = PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    require(np.array_equal(of_indices, sector.basis_indices), 'OF_SECTOR_BASIS')
    require(spec['sector'].endswith(_array_hash(of_indices)), 'SECTOR_HASH')
    require(np.array_equal(indices, reverse_bits(8)[of_indices]), 'QISKIT_BIT_ORDER')
    require(np.array_equal(state[indices], sector_state), 'FULL_SECTOR_STATE')
    outside = np.setdiff1d(np.arange(256), indices)
    require(np.count_nonzero(state[outside]) == 0 and np.isfinite(state).all(), 'STATE_SECTOR_SUPPORT')
    require(abs(np.vdot(state, state).real-1) <= 1e-12, 'STATE_NORM')
    raw = git('show', DF_COMMIT+':'+DF_PATH)
    require(hashlib.sha256(raw).hexdigest() == DF_SHA, 'DF_FILE_SHA256')
    with tempfile.TemporaryDirectory(prefix='b2-p1-public-df-') as temp:
        path = Path(temp)/'input.npz'
        path.write_bytes(raw)
        ham, df_sector, discarded_df_state, metadata = load_snapshot(path)
        del discarded_df_state
    require(ham.n_blocks == 12 and metadata['hamiltonian_hash'] == spec['df_hamiltonian'], 'DF_IDENTITY_RANK')
    require(metadata['fragment_order'] == list(range(12)), 'DF_GENERATION_ORDER')
    require(np.array_equal(df_sector.basis_indices, of_indices), 'DF_SECTOR_IDENTITY')
    for name, expected in operator_prior['identity']['execution_SOURCE_hashes'].items():
        if name in ('scripts/run_o1_m1_h4_operator_correctness.py', 'src/trotterlib/h4_molecular_inputs.py'):
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'REUSED_OPERATOR_SOURCE')
    molecule = MolecularData(filename=str(files['H4_sto-3g_singlet_df_d100.hdf5'].with_suffix('')))
    one, two_of = molecule.get_integrals()  # Saved integrals only; no SCF call.
    for name, value in (('one_MO', one), ('two_MO_OF', two_of),
                        ('MO', molecule.canonical_orbitals), ('AO_overlap', molecule.overlap_integrals)):
        require(_array_hash(value) == operator_prior['identity']['original_integral_hashes'][name], 'ORIGINAL_'+name+'_HASH')
    require(float(molecule.nuclear_repulsion).hex() == float(ham.constant).hex(), 'NUCLEAR_SCALAR')
    spin_one, spin_two = spin_lift_unthresholded(one, two_of.transpose(0, 3, 1, 2))
    ref = car_reference()
    a = ref.annihilators(8)
    identity = np.eye(256, dtype=complex)
    physical = ham.constant*identity + ref.one_body(spin_one, a).toarray() + ref.quartic(spin_two/2, a).toarray()
    require('H_phys:S0_array_hash:'+_array_hash(physical) == spec['original_hamiltonian'], 'RAW_HPHYS_OPERATOR_HASH')
    one_body = ref.one_body(ham.one_body, a).toarray()
    fragments = tuple(w*(ref.one_body(g, a)@ref.one_body(g, a)).toarray()
                      for w, g in zip(ham.lambdas, ham.g_matrices, strict=True))
    df = ham.constant*identity+one_body+sum(fragments, np.zeros_like(one_body))
    require(_array_hash(df) == operator_prior['identity']['operator_matrix_hashes']['H_DF12'], 'DF_MATRIX_HASH')
    residual = float(np.linalg.norm(physical@state-energy*state))
    require(residual <= 1e-10, 'ORIGINAL_H_STATE_RESIDUAL')
    public = dict(status='PASS', file_SHA256=hashes, original_hamiltonian=spec['original_hamiltonian'],
        original_state=spec['original_state'], df_hamiltonian=spec['df_hamiltonian'],
        reconstructed_DF_matrix_hash=_array_hash(df), state_role=spec['state_role'], sector=spec['sector'],
        spin_orbital_order=spec['spin_orbital_order'], state_generation_SOURCE=spec['source'],
        raw_Hphys_state_residual_Ha=residual, raw_Hphys_Hermiticity_spectral=float(np.linalg.norm(physical-physical.conj().T,2)),
        state_preparation_included=False, DF_state_used=False,
        input_modes={name: oct(path.stat().st_mode&0o777) for name,path in files.items()})
    return dict(ham=ham, state=state, sector_state=sector_state, indices=indices, energy=energy,
        physical=physical, df=df, one_body=one_body, fragments=fragments, spec=spec,
        public=public, files=files, ref=ref)


def compose_step(halves, tail, phase):
    """Adapter for the existing forward/tail/reverse circuit append order."""
    result = phase*np.eye(len(tail), dtype=complex)
    for half in halves:
        result = half@result
    result = tail@result
    for half in reversed(halves):
        result = half@result
    return result


def legacy_mean_steps(preparation, deterministic, tail_eigensystem):
    # Preserve the historical B2 SOURCE/hash contract. Its mean-only extraction
    # delegates polynomial arithmetic to the existing canonical RTE function.
    tau=preparation.exact_rte_lambda_r*DELTA/R
    config,distribution=make_rte_config(preparation.rte_preparation.symbolic_tail,
        evolution_time=DELTA,rte_steps=R,finite_taylor_order=K,
        truncation_tolerance=legacy._explicit_cutoff_tolerance(tau,K),seed=0)
    polynomial=np.diag(finite_taylor_operator(np.diag(tail_eigensystem[0]),
        config.dimensionless_step_time,K))
    corrected=polynomial**R
    log_normalization=R*math.log(distribution.exact_finite_distribution)
    kernel=dict(corrected=corrected,raw=corrected/math.exp(log_normalization),
        ideal_tail=np.exp(-1j*DELTA*preparation.exact_rte_lambda_r*tail_eigensystem[0]),
        phase=np.exp(-1j*DELTA*(preparation.constant_coefficient+preparation.extracted_identity_coefficient)),
        occurrence_log_normalization=log_normalization)
    eye = np.eye(len(tail_eigensystem[0]), dtype=complex)
    steps = {}
    for name in ('ideal_tail', 'corrected', 'raw'):
        def action(vector):
            return legacy._apply_spectral_values(tail_eigensystem, vector, kernel[name])
        steps[name] = np.column_stack([legacy._apply_outer_step(eye[:, j], deterministic,
            delta=DELTA, phase=kernel['phase'], tail_action=action) for j in range(len(eye))])
    return config, distribution, kernel, steps


def qualify_signals(round_, record):
    bounds = record.bounds(round_, radius_allowance=U, complex_allowance=U,
                          phase_allowance=U, normalization_tolerance=U)
    if bounds['reference_radius_lower_bound'] <= 0:
        return dict(feasible=False, reasons=['nonpositive_PF_radius'], bounds=bounds)
    diagnostic = radius_phase_diagnostics(specification=round_.specification, allocation=ALLOCATION,
        beta_rpe=.4, config=round_.config, **{k: bounds[k] for k in
        ('reference_radius_lower_bound', 'corrected_signal_error_bound', 'pf_phase_bound',
         'model_phase_bound', 'implementation_phase_bound')})
    reasons = list(diagnostic['reasons'])
    if bounds['model_phase_bound'] > CAPS['model']:
        reasons.append('model_phase_cap')
    if bounds['implementation_phase_bound'] > CAPS['implementation']:
        reasons.append('implementation_phase_cap')
    total = ALLOCATION.beta_total+bounds['model_phase_bound']+bounds['implementation_phase_bound']+CAPS['reserve']
    if total > .4:
        reasons.append('P1_total_with_reserve')
    return dict(**{k:v for k,v in diagnostic.items() if k not in ('feasible','reasons')},
        bounds=bounds, phase_total_with_reserved_statistical=total, feasible=not reasons,
        reasons=reasons, guarantee_status='not_certified', shot_accounting_called=False)


def check_record(name, left, right, *, tolerance=MATCH):
    delta = np.asarray(left)-np.asarray(right)
    value = float(np.max(np.abs(delta), initial=0))
    return dict(name=name, max_abs_entry=value, tolerance=tolerance, passed=value<=tolerance)


def controlled_checks(inputs, preparation, round0, det_full):
    """Representative ideal native semantics; not all-support or compiled audit."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Pauli, Statevector
    checks, halves = [], []
    d, n = len(inputs['state']), preparation.num_system_qubits
    for number, (block, matrix) in enumerate(zip(preparation.deterministic_blocks, det_full, strict=True)):
        circuit = QuantumCircuit(n+1)
        native._append_block(circuit, block, DELTA/2, 'ORDINARY', n)
        actual = np.asarray(Operator(circuit).data)
        exact = expm(-.5j*DELTA*matrix)
        checks.append(check_record('deterministic_half_'+str(number), actual, controlled_reference(exact)))
        halves.append(actual[d:,d:])
    extraction = preparation.tail_extraction
    bases = {key:basis_change_unitary(extraction.basis_definition(key)) for key in extraction.referenced_basis_ids}
    diagonals = {c.component_id:np.diag(diagonal_pauli_matrix(n,c.diagonal_pauli_support)).copy()
                 for c in extraction.components}
    reconstructed_tail = np.zeros((d,d),complex)
    for key,basis in bases.items():
        diagonal = sum((c.coefficient*diagonals[c.component_id] for c in extraction.components if c.basis_id==key), np.zeros(d))
        reconstructed_tail += (basis*diagonal)@basis.conj().T
    tail_car = sum(inputs['fragments'][LD:],np.zeros((d,d),complex))-preparation.extracted_identity_coefficient*np.eye(d)
    checks.append(check_record('all_tail_coefficients_Gaussian_sum_vs_CAR', reconstructed_tail, tail_car))
    component_ops = {}
    components = {c.component_id:c for c in extraction.components}
    def event_operator(event):
        for cid in event.selected_component_ids:
            if cid not in component_ops:
                c=components[cid]; b=bases[c.basis_id]
                component_ops[cid]=c.coefficient_sign*(b*diagonals[cid])@b.conj().T
        return event_unitary(event, component_ops)
    # The existing selection supplies coverage; reduce only repeated primitive
    # classes, before evaluating their circuits. No random event sampling.
    selected = {}
    for event in representative_events(round0.tail.components,round0.distribution):
        cs=[components[cid] for cid in event.selected_component_ids]
        key=(event.taylor_order,event.phase,tuple(sorted({c.coefficient_sign for c in cs})),
             tuple(sorted({len(c.diagonal_pauli_support) for c in cs})))
        selected.setdefault(key,event)
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=preparation.rte_preparation.basis_registry)
    events=list(selected.values()); event_diagnostics=[]
    for number,event in enumerate(events):
        expected=event_operator(event)
        built=builder.build_event(preparation.rte_preparation.request_for_event(event,controlled=True,ancilla_qubit=n))
        checks.append(check_record('controlled_event_'+str(number),Operator(built.circuit).data,controlled_reference(expected)))
        event_diagnostics.append(dict(taylor_order=event.taylor_order,phase_real=float(complex(event.phase).real),
            phase_imag=float(complex(event.phase).imag),component_ids=list(event.selected_component_ids)))
    phase=np.exp(-1j*DELTA*(preparation.constant_coefficient+preparation.extracted_identity_coefficient))
    outer_events=[(events[0],)*R]
    negative=next((e for e in events if e.taylor_order==2 and e.phase==-1),None)
    require(negative is not None, 'REPRESENTATIVE_NEGATIVE_ORDER2_REQUIRED')
    outer_events.append((events[0],negative,events[-1],events[0]))
    for case,occurrence in enumerate(outer_events):
        request=round0.explicit_request(occurrence)
        # Instruction bound follows the one-step request, never a CPU/RAM cap.
        from .df_rte_qiskit import estimate_df_rte_untranspiled_size_upper_bound
        bound=2*sum(native.block_instruction_bound(b,'ORDINARY') for b in preparation.deterministic_blocks)+1
        bound+=sum(estimate_df_rte_untranspiled_size_upper_bound(o) for o in request.rte_occurrences)
        evolution=round0.native(request,max_instructions=bound)
        tail=np.eye(d,dtype=complex)
        for event in occurrence:
            tail=event_operator(event)@tail
        exact=compose_step([expm(-.5j*DELTA*b) for b in det_full],tail,phase)
        checks.append(check_record('one_outer_step_'+str(case),Operator(evolution.circuit).data,controlled_reference(exact)))
        z=complex(np.vdot(inputs['state'],exact@inputs['state']))
        for axis,target in (('cosine',z.real),('sine',z.imag)):
            wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=bound+4)
            measured=Statevector(np.r_[inputs['state'],np.zeros(d)]).evolve(wrapper).expectation_value(Pauli('Z'+'I'*n)).real
            checks.append(check_record('wrapper_'+str(case)+'_'+axis,measured,target))
    return dict(status='PASS' if all(c['passed'] for c in checks) else 'STOP',checks=checks,
        representative_events=event_diagnostics,outer_step_cases=len(outer_events),
        scope='uncompiled selected primitives and Q=1 complete outer steps; not all support, trajectories, or compiler'), halves, reconstructed_tail


def run(private_dir, output):
    """Execute only m=0 and m=17. Stop on first failed scientific qualification."""
    started=time.perf_counter()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source=git('rev-parse','HEAD').decode().strip()
    source_paths=['src/trotterlib/b2_p1_signal_validation.py','scripts/run_b2_p1_signal_validation.py',
        'src/trotterlib/pr2_matched_accuracy_m1_execution.py','src/trotterlib/o1_m1_resource_bridge.py']
    for path in source_paths:
        require(git('show',source+':'+path)==(ROOT/path).read_bytes(),'COMMITTED_SOURCE_REQUIRED')
    result=dict(schema_version='b2_p1_two_round_signal_v1',started_at_UTC=datetime.now(timezone.utc).isoformat(),
        execution_SOURCE=source,SOURCE_SHA256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        fixed_conditions=dict(target='H_phys',DF_rank=12,L_D=LD,delta=DELTA,r=R,K=K,requested_rounds=[0,17],
            energy_precision_Ha=.00015936001019904,success_probability_target=.95,beta_RPE=.4,
            energy_prior_Ha=[-128,128],energy_shift_Ha=0,phase_convention='exp(-itH); phi=-delta E',
            caps_rad=CAPS,allocation=asdict(ALLOCATION),numerical_allowance=U,independent_match_tolerance=MATCH),
        rounds=[],guarantee_status='not_certified',mandatory_stop=True,
        forbidden_work=dict(SCF=0,DF_fit=0,ground_state_solve=0,other_round_signals=0,
            branch_success=0,shot_accounting=0,compile=0,sampling=0,price_MC=0,total_cost=0,scientific_review=0))
    try:
        inputs=load_private_inputs(private_dir)
        result['input_identity']=inputs['public']
        print('Private input identity: PASS',flush=True)
        preparation=connection.prepare_b2(inputs['ham'],L_D=LD)
        require(math.isclose(preparation.exact_rte_lambda_r,.5832704246339983,rel_tol=0,abs_tol=MATCH),'LAMBDA_R_SCREENING_REFERENCE')
        indices=inputs['indices']; ix=np.ix_(indices,indices)
        one=inputs['one_body'][ix]; fragments=tuple(b[ix] for b in inputs['fragments'])
        det_full=[inputs['one_body'],*inputs['fragments'][:LD]]
        det=[one,*fragments[:LD]]
        tail=sum(fragments[LD:],np.zeros_like(one))-preparation.extracted_identity_coefficient*np.eye(len(one))
        deterministic=legacy._preparation_eigensystems(preparation,one,fragments,{})
        tail_eigensystem=legacy._eigendecomposition(tail/preparation.exact_rte_lambda_r)
        config,distribution,kernel,steps=legacy_mean_steps(preparation,deterministic,tail_eigensystem)
        ctx=connection.RPEConnectionContext(inputs['spec']['original_hamiltonian'],preparation.hamiltonian_hash,
            inputs['spec']['original_state'],inputs['spec']['sector'],inputs['spec']['spin_orbital_order'],source,
            CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'),'no_backend','rz_count',
            None,.00015936001019904,.95,'P1 [-128,128] Ha; shift0; beta<pi/3; full branch untested')
        def round_at(m):
            spec=RPERoundSpecification(m,DELTA)
            return connection.B2RPERound(ctx,preparation,spec,LD,R,K,DELTA,spec.q_m,config,distribution)
        controlled, native_halves, native_tail=controlled_checks(inputs,preparation,round_at(0),det_full)
        result['controlled_correctness']=controlled
        require(controlled['status']=='PASS','CONTROLLED_CORRECTNESS')
        print('Representative controlled correctness: PASS',flush=True)
        phase=kernel['phase']
        halves=[expm(-.5j*DELTA*b) for b in det]
        independent={
            'ideal_tail':compose_step(halves,expm(-1j*DELTA*tail),phase),
            'corrected':compose_step(halves,paired_polynomial_action(tail,np.eye(len(tail)),DELTA,R,K),phase)}
        moments=finite_rte_operator_moments(tail/preparation.exact_rte_lambda_r,config)
        independent['raw']=compose_step(halves,moments.attenuated_event_mean_operator,phase)
        native_moments=finite_rte_operator_moments(native_tail/preparation.exact_rte_lambda_r,config)
        native_step=compose_step(native_halves,native_moments.corrected_operator,phase)[ix]
        psi=inputs['sector_state']
        signal=lambda op:complex(np.vdot(psi,op@psi))
        result['normalization_law']=dict(lambda_R=preparation.exact_rte_lambda_r,
            tail_identity=preparation.extracted_identity_coefficient,tau=config.dimensionless_step_time,
            b_event=config.distribution_normalization,K_mean_polynomial_degree=K+1,
            preparation_hash=preparation.preparation_hash,tail_hash=round_at(0).tail.tail_hash)
        for m in (0,17):
            round_=round_at(m);q=round_.q;t=round_.specification.t_m
            print('Evaluating permitted round m='+str(m),flush=True)
            values={'phys':signal(expm(-1j*t*inputs['physical'][ix])),
                    'DF':signal(expm(-1j*t*inputs['df'][ix]))}
            other={name:complex(np.vdot(inputs['state'],expm_multiply(-1j*t*csr_matrix(inputs[key]),inputs['state'])))
                   for name,key in (('phys','physical'),('DF','df'))}
            for name,key in (('PF','ideal_tail'),('finite','corrected'),('raw','raw')):
                values[name]=signal(np.linalg.matrix_power(steps[key],q))
                other[name]=signal(np.linalg.matrix_power(independent[key],q))
            comparisons={name:signal_difference(values[name],other[name]) for name in values}
            normalization=math.exp(q*kernel['occurrence_log_normalization'])
            direct_B=config.distribution_normalization**(q*R)
            A=connection.finite_rte_attenuation(config,tail_evolutions=q)
            normalization_checks=dict(B_log_vs_direct=abs(normalization-direct_B),
                A_log_vs_inverse_direct=abs(A-1/direct_B),BA_residual=abs(normalization*A-1))
            numerical_pass=(all(c['complex_absolute']<=MATCH and c['radius_absolute']<=MATCH and
                               c['phase_rad'] is not None and c['phase_rad']<=MATCH for c in comparisons.values())
                            and all(v<=MATCH for v in normalization_checks.values()))
            native_value=signal(np.linalg.matrix_power(native_step,q))
            implementation=phase_distance(native_value,values['finite'])
            require(implementation is not None,'UNDEFINED_IMPLEMENTATION_PHASE')
            fields=[connection.SignalValue(values[name],ctx.original_state,
                ctx.original_hamiltonian if name=='phys' else ctx.df_hamiltonian)
                for name in ('phys','DF','PF','finite','raw')]
            record=connection.B2SignalRecord(round_.binding,*fields,normalization,implementation)
            item=dict(m=m,Q=q,T_Ha_inverse=t,signals={k:signal_record(v) for k,v in values.items()},
                exact_ground_assumption_reference=signal_record(complex(np.exp(-1j*t*inputs['energy']))),
                independent_differences=comparisons,independent_numerical_match=numerical_pass,
                adjacent_errors={name:signal_difference(values[a],values[b]) for name,a,b in
                    (('model','DF','phys'),('PF','PF','DF'),('RTE','finite','PF'),('normalization','raw','finite'))},
                normalization=normalization,attenuation=connection.finite_rte_attenuation(config,tail_evolutions=q),
                independent_normalization_differences=normalization_checks,
                raw_corrected_residual=float(abs(values['raw']-values['finite']/normalization)),
                independent_raw_path='finite_rte_operator_moments attenuated event mean, composed before Q power',
                implementation_native_mean=signal_record(native_value),implementation_phase_observed=implementation,
                implementation_scope='conditional ideal native mean Q power; selected native semantics checked; no full long circuit/all-support floating point certificate')
            result['rounds'].append(item)
            if not numerical_pass:
                item['status']='STOP_NUMERICAL_MATCH';result['status']='STOP_NUMERICAL_MATCH';break
            item['qualification']=qualify_signals(round_,record)
            item['status']='CONDITIONAL_PASS' if item['qualification']['feasible'] else 'STOP_P1_INELIGIBLE'
            print('Round m='+str(m)+': '+item['status'],flush=True)
            if item['status']!='CONDITIONAL_PASS':
                result['status']=item['status'];break
        else:
            result['status']='TWO_ROUND_CONDITIONAL_PASS'
        for name,path in inputs['files'].items():
            require(hashlib.sha256(path.read_bytes()).hexdigest()==inputs['public']['file_SHA256'][name], 'PRIVATE_INPUT_CHANGED')
    except Exception as error:
        result['status']='STOP_'+type(error).__name__
        # Explicit diagnostic names contain no private values or file contents.
        result['stop_reason']=str(error) if isinstance(error,ValueError) else type(error).__name__
        raise_after_write=error
    else:
        raise_after_write=None
    result['environment']=dict(host=platform.node(),python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','qiskit','openfermion','h5py')},
        cpu_affinity=len(os.sched_getaffinity(0)),BLAS_env={k:os.environ.get(k) for k in
        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},gpu_used=False)
    result['elapsed_seconds']=time.perf_counter()-started
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if raise_after_write is not None:
        raise raise_after_write
    return result
