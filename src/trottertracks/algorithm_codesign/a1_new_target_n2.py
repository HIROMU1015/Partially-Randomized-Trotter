"""Limited N2: one full-DF ground solve and certified sector/state checks.

N1 payloads are read only. No molecular circuits, path means, shots or RNG.
The unknown future selected-circuit probability error is never replaced by zero.
"""
from contextlib import ExitStack, contextmanager
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
import mpmath as mp
from . import a1_new_target_n1 as n1
from .df_correction_comparison.intervals import configure, iv, upper, outward_upper

N1_COMMIT='744e1a09661ebba6841fe5faf456f88c41c47780'
N1_SOURCE='b6dbec9f80da35a88fba28795ad02ca74c9adfa0'
N1_DIR=Path('artifacts/o2_a1_new_target/2026-11-06/n1')
DOC=Path('docs/research/o2_a1_new_target_n2_20261106')


def read(p):return json.loads(Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rational(q):return dict(exact=str(q),float_preview=float(q))
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root).decode().strip()


def profile(root):
    p=n1.profile(root)
    for path in ('scripts/tracks/algorithm_codesign/run_a1_new_target_n2.py',
                 'tests/tracks/algorithm_codesign/test_a1_new_target_n2.py'):
        p['source_sha256'][path]=sha((Path(root)/path).read_bytes())
    p['source_sha256']=dict(sorted(p['source_sha256'].items()))
    return p


@contextmanager
def n2_only(dep):
    def forbidden(*a,**kw):raise RuntimeError('N2 forbids N1 recompute/RNG/full molecular circuits/measurements/compile')
    from qiskit.quantum_info import Statevector
    with ExitStack() as stack:
        for owner,names in ((np.random,('default_rng','Generator','PCG64','RandomState','seed','random','choice','uniform','normal')),
                            (random,('Random','seed','random','choice','randrange'))):
            for name in names:stack.enter_context(patch.object(owner,name,forbidden))
        for owner,name in ((dep.split,'split_df_hamiltonian_by_ld'),(dep.partial,'prepare_df_partial_s2'),
            (dep.policy,'register_support_restricted_bases'),(n1,'connect_sampler_inputs'),
            (dep.rte,'finite_rte_distribution'),(dep.rte,'iter_sample_rte_events'),(dep.rte,'sample_rte_events'),
            (dep.partial.QiskitDFPartialS2CircuitBuilder,'build_step'),
            (dep.partial.QiskitDFPartialS2CircuitBuilder,'build_additive_circuits'),
            (dep.policy.QiskitDFRTEEventCircuitBuilder,'build_sequence'),
            (dep.repeated.QiskitDFPartialS2RepeatedCircuitBuilder,'build'),
            (dep.hadamard.QiskitRPEHadamardInterrogationBuilder,'build'),
            (dep.benchmark.QiskitRPEHadamardBenchmarkCircuitBuilder,'build'),
            (dep.compiler,'transpile_and_measure_cost'),(dep.policy,'transpile_and_measure_cost'),
            (Statevector,'evolve'),(Statevector,'sample_counts'),(Statevector,'sample_memory')):
            stack.enter_context(patch.object(owner,name,forbidden))
        yield


def exact_sector_matrix(h,sector,table):
    """Exact dyadic DF assembly using the existing fermionic excitation table.

    Real Hermitian saved input only; complex/asymmetric inputs STOP. No 1/2.
    """
    n=sector.dimension
    def one(matrix):
        a=np.asarray(matrix)
        if not np.isfinite(a).all() or np.any(a.imag!=0) or not np.array_equal(a,a.conj().T):
            raise ValueError('exact real Hermitian input required; no symmetrization fallback')
        values=[[F(float(x.real)) for x in row] for row in a]
        out=[[F(0) for _ in range(n)] for _ in range(n)]
        for i,j,p,q,s in zip(table.target_positions,table.source_positions,table.p_indices,table.q_indices,table.signs):
            out[int(i)][int(j)]+=int(s)*values[int(p)][int(q)]
        return out
    out=one(h.one_body)
    for i in range(n):out[i][i]+=F(float(h.constant))
    for lam,g in zip(h.lambdas,h.g_matrices):
        a=one(g);nz=[[(j,x) for j,x in enumerate(row) if x] for row in a];weight=F(float(lam))
        for i in range(n):
            for k,x in nz[i]:
                for j,y in nz[k]:out[i][j]+=weight*x*y
    if any(out[i][j]!=out[j][i] for i in range(n) for j in range(n)):
        raise ValueError('exact DF matrix is not symmetric')
    return out


def spectral_enclosure(a,values,vectors):
    """Exact Gram/reconstruction errors + polar orthogonalization, then Weyl.

    Reuses Spectrum's polar-perturbation idea, but evaluates all matrix sums
    as exact Fractions and square roots by existing directed iv helpers.
    Assembly is already exact, so no uncharged BLAS/matvec assembly error.
    """
    configure();n=len(a)
    d=[F(float(x)) for x in values];q=[[F(float(x)) for x in row] for row in vectors]
    if len(d)!=n or any(d[i]>d[i+1] for i in range(n-1)) or np.asarray(vectors).shape!=(n,n):
        raise ValueError('invalid complete eigensystem')
    gram=[[sum((q[k][i]*q[k][j] for k in range(n)),F(0))-int(i==j) for j in range(n)] for i in range(n)]
    g=max(sum(map(abs,row)) for row in gram)
    if g>=1:raise ValueError('eigenvector invertibility not certified')
    recon=[[a[i][j]-sum((q[i][k]*d[k]*q[j][k] for k in range(n)),F(0)) for j in range(n)] for i in range(n)]
    e=max(sum(map(abs,row)) for row in recon)
    delta=iv(g)/(1+mp.iv.sqrt(1-iv(g)))
    error=outward_upper(upper(iv(e)+iv(max(map(abs,d)))*delta*(mp.iv.sqrt(1+iv(g))+1)))
    return dict(gram_row_norm_upper=str(g),reconstruction_row_norm_upper=str(e),
        matrix_operator_error_upper=str(error),eigenvalue_intervals=[dict(lower=str(x-error),upper=str(x+error)) for x in d],
        method='exact real rational assembly/Gram/reconstruction; polar Q to O; Weyl ordered eigenvalue intervals',
        rounding_model='exact stored binary64 input; mpmath1.3.0 directed iv100dps; outward256bit endpoints')


def state_certificate(a,vector,spectrum):
    configure();n=len(a);v=np.asarray(vector,dtype=complex)
    if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('invalid sector state')
    re=[F(float(x.real)) for x in v];im=[F(float(x.imag)) for x in v]
    norm2=sum((x*x+y*y for x,y in zip(re,im)),F(0))
    if norm2<=0:raise ValueError('zero state')
    rayleigh=sum((a[i][j]*(re[i]*re[j]+im[i]*im[j]) for i in range(n) for j in range(n)),F(0))/norm2
    ev=spectrum['eigenvalue_intervals'];l0,u0,l1=F(ev[0]['lower']),F(ev[0]['upper']),F(ev[1]['lower'])
    if l1<=u0:raise ValueError('unique sector ground not spectrally separated')
    if not l0<=rayleigh<l1:raise ValueError('state is not certified in separated ground branch')
    weight=min(F(1),max(F(0),(rayleigh-l0)/(l1-l0)))
    bound=outward_upper(2*upper(mp.iv.sqrt(iv(weight))))
    return dict(status='STATE_CERTIFIED_FOR_SAVED_DF_NUMBER_SECTOR',norm_squared=rational(norm2),
        normalized_state_definition='exact vector / sqrt(exact norm_squared)',
        normalized_rayleigh=rational(rayleigh),ground_energy_interval=ev[0],first_excited_lower=rational(l1),
        gap_lower=rational(l1-u0),unique_ground_certified=True,excited_weight_upper=rational(weight),
        coordinate_state_error_upper=rational(bound),
        error_meaning='uniform X/Y expectation distance for any same ideal unitary; 2*sqrt(excited weight)',
        precision_budget_accepted='UNKNOWN_NOT_ADOPTED',
        applies_to='all 4-electron number sector of saved DF Hamiltonian only; not original unapproximated Hamiltonian')


def validate_n1(root):
    p=root/N1_DIR;r=read(p/'RESULT.json');source=read(p/'SOURCE.json')
    if r['source_commit']!=N1_SOURCE or source['adopted_contract_commit']!=n1.CONTRACT:
        raise ValueError('N1 source/contract binding mismatch')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',N1_COMMIT,'--',str(N1_DIR)],cwd=root,text=True).splitlines()
    for path in paths:
        expected=subprocess.check_output(['git','show',N1_COMMIT+':'+path],cwd=root)
        if (root/path).read_bytes()!=expected:raise ValueError('N1 payload changed: '+path)
    if any(sha((p/f).read_bytes())!=h for f,h in r['artifact_sha256'].items()):raise ValueError('N1 artifacts hash mismatch')
    if read(p/'STOP.json')['status']!='MANDATORY_STOP':raise ValueError('old N1 STOP changed')
    return r,source,{path:sha((root/path).read_bytes()) for path in paths}


def run(output_dir,profile_lock,root=None):
    root=Path(root or Path(__file__).resolve().parents[3]).resolve();out=Path(output_dir).resolve()
    dirty=git(root,'status','--porcelain=v1','--untracked-files=all');out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();counts=dict(snapshot_load=0,cache_vector_load=0,ground_state_solve=0,
        exact_sector_matrix=0,sector_spectral_eigh=0,partition=0,preparation=0,CDF_recompute=0,
        RNG_sampling=0,Hadamard_shots=0,molecular_full_build=0,compile=0,variance_N_cost=0)
    dep=None;source=dict(schema='a1_new_target_n2_source_v1',N1_commit=N1_COMMIT,N1_scientific_source=N1_SOURCE,
        adopted_contract_commit=n1.CONTRACT,N2_source_commit=git(root,'rev-parse','HEAD'),
        no_retry=True,N3_authorized=False)
    stop_reason='N2_INCOMPLETE';state_context={}
    def log(s):
        with (out/'RUN.log').open('a') as f:f.write(s+'\n');f.flush()
    try:
        dep=n1.dependencies();locked=read(profile_lock);actual=profile(root)
        if dirty or actual!=locked:raise ValueError('unclean worktree or source/environment lock mismatch')
        write(out/'PROFILE.json',locked);source['profile_sha256']=sha(canonical(locked).encode())
        r,s,payload_hashes=validate_n1(root);source.update(N1_identities=r['identities'],N1_payload_sha256=payload_hashes,
            input=s['snapshot'],input_sha256=s['snapshot_sha256'],scope=s['scope'])
        if sha((root/s['snapshot']).read_bytes())!=s['snapshot_sha256']:raise ValueError('N1 NPZ bytes mismatch')
        inventory=read(root/DOC/'STATE_INVENTORY.json');source['state_inventory_sha256']=sha((root/DOC/'STATE_INVENTORY.json').read_bytes())
        write(out/'RESOURCE_OBSERVATION.json',dict(cpu_count=__import__('os').cpu_count(),load_average=__import__('os').getloadavg(),memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))],resource_caps_added=False))
        with n2_only(dep):
            counts['snapshot_load']+=1;h=dep.loader.load_connected_cluster_hamiltonian_snapshot(root/s['snapshot'])
            if dep.split.df_hamiltonian_hash(h)!=r['identities']['hamiltonian_hash']:raise ValueError('N1 Hamiltonian identity changed')
            sector=dep.hamiltonian.PhysicalSector.number_sector(n_qubits=8,n_electrons=4)
            if h.n_qubits!=8 or h.n_blocks!=12 or sector.dimension!=70:raise ValueError('fixed model/number sector dimensions mismatch')
            if dep.split._sector_hash(sector)!=inventory['expected_sector_hash']:raise ValueError('sector ordering mismatch')
            ground_hash=dep.split._df_hamiltonian_hash(h,weight_rule='ground_state')
            candidates=[x for x in inventory['candidates'] if x.get('hamiltonian_hash')==ground_hash and x.get('sector_hash')==inventory['expected_sector_hash']]
            log('Fixed full H loaded; eligible saved-state metadata matches='+str(len(candidates)))
            if candidates:
                candidate=candidates[0];file=Path(candidate['path'])
                if sha(file.read_bytes())!=candidate['sha256']:raise ValueError('saved state bytes changed')
                counts['cache_vector_load']+=1
                with np.load(file,allow_pickle=False) as z:
                    full=np.asarray(z['state_vector'],complex);v=np.asarray(z['sector_state_vector'],complex)
                source['reused_state']=candidate
                solver_record=dict(mode='reused_identity_matched_saved_vector; validated below')
            else:
                counts['ground_state_solve']+=1;log('One existing solve_df_ground_state on full DF H, deterministic v0')
                # 70-dimensional Python matvec avoids large JIT/parallel dispatch overhead;
                # this is a backend choice, not a cap on CPU/RAM/threads.
                result=dep.hamiltonian.solve_df_ground_state(h,sector,matrix_free_backend='python',
                    v0=np.ones(sector.dimension,dtype=np.complex128),tol=1e-10,expand_state=True)
                full=result.state_vector;v=result.sector_state_vector
                solver_record=dict(mode='one solve_df_ground_state; full H passed unchanged',backend='python',solver=result.solver,
                    deterministic_v0='all ones; no RNG',tol=1e-10,converged=result.converged,
                    energy_diagnostic=result.energy,residual_norm_diagnostic=result.residual_norm,matvec_count=result.matvec_count,
                    elapsed_seconds=result.elapsed_s)
                if not result.converged:raise ValueError('ground solver did not converge; no retry')
            if full.shape!=(256,) or not np.isfinite(full).all() or not np.array_equal(full[sector.basis_indices],v):raise ValueError('state dimension/sector expansion mismatch')
            outside=full.copy();outside[sector.basis_indices]=0
            if np.any(outside!=0):raise ValueError('state has nonzero outside-sector component')
            perm=dep.split._bit_reverse_permutation(8);qstate=dep.split._to_qiskit_state_order(full,8)
            if not np.array_equal(perm[perm],np.arange(256)) or not np.array_equal(qstate[perm],full):raise ValueError('qubit-order permutation mismatch')
            internal=out/'engine_internal';internal.mkdir()
            vector_record=dict(scope='ENGINE_ONLY_NOT_ESTIMATOR_INPUT',hamiltonian_hash=r['identities']['hamiltonian_hash'],
                sector_hash=dep.split._sector_hash(sector),basis_indices=sector.basis_indices.tolist(),
                openfermion_full_complex_hex=[[z.real.hex(),z.imag.hex()] for z in full],
                qiskit_full_complex_hex=[[z.real.hex(),z.imag.hex()] for z in qstate],qiskit_bit_reversal=perm.tolist())
            write(internal/'STATE_VECTOR.json',vector_record);vector_sha=sha((internal/'STATE_VECTOR.json').read_bytes())
            state_context=dict(state_vector_sha256=vector_sha,hamiltonian_hash=r['identities']['hamiltonian_hash'],
                sector_hash=dep.split._sector_hash(sector),number_sector_dimension=sector.dimension,
                sector_leakage_exact='0',qubit_ordering_checked=True,solver=solver_record)
            log('Exact DF sector assembly begins')
            counts['exact_sector_matrix']+=1;a=exact_sector_matrix(h,sector,dep.hamiltonian._build_excitation_table(sector))
            ahash=sha(canonical([[str(x) for x in row] for row in a]).encode());log('Exact DF sector assembly completed')
            counts['sector_spectral_eigh']+=1;values,vectors=np.linalg.eigh(np.asarray([[float(x) for x in row] for row in a]))
            enclosure=spectral_enclosure(a,values,vectors);cert=state_certificate(a,v,enclosure)
            witness=dict(scope='ENGINE_ONLY_NOT_ESTIMATOR_INPUT',exact_H_sector_sha256=ahash,
                values_hex=[x.hex() for x in values],vectors_hex=[[x.hex() for x in row] for row in vectors],enclosure=enclosure)
            write(internal/'SPECTRAL_WITNESS.json',witness)
            cert.update(state_vector_sha256=vector_sha,hamiltonian_hash=r['identities']['hamiltonian_hash'],
                sector_hash=dep.split._sector_hash(sector),number_sector_dimension=sector.dimension,sector_leakage_exact='0',
                qubit_ordering='OpenFermion index bit n-1-mode; Qiskit mode bit mode; saved bit reversal',solver=solver_record,
                exact_H_sector_sha256=ahash,spectral_witness='engine_internal/SPECTRAL_WITNESS.json',
                oracle_isolation='engine_internal only; no CV fit/feature selection/allocation invoked')
            write(out/'STATE_RESULT.json',cert);log('State uniqueness and uniform coordinate error certified')
        tests=read(root/DOC/'TESTS.json')
        if not tests.get('passed'):raise ValueError('artificial correctness preflight failed')
        engine=dict(schema='a1_new_target_n2_engine_v1',status='PARTIAL_CONVENTIONS_VERIFIED_TOTAL_ERROR_UNKNOWN',
            N1_identities=r['identities'],state_identity=vector_sha,axes=['X','Y'],
            conventions=dict(controlled='diag(I,U); ancilla last',X='real; H readout',Y='imaginary; Sdg then H',bit_value_mapping={'0':1,'1':-1},
                scalar_identity_phase='existing _append_step_phases uses PhaseGate on controlled ancilla',
                event_phase='existing RTE n=2 phase=-1 retained as controlled relative phase',ordering='saved full-space bit reversal; exact permutation'),
            artificial_checks=tests,coordinate_error_terms=dict(state=cert['coordinate_state_error_upper'],
                measurement_uniform_53bit_grid=dict(upper_exact=str(F(2,2**53)),status='PROVED_CONDITIONAL_ON_U=k/2^53_AND_U<p_hat',
                    meaning='coordinate mean error; threshold count ceil(M*p_hat); endpoints included'),
                selected_gate_application_arithmetic='UNKNOWN',Born_probability_arithmetic='UNKNOWN',
                normalization_and_probability_range_guard='UNKNOWN_NOT_BOUND_TO_FUTURE_ENGINE',compiled_vs_logical='NOT_APPLICABLE_UNCOMPILED_IDEAL_TARGET'),
            total_b_engine='UNKNOWN',draft_budget=dict(rule='b_engine <= epsilon_axis/8',epsilon_candidate='1/4',candidate_threshold='1/32',adopted=False,comparison='UNKNOWN_TOTAL_BOUND_MISSING'),
            unresolved=['No end-to-end engine reconstructs the saved N1 selected operations here; no molecular path is built/evaluated.',
                'A uniform gate-application and Born-probability arithmetic bound tied to that implementation is required.',
                'Exact normalization, [0,1] probability validation and bit generation must be bound before N3; no unsupported clipping.',
                'Ideal IID word/Born model is an assumption; a PRNG independence proof is not supplied.'],
            N3_ready=False,oracle_information_sent_to_estimator=False,controls_changed=False,pilot_changed=False)
        write(out/'ENGINE_RESULT.json',engine);stop_reason='N2_PARTIAL_STOP_ENGINE_NUMERICAL_ERROR_UNKNOWN'
        log(stop_reason)
    except Exception as exc:
        stop_reason='N2_STOP_'+type(exc).__name__;source['error']=dict(type=type(exc).__name__,message=str(exc));log(stop_reason+': '+str(exc))
        for name in ('STATE_RESULT','ENGINE_RESULT'):
            if not (out/(name+'.json')).exists():
                write(out/(name+'.json'),dict(status='UNKNOWN',reason=str(exc),next_stage_started=False,
                    **(state_context if name=='STATE_RESULT' else {})))
    finally:
        source.update(actual_scientific_calls=counts,elapsed_seconds=time.monotonic()-started)
        write(out/'SOURCE.json',source);write(out/'CALL_COUNTS.json',counts)
        write(out/'STOP.json',dict(status='MANDATORY_STOP',reason=stop_reason,N3_authorized=False,N3_started=False,
            no_retry=True,controls_and_pilot_unchanged=True))
    return dict(status=stop_reason,counts=counts,source_commit=source['N2_source_commit'])
