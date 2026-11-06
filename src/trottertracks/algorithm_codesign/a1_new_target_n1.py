"""One authorized N1 transaction. No RNG/state/full interrogation/compile.

Reuse canonical DF preparation, CDF/CV interfaces and policy registration.
The source profile is frozen before NPZ input; outputs require a fresh directory.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

from .a1_connections import connect_sampler_inputs, proposed_control_banks
from .a1_law_validation import SNAPSHOT, SNAPSHOT_SHA, preparation_payload
from .shared_cpu import module as cpu_module

CONTRACT = '2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab'
A, B = 'df-fragment-3:Z0Z1', 'df-fragment-6:Z0'
POLICY_ATOL = 1e-10  # Existing support-completion / opt2 acceptance tolerance.
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','CUDA_VISIBLE_DEVICES',
               'MPLCONFIGDIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME')


def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def rational(q):
    q=F(q)
    return dict(exact=str(q), float_preview=float(q))


def dependencies():
    """Imports only. No snapshot, preparation, RNG or quantum interrogation."""
    import qiskit_nature.second_q.circuit.library  # freeze lazy Gaussian dependency
    from types import SimpleNamespace
    return SimpleNamespace(loader=cpu_module('rte_connected_cluster_cost_validation'),
        split=cpu_module('df_partial_randomized_pf'), partial=cpu_module('df_partial_s2'),
        tail=cpu_module('df_rte_tail'), rte=cpu_module('rte'),
        policy=cpu_module('research_direction_sequence_policy'),
        repeated=cpu_module('df_partial_s2_repeated'),
        hadamard=cpu_module('rpe_hadamard_interrogation'),
        benchmark=cpu_module('rpe_hadamard_compiled_cost_benchmark'),
        compiler=cpu_module('rte_compiled_cost'), hamiltonian=cpu_module('df_hamiltonian'))


def profile(root):
    """Static source/installed-distribution/native-library lock, no NPZ access."""
    import numpy as np
    root=Path(root)
    source_paths=sorted(set((root/'src/trotterlib').rglob('*.py')) |
                        set((root/'src/trottertracks/algorithm_codesign').rglob('*.py')) |
                        {root/'scripts/tracks/algorithm_codesign/run_a1_new_target_n1.py',
                         root/'tests/tracks/algorithm_codesign/test_a1_new_target_n1.py'})
    distributions={}; natives={}
    for dist in metadata.distributions():
        name=dist.metadata.get('Name','unknown')
        record=dict(version=dist.version, metadata_sha256={})
        for entry in dist.files or ():
            p=Path(dist.locate_file(entry))
            if p.name in ('METADATA','WHEEL','RECORD') and p.parent.name.endswith('.dist-info') and p.is_file():
                record['metadata_sha256'][p.name]=digest(p.read_bytes())
            if '.so' in p.name and p.is_file():
                natives[str(p.resolve())]=digest(p.read_bytes())
        distributions[name]=record
    return dict(schema='a1_n1_static_profile_v1', python=platform.python_version(),
        executable=str(Path(sys.executable).resolve()), executable_sha256=digest(Path(sys.executable).read_bytes()),
        platform=platform.platform(), node=platform.node(),
        thread_environment={k:os.environ.get(k) for k in THREAD_KEYS},
        numpy_build=np.__config__.CONFIG,
        distributions=dict(sorted(distributions.items())), native_sha256=dict(sorted(natives.items())),
        source_sha256={str(p.relative_to(root)):digest(p.read_bytes()) for p in source_paths})


@contextmanager
def n1_only(dep):
    """Fail before any prohibited operation; orbital basis preparation stays allowed."""
    def forbidden(*a,**kw): raise RuntimeError('N1 scope forbids RNG/state/full-build/compile')
    import numpy as np
    with ExitStack() as stack:
        for name in ('default_rng','Generator','PCG64','RandomState','seed','random','choice','uniform','normal'):
            stack.enter_context(patch.object(np.random,name,forbidden))
        for owner,name in ((dep.hamiltonian,'solve_df_ground_state'),
            (dep.partial.QiskitDFPartialS2CircuitBuilder,'build_step'),
            (dep.partial.QiskitDFPartialS2CircuitBuilder,'build_additive_circuits'),
            (dep.policy.QiskitDFRTEEventCircuitBuilder,'build_sequence'),
            (dep.repeated.QiskitDFPartialS2RepeatedCircuitBuilder,'build'),
            (dep.hadamard.QiskitRPEHadamardInterrogationBuilder,'build'),
            (dep.benchmark.QiskitRPEHadamardBenchmarkCircuitBuilder,'build'),
            (dep.compiler,'transpile_and_measure_cost'),
            (dep.policy,'transpile_and_measure_cost')):
            stack.enter_context(patch.object(owner,name,forbidden))
        yield


def hash_payloads(partition, prep, block_payload):
    """Serialize already computed values and reproduce all three NEW hashes."""
    part=dict(hamiltonian_hash=prep.hamiltonian_hash,ld=partition.ld,
        deterministic_fragments=[asdict(x) for x in partition.deterministic_fragments],
        randomized_fragments=[asdict(x) for x in partition.randomized_fragments],
        ranking_proxy_lambda_r=partition.ranking_proxy_lambda_r,weight_rule=partition.weight_rule,
        partition_hash_policy='ranked_df_prefix_v1')
    preparation=dict(hamiltonian_hash=prep.hamiltonian_hash,partition_hash=prep.partition_hash,
        deterministic_blocks=[block_payload(b) for b in prep.deterministic_blocks],
        tail_hash=prep.tail_extraction.tail_hash,identity_policy=prep.identity_policy,
        coefficient_atol=float(prep.coefficient_atol),basis_construction_policy=prep.basis_construction_policy,
        diagonal_sort=prep.diagonal_sort,product_formula='2nd',
        preparation_hash_policy='df_partial_s2_preparation_v1')
    tail=prep.tail_extraction.canonical_hash_payload_json
    if not tail or digest(tail.encode()) != prep.tail_extraction.tail_hash:
        raise ValueError('complete canonical tail hash input missing or changed')
    if digest(canonical(part).encode())!=prep.partition_hash or digest(canonical(preparation).encode())!=prep.preparation_hash:
        raise ValueError('new partition/preparation identity self-check failed')
    return part,preparation,tail


def check_controls(components, tail_payload, *, a=A, b=B):
    """No old numerical coefficient or basis hash is an acceptance criterion."""
    rows={x.component_id:x for x in components}
    original={x['component_id']:x for x in tail_payload['pre_threshold_components']}
    for label,fragment,support in ((a,'df-fragment-3',(0,1)),(b,'df-fragment-6',(0,))):
        c=rows[label];p=original[label]
        if (c.df_fragment_id!=fragment or tuple(c.diagonal_pauli_support)!=support
            or c.is_identity or c.coefficient_abs<=0 or c.coefficient_sign not in (-1,1)
            or c.coefficient_sign*c.coefficient_abs!=p['coefficient']
            or c.basis_hash!=p['basis_hash']):
            raise ValueError('fixed control semantic mismatch: '+label)
    return {label:dict(fragment=rows[label].df_fragment_id,
        support=list(rows[label].diagonal_pauli_support),sign=rows[label].coefficient_sign,
        coefficient_abs_hex=rows[label].coefficient_abs.hex(),basis_hash=rows[label].basis_hash)
        for label in (a,b)}


def law_payload(connection, *, a=A,b=B,law_id='artificial'):
    cv,order,strata=proposed_control_banks(connection,law_id=law_id,a=a,b=b)
    q0,q2=connection.law.orders.masses;qa=connection.law.labels.mass(a);qb=connection.law.labels.mass(b)
    means=cv.means; joint=tuple(tuple(cv.joint_mean((i,j)) for j in range(3)) for i in range(3))
    expected=((means[0],F(0),F(0)),(F(0),means[1],means[2]),(F(0),means[2],means[2]))
    order_patterns=((1,),(0,))
    patterns=((1,1,0,0),(1,0,0,0),(0,0,0,0),(0,0,1,0),(0,0,1,1))
    two=tuple(order.pattern_mass(x) for x in order_patterns)
    five=tuple(strata.pattern_mass(x) for x in patterns)
    if (joint!=expected or two!=(q0,q2) or five!=(q0*qa,q0*(1-qa),q2*(1-qa),q2*qa*(1-qb),q2*qa*qb)
        or sum(two)!=1 or sum(five)!=1 or min(five)<0):
        raise ValueError('control moments or exhaustive strata mismatch')
    positive_labels=sum(x>0 for x in connection.law.labels.masses)
    return dict(uniform_grid_size=2**53,application_label_order='products_then_rotation',
        Q_definition='Q(order)*product(Q(label_position)); independent uniform-word model',
        component_probability_sum_hex=connection.component_probability_sum.hex(),
        event_support_size=sum(positive_labels**(n+1) for n,q in zip(connection.law.orders.labels,connection.law.orders.masses) if q>0),
        orders=list(connection.law.orders.labels),
        order_masses=[rational(x) for x in connection.law.orders.masses],
        label_masses=[dict(label=l,**rational(q)) for l,q in zip(connection.law.labels.labels,connection.law.labels.masses)],
        order_choice_hex=[x.hex() for x in connection.order_choice_input],
        label_choice_hex=[x.hex() for x in connection.label_choice_input],
        order_cdf_hex=[x.hex() for x in connection.order_cdf],label_cdf_hex=[x.hex() for x in connection.label_cdf],
        positive_grid_support=sum(x>0 for x in connection.law.labels.masses),
        zero_grid_labels=[l for l,q in zip(connection.law.labels.labels,connection.law.labels.masses) if q==0],
        control_means=[rational(x) for x in means],joint_moments=[[rational(x) for x in row] for row in joint],
        order_patterns=order_patterns,order_strata=[rational(x) for x in two],
        five_patterns=patterns,five_strata=[rational(x) for x in five],
        assumptions=['independent uniform 53-bit words; not a PRNG independence proof'],
        unknown_quantum_variance_acquired=False)


def policy_payload(prep, definitions, proofs, *, source_commit):
    expected={(c.basis_id,str(c.basis_hash),tuple(c.diagonal_pauli_support)) for c in prep.rte_preparation.symbolic_tail.components}
    if set(definitions)!=expected or len(proofs)!=len(expected):
        raise ValueError('incomplete selected basis definition coverage')
    keys={(p['source_basis_id'],p['source_basis_hash'],tuple(p['support'])) for p in proofs}
    if keys!=expected or any(not math.isfinite(p['preserved_columns_max_abs_residual']) or
            p['preserved_columns_max_abs_residual']>POLICY_ATOL for p in proofs):
        raise ValueError('selected basis correspondence failed')
    rows=[]
    for key,definition in sorted(definitions.items()):
        if not any(p['selected_basis_hash']==definition.basis_hash and
                   p['source_basis_id']==key[0] and tuple(p['support'])==key[2] for p in proofs):
            raise ValueError('selected basis proof hash mismatch')
        rows.append(dict(source_basis_id=key[0],source_basis_hash=key[1],support=key[2],
                         definition=asdict(definition.metadata)))
    declaration=dict(schema='a1_new_target_selected_policy_v1',source_commit=source_commit,
        preparation_hash=prep.preparation_hash,tail_hash=prep.tail_extraction.tail_hash,
        rule='support_run_le_1',maximum_support_run_length=1,boundary='boundary_optimized',
        controlled='ordinary_diag_I_U',definition_sha256=digest(canonical(rows).encode()))
    return dict(declaration=declaration,policy_fingerprint=digest(canonical(declaration).encode()),
        definitions=rows,proofs=proofs,column_residual_acceptance_atol=POLICY_ATOL,
        maximum_preserved_columns_residual=max((p['preserved_columns_max_abs_residual'] for p in proofs),default=0.),
        full_interrogation_tested=False,full_numerical_operator_bias_certified=False)


def run(output_dir, profile_lock, *, root=None):
    root=Path(root or Path(__file__).resolve().parents[3]).resolve()
    initial_status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root,text=True).strip()
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    counts=dict(snapshot_load=0,partition=0,preparation=0,policy_registration=0,cdf_connection=0,
                sampling=0,state_solve=0,full_build=0,compile=0,measurement=0,variance_N_cost=0)
    started=time.monotonic()
    try:
        dep=dependencies()
        locked=json.loads(Path(profile_lock).read_text());actual_profile=profile(root)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        source=dict(schema='a1_new_target_n1_source_v1',source_commit=commit,adopted_contract_commit=CONTRACT,
            snapshot=SNAPSHOT,snapshot_sha256=SNAPSHOT_SHA,
            profile_sha256=digest(canonical(locked).encode()),profile_lock_sha256=digest(Path(profile_lock).read_bytes()),
            scope=dict(model='linear H4',distance_angstrom=1.,basis='STO-3G',df_rank=12,ld=3,
                m=0,q=1,delta=.02,r=1,K=2,coefficient_atol=0.,axes=['X','Y'],
                sector_electrons=4,system_qubits=8,identity_policy='extract_identity_phase'),
            excluded=['N2','N3','state','RNG','full interrogation build/compile','shots','variance/N/cost'],
            no_retry=True)
        write(out/'SOURCE.json',source)
        if actual_profile!=locked: raise ValueError('source/dependency profile differs from frozen lock')
        if locked['python']!='3.12.3' or locked['distributions']['numpy']['version']!='1.26.4' or locked['distributions']['qiskit']['version']!='1.3.0':
            raise ValueError('required numerical profile mismatch')
        if initial_status:
            raise ValueError('source worktree is not clean before scientific input')
        if digest((root/SNAPSHOT).read_bytes())!=SNAPSHOT_SHA:raise ValueError('snapshot bytes mismatch')
        write(out/'PROFILE.json',locked)
        write(out/'RESOURCE_OBSERVATION.json',dict(cpu_count=os.cpu_count(),load_average=os.getloadavg(),
            affinity_cpu_count=len(os.sched_getaffinity(0)),
            memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))],
            resource_caps_added=False))
        with n1_only(dep):
            counts['snapshot_load']+=1;h=dep.loader.load_connected_cluster_hamiltonian_snapshot(root/SNAPSHOT)
            if h.n_qubits!=8 or h.n_blocks!=12:raise ValueError('fixed H4 dimensions mismatch')
            counts['partition']+=1;partition=dep.split.split_df_hamiltonian_by_ld(h,3)
            counts['preparation']+=1;prep=dep.partial.prepare_df_partial_s2(h,partition,
                identity_policy='extract_identity_phase',coefficient_atol=0.,diagonal_sort='descending_abs',
                basis_construction_policy='fermionic_gaussian_jw',partition_policy='weight_ranked_prefix')
            prep_record=preparation_payload(h,partition,prep,dep.partial._block_hash_payload)
            prep_record.update(constant_coefficient=prep.constant_coefficient,constant_coefficient_hex=prep.constant_coefficient.hex(),
                identity_policy=prep.identity_policy,coefficient_atol=prep.coefficient_atol,diagonal_sort=prep.diagonal_sort,
                basis_construction_policy=prep.basis_construction_policy,
                extraction_metadata=asdict(prep.tail_extraction.extraction_metadata),
                normalization_metadata=asdict(prep.tail_extraction.normalization_metadata))
            write(out/'PREPARATION.json',prep_record)
            part,preparation,tail=hash_payloads(partition,prep,dep.partial._block_hash_payload)
            for name,value in (('PARTITION_HASH_INPUT.json',canonical(part)),
                               ('PREPARATION_HASH_INPUT.json',canonical(preparation)),('TAIL_HASH_INPUT.json',tail)):
                with (out/name).open('x') as f:f.write(value)
            controls=check_controls(prep.rte_preparation.symbolic_tail.components,json.loads(tail))
            write(out/'CONTROLS.json',controls)
            # Existing finite-input arithmetic, without importing or running the old worker.
            distribution0=dep.rte.finite_rte_distribution(prep.exact_rte_lambda_r*.02,2)
            tolerance=max(math.nextafter(distribution0.step_truncation_residual_bound,math.inf),math.ulp(0.))
            config,distribution=dep.rte.make_rte_config(prep.rte_preparation.symbolic_tail,evolution_time=.02,
                rte_steps=1,truncation_tolerance=tolerance,finite_taylor_order=2)
            write(out/'CONFIG.json',dict(config=asdict(config),distribution=asdict(distribution),
                config_float_hex={k:v.hex() for k,v in asdict(config).items() if isinstance(v,float)},
                order_weight_hex=[v.hex() for v in distribution.unnormalized_order_weights]))
            counts['cdf_connection']+=1
            connection=connect_sampler_inputs(prep.rte_preparation.symbolic_tail.components,distribution,
                source=commit+':'+prep.preparation_hash,validate_components=dep.rte._validate_components)
            law=law_payload(connection,law_id=commit+':'+prep.preparation_hash)
            write(out/'LAW.json',law)
            counts['policy_registration']+=1
            definitions,proofs=dep.policy.register_support_restricted_bases(h,prep)
            policy=policy_payload(prep,definitions,proofs,source_commit=commit)
            # Lossless compact metadata: avoid duplicating pretty-print whitespace for all support bases.
            with (out/'POLICY.json').open('x') as f:f.write(canonical(policy)+'\n')
        identities=dict(hamiltonian_hash=prep.hamiltonian_hash,partition_hash=prep.partition_hash,
            preparation_hash=prep.preparation_hash,tail_hash=prep.tail_extraction.tail_hash,
            catalog_sha256=digest((out/'PREPARATION.json').read_bytes()),law_sha256=digest((out/'LAW.json').read_bytes()),
            selected_policy_fingerprint=policy['policy_fingerprint'])
        result=dict(status='N1_PASS_MANDATORY_STOP',source_commit=commit,identities=identities,
            symbolic_components=len(prep.rte_preparation.symbolic_tail.components),
            positive_grid_support=law['positive_grid_support'],controls_eligible=True,
            elapsed_seconds=time.monotonic()-started,counts=counts,
            next_stage_started=False,N2_authorized=False,N3_authorized=False,
            artifact_sha256={p.name:digest(p.read_bytes()) for p in sorted(out.glob('*.json'))})
        write(out/'RESULT.json',result)
        write(out/'STOP.json',dict(status='MANDATORY_STOP',N1_completed=True,no_retry=True,
            next_scope_requires_permission=['N2','N3'],no_state_or_measurement_or_compile=True))
        return result
    except Exception as exc:
        write(out/'STOP.json',dict(status='STOPPED',error_type=type(exc).__name__,error=str(exc),
            no_retry=True,counts=counts,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        write(out/'CALL_COUNTS.json',counts)
