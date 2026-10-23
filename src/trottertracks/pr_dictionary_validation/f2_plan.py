"""F2 integer descriptors only. No events, laws, quantum IR or numeric target."""
ARMS=('SS','SP','PS','PP')
LENGTHS=dict(SS=14,SP=58,PS=26,PP=51)
CONTEXT=dict(T='1/5',q=1,r=2,K=0,delta='1/5',h='1/10')


def counts(lengths=None):
    lengths=LENGTHS if lengths is None else lengths
    if set(lengths)!=set(ARMS) or any(type(n) is not int or n<1 for n in lengths.values()):
        raise ValueError('Four positive integer dictionary lengths required')
    pairs=sum(n*n for n in lengths.values()); wrappers=2*pairs+2
    return dict(law_bindings=4,ordered_trajectory_pairs=pairs,
        full_wrapper_build_requests=wrappers,compile_requests=wrappers,
        raw_operator_checks=wrappers,raw_signal_checks=wrappers,
        compiled_operator_checks=wrappers,compiled_signal_checks=wrappers,
        reference_precision_evaluations=2*(wrappers+2),
        mean_comparisons=4,probability_mass_checks=4,shot_qualifications=4,
        physical_bias_diagnostics=1,conditional_selector=1,known_four_search=1,
        resource_aggregation=1,quantum_measurement_draws=0)


def iter_jobs(lengths=None):
    """One shared evolution for both axes; diagnostics have no physical evolution."""
    lengths=LENGTHS if lengths is None else lengths
    counts(lengths)
    yield ('diagnostic',0,0)
    for arm in ARMS:
        for i in range(lengths[arm]):
            for j in range(lengths[arm]):
                yield (arm,i,j)


def case_id(arm,i,j,axis):
    if axis not in ('X','Y'): raise ValueError('Axis')
    return f'{arm}_{i:03}_{j:03}_{axis}'


def compile_groups(records):
    """Deduplicate only keys produced from complete, bound IR after approval."""
    groups={}; seen=set()
    for record in records:
        if record['id'] in seen: raise ValueError('Duplicate logical wrapper')
        seen.add(record['id'])
        groups.setdefault(record['compile_key'],[]).append(record['id'])
    return tuple((key,tuple(sorted(ids))) for key,ids in sorted(groups.items()))

# Saved B input bytes; no F1 law or price outputs are reused.
SAVED_INPUT_HASHES={'handoff/f1_bundle.json': '9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0', 'inputs/physical.json': 'd796385d5bc5f52fa2fbffe9d4f327b738055dae0420f4d255b40d862ea14658', 'requirements.lock': 'c0e7e23e5c2c06931ab23cdf247ddea372d5c62b83c4a8cd07c42e3bf6c38eb0', 'arms/SS.json': 'e08985a9b560d49f7e6694d3d36c23e462bb1f996a9dd8f542e589d346e10358', 'arms/SP.json': 'df5dd1aa4896286015cd757bad303fee18745a80e6bca3f949038c7c64127c6d', 'arms/PS.json': '8c9f0a2e739f2829cece524568523afa66487dbd7dd52b1bfd22a4ca182e16fb', 'arms/PP.json': '4c036eb36262930669650e356accdc129aab0ec5f14bbbdac144fd55e1009a71'}

COMPILER={'approximation_degree': 1.0, 'basis_gates': ['rz', 'sx', 'x', 'cx'], 'optimization_level': 1, 'qiskit': '1.3.0', 'qubits_initially_zero': False, 'seed_transpiler': 17, 'topology': 'free'}
THRESHOLDS={'B_coefficient': '1e-12*max(1,reference_magnitude)', 'DP_double': '1e-10*max(1,price)', 'DP_high_precision': '1e-40*max(1,price)', 'H_Hermitian_involution': '1e-11', 'angle': '1e-12', 'compiled_operator_signal': '5e-10', 'controlled_unitarity': '1e-10', 'mean_signal': '2e-10', 'paired_reference': '1e-60', 'phase_fitting': False, 'probability_mass_normalization_L1': '1e-12', 'rtol': 0}
