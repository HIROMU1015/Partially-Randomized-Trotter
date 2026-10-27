"""C2-A technical adapters. No molecular decoder, build or science endpoint.

Proof-input previews do not verify proofs or qualify the saved H4 experiment.
The existing compiler is observed, never replaced or patched in production.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io

VERSION = 'C2A_fixed_P0_ZSXX_coverage_v1'
SCOPE = 'C_PR_AWARE_DF_C2A_SYNTHESIS_READINESS_V1'
FAMILY = 'artifacts/research_c_pr_aware_df_c2a/2026-10-27'
P0 = dict(optimization_level=1, basis_gates=['rz', 'sx', 'x', 'cx'],
          seed_transpiler=17, approximation_degree=1., qubits_initially_zero=False)
RAW_IDS = ('gaussian_basis', 'relative_gaussian', 'head_diagonal',
           'signed_event', 'scalar_identity')
P0_IDS = ('official_unroll', 'basis_translation', 'run_product', 'ZSXX_angles',
          'ZSXX_simplification', 'mod2pi_phase', 'phase_accumulation')
SHELLS = ('head', 'X', 'Y')


def check_graph(stages):
    expected = dict(pre_init=['ContainsInstruction','ConditionalController','Error'],
        init=['InverseCancellation'], translation=['UnitarySynthesis','HighLevelSynthesis','BasisTranslator'],
        pre_optimization=[], optimization=['Depth','FixedPoint','Size','FixedPoint',
            'DoWhileController','Optimize1qGatesDecomposition','InverseCancellation','GatesInBasis',
            'ConditionalController','FlowControllerLinear','UnitarySynthesis','HighLevelSynthesis',
            'BasisTranslator','Depth','FixedPoint','Size','FixedPoint'],
        scheduling=['ContainsInstruction','ConditionalController','TimeUnitConversion'])
    def flatten(rows):
        for row in rows:
            yield row
            yield from flatten(row.get('nested',[]))
    flat={k:list(flatten(v)) for k,v in stages.items()}
    mismatches=[k for k in set(expected)|set(flat)
                if [x['name'] for x in flat.get(k,[])]!=expected.get(k)]
    optimizers=[r for rows in flat.values() for r in rows if r['name']=='Optimize1qGatesDecomposition']
    cancellations=[r for rows in flat.values() for r in rows if r['name']=='InverseCancellation']
    if (len(optimizers)!=1 or not optimizers[0].get('target_none')
            or optimizers[0].get('decomposers')!=['ZSXX'] or not optimizers[0].get('error_map_none')
            or len(cancellations)!=2 or not all(r.get('parameter_free') for r in cancellations)):
        mismatches.append('NUMERIC_OPTIMIZER_OR_CANCELLATION_POLICY')
    return dict(status='UNRESOLVED_PINNED_GRAPH_MISMATCH' if mismatches else 'PINNED_GRAPH_MATCH_NOT_NUMERIC_CERTIFICATE',
                mismatches=mismatches, all_law_qualified=False)


def _positive(n):
    if type(n) is not int or n < 1:
        raise ValueError('POSITIVE_INTEGER')
    return n


def monotone_budget(initial_nodes, *, basis_closed, target_none,
                    zsxx_only, parameter_free_cancellation, no_calibrations):
    """Conditional combinatorial proof, not a numerical synthesis bound.

    In-basis target=None accepted rewrites strictly reduce node count. ZSXX
    emits <=5 nodes. Accepted input gates are removed once; all introduced
    nodes total <=5R, hence accepted matrix-product visits <=N+5R<=6N.
    Skipped rewrites do not affect the output or accumulate synthesis error.
    """
    _positive(initial_nodes)
    assumptions = (basis_closed, target_none, zsxx_only,
                   parameter_free_cancellation, no_calibrations)
    if any(x is not True for x in assumptions):
        return dict(status='UNRESOLVED_MONOTONE_GRAMMAR', accepted_rewrites=None,
                    accepted_gate_visits=None, max_run_length=None, iterations=None)
    return dict(status='CONDITIONAL_STRUCTURAL_COUNT', initial_nodes=initial_nodes,
                accepted_rewrites=initial_nodes, accepted_gate_visits=6*initial_nodes,
                max_run_length=initial_nodes, iterations=initial_nodes+1,
                output_gates_per_ZSXX=5, phase_additions=initial_nodes,
                numerical_error_bound=None, all_law_qualified=False)


def primitive_grammar():
    """Counts from emitter loops and fixed metadata, not saved H4 arrays."""
    return dict(status='RAW_GRAMMAR_REQUIREMENTS', modes=8, events=16,
        atom_alphabet=216, tail_frames=6, bases=9, directed_head_transitions=4,
        head_factors=6, tail_basis_runs=[1, 16],
        gaussian_occurrences_upper=40, gaussian_RZ_upper=8,
        gaussian_XXPlusYY_upper=28, head_RZ_upper_per_factor=8,
        head_RZZ_upper_per_factor=28, tail_RZ_upper_per_event=2,
        tail_CX_upper_per_event=4, scalar_identity_P_upper=7,
        arbitrary_tail_relative_fusion=False, head_tail_fusion=False,
        primitive_ids=list(RAW_IDS), compiler_ids=list(P0_IDS), shells=list(SHELLS),
        adjoint='actual inverse construction;separate rounding coverage required',
        input_restrictions='finite numeric,unitary target,no measure/reset/delay/control-flow/calibration',
        H4_numeric_decode=False, worst_case_translated_nodes=None,
        all_law_synthesis=None)


def compiler_coverage():
    """Read fixed preset attributes. No circuit is built or run."""
    from .c_pr_aware_df_stage_c2_audit import compiler_audit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    value = compiler_audit()
    pm = generate_preset_pass_manager(**P0)
    def graph(controller):
        out = []
        for task in controller.tasks:
            row = dict(name=type(task).__name__)
            if hasattr(task, 'tasks'):
                row['nested'] = graph(task)
            elif row['name'] == 'Optimize1qGatesDecomposition':
                row.update(target_none=task._target is None,
                           decomposers=task._global_decomposers,
                           error_map_none=task.error_map is None)
            elif row['name'] == 'InverseCancellation':
                gates = [*task.self_inverse_gates,
                         *(g for pair in task.inverse_gate_pairs for g in pair)]
                row['parameter_free'] = all(not g.params for g in gates)
                row['gates'] = [g.name for g in gates]
            out.append(row)
        return out
    stages = {name:graph(getattr(pm, name).to_flow_controller())
              for name in pm.expanded_stages if getattr(pm, name) is not None}
    value.update(expanded_stage_graph=stages, graph_check=check_graph(stages),
        active_1q_implementation='Rust optimize_1q_gates_decomposition',
        numerical_branches=['theta near0', 'theta near pi/2', 'theta near pi',
            'lambda+pi or phi near0', 'RZ drop', 'SX pair versus X',
            'atan2/arg/determinant', 'mod2pi endpoint', 'f64 phase plus MPFR expression'],
        inactive_under_grammar=['calibration skip', 'non-global target/error map',
            'layout/routing', 'opaque UnitaryGate synthesis', 'HLS plugins/control flow',
            'arbitrary parameterized inverse cancellation'],
        path_conditions='Requires all inputs lower to u/rz/sx/x/cx;unknown gate/target/path is not covered',
        structural_argument='once basis-closed,target=None:strict gate-count decrease;ZSXX emits at most5',
        uniform_ZSXX_error=None, initial_basis_translation_enclosure=None,
        reference_enclosure=None, scientific_calls=0, all_law_qualified=False)
    return value


def symbolic_templates():
    """Artificial symbolic gate definitions, not molecular circuit costs.

    Only unrolls official definitions to basis+u under the existing mutex.
    Does not assume the final BasisTranslator chooses a proved <=5 rule.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import XXPlusYYGate, RZGate, RZZGate, PhaseGate
    from qiskit.circuit.equivalence import EquivalenceLibrary
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollCustomDefinitions
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    mutex = SharedQiskitCalls(lambda kind, label, fn:fn())
    a, b = Parameter('invented_a'), Parameter('invented_b')
    gates = mutex('kernel_build', 'artificial symbolic controls', lambda:dict(
        XX=XXPlusYYGate(a, b), XX_inverse=XXPlusYYGate(a, b).inverse(),
        controlled_RZ=RZGate(a).control(1, annotated=False),
        controlled_RZZ=RZZGate(a).control(1, annotated=False), phase=PhaseGate(a)))
    result = {}
    for name, gate in gates.items():
        def lower(gate=gate):
            q = QuantumCircuit(gate.num_qubits); q.append(gate, range(gate.num_qubits))
            return PassManager([UnrollCustomDefinitions(EquivalenceLibrary(),
                P0['basis_gates']+['u'])]).run(q)
        lowered = mutex('kernel_build', 'artificial:'+name, lower)
        counts = dict(lowered.count_ops())
        if set(counts)-set(P0['basis_gates'])-{'u'}:
            raise ValueError('UNKNOWN_LOWERED_TEMPLATE')
        result[name] = dict(lowered_counts=counts, symbolic=True,
                           global_phase=str(lowered.global_phase),
                           explicit_control_definitions='annotated=False for symbolic proxy of numeric production controls')
    return dict(status='ARTIFICIAL_SYMBOLIC_TEMPLATES', templates=result,
                actual_H4_build=False, all_law_numeric_enclosure=None)


def trace_artificial(circuit):
    """Observe the actual fixed wrapper on explicitly artificial input.

    Caller owns input provenance. No saved-input loader exists in this module.
    The callback only observes the existing transpile call and never rewrites it.
    """
    import qiskit
    from unittest.mock import patch
    from .c1_h6_compiler_precision import compiled_wrapper
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    rows = []; original = qiskit.transpile
    def callback(**kw):
        dag = kw['dag']; runs = dag.collect_1q_runs()
        rows.append(dict(pass_name=type(kw['pass_']).__name__, count=kw['count'],
            size=dag.size(), depth=dag.depth(), basis=sorted(dag.count_ops()),
            phase=str(dag.global_phase), run_lengths=[len(r) for r in runs],
            runs=[dict(qubit=dag.find_bit(r[0].qargs[0]).index,
                       gates=[n.name for n in r],
                       parameters=[[str(p) for p in n.op.params] for n in r]) for r in runs]))
    def observed(*args, **kwargs):
        if 'callback' in kwargs:raise ValueError('CALLBACK_COLLISION')
        return original(*args, **kwargs, callback=callback)
    mutex = SharedQiskitCalls(lambda kind, label, fn:fn())
    with patch.object(qiskit, 'transpile', observed):
        compiled = mutex('compile', 'artificial fixed P0',
                         lambda:compiled_wrapper(circuit, CompilerProfile()))
    return compiled, dict(status='ARTIFICIAL_WHOLE_P0_TRACE', passes=rows,
                          profile=P0, all_law_proof=False)


def _proof(row):
    if row is None or row.get('epsilon') is None:return None
    if (row.get('kind') not in ('analytic_unitary_bound', 'enclosed_unitary_bound')
            or row.get('norm') != 'full_controlled_operator_2'
            or row.get('phase_sensitive') is not True or not row.get('provenance')
            or row.get('excludes_CAR_bridge') is not True):
        raise ValueError('LOCAL_PROOF_SCOPE_NOT_OBSERVED_RESIDUAL')
    from .c_pr_aware_df_stage_c2 import rational
    eps = rational(row['epsilon'])
    if eps < 0:raise ValueError('NEGATIVE_BOUND')
    return eps


def uniform_preview(*, m, raw, compiler, reference, shells, coverage,
                    exact_unitary_targets, reference_invariant):
    """Conditional proof-input telescope; no H4 qualification or shot call.

    Raw and P0 cover every supported trajectory with the same exact target.
    Reference is numerical evaluation uncertainty only;the CAR/atom bridge
    remains a separate round-level signal term. Each error appears once.
    """
    if type(m) is not int or not 0 <= m <= 12:raise ValueError('REGISTERED_ROUND')
    if set(raw) != set(RAW_IDS) or set(compiler) != set(P0_IDS) or set(shells) != set(SHELLS):
        raise ValueError('FINITE_PRIMITIVE_OR_REWRITE_COVERAGE')
    missing = []
    if exact_unitary_targets is not True or reference_invariant is not True:
        missing.append('exact_unitary_invariant_reference')
    for k in ('all_trajectory_words', 'whole_P0', 'initial_translation',
              'ZSXX_all_branches', 'rounding_model', 'monotone_conditions', 'phase_policy'):
        if coverage.get(k) is not True:missing.append(k)
    groups = dict(raw=raw, compiler=compiler, reference={'reference':reference}, shells=shells)
    sums = {}; primitive_values = {}
    for group, values in groups.items():
        total = Fraction()
        for key, row in values.items():
            value = _proof(row)
            if value is None:missing.append(group+':'+key)
            else:
                if group in ('raw', 'compiler', 'reference') and row.get('coverage') != 'all_trajectory_words':
                    raise ValueError('SAMPLE_OR_EVENT_P0_IS_NOT_WHOLE_LAW')
                if group=='shells' and row.get('coverage')!='fixed_shell:'+key:
                    raise ValueError('SHELL_SCOPE')
                total += value;primitive_values[group+':'+key] = value
        sums[group] = total
    base = dict(m=m, occurrences=4*(1<<m), missing=missing,
        proof_inputs_verified=False, CAR_bridge_included=False, law_included=False,
        attenuation_factor_applied=False, actual_H4_bound=None, error_evidence=None,
        guaranteed_shots=None, guaranteed_total=None, CA_achieved=None,
        usefulness='STRUCTURAL_PREVIEW_ONLY;no saved phase-room comparison')
    if missing:return dict(base, status='UNRESOLVED_INPUTS', candidate_raw_complex_bound=None)
    from .c_pr_aware_df_c2_bridge import uniform_synthesis_preview
    from .c1_subspace_synthesis_bound import repeat_bound
    from .c1_stage3_deterministic_accuracy import round_synthesis
    def packed(value):return dict(kind='analytic_unitary_bound', norm='full_controlled_operator_2',
        phase_sensitive=True, epsilon=str(value), reference_error='0',
        provenance='conditional supplied proofs;not certified here', compiler_path='whole_kernel_P0',
        coverage='all_final_whole_kernel_outputs')
    kernel = sums['raw']+sums['compiler']+sums['reference']
    preview = uniform_synthesis_preview([packed(kernel)],
        {k:packed(primitive_values['shells:'+k]) for k in SHELLS}, m)
    # Existing numerical surrogates are diagnostics,not rigorous promotion.
    diagnostic = repeat_bound(float(kernel), base['occurrences'], reference_invariant=True)
    shell_pilot = {k:dict(unitary_distance_pilot=float(primitive_values['shells:'+k])) for k in SHELLS}
    round_pilot = round_synthesis(dict(unitary_distance_pilot=float(kernel)), shell_pilot, base['occurrences'])
    return dict(base, status='CONDITIONAL_ON_SUPPLIED_PROOFS',
        kernel_components_exact={k:str(v) for k,v in sums.items() if k!='shells'},
        candidate_raw_complex_bound=preview['candidate_raw_complex_bound'],
        repeat_diagnostic=diagnostic, round_diagnostic=round_pilot)


def preflight(root, path):
    """Hash/metadata only. A blocked draft is not an executable contract."""
    root = Path(root).resolve(); c = io.read(path)
    if (c['scope'] != SCOPE or c['implementation'] != VERSION
            or c['status'] != 'BLOCKED_UNAPPROVED_SCIENCE_DESIGN'
            or c['executable'] is not False or c['maximum_logical_units'] is not None
            or c['compiler'] != P0 or c['retry'] or c['resume']):
        raise ValueError('C2A_TECHNICAL_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for p, h in group.items():
            target = (root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:
                raise ValueError('SOURCE_OR_INPUT:'+p)
    for p, want in c['packages'].items():
        if importlib.metadata.version(p)!=want:raise ValueError('ENVIRONMENT:'+p)
    from importlib.util import find_spec
    qiskit_root=Path(find_spec('qiskit').origin).parent
    for p, h in c['external_python_source_hashes'].items():
        target=(qiskit_root/p).resolve()
        if not target.is_relative_to(qiskit_root.resolve()) or io.sha(target)!=h:
            raise ValueError('PINNED_COMPILER_SOURCE:'+p)
    return dict(status='PASSIVE_IDENTITIES_PASS_SCIENCE_BLOCKED', scientific_calls=0,
                executable=False, maximum_logical_units=None, blockers=c['blockers'])
