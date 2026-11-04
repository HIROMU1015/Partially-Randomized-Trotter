"""B1: whole-P0 structural budgets and conditional reuse of synthesis tools.

No H4 scientific backend is executable here. A finite diagnostic, a rewrite
count, or a supplied argument's schema is not proof of a uniform error bound.
"""
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io

BASE='ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53'
CANDIDATE='a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
FAMILY='artifacts/research_c_p0_uniform_method/2026-11-03'
CONTRACT='docs/research/research_c_p0_uniform_method_contract_unapproved_20261103.json'
ENTRY='scripts/prepare_research_c_p0_uniform_method.py'
MISSING=('uniform finite-precision 1q product/ZSXX reconstruction and phase error',
         'full512 raw Gaussian/relative/diagonal to exact saved-atom error',
         'initial definition/translation and accumulated declared-phase error',
         'all-word successful compiler output and actual inverse/shell coverage')


def fixed():
    return dict(scope='C_B1_WHOLE_P0_UNIFORM_METHOD_V1',base_commit=BASE,
        candidate_sha256=CANDIDATE,status='P0_UNIFORM_NOT_ESTABLISHED',
        executable=False,maximum_logical_units=None,family_count=None,
        family='all216_atom_words_length32',controlled_dimension=512,
        Stage_C='NOT_READY',retry=0,resume=0,mandatory_stop=True,
        fresh_output=FAMILY+'/run01',missing=list(MISSING))


def rewrite_budget(oneq,cx):
    """Conditional count theorem AFTER native-basis translation, no bound yet.

    Fixed P0: no target/calibration; native RZ/SX/X/CX only. Rust substitutes
    an in-basis 1Q run only for a strictly shorter sequence. ZSXX emits <=5
    gates; inverse cancellation deletes gates. All later translation is idle.
    Each accepted rewrite consumes >=1 of the initial Q one-qubit nodes.
    Candidate matrix multiplications can still recur on unchanged runs.
    """
    if any(type(x) is not int or x<0 for x in (oneq,cx)):
        raise ValueError('NONNEGATIVE_GATE_COUNTS')
    from .c_pr_aware_df_c2a import monotone_budget
    from .c_pr_aware_df_zsxx_uniform import CONDITIONS
    total=oneq+cx
    structural=monotone_budget(total,**CONDITIONS) if total else None
    iterations=min(1000,total+1)
    return dict(initial_one_qubit=oneq,initial_cx=cx,initial_size=oneq+cx,
        max_1q_run_length=oneq,accepted_1q_rewrites_max=structural['accepted_rewrites'] if structural else 0,
        accepted_gate_visits_max=structural['accepted_gate_visits'] if structural else 0,
        phase_updates_max=structural['phase_additions'] if structural else 0,successful_output_iterations_max=iterations,
        candidate_matrix_products_max=iterations*oneq,
        candidate_resyntheses_max=iterations*oneq,ZSXX_output_gates_max=5,
        assumptions='post-translation native RZ/SX/X/CX;target=None;no calibrations/control flow;fixed opt1 passes',
        proves_all_words_compile_successfully=False,uniform_error_bound=None)


def primitive_translation_inventory():
    """Symbolic artificial standard gates ONLY; never any H4 input/circuit.

    Parameterized control uses annotated=False to select the ordinary gate
    definition used by bound numeric production inputs. No P0 settings change.
    This is a definition/count diagnostic, not a floating-error theorem.
    """
    from .c_pr_aware_df_c2a import symbolic_templates
    templates=symbolic_templates()['templates'];result={}
    for label,name in dict(xx_plus_yy='XX',crz='controlled_RZ',crzz='controlled_RZZ',p='phase').items():
        t=templates[name];counts=t['lowered_counts']
        if not set(counts)<=set(('u','rz','sx','x','cx')):raise ValueError('UNKNOWN_DEFINITION')
        # Reuse the prior SOURCE argument: U has a <=5 one-qubit-node
        # translation witness, with additive unit native gate costs.
        result[label]=dict(lowered_counts=counts,native_counts_upper=dict(
            one_qubit=5*counts.get('u',0)+sum(counts.get(x,0) for x in ('rz','sx','x')),
            cx=counts.get('cx',0)),declared_phase=t['global_phase'])
    result['rz']=dict(lowered_counts={'rz':1},native_counts_upper={'one_qubit':1,'cx':0},declared_phase='0')
    return dict(scope='artificial symbolic primitive definitions;not numerical H4',
        primitives=result,phase_error_enclosed=False,uniform_error_bound=None)


def grammar_budget(inventory):
    """Conservative syntactic counts for existing r32 native grammar.

    68 ordinary Gaussian occurrences+4 relative transitions; each emits
    <=8 RZ and28 XX+YY. Six head diagonals emit <=8 CRZ,28 controlled RZZ,
    one P each. Central K0 <=2 RZ+4 CX per event. Four extra P cover the
    two literal zero-time head scalars,event phase,and total scalar/identity.
    Inverse-cancellation before translation only removes nodes.
    """
    from collections import Counter
    from .c_pr_aware_df_c2a import primitive_grammar
    old=primitive_grammar();assert old['modes']==8 and old['head_factors']==6
    p=inventory['primitives'];counts=Counter()
    multiplicities=dict(rz=72*8+32*2,xx_plus_yy=72*28,
        crz=6*8,crzz=6*28,p=6+4)
    for name,n in multiplicities.items():
        for gate,value in p[name]['native_counts_upper'].items():
            if gate not in ('one_qubit','cx') or type(value) is not int or value<0:
                raise ValueError('PRIMITIVE_COUNT_SCHEMA')
            counts[gate]+=n*value
    counts['cx']+=32*4
    budget=rewrite_budget(counts['one_qubit'],counts['cx'])
    return dict(native_gate_upper_counts=dict(counts),primitive_multiplicities=multiplicities,
        rewrite_budget=budget,coverage='syntactic upper count under fixed standard definitions;not a synthesis-error proof',
        H4_arrays_decoded=0,H4_circuits_built=0,uniform_error_bound=None)


def conditional_compiler_bound(grammar,**proof_inputs):
    """Route current size to the unchanged prior all-domain composer.

    The old run-product/ZSXX/translation/phase engine already accounts for
    accepted visits and refuses captured images. Do not duplicate its algebra.
    """
    from .c_pr_aware_df_zsxx_uniform import uniform_whole_bound,CONDITIONS
    budget=grammar['rewrite_budget']
    if budget!=rewrite_budget(budget['initial_one_qubit'],budget['initial_cx']):
        raise ValueError('REWRITE_BUDGET_DRIFT')
    return uniform_whole_bound(budget['initial_size'],conditions=CONDITIONS,**proof_inputs)


def conditional_rows(kernel,shells,argument,*,expected_identity,observed_identity):
    """Reuse existing raw-complex propagation, never an H4 coverage factory."""
    from . import c_all_law_signal_ca as old
    if expected_identity!=observed_identity:raise ValueError('SOURCE_INPUT_COMPILER_IDENTITY')
    required=dict(family='all216_atom_words_length32',final_output='whole_kernel_P0',
        dimension=512,phase_sensitive=True,actual_inverse=True,shells=True,
        compiler_success_all_words=True,kind='supplied_structural_uniform_argument')
    if any(argument.get(k)!=v for k,v in required.items()) or not argument.get('provenance'):
        raise ValueError('UNIFORM_FINAL_OUTPUT_ARGUMENT_REQUIRED')
    if kernel.get('operator_bound_pilot') is None or any(shells.get(k,{}).get('operator_bound_pilot') is None for k in ('head','X','Y')):
        return dict(status='NOT_READY_BOUND_INPUTS',rows=None,actual_H4_qualification=None)
    coverage=dict(scope=old.COVERAGE,compiler_path='whole_kernel_P0',final_transform_covered=True,
        phase_sensitive=True,actual_adjoint_covered=True,dimension=512,
        provenance=argument['provenance'],kind='all_law_numerical_bound')
    rows=old.numerical_synthesis_rows(kernel,shells,coverage)
    useful=all(x['bound']<=math.sin(math.pi/12) for x in rows)
    return dict(status='CONDITIONAL_SYNTHESIS_ONLY_PHASE_ROOM_POSSIBLE' if useful else 'INFEASIBLE_SYNTHESIS_ONLY_BOUND',
        rows=rows,phase_room_test='synthesis alone versus sin(pi/12);not full RPE feasibility',
        proof_inputs_verified_by_schema=False,actual_H4_qualification=None,
        actual_CA_achieved=None,rigorous_CA=None,PCG64_joint_stream_qualified=False)


def diagnose_artificial_circuit(raw,reference):
    """Whole unchanged P0+full operator+actual inverse, finite diagnostic only."""
    from .c1_h4_native_fair import decode
    from .c1_h4_full_synthesis import segment
    from .c_pr_aware_df_stage_b2 import adjoint_pair_gate
    from .c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
    from .c_pr_aware_df_c2a import trace_artificial
    compiled,trace=trace_artificial(raw)
    R,C=decode(raw),decode(compiled)
    s=segment(R,C,reference,'B1 artificial whole-P0')
    lock=SharedQiskitCalls(lambda k,l,f:f())
    with lock.adjoint_builds():_,inverse=adjoint_pair_gate(raw,compiled,R,C,reference)
    return compiled,dict(segment=s,actual_inverse=inverse,trace=trace,
        coverage='finite_artificial_family_only',all_law_qualified=False,rigorous_bound=None)


def preflight(root,path):
    """Read metadata/bytes only; no candidate NPZ/QPY decoding or rerun."""
    root=Path(root).resolve();c=io.read(path)
    if any(c.get(k)!=v for k,v in fixed().items()):raise ValueError('FIXED_B1_DESIGN')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    source=io.read(root/c['source_manifest'])
    for group in (source['files'],c['saved_inputs']):
        for name,want in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=want:raise ValueError('SOURCE_INPUT:'+name)
    for name,want in c['packages'].items():
        if importlib.metadata.version(name)!=want:raise ValueError('PACKAGE:'+name)
    qroot=Path(importlib.metadata.distribution('qiskit').locate_file('qiskit'))
    for name,want in c['installed_compiler_sources'].items():
        if io.sha(qroot/name)!=want:raise ValueError('INSTALLED_COMPILER_SOURCE:'+name)
    a=c['stage_A'];stop=io.read(root/a['STOP']);saved=io.read(root/a['contract'])
    if (stop['failed'] or stop['scientific_units']!=38 or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or saved['candidate_sha256']!=CANDIDATE or saved['compiler']!=c['compiler']):
        raise ValueError('STAGE_A_OR_COMPILER_IDENTITY')
    if (root/c['fresh_output']).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return dict(status='PASSIVE_B1_DESIGN_CHECKS_PASS',scientific_calls=0,
        H4_execution_ready=False,uniform_coverage_established=False,
        source_files=len(source['files']),saved_inputs=len(c['saved_inputs']),
        method_status='P0_UNIFORM_NOT_ESTABLISHED',maximum_logical_units=None,
        Stage_C='NOT_READY',actual_CA_achieved=None,mandatory_stop=True)


def run_science(*,execute_science=False):
    if not execute_science:raise PermissionError('SEPARATE_SCIENTIFIC_PERMISSION_REQUIRED')
    raise NotImplementedError('P0_UNIFORM_NOT_ESTABLISHED;NO_H4_EXECUTABLE_CONTRACT')
