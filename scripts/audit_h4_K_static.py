#!/usr/bin/env python3
"""Extend the previous bounded historical compiler audit; never run Qiskit."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('previous_binding_CLI',ROOT/'scripts/verify_h4_evidence_binding.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def audit():
    manifest=previous.b.T.read_json_bytes((ROOT/'artifacts/non_df_h4_proof_binding/2026-10-23/evidence_manifest.json').read_bytes())
    store=previous.b.EvidenceStore(ROOT,manifest)
    report=previous.inspect_compiler(store)
    seal=store.document('C_seal');sources={}
    names=['optimization/inverse_cancellation.py','optimization/optimize_1q_decomposition.py',
           'basis/basis_translator.py','synthesis/high_level_synthesis.py','synthesis/unitary_synthesis.py']
    suffixes={name:'qiskit/transpiler/passes/'+name for name in names}
    suffixes.update(control='qiskit/circuit/add_control.py',equivalences='qiskit/circuit/library/standard_gates/equivalence_library.py',
                    RZ='qiskit/circuit/library/standard_gates/rz.py',XXPlusYY='qiskit/circuit/library/standard_gates/xx_plus_yy.py')
    for name,suffix in suffixes.items():
        paths=[p for p in seal['legacy_runtime_files'] if p.endswith('/'+suffix)]
        previous.b.require(len(paths)==1,'K_SEALED_SOURCE_MISSING')
        path=Path(paths[0]);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        previous.b.require(digest==seal['legacy_runtime_files'][str(path)],'K_RUNTIME_SOURCE_HASH')
        text=raw.decode();tree=ast.parse(text)
        sources[name]=dict(path=str(path),sha256=digest,functions=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno)
            for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)],text=text)
    def location(name,needle):
        d=sources[name];lines=[i for i,line in enumerate(d['text'].splitlines(),1) if needle in line]
        previous.b.require(bool(lines),'K_SOURCE_ANCHOR')
        return dict(path=d['path'],sha256=d['sha256'],lines=lines,needle=needle)
    def row(name,condition,math_claim,precision,phase,T_scope,missing,source,category):
        return dict(transformation=name,application_condition=condition,mathematical_identity=math_claim,
                    finite_precision_obligation=precision,controlled_phase_obligation=phase,
                    T_proved_scope=T_scope,unproved=missing,source=source,resolution_class=category,
                    actual_historical_invocation='UNKNOWN_WITHOUT_PROGRAM_OR_PASS_TRACE',
                    full_pass_semantics_proved=False,status='MISSING_EVIDENCE')
    rows=[row('InverseCancellation','Configured self-inverse gates or inverse pairs, adjacent matching DAG nodes',
        'A^-1 A=I for exact operators; tensor-Pauli and H self-inverse identities checked in T',
        'Fixed configured pairs have exact symbolic identities; compiled matching/DAG mutation are implemented in accelerated code, not verified here',
        'Cancelling an inverse pair must also cancel its full phase',
        'Adjacent free-word inverse/Pauli/H cancellation only',
        'Accelerated matcher, all configured SX/S/T/CX/etc pairs and actual old DAG applications are not covered by T',
        location('optimization/inverse_cancellation.py','from qiskit._accelerate.inverse_cancellation'), '1_exact_lemma_plus_2_binding_and_4_missing_trace'),
      row('Optimize1qGatesDecomposition','Collected1Q runs, available Euler bases and selected replacement',
        'Exact Euler identities hold only with correct parameters and global phase',
        'Accumulated2x2 binary64 matrix, accelerated Euler angles, selection and gate conversion need a validated uniform error bound',
        'out_dag.global_phase is copied; dropping system phase before control changes relative phase',
        'RZ/inverse word identities do not prove Euler resynthesis',
        'No accelerated numerical proof or old run/pass trace',location('optimization/optimize_1q_decomposition.py','unitary_to_gate_sequence'), '3_new_numeric_proof_and_4_missing_trace'),
      row('Controlled gate decomposition','Control construction unrolls gate definitions and tracks global phase',
        'Controlled(e^(i phi)U)=diag(I,e^(i phi)U); control0/1 must match',
        'Gate angles, pi constants and accumulated phase arithmetic need quantified errors',
        'Definition phase must become relative ancilla phase, including identity/sign',
        'Abstract controlled branches with exact phase only',
        'Real decomposition and all phase arithmetic are not certified',location('control','global_phase = 0'), '2_binding_plus_3_numeric_proof'),
      row('BasisTranslator/equivalence library','Source basis searched against target/equivalence library and DAG replacements selected',
        'Each exact equivalence rule requires correct angle/phase and arity',
        'Parameterized substitutions and stored floating constants need transfer; source itself is not a roundoff certificate',
        'Each rule must retain full phase in an already-controlled wrapper',
        'Limited symbolic word equality only',
        'Rule coverage, selected search path, accelerated translation and old DAG are missing',location('basis/basis_translator.py','return base_run('), '2_binding_plus_3_numeric_and_4_trace'),
      row('HighLevelSynthesis','High-level operations, synthesis plugins, controlled definitions and target-supported lowering',
        'Plugin-specific decompositions require individual proofs and ancilla preconditions',
        'Any numeric synthesis/angle manipulation remains unbounded here',
        'Control-state, clean ancilla and global phase must survive lowering',
        'No full HLS proof in T',
        'All plugin choices/branches and old trace not known',location('synthesis/high_level_synthesis.py','class HighLevelSynthesis'), '2_binding_plus_3_numeric_and_4_trace'),
      row('UnitarySynthesis','Translator includes synthesis path for selected unitary operations',
        'Exact decomposition versus approximated decomposition must be distinguished',
        'approximation_degree, plugins and floating kernels require guarantees',
        'Global phase and target mapping cannot be inferred from a seed',
        'Unsupported by T; source presence is only a candidate path',
        'Actual invocation, numerical synthesis bound and program absent',location('synthesis/unitary_synthesis.py','approximation_degree'), '3_new_numeric_and_4_trace'),
      row('DF basis / wrapper','Inverse local operations in reversed order, diagonal control and H/SDG wrapper',
        'Ideal basis inverse cancels and bit mean equals Re/Im controlled expectation',
        'Rounded forward/inverse matrices differ from ideal RZ/XXPlusYY; L isolates that gap at128bit',
        'Identity/event sign is relative phase on the control, not discardable global phase',
        'Formal basis/control and cosine/sine lemmas already checked in T',
        'Literal decomposition/compiled transfer still missing after L',location('XXPlusYY','def inverse('), 'L_local_proof_preparation_plus_K_missing'),
    ]
    return dict(kind='K_EXTENDED_STATIC_SOURCE_GUARANTEE_FEASIBILITY',previous_bounded_audit=report,
        rows=rows,source_files=[{k:v for k,v in d.items() if k!='text'} for d in sources.values()],
        proved='Exact symbolic Pauli/H/adjacent inverse, controlled-phase and wrapper lemmas under stated hypotheses; not whole runtime passes',
        old_resource_guarantee='MISSING_EVIDENCE; existing empirical costs preserved, no new certified status',
        original_program_target_unchanged=True,new_compiled_series_authorized=False,new_compile_calls=0,
        K_result_effect_on_L='Continue independent local gate-to-rounded-matrix implementation; exclude compiler error from L claim',
        minimal_alternative='First recover explicitly identified historical program/pass artifacts or prove restricted source rules. Any future new-program series needs separate user decision, fixed workload and capture; it does not recover historical identity.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit();args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'K_report.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    with (args.output/'K_pass_map.csv').open('x',newline='') as f:
        columns=['transformation','application_condition','mathematical_identity','finite_precision_obligation','controlled_phase_obligation','T_proved_scope','unproved','resolution_class','actual_historical_invocation','status']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(result['rows'])
    print(json.dumps(dict(status='K_STATIC_AUDIT_COMPLETE_MISSING_EVIDENCE_RETAINED',transformations=len(result['rows']),compile_calls=0)))


if __name__=='__main__':main()
