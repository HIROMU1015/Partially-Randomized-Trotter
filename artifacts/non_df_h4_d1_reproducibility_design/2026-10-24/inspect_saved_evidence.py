"""Read-only saved evidence inspection. No H4 generation or numerical analysis.

QPY deserialization is permitted; lazy definitions, control construction,
Operators and compilers are forbidden. Only saved parameter tokens are read.
Run with the original runtime and -B. Writes only this new task directory.
"""
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE='a5f77bbb99e23f15a2a465e0ab485777e7a6f40d'
SCIENCE='48d4d27a05a529e05b11599f5f3c671302826617'
ORIGINAL='0c329b30a81accb35c212be8defa66fee0dd0f12'
OLD=ROOT/'artifacts/non_df_h4_d1_execution/2026-10-24'

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data): (HERE/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def require(value,reason):
    if not value: raise ValueError(reason)
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)

def snapshot():
    seal=read(OLD/'seal.json')
    paths=set(read(ROOT/seal['preservation_map'])) | set(seal['code_files']) | set(seal['input_files'])
    # Fixed D1 subtree only; no traversal of other worktrees, /tmp or agent history.
    paths.update(git('ls-tree','-r','--name-only',BASE,'--',str(OLD.relative_to(ROOT))).decode().splitlines())
    return {p:sha(ROOT/p) for p in sorted(paths)}

def check_provenance():
    seal=read(OLD/'seal.json');checks={}
    for kind in ('code_files','input_files'):
        require(all(sha(ROOT/p)==h for p,h in seal[kind].items()),kind)
        checks[kind]=len(seal[kind])
    original_map=read(ROOT/seal['preservation_map'])
    require(all(sha(ROOT/p)==h for p,h in original_map.items()),'OLD_PRESERVATION')
    checks['old_preservation_files']=len(original_map)
    require(all(hashlib.sha256(git('show',SCIENCE+':'+p)).hexdigest()==h
                for p,h in seal['code_files'].items()),'SCIENCE_COMMIT_BYTES')
    original_modules=['pauli_signal_adapter.py','rte.py','rte_event_audit.py',
                      'h4_audited_evaluation.py','h4_pilot_science.py',
                      'fixed_signal_evaluation.py','rte_compiled_cost.py']
    for p in original_modules:
        path='src/trotterlib/'+p
        require((ROOT/path).read_bytes()==git('show',ORIGINAL+':'+path),'ORIGINAL_SOURCE:'+p)
    runtime=read(OLD/'runtime.json')
    require(sha(OLD/'runtime.json')==seal['runtime_manifest_sha256'],'RUNTIME_MANIFEST')
    require(all(sha(p)==h for p,h in runtime['files'].items()),'RUNTIME_BYTES')
    checks['runtime_files']=len(runtime['files']);checks['original_modules']=original_modules
    checks['runtime_version']=runtime['version']
    return checks

def bounded_recovery(selected):
    paths=set()
    for row in selected:
        folder=(ROOT/row['result_file']).parent
        paths.update(p for p in folder.iterdir() if p.is_file())
        paths.update(p for p in (ROOT/row['events_file']).parent.iterdir() if p.is_file())
    oldroot=ROOT/'artifacts/non_df_h4_pilot/2026-10-23/run01'
    paths.update(p for p in oldroot.iterdir() if p.is_file())
    textfiles={};candidates=[];name_hits=[]
    import re
    for p in sorted(paths):
        rel=str(p.relative_to(ROOT))
        if p.suffix.lower() in ('.qpy','.qasm','.pickle') or 'payload' in p.name.lower():
            candidates.append(rel)
        if p.suffix.lower() in ('.json','.txt','.log','.md') or p.name in ('stdout','stderr'):
            text=p.read_text(errors='replace');textfiles[rel]=sha(p)
            names=sorted(set(re.findall(r'circuit-\d+(?:-\d+)?',text)))
            if names:name_hits.append(dict(path=rel,names=names))
    return dict(scope='Original run01 direct files, four case direct files and selected index0 trajectory direct files only',
                examined={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},
                text_files_read=textfiles,payload_or_QPY_candidates=candidates,gate_name_hits=name_hits,
                recovery_status='NO_OLD_QPY_OR_RECURSIVE_PREIMAGE_IN_EXAMINED_SET',
                no_broad_search=True,no_name_bruteforce=True)

def source_locations():
    names={
        'h4_pilot_science.py':['event_gates','evaluate_h4_case'],
        'h4_audited_evaluation.py':['evaluate_audited_signal'],
        'rte_event_audit.py':['trajectory','observe','verify_join'],
        'pauli_signal_adapter.py':['append_pauli_evolution','append_pauli_event','__post_init__'],
        'fixed_signal_evaluation.py':['hadamard_wrapper','default_compiler'],
        'rte_compiled_cost.py':['_operation_payload','_circuit_payload'],
        'h4_legacy_export.py':['build_export'],
        'h4_d1_adapter.py':['event_from_record','grouped_events','build','save_input'],
    }
    result=[]
    for filename,functions in names.items():
        p=ROOT/'src/trotterlib'/filename;tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in functions:
                result.append(dict(path=str(p.relative_to(ROOT)),function=node.name,
                                   line=node.lineno,end_line=node.end_lineno,file_sha256=sha(p)))
    runtime=Path(sys.prefix)/'lib/python3.11/site-packages/qiskit'
    runtime_paths=['circuit/quantumcircuit.py','converters/circuit_to_gate.py',
                   'circuit/gate.py','circuit/add_control.py','circuit/library/pauli_evolution.py',
                   'qpy/binary_io/circuits.py','transpiler/passes/synthesis/high_level_synthesis.py',
                   'transpiler/passes/basis/basis_translator.py']
    return dict(project=result,runtime={str(runtime/p):sha(runtime/p) for p in runtime_paths})

def primitive_token(item,circuit):
    """Serialize existing high-level tokens. Never ask for .definition/.to_matrix."""
    from qiskit.circuit.library import PauliEvolutionGate
    op=item.operation
    base=dict(name=op.name,qubits=[circuit.find_bit(q).index for q in item.qubits],
              clbits=[circuit.find_bit(q).index for q in item.clbits])
    if isinstance(op,PauliEvolutionGate):
        base.update(operator=[dict(label=label,real=value.real,imag=value.imag)
                              for label,value in op.operator.to_list()],
                    time=op.params[0],synthesis=type(op.synthesis).__name__,settings=op.synthesis.settings)
    else: base['params']=list(op.params)
    return base

def pauli_recipe(model,method,join):
    """Expected instruction tokens from saved SOURCE operands, no H4 arithmetic.

    Fixed literal half-step token .05 comes from C export's `.1/2` expression;
    no new coefficient/angle multiplication, exp, phase sum or matrix is used.
    """
    case=model['cases'][method];expected=[];phase_ledger=[]
    def rotation(label,coefficient,time):
        expected.append(dict(name='PauliEvolution',qubits=list(range(8)),clbits=[],
            operator=[dict(label=label,real=coefficient,imag=0.)],time=time,
            synthesis='LieTrotter',settings=dict(reps=1,insert_barriers=False,cx_structure='chain',wrap=False)))
    for step in range(8):
        for b in case['blocks']:rotation(b['label'],b['coefficient'],.05)
        if method=='PR':
            for event in [e for e in join['events'] if e['outer_step_index']==step]:
                for app in event['application_sequence']:
                    label=case['components'][int(app['component_id'])]['label']
                    if app['role']=='product':
                        for q,axis in enumerate(reversed(label)):
                            if axis!='I':expected.append(dict(name=axis.lower(),qubits=[q],clbits=[],params=[]))
                        if app['coefficient_sign']<0:
                            phase_ledger.append(dict(step=step,owner='event',update='global_phase += math.pi',
                                                     event_position=event['event_position'],application=app['application_index']))
                    else:rotation(label,app['coefficient_sign'],event['rotation_angle'])
                phase_ledger.append(dict(step=step,owner='event',update='global_phase += float(np.angle(event.phase))',
                                         event_position=event['event_position'],saved_phase=event['phase']))
        for b in reversed(case['blocks']):rotation(b['label'],b['coefficient'],.05)
        phase_ledger.append(dict(step=step,owner='retained',update='global_phase -= constant * step_time',
                                 constant=model['hamiltonian']['constant'],step_time=model['step_time']))
    return expected,phase_ledger

def inspect_qpy(selected):
    import qiskit
    from qiskit import qpy,QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.circuit.library import PauliEvolutionGate
    import qiskit.quantum_info as qi
    from d1_private import h4_d1_adapter as A
    result={}
    def forbidden(*args,**kwargs):raise AssertionError('H4_NUMERIC_BUILD_CONTROL_SYNTHESIS_COMPILE_FORBIDDEN')
    with ExitStack() as stack:
        for obj,name in [(PauliEvolutionGate,'_define'),(QuantumCircuit,'to_gate'),
                         (QuantumCircuit,'control'),(Gate,'control'),(qiskit,'transpile'),
                         (qi,'Operator'),(A,'build'),(A,'compile_saved')]:
            stack.enter_context(patch.object(obj,name,side_effect=forbidden))
        for row in selected:
            case=row['case'];folder=OLD/'run01/variants'/(case+'_full')
            with (folder/'input.qpy').open('rb') as f:circuits=qpy.load(f)
            require(len(circuits)==1,'ONE_QPY');full=circuits[0]
            require(full.num_qubits==9 and full.num_clbits==1,'WRAPPER_DIMENSIONS')
            require(dict(full.count_ops())==read(folder/'input.json')['pretranspile_histogram'],'SAVED_HISTOGRAM')
            first=full.data[0];last=full.data[-1];penultimate=full.data[-2]
            require(first.operation.name=='h' and [full.find_bit(q).index for q in first.qubits]==[8],'WRAPPER_H')
            require(penultimate.operation.name=='h' and [full.find_bit(q).index for q in penultimate.qubits]==[8],'WRAPPER_H2')
            require(last.operation.name=='measure' and [full.find_bit(q).index for q in last.qubits]==[8]
                    and [full.find_bit(c).index for c in last.clbits]==[0],'WRAPPER_MEASUREMENT')
            r=dict(input_qpy_sha256=sha(folder/'input.qpy'),input_instruction_count=len(full.data),
                   wrapper_tokens='H(8), saved_body, H(8), measure(8->0)',
                   full_global_phase=full.global_phase,read_only_deserialization=True,
                   no_lazy_definition_requested=True)
            if case.startswith('pauli_jw'):
                method=case.rsplit('_',1)[1]
                with (folder/'body_before_control.qpy').open('rb') as f:body=qpy.load(f)[0]
                model=read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json')
                join=read(ROOT/row['price_join'])
                expected,ledger=pauli_recipe(model,method,join)
                actual=[primitive_token(i,body) for i in body.data]
                require(actual==expected,'SOURCE_RECIPE_TO_SAVED_BODY:'+case)
                require(len(full.data)==4,'GENERIC_WRAPPER_SHAPE')
                controlled=full.data[1].operation
                require(controlled.num_ctrl_qubits==1 and controlled.ctrl_state==1,'CONTROL_STATE')
                require([full.find_bit(q).index for q in full.data[1].qubits]==[8,*range(8)],'CONTROL_WIRES')
                require(controlled.base_gate._definition is not None and controlled._definition is not None,'SERIALIZED_GENERIC_DEFINITION')
                base=controlled.base_gate._definition
                require([primitive_token(i,base) for i in base.data]==actual,'BASE_BODY_TOKENS')
                naming=read(folder/'input.json')['sidecar']['naming']
                require(body.global_phase==base.global_phase==naming['global_phase'],'SAVED_BODY_PHASE')
                require(controlled.base_gate.name==body.name==naming['body_name'],'BODY_NAME_TO_GATE')
                require(controlled.name=='c'+body.name,'CONTROLLED_GATE_NAME')
                # Read already stored control definition only. Do not regenerate it.
                definition=controlled._definition
                r.update(body_qpy_sha256=sha(folder/'body_before_control.qpy'),
                         source_recipe_instruction_token_match=True,body_instruction_count=len(actual),
                         current_base_body_token_match=True,current_body_phase_match=True,
                         body_global_phase=body.global_phase,body_name=body.name,
                         controlled_name=controlled.name,control_state=controlled.ctrl_state,
                         control_definition_instruction_count=len(definition.data),
                         lazy_PauliEvolution_definitions_after_QPY=sum(
                             1 for i in body.data if isinstance(i.operation,PauliEvolutionGate) and i.operation._definition is None),
                         ordered_phase_SOURCE_ledger=ledger,
                         phase_ledger_status='SOURCE_EXPRESSION_CORRESPONDENCE_NOT_NEW_NUMERICAL_PHASE_SUM',
                         historical_recursive_definition_comparison='UNAVAILABLE_NO_OLD_PAYLOAD')
            else:
                r['semantic_basis']='Inherited exact recursive fingerprint match; no DF numeric/basis regeneration'
            result[case]=r
    return result

def main():
    require(git('rev-parse','HEAD').decode().strip()==BASE,'RUN_ON_UNCOMMITTED_DESIGN_AT_FIXED_BASE_ONLY')
    before=snapshot();save('preservation_before.json',before)
    checks=check_provenance()
    package=types.ModuleType('d1_private');package.__path__=[str(ROOT/'src/trotterlib')]
    sys.modules['d1_private']=package
    from d1_private.rte_event_audit import verify_join
    from d1_private import h4_d1_reproducibility as criteria
    selected=read(ROOT/'artifacts/non_df_h4_d1_preparation/2026-10-24/preparation.json')['selected']
    join_checks={}
    for s in selected:
        j=read(ROOT/s['price_join']);e=read(ROOT/s['events_file'])
        require(verify_join(j),'ORIGINAL_JOIN')
        require(e['events']==j['events'] and e['event_record_hash']==j['event_record_hash'],'EVENT_FILE_JOIN')
        representation,method=s['case'].rsplit('_',1)
        cmodel=read(ROOT/('artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/'+representation+'_model.json'))
        require(cmodel['cases'][method]['metadata']==read(ROOT/s['result_file'])['model'],'C_DESCRIPTOR_TO_ORIGINAL_MODEL')
        require(j['compiler']==read(ROOT/'artifacts/non_df_h4_d1_preparation/2026-10-24/contract_DRAFT.json')['fixed']['compiler'],'COMPILER_PROFILE')
        join_checks[s['case']]=dict(event_count=len(j['events']),trajectory_record_hash=j['trajectory_record_hash'],
                                   event_record_hash=j['event_record_hash'],saved_compiler=j['compiler'])
    model=read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json')
    original=read(ROOT/'artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/pauli.json')
    require(model['hamiltonian']['terms']==original['terms'] and model['hamiltonian']['constant']==original['constant'],'ORIGINAL_HAMILTONIAN_TOKENS')
    for method in ('PF','PR'):
        c=model['cases'][method]
        require(c['metadata']==c['expected_metadata'],'C_EXPORT_ORIGINAL_METADATA')
        terms=original['terms'] if method=='PF' else original['terms'][:27]
        require([[b['label'],b['coefficient']] for b in c['blocks']]==terms,'RETAINED_ORDER_COEFFICIENTS')
    for (label,coefficient),c in zip(original['terms'][27:],model['cases']['PR']['components'],strict=True):
        require(label==c['label'] and coefficient.hex().lstrip('-')==c['coefficient_abs'].hex()
                and c['sign']==(-1 if coefficient.hex().startswith('-') else 1),'TAIL_SIGN_COEFFICIENT_TOKENS')
    recovery=bounded_recovery(selected);save('bounded_recovery.json',recovery)
    save('source_locations.json',source_locations())
    structures=inspect_qpy(selected);save('saved_QPY_structure.json',structures)
    correspondence=read(HERE/'correspondence.json')
    source_links={link['id']:link['assessment']=='SUPPORTED_SOURCE_RULE'
                  for link in correspondence['links']}
    require(set(source_links)==set(criteria.REQUIRED_LINKS),'TWELVE_DOCUMENTED_LINKS')
    stage=read(OLD/'run01/stage_A.json')['full_inputs']
    rows={}
    for s in selected:
        old=stage[s['case']+'_full']
        rows[s['case']]=dict(old_fingerprint=old['expected_fingerprint'],D1_fingerprint=old['input_fingerprint'],
            historical_fingerprint_matches=old['input_fingerprint_matches'],saved_cosine_RZ=s['saved_cosine_RZ'],
            new_RZ=None,price_reproduction='NOT_EXECUTED',
            # The twelve links are documented individually in correspondence.json.
            source_evidence_level='SOURCE_RULES_PLUS_SAVED_TOKENS; PROPOSED_NOT_ADOPTED',
            assessment=criteria.assess(historical_match=old['input_fingerprint_matches'],
                                       source_links=source_links))
    after=snapshot();require(before==after,'PRESERVATION_AFTER')
    save('inspection.json',dict(status='STATIC_SOURCE_AND_SAVED_QPY_INSPECTION_COMPLETE',base_commit=BASE,
         D1_SOURCE=SCIENCE,original_SOURCE=ORIGINAL,provenance=checks,joins=join_checks,cases=rows,
         preserved_files=len(before),preservation=True,
         Pauli_PR_root_cause='UNKNOWN; AUTO_NAMES_AND_LAZY_DEFINITION_HISTORY_ARE_SUPPORTED_MECHANISMS_NOT_EXCLUSIVE_PROOF',
         historical_program_identity='UNRESOLVED',criterion_adopted=False,
         new_real_H4_build=0,H4_unitary_or_state=0,cost_compile=0,sampling=0,N_G=0,new_science_grants=0,
         mandatory_STOP=True))
    print(json.dumps(dict(status='PASS',preserved_files=len(before),cases=4,real_H4_build=0,cost_compile=0)))

if __name__=='__main__':main()
