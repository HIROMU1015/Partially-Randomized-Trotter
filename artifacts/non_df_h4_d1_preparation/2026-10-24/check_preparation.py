"""D1 preparation only: metadata joins, disabled contract and artificial checks.
No Qiskit, scientific imports, circuit generation, RNG or execution dispatcher.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '6cc5bded8e0ea2a4f0f1ad7e990f909eb1878d13'
OLD = '0c329b30a81accb35c212be8defa66fee0dd0f12'
PILOT = 'artifacts/non_df_h4_pilot/2026-10-23/run01'
C = 'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01'
TIDS = {
    'pauli_jw_PF': 'd761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d',
    'pauli_jw_PR': '93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97',
    'df_native_PF': '71567586d92a908244f28a7ed1339c7d93907f4d7e7f4912bf492421dea1b83f',
    'df_native_PR': 'ee4b33ef3f4ee363ae4a3d4c1d8b468dbbc0fe20a269361b22d68f1799e274cf',
}

def load(path):
    return json.loads((ROOT/path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extract(path, names, env):
    tree = ast.parse((ROOT/path).read_text())
    body = [ast.ImportFrom(module='__future__',
                           names=[ast.alias(name='annotations')], level=0)]
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    body.extend(selected)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),path,'exec'),env)
    return env

def variants():
    return ([dict(id=case+'_full', case=case, part='full', phase='all_original',
                  axis='cosine', historical_baseline=True) for case in TIDS]
        + [dict(id=case+'_'+part,case=case,part=part,
                phase='deterministic_scalar_and_block' if part=='retained' else 'all_RTE',
                axis='cosine',historical_baseline=False)
           for case in ('pauli_jw_PR','df_native_PR') for part in ('retained','event')]
        + [dict(id='empty',case=None,part='empty',phase='none',
                axis='cosine',historical_baseline=False)])

def require_baseline(saved, rebuilt):
    # Exact matching is a gate, never a preimage/name search or RZ-only fallback.
    keys = ('historical_SOURCE','reference_runtime_manifest_sha256','compiler','trajectory_id',
            'event_record_hash','trajectory_record_hash','input_fingerprint')
    for key in keys:
        if saved.get(key) is None or rebuilt.get(key) != saved[key]:
            raise ValueError('STOP_BASELINE_'+key)
    if rebuilt.get('compiled_RZ') != saved.get('compiled_RZ'):
        raise ValueError('STOP_BASELINE_RZ')
    if rebuilt.get('mapping_proved') is not True:
        raise ValueError('STOP_DESCRIPTOR_MAPPING')
    return 'SAVED_INPUT_AND_PRICE_REPRODUCED_NOT_OLD_PROGRAM_IDENTITY'

def interaction_counts(full, retained, event, empty):
    # Called in this preparation ONLY with invented integers.
    for value in (full,retained,event,empty):
        if type(value) is not int or value < 0:
            raise ValueError('NONNEGATIVE_INTEGER_COUNT_REQUIRED')
    return dict(retained_increment=retained-empty, event_increment=event-empty,
                isolated_prediction=retained+event-empty,
                interaction=full-retained-event+empty)

def phase_partition(atoms, part):
    # Symbolic ledger: preserves ordered opaque phase atoms, never adds H4 angles.
    if part not in ('full','retained','event','empty'):
        raise ValueError('PART')
    owners = {'constant':'retained','extracted_identity':'retained',
              'block_primitive':'retained','order':'event',
              'negative_product':'event','identity_rotation':'event'}
    if any(kind not in owners for kind,token in atoms):
        raise ValueError('UNKNOWN_PHASE')
    return [(kind,token) for kind,token in atoms
            if part=='full' or owners[kind]==part]

def static_audit():
    verify_env=extract('src/trotterlib/isolated_qiskit_execution.py',['digest'],
                       dict(hashlib=hashlib,json=json))
    verify_env=extract('src/trotterlib/rte_event_audit.py',['verify_join'],verify_env)
    previous='artifacts/non_df_h4_cost_reversal_structure/2026-10-24/verification.json'
    proof=load(previous)
    preserved_path=proof['reused_preservation_map']
    preserved=load(preserved_path)
    for path,expected in preserved.items():
        assert sha(ROOT/path)==expected,path
    # Include the preceding structural investigation and fixed original source.
    # Retain the existing 611-file map by reference, without copying it again.
    fixed_inputs={}
    for row in proof['sources']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
        fixed_inputs[row['path']]=row['sha256']
    more_sources=['df_rte_circuit.py','df_rte_tail.py','df_partial_s2_cost.py',
                  'df_partial_s2_repeated_cost.py','h4_legacy_export.py']
    for name in more_sources:
        path='src/trotterlib/'+name
        ref='e80d7510880f8db993b777c627bb8e95b05a070d' if name=='h4_legacy_export.py' else OLD
        assert (ROOT/path).read_bytes()==subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
        fixed_inputs[path]=sha(ROOT/path)
    receipt=load(str(HERE.relative_to(ROOT)/'github_verification.json'))
    for row in receipt['files']:
        b=(ROOT/row['path']).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob_sha']
        assert b==subprocess.check_output(['git','show',row['ref']+':'+row['path']],cwd=ROOT)
    models={rep:load(C+'/'+rep+'_model.json') for rep in ('pauli_jw','df_native')}
    pauli_path=PILOT+'/input_run/input_generation/data/pauli.json'
    original_pauli=load(pauli_path)
    assert models['pauli_jw']['hamiltonian']['constant']==original_pauli['constant']
    assert models['pauli_jw']['hamiltonian']['terms']==original_pauli['terms']
    fixed_inputs[pauli_path]=sha(ROOT/pauli_path)
    for case in models['pauli_jw']['cases'].values():
        blocks=[[b['label'],b['coefficient']] for b in case['blocks']]
        assert blocks==original_pauli['terms'][:case['metadata']['L_D']]
    assert models['df_native']['cases']['PR']['blocks']==models['df_native']['cases']['PF']['blocks'][:2]
    for case in models['df_native']['cases'].values():
        for block in case['blocks']:
            assert set(block['primitives'])=={'rz','rzz','global_phase'}
            for op in block['basis']['operations']:
                assert op['description']['name'] in ('rz','xx_plus_yy')
                assert op['qubits']==op['description']['qubits']
                assert 'matrix' in op and 'inverse_matrix' in op
    selected=[]
    for case,tid in TIDS.items():
        result_path=PILOT+'/run/'+case+'/result.json'
        price_path=PILOT+'/run/'+case+'/trajectories/'+tid+'/price_join.json'
        event_path=PILOT+'/run/'+case+'/trajectories/'+tid+'/events.json'
        d=load(price_path); result=load(result_path)
        assert verify_env['verify_join'](d,expected_case=case)
        assert d['trajectory_index']==0 and d['trajectory_id']==tid
        assert d['compiler']==result['compiler']
        row=result['circuit_cost']['samples'][0]
        assert row['trajectory_id']==tid and row['RZ']==d['compiled_RZ']
        assert row['event_record_hash']==d['event_record_hash']
        base={k:v for k,v in d.items() if k not in
              {'compiled_RZ','compiler','wrapper_fingerprints','event_costs_additive','trajectory_record_hash'}}
        assert base==load(event_path)
        assert d['information_missing']==[]
        rep=d['representation'];method=d['method']
        model=models[rep]['cases'][method]
        assert model['metadata']==result['model']==model['expected_metadata']
        for path in (result_path,price_path,event_path):
            fixed_inputs[path]=sha(ROOT/path)
        for e in d['events']:
            # Only token/identifier/descriptor joins; no price or probability arithmetic.
            if rep=='pauli_jw':
                for app in e['application_sequence']:
                    comp=model['components'][int(app['component_id'])]
                    assert comp['sign']==app['coefficient_sign']
                    assert comp['coefficient_abs']==app['coefficient_abs']
                    assert comp['label']==result['model']['tail_labels'][int(app['component_id'])]
            else:
                for app in e['application_sequence']:
                    descriptions=app['basis_change_operations']
                    candidates=[comp for comp in model['components']
                        if comp['basis']['basis_hash']==app['basis_hash']
                        and comp['support']==app['diagonal_pauli_support']
                        and comp['sign']==app['coefficient_sign']
                        and comp['coefficient_abs']==app['coefficient_abs']]
                    assert candidates,'MISSING_SAVED_DF_COMPONENT'
                    for comp in candidates:
                        assert [o['description'] for o in comp['basis']['operations']]==descriptions
                    assert app['basis_id']=='df-basis-'+app['basis_hash']
        selected.append(dict(case=case,trajectory_index=0,trajectory_id=tid,
            price_join=price_path,events_file=event_path,result_file=result_path,
            event_record_hash=d['event_record_hash'],trajectory_record_hash=d['trajectory_record_hash'],
            input_fingerprint=d['wrapper_fingerprints']['cosine'],
            input_fingerprint_policy='qiskit_recursive_numeric_circuit_v2',
            saved_cosine_RZ=d['compiled_RZ']['cosine'],cost_is='saved singleton/index0, not PR256 mean',
            saved_compiler=d['compiler'],scalar_summary=d['scalar_phase'],
            component_descriptor_join='PASS_TOKEN_EQUALITY_NO_BUILD'))
    for rep in models:
        path=C+'/'+rep+'_model.json';fixed_inputs[path]=sha(ROOT/path)
    for path in [C+'/seal.json',PILOT+'/SOURCE.json',PILOT+'/contract.json',
                 PILOT+'/input_run/input_generation/data/receipt.json',
                 'docs/non_df_h4_cost_reversal_structure_20261024.md',
                 'artifacts/non_df_h4_cost_reversal_structure/2026-10-24/next_contracts_DRAFT.json',
                 previous]:
        fixed_inputs[path]=sha(ROOT/path)
    runtime_seal=load(C+'/seal.json')
    runtime_rows=list(proof['runtime_files'])
    runtime_root='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/lib/python3.11/site-packages/qiskit/'
    for suffix in ('converters/circuit_to_gate.py','circuit/quantumcircuit.py','compiler/transpiler.py'):
        p=runtime_root+suffix
        assert p in runtime_seal['legacy_runtime_files']
        assert sha(p)==runtime_seal['legacy_runtime_files'][p]
        runtime_rows.append(dict(path=p,sha256=sha(p)))
    for row in runtime_rows:
        assert sha(row['path'])==row['sha256']
    return dict(status='D1_CONTRACT_DESIGN_COMPLETE_NO_SCIENTIFIC_EXECUTION',
        base_commit=BASE,original_science_SOURCE=OLD,selected=selected,
        previous_preservation_map=preserved_path,previous_preservation_map_sha256=sha(ROOT/preserved_path),
        preserved_previous_files=len(preserved),
        input_source_sha256=fixed_inputs,
        reference_runtime_seal=C+'/seal.json',reference_runtime_seal_sha256=sha(ROOT/C/'seal.json'),
        checked_runtime_sources=runtime_rows,GitHub_full_files_checked=len(receipt['files']),
        descriptor_information_missing=[],production_adapter_status='NOT_IMPLEMENTED_OR_SEALED',
        literal_gate_roundtrip_on_H4='NOT_EXECUTED',
        exact_full_input_reproduction='NOT_EXECUTED',
        historical_compiled_program_identity='UNPROVABLE_FROM_SAVED_RZ_AND_INPUT_HASH_ALONE',
        auto_gate_name_risk='Pauli to_gate takes circuit.name; recursive hash includes operation.name; old name/preimage not separately recorded in examined receipts',
        real_H4_build=0,compile=0,sampling=0,new_scientific_aggregation=0,N_G=0,new_grants=0)

class D1Tests(unittest.TestCase):
    def test_nine_fixed_variants(self):
        v=variants();self.assertEqual(len(v),9)
        self.assertEqual(len({x['id'] for x in v}),9)
        self.assertEqual(sum(x['historical_baseline'] for x in v),4)
        self.assertTrue(all(x['axis']=='cosine' for x in v))
        self.assertFalse(any(x['case'] and x['case'].endswith('_PF') and x['part']!='full' for x in v))

    def test_ablation_phase_partition(self):
        a=[('constant','c'),('block_primitive','p1'),('order','o'),
           ('negative_product','n'),('identity_rotation','i'),('extracted_identity','x')]
        self.assertEqual(phase_partition(a,'full'),a)
        self.assertEqual(phase_partition(a,'retained'),[a[0],a[1],a[5]])
        self.assertEqual(phase_partition(a,'event'),a[2:5])
        self.assertEqual(phase_partition(a,'empty'),[])
        with self.assertRaises(ValueError):phase_partition([('unowned','x')],'full')

    def test_signed_interactions_are_not_shares(self):
        self.assertEqual(interaction_counts(15,10,8,2)['interaction'],-1)
        self.assertEqual(interaction_counts(25,10,8,2)['interaction'],9)
        self.assertEqual(interaction_counts(1,0,4,2)['retained_increment'],-2)
        with self.assertRaises(ValueError):interaction_counts(1,-1,1,0)

    def baseline(self):
        return dict(historical_SOURCE='toy',reference_runtime_manifest_sha256='runtime',compiler={'seed':17},
            trajectory_id='id',event_record_hash='events',trajectory_record_hash='join',
            input_fingerprint='input',compiled_RZ=123,mapping_proved=True)

    def test_reproduction_is_strict_not_same_RZ(self):
        a=self.baseline();b=copy.deepcopy(a)
        self.assertEqual(require_baseline(a,b),'SAVED_INPUT_AND_PRICE_REPRODUCED_NOT_OLD_PROGRAM_IDENTITY')
        for k in ('historical_SOURCE','reference_runtime_manifest_sha256','compiler','trajectory_id','event_record_hash',
                  'trajectory_record_hash','input_fingerprint','compiled_RZ','mapping_proved'):
            b=copy.deepcopy(a);b[k]=None
            with self.subTest(key=k),self.assertRaises(ValueError):require_baseline(a,b)

    def test_missing_old_input_hash_never_passes(self):
        a=self.baseline();a['input_fingerprint']=None
        with self.assertRaises(ValueError):require_baseline(a,copy.deepcopy(a))

    def test_old_recursive_fingerprint_includes_custom_gate_name(self):
        env=extract('src/trotterlib/rte_compiled_cost.py',
            ['_numeric_parameter_payload','_condition_payload','_operation_payload',
             '_circuit_payload','canonical_qiskit_circuit_fingerprint'],
            dict(math=math,json=json,hashlib=hashlib,
                 np=NS(generic=type('UnusedScalar',(),{}),ndarray=type('UnusedArray',(),{}))))
        class FakeCircuit:
            parameters=();calibrations={};num_qubits=1;num_clbits=0;global_phase=0.0
            def __init__(self,name):
                op=NS(name=name,num_qubits=1,num_clbits=0,params=[.3],definition=None)
                self.data=[NS(operation=op,qubits=[0],clbits=[])]
            def find_bit(self,bit):return NS(index=bit)
        fp=env['canonical_qiskit_circuit_fingerprint']
        self.assertEqual(fp(FakeCircuit('circuit-a')),fp(FakeCircuit('circuit-a')))
        self.assertNotEqual(fp(FakeCircuit('circuit-a')),fp(FakeCircuit('circuit-b')))

    def test_nonexecution_contract(self):
        d=json.loads((HERE/'contract_DRAFT.json').read_text())
        self.assertEqual(d['variants'],variants())
        for k in ('execution_authorized','dispatch_enabled','grant_valid'):self.assertIs(d[k],False)
        for k in ('grant','science_SOURCE','production_adapter','launch_command'):self.assertIsNone(d[k])
        self.assertEqual(d['maximum_cost_transpile_calls'],9)
        self.assertEqual(d['compiler_trace']['policy'],'PRE_POST_ONLY_NO_COMPILER_WRAPPER_CHANGE')
        self.assertEqual(d['reproduction_gate']['fingerprint_mismatch'],'STOP_NO_FALLBACK')

def tests():
    spec=importlib.util.spec_from_file_location('d1_reused_artificial',ROOT/'tests/test_native_df_gate_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(D1Tests)
    for cls,name in [
        (module.MatrixChecks,'test_gaussian_forward_inverse_exact_relative_phase'),
        (module.MatrixChecks,'test_basis_cancellation_preserves_noncommuting_order'),
        (module.MatrixChecks,'test_identity_negative_product_phase_is_not_discarded'),
        (module.MatrixChecks,'test_artificial_df_squared_occupation_eigenvalues'),
        (module.CountChecks,'test_wrapper_axes'),
        (module.CountChecks,'test_identity_barrier_flushes_even_with_zero_phase'),
        (module.CountChecks,'test_existing_event_adapter_and_order_rejection')]:
        suite.addTest(cls(name))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    if not r.wasSuccessful():raise SystemExit(1)
    return dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),
        reused='tests/test_native_df_gate_model.py: 7 tests; invented local gates only',
        oracle='Existing declared-lowering algebra model, not Qiskit optimization or real H4 compilation')

if __name__=='__main__':
    result=static_audit()
    result['artificial_tests']=tests()
    result['checker_sha256']=sha(Path(__file__))
    result['contract_sha256']=sha(HERE/'contract_DRAFT.json')
    (HERE/'preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','preserved_previous_files',
                     'GitHub_full_files_checked','artificial_tests')},ensure_ascii=False))
