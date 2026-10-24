"""Approval, call accounting and failure preservation without science imports."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
from types import SimpleNamespace
from decimal import Decimal, localcontext
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import rustiq_execution as execution
from trottertracks.pr_dictionary_validation import reference


class ExecutionManagementTests(unittest.TestCase):
    def tearDown(self):
        for name in ('numpy', 'qiskit', 'mpmath', 'trotterlib'):
            self.assertNotIn(name, sys.modules)

    def test_foreign_approval_is_rejected_before_source_or_science(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            execution.write_json(root/'approval.json', dict(stage='F1', approved=True))
            with patch.object(execution.subprocess, 'check_output', side_effect=AssertionError('No git')), \
                    self.assertRaises(PermissionError):
                execution.verify_packet(root, root, root, root/'output')

    def test_bound_source_and_input_and_fresh_output_are_required(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'journals').mkdir()
            (root/'input').write_bytes(b'artificial input')
            contract = dict(cases=[dict(id=x) for x in execution.CASE_IDS],
                source_files={}, input_files={'input': execution.file_sha(root/'input')})
            (root/execution.CONTRACT).parent.mkdir(parents=True)
            execution.write_json(root/execution.CONTRACT, contract)
            sha = execution.file_sha(root/execution.CONTRACT)
            approval = dict(stage='C4_Q13_LIMITED_V1', authority='USER_EXPLICIT_C4_EXECUTION',
                approved=True, contract_sha256=sha, source_commit='synthetic-commit',
                output=str(root/'output'), execution_files={})
            execution.write_json(root/'approval.json', approval)
            def git(cmd, **kwargs):
                return 'synthetic-commit\n' if cmd[1] == 'rev-parse' else b''
            with patch.object(execution, 'CONTRACT_SHA', sha), \
                    patch.object(execution.subprocess, 'check_output', side_effect=git):
                verified = execution.verify_packet(root, root, root, root/'output')
                self.assertEqual(verified['inputs']['input'], b'artificial input')
                (root/'input').write_bytes(b'drift')
                with self.assertRaises(ValueError):
                    execution.verify_packet(root, root, root, root/'output')
                (root/'output').mkdir()
                with self.assertRaises(PermissionError):
                    execution.verify_packet(root, root, root, root/'output')

    def test_journal_counts_completed_requests_and_preserves_failed_attempt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'journals').mkdir()
            journal = execution.Journal(root, 'artificial')
            self.assertEqual(journal.call('mock', 'one', lambda: 'token'), 'token')
            self.assertEqual(execution.journal_counts(root), ({'mock': 1}, 2))
            def fail():
                raise TypeError('artificial technical interruption')
            with self.assertRaises(TypeError):
                journal.call('mock', 'two', fail)
            self.assertTrue((root/'STOP.json').exists())
            self.assertIn('artificial technical interruption', journal.path.read_text())
            with self.assertRaises(RuntimeError):
                journal.call('mock', 'no-new-dependency', lambda: None)
            self.assertEqual(len(journal.path.read_text().splitlines()), 4)

    def test_B2_rejects_old_grant_before_preflight_or_science(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            execution.write_json(root/'approval.json',dict(stage='C4_Q13_LIMITED_V1',approved=True))
            with patch.object(execution,'verify_b2_preflight',side_effect=AssertionError('No preparation')), \
                    self.assertRaises(PermissionError):
                execution.verify_packet(root,root,root,root/'out',scope=execution.B2_SCOPE)

    def test_B2_fresh_bound_grant_and_protected_output(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/execution.B2_CONTRACT).parent.mkdir(parents=True)
            execution.write_json(root/execution.B2_CONTRACT,dict(artificial=True))
            approval=dict(stage=execution.B2_SCOPE,authority='USER_EXPLICIT_C4_B2_EXECUTION',approved=True,
                contract_sha256=execution.file_sha(root/execution.B2_CONTRACT),source_commit='artificial',
                output=str(root/'out'))
            execution.write_json(root/'approval.json',approval)
            def git(cmd,**kwargs):
                return 'artificial\n' if cmd[1]=='rev-parse' else b''
            verified=dict(old_packet=root/'old',source_identity='hash')
            with patch.object(execution.subprocess,'check_output',side_effect=git), \
                    patch.object(execution,'verify_b2_preflight',return_value=verified):
                result=execution.verify_packet(root,root,root,root/'out',scope=execution.B2_SCOPE)
                self.assertEqual(result['source_identity'],'artificial:hash')
                (root/'out').mkdir()
                with self.assertRaises(PermissionError):
                    execution.verify_packet(root,root,root,root/'out',scope=execution.B2_SCOPE)

    def test_saved_reader_decimal_and_complex_grammar_is_strict(self):
        for text, parts in [('0.0',('0.0','0')),('-0.0',('-0.0','0')),
                            ('1.234e-128',('1.234e-128','0')),
                            ('(-1.25e+3 - 0.0004j)',('-1.25e+3','-0.0004')),
                            ('(0.0 + 1.0j)',('0.0','1.0'))]:
            self.assertEqual(reference.saved_scalar_parts(text),parts)
        for text in ('nan','+inf','(1.0 + -2.0j)','1+2j','__import__("os")',1.0,' 1.0','(1.0 + 2.0j)garbage'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                reference.saved_scalar_parts(text)

    def test_saved_reader_passes_original_decimal_text_to_context_without_float(self):
        text='0.123456789012345678901234567890123456789012345678901234567890'
        payload=dict(digits=128,values=dict(controlled=[[text]*32 for _ in range(32)]))
        ctx=SimpleNamespace(mpf=Decimal,mpc=lambda real,imag:(real,imag),
            isfinite=lambda value:all(x.is_finite() for x in value),matrix=lambda rows:rows)
        with localcontext() as decimal_context,patch.object(reference,'precision_context',return_value=ctx):
            decimal_context.prec=200
            restored=reference.restore_saved_reference(payload,full=False)
            self.assertEqual(str(restored['controlled'][0][0][0]),text)
        payload['digits']=90
        with patch.object(reference,'precision_context',side_effect=AssertionError('No context')), \
                self.assertRaises(ValueError):
            reference.restore_saved_reference(payload,full=False)

    def test_saved_reader_schema_and_shapes_fail_closed(self):
        payload=dict(digits=128,values=dict(controlled=[['0.0']*32 for _ in range(32)]))
        self.assertIs(reference.validate_saved_reference(payload,full=False),payload)
        for bad in (dict(digits=128,values=dict(controlled=[['0.0']])),
                    dict(digits=128,values=dict(controlled=payload['values']['controlled'],signal='0.0')),
                    dict(digits=True,values=payload['values'])):
            with self.assertRaises(ValueError):
                reference.validate_saved_reference(bad,full=False)

    def test_before_after_gate_materializes_Operator_once_with_mock_only(self):
        from trottertracks.pr_dictionary_validation import gates
        matrix=object(); operator=Mock(return_value=SimpleNamespace(data=matrix))
        numpy=SimpleNamespace(save=Mock())
        with TemporaryDirectory() as tmp, \
                patch.dict(sys.modules,{'numpy':numpy,'qiskit.quantum_info':SimpleNamespace(Operator=operator)}), \
                patch.object(gates,'matrix_gate',side_effect=[dict(label='reference'),dict(label='literal')]) as gate:
            report=execution._matrix(tmp,'artificial','circuit-token','independent-target',literal='saved-literal')
        operator.assert_called_once_with('circuit-token')
        self.assertEqual(gate.call_count,2)
        self.assertIs(gate.call_args_list[0].args[0],matrix)
        self.assertIs(gate.call_args_list[1].args[0],matrix)
        self.assertEqual(report['before_fusion_literal']['label'],'literal')

    def test_B2_local_dispatch_calls_existing_emitter_without_Rustiq_or_real_QC(self):
        from trottertracks.pr_dictionary_validation import rustiq_adapter as adapter
        from trottertracks.pr_dictionary_validation import circuits
        group=adapter.FusionGroup((0,),0.0,dict(policy='artificial'))
        record=dict(name='artificial',region='region-token',plan='plan-token',case_id='case',macro=0,
                    expected='independent-token',literal='literal-token')
        qc=object()
        with TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'journals').mkdir();(out/'reports').mkdir()
            with patch.dict(sys.modules,{'qiskit':SimpleNamespace(QuantumCircuit=Mock(return_value=qc))}), \
                    patch.object(adapter,'fusion_groups',return_value=(group,)), \
                    patch.object(adapter,'_q13',side_effect=AssertionError('No Rustiq')), \
                    patch.object(circuits,'append_central') as central, \
                    patch.object(execution,'_save_circuit'), \
                    patch.object(execution,'_matrix',return_value={}) as gate:
                result=execution._local((record,'B2',str(out)))
            central.assert_called_once_with(qc,(0,),4,0.0)
            self.assertEqual(gate.call_args.kwargs['literal'],'literal-token')
            self.assertEqual(result['fusion_ledger'],[group.ledger])
            self.assertEqual(execution.journal_counts(out)[0],dict(local_build=1,operator=1))

    def test_B2_contract_binds_current_source_and_zero_old_route_execution(self):
        contract=json.loads((ROOT/execution.B2_CONTRACT).read_bytes())
        self.assertFalse(contract['scientific_execution_authorized'])
        self.assertEqual(contract['schema'],execution.B2_SCOPE)
        for name,expected in contract['source_files'].items():
            self.assertEqual(execution.file_sha(ROOT/name),expected,name)
        self.assertEqual(contract['call_counts']['operator_materializations'],26)
        self.assertEqual(contract['call_counts']['Rustiq_synthesis_calls'],0)
        self.assertEqual(contract['call_counts']['new_references'],0)
        self.assertEqual(contract['call_counts']['old_route_execution'],0)

    def test_B2_controller_uses_only_opt_in_jobs_and_stops_at_failed_stage(self):
        """All worker operations are tokens; no circuit/reference numeric work."""
        from contextlib import ExitStack
        from trottertracks.pr_dictionary_validation import gates, f2_execution
        class TokenPool:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def map(self,function,jobs,**kwargs): return map(function,jobs)
        cases=[dict(id=cid,regions=n) for cid,n in zip(execution.CASE_IDS,(2,2,2,4))]
        def prepare(payload):
            case,shared,out=payload
            return dict(case=case,local=['local-token']*case['regions'],signal=0j)
        def full(payload):
            p,route,shared,out=payload
            self.assertEqual(route,'B2')
            return dict(name=p['case']['id']+'_B2',route=route,prepared=p,
                trace=[dict(kind='fusion')]*p['case']['regions'],operator=dict(unitarity={}))
        def wrapper(payload):
            item,axis,out=payload
            return dict(name=item['name']+'_'+axis,parent=item['name'],axis=axis,
                route=item['route'],raw_signal=0,expected_signal=0j,circuit='raw-token')
        def compile_token(payload):
            item,out=payload
            return dict(item,compiled_signal=0)
        for fail_local in (False,True):
            with self.subTest(fail_local=fail_local),TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp);old=root/'old';old.mkdir()
                with (old/'metrics.csv').open('w') as stream:
                    stream.write('case,route,axis,measurement_count\n')
                    for cid in execution.CASE_IDS:
                        for route in ('B0','B1','R1'):
                            for axis in ('X','Y'): stream.write(f'{cid},{route},{axis},1\n')
                contract=dict(environment=dict(python=execution.platform.python_version()),
                    cases=cases,source_files={},design_commit='artificial',changed_source_files=[],
                    journal_counts=dict(signal_pair=8,metrics=8),input_files={},saved_evidence={},
                    compiler={},fusion_policy='artificial',resources=dict(metrics=['measurement_count']))
                verified=dict(scope=execution.B2_SCOPE,contract=contract,root=root,old_packet=old,
                    output=root/'out',inputs={},input_identity='i',source_identity='s',
                    approval=dict(source_commit='artificial'))
                mocks={}
                replacements=dict(_prepare_saved=prepare,_full=full,_adjoint=lambda x:{},
                    _wrapper=wrapper,_compile=compile_token,_raw_diagnostic=lambda x:{})
                for name,function in replacements.items():
                    mocks[name]=stack.enter_context(patch.object(execution,name,side_effect=function))
                def local_token(payload):
                    self.assertEqual(payload[1],'B2')
                    if fail_local: raise ValueError('artificial local failure')
                    return dict(before_fusion_literal={},unitarity={})
                mocks['_local']=stack.enter_context(patch.object(execution,'_local',side_effect=local_token))
                stack.enter_context(patch.object(execution,'_prepare',side_effect=AssertionError('No new references')))
                stack.enter_context(patch.object(execution,'ProcessPoolExecutor',TokenPool))
                stack.enter_context(patch.object(execution,'_worker_init'))
                stack.enter_context(patch.object(execution.subprocess,'check_output',return_value=b'artificial diff'))
                stack.enter_context(patch.object(gates,'signal_gate',return_value=dict(error=0)))
                stack.enter_context(patch.object(f2_execution,'circuit_metrics',return_value=dict(measurement_count=1)))
                if fail_local:
                    with self.assertRaisesRegex(ValueError,'artificial local failure'):
                        execution.execute(verified)
                    mocks['_full'].assert_not_called()
                    self.assertEqual(json.loads((root/'out/STOP.json').read_bytes())['status'],
                                     'FAILED_ATTEMPT_PRESERVED')
                else:
                    result=execution.execute(verified)
                    self.assertEqual(result['Rustiq_calls'],0)
                    for name,expected in dict(_prepare_saved=4,_local=10,_full=4,_adjoint=4,
                                               _wrapper=8,_compile=8,_raw_diagnostic=8).items():
                        self.assertEqual(mocks[name].call_count,expected,name)

    def test_paired_cost_table_uses_only_same_case_axis_integer_vectors_on_toy_rows(self):
        saved=[];new=[]
        for cid in execution.CASE_IDS:
            for axis in ('X','Y'):
                for route in ('B0','B1','R1','B2'):
                    row=dict(case=cid,axis=axis,route=route,measurement_count=1,cx_count=7 if route=='B2' else '9')
                    (new if route=='B2' else saved).append(row)
        rows=execution.paired_metric_rows(saved,new,['cx_count','measurement_count'])
        self.assertEqual(len(rows),8)
        self.assertEqual(rows[0]['metrics']['B2']['cx_count'],7)
        self.assertEqual(rows[0]['metrics']['R1']['cx_count'],9)
        new[0]['measurement_count']=0
        with self.assertRaises(ValueError):
            execution.paired_metric_rows(saved,new,['cx_count','measurement_count'])


if __name__ == '__main__':
    unittest.main()
