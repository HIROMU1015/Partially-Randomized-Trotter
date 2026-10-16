"""Artificial Phase B authorization, exact threshold and fail-fast tests."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import track_a_v1_runner as r
import track_a_v1_compile as adapter
import track_a_v1_costs as costs

c=r.contract


class PhaseBTests(unittest.TestCase):
    def test_exact_boundary_and_both_sides(self):
        self.assertLess(c.axis_relative_error('101999/1000',100),Fraction(1,50))
        for prediction in ('102','98','102001/1000','97999/1000'):
            with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):c.axis_relative_error(prediction,100)

    def test_invalid_measurements(self):
        for n in (0,-1,True,1.0):
            with self.assertRaises(ValueError):c.axis_relative_error(1,n)

    def grant(self):
        cpus=list(os.sched_getaffinity(0))[:2];plan=c.phase_b_plan()
        config=dict(phase_b=True,plan_sha256=c.digest(plan),frozen_model_sha256='model')
        grant=dict(schema='track_a_v1_phase_b_grant_v1',approved_by_user=True,runtime_authorization=True,
            one_shot=True,retry=False,resume=False,scope='compile_validation',SOURCE='source',
            config_sha256='config',plan_sha256=c.digest(plan),call_caps=plan['call_caps'],
            resources=plan['resources'],allowed_cpus=cpus,driver_cpu=cpus[0],monitor_cpu=cpus[1],
            frozen_model_sha256='model')
        fresh=dict(resources_idle_verified=True,profile_verified=True,protected_jobs_unchanged=True,
            source='source',allowed_cpus=cpus,config_sha256='config',external_RSS_monitor_ready=True,
            monotonic_seconds=time.monotonic())
        return config,grant,fresh

    def test_phase_b_sealed_grant_pass(self):
        config,grant,fresh=self.grant()
        self.assertTrue(r.verify_grant(config,grant,config_sha256='config',source='source',fresh=fresh))

    def test_no_signal_or_extra_compile_in_phase_b(self):
        caps=c.phase_b_plan()['call_caps']
        self.assertEqual(caps['native_compile_max'],4)
        self.assertEqual({k:v for k,v in caps.items() if v},dict(snapshot_load=1,frozen_preparation_decode=1,
            native_evolution_max=2,measured_wrappers_max=4,control_probes_max=8,native_compile_max=4))

    def test_grant_mutations_fail(self):
        for key,value in [('retry',True),('scope','signal'),('SOURCE','other'),('frozen_model_sha256','other'),
                          ('approved_by_user',False),('call_caps',{'native_compile_max':5})]:
            config,grant,fresh=self.grant();grant[key]=value
            with self.assertRaises(ValueError):r.verify_grant(config,grant,config_sha256='config',source='source',fresh=fresh)

    def test_stale_gate_and_duplicate_cpu_fail(self):
        config,grant,fresh=self.grant();fresh['monotonic_seconds']-=31
        with self.assertRaises(ValueError):r.verify_grant(config,grant,config_sha256='config',source='source',fresh=fresh)
        config,grant,fresh=self.grant();grant['monitor_cpu']=grant['driver_cpu']
        with self.assertRaises(ValueError):r.verify_grant(config,grant,config_sha256='config',source='source',fresh=fresh)

    def test_validation_uses_exact_prediction(self):
        axes=c.validation_axes();rows=[dict(cell_id=a['cell']['id'],axis=a['axis'],status='MEASURED_COMPLETE',
            model_sha256='fixed',identity_verified=True,predicted=100,exact_prediction='102',measured=100) for a in axes]
        with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):
            c.validate_predictions(rows,expected_axes=axes,frozen_model_sha256='fixed')

    def future_inputs(self):
        axes=c.select_validation_axes({'B0_p11_q1_4th'})
        rows=[dict(cell_id=a['cell']['id'],axis=a['axis'],status='MEASURED_COMPLETE',
                   model_sha256='fixed',identity_verified=True,predicted=100,measured=100) for a in axes]
        frozen=dict(validation_axes=axes,fitted_after_validation=False,models=[
            dict(stratum=dict(method='B0',axis=axis,formula='2nd'),coefficients=['7','3','19']) for axis in c.AXES])
        signals=[dict(cell=axes[0]['cell'],precision_rows={'.05':{'1.0':dict(cost_acquisition_eligible=True)}})]
        return frozen,rows,signals

    def test_future_model_costs_require_complete_validation(self):
        frozen,rows,signals=self.future_inputs()
        with self.assertRaises(ValueError):costs.validated_additional_cost_rows(frozen,rows[:1],signals,
            contract=c,frozen_model_sha256='fixed')
        frozen['fitted_after_validation']=True
        with self.assertRaisesRegex(ValueError,'REFITTED'):costs.validated_additional_cost_rows(frozen,rows,signals,
            contract=c,frozen_model_sha256='fixed')

    def test_future_estimates_keep_wrapper_once_and_label(self):
        frozen,rows,signals=self.future_inputs()
        out=costs.validated_additional_cost_rows(frozen,rows,signals,contract=c,frozen_model_sha256='fixed')
        self.assertEqual([x['RZ'] for x in out],['643','643'])
        self.assertTrue(all(not x['measured'] and not x['population_error_bound'] for x in out))

    def test_fail_fast_adapter_does_not_compile_second_axis(self):
        # Deliberately disagreeing exact/float fields prove the rational field
        # controls STOP; the second axis must never reserve or compile.
        axes=c.select_validation_axes({'B0_p11_q1_4th'})
        transfer=dict(status='AUDITED_CONDITIONAL_TRANSFER_REQUIRES_COMPILE_VALIDATION',current_runtime_closure_digest=c.digest({}))
        frozen=dict(validation_axes=axes,phase_and_profile_transfer_verified=True,
            transfer_receipts=[dict(receipt_sha256=c.digest(transfer))],predictions=[
                dict(cell_id=a['cell']['id'],axis=a['axis'],predicted=100,exact_prediction='102') for a in axes])
        config=dict(frozen_model_digest=c.digest(frozen),runtime_file_hashes={},
            input_mapping={'h6_input_snapshot.npz':'unused'},archive=dict(path='unused',sha256='fixed',binding={}),
            binding=dict(hamiltonian='H',compiler='compiler'),compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
            frozen_model_sha256='fixed')
        record=dict(cell=axes[0]['cell'],signals={'corrected':dict(real=1,imag=0)},
            precision_rows={'.05':{'1.0':dict(cost_acquisition_eligible=True)}})
        inputs={'frozen_model.json':frozen,'transfer_receipt.json':transfer,
            'V1_signal_complete.json':dict(status='COMPLETE_MANDATORY_STOP',signals=[record]),
            'V1_signal_terminal.json':dict(status='COMPLETE_MANDATORY_STOP',scope='signal',exit_code=0),
            'snapshot_receipt.json':{},'df_receipt.json':{}}
        calls=[];written=[]
        permit=NS(require=lambda scope:None,json_input=lambda name:inputs[name],take=lambda name:calls.append(name))
        writer=NS(write=lambda name,value:written.append((name,value)))
        class Vector(list):
            def __sub__(self,other):return Vector(a-b for a,b in zip(self,other))
            def __getitem__(self,key):
                result=super().__getitem__(key)
                return Vector(result) if isinstance(key,slice) else result
        class Circuit:
            data=[]
            def remove_final_measurements(self,inplace):return self
        np=NS(concatenate=lambda v:Vector(x for seq in v for x in seq),zeros_like=lambda v:Vector(0 for x in v),
              vdot=lambda a,b:1,linalg=NS(norm=lambda a:0))
        circuit=Circuit();compiles=[]
        def compile_one(*args,**kwargs):
            compiles.append(1)
            return NS(compiler_settings_hash='compiler',actual_circuit_fingerprint='fp',transpiled_circuit=circuit,rz_count=100,cx_count=5)
        m=NS(np=np,load_h6_snapshot=lambda *a:(NS(n_qubits=12,lambdas=[0]*19),NS(dimension=400),None,None,None,None),
            checked_basis_bridge=lambda *a:(None,Vector([1,0])),CompilerSettings=lambda *a:None,
            simulate_statevector=lambda circuit,state:state,hadamard_expectation=lambda *a:1,
            finite_scale_guard=lambda **kw:None,canonical_qiskit_circuit_fingerprint=lambda *a:'fp',
            transpile_and_measure_cost=compile_one)
        whole=NS(constant_coefficient=1,deterministic_blocks=[0]*20,hamiltonian_hash='H',deterministic_fragment_indices=tuple(range(19)))
        native=NS(build_deterministic_native=lambda *a,**k:NS(circuit=circuit),build_native_hadamard_wrapper=lambda *a,**k:circuit)
        codec=NS(read_preparations=lambda *a:({'faithful:p19':whole},{'binding':{}}))
        modules=dict(h6_matched_port_v1=m,ax2b_native_df_v5=native,h6_roundoff_preparation_v1=codec,qiskit=NS(__version__='1.3.0'))
        with patch.object(adapter.importlib,'import_module',side_effect=lambda name:modules[name.split('.')[-1]]):
            with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):adapter.evaluate(config,permit,c,writer)
        self.assertEqual(len(compiles),1)
        self.assertEqual(calls.count('native_compile_max'),1)
        self.assertEqual(calls.count('measured_wrappers_max'),1)
        self.assertFalse(any(name=='validation_complete.json' for name,_ in written))
        measured=written[-1][1];self.assertEqual(measured['exact_relative_error'],'1/50')


if __name__=='__main__':unittest.main()
