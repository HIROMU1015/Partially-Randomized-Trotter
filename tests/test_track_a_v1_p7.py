"""Artificial-only p7 grant, two-axis fail-fast, phase and binding checks."""
from copy import deepcopy
from fractions import Fraction
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import track_a_v1_p7_runner as runner
import track_a_v1_p7_compile as adapter

c=runner.contract


def frozen():
    return dict(fitted_after_validation=False,validation_axes=runner.base.contract.validation_axes(),models=[
        dict(stratum=dict(method='B0I',axis=axis,formula='2nd',control='symmetric_directional',
            compiler_hash=c.COMPILER,profile_hash='saved-profile',phase_policy='single_scalar_on_ancilla'),
             coefficients=['797','269','9613/5']) for axis in c.AXES])


def saved_fixture():
    f=frozen();signal=dict(cell=c.cell(),signals={'corrected':dict(real=1.,imag=0.)},
        reference=dict(real=.3,imag=.2),scalar_receipt=dict(C=.25,full_Fock_identity=True),
        allowance=dict(kind='EMPIRICAL',certified=False))
    terminal=dict(status='COMPLETE_MANDATORY_STOP',exit_code=0)
    cleanup=dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,all_reaped=True,first_stop=None,cleanup_errors=[])
    validation=dict(rows=[dict(cell_id=x['cell']['id'],axis=x['axis'],predicted=100,measured=100,
        model_sha256=c.MODEL_SHA,status='MEASURED_COMPLETE',identity_verified=True) for x in f['validation_axes']])
    inputs={'frozen_model.json':f,'p7_signal.json':signal,'phase_b_validation.json':validation,
        'phase_a_terminal.json':deepcopy(terminal),'phase_b_terminal.json':deepcopy(terminal),
        'reaccount_terminal.json':deepcopy(terminal),'phase_a_cleanup.json':deepcopy(cleanup),
        'phase_b_cleanup.json':deepcopy(cleanup),'snapshot_receipt.json':{},'df_receipt.json':{}}
    config=dict(frozen_model_digest=c.digest(f),p7_signal_digest=c.digest(signal),runtime_file_hashes={},
        input_mapping={'h6_input_snapshot.npz':'never-read'},archive=dict(path='never-read',sha256='archive',binding={}),
        compiler_options=deepcopy(c.OPTIONS),binding=dict(hamiltonian='H',compiler=c.COMPILER))
    inputs['p7_saved_binding.json']=dict(p7_signal_digest=c.digest(signal),predictions=c.predictions(f),
        runtime_closure_digest=c.digest({}))
    counts={};written={}
    def take(name):
        counts[name]=counts.get(name,0)+1
        if counts[name]>c.plan()['call_caps'].get(name,0):raise ValueError('CALL_CAP')
    permit=NS(require=lambda scope:None,json_input=lambda name:inputs[name],take=take,base_contract=runner.base.contract)
    writer=NS(write=lambda name,value:written.__setitem__(name,value))
    return config,inputs,permit,writer,counts,written


def fake_science(measurements,*,compiler_changed=False,overlap=1.,mutate=False):
    build_args=[];compiles=[]
    class Vector(list):
        def __sub__(self,other):return Vector(a-b for a,b in zip(self,other))
        def __getitem__(self,key):
            result=super().__getitem__(key)
            return Vector(result) if isinstance(key,slice) else result
    class Circuit:
        def __init__(self,axis=None):self.axis=axis;self.data=[];self.changed=False
        def remove_final_measurements(self,inplace):return self
    np=NS(concatenate=lambda v:Vector(x for seq in v for x in seq),zeros_like=lambda v:Vector(0 for x in v),
        vdot=lambda *a:overlap,linalg=NS(norm=lambda x:0.))
    def scale(**kwargs):
        if kwargs['absolute_discrepancy']>1.1e-9:raise ValueError('SCALE_AGREEMENT')
    def compile_one(wrapper,*args,**kwargs):
        value=measurements[len(compiles)];compiles.append(wrapper.axis)
        if mutate:wrapper.changed=True
        return NS(rz_count=value,cx_count=4,compiler_settings_hash='changed' if compiler_changed else c.COMPILER,
            actual_circuit_fingerprint=kwargs['actual_circuit_fingerprint'],transpiled_circuit=Circuit())
    def simulate(circuit,state):
        out=Vector(state);out.axis=circuit.axis;return out
    m=NS(np=np,load_h6_snapshot=lambda *a:(NS(n_qubits=12,lambdas=[0]*19),NS(dimension=400),None,None,None,None),
        checked_basis_bridge=lambda *a:(None,Vector([1,0])),CompilerSettings=lambda *a:None,
        simulate_statevector=simulate,
        hadamard_expectation=lambda probe,n:1. if probe.axis=='cosine' else 0.,finite_scale_guard=scale,
        canonical_qiskit_circuit_fingerprint=lambda circuit:('changed-' if circuit.changed else '')+str(circuit.axis),
        transpile_and_measure_cost=compile_one)
    whole=NS(constant_coefficient=2.,hamiltonian_hash='H',deterministic_fragment_indices=tuple(range(19)),
        deterministic_blocks=list(range(20)))
    def build(blocks,**kwargs):build_args.append(dict(blocks=blocks,**kwargs));return NS(circuit=Circuit())
    native=NS(build_deterministic_native=build,
        build_native_hadamard_wrapper=lambda evolution,axis,**kw:Circuit(axis))
    codec=NS(read_preparations=lambda *a:({'faithful:p19':whole},{'binding':{}}))
    modules=dict(h6_matched_port_v1=m,ax2b_native_df_v5=native,h6_roundoff_preparation_v1=codec,qiskit=NS(__version__='1.3.0'))
    return modules,build_args,compiles


class P7Tests(unittest.TestCase):
    def grant(self):
        p=c.plan();cpus=sorted(os.sched_getaffinity(0))[:2]
        config=dict(plan_sha256=c.digest(p),frozen_model_sha256=c.MODEL_SHA,p7_signal_digest='signal')
        grant=dict(schema='track_a_v1_p7_compile_grant_v1',approved_by_user=True,runtime_authorization=True,
            one_shot=True,retry=False,resume=False,scope=p['scope'],SOURCE='source',config_sha256='config',
            plan_sha256=c.digest(p),call_caps=p['call_caps'],resources=p['resources'],frozen_model_sha256=c.MODEL_SHA,
            p7_signal_digest='signal',next_stage_authorized=False,allowed_cpus=cpus,driver_cpu=cpus[0],monitor_cpu=cpus[1])
        fresh=dict(source='source',config_sha256='config',allowed_cpus=cpus,monotonic_seconds=100.,
            profile_verified=True,resources_idle_verified=True,protected_jobs_unchanged=True,external_RSS_monitor_ready=True)
        return config,grant,fresh

    def axis(self,axis='cosine',measured=14524):
        return dict(cell_id=c.CELL_ID,axis=axis,measured=measured,exact_prediction=c.PREDICTION,
            model_sha256=c.MODEL_SHA,compiler_hash=c.COMPILER,status='MEASURED_COMPLETE',identity_verified=True)

    def test_plan_has_no_authority_and_exact_scope(self):
        p=c.plan();self.assertFalse(p['approved']);self.assertFalse(p['runtime_authorization'])
        self.assertEqual(p['allowed_cpus'],[]);self.assertIsNone(p['absolute_launch_command'])
        self.assertEqual(p['axes'],['cosine','sine']);self.assertEqual(p['cell'],c.cell())
        self.assertEqual(p['call_caps']['native_compile_max'],2)
        self.assertEqual(p['call_caps']['native_evolution_max'],1)
        self.assertTrue(all(p['call_caps'][x]==0 for x in ('new_signal','new_scalar','new_reference','new_fit','new_sampling','cost_reaccounting','V2')))

    def test_prediction_is_frozen_14524_point_2(self):
        p=c.predictions(frozen());self.assertEqual([x['exact_prediction'] for x in p],['72621/5']*2)
        self.assertEqual([x['predicted'] for x in p],[14524.2]*2)

    def test_wrong_model_formula_phase_or_coefficients_stop(self):
        for key,value in [('formula','4th'),('phase_policy','other'),('control','ordinary'),('compiler_hash','other')]:
            f=frozen();f['models'][0]['stratum'][key]=value
            with self.assertRaises(ValueError):c.predictions(f)
        f=frozen();f['models'][0]['coefficients'][0]='798'
        with self.assertRaisesRegex(ValueError,'PREDICTION_CHANGED'):c.predictions(f)

    def test_refit_forbidden(self):
        f=frozen();f['fitted_after_validation']=True
        with self.assertRaisesRegex(ValueError,'REFITTED'):c.predictions(f)

    def test_exact_two_percent_and_neighbors(self):
        with patch.object(c,'PREDICTION','102'):
            row=self.axis(measured=100)
            with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):c.validate_axis(row)
        with patch.object(c,'PREDICTION','101999/1000'):
            row=self.axis(measured=100);self.assertLess(Fraction(c.validate_axis(row)),Fraction(1,50))

    def test_positive_integer_and_identity(self):
        for measured in (0,-1,True,14524.):
            with self.assertRaisesRegex(ValueError,'POSITIVE_INTEGER'):c.validate_axis(self.axis(measured=measured))
        row=self.axis();row['cell_id']='B0I_p8_q1_2nd'
        with self.assertRaises(ValueError):c.validate_axis(row)

    def test_complete_pair_only(self):
        rows=[self.axis(a) for a in c.AXES];c.validate_pair(rows)
        for bad in (rows[:1],list(reversed(rows)),rows+[rows[0]],[rows[0],rows[0]]):
            with self.assertRaises(ValueError):c.validate_pair(bad)

    def test_sealed_grant_and_stale_gate(self):
        config,grant,fresh=self.grant()
        runner.verify_grant(config,grant,fresh,source='source',config_sha='config',now=110.)
        with self.assertRaisesRegex(ValueError,'FRESH_P7_GATE'):
            runner.verify_grant(config,grant,fresh,source='source',config_sha='config',now=131.)

    def test_unapproved_grant_extra_compile_and_retry_denied(self):
        for key,value in [('approved_by_user',False),('runtime_authorization',False),('scope','compile_validation'),
            ('SOURCE',None),('retry',True),('resume',True),('frozen_model_sha256','other'),('p7_signal_digest','other')]:
            config,grant,fresh=self.grant();grant[key]=value
            with self.assertRaises(ValueError):runner.verify_grant(config,grant,fresh,source='source',config_sha='config',now=110.)
        config,grant,fresh=self.grant();grant['call_caps']['native_compile_max']=3
        with self.assertRaises(ValueError):runner.verify_grant(config,grant,fresh,source='source',config_sha='config',now=110.)

    def test_duplicate_cpu_denied(self):
        config,grant,fresh=self.grant();grant['monitor_cpu']=grant['driver_cpu']
        with self.assertRaisesRegex(ValueError,'DISJOINT'):runner.verify_grant(config,grant,fresh,source='source',config_sha='config',now=110.)

    def test_adapter_denied_before_import(self):
        def deny(scope):raise ValueError('DENIED')
        permit=NS(require=deny)
        with patch.object(adapter.importlib,'import_module',side_effect=AssertionError('science import')):
            with self.assertRaisesRegex(ValueError,'DENIED'):adapter.evaluate({},permit,c,None)

    def test_changed_saved_signal_denied_before_import(self):
        config,inputs,permit,writer,counts,written=saved_fixture();inputs['p7_signal.json']['reference']['real']+=1
        with patch.object(adapter.importlib,'import_module',side_effect=AssertionError('science import')):
            with self.assertRaisesRegex(ValueError,'SAVED_P7_SIGNAL_CHANGED'):adapter.evaluate(config,permit,c,writer)
        self.assertEqual(counts,{})

    def test_incomplete_old_phase_denied(self):
        config,inputs,permit,writer,counts,written=saved_fixture();inputs['phase_a_terminal.json']['exit_code']=1
        with patch.object(adapter.importlib,'import_module',side_effect=AssertionError('science import')):
            with self.assertRaisesRegex(ValueError,'COMPLETE_PREVIOUS'):adapter.evaluate(config,permit,c,writer)

    def exercise(self,measurements,**kwargs):
        config,inputs,permit,writer,counts,written=saved_fixture();modules,build_args,compiles=fake_science(measurements,**kwargs)
        with patch.object(adapter.importlib,'import_module',side_effect=lambda name:modules[name.split('.')[-1]]):
            error=None
            try:adapter.evaluate(config,permit,c,writer)
            except ValueError as exc:error=exc
        return error,counts,written,build_args,compiles,inputs

    def test_success_two_axes_literal_scalar_and_eight_blocks(self):
        error,counts,written,builds,compiles,inputs=self.exercise([14524,14525]);self.assertIsNone(error)
        self.assertEqual(compiles,['cosine','sine']);self.assertEqual(counts,{k:v for k,v in c.plan()['call_caps'].items() if v})
        self.assertEqual(builds[0]['blocks'],list(range(8)));self.assertEqual(builds[0]['scalar'],2.25)
        self.assertEqual((builds[0]['T'],builds[0]['q'],builds[0]['formula'],builds[0]['control_policy']),(.8,1,'2nd','symmetric_directional'))
        self.assertFalse(written['validation_complete.json']['cost_reaccounting_performed'])
        self.assertEqual(inputs['p7_signal.json']['reference'],dict(real=.3,imag=.2))

    def test_first_axis_fail_stops_before_second_wrapper(self):
        error,counts,written,builds,compiles,_=self.exercise([15000,14524])
        self.assertIn('GE_2_PERCENT',str(error));self.assertEqual(compiles,['cosine'])
        self.assertEqual(counts['measured_wrappers_max'],1);self.assertEqual(counts['native_compile_max'],1)
        self.assertIn('cosine_measurement.json',written);self.assertNotIn('validation_complete.json',written)

    def test_second_axis_error_cannot_average_with_first(self):
        error,counts,written,builds,compiles,_=self.exercise([14524,15000])
        self.assertIn('GE_2_PERCENT',str(error));self.assertEqual(len(compiles),2)
        self.assertNotIn('validation_complete.json',written)

    def test_compiler_change_and_input_mutation_stop(self):
        for kwargs in [dict(compiler_changed=True),dict(mutate=True)]:
            error,counts,written,builds,compiles,_=self.exercise([14524,14524],**kwargs)
            self.assertIn('IDENTITY',str(error));self.assertEqual(len(compiles),1)
            self.assertEqual(written['cosine_measurement.json']['status'],'STOP_IDENTITY')

    def test_bad_control_probe_stops_before_compile(self):
        error,counts,written,builds,compiles,_=self.exercise([14524,14524],overlap=1.+2e-9)
        self.assertIn('SCALE_AGREEMENT',str(error));self.assertEqual(compiles,[])
        self.assertNotIn('native_compile_max',counts)

    def test_cli_draft_cannot_reach_execute(self):
        config,grant,fresh=self.grant();grant['approved_by_user']=False
        argv=['p7','--execute-approved','--config','unused','--grant','unused','--fresh-gate','unused']
        with patch.object(sys,'argv',argv),patch.object(runner.base,'read_json',side_effect=[config,grant,fresh]),\
                patch.object(runner.base,'sha',return_value='config'),patch.object(runner,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'NEW_USER_APPROVED'):runner.main()
            execute.assert_not_called()


if __name__=='__main__':unittest.main()
