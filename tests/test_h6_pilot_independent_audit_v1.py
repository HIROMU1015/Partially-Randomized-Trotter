"""Independent tiny-fixture challenge tests; no scientific imports or operations.

Execution is delegated to the parent under its bounded, import-blocked harness.
All files are fabricated below TMPDIR, which must be a directory under home.
"""
from copy import deepcopy
import errno
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_pilot_runner', ROOT/'scripts/h6_b0i_pilot_runner_v1.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
b = r.adapter_module.b
launch_spec=importlib.util.spec_from_file_location('independent_pilot_launcher', ROOT/'scripts/h6_b0i_pilot_launcher_v1.py')
l=importlib.util.module_from_spec(launch_spec);launch_spec.loader.exec_module(l)


class Monitor:
    def __init__(self, grant):
        self.phase = 'PRELAUNCH'; self.first_stop = None; self.history = []
    def tick(self, **kwargs): pass
    def close(self): pass


def prepared(prefix):
    return SimpleNamespace(preparation_hash='2'*64, constant_coefficient=3.,
        extracted_identity_coefficient=0., deterministic_fragment_indices=tuple(range(prefix)),
        randomized_block_indices=(), ld=prefix,
        deterministic_block_order=tuple(['one']+[f'df{i}' for i in range(prefix)]),
        deterministic_blocks=[{'id':i, 'phase':.125} for i in range(prefix+1)], num_system_qubits=2)


def binding():
    value = dict(source='a'*40, hamiltonian='b'*64, df='c'*64, state='d'*64,
        input='e'*64, compiler='f'*64, fragment_order=[str(i) for i in range(19)])
    value['b0_fingerprints'] = {f'{p}/{q}':b.g.d0.digest(b.blueprint(prepared(p), value, prefix=p, q=q))
        for p,q in r.PILOT}
    return value


def authorized(contract):
    return dict(scope=r.SCOPE, approved_by_user=True, runtime_authorization=True,
        contract_sha256=r.g.d0.digest(contract), SOURCE=contract['SOURCE'], automatic_retry=False,
        exclusive_output=contract['exclusive_output'], approval_evidence_sha256='0'*64,
        review_commit='1'*40, synthetic_only=True,
        **{k+'_digest':r.g.d0.digest(contract[k]) for k in
            ('input_receipts','budget','binding','profile','source_hashes','allowed_cpus','carry')})


class FakeNative:
    """Uses the real stdlib bridge prepare, then fabricated native records."""
    events = None
    failure = None
    def __init__(self, grant, verified, mapping, calls, guard):
        self.grant, self.calls = grant, calls
        calls.take('snapshot_load'); self.events.append(('factory',))
    def prepare(self, prefix, q):
        self.events.append(('prepare', prefix))
        record = dict(signals=dict(corrected=dict(real=.7, imag=.2),
            exact_truncated=dict(real=.6, imag=.1)), reference=dict(real=.5, imag=.3),
            allowance=dict(log_B_margin=1e-11),
            error_components=dict(kind='EMPIRICAL', certified=False,
                B0_signal=dict(real=.000001, imag=.000002),
                full_H_reference=dict(real=.000003, imag=.000004), signal_abs_upper=1.,
                evidence=dict(kind='INDEPENDENT_ARTIFICIAL_COMPONENTS')))
        fragments = [dict(id=str(i), G=((1,1j),(-1j,2)),
            **{'lambda':(-1. if i%2 else .5)}) for i in range(19)]
        request = b.prepare(self.grant, prefix=prefix, q=q,
            binding=b.binding_for(self.grant,prefix,q), load_fragments=lambda _:fragments,
            load_prepared=lambda _:prepared(prefix), saved_record=record,
            scalar_input_error_upper=0., rotation_roundoff_upper=1e-12)
        # Deliberately preserve bridge output schema: no fixture-only cell added.
        return dict(request=request, independent=dict(real=.7, imag=.2))
    def build(self, case):
        self.events.append(('build',case['prefix']))
        if self.failure == ('build',case['prefix']): raise RuntimeError('ARTIFICIAL_PROVIDER_FAILURE')
        return dict(wrappers={axis:dict(prefix=case['prefix'],axis=axis) for axis in ('cosine','sine')})
    def validate(self, case, built):
        self.events.append(('validate',case['prefix']))
        self.calls.take('control_probe',4); self.calls.take('probe_wrapper_copy',2)
        if self.failure == ('validate',case['prefix']): raise r.g.Stop('ARTIFICIAL_LAST_CASE_PHASE_FAILURE')
        return dict(status='PASS', fingerprints={a:self.fingerprint(w) for a,w in built['wrappers'].items()})
    def fingerprint(self, wrapper):
        if self.failure == ('mutate',wrapper['prefix']):
            count = sum(e == ('fingerprint',wrapper['prefix'],wrapper['axis']) for e in self.events)
            self.events.append(('fingerprint',wrapper['prefix'],wrapper['axis']))
            if count: return '0'*64
        return r.g.d0.digest(wrapper)
    def compile(self, wrapper, fingerprint):
        self.events.append(('compile',wrapper['prefix'],wrapper['axis']))
        metrics = {m:5 for m in r.adapter_module.METRICS}
        if self.failure == 'invalid_metric': metrics['CX_depth'] = True
        if self.failure == 'missing_metric': metrics.pop('RZ_depth')
        return dict(metrics=metrics, fingerprint='0'*64 if self.failure=='bad_fp' else fingerprint,
            compiler_hash='0'*64 if self.failure=='bad_compiler' else self.grant.contract['binding']['compiler'])
    def legacy_b0(self, prefix, q):
        return dict(status='ARTIFICIAL_LEGACY_METADATA_ONLY',deltas=None)
    def verify_bindings(self):self.events.append(('final_bindings',))
    def close(self): self.events.append(('close',))


class IndependentPilotAudit(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ['TMPDIR']).resolve()
        self.assertTrue(scratch.is_relative_to('/home/AbeHiromu'))
        self.tmp = tempfile.TemporaryDirectory(prefix='h6-independent-pilot-',dir=scratch)
        self.base = Path(self.tmp.name); self.root = self.base/'source'; self.root.mkdir()
        raw = b'INDEPENDENT ARTIFICIAL SOURCE\n'; (self.root/'source.txt').write_bytes(raw)
        self.output = self.base/'output'
        self.contract = dict(scope=r.SCOPE, approved=True, runtime_authorization=True,
            execution_plan_sealed=True, synthetic_only=True, SOURCE='a'*40,
            automatic_retry=False, next_stage_authorized=False, exclusive_output=str(self.output),
            input_receipts=[], binding=binding(), source_hashes={'source.txt':r.g.sha(raw)},
            runtime_root=str(self.root), allowed_cpus=[], carry=dict(actual=0,reset=False,refund=0),
            profile=dict(python=__import__('sys').version, executable=__import__('sys').executable,
                runtime_file_hashes={}), budget=dict(output_bytes=2**20,terminal_reserve_bytes=2**16,
                output_inode_cap=100, terminal_inode_reserve=4),
            candidates=[dict(prefix=p,q=q,formula='2nd',T=.8) for p,q in r.PILOT],
            control='symmetric_directional', operation_limits=deepcopy(r.LIMITS), workers=1,
            matched_B0=dict(automatic_build=False,approved=False),
            precision_domain=dict(epsilon=[.05,.01,.005,.001,.0005],factors=[1.,10.]),
            compiler_options=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17),
            control_tolerance=1e-8,max_instructions=100000)
        self.auth = authorized(self.contract); self.events = []
        self.kind = type('IndependentNative',(FakeNative,),dict(events=self.events,failure=None))
        self.mapping = Mock(return_value={})
        self.env = patch.dict(os.environ,{k:'1' for k in r.g.THREAD_ENV}); self.env.start()
    def tearDown(self): self.env.stop(); self.tmp.cleanup()
    def execute(self):
        return r.run(self.contract,self.auth,contract_sha=r.g.d0.digest(self.contract),
            authorization_sha=r.g.d0.digest(self.auth),output=self.output,root=self.root,
            mapping_loader=self.mapping,synthetic=True,monitor_factory=Monitor,factory=self.kind)
    def terminal(self): return json.loads((self.output/'terminal_v1.json').read_text())
    def test_real_bridge_output_schema_reaches_all_six_mock_compiles(self):
        self.assertEqual(self.execute(),0,self.terminal())
        self.assertEqual(sum(e[0]=='compile' for e in self.events),6)
        self.assertEqual(self.terminal()['calls_reserved_before_operations'],r.LIMITS)
        first_compile = next(i for i,e in enumerate(self.events) if e[0]=='compile')
        self.assertEqual([e[1] for e in self.events[:first_compile] if e[0]=='validate'],[5,10,15])
        results = json.loads((self.output/'pilot_results_v1.json').read_text())
        self.assertTrue(all(c['matched_B0']['deltas'] is None for c in results['cases']))
        self.assertEqual(self.events[-2:],[('final_bindings',),('close',)])
    def test_last_case_failure_prevents_every_compile(self):
        self.kind.failure=('validate',15)
        self.assertEqual(self.execute(),2)
        self.assertIn('ARTIFICIAL_LAST_CASE_PHASE_FAILURE',self.terminal()['first_stop']['reason'])
        self.assertEqual(sum(e[0]=='compile' for e in self.events),0)
        self.assertEqual(self.events[-1],('close',))
    def test_provider_failure_is_not_retried_and_marker_is_consumed(self):
        self.kind.failure=('build',10)
        self.assertEqual(self.execute(),2)
        self.assertEqual(sum(e==('build',10) for e in self.events),1)
        self.assertEqual(sum(e[0]=='compile' for e in self.events),0)
        before=list(self.events)
        with self.assertRaisesRegex(r.g.Stop,'ONE_SHOT_ALREADY_USED'): self.execute()
        self.assertEqual(self.events,before)
    def test_unapproved_rejected_before_mapping_and_factory(self):
        self.contract['approved']=False
        with self.assertRaisesRegex(r.g.Stop,'EXPLICIT_NEW_EXECUTION_AUTHORIZATION_REQUIRED'): self.execute()
        self.mapping.assert_not_called(); self.assertEqual(self.events,[])
    def test_changed_source_rejected_before_mapping_and_factory(self):
        (self.root/'source.txt').write_text('CHANGED')
        self.assertEqual(self.execute(),2)
        self.mapping.assert_not_called(); self.assertEqual(self.events,[])
        self.assertIn('SOURCE_BYTES_SHA',self.terminal()['first_stop']['reason'])
    def test_input_profile_and_compiler_mutation_fail_before_mapping(self):
        for field in ('input_receipts','profile','binding'):
            with self.subTest(field=field):
                old=deepcopy(self.contract[field])
                if field=='input_receipts':self.contract[field]=[dict(logical_name='fabricated')]
                elif field=='profile':self.contract[field]['python']='CHANGED'
                else:self.contract[field]['compiler']='0'*64
                # Let the top-level contract SHA agree while preserving the old
                # field digest, so this specifically challenges AUTH_BINDING.
                self.auth['contract_sha256']=r.g.d0.digest(self.contract)
                with self.assertRaisesRegex(r.g.Stop,'AUTH_BINDING:'+field):self.execute()
                self.contract[field]=old
                self.auth['contract_sha256']=r.g.d0.digest(self.contract)
        self.mapping.assert_not_called(); self.assertEqual(self.events,[])
    def test_wrapper_mutation_before_first_compile_rejected(self):
        self.kind.failure=('mutate',5)
        self.assertEqual(self.execute(),2)
        self.assertIn('WRAPPER_CHANGED_BEFORE_COMPILE',self.terminal()['first_stop']['reason'])
        self.assertEqual(sum(e[0]=='compile' for e in self.events),0)
    def test_missing_metric_is_not_zero(self):
        self.kind.failure='missing_metric'
        self.assertEqual(self.execute(),2)
        self.assertIn('SIX_NONNEGATIVE_INTEGER_METRICS_REQUIRED',self.terminal()['first_stop']['reason'])
        self.assertFalse((self.output/'pilot_results_v1.json').exists())
    def test_boolean_metric_is_invalid(self):
        self.kind.failure='invalid_metric'
        self.assertEqual(self.execute(),2)
        self.assertIn('SIX_NONNEGATIVE_INTEGER_METRICS_REQUIRED',self.terminal()['first_stop']['reason'])
    def test_compile_return_identity_must_match(self):
        self.kind.failure='bad_compiler'
        self.assertEqual(self.execute(),2)
        self.assertIn('COMPILED_IDENTITY_MISMATCH',self.terminal()['first_stop']['reason'])
    def test_saved_independent_reference_cannot_fall_back_to_corrected(self):
        record=dict(signals=dict(corrected=dict(real=.7,imag=.2)))
        with self.assertRaisesRegex(r.g.Stop,'SAVED_INDEPENDENT_REFERENCE_REQUIRED'):
            r.adapter_module.stored_independent(record)
    def test_saved_independent_reference_finite_numeric_schema(self):
        for value in (dict(real=True,imag=0.),dict(real=float('nan'),imag=0.),
                dict(real=.1,imag=.2,unexpected=.3)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    r.adapter_module.stored_independent(dict(independent_forward_signals=dict(corrected=value)))
        result=r.adapter_module.stored_independent(dict(independent_forward_signals=dict(corrected=dict(real=.1,imag=-.2))))
        self.assertEqual(result,complex(.1,-.2))


class Vector(list):
    def __getitem__(self,key):
        result=super().__getitem__(key)
        return Vector(result) if isinstance(key,slice) else result
    def __sub__(self,other): return Vector(a-b for a,b in zip(self,other))


class TinyArrays:
    """A two-amplitude stdlib stand-in; does not emulate a science library import."""
    linalg=SimpleNamespace(norm=lambda v:math.sqrt(sum(abs(z)**2 for z in v)))
    @staticmethod
    def concatenate(parts):return Vector(z for p in parts for z in p)
    @staticmethod
    def zeros_like(v):return Vector(0j for _ in v)
    @staticmethod
    def isfinite(v):return SimpleNamespace(all=lambda:all(math.isfinite(complex(z).real) and math.isfinite(complex(z).imag) for z in v))
    @staticmethod
    def vdot(a,b):return sum(complex(x).conjugate()*y for x,y in zip(a,b))


class Wrapper:
    def __init__(self,axis):self.axis=axis
    def remove_final_measurements(self,inplace=False):
        if inplace is not False:raise AssertionError('original measured wrapper mutated')
        return self.axis


class IndependentNativeValidation(unittest.TestCase):
    def adapter(self, *, wrong_sine=False, wrong_phase=False, low_phase=False):
        api=r.adapter_module.NativeAdapter.__new__(r.adapter_module.NativeAdapter)
        api.np=TinyArrays;api.qstate=Vector([1.+0j]);api.guard=Mock()
        used={}
        def take(name,count=1):used[name]=used.get(name,0)+count
        api.calls=SimpleNamespace(take=take);api.grant=SimpleNamespace(contract=dict(control_tolerance=1e-8))
        def simulate(circuit,state):
            if circuit=='evolution':
                if state[0]:return Vector([1j if low_phase else 1.,0.])
                return Vector([0.,complex(.6,-.8 if wrong_phase else .8)])
            target=.6 if circuit=='cosine' else (-.8 if wrong_sine else .8)
            return Vector([math.sqrt((1+target)/2),math.sqrt((1-target)/2)])
        api.port=SimpleNamespace(simulate_statevector=simulate,
            hadamard_expectation=lambda state,n:abs(state[0])**2-abs(state[1])**2,
            finite_scale_guard=lambda **kwargs:None)
        api.fp_api=SimpleNamespace(canonical_qiskit_circuit_fingerprint=lambda w:r.g.sha(w.axis.encode()))
        case=dict(request=dict(signal=dict(signals=dict(corrected=dict(real=.6,imag=.8)))),
            independent=dict(real=.6,imag=.8))
        built=dict(evolution=SimpleNamespace(circuit='evolution'),
            wrappers={a:Wrapper(a) for a in ('cosine','sine')})
        return api,case,built,used
    def test_production_validation_four_probes_and_positive_sine(self):
        api,case,built,used=self.adapter();result=api.validate(case,built)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(used,dict(control_probe=4,probe_wrapper_copy=2))
        self.assertLess(result['max_error'],1e-12)
    def test_production_validation_rejects_wrong_sine(self):
        api,case,built,_=self.adapter(wrong_sine=True)
        with self.assertRaisesRegex(r.g.Stop,'NATIVE_CONTROL_REFERENCE_MISMATCH'):api.validate(case,built)
    def test_production_validation_rejects_wrong_relative_phase(self):
        api,case,built,_=self.adapter(wrong_phase=True)
        with self.assertRaisesRegex(r.g.Stop,'NATIVE_CONTROL_REFERENCE_MISMATCH'):api.validate(case,built)
    def test_production_validation_rejects_low_branch_phase(self):
        api,case,built,_=self.adapter(low_phase=True)
        with self.assertRaisesRegex(r.g.Stop,'NATIVE_CONTROL_REFERENCE_MISMATCH'):api.validate(case,built)
    def test_final_provider_bytes_are_rechecked_without_scientific_import(self):
        scratch=Path(os.environ['TMPDIR']).resolve()
        self.assertTrue(scratch.is_relative_to('/home/AbeHiromu'))
        with tempfile.TemporaryDirectory(prefix='h6-independent-provider-',dir=scratch) as value:
            checkout=Path(value);path=checkout/'provider.txt';fixed=b'ARTIFICIAL PINNED PROVIDER\n'
            path.write_bytes(fixed)
            api=r.adapter_module.NativeAdapter.__new__(r.adapter_module.NativeAdapter)
            api.guard=Mock();api.grant=SimpleNamespace(contract=dict(native_provider=dict(
                checkout=str(checkout),SOURCE='a'*40,source_hashes={'provider.txt':r.g.sha(fixed)})))
            with patch('subprocess.check_output',return_value=fixed) as git:
                api.verify_bindings()
                self.assertEqual(git.call_count,1)
                path.write_text('ARTIFICIAL MUTATION')
                with self.assertRaisesRegex(r.g.Stop,'FINAL_NATIVE_PROVIDER_BINDING'):api.verify_bindings()


class IndependentLauncherAudit(unittest.TestCase):
    def setUp(self):
        scratch=Path(os.environ['TMPDIR']).resolve()
        self.assertTrue(scratch.is_relative_to('/home/AbeHiromu'))
        self.tmp=tempfile.TemporaryDirectory(prefix='h6-independent-launcher-',dir=scratch)
        self.base=Path(self.tmp.name);self.control=self.base/'control';self.control.mkdir()
        self.contract=dict(synthetic_only=False,runtime_root=str(self.base),
            control_directory=str(self.control),budget=dict(address_space_bytes=32*2**30,wall_seconds=1.,cpu_seconds=14400),
            launcher_budget=dict(address_space_bytes=256*2**20,rss_bytes=64*2**20,cpu_seconds=10),
            log_cap_bytes=1024)
        self.contract_path=self.base/'contract.json';self.contract_path.write_text(json.dumps(self.contract))
        self.auth_path=self.base/'auth.json';self.auth_path.write_text('{}')
        self.mapping_path=self.base/'mapping.json';self.mapping_path.write_text('{}')
        self.output=self.base/'output'
        self.identity=dict(pid=987654321,start_ticks=123,parent_pid=456,uid=os.getuid(),boot_id='ARTIFICIAL')
    def tearDown(self):self.tmp.cleanup()
    def test_esrch_before_and_after_pidfd_open_is_benign(self):
        with patch.object(l.g,'identity',return_value=None),patch.object(l.os,'pidfd_open') as opened:
            l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
            opened.assert_not_called()
        with patch.object(l.g,'identity',return_value=self.identity),patch.object(l.os,'pidfd_open',
                side_effect=OSError(errno.ESRCH,'ARTIFICIAL EXIT RACE')):
            l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
        with patch.object(l.g,'identity',return_value=self.identity),patch.object(l.os,'pidfd_open',return_value=987),\
                patch.object(l.signal,'pidfd_send_signal',side_effect=OSError(errno.ESRCH,'ARTIFICIAL EXIT RACE')),\
                patch.object(l.os,'close') as closed:
            l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGKILL)
            closed.assert_called_once_with(987)
    def test_non_esrch_pidfd_errors_propagate(self):
        with patch.object(l.g,'identity',return_value=self.identity),patch.object(l.os,'pidfd_open',
                side_effect=OSError(errno.EPERM,'ARTIFICIAL PERMISSION FAILURE')):
            with self.assertRaises(OSError):l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
        with patch.object(l.g,'identity',return_value=self.identity),patch.object(l.os,'pidfd_open',return_value=987),\
                patch.object(l.signal,'pidfd_send_signal',side_effect=OSError(errno.EPERM,'ARTIFICIAL PERMISSION FAILURE')),\
                patch.object(l.os,'close') as closed:
            with self.assertRaises(OSError):l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
            closed.assert_called_once_with(987)
    def test_owned_identity_mismatch_before_or_after_open_rejected(self):
        impostor=dict(self.identity,start_ticks=999)
        with patch.object(l.g,'identity',return_value=impostor),patch.object(l.os,'pidfd_open') as opened:
            with self.assertRaisesRegex(l.g.Stop,'OWN_CHILD_IDENTITY_CHANGED'):
                l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
            opened.assert_not_called()
        with patch.object(l.g,'identity',side_effect=[self.identity,impostor]),\
                patch.object(l.os,'pidfd_open',return_value=987),patch.object(l.os,'close') as closed,\
                patch.object(l.signal,'pidfd_send_signal') as sent:
            with self.assertRaisesRegex(l.g.Stop,'OWN_CHILD_IDENTITY_CHANGED'):
                l.terminate_owned(self.identity['pid'],self.identity,l.signal.SIGTERM)
            sent.assert_not_called();closed.assert_called_once_with(987)
    def test_unauthorized_launcher_never_spawns(self):
        with patch.object(l.g,'authorization_gate',side_effect=l.g.Stop('ARTIFICIAL_UNAUTHORIZED')),\
                patch.object(l.signal,'signal',return_value=None),\
                patch.object(l.subprocess,'Popen') as spawned,patch.object(l.subprocess,'run') as recorded:
            with self.assertRaisesRegex(l.g.Stop,'ARTIFICIAL_UNAUTHORIZED'):
                l.launch(self.contract_path,self.auth_path,self.mapping_path,self.output)
            spawned.assert_not_called();recorded.assert_not_called()
    def test_wall_failure_term_kill_wait_then_recorder_without_retry(self):
        events=[];wait_calls=[subprocess.TimeoutExpired('ARTIFICIAL',1),
            subprocess.TimeoutExpired('ARTIFICIAL',5),-9]
        def wait(timeout=None):
            events.append(('wait',timeout));value=wait_calls.pop(0)
            if isinstance(value,Exception):raise value
            return value
        child=SimpleNamespace(pid=self.identity['pid'],wait=wait)
        def terminate(pid,identity,sig):
            self.assertEqual((pid,identity),(child.pid,self.identity));events.append(('signal',sig))
        def record(command,**kwargs):
            events.append(('record',command[-2:]));return SimpleNamespace(returncode=0)
        with patch.object(l.g,'authorization_gate',return_value=SimpleNamespace()),\
                patch.object(l.r,'validate_contract'),patch.object(l.g,'runtime_gate'),\
                patch.object(l.signal,'signal',return_value=None),\
                patch.object(l.resource,'getrlimit',return_value=(l.resource.RLIM_INFINITY,l.resource.RLIM_INFINITY)),\
                patch.object(l.resource,'setrlimit') as limited,\
                patch.object(l.resource,'getrusage',return_value=SimpleNamespace(ru_maxrss=1,ru_utime=0.,ru_stime=0.)),\
                patch.object(l.time,'monotonic',side_effect=[0.,2.,3.]),\
                patch.object(l.subprocess,'Popen',return_value=child) as spawned,\
                patch.object(l.subprocess,'run',side_effect=record) as recorded,\
                patch.object(l.g,'identity',side_effect=[self.identity,None]),\
                patch.object(l,'terminate_owned',side_effect=terminate):
            status=l.launch(self.contract_path,self.auth_path,self.mapping_path,self.output)
        self.assertEqual(status,2)
        self.assertEqual(events,[('wait',1),('signal',l.signal.SIGTERM),('wait',5),
            ('signal',l.signal.SIGKILL),('wait',5),('record',['--record-wait-status','137'])])
        self.assertEqual(spawned.call_count,1);self.assertEqual(recorded.call_count,1)
        limited.assert_called_once_with(l.resource.RLIMIT_AS,(256*2**20,l.resource.RLIM_INFINITY))
        receipt=json.loads((self.control/'launcher_wait_v1.json').read_text())
        self.assertEqual(receipt['worker_wait_status'],137)
        self.assertEqual(receipt['first_launcher_stop'],'OWN_WORKER_WALL_CAP')
        self.assertTrue(receipt['worker_identity_absent'])
        self.assertFalse(receipt['automatic_retry'])
    def test_launcher_term_runs_owned_cleanup_and_restores_handlers(self):
        events=[];previous={l.signal.SIGTERM:object(),l.signal.SIGINT:object()};installed={}
        def signal_handler(sig,handler):
            old=installed.get(sig,previous[sig]);installed[sig]=handler
            events.append(('handler',sig,handler));return old
        waits=[0]
        def wait(timeout=None):
            events.append(('wait',timeout));waits[0]+=1
            if waits[0]==1:installed[l.signal.SIGTERM](l.signal.SIGTERM,None)
            return -15
        child=SimpleNamespace(pid=self.identity['pid'],wait=wait)
        def terminate(pid,identity,sig):
            self.assertEqual((pid,identity),(child.pid,self.identity));events.append(('signal',sig))
        with patch.object(l.g,'authorization_gate',return_value=SimpleNamespace()),\
                patch.object(l.r,'validate_contract'),patch.object(l.g,'runtime_gate'),\
                patch.object(l.signal,'signal',side_effect=signal_handler),\
                patch.object(l.resource,'getrlimit',return_value=(l.resource.RLIM_INFINITY,l.resource.RLIM_INFINITY)),\
                patch.object(l.resource,'setrlimit'),patch.object(l.time,'monotonic',return_value=0.),\
                patch.object(l.subprocess,'Popen',return_value=child) as spawned,\
                patch.object(l.subprocess,'run') as recorded,patch.object(l.g,'identity',return_value=self.identity),\
                patch.object(l,'terminate_owned',side_effect=terminate):
            with self.assertRaisesRegex(l.g.Stop,'LAUNCHER_SIGNAL:'+str(l.signal.SIGTERM)):
                l.launch(self.contract_path,self.auth_path,self.mapping_path,self.output)
        self.assertEqual([e for e in events if e[0] in ('wait','signal')],
            [('wait',1),('signal',l.signal.SIGTERM),('wait',5)])
        self.assertEqual(installed,previous)
        self.assertEqual(spawned.call_count,1);recorded.assert_not_called()


if __name__=='__main__':unittest.main()
