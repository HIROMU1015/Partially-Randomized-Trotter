"""F01 artificial failures only: no real G7, F3, sampling or profiling."""
from copy import deepcopy
from contextlib import ExitStack
from hashlib import sha256
from itertools import cycle
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_g6_w2c_b3_driver as T
M, A = T.M, T.A


class Fault(RuntimeError):
    pass


class ToyEndpoint:
    def __init__(self, fail_sample=None, fail_identity=None, fail_angle=False,
                 invalid_result=False, invalid_event=False, bad_evidence=False):
        self.fail_sample, self.fail_identity = fail_sample, fail_identity
        self.fail_angle, self.invalid_result = fail_angle, invalid_result
        self.invalid_event, self.bad_evidence = invalid_event, bad_evidence
        self.samples, self.identities = [], 0

    def shared_identity(self):
        self.identities += 1
        if self.identities == self.fail_identity:raise Fault('identity original')
        return 'synthetic-state'

    def sample(self, bits, trial_id):
        self.samples.append(trial_id)
        if self.fail_sample and self.fail_sample(trial_id):raise Fault('sample original')
        bits.bits(8)
        event = T.event();event['synthetic_trial_id'] = trial_id
        if self.invalid_event:event['phase_i_power'] = 0
        if self.bad_evidence:
            class Bad:
                @property
                def document(self):raise Fault('evidence original')
            evidence = {'bad': Bad()}
        else:evidence = {'synthetic': True}
        return M.Outcome('INVALID' if self.invalid_result else 'EVENT', event, evidence)

    def angle_for(self, result):
        if self.fail_angle:raise Fault('angle original')
        return {'synthetic_angle': True}


class FailingFile:
    """Private test file: one successful row, then partial write or flush failure."""
    def __init__(self, stream, mode):
        self.stream, self.actual, self.mode = stream, stream.file, mode
        self.partial = False
    def write(self, value):
        if self.mode == 'partial' and self.stream.rows == 1:
            if self.partial:raise OSError('partial write original')
            self.partial = True
            return self.actual.write(value[:13])
        return self.actual.write(value)
    def flush(self):
        if self.mode == 'flush' and self.stream.rows == 2:raise OSError('flush original')
        return self.actual.flush()
    def close(self):return self.actual.close()


class F01(unittest.TestCase):
    def trial_id(self, axis='X', trial=0, layer='A', phase='profile'):
        return f'{phase}/baseline/full_return/{layer}/{axis}/{trial}'

    def exercise(self, endpoint, *, n=2, layer='A', phase='profile', bits_factory=None,
                 configure=None, patches=(), memory_read=lambda:{}):
        with tempfile.TemporaryDirectory(prefix='b3-f01-failure-') as d:
            meter = M.Meter(T.FakeClock())
            stream = M.Stream(Path(d)/'mock.jsonl', meter, byte_limit=16000000, flush_rows=1)
            if configure:configure(stream)
            cells = [{'arm': 'full_return', 'layer': layer, 'axis': x, 'N': n} for x in M.AXES]
            with ExitStack() as stack:
                for item in patches:stack.enter_context(item)
                try:
                    result = M.run_pair(endpoint, bits_factory or (lambda axis:T.Bits(cycle((2,)))),
                        cells, stream, meter, phase=phase, pass_name='baseline', correctness_trials=16,
                        memory_read=memory_read)
                except BaseException as exc:
                    failure = getattr(exc, 'b3_failure_provenance', None)
                    self.assertIsNotNone(failure, str(exc))
                    self.assertTrue(failure['mandatory_STOP']);self.assertFalse(failure['is_random_zero'])
                    self.assertEqual((failure['retry'], failure['resume']), (0, False))
                    self.assertEqual(failure['trace_scope'], 'COMPLETED_TRIALS_AFTER_NORMAL_EMIT_RETURN')
                    attempt=failure['current_attempt'];count=sum(failure['completed_prefix'].values())
                    self.assertEqual(failure['last_completed_trial']is None,count==0)
                    self.assertEqual(len(failure['trace_sha256']),64)
                    if count==0:self.assertEqual(failure['trace_sha256'],sha256(b'').hexdigest())
                    if not attempt['result_acquired']:
                        self.assertIsNone(failure['inflight_event']);self.assertFalse(attempt['event_present'])
                    if attempt['trial_id']is None:
                        self.assertFalse(attempt['result_acquired']);self.assertIsNone(attempt['trial_index'])
                    else:self.assertIn('/'+attempt['axis']+'/',attempt['trial_id'])
                    if attempt['trial_completed']:
                        self.assertTrue(attempt['stream_emit_returned'])
                        self.assertEqual(failure['last_completed_trial']['trial_id'],attempt['trial_id'])
                    elif failure['last_completed_trial']is not None and attempt['trial_id']is not None:
                        self.assertNotEqual(failure['last_completed_trial']['trial_id'],attempt['trial_id'])
                    result = exc
                else:failure = None
            try:stream.close()
            except BaseException:pass
            return result, failure, stream.path.read_bytes(), stream

    def check_prefix(self, f, completed, last=None):
        self.assertEqual(sum(f['completed_prefix'].values()), completed)
        self.assertEqual(f['last_completed_trial']['trial_id'] if f['last_completed_trial'] else None, last)
        self.assertEqual(f['trace_sha256'], sha256(b''.join(
            A.canonical(M.exact({'kind':'EVENT','event':self.event_for(t)})).encode()
            for t in completed_ids(completed, last))).hexdigest())

    def event_for(self, trial_id):
        event=T.event();event['synthetic_trial_id']=trial_id
        return event

    def test_next_sample_failure_does_not_reuse_previous_event(self):
        endpoint=ToyEndpoint(fail_sample=lambda t:t==self.trial_id(trial=1))
        exc,f,_,_=self.exercise(endpoint)
        self.assertIsInstance(exc,Fault);self.assertEqual(str(exc),'sample original')
        self.check_prefix(f,1,self.trial_id())
        self.assertEqual(f['current_attempt']['trial_id'],self.trial_id(trial=1))
        self.assertFalse(f['current_attempt']['result_acquired']);self.assertIsNone(f['inflight_event'])
        self.assertFalse(f['current_attempt']['trial_completed']);self.assertEqual(len(endpoint.samples),2)

    def test_first_sample_failure_has_no_result_or_completed_trial(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_sample=lambda t:True))
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'SAMPLING')
        self.assertEqual(f['current_attempt']['axis'],'X');self.assertIsNone(f['inflight_event'])

    def test_first_Y_sample_failure_keeps_X_only_in_completed_prefix(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_sample=lambda t:t==self.trial_id('Y')))
        self.check_prefix(f,2,self.trial_id(trial=1))
        self.assertEqual(f['current_attempt']['axis'],'Y')
        self.assertEqual(f['current_attempt']['trial_id'],self.trial_id('Y'))
        self.assertFalse(f['current_attempt']['event_present']);self.assertIsNone(f['inflight_event'])

    def test_Y_bits_factory_failure_has_axis_but_no_trial_or_result(self):
        def factory(axis):
            if axis=='Y':raise Fault('bits original')
            return T.Bits(cycle((2,)))
        _,f,_,_=self.exercise(ToyEndpoint(),bits_factory=factory)
        self.check_prefix(f,2,self.trial_id(trial=1))
        self.assertEqual(f['current_attempt']['stage'],'BITS_FACTORY')
        self.assertEqual(f['current_attempt']['axis'],'Y');self.assertIsNone(f['current_attempt']['trial_id'])
        self.assertFalse(f['current_attempt']['result_acquired']);self.assertIsNone(f['inflight_event'])

    def test_Y_identity_failure_does_not_inherit_X_context(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_identity=3))
        self.check_prefix(f,2,self.trial_id(trial=1))
        self.assertEqual(f['current_attempt']['stage'],'AXIS_IDENTITY')
        self.assertEqual(f['current_attempt']['axis'],'Y');self.assertIsNone(f['inflight_event'])

    def test_initial_identity_failure_is_preserved_before_any_sample(self):
        endpoint=ToyEndpoint(fail_identity=1)
        _,f,_,_=self.exercise(endpoint)
        self.check_prefix(f,0);self.assertEqual(endpoint.samples,[])
        self.assertEqual(f['current_attempt']['stage'],'PAIR_IDENTITY')
        self.assertIsNone(f['current_attempt']['axis']);self.assertIsNone(f['inflight_event'])

    def test_final_identity_failure_has_completed_trials_no_inflight(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_identity=4))
        self.check_prefix(f,4,self.trial_id('Y',1))
        self.assertEqual(f['current_attempt']['stage'],'FINAL_IDENTITY')
        self.assertIsNone(f['current_attempt']['axis']);self.assertIsNone(f['inflight_event'])

    def test_result_validation_failure_keeps_current_event_uncommitted(self):
        _,f,_,_=self.exercise(ToyEndpoint(invalid_result=True))
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'RESULT_VALIDATION')
        self.assertTrue(f['current_attempt']['result_acquired'])
        self.assertEqual(f['inflight_event']['synthetic_trial_id'],self.trial_id())
        self.assertFalse(f['current_attempt']['trial_completed'])

    def test_event_validation_failure_keeps_current_correctness_id(self):
        _,f,_,_=self.exercise(ToyEndpoint(invalid_event=True),phase='correctness')
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'EVENT_VALIDATION')
        self.assertEqual(f['inflight_event']['synthetic_trial_id'],self.trial_id(phase='correctness'))

    def test_angle_failure_has_acquired_event_but_no_completed_trial(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_angle=True),layer='B')
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'ANGLE_AND_REQUERY')
        self.assertTrue(f['current_attempt']['event_present'])
        self.assertEqual(f['inflight_event']['synthetic_trial_id'],self.trial_id(layer='B'))
        self.assertFalse(f['current_attempt']['stream_emit_started'])

    def test_evidence_failure_does_not_advance_digest_or_counts(self):
        _,f,_,_=self.exercise(ToyEndpoint(bad_evidence=True))
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'EVIDENCE_CONSTRUCTION')
        self.assertFalse(f['current_attempt']['record_constructed'])
        self.assertIsNone(f['candidate_trace_sha256'])

    def test_serialization_failure_keeps_pre_emit_committed_digest(self):
        original=M.serialization.canonical
        def canonical(value):
            if isinstance(value,dict)and'generator_exclusive_ns'in value:raise Fault('serialize original')
            return original(value)
        _,f,_,_=self.exercise(ToyEndpoint(),patches=(patch.object(M.serialization,'canonical',canonical),))
        self.check_prefix(f,0)
        self.assertTrue(f['current_attempt']['record_constructed'])
        self.assertTrue(f['current_attempt']['candidate_ready'])
        self.assertIsNotNone(f['candidate_trace_sha256'])
        self.assertEqual(f['stream_prefix_before_failure_record']['failure']['stage'],'SERIALIZATION')
        self.assertFalse(f['current_attempt']['stream_emit_returned'])

    def test_digest_failure_mutates_only_candidate_copies(self):
        updates=[]
        class Digest:
            def __init__(self,inner):self.inner=inner
            def copy(self):return Digest(self.inner.copy())
            def update(self,value):
                updates.append(value)
                if len(updates)==2:raise Fault('digest original')
                self.inner.update(value)
            def hexdigest(self):return self.inner.hexdigest()
        _,f,_,_=self.exercise(ToyEndpoint(),patches=(patch.object(M,'sha256',lambda *a:Digest(sha256(*a))),))
        self.check_prefix(f,0)
        self.assertEqual(f['current_attempt']['stage'],'CANDIDATE_DIGEST')
        self.assertTrue(f['current_attempt']['record_constructed'])
        self.assertFalse(f['current_attempt']['candidate_ready'])

    def test_emit_failure_keeps_previous_committed_trace_and_current_inflight(self):
        original=M.Stream.emit
        def emit(stream,payload):
            if payload.get('trial_id')==self.trial_id(trial=1):raise Fault('emit original')
            return original(stream,payload)
        _,f,_,_=self.exercise(ToyEndpoint(),patches=(patch.object(M.Stream,'emit',emit),))
        self.check_prefix(f,1,self.trial_id())
        self.assertEqual(f['inflight_event']['synthetic_trial_id'],self.trial_id(trial=1))
        self.assertFalse(f['current_attempt']['stream_emit_returned'])
        self.assertNotEqual(f['trace_sha256'],f['candidate_trace_sha256'])

    def test_partial_write_prefix_is_not_a_completed_trial(self):
        _,f,data,_=self.exercise(ToyEndpoint(),configure=lambda s:setattr(s,'file',FailingFile(s,'partial')))
        self.check_prefix(f,1,self.trial_id())
        p=f['stream_prefix_before_failure_record']
        self.assertEqual(p['complete_rows'],1);self.assertEqual(p['failure']['stage'],'WRITE')
        self.assertEqual(p['written_bytes_including_possible_partial_tail'],len(data))
        self.assertEqual(p['written_bytes_sha256'],sha256(data).hexdigest())
        self.assertFalse(data.endswith(b'\n'));json.loads(data.splitlines()[0])
        with self.assertRaises(json.JSONDecodeError):json.loads(data.splitlines()[1])
        self.assertFalse(p['readback_verified']);self.assertFalse(p['fsync_durability_verified'])

    def test_flush_failure_can_have_complete_written_row_without_trial_commit(self):
        _,f,data,_=self.exercise(ToyEndpoint(),configure=lambda s:setattr(s,'file',FailingFile(s,'flush')))
        self.check_prefix(f,1,self.trial_id())
        p=f['stream_prefix_before_failure_record']
        self.assertEqual(p['complete_rows'],2);self.assertEqual(p['failure']['stage'],'FLUSH')
        self.assertFalse(f['current_attempt']['stream_emit_returned'])
        self.assertEqual(len([json.loads(x)for x in data.splitlines()]),2)
        self.assertFalse(f['current_attempt']['trial_completed'])

    def test_file_hash_failure_marks_written_bytes_hash_unavailable(self):
        def configure(stream):
            class Hash:
                n=0
                def update(self,value):
                    self.n+=1
                    if self.n==2:raise Fault('file hash original')
                    underlying.update(value)
                def hexdigest(self):return underlying.hexdigest()
            underlying=stream.file_hash;stream.file_hash=Hash()
        _,f,data,_=self.exercise(ToyEndpoint(),configure=configure)
        self.check_prefix(f,1,self.trial_id())
        p=f['stream_prefix_before_failure_record']
        self.assertEqual(p['failure']['stage'],'FILE_HASH')
        self.assertEqual(p['written_bytes_including_possible_partial_tail'],len(data))
        self.assertFalse(p['file_hash_covers_written_bytes']);self.assertIsNone(p['written_bytes_sha256'])

    def test_failure_record_write_failure_preserves_original_exception_and_context(self):
        original=M.Stream.emit
        def emit(stream,payload):
            if payload.get('kind')=='TYPED_FAILURE':raise OSError('secondary failure record')
            return original(stream,payload)
        exc,f,_,_=self.exercise(ToyEndpoint(fail_sample=lambda t:True),patches=(patch.object(M.Stream,'emit',emit),))
        self.assertIsInstance(exc,Fault);self.assertEqual(str(exc),'sample original')
        self.assertEqual(f['failure_record_write_error']['reason'],'secondary failure record')
        self.check_prefix(f,0);self.assertIsNone(f['inflight_event'])

    def test_frozen_failure_descriptor_failure_does_not_hide_original(self):
        class Bad(Fault):
            def to_dict(self):raise OSError('secondary descriptor')
        class E(ToyEndpoint):
            def sample(self,*args):raise Bad('primary original')
        exc,f,_,_=self.exercise(E())
        self.assertEqual(str(exc),'primary original')
        self.assertEqual(f['frozen_failure_evidence_error']['reason'],'secondary descriptor')

    def test_many_trials_do_not_leak_X_or_previous_Y_event(self):
        _,f,_,_=self.exercise(ToyEndpoint(fail_sample=lambda t:t==self.trial_id('Y',2)),n=5)
        self.assertEqual(sum(f['completed_prefix'].values()),7)
        self.assertEqual(f['last_completed_trial']['trial_id'],self.trial_id('Y',1))
        self.assertEqual(f['current_attempt']['trial_id'],self.trial_id('Y',2))
        self.assertIsNone(f['inflight_event']);self.assertFalse(f['current_attempt']['result_acquired'])

    def test_zero_outcome_can_complete_without_event(self):
        class E(ToyEndpoint):
            def sample(self,*args):return M.Outcome('ZERO_R')
        result,f,_,_=self.exercise(E())
        self.assertIsNone(f);self.assertEqual(result['counts'],{'ZERO_R':4})

    def test_result_returned_but_event_access_failed_is_explicit(self):
        class E(ToyEndpoint):
            def sample(self,*args):
                class Result:
                    @property
                    def event(self):raise Fault('event access original')
                return Result()
        exc,f,_,_=self.exercise(E())
        self.check_prefix(f,0);self.assertEqual(str(exc),'event access original')
        self.assertTrue(f['current_attempt']['result_acquired'])
        self.assertEqual(f['current_attempt']['stage'],'RESULT_ACCESS')
        self.assertIsNone(f['inflight_event'])

    def test_memory_failure_after_trial_commit_keeps_completed_status(self):
        class E(ToyEndpoint):
            def sample(self,*args):return M.Outcome('ZERO_R')
        def fail_memory():raise Fault('memory observer original')
        _,f,_,_=self.exercise(E(),n=4096,memory_read=fail_memory)
        self.assertEqual(f['completed_prefix'],{'ZERO_R':4096})
        self.assertEqual(f['current_attempt']['stage'],'MEMORY_OBSERVATION')
        self.assertTrue(f['current_attempt']['trial_completed'])
        self.assertTrue(f['current_attempt']['stream_emit_returned'])
        self.assertEqual(f['last_completed_trial']['trial_id'],self.trial_id(trial=4095))

    def test_child_failure_log_retains_trial_context_when_cleanup_also_fails(self):
        class E(ToyEndpoint):
            def sample(self,*args):raise Fault('primary child sample')
        with tempfile.TemporaryDirectory(prefix='b3-f01-mock-child-')as d:
            root=Path(d);(root/'artificial.txt').write_text('fixture only')
            c=T.accepted_candidate();c.update(passes=list(M.PASSES),correctness_trials_per_cell=16)
            c['source_pins']={'artificial.txt':sha256((root/'artificial.txt').read_bytes()).hexdigest()}
            c['schedule_proposal']={'result_bytes':1000000,'log_bytes':1000000,'flush_rows':16}
            c['decisions']['schedule_and_output']=c['schedule_proposal']
            raw=json.dumps({'rows':[{'input':M.INPUT['id'],'arm':a,
                **{k:M.INPUT[k]for k in ('p','x','m')},'budget':{
                'N_per_axis':n,'quantum_call_hard_cap_per_axis':n,
                'alpha_axis':'1/320','alpha_familywise_16_axes':'1/20','uses_reference_B_new_or_m2':False}}
                for a,n in M.SAVED_N.items()]}).encode()
            (root/'saved.json').write_bytes(raw)
            c['saved_evidence']={'path':'saved.json','sha256':sha256(raw).hexdigest()}
            mods=T.modules();mods['g7'].DeterministicBits=lambda seed:T.Bits(cycle((2,)))
            with patch.object(M,'candidate_errors',return_value=()),patch.object(M,'load_modules',return_value=mods),\
                 patch.object(M,'LegacyEndpoint',return_value=E()),patch.object(M,'clock',T.FakeClock()),\
                 patch.object(M.Stream,'close',side_effect=OSError('secondary close')):
                with self.assertRaisesRegex(M.Stop,'primary child sample'):
                    M.execute(c,root=root,output=root/'mock_output',approval_port=T.FakePort())
            child=json.loads((root/'mock_output/correctness_baseline_ordinary_A.failure.json').read_text())
            self.assertEqual(child['reason'],'primary child sample')
            failure=json.loads(child['trial_failure_provenance'])
            self.assertEqual(failure['current_attempt']['axis'],'X')
            self.assertFalse(failure['current_attempt']['result_acquired'])
            self.assertIsNone(failure['inflight_event'])
            self.assertTrue((root/'mock_output/failure_prefix.json').is_file())

    def test_normal_all_four_arms_layers_match_frozen_pre_fix_reference(self):
        p=Path(__file__).parent/'fixtures/g6_w2c_b3_f01_normal_reference.json'
        refs=json.loads(p.read_text())
        self.assertEqual(refs['driver_sha256'],'d7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee')
        self.assertEqual(len(refs['cases']),8)
        for ref in refs['cases']:
            with self.subTest(arm=ref['arm'],layer=ref['layer']),tempfile.TemporaryDirectory()as d:
                meter=M.Meter(T.FakeClock());stream=M.Stream(Path(d)/'mock',meter,byte_limit=1000000,flush_rows=2)
                e=M.LegacyEndpoint(T.modules(),ref['arm'],ref['layer'],
                    T.accepted_candidate()['decisions']['angle']if ref['layer']=='B'else None,lambda:'source')
                cells=[{'arm':ref['arm'],'layer':ref['layer'],'axis':x,'N':4}for x in M.AXES]
                result=M.run_pair(e,lambda axis:T.Bits(cycle((0,1,2))),cells,stream,meter,
                    phase='profile',pass_name='baseline',correctness_trials=2)
                stream.close();data=stream.path.read_bytes()
                self.assertEqual(result,ref['summary']);self.assertEqual(meter.rows,ref['coarse'])
                self.assertEqual(sha256(data).hexdigest(),ref['stream_sha256'])
                self.assertEqual(stream.previous,ref['chain_sha256'])
                self.assertEqual((stream.rows,len(data)),(ref['stream_rows'],ref['stream_bytes']))
                self.assertEqual((e.g.packet_queries,e.g.event_queries),(ref['packet_queries'],ref['event_queries']))

    def test_no_frozen_scientific_modules_imported(self):
        for name in ('g7_generator','g6_f3_generator','g6_f3_precision','return_aggregation','g9_native'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+name,sys.modules)


def completed_ids(count,last):
    """Only the fixed two-trials-per-axis toy cases used by check_prefix."""
    if not count:return []
    prefix=last.rsplit('/',2)[0]
    return [prefix+'/X/'+str(i)for i in range(min(count,2))]+[prefix+'/Y/'+str(i)for i in range(max(0,count-2))]


if __name__=='__main__':unittest.main()
