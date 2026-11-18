"""Artificial Q3 artifacts through the ordinary worker adapter and pinned validator."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_synthetic_compact_q3_adapter', ROOT/'scripts/run_rpe_optimizer_q3_job.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
E = A.EVIDENCE


def fixture(case, inputs, source):
    closure = dict(status='VERIFIED_FINITE_BINARY64_SUPPORT', checked_pairs=1)
    coarse = dict(decoder_certificate=closure, first_unsafe=[dict(exact=0)], all_good=dict(exact=1),
                  rounds=[dict(good=dict(exact=1), unsafe=dict(exact=0))])
    dp = dict(retained_MSE_contribution=1, residual_loss_contribution_upper=0, MSE_upper=1,
              residual_probability=0, transition_pairs=1, any_zero=0, transition_allowance=0,
              rounds=[dict(retained_states=1)])
    risk = dict(bounds=dict(total_state_dp=dp, lemma=dict(exact=2**18000), upper_candidates={'dp':1}),
                MSE_lower=1, MSE_upper=1, moments=dict(bias_squared=[0,1],variance=[0,1]))
    record = dict(configuration_id='total', canonical_binding='binding', evaluation_target='synthetic',
                  allocation=case['allocation'], source=source, statistical_contract=inputs['statistical_contract'],
                  deterministic_input=dict(options=inputs['options']),
                  diagnostic=dict(deterministic_risk=risk, partial_decoder_risk=coarse),
                  unicode='人工', signed_zero=-0., padding='abcdefgh' * 16384)
    return dict(status=A.PASS, record=record, scientific_evaluation_calls=1, source=source,
                allocation=case['allocation'], sufficient_N=None, global_RMSE_bound=None,
                G=None, G_qualified=None, winner=None)


class EvidenceTests(unittest.TestCase):
    def exercise(self, output, kind, compact, mode='execute', fault=None, export_directory=None):
        writer = SimpleNamespace(write_new=A.write)
        label = 'fixture-job'
        case = dict(label=label, configuration_id='cfg', signal_packet='packet', allocation=dict(shots=[]))
        inputs = dict(template_source='synthetic', statistical_contract=dict(domain='ARTIFICIAL'),
                      adopted_evaluation=dict(MSE_upper=1), options=dict(maximum_round_outcomes=5,
                      maximum_verification_pairs=5, total_state_refinement=dict(maximum_transition_pairs=5)))

        def supervise(cases, values, source, out):
            self.assertEqual(mode, 'execute', 'collection called the supervisor')
            folder = out/label
            result = fixture(case, inputs, source)
            sidecars = E.source_sidecars(result)
            if fault=='bad_api':
                sidecars['API_RETURN.json'] = [dict(bad=True)]
            elif fault=='extra_sidecar':
                sidecars['PARTIAL_LEMMA.json'] = dict(audit='other worker evidence')
            def child():
                for name,value in sidecars.items():
                    writer.write_new(folder/name,value)
                    if fault=='interrupted':
                        return
                writer.write_new(folder/'RESULT.json',result)
                writer.write_new(folder/'PERSISTED.json',dict(operational_status='PERSISTED'))
                writer.write_new(folder/'RESOURCE_FINAL.json',dict(source=source))
            proc = multiprocessing.get_context('fork').Process(target=child)
            proc.start()
            proc.join(15)
            self.assertEqual(proc.exitcode, 0)
            row = dict(exitcode=0, signal=None)
            if kind == 'pauli_b_common_entry':
                (out/'PROCESS_EXITS_fixture.jsonl').write_text(json.dumps(dict(children=[row]))+'\n')
            return dict(receipts=[{}], errors=[], monitor_errors=[], process_exitcodes=[row])

        validate=A.pinned_validator(ROOT,SimpleNamespace(assess=lambda *_:dict(status=A.PASS)),lambda value:value)
        route=SimpleNamespace(metadata=lambda _:None, runtime=None, installed=lambda _:nullcontext(),
                supervise=supervise, bounds=lambda _:dict(maximum_round_outcomes=1,verification_pairs_upper=1,
                transition_pairs_upper=1), validate=validate,evidence_writer=writer)
        request=dict(request_id=label,configuration_id='cfg',route=dict(kind=kind,view=str(ROOT),view_commit='fixture'),
                     execution_profile=dict(compact_q3_evidence=compact))
        request_path=output.parent/(output.name+'_request.json')
        request_path.write_bytes(E.encode(request))
        argv=['adapter','--request',str(request_path),'--output',str(output),'--mode',mode]
        if export_directory is not None:
            argv+=['--evidence-directory',str(export_directory)]
        with patch.object(A,'route',return_value=route),patch.object(A,'build',return_value=(case,inputs)), \
             patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
            code=A.main()
        path=output/('COLLECTED_OUTCOME.json' if mode=='collect' else 'OUTCOME.json')
        return code,json.loads(path.read_bytes()) if path.is_file() else None

    def test_existing_raw_df_route_can_execute_and_collect_without_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'sqlite3':None}):
            output = Path(tmp)/'raw'
            code, answer = self.exercise(output, 'prdf_d2_split', False)
            self.assertEqual((code, answer['status']), (0, A.PASS))
            code, recovered = self.exercise(output, 'prdf_d2_split', False, mode='collect')
            self.assertEqual((code, recovered['status']), (0, A.PASS))
            self.assertEqual(answer['MSE_upper'], recovered['MSE_upper'])

    def test_pinned_gate_and_complete_collect_and_export_on_both_routes(self):
        for kind in ('pauli_b_common_entry','prdf_d2_common_entry'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                ordinary,compact=Path(tmp)/'ordinary',Path(tmp)/'compact'
                code,old=self.exercise(ordinary,kind,False)
                self.assertEqual(code,0)
                code,new=self.exercise(compact,kind,True)
                self.assertEqual(code,0)
                for key in ('status','MSE_lower','MSE_upper','scientific_evaluation_calls','identity','closure'):
                    self.assertEqual(old[key],new[key])
                leaf=compact/'fixture-job'
                self.assertTrue((leaf/E.WITNESS).is_file())
                self.assertFalse((leaf/E.CHECKPOINTS).exists())
                self.assertFalse(any((leaf/name).exists() for name in E.SIDECARS))
                code,resumed=self.exercise(compact,kind,True,mode='collect')
                self.assertEqual(code,0)
                self.assertEqual(resumed['status'],old['status'])
                self.assertEqual(resumed['original_outcome_sha256'],E.sha((compact/'OUTCOME.json').read_bytes()))
                copies=Path(tmp)/'export'
                self.assertEqual(self.exercise(compact,kind,True,'export-evidence',export_directory=copies)[0],0)
                actual=E.source_sidecars(json.loads((leaf/'RESULT.json').read_bytes()))
                for name in E.SIDECARS:
                    self.assertEqual((copies/name).read_bytes(),E.encode(actual[name]))
                    old_payload=json.loads((ordinary/'fixture-job'/name).read_bytes())
                    if name=='API_RETURN.json':
                        old_payload[0]['source']['execution_profile']['compact_q3_evidence']=True
                    self.assertEqual((copies/name).read_bytes(),E.encode(old_payload))
                self.assertTrue((leaf/'RESULT.json').is_file())

    def test_mismatched_api_and_interruption_keep_worker_bytes(self):
        for fault in ('bad_api','interrupted','extra_sidecar'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'attempt'
                code,result=self.exercise(out,'prdf_d2_common_entry',True,fault=fault)
                self.assertEqual(result['status'],A.PASS if fault=='extra_sidecar' else 'UNKNOWN')
                self.assertEqual(code,0 if fault=='extra_sidecar' else 2)
                self.assertFalse((out/'fixture-job'/E.WITNESS).exists())
                kept=E.checkpoints(out/'fixture-job')
                self.assertTrue(kept)
                if fault=='bad_api':
                    self.assertEqual(kept['API_RETURN.json'],E.encode([dict(bad=True)]))
                if fault=='interrupted':
                    target=Path(tmp)/'partial'
                    E.export(out/'fixture-job',target)
                    self.assertEqual([p.name for p in target.iterdir()],['API_RETURN.json'])
                if fault=='extra_sidecar':
                    self.assertEqual(kept['PARTIAL_LEMMA.json'],E.encode(dict(audit='other worker evidence')))

    def test_corrupt_witness_cannot_validate_or_overwrite_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'attempt'
            self.assertEqual(self.exercise(out,'prdf_d2_common_entry',True)[0],0)
            original=(out/'OUTCOME.json').read_bytes()
            marker=out/'fixture-job'/E.WITNESS
            row=json.loads(marker.read_bytes())
            row['sha256']['API_RETURN.json']='invalid'
            marker.write_bytes(E.encode(row))
            code,result=self.exercise(out,'prdf_d2_common_entry',True,'collect')
            self.assertEqual(code,2)
            self.assertEqual(result['status'],'UNKNOWN')
            self.assertIn('checksum differs',result['reason'])
            self.assertEqual((out/'OUTCOME.json').read_bytes(),original)


if __name__=='__main__':
    unittest.main(verbosity=2)
