"""PBS release checks precede existing worker acquisition; no scientific leaf."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_pbs_worker as W
from trottertracks.rpe_pbs import PbsManager
import test_rpe_pbs as F
import check_rpe_pbs as P
import check_rpe_romutail_distributed as H


class PbsReleaseBinding(unittest.TestCase):
    def worker(self,directory):
        base=Path(directory);manifest=base/'MANIFEST.json';layout=base/'LAYOUT.json'
        manifest.write_text(json.dumps(dict(schema='RPE_RELEASE_COMPONENTS_V1',release_id='SELECTED')))
        layout.write_text(json.dumps(dict(host_id='pbs-0')))
        worker=W.AllocationWorker.__new__(W.AllocationWorker)
        worker.profile=dict(host_id='pbs-0',pbs=dict(deployment=dict(manifest=str(manifest),layout=str(layout))))
        return worker

    def test_wrong_release_cannot_reach_original_worker_acquisition(self):
        with tempfile.TemporaryDirectory()as directory:
            worker=self.worker(directory)
            with patch.object(W.Worker,'_acquire',side_effect=AssertionError('Scientific worker must remain uncalled')):
                with self.assertRaisesRegex(ValueError,'Assigned release'):
                    worker._acquire(dict(required_release=dict(id='WRONG',manifest_sha256='WRONG')))

    def test_successful_binding_preserves_assignment_and_legacy_acquisition(self):
        with tempfile.TemporaryDirectory()as directory:
            worker=self.worker(directory);assignment=dict(request={'request_id':'UNCHANGED'},required_release={'id':'SELECTED'})
            before=deepcopy(assignment)
            with patch('trottertracks.rpe_release_deployment.validate_assignment_release',return_value={'status':'PASS'})as check,patch.object(W.Worker,'_acquire')as acquire:
                worker._acquire(assignment)
                check.assert_called_once();acquire.assert_called_once_with(assignment)
            self.assertEqual(assignment,before)
            worker.profile['pbs'].pop('deployment')
            with patch('trottertracks.rpe_release_deployment.validate_assignment_release',side_effect=AssertionError('Legacy contract must stay unchanged')),patch.object(W.Worker,'_acquire')as acquire:
                worker._acquire(assignment);acquire.assert_called_once_with(assignment)

    def test_missing_selected_components_hold_before_worker_startup(self):
        with tempfile.TemporaryDirectory()as directory:
            worker=self.worker(directory);profile=Path(directory)/'PROFILE.json';profile.write_text(json.dumps(worker.profile))
            with patch('trottertracks.rpe_release_deployment.ensure_components',return_value=dict(status='HOLD',holds=['MISSING_FIXED_SOURCE'])),patch.object(W,'fixed_worker',side_effect=AssertionError('Worker must not start')):
                with self.assertRaisesRegex(ValueError,'PBS release deployment HOLD'):W.run(profile)

    def test_manager_copies_operational_binding_without_changing_scientific_profile(self):
        with tempfile.TemporaryDirectory()as directory:
            settings=F.settings();settings['deployment']=dict(manifest='/FIXED/MANIFEST',layout='/FIXED/PBS_LAYOUT')
            before=deepcopy(settings['workers'][0]['profile']);gateway=F.FakeGateway()
            manager=PbsManager(F.FakeExecutor(directory),settings,gateway=gateway)
            try:
                manager._submit('pbs-0',settings['workers'][0],dict(artificial=True))
                spec=gateway.calls[-1]['spec']
                self.assertEqual(spec['profile']['pbs']['deployment'],settings['deployment'])
                self.assertEqual(settings['workers'][0]['profile'],before)
            finally:manager.close()

    def test_normal_gateway_planonly_with_pbs_has_no_allocation_or_scientific_queue(self):
        with tempfile.TemporaryDirectory()as directory:
            fixture=Path(directory)/'fixture'
            P.prepare(fixture,ROOT,F.settings())
            control=Path(directory)/'gateway';control.mkdir()
            for name in ['PROBLEM.json','ROUND.json']:
                (control/('ROUND.requested.json'if name=='ROUND.json'else name)).write_bytes((fixture/name).read_bytes())
            with patch('trottertracks.rpe_pbs.PbsManager',side_effect=AssertionError('PlanOnly must not allocate PBS')),redirect_stdout(io.StringIO()):
                H.controller(fixture,plan_only=True,control_run=control,max_cycles=1)
            self.assertTrue((control/'plan-only/KAPPA_STATE.json').exists())
            self.assertFalse((fixture/'run').exists())
            self.assertFalse((control/'plan-only/distributed/QUEUE.sqlite3').exists())


if __name__=='__main__':unittest.main()
