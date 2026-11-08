"""Small artificial archive/descriptor tests; no molecular payload or circuit."""
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import run_track_d_o5_q4_metadata as runner
SAVED,SELECTOR,FEATURES=runner.bootstrap()


def metadata(lengths):
    return [[dict(rotation_angle=.2,application_sequence=[dict(basis_id='invented',
        diagonal_pauli_support=[0],coefficient_sign=1)]) for _ in range(n)] for n in lengths]


class MetadataTests(unittest.TestCase):
    def archive(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        path=Path(temp.name)/'invented.tar.gz'
        payload={'events.json':b'invented metadata','circuits/fake.qpy':b'NOT A CIRCUIT','references.npz':b'NOT AN ARRAY'}
        with tarfile.open(path,'w:gz') as tar:
            for name,data in payload.items():
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        return path,dict(bytes=path.stat().st_size,sha256=runner.sha(path.read_bytes()),
            members=[dict(path=k,bytes=len(v),sha256=runner.sha(v)) for k,v in payload.items()])

    def test_reader_extracts_only_event_member(self):
        path,manifest=self.archive();calls=[];original=tarfile.TarFile.extractfile
        def extract(tar,member):calls.append(member.name);return original(tar,member)
        with patch.object(tarfile.TarFile,'extractfile',extract):
            self.assertEqual(runner.event_only(path,manifest),b'invented metadata')
        self.assertEqual(calls,['events.json'])

    def test_archive_and_event_hash_mismatch_stop(self):
        path,manifest=self.archive();good=manifest['sha256'];manifest['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'ARCHIVE_IDENTITY'):runner.event_only(path,manifest)
        manifest['sha256']=good;manifest['members'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'EVENT_MEMBER_IDENTITY'):runner.event_only(path,manifest)

    def test_frozen_rule_can_choose_noncentral_and_resolves_minimum_tie(self):
        off=SELECTOR.select_cut(metadata([100,1,1,1]),FEATURES.event_features)
        self.assertEqual(off.selected,1)
        tie=SELECTOR.select_cut(metadata([4,1,1,1]),FEATURES.event_features)
        self.assertEqual(tie.selected,2)
        self.assertEqual(tie.candidates[0][1],tie.candidates[1][1])

    def test_feature_capture_is_once_per_step_and_prices_are_ignored(self):
        observed=[]
        def capture(record):value=FEATURES.event_features(record);observed.append(value);return value
        events=metadata([4,1,1,1])
        for step in events:
            for event in step:event['compiled_depth']=-99999
        decision=SELECTOR.select_cut(events,capture)
        self.assertEqual(len(observed),4)
        row=runner.decision_row('invented',asdict(decision),observed,False)
        self.assertEqual(row['step_basis_runs'],[1]*4);self.assertEqual(row['step_support_work'],[4,1,1,1])
        self.assertEqual(row['step_loads'],[6,3,3,3]);self.assertTrue(row['selection_tie'])
        self.assertEqual(row['minimum_score_tied_cuts'],[1,2])

    def test_followup_candidate_uses_sample_id_not_score(self):
        rows=[dict(sample_id='q4_sample15',selected_cut=1,selection_tie=False,any_candidate_score_tie=False,
                   best_noncentral_minus_center=-100,step_load_range=100,center_half_load_imbalance=100),
              dict(sample_id='q4_sample02',selected_cut=3,selection_tie=False,any_candidate_score_tie=False,
                   best_noncentral_minus_center=-1,step_load_range=1,center_half_load_imbalance=1)]
        summary=runner.summarize(rows)
        self.assertEqual(summary['next_B_candidate'],'q4_sample02');self.assertEqual(summary['case'],'B')


if __name__=='__main__':unittest.main()
