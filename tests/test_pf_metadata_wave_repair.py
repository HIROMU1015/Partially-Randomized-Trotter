"""Failed PF owner metadata checks do not become permanent science holds."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pf_candidate_selection import repair_metadata_waves


class MetadataWaveRepair(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name)/'parent';self.path=self.parent/'pf_auto/WAVES.json'
        self.cid='a'*64;self.directory=self.parent/'pf_auto/candidates'/self.cid
        self.row=dict(status='PREPARATION_SCHEDULED',key=dict(p=9,formula='4th(m5_best)',delta_over_h='16'),configurations={})
        self.records={self.cid:self.row,'untouched':dict(status='PREPARATION_SCHEDULED',key={},configurations={})}
        self.put(self.path,self.records)
        self.put(self.parent/'LAUNCH.json',dict(pid=99999999,invocation_id='original'))
        self.put(self.parent/'EXIT.json',dict(returncode=1,invocation_id='original'))
        self.put(self.parent/'KAPPA_STATE.json',dict(stop=dict(reason='RUNTIME_EXCEPTION',error_type='CalledProcessError',
            message='check_pf_preparation_owner.py '+self.cid+' returned non-zero exit status 1')))
        self.put(self.directory/'pf_auto/SELECTION.json',dict(selected=dict(configuration_id=self.cid,key=self.row['key'])))
        for name in ['REQUEST.json','ORIGINAL_OWNER_REQUEST.json']:
            self.put(self.directory/'pf_auto/pf_completion'/name,dict(selected=[dict(configuration_body=dict(id=self.cid))]))
        self.repair=dict(state_file=str(self.path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),configuration_ids=[self.cid])

    @staticmethod
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_only_failed_metadata_candidate_can_be_selected_again_and_original_kept(self):
        before={p:p.read_bytes() for p in self.parent.rglob('*') if p.is_file()}
        records,repairs=repair_metadata_waves(self.records,[self.repair])
        self.assertEqual(set(records),{'untouched'})
        self.assertEqual(repairs[self.cid]['original_record'],self.row)
        self.assertEqual(self.records[self.cid],self.row)
        self.assertEqual(before,{p:p.read_bytes() for p in self.parent.rglob('*') if p.is_file()})

    def test_scientific_marker_or_unknown_outcome_cannot_be_dropped(self):
        marker=self.directory/'pf_auto/pf_completion/SIGNAL_STARTED.json';self.put(marker,dict(science=True))
        with self.assertRaisesRegex(ValueError,'retry a stage'):repair_metadata_waves(self.records,[self.repair])

    def test_live_owner_or_changed_wave_is_rejected(self):
        with patch('trottertracks.rpe_search_continuation._has_local_owner',return_value=True):
            with self.assertRaisesRegex(ValueError,'ended owner-check'):repair_metadata_waves(self.records,[self.repair])
        changed=deepcopy(self.records);changed[self.cid]['key']['p']=8
        with self.assertRaisesRegex(ValueError,'change its selection'):repair_metadata_waves(changed,[self.repair])


if __name__=='__main__':unittest.main()
