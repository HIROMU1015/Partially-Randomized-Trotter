"""Reuse completed PF compile bytes after an independent signal failure."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_distributed_preparation as P
from trottertracks.pr_candidate_completion import Held


class PartialCalibrationReuse(unittest.TestCase):
    def test_lookup_selects_only_the_matching_canonical_configuration(self):
        receipt=dict(status='COMPILE_RECEIPTS_READY',raw_commit='FIXED_RAW')
        settings=dict(saved_calibrations={'C1':receipt})
        before=deepcopy(settings)
        self.assertEqual(P.saved_for_configuration(settings,'saved_calibration','C1'),receipt)
        self.assertIsNone(P.saved_for_configuration(settings,'saved_calibration','C2'))
        self.assertEqual(settings,before)
        # Existing explicitly scoped single-candidate callers retain semantics.
        self.assertEqual(P.saved_for_configuration(dict(saved_calibration=receipt),'saved_calibration','C1'),receipt)
        for field, plural in [('saved_signal','saved_signals'),('saved_holdout_compile','saved_holdout_compiles')]:
            scoped={plural:{'C1':receipt}}
            self.assertEqual(P.saved_for_configuration(scoped,field,'C1'),receipt)
            self.assertIsNone(P.saved_for_configuration(scoped,field,'C2'))
        audits=dict(saved_calibration_audits={'C1':dict(status='AUDIT_PASS')})
        self.assertEqual(P.saved_for_configuration(audits,'saved_calibration_audit','C1')['status'],'AUDIT_PASS')
        with self.assertRaisesRegex(ValueError,'must map canonical'):
            P.saved_for_configuration(dict(saved_calibrations=[]),'saved_calibration','C1')

    def test_recovered_exact_raw_commit_is_restored_without_compile(self):
        for phase in ('calibration','holdout'):
            with self.subTest(phase=phase):
                self._check_phase(phase)

    def _check_phase(self,phase):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);source=base/'source';source.mkdir()
            def git(*args,cwd=source):
                return subprocess.check_output(['git',*args],cwd=cwd,text=True,stderr=subprocess.PIPE).strip()
            git('init','-q');git('config','user.name','Synthetic');git('config','user.email','synthetic@example.invalid')
            folder=source/'docs/research/distributed_selected_pf/C1';folder.mkdir(parents=True)
            plan=dict(configuration=dict(id='C1'),prepared=dict(pending=dict(requests=[dict(m=0),dict(m=1)])))
            (folder/'PLAN.json').write_text(json.dumps(plan))
            raw=folder/phase/'run01';raw.mkdir(parents=True)
            payload=json.dumps(dict(child_exit_code=0,results=[dict(ok=True,wrappers=dict(X={},Y={}))for _ in range(2)]))+'\n'
            (raw/'RESULT.json').write_text(payload);(raw/'PRICES.json').write_bytes(b'[]\n')
            git('add','.');git('commit','-qm','synthetic completed raw compile')
            commit=git('rev-parse','HEAD')
            workspace=base/'new-owner';git('clone','-q',str(source),str(workspace))
            target=workspace/'docs/research/distributed_selected_pf/C1'
            # Simulate a new workspace with metadata plan only.
            for f in (target/phase/'run01').iterdir():f.unlink()
            receipt=dict(status='COMPILE_RECEIPTS_READY',phase=phase,rounds=[0,1],
                raw_commit=commit,relative='docs/research/distributed_selected_pf/C1/'+phase+'/run01',route='TRUNCATED_PF')
            result=P.reuse_calibration(source,workspace,target,plan,[0,1],receipt,phase=phase)
            self.assertEqual(result,receipt)
            self.assertEqual((target/phase/'run01/RESULT.json').read_text(),payload)
            self.assertEqual(json.loads((target/('PF_COMPILE.RESULT.json' if phase=='calibration' else 'PF_HOLDOUT_COMPILE.RESULT.json')).read_text()),receipt)
            self.assertEqual((raw/'RESULT.json').read_text(),payload)
            wrong=deepcopy(plan);wrong['configuration']['id']='C2'
            with self.assertRaisesRegex(Held,'PF_SAVED_CALIBRATION_REUSE_HOLD'):
                P.reuse_calibration(source,workspace,target,wrong,[0,1],receipt,phase=phase)
            with self.assertRaisesRegex(Held,'PF_SAVED_CALIBRATION_REUSE_HOLD'):
                P.reuse_calibration(source,workspace,target,plan,[1],receipt,phase=phase)


if __name__=='__main__':unittest.main()


def test_publish_pf_locators_without_unrelated_history(tmp_path):
    import json
    import subprocess
    from trottertracks.pf_distributed_preparation import publish_pf_locators

    source, destination = tmp_path/'source', tmp_path/'destination'
    for root in (source, destination):
        subprocess.run(['git','init','-q',str(root)],check=True)
        subprocess.run(['git','config','user.email','test@example.invalid'],cwd=root,check=True)
        subprocess.run(['git','config','user.name','Test'],cwd=root,check=True)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=source,text=True).strip()
    (source/'irrelevant').write_text('not a scientific input')
    (source/'signal.json').write_text('{"signal": 1}')
    git('add','.');git('commit','-qm','fixed signal')
    first=git('rev-parse','HEAD')
    missing=git('rev-parse','HEAD:irrelevant')
    (source/'index.json').write_text(json.dumps({'signal':{'commit':first,'path':'signal.json'}}))
    git('add','index.json');git('commit','-qm','publish index')
    last=git('rev-parse','HEAD')
    (source/'.git/objects'/missing[:2]/missing[2:]).unlink()
    publish_pf_locators(source,destination,{'commit':last,'path':'index.json'})
    for ref,path in [(first,'signal.json'),(last,'index.json')]:
        before=git('show',ref+':'+path)
        after=subprocess.check_output(['git','show',ref+':'+path],cwd=destination,text=True).strip()
        assert before==after
