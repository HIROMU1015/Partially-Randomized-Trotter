"""Artificial byte/record tests only; no scientific calculation or circuits."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import gzip,hashlib,json,multiprocessing as mp,os,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import publish_full_h_d7_holdout as Pub
import check_full_h_d7_holdout as T

_BARRIER=None
_ROWS=None

def artificial_cell(task):
    _,_,cid,_=task
    _BARRIER.wait(timeout=15)
    return [r for r in _ROWS if r['cell_id']==cid],dict(cell_id=cid,status='PASS',worker_pid=os.getpid()),None


class Tests(unittest.TestCase):
    def test_parallel_gzip_preserves_raw_bytes_hash_and_part_order(self):
        raw=(bytes(range(256))*4096+b'ARTIFICIAL_NO_H4\n')*16
        with TemporaryDirectory() as temp,patch.object(Pub.R,'ROOT',Path(temp)):
            observations=[]
            for parallel in (False,True):
                directory=Path(temp)/str(parallel);directory.mkdir();(directory/'RESULT.json').write_bytes(raw)
                manifest=Pub.R.archive_result(directory,part_bytes=1024,parallel_gzip=parallel)
                data=b''.join((Path(temp)/p['path']).read_bytes() for p in manifest['parts'])
                self.assertEqual(gzip.decompress(data),raw)
                self.assertEqual(manifest['raw_sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(hashlib.sha256(data).hexdigest(),manifest['compressed_sha256'])
                observations.append(manifest['raw_sha256'])
                if parallel:self.assertGreater(manifest['compression']['workers'],1)
            self.assertEqual(*observations)
            print('ARTIFICIAL_GZIP',json.dumps(manifest['compression']))

    def test_published_cell_checks_run_concurrently_and_keep_four_exact_rows(self):
        global _BARRIER,_ROWS
        fixture=T.Tests();fixture.setUp();bound=fixture.bound();_ROWS=fixture.observations()
        for row in _ROWS:row['compile_receipt']=dict(reused=False)
        _BARRIER=mp.get_context('fork').Barrier(2)
        frozen=dict(prediction_bindings=bound,jobs={c:dict(plan=fixture.plan) for c in ('M5_ROUND','M6_ROUND')})
        with TemporaryDirectory() as temp,patch.object(Pub,'ROOT',Path(temp)), \
             patch.object(Pub.P.P1,'blob',return_value=json.dumps(frozen).encode()), \
             patch.object(Pub.H,'prediction_bindings',return_value=bound),patch.object(Pub,'bind_cell',artificial_cell):
            result=Pub.bind('ARTIFICIAL_PUBLICATION','ARTIFICIAL_EXECUTION')
            self.assertEqual(result['status'],'DESCRIPTIVE_COMPLETE')
            comparison=json.loads((Path(temp)/Pub.H.STUDY/'COMPARISON.json').read_text())
            pids={r['worker_pid'] for r in comparison['publication_checks']}
            self.assertEqual(len(pids),2);self.assertNotIn(os.getpid(),pids)
            expected=Pub.H.compare(bound,_ROWS)
            self.assertEqual(comparison['rows'],expected['rows'])
            self.assertEqual(comparison['tolerance'],None)

    def test_completed_cell_staging_copies_existing_bytes_and_never_runs_science(self):
        with TemporaryDirectory() as temp:
            base=Path(temp);publication=base/'publication';checkout=base/'scientific'
            source=checkout/Pub.H.STUDY/'run01'/'M5_ROUND';source.mkdir(parents=True)
            (source/'RESULT.json').write_bytes(b'{"status":"ARTIFICIAL_SAVED_ONLY"}')
            run=source.parent
            (run/'M5_ROUND.time.txt').write_text('Exit status: 0\n');(run/'M5_ROUND.console.txt').write_text('ARTIFICIAL')
            private=base/'private';private.mkdir()
            job=dict(output=Pub.H.STUDY+'/run01/M5_ROUND',private=str(private))
            (run/'JOB_M5_ROUND.json').write_text(json.dumps(job))
            (publication/Pub.H.STUDY).mkdir(parents=True)
            (publication/Pub.H.FROZEN).write_text(json.dumps(dict(jobs=dict(M5_ROUND=job))))
            with patch.object(Pub,'ROOT',publication),patch.object(Pub.subprocess,'check_output',return_value='a'*40), \
                 patch.object(Pub,'prepare',return_value={'ARTIFICIAL':'NO_COMPUTATION'}) as prepare:
                self.assertEqual(Pub.stage_when_finished('M5_ROUND','a'*40,checkout),{'ARTIFICIAL':'NO_COMPUTATION'})
                prepare.assert_called_once_with('M5_ROUND','a'*40)
            self.assertEqual((publication/job['output']/'RESULT.json').read_bytes(),(source/'RESULT.json').read_bytes())
            other=base/'missing_receipt_publication';(other/Pub.H.STUDY).mkdir(parents=True)
            (other/Pub.H.FROZEN).write_text(json.dumps(dict(jobs=dict(M5_ROUND=job))))
            (source/'RESULT.json').unlink()
            with patch.object(Pub,'ROOT',other),patch.object(Pub.subprocess,'check_output',return_value='a'*40), \
                 patch.object(Pub,'prepare') as prepare:
                with self.assertRaisesRegex(ValueError,'without final receipt'):
                    Pub.stage_when_finished('M5_ROUND','a'*40,checkout)
                prepare.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
