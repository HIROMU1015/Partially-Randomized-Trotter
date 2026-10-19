"""Re-extract ONLY new artificial raw and verify without producer imports."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tarfile
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign import g6_w2c_b3_pilot_offline as V
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_offline_adapter as W


def main():
    results=[]
    for case in sorted((HERE/'forward_evidence').iterdir()):
        c=json.loads((case/'pilot_contract.json').read_text())
        inner=json.loads((case/'inner_contract.json').read_text())
        obs=W.safe_values(json.loads((case/'observations.json').read_text()))
        expected='RESERVED'if case.name in ('commit_ack_uncertain','parent_crash')else'STOPPED'
        admission=V.admission_snapshot(case/'FAKE_ONLY_claims.sqlite3',inner,expected_status=expected)
        raw=None
        if 'report'in obs:
            with tempfile.TemporaryDirectory(prefix='b3-pilot-saved-offline-')as t:
                with tarfile.open(case/'artificial_raw.tar.gz')as tf:
                    tf.extractall(t,filter='data')
                raw=V.verify(Path(t),c,inner,obs['report'],reference_root=ROOT)
                assert raw['status']==obs['offline']['status']
                assert raw['history']==obs['offline']['history']
                assert raw['physical_retained_bytes']==obs['offline']['physical_retained_bytes']
        results.append(dict(case=case.name,admission=admission,
            raw_status=raw['status']if raw else'NOT_A_RAW_COMPLETION_CASE',
            counts=raw['history']['counts']if raw else None,
            raw_sha256=sha256((case/'artificial_raw.tar.gz').read_bytes()).hexdigest()if(case/'artificial_raw.tar.gz').exists()else None))
    forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1]in (
        'g7_generator','g7_provider','return_aggregation','g6_f3_contract','g6_f3_precision',
        'g6_f3_generator','g9_native','g6_w2c_b3_pilot_candidate','g6_w2c_b3_parallel_scheduler',
        'g6_w2c_b3_parallel_admission','g6_w2c_b3_parallel_host')]
    assert not forbidden
    (HERE/'saved_forward_verification.json').write_text(json.dumps(dict(cases=results,
        forbidden_science_or_producer_modules_imported=forbidden,successful=True,
        independent_implementation=True,separate_human_reviewer=False,historical_pending_resolved=False),indent=2)+'\n')
    print('Saved artificial cases independently re-read:',len(results))


if __name__=='__main__':main()
