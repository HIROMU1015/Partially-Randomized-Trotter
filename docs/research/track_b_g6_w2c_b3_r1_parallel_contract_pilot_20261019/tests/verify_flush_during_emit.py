"""Supplemental artificial probe: flush failure inside emit, not just close."""
from pathlib import Path
from hashlib import sha256
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_parallel_artificial_fixture as F


def main():
    original = F.D.Stream
    class EveryRow(original):
        def __init__(self, *args, **kwargs):
            kwargs['flush_rows'] = 1  # Probe-only; production/pilot default stays 4096.
            super().__init__(*args, **kwargs)
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as tmp:
        root = Path(tmp)
        settings = F.P.Settings(profile_n=16,workload='light',faults=((0,0,'flush',0),))
        c, ledger, auth = F.fixture(root, settings)
        try:
            with patch.object(F.D,'Stream',EveryRow):
                F.invoke(root,c,ledger,auth,settings)
        except F.P.BundleFailure as exc:
            row = next(r for r in exc.report['pairs'] if r['task']['pair_index']==0)
            provenance = row['failure']['trial_failure_provenance']
            assert provenance['current_attempt']['stage']=='STREAM_EMIT'
            assert not provenance['current_attempt']['stream_emit_returned']
            assert sum(provenance['completed_prefix'].values())==0
            assert provenance['last_completed_trial'] is None
            raw = root/'artificial-output'/('correctness_baseline_ordinary_A.jsonl')
            parsed = F.read_stream(raw)
            assert len(parsed['rows'])==1 and parsed['partial_bytes']==0
            assert parsed['chain']==row['chain'] and parsed['file_sha']==row['file_sha']
            assert ledger.row(c['scope'])[4]=='STOPPED'
            target = Path(__file__).parent
            (target/'flush_emit_pair_raw.jsonl').write_bytes(raw.read_bytes())
            result = dict(status='PASS',artificial_only=True,suite_test_count_unchanged=311,
                supplemental_probe_count=1,raw_complete_rows=1,logical_completed_trials=0,
                raw_file_sha256=parsed['file_sha'],raw_chain=parsed['chain'],
                f01=provenance,claim_status='STOPPED',scientific_calls=0,real_claims=0,
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
            (target/'flush_emit_verification.json').write_bytes(F.P.safe_json(result)+b'\n')
            print('PASS: one complete hashed row retained, zero logical commits, spent fake claim')
        else:
            raise AssertionError('FLUSH_FAULT_NOT_OBSERVED')


if __name__=='__main__':main()
