"""Synthetic interruption cases; no scientific calls or historical outputs."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h46_long_time_recover as recovery


def test_completed_trial_is_reused_and_interrupted_bytes_are_preserved(tmp_path):
    row = dict(id='artificial', method='PR')
    plan = dict(candidates=[row], screening_trajectories_per_new_PR=3)
    root = tmp_path / 'cost'
    group = root / 'ARTIFICIAL' / row['id']
    recovery.native.save(group / '0/result.json', dict(status='PASS', candidate=row,
        cohort='ARTIFICIAL', index=0))
    recovery.native.save(group / '1/events_PRIVATE.json', dict(seeds=[17]))
    original = (group / '1/events_PRIVATE.json').read_bytes()
    complete, interrupted, jobs = recovery.inventory(plan, 'ARTIFICIAL', {}, root)
    assert [t['index'] for t in complete[row['id']]] == [0]
    assert [j[2] for j in jobs] == [1, 2]
    recovery.preserve_interrupted(interrupted, root, tmp_path / 'archive')
    assert (group / '0/result.json').exists()
    assert (tmp_path / 'archive/cost/ARTIFICIAL/artificial/1/events_PRIVATE.json').read_bytes() == original
    assert not (group / '1').exists()


def test_existing_scientific_stop_is_never_reopened(tmp_path):
    row = dict(id='artificial', method='PR')
    root = tmp_path / 'cost'
    recovery.native.save(root / 'ARTIFICIAL/artificial/1/STOP.json', dict(reason='synthetic mismatch'))
    with pytest.raises(ValueError, match='EXISTING_SCIENTIFIC_STOP'):
        recovery.inventory(dict(candidates=[row]), 'ARTIFICIAL', {}, root)
    assert (root / 'ARTIFICIAL/artificial/1/STOP.json').exists()
