#!/usr/bin/env python3
"""Reuse the saved artificial54 differential pipeline, writing only new output."""
from unittest.mock import patch

import run_track_d_stream_bridge_fast as campaign
from run_track_d_h4_native_bridge import ROOT, write, sha, git
from trottertracks.track_d_stream_bridge_fast import optimize_native_stream
from trottertracks.track_d_bridge_run_reuse import optimize_native_runs


def main():
    # Existing private artificial fixture root; never publish the raw payload.
    toy = (ROOT.parent/'track-d-c7-2-cx-structure-20261107' /
           'artifacts/track_d_c7_2_toy/2026-11-07/run01')
    out = ROOT/'artifacts/track_d_bridge_run_reuse/2026-11-10/technical'
    def baseline(ops,nq,nc,phase):
        return optimize_native_stream(ops,nq,nc,global_phase=phase)
    def save(path, value):
        expected = ROOT/'artifacts/track_d_stream_bridge_fast/2026-11-10/technical/artificial54.json'
        if path != expected:
            raise ValueError('unexpected old writer path')
        value['execution_source'] = git('rev-parse','HEAD').decode().strip()
        value['pipeline_source_sha256'] = sha((ROOT/'scripts/run_track_d_stream_bridge_fast.py').read_bytes())
        write(out/'saved54.json', value)
    with (patch.object(campaign,'baseline',baseline),
          patch.object(campaign,'optimize_native_stream',optimize_native_runs),
          patch.object(campaign,'write',save)):
        campaign.artificial(str(toy))
    print('saved artificial54: word/phase/rounds/all7 and saved P0 PASS')


if __name__ == '__main__':
    main()
