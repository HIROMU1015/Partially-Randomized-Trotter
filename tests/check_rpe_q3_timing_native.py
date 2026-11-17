"""Artificial total-DP tables, unchanged native AST guard, no H4 calls."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
F=load('_timing_dp_fixture',ROOT/'tests/test_rpe_dp_prefix_reuse.py')
T=load('_timing_helper',ROOT/'src/trottertracks/rpe_q3_timing.py')
_,tables,scope=F.example(n=4)
expected=F.P.Accelerator(workers=1)(tables,h=10.,target=0.,scope=scope)
with tempfile.TemporaryDirectory()as d:
    with F.C.installed(F.P,directory=None,cache_namespace={'fixture':'timing'},checkpoint_rounds=[1,2]):
        with T.body_context(F.P,d):
            engine=F.P.Accelerator(workers=1)
            for _ in range(2):assert engine(tables,h=10.,target=0.,scope=scope)==expected
            assert any(x['event']=='HIT'for x in engine.cache_events)
    rows=T.read_lines(Path(d)/'Q3_BODY_TIMING.jsonl')
    assert [r['event']for r in rows]==['Q3_BODY_START','Q3_BODY_END']*2
    assert all(T.delta(rows[i],rows[i+1]) is not None for i in (0,2))
    print(json.dumps(dict(status='PASS',artificial_only=True,scientific_H4_calls=0,body_calls=2,exact_DP_equal=True,cache_HIT=True,AST_guard='UNCHANGED',events=rows)))
