#!/usr/bin/env python3
"""Read-only phase/order diagnostics of the same four saved q4 predictions.

No prediction rewrite, optimizer call, circuit generation or state simulation.
"""
import argparse
import gzip
import io
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

from run_track_d_h4_native_bridge import ROOT,OUT,read,write,git,sha,member,archive_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True)
    p.add_argument('--prediction-commit',required=True);a=p.parse_args()
    prediction_commit=git('rev-parse',a.prediction_commit).decode().strip()
    frozen=(ROOT/OUT/'PREDICTIONS.json').read_bytes()
    if frozen!=git('show',prediction_commit+':'+OUT+'/PREDICTIONS.json'):raise ValueError('fixed prediction bytes')
    source=git('rev-parse','HEAD').decode().strip();relative='scripts/check_track_d_h4_native_phase.py'
    code=(ROOT/relative).read_bytes()
    if code!=git('show',source+':'+relative):raise ValueError('unfixed diagnostic SOURCE')
    comparisons=read(ROOT/OUT/'COMPARISON.json');predictions=json.loads(frozen)
    manifest=archive_manifest(a.archive)
    import qiskit
    from qiskit import qpy
    from trottertracks.track_d_native_resources import iter_qiskit_native
    rows=[];started=time.perf_counter()
    with (patch.object(qiskit,'transpile',side_effect=AssertionError('new P0 forbidden')),
          patch.object(qiskit.compiler,'transpile',side_effect=AssertionError('new P0 forbidden'))):
        for prediction in predictions['records']:
            method,axis=prediction['method'],prediction['axis'];name=method+'_'+axis
            public=next(r for r in comparisons['records'] if (r['method'],r['axis'])==(method,axis))
            saved=member(a.archive,manifest,'circuits/'+name+'_compiled.qpy')
            c=qpy.load(io.BytesIO(saved))[0]
            path=ROOT/OUT/'local'/(name+'_bridge.jsonl.gz')
            if sha(path.read_bytes())!=prediction['private_payload_sha256']['bridge']:
                raise ValueError('unchanged predicted native word required')
            with gzip.open(path,'rt') as f:word=[json.loads(line) for line in f]
            oracle=[[op.name,list(op.qubits),list(op.clbits),list(op.params)] for op in iter_qiskit_native(c)]
            def lanes(ops):
                result=[[] for _ in range(c.num_qubits+c.num_clbits)]
                for i,op in enumerate(ops):
                    for w in op[1]+[c.num_qubits+b for b in op[2]]:result[w].append((i,op))
                return result
            left,right=lanes(word),lanes(oracle)
            same=all([[o[:3] for _,o in l]==[o[:3] for _,o in r] for l,r in zip(left,right)])
            diagnostic=dict(method=method,axis=axis,wirewise_opcode_order_matches=same,
                phase_difference_mod_2pi=public['phase_difference_mod_2pi'],new_correctness_PASS=False,
                phase_or_angles_fitted=False,prediction_changed=False)
            if same:
                angles={}
                for l,r in zip(left,right):
                    for (i,x),(_,y) in zip(l,r):
                        if x[0]=='rz':angles[i]=x[3][0]-y[3][0]
                turns=sum(round(d/(2*math.pi)) for d in angles.values())
                diagnostic.update(matched_rz=len(angles),max_rz_angle_difference=max(map(abs,angles.values()),default=0.),
                    max_rz_angle_difference_mod_2pi=max((abs(math.remainder(d,2*math.pi)) for d in angles.values()),default=0.),
                    analytic_2pi_RZ_turns=turns,
                    phase_difference_after_analytic_RZ_turns=math.remainder(public['phase_difference_mod_2pi']+math.pi*turns,2*math.pi))
            rows.append(diagnostic)
    write(ROOT/OUT/'DIAGNOSTICS.json',dict(execution_source=source,diagnostic_source_sha256=sha(code),
        prediction_commit_resolved=prediction_commit,prediction_sha256=sha(frozen),
        original_comparison_argument=comparisons['prediction_commit'],
        comparison_sha256=sha((ROOT/OUT/'COMPARISON.json').read_bytes()),records=rows,
        wall_seconds=time.perf_counter()-started,new_optimizer_calls=0,new_transpile=0,new_simulation=0,
        interpretation='finite opcode/angle/declared-phase diagnostic; no full-input correctness certificate'))


if __name__=='__main__':main()
