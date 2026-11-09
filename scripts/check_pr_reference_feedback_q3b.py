"""Check Q3-B coverage/extremal witnesses and exact B2 arithmetic only.

Does not repeat the count grid, old Q0 inversion/scan, or any circuit work.
"""
from collections import defaultdict
from fractions import Fraction as F
import hashlib,json,lzma
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_reference_feedback_q3b as b
out=Path(sys.argv[1]).resolve()
source=json.loads((out/'SOURCE.json').read_text());inp=json.loads((out/'INPUTS.json').read_text())
b1=json.loads((out/'B1.json').read_text());b2=json.loads((out/'B2.json').read_text())
q=b.q1._q0()
for p,h in source['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
assert b1['status']=='PASS_FIXED_DIRECTED_INTERVAL_MODEL'
assert b2['status']=='PASS_FIXED_INTERVAL_MODEL'
rows=defaultdict(list)
with lzma.open(out/'B1_ROWS.jsonl.xz','rt') as stream:
    for line in stream:
        row=json.loads(line);assert row['status']=='PASS' and row['all_nonzero_cases_pass']
        n=row['n'];kx=row['kx'];rows[n].append(row)
        assert row['ky_domain']==[0,n] and row['nonzero']+row['zero']==n+1
        assert row['zero']==int(n%2==0 and kx==n//2)
        lo,hi=map(F,row['phase_witness']['phase']['exact_interval'])
        assert hi-lo==F(row['phase_width_upper'])<=b.PHASE_WIDTH_LIMIT
assert set(rows)==set(inp['unique_denominators'])
unique_nonzero=unique_zero=0;max_phase=F(0);max_feedback=None
replayed=[]
for n,batch in rows.items():
    assert sorted(r['kx'] for r in batch)==list(range(n+1))
    t=b1['tables'][str(n)]
    assert sum(r['nonzero'] for r in batch)==t['nonzero']
    assert sum(r['zero'] for r in batch)==t['zero']
    assert t['all_pairs']==(n+1)**2
    assert max(F(r['phase_width_upper']) for r in batch)==F(t['max_phase_width'])
    unique_nonzero+=t['nonzero'];unique_zero+=t['zero'];max_phase=max(max_phase,F(t['max_phase_width']))
    w=t['phase_witness']
    phase=b.count_phase(b.q1.RoundCounts(0,b.q1.AxisCounts(n,w['kx']),b.q1.AxisCounts(n,w['ky'])))
    assert q.pack(phase)==w['phase'] and b.width(phase)==F(t['max_phase_width'])
    replayed.append(dict(n=n,kx=w['kx'],ky=w['ky'],kind='max_phase_width'))
    if t['max_feedback_error'] is not None:
        assert max(F(r['feedback_error_upper']) for r in batch)==F(t['max_feedback_error'])
        w=t['feedback_witness']
        p=b.count_phase(b.q1.RoundCounts(16,b.q1.AxisCounts(n,w['kx']),b.q1.AxisCounts(n,w['ky'])))
        f=b.q1._feedback_phase(p)
        assert f.arithmetic_error_upper==F(t['max_feedback_error'])<=b.ARITHMETIC_ANGLE_LIMIT<b.q1.FEEDBACK_CAP
        max_feedback=f.arithmetic_error_upper if max_feedback is None else max(max_feedback,f.arithmetic_error_upper)
        replayed.append(dict(n=n,kx=w['kx'],ky=w['ky'],kind='max_feedback_error'))
    else:assert n not in inp['feedback_denominators']
assert unique_nonzero+unique_zero==inp['unique_full_grid_cases']
mapped_nonzero=mapped_zero=0
for arm,ns in inp['schedules'].items():
    assert len(ns)==17
    for m,(n,rec) in enumerate(zip(ns,b1['per_arm_round_bounds'][arm])):
        assert rec['m']==m and rec['n']==n and rec['q']==1<<m
        t=b1['tables'][str(n)];mapped_nonzero+=t['nonzero'];mapped_zero+=t['zero']
        pw=F(t['max_phase_width'])
        wc=pw if m==0 else pw/(1<<m)+2*b.width(q.mp.iv.pi)+b.BASIC_SLACK
        assert F(rec['base_center_width_upper'])==wc<b.CENTER_WIDTH_LIMIT
        if m:
            cell=((1<<m)*b.CENTER_WIDTH_LIMIT+pw)/6+b.BASIC_SLACK
            assert F(rec['cell_width_upper'])==cell<F(1,1<<130)<F(1,4)
        assert F(rec['energy_center_width_upper'])==wc/b.q1.REFERENCE_TIME+b.OMEGA_E<b.KAPPA_LIMIT
assert mapped_nonzero+mapped_zero==inp['mapped_arm_round_cases']
for row in b2['rows']:
    arm=row['arm'];kappa=F(b1['per_arm_round_bounds'][arm][-1]['energy_center_width_upper'])
    time17=(1<<17)*b.q1.REFERENCE_TIME
    reserve=F(row['delta_lo'])-b.q1.FEEDBACK_CAP
    lhs=kappa*time17+b.OMEGA_E*time17+b.OMEGA_Q0
    assert F(row['kappa_Ha']['exact'])==kappa
    assert F(row['reserve']['exact'])==reserve and F(row['lhs_rad']['exact'])==lhs
    assert F(row['residual_rad']['exact'])==reserve-lhs>0 and row['status']=='PASS'
assert b.OMEGA_E>0 and b.OMEGA_Q0>0 and max_feedback is not None
output=dict(status='PASS',unique_nonzero_pairs=unique_nonzero,unique_zero_vectors=unique_zero,
    mapped_nonzero_pairs=mapped_nonzero,mapped_zero_vectors=mapped_zero,
    maximum_phase_width=str(max_phase),maximum_feedback_arithmetic_error=str(max_feedback),
    witness_replays=replayed,uniform_integer_cell_acceptance='PROVED_ON_FIXED_SUCCESS_EVENT_IN_192_BIT_MODEL',
    numerical_basis='directed interval primitive contract retained; integer-coordinate overlap is a consistency check, not independent libmp verification',
    B2_exact_rational_comparisons=True,full_grid_or_Q0_campaign_repeated=False,
    new_circuit_or_RPE_or_price=False,native_all_contexts_qualified=False,
    checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    result_hashes={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ('INPUTS.json','B1.json','B1_ROWS.jsonl.xz','B2.json','RESULTS.json')})
target=out/'CHECKS.json';assert not target.exists(),'Preserve completed check record'
target.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ('witness_replays','result_hashes')},indent=2))
