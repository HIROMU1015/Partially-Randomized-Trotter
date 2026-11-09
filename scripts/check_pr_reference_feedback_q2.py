"""Check the saved Q2 records and QPY, without state action or compilation."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from qiskit import qpy

out=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parents[1]
raw=(out/'RESULTS.json').read_bytes();result=json.loads(raw)
source=json.loads((out/'SOURCE.json').read_text())
assert result['status']=='FINITE_Q2_CORRECTNESS_PASS'
assert source['base_commit']==result['base_commit']=='f3705321515f35c08c702e377739b2c3e6e9df13'
for p,h in source['source_sha256'].items():
    assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
for p,meta in source['inputs'].items():
    blob=subprocess.check_output(['git','show',meta['commit']+':'+p],cwd=root)
    assert len(blob)==meta['bytes'] and hashlib.sha256(blob).hexdigest()==meta['sha256'],p
rows=result['rows'];assert len(rows)==16
angles={'X':'0x0.0p+0','Y':'0x1.921fb54442d18p+0','positive':'0x1.b4395810624ddp-3','negative':'-0x1.7020c49ba5e35p-1'}
expected={(r,a,k) for r in (0,1) for a in ('Bprime','Reference_B1') for k in angles}
actual={(r['identity']['rare'],r['identity']['arm'],r['identity']['angle_name']) for r in rows}
assert actual==expected
assert result['calls']==dict(CT=0,RPE=0,baseline_compiles=0,cache_state_load=1,
    native_compiles=16,native_state_actions=16,old_science_runners=0,
    raw_core_state_actions=4,raw_cores=4,sampling=0,wrapper_builds=16)
checks=[]
with tarfile.open(out/'native_qpy.tar.xz','r:xz') as tar:
    members={Path(m.name).stem:m for m in tar if m.isfile()}
    assert len(members)==16
    for row in rows:
        ident=row['identity'];rare=ident['rare'];arm=ident['arm'];name=ident['angle_name']
        assert ident['q']==ident['r']==1 and ident['j']==0
        assert ident['angle_hex']==angles[name]
        assert row['passed'] and row['tolerance']==1e-10
        assert max(row[k] for k in ('raw_error','native_error','raw_native_difference'))<1e-10
        assert abs(row['native']-row['expected'])==row['native_error']
        assert abs(row['raw']-row['expected'])==row['raw_error']
        if name in ('X','Y'):
            assert abs(row['expected']-row['mathematical_XY_saved_coherence_component'])<1e-10
        stem=f'rare{rare}_{arm}_{name}'
        payload=tar.extractfile(members[stem]).read();c=qpy.load(io.BytesIO(payload))[0]
        ops=dict(c.count_ops());price=row['price']
        assert c.num_qubits==price['qubits'] and c.num_clbits==1
        assert ops==row['ops'] and ops['rz']==price['rz'] and ops['cx']==price['cx']
        assert c.depth()==price['depth'] and c.size()==price['size']
        assert sum(ops.values())==price['size']
        resets=[] if arm=='Bprime' else list(range(8,16))
        assert [c.find_bit(i.qubits[0]).index for i in c.data if i.operation.name=='reset']==resets
        assert all(i.operation.name=='reset' for i in c.data[:len(resets)])
        a=c.num_qubits-1
        assert c.data[-1].operation.name=='measure' and c.find_bit(c.data[-1].qubits[0]).index==a
        ancillary=[i for i in c.data if any(c.find_bit(q).index==a for q in i.qubits)]
        tail=ancillary[-5:]
        assert [i.operation.name for i in tail]==['rz','rz','sx','rz','measure']
        assert float(tail[0].operation.params[0])==-float.fromhex(angles[name])
        # Isolated readout RZ survives; no cancellation or fusion into the
        # adjacent core/H angles in these sixteen concrete lowerings.
        baseline_arm='Bprime' if arm=='Bprime' else 'reference'
        for axis in 'XY':
            bp=result['baseline'][f'rare{rare}_{baseline_arm}_{axis}']['price']
            assert row['baseline_differences'][axis]=={k:price[k]-bp[k] for k in price}
        assert row['baseline_differences']['Y']=={k:0 for k in price}
        assert row['baseline_differences']['X']==dict(rz=1,cx=0,size=1,
            depth=0 if arm=='Bprime' else 1,qubits=0)
        checks.append(dict(context=stem,qpy_sha256=hashlib.sha256(payload).hexdigest(),
            readout_native_RZ=-float.fromhex(angles[name]),QPY_price_and_layout=True,
            direct_saved_coherence_agrees=True,readout_RZ_not_cancelled_or_fused=True))
assert not result['m17_feedback_340_or_allround_cost_qualified']
assert not result['phase_fit']
summary=dict(status='PASS',checks=checks,results_sha256=hashlib.sha256(raw).hexdigest(),
    native_archive_sha256=hashlib.sha256((out/'native_qpy.tar.xz').read_bytes()).hexdigest(),
    checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    max_raw_error=max(r['raw_error'] for r in rows),
    max_native_error=max(r['native_error'] for r in rows),
    old_input_and_executed_source_hashes=True,artificial_tests=4,
    metadata_clarification='cores.actual_inverse_SOURCE identifies Reference construction; Bprime uses its unchanged controlled builder',
    independent_scientific_review=False,new_simulation_or_compile=False)
target=out/'CHECKS.json'
assert not target.exists(),'Preserve completed checker record'
target.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
