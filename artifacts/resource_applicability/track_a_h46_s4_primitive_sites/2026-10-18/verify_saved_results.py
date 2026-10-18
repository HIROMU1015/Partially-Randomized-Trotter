"""Portable stdlib-only verification of saved evidence; no scientific calls."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
read=lambda name:json.loads((HERE/name).read_text())
sha=lambda data:hashlib.sha256(data).hexdigest()


def main():
    manifest=read('manifest.json')
    for name,receipt in manifest['files'].items():
        raw=(HERE/name).read_bytes()
        assert len(raw)==receipt['bytes'] and sha(raw)==receipt['sha256'],name
    binding=read('source_input_binding.json')
    source=binding['execution_SOURCE']
    for path,expected in binding['own_sources_sha256'].items():
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',f'{source}:{path}'])
        assert sha(raw)==expected,path
    terminal,native,ledger=(read(n) for n in ('terminal.json','native_exit.json','call_ledger_summary.json'))
    assert terminal['SOURCE']==native['SOURCE']==ledger['SOURCE']==source
    assert terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['first_stop'] is None
    assert ledger['primitive_evaluation_reserved']==ledger['primitive_evaluation_completed']==terminal['primitive_reservations']==66
    assert ledger['one_shot_dispatches']==1 and not ledger['automatic_retry']
    assert native['native_exit_code']==0 and native['wait_reap_completed'] and not native['same_child_still_live']
    assert not terminal['qiskit_imported'] and terminal['inputs_and_source_verified_after']
    for key in ('compile','new_trajectory','full_wrapper_build','full_theoretical_recount'):assert terminal[key]==0
    result=read('primitive_sites.json')
    for system,nblocks,q in (('H4',13,1),('H6',20,8)):
        record=result[system];assert len(record['blocks'])==nblocks
        net=dict(rz=0,rzz=0,phase=0)
        for b in record['blocks']:
            model,native=b['model'],b['native']
            assert model['nonzero_rzz_pairs']==native['nonzero_rzz_pairs']
            assert model['phase_nonzero']==native['phase_nonzero']
            observed=dict(rz=len(native['rz'])-len(model['rz']),rzz=0,phase=0)
            assert observed==b['cardinality_delta_native_minus_model']
            for key,value in observed.items():net[key]+=value
        assert net==record['net_per_outer_step']
        assert net==dict(rz=1 if system=='H4' else -1,rzz=0,phase=0)
        da=net['rz']*q
        assert record['inferred_full_saved_delta']==dict(raw=4*da,rz=2*da,cx=2*da,sx=0,x=0)
    rows=read('eight_row_reconciliation.json');assert len(rows)==8
    for row in rows:
        assert row['reconciled'] and row['new_local_delta']==row['native']-row['theory']
        assert row['new_local_delta']==result[row['system']]['inferred_full_saved_delta'][row['metric'].lower()]
    changed=read('changed_sites.json');assert len(changed)==4
    assert {(r['system'],r['fragment_index'],r['qubit']) for r in changed}=={('H4',1,0),('H4',1,1),('H4',1,7),('H6',5,11)}
    print('PASS: hashes, fixed SOURCE, 66 calls, 33 blocks, 4 changed sites, 8 reconciled rows, native exit/reap; no scientific calls')


if __name__=='__main__':main()
