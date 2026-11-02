#!/usr/bin/env python3
"""Read saved new results/QPY and protected inputs; no compile/state action."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from fractions import Fraction as Q

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/pauli_hybrid_rte_comparison/2026-10-25'
PREP = ROOT/'artifacts/pauli_hybrid_rte_preparation/2026-10-25'


def read(p):
    return json.loads(Path(p).read_text())


def digest(b):
    return hashlib.sha256(b).hexdigest()


def csvrows(p):
    with Path(p).open() as f:
        return [{k:None if v in ('','null') else v for k,v in x.items()} for x in csv.DictReader(f)]


def main():
    from qiskit import qpy
    import io
    import qiskit
    assert qiskit.__version__=='1.3.0'
    assert read(OUT/'STUDY_STOP.json')['mandatory_STOP'] is True
    source=read(OUT/'SOURCE.json')
    for p,h in source['input_sha256'].items():assert digest((ROOT/p).read_bytes())==h,p
    for p,h in source['code_sha256'].items():assert digest((ROOT/p).read_bytes())==h,p
    for x in read(PREP/'source_manifest.json'):assert digest((ROOT/x['local_path']).read_bytes())==x['sha256'],x['key']
    saved=csvrows(PREP/'candidate_evidence_inventory.csv')
    nominations=read(OUT/'nominations.json')['candidates']
    schedule={(x['molecule'],x['candidate_id']):x['planned_confirmation_seeds']
              for x in read(OUT/'seed_schedule_verification.json')['candidates']}
    protected=('N_cos','N_sin','B_exact','B_saved_decimal','bias_cos_exact','bias_sin_exact',
               'u_cos_exact','u_sin_exact','eligibility','L_D','q','r','K')
    fresh=csvrows(OUT/'new_candidate_accounting.csv')
    for row in fresh:
        old=next(x for x in saved if (x['molecule'],x['candidate_id'],x['epsilon_sig'],x['cohort'])
                 ==(row['molecule'],row['candidate_id'],row['epsilon_sig'],'exploration'))
        assert all(old[k]==row[k] for k in protected),row['candidate_id']
        if row['G_exact'] is not None:
            assert Q(row['G_exact'])==int(row['N_cos'])*Q(row['C_cos_exact'])+int(row['N_sin'])*Q(row['C_sin_exact'])
    qpy_checked=0;count=0;raw_records={};observed_orders={}
    for candidate in nominations:
        m,cid=candidate['molecule'],candidate['candidate_id']
        files=sorted((OUT/m/'confirmation01/cases'/cid/'prices/confirmation').glob('*/result.json'))
        assert len(files) in (48,96,192,384)
        assert candidate['group']==('Hybrid' if m=='H4' else 'RTE')
        assert candidate['saved_rows'][0]['existing_confirmation_n'] is None
        raw_records[(m,cid)]=[read(p) for p in files]
        observed_orders[m+':'+cid]={str(k):sum(x['orders'].count(k) for x in raw_records[(m,cid)]) for k in (0,2,4)}
        for i,p in enumerate(files):
            x=read(p);assert x['index']==i and x['seed']==schedule[(m,cid)][i]
            assert x['qiskit_version']=='1.3.0'
            for axis in ('cosine','sine'):
                for name in ('input','compiled'):
                    compressed=(p.parent/(axis+'_'+name+'.qpy.gz')).read_bytes()
                    meta=x['axes'][axis][name+'_qpy'];assert digest(compressed)==meta['gzip_sha256']
                    payload=gzip.decompress(compressed);assert digest(payload)==meta['raw_sha256']
                    if name=='compiled':
                        circuit=qpy.load(io.BytesIO(payload))[0];ops=circuit.count_ops()
                        assert circuit.num_qubits=={'H4':9,'H6':13}[m]
                        assert ops.get('rz',0)==x['axes'][axis]['rz_count']
                        assert ops.get('cx',0)==x['axes'][axis]['cx_count']
                        assert set(ops)<=set(('rz','sx','x','cx','measure','barrier'))
                        qpy_checked+=1
            count+=1
    for row in fresh:
        raw=raw_records[(row['molecule'],row['candidate_id'])]
        for axis,short in (('cosine','cos'),('sine','sin')):
            assert Q(row['C_'+short+'_exact'])==Q(sum(x['axes'][axis]['rz_count'] for x in raw),len(raw))
    native=[r for p in (OUT/'stages').glob('n*/native.json') for r in read(p)['results']]
    assert len(native)==count
    assert all(r['exit_code']==0 and r['reaped'] for r in native)
    assert count<=2688 and qpy_checked==2*count<=5376
    table=csvrows(OUT/'comparison_by_precision.csv')
    assert len(table)==len({(r['molecule'],r['epsilon_sig']) for r in table})==15
    for row in table:
        for cohort in ('exploration','confirmation'):
            h,r=row[cohort+'_Hybrid_G_exact'],row[cohort+'_RTE_G_exact']
            ratio=row[cohort+'_Hybrid_over_RTE_exact']
            assert (Q(ratio)==Q(h)/Q(r)) if h is not None and r is not None else ratio is None
    result=dict(status='PASS',rows=15,new_trajectories=count,compiled_QPY_count_checked=qpy_checked,
                all_native_exit_zero_and_reaped=True,all_QPY_bytes_SHA_match=True,
                compiled_RZ_CX_and_wire_counts_match=True,seed_schedule_match=True,
                saved_N_B_bias_allowance_unchanged=True,old_source_input_evidence_unchanged=True,
                exact_mean_G_ratio_identities=True,observed_orders=observed_orders,
                scientific_state_actions=0,compile_or_sampling_in_verifier=0,
                verification_scope='Primary Codex checks of saved artifacts; not independent scientific review or certificate')
    (OUT/'completion_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
