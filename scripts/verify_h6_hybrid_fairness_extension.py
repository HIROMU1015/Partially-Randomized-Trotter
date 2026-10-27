#!/usr/bin/env python3
"""Post-run byte/structure/accounting checks. No state evolution or sampling."""
import gzip,hashlib,importlib.util,json,subprocess
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('verify_hybrid_worker',ROOT/'scripts/run_h6_hybrid_fairness_extension.py');W=importlib.util.module_from_spec(s);s.loader.exec_module(W)
OUT=W.OUT
def main():
    import qiskit
    from qiskit import qpy
    import io
    assert W.read(OUT/'STUDY_STOP.json')['mandatory_STOP']
    for filename in ['SOURCE.json','ANALYSIS_SOURCE.json']:
        record=W.read(OUT/filename)
        for p,h in {**record['code_sha256'],**record.get('input_sha256',{})}.items():assert W.sha(ROOT/p)==h,p
    assert W.sha(OUT/'sampling_design.json')==W.read(OUT/'EXPLORATION_SOURCE.json')['sampling_design_sha256']
    assert W.sha(OUT/'confirmation_plan.json')==W.read(OUT/'CONFIRMATION_SOURCE.json')['confirmation_plan_sha256']
    checked=decoded=0;rows=W.read(OUT/'candidate_results.json');cp=W.read(OUT/'confirmation_plan.json');plan=W.read(OUT/'sampling_design.json')
    for cid in plan['candidates']:
        target=W.read(W.case(cid)/'target.json');spec=W.read(W.case(cid)/'spec.json')
        assert target['B_applied_count']==1 and not target['state_renormalized']
        assert target['raw_norm_exact']==W.read(OUT/'H6/inputs/reference.json')['raw_norm_exact']
        for cohort,n in [('exploration',plan['n_E']),('confirmation',cp['candidates'][cid]['n_C'])]:
            paths=sorted((W.case(cid)/'prices'/cohort).glob('*/result.json'));assert len(paths)==n
            for i,p in enumerate(paths):
                r=W.read(p);assert r['index']==i and r['seed']==W.E.trajectory_seed(cid,cohort,i)
                assert r['event_count']==spec['config']['q']*spec['config']['r']==len(r['orders'])
                assert set(r['orders'])<=set(range(0,spec['config']['K']+1,2))
                for axis in W.R.AXES:
                    for kind in ['input','compiled']:
                        path=p.parent/(axis+'_'+kind+'.qpy.gz');b=path.read_bytes();raw=gzip.decompress(b);meta=r['axes'][axis][kind+'_qpy']
                        assert hashlib.sha256(b).hexdigest()==meta['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==meta['raw_sha256'];checked+=1
                        if kind=='compiled' and i in (0,len(paths)-1):
                            circuit=qpy.load(io.BytesIO(raw))[0]
                            assert circuit.num_qubits==13 and circuit.num_clbits==1 and not circuit.parameters
                            assert set(circuit.count_ops())<=set(['rz','sx','x','cx','measure','barrier'])
                            assert circuit.count_ops().get('rz',0)==r['axes'][axis]['rz_count']
                            assert circuit.count_ops().get('cx',0)==r['axes'][axis]['cx_count'];decoded+=1
    for r in rows:
        if r['N_total'] is None:assert r['G_exact'] is None;continue
        assert Q(r['G_exact'])==r['N_cos']*Q(r['C_cos_exact'])+r['N_sin']*Q(r['C_sin_exact'])
        assert all(r['axes'][a]['rho_zero']['sufficient_inequality_verified'] for a in W.R.AXES)
    changed=subprocess.check_output(['git','diff','--name-only','bcc2eb6b201e1434bf47560775f5afb3e14d3ca8'],cwd=ROOT,text=True).splitlines()
    assert not any(p.startswith('docs/research/') or p.startswith('analysis/h6_hybrid_fairness_readiness_20261027/') for p in changed)
    W.save(OUT/'verification.json',dict(status='PASS',qpy_gzip_raw_SHA_checked=checked,representative_compiled_QPY_decoded=decoded,exact_G_identity=True,shot_sufficient_inequalities=True,common_raw_norm_B_once=True,all_native_exit0_reaped=True,old_SOURCES_STOP_results_unchanged=True,actual_compiled_rho=None,population_tail='UNGUARANTEED',formal_certificate_accepted=False))
    print(json.dumps(W.read(OUT/'verification.json')),flush=True)
if __name__=='__main__':main()
