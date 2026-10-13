"""Recheck published scalar-phase sensitivity without any private runtime or NPZ."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45));signal.alarm(60)
    bundle=Path(__file__).resolve().parent
    pub=bundle.parent.parent/'track_a_h4_geometry_results/2026-10-13'
    manifest=json.loads((bundle/'artifact_manifest_v1.json').read_text())
    for name,e in manifest['files'].items():
        raw=(bundle/name).read_bytes()
        assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256'],name
    source=json.loads((pub/'saved_candidates_v1.json').read_text())
    expected=json.loads((bundle/'diagnostic_summary_v1.json').read_text())
    spec=importlib.util.spec_from_file_location('cost_arithmetic',pub/'verify_saved_result_projection.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    lookup={};b2={}
    for r in source['candidates']:
        r['_statistics']=api.statistics(r['saved']['paired_costs'])
        lookup[(r['geometry'],r['template_id'])]=r
        if r['template_id'].startswith('B2-'):b2.setdefault(r['geometry'],[]).append(r)
    scalars={r['geometry']:r for r in json.loads((bundle/'tail_scalar_metadata_v1.json').read_text())}
    aux={};phase_count=0
    for r in json.loads((bundle/'saved_b0_phase_sensitivity_v1.json').read_text()):
        g=r['geometry'];old=lookup[(g,r['original_template_id'])]
        c=scalars[g]
        tail=math.fsum(c['DF_fragment_identity_coefficients'][r['prefix']:])
        assert math.isclose(tail,r['tail_identity'],abs_tol=1e-14)
        target=complex(math.cos(-c['T']*c['energy']),math.sin(-c['T']*c['energy']))
        original=complex(*r['saved_corrected_signal'])
        result=original*complex(math.cos(-c['T']*tail),math.sin(-c['T']*tail))
        assert abs(result-complex(*r['scalar_phase_counterfactual']))<1e-14
        bias={'cosine':abs(result.real-target.real),'sine':abs(result.imag-target.imag)}
        for a in bias:assert math.isclose(bias[a],r['counterfactual_bias'][a],abs_tol=1e-14)
        candidate={'template_id':r['original_template_id']+'+I',
                   'saved':{'bias':bias,'normalization':1.,'paired_costs':old['saved']['paired_costs']},
                   '_statistics':old['_statistics']}
        aux.setdefault(g,[]).append(candidate);phase_count+=1
    assert phase_count==120
    def best(rows,e):
        points=[(r,api.evaluate(r,e)) for r in rows]
        return min(((r,p) for r,p in points if p is not None),key=lambda x:x[1]['work'],default=None)
    with (bundle/'scalar_reference_envelope_1812.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1812
    reversals=0
    for r in rows:
        g,e=r['geometry'],float(r['epsilon'])
        rb,pb=best(b2[g],e);ri=best(aux[g],e)
        assert rb['template_id']==r['B2_template']
        api.near(pb['work'],float(r['B2_G']))
        if ri is None:assert r['B0_I_reference_template']=='';continue
        candidate,p=ri
        assert candidate['template_id']==r['B0_I_reference_template']
        api.near(p['work'],float(r['B0_I_reference_G_zero_extra']))
        api.near((pb['work']-p['work'])/sum(p['shots']),float(r['common_extra_RZ_per_shot_break_even']))
        reversals+=p['work']<pb['work']
    assert reversals==expected['zero_extra_reference_reversals']==397
    assert expected['B0_I_actual_compiled_cost']=='UNMEASURED'
    assert expected['status']=='SCIENTIFIC_DECISION_STOP_B0_IDENTITY_SENSITIVITY'
    print(json.dumps({'status':'PASS_PUBLIC_SCALAR_SENSITIVITY_ARITHMETIC','phase_cases':120,
                      'grid_rows':1812,'zero_extra_reference_reversals':reversals,
                      'private_runtime_NPZ_quantum_science_compile_access':0,
                      'actual_B0_I_cost_measured':False}))


if __name__=='__main__':main()
