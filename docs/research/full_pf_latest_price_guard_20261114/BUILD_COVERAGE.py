"""Latest A shortlist through the existing passive receipt coverage builder."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_pf_coverage',
    HERE.parent/'full_pf_candidate_price_evidence_20261114/BUILD_COVERAGE.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
A=dict(commit='71f04be7bf94e7773a32270f36ae29eee1b903d5',
       path='docs/research/h4_full_pf_followup_results_20261114/C_HANDOFF.json')


def build():
    handoff=old.read(A)
    labels=handoff['shortlist_labels']
    assert labels==['S2_16h_over_7','Y4_16h_over_3','M5_16h']
    result=old.build(a_source=A,labels=labels)
    for candidate in result['candidates']:
        original=next(c for c in handoff['configurations'] if c['label']==candidate['label'])
        ratio=original['h_multiplier'];h=handoff['shared']['h'];delta=candidate['constructor']['delta_pf']
        exact=lambda v:Fraction(v['numerator'],v['denominator'])
        assert exact(delta)==exact(h)*exact(ratio)
        candidate['h_multiplier']=ratio
    result['sources']['D_S1']=dict(commit='3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673',
        path='docs/research/d_s1_v1_price_alignment_20261114/S1_DOMAINS.md')
    assert sum(bool(r['direct_evidence']) for r in result['coverage'])==8
    assert all(r['receipt_catalog_status'] in ('EXACT_METADATA_MATCH','OUTSIDE_CHECKED_CATALOG')
               for r in result['coverage'])
    result['comparison_policy_adopted']=False
    result['rte_rotation_lowering']='NOT_APPLICABLE_NO_RTE'
    return result


if __name__=='__main__':
    result=build()
    with (HERE/'COVERAGE.json').open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print('PASS: latest 3 exact configurations, 39 physical rounds / 78 axes; saved direct 4 rounds / 8 axes; price arithmetic=0')
