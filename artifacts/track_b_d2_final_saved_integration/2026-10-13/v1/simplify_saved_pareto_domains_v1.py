"""Remove redundant saved scalar dominance boxes; no price or pair evaluation."""
from fractions import Fraction as F
import json
from pathlib import Path


def contains(outer,inner):
    # A deleted full-equality point could invalidate containment. Keep such
    # outer boxes conservatively. Inner point exclusions cannot enlarge it.
    if outer['exclude_all_coordinate_equality_point'] is not None:return False
    for k,a in outer['coordinate_box'].items():
        b=inner['coordinate_box'][k]
        if F(a['lo'])>F(b['lo']):return False
        if a['hi'] is not None and (b['hi'] is None or F(a['hi'])<F(b['hi'])):return False
    return True


def simplify(regions,labels):
    answer={};verified=0
    for target in labels:
        boxes=[v for v in regions['directed_dominance_regions'] if v['dominated']==target]
        keep=[]
        for i,v in enumerate(boxes):
            if any(contains(w,v) and (not contains(v,w) or j<i) for j,w in enumerate(boxes) if i!=j):continue
            keep.append(v)
        # Equality of unions: every kept box is an original region, and every
        # original region is retained or contained in an actually kept region.
        for v in boxes:
            assert v in keep or any(contains(w,v) for w in keep)
            verified+=1
        full=any(v['exclude_all_coordinate_equality_point'] is None and all(F(a['lo'])==0 and a['hi'] is None for a in v['coordinate_box'].values()) for v in keep)
        answer[target]={'front_always':not keep,'front_never':full,'nondominated_iff':'outside union of retained closed dominance regions (respect equality exclusions)',
                        'minimal_retained_dominance_regions':keep}
    return {'arms':answer,'original_region_containment_checks':verified,
            'full_union_exactly_preserved':True,'scope':'known quantum coordinates only; unknown classical excluded'}


if __name__=='__main__':
    d=Path(__file__).parent
    packet=json.loads((d/'saved_affine_rows_and_budget_metadata_v1.json').read_text())
    groups={'D2_r2':packet['D2_r2_boundary'],'D1r1_D2r2_joint12':packet['D1_r1']+packet['D2_r2_boundary']}
    result={}
    for name,rows in groups.items():
        labels=['r'+str(v['steps'])+':'+v['arm'] for v in rows]
        result[name]=simplify(json.loads((d/('pareto_'+name+'_v1.json')).read_text()),labels)
    with (d/'simplified_pareto_domains_v1.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({g:{k:len(v['minimal_retained_dominance_regions']) for k,v in a['arms'].items() if not v['front_never']} for g,a in result.items()}))
