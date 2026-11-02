"""Independent Decimal/rational accounting checks on saved results; no science calls."""
from decimal import Decimal,localcontext
from fractions import Fraction as F
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22'
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def archived_sha(p):
    p=Path(p)
    if p.exists():return sha(p)
    compressed=p.with_suffix(p.suffix+'.gz')
    assert p.name=='context.pkl' and compressed.exists()
    h=hashlib.sha256()
    with gzip.open(compressed,'rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def decimal(x):
    f=F(x);return Decimal(f.numerator)/Decimal(f.denominator)


def verify():
    data=read(OUT/'figure_data.json');rows=data['rows']
    old=read(ROOT/'docs/progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022/figure_data.json')['rows']
    expected={(x['system'],x['epsilon']):x for x in old}
    assert len(rows)==len({(x['system'],x['epsilon']) for x in rows})==20
    numeric=0;identities=0
    with localcontext() as context:
        context.prec=90
        for r in rows:
            source=expected[r['system'],r['epsilon']]
            assert all(r[k]==source[k] for k in ['PR_candidate_id','PF_candidate_id','PR_L_D','PF_L_D','PR_q','PF_q','PR_r','PR_K'])
            for role in ['PR','PF']:
                v=r[role]
                if v['theory'] is not None and v['shots'] is not None:
                    for bound in ['lower','upper']:
                        total=sum((F(v['theory'][axis][bound])*v['shots'][axis] for axis in ['cosine','sine']),F())
                        assert total==F(r['G_'+role+'_theory_'+bound])
                if 'G_'+role+'_compiled' in r:
                    total=F(v['compiled']['cosine_mean'])*v['shots']['cosine']+F(v['compiled']['sine_mean'])*v['shots']['sine']
                    assert total==F(r['G_'+role+'_compiled'])
            if 'rho_compiled' not in r:continue
            numeric+=1
            pr,pf=decimal(r['G_PR_compiled']),decimal(r['G_PF_compiled'])
            assert math.isclose(float(pr/pf),r['rho_compiled'],rel_tol=1e-14)
            assert F(r['rho_compiled_exact'])==F(r['G_PR_compiled'])/F(r['G_PF_compiled'])
            for role in ['PR','PF']:
                for bound in ['lower','upper']:
                    theory=decimal(r['G_'+role+'_theory_'+bound]);compiled=decimal(r['G_'+role+'_compiled'])
                    decrease=(theory-compiled)*100/theory
                    assert math.isclose(float(decrease),r[role+'_reduction_'+bound],rel_tol=1e-13,abs_tol=1e-12)
            if all(r['G_'+role+'_theory_lower']==r['G_'+role+'_theory_upper'] for role in ['PR','PF']):
                gamma_pr=F(r['G_PR_compiled'])/F(r['G_PR_theory_lower'])
                gamma_pf=F(r['G_PF_compiled'])/F(r['G_PF_theory_lower'])
                theory_ratio=F(r['G_PR_theory_lower'])/F(r['G_PF_theory_lower'])
                assert F(r['rho_compiled_exact'])/theory_ratio==gamma_pr/gamma_pf
                identities+=1
    units=[];seen=set();active=[]
    for marker in sorted((BASE/'units').rglob('marker.json')):
        p=marker.parent;plan=read(p/'plan.json');grant=read(p/'grant.json')
        key=tuple(plan.get(k) for k in ['cohort','system','candidate','stage','epsilon'])
        assert key not in seen;seen.add(key)
        assert grant['one_shot'] and grant['automatic_retry'] is False
        assert grant['plan_sha256']==digest(plan) and read(marker)['grant_id']==grant['grant_id']
        assert grant['authorization_sha256']==sha(BASE/'user_authorization.md')
        if 'context' in plan['input_hashes']:
            expected_context=read(BASE/'units'/plan['system']/'SYSTEM/input_preparation/result.json')['context_sha256']
            assert plan['input_hashes']['context']==expected_context
        if not (p/'STOP.json').exists():active.append(str(p.relative_to(ROOT)));continue
        stop=read(p/'STOP.json');assert stop['SOURCE']==plan['SOURCE']
        if stop['status']=='COMPLETE_MANDATORY_STOP_UNIT':
            for name,h in stop['result_files'].items():assert archived_sha(p/name)==h
        units.append(dict(path=str(p.relative_to(ROOT)),status=stop['status']))
    seeds=set();samples=0;main_groups=0;target_groups=[]
    candidates={(c['system'],c['id']):c for c in read(BASE/'candidates.json')}
    for p in sorted((BASE/'units').glob('*/*/MC_*/result.json')):
        system,cid,stage=p.parts[-4:-1];c=candidates[system,cid];result=read(p)
        assert len(result['values'])==result['n'] and sorted(x['index'] for x in result['values'])==list(range(result['n']))
        assert result['n']>=128
        if stage=='MC_pilot':assert result['n']==128
        if stage=='MC_main':
            plan=read(BASE/'main_plans'/system/(cid+'.json'));assert result['n']==plan['n'] and result['pilot_pooled'] is False
            assert result['automatic_additional_sampling'] is False
            main_groups+=1;target_groups.append(all(x['target_met'] for x in result['statistics'].values()))
        for value in result['values']:
            assert len(value['orders'])==c['q']*c['r'] and all(k in range(0,c['K']+1,2) for k in value['orders'])
            for outer in range(c['q']):
                text='/'.join(map(str,('FIXED_CANDIDATES_RECOMPUTED_NATIVE_20261022',system,cid,stage,value['index'],outer)))
                key=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
                assert key not in seeds;seeds.add(key)
            samples+=1
    for file in ['SOURCE.json','SOURCE_v2.json','SOURCE_physics_validation.json']:
        for p,h in read(BASE/file)['files'].items():assert sha(ROOT/p)==h
    changed=subprocess.check_output(['git','diff','2d4ff27cb630939f87e08ab5881404c72ad2fa19','--name-only','--',
        'artifacts/resource_applicability/track_a_current_s2_theory_compiled',
        'docs/progress/track_a/figures/pr_pf_s2_theory_compiled_main_20261022',
        'src/trotterlib/current_s2_theory.py'],cwd=ROOT,text=True)
    assert not changed.strip()
    figures=list(OUT.glob('*/*.png'));assert len(figures)==30
    from PIL import Image
    for p in figures:
        with Image.open(p) as image:image.verify()
        assert p.with_suffix('.svg').stat().st_size>1000
        assert ET.parse(p.with_suffix('.svg')).getroot().tag.endswith('svg')
        pdf=p.with_suffix('.pdf').read_bytes()
        assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-128:]
        assert len(re.findall(rb'/Type\s*/Page\b',pdf))==1 and b'startxref' in pdf[-128:]
    result=dict(status='ACCOUNTING_PASS_WITH_EXPLICIT_PHYSICS_CAVEATS',numerical_pairs=numeric,
        point_theory_ratio_reduction_identities=identities,scientific_units=len(units),
        failed_units=[x for x in units if x['status']=='FAILED_NO_RETRY'],active_units=active,
        new_main_groups=main_groups,new_main_groups_meeting_design_goal=sum(target_groups),
        completed_pilot_and_main_trajectories=samples,unique_actual_occurrence_seeds=len(seeds),
        original_sources_preserved=True,frozen_science_SOURCE_hashes_PASS=True,
        PNG_SVG_PDF_triplets_verified=len(figures),independent_scientific_review=False,
        format_check_scope='PNG decoding, SVG XML, PDF header/page object/xref/EOF structure; visual inspection uses PNG exports',
        failed_units_retried=False,fully_validated_pairs=data['complete_pairs'],conditional_pairs=data['conditional_pairs'])
    (OUT/'numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
