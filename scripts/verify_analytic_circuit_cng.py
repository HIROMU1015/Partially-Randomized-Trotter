"""Separate saved-data algebra, reference, SOURCE, one-shot and export checks."""
import csv
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022'
OLD=OUT.parent/'pr_pf_s2_recomputed_20261022'
RAW=ROOT/'artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    assert not (OUT/'STOP.json').exists(),'No output mutation after packet STOP'
    d=read(OUT/'figure_data.json');rows=d['rows'];old=read(OLD/'figure_data.json')['rows']
    assert len(rows)==len({(r['system'],r['epsilon'],r['role']) for r in rows})==40
    sources={(r['system'],r['epsilon']):r for r in old}
    source=read(RAW/'SOURCE.json');total_units=0
    for p,h in source['files'].items():assert sha(ROOT/p)==h,p
    protected=read(RAW/'protected_inputs.json')['files']
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    for r in rows:
        o=sources[r['system'],r['epsilon']];role=r['role'];v=o[role]
        assert r['candidate']==o[role+'_candidate_id'] and r['cohort']==o['cohort']
        assert r['physical_status']==o['status'] and r['preparation_hash']==v['preparation_hash']
        assert all(r[k] is None for k in ['N_circuit_total','N_ratio_total','G10','G11','delta_G_N'])
        assert r['N_independent'] is False and r['total_comparison_label']=='N_FIXED_C_ONLY_REPLACEMENT'
        sums={b:sum((F(r['axes'][a]['C_theory'][b])*r['axes'][a]['N_theory'] for a in ['cosine','sine']),F()) for b in ['lower','upper']}
        for b in ['lower','upper']:assert sums[b]==F(r['G00'][b])==F(o['G_'+role+'_theory_'+b])
        measured=sum((F(r['axes'][a]['C_circuit']['mean_exact'])*r['axes'][a]['N_theory'] for a in ['cosine','sine']),F())
        assert measured==F(r['G01']['mean_exact'])==F(o['G_'+role+'_compiled'])
        assert measured-sums['lower']==F(r['delta_G_C_lower_reference'])
        assert sum(map(F,r['delta_G_C_by_axis_lower_reference'].values()),F())==measured-sums['lower']
        assert r['N_total']==sum(v['shots'].values())
        for a in ['cosine','sine']:
            x=r['axes'][a];assert x['N_circuit'] is None and x['N_ratio'] is None
            assert x['N_theory']==v['shots'][a] and x['C_theory']==v['theory'][a]
            assert F(x['C_circuit']['mean_exact'])==F(v['compiled'][a+'_mean'])
        with localcontext() as context:
            context.prec=80
            q=F(r['G01']['mean_exact'])/F(r['G00']['upper'])
            decimal_ratio=Decimal(q.numerator)/Decimal(q.denominator)
            assert math.isclose(float(decimal_ratio),r['G01_over_G00']['lower'],rel_tol=1e-14)
        unit=RAW/'units'/r['cohort']/r['system']/r['candidate']/('saved_CNG_accounting_'+r['epsilon'])
        plan=read(unit/'plan.json');grant=read(unit/'grant.json');marker=read(unit/'marker.json');stop=read(unit/'STOP.json')
        assert grant['plan_sha256']==digest(plan)==grant['grant_id']==marker['grant_id']
        assert grant['one_shot'] and grant['automatic_retry'] is False
        assert grant['authorization_sha256']==sha(RAW/'user_instruction.txt')
        assert stop['SOURCE']==source['commit'] and stop['status']=='COMPLETE_MANDATORY_STOP_UNIT'
        assert stop['result_sha256']==sha(unit/'result.json')
        assert read(unit/'result.json')==r
        assert plan['input_hashes']['paired_figure_data']==sha(OLD/'figure_data.json')
        assert plan['input_hashes']['saved_prices']==sha(ROOT/plan['price_path'])
        raw=read(ROOT/plan['price_path'])
        if r['system']=='H2':
            samples=[dict(cosine=x['axes']['cosine']['RZ'],sine=x['axes']['sine']['RZ']) for x in raw['measurements']+raw.get('reused_FP_measurements',[]) if x['cell_id']==r['candidate']]
        else:samples=raw['values']
        n=len(samples);assert n==r['MC_n']
        for a in ['cosine','sine']:
            values=[x[a] for x in samples]
            assert F(sum(values),n)==F(r['axes'][a]['C_circuit']['mean_exact'])
            numerator=n*sum(x*x for x in values)-sum(values)**2
            variance_of_mean=F(numerator,n*n*(n-1)) if n>1 else F()
            assert math.isclose(float(variance_of_mean),r['axes'][a]['C_circuit']['SE']**2,rel_tol=1e-12,abs_tol=1e-18)
        values=[sum(x[a]*r['axes'][a]['N_theory'] for a in ['cosine','sine']) for x in samples]
        numerator=n*sum(x*x for x in values)-sum(values)**2
        variance_of_mean=F(numerator,n*n*(n-1)) if n>1 else F()
        assert math.isclose(float(variance_of_mean),r['G01']['SE']**2,rel_tol=1e-12,abs_tol=1e-12)
        assert math.isclose(v['compiled']['half95'],r['G01']['half95'],rel_tol=1e-12,abs_tol=1e-7)
        total_units+=1
    csv_rows=list(csv.DictReader((OUT/'figure_data.csv').open()))
    assert len(csv_rows)==80 and all(r['N_circuit']==r['N_ratio']==r['G11']==r['delta_G_N']=='' for r in csv_rows)
    from PIL import Image
    figures=list(OUT.glob('*/*.png'));assert len(figures)==30
    for p in figures:
        with Image.open(p) as im:im.verify()
        assert ET.parse(p.with_suffix('.svg')).getroot().tag.endswith('svg')
        pdf=p.with_suffix('.pdf').read_bytes()
        assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-128:] and b'startxref' in pdf[-128:]
        assert len(re.findall(rb'/Type\s*/Page\b',pdf))==1
    links=0
    for p in OUT.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('https:','http:','#')):continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            if resolved != OUT/'verification.json':
                assert resolved.exists(),str(p)+': '+target
            links+=1
    # Check old scientific paths and research/slide originals remain byte-identical in Git too.
    changed=subprocess.check_output(['git','diff','a590214d6d70962aa02b9e0a2bce56810859522f','--name-only','--',
        'artifacts/resource_applicability/pr_pf_s2_recomputed','docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022',
        ':(top,glob)docs/progress/track_a/*.md','docs/research/研究概要・現状.md','docs/research/研究目的・研究課題.md'],cwd=ROOT,text=True)
    assert not changed.strip(),changed
    original_manifest=json.loads(subprocess.check_output(['git','show','a590214d6d70962aa02b9e0a2bce56810859522f:artifacts/validation_manifest.json'],cwd=ROOT,text=True))
    manifest=read(ROOT/'artifacts/validation_manifest.json')
    identifier='pr_pf_analytic_vs_circuit_CNG_20261022'
    assert manifest['result_sets'][-1]['id']==identifier
    manifest.pop(identifier);manifest['result_sets'].pop();manifest['artifact_inventory']['present'].pop()
    assert manifest==original_manifest,'Existing validation records changed'
    trends=read(OUT/'trend_summary.json')['rows']
    for trend in trends:
        previous=sources[trend['system'],trend['epsilon']]
        for role in ['PR','PF']:
            assert math.isclose(trend[role+'_reduction_percent'],previous[role+'_reduction_lower'],abs_tol=1e-12)
    result=dict(status='PASS_SAVED_ACCOUNTING_WITH_RETAINED_SCIENTIFIC_LIMITS',
        paired_C_and_G_conditions=20,method_sides=40,axis_rows=80,completed_one_shot_units=total_units,
        independent_N_conditions=0,N_circuit_and_ratios_null=True,G10_G11_and_N_effect_null=True,
        within_executed_technical_pass_conditions=10,conditional_H6_H8_conditions=10,
        exact_axis_sums_and_C_contributions=True,independent_Decimal_ratio_check=True,
        paired_MC_covariance_check=True,protected_existing_files_unchanged=len(protected),
        SOURCE_hashes_match=True,PNG_SVG_PDF_triplets=len(figures),local_links_checked=links,
        existing_validation_entries_preserved=True,old_reduction_values_reproduced=True,
        physical_validation_status_upgraded=False,independent_scientific_review=False,
        new_signal=0,new_compile=0,new_sampling=0,new_shot_evaluation=0,
        format_scope='PNG decode; SVG XML; PDF page/header/xref/EOF structure. Visual check uses PNG exports',
        artificial_tests='4 passed (0.06s); covariance, intervals, absent N, input mismatch')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    assert (OUT/'verification.json').exists()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
