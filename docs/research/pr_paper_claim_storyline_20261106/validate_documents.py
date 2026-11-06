"""Document/reference/literal checks only. No scientific arithmetic or runners."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
OLD='3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd'
CAT='2779710a86d380bb484accc51d3dccba183370b8'
VIZ='b3b194dbf405285239326fd25b671d30957b801c'
REAN='29c3bf0ecc687d458df55cf967ff38f990130b81'
SP='docs/research/pr_research_paper_evidence_synthesis_20261103/'
CP='docs/research/o2_data_catalog_20261105/'
VP='docs/research/o2_visualization_20261106/'
RP='docs/research/o2_b1_b2a_saved_reanalysis_20261106/'


def reused_functions(path,names):
    tree=ast.parse(path.read_bytes())
    nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    scope=dict(csv=csv)
    import io
    scope['io']=io
    exec(compile(ast.Module(body=[nodes[n] for n in names],type_ignores=[]),str(path),'exec'),scope)
    return scope


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    a=p.parse_args();root=a.inputs
    index=json.loads((HERE/'SOURCE_INDEX.json').read_text())['references']
    for r in index:
        data=(root/r['commit']/r['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['blob']
    # The original validators hard-code their historical output/schema. Reuse
    # pure helpers, never run their main functions or data/figure generators.
    table=reused_functions(root/OLD/SP/'validate_synthesis.py',['table'])['table']
    helper=reused_functions(root/VIZ/VP/'scripts/extract_saved_display.py',['pointer','scalar'])
    _,old=table((root/OLD/SP/'CLAIM_EVIDENCE_MAP.csv').read_bytes())
    _,new=table((HERE/'CLAIM_EVIDENCE_LIMITATION.csv').read_bytes())
    by_component={r['component_id']:r for r in old}
    studies={r['study_id'] for r in csv.DictReader((root/CAT/CP/'study_catalog.csv').read_text().splitlines())}
    comparisons={r['comparison_id'] for r in csv.DictReader((root/CAT/CP/'comparison_catalog.csv').read_text().splitlines())}
    assert len({r['paper_row_id'] for r in new})==len(new)
    for r in new:
        components=r['existing_component_ids'].split(';')
        assert all(c in by_component for c in components)
        assert set(r['existing_claim_ids'].split(';'))=={by_component[c]['claim_id'] for c in components}
        assert set(r['question_ids'].split(';'))<=set(['Q1','Q2','Q3','Q4','Q5','Q6'])
        assert r['formal_status']=='NOT_ADOPTED_EDITORIAL_MAPPING;CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT_UNCHANGED'
        for c in r['comparison_ids'].split(';'):
            assert c in comparisons or c=='NOT_APPLICABLE_BOUNDARY_NOTE'
        for s in r['study_ids'].split(';'):
            assert s in studies or s=='SEE_R02_ORIGINAL_REFERENCES'
    narrative=(HERE/'NARRATIVE_OPTIONS.md').read_text()
    for option in ['S1','S2','S3']:
        for num in ['1','2','3','4','5','6']:
            block=narrative.split('### '+option+'-'+num+'. ',1)[1].split('\n### ',1)[0].split('\n## ',1)[0]
            for field in ['目的','問い','方法・方式','直接証拠','図表候補','先行関係','限定','次節への接続']:
                assert '**'+field+'**' in block
    figure=(HERE/'FIGURE_CLAIM_MAP.md').read_text()
    for f in csv.DictReader((root/VIZ/VP/'FIGURE_MANIFEST.csv').read_text().splitlines()):
        assert '## '+f['figure_id']+'：' in figure
        assert ET.fromstring((root/VIZ/VP/f['svg_path']).read_bytes()).tag.endswith('svg')
    # Check quoted numeric strings, not their arithmetic meaning or new counts.
    quote_specs=[('155529e20a8fac39e00786c40b57830b71203c4f','REPORT.md','33/40'),('155529e20a8fac39e00786c40b57830b71203c4f','docs/research/track_a_h246_pf_common_final_20261022.md','+40.9592%'),('155529e20a8fac39e00786c40b57830b71203c4f','docs/research/track_a_h246_pf_common_final_20261022.md','-53.8932%'),('d3fdfad880902b7ff35b710d22e1c9759f7087c2','docs/non_df_h4_pauli_pr_search_20261025.md','48.42%'),('d3fdfad880902b7ff35b710d22e1c9759f7087c2','docs/non_df_h4_pauli_pr_search_20261025.md','95.27%'),('eb07156a37b1938ed3aed95c13abbb545466c573','docs/research/h8_stage_c_execution_20261105/REPORT.md','24.58909439%'),(REAN,RP+'REPORT.md','39/40'),(REAN,RP+'REPORT.md','14/15'),(REAN,RP+'REPORT.md','432/432')]
    contribution=(HERE/'DF_PAULI_CONTRIBUTION.md').read_text()
    prior=(HERE/'PRIOR_ART_POSITIONING.md').read_text()
    for sha,path,literal in quote_specs:
        assert literal in (root/sha/path).read_text(),(path,literal)
        assert literal in contribution+prior,(literal,'destination')
    h4=list(csv.DictReader((root/'d3fdfad880902b7ff35b710d22e1c9759f7087c2'/'artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv').read_text().splitlines()))
    assert h4[0]['selected_PR']=='p036_q2_r8_K2' and h4[0]['confirmation_n']=='48'
    assert h4[-1]['selected_PR']=='p000_q8_r8_K4' and h4[-1]['confirmation_n']=='384'
    frozen=json.loads((root/REAN/RP/'B1_FIXED_COMPARISONS.json').read_text(),parse_int=str,parse_float=str)
    h4latest=next(r for r in frozen['comparisons'] if r['study']=='PAULI_LATEST_15' and r['system']=='H4' and r['epsilon']=='0.05')
    assert h4latest['arms']['Hybrid']['candidate_id']=='p036_q2_r8_K4'
    assert h4latest['arms']['Hybrid']['cohort']=='confirmation'
    assert helper['pointer']({'a/b':['saved']},'/a~1b/0')=='saved'
    assert helper['scalar'](None)=='null'
    # Check local and reference-style links; fixed remote URLs retain full SHAs.
    for path in HERE.glob('*.md'):
        text=path.read_text()
        definitions=set(re.findall(r'^\[([^]]+)\]:',text,re.M))
        for label in re.findall(r'\]\[([^]]+)\]',text):assert label in definitions,(path.name,label)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http://','https://','#')):continue
            assert (HERE/target.split('#')[0]).exists(),(path.name,target)
        for sha in re.findall(r'github\.com/HIROMU1015/Partially-Randomized-Trotter/blob/([^/]+)/',text):assert re.fullmatch('[0-9a-f]{40}',sha)
        assert all(line==line.rstrip() for line in text.splitlines()),path.name
    assert not list(HERE.glob('*.svg')) and not list(HERE.glob('*.png'))
    print(json.dumps(dict(status='DOCUMENT_CHECKS_PASS',fixed_public_bytes_checked=len(index),editorial_claim_rows=len(new),old_claim_components_preserved=len(old),figures_referenced=11,quote_literal_checks=len(quote_specs),section_blocks_with_eight_fields=18,scientific_execution=False,new_numeric_results=False,formal_adoption=False),ensure_ascii=False))


if __name__=='__main__':
    main()
