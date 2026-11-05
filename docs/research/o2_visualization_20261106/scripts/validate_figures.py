"""Validate fixed-field display and output artifacts; no scientific arithmetic."""
import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[2]
BASE = '59f7083dacf6fdf25b6b593608d60c1e816846a3'
REL = 'docs/research/o2_visualization_20261106'
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1', PYTHONDONTWRITEBYTECODE='1')
NS = '{http://www.w3.org/2000/svg}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-checkout', type=Path,
                        help='Optional clean checkout of the fixed plan commit for its unchanged validator')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('catalog_metadata', ROOT / 'docs/research/o2_data_catalog_20261105/validate_catalog.py')
    catalog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalog)
    meta = json.loads((OUT / 'FIGURE_VALIDATION.json').read_text())
    assert meta['base_plan'] == BASE and meta['new_scientific_PASS'] is False
    assert all(value is False for value in meta['scope'].values())

    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], env=ENV)

    changes = git('diff', '--name-only', BASE).decode().splitlines()
    assert all(p.startswith(REL + '/') or p in {'PROJECT_MAP.md', 'docs/research/README.md'} for p in changes)
    protected = ['docs/research/o2_data_catalog_20261105', 'docs/research/o2_figure_planning_20261105',
                 'docs/research/pr_research_paper_evidence_synthesis_20261103', 'src', 'scripts',
                 'tests', 'artifacts', 'AGENTS.md', 'VALIDATION_STATUS.md']
    assert not git('diff', '--name-only', BASE, '--', *protected).strip()
    for directory in protected[:3]:
        for path in git('ls-tree', '-r', '--name-only', BASE, '--', directory).decode().splitlines():
            assert (ROOT / path).read_bytes() == git('show', BASE + ':' + path)
    if args.baseline_checkout:
        baseline = args.baseline_checkout.resolve()
        assert subprocess.check_output(['git', '-C', str(baseline), 'rev-parse', 'HEAD'], env=ENV).decode().strip() == BASE
        script = baseline / 'docs/research/o2_figure_planning_20261105/validate_figure_plan.py'
        completed = subprocess.run([sys.executable, str(script), '--repo', str(baseline)],
                                   env=ENV, check=True, stdout=subprocess.PIPE, text=True)
        assert json.loads(completed.stdout)['status'] == 'PASS_METADATA_AND_LITERAL_ONLY'

    refs = meta['fixed_refs']
    batch = subprocess.run(['git', '-C', str(ROOT), 'cat-file',
                            '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                           input=('\n'.join(e['commit']+':'+e['path'] for e in refs.values())+'\n').encode(),
                           stdout=subprocess.PIPE, check=True, env=ENV)
    lines = batch.stdout.decode().splitlines()
    assert len(lines) == len(refs)
    for e, line in zip(refs.values(), lines):
        assert line.split() == [e['blob'], 'blob', str(e['bytes'])]
        assert e['url'].endswith('/'+e['commit']+'/'+e['path'])
    cache = {}

    def raw(key):
        if key not in cache:
            e=refs[key]
            cache[key]=git('show', e['commit']+':'+e['path']).decode('utf-8-sig')
        return cache[key]

    def source(key, selector):
        if selector['type']=='csv_row':
            row=list(csv.DictReader(io.StringIO(raw(key))))[selector['data_row']-1]
            assert all(row[k]==v for k,v in selector.get('where',{}).items())
            return row
        assert selector['type']=='json_pointer'
        return catalog.pointer(json.loads(raw(key),parse_int=str,parse_float=str),selector['pointer'])

    def number(literal):
        return None if literal in {'','null','None','UNKNOWN','NOT_ACQUIRED','NOT_RECORDED'} else float(Fraction(literal))

    points=catalog.read_csv(OUT/'DISPLAY_DATA.csv')
    manifest=catalog.read_csv(OUT/'FIGURE_MANIFEST.csv')
    by_id={p['point_id']:p for p in points}
    assert len(points)==len(by_id)==meta['display_rows']
    assert {p['figure_id'] for p in points} == set(meta['figures'])-{'V07'}
    assert set(c for r in manifest for c in json.loads(r['figure_candidate_ids'])) == set(meta['target_candidates'])
    assert len(manifest)==meta['figure_count']==11
    literal_checks=0
    for p in points:
        r=source(p['ref_id'],json.loads(p['source_selector']))
        assert p['y_literal']==catalog.scalar(r[p['y_field']]),p['point_id']
        literal_checks+=1
        if p['x_field']:
            assert p['x_literal']==catalog.scalar(r[p['x_field']])
            literal_checks+=1
        if p['se_field']:
            assert p['se_literal']==catalog.scalar(r[p['se_field']])
            literal_checks+=1
        else:
            assert p['se_literal']==''
        if 'cohort' in r:
            assert p['source_cohort']==r['cohort']
        for binding in json.loads(p['metadata_literals']).get('saved_candidate_identity_bindings',[]):
            original=source(binding['ref_id'],binding['selector'])
            assert all(original[k]==v for k,v in binding['fields'].items())
            literal_checks+=len(binding['fields'])
        assert p['unit'] in {'RZ×shot','analytic quantum shots','compiled RZ/shot',
                             'saved higher/S2 ratio','dimensionless signal'}
    # Ensure the boundary display uses precisely the original catalog's fixed cohorts.
    old_candidates=catalog.read_csv(ROOT/'docs/research/o2_data_catalog_20261105/candidate_results.csv')
    for p in points:
        if p['figure_id']=='V09':
            original=next(r for r in old_candidates if r['record_kind']=='LITERAL_EXTRACT' and
                          r['ref_id']=='BOUNDARY_RESULTS' and r['candidate_id']==p['candidate_id'])
            assert p['source_cohort']==original['cohort_id']
    boundary=[p for p in points if p['figure_id']=='V09' and p['panel_id'].endswith('|G')]
    assert len(boundary)==5 and len({p['candidate_id'] for p in boundary})==5
    h6=next(p for p in boundary if p['x_literal']=='11')
    assert json.loads(h6['metadata_literals'])['diagnostic']=='INCONCLUSIVE'
    assert json.loads(h6['metadata_literals'])['point_lower']=='True'
    assert len([p for p in points if p['figure_id']=='V02' and p['series_id']=='PR_fixed_confirmation'])==33
    assert not any(p['se_field']=='difference_SE' for p in points)
    assert next(p for p in points if p['figure_id']=='V04' and p['series_id']=='PF')['se_literal']=='null'
    assert not any(p['figure_id']=='V04' and p['y_field'].startswith('mean_C') for p in points)
    assert all(p['candidate_id']=='PR_p0078_q8_r32_K2' for p in points
               if p['figure_id']=='V05' and p['panel_id'].startswith('H6|') and p['series_id'].startswith('Hybrid'))
    matching=list(csv.DictReader(io.StringIO(raw('VIS_MATCHING'))))
    assert next(r for r in matching if r['matching_id']=='H4_P3_common_hold')['comparison_class']=='CONDITIONAL_MATCHED'
    assert next(r for r in matching if r['matching_id']=='DF_B3_vs_Pauli_pure_RTE')['comparison_class']=='CONTEXTUAL'
    assert hashlib.sha256(raw('VIS_P3_PRICES').encode()).hexdigest()==meta['P3_price_input_sha256']

    data_hash=hashlib.sha256((OUT/'DISPLAY_DATA.csv').read_bytes()).hexdigest()
    svg_marks=0
    output_files={}
    required_labels={'V02':['33 PR-vs-PF confirmation','40 exploration'],
        'V04':['24.58909439%','sample SE is null'],
        'V05':['14/15','hypothetical compiled ρ=0','not 15 independent'],
        'V06':['CONDITIONAL_MATCHED','finite-bit law/angle gap = 0','compiled residual ρ = 0','B engineering-candidate coverage'],
        'V07':['CONTEXTUAL','One-body Gaussian block','Constant scalar only','not jointly matched'],
        'V09':['INCONCLUSIVE','zero-based linear axes'],
        'V10':['UNRESOLVED','saved 0'], 'V11':['not stacked','Δinteraction']}
    for entry in manifest:
        fid=entry['figure_id']
        svg=OUT/entry['svg_path'];png=OUT/entry['png_path']
        tree=ET.fromstring(svg.read_text())
        text=' '.join(''.join(node.itertext()) for node in tree.iter(NS+'text'))
        for phrase in required_labels.get(fid,[]):
            assert phrase in text,(fid,phrase)
        ids={node.attrib['id'] for node in tree.iter() if 'id' in node.attrib}
        metadata=next(node for node in tree.iter(NS+'metadata') if node.attrib.get('id')=='o2-saved-fields')
        rendered=json.loads(metadata.text)
        assert rendered['display_data_sha256']==data_hash
        assert rendered['new_numeric_analysis'] is rendered['new_scientific_PASS'] is False
        expected={p['point_id'] for p in points if p['figure_id']==fid}
        assert {r['point_id'] for r in rendered['rendered_fields']}==expected
        for mark in rendered['rendered_fields']:
            assert mark['point_id'] in ids,(fid,mark['point_id'])
            p=by_id[mark['point_id']]
            assert mark['y_literal']==p['y_literal'] and mark['se_literal']==p['se_literal']
            if mark['kind']!='saved_text_annotation':
                assert mark['y']==number(p['y_literal'])
                assert mark['se']==number(p['se_literal'])
            svg_marks+=1
        data=png.read_bytes()
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',data[16:24]);assert width>=1000 and height>=700
        assert entry['new_numeric_analysis']=='false' and entry['existing_figure_reused']=='false'
        assert entry['status']=='CREATED_FROM_SAVED_DATA'
        for path in [svg,png]:output_files[str(path.relative_to(OUT))]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    assert svg_marks==len(points)
    # The extraction layer cannot evaluate scientific values; the renderer only parses/draws.
    for name in ['extract_saved_display.py','render_saved_figures.py']:
        tree=ast.parse((OUT/'scripts'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any(m.split('.')[0] in {'qiskit','pyscf','trotterlib','scipy','numpy'} for m in modules)
            if isinstance(node,ast.Call):
                called=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
                assert called not in {'mean','average','std','var','cov','median','min','max','sum','sample','choice','seed','compile','simulate'},(name,called)
                if name=='extract_saved_display.py':assert called not in {'float','Fraction'}
    for path in OUT.rglob('*'):
        if path.is_file() and path.suffix in {'.md','.csv','.json','.py'}:
            assert not re.search('/'+r'(?:home|tmp|mnt|scratch|Users)/|gh[pousr]_[A-Za-z0-9]{20,}',path.read_text()),path
    result=dict(status='PASS_SAVED_FIELD_DISPLAY_ONLY',figures=len(manifest),candidate_ids=meta['target_candidates'],
                display_rows=len(points),literal_fields_checked=literal_checks,svg_marks_verified=svg_marks,
                fixed_refs=len(refs),original_claim_rows_unchanged=35,
                scientific_execution=False,numeric_reanalysis=False,new_scientific_PASS=False,
                output_files=output_files)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
