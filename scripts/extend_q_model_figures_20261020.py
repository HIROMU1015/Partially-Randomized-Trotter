#!/usr/bin/env python3
"""Edit text in fixed SVGs and export: no model/error/statistics evaluation."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE='55b719c8e0881bda848a8ed721b0cd452267b801'
Q='7430a386bcd4dc545107e413cd022ab7cb261968'
INTEGRATION='80b7a35c4dbf683f952293eeb55944bc10b9efc8'
B='3983fe4f1ede003953735e757d2c098c12b4644f'
QDIR='artifacts/resource_applicability/track_a_pf_q_model_diagnostics/2026-10-20/'
OLD='artifacts/resource_applicability/track_a_clean_figures_ja/2026-10-20/'
OUT=ROOT/'artifacts/resource_applicability/track_a_clean_figures_ja/2026-10-20_q_model_extension'
SCRIPT='scripts/extend_q_model_figures_20261020.py'
SVG='http://www.w3.org/2000/svg'
ET.register_namespace('',SVG);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
FIGURES={
 'E':('predicted_vs_measured','E_predicted_vs_measured_clean','PF model vs direct compile'),
 'E2':('relative_error_distribution','E2_relative_error_distribution_clean','Signed relative error'),
 'E3':('error_vs_q','E3_error_vs_q_clean','Relative error vs q'),
 'E4':('error_vs_prefix','E4_error_vs_prefix_clean','Relative error vs prefix'),
}
INPUTS={name:(Q,QDIR+name) for name in (
 'axis_errors.csv','axis_errors.json','grouped_errors.csv','grouped_errors.json','summary.json',
 'figure_data.json','numeric_completion.json','publication_scope.json','STOP.json')}
INPUTS.update({
 'display_figure_data.json':(Q,QDIR+'display/figure_data.json'),
 'display_provenance.json':(Q,QDIR+'display/display_provenance.json'),
 'display_rendering_completion.json':(Q,QDIR+'display/rendering_completion.json'),
 'q_diagnostics_report.md':(Q,'docs/research/track_a_pf_q_model_diagnostics_20261020.md'),
 'q_comparison_preparation.md':(Q,'docs/research/track_a_pf_q_model_comparison_preparation_20261020.md'),
 'B_summary.json':(B,'artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/summary.json'),
 'B_evidence_integration.md':(INTEGRATION,'docs/research/track_a_pf_q_model_evidence_integration_20261020.md'),
})
for _name in ('README.md','captions_ja.md','figure_notes_ja.md','figure_mapping.md','provenance.json'):
    INPUTS['existing_'+_name]=(BASE,OLD+_name)
def fixed(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def sha(data):return hashlib.sha256(data).hexdigest()
def jbytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write(path,obj):path.write_bytes(jbytes(obj))
def labels(root):return [''.join(t.itertext()).strip() for t in root.iter('{'+SVG+'}text')]

def geometry(node):
    """Canonical non-text SVG tree, including all paths/uses/clips/transforms."""
    if node.tag.rsplit('}',1)[-1] in {'metadata','text','tspan'} or node.get('id','').startswith('text_'):return None
    children=[c for item in node if (c:=geometry(item)) is not None]
    return [node.tag,sorted(node.attrib.items()),children]

def edit_svg(data,key):
    root=ET.fromstring(data);before=geometry(root);moves=[]
    for parent in root.iter():
        for child in list(parent):
            if child.tag!='{'+SVG+'}text':continue
            text=''.join(child.itertext()).strip();replacement=text
            if text.startswith(('Conditional saved-value diagnostic;','Full Hadamard-wrapper RZ,')):
                parent.remove(child);replacement=None
            elif text.startswith(('Saved q-dependent model vs','Relative-error distribution;','Error vs q;','Error vs prefix;')):
                replacement=FIGURES[key][2]
            elif 'S4 (composition hypothesis)' in text:replacement=text.replace('S4 (composition hypothesis)','S4')
            elif text.startswith('Signed error: 100'):replacement='Signed relative error (%)'
            if replacement is not None and 'B0I' in replacement:replacement=replacement.replace('B0I','B0+I')
            if replacement!=text:
                moves.append(dict(figure=key,old_english=text,new_label=replacement,notes='figure_notes_ja.md#'+key.lower()))
                if replacement is not None:
                    assert len(child)==0,('Unexpected nested changed label',text)
                    child.text=replacement
                    if key!='E' and 'B0I' in text and child.get('x'):
                        # Fit the longer B0+I alias inside the existing legend box.
                        child.set('x',str(float(child.get('x'))-2.5))
    if key=='E':
        # A compact encoding key; it uses the original filled/open marker convention.
        group=ET.SubElement(root,'{'+SVG+'}g',{'id':'text_encoding_E'})
        ET.SubElement(group,'{'+SVG+'}text',{
            'x':'468','y':'385','text-anchor':'middle',
            'style':"font-size:9px;font-family:'DejaVu Sans';fill:#333333"}).text='Filled: cosine  |  Open: sine'
    assert geometry(root)==before,('Non-text SVG changed',key)
    after=labels(root)
    assert not any(re.search(r'[ぁ-ゟ゠-ヿ一-鿿]',text) for text in after)
    assert not any(s in '\n'.join(after) for s in ('Conditional saved-value','runtime equivalence','no fit','composition hypothesis','no state preparation'))
    serialized=ET.tostring(root,encoding='utf-8',xml_declaration=True).decode()
    clean=('\n'.join(line.rstrip() for line in serialized.splitlines())+'\n').encode()
    assert geometry(ET.fromstring(clean))==before
    return clean,dict(
        status='EXACT_NON_TEXT_SVG_TREE_MATCH',before_sha256=sha(jbytes(before)),after_sha256=sha(jbytes(geometry(root))),
        preserved='All paths, scatter uses, histogram outlines, mean curves, equality/2% lines/bands, axes, ticks and clips',
        text_moves=moves,retained_labels=after)

def validate_saved(raw):
    completion=json.loads(raw['numeric_completion.json'])
    for name,digest in completion['result_hashes'].items():
        if name in raw:assert sha(raw[name])==digest,name
    assert raw['figure_data.json']==raw['display_figure_data.json']
    fd=json.loads(raw['display_figure_data.json']);axes=json.loads(raw['axis_errors.json'])['rows']
    groups=json.loads(raw['grouped_errors.json'])['rows'];summary=json.loads(raw['summary.json'])
    assert len(axes)==450 and len(groups)==220 and len(fd['points'])==450
    by_id={r['task_id']:r for r in axes}
    for point in fd['points']:assert all(point[k]==by_id[point['task_id']][k] for k in fd['point_columns'])
    assert list(csv.DictReader(io.StringIO(raw['axis_errors.csv'].decode())))
    expected=[('H6_DF19',192,'0.353456','0.805394'),('H8_DF28',258,'0.316507','1.488276')]
    for name,n,mean,maximum in expected:
        row=summary['systems'][name]
        assert row['n_axes']==row['below_2_percent']==n
        assert format(row['mean_absolute_relative_error_percent'],'.6f')==mean
        assert format(row['max_absolute_relative_error_percent'],'.6f')==maximum
    assert summary['all_axes']['n_axes']==summary['all_axes']['below_2_percent']==450
    assert format(summary['all_axes']['mean_absolute_relative_error_percent'],'.6f')=='0.332272'
    assert format(summary['all_axes']['max_absolute_relative_error_percent'],'.6f')=='1.488276'
    # Deliberately no sums, averages, quantiles, bins, error formulas or model evaluation.
    return dict(status='PASS_LITERAL_AND_SAVED_COORDINATE_CHECKS',axes=450,groups=220,
                mean_max_origin='Stored summary fields; six-decimal formatting only',statistics_recomputed=False)

def render():
    import cairosvg
    for sub in ('figures','inputs','verification'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    raw={name:fixed(*spec) for name,spec in INPUTS.items()}
    for name,data in raw.items():(OUT/'inputs'/name).write_bytes(data)
    data_check=validate_saved(raw);audit={};sources=[]
    for key,(stem,new,_) in FIGURES.items():
        source=fixed(Q,QDIR+'display/'+stem+'.svg')
        (OUT/'inputs'/(stem+'_original.svg')).write_bytes(source)
        (OUT/'inputs'/(stem+'_original.png')).write_bytes(fixed(Q,QDIR+'display/'+stem+'.png'))
        clean,audit[key]=edit_svg(source,key)
        (OUT/'figures'/(new+'.svg')).write_bytes(clean)
        cairosvg.svg2png(bytestring=clean,write_to=str(OUT/'figures'/(new+'.png')),dpi=300)
        cairosvg.svg2pdf(bytestring=clean,write_to=str(OUT/'figures'/(new+'.pdf')))
        sources.append(dict(figure=key,commit=Q,source=QDIR+'display/'+stem+'.svg',source_sha256=sha(source),new_stem='figures/'+new))
    preserved=[]
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,OLD],cwd=ROOT).decode().splitlines()
    for path in paths:
        if '/figures/' in path:preserved.append(dict(path=path,commit=BASE,sha256=sha(fixed(BASE,path))))
    assert len(preserved)==18
    write(OUT/'verification/svg_equivalence.json',audit)
    write(OUT/'verification/saved_value_check.json',data_check)
    write(OUT/'provenance.json',dict(base_commit=BASE,q_result_commit=Q,q_analysis_SOURCE='274f89774ae3f65b5fd0612f93dfbfe6979db19a',
        B_result_commit=B,B_evidence_integration_commit=INTEGRATION,input_sources=[dict(label=k,commit=INPUTS[k][0],path=INPUTS[k][1],sha256=sha(v)) for k,v in raw.items()],
        new_figures=sources,original_six_figure_files=preserved,renderer='Text-only SVG editing then CairoSVG PNG/PDF export',
        new_statistics=0,new_model_predictions=0,new_errors=0,new_bins=0,new_fits=0,
        new_shot_accounting=0,new_PR_PF_comparisons=0,main_table_changed=False,
        cairosvg_version=cairosvg.__version__,original_matplotlib_version='3.11.2',
        new_publication_authorization='NOT_CONFIRMED_LOCAL_ONLY_NO_PUSH'))
    print('PASS: four SVG text revisions; all non-text SVG trees identical; twelve exports saved.')

def check():
    for name,spec in INPUTS.items():assert (OUT/'inputs'/name).read_bytes()==fixed(*spec),name
    validate_saved({n:(OUT/'inputs'/n).read_bytes() for n in INPUTS})
    pdf_fonts={}
    for key,(stem,new,_) in FIGURES.items():
        original=fixed(Q,QDIR+'display/'+stem+'.svg')
        data=(OUT/'figures'/(new+'.svg')).read_bytes()
        assert geometry(ET.fromstring(original))==geometry(ET.fromstring(data)),key
        png=(OUT/'figures'/(new+'.png')).read_bytes()
        assert png.startswith(b'\x89PNG\r\n\x1a\n')
        w,h=struct.unpack('>II',png[16:24]);assert w>=3890 and h>=1700
        pdf=(OUT/'figures'/(new+'.pdf')).read_bytes()
        assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
        fonts=re.findall(rb'/FontName\s*/([^\s/]+)',pdf);assert fonts and b'/FontFile' in pdf
        pdf_fonts[key]=[s.decode('ascii') for s in fonts]
        text=' '.join(labels(ET.fromstring(data)))
        assert not re.search(r'[ぁ-ゟ゠-ヿ一-鿿]',text)
    prov=json.loads((OUT/'provenance.json').read_text())
    for r in prov['original_six_figure_files']:
        assert sha((ROOT/r['path']).read_bytes())==r['sha256']==sha(fixed(BASE,r['path']))
    slide=json.loads((OUT/'verification/slide_source.json').read_text())
    assert sha((OUT/'inputs/slide_plan_original.md').read_bytes())==slide['sha256']
    assert sha((ROOT.parent/slide['filename']).read_bytes())==slide['sha256']
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['script_sha256']==sha((ROOT/SCRIPT).read_bytes())
    for path,digest in manifest['output_sha256'].items():assert sha((OUT/path).read_bytes())==digest,path
    assert json.loads((OUT/'visual_review.json').read_text())['status']=='PASS_PRIMARY_AGENT_VISUAL_INSPECTION'
    assert json.loads((OUT/'publication_scope.json').read_text())['decision']=='LOCAL_ONLY_NO_PUSH'
    assert json.loads((OUT/'STOP.json').read_text())['status']=='COMPLETE_LOCAL_MANDATORY_STOP'
    count=0
    for name in ('README.md','captions_ja.md','figure_notes_ja.md','figure_mapping.md','slides_7_9_update_proposal_ja.md','publication_scope.md'):
        text=(OUT/name).read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            assert (OUT/target.split('#')[0]).exists(),(name,target)
            count+=1
    notes=(OUT/'figure_notes_ja.md').read_text();assert '450軸モデル精度検証は別図E' in notes
    for value in ('0.353456','0.805394','0.316507','1.488276','0.332272','225','788','338'):
        assert value in notes,value
    for key in FIGURES:
        assert '## '+key in notes and '## '+key in (OUT/'captions_ja.md').read_text()
    changes=subprocess.check_output(['git','diff','--name-status','-z',BASE],cwd=ROOT).decode().rstrip('\0').split('\0')
    if changes!=['']:
        for status,path in zip(changes[::2],changes[1::2]):
            assert status=='A' and (path==SCRIPT or path.startswith(str(OUT.relative_to(ROOT))+'/')),(status,path)
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check',BASE],cwd=ROOT,check=True,capture_output=True)
    tree=ast.parse((ROOT/SCRIPT).read_text())
    called={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not called & {'mean','sum','max','min','quantile','histogram','hist','linspace','bootstrap','polyfit','fit','predict'}
    return dict(status='PASS_SAVED_VALUES_SVG_GEOMETRY_FORMATS_FONTS_LINKS_PRESERVATION',
        new_figures=4,new_PNG=4,new_SVG=4,new_PDF=4,existing_six_files_unchanged=18,
        scientific_SVG_geometry='EXACT_NON_TEXT_TREE_MATCH',new_statistics=0,new_model_predictions=0,new_errors_or_bins=0,
        pdf_embedded_fonts=pdf_fonts,local_links=count,publication='LOCAL_ONLY_NO_PUSH',
        reviewer='Codex primary agent technical/display checks; not independent science review')

def finalize():
    write(OUT/'manifest.json',dict(script_sha256=sha((ROOT/SCRIPT).read_bytes()),output_sha256={str(p.relative_to(OUT)):sha(p.read_bytes())
          for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'manifest.json','verification.json'}}))
    write(OUT/'verification.json',dict(status='CHECK_IN_PROGRESS'))
    result=check();write(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.check:print(json.dumps(check(),ensure_ascii=False,indent=2))
    elif a.finalize:finalize()
    else:render()
if __name__=='__main__':main()
