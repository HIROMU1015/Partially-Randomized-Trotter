#!/usr/bin/env python3
"""Remove prose from six saved-result figures; retain their scientific artists."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE='0566c3bb6c9c1a2d634519db6aabc5d2ef412798'
OLD='artifacts/resource_applicability/track_a_presentation_figure_updates/2026-10-20/'
OUT=ROOT/'artifacts/resource_applicability/track_a_clean_figures_ja/2026-10-20'
SCRIPT='scripts/render_clean_pr_figures_ja_20261020.py'
LEGACY='scripts/render_pr_presentation_updates_20261020.py'
FIGURES={
 'A1':('A_absolute/figures/absolute_rz_eps_0p05','A1_absolute_rz_eps_0p05_clean'),
 'A2':('A_absolute/figures/absolute_rz_eps_0p0005','A2_absolute_rz_eps_0p0005_clean'),
 'B':('figures/B_price_replacement_summary100','B_price_replacement_summary100_clean'),
 'C1':('figures/C_pr_mean_cost_vs_n_presentation','C1_pr_mean_cost_vs_n_clean'),
 'C2':('figures/C_pr_standard_error_vs_n_supplement','C2_pr_standard_error_vs_n_clean'),
 'D':('figures/D_independent_confirmation8_auxiliary','D_independent_confirmation8_clean'),
}
DATA=[
 'data/A_comparisons_with_H2_P.csv','data/A_plot_points_with_H2_P.csv',
 'data/B_change_counts.csv','data/B_H8_prefix_examples.csv','data/C_plot_coordinates.csv',
 'data/D_independent_confirmation8.csv','inputs/B_summary.json',
 'inputs/C_bootstrap_summary.csv','inputs/C_full_sample_reference.csv',
]

def fixed(path):
    return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
def digest(data):return hashlib.sha256(data).hexdigest()
def jbytes(obj):return (json.dumps(plain(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write_json(path,obj):path.write_bytes(jbytes(obj))
def rows(data):return list(csv.DictReader(io.StringIO(data.decode())))
def plain(value):
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    return value
def canonical_label(s):return s.replace('†','').replace('*','')

def scene(fig):
    """Scientific artist coordinates and styles, independent of page layout."""
    axes=[]
    for ax in fig.axes:
        lines=[dict(x=plain(line.get_xdata(orig=False)),y=plain(line.get_ydata(orig=False)),
                    color=plain(line.get_color()),marker=line.get_marker(),linestyle=line.get_linestyle(),
                    width=line.get_linewidth(),marker_size=line.get_markersize(),
                    marker_face=plain(line.get_markerfacecolor()),marker_edge=plain(line.get_markeredgecolor()))
               for line in ax.lines]
        collections=[]
        for coll in ax.collections:
            collections.append(dict(kind=type(coll).__name__,
                paths=[dict(vertices=plain(p.vertices),codes=plain(p.codes)) for p in coll.get_paths()],
                offsets=plain(coll.get_offsets()),segments=plain(coll.get_segments()) if hasattr(coll,'get_segments') else None,
                colors=plain(coll.get_facecolors()),edges=plain(coll.get_edgecolors()),widths=plain(coll.get_linewidths())))
        bars=[dict(kind=type(p).__name__,xy=plain(p.get_xy()),width=p.get_width(),height=p.get_height(),
                   color=plain(p.get_facecolor()),edge=plain(p.get_edgecolor())) for p in ax.patches]
        tables=[{str(key):dict(text=canonical_label(cell.get_text().get_text()),
                             face=plain(cell.get_facecolor()),edge=plain(cell.get_edgecolor()))
                 for key,cell in table.get_celld().items()} for table in ax.tables]
        axes.append(dict(xscale=ax.get_xscale(),yscale=ax.get_yscale(),xlim=plain(ax.get_xlim()),ylim=plain(ax.get_ylim()),
            xticks=plain(ax.get_xticks()),yticks=plain(ax.get_yticks()),
            xticklabels=[canonical_label(t.get_text()) for t in ax.get_xticklabels()],
            yticklabels=[t.get_text() for t in ax.get_yticklabels()],xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),
            lines=lines,collections=collections,bars=bars,tables=tables))
    return dict(axes=axes,legends=[[t.get_text() for t in leg.get_texts()] for leg in fig.legends])

def clean(fig,key):
    """Edit only prose/title text and page layout; do not rebuild any data artist."""
    from matplotlib.ticker import FixedLocator,FixedFormatter
    moves=[]
    def change(artist,new,role):
        old=artist.get_text()
        if old==new:return
        moves.append(dict(figure=key,role=role,old_english=old,new_figure_label=new,
                          japanese_notes='figure_notes_ja.md#'+key.lower()))
        if new is None:artist.remove()
        else:artist.set_text(new)
    titles={'A1':r'PR vs deterministic PF  |  $\epsilon = 0.05$',
            'A2':r'PR vs deterministic PF  |  $\epsilon = 0.0005$',
            'B':'Unit-price replacement sensitivity','C1':'H4 PR mean cost',
            'C2':'H4 PR standard error','D':'Independent confirmation'}
    for t in list(fig.texts):
        if t is fig._suptitle:change(t,titles[key],'title')
        elif t.get_text().startswith('Linear H-chains') or t.get_text().startswith('Linear H4'):
            change(t,'1 Å  ·  STO-3G  ·  T = 0.8  ·  P = 0','short_condition')
        elif key=='B' and t.get_text().startswith('100 overlapping scopes'):
            change(t,'100 scopes  ·  40 unique conditions','short_scope_label')
        else:change(t,None,'paragraph_or_footnote')
    for i,ax in enumerate(fig.axes):
        # Fix the existing tick locations before changing only the page size.
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
        ax.xaxis.set_minor_locator(FixedLocator(ax.xaxis.get_minorticklocs()))
        ax.yaxis.set_minor_locator(FixedLocator(ax.yaxis.get_minorticklocs()))
        if key.startswith('A'):
            change(ax.title,'Independent P' if i==0 else 'Exploratory A  |  n = 8','panel_title')
            for t in list(ax.texts):
                if t.get_text().startswith('Separate A cohorts'):change(t,None,'cohort_explanation')
            for t in ax.get_xticklabels():
                if '†' in t.get_text():change(t,canonical_label(t.get_text()),'footnote_marker')
            ax.xaxis.set_major_formatter(FixedFormatter([canonical_label(t.get_text()) for t in ax.get_xticklabels()]))
        elif key=='B':
            if i==0:change(ax._left_title,'Saved changes','panel_title')
            for t in list(ax.texts):
                text=t.get_text()
                if text=='Saved point-cost directions':change(t,'Point-cost direction','panel_title')
                elif text=='Same directions under both price policies':change(t,None,'interpretation')
                elif text=='H8: two saved prefix changes':change(t,'H8  ·  2 scopes','short_scope_label')
                elif text=='q = 4  ·  fourth-order PF\nPF remains lower-cost':
                    change(t,'q = 4  ·  PF4\nPF lower','short_condition_and_category')
                elif text.startswith('Noncancellation, ε ='):change(t,'ε = 0.0005','short_condition')
        elif key=='D':
            for table in ax.tables:
                for (row,col),cell in table.get_celld().items():
                    if col==-1:change(cell.get_text(),canonical_label(cell.get_text().get_text()),'footnote_marker')
    if key.startswith('A'):
        fig.set_size_inches(12,5.6);fig.subplots_adjust(bottom=.18,top=.74)
    elif key=='B':
        fig.set_size_inches(13,6.4)
        fig.axes[0].set_position([.22,.15,.43,.59])
        fig.axes[1].set_position([.715,.47,.26,.28])
        fig.axes[2].set_position([.715,.14,.26,.26])
    elif key.startswith('C'):
        fig.set_size_inches(12.8,5.8);fig.subplots_adjust(bottom=.19,top=.66)
    else:
        fig.set_size_inches(11,5.7);fig.axes[0].set_position([.04,.04,.92,.84])
    return moves

def render():
    for sub in ('data','figures','verification'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    originals={p:fixed(OLD+p) for p in DATA}
    for p,data in originals.items():(OUT/'data'/Path(p).name).write_bytes(data)
    before={};after={};moves=[];retained={}
    with tempfile.TemporaryDirectory(prefix='pr-clean-figures-') as tmp:
        tmp=Path(tmp);os.environ['MPLCONFIGDIR']=str(tmp/'mpl')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.text import Text
        # Only the saved display functions are invoked. main/prepare/check are not called.
        spec=importlib.util.spec_from_file_location('pinned_display',ROOT/LEGACY)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        assert (ROOT/LEGACY).read_bytes()==fixed(LEGACY)
        legacy.OUT=tmp
        for sub in ('figures','A_absolute/figures'):(tmp/sub).mkdir(parents=True,exist_ok=True)
        saver=Figure.savefig
        def save(fig,path,*args,**kwargs):
            relative=str(Path(path).relative_to(tmp));stem=str(Path(relative).with_suffix(''))
            key=next(k for k,v in FIGURES.items() if v[0]==stem)
            if key not in before:
                fig.canvas.draw();before[key]=scene(fig)
                moves.extend(clean(fig,key));fig.canvas.draw();after[key]=scene(fig)
                assert before[key]==after[key],('Scientific artist mismatch',key)
                retained[key]=[t.get_text() for t in fig.findobj(match=Text) if t.get_text() and t.get_visible()]
                assert not any(re.search(r'[ぁ-ゟ゠-ヿ一-鿿]',s) for s in retained[key])
            return saver(fig,path,*args,**kwargs)
        Figure.savefig=save
        try:
            legacy.old_a_renderer({'A_renderer.py':fixed(OLD+'inputs/A_renderer.py')})(
                rows(originals[DATA[0]]),rows(originals[DATA[1]]))
            bs=rows(originals[DATA[2]])
            for row in bs:
                for field in ('previous70','new30','all100'):row[field]=int(row[field])
            legacy.render_b(bs,json.loads(originals['inputs/B_summary.json']),rows(originals[DATA[3]]))
            legacy.render_c(rows(originals['inputs/C_bootstrap_summary.csv']),rows(originals['inputs/C_full_sample_reference.csv']))
            legacy.render_d(rows(originals['data/D_independent_confirmation8.csv']))
        finally:Figure.savefig=saver;plt.close('all')
        for old,new in FIGURES.values():
            for ext in ('png','svg','pdf'):(OUT/'figures'/(new+'.'+ext)).write_bytes((tmp/(old+'.'+ext)).read_bytes())
    write_json(OUT/'verification/scenes_before.json',before)
    write_json(OUT/'verification/scenes_after.json',after)
    write_json(OUT/'verification/text_moves.json',moves)
    write_json(OUT/'verification/retained_labels.json',retained)
    provenance=dict(status='DISPLAY_ONLY_FROM_PUBLIC_FIXED_RESULTS',base_commit=BASE,old_directory=OLD,
        legacy_renderer=dict(path=LEGACY,sha256=digest(fixed(LEGACY))),
        original_figures={key:dict(old_stem=old,new_stem='figures/'+new,
            original_sha256={ext:digest(fixed(OLD+old+'.'+ext)) for ext in ('png','svg','pdf')})
                          for key,(old,new) in FIGURES.items()},
        saved_data=[dict(commit=BASE,path=OLD+p,new_path='data/'+Path(p).name,sha256=digest(data)) for p,data in originals.items()],
        scientific_input_sources=json.loads(fixed(OLD+'input_sources.json')),
        software=dict(matplotlib=matplotlib.__version__,dpi=300),
        display_changes='Prose/title text and page layout only; all scientific artists retained.',
        science_calls=0,new_statistics=0,new_accounting=0,new_ranking=0,new_PR_PF_comparisons=0,
        review='Primary-agent technical checks only; no detailed science review or final evidence freeze.')
    write_json(OUT/'provenance.json',provenance)
    inventory()
    print('Rendered six clean figures; scientific artist equivalence PASS. Visual inspection and --finalize remain.')

def inventory():
    groups=[
      ('H-chain absolute cost v2',BASE,'track_a_pr_pf_absolute_cost_v2/2026-10-19/figures/'),
      ('H8 cost breakdown / Figure 2',BASE,'track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/'),
      ('Theory / compiled totals','f15752625613956d087bab221e48f1ce7dd59fef','track_a_h46_theory_compiled_total_plots/2026-10-19/figures/'),
      ('H4 PR variability','295b93d50662a0f8af3ec469df43099a3f04af60','track_a_h4_pr_cost_variability/2026-10-19/'),
      ('q-dependent unit-price sensitivity','3983fe4f1ede003953735e757d2c098c12b4644f','track_a_h6_noncancellation_stage_b_rerun/2026-10-20/'),
    ]
    records=[]
    for family,commit,suffix in groups:
        directory='artifacts/resource_applicability/'+suffix
        paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit,directory],cwd=ROOT).decode().splitlines()
        for path in paths:
            if not path.endswith('.svg'):continue
            data=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
            root=ET.fromstring(data)
            labels=[''.join(t.itertext()) for t in root.iter('{http://www.w3.org/2000/svg}text')]
            labels+=[html.unescape(s.strip()) for s in re.findall(r'<!--(.*?)-->',data.decode(),re.S)]
            excerpts=[s for s in labels if len(s)>95 or any(w in s for w in
                      ('not a population','not confidence','receipts not','final gate','no new analysis','No new fit'))]
            records.append(dict(family=family,commit=commit,path=path,sha256=digest(data),
                long_annotation_examples=list(dict.fromkeys(excerpts)),
                action='SURVEY_ONLY_ORIGINAL_PRESERVED',text_extraction='SVG text plus Matplotlib text comments'))
    write_json(OUT/'verification/existing_figure_inventory.json',records)

def check():
    prov=json.loads((OUT/'provenance.json').read_text())
    for p in DATA:assert (OUT/'data'/Path(p).name).read_bytes()==fixed(OLD+p)
    assert json.loads((OUT/'verification/scenes_before.json').read_text())==json.loads((OUT/'verification/scenes_after.json').read_text())
    for key,(old,new) in FIGURES.items():
        for ext in ('png','svg','pdf'):
            assert digest(fixed(OLD+old+'.'+ext))==prov['original_figures'][key]['original_sha256'][ext]
            data=(OUT/'figures'/(new+'.'+ext)).read_bytes()
            if ext=='png':
                assert data[:8]==b'\x89PNG\r\n\x1a\n';w,h=struct.unpack('>II',data[16:24]);assert w>=3300 and h>=1650
            elif ext=='svg':
                xml=ET.fromstring(data);assert xml.tag=='{http://www.w3.org/2000/svg}svg'
                labels=' '.join(xml.itertext())
                assert not re.search(r'[ぁ-ゟ゠-ヿ一-鿿]',labels)
                assert not any(s in labels for s in ('not a population','Limits retained:','receipts not reverified','Lines guide the eye','Unobserved rare-event','Which decisions changed?'))
            else:assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['script_sha256']==digest((ROOT/SCRIPT).read_bytes())
    for path,sha in manifest['output_sha256'].items():assert digest((OUT/path).read_bytes())==sha,path
    assert json.loads((OUT/'visual_review.json').read_text())['status']=='PASS_PRIMARY_AGENT_VISUAL_INSPECTION'
    assert json.loads((OUT/'STOP.json').read_text())['status']=='COMPLETE_MANDATORY_STOP'
    retained=json.loads((OUT/'verification/retained_labels.json').read_text())
    for move in json.loads((OUT/'verification/text_moves.json').read_text()):
        assert move['old_english'] not in retained[move['figure']]
    for r in json.loads((OUT/'verification/existing_figure_inventory.json').read_text()):
        assert digest(subprocess.check_output(['git','show',r['commit']+':'+r['path']],cwd=ROOT))==r['sha256']
    links=0
    for name in ('README.md','captions_ja.md','figure_notes_ja.md','figure_mapping.md','remaining_candidates_ja.md'):
        text=(OUT/name).read_text()
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            assert (OUT/target.split('#')[0]).exists(),(name,target)
            links+=1
        if name in ('captions_ja.md','figure_notes_ja.md'):
            for key in FIGURES:assert '## '+key in text,(name,key)
        if name=='figure_mapping.md':
            for key,(old,new) in FIGURES.items():assert '|'+key+'|' in text and old in text and new in text
    conditions=rows(fixed(OLD+'data/D_independent_confirmation8.csv'))
    for line in (OUT/'figure_notes_ja.md').read_text().splitlines():
        if not re.match(r'\|H[2468] / ',line):continue
        cells=[s.strip() for s in line.strip('|').split('|')]
        row=next(r for r in conditions if r['system']==cells[0].split()[0] and r['epsilon']==cells[1])
        assert cells[2:]==[row['L_D'],row['delta'],row['n']]
    delta=subprocess.check_output(['git','diff','--name-status','-z',BASE],cwd=ROOT).decode().rstrip('\0').split('\0')
    if delta!=['']:
        for status,path in zip(delta[::2],delta[1::2]):
            assert status=='A' and (path==SCRIPT or path.startswith(str(OUT.relative_to(ROOT))+'/')),(status,path)
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                    'diff','--check',BASE],cwd=ROOT,capture_output=True,check=True)
    tree=ast.parse((ROOT/SCRIPT).read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls & {'mean','std','quantile','bootstrap','sample','choice','polyfit','curve_fit','prepare'}
    return dict(status='PASS_SAVED_DATA_ARTIST_EQUIVALENCE_FORMAT_LINKS_PRESERVATION',
        figures=list(FIGURES),formats=dict(PNG=6,SVG=6,PDF=6),data_files_byte_identical=len(DATA),surveyed_legacy_SVG=21,
        numeric_series_errors_bands_bars_axis_limits_scales_legend_meanings='EXACT_SCIENTIFIC_ARTIST_MATCH',
        prose_removed=True,Japanese_in_figures=False,Japanese_markdown=True,local_links=links,
        original_tracked_files_modified=False,new_science_calls=0,new_statistics=0,
        visual_review='PASS_PRIMARY_AGENT_VISUAL_INSPECTION',
        reviewer='Codex primary agent; not independent scientific review',
        github_preview='PNG/SVG/PDF exported; repository-relative links checked. No browser screenshot claim.')

def finalize():
    write_json(OUT/'manifest.json',dict(base_commit=BASE,script_sha256=digest((ROOT/SCRIPT).read_bytes()),
        output_sha256={str(p.relative_to(OUT)):digest(p.read_bytes()) for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name not in {'manifest.json','verification.json'}}))
    write_json(OUT/'verification.json',dict(status='CHECK_IN_PROGRESS'))
    result=check();write_json(OUT/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.check:print(json.dumps(check(),ensure_ascii=False,indent=2))
    elif args.finalize:finalize()
    else:render()
if __name__=='__main__':main()
