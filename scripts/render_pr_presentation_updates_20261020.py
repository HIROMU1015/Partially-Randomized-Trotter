#!/usr/bin/env python3
"""A-D figures from fixed saved tokens only; no science or new statistics."""
from __future__ import annotations
import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE="c5ad79457aea87cc36b38022a2821ebb42eb276d"
H2="f2d1a86c431f835db700f348ce618d8604e7ee23"
INTEGRATION="8fbe7d77556c2353be837dbd9ed0fddc02ee81c4"
B_COMMIT="3983fe4f1ede003953735e757d2c098c12b4644f"
C_COMMIT="295b93d50662a0f8af3ec469df43099a3f04af60"
MAIN="1923fce8f7461513d1bc47b5061e87cd70917e87"
OUT=ROOT/"artifacts/resource_applicability/track_a_presentation_figure_updates/2026-10-20"
EPS=("0.05","0.0005")
A_DIR="artifacts/resource_applicability/track_a_pr_pf_absolute_cost_v2/2026-10-19/"
H2_DIR="artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/"
B_DIR="artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/"
C_DIR="artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/"
SPECS={
 "A_comparisons.csv":(BASE,A_DIR+"data/comparisons.csv"),
 "A_plot_points.csv":(BASE,A_DIR+"data/plot_points.csv"),
 "A_selection_decisions.csv":(BASE,A_DIR+"data/selection_decisions.csv"),
 "A_renderer.py":(BASE,"scripts/plot_pr_pf_absolute_cost_v2.py"),
 "A_report.md":(BASE,"docs/research/track_a_pr_pf_absolute_cost_v2_20261019.md"),
 "H2_P_confirmation.json":(H2,"artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json"),
 "H2_adopted_points.csv":(INTEGRATION,H2_DIR+"H2_adopted_points.csv"),
 "H2468_comparison_separate.csv":(INTEGRATION,H2_DIR+"H2468_comparison_separate.csv"),
 "H2_integration_report.md":(INTEGRATION,"docs/research/track_a_h2_results_integration_20261019.md"),
 "H468_main_table.csv":(MAIN,"artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/main_table.csv"),
 "B_summary.json":(B_COMMIT,B_DIR+"summary.json"),
 "B_figure_data.json":(B_COMMIT,B_DIR+"figure_data.json"),
 "B_combined_selection100.csv":(B_COMMIT,B_DIR+"combined_selection100.csv"),
 "B_selection_comparison30.csv":(B_COMMIT,B_DIR+"selection_comparison30.csv"),
 "B_rank_relation_changes.csv":(B_COMMIT,B_DIR+"rank_relation_changes.csv"),
 "B_report.md":(B_COMMIT,"docs/research/track_a_h6_noncancellation_stage_b_rerun_20261020.md"),
 "C_bootstrap_summary.csv":(C_COMMIT,C_DIR+"pr_bootstrap_summary.csv"),
 "C_full_sample_reference.csv":(C_COMMIT,C_DIR+"pr_full_sample_reference.csv"),
 "C_renderer.py":(C_COMMIT,"scripts/render_h4_pr_cost_variability.py"),
 "C_report.md":(C_COMMIT,"docs/research/track_a_h4_pr_cost_variability_20261019.md"),
}

def git_bytes(commit,path):
    return subprocess.check_output(["git","show",commit+":"+path],cwd=ROOT)
def sha(data):
    return hashlib.sha256(data).hexdigest()
def jbytes(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode())))
def cbytes(rows):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,list(rows[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()
def number(token):
    x=float(Fraction(str(token)))
    if not math.isfinite(x):raise ValueError("Nonfinite saved token")
    return x


def prepare():
    raw={label:git_bytes(*spec) for label,spec in SPECS.items()}
    sources=[dict(label=k,commit=SPECS[k][0],path=SPECS[k][1],sha256=sha(v),bytes=len(v)) for k,v in raw.items()]
    old=csv_rows(raw['A_comparisons.csv'])
    a=[dict(r) for r in old if r['epsilon'] in EPS]
    h2=json.loads(raw['H2_P_confirmation.json'])
    adopted=csv_rows(raw['H2_adopted_points.csv'])
    h2468=csv_rows(raw['H2468_comparison_separate.csv'])
    main=csv_rows(raw['H468_main_table.csv'])
    assert h2['status']=='COMPLETE_MANDATORY_STOP' and h2['old_cohorts_pooled'] is False
    assert h2['accounting']['candidate_reselected'] is False
    assert [r for r in h2468 if r['system']!='H2']==main
    assert [r for r in h2468 if r['system']=='H2']==adopted
    for i,r in enumerate(adopted):
        saved=h2['accounting']['comparisons'][i]
        assert r['G_PR_exact']==saved['G_exact'] and r['G_PF_exact']==saved['PF']['G_RZ_exact']
        assert r['paired_SE']==str(saved['paired_SE']) and r['n']==str(saved['n'])
        assert r['PR_id']==saved['cell_id'] and r['PF_id']==saved['PF']['cell_id']
        assert r['saved_direction_status']==saved['status'] and r['lower_saved_point_cost']=='PR'
        template=next(o for o in old if o['system']=='H2' and o['epsilon']==r['epsilon'])
        new=dict(template)
        assert template['hamiltonian_hash']==h2['binding']['hamiltonian']
        new.update(comparison_id='independent_H2_eps_'+r['epsilon'], panel='independent',
            PR_id=r['PR_id'],PF_id=r['PF_id'],L_D=r['L_D'],q='1' if r['epsilon']=='0.05' else '4',delta=r['delta'],
            n=r['n'],PR_cohort=r['cohort'],G_PR_exact=r['G_PR_exact'],G_PR_paired_SE_saved=r['paired_SE'],
            G_PF_exact=r['G_PF_exact'],PF_cost_class=r['PF_cost_class'],PF_cost_status_saved=r['PF_cost_class'],
            PF_selection_scope='PREREGISTERED_FIXED_P_CONTROL',saved_direction_status=r['saved_direction_status'],
            audit_status=r['audit_status'],result_commit=r['result_commit'],result_file=r['source_file'],
            result_record=r['source_record'],scientific_SOURCE=h2['SOURCE'],limits=r['interpretation_limits'])
        a.append(new)
    a.sort(key=lambda r:(EPS.index(r['epsilon']),r['panel'],int(r['hydrogen_atoms'])))
    assert len(a)==16
    assert [r for r in a if not(r['panel']=='independent' and r['system']=='H2')]==sorted(
        [r for r in old if r['epsilon'] in EPS],key=lambda r:(EPS.index(r['epsilon']),r['panel'],int(r['hydrogen_atoms'])))
    points=[]
    for r in a:
        for method in ('PR','PF'):
            token=r['G_'+method+'_exact']
            points.append(dict(comparison_id=r['comparison_id'],epsilon=r['epsilon'],panel=r['panel'],system=r['system'],
                hydrogen_atoms=r['hydrogen_atoms'],method=method,G_RZ_exact=token,G_RZ_plot=repr(number(token)),
                display_text=format(number(token),'.3g'),saved_paired_SE=r['G_PR_paired_SE_saved'] if method=='PR' else '',
                cost_class=r['PF_cost_class'] if method=='PF' else 'FINITE_SAMPLE_MEAN',PR_cohort=r['PR_cohort'],n=r['n'],
                result_commit=r['result_commit'],result_file=r['result_file'],result_record=r['result_record']))
    oldpoints=csv_rows(raw['A_plot_points.csv'])
    for p in points:
        if p['comparison_id'].startswith('independent_H2'):continue
        assert p==next(q for q in oldpoints if q['comparison_id']==p['comparison_id'] and q['method']==p['method'])
    b=json.loads(raw['B_summary.json'])
    expected={'best_candidate_changed':2,'q_changed':0,'PF_changed':0,'winner_total_changed':50,'ordering_changed':50,'PR_PF_relation_changed':0}
    for key,value in expected.items():assert b['all100'][key]==value
    assert b['all100']['relation_after']=={'PR_LOWER':90,'PF_LOWER':10}
    assert b['all100']['relation_before']==b['all100']['relation_after']
    assert b['scope_counts_overlap'] and b['unique_endpoints']==40
    summary=[dict(metric=k,previous70=b['previous70'][k],new30=b['new30'][k],all100=b['all100'][k],
                  source_pointer='/all100/'+k) for k in expected]
    combined=csv_rows(raw['B_combined_selection100.csv'])
    changes=[dict(r) for r in combined if r['best_candidate_changed']=='true']
    assert len(combined)==100 and len(changes)==2
    for r in changes:
        before,after=json.loads(r['baseline']),json.loads(r['replacement'])
        assert before['prefix']=='16' and after['prefix']=='15' and before['q']==after['q']=='4'
        assert before['PF']==after['PF']=='4th' and r['numeric_origin']=='VERBATIM_PREVIOUS70'
        assert r['PR_PF_relation_baseline']==r['PR_PF_relation_replacement']=='PF_LOWER'
    c=csv_rows(raw['C_bootstrap_summary.csv']);ref=csv_rows(raw['C_full_sample_reference.csv'])
    assert len(c)==18 and len(ref)==2 and all(r['B']=='2000' for r in c)
    assert {r['epsilon'] for r in c}==set(EPS)
    # Record each plotted numeric token, retaining the literal saved field.
    coordinates=[]
    for row_index,r in enumerate(c):
        for field in ('bootstrap_mean','cumulative_mean','q025','q975','bootstrap_SE','conditional_empirical_SE','cumulative_SE'):
            coordinates.append(dict(figure_group='C',epsilon=r['epsilon'],n=r['n'],source='C_bootstrap_summary.csv',
                record=str(row_index+1),field=field,saved_token=r[field],plot_coordinate=repr(number(r[field]))))
    for row_index,r in enumerate(ref):
        coordinates.append(dict(figure_group='C',epsilon=r['epsilon'],n=r['N'],source='C_full_sample_reference.csv',
            record=str(row_index+1),field='G_exact',saved_token=r['G_exact'],plot_coordinate=repr(number(r['G_exact']))))
    return raw,sources,a,points,b,summary,changes,c,ref,h2468,coordinates


def old_a_renderer(raw):
    tree=ast.parse(raw['A_renderer.py'].decode())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render')
    code=ast.get_source_segment(raw['A_renderer.py'].decode(),node)
    code=code.replace('Separate A cohorts; H2 P is incomplete','Separate A cohorts; H2 P added at left')
    code=code.replace('Independent confirmation P','Independent P  |  H2 added')
    namespace=dict(ROOT=ROOT,OUT=OUT/'A_absolute',EPSILONS=EPS,os=os,Fraction=Fraction)
    exec(compile(code,'saved_A_renderer_with_H2_display_labels','exec'),namespace)
    return namespace['render']


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'presentation-updates-20261020','savefig.facecolor':'white'})
    return plt


def export(fig,name):
    files=[]
    for ext in ('png','svg','pdf'):
        path=OUT/'figures'/(name+'.'+ext)
        metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {}
        fig.savefig(path,dpi=300,metadata=metadata)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        files.append(str(path.relative_to(OUT)))
    return files


def render_b(summary,stats,changes):
    plt=style()
    from matplotlib.patches import Patch
    labels=['Best PF candidate','Selected q','PF order','Best PF total cost','Candidate ordering','PR/PF direction']
    fig=plt.figure(figsize=(13,7.7))
    grid=fig.add_gridspec(2,2,left=.22,right=.975,bottom=.23,top=.79,width_ratios=[1.6,1],hspace=.22,wspace=.18)
    ax=fig.add_subplot(grid[:,0]);card=fig.add_subplot(grid[0,1]);example=fig.add_subplot(grid[1,1])
    ys=list(range(6))
    ax.barh(ys,[r['previous70'] for r in summary],height=.55,color='#557CA3')
    ax.barh(ys,[r['new30'] for r in summary],left=[r['previous70'] for r in summary],height=.55,color='#00A08A')
    for y,r in zip(ys,summary):
        ax.text(r['all100']+1.4,y,str(r['all100'])+'/100',va='center',fontsize=12,weight='bold',color='#213547')
    ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Scopes with a saved change');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Which decisions changed?',loc='left',weight='bold',pad=15)
    card.axis('off');example.axis('off')
    card.text(0,1,'Saved point-cost directions',fontsize=13,weight='bold',va='top')
    card.text(0,.70,str(stats['all100']['relation_after']['PR_LOWER']),fontsize=38,color='#007E73',weight='bold')
    card.text(.30,.73,'PR lower',fontsize=14)
    card.text(0,.25,str(stats['all100']['relation_after']['PF_LOWER']),fontsize=32,color='#B65524',weight='bold')
    card.text(.30,.28,'PF lower',fontsize=14)
    card.text(0,-.04,'Same directions under both price policies',fontsize=10,color='#555555')
    example.text(0,1,'H8: two saved prefix changes',fontsize=13,weight='bold',va='top')
    example.text(0,.65,'prefix 16  →  15',fontsize=22,weight='bold',color='#3C5878')
    example.text(0,.35,'q = 4  ·  fourth-order PF\nPF remains lower-cost',fontsize=12,linespacing=1.5)
    example.text(0,.00,'Noncancellation, ε = 0.0005\nStrengthened184; allowance factors 1 and 10\nBoth examples come from previous70',fontsize=9.5,color='#555555',linespacing=1.5)
    fig.suptitle('Sensitivity to replacing mixed unit prices with direct measurements',fontsize=17,weight='bold',y=.97)
    fig.text(.5,.905,'100 overlapping scopes  ·  40 unique system/bias/accuracy/allowance endpoints',ha='center',fontsize=12)
    fig.legend(handles=[Patch(color='#557CA3',label='Previous 70: copied saved results'),Patch(color='#00A08A',label='New 30: saved Stage B results')],
        ncol=2,loc='upper center',bbox_to_anchor=(.52,.87),frameon=False,fontsize=10.5)
    fig.text(.06,.035,'Saved summary only: q and PF order are unchanged; some PF costs and candidate rankings change.\n'
        'Replacement sensitivity, not isolated model-error validation. Scope overlap is not 100 independent experiments.\n'
        'Limits retained: profile/runtime differences; S4 hypothesis; missing initial final gate; pending old-profile axes.\n'
        'Fixed H6/H8 domains; saved union8/union6 PR n32; no population-mean, unseen-event, or global-optimum guarantee.',
        fontsize=9.2,color='#444444',va='bottom',linespacing=1.4)
    files=export(fig,'B_price_replacement_summary100');plt.close(fig)
    return files


def render_c(rows,refs):
    plt=style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    files=[]
    for mode in ('mean','se'):
        fig,axes=plt.subplots(1,2,figsize=(12.8,7))
        fig.subplots_adjust(top=.71,bottom=.27,left=.09,right=.98,wspace=.22)
        for ax,ep in zip(axes,refs):
            eps=ep['epsilon'];selected=[r for r in rows if r['epsilon']==eps]
            ns=[int(r['n']) for r in selected]
            ticks=[n for n in ns if n not in (12,1024)] if ep['N']=='1108' else ns
            ax.set_xscale('log',base=2);ax.set_xticks(ticks,[str(n) for n in ticks],rotation=40,ha='right',fontsize=9)
            ax.set_xlabel('MC sample size n (paired trajectories)',labelpad=10);ax.grid(alpha=.16)
            ax.set_title('ε = '+eps+'  |  full saved N = '+ep['N']+'\n'+ep['cell_id'],fontsize=10.5,weight='bold',pad=13)
            if mode=='mean':
                ax.fill_between(ns,[number(r['q025']) for r in selected],[number(r['q975']) for r in selected],color='#0072B2',alpha=.18)
                ax.plot(ns,[number(r['bootstrap_mean']) for r in selected],'o-',color='#0072B2',ms=4.5,lw=1.7)
                ax.plot(ns,[number(r['cumulative_mean']) for r in selected],'s-',color='#008C79',ms=4,lw=1.6)
                ax.axhline(number(ep['G_exact']),color='#6B6B6B',ls=':',lw=1.5)
                scale=1e6 if eps=='0.05' else 1e12;power='6' if eps=='0.05' else '12'
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_,s=scale:format(y/s,'.2f')))
                ax.set_ylabel('Mean PR cost $G_{\\mathrm{RZ}}$  ($10^{'+power+'}$)',labelpad=10)
            else:
                ax.plot(ns,[number(r['bootstrap_SE']) for r in selected],'o-',color='#0072B2',ms=4.5,lw=1.7)
                ax.plot(ns,[number(r['conditional_empirical_SE']) for r in selected],'--',color='#666666',lw=1.6)
                ax.plot(ns,[number(r['cumulative_SE']) for r in selected],'s:',color='#008C79',ms=4,lw=1.5)
                ax.set_yscale('log');ax.set_ylabel('SE of mean PR cost ($G_{\\mathrm{RZ}}$ units)',labelpad=10)
        if mode=='mean':
            title='H4: variability of the PR mean-cost estimate'
            handles=[Patch(color='#0072B2',alpha=.18,label='Saved bootstrap 2.5–97.5 percentiles'),
                Line2D([],[],color='#0072B2',marker='o',label='Saved mean of bootstrap means'),
                Line2D([],[],color='#008C79',marker='s',label='Saved cumulative MC mean'),
                Line2D([],[],color='#666666',ls=':',label='Full saved PR mean')]
            note='Band: conditional on the saved empirical distribution; not a population confidence interval.'
            name='C_pr_mean_cost_vs_n_presentation'
        else:
            title='H4: saved standard errors across MC sample sizes'
            handles=[Line2D([],[],color='#0072B2',marker='o',label='Saved SD of bootstrap means'),
                Line2D([],[],color='#666666',ls='--',label='Saved empirical SD / sqrt(n) column'),
                Line2D([],[],color='#008C79',marker='s',ls=':',label='Saved cumulative paired SE')]
            note='The reference curve is a stored column; no new SE, fit, bootstrap, or required-n calculation.'
            name='C_pr_standard_error_vs_n_supplement'
        fig.suptitle(title,fontsize=18,weight='bold',y=.975)
        fig.text(.5,.911,'Linear H4, 1 Å / STO-3G / tol-only DF10 / T = 0.8 / P = 0 / usual signed bias, factor 1',ha='center',fontsize=10.5)
        fig.legend(handles=handles,ncol=2 if mode=='mean' else 3,loc='upper center',bbox_to_anchor=(.53,.865),frameon=False,fontsize=9.8)
        fig.text(.09,.03,note+'\nSaved 2,000 resamples per n; the cumulative curve is one nested saved sequence.\n'
            'Mean-cost estimator, not the aggregate cost of n runs or server CPU time. Full prefix + tail + Hadamard wrapper.\n'
            'Unobserved rare-event costs remain unknown. State preparation and FTQC are excluded; no population guarantee.',
            fontsize=9.2,color='#444444',linespacing=1.5,va='bottom')
        files+=export(fig,name);plt.close(fig)
    return files


def render_d(rows):
    plt=style()
    fig,ax=plt.subplots(figsize=(11,7.6));ax.axis('off')
    ax.set_position([.04,.20,.92,.64])
    order=['H2','H4','H6','H8'];texts=[];colors=[]
    for system in order:
        cells=[];cs=[]
        for eps in EPS:
            r=next(r for r in rows if r['system']==system and r['epsilon']==eps)
            cells.append(r['lower_saved_point_cost']+' lower\n'+r['saved_direction_status']+'\nn = '+r['n'])
            cs.append('#E6F3EF' if r['lower_saved_point_cost']=='PR' else '#FFF0E5')
        texts.append(cells);colors.append(cs)
    table=ax.table(cellText=texts,rowLabels=['H2*  / DF3','H4  / DF10','H6  / DF19','H8†  / DF28'],
        colLabels=['ε = 0.05','ε = 0.0005'],cellColours=colors,cellLoc='center',rowLoc='center',loc='center',bbox=[.18,.01,.79,.97])
    table.auto_set_font_size(False);table.set_fontsize(12)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D4DDE3');cell.set_linewidth(.8)
        if row==0:cell.set_facecolor('#234058');cell.set_text_props(color='white',weight='bold',fontsize=14)
        if col==-1:cell.set_text_props(weight='bold',fontsize=11);cell.set_facecolor('#F4F6F8')
    fig.suptitle('Eight independent-confirmation conditions: saved point results',fontsize=17,weight='bold',y=.965)
    fig.text(.5,.895,'Supplementary presentation table  ·  existing six-row manuscript main table is unchanged',ha='center',fontsize=11,color='#444444')
    fig.text(.07,.025,'* H2: additional independent P, n80/12; not formal inclusion in the manuscript main table.\n'
        'H4/H6/H8: existing main-table series. REPRODUCED refers to the prespecified direction, not always PR advantage.\n'
        '† H8: full public PR execution receipts not reverified. Finite-sample points; no population-mean or unseen-event guarantee.\n'
        'Linear 1 Å / STO-3G / T=0.8 / P=0; stored analytic shots × compiled RZ. Input-generation differences remain.',
        fontsize=9.1,linespacing=1.5,va='bottom',color='#444444')
    files=export(fig,'D_independent_confirmation8_auxiliary');plt.close(fig)
    return files


def materialize(prepared):
    raw,sources,a,points,b,summary,changes,c,refs,h2468,coords=prepared
    for sub in ('inputs','data','figures','A_absolute/figures'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    for label,data in raw.items():(OUT/'inputs'/label).write_bytes(data)
    (OUT/'input_sources.json').write_bytes(jbytes(sources))
    (OUT/'data/A_comparisons_with_H2_P.csv').write_bytes(cbytes(a))
    (OUT/'data/A_plot_points_with_H2_P.csv').write_bytes(cbytes(points))
    (OUT/'data/B_change_counts.csv').write_bytes(cbytes(summary))
    (OUT/'data/B_H8_prefix_examples.csv').write_bytes(cbytes(changes))
    (OUT/'data/C_plot_coordinates.csv').write_bytes(cbytes(coords))
    (OUT/'data/D_independent_confirmation8.csv').write_bytes(raw['H2468_comparison_separate.csv'])
    with tempfile.TemporaryDirectory(prefix='pr-presentation-mpl-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        afiles,version=old_a_renderer(raw)(a,points)
        figures=['A_absolute/'+p for p in afiles]
        figures+=render_b(summary,b,changes)
        figures+=render_c(c,refs)
        figures+=render_d(h2468)
    return figures,version


def check(prepared):
    raw,sources,a,points,b,summary,changes,c,refs,h2468,coords=prepared
    for label,data in raw.items():assert (OUT/'inputs'/label).read_bytes()==data,label
    expected={'A_comparisons_with_H2_P.csv':cbytes(a),'A_plot_points_with_H2_P.csv':cbytes(points),
              'B_change_counts.csv':cbytes(summary),'B_H8_prefix_examples.csv':cbytes(changes),
              'C_plot_coordinates.csv':cbytes(coords),'D_independent_confirmation8.csv':raw['H2468_comparison_separate.csv']}
    for label,data in expected.items():assert (OUT/'data'/label).read_bytes()==data,label
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['input_sources']==sources
    assert (OUT/'input_sources.json').read_bytes()==jbytes(sources)
    assert manifest['render_script_sha256']==sha(Path(__file__).read_bytes())
    assert len(manifest['figure_files'])==18
    assert all(sum(name.endswith('.'+ext) for name in manifest['figure_files'])==6 for ext in ('png','svg','pdf'))
    for path,digest in manifest['output_sha256'].items():assert sha((OUT/path).read_bytes())==digest,path
    for name in manifest['figure_files']:
        data=(OUT/name).read_bytes()
        if name.endswith('.png'):
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
            width,height=struct.unpack('>II',data[16:24]);assert width>=3300 and height>=2000
        elif name.endswith('.svg'):assert ET.fromstring(data).tag=='{http://www.w3.org/2000/svg}svg'
        else:assert data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF')
    # No modifications to any tracked old source, figure, table, inventory or STOP.
    delta=subprocess.check_output(['git','diff','--name-status','-z',BASE],cwd=ROOT).decode().split('\0')
    delta=[d for d in delta if d]
    for status,path in zip(delta[::2],delta[1::2]):
        assert status=='A',('Existing file changed',status,path)
        assert path.startswith(str(OUT.relative_to(ROOT))+'/') or path=='scripts/render_pr_presentation_updates_20261020.py'
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                    'diff','--check',BASE],cwd=ROOT,check=True,capture_output=True)
    tree=ast.parse(Path(__file__).read_text())
    blocked={'mean','std','stdev','variance','quantile','bootstrap','sample','choice','polyfit','curve_fit','default_rng'}
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not(blocked & calls)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert not any(n.name.split('.')[0] in {'qiskit','scipy','trotterlib'} for n in node.names)
        elif isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in {'qiskit','scipy','trotterlib'}
    assert json.loads((OUT/'visual_review.json').read_text())['status']=='PASS_PRIMARY_AGENT_VISUAL_INSPECTION'
    assert json.loads((OUT/'STOP.json').read_text())['status']=='COMPLETE_MANDATORY_STOP'
    links=0
    for path in [OUT/'README.md',OUT/'captions.md',OUT/'publication_scope.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in target or target.startswith('#'):continue
            assert (path.parent/target.split('#')[0]).exists(),(path,target)
            links+=1
    return dict(status='PASS_SAVED_TOKEN_STRUCTURE_AND_PRESERVATION',figures=6,PNG=6,SVG=6,PDF=6,
        A_comparisons=16,A_points=32,A_existing_independent_conditions_unchanged=6,A_new_H2_P_conditions=2,A_exploratory_conditions_unchanged=8,
        B_summary_source='/all100; /previous70; /new30',B_examples=2,C_stored_n_rows=18,D_independent_conditions=8,
        sources=len(sources),local_links=links,new_scientific_calls=0,new_statistics=0,new_comparisons_or_rankings=0,
        whitespace_check='PASS; command-local cr-at-eol preserves source CSV CRLF bytes',
        visual_review='PASS_PRIMARY_AGENT_VISUAL_INSPECTION',
        original_source_figures_tables_ledgers_STOPs_modified=False,reviewer='Codex primary agent technical checks; not independent scientific review')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();prepared=prepare()
    if args.check:print(json.dumps(check(prepared),ensure_ascii=False,indent=2));return
    figures,version=materialize(prepared)
    manifest=dict(status='PRESENTATION_AND_PAPER_CANDIDATE_DISPLAY_ONLY',base_commit=BASE,
        date='2026-10-20 Asia/Tokyo',figure_files=figures,input_sources=prepared[1],matplotlib=version,dpi=300,
        source_reuse=dict(A_renderer=BASE,C_renderer=C_COMMIT),scientific_calls=0,new_bootstrap=0,new_accounting=0,
        render_script_sha256=sha(Path(__file__).read_bytes()),
        new_PR_PF_comparison_or_PF_selection=0,main_table_replaced=False,H2_main_table_inclusion_adopted=False,
        original_evidence_inventory_modified=False,detailed_science_review=False,final_evidence_freeze=False,
        output_sha256={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()
                       and p.name not in {'manifest.json','verification.json'}})
    (OUT/'manifest.json').write_bytes(jbytes(manifest))
    (OUT/'verification.json').write_bytes(jbytes(dict(status='CHECK_IN_PROGRESS')))
    result=check(prepared);(OUT/'verification.json').write_bytes(jbytes(result))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
