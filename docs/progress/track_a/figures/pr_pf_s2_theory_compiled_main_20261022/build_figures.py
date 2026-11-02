"""Render actually acquired current-S2 ratios. No scientific execution/retry."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
RAW=ROOT/'artifacts/resource_applicability/track_a_current_s2_theory_compiled/2026-10-22/run01'
EPS=['0.05','0.01','0.005','0.001','0.0005']
SYSTEMS=['H2','H4','H6','H8']


def read(p):return json.loads(Path(p).read_text())
def dump(name,value):(HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csv_write(name,rows):
    with (HERE/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def figure_data():
    selected=read(RAW/'output/selected_conditions.json');rows=[]
    for r in selected:
        assert r['PR']['PF_order']==r['PF']['PF_order']=='S2'
        ratio=F(r['PR_compiled_G'])/F(r['PF_compiled_G'])
        assert str(ratio)==r['rho_compiled']
        rows.append(dict(system=r['system'],epsilon=r['epsilon'],PR_candidate_id=r['PR']['candidate_id'],
            PF_candidate_id=r['PF']['candidate_id'],PF_method=r['PF']['method'],DF_rank=r['PR']['DF_rank'],
            PR_L_D=r['PR']['L_D'],PR_q=r['PR']['q'],PR_delta=r['PR']['delta'],PR_r=r['PR']['r'],PR_K=r['PR']['K'],
            PF_L_D=r['PF']['L_D'],PF_q=r['PF']['q'],PF_delta=r['PF']['delta'],
            PR_cohort=r['PR']['cohort'],PF_cohort=r['PF']['cohort'],PR_n=r['PR']['n'],PF_n=r['PF']['n'],
            G_PR_compiled=r['PR_compiled_G'],G_PF_compiled=r['PF_compiled_G'],
            G_PR_theory_lower=None,G_PR_theory_upper=None,G_PF_theory_lower=None,G_PF_theory_upper=None,
            PR_theory_units=None,PF_theory_units=None,PR_shots=r['PR_shots'],PF_shots=r['PF_shots'],
            rho_compiled_exact=str(ratio),rho_compiled_display=float(ratio),rho_theory_lower=None,rho_theory_upper=None,
            PR_change_percent=None,PF_change_percent=None,
            A_status='COMPILED_SERIES_ONLY_THEORY_MISSING',B_status='NOT_CREATED_MISSING_THEORY_G',
            four_total_comparison=False,reasons=r['reasons'],PR_selection_source=r['PR_selection_source'],PF_selection_source=r['PF_selection_source'],
            PR_original_record=r['PR'],PF_original_record=r['PF'],
            raw_selected_record=f'/output/selected_conditions/{selected.index(r)}'))
    return dict(schema='CURRENT_S2_PARTIAL_DISPLAY_V1',scope='Current exploration A S2 versus registered best eligible deterministic S2; signed factor1',
                theory_G_acquired_conditions=0,complete_four_total_pairs=0,compiled_only_conditions=20,
                status='PARTIAL_MISSING_FIXED_INPUTS',epsilons=EPS,rows=rows,
                figures_A_compiled_only=5,figures_A_theory_and_compiled=0,figures_B=0,
                scientific_SOURCE=read(RAW/'output/terminal.json')['SOURCE'])


def render_one(payload):
    epsilon,rows=payload
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'current-s2-partial-20261022'})
    fig,ax=plt.subplots(figsize=(7.6,5.2))
    values=[r['rho_compiled_display'] for r in rows]
    ax.axhline(1,color='#555e68',lw=1.2,linestyle='--',zorder=1)
    ax.plot(range(4),values,'o-',color='#008c95',markersize=8,lw=1.6,label='Compiled',zorder=3)
    for i,v in enumerate(values):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=12)
    ax.set_xticks(range(4),SYSTEMS);ax.set_ylabel('PR / PF total RZ cost')
    ax.set_ylim(.45,1.95);ax.set_xlim(-.22,3.22)
    ax.set_title(r'$\epsilon='+epsilon+r'$  |  Compiled only',fontsize=16,pad=16)
    ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='upper right',fontsize=13)
    fig.subplots_adjust(left=.15,right=.97,bottom=.14,top=.86)
    stem='A_compiled_only_eps_'+epsilon.replace('.','p')
    for ext in ['png','svg','pdf']:fig.savefig(HERE/(stem+'.'+ext),dpi=220)
    plt.close(fig)
    return dict(figure=stem,epsilon=epsilon,system_order=SYSTEMS,compiled_values_exact=[r['rho_compiled_exact'] for r in rows],
                compiled_values_display=values,theory_values_display=[None]*4,series_drawn=['Compiled'])


def check():
    d=figure_data();assert read(HERE/'figure_data.json')==d
    coords=read(HERE/'display_coordinates.json');assert len(coords)==5
    for c in coords:
        assert c['theory_values_display']==[None]*4
        assert c['compiled_values_display']==[float(F(v)) for v in c['compiled_values_exact']]
        for ext in ['png','svg','pdf']:assert (HERE/(c['figure']+'.'+ext)).stat().st_size>1000
    assert not list(HERE.glob('B_*.png'))
    return dict(status='PASS_TECHNICAL_PARTIAL',compiled_points=20,theory_points=0,A_compiled_only_figures=5,
                A_complete_figures=0,B_figures=0,exports=15,complete_current_pairs=0,
                missing_cost_zero_filling=False,independent_scientific_review=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:print(json.dumps(check()))
    else:
        d=figure_data();dump('figure_data.json',d)
        rows=[{k:v for k,v in r.items() if k not in ['PR_original_record','PF_original_record']} for r in d['rows']]
        csv_write('figure_data.csv',rows);csv_write('coverage_matrix.csv',rows)
        jobs=[(e,[next(r for r in d['rows'] if r['system']==h and r['epsilon']==e) for h in SYSTEMS]) for e in EPS]
        # Five independent small figures; workers equal the number of jobs, not a CPU quota.
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:coordinates=list(pool.map(render_one,jobs))
        dump('display_coordinates.json',coordinates);dump('verification_display_data.json',check())
        print(json.dumps(check()))
