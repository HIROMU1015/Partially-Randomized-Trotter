"""Revision figures: plot frozen published values and explanatory schematics.

No research module is imported. No candidate search, simulation, refitting,
new confidence interval, or new cost estimate is performed. Fraction conversion,
axis rescaling, and addition of the two saved axis shot counts are display only.
Saved ratios, intervals and decisions are quoted into plot_values.csv, never recomputed.
Run with the repository's .venv311/bin/python.
"""
from pathlib import Path
from fractions import Fraction
import csv,io,json,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures_v2';OUT.mkdir(exist_ok=True)
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'font.size':14,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CAD4DE',
 'axes.labelcolor':'#19334D','text.color':'#19334D','xtick.color':'#53677A',
 'ytick.color':'#53677A','figure.facecolor':'white','axes.facecolor':'white',
 'svg.fonttype':'none'})
TEAL,BLUE,AMBER,GRAY,CORAL='#009C91','#446CB3','#CE8530','#8998A8','#C85B65'
# Match method colors to the quoted, published three-method figure.
DF_CDF_COLOR,DF_AWARE_COLOR,DF_ENTRY_COLOR='#929eab','#477a9b','#167d75'
PAULI_SHA='4591b8d75a2a6a55c74a51225f0c61648023c8ee'
PAULI_PATH='docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md'
DF_SHA='ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed'
DF_PATH='artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/comparisons.csv'
LONG_SHA='155529e20a8fac39e00786c40b57830b71203c4f'
LONG_PATH='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/best_configurations.csv'
H8_SHA='eb07156a37b1938ed3aed95c13abbb545466c573'
H8_PATH='artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/confirmation.json'
# Later version of the integrated three-method report. Its PNG/SVG/render script
# are byte-identical to the initial 709813805d41e470676e311cd63a740af23bf33c.
THREE_METHOD_SHA='776e0a16ac98762631fe9307672725fdfee3b73d'
THREE_METHOD_INITIAL_SHA='709813805d41e470676e311cd63a740af23bf33c'
THREE_METHOD_DIR='docs/research/research_c_three_method_z_relocated_comparison_20261108'
# Full-Hamiltonian pure RTE: fixed T=.8 candidates, and the T=2.0 q/R/K search.
FULLH_SHA='9b83b96b83f62142c9d87858dc24a94db7ac5d61'
FULLH_DIR='artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09'
FULLH_T20_SHA='7a8bc7f132e166390c943e6b32c92cc6abe1baf4'
FULLH_T20_DIR='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09'
# Independent fresh128 CDF0 versus ENTRY comparison.
INDEP_SHA='e09f3bea9df5f2f545f1b928e6cf164d4e5d4551'
INDEP_SUMMARY='artifacts/research_c_independent_price_precision/2026-11-08/publication/summary.json'
INDEP_REPORT='docs/research/research_c_cdf0_entry_independent_price_results_20261108.md'
DISPLAY=[]
def source(sha,path):return subprocess.check_output(['git','show',sha+':'+path],text=True)
def num(x):return float(Fraction(str(x)))
def record(fig,panel,series,x,metric,token,unit,sha,path,selector):
    DISPLAY.append(dict(figure=fig,panel=panel,series=series,x=x,metric=metric,
        saved_value=str(token),display_value=num(token) if token is not None else '',
        unit=unit,source_commit=sha,source_path=path,selector=selector))
def save(fig,name):
    for ext in ('png','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def grid(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)

# S1 Table 1: quote all four existing family columns, without selecting a winner.
text=source(PAULI_SHA,PAULI_PATH)
section=text.split('### Table 1：',1)[1].split('### H2',1)[0]
pauli={}
for line in section.splitlines():
    if not line.startswith(('| H2 |','| H4 |','| H6 |')):continue
    cells=[x.strip() for x in line.strip('|').split('|')]
    pauli[(cells[0],cells[1])]=cells
eps=['0.05','0.005','0.0005']
fig,axs=plt.subplots(1,3,figsize=(13.6,4.8))
series=[('Full PF',2,GRAY,'s'),('切捨てPF',3,AMBER,'D'),('Hybrid PR',4,BLUE,'o'),('純RTE',5,TEAL,'^')]
for ax,e in zip(axs,eps):
    for k,(label,col,color,marker) in enumerate(series):
        xs=[];ys=[]
        for j,mol in enumerate(['H2','H4','H6']):
            token=pauli[(mol,e)][col]
            if token=='—':continue
            xs.append(j+(k-1.5)*.075);ys.append(num(token))
            record('pauli_precision',e,label,mol,'G',token,'RZ',PAULI_SHA,PAULI_PATH,f'Table1 {mol}/epsilon={e}/column={col}')
        ax.scatter(xs,ys,color=color,marker=marker,s=65,label=label,zorder=3)
    ax.set(title='要求信号精度 ε='+e,xticks=[0,1,2],xticklabels=['H2','H4','H6'],yscale='log',xlim=(-.35,2.35))
    grid(ax)
axs[0].set_ylabel('測定込み総RZ費用 G [RZ]')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.08))
fig.tight_layout(w_pad=2);save(fig,'pauli_precision')

# S3 comparisons.csv: B2, B3 and already recorded deterministic envelopes.
df=list(csv.DictReader(io.StringIO(source(DF_SHA,DF_PATH))))
df_configs=[]
# Look up conditions by the IDs already selected in the published table.
# This is provenance lookup only; plot G stays the recorded comparisons.csv G.
df_selected=[r for r in df if r['T']=='0.8' and r['epsilon'] in eps]
needed={(r[key],r['epsilon']) for r in df_selected
        for key in ['Full_PF_id','PF_envelope_id','B2_id','B3_id']}
meta={}
with subprocess.Popen(['git','show',LONG_SHA+':artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/candidate_precision_accounting.csv'],
                      text=True,stdout=subprocess.PIPE) as process:
    for r in csv.DictReader(process.stdout):
        key=(r['id'],r['epsilon'])
        if key in needed:meta[key]=r
for r in csv.DictReader(io.StringIO(source(DF_SHA,
    'artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/best_configurations.csv'))):
    if (r['id'],r['epsilon']) in needed:meta[(r['id'],r['epsilon'])]=r
DF_SERIES=[('Full PF','Full_PF_G','Full_PF_id','SAVED_FULL_PF_BEST',GRAY,'s'),
           ('決定論PF対照','PF_envelope_G','PF_envelope_id','SAVED_PF_ENVELOPE_BEST',AMBER,'D'),
           ('Hybrid B2','B2_G','B2_id','SAVED_B2_BEST',BLUE,'o'),
           ('DF-native B3','B3_G','B3_id','SAVED_B3_BEST',CORAL,'^')]
def df_saved_row(mol,t,e):
    selected=[r for r in df if r['system']==mol and r['T']==t and r['epsilon']==e]
    assert len(selected)==1
    return selected[0]
def check_baselines(comparisons,t):
    # Provenance check only: the full-H reports must quote the same saved S13 IDs/G.
    for c in comparisons:
        if c['epsilon'] not in eps:continue
        for label,col,id_col,role,_,_ in DF_SERIES:
            if c['role']!=role:continue
            r=df_saved_row(c['system'],t,c['epsilon'])
            assert c['baseline_id']==r[id_col] and num(c['baseline_G'])==num(r[col]),(t,role,c['system'],c['epsilon'])
def df_figure(t,name,fullh,fullh_label,fullh_sha,fullh_path,title_prefix,footnote):
    fig,axs=plt.subplots(1,3,figsize=(13.6,4.8))
    series=DF_SERIES+[(fullh_label,None,None,None,TEAL,'v')]
    for ax,e in zip(axs,eps):
        for k,(label,col,id_col,role,color,marker) in enumerate(series):
            xs=[];ys=[]
            for j,mol in enumerate(['H4','H6']):
                x=j+(k-2)*.075
                if col is None:
                    f=fullh[(mol,e)];token=f['G_exact'];xs.append(x);ys.append(num(token))
                    selector=f"{mol}/T={t}/epsilon={e};id={f['id']};LD=0;q={f['q']};delta={f['delta']};R={f['R_outer']};K={f['K']};n={f['n']};cohort={f['cohort']}"
                    record(name,e,label,mol,'G',token,'RZ',fullh_sha,fullh_path,selector)
                    if t=='0.8':
                        df_configs.append(dict(system=mol,epsilon=e,series=label,candidate_id=f['id'],
                            formula=f['formula'],L_D=f['prefix'],q=f['q'],delta=f['delta'],
                            R_outer=f['R_outer'],K=f['K'],cohort=f['cohort'],reported_G=f['G'],
                            source_commit=fullh_sha))
                    continue
                r=df_saved_row(mol,t,e);token=r[col];xs.append(x);ys.append(num(token))
                selector=f'{mol}/T={t}/epsilon={e}/{col};id='+r[id_col]
                if t=='0.8':
                    cfg=meta[(r[id_col],e)]
                    df_configs.append(dict(system=mol,epsilon=e,series=label,candidate_id=r[id_col],
                        formula=cfg['formula'],L_D=cfg['prefix'],q=cfg['q'],delta=cfg['delta'],
                        R_outer=cfg['R_outer'],K=cfg['K'],cohort=cfg['cohort'],reported_G=token,
                        source_commit=DF_SHA))
                    selector+=f";LD={cfg['prefix']};q={cfg['q']};delta={cfg['delta']};R={cfg['R_outer']};K={cfg['K']}"
                else:
                    c=t20_baseline[(mol,e,role)]
                    selector+=f";LD={c['baseline_prefix']};formula={c['baseline_formula']};q={c['baseline_q']};R={c['baseline_R']};K={c['baseline_K']};same_as_T20_report_comparisons"
                record(name,e,label,mol,'G',token,'RZ',DF_SHA,DF_PATH,selector)
            ax.scatter(xs,ys,color=color,marker=marker,s=65,label=label,zorder=3)
        ax.set(title=title_prefix+'要求信号精度 ε='+e,xticks=[0,1],xticklabels=['H4 · DF10','H6 · DF19'],yscale='log',xlim=(-.4,1.4))
        grid(ax)
    axs[0].set_ylabel('測定込み総RZ費用 G [RZ]')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,1.08),columnspacing=1.2,handletextpad=.3)
    fig.tight_layout(w_pad=2)
    fig.text(.01,-.02,footnote,fontsize=11.5,color='#53677A',va='top',linespacing=1.5)
    save(fig,name)

# S20: fixed full-H pure RTE candidates at T=.8 (same candidate for all precisions, n=9).
fullh_rows=list(csv.DictReader(io.StringIO(source(FULLH_SHA,FULLH_DIR+'/RESULTS.csv'))))
fullh={(r['system'],r['epsilon']):r for r in fullh_rows if r['comparison_role']=='FULL_H_FIXED'}
fullh_cmp=list(csv.DictReader(io.StringIO(source(FULLH_SHA,FULLH_DIR+'/comparisons.csv'))))
check_baselines(fullh_cmp,'0.8')
df_figure('0.8','df_precision',fullh,'全H純RTE（固定候補）',FULLH_SHA,FULLH_DIR+'/RESULTS.csv','',
  '全H純RTE：一体項の非scalar成分もsampling、決定論ブロック空。B3のT=.8・ε=.0005探索nominee（H4 q16/R16/K4、H6 q32/R32/K6）を3精度で共通に使う事前固定候補、各n=9。\n'
  '他の4系列は各方式の保存選択値（公式・保持数・q/R/Kは方式ごとに異なる）。T=.8、状態準備費用を除外。')
for c in fullh_cmp:
    if c['epsilon'] in eps:
        record('df_full_h_ratio_table',c['epsilon'],c['role'],c['system'],'G_ratio',c['G_ratio'],'ratio',FULLH_SHA,FULLH_DIR+'/comparisons.csv',
               f"saved ratio full_H/baseline;baseline_id={c['baseline_id']};interpretation={c['interpretation']}")
for r in fullh_rows:
    if r['comparison_role'] in ('FULL_H_FIXED','MATCHED_B3') and r['epsilon']=='0.05':
        for metric in ('lambda_R','B','N_cos','N_sin','C_cos'):
            record('df_full_h_text','T=.8',r['comparison_role'],r['system'],metric,r[metric],'',FULLH_SHA,FULLH_DIR+'/RESULTS.csv',f"id={r['id']};q={r['q']};R={r['R_outer']};K={r['K']}")

# S21: T=2.0 q/R/K search, point-best among priced candidates (not a global optimum).
t20_best={(r['system'],r['epsilon']):r for r in csv.DictReader(io.StringIO(source(FULLH_T20_SHA,FULLH_T20_DIR+'/best_configurations.csv')))}
t20_cmp=list(csv.DictReader(io.StringIO(source(FULLH_T20_SHA,FULLH_T20_DIR+'/comparisons.csv'))))
t20_baseline={(c['system'],c['epsilon'],c['role']):c for c in t20_cmp}
check_baselines(t20_cmp,'2.0')
df_figure('2.0','df_full_h_t20',t20_best,'全H純RTE（探索の点最良）',FULLH_T20_SHA,FULLH_T20_DIR+'/best_configurations.csv','T=2.0 · ',
  '全H純RTE：T=2.0でq/R/Kを探索し、価格取得した10候補（各n=9）内の精度別点最良。全域最適ではない。他の4系列は各方式の保存選択値。\n'
  'T=.8の固定候補とは選び方が異なり、同じ最適化条件の時間変化としては読まない。状態準備費用を除外。')
for c in t20_cmp:
    if c['epsilon'] in eps:
        record('df_full_h_t20_ratio_table',c['epsilon'],c['role'],c['system'],'G_ratio',c['G_ratio'] or None,'ratio',FULLH_T20_SHA,FULLH_T20_DIR+'/comparisons.csv',
               f"saved ratio full_H/baseline;baseline_id={c['baseline_id']};status={c['comparison_status']}")
conf=json.loads(source(FULLH_T20_SHA,FULLH_T20_DIR+'/confirmation_summary.json'))
for conf_key in ('G_R448','G_R512','ratio','difference','difference_MCSE'):
    record('df_full_h_t20_confirmation','H4/epsilon=.0005/q8/K2','R448 vs R512',conf['trials'],conf_key,conf[conf_key],'RZ' if conf_key!='ratio' else 'ratio',
           FULLH_T20_SHA,FULLH_T20_DIR+'/confirmation_summary.json','fixed two candidates, n=33 each; internal to pure RTE; not method superiority')
for j,v in enumerate(conf['normal_95_diagnostic']):
    record('df_full_h_t20_confirmation','H4/epsilon=.0005/q8/K2','R448 vs R512',conf['trials'],['diagnostic_lower','diagnostic_upper'][j],v,'RZ',
           FULLH_T20_SHA,FULLH_T20_DIR+'/confirmation_summary.json','existing normal-approx finite-sample diagnostic; no new interval')

# S1 §6.3, two representative cases. Only add the stored Ncos and Nsin for display.
cases=[
 ('H4・ε=.05',[(14384,13570,'10291/24','47945769/4'),(19369,19370,'41639/128','1613053221/128')]),
 ('H6・ε=.0005',[(177727546,247950732,'299249/64','63691899506611/32'),(152683318,152680785,'149519/48','45657735316457/48')])]
fig,axs=plt.subplots(2,3,figsize=(13.2,7.2))
for row,(name,values) in enumerate(cases):
    for col,(metric,label) in enumerate([('N','予定shot数 Ncos+Nsin'),('C','回路単価 C [RZ/shot]'),('G','総費用 G [RZ]')]):
        ax=axs[row,col]
        tokens=[str(a+b) if col==0 else c if col==1 else g for a,b,c,g in values]
        ys=[num(v) for v in tokens]
        ax.bar([0,1],ys,color=[BLUE,TEAL],width=.54)
        for j,v in enumerate(ys):ax.text(j,v*1.035,f'{v:.3g}',ha='center',fontsize=13)
        ax.set(xticks=[0,1],xticklabels=['Hybrid PR','純RTE'],ylim=(0,max(ys)*1.24),title=(name+'\n' if col==0 else '')+label)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));grid(ax)
        for label_,token in zip(['Hybrid PR','純RTE'],tokens):
            record('ncg_cases',name,label_,metric,metric,token,'shots' if col==0 else 'RZ/shot' if col==1 else 'RZ',PAULI_SHA,PAULI_PATH,'section6.3 saved table; N is sum of saved axes')
fig.tight_layout(h_pad=2.2,w_pad=2);save(fig,'ncg_cases')

# S2 already selected rows by formula: no new min/search.
long=list(csv.DictReader(io.StringIO(source(LONG_SHA,LONG_PATH))))
fig,axs=plt.subplots(2,2,figsize=(12.2,7.6))
for row,e in enumerate(['0.05','0.0005']):
    for col,mol in enumerate(['H4','H6']):
        ax=axs[row,col]
        for family,formula,label,color,style in [
          ('PF','2nd','二次PF',AMBER,'--s'),('PF','4th(m5_best)','四次PF m5',AMBER,'-D'),
          ('PR','2nd','二次Hybrid PR',BLUE,'--o'),('PR','4th(m5_best)','四次Hybrid PR m5',TEAL,'-^')]:
            xs=[];ys=[]
            for t in ['0.8','1.2','1.6','2.0']:
                sel=[r for r in long if (r['system'],r['T'],r['epsilon'],r['family'],r['formula_group'])==(mol,t,e,family,formula)]
                assert len(sel)==1
                r=sel[0];xs.append(float(t));ys.append(num(r['G']))
                record('high_order_absolute',mol+'/epsilon='+e,label,t,'G',r['G'],'RZ',LONG_SHA,LONG_PATH,'candidate='+r['id']+';cohort='+r['cohort']+';LD='+r['prefix']+';q='+r['q']+';R='+r['R_outer']+';K='+r['K'])
            ax.plot(xs,ys,style,color=color,label=label,lw=2,ms=6)
        ax.set(title=mol+' · DF'+('10' if mol=='H4' else '19')+' / ε='+e,xlabel='発展時間 T',xticks=[.8,1.2,1.6,2],yscale='log')
        if col==0:ax.set_ylabel('測定込み総費用 G [RZ]')
        grid(ax)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.03))
fig.tight_layout(h_pad=2,w_pad=2);save(fig,'high_order_absolute')

# S4: use the two saved PR cohorts separately, with existing one-SE whiskers.
h8=json.loads(source(H8_SHA,H8_PATH))
items=[h8['fixed_PF'],h8['exploration_comparison']['saved_Stage_B'],h8['PR_summary']]
fig,ax=plt.subplots(figsize=(10.8,4.8))
ys=[num(x['mean_G_exact'])/1e12 for x in items]
labels=['固定PF対照','PR・探索価格','PR・独立確認価格']
ax.barh(range(3),ys,color=[AMBER,'#A6C6E2',TEAL],height=.52)
for j,(row,v) in enumerate(zip(items,ys)):
    se=row['sample_SE_G']
    if se is not None:ax.errorbar(v,j,xerr=se/1e12,fmt='none',ecolor='#19334D',capsize=5,lw=1.5)
    ax.text(v+1.4,j,f'{v:.3f}',va='center',fontsize=15,fontweight='bold')
    record('h8_cost','H8/T1.6/epsilon=.0005',labels[j],j,'G',row['mean_G_exact'],'RZ',H8_SHA,H8_PATH,['/fixed_PF','/exploration_comparison/saved_Stage_B','/PR_summary'][j])
    if se is not None:record('h8_cost','H8/T1.6/epsilon=.0005',labels[j],j,'sample_SE_G',se,'RZ',H8_SHA,H8_PATH,'existing sample_SE_G; no new interval')
ax.set(yticks=range(3),yticklabels=labels,xlabel='測定込み総RZ費用 G [10^12 RZ]',xlim=(0,105));ax.invert_yaxis()
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();save(fig,'h8_cost')

# Explanatory diagrams: symbolic blocks only, no molecular coefficient data.
def diagram():
    fig,ax=plt.subplots(figsize=(12.2,5.6));ax.set(xlim=(0,12.2),ylim=(0,5.6));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,title,body,color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.07',facecolor='#F5F8FB',edgecolor=color,lw=1.6))
    ax.text(x+.18,y+h-.24,title,color=color,fontsize=17,fontweight='bold',va='top')
    ax.text(x+.18,y+h-.84,body,fontsize=15,va='top',linespacing=1.55)
def arrow(ax,x1,y1,x2,y2,color=GRAY):
    ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',color=color,lw=2))

fig,ax=diagram()
box(ax,.15,2.0,2.8,2.4,'分子の二電子積分','g[p,q,r,s]\n4添字の相互作用',BLUE)
arrow(ax,3.1,3.2,3.9,3.2)
for j in range(3):
    x=4.1+2.55*j
    box(ax,x,2.0,2.3,2.4,['因子 0','因子 1','… 因子 R−1'][j],'基底変換 U_t\n＋ 係数 J_t',TEAL)
ax.text(6.5,4.95,'一般core Jを持つCDFの模式図',ha='center',fontsize=18,fontweight='bold')
ax.text(6.15,1.23,'通常CDF：gとg_DFのずれ f を小さくする',ha='center',fontsize=19)
ax.text(6.15,.55,'R：因子数　　U_t：回転する基底　　J_t：基底内の相互作用係数',ha='center',fontsize=15)
save(fig,'df_representation')

fig,ax=diagram()
box(ax,.15,2.0,3.45,2.9,'通常CDF（γ=0）','同じJ/U表現を使う\n\n目的：tensor fit誤差 f',DF_CDF_COLOR)
box(ax,4.05,2.0,3.45,2.9,'PR-aware DF','通常CDFの目的に追加\n\n写像後tail massのpenalty',DF_AWARE_COLOR)
box(ax,7.95,2.0,3.95,2.9,'ENTRY DF','通常CDFの目的に追加\n\nJのentry massのpenalty',DF_ENTRY_COLOR)
ax.text(6.1,1.2,'共通の形： F = f/f* + γ (M/M*)²',ha='center',fontsize=23,fontweight='bold')
ax.text(6.1,.45,'どちらもJ/Uを調整し、物理DF誤差予算を別に検査する',ha='center',fontsize=16)
save(fig,'df_objectives')

fig,ax=diagram()
box(ax,.15,1.95,2.3,2.8,'共通のJ_t','対角・非対角\n符号を持つ係数',DF_CDF_COLOR)
arrow(ax,2.6,4.2,3.4,4.2)
arrow(ax,2.6,2.8,3.4,2.8)
box(ax,3.65,3.3,4.0,1.8,'PR-aware','J → Z/ZZ係数へ写像',DF_AWARE_COLOR)
box(ax,3.65,.9,4.0,1.8,'ENTRY','Jのentryごとに重み付け',DF_ENTRY_COLOR)
arrow(ax,7.8,4.2,8.5,4.2);arrow(ax,7.8,1.8,8.5,1.8)
box(ax,8.7,3.3,3.2,1.8,'tail mass','線形Z：行和 → | |\nZZ：entry → | |',DF_AWARE_COLOR)
box(ax,8.7,.9,3.2,1.8,'entry mass','entryごと | |\n→ 重み付き和',DF_ENTRY_COLOR)
ax.text(6.1,.25,'線形ZにはJの行和が現れる。ZZ側はentryに比例。実目的は平滑化を使用。',ha='center',fontsize=13)
save(fig,'mass_pipeline')

fig,ax=diagram()
ax.text(.15,5.12,'再配置前',fontsize=20,fontweight='bold')
ax.text(.15,2.57,'再配置後',fontsize=20,fontweight='bold')
box(ax,.25,3.28,3.1,1.45,'決定論 head','一体項 ＋ 保持因子',BLUE)
box(ax,4.2,3.28,7.5,1.45,'ランダム tail（非identity）','線形Z ｜ ZZ',TEAL)
box(ax,.25,.72,5.45,1.45,'決定論 headへ追加','一体項 ＋ 保持因子 ＋ tailの線形Z',BLUE)
box(ax,6.35,.72,5.35,1.45,'ランダム tailに残る','ZZ',TEAL)
arrow(ax,7.7,3.15,3.8,2.3,AMBER)
ax.text(8.8,2.43,'線形Zと対応scalarを移す',color=AMBER,ha='center',fontsize=15)
ax.text(6.1,.22,'Iは既知の位相として別処理。各候補の全Hを保存する。',ha='center',fontsize=14)
ax.text(6.1,-.18,'(H_D + L) + (H_R − L) = H_D + H_R',ha='center',fontsize=17)
save(fig,'z_relocation')

# Side-by-side saved comparisons. Different r policies are stated, not treated
# as a controlled causal intervention or pooled statistical comparison.
fig,axs=plt.subplots(1,2,figsize=(12.0,4.8),sharey=True)
for ax,title,values,labs,sha in [
 (axs[0],'Z再配置前・各目的でrを選択',[93007350398.25,170559917333],['PR-aware\nr16','ENTRY\nr64'],'417fa459b823509be5c2010c5836e58955088282'),
 (axs[1],'Z再配置後・各目的でrを選択',[87514417699,75621395471],['PR-aware\nr16','ENTRY\nr16'],'3015d4fb342d4d98b58e41d57553e75539a07655')]:
    ys=[v/1e9 for v in values];ax.bar(range(2),ys,color=[DF_AWARE_COLOR,DF_ENTRY_COLOR],width=.55)
    for j,v in enumerate(ys):ax.text(j,v+3,f'{v:.3f}',ha='center',fontsize=15)
    ax.set(title=title,xticks=[0,1],xticklabels=labs,ylim=(0,200));grid(ax)
    report_path=('docs/research/research_c_integrated_h4_100_results_20261106.md'
                 if sha.startswith('417fa459') else
                 'docs/research/research_c_h4_relocated_downstream_results_20261106.md')
    for j,v in enumerate(values):record('dictionary_comparison',title,labs[j],j,'G',v,'RZ',sha,report_path,'main fair-retuned saved comparison; separate policies/cohorts')
axs[0].set_ylabel('条件付き総RZ [10^9 RZ]');fig.tight_layout(w_pad=2);save(fig,'dictionary_comparison')

# Quote the published three-method figure byte for byte; do not rerun its study.
for ext in ('png','svg'):
    blob=subprocess.check_output(['git','show',f'{THREE_METHOD_SHA}:{THREE_METHOD_DIR}/comparison_total_rz.{ext}'])
    assert blob==subprocess.check_output(['git','show',f'{THREE_METHOD_INITIAL_SHA}:{THREE_METHOD_DIR}/comparison_total_rz.{ext}'])
    (OUT/f'df_three_method_z_relocated.{ext}').write_bytes(blob)
three_table=source(THREE_METHOD_SHA,THREE_METHOD_DIR+'/REPORT.md').split('## 4. 主比較表：',1)[1].split('## 5.',1)[0]
for line in three_table.splitlines():
    if not line.startswith(('| 通常CDF0 |','| PR-aware P |','| ENTRY E2 |')):
        continue
    cells=[x.strip().replace('**','') for x in line.strip('|').split('|')]
    label,r,total=cells[0],cells[1],cells[2].replace(',','')
    record('df_three_method_z_relocated','Z再配置後',label,r,'G',total,'RZ',
           THREE_METHOD_SHA,THREE_METHOD_DIR+'/REPORT.md',
           f'section4 saved {label}/r={r}; published PNG/SVG reused unchanged')

# Separate fresh128 two-arm table (main slide 12). Not merged into the three-method figure.
indep=json.loads(source(INDEP_SHA,INDEP_SUMMARY))
comparison=indep['result']['comparison']
for arm,point_key,arm_key in (('通常CDF0','PR','CDF0'),('ENTRY E2','ENTRY','ENTRY')):
    record('df_cdf0_entry_independent128','fresh128 each',arm,indep['arms'][arm_key]['r'],'G',comparison['total_points'][point_key],'RZ',INDEP_SHA,INDEP_SUMMARY,
           f"/result/comparison/total_points/{point_key} (legacy PR key means CDF0); samples={comparison['samples_per_arm']}; old samples pooled=False")
for metric,token in (('halfwidth',comparison['halfwidth']),('difference_point',comparison['difference_point']),
                     ('difference_lower',comparison['difference_interval'][0]),('difference_upper',comparison['difference_interval'][1])):
    record('df_cdf0_entry_independent128','fresh128 each','CDF0 minus ENTRY','',metric,token,'RZ',INDEP_SHA,INDEP_SUMMARY,
           f"/result/comparison; decision={comparison['decision']}; existing nominal 95% Welch output, not recomputed")
fraction=source(INDEP_SHA,INDEP_REPORT).split('半幅/新ENTRY点値=',1)[1].split('≤',1)[0]
record('df_cdf0_entry_independent128','fresh128 each','CDF0 minus ENTRY','', 'halfwidth_over_new_ENTRY',fraction,'fraction',INDEP_SHA,INDEP_REPORT,
       'section4 saved text; registered target 0.05')

fields=['figure','panel','series','x','metric','saved_value','display_value','unit','source_commit','source_path','selector']
with (OUT/'plot_values.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(DISPLAY)
with (OUT/'df_selected_configs.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(df_configs[0]));w.writeheader();w.writerows(df_configs)
print('Created 11 figures, copied the published three-method PNG/SVG, and wrote the saved-value citation table.')
