"""Render saved display rows only. No scientific modules or data accounting.

Adapted display patterns: plot_pauli_rte_event_zoom.py (Fraction-to-float,
log axes, distinct markers) and h46_long_time_report.py's figure section
(faceted axes). Their statistics, selection and doubled-SE paths are excluded.
"""
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
import textwrap

OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'o2-visualization-mpl-20261106'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

EPS = ['0.05', '0.01', '0.005', '0.001', '0.0005']
TIMES = ['0.8', '1.2', '1.6', '2.0']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#555555']
MISSING = {'', 'null', 'None', 'UNKNOWN', 'NOT_ACQUIRED', 'NOT_RECORDED'}


def number(literal):
    """Allowed display conversion, including original rational strings."""
    return None if literal in MISSING else float(Fraction(literal))


def load_points(fid):
    return [p for p in csv.DictReader((OUT / 'DISPLAY_DATA.csv').open()) if p['figure_id'] == fid]


def style():
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'o2-saved-20261106',
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'figure.facecolor': 'white'})


def axis(ax, ylabel, *, epsilon=False, log=True):
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', which='major', alpha=.18)
    if log:
        ax.set_yscale('log')
    if epsilon:
        ax.set_xscale('log')
        ax.set_xlim(.075, .00035)
        ax.xaxis.set_major_locator(FixedLocator([number(e) for e in EPS]))
        ax.xaxis.set_major_formatter(FixedFormatter(EPS))
        ax.minorticks_off()
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel('Required signal precision ε')


def show_point(ax, p, bindings, *, x=None, color=COLORS[0], marker='o', hollow=False, log=True):
    y = number(p['y_literal'])
    x = number(p['x_literal']) if x is None else x
    err = number(p['se_literal'])
    if y is None:
        item = ax.text(x, .04, 'missing', transform=ax.get_xaxis_transform(), fontsize=7,
                       rotation=90, color='#666666')
        kind = 'missing_literal'
    elif log and y == 0:
        item = ax.text(x, .035, 'saved 0', transform=ax.get_xaxis_transform(), ha='center',
                       fontsize=8, color=color)
        kind = 'zero_not_placed_on_log_axis'
    else:
        options = dict(marker=marker, color=color, markersize=5,
                       markerfacecolor='white' if hollow else color, linestyle='none',
                       zorder=4 if hollow else 3)
        if err is None:
            item, = ax.plot([x], [y], **options)
        else:
            container = ax.errorbar([x], [y], yerr=[err], capsize=2.5, elinewidth=.9, **options)
            item = container.lines[0]
        assert float(item.get_xdata()[0]) == x and float(item.get_ydata()[0]) == y
        kind = 'numeric_artist'
    item.set_gid(p['point_id'])
    bindings.append(dict(point_id=p['point_id'], kind=kind, x=x, y=y, se=err,
                         y_literal=p['y_literal'], se_literal=p['se_literal']))


def guide(ax, group, *, color, linestyle='-', categorical=None):
    # The order is the display axis, never a candidate cost ranking.
    order = EPS if group and group[0]['x_literal'] in EPS else TIMES
    if categorical is not None:
        order = categorical
    positions = {x: i for i, x in enumerate(order)}
    rr = sorted(group, key=lambda p: positions[p['x_literal']])
    xs = [positions[p['x_literal']] if categorical is not None else number(p['x_literal']) for p in rr]
    ys = [number(p['y_literal']) for p in rr]
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.15, alpha=.8, zorder=1)


def header(fig, fid, title):
    fig.text(.035, .973, fid + '  |  DRAFT · SAVED VALUES ONLY', fontsize=10, color='#666666')
    fig.text(.035, .935, title, fontsize=16, weight='bold')


def footer(fig, lines):
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=170))
    fig.text(.035, .028, '\n'.join(wrapped), fontsize=8.4, color='#333333', va='bottom', linespacing=1.4)


def legend(fig, handles):
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .902), ncol=len(handles), fontsize=9)


def handle(label, color, marker, hollow=False, line='-'):
    return Line2D([], [], label=label, color=color, marker=marker, linestyle=line,
                  markerfacecolor='white' if hollow else color, markersize=5)


def build(fid):
    style()
    points = load_points(fid)
    bindings = []
    if fid == 'V01':
        fig, axs = plt.subplots(3, 4, figsize=(18, 12))
        scopes = ['2nd', '4th', '4th(m5_best)', 'ALL_FORMULAS']
        for i, system in enumerate(['H2', 'H4', 'H6']):
            for j, scope in enumerate(scopes):
                ax = axs[i, j]
                panel = system + '|' + scope
                for series, color, mark in [('PF', COLORS[0], 's'), ('PR', COLORS[1], 'o')]:
                    pp = [p for p in points if p['panel_id'] == panel and p['series_id'] == series]
                    guide(ax, pp, color=color, linestyle='--' if series == 'PF' else '-')
                    for p in pp:
                        show_point(ax, p, bindings, color=color, marker=mark)
                axis(ax, 'G [compiled RZ × shots]', epsilon=True)
                ax.set_title(system + ' · ' + {'2nd':'S2', '4th':'Y4', '4th(m5_best)':'m5', 'ALL_FORMULAS':'all formulas'}[scope])
        header(fig, fid, 'DF: strong PF comparators across precision conditions')
        legend(fig, [handle('Saved PF selection', COLORS[0], 's', line='--'), handle('Saved PR exploration mean ± 1 price SE', COLORS[1], 'o')])
        footer(fig, ['H2/H4/H6: linear 1 Å, STO-3G, tol-only DF ranks 3/10/19; H6 weighted-Hermitian completion. T=0.8; P=0; full measured Hadamard RZ; Qiskit1.3/opt1/seed17.',
                     'All 60 saved comparison rows are shown: original common 588-PF domain and saved PR subset. Original formula scopes are separate; no candidate selection rerun.',
                     'N is analytic sufficient quantum shots; SE is from classical price samples, not a population confidence interval. Shared prices across precision; no system-size scaling claim.'])
        fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.16, hspace=.62, wspace=.35)
    elif fid in {'V02', 'V03'}:
        fig, axs = plt.subplots(2, 5, figsize=(19, 10))
        if fid == 'V02':
            series_styles = [('PF_exploration',COLORS[0],'s',False,'--'),
                             ('PR_exploration',COLORS[1],'o',False,'-'),
                             ('PR_fixed_confirmation',COLORS[1],'D',True,None),
                             ('PF_saved_comparator',COLORS[0],'^',True,None)]
        else:
            series_styles = [('B3',COLORS[1],'o',False,'-'),('B2',COLORS[0],'s',False,'--'),
                             ('Full_PF',COLORS[2],'^',False,':'),('PF_envelope',COLORS[3],'D',True,'-.')]
        for i, system in enumerate(['H4','H6']):
            for j, e in enumerate(EPS):
                ax = axs[i,j]
                for series, color, mark, hollow, line in series_styles:
                    pp = [p for p in points if p['panel_id']==system+'|'+e and p['series_id']==series]
                    if line:
                        guide(ax, pp, color=color, linestyle=line)
                    for p in pp:
                        show_point(ax,p,bindings,color=color,marker=mark,hollow=hollow)
                axis(ax,'G [compiled RZ × shots]')
                ax.set_xticks([number(t) for t in TIMES],TIMES)
                ax.set_xlabel('T [source time convention]')
                ax.set_title(system+' · ε='+e)
        if fid == 'V02':
            header(fig,fid,'DF: time / precision with exploration and fixed confirmation kept distinct')
            legend(fig,[handle('PF saved deterministic',COLORS[0],'s',line='--'),
                        handle('PR exploration ±1 saved SE',COLORS[1],'o'),
                        handle('PR fixed confirmation mean',COLORS[1],'D',True,line='none'),
                        handle('PF saved confirmation comparator',COLORS[0],'^',True,line='none')])
            footer(fig,['H4 DF10 / H6 DF19 (weighted completion), linear1 Å/STO-3G; original T and 5 precisions; P=0; full Hadamard RZ, opt1/seed17. ALL is the original saved selection scope.',
                        '40 exploration cells; the fixed comparison CSV contains 33 PR-vs-PF confirmation records. No confirmation is imputed for the other cells; no pooling or new success rate.',
                        'Confirmation CSV stores SE of the comparison difference, not separate-side SE. Hollow means therefore have no invented individual error bars. Old 33/40, G01 expansion and R-boundary sets remain separate.'])
        else:
            header(fig,fid,'DF B3: saved finite-domain negative comparisons')
            legend(fig,[handle(s,c,m,h,line=l) for s,c,m,h,l in series_styles])
            footer(fig,['H4 DF10 / H6 DF19 (weighted completion), linear1 Å/STO-3G; P=0; full Hadamard RZ, opt1/seed17. B3 retains the one-body block; it is not whole-Hamiltonian pure RTE.',
                        'All original 40 cells and three saved baseline families are shown. The repeated B3 value is drawn once per cell; baseline cohorts are retained in DISPLAY_DATA.csv.',
                        'Original finite-domain result: B3 lower G in 0/40 conditions against each saved baseline family. No universal impossibility claim. Only comparison-difference SE is saved here; no individual SE is invented.'])
        fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.2,hspace=.55,wspace=.4)
    elif fid == 'V04':
        fig, ax = plt.subplots(figsize=(11,7))
        for i, arm in enumerate(['PR','PF']):
            p = next(p for p in points if p['series_id']==arm)
            show_point(ax,p,bindings,x=i,color=COLORS[i],marker='o' if arm=='PR' else 's',log=False)
            ax.annotate('Saved G = '+p['y_literal'],(i,number(p['y_literal'])),xytext=(0,18),textcoords='offset points',ha='center',fontsize=10)
        axis(ax,'G [compiled RZ × analytic shots]',log=False)
        ax.set_ylim(0,1.1e14);ax.set_xlim(-.6,1.6)
        ax.set_xticks([0,1],['PR · Stage C confirmation n=9','Fixed PF · saved Stage B price n=1'])
        header(fig,fid,'H8 Stage C: fixed PR / PF saved resource comparison')
        ax.text(.04,.91,'Saved observed reduction: 24.58909439%',transform=ax.transAxes,fontsize=11)
        footer(fig,['H8/1 Å/STO-3G/tol-only DF28; T=1.6; ε=0.0005; P=0; full Hadamard RZ, Qiskit1.3/opt1/seed17.',
                    'PR m5/p8/q1/R32/K4 vs PF B0I/m5/p15/q1. PR error bar: ±1 saved paired-price G SE = 103180211675.02368 RZ×shot. PF sample SE is null, not zero.',
                    'PR confirmation n=9 is not pooled with exploration n=9. PR mean C is not reconstructed. Unobserved rare-order costs and population guarantees remain unknown.'])
        fig.subplots_adjust(left=.12,right=.96,top=.83,bottom=.25)
    elif fid == 'V05':
        fig,axs=plt.subplots(3,3,figsize=(17,13))
        for j,system in enumerate(['H2','H4','H6']):
            for i,quantity in enumerate(['G','N','C']):
                ax=axs[i,j]
                for role,color,mark in [('Hybrid',COLORS[0],'s'),('RTE',COLORS[1],'o')]:
                    for component in ([''] if quantity=='G' else ['cos','sin']):
                        series=role+('|' + component if component else '')
                        pp=[p for p in points if p['panel_id']==system+'|'+quantity and p['series_id']==series]
                        guide(ax,pp,color=color,linestyle=':' if component=='sin' else '-')
                        for p in pp:
                            show_point(ax,p,bindings,color=color,marker=mark if component!='sin' else '+',hollow=component=='sin')
                axis(ax,{'G':'G [compiled RZ × shots]','N':'N per axis [analytic quantum shots]','C':'C per axis [compiled RZ/shot]'}[quantity],epsilon=True,log=quantity!='C')
                if quantity=='C':
                    ax.set_ylim(bottom=0)
                if system=='H4' and quantity=='G':
                    ax.text(.03,.9,'Saved exception: ε=.05 Hybrid lower',transform=ax.transAxes,fontsize=9)
                ax.set_title(system+' · '+quantity)
        header(fig,fid,'Pauli-JW: saved Hybrid / pure-RTE comparisons and N / C / G')
        legend(fig,[handle('Hybrid',COLORS[0],'s'),handle('Pure RTE',COLORS[1],'o'),handle('cosine axis', '#555555','o'),handle('sine axis (may overlap)', '#555555','+',line=':')])
        footer(fig,['H2/H4/H6: linear1 Å/STO-3G; Pauli14/184/918; saved molecule-specific H/state/reference; T=.8/S2; central control; P=0; Qiskit1.3/opt1/seed17; hypothetical compiled ρ=0.',
                    'Original saved comparison direction: pure RTE lower G in 14/15 conditions; H4 ε=.05 is the Hybrid exception. H6 uses the later p78/q8/r32/K2 Hybrid, not the older anchor.',
                    'G bars are ±1 saved price-sample SE, not population intervals. N is analytic, not observed shots. Same prices are reused across precision; these are not 15 independent experiments.',
                    'Original fixed nominees / confirmation-after-selection / sequential extension are retained in the data manifest. Pauli constant is retained. Coincident cos/sin means are drawn with separate markers.'])
        fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.18,hspace=.55,wspace=.3)
    elif fid == 'V06':
        fig,axs=plt.subplots(1,3,figsize=(17,7.5))
        for index,quantity in [(0,'N'),(2,'G')]:
            ax=axs[index]
            for arm,color,mark in [('P',COLORS[0],'s'),('DF',COLORS[1],'o')]:
                for component in ([''] if quantity=='G' else ['cos','sin']):
                    pp=[p for p in points if p['panel_id']==quantity and p['series_id']==arm+('|' + component if component else '')]
                    guide(ax,pp,color=color,linestyle=':' if component=='sin' else '-')
                    for p in pp:show_point(ax,p,bindings,color=color,marker=mark if component!='sin' else '+')
            axis(ax,'N per axis [analytic quantum shots]' if quantity=='N' else 'G [compiled RZ × shots]',epsilon=True)
            ax.set_title('Saved '+quantity)
        ax=axs[1]
        for i,arm in enumerate(['P','DF']):
            for axisname,mark in [('cosine','o'),('sine','+')]:
                p=next(p for p in points if p['panel_id']=='C' and p['series_id']==axisname and p['x_literal']==arm)
                show_point(ax,p,bindings,x=i,color=COLORS[i],marker=mark,log=False)
            ax.text(i, .06, 'n=32',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
        axis(ax,'C [compiled RZ/shot]',log=False);ax.set_ylim(0,10500);ax.set_xlim(-.5,1.5)
        ax.set_xticks([0,1],['Pauli residual','DF residual']);ax.set_title('Original saved price means ±1 SE')
        header(fig,fid,'H4 P3 · CONDITIONAL_MATCHED · saved N / C / G')
        legend(fig,[handle('Pauli residual (P)',COLORS[0],'s'),handle('DF residual',COLORS[1],'o'),handle('sine N/C (may overlap)', '#555555','+',line=':')])
        footer(fig,['H4/1 Å/STO-3G/DF10; common raw ray and Gaussian T_eff hold; V_square residual; Pauli184 vs DF360 rotated Z/ZZ in 10 bases; T=.8/S2/q8/r4/K4; P=0; opt1/seed17.',
                    'Assumptions: finite-bit law/angle gap = 0; compiled residual ρ = 0; B engineering-candidate coverage. CONDITIONAL_MATCHED does not establish a representation-only causal effect.',
                    'Each arm uses its original Stage3/run04 n=32 prices across all five precisions. G bars: ±1 saved price SE. C comes from the exact price_statistics.json input of Stage4, not G/N or a new average.',
                    'No population ordering, implementation-wide accuracy or finite-bit guarantee. The finite-bit addendum does not update these N/G values or discharge the assumptions.'])
        fig.subplots_adjust(left=.065,right=.98,top=.79,bottom=.3,wspace=.35)
    elif fid == 'V07':
        fig,ax=plt.subplots(figsize=(15,9));ax.axis('off')
        content=[
            ['Deterministically retained','One-body Gaussian block + scalar','Constant scalar only'],
            ['Randomized tail','DF two-body fragments (rotated Z/ZZ)','All nonidentity Pauli terms'],
            ['Basis / native structure','Rotated Gaussian bases / DF fragments','Jordan–Wigner Pauli products'],
            ['Law / prefix meaning','DF L_D=0 excludes one-body from tail','Pauli L_D=0 leaves no nonidentity hold'],
            ['Saved systems / representation','H4 DF10; H6 DF19 (weighted completion)','H2/H4/H6: Pauli14/184/918'],
            ['Time conditions','T=.8/1.2/1.6/2 in the saved study','T=.8 in the saved study'],
            ['H / state / reference / cohorts','Preserved within the DF studies','Preserved within the Pauli studies'],
            ['Control / compilation','DF-native symmetric-directional; full wrapper','Pauli central; full wrapper'],
            ['Permitted use','Within-study B3 negative comparisons','Within-study Hybrid / pure-RTE comparisons'],
            ['Cross-study classification','CONTEXTUAL','Not a shared-program cost ranking'],
        ]
        tab=ax.table(cellText=content,colLabels=['Comparison item','DF B3','Pauli pure RTE'],cellLoc='left',colLoc='left',bbox=[.015,.035,.97,.88],colWidths=[.24,.38,.38])
        tab.auto_set_font_size(False);tab.set_fontsize(10)
        for (r,c),cell in tab.get_celld().items():
            cell.set_edgecolor('#D0D4D7');cell.PAD=.07
            if r==0:cell.set_facecolor('#E8EFF4');cell.set_text_props(weight='bold')
            elif r%2==0:cell.set_facecolor('#F5F7F8')
        header(fig,fid,'Different randomized boundaries: definition and comparability table')
        footer(fig,['XP04 · Original matching classification: CONTEXTUAL. Tail, basis, law, control, H/state/reference and price cohorts are not jointly matched.',
                    'This table contains no new numerical cost comparison. It does not establish a DF–Pauli representation-only causal effect, a universal pure-RTE result or a common winner.',
                    'The exact original matching row and scientific/result references are retained in FIGURE_MANIFEST.csv and FIGURE_VALIDATION.json.'])
        fig.subplots_adjust(left=.025,right=.985,top=.86,bottom=.17)
    elif fid == 'V08':
        fig,axs=plt.subplots(1,3,figsize=(17,14))
        rows=[s+'|'+fam+'|'+t+'|'+formula for s in ['H4','H6'] for fam in ['PF','PR'] for t in TIMES for formula in ['4th','4th(m5_best)']]
        for ax,quantity in zip(axs,['N','C_eff','G']):
            values=[]
            for row in rows:
                vals=[]
                for e in EPS:
                    p=next(p for p in points if p['panel_id']==quantity and p['series_id']==row and p['x_literal']==e)
                    value=number(p['y_literal']);vals.append(value)
                values.append(vals)
            mesh=ax.imshow(values,aspect='auto',cmap='RdBu_r',norm=LogNorm(vmin=.001,vmax=1000),interpolation='none')
            ax.set_xticks(range(5),EPS);ax.set_xlabel('Required signal precision ε')
            ax.set_yticks(range(len(rows)),[r.replace('4th(m5_best)','m5').replace('4th','Y4').replace('|',' · ') for r in rows] if quantity=='N' else ['']*len(rows))
            ax.tick_params(axis='y',labelsize=8)
            ax.set_title('Saved higher / S2 '+quantity+' ratio')
            for i,row in enumerate(rows):
                for j,e in enumerate(EPS):
                    p=next(p for p in points if p['panel_id']==quantity and p['series_id']==row and p['x_literal']==e)
                    value=number(p['y_literal'])
                    artist=ax.text(j,i,'missing' if value is None else format(value,'.2g'),ha='center',va='center',fontsize=7,color='black')
                    artist.set_gid(p['point_id'])
                    bindings.append(dict(point_id=p['point_id'],kind='heatmap_saved_value',x=j,y=value,se=None,y_literal=p['y_literal'],se_literal=''))
        header(fig,fid,'Saved N / C_eff / G competition across higher-order formulas')
        footer(fig,['ME01 · H4DF10 / H6DF19 (weighted completion), linear1 Å/STO-3G; T=.8/1.2/1.6/2; P=0; original saved higher-vs-S2 selections, full Hadamard RZ/opt1/seed17.',
                    'All 160 original ratio rows are shown in three separate dimensionless panels. Values are saved ratios, not newly divided fields. Text is rounded for display only; exact strings remain in DISPLAY_DATA.csv.',
                    'C_eff is the original shot-weighted effective cost, not C_cos or C_sin. q/prefix may also change; these are finite-sample accounting comparisons, not causal factor shares. Color range has extended ends.'])
        fig.subplots_adjust(left=.16,right=.91,top=.86,bottom=.13,wspace=.13)
        cax=fig.add_axes([.935,.21,.012,.56])
        fig.colorbar(mesh,cax=cax,extend='both',label='Saved ratio · logarithmic color scale')
    elif fid == 'V09':
        fig,axs=plt.subplots(2,3,figsize=(16,10))
        for i,system in enumerate(['H4','H6']):
            rs=['16','64','128'] if system=='H4' else ['11','16']
            for j,quantity in enumerate(['N','C','G']):
                ax=axs[i,j]
                for series,color,mark in ([('G',COLORS[0],'o')] if quantity=='G' else [('cos',COLORS[0],'o'),('sin',COLORS[1],'+')]):
                    pp=[p for p in points if p['panel_id']==system+'|'+quantity and p['series_id']==series]
                    for p in pp:show_point(ax,p,bindings,x=rs.index(p['x_literal']),color=color,marker=mark,log=False)
                axis(ax,{'N':'N per axis [analytic shots]','C':'C per axis [compiled RZ/shot]','G':'G [compiled RZ × shots]'}[quantity],log=False)
                bounds={'H4':{'N':1.8e8,'C':60000,'G':1.2e13},'H6':{'N':1.4e8,'C':220000,'G':4.7e13}}
                ax.set_ylim(0,bounds[system][quantity]);ax.set_xlim(-.5,len(rs)-.5)
                ax.set_xticks(range(len(rs)),['R'+r for r in rs]);ax.set_title(system+' · '+quantity)
                ax.set_xlabel('Original R_outer allocation (discrete saved candidates)')
                if quantity=='G':
                    for p in pp:
                        metadata=json.loads(p['metadata_literals']);diag=metadata['diagnostic'] or 'SAVED BASELINE'
                        ax.annotate(diag+'\nn='+metadata['n'],(rs.index(p['x_literal']),number(p['y_literal'])),xytext=(0,17),textcoords='offset points',ha='center',fontsize=8)
        header(fig,fid,'DF R boundaries: fixed-condition saved N / C / G')
        legend(fig,[handle('cosine / G',COLORS[0],'o',line='none'),handle('sine (may overlap)',COLORS[1],'+',line='none')])
        footer(fig,['ME03/ME04 · H4DF10 L_D3/K2 and H6DF19 weighted completion L_D7/K6; linear1 Å/STO-3G; T2/q1/m5; ε=.0005; P=0; full measured Hadamard RZ, opt1/seed17.',
                    'Saved H4 diagnostics: R16 and R128 HIGHER than R64. Saved H6 R11: point_lower=True, diagnostic=INCONCLUSIVE. R64 is not asserted globally optimal.',
                    'G error bars are ±1 saved price SE. R11/R16 use different original n/cohorts. No new differences/ratios/CI; zero-based linear axes avoid magnifying the small H6 point gap. Boundary complete SOURCE remains private.'])
        fig.subplots_adjust(left=.075,right=.98,top=.81,bottom=.2,hspace=.65,wspace=.32)
    elif fid == 'V10':
        fig,axs=plt.subplots(1,4,figsize=(18,7.5))
        labels=['H16','H32','H64','R64']
        for ax,component in zip(axs,['outer','hold','trunc','grid']):
            for series,color in [('Absolute difference',COLORS[0]),('Diagnostic resolution',COLORS[1])]:
                for p in [p for p in points if p['panel_id']==component and p['series_id']==series]:
                    mark='x' if series=='Diagnostic resolution' else ('o' if p['original_status']=='RESOLVED_EMPIRICAL' else 'D')
                    show_point(ax,p,bindings,x=labels.index(p['x_literal']),color=color,marker=mark,hollow=series=='Absolute difference' and p['original_status']=='UNRESOLVED')
            axis(ax,'Saved |signal difference| / diagnostic resolution')
            ax.set_ylim(1e-16,1e-3);ax.set_xticks(range(4),labels);ax.set_xlim(-.5,3.5)
            ax.set_title({'outer':'Outer split','hold':'Commuting hold','trunc':'Finite RTE','grid':'Finite grid'}[component])
        header(fig,fid,'H6 P1: saved signal differences and their resolution limits')
        legend(fig,[handle('RESOLVED_EMPIRICAL difference',COLORS[0],'o',line='none'),
                    handle('UNRESOLVED difference',COLORS[0],'D',True,line='none'),
                    handle('Saved diagnostic resolution',COLORS[1],'x',line='none')])
        footer(fig,['ME05 · H6Pauli918, saved full-H state/reference; T=.8/S2/q8/K2; H16/H32/H64: prefix78,r16/32/64; R64: pure RTE,r64. Engineering diagnostic floor=1e-10.',
                    'Absolute differences and resolutions are both original saved fields. UNRESOLVED is not zero/equivalent. Literal zero differences for R64 outer/hold are labeled at the axis edge, not given a false positive log value.',
                    'The retained commuting hold has a structural-zero ideal PF contribution; small floating residuals remain separate. No new signal differences, N/G, confidence intervals or causal shares are computed.'])
        fig.subplots_adjust(left=.075,right=.98,top=.78,bottom=.28,wspace=.36)
    elif fid == 'V11':
        fig,axs=plt.subplots(3,2,figsize=(13,12))
        fields=['full','retained','event','empty']
        for i,label in enumerate(['H32','H64','R64']):
            for j,axisname in enumerate(['cosine','sine']):
                ax=axs[i,j];pp=[p for p in points if p['panel_id']==label+'|'+axisname]
                for k,field in enumerate(fields):
                    p=next(p for p in pp if p['series_id']==field);y=number(p['y_literal'])
                    artist=ax.bar(k,y,color=COLORS[k],width=.6,hatch='//' if field=='empty' else None)[0]
                    artist.set_gid(p['point_id']);assert float(artist.get_height())==y
                    bindings.append(dict(point_id=p['point_id'],kind='bar_artist',x=k,y=y,se=None,y_literal=p['y_literal'],se_literal=''))
                    ax.text(k,y,p['y_literal'],ha='center',va='bottom',fontsize=10)
                p=next(p for p in pp if p['display_kind']=='annotation')
                artist=ax.text(.97,.88,'Saved Δinteraction = '+p['y_literal']+' RZ',transform=ax.transAxes,ha='right',fontsize=10,
                               bbox=dict(facecolor='white',edgecolor='#DDDDDD',pad=5))
                artist.set_gid(p['point_id']);assert p['y_literal'] in artist.get_text()
                bindings.append(dict(point_id=p['point_id'],kind='saved_text_annotation',x=None,y=None,se=None,y_literal=p['y_literal'],se_literal=''))
                axis(ax,'Compiled RZ per saved program',log=False);ax.set_ylim(0,8500)
                ax.set_xticks(range(4),['Full','R: retained','E: event','Empty']);ax.set_title(label+' · '+axisname)
        header(fig,fid,'H6 P2: component prices are not an additive cost allocation')
        footer(fig,['ME06 · H6Pauli918; T=.8/S2/q8/K2; original confirmation/index0000 of H32/H64/R64. These are fixed-program observations, not population means or full-task N/G.',
                    'Full, retained R, event E and Empty are separate original compiled programs. Bars are not stacked. Δinteraction is quoted from the saved interaction table; it is not recomputed here.',
                    'Shared retained programs and compiler cancellations do not define universal causal shares. Original Qiskit1.3/opt1/seed17; no new sampling, circuit build or compile.'])
        fig.subplots_adjust(left=.095,right=.97,top=.86,bottom=.14,hspace=.53,wspace=.25)
    else:
        raise ValueError(fid)

    assert len(bindings)==len(points), (fid,len(bindings),len(points))
    svg=OUT/'figures'/ (fid+'.svg');png=OUT/'figures'/ (fid+'.png')
    fig.savefig(svg,metadata={'Date':'2026-11-06','Creator':'O2 saved-data display only','Title':fid})
    fig.savefig(png,dpi=130,metadata={'Software':'O2 saved-data display only'})
    plt.close(fig)
    metadata=dict(figure_id=fid,display_data_sha256=hashlib.sha256((OUT/'DISPLAY_DATA.csv').read_bytes()).hexdigest(),
                  rendered_fields=bindings,new_numeric_analysis=False,new_scientific_PASS=False)
    text=svg.read_text();tag=re.search(r'<svg\b[^>]*>',text).group()
    text=text.replace(tag,tag+'\n<metadata id="o2-saved-fields">'+html.escape(json.dumps(metadata,separators=(',',':')))+ '</metadata>',1)
    svg.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    return dict(figure_id=fid,display_rows=len(points),svg_bytes=svg.stat().st_size,png_bytes=png.stat().st_size)


def main():
    # Independent small rendering jobs, not scientific workers. No global settings changed.
    figure_ids=json.loads((OUT/'FIGURE_VALIDATION.json').read_text())['figures']
    style()  # Build the local font cache once before workers fork.
    with ProcessPoolExecutor(max_workers=len(figure_ids)) as pool:
        for result in pool.map(build,figure_ids):
            print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
