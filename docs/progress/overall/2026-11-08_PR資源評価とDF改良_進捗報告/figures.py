"""Presentation figures: display previously reported values, without reanalysis.

Run with the repository's .venv311/bin/python. No scientific project module is
imported. No fitting, selection, statistics, simulations, or new cost accounting.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'font.size': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#CFD7E0', 'axes.labelcolor': '#20354D',
                     'text.color': '#20354D', 'xtick.color': '#53657A',
                     'ytick.color': '#53657A', 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'svg.fonttype': 'none'})
TEAL, BLUE, AMBER, CORAL = '#009C91', '#446CB3', '#CE8530', '#C85B65'

def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / (name + '.' + ext), dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# S2 REPORT.md: "m5とS2の厳しい精度での比較". These are saved ratios.
ts = [.8, 1.2, 1.6, 2.0]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
for ax, name, pf, pr in [
    (axes[0], 'H4 · DF10', [1.0362, .5833, .3424, .3439], [1.0025, .5525, .3880, .3249]),
    (axes[1], 'H6 · DF19', [.9118, .6015, .2945, .3868], [.6902, .4168, .3348, .2475])]:
    ax.axhline(1, color='#8291A1', lw=1.4, ls='--')
    ax.plot(ts, pf, 's--', color=AMBER, label='決定論PF', ms=7, lw=2)
    ax.plot(ts, pr, 'o-', color=TEAL, label='Hybrid PR', ms=7, lw=2.4)
    for t, y in zip(ts, pr):
        ax.annotate(f'{y:.4f}', (t, y), xytext=(0, -20), textcoords='offset points', ha='center', fontsize=11, color=TEAL)
    ax.set(title=name, xlabel='発展時間 T', xticks=ts, ylim=(.1, 1.2))
    ax.grid(axis='y', alpha=.16)
axes[0].set_ylabel('保存総RZ費用比  m5 / S2')
axes[1].legend(frameon=False, loc='upper right', fontsize=11)
fig.tight_layout(w_pad=2)
save(fig, 'higher_order_time')

def bars(name, labels, values, colors, xmax=None):
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ys = list(range(len(values)))
    ax.barh(ys, values, height=.52, color=colors)
    ax.set(yticks=ys, yticklabels=labels, xlabel='条件付き総RZ  [10^9 RZ]', xlim=(0, xmax or max(values)*1.25))
    ax.tick_params(axis='y', labelsize=14)
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for y, v in zip(ys, values):
        ax.text(v + (xmax or max(values)*1.25)*.017, y, f'{v:.3f}', va='center', fontweight='bold', fontsize=15)
    fig.tight_layout()
    save(fig, name)

# S5 main C0/64 fair-retuned final64; division by 10^9 is display units only.
bars('df_original_dictionary', ['PR-aware DF ＋ PR', 'ENTRY DF ＋ PR'],
     [93007350398.25/1e9, 170559917333/1e9], [TEAL, BLUE], 210)
# S6 main comparison B; candidates were not refitted after relocation.
bars('df_relocated_dictionary', ['PR-aware由来 P ＋ PR/r16', 'ENTRY由来 E2 ＋ PR/r16'],
     [87514417699/1e9, 75621395471/1e9], [TEAL, BLUE], 112)
# S8 final64. No new ratio, interval, or statistical decision is computed.
bars('cdf0_latest', ['通常CDF0 ＋ Z再配置 ＋ PR/r64', 'ENTRY E2 ＋ Z再配置 ＋ PR/r16'],
     [231951425282.5/1e9, 75621395471/1e9], [AMBER, BLUE], 290)
# S7 fixed point comparisons only. No CI is attached to these new D baselines.
bars('strong_d', ['ENTRY E2 ＋ PR/r16', '保存XDF8 ＋ 強D', '保存CDF8 ＋ 強D'],
     [75621395471/1e9, 101780388260/1e9, 108561283488/1e9], [BLUE, '#8D9BA9', '#BCC6D0'], 140)

# S2 and S3 tables, epsilon=.0005. Already reported rounded G values.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, name, pf, pr, b3 in [
    (axes[0], 'H4 · DF10', [8.4146e12,8.74132e12,8.66047e12,9.58334e12],
     [5.1495e12,5.54978e12,6.1726e12,7.46061e12],
     [1.47868e13,3.31139e13,5.74925e13,8.67851e13]),
    (axes[1], 'H6 · DF19', [3.67856e13,3.70685e13,3.52718e13,3.99801e13],
     [2.16718e13,2.44644e13,2.84996e13,3.46305e13],
     [1.7003e14,3.7721e14,6.7397e14,1.11839e15])]:
    for y, c, m, lab in [(pf,AMBER,'s','決定論PF envelope'),(pr,TEAL,'o','Hybrid B2'),(b3,CORAL,'^','DF B3')]:
        ax.plot(ts,y,marker=m,color=c,label=lab,lw=2,ms=6)
    ax.set(title=name,xlabel='発展時間 T',xticks=ts,yscale='log')
    ax.grid(axis='y',which='both',alpha=.12)
axes[0].set_ylabel('探索点費用 G  [RZ]')
axes[1].legend(frameon=False,fontsize=10,loc='upper left')
fig.tight_layout()
save(fig,'long_time_absolute')
print('Created 6 presentation figures in PNG/SVG; saved-value display only.')
