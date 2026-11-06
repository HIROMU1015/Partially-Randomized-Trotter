"""Render saved H4 summaries only: no scientific engine or new statistics.

Inputs are frozen at 417fa459b. Only field selection, decimal display, layout,
and axis-unit conversion are performed. All intervals are read, never fitted.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_COMMIT = '417fa459b823509be5c2010c5836e58955088282'
LITERATURE_COMMIT = '1a588f5491bce394ecf039ee4d2de57581e2e0b1'
BASE = Path('artifacts/research_c_integrated/2026-11-06-picklefix')
SAVED = BASE / 'summary_h4_100'
OUT = Path('artifacts/research_c_h4_paper/2026-11-06')
PR, ENTRY, D = '#24689B', '#D58A2D', '#626B73'


def read(path):
    return json.loads((ROOT / path).read_text())


def write_table(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t',
                                lineterminator='\n', restval='NA')
        writer.writeheader()
        writer.writerows({k: 'NA' if v is None else v for k, v in row.items()}
                         for row in rows)


def inputs():
    summary = read(SAVED / 'summary.json')
    comparisons = summary['comparisons']
    finals = summary['final_costs']
    def comparison(start, budget, regime):
        return next(x for x in comparisons if
                    (x['start'], x['budget'], x['regime']) == (start, budget, regime))
    def final(comp, arm):
        return next(x for x in finals if
                    (x['id'], x['r']) == (comp[arm + '_id'], comp[arm + '_r']))
    primary = [comparison('C0', 64, regime) for regime in ('common_r32', 'fair_retuned')]
    assert all(x['confidence'] == .9875 for x in primary)
    assert read(BASE / 'INTEGRATED_STOP.json')['status'] == 'FAILED_MANDATORY_STOP'
    assert summary['scientific_SOURCE_sha256'] == hashlib.sha256(
        (ROOT / BASE / 'technical/SOURCE.json').read_bytes()).hexdigest()
    checkpoints = []
    for row in summary['optimization_checkpoints']:
        original = read(Path(row['report']))
        state = original['checkpoints'][str(row['budget'])]['evidence']['state']
        copied = dict(row)
        for variable in ('J', 'U'):
            for key in ('value', 'threshold', 'engineering_margin', 'status'):
                copied[variable + '_' + key] = state[variable + '_stationarity'][key]
        checkpoints.append(copied)
    policies = []
    for row in read(BASE / 'development/run01/downstream/selection_rows.json'):
        copied = dict(row)
        reference = BASE / 'development/run01/downstream/selection' / row['id'] / f"r{row['r']}" / 'references.json'
        saved = read(reference)
        lane = saved['lanes']['ideal_exact_native']
        copied['saved_energy_status'] = lane['energy']['status']
        copied['saved_round_failures'] = ';'.join(
            f"m{x['m']}:{x['status']}" for x in lane['rounds']
            if x['status'] != 'CONDITIONAL_ROUND_PASS') or 'none'
        copied['reference_json'] = str(reference)
        policies.append(copied)
    return summary, comparison, final, primary, checkpoints, policies


def render(summary, comparison, final, primary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'research-c-h4-100-paper-20261106'})
    figures = ROOT / OUT / 'figures'
    figures.mkdir(exist_ok=True)
    generated = []

    def save(fig, name, footer):
        fig.text(.5, .012, footer + '\nFixed results: 417fa459b; SOURCE 8113f23e66c5; see figure_sources.json and captions.md',
                 ha='center', va='bottom', fontsize=7, color='#50575D')
        for extension in ('svg', 'pdf', 'png'):
            target = figures / f'{name}.{extension}'
            metadata = {'Creator': 'Matplotlib; saved-result transcription only'}
            if extension == 'pdf':
                metadata['CreationDate'] = datetime(2026, 11, 6, tzinfo=timezone.utc)
            if extension == 'svg':
                metadata = {'Creator': metadata['Creator'], 'Date': '2026-11-06'}
            fig.savefig(target, dpi=240, bbox_inches='tight', metadata=metadata)
            generated.append(str(target.relative_to(ROOT)))
        plt.close(fig)

    # Figure 1: conceptual dependencies, not a new numerical experiment.
    fig, ax = plt.subplots(figsize=(13.3, 7.3)); ax.set_axis_off()
    def box(x, y, w, h, label, face='#EDF3F7'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008',
            facecolor=face, edgecolor='#8D9BA5', linewidth=1))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13,
                                   color='#5A6873', linewidth=1.25))
    ax.set_title('PR-aware DF: a mapped-tail objective evaluated through downstream resources', pad=14)
    box(.01,.80,.21,.13,'Signed symmetric $J$\nFrame-specific orthogonal $U$')
    box(.29,.80,.32,.13,'Tensor fit + mapped-tail penalty\n$f/f_* + \\gamma(M_\\epsilon/M_*)^2$')
    box(.69,.80,.29,.13,'Finite update budget\nDF admission + J/U diagnostics')
    arrow((.22,.865),(.28,.865)); arrow((.61,.865),(.68,.865))
    box(.69,.56,.29,.14,'Number DF $\\to$ Z / ZZ + identity\nHead: deterministic / tail: RTE\nOriginal frame order, $L_D=2$')
    arrow((.835,.80),(.835,.71))
    ax.text(.01,.65,'$M_\\epsilon(J)$ uses mapped, nonidentity tail coefficients.\n'
        '$U$ affects tensor fitting, PF/RTE references and native rotations.\n'
        'Original-g one-body correction; scalar counted once.',va='center', fontsize=10)
    box(.01,.31,.27,.14,'Candidate PF/RTE reference\nOwn-DF ground oracle')
    box(.365,.31,.27,.14,'Native kernel / P0 compilation\nSame numerical checks per sample')
    box(.72,.31,.26,.14,'Head and X/Y shells\nSeparate accounting')
    for end in [(.145,.455),(.50,.455),(.85,.455)]: arrow((.835,.56),end)
    box(.01,.07,.27,.14,'RPE shot allocation\nshots and repetition weight $W$')
    box(.365,.07,.27,.14,'Independent final64\nMean native kernel RZ')
    box(.72,.07,.26,.14,'Conditional total RZ\n$C=W\\,\\overline{c}_{kernel}+S_{shell}$', '#DCEAF4')
    arrow((.145,.31),(.145,.215)); arrow((.50,.31),(.50,.215)); arrow((.85,.31),(.85,.215))
    arrow((.635,.14),(.712,.14))
    ax.plot([.145,.145,.85,.85],[.07,.027,.027,.062],color='#5A6873',lw=1.2)
    ax.annotate('',xy=(.85,.07),xytext=(.85,.027),arrowprops={'arrowstyle':'-|>','color':'#5A6873'})
    fig.subplots_adjust(bottom=.09,top=.91)
    save(fig,'F1_method','Conceptual diagram of existing code. Conditional shot model; state-preparation cost excluded.')

    # Figure 2: saved point totals and saved Welch difference intervals only.
    fig, axes = plt.subplots(1,2,figsize=(10.5,5.2),sharey=True)
    for ax, comp, title in zip(axes,primary,['Common $r=32$','Fair retuning over $r=16,32,64$']):
        for pos, arm, color in [(0,'PR',PR),(1,'ENTRY',ENTRY)]:
            value = comp[arm+'_total_RZ']/1e9
            ax.bar(pos,value,width=.58,color=color)
            ax.text(pos,value+3,f'{value:.3f}',ha='center',fontsize=11)
        ax.set_xticks([0,1],['PR-aware','ENTRY']);ax.set_ylim(0,260)
        ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
        ax.text(.5,.94,f"Selected strength: {comp['PR_strength']} / {comp['ENTRY_strength']}\n"
            f"Selected r: {comp['PR_r']} / {comp['ENTRY_r']}",transform=ax.transAxes,ha='center',va='top',fontsize=9)
        ax.text(.5,-.15,f"PR/ENTRY = {comp['point_ratio_PR_ENTRY']:.6f}\n"
            f"Stored 98.75% CI for PR minus ENTRY:\n[{comp['difference_low']/1e9:.3f}, {comp['difference_high']/1e9:.3f}] billion RZ",
            transform=ax.transAxes,ha='center',va='top',fontsize=9)
    axes[0].set_ylabel('Conditional total RZ ($10^9$)')
    fig.suptitle('C0, 64 accepted updates: selected on native32, evaluated on independent native64',fontsize=12)
    fig.subplots_adjust(bottom=.30,top=.85,wspace=.15)
    save(fig,'F2_fair_comparison','H4 1.00 Å / STO-3G / CDF8 / LD2 / S2,s1,h=0.1,K0 / m=0..12 / conditional ideal-law model.')

    # Figure 3: a cell display, with no pooled estimate or fitted curve.
    fig, axes = plt.subplots(1,2,figsize=(10.6,5.0))
    norm=Normalize(.35,1.0);cmap=plt.get_cmap('Blues_r')
    for ax, regime, title in zip(axes,('common_r32','fair_retuned'),('Common $r=32$','Fair retuning')):
        for y,start in enumerate(('C0','P1','P2')):
            for x,budget in enumerate((16,64)):
                row=comparison(start,budget,regime);value=row.get('point_ratio_PR_ENTRY')
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor='#EEEEEE' if value is None else cmap(norm(value)),edgecolor='white',lw=3))
                label='Not eligible\nENTRY at r32' if value is None else f"{value:.4f}\n{row['PR_reduction_percent']:.2f}% lower"
                ax.text(x,y,label,ha='center',va='center',fontsize=10,
                        color='white' if value is not None and value<.52 else '#172C3B')
        ax.set(xlim=(-.5,1.5),ylim=(2.5,-.5),xticks=[0,1],xticklabels=['16','64'],yticks=[0,1,2],yticklabels=['C0','P1','P2'],xlabel='Accepted-update budget',title=title)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('Three fixed initializations; each cell has its own selection and final cohorts',fontsize=12)
    fig.subplots_adjust(left=.07,right=.83,bottom=.18,top=.84,wspace=.30)
    cax=fig.add_axes([.87,.26,.022,.50])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Saved PR / ENTRY total-RZ point ratio')
    save(fig,'F3_robustness','Final64 per available arm. Primary C0/64: stored 98.75% Welch differences; other comparisons: individual 95%.')

    # Figure 4: separate axes and units; no newly computed decomposition.
    fig, axes=plt.subplots(1,5,figsize=(15,4.4))
    rows=[final(c,a) for c in primary for a in ('PR','ENTRY')]
    measures=[('shots','Allocated shots',1e3,'$10^3$ shots'),('W','Repetition weight $W$',1e6,'$10^6$ occurrences'),
              ('rz_count_mean_kernel','Mean kernel RZ',1e3,'$10^3$ RZ / kernel'),
              ('rz_count_shell','Shell RZ',1e3,'$10^3$ RZ'),('rz_count_total','Conditional total',1e9,'$10^9$ RZ')]
    for ax,(key,title,scale,unit) in zip(axes,measures):
        values=[x[key]/scale for x in rows]
        ax.bar([0,1,3,4],values,color=[PR,ENTRY,PR,ENTRY],width=.75)
        for x,value in zip((0,1,3,4),values):ax.annotate(f'{value:.2f}',(x,value),xytext=(0,4 if x in (0,3) else 14),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([.5,3.5],['Common\nr32','Fair\nretuned']);ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.margins(y=.22);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PR,label='PR-aware'),Patch(color=ENTRY,label='ENTRY')],loc='upper center',ncol=2,bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('C0 / 64 updates: five saved accounting quantities on separate scales',y=1.01,fontsize=13)
    fig.subplots_adjust(bottom=.22,top=.76,wspace=.62)
    save(fig,'F4_accounting','C_total = W × mean_kernel_cost + shell_cost. Allocated shots are model inputs; no hardware shot outcomes were drawn.')

    # Figure 5: stored one-sample Student-t intervals; no new inference.
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.0),gridspec_kw={'width_ratios':[1,1.3]})
    comp=primary[1];row=final(comp,'PR')
    nominal=read(Path(row['report']))['statistics']['lanes']['ideal_exact_native']['RZ_PR_D_nominal']
    ax=axes[0];values=[row['rz_count_total']/1e9,row['rz_count_same_JU_D']/1e9]
    ax.bar([0,1],values,color=[PR,D],width=.6)
    lo,hi=nominal['total_interval'];ax.vlines(0,lo/1e9,hi/1e9,color='black',lw=1.4)
    ax.hlines([lo/1e9,hi/1e9],-.10,.10,color='black',lw=1.4)
    for x,value in enumerate(values):ax.text(x,value+3,f'{value:.3f}',ha='center')
    ax.set_xticks([0,1],['PR-aware\nr16','Same-J/U D\nall-frame S2']);ax.set_ylim(0,125)
    ax.set_ylabel('Conditional total RZ ($10^9$)');ax.set_title('C0, 64 updates, strength 0.1')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[1]
    for y,start in enumerate(('C0','P1','P2')):
        c=comparison(start,64,'fair_retuned');x=final(c,'PR')
        point=x['rz_count_ratio_to_D'];low=x['D_ratio_low'];high=x['D_ratio_high']
        ax.hlines(y,low,high,color=PR,lw=2);ax.plot(point,y,'o',color=PR)
        ax.text(high+.013,y,f'{point:.4f}',va='center',fontsize=9)
    ax.axvline(1,color=D,ls='--',lw=1);ax.set_xlim(.68,1.06);ax.set_ylim(2.5,-.5)
    ax.set_yticks([0,1,2],['C0','P1','P2']);ax.set_xlabel('PR / own same-J/U D total RZ')
    ax.set_title('Retuned PR, 64 updates\nStored individual nominal 95% intervals')
    fig.subplots_adjust(bottom=.23,top=.84,wspace=.35)
    save(fig,'F5_same_JU_D','Candidate-specific D denominators. Fixed S2/s1 control, not a global PF optimum. Historical negative/unresolved cases: Supplement S3.')
    return generated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    summary,comparison,final,primary,checkpoints,policies=inputs()
    target=ROOT/OUT;target.mkdir(parents=True,exist_ok=True)
    if args.check_only:
        assert len(checkpoints)==36 and len(policies)==108
        print('Saved JSON fields and SOURCE binding checked; no scientific evaluation.');return
    write_table('S1_checkpoint_qualifications.tsv',checkpoints)
    write_table('S2_policy_qualifications.tsv',policies)
    write_table('T1_primary_comparisons.tsv',primary)
    write_table('T2_primary_accounting.tsv',[final(c,a) for c in primary for a in ('PR','ENTRY')])
    round_examples=[]
    for comp in primary:
        for arm in ('PR','ENTRY'):
            path=BASE/'development/run01/downstream/selection'/comp[arm+'_id']/f"r{comp[arm+'_r']}"/'references.json'
            saved=read(path)['rounds'][-1]
            assert saved['m']==12
            ref=saved['reference']
            round_examples.append(dict(regime=comp['regime'],arm=arm,id=comp[arm+'_id'],
                r=comp[arm+'_r'],m=12,pf_operator_error=ref['pf_operator_error'],
                rte_corrected_operator_error=ref['rte_corrected_operator_error'],
                attenuation=ref['attenuation'],reference_json=str(path)))
    write_table('T3_saved_round12_references.tsv',round_examples)
    files=render(summary,comparison,final,primary)
    c0=[final(c,a) for c in primary for a in ('PR','ENTRY')]
    robustness=[comparison(s,b,g) for s in ('C0','P1','P2') for b in (16,64)
                for g in ('common_r32','fair_retuned')]
    drows=[final(comparison(s,64,'fair_retuned'),'PR') for s in ('C0','P1','P2')]
    index=dict(science_commit=SCIENCE_COMMIT,literature_commit=LITERATURE_COMMIT,
        scientific_SOURCE_sha256=summary['scientific_SOURCE_sha256'],
        contract_sha256=summary['contract_sha256'],input_sha256=read(BASE/'development/run01/downstream/final'/c0[0]['id']/f"r{c0[0]['r']}"/'result.json')['input_sha256'],
        conditions=summary['scope'],model='ideal_exact_native conditional finite-bit price proxy',
        statistics='Stored intervals only: primary differences98.75%;other differences95%;same-JU D individual nominal95%;coverage unvalidated',
        source_files=[str(SAVED/name) for name in ('summary.json','comparisons.tsv','final_costs.tsv','deterministic_costs.tsv','optimization_checkpoints.tsv','selection_rows.tsv')],
        figures={'F1_method':dict(type='existing-code concept diagram',code=['src/trottertracks/c_pr_aware_df.py','src/trottertracks/c_near_objective.py','src/trottertracks/c_near_r32_statistics.py']),
                 'F2_fair_comparison':dict(comparisons=primary,candidates=c0),
                 'F3_robustness':dict(comparisons=robustness,candidates=summary['final_costs']),
                 'F4_accounting':dict(candidates=c0),
                 'F5_same_JU_D':dict(candidates=drows,stored_C0_absolute_interval=read(Path(drows[0]['report']))['statistics']['lanes']['ideal_exact_native']['RZ_PR_D_nominal']['total_interval'])},
        outputs=files,new_scientific_computations=0,new_statistics=0,raw_arrays_decoded=False,
        working_language='Japanese prose;English figure labels;neither final manuscript language nor journal fixed')
    (target/'figure_sources.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(f'Rendered 5 figures in SVG/PDF/PNG; 5 direct-transcription tables; {target}')


if __name__=='__main__':main()
