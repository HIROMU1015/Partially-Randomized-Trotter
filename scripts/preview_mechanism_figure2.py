#!/usr/bin/env python3
"""Figure 2 from pinned saved tokens, or an explicitly artificial preview."""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

_spec=importlib.util.spec_from_file_location('mechanism_paper_preview_schema',
    Path(__file__).resolve().parents[1]/'src/trotterlib/mechanism_paper.py')
_schema=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schema)
AXES,FACTOR_COLUMNS=_schema.AXES,_schema.FACTOR_COLUMNS


def synthetic_fixture():
    rows = []
    for system in ('H4', 'H6', 'H8'):
        for epsilon in ('0.05','0.005','0.0005'):
            for role in ('OLD_PR','NEW_PR'):
                rows.append(dict(system=system, epsilon=epsilon, PR_role=role,
                    PF_cost_class='VALIDATED_MODEL_ESTIMATE' if system=='H6' and epsilon=='0.005' else 'DIRECT_MEASURED',
                    R_C_axis={'cosine':'3/4','sine':'4/5'},
                    R_N_axis={'cosine':'5/4','sine':'3/2'}, R_G='15/16', R_N='5/4', R_Ceff='3/4'))
    examples = []
    for epsilon in ('0.05','0.0005'):
        examples.append(dict(epsilon=epsilon, R_G='15/16', R_N='5/4', R_Ceff='3/4',
            factors={axis:dict(zip(FACTOR_COLUMNS, ('6/5','1','6/5','4/5','1','24/25','1'))) for axis in AXES}))
    return dict(origin='SYNTHETIC_ONLY', rows=rows, examples=examples,
                missing_annotation='A82 original explanation: gap retained (fixture label only)')


def render_synthetic(data, output):
    if data.get('origin') != 'SYNTHETIC_ONLY':
        raise ValueError('REAL_EVIDENCE_RENDERING_NOT_ENABLED')
    return render(data,output,synthetic=True)


def saved_bundle(directory):
    directory=Path(directory)
    binding=json.loads((directory/'render_binding.json').read_text())
    if binding.get('scope')!='SAVED_FIGURE2_COORDINATES_ONLY' or binding.get('core_commit')!='2bf5d83293b3a47f672805fae956ff2e50df2878' or binding.get('explanation_commit')!='0edd62403c35c5ee18b2a2a152158faca5c2ebd8':
        raise ValueError('FIXED_FIGURE_SOURCE_BINDING')
    names=('main_table.csv','core_axes.csv','explanation_factors.csv','display_plan.csv')
    if set(binding['files_sha256'])!=set(names):raise ValueError('FIGURE_FILE_CLOSURE')
    tables=[]
    for name in names:
        path=directory/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['files_sha256'][name]:
            raise ValueError('FIGURE_INPUT_SHA_MISMATCH')
        with path.open() as stream:tables.append(list(csv.DictReader(stream)))
    result=_schema.figure_bundle(*tables)
    result['input_binding']=binding
    return result


def render(data, output, *, synthetic=False):
    if data.get('origin')!=('SYNTHETIC_ONLY' if synthetic else 'SAVED_FIXED_EVIDENCE'):
        raise ValueError('FIGURE_DATA_ORIGIN')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Imports are deferred; technical tests use an existing project environment.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, FuncFormatter
    plt.rcParams.update({'font.size':9, 'svg.fonttype':'none','savefig.facecolor':'white'})
    colors = {'OLD_PR':'#0072B2', 'NEW_PR':'#D55E00'}
    plotted=[]
    def coordinate(token, panel, record, axis, metric, *, precise=False):
        number=float(Fraction(token))  # representation conversion, no new scientific ratio
        if number<=0:raise ValueError('POSITIVE_SAVED_PLOT_TOKEN_REQUIRED')
        label=format(number,'.13g' if precise else '.5g')
        plotted.append(dict(panel=panel,contrast_id=record.get('contrast_id','SYNTHETIC'),system=record.get('system','H8'),
            epsilon=record['epsilon'],axis=axis,metric=metric,saved_value_text=token,
            plot_coordinate=repr(number),display_text=label,PF_cost_class=record.get('PF_cost_class','SYNTHETIC'),
            PR_cohort=record.get('PR_cohort','SYNTHETIC'),n=record.get('n','SYNTHETIC')))
        return number,label
    fig, panels = plt.subplots(3, 3, figsize=(11.8,8.5), sharey='row')
    fig.subplots_adjust(top=.84,bottom=.16,hspace=.22,wspace=.10,left=.09,right=.985)
    row_labels=('Circuit unit-cost ratio\n$R_{C,a}$ (PR / PF)', 'Shot ratio\n$R_{N,a}$ (PR / PF)', 'Total cost ratio\n$R_G$ (PR / PF)')
    for col, system in enumerate(('H4','H6','H8')):
        lookup={(r['epsilon'],r['PR_role']):r for r in data['rows'] if r['system']==system}
        selected=[lookup[(eps,role)] for eps in _schema.MAIN_EPSILON for role in ('OLD_PR','NEW_PR')]
        for row, field in enumerate(('R_C_axis','R_N_axis','R_G')):
            ax = panels[row,col]
            for x, record in enumerate(selected):
                axes = AXES if field != 'R_G' else ('total',)
                for axis in axes:
                    number = record[field][axis] if axis != 'total' else record[field]
                    fill = colors[record['PR_role']] if record['PF_cost_class']=='DIRECT_MEASURED' else 'none'
                    y,_=coordinate(number,'2A',record,axis,field)
                    offset=-.09 if axis=='cosine' else .09 if axis=='sine' else 0
                    ax.scatter(x+offset,y,marker='o' if axis!='sine' else 's',s=35,
                        facecolors=fill,edgecolors=colors[record['PR_role']],linewidths=1.1,zorder=3)
            ax.axhline(1,color='#777777',linewidth=.8,zorder=1)
            ax.set_yscale('log');ax.grid(axis='y',which='major',alpha=.16)
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda tick,position:format(tick,'.2g')))
            ax.set_xticks([.5,2.5,4.5],['0.05','0.005','0.0005'])
            if col==0:ax.set_ylabel(row_labels[row])
            if row<2:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('Target signal accuracy $\\epsilon$')
            if row==0: ax.set_title(system+(' / legacy DF12' if system=='H4' else ' / DF19' if system=='H6' else ' / DF28'),weight='bold')
    title='SYNTHETIC FIXTURE — NO SCIENTIFIC EVIDENCE' if synthetic else 'Figure 2A  |  Saved cost decomposition across three systems'
    fig.suptitle(title,y=.975,fontsize=13,weight='bold')
    legend=[Line2D([],[],color=c,marker='o',linestyle='None',label=l) for c,l in [(colors['OLD_PR'],'Old PR: n32'),(colors['NEW_PR'],'New A: exploratory n8')]]
    legend += [Line2D([],[],color='#555555',marker=m,linestyle='None',label=l) for m,l in [('o','cosine / total'),('s','sine')]]
    legend += [Line2D([],[],color='#555555',marker='o',markerfacecolor=f,linestyle='None',label=l) for f,l in [('#555555','PF directly measured'),('none','PF model estimate')]]
    fig.legend(handles=legend,ncol=3,loc='upper center',bbox_to_anchor=(.52,.945),frameon=False,fontsize=9)
    footer=('Linear chains, 1 Angstrom, STO-3G; T=0.8; P=0; usual signed / factor 1. Cost ratios retain the original n8/n32 cohorts.\n'
        'H4: old n32 is fixed-anchor selection/estimation; independent P completed separately (not plotted). H6/H8: old n32 is post-selection.\n'
        'A82 core is present; its archived explanation remains missing pending separate numerical approval. Empirical, finite-sample evidence.')
    fig.text(.09,.035,footer,fontsize=8.2,va='bottom',linespacing=1.45)
    suffix='_synthetic' if synthetic else ''
    fig.savefig(output/('figure2A'+suffix+'.svg'));fig.savefig(output/('figure2A'+suffix+'.png'),dpi=220)
    if not synthetic:fig.savefig(output/'figure2A.pdf')
    plt.close(fig)
    fig, panels = plt.subplots(2,2,figsize=(11.8,7.8))
    fig.subplots_adjust(top=.88,bottom=.26,hspace=.44,wspace=.23,left=.09,right=.98)
    for col, record in enumerate(data['examples']):
        ax = panels[0,col]
        for x, field in enumerate(('R_N','R_Ceff','R_G')):
            number,label=coordinate(record[field],'2B',record,'total',field)
            ax.scatter(x,number,color='#0072B2',s=42)
            ax.annotate(label,(x,number),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks(range(3),('$R_N$\nshots','$R_{C,eff}$\nunit cost','$R_G$\ntotal cost'))
        ax.axhline(1,color='#777777',linewidth=.7); ax.set_yscale('log')
        ax.set_xlim(-.45,2.45);ax.margins(y=.25);ax.set_ylabel('PR / fixed PF ratio');ax.grid(axis='y',alpha=.16)
        ax.set_title('H8 independent n48: $\\epsilon$ = '+record['epsilon'],weight='bold')
        ax = panels[1,col]
        fields = ('nominal_normalization_squared','normalization_margin_squared','headroom_squared','alpha_log','rounding_factor')
        precision=[]
        for axis,marker,color,offset in zip(AXES,('o','s'),('#0072B2','#D55E00'),(-.10,.10)):
            ys=[]
            for k in fields:
                y,label=coordinate(record['factors'][axis][k],'2B',record,axis,k,precise=True);ys.append(y)
                if k in ('normalization_margin_squared','rounding_factor'):precision.append(axis+' '+k.replace('normalization_margin_squared','margin').replace('rounding_factor','rounding')+' = '+label)
            ax.scatter([x+offset for x in range(len(fields))],ys,marker=marker,color=color,label=axis,s=35)
        ax.set_xticks(range(len(fields)),('Nominal\nnorm²','Empirical\nmargin exp','Headroom²','Alpha log','Integer\nrounding'))
        ax.axhline(1,color='#777777',linewidth=.7);ax.set_yscale('log');ax.set_ylabel('Saved explanatory factor')
        ax.grid(axis='y',alpha=.16);ax.legend(loc='lower left',fontsize=8);ax.margins(y=.20)
        fig.text(.09+col*.49,.075,'\n'.join(precision),fontsize=7.7,linespacing=1.4)
    title='SYNTHETIC FIXTURE — NO SCIENTIFIC EVIDENCE' if synthetic else 'Figure 2B  |  Two fixed H8 confirmations: cost and shot factors'
    fig.suptitle(title,y=.975,fontsize=13,weight='bold')
    fig.text(.09,.018,'Saved factor values only; independent n48 is separate from exploratory n8. High-order events unobserved; population advantage not certified.',fontsize=8)
    fig.savefig(output/('figure2B'+suffix+'.svg'));fig.savefig(output/('figure2B'+suffix+'.png'),dpi=220)
    if not synthetic:fig.savefig(output/'figure2B.pdf')
    plt.close(fig)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    with (output/'plot_values.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plotted[0]),lineterminator='\n');writer.writeheader();writer.writerows(plotted)
    receipt=dict(origin=data['origin'],plotted_points=len(plotted),coordinate_conversion='float(Fraction(saved_token))',
        new_ratios=0,new_statistics=0,new_decompositions=0,exact_tokens_preserved=True,
        input_binding=data.get('input_binding'),A82_explanation_status='EXPLANATORY_INPUT_MISSING',
        H4_P_costs_used_in_plot=False)
    (output/'render_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (output/'README.txt').write_text(('Artificial fixture only.\n' if synthetic else 'Fixed saved scientific values; representation conversion only.\n')+data['missing_annotation']+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic-preview',action='store_true')
    mode.add_argument('--saved-data',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.synthetic_preview:render_synthetic(synthetic_fixture(),args.output)
    else:render(saved_bundle(args.saved_data),args.output)


if __name__=='__main__':
    main()
