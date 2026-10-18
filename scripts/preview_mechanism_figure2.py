#!/usr/bin/env python3
"""Technical Figure 2 preview. This CLI has no real-data input route."""
import argparse
from fractions import Fraction
import importlib.util
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Imports are deferred; technical tests use an existing project environment.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8, 'svg.fonttype':'none'})
    colors = {'OLD_PR':'#33658A', 'NEW_PR':'#B35C1E'}
    fig, panels = plt.subplots(3, 3, figsize=(11,8), layout='constrained')
    for col, system in enumerate(('H4','H6','H8')):
        selected = [r for r in data['rows'] if r['system']==system]
        labels = [r['epsilon']+'\n'+r['PR_role'] for r in selected]
        for row, field in enumerate(('R_C_axis','R_N_axis','R_G')):
            ax = panels[row,col]
            for x, record in enumerate(selected):
                axes = AXES if field != 'R_G' else ('total',)
                for axis in axes:
                    number = record[field][axis] if axis != 'total' else record[field]
                    fill = colors[record['PR_role']] if record['PF_cost_class']=='DIRECT_MEASURED' else 'none'
                    ax.scatter(x, float(Fraction(number)), marker='o' if axis!='sine' else 's',
                        facecolors=fill, edgecolors=colors[record['PR_role']])
            ax.axhline(1, color='#777777', linewidth=.7)
            ax.set_yscale('log'); ax.set_ylabel(field); ax.set_xticks(range(len(labels)),labels)
            if row==0: ax.set_title(system+' synthetic')
    fig.suptitle('SYNTHETIC FIXTURE — NO SCIENTIFIC EVIDENCE\nFigure 2A layout; circles cosine/total, squares sine; open markers model PF')
    fig.savefig(output/'figure2A_synthetic.svg'); fig.savefig(output/'figure2A_synthetic.png',dpi=120)
    plt.close(fig)
    fig, panels = plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for col, record in enumerate(data['examples']):
        ax = panels[0,col]
        for x, field in enumerate(('R_N','R_Ceff','R_G')):
            ax.scatter(x,float(Fraction(record[field])),color='#33658A')
        ax.set_xticks(range(3),('shot total ratio','effective unit cost ratio','total cost ratio'))
        ax.axhline(1,color='#777777',linewidth=.7); ax.set_yscale('log')
        ax.set_title('H8 fixed n48 — epsilon '+record['epsilon']+' (synthetic values)')
        ax = panels[1,col]
        fields = ('nominal_normalization_squared','normalization_margin_squared','headroom_squared','alpha_log','rounding_factor')
        for axis,marker in zip(AXES,('o','s')):
            ax.scatter(range(len(fields)),[float(Fraction(record['factors'][axis][k])) for k in fields],marker=marker,label=axis)
        ax.set_xticks(range(len(fields)),('nominal norm²','margin exp ratio','headroom²','alpha log','rounding'),rotation=12)
        ax.axhline(1,color='#777777',linewidth=.7);ax.set_yscale('log');ax.legend()
    fig.suptitle('SYNTHETIC FIXTURE — NO SCIENTIFIC EVIDENCE\nFigure 2B: independent n48 examples remain separate from exploratory n8')
    fig.savefig(output/'figure2B_synthetic.svg');fig.savefig(output/'figure2B_synthetic.png',dpi=120)
    plt.close(fig)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    (output/'README.txt').write_text('Artificial layout preview only. No saved molecular values were read.\n'+data['missing_annotation']+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-preview',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render_synthetic(synthetic_fixture(),args.output)


if __name__=='__main__':
    main()
