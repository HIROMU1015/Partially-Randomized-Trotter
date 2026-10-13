"""Plot published scalar tables only. Matplotlib is required; no science imports."""
import argparse
import csv
import os
from pathlib import Path
import resource
import signal


def load(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    resource.setrlimit(resource.RLIMIT_AS, (1024*2**20, 1024*2**20))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(180)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Callers must choose private directories under their home, including cache/temp.
    for variable in ('MPLCONFIGDIR', 'TMPDIR'):
        if variable not in os.environ:
            raise ValueError(f'{variable} must be set to a private home directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'svg.hashsalt': 'h4-geometry-technical-v1'})
    args.output.mkdir(exist_ok=False)
    table = load(args.bundle/'precision_grid_1812.csv')
    scalar = load(args.bundle/'scalar_classification_1812.csv')
    deltas = load(args.bundle/'delta_sensitivity.csv')
    geometries = ('0.70', '0.80', '0.90', '1.10', '1.40', '1.60')
    groups = {g: [r for r in table if r['geometry'] == g] for g in geometries}
    colors = {'B0': '#6f6f6f', 'B1': '#c17c16', 'B2': '#1974b0', 'B3': '#8a559b'}

    def save(fig, name):
        for ext in ('png', 'svg', 'pdf'):
            path=args.output/f'{name}.{ext}'
            fig.savefig(path, dpi=170, bbox_inches='tight',
                        metadata={'Creator': 'saved-value scalar plot'})
            if ext=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for ax, g in zip(axes.flat, geometries):
        rows = groups[g]
        for method, color in colors.items():
            eligible = [r for r in rows if r[method+'_G']]
            ax.plot([float(r['epsilon']) for r in eligible],
                    [float(r[method+'_G']) for r in eligible], color=color, label=method)
        ax.set(xscale='log', yscale='log', title=f'{g} A', xlabel='Signal precision epsilon', ylabel='Total RZ work G')
        ax.grid(alpha=.18)
    axes[0, 0].legend(ncol=2)
    fig.suptitle('Registered method point minima: fixed DF12, T=0.8, P=0\n302 precisions reuse each candidate\'s same saved cost sample')
    fig.tight_layout(rect=(0, 0, 1, .92));save(fig, 'figure_1_registered_method_costs')

    templates = sorted({r['winner'] for r in table})
    cmap = ListedColormap(plt.get_cmap('tab10').colors[:len(templates)])
    norm = BoundaryNorm([i-.5 for i in range(len(templates)+1)], cmap.N)
    xs = [float(r['epsilon']) for r in groups[geometries[0]]]
    import math
    edges = [xs[0]**2/math.sqrt(xs[0]*xs[1])]
    edges += [math.sqrt(a*b) for a,b in zip(xs,xs[1:])]
    edges += [xs[-1]**2/edges[-1]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True)
    reduction = [[float(r['reduction_percent']) for r in groups[g]] for g in geometries]
    ids = [[templates.index(r['winner']) for r in groups[g]] for g in geometries]
    mesh = axes[0].pcolormesh(edges, list(range(7)), reduction, shading='flat', cmap='viridis')
    fig.colorbar(mesh, ax=axes[0], label='Below registered deterministic minimum (%)')
    mesh2 = axes[1].pcolormesh(edges, list(range(7)), ids, shading='flat', cmap=cmap, norm=norm)
    bar=fig.colorbar(mesh2, ax=axes[1], ticks=list(range(len(templates))))
    bar.ax.set_yticklabels([t.removeprefix('B2-rank3-') for t in templates])
    for ax in axes:
        ax.set(xscale='log', yticks=[i+.5 for i in range(6)], yticklabels=geometries, ylabel='Distance (A)')
    axes[0].set_title('Point reduction inside the registered comparison set')
    axes[1].set(title='Best registered template (all prefix3); overlapping alternatives at all points', xlabel='Signal precision epsilon')
    fig.tight_layout();save(fig, 'figure_2_precision_configuration_map')

    chosen = [r for r in table if float(r['epsilon']) == .05]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    x=list(range(6))
    axes[0].bar([i-.24 for i in x], [float(r['ratio_N_det']) for r in chosen], .24, label='Shot ratio', color='#50a293')
    axes[0].bar(x, [float(r['ratio_C_effective_det']) for r in chosen], .24, label='Effective RZ/shot ratio', color='#dfad53')
    axes[0].bar([i+.24 for i in x], [float(r['ratio_det']) for r in chosen], .24, label='Work ratio = product', color='#1974b0')
    axes[0].set(xticks=x, xticklabels=geometries, ylabel='Ratio to registered deterministic point minimum', xlabel='Distance (A)')
    axes[0].axhline(1, color='grey', linewidth=.7);axes[0].legend(fontsize=8)
    fixed=load(args.bundle/'fixed_configuration_factors.csv')
    for name,color in [('N_total','#50a293'),('C_effective','#dfad53'),('G','#1974b0')]:
        rows=[r for r in fixed if r['fixed_template']=='B2-rank3-q1-r4-K4' and float(r['epsilon'])==.05]
        reference=float(rows[0][name])
        axes[1].plot(x,[float(r[name])/reference for r in rows], 'o-', label=name, color=color)
    axes[1].set(xticks=x,xticklabels=geometries,xlabel='Distance (A)',ylabel='Ratio to 0.70 A at the same fixed template')
    axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('Cost factors at epsilon=0.05, P=0: reselected minima vs fixed p3/q1/r4/K4')
    fig.tight_layout(rect=(0, 0, 1, .93));save(fig,'figure_3_shot_circuit_factors')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for g in geometries:
        rows=[r for r in scalar if r['geometry']==g]
        axes[0].plot([float(r['epsilon']) for r in rows],
                     [float(r['ratio_reference_to_B2']) for r in rows], label=f'{g} A')
    axes[0].axhline(1,color='black',linewidth=.8)
    axes[0].set(xscale='log',xlabel='Signal precision epsilon',ylabel='Reference B0+I / registered B2 work',
                title='Unmeasured B0+I, extra compiled RZ Delta=0')
    axes[0].legend(ncol=2,fontsize=8);axes[0].grid(alpha=.2)
    for column,label,style in [('reoptimized_auxiliary_points_below_B2','Auxiliary reselected','o-'),
                                ('fixed_zero_delta_candidate_points_below_B2','Delta=0 template held fixed','x--')]:
        axes[1].plot([float(r['common_delta_RZ_per_axis_shot']) for r in deltas],
                     [int(r[column]) for r in deltas],style,label=label)
    axes[1].set(xscale='symlog',xlabel='Assumed common extra RZ per axis shot (Delta)',
                ylabel='Grid points below B2 / 1812', title='Common nonnegative Delta is an assumption')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.tight_layout();save(fig,'figure_4_unmeasured_scalar_sensitivity')
    total=sum(p.stat().st_size for p in args.output.iterdir())
    if total>12*2**20:
        raise ValueError('figure output cap exceeded')
    print(f'12 figure assets, {total} bytes; science/compile/sampling calls=0')


if __name__=='__main__':
    main()
