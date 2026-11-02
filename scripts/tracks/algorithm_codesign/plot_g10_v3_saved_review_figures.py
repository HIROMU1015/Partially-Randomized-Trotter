"""Static note figures from saved-field arithmetic; no simulation/optimization."""
from fractions import Fraction as F
import json
from pathlib import Path
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/track_b_g10_v3_scientific_review_intake/2026-10-10'


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'g10-v3-saved-review-20261010'})
    data = json.loads((OUT/'saved_arithmetic.json').read_text())
    assert data['lower_recomputed_or_sampling_optimized'] is False
    rows = {(r['degree'],r['arm']):r for r in data['all_row_saved_affine_fields']}
    files = []

    def save(fig, name):
        for suffix in ['svg', 'png']:
            path = OUT/f'{name}.{suffix}'
            assert not path.exists()
            fig.savefig(path, dpi=160, bbox_inches='tight', metadata={'Title': 'G10 v3 saved-value documentation'})
            if suffix == 'svg':
                # Matplotlib emits trailing spaces in multiline path attributes.
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            files.append(path.name)
        plt.close(fig)

    # h display ranges show existing affine equations; not new registered conditions.
    hs = list(range(0, 601, 5))
    p, c = rows[3,'partial_return_tail'], rows[3,'closed_P3_tail']
    dT, dK = F(c['T_intercept'])-F(p['T_intercept']), F(c['prep_T_slope'])-F(p['prep_T_slope'])
    fig, ax = plt.subplots(figsize=(7,3.7), layout='constrained')
    ax.plot(hs,[float((dT+dK*h)/1000000) for h in hs],color='#176b9b',label='closed P3 minus partial')
    edge=float(F(data['m3_partial_P3_prep_T_boundary']['exact_rational']))
    ax.axhline(0,color='black',linewidth=.8);ax.axvline(edge,color='#999999',linestyle=':')
    ax.set(xlabel='Common prep/readout T cost per call, h (display range)',ylabel='Expected two-axis T difference (millions)',
           title=f'm=3 registered T trade-off; boundary h={edge:.4f}')
    ax.legend();ax.grid(alpha=.2);save(fig,'m3_prep_T_boundary')

    hs=list(range(0,1201,5));full,p5=rows[7,'full_return'],rows[7,'closed_P5_tail']
    lo=full['fixed_dictionary_policy_lower'];fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    def delta(intercept,slope,h):
        return float((F(intercept)-F(p5['T_intercept'])+(F(slope)-F(p5['prep_T_slope']))*h)/1000000)
    ax.plot(hs,[delta(full['T_intercept'],full['prep_T_slope'],h) for h in hs],color='#d37c12',label='registered canonical full minus P5+tail')
    ax.plot(hs,[delta(lo['intercept_lower'],lo['prep_slope_lower'],h) for h in hs],color='#176b9b',label='saved full dictionary lower minus P5+tail')
    ax.axhline(0,color='black',linewidth=.8);ax.axvspan(0,970,color='#176b9b',alpha=.07,label='conservative lower separation: 0 <= h <= 970')
    edge=float(F(data['m7_saved_full_lower_minus_registered_P5']['lower_separation_endpoint']['exact_rational']))
    ax.axvline(edge,color='#999999',linestyle=':')
    ax.set(xlabel='Common prep/readout T cost per call, h (display range)',ylabel='Difference from registered P5+tail (million T)',
           title='m=7: lower-bound endpoint is not an actual winner crossover')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'m7_registered_and_lower_T_gaps')

    labels=['T','CX','native 1Q','K'];x=list(range(4));fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    for pos,(arm,color,label) in enumerate([('closed_P5_tail','#176b9b','P5+tail'),('full_return','#d37c12','full'),('matched_CTS','#468b62','literal CTS')]):
        r=rows[7,arm];values=[]
        for field in ['T','CX','1Q','K']:
            a,b=(r['prep_T_slope'],p5['prep_T_slope']) if field=='K' else (r['native_vector'][field],p5['native_vector'][field])
            values.append(float(F(a)/F(b)))
        ax.bar([v+(pos-1)*.24 for v in x],values,width=.24,label=label,color=color)
    ax.axhline(1,color='black',linewidth=.8);ax.set_xticks(x,labels)
    ax.set(ylabel='Ratio to P5+tail (separate resource units)',title='m=7 saved native vector; common prep gates excluded')
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,'m7_native_resource_vector')
    return {'plotting_only': True, 'matplotlib_version':matplotlib.__version__, 'python_executable':sys.executable,
            'files':files, 'numeric_conversion':'Exact saved rationals converted to float only for display.',
            'h_plot_domains':[{'figure':'m3','range':[0,600]},{'figure':'m7_lower','range':[0,1200]}],
            'display_domains_not_new_scientific_conditions':True,
            'SVG_trailing_whitespace_normalized_without_geometry_changes':True,
            'new_science_calls':0,'mandatory_STOP':True}


if __name__ == '__main__':
    print(json.dumps(render(),indent=2))
