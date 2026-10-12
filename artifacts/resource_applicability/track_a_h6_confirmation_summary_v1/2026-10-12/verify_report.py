"""Separate CSV/Fraction arithmetic and saved signal-shot checks; no science."""
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (2**30, 2**30))
    signal.alarm(60)
    audit=read(HERE/'summary_audit_v1.json')
    for path, expected in audit['generated_file_hashes'].items():
        assert sha(HERE/path)==expected, path
    summary=read(HERE/'result_summary_v1.json')
    maps=read(HERE/'confirmation_resource_map_v1.json')
    runtime=ROOT/'artifacts/resource_applicability/track_a_h6_confirmation_fast_v1/2026-10-12/launch_confirmation_fast_v1'
    m=read(runtime/'frozen_manifest.json')
    registry=m['base_manifest']['reuse_registry']
    signal_path=ROOT/registry['original_raw_path']/'signal/signal_summary.json'
    assert sha(signal_path)==registry['raw_hashes'][str(signal_path.relative_to(ROOT))]
    signals={s['cell']['id']:s for s in read(signal_path)['signals']}
    with (HERE/'wrapper_metrics_v1.csv').open() as f:
        confirmation=list(csv.DictReader(f))
    assert len(confirmation)==512
    assert len({(x['cell_id'],x['replica'],x['axis']) for x in confirmation})==512
    assert {x['phase'] for x in confirmation}=={'confirmation'}
    old=ROOT/'artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv'
    with old.open() as f:
        baseline=[x for x in csv.DictReader(f) if not x['cell_id'].startswith('B2_') and x['control']=='symmetric_directional']
    raw=defaultdict(list)
    for x in confirmation+baseline:
        raw[x['cell_id']].append(x)
    checked=0
    for r in maps['rows']:
        precision=signals[r['cell_id']]['precision_rows'][str(r['epsilon'])][r['allowance_factor']]
        assert precision['cost_acquisition_eligible']
        # Saved precision record names are real/imag; wrapper names cosine/sine.
        assert r['axes']['cosine']['shots']==precision['axes']['real']['shots']
        assert r['axes']['sine']['shots']==precision['axes']['imag']['shots']
        paired=defaultdict(dict)
        for x in raw[r['cell_id']]:
            paired[x['replica']][x['axis']]=int(x[r['metric']])
        totals=[sum(r['axes'][a]['shots']*x[a] for a in ('cosine','sine')) for x in paired.values()]
        point=float(Fraction(sum(totals),len(totals)))
        assert math.isclose(point,r['G_point'],rel_tol=2e-15,abs_tol=1e-9)
        se=statistics.stdev(totals)/math.sqrt(len(totals)) if len(totals)>1 else None
        assert (se is None and r['G_standard_error'] is None) or math.isclose(se,r['G_standard_error'],rel_tol=2e-15,abs_tol=1e-9)
        checked+=1
    for a,b in zip(summary['comparisons'][:4],summary['comparisons'][4:]):
        assert a['eligible_counts']==b['eligible_counts']
        assert {k:v['cell_id'] for k,v in a['family_best_point'].items()}=={k:v['cell_id'] for k,v in b['family_best_point'].items()}
    result=dict(status='LOCAL_CSV_FRACTION_AND_SAVED_SHOTS_MATCH',checked_resource_rows=checked,
        confirmation_wrapper_csv_rows=512,scientific_calls=0,scientific_array_decodes=0,
        source_changed=False,source_commit=summary['source_commit'],CI=False,independent_scientific_review=False,
        signal_summary_sha256=sha(signal_path),csv_sha256=sha(HERE/'wrapper_metrics_v1.csv'),
        report_map_sha256=sha(HERE/'confirmation_resource_map_v1.json'),verification_script_sha256=sha(__file__))
    with (HERE/'report_consistency_v1.json').open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
