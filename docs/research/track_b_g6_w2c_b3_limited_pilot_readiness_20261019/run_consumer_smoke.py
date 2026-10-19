"""Run the fixed consumer CLI against all 47 public artificial archives."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PACKET = ROOT / 'docs/research/track_b_g6_w2c_b3_v3_offline_acceptance_20261019'


def main():
    target = HERE / 'consumer_smoke'
    target.mkdir(exist_ok=False)
    inputs = [PACKET / p for p in json.loads((PACKET / 'published_cases.json').read_text())]
    def one(manifest):
        name = manifest.parent.name
        watched = [p for p in manifest.parent.iterdir() if p.is_file()]
        before = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in watched}
        output = target / name
        run = subprocess.run([sys.executable, '-m',
            'trottertracks.algorithm_codesign.g6_w2c_b3_v3_offline_cli',
            '--artificial-manifest', str(manifest), '--fixed-reference', str(PACKET / 'fixed_reference'),
            '--new-output', str(output)], cwd=ROOT, capture_output=True,
            env=dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONDONTWRITEBYTECODE='1'))
        after = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in watched}
        assert before == after, name
        assert run.returncode == 0, (name, run.stderr.decode())
        (target / (name+'.cli.log')).write_bytes(run.stdout+run.stderr)
        v = json.loads((output / 'verification_report.json').read_text())
        c = json.loads((output / 'cost_summary.json').read_text())
        a = json.loads((output / 'integration_acceptance_report.json').read_text())
        expected = 'VERIFIED_COMPLETE' if json.loads(manifest.read_text())['case_kind']=='normal' else 'TECHNICAL_FAILURE'
        assert v['status'] == expected, (name,v['status'])
        assert c['status'] == ('INSUFFICIENT_EVIDENCE' if name=='fused-parent_crash' else 'VERIFIED_PARALLEL_COST_BOUNDARIES')
        return dict(case=name, verification=v['status'], cost=c['status'], pending=a['pending'], input_sha256=before)
    with ThreadPoolExecutor(max_workers=min(16,len(os.sched_getaffinity(0)))) as pool:
        rows = list(pool.map(one,inputs))
    result = dict(cases=rows, classifications=dict(Counter(r['verification'] for r in rows)),
        cost_classifications=dict(Counter(r['cost'] for r in rows)), inputs_unchanged=True,
        consumer_commit='bf601f8a9723a61b7a05b534019f8e036a362735', artificial_only=True,
        historical_pending_reclassified=False, mandatory_STOP=True)
    (HERE/'consumer_smoke_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__ == '__main__':
    main()
