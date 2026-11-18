"""Prepare a controller's saved-result index for the ordinary C 10-minute sync.

This does not run, modify, or stop the controller, nor commit or push for it.
Publish the generated file with the referenced metadata via ordinary Git.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

FILES = ('ROUND_RESULT.json', 'ROUND_RESULT_INTERRUPTED.json', 'EVALUATIONS.json',
         'NEEDS.json', 'SCOPE_METADATA.json', 'SEARCH_STATE.json', 'STATE.json',
         'ROUND.json', 'STATUS.json', 'LIVE_START_SNAPSHOT.json', 'PRICE_MODEL_VALIDATIONS.json')


def search_progress(path, root, run_id):
    """Publish the latest controller decision, without copying histories or raw results."""
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        state = json.loads(raw)
        if not isinstance(state, dict) or not isinstance(state.get('families'), dict):
            raise ValueError('Expected a controller state object')
    except (OSError, ValueError):
        return dict(schema='H4_SEARCH_PROGRESS_V1', status='STATE_UNREADABLE')
    if state.get('artificial') is True or state.get('artificial_only') is True:
        return None
    if state.get('round_id') != run_id:
        return dict(schema='H4_SEARCH_PROGRESS_V1', status='STATE_RUN_ID_MISMATCH')

    try:
        return dict(_search_progress_summary(state), state_path=path.relative_to(root).as_posix(),
                    state_sha256=hashlib.sha256(raw).hexdigest())
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return dict(schema='H4_SEARCH_PROGRESS_V1', status='STATE_UNREADABLE')


def _search_progress_summary(state):
    latest = (state.get('history') or [{}])[-1]
    report = latest.get('report') or {}
    proposals = defaultdict(list)
    decisions = defaultdict(list)
    point_status = defaultdict(Counter)
    points = {}
    for row in latest.get('proposals', []):
        proposals[str(row.get('family', row.get('L_D')))].append(
            {k: row[k] for k in ('kappa', 'reason') if k in row})
    for row in latest.get('decisions', []):
        decisions[str(row.get('family', row.get('L_D')))].append(
            {k: row[k] for k in ('decision', 'kappa', 'lower', 'incumbent') if k in row})
    for row in state.get('points', []):
        family = str(row.get('family', row.get('L_D')))
        point_status[family][row.get('status') or 'UNKNOWN'] += 1
        points[family, str(row.get('kappa'))] = row.get('status')

    families = []
    for family in sorted(set(state.get('families', {})) | set(report.get('families', {}))):
        metadata = state.get('families', {}).get(family, {})
        summary = report.get('families', {}).get(family, {})
        row = dict(family=family, point_status=dict(point_status[family]),
                   next_candidates=proposals[family], decisions=decisions[family])
        row.update({k: metadata[k] for k in ('method', 'L_D', 'formula', 'K', 'requested_delta_pf', 'status') if k in metadata})
        row.update({k: summary[k] for k in ('requested_kappa_bounds', 'kappa_range_evaluated', 'upper_limit',
                    'upper_basis', 'refinement', 'open_gaps', 'floor_end') if k in summary})
        best = summary.get('best')
        if isinstance(best, dict):
            row['reference_kappa'] = best.get('kappa')
        for candidate in row['next_candidates']:
            value = candidate.get('kappa')
            candidate['status'] = points.get((family, str(value)))
            try:
                ref = row.get('reference_kappa')
                if isinstance(ref, dict):
                    ref = Fraction(int(ref['numerator']), int(ref['denominator']))
                kappa = Fraction(value)
                if ref is not None:
                    candidate['direction'] = 'UP' if kappa > Fraction(ref) else 'DOWN' if kappa < Fraction(ref) else 'SAME'
            except (TypeError, ValueError, KeyError, ZeroDivisionError):
                pass
        families.append(row)
    stop = state.get('stop') or {}
    return dict(schema='H4_SEARCH_PROGRESS_V1', status='RECORDED',
                controller_cycle=state.get('cycle'), planner_cycle=latest.get('cycle'),
                controller_stop={k: stop[k] for k in ('reason', 'UTC') if k in stop},
                planner_stop=report.get('stop'), families=families)


def prepare(root, run_directory, output, *, run_id=None, execution_hosts=()):
    root=Path(root).resolve(); run=Path(run_directory).resolve(); output=Path(output).resolve()
    relative=run.relative_to(root).as_posix()
    if not relative.startswith('docs/research/'):
        raise ValueError('Publish owned controller metadata under docs/research; private paths are not scanned')
    out_relative=output.relative_to(root).as_posix()
    if not out_relative.startswith('docs/research/')or output.name!='SEARCH_PUBLICATION.json':
        raise ValueError('Public index must be docs/research/.../SEARCH_PUBLICATION.json')
    files={}
    for name in FILES:
        p=run/name
        if p.exists():
            value=json.loads(p.read_text())
            if value.get('artificial_only') or value.get('artificial'):
                raise ValueError('Artificial fixtures cannot be published to the production map')
            files[name]=dict(path=p.relative_to(root).as_posix(),
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    if not files:raise ValueError('No saved controller metadata found')
    # No host guess from absolute run paths or usernames. A producer may list
    # several executor hosts, but it publishes just one controller adoption.
    value=dict(schema='H4_SEARCH_PUBLICATION_V1',role='CONTROLLER',controller_adopted=True,
        run_id=run_id or relative,run_directory=relative,execution_hosts=sorted(set(execution_hosts)),
        files=files,scope='SAVED_METADATA_ONLY_NO_SCIENTIFIC_EXECUTION',
        new_scientific_calls=0)
    progress = search_progress(run / 'KAPPA_STATE.json', root, value['run_id'])
    if progress is not None:
        value['search_progress'] = progress
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if output.exists()and output.read_text()==text:return 'UNCHANGED',value
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
    return 'UPDATED',value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path.cwd())
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,
                   help='Publish as docs/research/.../SEARCH_PUBLICATION.json')
    p.add_argument('--run-id');p.add_argument('--execution-host',action='append',default=[])
    a=p.parse_args();root=a.repository.resolve()
    def path(v):return v if v.is_absolute()else root/v
    status,value=prepare(root,path(a.run_dir),path(a.output),run_id=a.run_id,execution_hosts=a.execution_host)
    print(json.dumps(dict(status=status,run_id=value['run_id'],files=len(value['files']),
        next='Commit/push this index and referenced controller metadata on any public branch; C ordinary sync detects it')))


if __name__=='__main__':main()
