#!/usr/bin/env python3
"""Capture real optional-validation metadata without a queue or scientific body.

Uses saved LD2 prices, not a new model prediction.  The capture executor only
records Service._submit's portable request; it cannot start any worker.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks.pr_price_rules import GitReader, family, point
from trottertracks.rpe_price_validation import Service, canonical, digest, _coefficients
from trottertracks.rpe_preparation_artifacts import ArtifactCache

REGISTRY = dict(commit='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa',
                path='configs/pr_price_rules/h4_saved_rules.json')
PRICE_PYTHON = '/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class CaptureExecutor:
    """The production queue API is deliberately not imported or constructed."""
    def __init__(self, root, source):
        self.run_dir = Path(root)
        self.source_manifest = dict(commit=source)
        self.artifact_cache = ArtifactCache(self.run_dir/'artifacts')
        self.requests = []

    def submit_preparation(self, request, **_):
        self.requests.append(deepcopy(request))
        return request['request_id']

    def preparation_result(self, job, *, wait):
        if wait is not False:
            raise ValueError('Capture-only inspection cannot wait for a worker')
        return None


def capture(root, source, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != source:
        raise ValueError('Capture checkout must have the fixed implementation SOURCE')
    reader = GitReader(root)
    entries = {e['name']: e for e in reader.read(REGISTRY)['entries']}
    executor = CaptureExecutor(output, source)
    providers = {
        'PR_PAULI': dict(kind='PR_PAULI',backend='BUILTIN_PAULI_NORMAL_PATH',
            legacy_view=str(root.parent/'a-pauli-ld3-cal-source-13ab'),interpreter=PRICE_PYTHON),
        'PR_DF': dict(kind='PR_DF',backend='BUILTIN_DF_NORMAL_PATH',
            legacy_view=str(root.parent/'d2-pr-df-ld1-m3-acquisition-20261115'),interpreter=PRICE_PYTHON),
    }
    spec = dict(price_validation=dict(source=source,store=str(output/'capture-only.sqlite3'),providers=providers),
        executor=dict(distributed=dict(preparation=dict(budgets=dict(compile=0),
            runtime_requirements=dict(compile=dict(version_info_prefix=[3],
                packages=dict(qiskit='1.3.0',numpy=None,scipy=None)))))))
    service = Service(root, output, spec, executor=executor)
    rows = []
    for name, m in [('PAULI_LD2', 5), ('DF_LD2', 4)]:
        entry = entries[name]
        config = reader.read(entry['configuration_source'])
        prices = reader.read(entry['price_source'])['rows']
        prices = [r for r in prices if r['m']==m and r.get('L_D',2)==2]
        if len(prices)!=2 or {r['axis'] for r in prices}!={'X','Y'}:
            raise ValueError('Saved X/Y price metadata incomplete: '+name)
        r = config['rounds'][m]; p = point(r)
        model = reader.read(entry['model_source'])
        identity = canonical(dict(family=family(config,entry['implementation']),
            actual_step=p['step'],piece_pattern=p['piece_pattern'],event_recipe=p['event_recipe'],
            mode=entry.get('mode'),coefficients=_coefficients(model),model_source=entry['model_source']))
        mid = digest(identity)
        desc = dict(model_id=mid,identity=identity,registry=name,
            calibration_sources=[entry[k] for k in ('model_source','calibration_evidence','calibration_handoff','direct_path_means') if entry.get(k)])
        # The Pauli bundle is the actual packet container. A configuration-only
        # selector is insufficient to build its original physical request.
        signal = deepcopy(entry['configuration_source'])
        if name=='PAULI_LD2':
            signal['selector'] = ['packets',1]
        else:
            signal = dict(commit='91349e703e0d09ef702ccd108745dfaf1acc29f7',
                path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD2.json')
        selected = dict(m=m,q=r['q_PF'],R=r['R_outer'],physical_id=r['candidate']['id'],
            predicted_RZ={row['axis']:row['C_ref'] for row in prices},
            selection_reason=dict(rule='SAVED_LD2_METADATA_DEPLOYMENT_INSPECTION_ONLY',
                source=entry['price_source'],not_a_new_validation_selection=True))
        service.store.register(desc,lambda:([],[]))
        row = service.store.row(mid)
        row.update(status='PENDING',selected_round=selected,selected_configuration=config,
            selected_configuration_source=entry['configuration_source'],selected_configuration_id=config['id'],
            selected_signal_source=signal)
        candidate = dict(method=identity['family']['method'],family='PR',configuration_id=config['id'],signal_source=signal)
        before = len(executor.requests)
        service._submit(mid,row,desc,candidate)
        saved = service.store.row(mid)
        if len(executor.requests)!=before+1:
            raise ValueError(name+' metadata request HOLD: '+str(saved.get('pending_reason')))
        request = executor.requests[-1]
        rows.append(dict(method=candidate['method'],model_id=mid,request_id=request['request_id'],
            configuration_id=config['id'],m=m,q=r['q_PF'],R=r['R_outer'],
            registry=REGISTRY,price_source=entry['price_source'],model_source=entry['model_source'],
            captured_without_queue=True,scientific_calls=0,compile_calls=0,queue_operations=0))
    write(output/'TASKS.json',dict(tasks=executor.requests,scientific_calls=0,compile_calls=0,queue_operations=0))
    write(output/'CAPTURE_SUMMARY.json',dict(source=source,rows=rows,
        purpose='REAL_REQUEST_METADATA_INSPECTION_NOT_NEW_MODEL_VALIDATION',
        production_store_touched=False,scientific_calls=0,compile_calls=0,queue_operations=0))
    return executor.requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tasks=capture(ROOT,args.source,args.output)
    print(json.dumps(dict(tasks=len(tasks),source=args.source,scientific_calls=0,compile_calls=0,queue_operations=0)))


if __name__=='__main__': main()
