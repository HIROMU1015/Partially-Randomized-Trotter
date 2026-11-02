"""Confirmation scope around the unchanged, validated native acquisition kernel."""
from functools import partial
import json
from pathlib import Path

import h46_acquire_native as frozen
from h46_confirmation_contract import authorize, task_id, write, sha


def initialize(config_path, stop):
    config = json.loads(Path(config_path).read_text())
    authorize(config)
    frozen.initialize(config_path, stop)
    # Only registration changes; preparation, law, lowering and control stay frozen.
    frozen.task_id = partial(task_id, config['plan'])


def run_task(group, sample):
    c = frozen.CONFIG
    receipt = frozen.run_task(group, sample)
    directory = Path(c['output'])/task_id(c['plan'], group, sample)
    try:
        raw = json.loads((directory/'result.json').read_text())
        if raw['SOURCE'] != c['SOURCE'] or raw['seed_namespace'] != c['plan']['seed_namespace']:
            raise ValueError('NEW_CONFIRMATION_PROVENANCE_REQUIRED')
        if raw['plan_sha256'] != c['plan_sha256'] or raw['forced_events_enter_cost_sample'] is not False:
            raise ValueError('CONFIRMATION_PLAN_OR_FORCED_SAMPLE')
        result = {'status': 'CONFIRMATION_COST_VALIDATED', 'stage': 'CONFIRMATION',
                  'study_id': c['plan']['study_id'], 'SOURCE': c['SOURCE'],
                  'plan_sha256': c['plan_sha256'], 'seed_scope_digest': c['plan']['seed_scope_digest'],
                  'group': group, 'sample': sample, 'raw_native_result_sha256': sha(directory/'result.json'),
                  'legacy_kernel_receipt_retained_unchanged': True}
        write(directory/'confirmation.json', result)
        return {**receipt, 'stage': 'CONFIRMATION', 'confirmation_receipt_sha256': sha(directory/'confirmation.json')}
    except BaseException as exc:
        frozen.STOP.set()
        write(directory/'CONFIRMATION_STOP.json', {'reason': str(exc), 'retry': False})
        raise
