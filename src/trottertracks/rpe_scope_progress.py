"""Read a finite scope snapshot alongside current operational observations."""
from collections import Counter

def scope_progress(document, preparation_records=(), observations=None):
    """Join the finite map with durable live preparation/queue observations.

    A completed Q3 job is not a completed configuration. Do not promote an
    unevaluated configuration or claim a new minimum from a preparation receipt.
    """
    preparation = {}
    for record in preparation_records:
        method = record.get('method')
        for cid in (record.get('configuration_id'), record.get('installed_configuration_id')):
            if method and cid:
                preparation[method, cid] = record
    observations = observations if observations is not None else {}
    summaries = {}
    for method, scope in document['methods'].items():
        counts = Counter()
        for row in scope['rows']:
            status = row['status']
            cid = row.get('configuration_id') or row.get('evaluation_configuration_id')
            key = row.get('logical_configuration_id') or cid
            old = observations.get(key, {}) if key else {}
            record = preparation.get((method, key)) or preparation.get((method, cid)) or {}
            stage_progress = record.get('stage_progress') or {}
            # A changed price is not an optimizer result until re-quoted.
            if row.get('price_required') and stage_progress.get('price') == 'AVAILABLE':
                status = 'PRICE_AVAILABLE_PENDING_REPRICE'
            elif row.get('price_required') and stage_progress.get('price_requested'):
                status = stage_progress.get('price') or status
            if old.get('state') in ('PREPARATION_RUNNING','EVALUATION_RUNNING'):
                status = old['state']
            counts[status] += 1
            if not key:
                continue
            observation = observations.setdefault(key, dict(candidate_id=key, method=method))
            observation.setdefault('scope_rows', []).append(dict(
                coordinates=row['coordinates'], state=status, snapshot_state=row['status'],
                allocations=row.get('allocations', {})))
            if stage_progress:
                observation['stage_progress'] = stage_progress
            # Only a live running job beats the finite-scope classification.
            if observation.get('state') not in ('PREPARATION_RUNNING','EVALUATION_RUNNING'):
                states = {item['state'] for item in observation['scope_rows']}
                observation['state'] = next(iter(states)) if len(states)==1 else 'MIXED_SCOPE_PROGRESS'
        summaries[method] = {k:v for k,v in scope.items() if k!='rows'}
        summaries[method]['live_counts'] = dict(counts)
    return summaries
