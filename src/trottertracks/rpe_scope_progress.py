"""Read a finite scope snapshot alongside current operational observations."""
from collections import Counter
from fractions import Fraction


def normalized_kappa(value):
    return str(Fraction(value['numerator'], value['denominator']) if isinstance(value, dict) else Fraction(value))

def scope_progress(document, preparation_records=(), observations=None, *, families=None, points=()):
    """Join the finite map with durable live preparation/queue observations.

    A completed Q3 job is not a completed configuration. Do not promote an
    unevaluated configuration or claim a new minimum from a preparation receipt.
    """
    preparation = {}
    by_design = {}
    for record in preparation_records:
        method = record.get('method')
        for cid in (record.get('configuration_id'), record.get('installed_configuration_id')):
            if method and cid:
                preparation[method, cid] = record
        request = record.get('request') or {}
        family = (request.get('selection') or {}).get('family')
        kappa = (request.get('design') or {}).get('kappa')
        if method and family and kappa is not None:
            by_design.setdefault((method, family, normalized_kappa(kappa)), []).append(record)
    families = families or {}
    planned = {(p['method'], p['family'], normalized_kappa(p['kappa'])): p
               for p in points if p.get('method') and p.get('family') and p.get('kappa') is not None}
    observations = observations if observations is not None else {}
    summaries = {}
    for method, scope in document['methods'].items():
        counts = Counter()
        horizons_by_family = {}
        for row in scope['rows']:
            horizons_by_family.setdefault(row.get('family'), set()).add(row['coordinates'].get('M'))
        for row in scope['rows']:
            status = row['status']
            cid = row.get('configuration_id') or row.get('evaluation_configuration_id')
            key = row.get('logical_configuration_id') or cid
            horizon = row['coordinates'].get('M')
            family = row.get('family')
            kappa = row['coordinates'].get('absolute_kappa')
            live_record = None
            # The publication snapshot can predate a mapper release. Its
            # unsupported rows have no physical ID, so join the exact declared
            # family/kappa to the current planner before reporting live state.
            if (status == 'MAPPER_UNSUPPORTED' and family and kappa is not None
                    and families.get(family, {}).get('status') == 'READY_METADATA'):
                design_key = (method, family, normalized_kappa(kappa))
                point = planned.get(design_key)
                status = 'SCHEDULE_GENERATION_REQUIRED' if point is None else 'PREPARATION_WAITING'
                matches = []
                for candidate in by_design.get(design_key, ()):
                    request = candidate.get('request') or {}
                    rounds = candidate.get('schedule') or ()
                    candidate_horizon = request.get('M') if not rounds else rounds[-1].get('m')
                    if candidate_horizon is None:
                        # A single declared horizon can identify an unfinished
                        # record even before it has a physical schedule.
                        family_horizons = horizons_by_family[family]
                        if len(family_horizons) != 1:
                            continue
                        candidate_horizon = next(iter(family_horizons))
                    if candidate_horizon == horizon:
                        matches.append(candidate)
                if len(matches) == 1:
                    live_record = matches[0]
                    cid = live_record.get('configuration_id') or live_record.get('installed_configuration_id')
                    key = cid or live_record.get('key')
                    state = live_record.get('status', '')
                    if state == 'READY_FOR_SEARCH':
                        status = 'EVALUATION_WAITING'
                    elif state.startswith(('HELD', 'STOP')):
                        status = ('SCIENTIFIC_FAILURE' if live_record.get('scientific_failure') else
                                  'TECHNICAL_FAILURE')
            old = (observations.get(cid) or observations.get(key) or {}) if key else {}
            record = live_record or preparation.get((method, key)) or preparation.get((method, cid)) or {}
            stage_progress = record.get('stage_progress') or {}
            # A changed price is not an optimizer result until re-quoted.
            if row.get('price_required') and stage_progress.get('price') == 'AVAILABLE':
                status = 'PRICE_AVAILABLE_PENDING_REPRICE'
            elif row.get('price_required') and stage_progress.get('price_requested'):
                status = stage_progress.get('price') or status
            live = (old.get('q3_horizons') or {}).get(horizon, {})
            if live.get('EVALUATION_RUNNING'):
                status = 'EVALUATION_RUNNING'
            elif live.get('UNKNOWN_ATTEMPT'):
                status = 'UNKNOWN_ATTEMPT'
            elif old.get('state') == 'PREPARATION_RUNNING' and row.get('price_required'):
                status = 'PREPARATION_RUNNING'
            counts[status] += 1
            if not key:
                continue
            observation = observations.setdefault(key, dict(candidate_id=key, method=method))
            observation.setdefault('scope_rows', []).append(dict(
                coordinates=row['coordinates'], state=status, snapshot_state=row['status'],
                allocations=row.get('allocations', {})))
            if stage_progress:
                observation['stage_progress'] = stage_progress
            if live:
                observation.setdefault('q3_horizons', {})[horizon] = live
            # Only a live running job beats the finite-scope classification.
            if observation.get('state') != 'PREPARATION_RUNNING':
                states = {item['state'] for item in observation['scope_rows']}
                observation['state'] = next(iter(states)) if len(states)==1 else 'MIXED_SCOPE_PROGRESS'
        summaries[method] = {k:v for k,v in scope.items() if k!='rows'}
        summaries[method]['live_counts'] = dict(counts)
    return summaries
