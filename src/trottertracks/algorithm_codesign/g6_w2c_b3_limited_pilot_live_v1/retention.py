"""Adopted compact retention after the first16 trials on each profile axis."""
from . import spec as S
from .guard import require
from ..g6_w2c_b3_real_adapter_v1 import codec as C

def compact(payload):
    ev=payload['evidence'];a=ev['adapted']
    return dict(schema='b3_pilot_compact_trial_v1',retention='FULL_CONSTRUCTED_HASHED_NOT_RECONSTRUCTABLE',
        trial_id=payload['trial_id'],axis=payload['axis'],kind=payload['kind'],event=payload['event'],
        generator_exclusive_ns=payload['generator_exclusive_ns'],
        exact_event=C.Document.create(payload['event']).transport(),
        classification=a['classification'],consumption=ev['consumption'],connection=ev['connection'],
        boundary_counts=ev['boundary_counts'],setup_digest=ev['setup_digest'],
        original_digest=a['codec']['original_fingerprint'],
        exact_codec_sha256=C.fingerprint(a['codec']),full_evidence_sha256=C.fingerprint(ev),
        angle_sha256=C.fingerprint(payload['angle']),angle_present=payload['angle'] is not None,
        original_projection=dict(phase_i_power=a['event']['phase_i_power'] if a['event'] else None,
            source_id=a['source_id'] if 'source_id' in a else None),
        artificial_only=ev['artificial_only'],runtime_origin=ev['runtime_origin'],
        fixed_invariants_checked_in_worker=True,general_correctness_certified=False)
