"""Future connection specifications. Only artificial documents are executable here."""
from dataclasses import dataclass, asdict
from copy import deepcopy
from .guard import require
from . import spec as S
from ..g6_w2c_b3_real_adapter_v1 import codec as C, fixture as X, result as R


@dataclass(frozen=True)
class ConnectionIdentity:
    source: str
    input: str
    policy: str
    runtime: str
    run: str
    scope_claim: str
    mode: str
    arm: str
    layer: str
    axis: str
    phase: str
    pass_name: str
    local_trial: int
    session: str

    def global_id(self):
        require(self.mode in ('S','P1') and self.arm in S.ARMS and self.layer in ('A','B')
                and self.axis in ('X','Y') and self.phase == 'correctness'
                and self.pass_name == 'baseline' and type(self.local_trial) is int
                and 0 <= self.local_trial < 16, 'TRIAL_IDENTITY')
        require(all(type(x) is str and x for x in
            (self.source,self.input,self.policy,self.runtime,self.run,self.scope_claim,self.session)), 'IDENTITY_FIELDS')
        return '/'.join((self.run,self.mode,self.phase,self.pass_name,self.arm,self.layer,self.axis,str(self.local_trial)))

    def binding(self):
        self.global_id()
        return S.digest(asdict(self))

    def proposed_seed_domain(self):
        self.global_id()
        # A string specification only. This release never feeds it to a bits object.
        return S.SEED + '/' + self.arm + '/' + self.axis


class FutureScientificResolver:
    """Public names/signatures are data; no module path loader/callable injection."""
    def describe(self, arm, layer):
        return S.route(arm,layer)

    def resolve(self, *args, **kwargs):
        raise RuntimeError('SEPARATE_VERSIONED_SCIENTIFIC_RELEASE_REQUIRED')


def connect_artificial(record, *, trial_id, arm, layer):
    require(type(record) is C.Document, 'DOCUMENT_REQUIRED')
    data = record.read()
    require(data.get('source_id') == X.SOURCE, 'ONLY_ARTIFICIAL_SOURCE_IN_THIS_RELEASE')
    return X.adapt(record,trial_id=trial_id,arm=arm,layer=layer)


def validate_connection(adapted, consumption, *, trial_id, arm, layer, axis,
                        angle=None, audit=None):
    """Revalidate documents and sidecars; no inference from driver projection alone."""
    from ..g6_w2c_b3_real_adapter_v1 import bits as B
    B.validate_trace(consumption,arm=arm,axis=axis)
    require(adapted['trial_id'] == trial_id and adapted['source_identity'] == X.SOURCE, 'RESULT_CONTEXT')
    original = C.restore(adapted['codec'], source_identity=X.SOURCE)
    data = original.read()
    if arm == 'full_return' and layer == 'A':
        observation = C.restore(adapted['g7_observation'],source_identity=X.SOURCE)
        data = observation.read()
        fresh = R.adapt_g7(data.get('event'),observation=observation,expected=X.context(trial_id),arm=arm)
    elif data.get('schema') == 'g7_observation_candidate_v1':
        fresh = R.adapt_g7(data.get('event'),observation=original,expected=X.context(trial_id),arm=arm)
    else:
        fresh = R.adapt_f3(original,expected=X.context(trial_id))
    require(C.fingerprint(fresh) == C.fingerprint(adapted), 'ORIGINAL_ADAPTER_MISMATCH')
    if layer == 'B':
        require(type(audit) is dict, 'AUDIT_REQUIRED')
        calls=int(trial_id.rsplit('/',1)[-1])+(16 if axis == 'Y' else 0)
        seal=C.DriverDocument.create(dict(arm=arm,identity=C.fingerprint(dict(arm=arm,source=X.SOURCE)),
            state_id=C.fingerprint(dict(artificial=True,arm=arm,layer=layer)),payload=original.document,
            completed_calls_before=calls,confidence_certified=False))
        expected_audit=R.canonical_audit(seal,arm=arm,identity=C.fingerprint(dict(arm=arm,source=X.SOURCE)),
            state_id=C.fingerprint(dict(artificial=True,arm=arm,layer=layer)),completed_calls_before=calls)
        require(C.fingerprint(audit) == C.fingerprint(expected_audit), 'AUDIT_SEAL_BINDING')
    if layer == 'B' and adapted['classification'] == 'ACCEPTED':
        require(type(angle) is dict, 'ANGLE_EVIDENCE_REQUIRED')
        query, original_angle = X.synthetic_angle(adapted)
        if arm == 'full_return':
            expected_angle = R.angle_binding(adapted,requery_input=query,
                requery_event=adapted['event'],angle_record=original_angle,synthetic=True)
        else:
            expected_angle = R.canonical_angle_binding(adapted,requery_input=query,
                requery_event=adapted['event'],angle_record=original_angle,
                input_payload={'id':'ARTIFICIAL_INPUT_V1'},policy_payload={'id':'ARTIFICIAL_PRECISION_V1'},synthetic=True)
        require(C.fingerprint(angle) == C.fingerprint(expected_angle), 'ANGLE_REQUERY_BINDING')
    else:
        require(angle is None, 'UNEXPECTED_ANGLE')
    return dict(original_digest=original.digest, classification=adapted['classification'],
                event_fingerprint=adapted['event_fingerprint'], live_compatibility='NOT_ESTABLISHED')


def coverage(rows):
    """Candidate classification, with no scheduling or retry side effect."""
    require(type(rows) is list, 'COVERAGE_ROWS')
    if any(r.get('unknown') for r in rows):
        return 'TECHNICAL_STOP_UNKNOWN'
    if any(r.get('technical_failure') for r in rows):
        return 'TECHNICAL_STOP'
    if any(r.get('scientific_mismatch') for r in rows):
        return 'SCIENTIFIC_MISMATCH_STOP'
    expected = {(mode,a,l,x) for mode in ('S','P1') for a in S.ARMS for l in ('A','B') for x in ('X','Y')}
    require(len(rows) == len(expected) and {tuple(r['cell']) for r in rows} == expected, 'COVERAGE_CELL_COMPLETENESS')
    require(all(type(r['completed']) is int and 0 <= r['completed'] <= 16 and
                type(r['accepted']) is int and 0 <= r['accepted'] <= r['completed'] for r in rows), 'COVERAGE_COUNTS')
    if any(r['completed'] != 16 for r in rows):
        return 'TECHNICAL_INCOMPLETE'
    return 'INCONCLUSIVE_COVERAGE' if any(r['accepted'] == 0 for r in rows) else 'CANDIDATE_CORRECTNESS_PASS'
