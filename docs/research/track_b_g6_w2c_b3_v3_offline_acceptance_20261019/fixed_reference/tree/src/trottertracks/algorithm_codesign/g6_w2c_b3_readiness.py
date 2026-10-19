"""Read-only B3 adoption checks and a strictly artificial admission model.

No G7/F3/kernel import, sampler, bitstream generator or production admission.
The mock accepts fixture-only plans in a private /tmp directory and has no
admit_once method compatible with the production driver. Science always rejects.
"""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from . import g6_w2c_b3_driver as D

BASE_COMMIT = 'e4af6c064db805cbec611bcc8f0ff75a4754d38b'
BASE_CONTRACT_ID = '28d84d1802ee4651aabf2bb5d497d6c486d1a8b00d42a06eb47c00ac6d6964a7'
MODE = 'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE'
ANGLE_PROFILES = (
    {'id': 'A6', 'tau': '1/1000000', 'K_phi': 32, 'J': 32, 'mode': MODE},
    {'id': 'A9', 'tau': '1/1000000000', 'K_phi': 48, 'J': 64, 'mode': MODE},
    {'id': 'A12', 'tau': '1/1000000000000', 'K_phi': 64, 'J': 96, 'mode': MODE},
)
FIXED_FIELDS = ('input', 'policy', 'source_pins', 'saved_evidence', 'saved_rows', 'cells',
                'passes', 'correctness_trials_per_cell', 'call_budget', 'acceptance', 'one_shot', 'output_relative')
MOCK_BINDING_FIELDS = ('contract_digest', 'source_sha256', 'input_sha256', 'schedule_sha256',
                       'call_budget_sha256', 'guard_sha256', 'angle_sha256')


class ReadinessStop(ValueError):
    pass


def require(ok, reason):
    if not ok:raise ReadinessStop(reason)


def angle_bound(profile):
    """Parameter-only sufficient inequality from existing F3 endpoint bounds.

    This does not evaluate an input/event/ratio or run the frozen enclosure.
    K>=3 gives rounded half-angle endpoint <=3/4; see the accompanying derivation.
    """
    k, j = profile['K_phi'], profile['J']
    require(type(k) is int and k >= 3 and type(j) is int and j > 0, 'ANGLE_PARAMETER_DOMAIN')
    require(type(profile['tau']) is str and F(profile['tau']) > 0 and profile['mode'] == MODE,
            'ANGLE_SCOPE_OR_TOLERANCE')
    bound = F(1, 1 << (k+1)) + 2*F(3, 4)**(2*j+1)/(2*j+1)
    return {'radius_upper': bound, 'tau': F(profile['tau']), 'sufficient': bound <= F(profile['tau']),
            'ratio_domain': '[0,2]', 'native_error_certificate': False, 'G7_event_evaluated': False}


def fixed_science(base):
    return {key: deepcopy(base[key]) for key in FIXED_FIELDS}


def binding_manifest(base):
    return {'base_contract_sha256': D.identity(base), 'input_sha256': D.identity(base['input']),
            'science_policy_sha256': D.identity(base['policy']),
            'sources_sha256': D.identity(base['source_pins']), 'cells_sha256': D.identity(base['cells']),
            'call_budget_sha256': D.identity(base['call_budget']),
            'schedule_policy_sha256': D.identity(base['schedule_proposal']),
            'seed_rule_sha256': D.identity(base['seed_proposal']),
            'digest_scheme': 'frozen driver identity: exact canonical tagged values, SHA256'}


def refine_draft(base):
    """A new data-only candidate; never overwrites/adopts the base contract."""
    require(D.identity(base) == BASE_CONTRACT_ID, 'FROZEN_BASE_CONTRACT_MISMATCH')
    return {'schema': 'g6_w2c_b3_adoption_candidate_v1', 'base_commit': BASE_COMMIT,
            'adopted': False, 'contract_grants_execution': False, 'separate_execution_approval': False,
            'fixed_science': fixed_science(base), 'binding_manifest': binding_manifest(base),
            'technically_resolved': {'estimand': 'USER_SPECIFIED_DESCRIPTIVE_FIXED_WORKLOAD',
                'angle_scope': MODE, 'sample_budget_unchanged': 31710848,
                'angle_support': 'EXISTING_F3_LEMMAS_WITH_ELEMENTARY_OUTWARD_BOUND',
                'production_science_entry_enabled': False},
            'proposed_values': {'angle_profiles': list(deepcopy(ANGLE_PROFILES)),
                'seed': deepcopy(base['seed_proposal']), 'schedule_and_output': deepcopy(base['schedule_proposal']),
                'package_R1': {'angle_id': 'A6', 'policy_bundle': 'UNCHANGED_B3_DESCRIPTIVE_V1'}},
            'adoption_pending': {'package': None},
            'execution_approval_pending': {'approval': None,
                'production_admission': 'NOT_IMPLEMENTED_NOT_AUTHORIZED',
                'production_source_commit': None, 'formal_contract_digest': None,
                'authorization_reference': None, 'fresh_scope_and_claim': None},
            'unsupported_unknown': {'real_G7_correctness': 'NOT_ESTABLISHED',
                'actual_runtime_memory_output': 'NOT_MEASURED', 'fresh_IID_cost_certificate': 'OUT_OF_SCOPE',
                'physical_RZ_or_CQ_guarantee': 'OUT_OF_SCOPE', 'power_loss_durability': 'NOT_GUARANTEED'},
            'STOP_conditions': ['unadopted_or_unapproved', 'source_input_contract_policy_mismatch',
                'scope_or_claim_spent', 'angle_or_preflight_failure', 'no_accepted_in_16_trials',
                'output_collision_or_save_failure', 'partial_failure', 'budget_exhausted',
                'STOP_no_retry_no_resume'], 'mandatory_STOP': True}


def validate_refined(candidate, base):
    """Check a draft only; a successful return carries no execution capability."""
    require(D.identity(base) == BASE_CONTRACT_ID, 'FROZEN_BASE_CONTRACT_MISMATCH')
    require(candidate['schema'] == 'g6_w2c_b3_adoption_candidate_v1', 'REFINED_SCHEMA')
    require(candidate['base_commit'] == BASE_COMMIT, 'BASE_COMMIT_MISMATCH')
    require(all(candidate[k] is False for k in ('adopted', 'contract_grants_execution',
                                               'separate_execution_approval')), 'DRAFT_IS_NOT_AUTHORITY')
    require(D.identity(candidate['fixed_science']) == D.identity(fixed_science(base)), 'FIXED_SCIENCE_OR_BUDGET_MISMATCH')
    require(candidate['binding_manifest'] == binding_manifest(base), 'CONTRACT_SOURCE_INPUT_BINDING_MISMATCH')
    proposed = candidate['proposed_values']
    require(proposed['angle_profiles'] == list(ANGLE_PROFILES), 'ANGLE_PROFILE_NOT_FIXED_OR_SUPPORTED')
    require(all(angle_bound(p)['sufficient'] for p in ANGLE_PROFILES), 'ANGLE_BOUND_NOT_SUFFICIENT')
    require(D.identity(proposed['seed']) == D.identity(base['seed_proposal']), 'SEED_NAMESPACE_OR_REUSE_MISMATCH')
    require(D.identity(proposed['schedule_and_output']) == D.identity(base['schedule_proposal']), 'SCHEDULE_CACHE_RETENTION_MISMATCH')
    require(proposed['package_R1'] == {'angle_id': 'A6', 'policy_bundle': 'UNCHANGED_B3_DESCRIPTIVE_V1'},
            'PACKAGE_NOT_FIXED')
    require(candidate['adoption_pending'] == {'package': None}, 'FORMAL_ADOPTION_PENDING')
    require(candidate['execution_approval_pending']['approval'] is None, 'SCIENCE_APPROVAL_NOT_ISSUED')
    require(candidate['execution_approval_pending']['production_admission'] == 'NOT_IMPLEMENTED_NOT_AUTHORIZED',
            'PRODUCTION_ADMISSION_MUST_STAY_DISABLED')
    require(candidate['mandatory_STOP'] is True, 'STOP_REQUIRED')
    return {'technical_candidate_consistent': True, 'adoption_required': True,
            'separate_execution_approval_required': True, 'production_admission_enabled': False,
            'science_calls': 0}


def reject_science(*args, **kwargs):
    raise PermissionError('PREPARATION_ONLY_PRODUCTION_ADMISSION_NOT_IMPLEMENTED_OR_ENABLED')


def check_source_snapshot(root, pins):
    """Read exact supplied files only, never import or evaluate scientific source."""
    return D.check_pins(root, pins)


def validate_mock_plan(plan, observed, authorization):
    require(plan['artificial_only'] is True and authorization['fixture_only'] is True, 'MOCK_ONLY')
    require(plan['scope'].startswith('ARTIFICIAL_SCOPE_') and
            plan['claim_id'].startswith('ARTIFICIAL_CLAIM_'), 'REAL_SCOPE_OR_CLAIM_FORBIDDEN')
    require(plan['angle_adopted_in_fixture'] is True, 'MOCK_ANGLE_UNADOPTED')
    require(plan['angle'] in ANGLE_PROFILES, 'MOCK_ANGLE_VALUES_NOT_BOUND')
    require(plan['seed_namespace'] == 'ARTIFICIAL_BIT_VALUES_NO_G7_STREAM', 'MOCK_SEED_NAMESPACE')
    require(type(plan['mock_trial_budget']) is int and plan['mock_trial_budget'] > 0, 'MOCK_CALL_BUDGET')
    require(set(plan['bindings']) == set(MOCK_BINDING_FIELDS), 'MOCK_BINDING_SET')
    require(all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in plan['bindings'].values()),
            'MOCK_COMPLETE_SHA_REQUIRED')
    require(observed == plan['bindings'], 'MOCK_OBSERVED_CONTRACT_SOURCE_INPUT_SCHEDULE_MISMATCH')
    require(authorization['approved_in_fixture'] is True and authorization['consumed_in_fixture'] is False,
            'MOCK_AUTHORIZATION_MISSING_OR_OLD')
    require(authorization['scope'] == plan['scope'] and
            authorization['contract_digest'] == plan['bindings']['contract_digest'], 'MOCK_AUTHORIZATION_BINDING')
    require(type(plan['retry']) is int and plan['retry'] == 0 and plan['resume'] is False,
            'MOCK_RETRY_OR_RESUME_FORBIDDEN')


class MockAtomicScope:
    """O_EXCL uniqueness demonstration on own private local temporary files only.

    Existence is spent even for a torn/empty write. No deletion, recovery or retry.
    This is not a production authority port and cannot satisfy Boundary.admit_once.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()) and
                self.root.name.startswith('b3-artificial-') and self.root.is_dir() and
                self.root.stat().st_uid == os.getuid() and (self.root.stat().st_mode & 0o077) == 0,
                'PRIVATE_ARTIFICIAL_TEMP_REQUIRED')

    def admit_mock(self, plan, observed, authorization, fake_clock):
        plan, observed, authorization = deepcopy((plan, observed, authorization))
        validate_mock_plan(plan, observed, authorization)
        # Scope key deliberately excludes contract digest: changing the contract
        # does not provide a second chance under the same approved logical scope.
        key = sha256(plan['scope'].encode()).hexdigest()
        path = self.root/('mock_scope_'+key+'.json')
        payload = json.dumps({'fixture_only': True, 'scope': plan['scope'], 'claim_id': plan['claim_id'],
                              'plan_digest': D.identity(plan), 'bindings': plan['bindings']}, sort_keys=True).encode()
        claim_key = sha256(plan['claim_id'].encode()).hexdigest()
        claim_path = self.root/('mock_claim_'+claim_key+'.json')
        # No rollback: a partial two-file reservation burns the fixture scope.
        # No execution/session is returned until both immutable bindings persist.
        for current in (path, claim_path):
            try:fd = os.open(current, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError as exc:raise ReadinessStop('MOCK_SCOPE_OR_CLAIM_ALREADY_SPENT_NO_RETRY') from exc
            try:
                view = memoryview(payload)
                while view:
                    count = os.write(fd, view)
                    require(type(count) is int and count > 0, 'MOCK_SHORT_CLAIM_WRITE')
                    view = view[count:]
                os.fsync(fd)
            finally:os.close(fd)
        return MockSession(self.root, key, plan, fake_clock)


class MockSession:
    """Records supplied artificial events; never calls a generator or bit source."""
    def __init__(self, root, key, plan, fake_clock):
        self.root, self.key, self.plan = root, key, deepcopy(plan)
        self.plan_id = D.identity(self.plan)
        self.clock = fake_clock
        self.meter = D.Meter(fake_clock)
        self.stream = D.Stream(root/('mock_prefix_'+key+'.jsonl'), self.meter,
                               byte_limit=1000000, flush_rows=1)
        self.state, self.completed, self.spent = 'ACTIVE', [], [0, 0]
        self.failure = None

    def record_mock_trial(self, kind, event=None):
        require(self.state == 'ACTIVE', 'MOCK_STOP_NO_RETRY_OR_RESUME')
        before = self.clock()
        try:
            require(D.identity(self.plan) == self.plan_id, 'MOCK_PLAN_DRIFT')
            require(len(self.completed) < self.plan['mock_trial_budget'], 'MOCK_CALL_BUDGET_EXHAUSTED')
            D.Outcome(kind, event).validate()
            if event is not None:D.audit.event_witness(event, full=True, degree_cap=5, labels=3)
            row = {'fixture_only': True, 'trial': len(self.completed), 'kind': kind, 'event': event}
            self.stream.emit(row)
            self.completed.append(row)
            elapsed = D.delta(before, self.clock())
            self.spent = [a+b for a, b in zip(self.spent, elapsed)]
        except BaseException as exc:
            elapsed = D.delta(before, self.clock())
            self.spent = [a+b for a, b in zip(self.spent, elapsed)]
            self.stop_mock(type(exc).__name__+':'+str(exc))
            raise

    def stop_mock(self, reason='MOCK_COMPLETED_STOP'):
        require(self.state == 'ACTIVE', 'MOCK_STOP_NO_RETRY_OR_RESUME')
        self.state = 'STOPPED'
        self.failure = {'fixture_only': True, 'reason': reason, 'completed_trials': len(self.completed),
                        'fake_cpu_wall_prefix_ns': tuple(self.spent), 'retry': 0, 'resume': False,
                        'prefix_rows': self.stream.rows, 'prefix_hash': self.stream.previous,
                        'save_failure': self.stream.failed, 'control_self_write_cost_ns': None,
                        'mandatory_STOP': True, 'is_random_zero': False}
        D.write_control_metadata(self.root/('mock_stop_'+self.key+'.json'), self.failure, 100000)
        self.stream.close()
        return deepcopy(self.failure)

    def finish_mock(self):
        require(self.state == 'ACTIVE', 'MOCK_STOP_NO_RETRY_OR_RESUME')
        try:
            require(len(self.completed) == self.plan['mock_trial_budget'], 'MOCK_INCOMPLETE_PREFIX')
            require(any(r['kind'] == 'EVENT' for r in self.completed), 'MOCK_NO_ACCEPTED_NO_EXTRA_TRIALS')
        except ReadinessStop as exc:
            self.stop_mock(str(exc))
            raise
        return self.stop_mock('MOCK_COMPLETED_STOP')
