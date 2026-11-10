"""Passive fixed-A0 diagnostic intake. Never calls normalization/N/C/G code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import pf_pr_rpe_pipeline as p
from . import rpe_r_selection as r
from . import rpe_provider_integration as b

A0 = '6a75ebf967aaae2698ce6d295bf6fbec9d5e9929'
B_ALIGNMENT = '328cb26fa893e49197d15e0e075e3d9b43213980'
PLAN = 'docs/research/pr_r_normalization_phase0_preparation_20261111/PLAN.json'
ALIGNMENT = 'docs/research/r_screening_price_alignment_20261111/ALIGNMENT.json'
ROW_FIELDS = ('system', 'm', 'T', 'q_RPE', 'q_PF', 'delta', 'candidate_id',
              'R_outer', 'formula', 'prefix', 'K', 'structure', 'request')
SCORED = 'NORMALIZATION_SCORED_NOT_N_QUALIFIED'
COMPLETE = 'PHASE0_NORMALIZATION_RECORDED'


def _numeric_record_complete(record):
    """Check field availability/status only, without recomputing any diagnostic."""
    normal = record.get('normalization') or {}
    if normal.get('status') != SCORED:
        return False
    for field in ('log_B', 'optimistic_shot_inflation_log', 'one_ulp_accumulation_log_sensitivity'):
        if type(normal.get(field)) not in (int, float):
            return False
    for field in ('attenuation', 'shot_inflation'):
        display = record.get(field) or {}
        value, status = display.get('value'), display.get('status')
        if status == 'FINITE_BINARY64' and type(value) in (int, float):
            continue
        if status == 'UNDERFLOW_TO_ZERO_KEEP_LOG' and type(value) in (int, float) and value == 0:
            continue
        if status == 'OVERFLOW_BINARY64_KEEP_LOG' and value is None:
            continue
        return False
    return True


class A0Intake:
    """Reads committed metadata only; accepts diagnostic copies, never N records.

    Source hashes describe the scientific implementation, not user permission.
    No live producer paths, input arrays, checkpoints, or producer imports.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.store = p.SavedArtifacts(root)
        self.plan = self.store.read(dict(commit=A0, path=PLAN))
        self.alignment = self.store.read(dict(commit=B_ALIGNMENT, path=ALIGNMENT))
        self.expected = {}
        self.B_requests = {}
        for system, values in self.plan['fixed_R_menu'].items():
            proposal = self.store.read(dict(commit=self.plan['upstream_commit'],
                path='docs/research/pr_r_screening_20261111/' + system + '_METADATA_PROPOSAL.json'))
            menu = r.build_menu(proposal['base'], {int(m): rs for m, rs in values.items()},
                                sampling_law=proposal['menu']['sampling_law'])
            bmenu = deepcopy(menu)
            bmenu['sampling_law'] = deepcopy(b.LAW)
            for row in self.plan['requests']:
                if row['system'] != system:
                    continue
                cid = row['candidate_id']
                target = next(v for v in menu['candidates'] if v['candidate']['id'] == cid)
                if (cid in self.expected or not p.same(row['request'], r.request_context(menu, target))
                        or not p.same(row['structure'], target['structure'])):
                    raise ValueError('A0 candidate/request/structure differs from B metadata')
                c = target['candidate']
                expected = dict(system=system, m=target['m'], T=c['T'], q_RPE=target['q_RPE'],
                    q_PF=target['q_PF'], delta=c['delta'], candidate_id=cid, R_outer=c['R_outer'],
                    formula=c['formula'], prefix=c['prefix'], K=c['K'], structure=target['structure'],
                    request=r.request_context(menu, target))
                if any(not p.same(row[k], expected[k]) for k in ROW_FIELDS):
                    raise ValueError('A0 descriptor differs from B candidate')
                self.expected[cid] = deepcopy(row)
                self.B_requests[cid] = r.request_context(bmenu, target)
        ids = {v['candidate_id'] for v in self.alignment['A_normalization24']}
        if len(self.expected) != 24 or set(self.expected) != ids:
            raise ValueError('Keep the fixed A0 24-candidate scope')
        self.cells = {v['candidate_id']: v for v in self.alignment['C_phase1_cells']}
        self.overlap = sorted(set(self.expected) & self.cells.keys())
        if self.overlap != self.alignment['sets']['A24_C_overlap']:
            raise ValueError('C1/A0 overlap differs from fixed alignment')
        for cid in self.overlap:
            cell = self.cells[cid]
            ident = self.expected[cid]['request']['identity']
            if (any(not p.same(cell[k], ident[k]) for k in
                    ('candidate', 'structure', 'binding', 'profile', 'm', 'q_RPE', 'q_PF'))
                    or not p.same(cell['A_request_law_binding'], self.expected[cid]['request']['law_binding'])):
                raise ValueError('C1 candidate/H/state/compiler/law differs from A0')
        self._source_cache = set()

    def _source(self, source):
        if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{40}', source):
            raise ValueError('Pinned scientific SOURCE required')
        if source not in self._source_cache:
            for path, expected in self.plan['sources'].items():
                try:
                    raw = subprocess.check_output(['git', 'show', source + ':' + path],
                                                  cwd=self.root, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as exc:
                    raise ValueError('Scientific SOURCE unavailable: ' + path) from exc
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError('Scientific SOURCE differs from fixed A0: ' + path)
            self._source_cache.add(source)

    def readiness(self):
        return dict(status='A0_INTAKE_READY_NO_RESULTS_CONSUMED', A0_plan=dict(commit=A0, path=PLAN),
            alignment=dict(commit=B_ALIGNMENT, path=ALIGNMENT), candidate_count=24,
            candidates=[dict(candidate_id=cid, request=deepcopy(row['request']),
                B_provider_law_binding=deepcopy(self.B_requests[cid]['law_binding']),
                raw_law_alias_applied=False, normalization=None, intake_status='NOT_RECEIVED')
                for cid, row in self.expected.items()],
            C1_cells=[dict(candidate_id=cid, role=v['role'], intake_lane=v['intake_lane'],
                in_A0=cid in self.expected) for cid, v in self.cells.items()],
            C1_overlap=self.overlap, N_C_G=None, recommended_R=None,
            scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)

    def accept(self, handoff, result=None):
        """Copy diagnostics verbatim; companion result supplies preservation evidence.

        Missing, STOP, unresolved, and unrecognized numeric statuses stay visible.
        A HANDOFF alone is retained but cannot qualify diagnostic sharing.
        """
        # Standard JSON excludes NaN/Infinity while retaining null and integer precision.
        json.dumps([handoff, result], allow_nan=False)
        source = handoff['scientific_source']
        self._source(source)
        if (handoff.get('N_C_G') is not None or handoff.get('scientific_R_adopted') is not False
                or handoff.get('full_RPE_qualified') is not False
                or handoff.get('no_threshold_no_candidate_selection') is not True):
            raise ValueError('A0 must remain normalization-only without scientific adoption')
        preserved = False
        if result is not None:
            for key in ('scientific_source', 'status', 'records'):
                if not p.same(result[key], handoff[key]):
                    raise ValueError('HANDOFF/result mismatch: ' + key)
            for key, plan_key in (('source_files', 'sources'), ('input_hashes', 'input_hashes')):
                if not p.same(result.get(key), self.plan[plan_key]):
                    raise ValueError('Result SOURCE/input provenance mismatch: ' + key)
            preserved = result.get('source_preserved') is True and result.get('input_preserved') is True
            if any(result.get(k) != 0 for k in ('real_N_C_G', 'signal_phase_radius_actions', 'H_state_array_loads')):
                raise ValueError('Result exceeds normalization-only scope')
        received = {}
        for record in handoff['records']:
            cid = record['candidate_id']
            if cid not in self.expected or cid in received:
                raise ValueError('Unknown/duplicate A0 candidate: ' + cid)
            expected = self.expected[cid]
            if any(not p.same(record[k], expected[k]) for k in ROW_FIELDS):
                raise ValueError('Candidate/SOURCE binding/law mismatch: ' + cid)
            if (any(record.get(k) is not None for k in ('N', 'C', 'G', 'recommended_R', 'scientific_candidate_PASS_FAIL'))
                    or record.get('phase_radius_N_status') != 'NOT_EVALUATED'
                    or record.get('scientific_R_adopted') is not False):
                raise ValueError('Normalization must not supply N/C/G/PASS/selection')
            normal = record.get('normalization')
            if normal is not None:
                if normal.get('method') != expected['normalization']['method']:
                    raise ValueError('Normalization method differs')
                if normal.get('status') != 'NOT_EVALUATED' and not p.same(
                        normal.get('lambda_source'), self.plan['saved_tails'][record['system']]['source']):
                    raise ValueError('Normalization tail SOURCE differs')
            received[cid] = deepcopy(record)
        missing = [cid for cid in self.expected if cid not in received]
        all_scored = not missing and all(_numeric_record_complete(v) for v in received.values())
        blocks = []
        if handoff['status'] != COMPLETE:
            blocks.append('A0_ROOT_NOT_COMPLETE_RECORDED')
        if not all_scored:
            blocks.append('MISSING_OR_NUMERICALLY_UNQUALIFIED_RECORDS')
        if not preserved:
            blocks.append('MISSING_OR_FAILED_SOURCE_INPUT_PRESERVATION_EVIDENCE')
        if handoff.get('normalization_evidence_valid') is not True:
            blocks.append('A0_NORMALIZATION_EVIDENCE_NOT_VALID')
        usable = not blocks
        if handoff['status'].startswith('PHASE0_STOP'):
            completion = 'STOP'
        elif missing or handoff['status'] == 'PHASE0_RUNNING':
            completion = 'PARTIAL'
        elif handoff['status'] == COMPLETE and all_scored:
            completion = 'COMPLETE_RECORDED'
        else:
            completion = 'COMPLETE_UNQUALIFIED'
        rows = []
        for cid, expected in self.expected.items():
            record = received.get(cid)
            rows.append(dict(candidate_id=cid, expected=deepcopy(expected), record=record,
                intake_status='NOT_RECEIVED' if record is None else 'RECORDED',
                normalization_status=None if record is None else (record.get('normalization') or {}).get('status'),
                diagnostic_share_eligible=usable and record is not None,
                N_C_G_qualified=False))
        return dict(status='A0_DIAGNOSTICS_ONLY', completion=completion, scientific_source=source,
            source_preserved=None if result is None else result.get('source_preserved'),
            input_preserved=None if result is None else result.get('input_preserved'),
            normalization_evidence_usable=usable, evidence_blocks=blocks,
            received_count=len(received), missing_ids=missing,
            rows=rows, original_handoff=deepcopy(handoff), original_result=deepcopy(result),
            N_records=[], C_records=[], N_C_G=None, recommended_R=None,
            scientific_candidate_PASS_FAIL=None, scientific_R_adopted=False,
            full_RPE_qualified=False, mandatory_stop=True)

    def for_c1(self, accepted, candidate_id, *, request, normalization_source):
        """Explicit same-condition diagnostic link; never returns a price or N."""
        if candidate_id not in self.overlap:
            raise ValueError('C1 candidate is outside the fixed A0/C1 overlap')
        if (normalization_source != accepted['scientific_source']
                or not p.same(request, self.expected[candidate_id]['request'])):
            raise ValueError('C1 normalization SOURCE/request/law mismatch')
        row = next(v for v in accepted['rows'] if v['candidate_id'] == candidate_id)
        if row['record'] is not None and any(not p.same(row['record'][k], self.expected[candidate_id][k])
                                             for k in ROW_FIELDS):
            raise ValueError('C1 diagnostic record no longer matches accepted candidate')
        return dict(candidate_id=candidate_id, normalization_source=normalization_source,
            request=deepcopy(request), share_eligible=row['diagnostic_share_eligible'],
            diagnostic_record=deepcopy(row['record']), C=None, N=None, G=None,
            long_q_transfer_qualified=False, full_law_expected_price_qualified=False,
            scientific_R_adopted=False)

    def read_published(self, handoff_ref, result_ref=None):
        """Only explicit immutable publication pins; never reads A's live output."""
        accepted = self.accept(self.store.read(handoff_ref),
            None if result_ref is None else self.store.read(result_ref))
        accepted['publication'] = dict(handoff=deepcopy(handoff_ref), result=deepcopy(result_ref))
        return accepted
