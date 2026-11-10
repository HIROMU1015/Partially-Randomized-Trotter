"""Passive C1 R-price intake and R-scoped fixed prices for B. No science here.

B receives prices; it never samples, compiles, counts S, fits, predicts,
re-evaluates residuals, solves N or sums N*C in this module. C1 records are
copied from pinned Git JSON after candidate/law/role checks. A new R has a
price only when C publishes one; fixed-R coefficients are never moved.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import types

from . import pf_pr_rpe_pipeline as p
from . import rpe_r_selection as r
from . import rpe_provider_integration as b
from .r_variable_compiled_price import describe_request

C_PLAN_COMMIT = 'c596e88d2a88fb219c2e373fb0d72c6e39cb723a'
C_SOURCE = '1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7'
C_PLAN = dict(commit=C_PLAN_COMMIT, path='docs/research/r_price_phase1_preparation_20261111/PLAN.json')
C_MODULE = 'src/trottertracks/r_price_phase1.py'
RTE = 'src/trotterlib/rte.py'
# C's declared sampler. B's N request law stays b.LAW; neither is aliased.
C_LAW = dict(model='ideal_finite_RTE_law', sampling='PCG64_numpy_float64_CDF_saved_cohort',
             source=dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe', path=RTE))
C1_R_MENU = {'H4': {3: [4, 8], 4: [4, 8]}, 'H6': {4: [16, 32], 5: [16, 32]}}
FIXED_R = {'H4': 4, 'H6': 16}
FILES = dict(plan='PLAN.json', frozen='FROZEN_MODELS.json', CALIBRATION='CALIBRATION_EVALUATION.json',
             VALIDATION='VALIDATION.json', rare='RARE_DIAGNOSTICS.json', stop='STOP.json')
RAW = dict(CALIBRATION='CALIBRATION.json', VALIDATION='VALIDATION.json')
EVALUATED = dict(CALIBRATION='CALIBRATION', VALIDATION='FINITE_INDEPENDENT_POINT_EVALUATED_NOT_ADOPTED')
KINDS = {'DIRECT_COMPILED_FINITE_SAMPLE': ('DIRECT:', 'observed_cohort_mean'),
         'MODEL_PREDICTION': ('MODEL:', 'predicted_cohort_mean')}
NULL_FIELDS = ('R_q_systematic', 'joint_covariance', 'total_CI', 'all_law_expectation')
UNQUALIFIED = dict(R_q_systematic=None, cross_R_covariance=None, cross_round_covariance=None,
                   joint_covariance=None, total_CI=None, all_law_expected_price=None)
REUSE_KEYS = ['wrapper_fingerprint', 'compiler', 'cohort', 'path_index', 'event_digest']
OUTSIDE_DESIGN = ('OUTSIDE_R_VARIABLE_PRICE_DESIGN_RECIPE', 'H4_DF10_OR_H6_DF19_PRICE_DESIGN_ONLY')


def _blob(root, commit, path):
    try:
        return subprocess.check_output(['git', 'show', commit+':'+path], cwd=root, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None


def pinned_phase1(root):
    """C's own pinned plan checks/digest. Its fit/price functions are never called."""
    module = types.ModuleType('trottertracks._c1_pinned_phase1')
    module.__package__ = 'trottertracks'
    module.__file__ = str(Path(root)/C_MODULE)
    exec(compile(_blob(root, C_SOURCE, C_MODULE), C_SOURCE+':'+C_MODULE, 'exec'), module.__dict__)
    return module


def role(cell):
    if cell['role'] == 'VALIDATION':
        return 'INDEPENDENT_VALIDATION'
    return 'SAVED_CALIBRATION' if cell['saved'] else 'NEW_CALIBRATION'


def price_kind(cell, classification):
    if classification == 'MODEL_PREDICTION':
        return 'MODEL_PREDICTION_AT_' + ('VALIDATION' if cell['role'] == 'VALIDATION' else 'CALIBRATION') + '_POINT'
    return role(cell) + '_DIRECT'


def _number_or_null(value):
    return value is None or (not isinstance(value, bool) and isinstance(value, (int, float))
                             and math.isfinite(value) and value >= 0)


def _fail(reason):
    raise ValueError('C1_INTAKE_REJECTED: ' + reason)


class C1PriceIntake:
    """Pinned C1 PLAN/result reader. Copies records; no numerical work.

    `accept` takes already-loaded public files; `read_published` takes one
    explicit commit/run directory. Missing stages stay NOT_RECEIVED.
    """
    def __init__(self, root, *, plan_ref=C_PLAN):
        self.root = Path(root)
        self.store = p.SavedArtifacts(root)
        self.c = pinned_phase1(root)
        self.plan_ref = deepcopy(plan_ref)
        self.plan = self.store.read(plan_ref)
        self.c.validate_plan(self.plan)
        if self.plan['source'] != C_SOURCE or self.plan['acceptance_threshold'] is not None:
            _fail('pinned C SOURCE without an adoption threshold required')
        self.digest = self.c.digest(self.plan)
        self.law = self._law_correspondence()
        self.menus = {s: r.build_menu(b.nmodel.saved_task(root, s), C1_R_MENU[s], sampling_law=b.LAW)
                      for s in C1_R_MENU}
        self.cells = {}
        for cell in self.plan['cells']:
            menu = self.menus[cell['system']]
            row = next(v for v in menu['candidates'] if v['candidate']['id'] == cell['candidate_id'])
            request = r.request_context(menu, row)
            law = cell['request']['law_binding']
            if not p.same(cell['request']['identity'], request['identity']):
                _fail('C1 cell identity differs from B menu: ' + cell['id'])
            if (set(law) != set(request['law_binding']) or not p.same(law['definition'], C_LAW)
                    or any(not p.same(law[k], request['law_binding'][k]) for k in law if k != 'definition')):
                _fail('C1 law binding differs from B request: ' + cell['id'])
            self.cells[cell['id']] = dict(cell=deepcopy(cell), role=role(cell), row=deepcopy(row),
                                          request=request)
        self.rare_cohorts = {v['cohort'] for v in self.plan['rare_groups']}

    def _law_correspondence(self):
        """Explicit N-law/C-sampler map. Same rte.py blob is evidence, not a distance bound."""
        hashes = {}
        for name, commit in (('N_law', b.LAW['source']['commit']), ('C_sampler', C_LAW['source']['commit'])):
            data = _blob(self.root, commit, RTE)
            if data is None:
                _fail('law definition SOURCE unavailable: ' + commit)
            hashes[name] = hashlib.sha256(data).hexdigest()
        if hashes['N_law'] != hashes['C_sampler']:
            _fail('N law and C sampler use different rte.py definitions')
        return dict(N_law=deepcopy(b.LAW), price_sampler_law=deepcopy(C_LAW), rte_path=RTE,
                    rte_sha256=hashes['N_law'], rte_commits=[b.LAW['source']['commit'], C_LAW['source']['commit']],
                    basis='SAME_CANDIDATE_R_K_QPF_AND_SAME_rte.py_BLOB',
                    ideal_sampler_distance_qualified=False, raw_law_alias_applied=False)

    def cell_for(self, row):
        return next((v for v in self.cells.values() if p.key(v['row']) == p.key(row)
                     and all(p.same(v['row'][k], row[k]) for k in ('q_RPE', 'q_PF', 'candidate', 'structure'))), None)

    def readiness(self):
        return dict(status='C1_PRICE_INTAKE_READY_NO_RESULTS_CONSUMED', plan=deepcopy(self.plan_ref),
                    plan_digest=self.digest, C_source=C_SOURCE, law_correspondence=deepcopy(self.law),
                    cells=[dict(cell_id=k, candidate_id=v['cell']['candidate_id'], system=v['cell']['system'],
                                m=v['row']['m'], T=v['row']['candidate']['T'], q_PF=v['row']['q_PF'],
                                q_RPE=v['row']['q_RPE'], delta=v['row']['candidate']['delta'],
                                R_outer=v['row']['candidate']['R_outer'], allocation=v['row']['structure']['allocation'],
                                event_count=v['row']['structure']['event_count'], role=v['role'],
                                cohort=v['cell']['cohort'], classical_paths=v['cell']['n'],
                                B_request=deepcopy(v['request']), C_law_binding=deepcopy(v['cell']['request']['law_binding']),
                                join_keys={a: describe_request(self.menus[v['cell']['system']], v['row'], a, pipeline=p)['join_key']
                                           for a in p.AXES},
                                intake_status='NOT_RECEIVED') for k, v in self.cells.items()],
                    prices=[], rare_diagnostics=[], model_qualified_for_candidate=False, N_C_G=None,
                    scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)

    def _frozen(self, frozen, committed):
        if frozen is None:
            _fail('FROZEN_MODELS.json is required with any evaluated stage')
        if committed is not None and not p.same(frozen, committed):
            _fail('frozen model differs from the frozen commit')
        if (frozen.get('plan_digest') != self.digest or frozen.get('validation_prices_observed') is not False
                or frozen.get('validation_refit') is not False or frozen.get('acceptance_threshold') is not None
                or frozen.get('full_RPE_qualified') is not False or frozen.get('model') != self.plan['model']
                or set(frozen.get('models', {})) != set(C1_R_MENU)):
            _fail('frozen model plan/role/threshold binding')
        groups = {}
        for system, model in frozen['models'].items():
            expected = {k for k, v in self.cells.items() if v['cell']['system'] == system and v['cell']['role'] == 'CALIBRATION'}
            roles = {k: v['cell']['role'] for k, v in self.cells.items() if v['cell']['system'] == system}
            if (set(model.get('calibration_cells', [])) != expected or len(model['calibration_cells']) != 3
                    or model.get('point_roles') != roles or model.get('qualified_for_candidate') is not False
                    or model.get('unobserved_q_R_qualified') is not False
                    or any(model.get(k) is not None for k in ('R_q_systematic', 'total_CI', 'all_law_expected_price'))
                    or not str(model.get('shared_calibration_group', '')).startswith('R_PRICE_P1:'+system+':')):
                _fail('frozen calibration cells/roles/qualification: ' + system)
            groups[system] = model['shared_calibration_group']
        return groups

    def _stage(self, stage, document, groups, publication, frozen_commit):
        json.dumps(document, allow_nan=False)
        cells = {k for k, v in self.cells.items() if v['cell']['role'] == stage}
        if document.get('full_rpe_resources') is not None:
            _fail(stage + ' carries full RPE resources')
        if stage == 'VALIDATION' and (document.get('frozen_model_commit') != frozen_commit
                                      or document.get('validation_refit') is not False
                                      or any(document.get(k) is not None for k in ('all_law_expected_price', 'total_CI'))):
            _fail('validation must use the committed frozen model without refit or CI')
        evaluations = {}
        for ev in document['evaluations']:
            cid = ev['cell_id']
            if cid not in cells or cid in evaluations:
                _fail(stage + ' evaluation outside its role or repeated: ' + str(cid))
            if (ev['role'] != self.cells[cid]['cell']['role'] or ev['price_unit'] != 'FINITE_COHORT_ONLY'
                    or ev['qualified_for_candidate'] is not False or ev['validation'] != EVALUATED[stage]
                    or ev['total_CI'] is not None or ev['all_law_expected_price'] is not None):
                _fail(stage + ' evaluation qualification/role: ' + cid)
            evaluations[cid] = ev
        if set(evaluations) != cells:
            _fail(stage + ' requires its complete fixed cell batch')
        out = []
        seen = set()
        for record in document['prices']:
            item = self._record(stage, record, evaluations, groups, publication, frozen_commit)
            k = item['c1']['cell_id'], item['axis'], item['classification']
            if k in seen:
                _fail('repeated C1 price record: ' + str(k))
            seen.add(k)
            out.append(item)
        if seen != {(c, a, k) for c in cells for a in p.AXES for k in KINDS}:
            _fail(stage + ' requires direct and model X/Y records for every cell')
        return out, evaluations

    def _record(self, stage, record, evaluations, groups, publication, frozen_commit):
        json.dumps(record, allow_nan=False)
        entry = next((v for v in self.cells.values() if p.same(record.get('identity'), v['cell']['request']['identity'])), None)
        if entry is None:
            _fail('price identity outside the fixed C1 cells')
        cell, row = entry['cell'], entry['row']
        if cell['role'] != stage:
            _fail('price record from the other calibration/validation role: ' + cell['id'])
        axis, kind = record.get('axis'), record.get('classification')
        if axis not in p.AXES or kind not in KINDS:
            _fail('axis/classification: ' + cell['id'])
        prefix, field = KINDS[kind]
        measured = p.measurement(row, axis, saved_sampler=True)
        if any(not p.same(record.get(k), v) for k, v in measured.items()):
            _fail('measurement axis/sign/unit/law context: ' + cell['id'])
        if (not p.same(record.get('law_binding'), cell['request']['law_binding'])
                or record['law_context']['price_law'] != C_LAW['sampling']):
            _fail('C law binding or sampler differs: ' + cell['id'])
        if (record.get('cohort') != prefix + cell['cohort'] or record['cohort'][len(prefix):] in self.rare_cohorts
                or 'FORCED' in record['cohort'] or record.get('n_paths') != cell['n']
                or record.get('expectation') != 'FINITE_COHORT_ONLY'
                or record.get('normalization_is_price_multiplier') is not False):
            _fail('cohort/path count/expectation/normalization: ' + cell['id'])
        value = record.get('mean', {}).get('rz_count')
        ev = evaluations[cell['id']]
        if not _number_or_null(value) or value != ev[field][list(p.AXES).index(axis)]:
            _fail('price value differs from its saved evaluation: ' + cell['id'])
        source = record.get('source') or {}
        if (source.get('execution_source') != self.plan['source']
                or source.get('result_path') != publication['run_dir'] + '/' + RAW[stage]
                or (stage == 'VALIDATION' and source.get('frozen_model_commit') != frozen_commit)):
            _fail('execution SOURCE/result path/frozen commit: ' + cell['id'])
        unc = record.get('uncertainty') or {}
        group = groups[cell['system']]
        if any(unc.get(k) is not None for k in NULL_FIELDS):
            _fail('unqualified uncertainty supplied as known: ' + cell['id'])
        if kind == 'MODEL_PREDICTION':
            model = record.get('model') or {}
            if (model.get('name') != self.plan['model'] or model.get('q_variable') != 'q_PF'
                    or model.get('q') != row['q_PF'] or not p.same(model.get('domain'), row['candidate'])
                    or not p.same(model.get('binding'), self.menus[cell['system']]['binding'])
                    or model.get('compiler') != p.PROFILE['compiler'] or model.get('coverage') != ev['validation']
                    or not p.same(model.get('source'), source) or not p.same(model.get('validation_source'), source)
                    or unc.get('shared_calibration_group') != group):
                _fail('model domain/compiler/source/shared group: ' + cell['id'])
            if model.get('qualified_for_candidate') is not False:
                _fail('C has no adoption threshold; model qualification must stay false')
        else:
            data_group = group if cell['role'] == 'CALIBRATION' else None
            if (record.get('model') is not None or unc.get('paired_group') != 'DIRECT:'+cell['id']
                    or unc.get('calibration_data_group') != data_group):
                _fail('direct pairing/calibration-data group: ' + cell['id'])
        out = deepcopy(record)
        out['c1_original_law_binding'] = deepcopy(record['law_binding'])
        out['law_binding'] = deepcopy(entry['request']['law_binding'])
        out['law_correspondence'] = deepcopy(self.law)
        out['source_cohort'] = record['cohort']
        out['cohort'] = 'C1:' + record['cohort']
        out['provider_quote'] = dict(provider='C1_PHASE1_INTAKE', price_unit='cohort_mean',
                                     validation=ev['validation'], reason=None)
        out['c1'] = dict(cell_id=cell['id'], role=entry['role'], stage=stage, price_kind=price_kind(cell, kind),
                         plan=deepcopy(self.plan_ref), plan_digest=self.digest, publication=deepcopy(publication),
                         frozen_model_commit=frozen_commit, model_application_domain='C1_EVALUATED_POINT_ONLY',
                         same_saved_cohort_as_fixed_catalog=cell['cohort'] if cell['saved'] else None,
                         classical_paths=cell['n'], classical_paths_are_quantum_shots=False,
                         xy_pair=cell['id']+':'+kind, independent_classical_samples=cell['n'],
                         join_key=describe_request(self.menus[cell['system']], row, axis, pipeline=p)['join_key'],
                         join_key_is_cache_evidence=False, compiled_price_reuse_requires=list(REUSE_KEYS),
                         uncertainty=dict(path_price_covariance=deepcopy(ev['paired_price_covariance']),
                                          nominal_mean_price_covariance=deepcopy(ev['nominal_mean_price_covariance']),
                                          paired_residual_covariance=deepcopy(ev['paired_residual_covariance']),
                                          nominal_mean_residual_covariance_given_frozen_fit=deepcopy(
                                              ev['nominal_mean_residual_covariance_given_frozen_fit']),
                                          validation_summaries=deepcopy(ev['summaries']) if stage == 'VALIDATION' else None,
                                          shared_calibration_group=group if kind == 'MODEL_PREDICTION' else None,
                                          calibration_data_group=unc.get('calibration_data_group'),
                                          nominal_covariance_is_guarantee=False, **UNQUALIFIED),
                         qualified_for_candidate=False, full_law_expected_price_qualified=False)
        return out

    def _rare(self, document):
        json.dumps(document, allow_nan=False)
        pairs = document.get('pairs') or []
        expected = {(v['system'], v['pair']): v for v in self.plan['rare_groups']}
        if (document.get('cohort_pooling') is not False or document.get('all_law_contribution') is not None
                or document.get('total_CI') is not None or len(pairs) != len(expected)
                or {(v.get('system'), v.get('pair')) for v in pairs} != set(expected)):
            _fail('rare diagnostics must be the four separate forced pairs')
        out = []
        for pair in pairs:
            group = expected[pair['system'], pair['pair']]
            if (pair.get('unit') != 'FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT'
                    or pair.get('all_law_weighted_contribution') is not None or len(pair.get('paths', [])) != 2
                    or any(v.get('conditional') is not True or v.get('cohort') != group['cohort']
                           or v.get('candidate_id') != group['candidate_id'] for v in pair['paths'])):
                _fail('forced rare pair unit/cohort/conditional binding')
            out.append(dict(diagnostic=deepcopy(pair), unit=pair['unit'], price_record=False,
                            pooled_with_cohort=False, used_in_fit=False, all_law_contribution=None))
        return out

    def accept(self, files, *, publication, frozen_committed=None):
        """files: {name: parsed public JSON}. Nothing is computed or fitted."""
        if not re.fullmatch('[0-9a-f]{40}', str(publication.get('commit'))) or not publication.get('run_dir'):
            _fail('pinned publication commit/run directory required')
        published = any(files.get(k) is not None for k in FILES)
        if published and (files.get('plan') is None or not p.same(files['plan'], self.plan)):
            _fail('published run PLAN differs from the pinned C1 PLAN')
        calibration, validation, rare = files.get('CALIBRATION'), files.get('VALIDATION'), files.get('rare')
        if (validation is not None and calibration is None) or (rare is not None and validation is None):
            _fail('stage order calibration -> validation -> rare required')
        stop = files.get('stop')
        if stop is not None and stop.get('status') not in ('PHASE1_FAILED_NO_RETRY', 'PHASE1_COMPLETE_MANDATORY_STOP'):
            _fail('unknown C1 STOP status')
        if stop is not None and stop['status'] == 'PHASE1_COMPLETE_MANDATORY_STOP' and rare is None:
            _fail('complete STOP without rare diagnostics')
        frozen_commit = None if validation is None else validation.get('frozen_model_commit')
        if validation is not None and not re.fullmatch('[0-9a-f]{40}', str(frozen_commit)):
            _fail('validation requires a 40-hex frozen model commit')
        if validation is not None and frozen_committed is None:
            _fail('validation requires the committed frozen model bytes')
        groups = None if calibration is None else self._frozen(files.get('frozen'), frozen_committed)
        prices, evaluations, received = [], {}, {}
        for stage, document in (('CALIBRATION', calibration), ('VALIDATION', validation)):
            received[stage] = document is not None
            if document is None:
                continue
            records, evs = self._stage(stage, document, groups, publication, frozen_commit)
            prices += records
            evaluations.update(evs)
        rare_rows = [] if rare is None else self._rare(rare)
        if stop is not None and stop['status'] == 'PHASE1_FAILED_NO_RETRY':
            completion = 'STOP'
        elif rare is not None and stop is not None:
            completion = 'PHASE1_COMPLETE_RECORDED'
        elif validation is not None:
            completion = 'VALIDATION_RECEIVED_RARE_PENDING'
        elif calibration is not None:
            completion = 'CALIBRATION_RECEIVED_VALIDATION_PENDING'
        else:
            completion = 'NOT_PUBLISHED'
        dependence = {}
        for v in prices:
            g = v['c1']['uncertainty']['shared_calibration_group'] or v['c1']['uncertainty']['calibration_data_group']
            if g:
                lane = 'model_predictions' if v['classification'] == 'MODEL_PREDICTION' else 'calibration_data'
                dependence.setdefault(g, dict(model_predictions=[], calibration_data=[]))[lane].append(
                    v['c1']['cell_id']+':'+v['axis']+':'+v['classification'])
        cells = []
        for k, v in self.cells.items():
            got = received[v['cell']['role']]
            cells.append(dict(cell_id=k, candidate_id=v['cell']['candidate_id'], system=v['cell']['system'],
                              m=v['row']['m'], q_PF=v['row']['q_PF'], R_outer=v['row']['candidate']['R_outer'],
                              role=v['role'], intake_status='RECEIVED' if got else 'NOT_RECEIVED',
                              price_kinds=sorted({x['c1']['price_kind'] for x in prices if x['c1']['cell_id'] == k}),
                              model_qualified_for_candidate=False))
        return dict(status='C1_PRICE_RECORDS_ONLY', completion=completion, publication=deepcopy(publication),
                    plan=deepcopy(self.plan_ref), plan_digest=self.digest, received=received,
                    rare_received=rare is not None, stop=deepcopy(stop), frozen_model_commit=frozen_commit,
                    shared_calibration_groups=groups, dependence_groups=dependence, cells=cells, prices=prices,
                    rare_diagnostics=rare_rows, law_correspondence=deepcopy(self.law), N_records=[], N_C_G=None,
                    totals=None, recommended_R=None, model_qualified_for_candidate=False,
                    long_q_or_new_R_transfer_qualified=False, full_law_expected_price_qualified=False,
                    scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)

    def read_published(self, commit, run_dir):
        """Read one explicit public C1 run; never the live C worktree or private raw."""
        if not re.fullmatch('[0-9a-f]{40}', commit) or run_dir.startswith('/') or '..' in run_dir.split('/'):
            _fail('pinned commit and repository-relative run directory required')
        files = {}
        for name, filename in FILES.items():
            data = _blob(self.root, commit, run_dir+'/'+filename)
            files[name] = None if data is None else json.loads(data)
        committed = None
        if files['VALIDATION'] is not None:
            frozen_commit = str(files['VALIDATION'].get('frozen_model_commit'))
            if not re.fullmatch('[0-9a-f]{40}', frozen_commit):
                _fail('validation requires a 40-hex frozen model commit')
            data = _blob(self.root, frozen_commit, run_dir+'/'+FILES['frozen'])
            if data is None or data != _blob(self.root, commit, run_dir+'/'+FILES['frozen']):
                _fail('frozen model bytes differ between frozen and publication commits')
            committed = json.loads(data)
        return self.accept(files, publication=dict(commit=commit, run_dir=run_dir), frozen_committed=committed)


def records_for(accepted, menu):
    """Exact-identity C1 records for a B menu; never moved to another q/R/round."""
    b._law(menu)
    out = []
    for record in accepted['prices']:
        row = next((v for v in menu['candidates'] if p.key(v) == p.key(record['identity'])), None)
        if row is None:
            continue
        request = r.request_context(menu, row)
        if not p.same(request['identity'], record['identity']) or not p.same(request['law_binding'], record['law_binding']):
            _fail('C1 record does not match this menu row: ' + record['c1']['cell_id'])
        out.append(deepcopy(record))
    return out


def price_scope(menu, row, axis, *, intake=None):
    """C's describe_request plus B law/C1 status. Metadata only."""
    c = row['candidate']
    try:
        scope = describe_request(menu, row, axis, pipeline=p)
    except ValueError as exc:
        if str(exc) not in OUTSIDE_DESIGN:
            raise
        scope = dict(identity=p.identity(menu, row), axis=axis, join_key=None, T=c['T'], q_PF=c['q'],
                     delta=c['delta'], R_outer=c['R_outer'], q_RPE=row['q_RPE'],
                     allocation=deepcopy(row['structure']['allocation']), event_count=row['structure']['event_count'],
                     R_evidence_status='OUTSIDE_R_VARIABLE_PRICE_DESIGN', design_reason=str(exc),
                     R_systematic_uncertainty=None, full_RPE_price_adopted=False)
    entry = None if intake is None else intake.cell_for(row)
    if p.arm(c) == 'PF':
        status = 'PF_DETERMINISTIC_NO_R'
    elif entry is not None:
        status = 'C1_' + entry['role']
    elif scope['R_evidence_status'] == 'OUTSIDE_R_VARIABLE_PRICE_DESIGN':
        status = 'OUTSIDE_R_VARIABLE_PRICE_DESIGN'
    elif c['R_outer'] == FIXED_R[c['system']]:
        status = 'FIXED_R_ANCHOR_PROVIDER_Q_DOMAIN'
    else:
        status = 'NEW_R_PRICE_NOT_ACQUIRED'
    scope.update(system=c['system'], m=row['m'], formula=c['formula'], prefix=c['prefix'], K=c['K'],
                 candidate_id=c['id'], signed_weights=deepcopy(row['structure']['signed_weights']),
                 compiler_binding={k: menu['profile'][k] for k in ('compiler', 'control', 'wrapper', 'scope', 'P')},
                 N_law_binding=r.request_context(menu, row)['law_binding'],
                 price_sampler_law=deepcopy(C_LAW) if p.arm(c) == 'PR' else None,
                 C1_cell=None if entry is None else dict(cell_id=entry['cell']['id'], role=entry['role']),
                 R_price_status=status, join_key_is_cache_evidence=False,
                 compiled_price_reuse_requires=list(REUSE_KEYS), unqualified=deepcopy(UNQUALIFIED))
    return scope


class RScopedFixedPrices:
    """B c_provider: existing fixed H4 R4/H6 R16 lookup plus R scope and transplant guard.

    This replaces C's RVariableSavedPriceProvider in B because B's provider
    path also binds law/domain (FixedSavedPrices). No new-R model is installed.
    """
    def __init__(self, read_json, system, *, selections=None, intake=None):
        self.fixed = b.FixedSavedPrices(read_json, system, selections=selections)
        self.intake = intake

    def __call__(self, menu, rows):
        b._law(menu)
        scopes = {(v['candidate']['id'], a): price_scope(menu, v, a, intake=self.intake) for v in rows for a in p.AXES}
        records = self.fixed(menu, rows)
        for record in records:
            c = record['identity']['candidate']
            scope = scopes[c['id'], record['axis']]
            if not p.same(record['identity'], scope['identity']):
                raise ValueError('FIXED_PROVIDER_RETURNED_FOREIGN_R_OR_BINDING')
            if (p.arm(c) == 'PR' and c['R_outer'] != FIXED_R[c['system']]
                    and record.get('mean', {}).get('rz_count') is not None):
                raise ValueError('FIXED_R_PRICE_CANNOT_BE_TRANSPLANTED')
            record['R_price_scope'] = scope
        return records


def coverage(menu, shots, prices, *, cohort_selection, price_policy='qualified_expectation',
             intake=None, accepted=None, a0=None):
    """B coverage plus normalization/N/C/G lanes. No N, C, G or provider calls."""
    out = r.coverage(menu, shots, prices, cohort_selection=cohort_selection, price_policy=price_policy)
    table = {(v['m'], v['candidate']['id']): v for v in menu['candidates']}
    a0rows = {v['candidate_id']: v for v in (a0 or {}).get('rows', [])}
    received = {v['c1']['cell_id'] for v in (accepted or {}).get('prices', [])}
    for item in out['rows']:
        row = table[item['m'], item['candidate_id']]
        scope = price_scope(menu, row, 'X', intake=intake)
        a = a0rows.get(item['candidate_id'])
        values = [v for v in item['price_contexts'] if v['value_available']]
        cell = scope['C1_cell']
        if values:
            c_status = 'PRICE_SCENARIOS_AVAILABLE'
        elif cell is not None:
            c_status = 'C1_' + ('RECEIVED_WITHOUT_VALUE' if cell['cell_id'] in received else 'NOT_INGESTED')
        elif item['price_contexts']:
            c_status = 'NULL_OR_OUTSIDE_DOMAIN_RECORD_ONLY'
        else:
            c_status = 'NOT_ACQUIRED'
        item.update(R_price_status=scope['R_price_status'], C1_cell=cell, C_status=c_status,
                    normalization=dict(status=None if a is None else a['normalization_status'],
                                       in_A0=a is not None, diagnostic_only=True,
                                       used_as_N=False, used_as_C_multiplier=False),
                    signal_phase_radius_N=('NOT_EVALUATED' if item['N_status'] == 'MISSING' else item['N_status']),
                    G='NOT_EVALUATED', long_q_or_new_R_transfer_qualified=False)
    out.update(status='R_PRICE_INTAKE_METADATA_ONLY', costs=None, G=None, totals=None,
               C1_completion=None if accepted is None else accepted['completion'],
               normalization_is_price_multiplier=False, classical_samples_are_shot_multipliers=False,
               scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)
    return out
