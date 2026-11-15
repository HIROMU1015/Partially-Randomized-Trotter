"""B2 PR-DF configuration -> reuse/request/unsupported -> existing Q3 bundle.

Only metadata, saved records and field-preserving carriers are handled here.
No generate/evaluate/load_h4_context/compile/Q3 leaf is called. A REQUEST is
input to the unchanged provider, not a numerical-gate PASS or permission.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .pr_price_rules import GitReader, frac

SIGNAL = 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
B_RELEASE = '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
PROVIDER_PATH = 'src/trottertracks/pr_rpe_signal_provider.py'
RECIPE = 'DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1'
IMPLEMENTATION = dict(commit=SIGNAL, path=PROVIDER_PATH, recipe=RECIPE)
INDEX = dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',
             path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
SCALAR = dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',
              path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/SCALAR.json')
LOCAL = 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
FORMULAS, ORDERS = ('2nd', '4th(m5_best)'), (2, 4, 6)
SAVED = (
    dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af', path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json'),
    dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af', path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD2.json'),
    dict(commit='0defa286c53a1b9acf513be453907bcad83102b3', path='docs/research/d_pr_df_ld7_v0_results_20261116/B_MEANS_LD7.json'),
    dict(commit='460d77d8f56ef0bde41dbddaa27852dd8b05513e', path='docs/research/d_pr_df_ld9_v0_results_20261116/B_MEANS_LD9.json'),
)


def _package(view, commit, name):
    """Isolate pinned relative imports from B2's newer trottertracks modules."""
    view = Path(view)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip() != commit:
        raise ValueError('Pinned runtime view differs: ' + str(view))
    if name not in sys.modules:
        directory = view / 'src/trottertracks'
        spec = importlib.util.spec_from_file_location(name, directory / '__init__.py',
            submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


class FixedRuntime:
    """Reuse the LD7/LD9 planner and B-metadata loader, without their CLI globals."""
    def __init__(self, root, view=None):
        root = Path(root).resolve()
        self.view = Path(view) if view else root.parent / 'd2-pr-df-ld7-signal-source-fefd-20261116'
        for path in (PROVIDER_PATH, 'src/trottertracks/pr_rpe_signal_plan.py',
                     'src/trottertracks/pr_rpe_rmse_s0_v2.py'):
            if (self.view / path).read_bytes() != subprocess.check_output(['git', 'show', SIGNAL+':'+path], cwd=root):
                raise ValueError('Pinned provider/metadata bytes differ: ' + path)
        name = _package(self.view, SIGNAL, '_pr_df_signal_fixed_fefd')
        fixed = importlib.import_module(name+'.pr_rpe_rmse_s0_v2')
        self.metadata = importlib.import_module(name+'.pr_rpe_signal_plan')
        self.provider = importlib.import_module(name+'.pr_rpe_signal_provider')
        self.common = self.metadata.common
        # Same extraction and B_RELEASE as run_pr_df_ld7_v0.signal_modules.
        node = next(n for n in ast.parse((self.view/'src/trottertracks/pr_rpe_rmse_s0_v2.py').read_bytes()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'b_metadata')
        scope = dict(vars(fixed), B_RELEASE=B_RELEASE)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'fixed:B_metadata', 'exec'), scope)
        self.b = scope['b_metadata'](root)

    def jobs(self, configuration):
        return self.metadata.plan(configuration,
            {r['m']: r['physical_candidate'] for r in configuration['rounds']},
            b_validate=self.b.validate, b_exact_scalar=frac,
            numerical_recipe=self.provider.numerical_recipe(IMPLEMENTATION))['jobs']


def _numbers(value):
    if isinstance(value, dict):
        if set(value) == {'numerator', 'denominator'}:
            f = frac(value); return ['NUMBER', str(f.numerator), str(f.denominator)]
        return {k: _numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_numbers(v) for v in value]
    if type(value) in (int, float):
        f = Fraction(value); return ['NUMBER', str(f.numerator), str(f.denominator)]
    return value


def _same(a, b):
    return _numbers(a) == _numbers(b)


def _definition(value):
    if isinstance(value, dict):
        return {k: _definition(v) for k, v in value.items() if k != 'candidate_id'}
    return value


class DFSignalEntry:
    def __init__(self, root, *, runtime=None, reader=None, metadata=None, saved=SAVED):
        self.root = Path(root).resolve()
        self.reader = reader or GitReader(root)
        self.runtime = runtime or FixedRuntime(root)
        if metadata is None:
            index = self.reader.read(INDEX)
            desc = index['T20_reference_files']['H4']
            data = Path(desc['path']).read_bytes()  # scalar/lambda metadata, no arrays/operators
            if hashlib.sha256(data).hexdigest() != desc['sha256']:
                raise ValueError('Saved tail descriptor differs')
            metadata = json.loads(data)
            metadata['descriptor_sha256'] = desc['sha256']
            metadata['hamiltonian_constant'] = self.reader.read(SCALAR)['hamiltonian_constant']
        self.info = deepcopy(metadata)
        self.saved = list(saved)
        self._index = None

    def configuration(self, value):
        """Canonical object, locator, B2 generated item, or {configuration: object}."""
        if 'commit' in value and 'path' in value:
            return self.reader.read(value)
        if 'configuration_source' in value:
            return self.configuration(value['configuration_source'])
        if isinstance(value.get('configuration'), dict):
            return self.configuration(value['configuration'])
        return deepcopy(value)

    def _support(self, config):
        s = config['spec']; split = s['split']; reasons = []
        if s['method'] != 'PR' or split.get('tail_dictionary'):
            reasons.append('NOT_PR_DF')
        if s['formula'] not in FORMULAS:
            reasons.append('FORMULA_NOT_SUPPORTED')
        if s['finite_K'] not in ORDERS:
            reasons.append('FINITE_K_NOT_SUPPORTED')
        ld = s['L_D']; tail = self.info['prefixes'].get(str(ld))
        if type(ld) is not int or not 1 <= ld < s['binding']['rank'] or tail is None:
            reasons.append('NONEMPTY_PREFIX_AND_TAIL_REQUIRED')
        expected = self.info['binding']
        if any(not _same(s['binding'].get(k), expected.get(k)) for k in
               ('system','basis','geometry','rank','rank_policy','hamiltonian','state','snapshot')):
            reasons.append('H_STATE_SNAPSHOT_BINDING_NOT_SUPPORTED')
        contract = s['binding'].get('decomposition', {}).get('contract', {})
        if contract.get('fragment_order', 'saved_snapshot_order') != 'saved_snapshot_order':
            reasons.append('FRAGMENT_ORDER_NOT_SUPPORTED')
        if (split.get('deterministic_scope') != 'PREFIX' or split.get('randomized_scope') != 'TAIL_AFTER_PREFIX'
                or split.get('one_body_policy') not in (None,'DETERMINISTIC')
                or split.get('original_PF_method') != 'B0'):
            reasons.append('HEAD_TAIL_POLICY_NOT_SUPPORTED')
        if s['law']['model'] != 'ideal_finite_RTE_law':
            reasons.append('LAW_NOT_SUPPORTED')
        law_source = s['law'].get('source', {})
        if law_source.get('composition_recipe') != IMPLEMENTATION or set(s['law']) != {'model','source'}:
            reasons.append('LAW_SOURCE_OR_EVENT_RULE_NOT_SUPPORTED')
        split_keys = {'L_D','source','randomized_scope','deterministic_scope','original_PF_method',
                      'lambda_R','lambda_source','one_body_policy','scalar'}
        if set(split)-split_keys:
            reasons.append('ADDITIONAL_SPLIT_RULE_NOT_SUPPORTED')
        if tail is not None:
            source = dict(index=INDEX, context_sha256=self.info['descriptor_sha256'], selector=['prefixes',str(ld)])
            if not _same(split['lambda_source'], source) or frac(split['lambda_R']) != Fraction(tail['lambda_R']):
                reasons.append('TAIL_LAMBDA_OR_SOURCE_NOT_SUPPORTED')
            if 'scalar' in split and not _same(split['scalar'], tail['scalar']):
                reasons.append('TAIL_SCALAR_NOT_SUPPORTED')
        return reasons

    def physical_key(self, record):
        c = record['physical_context']; p = c['physical_candidate']; ld = p['prefix']
        tail = self.info['prefixes'][str(ld)]
        value = dict(binding=c['binding'], split=c['split'], law=c['law'], profile=c['profile'],
            physical_candidate={k:v for k,v in p.items() if k != 'id'}, structure=c['structure'],
            numerical_recipe=c['numerical_recipe'],
            event_definition=_definition(record['B_law_binding']['definition']),
            head=dict(one_body='SEPARATE_HEAD', fragments=list(range(ld))),
            tail=dict(fragments=list(range(ld,self.info['binding']['rank'])), **tail),
            hamiltonian_constant=self.info['hamiltonian_constant'], scalar_application='ONCE_AT_TOTAL_T')
        text = json.dumps(_numbers(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def saved_index(self):
        if self._index is not None:
            return self._index
        index = {}
        for packet_source in self.saved:
            packet = self.reader.read(packet_source)
            config = self.reader.read(packet['configuration_source'])
            if self._support(config):
                continue
            jobs = self.runtime.jobs(config)
            for mean in packet['means']:
                rec = self.reader.read(mean['source']); m = mean['m']
                self._check_record(rec, jobs[m])
                if mean != mean_from_record(config['rounds'][m], rec, mean['source'], packet['target']):
                    raise ValueError('Saved bundle differs from raw record at m='+str(m))
                key = self.physical_key(rec)
                item = dict(source=mean['source'], packet=packet_source,
                            configuration_id=config['id'], m=m)
                if key in index and self.reader.read(index[key]['source']) != rec:
                    raise ValueError('Conflicting saved means: select explicit saved inventory')
                index.setdefault(key, item)
        self._index = index
        return index

    def _check_record(self, rec, job):
        for key in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context'):
            if not _same(rec.get(key), job[key]):
                raise ValueError('Provider record/job mismatch: '+key)
        if not _same(rec.get('candidate'),job['physical_context']['physical_candidate']):
            raise ValueError('Provider candidate mismatch')
        if (rec.get('status') != LOCAL or rec.get('numerical',{}).get('gate_pass') is not True
                or rec.get('scalar_once') is not True or not rec.get('input_source')
                or not rec.get('reference_source') or rec.get('source') != IMPLEMENTATION
                or rec.get('signal') is None):
            raise ValueError('Invalid/failed provider record; no automatic retry')

    def plan(self, configuration):
        cfg = self.configuration(configuration); reasons = self._support(cfg)
        jobs = []
        if not reasons:
            try:
                jobs = self.runtime.jobs(cfg)
            except (ValueError, KeyError, TypeError) as exc:
                reasons = ['CANONICAL_OR_PHYSICAL_INPUT_INVALID: '+str(exc)]
        rows = []; index = self.saved_index() if not reasons else {}
        for row in cfg['rounds']:
            m = row['m']
            if reasons:
                rows.append(dict(m=m, decision='UNSUPPORTED', reasons=reasons)); continue
            job = jobs[m]; key = self.physical_key(job)
            definition = job['B_law_binding']['definition']
            rte_source = dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',path='src/trotterlib/rte.py')
            decl = definition.get('A_declaration', {})
            if (definition.get('model') != 'ideal_finite_RTE_law' or definition.get('source') != rte_source
                    or set(definition) != {'model','source','A_declaration'}
                    or set(decl) != {'model','candidate_id','R_outer','K','q_PF','source'}
                    or decl.get('source') != rte_source or decl.get('K') != cfg['spec']['finite_K']
                    or decl.get('model') != 'ideal_finite_RTE_law'
                    or decl.get('q_PF') != row['q_PF'] or decl.get('R_outer') != row['R_outer']):
                rows.append(dict(m=m,decision='UNSUPPORTED',reasons=['ROUND_LAW_SOURCE_NOT_SUPPORTED']));continue
            if key in index:
                rows.append(dict(m=m, decision='REUSE', physical_key=key, job=job,
                                 saved=deepcopy(index[key]))); continue
            try:
                # Saved pair lookup only; never a new Full-H reference action.
                ref = self.runtime.provider.saved_h4_reference(self.root, job)
            except (ValueError, KeyError, IndexError) as exc:
                rows.append(dict(m=m, decision='UNSUPPORTED', reasons=['SAVED_REFERENCE_UNAVAILABLE: '+str(exc)])); continue
            rows.append(dict(m=m, decision='REQUEST', physical_key=key, job=job,
                request=dict(provider=IMPLEMENTATION, function='evaluate',
                    context_factory='load_h4_context(root, configuration)', reference=ref,
                    numerical_gate='PENDING_PROVIDER_EVALUATION', automatic_retry=False)))
        counts = {name:sum(r['decision']==name for r in rows) for name in ('REUSE','REQUEST','UNSUPPORTED')}
        return dict(schema='PR_DF_SIGNAL_ENTRY_PLAN_V1', configuration_id=cfg['id'],
                    rounds=rows, counts=counts, provider=IMPLEMENTATION,
                    supported_formulas=list(FORMULAS), supported_K=list(ORDERS),
                    scope='Saved replay: S2/K2. Other provider formulas/K: metadata/fixture coverage only; REQUEST gates remain pending.',
                    new_signal_calls=0, compile=0, Q3=0)

    def stage(self, configuration, output, *, obtained=None):
        """Copy/rebind only. Publish this directory before building the Q3 bundle.

        obtained maps requested m to a fixed raw-record locator. No provider call
        is implicit. A new ID over identical physics keeps every numerical field.
        """
        cfg = self.configuration(configuration); plan = self.plan(cfg)
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
        write(output/'CONFIGURATION.json', cfg); write(output/'PLAN.json', plan)
        obtained = {int(m):src for m,src in (obtained or {}).items()}
        wanted = {r['m'] for r in plan['rounds'] if r['decision']=='REQUEST'}
        if set(obtained)-wanted:
            raise ValueError('Unexpected obtained round; do not replace saved evidence')
        result = dict(status='CARRIERS_READY', staged=[], missing=[], invalid=[], new_signal_calls=0)
        for row in plan['rounds']:
            m = row['m']; decision = row['decision']
            src = row['saved']['source'] if decision=='REUSE' else obtained.get(m)
            if decision=='UNSUPPORTED' or src is None:
                result['missing'].append(m); continue
            rec = self.reader.read(src)
            original = deepcopy(rec)
            try:
                if decision=='REUSE':
                    if self.physical_key(rec) != row['physical_key']:
                        raise ValueError('Saved physical key differs')
                    job = row['job']
                    for key in ('m','B_configuration_id','B_candidate_id','B_identity','B_law_binding','physical_context'):
                        rec[key] = deepcopy(job[key])
                    rec['candidate'] = deepcopy(job['physical_context']['physical_candidate'])
                    if rec != original:
                        rec['reuse'] = dict(source=src, kind='SAME_PHYSICS_IDENTITY_REBOUND')
                self._check_record(rec,row['job'])
                write(output/'raw'/('m'+str(m)+'.json'), rec)
                result['staged'].append(dict(m=m, original_source=src, kind=decision))
            except (ValueError, KeyError) as exc:
                write(output/'raw'/('invalid_m'+str(m)+'.json'), original)
                result['invalid'].append(dict(m=m,reason=str(exc))); break
        if result['missing'] or result['invalid'] or len(result['staged']) != len(cfg['rounds']):
            result['status']='PARTIAL_OR_INVALID_NO_BUNDLE'
        write(output/'STAGING.json',result)
        return result

    def assemble(self, configuration_source, raw_sources):
        """Existing A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED layout, no new gates."""
        if configuration_source.get('selector'):
            raise ValueError('Existing DF intake needs a top-level CONFIGURATION.json carrier')
        cfg = self.reader.read(configuration_source); plan = self.plan(cfg)
        raw_sources = {int(m):src for m,src in raw_sources.items()}
        if set(raw_sources) != {r['m'] for r in cfg['rounds']}:
            raise ValueError('Complete raw rounds required; no partial bundle')
        commits = {s['commit'] for s in raw_sources.values()}
        if len(commits) != 1 or any(s.get('selector') for s in raw_sources.values()):
            raise ValueError('Existing DF intake requires one raw publication commit, top-level record files; use stage()')
        template = self.reader.read(SAVED[1])
        packet = {k:deepcopy(v) for k,v in template.items() if k!='means'}
        packet.update(configuration_source=configuration_source,configuration_id=cfg['id'],
                      raw_result_carrier=next(iter(commits)),means=[])
        packet['target']['binding']=deepcopy(cfg['spec']['binding'])
        for row, planned in zip(cfg['rounds'],plan['rounds']):
            if planned['decision']=='UNSUPPORTED':
                raise ValueError('Unsupported round cannot enter Q3 bundle')
            src=raw_sources[row['m']];rec=self.reader.read(src)
            self._check_record(rec,planned['job'])
            packet['means'].append(mean_from_record(row,rec,src,packet['target']))
        return packet


def mean_from_record(row, rec, source, target):
    """Unchanged field projection used by the LD7 bundle() / LD9 bundle route."""
    s=rec['signal']
    return dict(m=row['m'],identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
        source=deepcopy(source),implementation_source=deepcopy(IMPLEMENTATION),
        **{k:deepcopy(s[k]) for k in ('raw','corrected','reference','exact_tail')},
        normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_PRODUCER',
            **{k:s[k] for k in ('log_B','B','attenuation')},log_B_margin=rec['allowance']['log_B_margin'],
            source=deepcopy(source),normalizer_one_ulp_log_sensitivity=s['allowance_evidence']['normalizer_one_ulp_log_sensitivity']),
        bias={k:deepcopy(s[k]) for k in ('total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')},
        uncertainty=dict(empirical_allowance=deepcopy(rec['allowance']),numerical_observations=deepcopy(rec['numerical']),
            maximum_norm_domain=s['allowance_evidence']['maximum_norm_domain'],raw_rigorous_error_bound=None,
            ground_identification_error=None,actual_sampler_law_error=None,finite_K_RMSE_transfer_bound=None),
        target=deepcopy(target),qualification='UNQUALIFIED',numerical_gate_pass=True,
        reference_source=deepcopy(rec['reference_source']),state_evidence=deepcopy(rec['state_evidence']))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
