"""Source-bound PR calibration intake, scoped to exact actual steps.

This is a passive consumer of the existing Pauli S0-r1 and DF calibration /
holdout products. It never fits a model or invokes scientific acquisition.
The canonical PriceRules point/family/model functions remain authoritative.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

PRICE_SOURCE = 'c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa'
PRICE_MODULE = 'src/trottertracks/pr_price_rules.py'
SCHEMA = 'PR_ACTUAL_STEP_PRICE_REGISTRY_V1'
AXES = ('X', 'Y')
PRICE_SOURCE_CONFLICT = 'MULTIPLE_SCOPED_MODELS_CHOOSE_SOURCE_NO_POOLING'
PRICE_EXTRAPOLATION_HOLD = 'SCOPED_MODEL_EXTRAPOLATION_REQUIRES_EVIDENCE'
QUOTE_POLICY_REVISION = 4


def frac(value):
    return Fraction(int(value['numerator']), int(value['denominator'])) if isinstance(value, dict) else Fraction(value)


def exact(value):
    value = frac(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def canonical(value):
    if isinstance(value, dict):
        if set(value) == {'numerator', 'denominator'}:
            return ('number', frac(value))
        return tuple(sorted((k, canonical(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(canonical(v) for v in value)
    if type(value) in (int, float):
        return ('number', frac(value))
    return value


def same(a, b):
    return canonical(a) == canonical(b)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_ref(value):
    require(isinstance(value, dict) and value.get('commit') and value.get('path'), 'FIXED_EVIDENCE_LOCATOR_REQUIRED')
    return value


@lru_cache(maxsize=None)
def price_api(root):
    raw = subprocess.check_output(['git', 'show', PRICE_SOURCE+':'+PRICE_MODULE], cwd=root)
    scope = dict(__name__='_fixed_actual_step_price_rules', __file__=str(Path(root)/PRICE_MODULE))
    exec(compile(raw, PRICE_SOURCE+':'+PRICE_MODULE, 'exec'), scope)
    legacy_point=scope['point']
    def point(row):
        if (row.get('mapping') or {}).get('pf_grid_policy')!='PR_DF_REQUESTED_MAX_WIDTH_V1':
            return legacy_point(row)
        # The fixed API's consumers share the same global point function.
        # Only this explicit mapper may encode its physical step as binary64.
        import importlib,sys,types
        package='_rpe_step_price_max_width'
        if package not in sys.modules:
            module=types.ModuleType(package);module.__path__=[str(Path(__file__).resolve().parent)];sys.modules[package]=module
        maximum=importlib.import_module(package+'.pr_df_max_width')
        if not maximum.actual_step_matches(row):raise ValueError('Invalid exact max-width/B64 price row')
        return importlib.import_module(package+'.pr_price_rules').point(row)
    scope['point']=point
    return scope


@lru_cache(maxsize=None)
def df_validator(root):
    path = Path(root)/'scripts/run_pr_df_candidate_prices.py'
    spec = importlib.util.spec_from_file_location('_step_registry_df_receipt_validator', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.validate_axis


class StepPriceRegistry:
    def __init__(self, reader, registry=None, state=None, *, root=None, exploratory=False):
        self.reader = reader
        self.exploratory = exploratory
        self.root = str(root or getattr(reader, 'root', Path(__file__).resolve().parents[2]))
        self.registry = registry or dict(entries=[])
        self.api = price_api(self.root)
        self.state = deepcopy(state) if state is not None else dict(schema=SCHEMA, entries=[])
        require(self.state.get('schema') == SCHEMA, 'STEP_REGISTRY_SCHEMA_DIFFERS')

    def read(self, value):
        return deepcopy(self.reader.read(fixed_ref(value)))

    def resolve(self, value):
        return self.read(value) if isinstance(value, dict) and value.get('commit') and value.get('path') else deepcopy(value)

    def configuration(self, package):
        cfg = self.resolve(package.get('configuration_source') or package.get('configuration'))
        require(isinstance(cfg, dict) and cfg.get('id') and cfg.get('spec') and cfg.get('rounds'), 'CANONICAL_CONFIGURATION_REQUIRED')
        require(cfg['spec']['method'] == 'PR', 'PR_CONFIGURATION_REQUIRED')
        require(len(cfg['rounds']) == cfg['spec']['M']+1 and {r['m'] for r in cfg['rounds']} == set(range(cfg['spec']['M']+1)),
                'CANONICAL_ROUND_COVERAGE_DIFFERS')
        for row in cfg['rounds']:
            self.api['point'](row)
        return cfg

    def _domain(self, cfg, covered, implementation):
        require(covered and len(set(covered)) == len(covered) and all(type(m) is int and m in range(len(cfg['rounds'])) for m in covered),
                'COVERED_ROUNDS_REQUIRED')
        rows = [cfg['rounds'][m] for m in covered]
        step = frac(rows[0]['actual_delta_pf'])
        require(all(frac(r['actual_delta_pf']) == step for r in rows), 'ONE_ACTUAL_STEP_PER_PRICE_ENTRY_REQUIRED')
        points = [self.api['point'](r) for r in rows]
        first = points[0]
        require(all(same(p['piece_pattern'], first['piece_pattern']) and same(p['event_recipe'], first['event_recipe']) for p in points),
                'ONE_SIGNED_PATTERN_AND_LAW_PER_PRICE_ENTRY_REQUIRED')
        return dict(family=self.api['family'](cfg, implementation), actual_step=exact(step),
            piece_pattern=first['piece_pattern'], event_recipe=first['event_recipe'],
            producer_configuration_id=cfg['id'], covered_rounds=sorted(covered))

    def _save(self, entry):
        entry = deepcopy(entry)
        entry['id'] = digest({k:entry.get(k) for k in ('domain', 'mode', 'model', 'provenance', 'wrapper_times', 'calibration_domain', 'direct_prices')})
        old = next((e for e in self.state['entries'] if e['id'] == entry['id']), None)
        if old:
            require(old == entry, 'STEP_ENTRY_ID_BODY_DIFFERS')
            return dict(status='REUSED', entry_id=entry['id'], covered_rounds=entry['domain']['covered_rounds'])
        # A repeated cell/cohort cannot be silently reinterpreted with different coefficients.
        for old in self.state['entries']:
            if old['provenance'] == entry['provenance']:
                require(same(old['model'], entry['model']) and same(old['domain'], entry['domain']), 'CONFLICTING_REPEATED_CALIBRATION_RESULT')
        self.state['entries'].append(entry)
        return dict(status='REGISTERED', entry_id=entry['id'], covered_rounds=entry['domain']['covered_rounds'])

    def ingest(self, package):
        package = self.resolve(package)
        kind = package.get('kind')
        if kind == 'PR_PAULI_STEP_AUDIT':
            return self.ingest_pauli(package)
        if kind == 'PR_DF_STEP_MODEL':
            return self.ingest_df(package)
        raise ValueError('KNOWN_STEP_EVIDENCE_KIND_REQUIRED')

    def ingest_pauli(self, package):
        cfg = self.configuration(package); spec = cfg['spec']; prefix = spec['L_D']
        require(spec['split'].get('tail_dictionary') and spec['formula'] == '2nd' and spec['finite_K'] == 2,
                'PAULI_S2_K2_PRICE_CONTRACT_REQUIRED')
        imp = package['implementation']
        price = self.read(package['price_source']); audit = self.read(package['audit_source'])
        inputs = self.read(package['audit_inputs_source']); binding = self.read(package['binding_source'])
        raw = self.read(package['raw_source'])
        require(price.get('status') == 'PRICE_RULE_SUPPORTED_MODEL_REFERENCE' and price.get('configuration_id') == cfg['id'],
                'PAULI_PRICE_SOURCE_CONFIGURATION_OR_STATUS_DIFFERS')
        require(self.resolve(price['configuration_source']) == cfg and self.resolve(inputs['configuration']) == cfg,
                'PAULI_AUDIT_CONFIGURATION_SOURCE_DIFFERS')
        require(binding.get('status') == 'BINDING_FIXED_BEFORE_SAMPLING' and binding['configuration']['id'] == cfg['id']
                and same(binding['dictionary_identity'], spec['split']['tail_dictionary']), 'PAULI_BINDING_OR_DICTIONARY_DIFFERS')
        require(raw.get('status') == 'FINITE_LD%d_CAL_PATHS_COMPLETE' % prefix and raw.get('compiles') == 4,
                'PAULI_RAW_CALIBRATION_INCOMPLETE')
        require(self.read(raw['binding']) == binding and inputs['raw'] == package['raw_source']['commit'],
                'PAULI_RAW_BINDING_SOURCE_DIFFERS')
        require(not audit.get('stops') and audit.get('rule_supported') is True and len(audit.get('paths_ok', [])) == 2,
                'PAULI_AUDIT_NOT_SUPPORTED')
        require(audit.get('checks') and all(v is True for v in audit['checks'].values()), 'PAULI_P1_P2_CHECK_FAILED')
        require(all(audit.get(k, {}).get('status') == 'PASS' for k in ('P3', 'P5', 'P6')), 'PAULI_P3_P5_P6_FAILED')
        require(audit['P5'].get('margins_all_ok') is True and audit['P6'].get('decomposition')
                and all(v.get('exact') is True for v in audit['P6']['decomposition'].values()), 'PAULI_WRAPPER_OR_DECOMPOSITION_FAILED')
        req = binding['request']; cell = req['physical_cell']
        for key in ('binding', 'split', 'profile'):
            require(same(cell[key], spec[key]), 'PAULI_CELL_'+key.upper()+'_DIFFERS')
        require(cell.get('lowering') == imp['lowering_id'] and cell.get('route') == 'PR_PAULI_TAIL'
                and imp.get('sampler') == 'ORIGINAL_NORMAL_PCG64_NUMPY_CHOICE'
                and imp.get('scalar') == 'HAMILTONIAN_CONSTANT_PLUS_SAVED_sR_ONCE_AT_TOTAL_T'
                and imp.get('fragment_order') == 'SAVED_SNAPSHOT_ORDER_ONE_BODY_SEPARATE_HEAD', 'PAULI_IMPLEMENTATION_DIFFERS')
        require(req.get('prefix')==prefix and frac(req['lambda_P'])==frac(spec['split']['lambda_R']),
                'PAULI_DICTIONARY_LAMBDA_OR_PREFIX_DIFFERS')
        require(binding['scalar']['application'] == 'ONCE_AT_TOTAL_T' and inputs['scalar']['value_hex'] == float(binding['scalar']['value']).hex(),
                'PAULI_SCALAR_APPLICATION_DIFFERS')
        step = frac(req['step_time']); require(step > 0 and Fraction(float(step)) == step, 'PAULI_EXACT_STEP_NOT_BINARY64_REPRESENTABLE')
        group = self.resolve(package.get('preparation_group'))
        if group:
            ds = group['driver_request']['driver_spec']; expected = group['driver_metadata']['request']
            require(expected == req and ds['configuration'] == cfg and ds['config_id'] == cfg['id'], 'PAULI_REQUEST_BINDING_DIFFERS')
            require(same(ds['implementation'], imp), 'PAULI_REQUEST_IMPLEMENTATION_DIFFERS')
            covered = group['covered_rounds']
        else:
            covered = [r['m'] for r in cfg['rounds'] if frac(r['actual_delta_pf']) == step]
        domain = self._domain(cfg, covered, imp)
        require(frac(domain['actual_step']) == step, 'PAULI_CELL_ACTUAL_STEP_DIFFERS')
        require(req['q'] == cell['q'] and req['R'] == cell['R'] and frac(req['T']) == step*req['q']
                and req['K'] == cell['K'] == 2, 'PAULI_CELL_LAYOUT_DIFFERS')
        # Legacy audits verified every wrapper time. New actual-step audits
        # intentionally inspect only their covered times; retain that scope.
        require(all(str(m) in inputs['round_T'] and same(inputs['round_T'][str(m)],cfg['rounds'][m]['T']) for m in covered),
                'PAULI_WRAPPER_TIME_COVERAGE_DIFFERS')
        require('covered_rounds'not in inputs or inputs['covered_rounds']==sorted(covered), 'PAULI_AUDIT_ROUND_SCOPE_DIFFERS')
        paths = inputs['paths']; rows = price['calibration_cell']['rows']
        require(price['calibration_cell'].get('decomposition_supported') is True and len(paths) == 2 and len(rows) == 4,
                'PAULI_CALIBRATION_RECEIPTS_INCOMPLETE')
        require({(r['path_id'], r['axis']) for r in rows} == {(p['path_id'], a) for p in paths for a in AXES},
                'PAULI_CALIBRATION_PATH_AXIS_COVERAGE_DIFFERS')
        for row in rows:
            path = next(p for p in paths if p['path_id'] == row['path_id'])
            require(row['physical_id'] == req['candidate_id'] and row['q'] == req['q'] and row['R'] == req['R'],
                    'PAULI_CALIBRATION_CELL_ID_DIFFERS')
            require(row['receipt'] == path['axes'][row['axis']]['receipt'] and row['receipt']['commit'] == inputs['raw'],
                    'PAULI_CALIBRATION_RECEIPT_SOURCE_DIFFERS')
            receipt = self.read(row['receipt'])
            require(receipt['request'] == req and receipt['cohort'] == binding['cohort']
                    and receipt['record']['metrics']['rz_count'] == row['compiled_RZ'], 'PAULI_COMPILED_RECEIPT_DIFFERS')
        # Recheck the already-counted decomposition algebra; this is not a new fit.
        constants = audit['edges']['constants']; final = audit['P5']['final_run_rz']
        require(len(final) == 1 and final[0] == 2, 'PAULI_FINAL_RUN_CONTRACT_DIFFERS')
        H = constants['later_prefix_rz']+constants['suffix_rz']
        A = constants['first_prefix_rz']+constants['suffix_rz']+constants['end_rz']+final[0]-H
        later = [r['body_rz'] for values in audit['segments'].values() for r in values if r['context'] == 'LATER']
        require(len(later) == 2*(req['q']-1) and later, 'PAULI_EVENT_INTERIOR_COVERAGE_DIFFERS')
        E = Fraction(sum(later), len(later)*req['R']); co = price['coefficients']
        require(frac(co['A%d'%prefix]) == A and frac(co['H%d'%prefix]) == H and frac(co['E%d'%prefix]) == E,
                'PAULI_AUDIT_COEFFICIENTS_DIFFER')
        model = dict(coefficients={'A%d'%prefix:exact(A),'H%d'%prefix:exact(H),'E%d'%prefix:exact(E)})
        for row in price.get('rows') or []:
            if row['m'] not in covered:
                continue  # Old audit writers emitted all26; never widen their actual-step scope.
            target = cfg['rounds'][row['m']]
            require(frac(row.get('C_ref', row.get('C_ref_exact'))) == A+H*target['q_PF']+E*target['q_PF']*target['R_outer'],
                    'PAULI_SCOPED_MODEL_ROW_DIFFERS')
        return self._save(dict(domain=domain, mode='PAULI_AUDIT', model=model,
            wrapper_times=[cfg['rounds'][m]['T'] for m in sorted(covered)], calibration_domain=dict(q=[req['q'],req['q']],R=[req['R'],req['R']]),
            calibration_R=[req['R']], provenance={k:deepcopy(package[k]) for k in ('price_source','audit_source','audit_inputs_source','binding_source','raw_source')},
            request_id=req['candidate_id'], qualification='FINITE_COHORT_MODEL_REFERENCE_NOT_EXPECTATION_OR_CI'))

    def ingest_df(self, package):
        cfg = self.configuration(package); spec = cfg['spec']; imp = package['implementation']
        require(not spec['split'].get('tail_dictionary') and spec['formula'] == '2nd' and spec['finite_K'] == 2,
                'DF_S2_K2_PRICE_CONTRACT_REQUIRED')
        inputs = self.read(package['inputs_source']); model = self.read(package['model_source'])
        prices = self.read(package['price_source'])
        mode = model['mode']; direct = mode == 'DIRECT_FINITE_COHORT'
        hold = self.read(package['holdout_source']) if package.get('holdout_source') else None
        require(inputs['configuration_id'] == cfg['id'] and self.resolve(inputs['configuration']) == cfg,
                'DF_INPUT_CONFIGURATION_DIFFERS')
        require(model.get('status') == 'FROZEN_2_AXIS_MODELS' and model.get('holdout_not_opened') is True
                and model.get('formula') == 'FULL_WRAPPER_C_OFFSET_ZERO_NO_S_ADDITION', 'DF_FROZEN_MODEL_CONTRACT_DIFFERS')
        if direct:
            require(hold is None and all(m.get('direct') is True and m.get('coefficients') == {} for m in model['models'].values()),
                    'DF_DIRECT_SCOPE_CONTRACT_DIFFERS')
        elif hold is not None or not self.exploratory:
            require(hold is not None and hold.get('status') in ('DESCRIPTIVE_COMPLETE','DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED')
                    and hold.get('refit') is False, 'DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS')
            frozen = hold.get('model_commit')
            if frozen != package['model_source']['commit']:
                # A resumed publish can return the current HEAD when all
                # outputs were already committed. Recover the original
                # freeze from the independent comparison receipt.
                package = dict(package)
                require(isinstance(frozen, str) and len(frozen) == 40
                        and all(c in '0123456789abcdef' for c in frozen),
                        'DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS')
                blob = getattr(self.reader, 'blob', None)
                if blob is None:
                    def blob(commit, path):
                        return subprocess.check_output(['git','show',commit+':'+path],cwd=self.root)
                committed_paths = None
                for role in ('model_source','price_source'):
                    later = fixed_ref(package[role])
                    original = fixed_ref(dict(later, commit=frozen))
                    try:
                        original_bytes = blob(original['commit'], original['path'])
                    except (KeyError, subprocess.CalledProcessError):
                        if committed_paths is None:
                            committed_paths = subprocess.check_output(
                                ['git','diff-tree','--root','--no-commit-id','--name-only','-r',frozen],
                                cwd=self.root,text=True).splitlines()
                        suffix = Path(later['path']).parts[2:]
                        require(len(suffix) >= 4 and suffix[0] == 'completion',
                                'DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS')
                        matches = [name for name in committed_paths if Path(name).parts[2:] == suffix]
                        require(len(matches) == 1, 'DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS')
                        original['path'] = matches[0]
                        original_bytes = blob(original['commit'], original['path'])
                    require(original_bytes == blob(later['commit'], later['path']),
                            'DF_INDEPENDENT_HOLDOUT_SOURCE_DIFFERS')
                    package[role] = original
        require(mode in ('FULL_AFFINE_Q_R1','FULL_AFFINE_QR','DIRECT_FINITE_COHORT'), 'DF_MODEL_MODE_NOT_SUPPORTED')
        require(set(model['models']) == set(AXES), 'DF_BOTH_AXIS_MODELS_REQUIRED')
        calibration = [r for a in AXES for r in model['models'][a]['calibration']]
        require(calibration and {r['axis'] for r in calibration} == set(AXES), 'DF_CALIBRATION_AXIS_COVERAGE_DIFFERS')
        ms = {r['m'] for r in calibration}; steps = {frac(cfg['rounds'][m]['actual_delta_pf']) for m in ms}
        require(len(steps) == 1, 'DF_MIXED_STEP_MODEL_REJECTED')
        step = next(iter(steps)); covered = inputs.get('covered_rounds') or [r['m'] for r in cfg['rounds'] if frac(r['actual_delta_pf']) == step]
        domain = self._domain(cfg, covered, imp)
        require(frac(domain['actual_step']) == step, 'DF_MODEL_ACTUAL_STEP_DIFFERS')
        require(hold is None or all(frac(cfg['rounds'][r['m']]['actual_delta_pf']) == step for r in hold['rows']), 'DF_HOLDOUT_STEP_DIFFERS')
        require(not direct or len(ms) == 1, 'DF_DIRECT_SINGLE_PHYSICAL_POINT_REQUIRED')
        validate = df_validator(self.root)
        source_reader = (lambda c,p:self.reader.blob(c,p)) if hasattr(self.reader,'blob') else lambda c,p:subprocess.check_output(['git','show',c+':'+p],cwd=self.root)
        used_seeds = set(); cal_cohorts = set()
        for role, records in [('CALIBRATION',calibration)]+([] if hold is None else [('HOLDOUT',hold['rows'])]):
            require({r['axis'] for r in records} == set(AXES), 'DF_HOLDOUT_AXIS_COVERAGE_DIFFERS')
            for rec in records:
                cell = next(c for c in inputs['cells'] if c['m']==rec['m'] and c['role']==role)
                context = cell['request']['physical_context']
                target = cfg['rounds'][rec['m']]
                expected_profile = dict(spec['profile'],lowering_id=imp['lowering_id'])
                require(context['implementation'].get('adapter') == 'PR_CANDIDATE_PRICE_CONTEXT_V1'
                        and imp.get('scalar') == 'HAMILTONIAN_CONSTANT_PLUS_SAVED_sR_ONCE_AT_TOTAL_T'
                        and imp.get('fragment_order') == 'SAVED_SNAPSHOT_ORDER_ONE_BODY_SEPARATE_HEAD'
                        and imp.get('sampler') == 'ORIGINAL_NORMAL_PCG64_NUMPY_CHOICE'
                        and same(context['binding'],spec['binding'])
                        and same(context['split'],spec['split']) and same(context['law'],target['law_binding'])
                        and same(context['price_profile'],expected_profile), 'DF_RECEIPT_IMPLEMENTATION_FAMILY_DIFFERS')
                candidate = context['physical_candidate']
                require(candidate == target['physical_candidate'] and rec['q']==target['q_PF'] and rec['R']==target['R_outer'],
                        'DF_CALIBRATION_PHYSICAL_POINT_DIFFERS')
                require(len(rec['paths']) == len(cell['indices']) == rec['n_paths'], 'DF_PATH_COUNT_DIFFERS')
                checked=[]
                for path in rec['paths']:
                    checked.append(validate(cell,path['index'],rec['axis'],path,reader=self.reader.read,source_reader=source_reader))
                require(frac(rec['mean_C']) == sum(Fraction(p['C']) for p in checked)/len(checked), 'DF_MEAN_RECEIPTS_DIFFER')
                if role=='CALIBRATION':cal_cohorts.add(rec['cohort'])
                else:require(rec['cohort'] not in cal_cohorts, 'DF_HOLDOUT_COHORT_NOT_INDEPENDENT')
                for path in checked:
                    for seed in path['seeds']:
                        # X/Y from one path share seeds; only cross-role overlap is prohibited.
                        ident = (role,rec['m'],path['index'],seed)
                        require(not any(other[-1]==seed and other[0]!=role for other in used_seeds), 'DF_HOLDOUT_SEED_OVERLAP')
                        used_seeds.add(ident)
                if direct:
                    continue
                price_entry=dict(mode=mode,model=model,family=domain['family'])
                predicted,missing=self.api['PriceRules']._model(None,price_entry,self.api['point'](target),rec['axis'])
                require(not missing and predicted is not None, 'DF_FROZEN_MODEL_NOT_APPLICABLE')
                if role=='CALIBRATION':require(predicted==frac(rec['mean_C']), 'DF_FROZEN_CALIBRATION_NOT_REPRODUCED')
                else:require(predicted==frac(rec['prediction']) and frac(rec['observed'])==frac(rec['mean_C'])
                             and frac(rec['signed_error'])==predicted-frac(rec['observed']), 'DF_HOLDOUT_COMPARISON_DIFFERS')
        require(prices.get('rows'), 'DF_SOURCED_AXIS_PRICES_REQUIRED')
        for row in prices['rows']:
            if row['m'] not in covered:continue
            target=cfg['rounds'][row['m']]
            if direct:
                predicted=frac(next(c['mean_C'] for c in calibration if c['m']==row['m'] and c['axis']==row['axis']))
            else:
                predicted,_=self.api['PriceRules']._model(None,dict(mode=mode,model=model,family=domain['family']),self.api['point'](target),row['axis'])
            require(row['configuration_id']==cfg['id'] and frac(row['C_ref'])==predicted, 'DF_SCOPED_MODEL_ROW_DIFFERS')
        q=[r['q']for r in calibration];R=[r['R']for r in calibration]
        return self._save(dict(domain=domain,mode=mode,model=deepcopy(model),wrapper_times=[cfg['rounds'][m]['T']for m in covered],
            calibration_domain=dict(q=[min(q),max(q)],R=[min(R),max(R)]),calibration_R=sorted(set(R)),
            provenance={k:deepcopy(package.get(k))for k in ('inputs_source','model_source','price_source','holdout_source')},
            direct_prices=([dict(axis=c['axis'],point=self.api['physical'](self.api['point'](cfg['rounds'][c['m']])),value=c['mean_C'],paths=c['paths']) for c in calibration] if direct else []),
            request_id=digest(dict(family=domain['family'],step=domain['actual_step'],calibration=calibration)),
            qualification=('EXPLORATORY_CALIBRATION_ONLY' if hold is None and not direct
                           else 'DESCRIPTIVE_HOLDOUT_FINITE_COHORT_NOT_EXPECTATION_OR_CI')))

    def common_pauli_model_terms(self, configuration, implementation, rules=None):
        """Return source-bound MODEL terms only for one-piece Pauli families.

        A step-specific rule or a missing base axis leaves the proof unavailable.
        This is a bound on the search's model cost under the current registry.
        """
        if configuration['spec']['M']!=12 or configuration['spec']['formula'] not in ('2nd','S2'):
            return None
        family=self.api['family'](configuration,implementation)
        if family['method']!='PR_PAULI' or any(same(e['domain']['family'],family) for e in self.state['entries']):
            return None
        rules=self.api['PriceRules'](self.reader,self.registry) if rules is None else rules
        match=[e for e in rules.entries if same(e['family'],family)]
        if len(match)!=1 or match[0]['mode'] not in ('PAULI_AUDIT','PAULI_C2'):
            return None
        entry=match[0]
        # An identical-physical saved value can override the common formula.
        for (m,axis),saved in entry['saved'].items():
            model,_=rules._model(entry,entry['points'][m],axis)
            if model is None or frac(saved['value'])!=model:
                return None
        quote=rules.quote(dict(configuration=configuration,implementation=implementation,policy='MODEL'))
        if not quote['complete']:
            return None
        terms=[]
        rounds={r['m']:r for r in configuration['rounds']}
        for row in quote['rows']:
            p=self.api['point'](rounds[row['m']])
            if (p['structure']['signed_weights']!=[1.] or len(p['structure']['allocation'])!=1
                    or row.get('classification')!='MODEL_REFERENCE'):
                return None
            one,_=rules._model(entry,dict(p,R=1),row['axis'])
            two,_=rules._model(entry,dict(p,R=2),row['axis'])
            if one is None or two is None or one<=0 or two<one:
                return None
            slope=two-one
            if frac(row['unit_price_RZ'])!=one+slope*(p['R']-1):
                return None
            terms.append((one,slope))
        return tuple(terms) if len(terms)==26 else None

    def _scoped_price(self, row, point, matching, family, preferred=None, *, extrapolated_time=False):
        axis = row['axis']
        direct = []
        models = []
        rejected = []
        for entry in matching:
            if entry['mode'] == 'DIRECT_FINITE_COHORT':
                direct.extend((entry, observed) for observed in entry['direct_prices']
                              if observed['axis'] == axis and same(observed['point'], self.api['physical'](point)))
                continue
            value, missing = self.api['PriceRules']._model(None, dict(entry, family=family), point, axis)
            if missing or value is None or value <= 0:
                rejected.extend(missing or ['SCOPED_MODEL_NONPOSITIVE_NO_CLAMP'])
                continue
            extrapolation = [k for k in ('q', 'R') if not
                             entry['calibration_domain'][k][0] <= point[k] <= entry['calibration_domain'][k][1]]
            models.append((entry, value, extrapolation))

        if direct:
            if len(direct) > 1:
                row['needs'] = dict(reasons=[PRICE_SOURCE_CONFLICT], source_kind='DIRECT_RECEIPTS',
                                    entry_ids=[entry['id'] for entry, _ in direct])
                return
            entry, observed = direct[0]
            row.update(unit_price_RZ=observed['value'], classification='DIRECT_FINITE_COHORT',
                       source=deepcopy(entry['provenance']), registered_rule='ACTUAL_STEP:'+entry['id'],
                       qualified_E_C=False, CI=None, direct_alternatives=[deepcopy(observed)])
        elif models:
            if extrapolated_time and len(models) != 1:
                row['needs'] = dict(reasons=[PRICE_SOURCE_CONFLICT], entry_ids=[e['id'] for e, _, _ in models])
                return
            # A previously ambiguous scope may only be resolved with
            # in-range evidence. Keep the existing single-entry policy.
            competing = sum(entry['mode'] != 'DIRECT_FINITE_COHORT' for entry in matching) > 1
            choices = [model for model in models if not model[2]] if competing else models
            if not choices:
                row['needs'] = dict(reasons=[PRICE_EXTRAPOLATION_HOLD],
                    entry_ids=[entry['id'] for entry, _, _ in models],
                    extrapolation={entry['id']: axes for entry, _, axes in models})
                return
            if len(choices) > 1:
                agreement = [item for item in choices
                             if preferred and preferred['registered_rule'] == 'ACTUAL_STEP:'+item[0]['id']
                             and same(preferred['source'], item[0]['provenance'])
                             and frac(preferred['unit_price_RZ']) == item[1]]
                if len(agreement) != 1:
                    row['needs'] = dict(reasons=[PRICE_SOURCE_CONFLICT], entry_ids=[entry['id'] for entry, _, _ in choices])
                    return
                choices = agreement
                row['parent_price_quote'] = deepcopy(preferred['quote'])
            entry, value, extrapolation = choices[0]
            row.update(unit_price_RZ=exact(value), classification='MODEL_REFERENCE', source=deepcopy(entry['provenance']),
                       registered_rule='ACTUAL_STEP:'+entry['id'], qualified_E_C=False, CI=None,
                       model_alternative=dict(value=exact(value), rule=entry['mode'], source=deepcopy(entry['provenance']),
                           actual_step=entry['domain']['actual_step'], covered_rounds=entry['domain']['covered_rounds'],
                           calibration_domain=entry['calibration_domain'], extrapolation=extrapolation,
                           event_step_changed_from_calibration=point['R'] not in entry['calibration_R'],
                           qualification=entry['qualification'],
                           application=('EXPLORATORY_UNVALIDATED_TERMINAL_EXTRAPOLATION' if extrapolated_time
                                        else 'SOURCE_BOUND_ACTUAL_STEP_MODEL')))
        else:
            if rejected:
                row['needs'] = dict(reasons=list(dict.fromkeys(rejected)))
            return
        row.pop('needs', None)

    def _df_calibration_alternatives(self, round_row, axis, matching):
        """Offer original measured paths to the existing cross-cohort receipt gate."""
        alternatives = []
        for entry in matching:
            if entry['mode'] == 'DIRECT_FINITE_COHORT':
                continue
            model = entry['model']['models'].get(axis, {})
            measured = [c for c in model.get('calibration', [])
                        if c.get('m') == round_row['m'] and c.get('physical_id') == round_row['physical_candidate']['id']
                        and c.get('paths') and c.get('cohort')]
            for cell in measured:
                alternatives.append(dict(cohort=cell['cohort'], role='CALIBRATION',
                                         source=entry['provenance']['price_source'],
                                         path_witnesses=deepcopy(cell['paths'])))
        return alternatives

    def quote(self, configuration, implementation, base_quote=None, *, preferred_models=None):
        config=configuration;family=self.api['family'](config,implementation)
        if base_quote is None:
            base_quote=self.api['PriceRules'](self.reader,self.registry).quote(dict(configuration=config,implementation=implementation,policy='MODEL'))
        require(base_quote.get('configuration_id')==config['id'] and base_quote.get('policy')=='MODEL', 'BASE_QUOTE_CONFIGURATION_POLICY_DIFFERS')
        rows={(r['m'],r['axis']):deepcopy(r)for r in base_quote['rows']}
        require(set(rows)=={(m,a)for m in range(config['spec']['M']+1)for a in AXES}, 'BASE_QUOTE_AXIS_COVERAGE_DIFFERS')
        for round_row in config['rounds']:
            p=self.api['point'](round_row)
            matching=[e for e in self.state['entries']if same(e['domain']['family'],family)
                and frac(e['domain']['actual_step'])==frac(p['step'])
                and same(e['domain']['piece_pattern'],p['piece_pattern'])
                and same(e['domain']['event_recipe'],p['event_recipe'])
                and (self.exploratory or e.get('qualification') != 'EXPLORATORY_CALIBRATION_ONLY')]
            source_bound = [e for e in matching if any(same(t, p['T']) for t in e['wrapper_times'])]
            estimate_terminal = (self.exploratory and family['method'] == 'PR_DF'
                                 and not source_bound)
            matching = ([e for e in matching if e['mode'] != 'DIRECT_FINITE_COHORT'
                         and p['R'] in e['calibration_R']
                         and all(frac(t) < frac(p['T']) for t in e['wrapper_times'])]
                        if estimate_terminal else source_bound)
            for axis in AXES:
                row=rows[round_row['m'],axis]
                require(row.get('physical_id')==round_row['candidate']['id'] and row.get('q')==p['q'] and row.get('R')==p['R'],
                        'BASE_QUOTE_PHYSICAL_AXIS_DIFFERS')
                if not matching:continue
                if row.get('unit_price_RZ') is not None:continue
                if family['method'] != 'PR_DF' and len(matching) > 1:
                    row['needs'] = dict(reasons=[PRICE_SOURCE_CONFLICT], entry_ids=[e['id'] for e in matching])
                    continue
                preferred = (preferred_models or {}).get((round_row['m'], axis))
                self._scoped_price(row, p, matching, family, preferred, extrapolated_time=estimate_terminal)
                if family['method'] == 'PR_DF' and row.get('unit_price_RZ') is None:
                    measured = self._df_calibration_alternatives(round_row, axis, matching)
                    if measured:
                        row['direct_alternatives'] = [*row.get('direct_alternatives', []), *measured]
        values=sorted(rows.values(),key=lambda r:(r['m'],r['axis']))
        from .rpe_price_validation import describe,describe_steps
        fixed=self.api['PriceRules'](self.reader,self.registry)
        legacy=[]
        for name in {r.get('registered_rule')for r in values if r.get('classification')=='MODEL_REFERENCE'}:
            if name and not name.startswith('ACTUAL_STEP:'):
                legacy+=describe(fixed,config,name,dict(base_quote,rows=values))
        return dict(base_quote,rows=values,model_validations=legacy+describe_steps(self,config,dict(base_quote,rows=values)),complete=all(r.get('unit_price_RZ')is not None for r in values),
            qualified_E_C=False,CI=None,Q3_executed=False,step_registry_schema=SCHEMA)
