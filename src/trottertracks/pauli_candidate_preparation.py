"""B2 request -> saved/on-demand Pauli split -> B canonical schedule and passive handoffs.

No search, mean, normalizer, price quote or Q3 is performed here.  The optional
split materializer is a separate scientific leaf; the default only describes it.
Canonical generation is B2's published generator, not a second mapper.
"""
from copy import deepcopy
from contextlib import nullcontext
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess

from . import pauli_signal_entry as E

MODULE = 'src/trottertracks/pauli_candidate_preparation.py'
B2 = '98b6f8e1ac6e68285d2c68a8e35125d02d6edd9c'
GENERATOR = 'scripts/run_rpe_optimizer_config_generator.py'
D2 = 'c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa'
REGISTRY = 'configs/pr_price_rules/h4_saved_rules.json'
VARIANTS = '31d1bcc0bf9f43fa0afdef9d9813d9f57b049f81'
VARIANT_DIR = 'docs/research/b2_pauli_kappa_signals_20261116/configurations/'
FORMULAS = ('2nd', '4th(m5_best)')
LEGACY_BUILDER = 'P0_LEGACY_OPENFERMION_DEFAULT_V1'
NO_DROP_BUILDER = 'P0_NO_INTERMEDIATE_DROP_V1'
DICTIONARY_REGISTRY = 'configs/pauli_candidate_preparation/h4_no_drop_splits.json'


def dictionary_builder(item):
    """Scientific construction policy; a legacy payload has no builder field."""
    value = item['payload'].get('builder')
    if value is None:
        return LEGACY_BUILDER
    if (not isinstance(value, dict) or value.get('id') != NO_DROP_BUILDER
            or not isinstance(value.get('source'), dict)
            or not value['source'].get('commit') or not value['source'].get('path')):
        raise ValueError('UNKNOWN_OR_UNSOURCED_DICTIONARY_BUILDER')
    return value['id']


def rational(value):
    if isinstance(value, dict):
        return Fraction(value['numerator'], value['denominator'])
    return Fraction(value)  # float -> its binary64 value, never str(float)


def encoded(value):
    value = rational(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


@lru_cache(maxsize=None)
def generator(root):
    """Load the fixed B2 metadata-only boundary; B's identity/mapper stays authoritative."""
    root = Path(root)
    source = E.p0.blob(root, B2, GENERATOR)
    scope = dict(__name__='fixed_b2_config_generator', __file__=str(root/GENERATOR))
    exec(compile(source, B2+':'+GENERATOR, 'exec'), scope)
    b, build = scope['generator']('pauli_native', root)
    build.structure_checks = scope['structure_checks']
    return b, build


class Assets:
    """Small request-session view over immutable artifacts and optional split cache.

    Cache entries locate scientific products; they do not certify approval.
    Dictionary cache keys are binding + prefix + dictionary rules, never kappa/Delta.
    """
    def __init__(self, root, *, cache=None, packets=(), configurations=(), builder_id=None):
        self.root = Path(root)
        if builder_id not in (None, LEGACY_BUILDER, NO_DROP_BUILDER):
            raise ValueError('UNKNOWN_DICTIONARY_BUILDER_REQUEST')
        self.builder_id = builder_id
        self.cache = Path(cache) if cache else None
        self.meta = E.p0.metadata(self.root)
        self.binding = self.meta['binding']
        self.context = json.loads(Path(self.meta['context_source']['path']).read_bytes())
        receipt = json.loads(Path(self.meta['inputs']['H4_tol_receipt.json']['path']).read_bytes())
        from .pauli_split_descriptor import fragment_count
        self.fragment_count = fragment_count(receipt)
        self.packets = tuple(E.SAVED_PACKETS) + tuple(p for p in packets if p not in E.SAVED_PACKETS)
        self.configurations = []
        self.local_configurations = []
        self.baselines = {}
        self.extra_splits = {}
        self.split_versions = {}
        self._legacy_splits = {}
        for ref in E.SAVED_PACKETS:
            obj = E.read_ref(self.root, ref)
            for i, packet in enumerate(obj['packets']):
                loc = dict(ref, selector=['packets', i, 'configuration'])
                config = packet['configuration']
                self.baselines[config['spec']['L_D']] = (config, loc)
                self.configurations.append((config, loc))
        for p in (2, 3, 5):
            for suffix in ('x2', 'x2o3'):
                loc = dict(commit=VARIANTS, path=VARIANT_DIR+'LD%d_S2_delta_h_kappa_%s.json' % (p, suffix))
                self.configurations.append((E.read_ref(self.root, loc), loc))
        for loc in configurations:
            config, source = E.load_configuration(self.root, loc)
            if config['spec']['binding'] != self.binding:
                raise ValueError('SUPPLIED_CONFIGURATION_H_STATE_SNAPSHOT_DIFFERS')
            self.configurations.append((config, source))
            p = config['spec']['L_D']
            if p not in self.baselines:
                item = E.dictionary_item(self.root, config)
                res = self.read(item['P0_result_source'])
                ix = next(i for i, r in enumerate(res['prefixes']) if r['prefix'] == p)
                product = dict(status='SPLIT_READY', item=item,
                    descriptor=dict(status='SAVED_P0_DESCRIPTOR', descriptor=res['prefixes'][ix]['saved_DF_descriptor'],
                        source=dict(item['P0_result_source'], selector=['prefixes', ix, 'saved_DF_descriptor'])))
                self._store_split(p, product, verify=False)
        self.price_registry_needs = []
        try:
            self.price_registry = E.L.read(self.root, D2, REGISTRY)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            self.price_registry = dict(entries=[])
            self.price_registry_needs = ['RECOVER_D2_REGISTRY_SOURCE: '+str(exc)]
        self._index = None
        if self.cache and self.cache.exists():
            saved = json.loads(self.cache.read_text())
            if saved['binding'] != self.binding:
                raise ValueError('CACHE_H_STATE_SNAPSHOT_BINDING_DIFFERS')
            for key, value in saved.get('splits', {}).items():
                if dictionary_builder(value['item']) != LEGACY_BUILDER:
                    raise ValueError('VERSIONED_BUILDER_IN_LEGACY_CACHE_SLOT')
                self._store_split(int(key), value, verify=False)
            for key, versions in saved.get('split_versions', {}).items():
                for builder, value in versions.items():
                    if dictionary_builder(value['item']) != builder:
                        raise ValueError('CACHE_DICTIONARY_BUILDER_TAG_DIFFERS')
                    self._store_split(int(key), value, verify=True)
            for value in saved.get('configurations', []):
                config = value['configuration']
                generator(str(self.root))[0].validate(config)
                if config['spec']['binding'] != self.binding:
                    raise ValueError('CACHED_CONFIGURATION_BINDING_DIFFERS')
                if json.loads(Path(value['file']).read_text()) != config:
                    raise ValueError('CACHED_LOCAL_CONFIGURATION_FILE_DIFFERS')
                self.local_configurations.append(value)
                self.configurations.append((config, None))
        registry = self.root/DICTIONARY_REGISTRY
        if registry.exists():
            for row in json.loads(registry.read_text()).get('entries', []):
                p = row['prefix']; result = self.read(row['p0_result_source'])
                record = next(v for v in result['prefixes'] if v['prefix'] == p)
                payload = self.read(row['dictionary_source'])
                product = dict(status='SPLIT_READY', item=dict(payload=payload,
                    dictionary_sha256=record['dictionary_sha256'], source=deepcopy(row['dictionary_source']),
                    P0_metrics=deepcopy(record['identity_metrics']), P0_result_source=deepcopy(row['p0_result_source'])),
                    descriptor=dict(status='REGISTERED_SAVED_DESCRIPTOR', descriptor=self.read(row['descriptor_source']),
                                    source=deepcopy(row['descriptor_source'])))
                if dictionary_builder(product['item']) != row['builder_id']:
                    raise ValueError('REGISTRY_DICTIONARY_BUILDER_TAG_DIFFERS')
                self._store_split(p, product, verify=True)
                if row.get('configuration_source'):
                    config, source = E.load_configuration(self.root, row['configuration_source'])
                    if (config['spec']['binding'] != self.binding or config['spec']['L_D'] != p
                            or config['spec']['split']['tail_dictionary'] != E.L.dictionary_identity(product['item'])):
                        raise ValueError('REGISTERED_CONFIGURATION_DICTIONARY_DIFFERS')
                    self.configurations.append((config, source))

    def read(self, ref):
        return E.read_ref(self.root, ref)

    def _validate_product(self, prefix, value):
        item = value['item']; payload = item['payload']; result = self.read(item['P0_result_source'])
        row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
        if (payload['binding'] != self.binding or payload['prefix'] != prefix
                or self.read(item['source']) != payload
                or E.L.canonical_sha(payload) != item['dictionary_sha256']
                or row['dictionary_sha256'] != item['dictionary_sha256']
                or result['binding'] != self.binding
                or not result['status'].endswith('_COMPLETE_NOT_P1_OR_RPE_QUALIFIED')
                or not result['checks'] or not all(v['PASS'] for v in result['checks'])
                or result.get('control', dict(PASS=True)).get('PASS') is not True
                or 'GATES_PASS' not in row['status']
                or row['identity_metrics'] != item['P0_metrics']):
            raise ValueError('VERSIONED_P0_DICTIONARY_SOURCE_OR_GATES_DIFFER')
        descriptor = value['descriptor']
        if (self.read(descriptor['source']) != descriptor['descriptor']
                or descriptor['descriptor']['scalar'] != payload['saved_DF_scalar']):
            raise ValueError('VERSIONED_DESCRIPTOR_SOURCE_OR_SCALAR_DIFFERS')
        if dictionary_builder(item) == NO_DROP_BUILDER:
            if result.get('builder') != payload['builder'] or row.get('builder') != payload['builder']:
                raise ValueError('VERSIONED_P0_BUILDER_PROVENANCE_DIFFERS')

    def _store_split(self, prefix, value, *, verify):
        if type(prefix) is not int or not 1 <= prefix < self.fragment_count:
            raise ValueError('CACHED_PREFIX_OUTSIDE_NONEMPTY_TAIL')
        builder = dictionary_builder(value['item'])
        if verify:
            self._validate_product(prefix, value)
        if not hasattr(self, 'split_versions'):
            self.split_versions = {}
            self._legacy_splits = deepcopy(getattr(self, 'extra_splits', {}))
        versions = self.split_versions.setdefault(prefix, {})
        old = versions.get(builder)
        if old is not None and old['item'] != value['item']:
            raise ValueError('DIFFERENT_DICTIONARY_FOR_SAME_BUILDER_SELECT_FIXED_SOURCE')
        versions[builder] = deepcopy(value)
        if builder == LEGACY_BUILDER:
            self._legacy_splits[prefix] = deepcopy(value)
        selected = self._selected_split(prefix)
        if selected is not None:
            self.extra_splits[prefix] = selected
        else:
            self.extra_splits.pop(prefix, None)

    def _selected_split(self, prefix):
        versions = getattr(self, 'split_versions', {}).get(prefix, {})
        requested = getattr(self, 'builder_id', None)
        if requested:
            return versions.get(requested)
        return versions.get(NO_DROP_BUILDER) or versions.get(LEGACY_BUILDER)

    def save_split(self, prefix, value):
        builder = dictionary_builder(value['item'])
        if getattr(self, 'builder_id', None) not in (None, builder):
            raise ValueError('MATERIALIZED_DICTIONARY_BUILDER_DIFFERS_FROM_REQUEST')
        self._store_split(prefix, value, verify=builder == NO_DROP_BUILDER)
        self.write_cache()

    def remember_configuration(self, config, path):
        if not any(c['id'] == config['id'] for c, _ in self.configurations):
            self.configurations.append((deepcopy(config), None))
        if not any(v['configuration']['id'] == config['id'] for v in self.local_configurations):
            self.local_configurations.append(dict(configuration=deepcopy(config), file=str(Path(path).resolve())))
        self.write_cache()

    def write_cache(self):
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            temp = self.cache.with_suffix(self.cache.suffix+'.tmp')
            legacy = getattr(self, '_legacy_splits', self.extra_splits)
            versions = {str(p): {k:v for k,v in values.items() if k != LEGACY_BUILDER}
                        for p, values in getattr(self, 'split_versions', {}).items()
                        if any(k != LEGACY_BUILDER for k in values)}
            temp.write_text(json.dumps(dict(binding=self.binding, splits=legacy, split_versions=versions,
                                           configurations=self.local_configurations), indent=2)+'\n')
            temp.replace(self.cache)

    def descriptor(self, prefix):
        if str(prefix) in self.context['prefixes']:
            return dict(status='SAVED', descriptor=deepcopy(self.context['prefixes'][str(prefix)]),
                        source=dict(index=deepcopy(E.p0.INDEX), context_sha256=self.meta['context_source']['sha256'],
                                    selector=['prefixes', str(prefix)]))
        product = self.extra_splits.get(prefix) or getattr(self, '_legacy_splits', {}).get(prefix)
        if product is None:
            versions = getattr(self, 'split_versions', {}).get(prefix, {})
            product = next(iter(versions.values()), {})
        saved = product.get('descriptor') if product else None
        if saved:
            if self.read(saved['source']) != saved['descriptor']:
                raise ValueError('CACHED_DESCRIPTOR_DIFFERS_FROM_IMMUTABLE_SOURCE')
            return deepcopy(saved)
        if prefix in (7, 8, 9):
            # FIXED_INPUTS is an input descriptor, not adoption of its failed P0 dictionary.
            source = dict(commit='9fa9525b3', path='docs/research/b2_pauli_split_p0_20261116/LD%d/FIXED_INPUTS.json' % prefix)
            fixed = self.read(source)
            if fixed['binding'] != self.binding:
                raise ValueError('SAVED_P0_INPUT_DESCRIPTOR_BINDING_DIFFERS')
            return dict(status='SAVED_P0_INPUT_DESCRIPTOR', descriptor=deepcopy(fixed['saved_tail_descriptors'][str(prefix)]),
                        source=dict(source, selector=['saved_tail_descriptors', str(prefix)]))
        return dict(status='NEEDS_DERIVATION', descriptor=None,
                    source=dict(commit='7f4c9a171a1d4caeaeaee6c443217edec761f4c9',
                                path='scripts/h46_new2_core.py', function='matrix_only_tail_coefficients'),
                    missing_inputs=[], additional_science='saved DF factor coefficient extraction (no fit/circuit)')

    def dictionary(self, prefix):
        saved = self._selected_split(prefix)
        if saved and dictionary_builder(saved['item']) == NO_DROP_BUILDER:
            # This immutable product was validated at registration/cache load/save.
            return deepcopy(saved['item'])
        if prefix in self.baselines:
            item = E.dictionary_item(self.root, self.baselines[prefix][0])
            if getattr(self, 'builder_id', None) in (None, dictionary_builder(item)):
                return item
            return None
        saved = saved or self.extra_splits.get(prefix)
        if not saved:
            return None
        # The signal entry checks hash, prefix, binding, scalar and P0 gates from its own locators.
        return E.dictionary_item(self.root, self.baseline_from_item(prefix, saved['item']))

    def baseline_from_item(self, prefix, item):
        # Reuse fixed P1's conversion, with a split-specific original DF spec.
        original = self.read(dict(commit=E.p0.MENU, path=E.p0.DIRECTORY+'CONFIGURATION_LD1.json'))
        original['spec']['L_D'] = prefix
        descriptor = self.descriptor(prefix)
        if descriptor['descriptor'] is None:
            raise ValueError('SPLIT_DESCRIPTOR_REQUIRED')
        d = descriptor['descriptor']
        original['spec']['split'].update(L_D=prefix, scalar=d['scalar'],
            lambda_R=encoded(d['lambda_R']), lambda_source=deepcopy(descriptor['source']))
        return E.L.generate_configuration(self.root, generator(str(self.root))[0], original, item)

    def signal_plan(self, config, locator, descriptor):
        from .pauli_split_descriptor import stage1_metadata
        if self._index is None:
            self._index = E.saved_index(self.root, self.packets)
        p = config['spec']['L_D']
        context = (nullcontext() if str(p) in self.context['prefixes'] else
                   stage1_metadata(self.root, p, descriptor['descriptor'], descriptor['source']))
        with context:
            plan = E.plan(self.root, config, packets=self.packets, index=self._index, configuration_source=locator)
        plan['split_signal_validation'] = ('SAVED_MEANS_FOR_THIS_DICTIONARY_EXIST' if any(
            v['record']['physical_context']['split']['tail_dictionary']['dictionary_sha256'] ==
            config['spec']['split']['tail_dictionary']['dictionary_sha256'] for v in self._index.values())
            else 'NO_SAVED_MEAN_FOR_THIS_SPLIT_DICTIONARY')
        return plan


def inventory(assets):
    rows = []
    for p in range(1, assets.fragment_count):
        item = assets.dictionary(p)
        identity = E.L.dictionary_identity(item) if item is not None else None
        matches = [(c, loc) for c, loc in assets.configurations
                   if c['spec']['L_D'] == p and c['spec']['split'].get('tail_dictionary') == identity]
        baseline = next((value for value in matches if value[1] is not None), matches[0] if matches else None)
        rows.append(dict(L_D=p, descriptor=assets.descriptor(p),
            dictionary_source=None if item is None else item['source'],
            dictionary_builder_id=None if item is None else dictionary_builder(item),
            available_builder_ids=sorted(getattr(assets, 'split_versions', {}).get(p, {})),
            baseline_configuration_source=None if baseline is None else baseline[1],
            baseline_configuration_id=None if baseline is None else baseline[0]['id']))
    return dict(fragment_count=assets.fragment_count, one_body='ALWAYS_DETERMINISTIC_HEAD',
                fragment_count_source=dict(index=deepcopy(E.p0.INDEX), input='H4_tol_receipt.json',
                    selectors=[['arrays', 'lambdas', 'shape'], ['arrays', 'g_matrices', 'shape']]),
                admissible_L_D=list(range(1, assets.fragment_count)),
                rows=rows)


def request_parts(item, assets):
    """Existing B2 generator items or unmodified B2 NEEDS candidate rows."""
    item = deepcopy(item)
    if 'method' in item and item['method'] != 'PR_PAULI':
        raise ValueError('ONLY_PR_PAULI_REQUESTS')
    base = None
    if item.get('base'):
        base = E.load_configuration(assets.root, item['base'])[0]
    elif item.get('configuration_id') is not None and item.get('signal_source'):
        base = E.load_configuration(assets.root, item['signal_source'], configuration_id=item['configuration_id'])[0]
    design = item.get('configuration') or item.get('key') or {}
    if 'spec' in design:
        base = design
        design = {}
    prefix = design.get('L_D', item.get('L_D', base['spec']['L_D'] if base else None))
    if type(prefix) is not int or not 1 <= prefix < assets.fragment_count:
        raise ValueError('L_D_MUST_HAVE_ONE_BODY_HEAD_AND_NONEMPTY_TAIL')
    if base and (base['spec']['binding'] != assets.binding or base['spec']['L_D'] != prefix
                 or base['spec']['split'].get('tail_dictionary', {}).get('kind') != E.KIND):
        raise ValueError('BASE_H_SPLIT_OR_DICTIONARY_RULES_DIFFER')
    for key in ('binding', 'hamiltonian', 'snapshot', 'state'):
        supplied = item.get(key, design.get(key))
        expected = assets.binding if key == 'binding' else assets.binding[key]
        if supplied is not None and supplied != expected:
            raise ValueError('REQUEST_'+key.upper()+'_DIFFERS')
    changes = deepcopy(item.get('changes', {}))
    for key in ('formula', 'delta_over_h', 'delta_pf', 'kappa_multiplier', 'finite_K', 'K'):
        if key in design:
            if key in changes and changes[key] != design[key]:
                raise ValueError('CONFLICTING_REQUEST_FIELD:'+key)
            changes[key] = design[key]
    unknown = set(changes)-{'formula', 'delta_over_h', 'delta_pf', 'kappa_multiplier', 'finite_K', 'K'}
    if unknown:
        raise ValueError('UNSUPPORTED_CHANGES:'+','.join(sorted(unknown)))
    if changes.get('formula', '2nd') not in FORMULAS:
        raise ValueError('UNSUPPORTED_FORMULA')
    return prefix, base, changes


def changed_configuration(assets, base, changes):
    """Call B2's exact generator; K is the sole additional native-spec input."""
    b, build = generator(str(assets.root))
    ch = {k: v for k, v in changes.items() if k in ('formula', 'delta_over_h', 'kappa_multiplier')}
    if 'delta_pf' in changes:
        relative = rational(changes['delta_pf']) / rational(base['spec']['h'])
        if 'delta_over_h' in ch and rational(ch['delta_over_h']) != relative:
            raise ValueError('CONFLICTING_ABSOLUTE_AND_RELATIVE_STEP')
        ch['delta_over_h'] = str(relative)
    for k in ('delta_over_h', 'kappa_multiplier'):
        if k in ch:
            if rational(ch[k]) <= 0:
                raise ValueError('POSITIVE_'+k.upper()+'_REQUIRED')
            if isinstance(ch[k], dict):
                ch[k] = str(rational(ch[k]))
    if build(base, {}) != base:
        raise ValueError('BASE_NOT_REPRODUCED_BY_B2_MAPPER')
    config = build(base, ch) if ch else base
    k = changes.get('K', changes.get('finite_K', base['spec']['finite_K']))
    if 'K' in changes and 'finite_K' in changes and changes['K'] != changes['finite_K']:
        raise ValueError('CONFLICTING_K')
    if type(k) is not int or k < 0:
        raise ValueError('FINITE_K_MUST_BE_NONNEGATIVE_INTEGER')
    if k != config['spec']['finite_K']:
        spec = deepcopy(config['spec']); spec['finite_K'] = k
        config = b.generate(spec)
    b.validate(config)
    return config


def price_input(assets, config):
    """Prepare a passive D2 request. No quote/model is evaluated by A."""
    s = config['spec']
    entries = [e for e in assets.price_registry['entries']
               if e['name'] == 'PAULI_LD%d' % s['L_D']]
    if entries:
        entry = entries[0]
        implementation = deepcopy(entry['implementation'])
        source = dict(commit=D2, path=REGISTRY, selector=['entries', assets.price_registry['entries'].index(entry), 'implementation'])
        evidence = dict(registry_entry=entry['name'], price_source=entry.get('price_source'),
                        model_source=entry.get('model_source'), direct_sources=entry.get('direct_sources', []))
        needs = []
        saved_config = assets.read(entry['configuration_source'])
        coverage = []
        if entry.get('price_source') and saved_config == config:
            price_rows = assets.read(entry['price_source'])['rows']
            for r in price_rows:
                if r.get('L_D', s['L_D']) != s['L_D']:
                    continue
                if r['m'] not in range(13) or r['axis'] not in ('X', 'Y'):
                    raise ValueError('SAVED_PRICE_AXIS_SCOPE_DIFFERS')
                cr = config['rounds'][r['m']]
                ids = {cr['candidate']['id'], cr.get('physical_candidate', {}).get('id')}
                if (r.get('q', cr['q_PF']) != cr['q_PF']
                        or r.get('R', r.get('R_outer', cr['R_outer'])) != cr['R_outer']
                        or r.get('physical_id') not in ids):
                    raise ValueError('SAVED_PRICE_PHYSICAL_ID_OR_SCHEDULE_DIFFERS')
                coverage.append((r['m'], r['axis']))
        saved_complete = sorted(coverage) == [(m, a) for m in range(13) for a in ('X', 'Y')]
        if not saved_complete:
            needs.append('MODEL_QUOTE_APPLICABILITY_OR_MISSING_26_AXIS_EVIDENCE')
    else:
        implementation = None; source = None; evidence = None
        needs = ['CANDIDATE_SPECIFIC_IMPLEMENTATION_CONTRACT', 'CANDIDATE_SPECIFIC_PRICE_EVIDENCE_OR_MODEL',
                 'CONTROL_LOWERING_SCALAR_FRAGMENT_ORDER_SAMPLER_AND_WRAPPER_SCOPE'] + getattr(assets, 'price_registry_needs', [])
        saved_complete = False
    return dict(entry=dict(commit=D2, path='src/trottertracks/pr_price_rules.py', function='PriceRules.quote'),
                request=dict(configuration=deepcopy(config), implementation=implementation, policy='MODEL'),
                implementation_source=source, implementation_status='SAVED_CONTRACT_PENDING_APPLICABILITY' if entries else 'MISSING',
                evidence=evidence, needs=needs, quote_executed=False, prices_26_complete=saved_complete,
                next_action='QUOTE_MODEL' if entries else 'D2_BIND_THIS_CANDIDATE_CONTRACT_THEN_QUOTE_MODEL',
                note='No coefficient or implementation qualification is transferred between L_D values.')


def prepare_item(assets, item, *, materialize=None):
    out = dict(item_id=item.get('item_id', item.get('name')), request=deepcopy(item),
               status='NOT_PREPARED', configuration=None, configuration_id=None,
               configuration_source=None, schedule=None,
               readiness=dict(configuration_complete=False, signal_13round_complete=False, price_26axis_complete=None),
               missing_inputs=[], next_actions=[])
    try:
        prefix, base, changes = request_parts(item, assets)
        out.update(L_D=prefix, requested_changes=changes,
                   requested_dictionary_builder_id=getattr(assets, 'builder_id', None))
        descriptor = assets.descriptor(prefix)
        dictionary = assets.dictionary(prefix)
        out.update(descriptor=descriptor, dictionary_source=None if dictionary is None else dictionary['source'])
        if dictionary is None:
            if materialize is None:
                out.update(status='NEEDS_SPLIT_PREPARATION',
                           next_actions=['DERIVE_DESCRIPTOR' if descriptor['descriptor'] is None else 'REUSE_DESCRIPTOR',
                                         'P0_DICTIONARY_AND_OPERATOR_GATES', 'P1_CONFIGURATION'],
                           scientific_request=dict(prefix=prefix, descriptor=descriptor,
                             stage1='scripts/run_df_head_pauli_tail_split_stage1.py:execute_p0',
                             route='scripts/prepare_pauli_candidate.py --prepare-missing --source SOURCE --request REQUEST --output OUT',
                             new_H4_dictionary_executed=False))
                out['missing_inputs'] = descriptor.get('missing_inputs', [])
                return out
            product = materialize(prefix, descriptor)
            if product.get('status') != 'SPLIT_READY':
                out.update(status=product.get('status', 'SPLIT_PREPARATION_FAILED'), split_preparation=product,
                           missing_inputs=product.get('missing_inputs', []))
                return out
            assets.save_split(prefix, product)
            out['split_preparation'] = product
            descriptor = assets.descriptor(prefix); dictionary = assets.dictionary(prefix)
        if (dictionary['payload']['binding'] != assets.binding or dictionary['payload']['prefix'] != prefix
                or dictionary['payload']['saved_DF_scalar'] != descriptor['descriptor']['scalar']):
            raise ValueError('DICTIONARY_H_SPLIT_OR_SCALAR_DIFFERS')
        selected_identity = E.L.dictionary_identity(dictionary)
        if base is None:
            previous = assets.baselines.get(prefix)
            base = (previous[0] if previous and previous[0]['spec']['split']['tail_dictionary'] == selected_identity
                    else assets.baseline_from_item(prefix, dictionary))
        if base['spec']['split']['tail_dictionary'] != selected_identity:
            raise ValueError('BASE_DICTIONARY_DIFFERS')
        config = changed_configuration(assets, base, changes)
        structure_ok, structure = generator(str(assets.root))[1].structure_checks(config)
        if not structure_ok:
            raise ValueError('B2_STRUCTURE_CHECK_FAILED')
        if item.get('configuration_id') is not None and not changes and item['configuration_id'] != config['id']:
            raise ValueError('REQUESTED_CONFIGURATION_ID_DIFFERS')
        hits = [(c, loc) for c, loc in assets.configurations if c['id'] == config['id']]
        locator = hits[0][1] if hits else None
        if hits and hits[0][0] != config:
            raise ValueError('SAVED_ID_HAS_DIFFERENT_CANONICAL_BODY')
        try:
            plan = assets.signal_plan(config, locator, descriptor)
        except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            reason = 'SIGNAL_EVIDENCE_SOURCE_UNRESOLVED: '+str(exc)
            plan = dict(schema='A_PAULI_SIGNAL_ENTRY_PLAN_V1', configuration_id=config['id'],
                        planning_status='SOURCE_UNRESOLVED', counts=dict(UNSUPPORTED=13),
                        reuse_rounds=[], rebind_rounds=[], request_rounds=[], unsupported_rounds=list(range(13)),
                        rounds=[dict(m=r['m'], candidate_id=r['candidate']['id'], decision='UNSUPPORTED',
                                     provider_support='UNSUPPORTED', reasons=[reason]) for r in config['rounds']],
                        configuration_reasons=[reason], new_mean_calls=0, compile=0, Q3=0)
        rows = [dict(m=r['m'], q_RPE=r['q_RPE'], T=r['T'], q_PF=r['q_PF'], actual_delta_pf=r['actual_delta_pf'],
                     R_outer=r['R_outer'], physical_id=r['candidate']['id'], structure=r['structure'],
                     mapping=r['mapping'], law_binding=r['law_binding']) for r in config['rounds']]
        try:
            price = price_input(assets, config)
        except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            price = dict(entry=dict(commit=D2, path='src/trottertracks/pr_price_rules.py', function='PriceRules.quote'),
                         request=dict(configuration=deepcopy(config), implementation=None, policy='MODEL'),
                         needs=['RECOVER_OR_CHECK_CANDIDATE_PRICE_CONTRACT_OR_SOURCE: '+str(exc)],
                         prices_26_complete=False, quote_executed=False,
                         next_action='RESOLVE_PRICE_EVIDENCE_WITHOUT_CHANGING_CONFIGURATION')
        resolution = ('REUSED_CONFIGURATION' if locator else 'REUSED_LOCAL_METADATA_CONFIGURATION') if hits else 'GENERATED_CONFIGURATION'
        out.update(status='GENERATED', preparation_status=resolution,
                   kind='pauli_native', base=deepcopy(item.get('base')), changes=deepcopy(changes),
                   base_reproduced=True, identical_to_base=config == base, supported_formulas=list(FORMULAS),
                   structure=structure,
                   descriptor=descriptor, dictionary_source=dictionary['source'], dictionary_identity=E.L.dictionary_identity(dictionary),
                   dictionary_builder_id=dictionary_builder(dictionary),
                   dictionary_builder=deepcopy(dictionary['payload'].get('builder', dict(id=LEGACY_BUILDER))),
                   lambda_P=dictionary['payload']['lambda_P'], lambda_DF=descriptor['descriptor']['lambda_R'],
                   configuration=config, configuration_id=config['id'], configuration_source=locator, schedule=rows,
                   signal_plan=plan, price_handoff=price,
                   readiness=dict(configuration_complete=True, signal_13round_complete=len(plan['reuse_rounds']) == 13,
                                  price_26axis_complete=price['prices_26_complete']),
                   next_actions=['ASSEMBLE_REUSED_SIGNAL_PACKET' if len(plan['reuse_rounds']) == 13 else 'PROCESS_SIGNAL_PLAN',
                                 'PROCESS_D2_PRICE_HANDOFF'],
                   B2_generator_source=dict(commit=B2, path=GENERATOR), B_canonical_source=E.p1.B,
                   signal_entry_source='4e5b8bef6', signal_price_Q3_calls=0)
    except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        out.update(status='INPUT_OR_SOURCE_UNRESOLVED', error=type(exc).__name__+': '+str(exc),
                   missing_inputs=['FIX_REQUEST_OR_RECOVER_NAMED_SOURCE: '+str(exc)])
    return out


def prepare(request, assets, *, materialize=None):
    if request.get('kind', 'pauli_native') != 'pauli_native':
        raise ValueError('ONLY_PAULI_NATIVE')
    items = request.get('items', [request])
    results = [prepare_item(assets, item, materialize=materialize) for item in items]
    return dict(schema='A_PAULI_CANDIDATE_PREPARATION_V1', kind='pauli_native',
                items=results, inventory=inventory(assets),
                split_materializer_called=any('split_preparation' in r for r in results),
                signal_price_Q3_evaluation_performed=False,
                quote_executed=False, N_MSE_G_Q3=None)
