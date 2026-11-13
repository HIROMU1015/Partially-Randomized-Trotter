"""SOURCE-bound implementation admission before existing B price arithmetic.

An externally selected comparison policy is required; this module neither
chooses a scientific scenario nor certifies prices, laws, intervals or N.
Historical records stay unchanged: a pinned producer/receipt sidecar supplies
the implementation facts, rather than a default inferred from compiler hash.
"""
from copy import deepcopy
from functools import lru_cache
import ast

from . import baseline_signal_pipeline as C

NO_RTE = 'NOT_APPLICABLE_NO_RTE'
QUANTITY = 'FULL_WRAPPER_MEAN_COMPILED_PRICE'
SCALAR_ONCE = 'ONCE_AT_TOTAL_T'
NEW_ZZ = 'CONTROLLED_ZZ_CX_CRZ_CX_V1'
OLD_ZZ = 'CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'


@lru_cache(maxsize=128)
def _document(root, commit, path):
    return C.p.SavedArtifacts(root).read(dict(commit=commit,path=path))


def evidence(root, locator):
    """Read only pinned public metadata. Never read input arrays or price raw."""
    if (not isinstance(locator, dict) or not locator.get('path')
            or not isinstance(locator.get('commit'), str) or len(locator['commit']) != 40):
        raise ValueError('Pinned implementation evidence SOURCE required')
    value = _document(str(root), locator['commit'], locator['path'])
    for key in locator.get('selector', ()):
        value = value[key]
    return deepcopy(value)


def _same(a, b, message):
    if not C.p.same(a, b):
        raise ValueError(message)


@lru_cache(maxsize=32)
def _rte_source(root, lowering, commit, path):
    """Reuse D S1's two known SOURCE routes; unknown routes need new evidence.

    Matching the emitter is necessary, not proof it was activated: the pinned
    execution/receipt assertion must also bind this route to the actual price.
    """
    references={
        NEW_ZZ:('01e068af7e06d9f1f5621a5b3800c30eb32ecb06','src/trottertracks/efficient_controlled_zz.py'),
        OLD_ZZ:('c31ef424175c02379eafebe8fc6c7548c1d9aabe','src/trotterlib/df_rte_qiskit.py')}
    if lowering not in references:
        raise ValueError('Unknown RTE lowering SOURCE route')
    def central(c,p):
        tree=ast.parse(C.blob(root,c,p))
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_append_central_application']
        if len(nodes)!=1:raise ValueError('RTE emitter SOURCE unresolved')
        return ast.dump(nodes[0],include_attributes=False)
    _same(central(commit,path),central(*references[lowering]),'RTE lowering contradicts emitter SOURCE')


def check(root, config, prices, *, cohorts, bindings, policy):
    """Check selected valued records before B.cost, with no price arithmetic.

    policy.by_method specifies native policy, RTE lowering and scalar quantity.
    implementation_evidence selects a fixed JSON assertion for this exact
    price SOURCE/physical context/axis/cohort/extension. A supplied extension
    alone is insufficient. The producer's scientific evidence remains external.
    """
    if (not isinstance(policy, dict) or not policy.get('source') or not policy.get('id')
            or not policy.get('profile') or not policy.get('by_method')):
        raise ValueError('Explicit price comparison implementation policy required')
    api = C.published_api(str(root)); B = api.B
    B._validate(config)
    method = B.comparison_method(config['spec'])
    required = policy['by_method'].get(method)
    if not required:
        raise ValueError('Method outside the explicit implementation policy')
    rte_domains={policy['by_method'][m].get('rte_rotation_lowering') for m in ('PR','FULL_H_RTE')
                 if m in policy['by_method']}
    if len(rte_domains)>1:
        raise ValueError('Mixed RTE lowering domains require separate policies; no cross-domain equivalence adopted')
    profile = config['spec']['profile']
    _same(profile, policy['profile'], 'Comparison compiler/control/wrapper/P0 profile differs')
    if (profile.get('P') != 0 or profile.get('wrapper') != 'full_Hadamard_with_measurement'
            or profile.get('scope') != 'single_hadamard_interrogation_without_state_preparation'):
        raise ValueError('Full measured wrapper and P0 required')
    rte = method in ('PR', 'FULL_H_RTE')
    if (not required.get('native_policy') or not required.get('scalar_quantity')
            or not required.get('rte_rotation_lowering')
            or (required['rte_rotation_lowering'] == NO_RTE) == rte):
        raise ValueError('RTE applicability/native/scalar policy is inconsistent')
    baseline = None
    if method in ('FULL_PF','FULL_H_RTE'):
        layouts = ({r['m']:{k:r['mapping']['physical_layout'][k] for k in ('q','R_outer')}
                    for r in config['rounds']} if method=='FULL_H_RTE' else None)
        baseline = api.D.physical_schedule(config['spec'],layout_by_round=layouts)
        api.D._check_against_b(baseline,config)
    checked = []
    for row in config['rounds']:
        for axis in B.p.AXES:
            cohort = cohorts.get((row['candidate']['id'], axis))
            matches = [v for v in prices if v.get('axis') == axis and v.get('cohort') == cohort
                       and C.p.same(v.get('identity'), row['identity'])]
            if len(matches) > 1:
                raise ValueError('Ambiguous selected price implementation')
            if not matches or (matches[0].get('mean') or {}).get('rz_count') is None:
                continue  # Existing B handles missing prices; never substitute zero.
            price = matches[0]
            sides = [v for v in bindings if v.get('axis') == axis and v.get('cohort') == cohort
                     and C.p.same(v.get('identity'), row['identity'])]
            if len(sides) != 1:
                raise ValueError('Missing or ambiguous implementation sidecar')
            side = sides[0]
            _same(side.get('price_source'), price.get('source'), 'Price implementation producer SOURCE differs')
            _same(side.get('law_binding'), row['law_binding'], 'Price implementation law differs')
            _same(price.get('law_binding'),row['law_binding'],'Price law binding differs')
            facts = side.get('implementation')
            if not facts:
                raise ValueError('Unknown price native/lowering implementation')
            proof = evidence(root, side.get('implementation_evidence'))
            expected = dict(price_source=price['source'], identity=row['identity'], law_binding=row['law_binding'],
                axis=axis, cohort=cohort, physical_context=side.get('physical_context'),
                profile_extension=side.get('profile_extension'), implementation=facts)
            _same(proof, expected, 'Pinned implementation evidence contradicts SOURCE/axis/scalar/domain')
            physical = side.get('physical_context') or {}
            if not physical.get('physical_candidate') or not physical.get('binding'):
                raise ValueError('Implementation physical candidate binding missing')
            _same(physical['binding'], config['spec']['binding'], 'Implementation H/state/SOURCE differs')
            _same(physical.get('profile'), profile, 'Implementation compiler profile differs')
            _same(physical.get('split'),config['spec']['split'],'Implementation PF/RTE split/scalar construction differs')
            _same(physical.get('law'),config['spec']['law'],'Implementation event law differs')
            c=physical['physical_candidate']
            if baseline is not None:
                _same(c,baseline['rounds'][row['m']]['physical_candidate'],'Implementation physical candidate differs')
            else:
                expected=dict(system=config['spec']['binding']['system'],method=(
                    config['spec']['split'].get('original_PF_method') if method=='TRUNCATED_PF' else 'PR'),
                    formula=config['spec']['formula'])
                if any(c.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Implementation physical method/formula differs')
                if (B.a.exact_scalar(c.get('T'))!=B.a.exact_scalar(row['T'])
                        or C.p.candidate_id({k:v for k,v in c.items() if k!='id'})!=c.get('id')):
                    raise ValueError('Implementation physical T/id differs')
            if method=='FULL_H_RTE':
                layout=row['mapping']['physical_layout']
                if (c.get('q'),c.get('R_outer'),c.get('K'))!=(layout['q'],layout['R_outer'],config['spec']['finite_K']):
                    raise ValueError('Implementation Full-H grouping/law differs')
                _same(physical.get('structure'),dict(row['structure'],allocation=layout['allocation'],deterministic_blocks=0),
                      'Implementation Full-H structure differs')
            elif (c.get('q')!=row['q_PF'] or c.get('prefix')!=config['spec']['L_D']
                  or float(B.a.exact_scalar(c.get('delta'))).hex()!=float(B.a.exact_scalar(row['actual_delta_pf'])).hex()):
                raise ValueError('Implementation physical q/prefix/actual width differs')
            for name in ('native_source', 'wrapper_source', 'scalar_source'):
                src = facts.get(name) or {}
                if not src.get('commit') or not src.get('path'):
                    raise ValueError('Implementation SOURCE missing: ' + name)
                # Existence is provenance, not a replacement for the producer's
                # source-route/receipt checks named by implementation_evidence.
                C.blob(root, src['commit'], src['path'])
            _same(facts.get('method'), method, 'Implementation METHOD differs')
            _same(facts.get('profile'), profile, 'Implementation compiler/control/wrapper differs')
            _same(facts.get('measurement'), B.p.measurement(row, axis), 'Implementation measurement/axis differs')
            for name in ('measurement_axis','observable','outcome_convention','units'):
                _same(price.get(name),B.p.measurement(row,axis)[name], 'Price measurement/axis differs')
            _same(price.get('quantity'), QUANTITY, 'Implementation price quantity differs')
            if price.get('normalization_is_price_multiplier') is not False:
                raise ValueError('Normalization is not a price multiplier')
            for name in ('native_policy', 'rte_rotation_lowering', 'scalar_quantity'):
                _same(facts.get(name), required[name], 'Comparison implementation differs: ' + name)
            if facts.get('scalar_application') != SCALAR_ONCE:
                raise ValueError('Scalar must be applied once at total T')
            if rte:
                src = facts.get('rte_lowering_source')
                if not src or not src.get('commit') or not src.get('path'):
                    raise ValueError('RTE lowering SOURCE missing')
                _rte_source(str(root),facts['rte_rotation_lowering'],src['commit'],src['path'])
            elif facts.get('rte_lowering_source') is not None:
                raise ValueError('Deterministic PF has no RTE lowering SOURCE')
            ext = side.get('profile_extension') or {}
            _same(ext.get('lowering_id'), facts['rte_rotation_lowering'] if rte else facts['native_policy'],
                  'Profile extension contradicts actual native/lowering')
            recorded = price.get('price_profile')
            if recorded is not None:
                _same(recorded, dict(profile, lowering_id=ext['lowering_id']), 'Record price profile contradicts implementation')
            # Optional resolved scalar must match the bound requested total;
            # quantities alone do not assert equality of different coefficients.
            if 'resolved_scalar' in facts or 'expected_scalar' in facts:
                if facts.get('resolved_scalar') is None or facts.get('expected_scalar') is None:
                    raise ValueError('Resolved/expected scalar missing')
                _same(facts['resolved_scalar'], facts['expected_scalar'], 'Resolved scalar differs from expected scalar')
            checked.append(dict(m=row['m'], axis=axis, evidence=deepcopy(side['implementation_evidence'])))
    return dict(status='IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION',
                policy=deepcopy(policy), checked=checked)


def cost(root, controller, config, n_packet, prices, *, cohorts, price_bindings, comparison_policy,
         price_policy='qualified_expectation'):
    """B handoff entry: check immediately before the unchanged B.cost call."""
    check(root, config, prices, cohorts=cohorts, bindings=price_bindings, policy=comparison_policy)
    return controller.cost(config, n_packet, prices, cohorts=cohorts, price_policy=price_policy)
