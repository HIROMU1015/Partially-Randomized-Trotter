"""A preparation + supplied D2 applicability -> saved reuse or split-calibration input.

Only metadata is processed. No path, circuit, fit, quote, Q3 or coefficient
transfer is performed. Keep top-level imports light for the provider-first child.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import subprocess

MODULE = 'src/trottertracks/pauli_calibration_request.py'
DRIVER_SOURCE = '1097c37f9'
LEGACY = '13ab1acfaebe5f33958e1db67862f5c45725ab91'
LOWERING = 'PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF'
COMMON = dict(commit='4220f3006d0d8d413e195ed948e250578513b557',
              path='docs/research/df_head_pauli_tail_p0_results_20261114/run01/COMMON_OPERATORS.npz')
SAVED_BINDINGS = (
    dict(commit='3074433d16dd8bbfa3fbf173e1da3606af6670d2',
         path='docs/research/a_pauli_ld3_cal_price_20261116/BINDING.json'),
    dict(commit='aedf55d9a', path='docs/research/a_pauli_ld5_cal_price_20261116/BINDING.json'))


def frac(v):
    return Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)


def exact(value):
    value=frac(value)
    return dict(numerator=value.numerator,denominator=value.denominator)


@lru_cache(maxsize=None)
def driver(root):
    path = Path(root)/'scripts/run_pauli_split_cal.py'
    spec = importlib.util.spec_from_file_location('_request_split_cal_'+str(abs(hash(str(root)))), path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def implementation():
    """The generic explicit-operators acquisition recipe, not the LD1/2 proposal()."""
    return dict(lowering_id=LOWERING, scalar='HAMILTONIAN_CONSTANT_PLUS_SAVED_sR_ONCE_AT_TOTAL_T',
                fragment_order='SAVED_SNAPSHOT_ORDER_ONE_BODY_SEPARATE_HEAD', sampler='ORIGINAL_NORMAL_PCG64_NUMPY_CHOICE',
                contract_source=dict(commit=LEGACY, path='scripts/run_pauli_b1_prices.py'))


def implementation_sources():
    return dict(acquire=dict(commit=LEGACY, path='scripts/run_pauli_b1_prices.py', function='acquire_path',
                             arguments='explicit operators and blocks; no proposal(prefix) lookup'),
                connection=dict(commit=LEGACY, path='src/trottertracks/pauli_tail_price_connection.py'),
                lowering=dict(commit='9c4006bf', path='src/trotterlib/h4_d2_pauli_control.py'),
                native=dict(commit='af284d853890155bacd19e61850c196bd1d5c660', path='scripts/h46_long_time_native.py'),
                wrapper=dict(commit='af284d853890155bacd19e61850c196bd1d5c660',
                             path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py'),
                provider=dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe'))


def scope_reasons(config, rounds=None, *, multiple_steps=False):
    s = config['spec']; reasons = []
    if s['method'] != 'PR' or not s['split'].get('tail_dictionary'):
        reasons.append('PR_PAULI_CONFIGURATION_REQUIRED')
    if s['formula'] != '2nd' or s['finite_K'] != 2:
        reasons.append('NEW_FORMULA_OR_K_BOUNDARY_EVENT_RULE_AND_CALIBRATION_REQUIRED')
    if s['M'] != 12 or [r['m'] for r in config['rounds']] != list(range(13)):
        reasons.append('FIXED_13ROUND_CONFIGURATION_REQUIRED')
    for row in config['rounds']:
        if rounds is not None and row['m'] not in rounds:
            continue
        step=frac(row['actual_delta_pf']);T=frac(row['T'])
        if (T!=row['q_PF']*step or step<=0 or not math.isfinite(float(step)) or
                not math.isfinite(float(T)) or Fraction(float(step))!=step or Fraction(float(T))!=T):
            reasons.append('EXACT_STEP_OR_T_NOT_BINARY64_DRIVER_REPRESENTABLE:m%d' % row['m'])
        if not multiple_steps and step != Fraction(.1):
            reasons.append('SAME_STEP_CALIBRATION_OR_AUDIT_EXTENSION_REQUIRED:m%d' % row['m'])
        st = row['structure']
        if (st.get('signed_weights') != [1.] or st.get('allocation') != [row['R_outer']]
                or st.get('event_count') != row['q_PF']*row['R_outer']):
            reasons.append('SIGNED_PIECE_OR_EVENT_RECIPE_EXTENSION_REQUIRED:m%d' % row['m'])
    p = s['profile']
    if (p.get('control') != 'symmetric_directional' or p.get('wrapper') != 'full_Hadamard_with_measurement'
            or p.get('P') != 0 or p.get('qiskit_version') != '1.3.0'
            or p.get('basis_gates') != ['rz', 'sx', 'x', 'cx']
            or p.get('optimization_level') != 1 or p.get('seed_transpiler') != 17):
        reasons.append('SOURCED_COMPILER_CONTROL_WRAPPER_IMPLEMENTATION_CONTRACT_REQUIRED')
    if (s['law'].get('model') != 'ideal_finite_RTE_law'
            or s['law'].get('event_model') != 'EXISTING_PAIRED_TAYLOR_SIGNED_PAULI_INVOLUTIONS'
            or s['law'].get('scalar') != 'SAVED_sR_AND_c_ONCE_AT_TOTAL_TIME'
            or s['split'].get('one_body_policy') != 'DETERMINISTIC'):
        reasons.append('SOURCED_SAMPLER_OR_SCALAR_IMPLEMENTATION_CONTRACT_REQUIRED')
    return list(dict.fromkeys(reasons))


def validate_driver_spec(root, spec):
    """Pure metadata guard used at the real driver's parent and worker boundary.

    B canonical validation and P0 SOURCE verification are performed by prepare;
    the scientific driver additionally checks its frozen binding and source.
    """
    config = spec['configuration']; s = config['spec']; cell = spec['cell']
    domain=spec.get('calibration_domain');covered=spec.get('covered_rounds')
    if domain is not None and (not covered or len(set(covered))!=len(covered) or
            any(type(m)is not int or m not in range(13)for m in covered)):
        raise ValueError('COVERED_CANONICAL_ROUNDS_REQUIRED')
    reasons = scope_reasons(config,covered,multiple_steps=domain is not None)
    if reasons:
        raise ValueError('; '.join(reasons))
    td = s['split']['tail_dictionary']; p = s['L_D']; last = config['rounds'][12]
    if (spec['config_id'] != config['id'] or spec['prefix'] != p or td['prefix'] != p
            or spec['p0']['commit'] != td['source']['commit']
            or spec['p0']['directory'] != td['source']['path'].rsplit('/', 1)[0]+'/'
            or td['scalar_policy'] != 'KEEP_SAVED_sR_AND_c_ONCE'
            or td['event_space'] != 'FULL_2_TO_N_NOT_PROJECTED_PAULI'):
        raise ValueError('DRIVER_CONFIGURATION_DICTIONARY_OR_OPERATOR_SOURCE_DIFFERS')
    step=Fraction(.1)if domain is None else frac(domain['actual_step'])
    anchor_R=last['R_outer']if domain is None else max(config['rounds'][m]['R_outer']for m in covered)
    if domain is not None:
        if (Fraction(float(step))!=step or domain.get('actual_step_hex')!=float(step).hex() or
                any(frac(config['rounds'][m]['actual_delta_pf'])!=step for m in covered)):
            raise ValueError('CALIBRATION_EXACT_STEP_DOMAIN_DIFFERS')
    if (type(cell['q']) is not int or cell['q'] != 8 or type(cell['R']) is not int
            or cell['R'] < 1 or cell['R'] != anchor_R or Fraction(cell['T']) != step*8 or cell['K'] != 2
            or spec['m12'] != dict(id=last['candidate']['id'], q=last['q_PF'], R=last['R_outer'])):
        raise ValueError('CALIBRATION_TEMPLATE_OR_M12_MAPPING_DIFFERS')
    if cell['cell_id'] != 'Q8R%d' % cell['R'] or not spec.get('cohort'):
        raise ValueError('CALIBRATION_CELL_OR_COHORT_REQUIRED')
    if not spec.get('base') or not spec.get('private') or not spec.get('old_seed_sources'):
        raise ValueError('FRESH_OUTPUT_LOCATIONS_AND_SEED_SOURCES_REQUIRED')
    return spec


def _api():
    from . import pauli_candidate_preparation as A
    return A


def validate_quote(config, quote):
    if (quote.get('schema') != 'PR_PRICE_APPLICABILITY_V1' or quote['configuration_id'] != config['id']
            or quote.get('policy') != 'MODEL' or quote.get('method') != 'PR_PAULI'
            or quote.get('L_D') != config['spec']['L_D']):
        raise ValueError('D2_CONFIGURATION_ID_POLICY_OR_METHOD_DIFFERS')
    seen = set(); reuse = []; missing = []
    for row in quote['rows']:
        key = row['m'], row['axis']
        if key in seen or key[0] not in range(13) or key[1] not in ('X', 'Y'):
            raise ValueError('D2_AXIS_SCOPE_OR_DUPLICATE')
        seen.add(key); r = config['rounds'][row['m']]
        if (row['physical_id'] != r['candidate']['id'] or row['q'] != r['q_PF'] or row['R'] != r['R_outer']):
            raise ValueError('D2_PHYSICAL_ID_OR_SCHEDULE_DIFFERS')
        if row.get('unit_price_RZ') is not None:
            if (row.get('classification') != 'MODEL_REFERENCE'
                    or frac(row['unit_price_RZ']) <= 0 or not row.get('source')):
                raise ValueError('POSITIVE_PRICE_AND_SOURCE_REQUIRED')
            reuse.append(deepcopy(row))
        else:
            if row.get('classification') != 'NEEDS_PRICE_EVIDENCE' or not row.get('needs'):
                raise ValueError('D2_MISSING_PRICE_REASONS_REQUIRED')
            missing.append(deepcopy(row))
    if len(seen) != 26 or quote.get('complete') != (not missing):
        raise ValueError('D2_COMPLETE_FLAG_OR_26AXIS_COVERAGE_DIFFERS')
    return reuse, missing


def build_spec(root, config, configuration_source, *, execution=None, saved_bindings=SAVED_BINDINGS, covered_rounds=None):
    """One S0-r1 cell from the candidate, with published scientific inputs reused if identical."""
    A = _api(); d = driver(str(Path(root).resolve())); P = d.D3.legacy().B0.connection()
    A.E.p1.b_api(root).validate(config)
    item = A.E.dictionary_item(root, config); s = config['spec']; p = s['L_D']
    if item['payload']['binding'] != s['binding'] or item['payload']['saved_DF_scalar'] != s['split']['scalar']:
        raise ValueError('P0_H_STATE_SNAPSHOT_OR_SCALAR_DIFFERS')
    if frac(s['split']['lambda_R']) != Fraction(item['payload']['lambda_P']):
        raise ValueError('SCHEDULE_MUST_USE_DICTIONARY_LAMBDA_P')
    profile = A.E.load_configuration(root, A.E.SAVED_PACKETS[0], configuration_id=A.E.L.SAVED_PAULI_IDS[1])[0]['spec']['profile']
    if s['profile'] != profile:
        raise ValueError('CANDIDATE_SPECIFIC_COMPILER_PROFILE_CONTRACT_UNRESOLVED')
    if (P.ROUTE != 'PR_PAULI_TAIL' or P.LOWERING_ID != LOWERING):
        raise ValueError('FIXED_ACQUISITION_IMPLEMENTATION_UNRESOLVED')
    # Check locators only; no np.load, input_preflight, native_context or circuit build.
    for ref in (COMMON, A.E.S3.operators_locator(item)):
        A.E.p0.blob(root, ref['commit'], ref['path'])
    last = config['rounds'][12]
    domain=None
    if covered_rounds is None:
        R=last['R_outer'];step=Fraction(.1)
    else:
        steps={frac(config['rounds'][m]['actual_delta_pf'])for m in covered_rounds}
        if len(steps)!=1:raise ValueError('ONE_EXACT_STEP_PER_CALIBRATION_REQUIRED')
        step=next(iter(steps));R=max(config['rounds'][m]['R_outer']for m in covered_rounds)
        domain=dict(actual_step=exact(step),actual_step_hex=float(step).hex())
    cell = dict(cell_id='Q8R%d' % R, q=8, R=R, T=float(step)*8, K=2)
    request = d.D3.cell_request(P, s, cell, item['payload']['lambda_P'])
    matched = []
    for loc in saved_bindings:
        old = A.E.read_ref(root, loc)
        if old['request'] == request and old['dictionary_identity'] == s['split']['tail_dictionary']:
            matched.append((old, deepcopy(loc)))
    if len(matched) > 1:
        raise ValueError('CHOOSE_SAVED_CALIBRATION_COHORT_NO_POOLING')
    old, old_source = matched[0] if matched else (None, None)
    execution = execution or {}
    if old and execution:
        raise ValueError('SAVED_CALIBRATION_IS_REUSE_ONLY_NOT_A_NEW_COHORT')
    identifier = config['id'][:20]
    cell_folder=cell['cell_id']if domain is None else cell['cell_id']+'_'+request['candidate_id'].split(':')[1][:16]
    cohort = old['cohort'] if old else execution.get('cohort', 'A_PAULI_CAL_'+request['candidate_id'].split(':')[1][:20]+'_N2_V1')
    spec = dict(prefix=p, config=deepcopy(configuration_source), configuration=deepcopy(config), config_id=config['id'],
                p0=dict(commit=item['source']['commit'], directory=item['source']['path'].rsplit('/', 1)[0]+'/'),
                cell=cell, cohort=cohort, m12=dict(id=last['candidate']['id'], q=last['q_PF'], R=last['R_outer']),
                total_view=None, plan=dict(commit='7cf4e1f0536567df6dcf2da6c7bc63f4547aad7b',
                    path='docs/research/pauli_candidate_preparation_20261116/HANDOFF.md'),
                base=execution.get('base', 'docs/research/pauli_candidate_calibrations/'+identifier+'/'+cell_folder),
                private=execution.get('private', str(Path(root).parent/'a-pauli-on-demand-calibrations'/identifier/cell_folder)),
                old_seed_sources=deepcopy(old['seed_checks']['old_sources'] if old else
                                         list(d.D3.OLD_SEEDS)+list(saved_bindings)),
                reuse_only=old is not None, saved_binding_source=old_source,
                implementation=implementation(), implementation_sources=implementation_sources())
    if old and configuration_source is not None and old['configuration']['id'] == config['id']:
        # Same canonical body under another locator; retain original producer for replay.
        spec['config'] = {k:v for k,v in old['configuration'].items() if k != 'id'}
    branch = execution.get('branch') or subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=root).decode().strip()
    if not branch:
        raise ValueError('PUBLISHED_EXECUTION_BRANCH_REQUIRED_FOR_DETACHED_VIEW')
    spec['execution_branch'] = branch
    if domain is not None:
        spec.update(calibration_domain=domain,covered_rounds=sorted(covered_rounds))
    validate_driver_spec(root, spec)
    return spec, item, old_source


def canonical_preparation(config,source=None):
    """Canonical input is already generated by B2; never reconstruct q from a width."""
    return dict(configuration=config,configuration_source=source,
        dictionary_source=config['spec']['split']['tail_dictionary']['source'],
        schedule=[dict(m=r['m'],q_RPE=r['q_RPE'],T=r['T'],q_PF=r['q_PF'],
            actual_delta_pf=r['actual_delta_pf'],R_outer=r['R_outer'],physical_id=r['candidate']['id'],
            structure=r['structure'],mapping=r['mapping'],law_binding=r['law_binding'])for r in config['rounds']])


def prepare(root, preparation, quote, *, execution=None, inspect_saved_cell=False):
    A = _api(); result = dict(schema='A_PAULI_CALIBRATION_PREPARATION_V1', status='NOT_PREPARED',
        driver_request=None,calibration_requests=[],round_actions=[],reused_prices=[], missing_prices=[], needs=[], science_calls=0,
        path_calls=0, compile_calls=0, fit_calls=0, Q3_calls=0, sufficient_N=None, G=None)
    result.update(execution_authorized=False,next_action='RESOLVE_INPUT_OR_IMPLEMENTATION_EVIDENCE')
    try:
        if preparation.get('schema')=='PR_RPE_RMSE_V2':preparation=canonical_preparation(preparation)
        if 'items' in preparation:
            ready = [x for x in preparation['items'] if x.get('configuration')]
            if len(ready) != 1:
                raise ValueError('SELECT_ONE_B2_CANDIDATE_BY_LOCATOR_SELECTOR')
            preparation = ready[0]
        config = preparation['configuration']; A.E.p1.b_api(root).validate(config)
        result['configuration_id'] = config['id']
        reference = A.E.load_configuration(root, A.E.SAVED_PACKETS[0],
                                           configuration_id=A.E.L.SAVED_PAULI_IDS[1])[0]['spec']
        if config['spec']['binding'] != reference['binding']:
            raise ValueError('FIXED_H4_HAMILTONIAN_STATE_SNAPSHOT_BINDING_REQUIRED')
        if config['spec']['profile'] != reference['profile']:
            raise ValueError('SOURCED_COMPILER_PROFILE_IMPLEMENTATION_CONTRACT_REQUIRED')
        if frac(config['spec']['h'])!=frac(reference['h']):
            raise ValueError('FIXED_RPE_H_REQUIRED_NO_TIME_GRID_CHANGE')
        if preparation.get('dictionary_source') != config['spec']['split']['tail_dictionary']['source']:
            raise ValueError('A_DICTIONARY_SOURCE_DIFFERS')
        if preparation.get('configuration_source'):
            if A.E.read_ref(root, preparation['configuration_source']) != config:
                raise ValueError('A_CONFIGURATION_SOURCE_DIFFERS')
        for row in preparation['schedule']:
            r = config['rounds'][row['m']]
            expected = dict(m=r['m'], q_RPE=r['q_RPE'], T=r['T'], q_PF=r['q_PF'],
                actual_delta_pf=r['actual_delta_pf'], R_outer=r['R_outer'], physical_id=r['candidate']['id'],
                structure=r['structure'], mapping=r['mapping'], law_binding=r['law_binding'])
            if row != expected:
                raise ValueError('A_SCHEDULE_DIFFERS_FROM_CANONICAL_CONFIGURATION')
        if [r['m'] for r in preparation['schedule']] != list(range(13)):
            raise ValueError('A_13ROUND_SCHEDULE_REQUIRED')
        supplied = preparation.get('price_handoff', {}).get('request', {}).get('implementation')
        if supplied is not None and supplied != implementation():
            result.update(status='NEEDS_IMPLEMENTATION_CONTRACT',
                          needs=['fixed explicit-operators acquisition recipe does not match requested implementation'])
            return result
        reuse, missing = validate_quote(config, quote)
        result.update(reused_prices=reuse, missing_prices=missing,
                      supplied_D2_response_recomputed=False, prices_complete=not missing)
        if not missing and not inspect_saved_cell:
            result['status'] = 'REUSE_SUPPLIED_PRICES_NO_CALIBRATION'
            result['next_action']='RETURN_SUPPLIED_PRICES_NO_ACQUISITION'
            result['round_actions']=[dict(m=r['m'],configuration_id=config['id'],physical_id=r['candidate']['id'],
                q=r['q_PF'],R=r['R_outer'],T=r['T'],actual_step=r['actual_delta_pf'],
                action='REUSE_SUPPLIED_PRICES',missing_axes=[])for r in config['rounds']]
            return result
        grouped={};missing_rounds={x['m']for x in missing}
        if inspect_saved_cell:missing_rounds=set(range(13))
        for r in config['rounds']:
            axes=[x['axis']for x in missing if x['m']==r['m']]
            action=dict(m=r['m'],configuration_id=config['id'],physical_id=r['candidate']['id'],
                q=r['q_PF'],R=r['R_outer'],T=r['T'],actual_step=r['actual_delta_pf'],missing_axes=axes)
            result['round_actions'].append(action)
            if r['m']not in missing_rounds:
                action['action']='REUSE_SUPPLIED_PRICES';continue
            reasons=scope_reasons(config,[r['m']],multiple_steps=True)
            if reasons:
                action.update(action='NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION',needs=reasons)
                result['needs'].extend(reasons);continue
            grouped.setdefault(frac(r['actual_delta_pf']),[]).append(r['m'])
        for step,needed in grouped.items():
            covered=[r['m']for r in config['rounds']if frac(r['actual_delta_pf'])==step]
            # Preserve the existing .1 whole-schedule request and saved metadata.
            legacy=len(covered)==13 and step==Fraction(.1)
            spec,item,old_source=build_spec(root,config,preparation.get('configuration_source'),execution=execution,
                covered_rounds=None if legacy else covered)
            d=driver(str(Path(root).resolve()));binding=d.metadata(spec)
            identifier=binding['request']['candidate_id']
            group=dict(request_id=identifier,status='REUSE_SAVED_CALIBRATION'if old_source else'CALIBRATION_REQUEST_READY_NOT_EXECUTED',
                next_action='D2_AUDIT_OR_REGISTER_SAVED_CALIBRATION'if old_source else'SELECT_FINITE_CALIBRATION_SCOPE_NOT_EXECUTE',
                execution_authorized=False,
                driver_request=dict(schema='A_PAULI_SPLIT_CAL_REQUEST_V1',driver_spec=spec),driver_metadata=binding,
                saved_calibration_source=old_source,covered_rounds=covered,requested_missing_rounds=needed,
                proposed_applicability=dict(configuration_id=config['id'],formula=config['spec']['formula'],K=2,
                    law=deepcopy(config['spec']['law']),implementation=implementation(),
                    actual_step=exact(step),actual_step_hex=float(step).hex(),rounds=covered,
                    q_values=[config['rounds'][m]['q_PF']for m in covered],R_values=[config['rounds'][m]['R_outer']for m in covered],
                    qualification='SAME_STEP_METADATA_ONLY_CALIBRATION_NOT_MODEL_OR_WRAPPER_QUALIFICATION'))
            result['calibration_requests'].append(group)
            for action in result['round_actions']:
                if action['m']in needed:action.update(action=group['status'],request_id=identifier)
        result.update(implementation=implementation(),implementation_sources=implementation_sources(),
            post_acquisition_needs=['same-step S0-r1 boundary/normal-body decomposition audit',
                'wrapper margins at all requested round T; no other step or L_D coefficients',
                'D2 model registration/applicability and validation; not an E[C] guarantee'],
            requested_wrapper_times=[dict(m=r['m'],T=r['T'])for r in config['rounds']if r['m']in missing_rounds])
        groups=result['calibration_requests']
        if len({g['driver_request']['driver_spec']['base']for g in groups})!=len(groups) or \
                len({g['driver_request']['driver_spec']['private']for g in groups})!=len(groups):
            raise ValueError('DISTINCT_OUTPUT_LOCATIONS_REQUIRED_FOR_STEP_GROUPS')
        if len(groups)==1:
            g=groups[0];result.update(driver_request=g['driver_request'],driver_metadata=g['driver_metadata'],
                saved_calibration_source=g['saved_calibration_source'],
                candidate_implementation_binding=dict(configuration_id=config['id'],dictionary_identity=A.E.L.dictionary_identity(item),
                    profile=config['spec']['profile'],operators=g['driver_metadata']['operators'],
                    qualification='SOURCE_BOUND_RECIPE_PENDING_NUMERICAL_INPUT_PREFLIGHT'))
        result['status']=('NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION'if result['needs']else
            groups[0]['status']if len(groups)==1 else 'CALIBRATION_GROUPS_READY_NOT_EXECUTED')
        result['next_action']=('RESOLVE_ROUND_NEEDS_KEEP_REUSED_PRICES'if result['needs']else
            groups[0]['next_action']if len(groups)==1 else'HANDLE_UNIQUE_CALIBRATION_REQUESTS')
    except (KeyError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        result.update(status='NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE', needs=[type(exc).__name__+': '+str(exc)],
            calibration_requests=[],driver_request=None,next_action='RESOLVE_INPUT_OR_IMPLEMENTATION_EVIDENCE')
    return result


def prepare_many(root,inputs,*,execution=None):
    """Deduplicate identical physical calibration requests across selected candidates.

    No configurations, rounds or price records are merged/relabelled. The one
    executable cell retains its canonical producer; consumers are metadata.
    """
    results=[prepare(root,x.get('preparation',x.get('configuration')),x['quote'],execution=execution)for x in inputs]
    groups={}
    for result in results:
        for group in result['calibration_requests']:
            request=group['driver_metadata']['request'];key=group['request_id']
            consumer=dict(configuration_id=result['configuration_id'],rounds=group['requested_missing_rounds'],
                proposed_applicability=group['proposed_applicability'])
            if key not in groups:groups[key]=dict(deepcopy(group),consumers=[])
            other=groups[key]
            if other['driver_metadata']['request']!=request or other['driver_metadata']['cohort']!=group['driver_metadata']['cohort']:
                raise ValueError('CALIBRATION_CELL_OR_COHORT_CONFLICT_NO_POOLING')
            if other['proposed_applicability']['law']!=group['proposed_applicability']['law']:
                raise ValueError('CALIBRATION_LAW_BINDING_CONFLICT_NO_IMPLICIT_TRANSFER')
            if consumer not in other['consumers']:other['consumers'].append(consumer)
    return dict(schema='A_PAULI_MULTI_CANDIDATE_CALIBRATION_PREPARATION_V1',results=results,
        calibration_requests=list(groups.values()),unique_calibration_requests=len(groups),
        maximum_new_paths=sum(2 for g in groups.values()if not g['saved_calibration_source']),
        maximum_new_compiles=sum(4 for g in groups.values()if not g['saved_calibration_source']),
        science_calls=0,execution_authorized=False)
