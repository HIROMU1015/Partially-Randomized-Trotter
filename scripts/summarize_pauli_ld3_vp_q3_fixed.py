"""Read saved LD3 API once and quote the saved LD2 VP; no scientific kernel."""
import importlib.util
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_ld3_summary_caller', ROOT/'scripts/run_pauli_ld3_vp_q3_fixed.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)


def exact(value):
    return C.T.Q.a.exact_scalar(value)


def display(value):
    q = exact(value)
    with localcontext() as ctx:
        ctx.prec = 28
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def read(path):
    return json.loads(path.read_text())


def main():
    p = ROOT/C.DIRECTORY; inputs = read(p/'INPUTS.json'); case = inputs['cases'][0]
    run = p/'run01'; folder = run/C.LABEL; result = read(folder/'RESULT.json')
    source = read(run/'EXECUTION.json')['source']; supervision = read(run/'SUPERVISION.json')
    result_source = read(p/'RESULT_SOURCE.json')
    if C.R.blob(result_source['commit'], result_source['path']) != (folder/'API_RETURN.json').read_bytes():
        raise ValueError('Published result SOURCE differs from saved API')
    receipt = read(folder/'PERSISTED.json'); usage = read(folder/'RESOURCE_FINAL.json')
    returned = read(p/'PROCESS_RETURN.json')
    gates = dict(process_return=returned['returncode'] == 0,
        recorded_DP_children_exit=supervision['process_exit_gate']['all_exit_codes_zero'],
        parent_monitor=not supervision['errors'] and not supervision['monitor_errors'],
        source=result['source'] == source == usage['source'],
        one_science_call=result['scientific_evaluation_calls'] == 1,
        worker_persistence=receipt['api_return_saved'] and receipt['resource_saved'],
        unqualified_outputs=all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner')))
    rows = []
    previous = C.document(inputs['comparison']['commit'], inputs['comparison']['path'])
    for label, value, src in [('LD2', previous, inputs['comparison']),
                             ('LD3', result, result_source)]:
        risk = value.get('record', {}).get('diagnostic', {}).get('deterministic_risk')
        if label == 'LD3' and risk is not None and result['status'] != 'UNKNOWN':
            api = read(folder/'API_RETURN.json'); coarse = read(folder/'PARTIAL_COARSE.json')
            dp = risk['bounds']['total_state_dp']; moments = risk['moments']
            gates.update(api_unmodified=api == [result['record']],
                closure=coarse['decoder_certificate']['status'] == 'VERIFIED_FINITE_BINARY64_SUPPORT',
                coarse_mass=sum(exact(v['exact']) for v in coarse['first_unsafe'])+exact(coarse['all_good']['exact']) == 1,
                round_mass=all(exact(v['good']['exact'])+exact(v['unsafe']['exact']) == 1 for v in coarse['rounds']),
                bounds=0 <= exact(risk['MSE_lower']) <= exact(risk['MSE_upper']),
                residual_loss=exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper']) == exact(dp['MSE_upper']),
                minimum_valid_upper=exact(risk['MSE_upper']) == min(exact(v) for v in risk['bounds']['upper_candidates'].values()),
                residual_probability=0 <= exact(dp['residual_probability']) <= 1,
                finite_transitions=dp['transition_pairs'] <= inputs['options']['total_state_refinement']['maximum_transition_pairs'],
                nonterminal_frontier=all(v['retained_states'] <= 8192 for v in dp['rounds'][:-1]),
                finite_verification=coarse['decoder_certificate']['checked_pairs'] <= inputs['options']['maximum_verification_pairs'],
                moments_lower=exact(moments['bias_squared'][0])+exact(moments['variance'][0]) <= exact(risk['MSE_upper']),
                moments_upper=exact(moments['bias_squared'][1])+exact(moments['variance'][1]) >= exact(risk['MSE_lower']))
            diagnostic = dict(closure=coarse['decoder_certificate'], moments=moments,
                residual_mass=dp['residual_probability'], residual_loss_upper=dp['residual_loss_contribution_upper'],
                retained_MSE_contribution=dp['retained_MSE_contribution'], any_zero=dp['any_zero'],
                per_round=dp['rounds'], transition_pairs=dp['transition_pairs'],
                transition_allowance=dp['transition_allowance'], upper_candidates=risk['bounds']['upper_candidates'])
        if risk is None or value['status'] == 'UNKNOWN':
            rows.append(dict(candidate=label,status='UNKNOWN',reason=value.get('reason'),source=src,
                             MSE_lower=None,MSE_upper=None)); continue
        assessment = C.H.assess(risk, inputs['adopted_evaluation'], source)
        if assessment['status'] != value['status']: raise ValueError('Saved assessment inconsistent')
        rows.append(dict(candidate=label, configuration_id=value['record']['configuration_id'],
            shots=deep_shots(value['record'], value, inputs), source=src, status=assessment['status'],
            MSE_lower=risk['MSE_lower'], MSE_upper=risk['MSE_upper'],
            MSE_lower_display_Ha2=display(risk['MSE_lower']), MSE_upper_display_Ha2=display(risk['MSE_upper'])))
    status = result['status'] if all(gates.values()) else 'UNKNOWN'
    summary = dict(status=status,source=source,comparison=rows,gates=gates,
        numerical_classification_separate_from_persistence=True, persistence=receipt,
        observation_scope=dict(recorded_OS_exits='104_DP_CHILDREN_ONLY',
            top_worker='PERSISTED_SUCCESSFUL_FUTURE_RETURN_NO_SEPARATE_OS_EXIT_RECORD',
            detailed_segment_hook='NOT_APPLIED_IN_RUN01_MODULE_ALIAS_FIXED_AFTER_COMPLETION_SYNTHETICALLY_ONLY'),
        resources=dict(**usage,sampled_process_tree_peak_RSS_KiB=supervision['sampled_process_tree_peak_RSS_KiB'][C.LABEL],
                       child_exit_count=supervision['process_exit_gate']['recorded_children']),
        exact_threshold=inputs['adopted_evaluation']['MSE_upper'],
        exact_threshold_display_Ha2=display(inputs['adopted_evaluation']['MSE_upper']),
        diagnostics=locals().get('diagnostic'), scientific_calls=result['scientific_evaluation_calls'],
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    C.R.write_new(p/'SUMMARY.json',summary)
    binding = read(p/'TOTAL_BINDING.json'); original = binding['original_configuration']; total = binding['total_configuration']
    packet = read(p/'TOTAL_SIGNAL_VIEW.json'); index=[]
    for old, new, mean, shot in zip(original['rounds'], total['rounds'], packet['means'],case['allocation']['shots']):
        for axis in ('X','Y'):
            index.append(dict(m=old['m'],axis=axis,N=shot['N_'+axis],allocation_id=case['allocation']['allocation_id'],
                original_configuration_id=original['id'],total_configuration_id=total['id'],binding_id=binding['id'],
                original_candidate_id=old['candidate']['id'],total_candidate_id=new['candidate']['id'],
                original_law_binding=old['law_binding'],total_law_binding=new['law_binding'],
                original_physical_context=mean['canonical_origin']['provider_binding']['physical_context'],
                signal_source=mean['source'],implementation_source=mean['implementation_source'],
                dictionary_identity=packet['dictionary_identity'],result_source=rows[-1]['source'],
                result_record_selector=[0],price=None,G=None))
    C.R.write_new(p/'C_RESULT_JOIN.json',dict(rows=index,source=source,price_evaluations=0,G=None,G_qualified=None))
    print('classification',status,'gates',gates)
    for row in rows: print(row['candidate'],row['status'],row.get('MSE_lower_display_Ha2'),row.get('MSE_upper_display_Ha2'))


def deep_shots(record, value, inputs):
    shots = record['allocation']['shots']
    if shots != inputs['cases'][0]['allocation']['shots']:
        raise ValueError('Comparison result is not the fixed common VP')
    return shots


if __name__ == '__main__': main()
