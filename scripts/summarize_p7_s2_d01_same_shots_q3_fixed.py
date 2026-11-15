"""Read one saved Q3 call; exact fine-minus-coarse MSE and threshold margins."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_p7_d01_result_caller',ROOT/'scripts/run_p7_s2_d01_same_shots_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
read,display=C.F.D.read,C.F.D.display
exact=C.T.B.a.exact_scalar


def intervals(fine,coarse,tau):
    fl,fu=map(exact,fine);cl,cu=map(exact,coarse);tau=exact(tau)
    if not (0<=fl<=fu and 0<=cl<=cu and tau>0):
        raise ValueError('Invalid model MSE intervals')
    delta=[fl-cu,fu-cl]
    return dict(MSE_difference_fine_minus_coarse=[C.F.exact(v) for v in delta],
        fine_margin=[C.F.exact(tau-fu),C.F.exact(tau-fl)],
        coarse_margin=[C.F.exact(tau-cu),C.F.exact(tau-cl)],
        margin_gain=[C.F.exact(-delta[1]),C.F.exact(-delta[0])],
        margin_change='INCREASED' if delta[1]<0 else 'DECREASED' if delta[0]>0 else 'INTERVALS_OVERLAP')


def summarize():
    folder=ROOT/C.DIRECTORY;inputs=read(folder/'INPUTS.json');run=folder/'run01';case=inputs['cases'][0]
    result=read(run/C.LABEL/'RESULT.json');receipt=read(run/C.LABEL/'PERSISTED.json')
    completion=read(run/'COMPLETION.json');execution=read(run/'EXECUTION.json')
    usage=read(run/C.LABEL/'RESOURCE_FINAL.json')
    coarse=C.T._selected(ROOT,inputs['comparison']['API_source'])[0]
    old=coarse['diagnostic']['deterministic_risk']
    assert coarse['allocation']['shots']==case['allocation']['shots']
    old_assessment=C.F.H.assess(old,inputs['adopted_evaluation'],coarse['source'])
    summary=dict(source=execution['source'],status=result['status'],completion=completion,
        process_return=read(folder/'PROCESS_RETURN.json'),case=case,
        original_API_return_saved=receipt['api_return_saved'],operational_status=receipt['operational_status'],
        scientific_evaluation_calls=result['scientific_evaluation_calls'],resources=usage,
        comparison_source=inputs['comparison'],coarse_model_status=old_assessment['status'],
        coarse_MSE=[old['MSE_lower'],old['MSE_upper']],fine_MSE=None,accuracy_comparison=None,
        fine_conditional_cost=None,sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    assert result['source']==usage['source']==execution['source']
    assert result['scientific_evaluation_calls']<=1
    assert all(result[k] is None for k in ('sufficient_N','global_RMSE_bound','G','G_qualified','winner'))
    if result['status']=='UNKNOWN':
        summary['reason']=result.get('reason')
    else:
        assert read(run/C.LABEL/'API_RETURN.json')==[result['record']]
        risk=result['record']['diagnostic']['deterministic_risk'];dp=risk['bounds']['total_state_dp']
        coarse_gate=read(run/C.LABEL/'PARTIAL_COARSE.json')
        assert C.F.H.assess(risk,inputs['adopted_evaluation'],execution['source'])['status']==result['status']
        assert coarse_gate['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT'
        assert coarse_gate['decoder_certificate']['checked_pairs']<=inputs['options']['maximum_verification_pairs']
        assert sum(exact(v['exact']) for v in coarse_gate['first_unsafe'])+exact(coarse_gate['all_good']['exact'])==1
        assert all(exact(v['good']['exact'])+exact(v['unsafe']['exact'])==1 for v in coarse_gate['rounds'])
        assert exact(dp['retained_MSE_contribution'])+exact(dp['residual_loss_contribution_upper'])==exact(dp['MSE_upper'])
        assert dp['transition_pairs']<=inputs['options']['total_state_refinement']['maximum_transition_pairs']
        assert len(dp['rounds'])==13 and all(v['retained_states']<=8192 for v in dp['rounds'][:-1])
        assert exact(risk['MSE_upper'])==min(exact(v) for v in risk['bounds']['upper_candidates'].values())
        summary.update(fine_MSE=[risk['MSE_lower'],risk['MSE_upper']],
            accuracy_comparison=intervals([risk['MSE_lower'],risk['MSE_upper']],
                [old['MSE_lower'],old['MSE_upper']],inputs['adopted_evaluation']['MSE_upper']),
            fine_moments=risk['moments'],coarse_moments=old['moments'],
            closure=coarse_gate['decoder_certificate'],residual_mass=dp['residual_probability'],
            residual_loss_upper=dp['residual_loss_contribution_upper'],any_zero=dp['any_zero'],
            transition_pairs=dp['transition_pairs'],per_round=dp['rounds'],
            numerical_allowance=dp['transition_allowance'],
            gates=dict(untouched_API=True,exact_mass_and_residual_accounting=True,finite_scope=True,exact_threshold=True))
    # Exact rational comparison is saved separately from the display projection.
    C.R.write_new(folder/'EXACT_COMPARISON.json',summary)
    projected=display(summary)
    C.R.write_new(folder/'SUMMARY.json',projected)
    C.R.write_new(folder/'C_RESULT_JOIN.json',dict(kind='ACCURACY_ONLY_NO_PRICE_OR_N_QUALIFICATION',
        source=execution['source'],configuration_id=case['configuration_id'],total_view_id=case['total_view_id'],
        binding_id=case['binding_id'],signal_packet_source=case['signal_packet'],allocation=case['allocation'],
        physical_rows=case['physical_rows'],model_status=result['status'],conditional_cost=None,
        sufficient_N=None,G=None,G_qualified=None))
    print(json.dumps(dict(status=projected['status'],fine_MSE=projected['fine_MSE'],
        comparison=projected['accuracy_comparison']),ensure_ascii=False,indent=2))


if __name__=='__main__':summarize()
