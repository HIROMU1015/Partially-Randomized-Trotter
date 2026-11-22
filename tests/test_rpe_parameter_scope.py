"""Scope recovery and native horizon dispatch with no scientific calls."""
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import pytest

from trottertracks import pf_horizon_preparation as pf, pr_df_horizon_preparation as df
from trottertracks import rpe_scaled_shots as scaled
from trottertracks.rpe_local0_scope_plan import verified_scope
from trottertracks.rpe_local0_search import RequestError, request_service
from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge
from trottertracks.rpe_release_registry import read, write
from trottertracks.pr_df_descriptor_extension import BINDING_KEYS


def fixture_bridge(tmp_path, settings=None):
    executor = SimpleNamespace(run_dir=tmp_path, state=SimpleNamespace(failure=None),
                               ready_preparations=lambda *args, **kwargs: [])
    return DistributedPreparationBridge(executor, tmp_path, settings, approved=False)

def test_known_candidates_do_not_prove_completion_of_declared_scope():
    tau = Fraction(scaled.EPSILON)**2
    policy = dict(M=[0,2],grid=dict(min_k=0,max_k=0,divisor=1),compare_paper=False)
    method = dict(p_values=[1,3], formulas=['2nd'], delta_over_h_values=['1','3/2'])
    result = scaled.run('TRUNCATED_PF', [], {}, None, {'scaled_shots':policy}, tau, None, method_scope=method)
    assert result['stop_reason'] == 'UNRESOLVED_SCALED_SCOPE'
    requested = result['scaled_shots']['scope_coverage']
    assert (requested['requested'], requested['represented'], requested['missing_count']) == (8,0,8)
    rows = [dict(M=M,configuration_id=f'c{M}',configuration=dict(p=3,formula='2nd',
                 delta_over_h={'numerator':3,'denominator':2})) for M in policy['M']]
    after = scaled.scope_coverage(method,policy['M'],rows + rows)
    assert after['represented'] == 2 and after['missing_count'] == 6
    whitelist = scaled.scope_coverage({'configuration_ids':['a','b']},[2],[
        dict(M=2,configuration_id='extended',source_configuration_id='a',configuration={})])
    assert whitelist['missing_sample'] == [dict(M=2,source_configuration_id='b')]


def test_df_only_and_mixed_plan_respect_independent_splits_and_grids():
    svc = dict(absolute_kappa_route={'entry':'schedule'})
    req = {'scope':{'methods':['PR_DF'],'per_method':{'PR_DF':{'kappa':{'mode':'absolute'}}}}}
    assert request_service(svc,req)['absolute_kappa_schedule']
    applied = dict(method_settings={'PR_PAULI':{'L_D_values':[5]},'PR_DF':{'L_D_values':[7,8]},
                   'FULL_PF':{}},search=dict(schedule_wide_by_method={
                       'PR_PAULI':dict(kappa_bounds=['1','8']),
                       'PR_DF':dict(kappa_bounds=['2','6'],kappa_values=['2','6'])}))
    plan = dict(requested_L_D={'PR_PAULI':[5],'PR_DF':[7,8],'FULL_PF':None},
        kappa_bounds_by_method={'PR_PAULI':['1','8'],'PR_DF':['2','6']},
        kappa_values_by_method={'PR_PAULI':None,'PR_DF':['2','6']})
    families = {str(i):dict(method=name,L_D=i) for i,name in [(5,'PR_PAULI'),(7,'PR_DF'),(8,'PR_DF')]}
    actual = dict(scope_plan=plan,families=families,history=[dict(proposals=[dict(family='7',kappa='6')])])
    assert verified_scope(applied,actual)[1]['PR_DF'] == ['2','6']
    actual['history'][0]['proposals'][0]['kappa']='4'
    with pytest.raises(RequestError):verified_scope(applied,actual)


def test_scoped_pf_completion_collects_new_parents_and_resumes(tmp_path, monkeypatch):
    import trottertracks.pf_distributed_preparation as prices
    from trottertracks.pr_candidate_completion import PreparationWait
    rt = fixture_bridge(tmp_path)
    methods = ['FULL_PF','TRUNCATED_PF']
    candidates = []
    for i, method in enumerate(methods*2):
        name = str(i)
        entry = dict(method=method,configuration={'commit':name},signal_packet={'commit':'sig'+name},prices=name+'p')
        candidates.append(dict(method=method,configuration_id=name,pf_evidence=entry))
    routes = {m:[dict(kind='pf_b_pf6')] for m in methods}
    started, ready = set(), []
    def signal(root, runtime, parent, entry, route, interpreter, stopped, owners):
        if parent not in started:
            started.add(parent)
            raise PreparationWait('owned')
        parent_locs={k:entry[k] for k in ('configuration','signal_packet')}
        return dict(entry=dict(horizon_parent=parent_locs, price_scope='FIXTURE'))
    def register(root,runtime,entries,*,selected,stop_requested,on_ready):
        path = tmp_path/'pf_auto'/'TERMINAL_PRICES.json'
        state = read(path) if path.exists() else {}
        for k in selected:
            state[k] = dict(status='READY',configuration_id=k+'extended',entry=selected[k]['entry'],parent_prices=entries[k]['prices'])
            on_ready(k+'extended',selected[k]['entry'])
        write(path,state)
    monkeypatch.setattr(pf,'produce',signal)
    monkeypatch.setattr(prices,'complete_terminal_prices',register)
    for _ in range(3):
        pf.complete_scoped(tmp_path,rt,candidates,methods,[11,12,13],routes,'python',[],
            stop_requested=lambda:False,on_ready=lambda cid,e:ready.append(cid))
    assert started == set('0123') and len(list((tmp_path/'pf_auto/horizon').glob('*/LEAF.json'))) == 4
    assert ready == [k+'extended' for k in '01230123']
    assert not pf.parents(candidates,methods,[11,12],{}, {})
    assert set(pf.parents(candidates,methods,[13],{'1':{}},{'2':{}})) == {'0','3'}


def test_df_shares_only_a_matching_terminal_reference():
    config = dict(spec=dict(M=12,binding={k:k for k in BINDING_KEYS}),rounds=[{'m':12,'T':'5/2'}])
    import json
    key = json.dumps(config['spec']['binding'],sort_keys=True), Fraction(5)
    refs = {key:[({'commit':'a'}, {'reference':42,'source':'a'}),({'commit':'b'}, {'reference':42,'source':'b'})]}
    assert df.select_reference(config,refs)=={'commit':'a'}
    altered=deepcopy(config);altered['spec']['binding']['state']='other'
    with pytest.raises(ValueError,match='missing'):df.select_reference(altered,refs)
    refs[key][1][1]['reference']=43
    with pytest.raises(ValueError,match='disagree'):df.select_reference(config,refs)


def test_unsupported_horizon_does_not_block_supported_horizon_or_create_work():
    from trottertracks.rpe_method_optimizer import FixtureBackend
    tau = Fraction(scaled.EPSILON)**2
    policy = scaled.settings(dict(M=[0,100000],grid=dict(min_k=0,max_k=0,divisor=1),compare_paper=False))
    parent = dict(method='PR_PAULI',configuration_id='parent',M=0,key=dict(L_D=4,formula='2nd'),
                  name='parent',ready=True,needs=[],labels=[],prices=[1,1],kappa_value='7')
    candidates = scaled.horizon_candidates([parent], {'rpe':{'M':0},'problem':{'system':'H4'}}, policy)
    result = scaled.run('PR_PAULI',candidates,{},FixtureBackend(lambda *_:(0,tau),tau),
                        {'scaled_shots':policy},tau,None,method_scope={'L_D_values':[4]})
    assert result['new_calls'] == 1 and result['input_requests'] == []
    assert [row['minimum_status'] for row in result['by_configuration']] == [
        'GRID_MINIMUM_CONFIRMED','OUTSIDE_QUALIFIED_EVALUATOR_DOMAIN']
    assert result['stop_reason'] == 'SCALED_SCOPE_INCLUDES_UNSUPPORTED_HORIZON'


def test_late_native_replaces_only_its_own_pending_projection():
    from trottertracks.rpe_method_optimizer import FixtureBackend
    tau = Fraction(scaled.EPSILON)**2
    policy = scaled.settings(dict(M=[1],grid=dict(min_k=0,max_k=0,divisor=1),compare_paper=False))
    seed = dict(method='TRUNCATED_PF',M=1,key={'p':3},ready=False,needs=[],prices=None,
                labels=[],required_rounds=[1],horizon_preparation=dict(extension_rounds=[1]))
    waiting = [dict(seed,configuration_id=k+'-projection',source_configuration_id=k,name=k) for k in ('A','B')]
    native = dict(seed,configuration_id='native',source_configuration_id='A',name='native',
                  required_rounds=[],prices=[1]*4,ready=True)
    native.pop('horizon_preparation')
    for order in (waiting+[native], [native]+waiting):
        out = scaled.run('TRUNCATED_PF',order,{},FixtureBackend(lambda *_:(0,tau),tau),
                         {'scaled_shots':policy},tau,None)
        assert out['new_calls'] == 1
        assert {r['configuration_id'] for r in out['by_configuration']} == {'native','B-projection'}
        assert [r['source_configuration_id'] for r in out['input_requests']] == ['B']


def test_index_derives_truncation_from_its_own_fixed_conditions():
    from trottertracks.pf_optimizer_connection import indexed_prefix, scoped_index_row
    reader = SimpleNamespace(read=lambda loc:{'spec':{'L_D':9}})
    assert indexed_prefix(reader, {'method':'B0I_P2'},10) == 2
    assert indexed_prefix(reader, {'method':'TRUNCATED_PF','configuration_locator':{'commit':'fixed'}},10) == 9
    with pytest.raises(ValueError,match='Conflicting'):
        indexed_prefix(reader, {'method':'B0I_P2','p':9},10)
    request = dict(index_method='B0I_P7', p_values=[2, 9])
    assert scoped_index_row('TRUNCATED_PF', request, {'method':'B0I_P2'})
    assert not scoped_index_row('TRUNCATED_PF', request, {'method':'B0I_P4'})
    assert not scoped_index_row('TRUNCATED_PF', dict(index_method='B0I_P7'), {'method':'B0I_P2'})


def test_df_completion_uses_scope_and_recovers_the_owned_parent_only(tmp_path, monkeypatch):
    import json
    from trottertracks.pr_price_rules import GitReader
    candidates = [dict(method='PR_DF', configuration_id=cid, configuration_source={'path':cid},
                       signal_source={'path':'prefix/'+cid}, M=12) for cid in ('owned','fresh','unmatched')]
    bindings = dict.fromkeys(BINDING_KEYS,'same')
    def configuration(self, loc):
        cid = loc['path']
        physical = dict(bindings, state='other') if cid=='unmatched' else bindings
        return dict(id=cid, spec=dict(M=12,binding=physical), rounds=[dict(m=12,T='4')])
    monkeypatch.setattr(GitReader,'read',configuration)
    refs = {(json.dumps(bindings,sort_keys=True),Fraction(8)):[({'commit':'ref'},{'reference':1})]}
    monkeypatch.setattr(df,'references',lambda *_:refs)
    rt = fixture_bridge(tmp_path, {'horizon_extensions':{'bindings':{
        'owned':{'rounds':{'13':{'signal':'existing'}}}}}})
    about = []
    def complete(request, *, existing_signal_only, saved_packets):
        cid=request['selection']['source_configuration_id']
        about.append((cid,existing_signal_only,request['horizon_parent']))
        return dict(key=cid,status='READY_FOR_SEARCH')
    recovered = []
    def recover_signal(record):
        recovered.append(record['key'])
        return dict(record, verified_parent=True)
    completion = SimpleNamespace(complete=complete, recover_horizon_signal=recover_signal)
    number=lambda:len(about)
    for iteration in range(2):
        records=df.prepare(tmp_path,rt,completion,candidates,[13],[],lambda:False)
        assert [r['key'] for r in records] == ['owned','fresh']
        assert all(r['verified_parent'] for r in records)
        assert number() == (iteration+1)*2
    assert about[:2] == about[2:]
    assert [v[:2] for v in about[:2]] == [('owned',True),('fresh',False)]
    assert recovered == ['owned','fresh'] * 2
    assert 'reference' in read(tmp_path/'completion/horizons/unmatched/STATUS.json')['reason']


def test_saved_catalog_parent_can_prepare_signal_before_prices_or_q3():
    c = dict(method='TRUNCATED_PF',configuration_id='parent', configuration_source={'commit':'C'},
             signal_source={'commit':'S'},key=dict(p=2,formula='2nd',delta_over_h='3/2'), M=12)
    selected=pf.parents([c],['TRUNCATED_PF'],[13],{}, {})
    assert selected['parent']['configuration'] == c['configuration_source']
    assert selected['parent']['prices'] is None and selected['parent']['q3_inputs'] is None


def test_completed_df_stage_restores_missing_output_without_launch(tmp_path):
    from trottertracks.pr_candidate_completion import Completion, PreparationWait

    completion = Completion.__new__(Completion)
    output = tmp_path/'result.json'
    marker = tmp_path/'DF_SIGNALS.DISTRIBUTED.json'
    write(marker, {'job_id': 'same-job'})
    write(tmp_path/'DF_SIGNALS.RETURN.json', {'returncode': 0})

    def restore(owner, original, stage):
        assert (owner, original, stage) == (completion, marker, 'DF_SIGNALS')
        write(output, {'status': 'SIGNALS_READY'})
        return True

    completion.stage_executor = SimpleNamespace(import_received_stage=restore)
    assert completion.call([], tmp_path, 'DF_SIGNALS', output, science=True)['status'] == 'SIGNALS_READY'
    with pytest.raises(PreparationWait, match='RESULT_ARTIFACTS_MISSING'):
        completion.call([], tmp_path, 'DF_SIGNALS', tmp_path/'absent.json', science=True)
