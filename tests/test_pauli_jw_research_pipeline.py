"""P2 technical-only mocks. No state/operator/law/RNG/Qiskit execution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import builtins
import inspect
import json
import os
import subprocess
import sys

import pytest

from trottertracks.pauli_jw.contexts import (
    CodeVersion, CompilerProfile, Locator, PauliContext, PipelineError,
    ScientificExecutionDisabled, StageResult,
)
from trottertracks.pauli_jw.adapters import PauliStudyAdapter, H6_CANDIDATES, h2_saved_index, BASE
from trottertracks.pauli_jw.pipeline import compose, write_new_result, STAGES
from trottertracks.pauli_jw.pipeline_fixture import synthetic_context, synthetic_fixture
from trottertracks.pauli_jw import core_stages as stages

ROOT = Path(__file__).resolve().parents[1]
SCOPES = ('h4-hybrid', 'h6-hybrid', 'h6-rte')


@pytest.mark.parametrize('scope', SCOPES)
def test_artificial_scope_end_to_end_uses_existing_wrapper_calls(scope):
    c = synthetic_context(scope); data, p = synthetic_fixture(c)
    result = compose(c, PauliStudyAdapter(scope), mode='synthetic', data=data, providers=p)
    assert result.qualification == 'SYNTHETIC_END_TO_END_PASS'
    assert tuple(x.stage for x in result.stages) == STAGES
    core = p.core
    ham = core.hamiltonian.call_args.kwargs
    assert ham['terms'] is data.terms and ham['constant'] == data.constant
    assert ham['num_qubits'] == c.num_qubits and data.terms[0][1] == -1.0
    assert len(data.terms) == c.term_count
    if scope == 'h4-hybrid':
        core.config.assert_called_once_with(36, 2, 8, 4)
        core.seed.assert_called_once_with(36, 2, 8, 4, 'confirmation', 0)
        assert core.trajectory.call_args.kwargs == {'seed': c.seed_value}
        assert core.mean.call_args.args == ('ARTIFICIAL_LAW', data.raw_state, data.reference, 128)
    elif scope == 'h6-hybrid':
        core.config.assert_called_once_with(c.candidate_id)
        assert core.trajectory.call_args.args[1:] == (c.candidate_id, 'confirmation', 0)
        assert core.mean.call_args.args == (data.model, data.reference, 'ARTIFICIAL_LAW', c.candidate_id)
        core.seed.assert_not_called()  # The existing H6 trajectory owns its seed address.
    else:
        core.config.assert_called_once_with(64, 2)
        assert core.trajectory.call_args.args[1:] == ('H6', 64, 2, 'confirmation', 0)
        assert core.mean.call_args.args == (data.model, data.reference, 'ARTIFICIAL_LAW')
        core.seed.assert_not_called()
    assert core.compile_cost.call_count == 2
    assert [x.args[0] for x in core.compile_cost.call_args_list] == [
        'ARTIFICIAL_COSINE_NO_CIRCUIT', 'ARTIFICIAL_SINE_NO_CIRCUIT']
    assert core.account.call_args.args == ('ARTIFICIAL_TARGET_NO_MEAN',
        {'cosine': None, 'sine': None}, '.05', 'ARTIFICIAL_LAW')
    record = result.to_dict()
    assert not record['science_executed'] and not record['numerical_equivalence_verified']
    assert not record['reproduces_saved_results']
    assert record['stages'][-1]['values']['scientific_qualification'] == 'UNQUALIFIED'
    assert all(record['stages'][-1]['values'][k] is None for k in ('N', 'G', 'price', 'SE', 'rare_probability'))
    encoded = json.dumps(record, allow_nan=False)
    assert 'metadata_json' not in encoded and 'ARTIFICIAL_TARGET_NO_MEAN' not in encoded


@pytest.mark.parametrize('candidate', tuple(H6_CANDIDATES))
def test_all_four_existing_h6_signatures_route_to_same_existing_adapter(candidate):
    c = synthetic_context('h6-hybrid', candidate_id=candidate); data, p = synthetic_fixture(c)
    result = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data, providers=p)
    assert result.qualification == 'SYNTHETIC_END_TO_END_PASS'
    assert (c.prefix, c.q, c.r, c.K) == H6_CANDIDATES[candidate]
    assert p.core.law.call_count == 1
    assert p.core.law.call_args.args[0] is result.stages[0].payload['ham']
    assert p.core.law.call_args.args[1] == candidate


def test_plan_and_context_stop_touch_no_input_or_provider():
    c = synthetic_context(); adapter = PauliStudyAdapter(c.scope)
    adapter.input = Mock(side_effect=AssertionError('input must remain unopened'))
    assert compose(c, adapter).qualification == 'PLAN_ONLY'
    assert compose(replace(c, stop_status='STOP'), adapter, mode='synthetic').qualification == 'STOP'
    adapter.input.assert_not_called()


@pytest.mark.parametrize('mode', ('scientific', 'real', 'resume', 'execute'))
def test_scientific_mode_refused_before_any_adapter_access(mode):
    adapter = Mock()
    with pytest.raises(ScientificExecutionDisabled):
        compose(None, adapter, mode=mode)
    assert not adapter.mock_calls
    with pytest.raises(ScientificExecutionDisabled):
        stages.scientific_providers()


@pytest.mark.parametrize('kind', ('real_input', 'real_core', 'real_provider'))
def test_real_input_or_callback_provider_never_runs(kind):
    c = synthetic_context(); data, p = synthetic_fixture(c)
    if kind == 'real_input': c = replace(c, input_kind='frozen_scientific')
    elif kind == 'real_core': p = replace(p, core=replace(p.core, kind='real_unexecuted'))
    else: p = replace(p, kind='real')
    with pytest.raises(ScientificExecutionDisabled):
        compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data, providers=p)
    p.core.hamiltonian.assert_not_called()


@pytest.mark.parametrize('name', ['prepare', 'event', 'signal', 'account_price'])
def test_missing_callback_refused_before_input(name):
    c=synthetic_context();data,p=synthetic_fixture(c)
    with pytest.raises(PipelineError,match=name):
        compose(c,PauliStudyAdapter(c.scope),mode='synthetic',data=data,
                providers=replace(p,**{name:None}))
    p.core.hamiltonian.assert_not_called()


@pytest.mark.parametrize('scope',['h4-hybrid','h6-hybrid'])
def test_wrong_seed_address_stops_before_mean(scope):
    c=synthetic_context(scope);data,p=synthetic_fixture(c)
    if scope=='h4-hybrid':p.core.seed.return_value=124
    else:
        body,wrapped,_=p.core.trajectory.return_value
        p.core.trajectory.return_value=(body,wrapped,{'seed':124})
    result=compose(c,PauliStudyAdapter(scope),mode='synthetic',data=data,providers=p)
    assert result.qualification=='FAILED'
    p.core.mean.assert_not_called()


@pytest.mark.parametrize('boundary',['price','account','account_exact'])
def test_mock_boundary_cannot_manufacture_real_price_or_N(boundary):
    c=synthetic_context();data,p=synthetic_fixture(c)
    if boundary=='price':p.core.compile_cost.return_value=SimpleNamespace(rz_count=0)
    elif boundary=='account':p.core.account.return_value={'N_total':0,'G':None,'SE':None}
    else:p.core.account.return_value={'N_total':None,'G_exact':'0'}
    result=compose(c,PauliStudyAdapter(c.scope),mode='synthetic',data=data,providers=p)
    assert result.qualification=='FAILED'


@pytest.mark.parametrize('changes', [
    {'source_record': None}, {'scientific_source': 'a'*40}, {'compiler': None},
    {'representation': 'df_native'}, {'cohort_id': ''}, {'input_id': ''},
    {'bit_order': 'big_endian'}, {'term_order_rule': 'reranked'}, {'pf_order': 'K4'},
    {'prefix': 0}, {'q': 1}, {'r': 16}, {'K': 2}, {'method': 'PF'},
    {'effective_law': 'different_law'}, {'cdf_graph': 'different_rng'},
    {'normalization_scope': 'twice'}, {'axes': ('sine', 'cosine')},
    {'state_preparation': 'included'}, {'control': 'whole_body_generic'},
    {'phase_scope': 'drop_constant'}, {'metric': 'cx_count'}, {'estimand': 'phase_rpe'},
    {'stop_status': 'CLEARED'}, {'molecule': 'H2'}, {'cohort_role': 'pooled'},
    {'q': True}, {'precision_bits': 64},
])
def test_incompatible_context_is_rejected_without_science(changes):
    with pytest.raises(PipelineError):
        c = replace(synthetic_context(), **changes)
        compose(c, PauliStudyAdapter('h4-hybrid'))


@pytest.mark.parametrize('field', ['input_id', 'hamiltonian_id', 'term_order_id', 'state_id',
                                   'norm_id', 'reference_id', 'bit_order', 'constant'])
def test_payload_identity_order_and_norm_mismatch_refused(field):
    c = synthetic_context(); data, p = synthetic_fixture(c)
    bad = replace(data, **{field: 9.0 if field == 'constant' else 'OTHER'})
    out = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=bad, providers=p)
    assert out.qualification == 'FAILED'
    p.core.hamiltonian.assert_not_called()


@pytest.mark.parametrize('changes', [{'cohort_role': 'exploration'}, {'cohort_id': 'OTHER'},
    {'input_id': 'OTHER'}, {'state_id': 'OTHER'}, {'norm_id': 'OTHER'}, {'term_order_id': 'OTHER'},
    {'scientific_source': 'b'*40}, {'seed_id': 'OTHER'},
    {'compiler': replace(synthetic_context().compiler, transpiler_seed=99)}])
def test_stage_binding_does_not_mix_scope_source_or_cohorts(changes):
    c = synthetic_context(); data, p = synthetic_fixture(c); other = replace(c, **changes)
    bad = Mock(return_value=StageResult('method_law', other.binding, 'SYNTHETIC'))
    out = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data,
                  providers=replace(p, prepare=bad))
    assert out.qualification == 'FAILED'
    p.core.trajectory.assert_not_called()


@pytest.mark.parametrize('status', ['MISSING', 'FAILED', 'STOP', 'UNQUALIFIED'])
def test_unqualified_missing_failure_and_stop_never_retry(status):
    c = synthetic_context(); data, p = synthetic_fixture(c)
    callback = Mock(return_value=StageResult('method_law', c.binding, status, {'N': None}))
    out = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data,
                  providers=replace(p, prepare=callback))
    assert out.qualification == status and callback.call_count == 1
    p.core.trajectory.assert_not_called(); p.core.mean.assert_not_called()


@pytest.mark.parametrize('value', [0, float('inf'), float('nan'), -1, True])
@pytest.mark.parametrize('name', ['N', 'G', 'rare_probability'])
def test_missing_science_and_unobserved_rare_events_are_not_imputed(name, value):
    c = synthetic_context(); data, p = synthetic_fixture(c)
    out = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data,
        providers=replace(p, prepare=lambda *_: StageResult('method_law', c.binding, 'SYNTHETIC', {name: value})))
    assert out.qualification == 'FAILED'
    p.core.trajectory.assert_not_called()


def test_axes_and_status_cannot_be_relabelled_as_scientific_pass():
    c = synthetic_context()
    for status, values in [('PASS', {}), ('SYNTHETIC', {'axes': ('sine', 'cosine')})]:
        data, p = synthetic_fixture(c)
        out = compose(c, PauliStudyAdapter(c.scope), mode='synthetic', data=data,
            providers=replace(p, prepare=lambda *_: StageResult('method_law', c.binding, status, values)))
        assert out.qualification == 'FAILED'


def _mock_modules():
    return {
        'trotterlib.pauli_signal_adapter': SimpleNamespace(PauliHamiltonian=Mock()),
        'trotterlib.rte': SimpleNamespace(CompilerSettings=Mock()),
        'trotterlib.rte_compiled_cost': SimpleNamespace(transpile_and_measure_cost=Mock()),
        'trotterlib.h4_pauli_pr_search': SimpleNamespace(config=Mock(), law_descriptor=Mock(),
            central_trajectory=Mock(), account=Mock(), trajectory_seed=Mock()),
        'trotterlib.h4_pauli_pr_mean': SimpleNamespace(_mean_enclosure=Mock()),
        'trotterlib.h6_hybrid_fairness_extension': SimpleNamespace(config=Mock(), law_descriptor=Mock(),
            trajectory=Mock(), mean_target=Mock(), account_target=Mock()),
        'trotterlib.pauli_rte_event_extension': SimpleNamespace(config=Mock(), law_descriptor=Mock(),
            trajectory=Mock(), h6_mean=Mock(), account_target=Mock()),
    }


@pytest.mark.parametrize('scope', SCOPES)
def test_real_binding_requests_existing_names_only_and_no_unrelated_science(monkeypatch, scope):
    c = synthetic_context(scope); modules = _mock_modules(); loader = Mock(side_effect=modules.__getitem__)
    monkeypatch.setattr(stages, 'import_module', loader)
    bound = stages.bind_existing_core(PauliStudyAdapter(scope), c)
    assert bound.kind == 'real_unexecuted' and bound.scope == scope
    assert bound.hamiltonian is modules['trotterlib.pauli_signal_adapter'].PauliHamiltonian
    requests = [call.args[0] for call in loader.call_args_list]
    assert not any('df_' in name or 'rpe_' in name for name in requests)
    assert not any(name.startswith('scripts') for name in requests)
    assert all(not fn.called for m in modules.values() for fn in vars(m).values())


def test_optional_dependency_failure_retains_cause_without_fake_fallback(monkeypatch):
    c = synthetic_context(); error = ModuleNotFoundError('absent', name='ARTIFICIAL_OPTIONAL')
    loader = Mock(side_effect=error); monkeypatch.setattr(stages, 'import_module', loader)
    with pytest.raises(PipelineError) as raised:
        stages.bind_existing_core(PauliStudyAdapter(c.scope), c)
    assert raised.value.__cause__ is error and loader.call_count == 1


def test_adapter_declared_missing_callback_is_refused(monkeypatch):
    c=synthetic_context();modules=_mock_modules()
    modules['trotterlib.h4_pauli_pr_search'].account=None
    monkeypatch.setattr(stages,'import_module',Mock(side_effect=modules.__getitem__))
    with pytest.raises(PipelineError,match='callback account'):
        PauliStudyAdapter(c.scope).bind_core(c)


def _signature(path, name):
    tree = ast.parse((ROOT/path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    args = fn.args; pos = args.posonlyargs+args.args
    params = [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None if i >= len(pos)-len(args.defaults) else inspect.Parameter.empty) for i,a in enumerate(pos)]
    params += [inspect.Parameter(a.arg, inspect.Parameter.KEYWORD_ONLY,
        default=None if d is not None else inspect.Parameter.empty) for a,d in zip(args.kwonlyargs,args.kw_defaults)]
    return inspect.Signature(params)


@pytest.mark.parametrize('scope', SCOPES)
def test_wrapper_arguments_bind_actual_fixed_source_signatures(scope):
    c = synthetic_context(scope); data, p = synthetic_fixture(c)
    compose(c, PauliStudyAdapter(scope), mode='synthetic', data=data, providers=p)
    if scope == 'h4-hybrid':
        module='h4_pauli_pr_search'; functions=('config','law_descriptor','central_trajectory','account')
        _signature('src/trotterlib/h4_pauli_pr_mean.py','_mean_enclosure').bind(*p.core.mean.call_args.args)
    elif scope == 'h6-hybrid':
        module='h6_hybrid_fairness_extension'; functions=('config','law_descriptor','trajectory','account_target')
        _signature('src/trotterlib/'+module+'.py','mean_target').bind(*p.core.mean.call_args.args)
    else:
        module='pauli_rte_event_extension'; functions=('config','law_descriptor','trajectory','account_target')
        _signature('src/trotterlib/'+module+'.py','h6_mean').bind(*p.core.mean.call_args.args)
    for fn,mock in zip(functions,(p.core.config,p.core.law,p.core.trajectory,p.core.account)):
        args=mock.call_args
        _signature('src/trotterlib/'+module+'.py',fn).bind(*args.args,**args.kwargs)
    _signature('src/trotterlib/rte_compiled_cost.py','transpile_and_measure_cost').bind(
        *p.core.compile_cost.call_args.args)
    model=ast.parse((ROOT/'src/trotterlib/pauli_signal_adapter.py').read_text())
    cls=next(n for n in model.body if isinstance(n,ast.ClassDef) and n.name=='PauliHamiltonian')
    assert set(p.core.hamiltonian.call_args.kwargs)=={n.target.id for n in cls.body if isinstance(n,ast.AnnAssign)}


def test_old_h6_candidate_set_and_global_initializers_are_unchanged():
    tree=ast.parse((ROOT/'src/trotterlib/h6_hybrid_fairness_extension.py').read_text())
    signatures=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SIGNATURES' for t in n.targets)))
    assert tuple(H6_CANDIDATES.values())==signatures
    for path in ('src/trotterlib/__init__.py','src/trottertracks/__init__.py'):
        p=subprocess.run(['git','show','b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b:'+path],cwd=ROOT,capture_output=True,check=True,
            env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_NO_LAZY_FETCH':'1'})
        assert p.stdout==(ROOT/path).read_bytes()


@pytest.mark.parametrize('scope',SCOPES)
def test_cli_plan_synthetic_records_actual_code_state_and_no_default_output(tmp_path, scope):
    for mode,qualification in [('plan','PLAN_ONLY'),('synthetic','SYNTHETIC_END_TO_END_PASS')]:
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/run_pauli_jw_research_pipeline.py'),
            '--study',scope,'--mode',mode],cwd=tmp_path,capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        record=json.loads(r.stdout)
        assert record['qualification']==qualification and not record['science_executed']
        assert record['context']['code']['state'] in ('CLEAN_CHECKOUT','DIRTY_WORKING_COPY')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mode',['scientific','real','resume'])
def test_cli_refuses_science_before_creating_output(tmp_path, mode):
    out=tmp_path/'forbidden'
    r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/run_pauli_jw_research_pipeline.py'),
        '--mode',mode,'--output-dir',str(out)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2 and not out.exists() and not r.stdout and 'disabled before' in r.stderr


def test_h2_is_a_versioned_index_and_not_a_manufactured_provider(tmp_path):
    entry=h2_saved_index(); assert entry['qualification']=='UNCONNECTED_SOURCE_INDEX'
    r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/run_pauli_jw_research_pipeline.py'),
        '--study','h2-index'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)['source_stop']['role']=='HISTORICAL_STOP'
    with pytest.raises(PipelineError):synthetic_context('h2-index')
    r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/run_pauli_jw_research_pipeline.py'),
        '--study','h2-index','--mode','synthetic'],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2 and not r.stdout


def test_no_site_imports_load_no_legacy_scientific_modules(tmp_path):
    code="import sys; sys.path.insert(0,sys.argv[1]); from trottertracks.pauli_jw import pipeline,contexts,adapters,core_stages,pipeline_fixture; assert not any(n in sys.modules for n in ('trotterlib','numpy','qiskit','scipy','flint','pyscf','openfermion'))"
    r=subprocess.run([sys.executable,'-I','-S','-B','-c',code,str(ROOT/'src')],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_output_dependency_cause_and_no_creation(monkeypatch,tmp_path):
    original=builtins.__import__; missing=ModuleNotFoundError('absent',name='ARTIFICIAL_OPTIONAL')
    def unavailable(name,*a,**kw):
        if name=='trotterlib.parallel_validation_executor':raise missing
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    out=tmp_path/'not_created'
    with pytest.raises(PipelineError) as e:write_new_result(compose(synthetic_context(),PauliStudyAdapter('h4-hybrid')),out)
    assert e.value.__cause__ is missing and not out.exists()


def test_new_directory_collision_with_artificial_writer_preserves_old_output(monkeypatch,tmp_path):
    original=builtins.__import__; writer=Mock(side_effect=lambda p,v:p.write_text(json.dumps(v)))
    def artificial(name,*a,**kw):
        if name=='trotterlib.parallel_validation_executor':return SimpleNamespace(atomic_write_json=writer)
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',artificial)
    result=compose(synthetic_context(),PauliStudyAdapter('h4-hybrid'));out=tmp_path/'new'
    target=write_new_result(result,out);before=target.read_bytes()
    with pytest.raises(FileExistsError):write_new_result(result,out)
    assert target.read_bytes()==before and writer.call_count==1


def test_real_existing_writer_if_normal_dependencies_are_available(tmp_path):
    result=compose(synthetic_context(),PauliStudyAdapter('h4-hybrid'));out=tmp_path/'new'
    try:target=write_new_result(result,out)
    except PipelineError as exc:
        if isinstance(exc.__cause__,ModuleNotFoundError):
            assert not out.exists();pytest.skip('legacy writer dependency unavailable: '+str(exc.__cause__.name))
        raise
    before=target.read_bytes()
    with pytest.raises(FileExistsError):write_new_result(result,out)
    assert target.read_bytes()==before


def test_cli_missing_output_dependency_is_explicit(tmp_path):
    out=tmp_path/'missing'
    r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/run_pauli_jw_research_pipeline.py'),
        '--output-dir',str(out)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==2 and not out.exists() and 'legacy writer dependency unavailable' in r.stderr
    assert 'Traceback' not in r.stderr


@pytest.mark.parametrize('values',[{'method':'PF'},{'K':0},{'PF_order':'Y4'},
    {'prefix':0},{'q':4},{'r':2},{'candidate_id':'OTHER'},{'K':True}])
def test_mislabelled_stage_stops_before_next_callback(values):
    c=synthetic_context();data,p=synthetic_fixture(c)
    def bad(ctx,prev):
        out=stages.prepare(p.core,ctx,prev)
        return replace(out,values={**out.values,**values})
    result=compose(c,PauliStudyAdapter(c.scope),mode='synthetic',data=data,providers=replace(p,prepare=bad))
    assert result.qualification=='FAILED' and len(result.stages)==2
    p.core.trajectory.assert_not_called();p.core.mean.assert_not_called()


@pytest.mark.parametrize('bad',['missing_core','missing_callback'])
def test_missing_core_or_callback_is_explicit_before_input(bad):
    c=synthetic_context();data,p=synthetic_fixture(c)
    core=None if bad=='missing_core' else replace(p.core,law=None)
    with pytest.raises(PipelineError):
        compose(c,PauliStudyAdapter(c.scope),mode='synthetic',data=data,providers=replace(p,core=core))
    p.core.hamiltonian.assert_not_called()


def test_missing_module_attribute_has_explicit_diagnostic(monkeypatch):
    c=synthetic_context();modules=_mock_modules();delattr(modules['trotterlib.h4_pauli_pr_search'],'account')
    monkeypatch.setattr(stages,'import_module',Mock(side_effect=modules.__getitem__))
    with pytest.raises(PipelineError) as error:PauliStudyAdapter(c.scope).bind_core(c)
    assert isinstance(error.value.__cause__,AttributeError)
