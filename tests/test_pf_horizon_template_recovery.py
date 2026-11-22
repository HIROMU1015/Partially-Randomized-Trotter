from copy import deepcopy
from hashlib import sha256
import sqlite3
from types import SimpleNamespace

import pytest

from trottertracks import pf_horizon_preparation as pf
from trottertracks import pf_optimizer_connection as pf_routes
from trottertracks.rpe_preparation_queue import normalize_preparation_request
from trottertracks.rpe_q3_coordinator import digest
from trottertracks.rpe_release_registry import write


def locator(name, **fields):
    return dict(commit='saved', path=name, **fields)


def saved_inputs():
    binding = dict(system='H4', snapshot='physical-input')
    config = dict(id='parent', spec=dict(M=12, binding=binding, law=dict(model='deterministic')),
                  rounds=[{'m': m} for m in range(13)])
    target = dict(binding=binding, source=locator('target'), reference_solver_energy_ha=-2)
    packet = dict(configuration_id='parent', target=target, means=config['rounds'])
    evaluation = dict(binding=binding, original_target_source=locator('target', selector=[]), energy_ha=-2)
    documents = dict(configuration=config, signal=packet, signal_copy=deepcopy(packet))
    parent = dict(configuration=locator('configuration'), signal_packet=locator('signal'))
    old = dict(parent, original_configuration_id='parent', evaluation_target=evaluation, label='A_SHAPE',
               allocation={'shots': [5]}, binding_id='binding', total_view_id='total')
    alternate = dict(old, configuration_id='parent', label='B_PRICE', allocation={'shots': [9]})
    alternate['configuration'] = locator('configuration', selector=[])
    alternate['signal_packet'] = locator('signal_copy')
    template = dict(kernel_source='kernel', C_source='reference', acceleration={},
                    statistical_contract={'law': 'deterministic'}, adopted_evaluation={'MSE': 'fixed'},
                    options={'phase_radii': [1.0]*13}, cases=[old, alternate])
    return documents, parent, template, evaluation


def test_historical_allocations_share_the_same_fixed_pf_evaluator():
    documents, parent, template, evaluation = saved_inputs()
    documents['template'] = template
    reader = SimpleNamespace(read=lambda loc: deepcopy(documents[loc['path']]))
    entry = dict(parent, q3_inputs=locator('template'))
    contract, target = pf.parent_q3_contract(reader, {}, entry, documents['configuration'])
    assert contract['options']['phase_radii'] == [1.0] * 13
    assert target == evaluation

    documents['signal_copy']['means'][0]['m'] = True
    with pytest.raises(ValueError, match='different fixed candidate evidence'):
        pf.parent_q3_contract(reader, {}, entry, documents['configuration'])


def test_canonical_pf_evaluator_requires_the_same_fixed_target(monkeypatch, tmp_path):
    import run_rpe_optimizer_q3_job as adapter

    documents, parent, template, evaluation = saved_inputs()
    for case in template['cases']:
        case.pop('original_configuration_id')
        case['configuration_id'] = 'another-case'
    reader = SimpleNamespace(read=lambda loc: deepcopy(documents[loc['path']]))
    monkeypatch.setattr(adapter, 'git_json', lambda view, loc: deepcopy(template))
    route = dict(view=str(tmp_path), templates=[dict(zip(('commit', 'path'), pf.PF_Q3_CONTRACT))])
    _, target = pf.parent_q3_contract(reader, route, parent, documents['configuration'])
    assert target == evaluation

    documents['signal']['target']['reference_solver_energy_ha'] = -3
    with pytest.raises(ValueError, match='evaluator target differs'):
        pf.parent_q3_contract(reader, route, parent, documents['configuration'])


def test_q3_template_block_preserves_collected_signal(monkeypatch, tmp_path):
    from run_rpe_pf_horizon_signal import PROVIDER

    documents, parent, _, _ = saved_inputs()
    config = documents['configuration']
    child = dict(config, id='child', spec=dict(config['spec'], M=13))
    entry = dict(parent, method='FULL_PF', p=10, formula='2nd', delta_over_h='1')
    root = tmp_path/'code'
    run = tmp_path/'run'
    base = run/'pf_auto'/'horizon'
    directory = base/'signal-provider'/'horizon-inputs'/'parent'
    request = dict(parent, physical_id='terminal')
    plan = dict(configuration=child, physical_round={'id': 'terminal'}, requested_rounds=[13])
    write(base/'parent'/'plan'/'PLAN.json', dict(plan, parent=parent))
    (base/'parent'/'evidence'/'.git').mkdir(parents=True)
    monkeypatch.setattr(pf, 'GitReader', lambda root: SimpleNamespace(
        read=lambda loc: deepcopy(documents[loc['path']])))
    providers = ('scripts/prepare_pf_candidate.py', 'src/trottertracks/pf_evidence_preparation.py')
    for name in providers:
        write(base/'signal-provider'/name, {'fixed': PROVIDER})
    monkeypatch.setattr(pf, 'fixed_workspace', lambda *args: None)
    monkeypatch.setattr(pf, 'publish', lambda workspace, root, path: locator(path.name))
    monkeypatch.setattr(pf_routes, 'physical_ids', lambda root, cfg: {13: 'terminal'})
    calls = []

    def collect(context, command, work, stage, **kwargs):
        calls.append(stage)
        write(directory/'result'/'PLAN.json', dict(plan, parent=request))
        means = [dict(m=m, physical_candidate={'id': 'terminal'}) for m in range(14)]
        write(directory/'result'/'MEAN_HANDOFF.json',
              dict(configuration_id='child', means=means, prefix_source=parent['signal_packet']))
        write(directory/'result'/'REFERENCE.json', {})

    runtime = SimpleNamespace(executor=SimpleNamespace(run_dir=run, log=lambda **_: None), call=collect)

    def blocked(*args):
        raise ValueError('fixed Q3 template unavailable')

    monkeypatch.setattr(pf, 'bind_terminal_q3', blocked)
    with pytest.raises(ValueError, match='fixed Q3 template unavailable'):
        pf.produce(root, runtime, 'parent', entry, {}, 'python', lambda: False)
    checkpoint = pf.read(base/'parent'/'LEAF.json')
    assert checkpoint['signal_only'] and checkpoint['entry']['signal_packet'] == locator('MEAN_HANDOFF.json')

    monkeypatch.setattr(pf, 'bind_terminal_q3',
                        lambda root, parent, entry, routes, interpreter, leaf, *args: dict(leaf, signal_only=False))
    assert not pf.produce(root, runtime, 'parent', entry, {}, 'python', lambda: False)['signal_only']
    assert calls == ['PF_SIGNALS']


def test_collected_parent_signal_is_usable_while_price_is_pending(monkeypatch, tmp_path):
    documents, parent, _, _ = saved_inputs()
    cid = 'parent'
    run = tmp_path/'run'
    relative = 'docs/research/distributed_selected_pf/'+cid
    folder = run/'pf_auto'/'candidates'/cid/'pf_auto/pf_completion/provider_workspace'/relative
    plan = locator(relative+'/PLAN.json')
    packet = dict(commit='signal-publication', path=relative+'/signal_run01/MEAN_HANDOFF.json')
    arguments = folder/'PF_SIGNALS.INPUT.json'
    result = folder/'PF_SIGNALS.RESULT.json'
    write(arguments, {'source': {'input_plan': plan}})
    write(result, dict(status='SIGNALS_READY', publication=packet['commit']))
    source = dict(commit='provider', files={})
    request = normalize_preparation_request(dict(route=dict(kind='preparation'),
        preparation=dict(stage='signal', provider='PF_SIGNALS', source=source, inputs={
            f'workspace/{relative}/{arguments.name}': sha256(arguments.read_bytes()).hexdigest()})))
    job = request['request_id']
    write(folder/'PF_SIGNALS.DISTRIBUTED.json', dict(job_id=job, request=request))
    original = run/'jobs'/'PREPARATION:signal'/job
    write(original/'OUTCOME.json', dict(request_id=job, request_sha256=digest(request),
        status='PREPARATION_COMPLETE', source_manifest=source,
        artifacts={f'workspace/{relative}/{result.name}': sha256(result.read_bytes()).hexdigest()}))
    write(original/'PROCESS_RETURN.json', dict(returncode=0))
    db = run/'distributed'/'QUEUE.sqlite3'
    db.parent.mkdir()
    with sqlite3.connect(db) as db_conn:
        db_conn.execute('CREATE TABLE jobs (job_id, method, state, attempt_id, request_sha256)')
        db_conn.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?)',
                        (job, 'PREPARATION:signal', 'ACCEPTED', 'original-attempt', digest(request)))
    monkeypatch.setattr(pf, 'GitReader', lambda _: SimpleNamespace(read=lambda loc:
        deepcopy({plan['path']: documents['configuration'], packet['path']: documents['signal']}[loc['path']])))
    candidate = dict(method='FULL_PF', configuration_id=cid, configuration_body=documents['configuration'],
                     key=dict(p=10, formula='2nd', delta_over_h='1'))
    admitted = pf.saved_parent_signals(tmp_path, run, [candidate], ('FULL_PF',))
    assert admitted[cid]['signal_packet'] == packet
    assert not admitted[cid].get('prices')
    assert pf.parents([candidate], ('FULL_PF',), [13], {}, {}, admitted) == admitted
    assert not pf.parents([candidate], ('FULL_PF',), [12], {}, {}, admitted)

    result.write_text('{}')
    assert not pf.saved_parent_signals(tmp_path, run, [candidate], ('FULL_PF',))
    write(result, dict(status='SIGNALS_READY', publication=packet['commit']))
    with sqlite3.connect(db) as db_conn:
        db_conn.execute("UPDATE jobs SET state='UNKNOWN'")
    assert not pf.saved_parent_signals(tmp_path, run, [candidate], ('FULL_PF',))


def test_scoped_producer_hands_off_signal_before_prices_and_resumes(monkeypatch, tmp_path):
    from trottertracks import pf_distributed_preparation

    _, parent, _, _ = saved_inputs()
    entry = dict(parent, method='FULL_PF', p=10, formula='2nd', delta_over_h='1')
    candidate = dict(method='FULL_PF', configuration_id='parent')
    produced, registered = [], []

    def produce(root, runtime, parent_id, request, *args, bind_q3):
        assert parent_id == 'parent'
        produced.append(bind_q3)
        return dict(entry=dict(horizon_parent=parent, price_scope='price-scope'), signal_only=not bind_q3)

    def complete(root, runtime, parents, *, selected, stop_requested, on_ready):
        if selected:
            registered.extend(parents)

    monkeypatch.setattr(pf, 'saved_parent_signals', lambda *args: {'parent': entry})
    monkeypatch.setattr(pf, 'produce', produce)
    monkeypatch.setattr(pf_distributed_preparation, 'complete_terminal_prices', complete)
    runtime = SimpleNamespace(executor=SimpleNamespace(run_dir=tmp_path), settings={},
                              follow_preparations=lambda items, advance, _: [advance(item) for item in items])
    options = dict(stop_requested=lambda: False, on_ready=lambda *_: None)

    pf.complete_scoped(tmp_path, runtime, [candidate], ('FULL_PF',), [13], {}, 'python', (), **options)
    assert produced == [False]
    assert registered == []
    leaf = tmp_path/'pf_auto'/'horizon'/'parent'
    assert pf.read(leaf/'PREPARATION.json')['status'] == 'WAITING_FOR_PARENT_PRICES'
    assert pf.read(leaf/'LEAF.json')['signal_only'] is True

    candidate['pf_evidence'] = dict(entry, prices=locator('prices'))
    pf.complete_scoped(tmp_path, runtime, [candidate], ('FULL_PF',), [13], {}, 'python', (), **options)
    assert produced == [False, True]
    assert registered == ['parent']
    assert pf.read(leaf/'LEAF.json')['signal_only'] is False


def test_new_parent_evidence_yields_watcher_without_stopping_preparation(monkeypatch, tmp_path):
    from trottertracks import pf_distributed_preparation

    checks = []
    monkeypatch.setattr(pf_distributed_preparation, 'complete_terminal_prices',
                        lambda *args, stop_requested, **kwargs: checks.append(stop_requested()))
    runtime = SimpleNamespace(settings={}, executor=SimpleNamespace(run_dir=tmp_path),
        follow_preparations=lambda items, advance, yield_requested: checks.append(yield_requested()))
    pf.complete_scoped(tmp_path, runtime, [], ('FULL_PF',), (13,), {}, 'python', (),
                       stop_requested=lambda: False, on_ready=lambda *_: None,
                       refresh_requested=lambda: True)
    assert checks == [False, True]
