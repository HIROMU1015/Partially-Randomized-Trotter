from copy import deepcopy
from types import SimpleNamespace

import pytest

from trottertracks import pf_horizon_preparation as pf
from trottertracks import pf_optimizer_connection as pf_routes
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
