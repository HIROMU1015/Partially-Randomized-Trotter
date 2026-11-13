"""Fixed A menu metadata and mock scalar check only; no H4 numerical leaf."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
from trottertracks import ld12_price_requests as L
from trottertracks import candidate_price_adapter as D

ROOT = Path(__file__).resolve().parents[1]
TECHNICAL_SOURCE = dict(fixture='TECHNICAL_METADATA_ONLY_NO_SCIENCE')
MOCK_CONTEXT_METHOD = D.NativeRuntime.context


def mock_constant(binding):
    return dict(quantity='HAMILTONIAN_CONSTANT', value=.125, source=dict(
        fixture='ARTIFICIAL_CONSTANT_NO_H4_ARRAY_READ', hamiltonian_hash=binding['hamiltonian'],
        snapshot=dict(sha256=binding['snapshot'])))


@pytest.fixture(autouse=True)
def forbid_scientific_entrypoints(monkeypatch):
    def fail(*a, **k):
        raise AssertionError('Metadata tests may not prepare/sample/build/compile/price H4')
    monkeypatch.setattr(D.NativeRuntime, 'context', fail)
    monkeypatch.setattr(D.NativeRuntime, 'sample', fail)
    monkeypatch.setattr(D.Adapter, 'acquire', fail)
    monkeypatch.setattr(L, '_saved_constant', lambda root,binding:mock_constant(binding))


@pytest.fixture(scope='module')
def assets():
    return L._read(ROOT)


@pytest.fixture(scope='module')
def prepared():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(L,'_saved_constant',lambda root,binding:mock_constant(binding))
        return L.prepare(ROOT, execution_source=TECHNICAL_SOURCE)


def test_all_26_rounds_52_axes_ready_but_none_selected(prepared):
    assert prepared['paired_round_requests'] == 26 and prepared['axis_cells'] == 52
    assert len({v['request_id'] for v in prepared['request_index']}) == 26
    assert len({v['context_key'] for v in prepared['request_index']}) == 26
    for label in L.LABELS:
        rows = [c for c in prepared['coverage'] if c['label'] == label]
        assert {(c['m'], c['axis']) for c in rows} == {(m,a) for m in range(13) for a in ('X','Y')}
    assert all(c['mean_C'] is None and c['N'] is None and not c['selected_for_execution'] for c in prepared['coverage'])
    assert all(c['preparation']['hash'] is None for c in prepared['coverage'])
    assert prepared['sampling_build_compile'] == prepared['scientific_preparations'] == 0


def test_canonical_and_physical_ids_law_time_allocation_match(assets, prepared):
    menu, configs = assets
    for cell in prepared['coverage']:
        row = configs[cell['label']]['rounds'][cell['m']]
        assert cell['configuration_id'] == configs[cell['label']]['id']
        assert cell['canonical_candidate_id'] == row['candidate']['id']
        assert cell['physical_id'] == row['physical_candidate']['id']
        assert cell['law_binding_key'] == D._key(row['law_binding'])
        assert cell['structure'] == row['structure']
        assert cell['T_binary64_hex'] == row['physical_candidate']['T'].hex()
        assert cell['signed_piece_times'][0]['time'] == row['physical_candidate']['T']/row['q_PF']
        assert cell['exact_kappa'] == menu['conditions']['kappa_paper_exact']
        assert cell['structure']['event_count'] == cell['q_PF']*cell['R_outer']
        assert cell['measurement']['measurement_axis'] == D.AXES[cell['axis']]


def test_prefix_specific_lambda_scalar_head_tail_and_profile(assets, prepared):
    menu, configs = assets
    for item in menu['candidates']:
        cell = next(c for c in prepared['coverage'] if c['label'] == item['label'])
        prep = cell['preparation']; ld = item['L_D']
        assert prep['deterministic_fragments'] == list(range(ld))
        assert prep['randomized_fragments'] == list(range(ld, 10))
        assert prep['lambda_R'] == configs[item['label']]['spec']['split']['lambda_R']
        assert prep['scalar_binding']['tail_identity']['value'] == item['tail_descriptor']['scalar']
        assert prep['scalar_expected'] == .125 + item['tail_descriptor']['scalar']
        assert cell['price_profile']['lowering_id'] == D.zz.LOWERING_ID
        assert cell['price_profile']['compiler'] == D.p.PROFILE['compiler']
        assert cell['price_profile']['wrapper'] == 'full_Hadamard_with_measurement'
    one, two = (next(c for c in prepared['coverage'] if c['label']==label) for label in L.LABELS)
    assert one['context_key'] != two['context_key']
    assert one['preparation']['lambda_R'] != two['preparation']['lambda_R']
    assert one['preparation']['scalar_expected'] != two['preparation']['scalar_expected']


def test_only_requested_metadata_materializes_and_no_price_relabelling(prepared):
    chosen = L.request_for(ROOT, 'PR-LD2', 12, axes=('Y',), execution_source=TECHNICAL_SOURCE)
    D.check_request(chosen)
    assert chosen['axes'] == ['Y'] and chosen['physical_context']['physical_candidate']['R_outer'] == 2559
    published = next(r for r in prepared['request_index'] if r['label']=='PR-LD2' and r['m']==12)
    # Axis subset changes request ID, while its physical context stays identical.
    assert chosen['request_id'] != published['request_id']
    assert chosen['context_key'] == published['context_key']
    pending = D.pending_price_records(chosen)
    assert len(pending) == 1 and pending[0]['mean']['rz_count'] is None
    assert all(not c['reuse']['accepted'] for c in prepared['coverage'])
    assert all(c['reuse']['structural']['status']=='OLD_LD1_STRUCTURAL_MEAN_NOT_REUSABLE'
        for c in prepared['coverage'] if c['label']=='PR-LD2')


@pytest.mark.parametrize('label,m', [('LD2',0),('PR-LD1',13),('PR-LD2',-1),('PR-LD1',True)])
def test_foreign_selection_rejected(label, m):
    with pytest.raises(ValueError):
        L.request_for(ROOT, label, m, execution_source=TECHNICAL_SOURCE)


def test_foreign_prefix_R_law_lowering_and_equal_event_count_rejected(assets):
    from trottertracks import pr_rpe_rmse_controller as B
    _, configs = assets
    one, two = configs['PR-LD1'], configs['PR-LD2']
    physical = deepcopy(one['rounds'][0]['physical_candidate'])
    with pytest.raises(ValueError):
        D.from_b(two,0,physical,input_source={'fixture':'A'},execution_source=TECHNICAL_SOURCE)
    # A proposed LD2 q1/R2 has the same event count as LD1, but is not A's LD2.
    physical = deepcopy(two['rounds'][0]['physical_candidate']); physical['R_outer']=2
    physical['id']=D.p.candidate_id(physical)
    assert physical['q']*physical['R_outer'] == one['rounds'][0]['structure']['event_count']
    with pytest.raises(ValueError):
        D.from_b(two,0,physical,input_source={'fixture':'A'},execution_source=TECHNICAL_SOURCE)
    changed=deepcopy(two);changed['rounds'][0]['law_binding']['candidate_id']='FOREIGN_LAW'
    with pytest.raises(ValueError): B._validate(changed)
    req=L.request_for(ROOT,'PR-LD1',0,execution_source=TECHNICAL_SOURCE)
    req['physical_context']['price_profile']['lowering_id']='LEGACY'
    with pytest.raises(ValueError): D.check_request(req)


def test_execution_SOURCE_change_cannot_reuse_published_context(prepared):
    one=L.request_for(ROOT,'PR-LD1',0,execution_source={'commit':'new_scientific_source'})
    published=next(r for r in prepared['request_index'] if r['label']=='PR-LD1' and r['m']==0)
    assert one['request_id']!=published['request_id'] and one['context_key']!=published['context_key']


def test_optional_scalar_is_bound_and_checked_before_head_build(assets, monkeypatch):
    # All objects are mock descriptors. No real H4 preparation/library import.
    _,configs=assets;config=configs['PR-LD2'];row=config['rounds'][0]
    scalar=.25
    req=D.from_b(config,0,row['physical_candidate'],input_source={'fixture':'MOCK'},
        execution_source=TECHNICAL_SOURCE,expected_scalar=scalar)
    runtime=D.NativeRuntime.__new__(D.NativeRuntime)
    runtime.binding=deepcopy(config['spec']['binding']);runtime.input_source={'fixture':'MOCK'}
    runtime.execution_source=TECHNICAL_SOURCE;runtime.contexts={};runtime.preparations={}
    runtime.ham=SimpleNamespace(constant=scalar,n_blocks=10)
    lam=float(D.p.accounting.exact_scalar(config['spec']['split']['lambda_R']))
    prep=SimpleNamespace(ld=2,num_system_qubits=8,hamiltonian_hash=runtime.binding['hamiltonian'],
        deterministic_fragment_indices=(0,1),randomized_block_indices=tuple(range(2,10)),
        rte_preparation=SimpleNamespace(symbolic_tail=SimpleNamespace(lambda_r=lam)),
        preparation_hash='MOCK_PREP',extracted_identity_coefficient=0.,deterministic_blocks=())
    tools=SimpleNamespace(h={'prepare':lambda *a:prep},Accounting=None,native=None,compile=None,
        CompilerSettings=None,fingerprint_context=None)
    monkeypatch.setattr(runtime,'_tools',lambda:tools)
    # The saved real method receives only these mock descriptors, not an H4 H.
    assert MOCK_CONTEXT_METHOD(runtime,req).scalar==scalar
    other=D.from_b(config,0,row['physical_candidate'],input_source={'fixture':'MOCK'},
        execution_source=TECHNICAL_SOURCE,expected_scalar=scalar+.1)
    assert other['context_key']!=req['context_key']
    with pytest.raises(ValueError,match='scalar'):MOCK_CONTEXT_METHOD(runtime,other)


def test_metadata_import_is_passive():
    code="import sys; import trottertracks.ld12_price_requests; assert not any(x in sys.modules for x in ('numpy','qiskit','pyscf'))"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
