"""Only invented inputs and mock full-wrapper prices; no molecular I/O."""
import numpy as np
import pytest
from qiskit import QuantumCircuit

from test_o1_m1_resource_bridge import bridge, rte, prepared, compiler, round_args, accounting, validation


@pytest.fixture(autouse=True)
def no_saved_inputs(monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('No saved input'))


def cost(value, settings):
    return rte.CircuitCost(value,value,0,0,value,value,settings,5,'invented_price')


@pytest.mark.parametrize('dictionary', ['P','DF'])
def test_fresh_stream_paired_axes_means_SE_and_separate_quantum_shots(monkeypatch,dictionary):
    p,c,d=prepared(dictionary,r=2,K=2)
    sequences=[]; prices=(2.,5.,2.,9.)
    def build(preparation,config,distribution,*,q,events):
        assert q==2 and len(events)==4
        bridge._canonical_events(preparation,config,distribution,events,q)
        sequences.append(tuple(events))
        return QuantumCircuit(3)
    calls=[]
    def compile_mock(circuit,settings,**kwargs):
        assert circuit.num_clbits==1 and circuit.data[-1].operation.name=='measure'
        assert kwargs['actual_circuit_fingerprint']
        i=len(calls); calls.append(circuit)
        return cost(prices[i//2]+(1 if i%2 else 0),settings)
    monkeypatch.setattr(bridge,'build_controlled_trajectory',build)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=compile_mock)
    result=provider.monte_carlo(p,c,d,q=2,sample_count=4,seed=37)
    expected=tuple(rte.iter_sample_rte_events(p.tail.components,d,sample_count=16,seed=37))
    assert tuple(e for trajectory in sequences for e in trajectory)==expected
    assert len(calls)==8 and result.classical_sample_count==4
    assert result.cosine_expected_cost.rz_count==pytest.approx(np.mean(prices))
    assert result.sine_expected_cost.rz_count==pytest.approx(np.mean(prices)+1)
    se=float(np.std(prices,ddof=1)/2)
    assert result.cosine_standard_error.rz_count==pytest.approx(se)
    assert result.sine_standard_error.rz_count==pytest.approx(se)
    args=round_args(p,c,m=1)
    record=bridge.account_radius_round(**args,cost=result)
    shots=record['shots']
    assert record['round_total_cost']==pytest.approx(shots['cosine']*np.mean(prices)
        +shots['sine']*(np.mean(prices)+1))
    assert record['classical_price_sample_count']==4
    assert record['price_standard_errors']['cosine']==pytest.approx(se)
    assert not record['price_uncertainty_qualified']
    assert record['guarantee_status']=='not_certified'


def test_MC_reproducible_and_duplicates_retained_without_probability_weights(monkeypatch):
    p,c,d=prepared()
    event=next(rte.iter_rte_events(p.tail.components,d))
    counts=[]
    def repeated(components,distribution,*,sample_count,seed):
        counts.append((sample_count,seed))
        yield from (event for _ in range(sample_count))
    monkeypatch.setattr(rte,'iter_sample_rte_events',repeated)
    built=[]
    def build(*args,**kwargs):
        built.append(kwargs['events']); return QuantumCircuit(3)
    monkeypatch.setattr(bridge,'build_controlled_trajectory',build)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=lambda circuit,settings,**k:cost(7.,settings))
    first=provider.monte_carlo(p,c,d,q=1,sample_count=3,seed=5)
    second=provider.monte_carlo(p,c,d,q=1,sample_count=3,seed=5)
    assert first==second and len(built)==6 and counts==[(3,5),(3,5)]
    assert first.cosine_expected_cost.rz_count==7.
    assert first.cosine_standard_error.rz_count==0.
    assert 'no confidence interval' in dict(first.metadata)['price_uncertainty']


@pytest.mark.parametrize('bad', [{'sample_count':1},{'seed':-1},{'q':0}])
def test_bad_MC_request_rejected_before_rng_or_compile(monkeypatch,bad):
    p,c,d=prepared()
    def forbidden(*a,**k): pytest.fail('Invalid request reached RNG/compiler')
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=forbidden)
    with pytest.raises(ValueError):
        provider.monte_carlo(p,c,d,**({'q':1,'sample_count':4,'seed':1}|bad))


def test_mismatched_dictionary_rejected_before_sampling(monkeypatch):
    p,c,d=prepared(); other,_,_=prepared('DF')
    def forbidden(*a,**k): pytest.fail('Wrong law reached sampler/compiler')
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)
    provider=bridge.MatchedNoPrepCostProvider(compiler(),compile_function=forbidden)
    with pytest.raises(ValueError,match='law'):
        provider.monte_carlo(other,c,d,q=1,sample_count=2,seed=0)


@pytest.mark.parametrize('dictionary',['P','DF'])
def test_existing_signal_record_connects_to_radius_accounting_without_q_extrapolation(dictionary):
    from test_o1_m1_circuit_signals import reference
    p,c,d=prepared(dictionary)
    payload=validation.validate_matched_circuit_signals(p.split,reference,
        step_times=(.13,),r=1,K_values=(0,),q=1)
    assert payload['status']=='PASS'
    record=payload['records'][0]
    spec=accounting.RPERoundSpecification(0,.13)
    margins=dict(radius_allowance=1e-10,complex_signal_allowance=1e-10,phase_allowance=1e-10)
    bounds=bridge.observed_signal_bounds(p,c,spec,record,**margins)
    assert bounds['reference_radius_lower_bound']==pytest.approx(record['PF_signal']['radius']-1e-10)
    assert bounds['corrected_signal_error_bound']==pytest.approx(
        record['dictionaries'][dictionary]['finite_RTE_signal_error_vs_PF']['complex_absolute']+1e-10)
    result=bridge.account_radius_round(**(round_args(p,c)|bounds))
    assert result['guarantee_status']=='not_certified'
    with pytest.raises(ValueError,match='no extrapolation'):
        bridge.observed_signal_bounds(p,c,accounting.RPERoundSpecification(1,.13),record,**margins)
    wrong=dict(record); wrong['PF_signal']={'radius':1e-11}
    with pytest.raises(ValueError,match='radius'):
        bridge.observed_signal_bounds(p,c,spec,wrong,**margins)
