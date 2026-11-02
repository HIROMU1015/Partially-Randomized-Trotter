"""Saved-data and draft schema tests; never performs molecular science."""
import json
from pathlib import Path

import pytest

from trotterlib.non_df_h4_readiness import h4_contract_draft,rare_event_acquisition_draft,structural_prefix
from trotterlib.saved_rte_cost_diagnostics import diagnose

ROOT=Path(__file__).resolve().parents[1]


def test_draft_concretizes_all_fixed_inputs_without_authorization():
    d=h4_contract_draft()
    assert not d["execution_authorized"] and not d["dispatch_enabled"] and not d["grant_valid"]
    assert d["system"]["geometry_Angstrom"][-1]==["H",[0.,0.,3.]]
    assert d["state"]["dimension"]==36
    assert d["DF"]["rank"] is None and d["Pauli"]["nonidentity_term_count"] is None
    assert d["evolution"]['q']==8 and d["evolution"]["r"]==4
    assert d["MC"]["trajectories_per_PR"]==256 and not d["MC"]["model_or_estimator_adopted"]
    assert d["execution"]["watch_policy"]["run_timeout"] is None
    assert d["numerical"]["compiled_probability_agreement"]==1e-9
    assert not d["readiness"]["H4_inputs_generated"]
    assert d["execution"]["approved_future_call_budget"]["fixed_cases"]==4
    json.dumps(d,allow_nan=False)


def test_fresh_draft_values_and_prefix_rule_are_not_H2_counts():
    x=h4_contract_draft();x["evolution"]["q"]=1
    assert h4_contract_draft()["evolution"]["q"]==8
    assert structural_prefix([9,1,1,1])==1
    assert structural_prefix([2,2,2,2,2,2])==3


@pytest.mark.parametrize("weights",[[],[1],[0,0],[1,-1],[1,2],[float('nan'),1],[1,0]])
def test_bad_or_degenerate_prefix_does_not_change_proposed_family(weights):
    with pytest.raises(ValueError):structural_prefix(weights)


def test_conditional_rare_cost_information_proposal_is_explicitly_not_population_estimator():
    d=rare_event_acquisition_draft()
    assert not d["execution_authorized"] and not d["scientific_definition_adopted"]
    assert d["proposed_diagnostic_grid"]["total_trajectories"]==256
    assert d["new_sampling_now"]==d["new_compile_now"]==0
    assert any("not a marginal" in x for x in d["limitations"])


def test_saved_diagnosis_does_not_sample_replay_or_transpile(monkeypatch):
    import numpy as np
    import qiskit
    import trotterlib.rte as rte
    import trotterlib.h2_non_df_pilot as pilot
    import trotterlib.fixed_signal_evaluation as evaluation

    def forbidden(*args,**kwargs): raise AssertionError("FORBIDDEN_NEW_SCIENCE_OR_RNG_REPLAY")
    for obj,names in [(np.random,["default_rng","PCG64"]),(qiskit,["transpile"]),
                      (rte,["sample_rte_events","iter_sample_rte_events"]),
                      (pilot,["build_h2_inputs","evaluate_case"]),(evaluation,["evaluate_fixed_signal"])]:
        for name in names:monkeypatch.setattr(obj,name,forbidden)
    d=diagnose(ROOT)
    assert d["new_sampling"]==d["RNG_replay"]==d["new_compile"]==0
    p,df=d["reports"]
    assert [x["observed"] for x in p["order_rows"]]==[510,2,0]
    assert [x["observed"] for x in df["order_rows"]]==[512,0,0]
    for case in d["reports"]:
        assert len(case["samples"])==64 and case["paired_axes"]
        assert not case["per_trajectory_event_log_available"]
    assert all(row["Taylor_orders"] is None for row in p["samples"])
    assert p["trajectory_order_cost_link_status"]=="NOT_IDENTIFIABLE_FROM_SAVED_SCHEMA"
    assert all(row["Taylor_orders"]==[0]*8 for row in df["samples"])
    assert df["trajectory_order_cost_link_status"]=="ORDERS_ALL_ZERO_FROM_AGGREGATE_COMPONENT_LOG_MISSING"
