"""Only synthetic matrices, fake decomposition/solver/ports, dummy processes.

No molecular I/O, random trajectories, Qiskit circuit construction or compile.
These checks do not add H4/H6 scientific evidence or numerical certification.
"""
import builtins
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trottertracks.resource_applicability.ax2b_numerical_accounting import u_aware_shots, propagate_allowance
from trottertracks.resource_applicability.ax1b_evaluation import reference_shots
from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix, reference_signal_mp
from trottertracks.resource_applicability.ax2b_h6_input import tol_only_df_from_integrals
from trottertracks.resource_applicability.ax2b_h6_contract import preparation_plan, reject_scientific_launch, primitive_time_schedule
from trottertracks.resource_applicability.ax2b_h6_controller import (BoundedWriter, bounded_matvec,
    bounded_sector_matrix, bounded_solver, finite_scale_guard, exercise_synthetic_controller)
from trottertracks.resource_applicability.ax2b_h6_watchdog import supervise_synthetic
from trottertracks.resource_applicability.ax2b_limits import CallBudget


@pytest.fixture(autouse=True)
def no_scientific_access(monkeypatch):
    original, original_path = builtins.open, Path.open
    def check(path):
        if isinstance(path,(str,Path)) and (str(path).endswith(".npz") or "/artifacts/" in str(path) or "/.runtime/" in str(path)):
            raise AssertionError("SCIENTIFIC_ARTIFACT_IO_FORBIDDEN")
    def open_(path,*args,**kwargs):
        check(path); return original(path,*args,**kwargs)
    def path_open(path,*args,**kwargs):
        check(path); return original_path(path,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError("MOLECULE_SAMPLING_OR_CIRCUIT_FORBIDDEN")
    monkeypatch.setattr(builtins,"open",open_)
    monkeypatch.setattr(Path,"open",path_open)
    monkeypatch.setattr(np,"load",forbidden)
    import qiskit
    import trotterlib.rte as rte
    import trotterlib.df_hamiltonian as df
    monkeypatch.setattr(qiskit.QuantumCircuit,"__init__",forbidden)
    monkeypatch.setattr(qiskit,"transpile",forbidden)
    for name in ("sample_rte_events","iter_sample_rte_events","sample_event_mean_operator"):
        monkeypatch.setattr(rte,name,forbidden)
    monkeypatch.setattr(df,"build_df_h_d_from_molecule",forbidden)
    monkeypatch.setattr(df,"low_rank_two_body_decomposition",forbidden)


@pytest.mark.parametrize("B",[1.,2.,100.])
@pytest.mark.parametrize("epsilon",[.005,.05,.1])
def test_u_zero_legacy_integer_reproduction(B,epsilon):
    bias={"real":epsilon*.1,"imag":epsilon*.2}
    log_B=math.log(B)
    old=reference_shots(bias,B,epsilon)
    new=u_aware_shots(bias,dict.fromkeys(bias,0.),epsilon=epsilon,log_B_upper=log_B,
                      evidence_kind="EMPIRICAL",evidence_ref="synthetic",normalization_upper=B)
    assert {a:r["shots"] for a,r in new["axes"].items()}==old["axis_shots"]
    assert new["N_total"]==old["N_total"]
    assert not new["certificate_verified_by_this_function"]


@pytest.mark.parametrize("bias,u,status",[(0.,0.,"ELIGIBLE"),(.1,0.,"INELIGIBLE"),(.07,.01,"UNDETERMINED")])
def test_three_state_boundary(bias,u,status):
    result=u_aware_shots(dict.fromkeys(("real","imag"),bias),dict.fromkeys(("real","imag"),u),
                        epsilon=.1,log_B_upper=0.,evidence_kind="EMPIRICAL",evidence_ref="fixture")
    assert result["eligibility_under_declared_allowance"]==status


def test_unknown_one_axis_and_large_normalization():
    result=u_aware_shots({"real":0.,"imag":0.},{"real":0.,"imag":None},epsilon=.1,
                        log_B_upper=1000.,evidence_kind="EMPIRICAL",evidence_ref="fixture")
    assert result["N_total"] is None
    assert result["axes"]["real"]["integer_status"]=="LOG_DOMAIN_ONLY"
    assert math.isfinite(result["axes"]["real"]["log_shot_bound"])
    assert result["eligibility_under_declared_allowance"]=="UNDETERMINED"
    unknown=u_aware_shots({"real":0.,"imag":0.},{"real":0.,"imag":0.},epsilon=.1,log_B_upper=0.)
    assert unknown["eligibility_under_declared_allowance"]=="UNDETERMINED"


@pytest.mark.parametrize("value",[True,-1.,float("nan"),float("inf")])
def test_invalid_allowance_rejected(value):
    with pytest.raises(ValueError):
        u_aware_shots({"real":0.,"imag":0.},{"real":value,"imag":0.},epsilon=.1,
                      log_B_upper=0.,evidence_kind="EMPIRICAL",evidence_ref="fixture")


def test_nonunitary_propagation():
    assert propagate_allowance([.1,.2],[2.,3.],initial_error=.5)["final"]==pytest.approx(3.5)
    with pytest.raises(ValueError,match="OVERFLOW"):
        propagate_allowance([0.],[1e308],initial_error=1e308)


def test_independent_occupation_sign_scalar_and_square():
    one=np.array([[.4,.2+.3j],[.2-.3j,-.7]])
    g=np.array([[.1,-.2j],[.2j,.5]])
    # Basis [|01>,|10>] reverses orbital order; independent analytic one-particle oracle.
    oracle=np.array(occupation_df_matrix(.3,one,[.8],[g],[1,2]))
    expected=(.3*np.eye(2)+one+.8*g@g)[::-1,::-1]
    np.testing.assert_allclose(oracle,expected,atol=1e-15)
    filled=np.array(occupation_df_matrix(.3,one,[.8],[g],[3]))
    np.testing.assert_allclose(filled,[[.3+np.trace(one)+.8*np.trace(g)**2]],atol=1e-15)


def test_square_before_projection_and_sector_rejection():
    zero=np.zeros((2,2)); flip=np.array([[0.,1.],[1.,0.]])
    assert occupation_df_matrix(0.,zero,[1.],[flip],[1])==[[1+0j]]
    with pytest.raises(ValueError,match="LEAVES_SECTOR"):
        occupation_df_matrix(0.,flip,[],[],[1])


def test_independent_reference_high_precision_fixture():
    a=reference_signal_mp(.25,[[.5,0.],[0.,-.5]],[],[],[1,2],[1.,0.],.8,dps=80)
    b=reference_signal_mp(.25,[[.5,0.],[0.,-.5]],[],[],[1,2],[1.,0.],.8,dps=120)
    assert complex(float(a["real"]),float(a["imag"]))==pytest.approx(complex(math.cos(.2),math.sin(.2)))
    assert float(a["real"])==float(b["real"])
    assert not a["certified"]


def test_tol_only_exact_kwargs_and_change_records():
    seen=[]
    def fake(two,**kwargs):
        seen.append(kwargs)
        return np.array([.2,.3]),np.array([np.eye(2),2*np.eye(2)]),.4*np.eye(2),1e-9
    h=tol_only_df_from_integrals(constant=.5,one_body=np.eye(2),two_body=np.zeros((2,)*4),decomposer=fake)
    assert seen==[{"truncation_threshold":1e-8}]
    assert h.n_blocks==2
    assert not h.metadata["final_rank_supplied"]
    assert len(h.metadata["hermitization"])==3
    np.testing.assert_allclose(h.one_body,1.4*np.eye(2))


def test_hermitization_over_policy_stops():
    def fake(*args,**kwargs):
        return np.array([1.]),np.array([[[0.,1.],[0.,0.]]]),np.zeros((2,2)),0.
    with pytest.raises(ValueError,match="HERMITIZATION_POLICY"):
        tol_only_df_from_integrals(constant=0.,one_body=np.eye(2),two_body=np.zeros((2,)*4),decomposer=fake)


@pytest.mark.parametrize("rank",[2,11,15,25])
def test_seven_cells_and_thirty_six_wrappers(rank):
    p=preparation_plan(rank)
    assert len(p["cells"])==7 and len(p["wrapper_tasks"])==36
    assert p["cells"][0]["prefix"]==(rank+1)//2
    assert len({(t["cell_id"],t["replica"],t["seed"]) for t in p["wrapper_tasks"] if t["seed"] is not None})==4
    assert p["status"]=="H6_NOT_AUTHORIZED" and p["input_binding"] is None
    assert p["quantum_shots"] is None and p["G"] is None
    with pytest.raises(RuntimeError,match="H6_NOT_AUTHORIZED"):
        reject_scientific_launch(p)


def test_actual_time_coverage_contains_s4_negative_and_merged_time():
    p=preparation_plan(11)
    c=next(c for c in p["cells"] if c["id"]=="H6_B1_S4_q2")
    schedule=primitive_time_schedule(c)
    w=1/(2-2**(1/3)); middle=1-2*w
    assert (0,.4*(w+middle)/2) in schedule["unique_primitive_times"]
    assert any(t<0 for _,t in schedule["unique_primitive_times"])
    for mode in ("ordinary_one_outer_step","directional_one_outer_step"):
        for term in range(12):
            assert sum(row[1] for row in schedule[mode] if row[0]==term)==pytest.approx(.4)


def test_before_call_matvec_caps_shared_directions():
    seen=[]; total=CallBudget(reference_matvec=2)
    apply,local=bounded_matvec(lambda v: seen.append(v) or v,total,name="reference_matvec",per_action_cap=1)
    apply(1)
    with pytest.raises(RuntimeError,match="CALL_BUDGET"):
        apply(2)
    assert seen==[1] and total.used["reference_matvec"]==1
    matrix,calls=bounded_sector_matrix(lambda v:v,1,total)
    assert matrix[0,0]==1 and calls==1
    with pytest.raises(RuntimeError,match="CALL_BUDGET"):
        bounded_sector_matrix(lambda v:seen.append(v),1,total)


def test_solver_wraps_both_directions_and_postsolve_calls():
    total=CallBudget(solver_matvec=2)
    def solver(op,**kwargs):
        assert kwargs["which"]=="SA" and kwargs["ncv"]==3
        op.matvec(np.array([1.,0.,0.])); op.rmatvec(np.array([1.,0.,0.]))
        return "fake-result"
    result,op,counter=bounded_solver(lambda v:v,3,[1.,0.,0.],total,solver=solver)
    assert result=="fake-result" and counter.used["calls"]==2
    with pytest.raises(RuntimeError,match="CALL_BUDGET"):
        op.matvec(np.array([1.,0.,0.]))


@pytest.mark.parametrize("log_B,reason",[(710.,"OVERFLOW"),(709.,"SUBNORMAL")])
def test_scale_failures(log_B,reason):
    with pytest.raises(ValueError,match=reason):
        finite_scale_guard(log_B=log_B,intermediate_norm=1.,absolute_discrepancy=0.)


def test_scale_and_output_cap_no_overwrite(tmp_path):
    assert finite_scale_guard(log_B=1.,intermediate_norm=100.,absolute_discrepancy=1e-9)["certified"] is False
    w=BoundedWriter(tmp_path,byte_cap=300,reserve=100,diagnostics_cap=1)
    w.write("a.json",{"a":1},diagnostic=True)
    with pytest.raises(RuntimeError,match="CALL_BUDGET"):
        w.write("b.json",{},diagnostic=True)
    assert not (tmp_path/"b.json").exists()
    with pytest.raises(RuntimeError,match="OUTPUT_WRITE_CAP"):
        w.write("big.json",{"data":"x"*300})
    with pytest.raises(FileExistsError):
        w.write("a.json",{})


class FixturePort:
    synthetic_only=True
    def __init__(self,fail=False,mutate=False):
        self.fail,self.mutate,self.groups=fail,mutate,0
    def setup(self,plan):
        assert plan["target"]["n_qubits"]==12
    def correctness(self,cell):
        return {"technical_pass":not self.fail,"fixture":cell["id"]}
    def occurrence(self,cell,seed,occurrence):
        return {"seed":seed,"occurrence":occurrence,"synthetic":True}
    def wrapper(self,task,trajectory):
        if self.mutate and trajectory:
            trajectory.append({"mutation":True})
        return {"fixture_task":task,"fake_cost":1}
    def release_group(self):
        self.groups+=1


def test_synthetic_controller_complete_and_no_real_port(tmp_path):
    port=FixturePort()
    report=exercise_synthetic_controller(port,BoundedWriter(tmp_path,byte_cap=2**20),actual_rank=11)
    assert report["status"]=="SYNTHETIC_CONTROLLER_COMPLETE"
    assert report["calls"]=={"compile":36,"trajectory":4,"occurrence":8}
    assert port.groups==9 and report["N"] is None and report["mandatory_stop"]
    with pytest.raises(RuntimeError,match="NOT_IMPLEMENTED_OR_AUTHORIZED"):
        exercise_synthetic_controller(SimpleNamespace(),None,actual_rank=11)


@pytest.mark.parametrize("fail,mutate",[(True,False),(False,True)])
def test_synthetic_failures_are_partial_stop(tmp_path,fail,mutate):
    report=exercise_synthetic_controller(FixturePort(fail,mutate),BoundedWriter(tmp_path,byte_cap=2**20),actual_rank=11)
    assert report["status"]=="SYNTHETIC_CONTROLLER_STOP"
    assert report["compiled_wrappers"]<36 and report["mandatory_stop"]


def test_watchdog_stops_dummy_worker_without_retry(tmp_path):
    r=supervise_synthetic([sys.executable,"-c","import time; time.sleep(2)"],tmp_path,
                          total_wall_seconds=1,phase_wall_seconds=dict.fromkeys(("input_reference","correctness","wrapper_cost"),.08),
                          output_bytes=2**20)
    assert r["status"]=="SYNTHETIC_WATCHDOG_STOP"
    assert r["reason"]=="PHASE_WALL_CAP:input_reference" and r["retry"] is False


@pytest.mark.parametrize("case",["complete","missing_terminal","phase_skip","log_overflow"])
def test_watchdog_dummy_terminal_phase_and_log_gates(tmp_path,case):
    terminal={"status":"SYNTHETIC_CONTROLLER_COMPLETE","completed_correctness_cells":7,
              "compiled_wrappers":36,"synthetic_only":True,"mandatory_stop":True,"next_stage_authorized":False}
    code="from pathlib import Path; import json; p=Path("+repr(str(tmp_path))+"); "
    if case=="phase_skip":
        code+="(p/'phase_wrapper_cost.json').write_text('{}'); "
    else:
        code+="[(p/('phase_'+s+'.json')).write_text('{}') for s in ('input_reference','correctness','wrapper_cost')]; "
    if case!="missing_terminal":
        code+="(p/'worker_terminal.json').write_text("+repr(json.dumps(terminal))+"); "
    if case=="log_overflow":
        code+="print('x'*100000,flush=True); "
    r=supervise_synthetic([sys.executable,"-c",code],tmp_path,total_wall_seconds=2,
                          phase_wall_seconds=dict.fromkeys(("input_reference","correctness","wrapper_cost"),1),output_bytes=2**20)
    if case=="complete":
        assert r["status"]=="SYNTHETIC_WATCHDOG_COMPLETE"
    else:
        assert r["status"]=="SYNTHETIC_WATCHDOG_STOP"
        assert r["reason"]=={"missing_terminal":"WORKER_FAILED_OR_INCOMPLETE","phase_skip":"INVALID_PHASE_ORDER",
                             "log_overflow":"WORKER_LOG_CAP"}[case]


def test_log_shots_cancel_large_scales_without_float_overflow():
    r=u_aware_shots({"real":0.,"imag":0.},{"real":0.,"imag":0.},epsilon=1e300,log_B_upper=800.,
                    evidence_kind="EMPIRICAL",evidence_ref="synthetic")
    assert r["N_total"] is None and r["axes"]["real"]["integer_status"]=="LOG_DOMAIN_ONLY"
    r=u_aware_shots({"real":0.,"imag":0.},{"real":0.,"imag":0.},epsilon=1e300,log_B_upper=600.,
                    evidence_kind="EMPIRICAL",evidence_ref="synthetic")
    assert r["N_total"]==2


def test_preparation_cli_is_stdlib_and_rejects_execute(tmp_path):
    script=Path(__file__).resolve().parents[3]/"scripts/resource_applicability/prepare_track_a_ax2b_post_review.py"
    # -S makes scientific packages unavailable; metadata path still works.
    run=subprocess.run([sys.executable,"-S",str(script),"--output",str(tmp_path/"plan.json")],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads((tmp_path/"plan.json").read_text())["actual_rank"] is None
    bad=subprocess.run([sys.executable,"-S",str(script),"--output",str(tmp_path/"x.json"),"--execute"],capture_output=True,text=True)
    assert bad.returncode!=0 and not (tmp_path/"x.json").exists()


@pytest.mark.parametrize("tau",[-.3,.3])
@pytest.mark.parametrize("K",[0,2,6])
def test_complete_finite_event_average_and_two_iid_occurrences(tau,K):
    from trotterlib.rte import (RTEComponent,finite_rte_distribution,enumerate_rte_events,
                               event_unitary,exact_enumerated_event_mean_operator,finite_taylor_operator)
    operators={"X":np.array([[0,1],[1,0]],dtype=complex),"Z":np.diag([1.,-1.]).astype(complex)}
    components=[RTEComponent("X",.4,.4,1),RTEComponent("Z",.6,.6,1)]
    distribution=finite_rte_distribution(tau,K)
    events=enumerate_rte_events(components,distribution,max_events=200)
    mean=exact_enumerated_event_mean_operator(events,operators)
    expected=finite_taylor_operator(.4*operators["X"]+.6*operators["Z"],tau,K)/distribution.exact_finite_distribution
    np.testing.assert_allclose(mean,expected,atol=2e-15)
    # Probability product represents independent occurrences, not one reused draw.
    paired=sum(a.event_probability*b.event_probability*event_unitary(b,operators)@event_unitary(a,operators)
               for a in events for b in events)
    np.testing.assert_allclose(paired,mean@mean,atol=3e-15)
    psi=np.array([1.,1j])/math.sqrt(2)
    # Hadamard X/Y expectations on each unitary controlled branch, including phase.
    plus=np.r_[psi,psi]/math.sqrt(2)
    X=np.kron(np.array([[0,1],[1,0]]),np.eye(2))
    Y=np.kron(np.array([[0,-1j],[1j,0]]),np.eye(2))
    axis_means=[0.,0.]
    for event in events:
        acted=np.r_[psi,event_unitary(event,operators)@psi]/math.sqrt(2)
        for i,measurement in enumerate((X,Y)):
            axis_means[i]+=event.event_probability*np.vdot(acted,measurement@acted).real
    signal=np.vdot(psi,mean@psi)
    np.testing.assert_allclose(axis_means,[signal.real,signal.imag],atol=3e-15)
