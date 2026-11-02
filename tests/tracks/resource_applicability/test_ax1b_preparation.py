"""All numbers, states/identities and costs here are constructed synthetic data."""
from copy import deepcopy
from collections import namedtuple
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trottertracks.resource_applicability import ax1b_contract as c
from trottertracks.resource_applicability import ax1b_execution as ex
from trottertracks.resource_applicability.ax1b_models import Features,fit_cost,kkt_residual,finite_normalization,complexity_gate,structural_na
from trottertracks.resource_applicability.ax1b_data import VerifiedReader,SavedCandidate,join_m1,check_membership,pm1_features,features_from_signal,folds,project_saved,validate_candidate_scope,canonical_compiler_identity
from trottertracks.resource_applicability.ax1b_evaluation import cost_error,error_summary,reference_shots,conditional_work,selection,paired_statistics,paired_total,rank_index,common_support_selection

SINGLE="PRED_BASE_SINGLE_COEFF"
FEW="PRED_BASE_FEW_PARAM"


def sample_features(n=24):
    return [Features(i+1,(i%5)**2+0.5,(i%7)+2,(i%3)+1) for i in range(n)]


def test_single_closed_form_axis_separate_and_no_intercept():
    fs=sample_features();actions=[f.actions()["A_exact"] for f in fs]
    for coefficient in [3.0,7.0]:
        model=fit_cost(SINGLE,fs,[coefficient*x for x in actions],list(map(str,range(len(fs)))))
        assert model.status=="FIT_OK"
        assert model.coefficients==pytest.approx({"A_exact":coefficient})
        assert model.predict(fs[0])["C_pred"]==pytest.approx(coefficient*actions[0])


def test_known_few_coefficients_and_train_scaling():
    fs=sample_features()
    theta=dict(intercept=2,n_det=3,E_rand=5,n_fixed=7,q=11)
    ys=[sum(theta[k]*v for k,v in f.mapping().items()) for f in fs]
    model=fit_cost(FEW,fs,ys,list(map(str,range(len(fs)))))
    assert model.status=="FIT_OK"
    assert model.coefficients==pytest.approx(theta,rel=1e-9,abs=1e-9)
    assert model.audit["kkt_residual"]<=1e-8
    assert model.scales["intercept"]==1


def test_nnls_nonnegative_when_unconstrained_slope_negative():
    fs=[Features(i,0,0,1) for i in range(6)]
    model=fit_cost(FEW,fs,[10-i for i in range(6)],list(map(str,range(6))))
    assert all(v>=0 for v in model.coefficients.values())
    assert model.coefficients["n_det"]==0


def test_dependency_priority_zero_features_and_test_relation_flag():
    fs=[Features(i,0,2+3*(i+1),i+1) for i in range(12)]
    model=fit_cost(FEW,fs,[f.actions()["A_exact"] for f in fs],list(map(str,range(12))))
    assert model.dropped["E_rand"]=="ALL_ZERO_TRAIN_COLUMN"
    assert "n_fixed" in model.dropped and "q" in model.dropped
    assert model.retained==["intercept","n_det"]
    old=deepcopy(model.record())
    p=model.predict(Features(3,5,999,50))
    assert p["C_pred"] is not None
    assert any(flag.startswith("OUTSIDE_TRAIN_FEATURE_RELATION") for flag in p["extrapolation_flags"])
    assert model.record()==old


@pytest.mark.parametrize("model_id",[SINGLE,FEW])
def test_zero_target_and_zero_reference(model_id):
    fs=sample_features();model=fit_cost(model_id,fs,[0]*len(fs),list(map(str,range(len(fs)))))
    assert model.status=="FIT_OK" and all(v==0 for v in model.coefficients.values())
    assert model.predict(fs[0])["C_pred"]==0
    assert cost_error(0,2)["signed_relative"] is None


def test_zero_action_single_and_insufficient_rows_are_not_substituted():
    model=fit_cost(SINGLE,[Features(0,0,0,1)]*3,[1,1,1],["a","b","c"])
    assert model.status=="FIT_UNIDENTIFIABLE" and model.predict(Features(1,1,1,1))["C_pred"] is None
    model=fit_cost(FEW,sample_features(2),[1,2],["a","b"])
    assert model.status=="FIT_UNIDENTIFIABLE"


@pytest.mark.parametrize("value",[-1,float("inf"),float("nan"),True])
def test_bad_feature_or_cost(value):
    with pytest.raises(c.Stop):Features(value,0,0,1)
    with pytest.raises(c.Stop):fit_cost(SINGLE,[Features(1,1,1,1)]*2,[value,1],["a","b"])


def test_missing_is_not_zero_and_reference_keys_rejected():
    f=Features(None,1,2,1)
    assert f.actions()["A_exact"] is None
    assert fit_cost(SINGLE,[f,f],[1,2],["a","b"]).status=="INPUT_MISSING"
    with pytest.raises(c.Stop):Features.from_mapping(dict(n_det=1,E_rand=2,n_fixed=3,q=1,reference_bias=0))


@pytest.mark.parametrize("key",["reference_bias","reference_shots","actual_compiled_cost"])
def test_reference_values_cannot_enter_feature_projection(key):
    with pytest.raises(c.Stop):Features.from_mapping({"n_det":1,"E_rand":2,"n_fixed":3,"q":1,key:999})


def test_test_truth_cannot_affect_fit_or_prediction():
    fs=sample_features();target=[2*f.actions()["A_exact"] for f in fs]
    fitted=fit_cost(SINGLE,fs,target,list(map(str,range(len(fs)))))
    before=fitted.record();p=fitted.predict(Features(9000,8000,7000,8))
    for test_cost in [0,1e12]:
        cost_error(test_cost,p["C_pred"])
        assert fitted.record()==before


def test_kkt_failure_nonnegative_and_solver_failure(monkeypatch):
    with pytest.raises(c.Stop,match="KKT"):kkt_residual(np.ones((3,1)),np.ones(3),np.array([0.0]))
    with pytest.raises(c.Stop,match="coefficient"):kkt_residual(np.ones((3,1)),np.ones(3),np.array([-1.0]))
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize,"nnls",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("synthetic failure")))
    with pytest.raises(c.Stop,match="no fallback"):fit_cost(FEW,sample_features(),[1]*24,list(map(str,range(24))))


def test_scipy_version_mismatch_no_silent_adapter(monkeypatch):
    import scipy
    monkeypatch.setattr(scipy,"__version__","0.synthetic")
    with pytest.raises(c.Stop,match="SciPy"):fit_cost(FEW,sample_features(),[1]*24,list(map(str,range(24))))


def test_action_rounding_and_structural_na():
    f=Features(2,2.25,3,1)
    assert f.actions()["A_exact"]==7.25 and f.actions()["A_ceil"]==8
    assert structural_na()["C_pred"] is None
    with pytest.raises(c.Stop):cost_error(1,1,unit="action_count_per_shot")


def test_finite_normalization_known_zero_and_closed_weights():
    z=finite_normalization(0,1,2,4,4)
    assert z["B"]==z["b"]==1 and z["probabilities"]==[1,0,0]
    r=finite_normalization(0.5,1,1,1,2)
    expected=math.hypot(1,0.5)+(0.5**2/2)*math.hypot(1,0.5/3)
    assert r["b"]==pytest.approx(expected)
    assert r["log_B"]==pytest.approx(math.log(expected))
    assert sum(r["probabilities"])==pytest.approx(1)
    assert r["log_bound_slack"]>=0


@pytest.mark.parametrize("args",[(1,1,0,1,2),(1,1,1,1,3),(float("inf"),1,1,1,2),(1000,1,1,1,1000)])
def test_bad_normalization_or_overflow(args):
    with pytest.raises((c.Stop,OverflowError)):finite_normalization(*args)


def test_total_normalization_overflow_stops_and_upper_bound_can_overflow():
    with pytest.raises(c.Stop,match="B overflow"):finite_normalization(100000,1,1000,1,2)
    r=finite_normalization(100,1,1,1,2)
    assert r["paper_upper_bound"] is None and r["paper_upper_bound_overflowed"]
    assert r["B"] is not None


def gate_scores(score):
    return {q:dict(score=score,full_coverage=True,support_hash=str(q)) for q in [1,2,4,8]}


def test_complexity_gate_pass_keep_simple_undetermined():
    assert complexity_gate(gate_scores(.2),gate_scores(.1))["selected_model"]==FEW
    assert complexity_gate(gate_scores(.2),gate_scores(.195))["selected_model"]==SINGLE
    assert complexity_gate(gate_scores(0),gate_scores(0))["selected_model"]==SINGLE
    bad=gate_scores(.1);bad[1]["full_coverage"]=False
    assert complexity_gate(gate_scores(.2),bad)["gate_status"]=="UNDETERMINED"
    assert complexity_gate({},bad)["gate_status"]=="UNDETERMINED"
    bad=gate_scores(.01);bad[1]["score"]=.23
    assert complexity_gate(gate_scores(.2),bad)["selected_model"]==SINGLE


def candidate(i=0,method="B0",q=1,rank=3,geometry="toy_dev"):
    row=dict(candidate_id=f"SYNTHETIC-{i}",method=method,q=q,r=2 if method in {"B2","B3"} else 0,K=2,
             rank=rank,T=.8,delta=.8/q,T_hex=(.8).hex(),delta_hex=(.8/q).hex(),hamiltonian_hash=geometry+"_ham",
             state_hash=geometry+"_state",state_vector_hash=geometry+"_vector",snapshot_sha256=geometry+"_snapshot",
             identity_policy="extract_identity_phase",outer_formula="symmetric_second_order_product_formula",coefficient_atol=0.0,
             compiler_identity=dict(basis_gates=["rz","sx","x","cx"],coupling_map=None,optimization_level=1,qiskit_version="1.3.0",transpiler_seed=17),
             wrapper_semantics="full_measured_hadamard_wrapper_without_state_preparation")
    row["candidate_fingerprint"]=c.digest(row)
    return row


def toy_join():
    row=candidate()
    a=dict(candidate_ledger=[row],signal_records=[dict(candidate=deepcopy(row),candidate_fingerprint=row["candidate_fingerprint"],n_fixed=5)])
    b=dict(compile_map=[dict(candidate=deepcopy(row))])
    return a,b


def test_correct_full_candidate_join():
    a,b=toy_join();assert len(join_m1(a,b))==1


@pytest.mark.parametrize("key,value",[("candidate_fingerprint","wrong"),("hamiltonian_hash","other"),("state_hash","other"),("snapshot_sha256","other"),
                                     ("q",2),("r",8),("K",4),("T",3.2),("delta",.4),("compiler_identity",{}),("wrapper_semantics","other")])
def test_identity_join_rejects_changes(key,value):
    a,b=toy_join();b["compile_map"][0]["candidate"][key]=value
    with pytest.raises(c.Stop):join_m1(a,b)


@pytest.mark.parametrize("which",["duplicate","missing"])
def test_duplicate_missing_candidate(which):
    a,b=toy_join()
    if which=="duplicate":b["compile_map"].append(deepcopy(b["compile_map"][0]))
    else:b["compile_map"]=[]
    with pytest.raises(c.Stop):join_m1(a,b)


def test_membership_rejects_foreign_geometry_even_if_three_way_join_consistent():
    a,b=toy_join();frozen=deepcopy(a["candidate_ledger"])
    for row in [a["candidate_ledger"][0],a["signal_records"][0]["candidate"],b["compile_map"][0]["candidate"]]:row["hamiltonian_hash"]="foreign_geometry"
    join_m1(a,b)
    with pytest.raises(c.Stop):check_membership(a["candidate_ledger"],frozen,"toy")


def test_pm1_action_source_invariant_and_missing_anchors():
    a,b=toy_join();anchors=join_m1(a,b);p=candidate(2,rank=4)
    f,provenance=pm1_features(p,anchors)
    assert f.n_det==8 and f.n_fixed==5 and provenance["no_truth_or_cost_imputation"]
    f,_=pm1_features(p,[]);assert f.n_fixed is None
    foreign=candidate(3,rank=4,geometry="toy_other");assert pm1_features(foreign,anchors)[0].n_fixed is None


def test_source_features_missing_E_and_deterministic_zero():
    random=candidate(method="B2")
    f,_=features_from_signal(random,dict(n_det=6,n_fixed=5,finite_distribution=dict(orders=[0,2],order_probabilities=[.75,.25])))
    assert f.E_rand==3
    missing,_=features_from_signal(random,dict(n_det=6,n_fixed=5));assert missing.E_rand is None
    deterministic,_=features_from_signal(candidate(),dict(n_det=6,n_fixed=5));assert deterministic.E_rand==0


def allow_entry(path,value):
    raw=(c.canonical(value)+"\n").encode()
    return dict(path=path,sha256=c.sha256(raw),schema=dict(format="json",root_keys=sorted(value),schema_version=value.get("schema_version")),
                planned_fields=[dict(selector="required",required=True)])


def test_reader_hash_schema_and_exact_allowlist(tmp_path):
    value=dict(schema_version="toy",required=1)
    entry=allow_entry("toy.json",value);(tmp_path/"toy.json").write_text(c.canonical(value)+"\n")
    reader=VerifiedReader(tmp_path,dict(entries=[entry]),dict(authorized=True))
    assert reader.read("toy.json")==value
    with pytest.raises(c.Stop):reader.read("not_allowlisted.json")
    (tmp_path/"toy.json").write_text("{}")
    with pytest.raises(c.Stop,match="hash"):reader.read("toy.json")
    value=dict(schema_version="toy",wrong=1);entry=allow_entry("toy.json",value);(tmp_path/"toy.json").write_text(c.canonical(value)+"\n")
    with pytest.raises(c.Stop,match="required"):VerifiedReader(tmp_path,dict(entries=[entry]),dict(authorized=True)).read("toy.json")


@pytest.mark.parametrize("path",["../escape.json","/absolute.json","toy.npz","x/.runtime/value.json","x/cache/y.json","x/foo_registry/y.json","x/../y.json","x\\y.json"])
def test_protected_or_noncanonical_paths_before_IO(path):
    with pytest.raises(c.Stop):c.relative_path(path)


def test_symlink_input_is_not_followed(tmp_path):
    (tmp_path/"value.json").write_text("{}")
    (tmp_path/"link.json").symlink_to(tmp_path/"value.json")
    with pytest.raises(c.Stop):c.safe_path(tmp_path,"link.json")


def test_denied_reader_does_not_open_science(monkeypatch):
    calls=[]
    monkeypatch.setattr(Path,"read_bytes",lambda self:calls.append(str(self)))
    with pytest.raises(c.Stop):VerifiedReader(".",dict(entries=[]),dict(authorized=False))
    assert calls==[]


def test_cost_errors_underestimation_missing_and_zero():
    r=cost_error(100,80)
    assert r["signed_relative"]==-.2 and r["absolute_rz"]==20 and r["underestimation_fraction"]==.2
    assert r["log_ratio"]==pytest.approx(math.log(.8))
    assert cost_error(100,0)["status"]=="ZERO_PREDICTION"
    rows=[r,cost_error(100,110),cost_error(100,None),cost_error(0,10)]
    s=error_summary(rows)
    assert s["positive_ref_denominator"]==2 and s["coverage"]==.75 and s["missing_count"]==1
    assert s["underestimate_more_than_10pct_rate"]==.5
    assert rank_index([1,2,2],[10,20,20])["spearman"]==pytest.approx(1)
    assert rank_index([1,1],[2,3])["spearman"] is None


def refs():
    return {"a":dict(candidate=dict(candidate_fingerprint="a"),eligible_ref=True,G_ref=10),
            "b":dict(candidate=dict(candidate_fingerprint="b"),eligible_ref=True,G_ref=20),
            "bad":dict(candidate=dict(candidate_fingerprint="bad"),eligible_ref=False,G_ref=None)}


def test_conditional_regret_false_acceptance_first_and_outside():
    r=selection(refs(),dict(a=30,b=10),"IN_SAMPLE")
    assert r["regret"]==1 and r["information_class"]=="I4_CONDITIONAL_ORACLE"
    r=selection(refs(),{},"TOY",chosen="bad")
    assert r["status"]=="SELECTED_REFERENCE_INELIGIBLE" and r["regret"] is None and r["false_acceptance"] is True
    assert selection(refs(),{},"TOY",chosen="outside")["status"]=="SELECTED_OUTSIDE_DIRECT_SET"


def test_selection_empty_rejected_missing_undefined_and_common_support():
    assert selection({}, {},"TOY")["status"]=="REF_ELIGIBLE_EMPTY"
    assert selection(refs(),{},"TOY")["status"]=="NO_PREDICTIONS"
    assert selection(refs(),dict(a=1),"TOY")["status"]=="INCOMPLETE_PREDICTION_COVERAGE"
    assert selection(refs(),dict(a=1),"TOY",chosen="b")["status"]=="MISSING_COST_PREDICTIONS"
    assert selection(refs(),dict(a=1,b=2),"TOY",predicted_eligible=dict(a=False,b=False))["status"]=="MODEL_ALL_REJECTED"
    zero=refs();zero["a"]["G_ref"]=0
    assert selection(zero,dict(a=1,b=2),"TOY")["status"]=="ZERO_REGRET_DENOMINATOR"
    missing=refs();missing["a"]["G_ref"]=None
    assert selection(missing,dict(a=1,b=2),"TOY")["status"]=="REFERENCE_COST_UNDEFINED"
    undetermined=refs();undetermined["a"]["eligible_ref"]=None
    assert selection(undetermined,{},"TOY",chosen="a")["status"]=="SELECTED_ELIGIBILITY_UNDETERMINED"
    shared=common_support_selection(refs(),dict(simple=dict(a=1,b=2),few=dict(b=2)),dict(simple={"a","b","bad"},few={"b","bad"}),"INTERNAL")
    assert shared[0]["direct_support_sha256"]==shared[1]["direct_support_sha256"]
    assert shared[1]["full_set_diagnostic"]["regret"] is None and shared[1]["common_set_diagnostic"]["regret"]==0


def test_selection_tie_break_and_reference_shots():
    s=selection(refs(),dict(a=1,b=1),"TOY");assert s["selected"]=="a"
    sh=reference_shots(dict(real=0,imag=0),1,.05)
    assert sh["eligible_ref"] and sh["axis_shots"]["real"]==math.ceil(2/(.05/math.sqrt(2))**2*math.log(80))
    assert conditional_work(sh["axis_shots"],dict(cosine=2,sine=3))==5*sh["axis_shots"]["real"]
    boundary=reference_shots(dict(real=.1,imag=0),1,math.sqrt(2)*.1)
    assert not boundary["eligible_ref"] and boundary["N_total"] is None
    assert reference_shots(dict(real=0,imag=0),1e308,.05)["status"]=="NUMERICAL_UNDEFINED"
    assert all(c.operational_na()[key] is None for key in ["axis_bias_pred","N_pred_by_axis","G_operational_pred","regret_operational"])


def pair_rows(n=32):
    left=[dict(trajectory_index=i,trajectory_seed=i+10,step_seeds=[i+100],evolution_circuit_semantics_fingerprint=f"toy{i}",cost=i+1) for i in range(n)]
    right=[dict(p,cost=2*p["cost"]+1) for p in left]
    return left,right


def test_paired_covariance_SE_and_rare_event_unresolved():
    left,right=pair_rows();stat=paired_statistics(left,right)
    assert stat["cov_cos_sin"]==pytest.approx(2*stat["var_cosine"])
    result=paired_total(stat,3,4)
    assert result["SE"]**2==pytest.approx(121*stat["var_cosine"]/32)
    assert not result["formal_ci"] and stat["cross_candidate_covariance"] is None
    assert stat["rare_event_cost_tail"]=="UNRESOLVED"
    assert not stat["quantum_shot_uncertainty"]


@pytest.mark.parametrize("field",["trajectory_seed","step_seeds","evolution_circuit_semantics_fingerprint"])
def test_pair_mismatch(field):
    left,right=pair_rows();right[0][field]="wrong"
    with pytest.raises(c.Stop):paired_statistics(left,right)


def test_deterministic_cost_and_missing_random_samples():
    left,right=pair_rows(1);s=paired_statistics(left,right,random=False)
    assert paired_total(s,3,4)["SE"]==0
    with pytest.raises(c.Stop):paired_statistics(left,right,random=True)


def saved_candidate(i,dataset="TRAIN_M1_210",method=None):
    method=method or ["B0","B1","B2","B3"][i%4]
    q=[1,2,4,8][(i//4)%4];prefix=[0,3,6,9,12][(i//16)%5]
    if method=="B3":prefix=0
    if method=="B1":prefix=12
    cnd=candidate(i,method,q,prefix,"synthetic_"+dataset)
    E=q*2 if method in {"B2","B3"} else 0
    f=Features(2*q*prefix,E,2+3*q,q);A=f.actions()["A_exact"]
    n=32 if method in {"B2","B3"} else 1
    left,right=pair_rows(n)
    left=[dict(p,cost=3*A) for p in left];right=[dict(p,cost=5*A) for p in right]
    finite=dict(exact_rte_lambda_r=0,normalization_log=0) if method in {"B2","B3"} else None
    return SavedCandidate(dataset,cnd,f,dict(synthetic_only=True),{a:{m:(3 if a=="cosine" else 5)*A for m in c.METRICS} for a in c.AXES},dict(real=0,imag=0),1,{m:(left,right) for m in c.METRICS},finite)


def test_group_folds_exclude_pm1_m2_and_keep_deterministic_K_training():
    rows=[saved_candidate(i) for i in range(40)];parts=folds(rows)
    assert sum(p["diagnostic_family"]=="leave_one_q_out" for p in parts)==4
    for p in parts:
        assert not ({r.fingerprint for r in p["train"]}&{r.fingerprint for r in p["test"]})
        if p["diagnostic_family"]=="leave_one_random_K_out":
            assert all(r.candidate["method"] in {"B2","B3"} for r in p["test"])
            assert all(r in p["train"] for r in rows if r.candidate["method"] in {"B0","B1"})
    with pytest.raises(c.Stop):folds([saved_candidate(1,"DIAG_PM1_8")])


def test_synthetic_full_orchestration_only_no_IO():
    from trottertracks.resource_applicability.ax1b_analysis import analyze
    rows=[saved_candidate(i) for i in range(210)]+[saved_candidate(1000+i,"DIAG_PM1_8",method="B0") for i in range(8)]+[saved_candidate(2000+i,"DIAG_M2_5") for i in range(5)]
    result=analyze(rows,dict(metrics=dict(epsilon_anchors=[.05])),{},dict(entries=[]))
    assert len(result["predictions.jsonl"])>210
    assert all(p["N_pred_by_axis"] is None and p["G_operational_pred"] is None for p in result["predictions.jsonl"])
    pooled=[r for r in result["conditional_oracle_selection.csv"] if r.get("fold_id")=="POOLED_Q_CROSS_FITTED"]
    assert pooled and all(r["single_frozen_model"] is False for r in pooled)
    assert all(r["information_class"]=="I4_CONDITIONAL_ORACLE" for r in result["conditional_oracle_selection.csv"] if r.get("row_kind")=="candidate_conditional_oracle_work")
    assert result["model_fits.json"]["complexity_gate"]["selected_model"]==SINGLE
    for filename,value in result.items():assert ex.encode_output(filename,value)


@pytest.fixture
def approved(tmp_path,monkeypatch):
    source=b"# synthetic frozen source\n";(tmp_path/"toy.py").write_bytes(source)
    bundle=dict(frozen_files=[dict(path="toy.py",sha256=c.sha256(source))])
    plan=dict(output=dict(directory="new_output"));allow=dict(entries=[])
    monkeypatch.setattr(ex,"validate_preparation",lambda *args:(allow,plan))
    commit="a"*40
    monkeypatch.setattr(ex.subprocess,"check_output",lambda command,**k:(commit+"\n").encode() if command[1]=="rev-parse" else source)
    version=namedtuple("Version","major minor micro releaselevel serial")(3,11,9,"final",0)
    monkeypatch.setattr(ex,"sys",SimpleNamespace(version_info=version))
    env=dict(python="3.11.9",python_releaselevel="final",numpy="1.26.4",scipy="1.14.1")
    monkeypatch.setattr(ex,"environment",lambda:env)
    auth=dict(ax1b_analysis_authorized=True,science_authorized=False,explicit_user_launch_required=True,mandatory_stop=True,next_stage_authorized=False,
              independent_review_decision="APPROVE_AX1B_EXECUTION",user_launch_record="SYNTHETIC ONLY",preparation_manifest_sha256=c.digest(bundle),source_commit=commit,
              environment=env,resources=dict(assigned_cpu_cores=1,cpu_affinity=[min(ex.os.sched_getaffinity(0))],processes=1,blas_threads=1,
                  ram_limit_bytes=1000000,wall_time_limit_seconds=10,output_disk_limit_bytes=1000000),output_directory="new_output")
    auth["analysis_environment_synthetic_audit"]=dict(python=env["python"],numpy=env["numpy"],scipy=env["scipy"],exit_code=0,passed=1,failed=0,skipped=0,
       protected_access_attempts=0,scientific_import_attempts=0,real_data_fit_executed=False,source_files_after_successful_test=bundle["frozen_files"])
    return tmp_path,bundle,auth


def test_unapproved_gate_never_reads_or_fits(monkeypatch):
    calls=[]
    monkeypatch.setattr(ex,"validate_preparation",lambda *args:calls.append("science read"))
    monkeypatch.setattr(Path,"read_bytes",lambda self:calls.append("read"))
    with pytest.raises(c.Stop):ex.execute(".",{},None)
    with pytest.raises(c.Stop):ex.execute(".",{},dict(c.FLAGS),True,"bad")
    assert calls==[]


def test_separate_launch_gate_synthetic_approved(approved):
    root,bundle,auth=approved
    permit=ex.authorize(root,bundle,auth,True,c.digest(auth))
    assert permit["authorized"] and permit["plan"]["output"]["directory"]=="new_output"
    assert not (root/"new_output").exists()
    with pytest.raises(c.Stop):ex.authorize(root,bundle,auth,False,c.digest(auth))
    with pytest.raises(c.Stop):ex.authorize(root,bundle,auth,True,"wrong")


@pytest.mark.parametrize("key",["assigned_cpu_cores","ram_limit_bytes","wall_time_limit_seconds","output_disk_limit_bytes"])
def test_unconfirmed_budget_blocks_before_reader(approved,key):
    root,bundle,auth=approved;auth["resources"][key]=None
    with pytest.raises(c.Stop,match="BUDGET"):ex.authorize(root,bundle,auth,True,c.digest(auth))


def test_source_environment_collision_next_stage_gates(approved,monkeypatch):
    root,bundle,auth=approved
    wrong=deepcopy(auth);wrong["source_commit"]=None
    with pytest.raises(c.Stop,match="IMPLEMENTATION"):ex.authorize(root,bundle,wrong,True,c.digest(wrong))
    wrong=deepcopy(auth);wrong["environment"]["scipy"]="other"
    with pytest.raises(c.Stop,match="ENVIRONMENT"):ex.authorize(root,bundle,wrong,True,c.digest(wrong))
    wrong=deepcopy(auth);wrong["next_stage_authorized"]=True
    with pytest.raises(c.Stop,match="AUTHORIZATION"):ex.authorize(root,bundle,wrong,True,c.digest(wrong))
    (root/"new_output").mkdir()
    with pytest.raises(c.Stop,match="OUTPUT_COLLISION"):ex.authorize(root,bundle,auth,True,c.digest(auth))


def test_source_blob_hash_mismatch(approved):
    root,bundle,auth=approved;bundle["frozen_files"][0]["sha256"]="0"*64;auth["preparation_manifest_sha256"]=c.digest(bundle)
    with pytest.raises(c.Stop,match="IMPLEMENTATION"):ex.authorize(root,bundle,auth,True,c.digest(auth))


def test_candidate_scope_compiler_and_literal_hex():
    row=candidate();validate_candidate_scope(row)
    for key,value in [("T_hex","wrong"),("delta",.01),("compiler_identity",{}),("wrapper_semantics","wrong"),("q",True)]:
        bad=deepcopy(row);bad[key]=value
        with pytest.raises(c.Stop):validate_candidate_scope(bad)


def test_pm1_missing_anchor_stays_na_even_if_feature_dropped():
    from dataclasses import replace
    from trottertracks.resource_applicability.ax1b_analysis import predict_row
    fs=[Features(i,0,2+3*(i+1),i+1) for i in range(12)]
    model=fit_cost(FEW,fs,[f.actions()["A_exact"] for f in fs],list(map(str,range(12))))
    row=replace(saved_candidate(1,"DIAG_PM1_8",method="B0"),features=Features(6,0,None,1))
    assert "n_fixed" in model.dropped
    assert predict_row(model,row)["C_pred"] is None


def test_preparation_validator_only_contracts_and_new_source(tmp_path,monkeypatch):
    (tmp_path/"contract").mkdir()
    configuration=dict(synthetic_only=True)
    allow=dict(entries=[dict(path=f"nonexistent_science_{i}.json") for i in range(45)],**c.FLAGS)
    plan=dict(model_configuration=configuration,model_configuration_sha256=c.digest(configuration),**c.FLAGS)
    hashes={}
    for name,value in [("input_allowlist_v1.json",allow),("execution_plan_draft_v1.json",plan)]:
        path="contract/"+name;raw=c.canonical(value).encode();(tmp_path/path).write_bytes(raw);hashes[path]=c.sha256(raw)
    monkeypatch.setattr(c,"AX1A_HASHES",hashes)
    source=b"# synthetic source\n";(tmp_path/"toy.py").write_bytes(source)
    audit=dict(exit_code=0,failed=0,skipped=0,passed=7,protected_access_attempts=0,scientific_import_attempts=0)
    raw=c.canonical(audit).encode();(tmp_path/"audit.json").write_bytes(raw)
    bundle=dict(schema_version="track_a_ax1b_preparation_manifest_v1",ax1a_commit=c.AX1A_COMMIT,model_configuration_sha256=c.digest(configuration),
                frozen_files=[dict(path="toy.py",sha256=c.sha256(source))],synthetic_test_audit=dict(path="audit.json",sha256=c.sha256(raw)),**c.FLAGS)
    shapes=dict(schemas=dict(preparation_manifest=dict(type="object",required=["schema_version"]),synthetic_test_audit=dict(type="object",required=["passed"])))
    raw=c.canonical(shapes).encode();(tmp_path/"schemas.json").write_bytes(raw)
    bundle["schemas"]=dict(path="schemas.json",sha256=c.sha256(raw))
    ex.validate_preparation(tmp_path,bundle)
    (tmp_path/"toy.py").write_bytes(b"changed")
    with pytest.raises(c.Stop,match="IMPLEMENTATION"):ex.validate_preparation(tmp_path,bundle)


def test_prerelease_python_remains_launch_blocker(approved,monkeypatch):
    root,bundle,auth=approved
    version=namedtuple("Version","major minor micro releaselevel serial")(3,11,0,"candidate",1)
    monkeypatch.setattr(ex,"sys",SimpleNamespace(version_info=version))
    with pytest.raises(c.Stop,match="ENVIRONMENT"):ex.authorize(root,bundle,auth,True,c.digest(auth))


def test_cli_default_no_contract_or_science_reads(monkeypatch,capsys):
    import importlib.util
    path=Path(__file__).resolve().parents[3]/"scripts/resource_applicability/run_track_a_ax1b.py"
    spec=importlib.util.spec_from_file_location("synthetic_ax1b_cli",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[];monkeypatch.setattr(Path,"read_text",lambda self:calls.append("read"))
    assert module.main([])==2 and module.main(["--execute-saved-analysis"])==2
    assert calls==[] and "AX1B_STOP_AUTHORIZATION" in capsys.readouterr().out


def toy_saved_values():
    cs=[candidate(1),candidate(2,method="B2")]
    def sig(row):
        return dict(candidate=deepcopy(row),candidate_fingerprint=row["candidate_fingerprint"],n_det=2*row["q"]*row["rank"],n_fixed=5,
                    expected_random_applications_exact=2 if row["method"]=="B2" else 0,axis_bias=dict(real=0,imag=0),normalization_multiplier=1)
    def axes(row):
        n=32 if row["method"]=="B2" else 1;left,right=pair_rows(n)
        result={}
        for axis,rs in zip(c.AXES,[left,right]):
            for p in rs:p["cost"]={m:10 for m in c.METRICS}
            result[axis]=dict(cost={m:10 for m in c.METRICS},retained_trajectory_records=rs,
                             state_preparation_included=False,measurement_included=True,backend_execution_included=False,trajectory_records_truncated=False)
        return result
    A=dict(candidate_ledger=cs,signal_records=[sig(x) for x in cs])
    B=dict(compile_map=[dict(candidate=deepcopy(x),compiled_axes=axes(x)) for x in cs])
    pc=candidate(3,rank=4);ps=sig(pc)
    P=dict(candidate_records=[dict(candidate=pc,signal=ps,signal_cost_candidate_fingerprint=pc["candidate_fingerprint"],axes=axes(pc))])
    mc=candidate(4,method="B2",geometry="toy_geometry")
    reduced={k:v for k,v in mc.items() if k not in ["T_hex","delta_hex","hamiltonian_hash","state_hash","state_vector_hash","snapshot_sha256"]}
    ms=sig(reduced);pairs=[]
    for i in range(32):
        pairs.append(dict(trajectory_index=i,trajectory_seed=i+10,axes={a:dict(metrics={m:10 for m in c.METRICS},step_seeds=[i+100],shared_evolution_fingerprint=f"toy{i}") for a in c.AXES}))
    M=dict(input_snapshot_identity=dict(hamiltonian_hash=mc["hamiltonian_hash"],state_hash=mc["state_hash"],state_vector_hash=mc["state_vector_hash"],file_sha256=mc["snapshot_sha256"]),
           candidate_results=[dict(signal=ms,execution_candidate_fingerprint=mc["candidate_fingerprint"],development_candidate_fingerprint=cs[1]["candidate_fingerprint"],
                    axis_one_shot_compiled_means={a:{m:10 for m in c.METRICS} for a in c.AXES},compiled=dict(paired_trajectory_rows=pairs))])
    schemas=["pr2_matched_accuracy_m1_a_result_v2","pr2_matched_accuracy_m1_b1_result_v2","track_a_pm1_discard_result_v1","pr2_matched_accuracy_m2_transfer_result_v2"]
    values=dict(zip(schemas,[A,B,P,M]))
    expected_m2=dict(mc,T_hex=None,delta_hex=None,T_literal=mc["T"],delta_literal=mc["delta"])
    allow=dict(entries=[dict(path=s,schema=dict(schema_version=s)) for s in schemas],membership=dict(
        TRAIN_M1_210=dict(rows=cs),DIAG_PM1_8=dict(rows=[pc]),DIAG_M2_5=dict(rows=[expected_m2])))
    return values,allow


def test_schema_adapters_pm1_and_m2_root_geometry_identity():
    values,allow=toy_saved_values();rows=project_saved(values,allow)
    assert len(rows)==4
    assert next(r for r in rows if r.dataset=="DIAG_PM1_8").features.n_fixed==5
    m2=next(r for r in rows if r.dataset=="DIAG_M2_5")
    assert m2.candidate["hamiltonian_hash"]=="toy_geometry_ham" and m2.candidate["T_hex"] is None
    left,right=m2.pairs_by_metric["rz_count"]
    assert paired_statistics(left,right)["mean_cosine"]==10


def test_schema_adapters_reject_m2_dev_execution_confusion_and_scope():
    values,allow=toy_saved_values()
    values["pr2_matched_accuracy_m2_transfer_result_v2"]["candidate_results"][0]["execution_candidate_fingerprint"]="wrong"
    with pytest.raises(c.Stop):project_saved(values,allow)
    values,allow=toy_saved_values()
    values["pr2_matched_accuracy_m1_b1_result_v2"]["compile_map"][0]["compiled_axes"]["cosine"]["measurement_included"]=False
    with pytest.raises(c.Stop):project_saved(values,allow)


def test_preparation_schema_types_flags_and_required_fields():
    schema=dict(type="object",required=["enabled","count","source"],properties=dict(enabled=dict(type="boolean",const=False),count=dict(type="integer",minimum=1),source=dict(type="string",pattern="^[0-9a-f]{40}$")))
    c.check_schema(dict(enabled=False,count=1,source="a"*40),schema)
    for bad in [dict(enabled=True,count=1,source="a"*40),dict(enabled=False,count=True,source="a"*40),dict(enabled=False,count=1,source="bad"),dict(enabled=False)]:
        with pytest.raises(c.Stop):c.check_schema(bad,schema)


def test_registered_new_output_schemas_on_synthetic_rows():
    path=Path(__file__).resolve().parents[3]/"artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/schemas_v1.json"
    schemas=json.loads(path.read_text())["schemas"]
    prediction=dict(schema_version="track_a_ax1b_prediction_v1",dataset_id="SYNTHETIC",candidate_fingerprint="toy",model_id=SINGLE,model_version=1,
                    fold_id="toy",diagnostic_kind="SYNTHETIC",C_pred_by_axis=dict(cosine=1,sine=2),N_pred_by_axis=None,G_operational_pred=None,G_conditional_oracle=None)
    c.check_schema(prediction,schemas["prediction"])
    bad=dict(prediction,G_operational_pred=999)
    with pytest.raises(c.Stop):c.check_schema(bad,schemas["prediction"])
    c.check_schema(dict(status="AX1B_COMPLETE_WITH_DECLARED_NA",mandatory_stop=True,next_stage_authorized=False),schemas["terminal"])
    with pytest.raises(c.Stop):c.check_schema(dict(status="AX1B_COMPLETE_WITH_DECLARED_NA",mandatory_stop=True,next_stage_authorized=True),schemas["terminal"])


def test_analysis_environment_needs_same_source_synthetic_success(approved):
    root,bundle,auth=approved;auth["analysis_environment_synthetic_audit"]=None
    with pytest.raises(c.Stop,match="ENVIRONMENT"):ex.authorize(root,bundle,auth,True,c.digest(auth))


@pytest.mark.parametrize("unknown_cost",[None,1e-200,1e200])
def test_prelaunch_unknown_reference_blocks_full_set_optimum(unknown_cost):
    values=refs();values["u"]=dict(candidate=dict(candidate_fingerprint="u"),eligible_ref=None,G_ref=unknown_cost)
    result=selection(values,dict(a=30,b=10,u=0),"INTERNAL_GROUP")
    assert result["status"]=="REF_ELIGIBILITY_UNDETERMINED"
    assert result["selected"]=="b" and result["regret"] is None
    assert result["full_set_min_G_ref"] is None and result["common_support_regret"] is None
    assert result["known_eligible_subset_regret"]==1
    assert result["known_eligible_prediction_subset_regret"]==1
    assert result["undetermined_reference_candidates"]==["u"]
    assert result["registered_direct_count"]==4


@pytest.mark.parametrize("states,status",[
    ([True,True],"VALID_CONDITIONAL_ORACLE"),
    ([False,False],"REF_ELIGIBLE_EMPTY"),
    ([None,None],"REF_ELIGIBILITY_UNDETERMINED"),
    ([False,None],"REF_ELIGIBILITY_UNDETERMINED"),
])
def test_prelaunch_empty_vs_unknown_and_normal_regret(states,status):
    values={k:dict(candidate=dict(candidate_fingerprint=k),eligible_ref=state,G_ref=cost) for k,state,cost in zip(["a","b"],states,[10,20])}
    result=selection(values,dict(a=30,b=10),"SYNTHETIC")
    assert result["status"]==status
    assert result["regret"]==(1 if status=="VALID_CONDITIONAL_ORACLE" else None)


def test_prelaunch_selected_unknown_and_false_acceptance_priority():
    values=refs();values["u"]=dict(candidate={},eligible_ref=None,G_ref=.0001)
    assert selection(values,{},"SYNTHETIC",chosen="u")["status"]=="SELECTED_ELIGIBILITY_UNDETERMINED"
    result=selection(values,{},"SYNTHETIC",chosen="bad")
    assert result["status"]=="SELECTED_REFERENCE_INELIGIBLE" and result["false_acceptance"] is True
    assert result["regret"] is None and result["undetermined_reference_candidates"]==["u"]


def test_prelaunch_unknown_survives_complete_common_cost_support():
    values=refs();values["u"]=dict(candidate={},eligible_ref=None,G_ref=.001)
    models={"simple":dict(a=3,b=1,u=None),"few":dict(a=4,b=2,u=None)}
    support={m:set(values) for m in models}
    results=common_support_selection(values,models,support,"CROSS_FITTED_INTERNAL_GROUP")
    for result in results:
        assert result["undetermined_reference_candidates_in_common_support"]==["u"]
        assert result["excluded_direct_candidates"]==[]
        assert result["common_set_diagnostic"]["status"]=="REF_ELIGIBILITY_UNDETERMINED"
        assert result["common_set_diagnostic"]["regret"] is None
        assert result["common_set_diagnostic"]["known_eligible_subset_regret"]==1


def test_prelaunch_excluded_unknown_only_certifies_explicit_common_subset():
    values=refs();values["u"]=dict(candidate={},eligible_ref=None,G_ref=.001)
    result=common_support_selection(values,{"simple":dict(a=3,b=1)},{"simple":{"a","b","bad"}},"SYNTHETIC")[0]
    assert result["excluded_direct_candidates"]==["u"]
    assert result["full_set_diagnostic"]["regret"] is None
    assert result["common_set_diagnostic"]["regret"]==1
    assert result["common_set_diagnostic"]["registered_direct_count"]==3


def test_prelaunch_unknown_and_incomplete_predictions_both_retained():
    values=refs();values["u"]=dict(candidate={},eligible_ref=None,G_ref=1e-200)
    result=selection(values,dict(a=3),"SYNTHETIC")
    assert result["status"]=="REF_ELIGIBILITY_UNDETERMINED" and result["regret"] is None
    assert result["missing_candidates"]==["b"] and result["known_eligible_subset_regret"] is None
    assert result["known_eligible_prediction_subset_regret"]==0


def test_prelaunch_reference_shot_overflow_propagates_to_selection():
    overflow=reference_shots(dict(real=0,imag=0),1e308,.05)
    assert overflow["eligible_ref"] is None and overflow["status"]=="NUMERICAL_UNDEFINED"
    result=selection({"u":dict(candidate={},G_ref=None,**overflow)},dict(u=None),"SYNTHETIC")
    assert result["status"]=="REF_ELIGIBILITY_UNDETERMINED" and result["regret"] is None


def test_prelaunch_unknown_prediction_eligibility_is_not_all_rejected():
    result=selection(refs(),dict(a=1,b=2),"SYNTHETIC",predicted_eligible=dict(a=False,b=None))
    assert result["status"]=="PREDICTED_ELIGIBILITY_UNDETERMINED"
    assert result["undetermined_predicted_candidates"]==["b"] and result["regret"] is None


@pytest.mark.parametrize("bad",[0,"True",float("nan")])
def test_prelaunch_invalid_eligibility_state_stops(bad):
    values=refs();values["a"]["eligible_ref"]=bad
    with pytest.raises(c.Stop,match="SCHEMA"):selection(values,dict(a=1),"SYNTHETIC")


def test_prelaunch_all_callers_csv_status_summary_and_operational_na(monkeypatch):
    from trottertracks.resource_applicability import ax1b_analysis as analysis
    from trottertracks.resource_applicability.ax1b_evaluation import selection_status_summary
    rows=[saved_candidate(i) for i in range(210)]+[saved_candidate(1000+i,"DIAG_PM1_8",method="B0") for i in range(8)]+[saved_candidate(2000+i,"DIAG_M2_5") for i in range(5)]
    for index,marker in [(0,.123),(210,.124),(218,.125)]:rows[index].reference_bias["real"]=marker
    original=analysis.reference_shots
    def overflow_marked(bias,B,epsilon):
        return original(dict(real=0,imag=0),1e308,epsilon) if bias["real"] in {.123,.124,.125} else original(bias,B,epsilon)
    monkeypatch.setattr(analysis,"reference_shots",overflow_marked)
    result=analysis.analyze(rows,dict(metrics=dict(epsilon_anchors=[.05,.01,.005,.001])),{},dict(entries=[]))
    records=result["conditional_oracle_selection.csv"]
    for epsilon in [.05,.01,.005,.001]:
        diagnostic=[r for r in records if r.get("epsilon")==epsilon and r.get("row_kind")=="selection_diagnostic"]
        assert all(r["status"]=="REF_ELIGIBILITY_UNDETERMINED" and r["regret"] is None for r in diagnostic if r["fold_id"]=="full210")
        pooled=[r for r in diagnostic if r["fold_id"]=="POOLED_Q_CROSS_FITTED"]
        assert pooled and all(r["status"]=="REF_ELIGIBILITY_UNDETERMINED" and r["single_frozen_model"] is False for r in pooled)
        fold=[r for r in diagnostic if r["fold_id"]=="leave_one_q_out:1"]
        assert fold and all(r["status"]=="REF_ELIGIBILITY_UNDETERMINED" for r in fold)
    counts=selection_status_summary(records)
    assert counts["registered_set"]["REF_ELIGIBILITY_UNDETERMINED"]>0
    assert counts["common_support"]["REF_ELIGIBILITY_UNDETERMINED"]>0
    assert result["shot_availability.json"]["selection_status_counts"]==counts
    csv_text=ex.encode_output("conditional_oracle_selection.csv",records).decode()
    assert "REF_ELIGIBILITY_UNDETERMINED" in csv_text and "known_eligible_subset_regret" in csv_text
    assert all(p["G_operational_pred"] is None for p in result["predictions.jsonl"])


def test_prelaunch_unknown_status_and_forged_success_are_rejected():
    from trottertracks.resource_applicability.ax1b_evaluation import validate_selection_record
    result=selection(refs(),dict(a=3,b=1),"SYNTHETIC")
    validate_selection_record(result)
    for bad in [dict(result,status="UNREGISTERED_STATUS"),dict(result,undetermined_reference_candidates=["u"]),dict(result,status="REF_ELIGIBILITY_UNDETERMINED")]:
        with pytest.raises(c.Stop,match="SCHEMA"):validate_selection_record(bad)


def test_prelaunch_schemas_reject_unknown_status_and_accept_amendment():
    schemas=json.loads((Path(__file__).resolve().parents[3]/"artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/schemas_v2.json").read_text())["schemas"]
    values=refs();values["u"]=dict(candidate={},eligible_ref=None,G_ref=None)
    result=selection(values,dict(a=3,b=1),"SYNTHETIC")
    c.check_schema(result,schemas["selection_diagnostic"])
    with pytest.raises(c.Stop):c.check_schema(dict(result,status="UNKNOWN"),schemas["selection_diagnostic"])
    with pytest.raises(c.Stop):c.check_schema(dict(status="UNKNOWN",mandatory_stop=True,next_stage_authorized=False),schemas["terminal"])


def test_prelaunch_dependency_origin_fingerprint_is_required(approved):
    root,bundle,auth=approved;bundle["schema_version"]="track_a_ax1b_prelaunch_manifest_v1"
    auth["preparation_manifest_sha256"]=c.digest(bundle)
    auth["analysis_environment_synthetic_audit"]["environment"]={"wrong":"dependency origin"}
    with pytest.raises(c.Stop,match="ENVIRONMENT"):ex.authorize(root,bundle,auth,True,c.digest(auth))


def test_prelaunch_resource_enforcement_without_changing_test_process(monkeypatch):
    calls=[];handlers={}
    monkeypatch.setattr(ex.os,"environ",{})
    monkeypatch.setattr(ex.os,"sched_setaffinity",lambda pid,cpus:calls.append(("affinity",pid,cpus)))
    monkeypatch.setattr(ex.resource,"setrlimit",lambda kind,limits:calls.append(("rlimit",kind,limits)))
    monkeypatch.setattr(ex.signal,"signal",lambda sig,handler:handlers.update({sig:handler}))
    monkeypatch.setattr(ex.signal,"setitimer",lambda kind,value:calls.append(("alarm",kind,value)))
    ex._limits(dict(resources=dict(cpu_affinity=[2],ram_limit_bytes=123456,wall_time_limit_seconds=30)))
    assert calls==[("affinity",0,[2]),("rlimit",ex.resource.RLIMIT_AS,(123456,123456)),("alarm",ex.signal.ITIMER_REAL,30)]
    assert all(ex.os.environ[var]=="1" for var in ex.THREAD_VARS)
    with pytest.raises(c.Stop,match="BUDGET"):handlers[ex.signal.SIGALRM](0,None)


def test_prelaunch_cannot_report_empty_while_reference_members_unknown():
    from trottertracks.resource_applicability.ax1b_evaluation import validate_selection_record
    result=selection({"u":dict(candidate={},eligible_ref=None,G_ref=None)}, {},"SYNTHETIC")
    with pytest.raises(c.Stop,match="SCHEMA"):validate_selection_record(dict(result,status="REF_ELIGIBLE_EMPTY"))


@pytest.mark.parametrize("error",[ex.importlib.metadata.PackageNotFoundError("scipy"),OSError("missing dependency file")])
def test_prelaunch_missing_dependency_identity_has_environment_stop(monkeypatch,error):
    def fail(name):raise error
    monkeypatch.setattr(ex.importlib.metadata,"distribution",fail)
    with pytest.raises(c.Stop,match="ENVIRONMENT"):ex.environment()

# Results-prior compatibility coverage: all values below are synthetic.
OPTIONAL_COMPILER = ("backend_name", "layout_method", "routing_method")


def pm1_compiler_form():
    return dict(candidate()["compiler_identity"], **{k: None for k in OPTIONAL_COMPILER})


def test_identity_compatibility_both_forms_scope_and_raw_preserved():
    base = candidate()["compiler_identity"]
    extended = pm1_compiler_form()
    before = deepcopy((base, extended))
    assert canonical_compiler_identity(base) == canonical_compiler_identity(extended) == base
    row = candidate(800, rank=4)
    row["compiler_identity"] = extended
    row["candidate_fingerprint"] = c.digest({k: v for k, v in row.items() if k != "candidate_fingerprint"})
    frozen = deepcopy(row)
    validate_candidate_scope(row)
    a, b = toy_join()
    f, provenance = pm1_features(row, join_m1(a, b))
    assert f.n_fixed == 5 and provenance["n_fixed_status"] == "SOURCE_INVARIANT_SHARED"
    assert row == frozen and (base, extended) == before
    view = canonical_compiler_identity(extended)
    view["basis_gates"].append("mutation")
    assert (base, extended) == before


@pytest.mark.parametrize("key", OPTIONAL_COMPILER)
@pytest.mark.parametrize("value", ["specified", False, 0])
def test_identity_compatibility_optional_nonnull_rejected(key, value):
    raw = pm1_compiler_form(); raw[key] = value
    with pytest.raises(c.Stop, match="INPUT_IDENTITY"):
        canonical_compiler_identity(raw)


@pytest.mark.parametrize("present", [1, 2, 3, 4, 5, 6])
def test_identity_compatibility_partial_optional_keys_rejected(present):
    raw = candidate()["compiler_identity"]
    raw.update({k: None for i, k in enumerate(OPTIONAL_COMPILER) if present & (1 << i)})
    with pytest.raises(c.Stop, match="INPUT_IDENTITY"):
        canonical_compiler_identity(raw)


@pytest.mark.parametrize("key,value", [
    ("unknown_field", None), ("qiskit_version", "1.3.1"), ("basis_gates", ["rz", "sx", "cx", "x"]),
    ("optimization_level", 2), ("transpiler_seed", 18), ("coupling_map", [[0, 1]]),
    ("qiskit_version", 1.3), ("basis_gates", ("rz", "sx", "x", "cx")), ("basis_gates", "rz,sx,x,cx"),
    ("basis_gates", ["rz", "sx", 1, "cx"]), ("optimization_level", True), ("optimization_level", 1.0),
    ("transpiler_seed", 17.0), ("transpiler_seed", "17"), ("coupling_map", False),
])
def test_identity_compatibility_values_and_exact_types_rejected(key, value):
    raw = pm1_compiler_form(); raw[key] = value
    with pytest.raises(c.Stop, match="INPUT_IDENTITY"):
        canonical_compiler_identity(raw)


@pytest.mark.parametrize("value", [None, [], "compiler"])
def test_identity_compatibility_non_dictionary_rejected(value):
    with pytest.raises(c.Stop, match="INPUT_IDENTITY"):
        canonical_compiler_identity(value)


def test_identity_compatibility_missing_base_and_wrapper_still_rejected():
    for key in candidate()["compiler_identity"]:
        raw = pm1_compiler_form(); del raw[key]
        with pytest.raises(c.Stop): canonical_compiler_identity(raw)
    row = candidate(); row["compiler_identity"] = pm1_compiler_form(); row["wrapper_semantics"] = "other"
    with pytest.raises(c.Stop): validate_candidate_scope(row)


@pytest.mark.parametrize("key,value", [
    ("hamiltonian_hash", "other"), ("state_hash", "other"), ("state_vector_hash", "other"),
    ("snapshot_sha256", "other"), ("identity_policy", "other"), ("outer_formula", "other"),
    ("coefficient_atol", 1e-12), ("wrapper_semantics", "other"), ("T", 1.6), ("q", 2),
])
def test_identity_compatibility_other_anchor_conditions_not_relaxed(key, value):
    row = candidate(801, rank=4); row["compiler_identity"] = pm1_compiler_form(); row[key] = value
    a, b = toy_join()
    f, provenance = pm1_features(row, join_m1(a, b))
    assert f.n_fixed is None and provenance["n_fixed_status"] == "N_A_ANCHOR_INVARIANT_UNPROVEN"


def test_identity_compatibility_all_same_q_anchors_must_agree():
    a, b = toy_join(); anchors = join_m1(a, b)
    other = candidate(802, rank=6)
    row = candidate(803, rank=4); row["compiler_identity"] = pm1_compiler_form()
    anchors.append((other, dict(candidate_fingerprint=other["candidate_fingerprint"], n_fixed=6), {}))
    assert pm1_features(row, anchors)[0].n_fixed is None
    anchors[-1][1]["n_fixed"] = None
    assert pm1_features(row, anchors)[0].n_fixed is None
    anchors[-1][1]["n_fixed"] = 5
    assert pm1_features(row, anchors)[0].n_fixed == 5
    assert pm1_features(row, [])[0].n_fixed is None


def test_identity_compatibility_foreign_geometry_m2_and_non_b0_excluded():
    row = candidate(804, rank=4); row["compiler_identity"] = pm1_compiler_form()
    foreign = candidate(805, geometry="toy_m2")
    non_b0 = candidate(806, method="B1")
    anchors = [(x, dict(candidate_fingerprint=x["candidate_fingerprint"], n_fixed=99), {}) for x in (foreign, non_b0)]
    f, provenance = pm1_features(row, anchors)
    assert f.n_fixed is None and provenance["anchor_fingerprints"] == []


def test_identity_compatibility_anchor_uses_only_fixed_action_metadata():
    class FixedOnly(dict):
        def get(self, key, default=None):
            assert key == "n_fixed"
            return super().get(key, default)
    anchor = candidate(807)
    signal = FixedOnly(n_fixed=5, candidate_fingerprint=anchor["candidate_fingerprint"],
                       reference_bias="FORBIDDEN", axis_shots="FORBIDDEN", compiled_cost="FORBIDDEN")
    row = candidate(808, rank=4); row["compiler_identity"] = pm1_compiler_form()
    f, provenance = pm1_features(row, [(anchor, signal, {"cost": "FORBIDDEN"})])
    assert f.n_fixed == 5 and provenance["no_truth_or_cost_imputation"]


def test_identity_compatibility_synthetic_210_and_5_scope_regression():
    for count, geometry in ((210, "toy_m1"), (5, "toy_m2")):
        rows = [candidate(900+i, geometry=geometry, q=[1, 2, 4, 8][i % 4], method=["B0", "B1", "B2", "B3"][i % 4]) for i in range(count)]
        for row in rows: validate_candidate_scope(row)
        check_membership(rows, deepcopy(rows), geometry)
        changed = deepcopy(rows); changed[0]["compiler_identity"]["transpiler_seed"] = 18
        with pytest.raises(c.Stop): check_membership(changed, rows, geometry)
        with pytest.raises(c.Stop): validate_candidate_scope(changed[0])


def test_identity_compatibility_raw_join_remains_exact():
    a, b = toy_join(); b["compile_map"][0]["candidate"]["compiler_identity"] = pm1_compiler_form()
    # Only scope/anchor comparison canonicalizes; stored full identity joins remain exact.
    with pytest.raises(c.Stop): join_m1(a, b)


def test_identity_compatibility_saved_projection_pm1_extended_form():
    values, allow = toy_saved_values()
    row = values["track_a_pm1_discard_result_v1"]["candidate_records"][0]["candidate"]
    row["compiler_identity"] = pm1_compiler_form()
    frozen = deepcopy(values)
    projected = project_saved(values, allow)
    assert next(r for r in projected if r.dataset == "DIAG_PM1_8").features.n_fixed == 5
    assert values == frozen


def test_identity_preflight_shapes_only_no_projection_fit_or_reference_calculation(monkeypatch):
    from trottertracks.resource_applicability import ax1b_preflight as pf, ax1b_data as data, ax1b_models as models
    def forbidden(*args, **kwargs): raise AssertionError("preflight attempted numerical analysis")
    for module, names in [(data, ["project_saved", "features_from_signal"]), (models, ["fit_cost"]), (ex, ["execute"])]:
        for name in names: monkeypatch.setattr(module, name, forbidden)
    values, allow = toy_saved_values()
    values["track_a_pm1_discard_result_v1"]["candidate_records"][0]["candidate"]["compiler_identity"] = pm1_compiler_form()
    frozen = deepcopy(values)
    audit = pf.inspect_metadata(values, allow)
    assert audit["precheck_status"] == "PRECHECK_PASS"
    assert [v["count"] for v in audit["datasets"].values()] == [2, 1, 1]
    assert audit["pm1_anchor_audit"][0]["n_fixed_shared"]
    assert all(x["status"] == "PRECHECK_UNVERIFIED" for x in audit["declared_unverified"])
    assert not audit["actual_feature_values_exported"] and values == frozen


def test_identity_preflight_missing_action_metadata_stays_na():
    from trottertracks.resource_applicability import ax1b_preflight as pf
    values, allow = toy_saved_values()
    values["pr2_matched_accuracy_m1_a_result_v2"]["signal_records"][0]["n_fixed"] = None
    audit = pf.inspect_metadata(values, allow)
    assert not audit["pm1_anchor_audit"][0]["n_fixed_shared"]
    assert audit["pm1_anchor_audit"][0]["n_fixed_status"] == "N_A_ANCHOR_INVARIANT_UNPROVEN"


@pytest.mark.parametrize("failure", ["compiler", "wrapper", "fingerprint", "field"])
def test_identity_preflight_failures_stop_without_fit(failure, monkeypatch):
    from trottertracks.resource_applicability import ax1b_preflight as pf, ax1b_models as models
    monkeypatch.setattr(models, "fit_cost", lambda *a, **k: pytest.fail("fit after preflight failure"))
    values, allow = toy_saved_values()
    pm = values["track_a_pm1_discard_result_v1"]["candidate_records"][0]
    if failure == "compiler": pm["candidate"]["compiler_identity"] = dict(pm1_compiler_form(), routing_method="sabre")
    elif failure == "wrapper": pm["candidate"]["wrapper_semantics"] = "other"
    elif failure == "fingerprint": pm["signal_cost_candidate_fingerprint"] = "other"
    else: del pm["signal"]["axis_bias"]
    with pytest.raises(c.Stop): pf.inspect_metadata(values, allow)


def test_identity_metadata_reader_requires_explicit_mode_and_hash_schema(tmp_path):
    from trottertracks.resource_applicability import ax1b_preflight as pf
    value = dict(schema_version="toy", required=None)
    entry = allow_entry("toy.json", value)
    (tmp_path/"toy.json").write_text(c.canonical(value)+"\n")
    with pytest.raises(c.Stop): pf.MetadataReader(tmp_path, dict(entries=[entry]))
    reader = pf.MetadataReader(tmp_path, dict(entries=[entry]), metadata_only=True)
    assert reader.read("toy.json") == value
    assert reader.audit[0]["schema_checked"]
    with pytest.raises(c.Stop): reader.read("outside.json")
    (tmp_path/"toy.json").write_text("{}")
    with pytest.raises(c.Stop): reader.read("toy.json")


def test_identity_preflight_cli_default_denial_without_reads(monkeypatch, capsys):
    import importlib.util
    path = Path(__file__).resolve().parents[3]/"scripts/resource_applicability/run_track_a_ax1b_identity_preflight.py"
    spec = importlib.util.spec_from_file_location("synthetic_identity_cli", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    reads = []
    monkeypatch.setattr(Path, "read_bytes", lambda p: reads.append(str(p)))
    assert module.main([]) == 2
    assert reads == [] and "AX1B_STOP_AUTHORIZATION" in capsys.readouterr().out


def test_identity_new_manifest_requires_exact_synthetic_environment(approved):
    root, bundle, auth = approved
    bundle["schema_version"] = "track_a_ax1b_identity_compatibility_manifest_v1"
    auth["preparation_manifest_sha256"] = c.digest(bundle)
    with pytest.raises(c.Stop, match="ENVIRONMENT"):
        ex.authorize(root, bundle, auth, True, c.digest(auth))


def test_identity_preflight_request_denies_analysis_flags_before_any_input(monkeypatch):
    from trottertracks.resource_applicability import ax1b_preflight as pf
    attempts = []
    monkeypatch.setattr(pf.MetadataReader, "read", lambda *a: attempts.append(a))
    request = dict(schema_version="track_a_ax1b_metadata_precheck_request_v1", metadata_only_preflight_authorized=True, **c.FLAGS)
    request["ax1b_analysis_authorized"] = True
    with pytest.raises(c.Stop, match="AUTHORIZATION"): pf.run_preflight(Path("/tmp"), request, [])
    assert attempts == []


def test_identity_preflight_request_bad_source_stops_before_science_reader(tmp_path, monkeypatch):
    from trottertracks.resource_applicability import ax1b_preflight as pf
    attempts = []
    monkeypatch.setattr(pf.MetadataReader, "read", lambda *a: attempts.append(a))
    (tmp_path/"toy.py").write_text("synthetic source")
    request = dict(schema_version="track_a_ax1b_metadata_precheck_request_v1", metadata_only_preflight_authorized=True,
                   source_files=[dict(path="toy.py", sha256="0"*64)], **c.FLAGS)
    with pytest.raises(c.Stop, match="IMPLEMENTATION"): pf.run_preflight(tmp_path, request, [])
    assert attempts == []


def test_identity_preflight_boundary_rejects_imports_and_real_calculation_entry():
    import importlib.util
    path = Path(__file__).resolve().parents[3]/"scripts/resource_applicability/run_track_a_ax1b_identity_preflight.py"
    spec = importlib.util.spec_from_file_location("synthetic_identity_boundary", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    boundary = module.NumericalBoundary()
    with pytest.raises(c.Stop): boundary.find_spec("trotterlib.rte", None)
    with pytest.raises(c.Stop): boundary.find_spec("trottertracks.resource_applicability.ax1b_analysis", None)
    assert boundary.import_attempts == 2
    frame = SimpleNamespace(f_globals={"__name__": "trottertracks.resource_applicability.ax1b_evaluation"},
                            f_code=SimpleNamespace(co_name="reference_shots"))
    with pytest.raises(c.Stop): boundary.profile(frame, "call", None)
    assert boundary.call_attempts == 1
