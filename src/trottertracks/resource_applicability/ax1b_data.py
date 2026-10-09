"""Allowlisted saved-value projection. Called only after execution authorization."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math

from .ax1b_contract import AXES, METRICS, digest, number, require, safe_path, select_fields, sha256
from .ax1b_models import Features


@dataclass(frozen=True)
class SavedCandidate:
    dataset: str
    candidate: dict
    features: Features
    feature_provenance: dict
    reference_costs: dict
    reference_bias: dict
    B: float
    pairs_by_metric: dict
    finite_input: dict | None

    @property
    def fingerprint(self):
        return self.candidate["candidate_fingerprint"]


class VerifiedReader:
    def __init__(self, root, allowlist, permit):
        require(permit is not None and permit.get("authorized") is True, "AUTHORIZATION", "science read requires prior permit")
        self.root, self.entries = root, {e["path"]:e for e in allowlist["entries"]}
        require(len(self.entries)==len(allowlist["entries"]),"SCHEMA","duplicate allowlist path")
        self.input_audit=[]

    def read(self, path):
        require(path in self.entries, "INPUT_IDENTITY", "path not in exact allowlist")
        entry=self.entries[path]
        raw=safe_path(self.root,path).read_bytes()
        require(sha256(raw)==entry["sha256"],"INPUT_IDENTITY","input hash differs")
        schema=entry["schema"]
        value=raw
        if schema["format"]=="json":
            value=json.loads(raw)
            require(sorted(value)==schema["root_keys"] and value.get("schema_version")==schema["schema_version"],"SCHEMA","JSON root/version mismatch")
            for field in entry["planned_fields"]:
                values=select_fields(value,field["selector"])
                if field["required"]:
                    require(bool(values),"SCHEMA","missing required field: "+field["selector"])
                if "static_presence" in field:
                    require(len(values)==field["static_presence"]["matched_value_count"] and sum(v is None for v in values)==field["static_presence"]["null_value_count"],
                            "SCHEMA","field presence/cardinality changed")
        elif schema["format"]=="csv":
            require(next(csv.reader(io.StringIO(raw.decode())))==schema["ordered_header"],"SCHEMA","CSV header mismatch")
        self.input_audit.append(dict(path=path,sha256=sha256(raw),schema_checked=True,embedded_paths_followed=False))
        return value


def _index(rows, candidate=lambda r:r):
    output={}
    for row in rows:
        c=candidate(row)
        fp=c.get("candidate_fingerprint")
        require(isinstance(fp,str) and bool(fp) and fp not in output,"INPUT_IDENTITY","duplicate/missing candidate fingerprint")
        output[fp]=row
    return output


def join_m1(a,b):
    ledger=_index(a["candidate_ledger"])
    signal=_index(a["signal_records"],lambda r:r["candidate"])
    compiled=_index(b["compile_map"],lambda r:r["candidate"])
    require(set(ledger)==set(signal)==set(compiled),"INPUT_IDENTITY","missing or foreign M1 candidate")
    output=[]
    for fp in sorted(ledger):
        require(ledger[fp]==signal[fp]["candidate"]==compiled[fp]["candidate"] and signal[fp]["candidate_fingerprint"]==fp,
                "INPUT_IDENTITY","M1 full candidate identity differs")
        output.append((ledger[fp],signal[fp],compiled[fp]))
    return output


def check_membership(candidates, expected_rows, label):
    actual=_index(candidates)
    frozen=_index(expected_rows)
    require(set(actual)==set(frozen),"INPUT_IDENTITY",f"{label} membership changed")
    for fp,c in actual.items():
        for key,value in frozen[fp].items():
            if key not in {"identity_provenance","development_candidate_fingerprint"}:
                require(c.get(key)==value,"INPUT_IDENTITY",f"{label} identity {key} differs")


def validate_candidate_scope(c):
    require(type(c["q"]) is int and c["q"]>0 and type(c["r"]) is int and c["r"]>=0,"INPUT_IDENTITY","invalid q/r identity")
    require(type(c["K"]) is int and c["K"]>=0 and c["K"]%2==0,"INPUT_IDENTITY","invalid cutoff identity")
    if c["method"] in {"B2","B3"}:
        require(c["r"]>0,"INPUT_IDENTITY","random r must be positive")
    require(c["delta"]==c["T"]/c["q"],"INPUT_IDENTITY","T/delta/q relation differs")
    for key in ["T","delta"]:
        number(c[key],key,positive=True)
        if c.get(key+"_hex") is not None:
            require(c[key+"_hex"]==float(c[key]).hex(),"INPUT_IDENTITY","literal/hex identity differs")
    expected=dict(basis_gates=["rz","sx","x","cx"],coupling_map=None,optimization_level=1,qiskit_version="1.3.0",transpiler_seed=17)
    require(c.get("compiler_identity")==expected and c.get("wrapper_semantics")=="full_measured_hadamard_wrapper_without_state_preparation",
            "INPUT_IDENTITY","compiler/full wrapper identity differs")


def features_from_signal(c,s):
    n_det,n_fixed=s.get("n_det"),s.get("n_fixed")
    if n_det is not None:
        require(n_det==2*c["q"]*c["rank"],"INPUT_IDENTITY","n_det/source prefix action invariant")
    if c["method"] in {"B0","B1"}:
        E=0.0
    else:
        E=s.get("expected_random_applications_exact")
        if E is None:
            dist=s.get("finite_distribution",{})
            orders,probs=dist.get("orders"),dist.get("order_probabilities")
            if orders is not None and probs is not None:
                require(len(orders)==len(probs),"SCHEMA","finite probability alignment")
                E=c["q"]*c["r"]*math.fsum(number(p,"p_n")*(n+1) for n,p in zip(orders,probs))
    return Features(n_det,E,n_fixed,c["q"]),dict(information_class="I1",source="allowlisted action fields/source invariant",input_procurement_new_size="UNKNOWN")


def pm1_features(c,anchors):
    keys=("hamiltonian_hash","state_hash","state_vector_hash","snapshot_sha256","identity_policy","outer_formula","coefficient_atol","compiler_identity","wrapper_semantics","q","T")
    matches=[s for ac,s,_ in anchors if ac["method"]=="B0" and all(ac.get(k)==c.get(k) for k in keys)]
    fixed=[s.get("n_fixed") for s in matches]
    value=fixed[0] if fixed and None not in fixed and len(set(fixed))==1 else None
    return Features(2*c["q"]*c["rank"],0.0,value,c["q"]),dict(information_class="I1",source="frozen _prepare_discard invariant: full one-body/constant; empty tail",
             anchor_fingerprints=[s["candidate_fingerprint"] for s in matches],n_fixed_status="SOURCE_INVARIANT_SHARED" if value is not None else "N_A_ANCHOR_INVARIANT_UNPROVEN",
             no_truth_or_cost_imputation=True,input_procurement_new_size="UNKNOWN")


def _axis_scope(axis):
    require(axis.get("state_preparation_included") is False and axis.get("measurement_included") is True and axis.get("backend_execution_included") is False,
            "INPUT_IDENTITY","wrapper scope mismatch")
    require(axis.get("trajectory_records_truncated") is False,"PAIR_IDENTITY","truncated trajectory records")


def _pairs(axes):
    pairs={}
    for a in AXES:
        _axis_scope(axes[a])
    for metric in METRICS:
        pairs[metric]=tuple([
            dict(trajectory_index=p["trajectory_index"],trajectory_seed=p["trajectory_seed"],
                 step_seeds=p["step_seeds"],evolution_circuit_semantics_fingerprint=p["evolution_circuit_semantics_fingerprint"],cost=p["cost"][metric])
            for p in axes[a]["retained_trajectory_records"]] for a in AXES)
    return pairs


def _m2_pairs(rows):
    pairs={}
    for metric in METRICS:
        pairs[metric]=tuple([
            dict(trajectory_index=p["trajectory_index"],trajectory_seed=p["trajectory_seed"],step_seeds=p["axes"][a]["step_seeds"],
                 evolution_circuit_semantics_fingerprint=p["axes"][a]["shared_evolution_fingerprint"],cost=p["axes"][a]["metrics"][metric]) for p in rows] for a in AXES)
    return pairs


def project_saved(values, allowlist):
    """Called by authorized analysis only. Schema adapters; no scientific actions."""
    def by_schema(name):
        found=[values[e["path"]] for e in allowlist["entries"] if e["schema"]["schema_version"]==name]
        require(len(found)==1,"SCHEMA","unique input schema missing")
        return found[0]
    A=by_schema("pr2_matched_accuracy_m1_a_result_v2")
    B=by_schema("pr2_matched_accuracy_m1_b1_result_v2")
    P=by_schema("track_a_pm1_discard_result_v1")
    M=by_schema("pr2_matched_accuracy_m2_transfer_result_v2")
    anchors=join_m1(A,B)
    check_membership([c for c,_,_ in anchors],allowlist["membership"]["TRAIN_M1_210"]["rows"],"M1")
    check_membership([r["candidate"] for r in P["candidate_records"]],allowlist["membership"]["DIAG_PM1_8"]["rows"],"PM1")
    output=[]
    for c,s,b in anchors:
        f,provenance=features_from_signal(c,s)
        output.append(SavedCandidate("TRAIN_M1_210",c,f,provenance,{a:b["compiled_axes"][a]["cost"] for a in AXES},s["axis_bias"],s["normalization_multiplier"],_pairs(b["compiled_axes"]),s if c["method"] in {"B2","B3"} else None))
    for r in P["candidate_records"]:
        c,s=r["candidate"],r["signal"]
        require(c["candidate_fingerprint"]==s["candidate_fingerprint"]==r["signal_cost_candidate_fingerprint"],"INPUT_IDENTITY","PM1 signal/compile identity")
        f,provenance=pm1_features(c,anchors)
        output.append(SavedCandidate("DIAG_PM1_8",c,f,provenance,{a:r["axes"][a]["cost"] for a in AXES},s["axis_bias"],s["normalization_multiplier"],_pairs(r["axes"]),None))
    identities=[]
    for r in M["candidate_results"]:
        s=r["signal"];c=s["candidate"]
        require(r["execution_candidate_fingerprint"]==s["candidate_fingerprint"]==c["candidate_fingerprint"],"INPUT_IDENTITY","M2 execution identity mismatch")
        require(r["development_candidate_fingerprint"] in {x[0]["candidate_fingerprint"] for x in anchors},"INPUT_IDENTITY","M2 development reference missing")
        c=dict(c,**{k:M["input_snapshot_identity"][k] for k in ["hamiltonian_hash","state_hash","state_vector_hash"]},snapshot_sha256=M["input_snapshot_identity"]["file_sha256"],
               T_hex=None,delta_hex=None,T_literal=c["T"],delta_literal=c["delta"])
        identities.append(c)
        f,provenance=features_from_signal(c,s)
        output.append(SavedCandidate("DIAG_M2_5",c,f,provenance,r["axis_one_shot_compiled_means"],s["axis_bias"],s["normalization_multiplier"],_m2_pairs(r["compiled"]["paired_trajectory_rows"]),s if c["method"] in {"B2","B3"} else None))
    check_membership(identities,allowlist["membership"]["DIAG_M2_5"]["rows"],"M2")
    require(len({r.fingerprint for r in output})==len(output),"INPUT_IDENTITY","cross-dataset duplicate identity")
    for r in output:
        validate_candidate_scope(r.candidate)
    return output


def folds(training):
    """Metadata partitions only. All inputs must be the registered M1 layer."""
    require(all(r.dataset=="TRAIN_M1_210" for r in training),"INPUT_IDENTITY","PM1/M2 cannot enter training folds")
    output=[]
    for name,key,groups in [("leave_one_q_out","q",[1,2,4,8]),("leave_one_prefix_out","rank",[0,3,6,9,12]),("leave_one_method_out","method",["B0","B1","B2","B3"]),("leave_one_random_K_out","K",[2,4])]:
        for group in groups:
            def held(r):
                return r.candidate[key]==group and (name!="leave_one_random_K_out" or r.candidate["method"] in {"B2","B3"})
            test=[r for r in training if held(r)];train=[r for r in training if not held(r)]
            output.append(dict(fold_id=f"{name}:{group}",diagnostic_family=name,group=group,train=train,test=test,
                               diagnostic_kind="INTERNAL_GROUP_DIAGNOSTIC",single_frozen_model=False))
    return output
