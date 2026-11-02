"""New synthetic checks only. Never run the historical v1 129-case runner."""
import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from collections import Counter
from jsonschema import Draft202012Validator, ValidationError

HERE = Path(__file__).parent
OLD = HERE.parent / "2026-10-06"
PROTECTED_ATTEMPTS = []
IMPORT_ATTEMPTS = []
READ_EVENTS = []


def install_guards():
    original_import = builtins.__import__
    forbidden = {"numpy", "scipy", "qiskit", "openfermion", "openfermionpyscf",
                 "pyscf", "trotterlib", "trottertracks", "torch", "cupy"}
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            IMPORT_ATTEMPTS.append(name)
            raise RuntimeError("scientific import forbidden")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    def protected(path):
        p = Path(os.fsdecode(path))
        return p.suffix.lower() in {".npz", ".npy", ".pickle", ".pkl", ".db", ".sqlite", ".sqlite3"} or any(
            part in {"runtime", ".runtime", "checkpoint", "checkpoints", "cache", "caches", "__pycache__"}
            for part in p.parts)
    original_stat = os.stat
    def guarded_stat(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and protected(path):
            PROTECTED_ATTEMPTS.append(str(path)); raise RuntimeError("protected stat forbidden")
        return original_stat(path, *args, **kwargs)
    os.stat = guarded_stat
    def audit(event, args):
        if event in {"subprocess.Popen", "os.system", "os.kill", "os.killpg"}:
            PROTECTED_ATTEMPTS.append(event); raise RuntimeError("process/job access forbidden")
        if event in {"open", "os.listdir", "os.scandir"} and args and isinstance(args[0], (str, bytes, os.PathLike)):
            if protected(args[0]):
                PROTECTED_ATTEMPTS.append(str(args[0])); raise RuntimeError("protected input forbidden")
            if event == "open": READ_EVENTS.append(os.fsdecode(args[0]))
    sys.addaudithook(audit)


def main():
    started = time.monotonic()
    install_guards()
    # Compile explicitly permitted source text in memory. Ordinary module loading
    # probes a .pyc cache even under -B; never probe or materialize those paths.
    def source_module(name, path):
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        return module
    source_module("contract_validator_v1", OLD/"contract_validator_v1.py")
    v = source_module("contract_validator_v2", HERE/"contract_validator_v2.py")
    oldv = v.inherited
    load = lambda p: json.loads(p.read_text())
    schema_names = ["scope", "plan", "stage_contract", "memory_observation", "checkpoint",
                    "completion_ledger", "numerical_circuit", "result"]
    schemas = {name: load(HERE / (name+"_schema_v2.json")) for name in schema_names}
    validators = {name: Draft202012Validator(s) for name, s in schemas.items()}
    cases = []
    def accept(group, name, fn):
        fn(); cases.append({"group":group,"name":name,"expected":"accept","passed":True})
    def reject(group, name, fn):
        try: fn()
        except (ValueError, ValidationError, KeyError, TypeError) as error:
            cases.append({"group":group,"name":name,"expected":"reject","passed":True,"reason":str(error).splitlines()[0]})
            return
        raise AssertionError("unexpected acceptance: "+name)
    def equals(actual, expected):
        if actual != expected: raise AssertionError((actual, expected))
    for name,schema in schemas.items():
        accept("schema",name+" structure",lambda s=schema:Draft202012Validator.check_schema(s))
    plan=load(HERE/"zero_compute_plan_v2.json")
    accept("scope","exact v2 scope",lambda:validators["scope"].validate(load(HERE/"scope_v2.json")))
    accept("plan","exact v2 plan+digest+coverage",lambda:v.validate_zero_plan(plan,validators["plan"]))
    stage=load(HERE/"stage_contract_v2.json")
    accept("stage","exact stage contract",lambda:validators["stage_contract"].validate(stage))
    accept("stage","plan and stage JSON agree",lambda:equals(plan["authorization_order"],stage))
    # Mutate and recompute the digest so scope/authorization failures do not rely on a stale hash.
    mutations={"distances_angstrom":["0.70"],"source_port_authorized":True,
        "input_generation_authorized":True,"science_execution_authorized":True,
        "execution_plan_sealed":True,"input_generation_plan_sealed":True,
        "next_stage_authorized":True,"automatic_research_decision_authorized":True,
        "research_decision":"GO","mandatory_stop":False,"master_seed":20261006,
        "actual_science_source_commit":"a"*40,"actual_science_source_hashes":{},
        "new_snapshot_identities":{},"execution_authorization":{},
        "input_generation_authorization":{},"signal_compile_authorization":{},
        "science_output_root":"/tmp/science","science_output_directory_created":True,
        "runtime_cache_registry_created":True,"contract_conditions_fully_closed":True,
        "actual_run_id":"fake","actual_allowed_cpu_count":12,"observed_memory_reserved":True,
        "unresolved_decisions":[],"additional_candidate":{},"stage":"SIGNAL_LAUNCH_APPROVED"}
    for field,value in mutations.items():
        changed=copy.deepcopy(plan);changed[field]=value;changed["plan_fingerprint"]=v.plan_fingerprint(changed)
        reject("plan","reject changed "+field,lambda p=changed:v.validate_zero_plan(p,validators["plan"]))
    changed=copy.deepcopy(plan);changed["plan_fingerprint"]="0"*64
    reject("plan","reject plan digest mismatch",lambda:v.validate_zero_plan(changed,validators["plan"]))
    for label,modify in [
        ("remove template",lambda p:p["templates"].pop()),
        ("duplicate template",lambda p:p["templates"].append(copy.deepcopy(p["templates"][0]))),
        ("replace template",lambda p:p["templates"][0].update(L_D=2)),
        ("compile cap 74785",lambda p:p["future_resource_caps"].update(actual_science_transpile_invocation_cap=74785)),
        ("trajectory 96",lambda p:p["future_resource_caps"].update(random_trajectories_per_cell=96)),
        ("ancilla count8",lambda p:p["model"].update(ancilla_count=8)),
        ("memory headroom0",lambda p:p["review_proposals"]["D4_MEMORY_WALL_OUTPUT"].update(fixed_host_headroom_bytes=0)),
        ("generation can compile",lambda p:p["authorization_order"]["input_generation"].update(signal_sampling_build_compile_permitted=True)),
        ("seal null inputs",lambda p:p.update(execution_plan_sealed=True)),
        ("workers13",lambda p:p["future_resource_caps"].update(cpu_worker_max=13)),
    ]:
        changed=copy.deepcopy(plan);modify(changed);changed["plan_fingerprint"]=v.plan_fingerprint(changed)
        reject("scope",label,lambda p=changed:v.validate_zero_plan(p,validators["plan"]))
    context=load(HERE/"synthetic_stage_fixtures_v2.json")
    for current,action,target in zip(v.STAGES[:-1],v.ACTIONS,v.STAGES[1:]):
        accept("stage","valid hypothetical "+action,lambda c=current,a=action,t=target:equals(v.validate_transition(c,a,context),t))
    # Explicitly reject all actions except the single ordered successor at every stage.
    forbidden=("input_generation","signal","sampling","build","compile","transpile","retry","resume")
    for index,current in enumerate(v.STAGES):
        for action in forbidden:
            reject("stage_preauthorization",current+" rejects "+action,lambda c=current,a=action:v.validate_transition(c,a,context))
        for offset in (2,3):
            if index+offset<len(v.ACTIONS):
                action=v.ACTIONS[index+offset]
                reject("stage_skip",current+" rejects "+action,lambda c=current,a=action:v.validate_transition(c,a,context))
    prerequisite_stage={"contract_conditions_approved":0,"separate_implementation_instruction":1,
        "synthetic_source_gates_passed":1,"generation_plan_source_bound":3,
        "generation_authorization_reviewed":4,"generation_explicit_launch":5,
        "resources_admitted":5,"generation_numerical_gates_passed":6,
        "generation_complete_stop":6,"input_bound_signal_plan_sealed":7,
        "signal_authorization_reviewed":8,"signal_explicit_launch":9}
    for flag,index in prerequisite_stage.items():
        altered=copy.deepcopy(context);altered["flags"][flag]=False
        reject("stage_prerequisite","missing "+flag,lambda x=altered,i=index:v.validate_transition(v.STAGES[i],v.ACTIONS[i],x))
    for field in v.HASH_FIELDS:
        altered=copy.deepcopy(context);altered["frozen_inputs"][0][field]=None
        reject("input_freeze","null "+field+" cannot seal",lambda x=altered:v.validate_transition(v.STAGES[7],v.ACTIONS[7],x))
    for label,modify,index in [
        ("missing one distance",lambda c:c["frozen_inputs"].pop(),7),
        ("duplicate distance",lambda c:c["frozen_inputs"][1].update(distance_angstrom="0.70"),7),
        ("null source",lambda c:c.update(source_commit=None),2),
        ("foreign source",lambda c:c.update(source_commit="b"*40),3),
        ("missing source hashes",lambda c:c.update(source_hashes={}),3),
        ("foreign generation plan",lambda c:c.update(generation_plan_fingerprint="f"*64),4),
        ("foreign input-bound plan",lambda c:c.update(signal_plan_fingerprint="f"*64),8),
        ("generation authorization used for signal",lambda c:c.update(signal_authorization_example=copy.deepcopy(c["generation_authorization_example"])),8),
        ("actual signal before authorization",lambda c:c.update(actual_signal_or_cost_before_authorization=True),1),
        ("attempt real authorization",lambda c:c.update(actual_authorizations=["REAL"]),4),
        ("missing synthetic marker",lambda c:c.update(artifact_scope="SCIENCE"),0),
    ]:
        altered=copy.deepcopy(context);modify(altered)
        reject("stage_binding",label,lambda x=altered,i=index:v.validate_transition(v.STAGES[i],v.ACTIONS[i],x))
    null_inputs=copy.deepcopy(context);null_inputs["frozen_inputs"]=None
    accept("stage","generation plan can precede new input hashes",lambda:v.validate_transition(v.STAGES[3],v.ACTIONS[3],null_inputs))
    for kind,index in [("generation",4),("signal",8)]:
        for field,val in [("result_prior",False),("source_commit","f"*40),("plan_fingerprint","f"*64),("marker","REAL"),("scope","ALL_SCIENCE")]:
            altered=copy.deepcopy(context);altered[kind+"_authorization_example"][field]=val
            reject("stage_binding",kind+" example rejects "+field,lambda x=altered,i=index:v.validate_transition(v.STAGES[i],v.ACTIONS[i],x))
    def observation(available,cpus=12,requested=12,age=0,pressure=False):
        return {"artifact_scope":"SYNTHETIC_CONTRACT_TEST_ONLY","available_bytes":available,
                "allowed_cpu_count":cpus,"requested_workers":requested,
                "observation_age_seconds":age,"memory_pressure":pressure}
    for workers in range(1,13):
        need=(24+8*workers)*v.GIB
        accept("memory_admission",f"exact boundary admits {workers}",lambda n=need,w=workers:equals(v.select_workers(observation(n),validators["memory_observation"]),w))
        if workers==1:
            reject("memory_admission","one byte below minimum STOP",lambda:v.select_workers(observation(need-1),validators["memory_observation"]))
        else:
            accept("memory_admission",f"one byte below {workers} lowers worker",lambda n=need,w=workers:equals(v.select_workers(observation(n-1),validators["memory_observation"]),w-1))
    for available,cpus,requested,w in [(64,12,12,5),(120,12,12,12),(1024,3,12,3),(120,12,4,4),(32,1,12,1),(120,1,12,1)]:
        accept("memory_admission",f"{available}GiB CPU{cpus} requested{requested} => {w}",lambda a=available,c=cpus,r=requested,e=w:equals(v.select_workers(observation(a*v.GIB,c,r),validators["memory_observation"]),e))
    for field,value in [("available_bytes",-1),("available_bytes",True),("allowed_cpu_count",0),
                        ("allowed_cpu_count",True),("requested_workers",13),("requested_workers",0),
                        ("observation_age_seconds",6),("observation_age_seconds",True),
                        ("memory_pressure",True),("artifact_scope","REAL")]:
        altered=observation(120*v.GIB);altered[field]=value
        reject("memory_admission","reject "+field+"="+str(value),lambda x=altered:v.select_workers(x,validators["memory_observation"]))
    healthy={"available_bytes":16*v.GIB,"worker_rss_bytes":[8*v.GIB]*12,
             "driver_rss_bytes":8*v.GIB,"admitted_workers":12,"psi_full_avg10":0,"oom_event_delta":0}
    accept("memory_pressure","exact RSS/headroom boundaries healthy",lambda:equals(v.pressure_requires_stop(**healthy),False))
    for label,modify in [("available below headroom",lambda c:c.update(available_bytes=16*v.GIB-1)),
        ("worker RSS exceeded",lambda c:c["worker_rss_bytes"].__setitem__(0,8*v.GIB+1)),
        ("driver RSS exceeded",lambda c:c.update(driver_rss_bytes=8*v.GIB+1)),
        ("PSI full positive",lambda c:c.update(psi_full_avg10=0.001)),
        ("OOM event",lambda c:c.update(oom_event_delta=1))]:
        altered=copy.deepcopy(healthy);modify(altered)
        accept("memory_pressure",label+" => STOP",lambda x=altered:equals(v.pressure_requires_stop(**x),True))
    gates={"state_norm_absolute":1e-12,"reference_residual_L2_Ha":1e-9,"relative_Hermiticity":1e-12,
           "imaginary_energy_Ha":1e-11,"outside_sector_max":1e-12,"full_sector_consistency_max":1e-12,
           "sector_gap_Ha":math.nextafter(1e-10,math.inf)}
    accept("numerical_gates","inclusive maxima, gap strictly greater",lambda:v.validate_numeric_gate_observations(gates))
    for field in gates:
        altered=dict(gates);altered[field]=1e-10 if field=="sector_gap_Ha" else math.nextafter(gates[field],math.inf)
        reject("numerical_gates","boundary violation "+field,lambda x=altered:v.validate_numeric_gate_observations(x))
    for value in [math.nan,math.inf,-1,True]:
        altered=dict(gates);altered["state_norm_absolute"]=value
        reject("numerical_gates","nonfinite/invalid observation "+str(value),lambda x=altered:v.validate_numeric_gate_observations(x))
    # Use unchanged *artificial* v1 records. Do not re-execute either v1 runner.
    fixture=load(OLD/"synthetic_records_v1.json")
    owner,reuse,baseline=fixture["records"]
    circuit=fixture["declarative_circuit"]
    records={r["wrapper_key"]:r for r in fixture["records"]}
    ledger=fixture["external_ledger"];registry=fixture["independent_numerical_registry"]
    master=fixture["master_seed_is_artificial"]
    def check(record=owner,circ=circuit,rs=records,led=ledger,reg=registry,expected=None):
        if expected is None: expected={k:owner[k] for k in oldv.IDENTITY_FIELDS}
        return v.validate_record(record,expected,circ,rs,led,reg,master,validators["checkpoint"])
    for name,record in [("owner",owner),("reuse",reuse),("baseline",baseline)]:
        accept("v1_identity_cache",name+" accepted by unchanged verifier",lambda r=record:check(r,expected={k:r[k] for k in oldv.IDENTITY_FIELDS}))
    for field in oldv.IDENTITY_FIELDS:
        altered=copy.deepcopy(owner);altered[field]=None
        reject("v1_identity_cache","reject null identity "+field,lambda r=altered:check(r))
    for label,modify in [("wrapper key",lambda r:r.update(wrapper_key="0"*64)),
        ("numerical circuit",lambda r:r.update(numerical_circuit_fingerprint="0"*64)),
        ("logical sample weight",lambda r:r.update(sample_weight={"numerator":1,"denominator":1})),
        ("compile invocation",lambda r:r.update(actual_transpile_invocation_id="ARTIFICIAL_WRONG")),
        ("unsupported wrapper semantics",lambda r:r.update(wrapper_semantics="ARTIFICIAL_FOREIGN")),
        ("changed metric invalidates external digest",lambda r:r["metrics"].update(rz_count=99))]:
        altered=copy.deepcopy(owner);modify(altered)
        reject("v1_identity_cache",label,lambda r=altered:check(r))
    altered=copy.deepcopy(ledger);altered["entries"][owner["wrapper_key"]]["record_sha256"]="0"*64
    reject("v1_identity_cache","external ledger digest altered",lambda:check(led=altered))
    altered=copy.deepcopy(registry);altered[owner["wrapper_key"]]="0"*64
    reject("v1_identity_cache","independent numerical registry altered",lambda:check(reg=altered))
    altered=copy.deepcopy(reuse);altered["cache_owner_wrapper_key"]="0"*64
    led=copy.deepcopy(ledger);led["entries"][altered["wrapper_key"]]["record_sha256"]=oldv.record_digest(altered)
    reject("v1_identity_cache","missing owner",lambda:check(altered,led=led,expected={k:reuse[k] for k in oldv.IDENTITY_FIELDS}))
    for field in ["geometry","candidate_template","axis","hamiltonian_sha256","df_sha256",
                  "state_sha256","source_commit","compiler_fingerprint","environment_fingerprint"]:
        foreign=copy.deepcopy(owner);circ=copy.deepcopy(circuit)
        if field=="geometry":foreign[field]="0.80"
        elif field=="candidate_template":foreign[field].update(L_D=6,template_id="B2-rank6-q1-r4-K2")
        elif field=="axis":foreign[field]=circ["axis"]="sine"
        else:foreign[field]="9"*(40 if field=="source_commit" else 64)
        foreign["candidate_fingerprint"]=oldv.candidate_fingerprint(foreign)
        foreign["trajectory_seed"]=oldv.trajectory_seed(foreign,master)
        foreign["wrapper_key"]=oldv.wrapper_key(foreign)
        foreign["numerical_circuit_fingerprint"]=oldv.circuit_fingerprint(circ)
        foreign["actual_transpile_invocation_id"]="ARTIFICIAL_V2_FOREIGN_"+field
        rs=dict(records);rs[foreign["wrapper_key"]]=foreign
        led=copy.deepcopy(ledger);led["entries"][foreign["wrapper_key"]]={"status":"COMPLETE","record_sha256":oldv.record_digest(foreign)}
        led["reservations"][foreign["actual_transpile_invocation_id"]]={"status":"COMPLETE","wrapper_key":foreign["wrapper_key"]}
        reg=dict(registry);reg[foreign["wrapper_key"]]=foreign["numerical_circuit_fingerprint"]
        accept("v1_foreign_owner","independently valid "+field,lambda r=foreign,c=circ,s=rs,l=led,n=reg:check(r,c,s,l,n,{k:r[k] for k in oldv.IDENTITY_FIELDS}))
        linked=copy.deepcopy(reuse);linked["cache_owner_wrapper_key"]=foreign["wrapper_key"]
        led["entries"][linked["wrapper_key"]]["record_sha256"]=oldv.record_digest(linked)
        reject("v1_foreign_owner","cross-scope reuse "+field,lambda r=linked,s=rs,l=led,n=reg:check(r,circuit,s,l,n,{k:reuse[k] for k in oldv.IDENTITY_FIELDS}))
    accept("v1_accounting","unchanged accounting",lambda:oldv.validate_accounting(fixture["records"],ledger))
    duplicate=copy.deepcopy(owner)
    reject("v1_accounting","duplicate logical records",lambda:oldv.validate_accounting([owner,duplicate],ledger))
    ambiguous=copy.deepcopy(ledger);ambiguous["reservations"]["ARTIFICIAL_ORPHAN"]={"status":"RESERVED","wrapper_key":owner["wrapper_key"]}
    reject("v1_accounting","unresolved reservation STOP",lambda:oldv.validate_accounting(fixture["records"],ambiguous))
    collide=copy.deepcopy(reuse);collide["trajectory_seed"]=owner["trajectory_seed"]
    reject("v1_accounting","duplicate trajectory seed STOP",lambda:oldv.validate_seed_uniqueness([owner,collide]))
    accept("v1_accounting","paired axes share seed",lambda:oldv.validate_seed_uniqueness([owner,{**owner,"axis":"sine"}]))
    if PROTECTED_ATTEMPTS or IMPORT_ATTEMPTS: raise AssertionError("protected/import attempts")
    result={"schema_version":"h4-contract-synthetic-tests-v2","status":"SYNTHETIC_CONTRACT_V2_TESTS_PASS",
            "evidence_scope":"local synthetic JSON only; not scientific, CI or independent external reproduction",
            "passed":len(cases),"failed":0,"skipped":0,"case_groups":dict(Counter(c["group"] for c in cases)),
            "historical_v1_cases_saved_not_rerun":129,"historical_v1_runner_executions":0,
            "actual_authorizations_issued":0,"real_trajectory_seeds_generated":0,
            "new_synthetic_transpiles":0,"scientific_import_attempts":IMPORT_ATTEMPTS,
            "protected_access_attempts":PROTECTED_ATTEMPTS,"elapsed_seconds":time.monotonic()-started,
            "python_executable":sys.executable,"command":"PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /home/AbeHiromu/venvs/trotter-common/bin/python -B run_contract_tests_v2.py",
            "new_test_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"cases":cases}
    with (HERE/"contract_tests_result_v2.json").open("x") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print(json.dumps({k:result[k] for k in ["status","passed","failed","skipped","case_groups","historical_v1_runner_executions","protected_access_attempts","scientific_import_attempts"]}))


if __name__=="__main__":
    main()
