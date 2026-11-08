"""Thin inheritance and progress adapter for the authorized C8-1 remainder.

No sampler/compiler/state propagation implementation. Numerical operations stay
in the fixed unfused Aer double path; progress observes completed batches only.
"""
from pathlib import Path
import shutil
import time

from . import track_d_c8_1_handoff as evidence
from .track_d_c8_1_nofusion import UnfusedStateActions
from .track_d_c8_1_diagnostics import save_json

REPAIR_SOURCE = "f1e89590c7daea12319352b85b959928ad166213"
ARMS = evidence.METHODS
AXES = evidence.AXES
PROGRESS = None


def inheritance(original, repair):
    original, repair = Path(original), Path(repair)
    prior = evidence.inheritance_plan(original)
    result = evidence.read(repair/"result.json")
    repaired_source=evidence.read(repair/"SOURCE.json")
    if (repaired_source["execution_source"] != REPAIR_SOURCE
            or result["completed"] != 4 or result["pass_count"] != 4
            or not result["prior_preserved"] or not evidence.read(repair/"STOP.json")["mandatory_stop"]):
        raise ValueError("FIXED_REPAIR_PASS4_REQUIRED")
    # Bind the inherited PASS to exactly the old raw, events and compiled QPYs.
    bindings={row["path"]:row["sha256"] for row in repaired_source["fixed_inputs"]["members"]}
    for row in prior["members"]:
        path=row["path"]
        if path.startswith("local/") and (path.endswith((".npz",".qpy")) or path.startswith("local/events/")
                                           or Path(path).name in ("conditions.json","precompile_checks.json")):
            if bindings.get(path)!=row["sha256"]:
                raise ValueError("REPAIR_PASS_BOUND_TO_DIFFERENT_INPUT:"+path)
    members = []
    def add(src, dest):
        p = repair/src
        members.append(dict(path=src, destination=dest, bytes=p.stat().st_size, sha256=evidence.sha(p)))
    for name in ("SOURCE.json","result.json","STOP.json","payload_index.json"):
        add(name,"inherited_repair_"+name)
    for arm in ARMS:
        for axis in AXES:
            name=arm+"_"+axis
            if evidence.read(repair/(name+"_check.json"))["status"] != "PASS":
                raise ValueError("REPAIR_POSTCHECK_PASS_REQUIRED")
            add(name+"_check.json","local/q64_sample00/"+name+"_check.json")
            add("local/"+name+"/observable.npz","local/q64_sample00/"+name+"_observable.npz")
            add("local/"+name+"/runtime_private.json","local/inherited_runtime/"+name+".json")
    return dict(original=prior, repair=dict(source=REPAIR_SOURCE,members=members),
                raw_reused=1,compiled_reused=4,postchecks_reused=4,
                new_sampling=0,new_raw_limit=7,new_compile_limit=28,new_postcheck_limit=28,
                final_paired_chi_limit=8,cut=32,q=64)


def verify(original, repair, packet):
    evidence.verify(original,packet["original"])
    evidence.verify(repair,packet["repair"])


def inherit_repair(repair, output, packet):
    for row in packet["repair"]["members"]:
        destination=Path(output)/row["destination"]
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            raise ValueError("INHERITED_DESTINATION_EXISTS")
        shutil.copyfile(Path(repair)/row["path"],destination)
        if evidence.sha(destination)!=row["sha256"]:
            raise ValueError("INHERITED_REPAIR_COPY_MISMATCH")


class _CompletedBatches(list):
    def __init__(self, callback):
        super().__init__(); self.callback=callback

    def append(self, value):
        super().append(value); self.callback(sum(self))


class ReportingStateActions(UnfusedStateActions):
    """Observe the existing batch-size list without changing Aer operations."""
    @property
    def last_batch_sizes(self):
        return self._completed_batches

    @last_batch_sizes.setter
    def last_batch_sizes(self, value):
        if value != []:
            raise ValueError("EXPECTED_NEW_BATCH_LIST")
        self._completed_batches=_CompletedBatches(self._report)

    def _report(self, completed):
        if PROGRESS is not None:
            save_json(PROGRESS,dict(completed_columns=completed,total_columns=self._total,
                elapsed_seconds=time.monotonic()-self._started,
                simulation_fusion_enabled=False,status="PROPAGATING"))

    def apply(self, circuit, states):
        self._total=len(states);self._started=time.monotonic();self._report(0)
        result=super().apply(circuit,states)
        if PROGRESS is not None:
            save_json(PROGRESS,dict(completed_columns=self._total,total_columns=self._total,
                elapsed_seconds=time.monotonic()-self._started,
                simulation_fusion_enabled=False,status="PROPAGATION_COMPLETE"))
        return result


def install(threads, progress, *, existing_install):
    global PROGRESS
    from . import track_d_c8_1_nofusion as repair
    from . import c1_h6_state_action as primitive
    PROGRESS=Path(progress)
    engine=repair.install(threads,existing_install=existing_install)
    primitive.StateActions=ReportingStateActions
    return engine


def allocation(existing, allocator, cpu, memory, pending, workers, active, estimates):
    """Keep the old CPU allocator; select the completed-peak RAM estimate by arm."""
    for task in pending:
        if task["stage"]=="post":
            key="reference_post_memory_estimate" if task["method"]=="reference" else "direct_post_memory_estimate"
            task["memory_bytes"]=estimates[key]
    return existing(allocator,cpu,memory,pending,workers,active)
