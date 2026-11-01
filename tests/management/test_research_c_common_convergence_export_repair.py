"""Real NumPy scalar/atomic export regressions; no molecular arrays."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/"src"))
from trottertracks import c_common_convergence_export_repair as m


def test_numpy_scalars_convert_without_changing_values():
    value=m.json_ready(dict(ok=np.bool_(True),x=np.float64(.25),n=np.int64(3),
                            other=[np.float32(.5),False]))
    assert type(value["ok"]) is bool and type(value["x"]) is float and type(value["n"]) is int
    assert json.loads(json.dumps(value))==dict(ok=True,x=.25,n=3,other=[.5,False])


@pytest.mark.parametrize("bad",[np.array([1.]),object(),float("nan"),np.complex128(1j)])
def test_invalid_payload_never_creates_partial_file(tmp_path,bad):
    with pytest.raises((TypeError,ValueError)):m.save_json(tmp_path/"result.json",dict(x=bad))
    assert not list(tmp_path.iterdir())


def test_link_failure_leaves_no_target_or_temporary_file(tmp_path):
    with patch.object(m.os,"link",side_effect=OSError("ARTIFICIAL_IO_ERROR")):
        with pytest.raises(OSError):m.save_json(tmp_path/"result.json",dict(ok=True))
    assert not list(tmp_path.iterdir())


def test_existing_partial_is_preserved_and_failure_has_separate_name(tmp_path):
    target=tmp_path/"result.json";target.write_bytes(b'{"partial": ')
    with pytest.raises(FileExistsError):m.save_json(target,dict(ok=True))
    m.save_json(target,dict(status="TECHNICAL_FAILURE",error="mock"))
    assert target.read_bytes()==b'{"partial": '
    assert json.loads((tmp_path/"result_failure.json").read_text())["status"]=="TECHNICAL_FAILURE"
    with pytest.raises(FileExistsError):m.save_json(target,dict(status="TECHNICAL_FAILURE"))


@pytest.mark.parametrize("fault",[None,"unqualified","budget","representation","worker"])
def test_existing_paired_pipeline_with_real_numpy_guard(tmp_path,monkeypatch,fault):
    # Reuse the original synthetic 25-unit pipeline, changing just the real
    # scalar type which its plain-float fixture missed.
    spec=importlib.util.spec_from_file_location("original_common_pilot_tests",
        ROOT/"tests/management/test_research_c_common_convergence_pilot.py")
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    run=m.legacy.run
    def numpy_guard_run(*args,**kwargs):
        assess=m.legacy.molecule.rank_assessment
        def typed(*a,**kw):
            row,G=assess(*a,**kw)
            return {**row,"numerical_guard":np.float64(row["numerical_guard"])},G
        with patch.object(m.legacy.molecule,"rank_assessment",side_effect=typed):
            return run(*args,**kwargs)
    monkeypatch.setattr(m.legacy,"run",numpy_guard_run)
    with m.json_exports():original.test_artificial_paired_25_call_route(tmp_path,monkeypatch,fault)


def test_context_restores_legacy_bindings_after_failure():
    old_decide,old_save=m.legacy.decide,m.io.save
    with pytest.raises(RuntimeError):
        with m.json_exports():raise RuntimeError("mock")
    assert m.legacy.decide is old_decide and m.io.save is old_save


def test_recovery_only_loads_json_and_keeps_old_failed_stop(tmp_path,monkeypatch):
    old=tmp_path/"old";old.mkdir()
    m.save_json(old/"STOP.json",dict(failed=True,scientific_units=25,retry=0,resume=0))
    (old/"result.json").write_bytes(b'{"status":"ONE_OR_BOTH_UNQUALIFIED",')
    for role in m.legacy.ROLES:
        d=old/role;d.mkdir()
        m.save_json(d/"STOP.json",dict(failed=False))
        m.save_json(old/(role+"_sector.json"),dict(sector_error=1e-7,numerical_guard=1e-9))
        m.save_json(d/"optimizer.json",dict(success=False,status=1))
        m.save_json(d/"final_diagnostic.json",dict(components=dict(exact_tail_mass_Ha=2. if role=="rc2" else 1.),assessment=dict(qualified=False)))
        m.save_json(d/"tensor.json",dict(frobenius=0.))
    m.save_json(tmp_path/"old_contract.json",dict(comparators={}))
    m.save_json(tmp_path/"source.json",dict(files={}))
    plan=dict(original_output="old",original_contract="old_contract.json",source_manifest="source.json",
              output=m.FAMILY+"/run01",expected_scientific_status="ONE_OR_BOTH_UNQUALIFIED")
    monkeypatch.setattr(m,"preflight",lambda *args:plan)
    before={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    with patch.object(np,"load",side_effect=AssertionError("NO_NPZ_DECODE")),\
            patch.object(m.legacy.policy,"fit",side_effect=AssertionError("NO_FIT")),\
            patch.object(m.legacy.molecule,"SectorCAR",side_effect=AssertionError("NO_OPERATOR")):
        m.recover(tmp_path,tmp_path/"plan.json",tmp_path/plan["output"])
    out=tmp_path/plan["output"]
    assert m.io.read(out/"result.json")["status"]=="ONE_OR_BOTH_UNQUALIFIED"
    assert m.io.read(out/"STOP.json")["scientific_units"]==1
    assert not m.io.read(out/"STOP.json")["failed"]
    assert m.io.read(out/"recovery.json")["new_fits"]==0
    assert before=={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match="FRESH"):m.recover(tmp_path,tmp_path/"plan.json",out)


def test_passive_import_needs_no_numeric_or_fitting_package():
    code="import sys;sys.path.insert(0,'src');import trottertracks.c_common_convergence_export_repair;assert not {'numpy','ffsim','jax','qiskit'}&set(sys.modules)"
    subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT,check=True)
