"""Two-qubit sine QPY -> unchanged provider compiler -> expected 2% STOP."""
import argparse,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_adapter as adapter
import h8_b2_saved_spool_acceptance as saved

def main():
 p=argparse.ArgumentParser();p.add_argument('--provider',type=Path,required=True);args=p.parse_args();sys.path.insert(0,str(args.provider/'src'))
 from qiskit import QuantumCircuit,qpy
 from trottertracks.resource_applicability import h6_matched_port_v1 as m
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);old=root/'old_read_only';old.mkdir();spool=old/'validated_wrapper.qpy'
  circuit=QuantumCircuit(2,1);circuit.h(0);circuit.crz(.3,0,1);circuit.sdg(0);circuit.h(0);circuit.measure(0,0)
  fp=m.canonical_qiskit_circuit_fingerprint(circuit)
  with spool.open('wb') as f:qpy.dump(circuit,f)
  digest=saved.file_sha(spool);evidence=dict(sine_spool=dict(bytes=spool.stat().st_size,sha256=digest),sine_fingerprint=fp,preparation_binding={'artificial':True},scalar_policy='ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE',max_control_error=1e-14)
  cfg=dict(provider_root=str(args.provider),SOURCE='a'*40,font_seed=None,input_mapping={'validated_sine_QPY':str(spool)})
  output=root/'new_output';output.mkdir();permit=r.Permit(cfg,dict(scope=c.SCOPE,runtime_authorization=True,resources=c.RESOURCES,call_caps=c.CAPS,worker_cpus=[min(os.sched_getaffinity(0))]),output);permit.child_CPU_seconds=0.
  with patch.object(saved,'accept_saved',return_value=evidence):
   try:adapter.evaluate(cfg,permit,r.Writer(output))
   except ValueError as exc:assert str(exc)=='AXIS_MODEL_ERROR_GE_2_PERCENT',str(exc)
   else:raise AssertionError('tiny circuit must fail H8 model comparison')
  result=r.read(output/'axis_01_accepted.json');assert result['axis']=='sine' and result['native_exit_code']==0 and result['reaped'] and not result['model_check']['PASS']
  assert result['fingerprint_before']==result['fingerprint_after']==fp and permit.actual()['compile']==permit.actual()['qpy_load']==1 and permit.actual()['control_probe']==0
  assert saved.file_sha(spool)==digest and not (output/'axis_02').exists()
  print(json.dumps(dict(status='PASS_TWO_QUBIT_SAVED_SPOOL_PROVIDER_ONLY',synthetic_compile=1,expected_model_STOP=True,original_QPY_unchanged=True,new_control=0,new_H8_compile=0,new_signal=0,native_exit0_reaped=True)))
if __name__=='__main__':main()
