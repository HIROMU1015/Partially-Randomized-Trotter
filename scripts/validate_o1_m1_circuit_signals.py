"""Prepared real H4 M1 circuit/signal entry; separate user execution permission required.

--help/import perform no H4 load/build/evaluation. Numerical conditions are
required arguments so a future command states the exact authorized scope.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--step-times',required=True,type=float,nargs='+')
    parser.add_argument('--r',required=True,type=int)
    parser.add_argument('--K-values',required=True,type=int,nargs='+')
    parser.add_argument('--q',required=True,type=int)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh result path required')
    import run_o1_m1_h4_operator_correctness as reference
    if hashlib.sha256(args.snapshot.read_bytes()).hexdigest()!=reference.INPUT_SHA256:
        raise ValueError('Fixed H4 snapshot bytes differ')
    # Passive package import reuses the prior driver's established pattern.
    namespace='o1_m1_signal_core'
    package=types.ModuleType(namespace)
    package.__path__=[str(ROOT/'src/trotterlib')]
    sys.modules[namespace]=package
    sub=types.ModuleType(namespace+'.df_trotter')
    sub.__path__=[str(ROOT/'src/trotterlib/df_trotter')]
    sys.modules[sub.__name__]=sub
    adapter=importlib.import_module(namespace+'.o1_m1_matched_split')
    validation=importlib.import_module(namespace+'.o1_m1_circuit_signal_validation')
    inputs=adapter.load_h4_a_snapshot(args.snapshot)
    split=adapter.prepare_matched_a(inputs)
    result=validation.validate_matched_circuit_signals(split,reference,
        step_times=args.step_times,r=args.r,K_values=args.K_values,q=args.q)
    paths=['src/trotterlib/o1_m1_matched_split.py','src/trotterlib/o1_m1_circuit_signal_validation.py',
           'src/trotterlib/pauli_signal_adapter.py',
           'scripts/run_o1_m1_h4_operator_correctness.py','scripts/validate_o1_m1_circuit_signals.py']
    result.update(scope='H4_A_CIRCUIT_AND_EXACT_FINITE_MEAN_ONLY',
        utc=datetime.now(timezone.utc).isoformat(),host=platform.node(),
        source=dict(base='74dc014627edfa3f7598747ded1b079ec0dbd6d9',
                    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}),
        input=dict(commit=reference.INPUT_COMMIT,path=reference.INPUT_PATH,sha256=reference.INPUT_SHA256,
                   rank=10,state_field='state',state_hash=importlib.import_module(namespace+'.pauli_signal_adapter').array_hash(inputs.state)),
        runtime={p:importlib.metadata.version(p) for p in ('numpy','scipy','openfermion','qiskit','qiskit-nature','qiskit-algorithms')},
        stopped_after='circuit correctness and deterministic finite mean; no sampling, shots or cost')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
