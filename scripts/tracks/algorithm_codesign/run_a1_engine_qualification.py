#!/usr/bin/env python3
"""One authorized fixed qualification; exact saved-state accounting afterward.

The 24d6bdf kernel and its raw result are preserved byte for byte. This runner
only adds forbidden-call guards, provenance and the user-requested exact sum.
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction as F
import hashlib,json,os
from pathlib import Path
import time
from unittest.mock import patch
from trottertracks.algorithm_codesign import a1_engine_connection as kernel


def exact_certificate(raw,state):
    if state['status']!='STATE_CERTIFIED_FOR_SAVED_DF_NUMBER_SECTOR':raise ValueError('N2 state guarantee not certified')
    b=F(state['coordinate_state_error_upper']['exact']);e=F(raw['product_vector_error_upper'])
    if not 0<=b<=F(raw['state_coordinate_upper']) or not 0<=e<1:raise ValueError('saved bound/E consistency failed')
    grid=F(2,2**53);total=b+4*e+grid
    return dict(status='CONDITIONAL_UNIFORM_ENGINE_BOUND_CERTIFIED',state_identity=state['state_vector_sha256'],
        state_bound_exact=str(b),maximum_local_operator_error=raw['maximum_local_operator_error'],
        max_gate_count=raw['max_gate_count'],E_upper=raw['product_vector_error_upper'],
        gate_and_final_normalization_coordinate_upper=str(4*e),conditional_grid_coordinate_upper=str(grid),
        gate_application_arithmetic='0 (exact Fraction kernel)',Born_sum_division_arithmetic='0 (exact Fraction kernel)',
        total_engine_coordinate_upper_exact=str(total),scope=raw['full_support_scope'],
        model='same N1 actual Q / selected logical gates; N2 normalized DF state; exact Fraction engine; independent uniform53bit measurement words',
        candidate_1_over_32_adopted=False,N3_authorized=False,
        raw_rounded_state_result='UNIFORM_ERROR.json retained unchanged; this certificate reuses N2 exact bound')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[3];out=(root/args.output_dir).resolve()
    dep=kernel.n2.n1.dependencies()
    lock=kernel.read(root/'docs/research/o2_a1_engine_qualification_20261106/RUNNER_LOCK.json')
    runner_bytes=Path(__file__).read_bytes()
    if hashlib.sha256(runner_bytes).hexdigest()!=lock['runner_sha256']:raise ValueError('execution runner differs from frozen lock')
    marker=Path('/tmp/a1-engine-qualification-invocation-20261106.json')
    with marker.open('x') as f:json.dump(dict(fixed_kernel_commit='24d6bdfa11ca5964c83b2639f060f4655ab1d4be',qualification_invocations=1,output=str(out)),f)
    started=time.monotonic()
    def forbidden(*a,**k):raise RuntimeError('scope forbids real state action/Born/path/solve/eigh/measurement/compile')
    try:
        import numpy as np
        # All allowed local-gate arithmetic is unchanged. Only forbidden entry
        # points are replaced by fail-only guards; no state/gate value is patched.
        with kernel.n2.n2_only(dep),ExitStack() as stack:
            for owner,name in ((dep.hamiltonian,'solve_df_ground_state'),(np.linalg,'eigh'),
                (kernel.SavedEngine,'probability'),(kernel.SavedEngine,'selected_circuit'),
                (kernel,'apply_exact'),(kernel,'born_probability'),(kernel,'norm_squared')):
                stack.enter_context(patch.object(owner,name,forbidden))
            raw=kernel.qualify_fixed_bundle(root,out)  # The sole real invocation.
        state_path=root/kernel.P2/'STATE_RESULT.json';state=kernel.read(state_path)
        cert=exact_certificate(raw,state)
        kernel.n2.write(out/'ENGINE_ERROR_CERTIFIED.json',cert)
        source=dict(runner_source_commit=kernel.n2.git(root,'rev-parse','HEAD'),fixed_kernel_commit='24d6bdfa11ca5964c83b2639f060f4655ab1d4be',runner_sha256=hashlib.sha256(runner_bytes).hexdigest(),
            N2_state_result_path=str(state_path.relative_to(root)),N2_state_result_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
            exact_state_bound_field='coordinate_state_error_upper.exact',qualification_invocations=1,exact_accounting_sums=1,
            prohibited_calls=dict(preparation=0,CDF=0,solve=0,eigh=0,real_state_action=0,real_Born_evaluation=0,real_path_build=0,RNG_sampling=0,shots=0,compile=0,variance_N_cost=0),
            elapsed_seconds=time.monotonic()-started,resource_caps_added=False)
        kernel.n2.write(out/'EXECUTION_SOURCE.json',source)
        kernel.n2.write(out/'RESOURCE_OBSERVATION.json',dict(cpu_count=os.cpu_count(),load_average=os.getloadavg(),memory=[x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))]))
        (out/'INVOCATION.json').write_bytes(marker.read_bytes())
        print(json.dumps(dict(status=cert['status'],unique_gate_keys=raw['local_gate_count'],delta_star=raw['maximum_local_operator_error'],L_max=raw['max_gate_count'],E_upper=raw['product_vector_error_upper'],b_engine=cert['total_engine_coordinate_upper_exact']),indent=2))
    except Exception as exc:
        if out.is_dir():
            kernel.n2.write(out/'EXECUTION_FAILURE.json',dict(error_type=type(exc).__name__,error=str(exc),qualification_invocations=1,no_retry=True))
            (out/'INVOCATION.json').write_bytes(marker.read_bytes())
        raise

if __name__=='__main__':main()
