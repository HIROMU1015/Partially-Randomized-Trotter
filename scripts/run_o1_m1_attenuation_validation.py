"""Prepared signal-only r4/r8 probe; real H4 execution needs user permission.

No price provider, shot accounting, compile, sampling or optimizer is called.
Controlled coverage is selected primitives/one complete step, not every q64
trajectory. The q64 finite mean has the existing independent CAR/polynomial check.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_o1_m1_c0_pilot as c0

BASE = '8820c41bc5beb32208447fe78fee6854088cbcd5'
INPUT_SHA = c0.INPUT_SHA
R_VALUES = (4, 8)
Q, K, DELTA = 64, 2, .05
PHASE = dict(PF=.06, RTE=.03, stat=.15, model=.03, implementation=.01, reserve=.02)
CORE = ('scripts/run_o1_m1_c0_pilot.py',
        'scripts/run_o1_m1_h4_operator_correctness.py',
        'src/trotterlib/o1_m1_matched_split.py',
        'src/trotterlib/o1_m1_circuit_signal_validation.py',
        'src/trotterlib/o1_m1_resource_bridge.py', 'src/trotterlib/rte.py')
COVERAGE = ('selected Gaussian/Pauli/DF primitives and one complete controlled '
            'step including scalar; independent q64 finite mean; '
            'not full q64 trajectory operator certification')


def modules():
    """Passive package import; the caller explicitly starts scientific work."""
    sys.path.insert(0, str(ROOT/'src'))
    if 'trotterlib' not in sys.modules:
        package = types.ModuleType('trotterlib')
        package.__path__ = [str(ROOT/'src/trotterlib')]
        sys.modules['trotterlib'] = package
    from trotterlib import o1_m1_matched_split as adapter
    from trotterlib import o1_m1_circuit_signal_validation as signals
    from trotterlib import o1_m1_resource_bridge as bridge
    import run_o1_m1_h4_operator_correctness as reference
    return adapter, signals, bridge, reference


def assess(payload, *, r, preparations, laws, bridge):
    """Reuse C0 allowances/bounds, stopping before its shot-accounting API."""
    if r not in R_VALUES:
        raise ValueError('Only fixed r4/r8 permitted')
    conditions = payload['conditions']
    if (payload['status'] != 'PASS' or not payload['checks']
            or not all(check['passed'] for check in payload['checks'])):
        raise ValueError('Existing operator/controlled validator failed')
    if (conditions['q'] != Q or conditions['r'] != r
            or tuple(conditions['step_times']) != (DELTA,)
            or tuple(conditions['K_values']) != (K,)
            or conditions['identity_policy'] != 'faithful_identity_in_tail'
            or conditions['original_state_field'] != 'state'
            or conditions['sector_projection'] is not False
            or len(payload['records']) != 1):
        raise ValueError('Validator conditions differ from the fixed probe')
    record = payload['records'][0]
    if (record['q'], record['r'], record['K'], record['delta']) != (Q, r, K, DELTA):
        raise ValueError('Signal record differs from the fixed probe')
    margins = c0.c0_margins(payload, record)
    spec = types.SimpleNamespace(q_m=Q, delta_time=DELTA)
    row = dict(r=r, K=K, q=Q, delta=DELTA, numerical_allowances=margins, arms={})
    for name in ('P', 'DF'):
        bounds = bridge.observed_signal_bounds(preparations[name], laws[name][0],
                                               spec, record, **margins)
        radius, error = bounds['reference_radius_lower_bound'], bounds['corrected_signal_error_bound']
        data = record['dictionaries'][name]
        attenuation = data['attenuation']
        rho = attenuation*(radius-error)
        rte_phase = math.asin(error/radius) if 0 <= error < radius else None
        reasons = []
        if not math.isfinite(rho) or rho <= 0 or attenuation <= 0:
            reasons.append('nonpositive_radius')
        if rte_phase is None or rte_phase > PHASE['RTE']:
            reasons.append('RTE_phase_budget')
        for key in ('PF', 'model', 'implementation'):
            bound = bounds['pf_phase_bound' if key == 'PF' else key+'_phase_bound']
            if bound > PHASE[key]:
                reasons.append(key+'_phase_budget')
        # Same C0 total-budget rule; reserve is not reassigned to statistics.
        if (PHASE['PF']+PHASE['RTE']+PHASE['stat']
                +bounds['model_phase_bound']+bounds['implementation_phase_bound'] > .30):
            reasons.append('total_phase_budget')
        row['arms'][name] = dict(feasible=not reasons, reasons=reasons, bounds=bounds,
            finite_rte_phase_bound=rte_phase, rho_observed_lower_bound=rho,
            attenuation=attenuation, normalization_product=data['normalization'])
    return row


def run(input_path, output_dir):
    """Scientific entry point; use only after the separate user permission."""
    if output_dir.exists():
        raise FileExistsError('Fresh output directory required; historical STOPs are preserved')
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != INPUT_SHA:
        raise ValueError('Fixed saved input hash differs')
    for path in CORE:
        if (ROOT/path).read_bytes() != subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT):
            raise ValueError('Fixed source differs: '+path)
    paths = CORE+('scripts/run_o1_m1_attenuation_validation.py',)
    result = dict(status='RUNNING', mandatory_stop=False,
        conditions=dict(r_values=R_VALUES, K=K, q=Q, delta=DELTA,
            dictionaries=['P', 'DF'], phase_budgets=PHASE, beta_RPE=.30,
            state_field='state', state_assumption='original H exact ground state',
            state_preparation_included=False, controlled_coverage=COVERAGE),
        provenance=dict(base=BASE,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            input=dict(commit=c0.INPUT_COMMIT,path=c0.INPUT_PATH,sha256=INPUT_SHA,local_path=str(input_path)),
            host=platform.node(),utc=datetime.now(timezone.utc).isoformat(),
            runtime={p:importlib.metadata.version(p) for p in ('numpy','scipy','openfermion','qiskit','qiskit-nature')}),
        c0_control=dict(commit='50ee55bc30f30838e1ac0704bc94f720914640de',
            path='docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md',
            use='quotation only; no saved-result load/reanalysis'), rows=[])
    if result['provenance']['runtime']['qiskit'] != '1.3.0':
        raise ValueError('Fixed Qiskit1.3.0 required')
    output_dir.mkdir(parents=True)
    try:
        c0.dump(output_dir/'result.json', result)
        return _run_probe(input_path, output_dir, result)
    except BaseException as exc:
        result.update(status='ERROR_STOP', mandatory_stop=True,
                      error_type=type(exc).__name__, error_message=str(exc))
        c0.dump(output_dir/'result.json', result)
        raise


def _run_probe(input_path, output_dir, result):
    adapter, signals, bridge, reference = modules()
    inputs = adapter.load_h4_a_snapshot(input_path)
    if inputs.num_qubits != 8 or len(inputs.df_lambdas) != 10:
        raise ValueError('Fixed H4/rank10 required')
    split = adapter.prepare_matched_a(inputs)
    preparations = {name:bridge.prepare_matched_resources(split,dictionary=name) for name in ('P','DF')}
    for r in R_VALUES:
        print(f'signal-only q={Q} r={r} K={K} delta={DELTA}', flush=True)
        laws = {name:bridge.configure_law(p,delta_time=DELTA,r=r,K=K) for name,p in preparations.items()}
        payload = signals.validate_matched_circuit_signals(split,reference,
            step_times=(DELTA,),r=r,K_values=(K,),q=Q)
        c0.dump(output_dir/f'signal_r{r}.json', payload)
        try:
            row = assess(payload,r=r,preparations=preparations,laws=laws,bridge=bridge)
        except ValueError as exc:
            result.update(status='LIMITED_PROBE_NOT_QUALIFIED', mandatory_stop=True,stop_reason=str(exc),failed_r=r)
            c0.dump(output_dir/'result.json', result)
            return result
        result['rows'].append(row)
        if any(not arm['feasible'] for arm in row['arms'].values()):
            result.update(status='LIMITED_PROBE_NOT_QUALIFIED',mandatory_stop=True,
                          stop_reason='Fixed C0 phase/radius conditions failed',failed_r=r)
            c0.dump(output_dir/'result.json',result)
            return result
        c0.dump(output_dir/'result.json',result)
    result.update(status='LIMITED_SIGNAL_PROBE_COMPLETE',mandatory_stop=True,
        limitation='No shots, prices, compile, sampling, all-round costs or optimization; not energy/success certification')
    c0.dump(output_dir/'result.json',result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.input.resolve(),args.output_dir.resolve())
    return 0 if result['status']=='LIMITED_SIGNAL_PROBE_COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())
