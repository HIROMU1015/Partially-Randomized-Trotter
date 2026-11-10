"""One new K0 m0 full no-prep trajectory price; not an expectation or total."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[1]
ART='artifacts/calpha_h4_step1/2026-11-10/'


def evaluate(private_dir,out):
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib.b2_rpe_connection import prepare_b2,B2RPERound,RPEConnectionContext
    from trotterlib.b2_p1_compiled_cost import price_explicit_trajectory
    from trotterlib.o1_m1_resource_bridge import MatchedNoPrepCostProvider
    from trotterlib.rpe_resource_accounting import RPERoundSpecification
    from trotterlib.rte import CompilerSettings,make_rte_config,sample_rte_events
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    accounting=json.loads((ROOT/(ART+'run10_rpe_accounting/result.json')).read_text())
    if accounting['status']!='ALL18_CONDITIONAL_GATE_MODEL_GUARANTEE_AND_SHOTS' or len(accounting['rounds'])!=18:
        raise ValueError('ALL18_GUARANTEE_AND_ACCOUNTING_REQUIRED')
    data=p1.load_private_inputs(private_dir)
    prep=prepare_b2(data['ham'],L_D=3);tail=prep.rte_preparation.symbolic_tail
    law=json.loads((ROOT/(ART+'run09_sampling_law/result.json')).read_text())
    if tail.tail_hash!=law['profile_summary']['tail_hash']:
        raise ValueError('QUALIFIED_EXECUTABLE_LAW_CHANGED')
    compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0')
    model_record=json.loads((ROOT/(ART+'run07_reconstruction_bridge/result.json')).read_text())
    model_bound=math.nextafter(float(Fraction(model_record['model_difference_upper']['upper_exact'])),math.inf)
    source='1b1449c9d8f04e4b574bc8a7c6b677572bfb0346+local:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    context=RPEConnectionContext(data['spec']['original_hamiltonian'],data['spec']['df_hamiltonian'],
        data['spec']['original_state'],data['spec']['sector'],data['spec']['spin_orbital_order'],
        source,compiler,'no_backend','rz_count',
        model_bound,.00015936001019904,.95,'all18 interval raw-radius certificate and beta=.4 branch induction')
    config,dist=make_rte_config(tail,evolution_time=.02,rte_steps=4,finite_taylor_order=0,
        truncation_tolerance=p1.legacy._explicit_cutoff_tolerance(tail.lambda_r*.02/4,0),seed=20261110)
    spec=RPERoundSpecification(0,.02)
    round_=B2RPERound(context,prep,spec,3,4,0,.02,1,config,dist)
    # Exactly one draw fixed before any price. It is not a sufficient MC size
    # for estimating an expected price or uncertainty and is never so labelled.
    events=sample_rte_events(tail.components,dist,sample_count=4,seed=20261110)
    (out/'trajectory.json').write_text(json.dumps([e.to_dict() for e in events],ensure_ascii=False,indent=2)+'\n')
    captures=[]
    def capture(circuit,settings,**kwargs):
        value=transpile_and_measure_cost(circuit,settings,**kwargs)
        captures.append(dict(qubits=value.qubit_count,measurements=value.transpiled_circuit.count_ops().get('measure',0),
            compiler_hash=value.compiler_settings_hash,
            input_fingerprint=value.actual_circuit_fingerprint,
            compiled_gate_counts=dict(value.posttranspile_gate_counts)))
        return value
    result=price_explicit_trajectory(round_,MatchedNoPrepCostProvider(compiler,compile_function=capture),events)
    if len(captures)!=2 or any(x['qubits']!=9 or x['measurements']!=1 for x in captures):
        raise ValueError('FULL_MEASURED_NO_PREP_SCOPE_NOT_PRESERVED')
    return dict(status='ONE_TRAJECTORY_M0_PRICE_OBTAINED',price=result,axes_scope=captures,
        seed=20261110,trajectory_samples=1,compiler_calls=2,
        expected_price=None,price_standard_error=None,total_RPE_cost=None,
        m17_price=None,cost_comparison='NOT_ESTABLISHED',
        same_known_K0_price='identical law/settings and same priced circuit; no distinct physical C-alpha price',
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_calpha_price_probe.py','src/trotterlib/b2_p1_compiled_cost.py',
           'src/trotterlib/b2_rpe_connection.py','src/trotterlib/o1_m1_resource_bridge.py',
           'src/trotterlib/rte_compiled_cost.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,status='STARTED')
    started=time.monotonic()
    try:result.update(evaluate(args.private_dir,out))
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
