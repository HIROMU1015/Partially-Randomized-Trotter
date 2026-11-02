"""R0 binary64 adapter. Only two saved cosine programs and five state actions.

Structural QPY validation performs no state action, build, sampling or compile.
Numerical imports are lazy. The preparation-only toy entrypoint stays unchanged.
"""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Mapping

from . import h4_certificate_control as C
from .h4_conditional_cng import extract
from . import h4_central_rho_plan as P

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_r0_empirical/2026-10-24'
KEYS=('PF0_cosine','PR0_cosine')
ACTION_LIMITS=dict(PF_input=1,PF_output=1,PR0_input=1,PR0_output=1,PR0_C_target=1)
LIMITS=dict(preflight=1,PF_diagnostic=1,PR0_diagnostic=1,**ACTION_LIMITS)
_API=None


def api():
    global _API
    if _API is None:
        import numpy as np
        import qiskit
        from qiskit import qpy,QuantumCircuit
        from qiskit.quantum_info import Statevector
        env=dict(np=np,math=math,json=json,hashlib=hashlib,Any=Any,Mapping=Mapping,QuantumCircuit=QuantumCircuit)
        for fn in ('_numeric_parameter_payload','_condition_payload','_operation_payload','_circuit_payload','canonical_qiskit_circuit_fingerprint'):
            extract('src/trotterlib/rte_compiled_cost.py',fn,env)
        inspector='artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/inspect_saved_evidence.py'
        token=extract(inspector,'primitive_token',{})
        recipe=extract(inspector,'pauli_recipe',{})
        signature=extract('src/trotterlib/h4_a_axes.py','circuit_signature',{})
        pauli=extract('src/trotterlib/h2_non_df_pilot.py','pauli_matrix',dict(np=np))
        event=extract('src/trotterlib/rte.py','event_unitary',dict(np=np,math=math,RTEEvent=object,Mapping=Mapping))
        from .h4_d2_pauli_control import _tokens,_rotation
        _API=SimpleNamespace(np=np,qiskit=qiskit,qpy=qpy,Statevector=Statevector,token=token,recipe=recipe,
            fingerprint=env['canonical_qiskit_circuit_fingerprint'],signature=signature,pauli=pauli,event=event,tokens=_tokens,rotation=_rotation)
    return _API


class TokenRecorder:
    """Read-only expected token pattern, not a QuantumCircuit builder."""
    def __init__(self):self.data=[]
    def __getattr__(self,name):
        def record(*args):
            params=list(args[:1]) if name in ('crz','p') else []
            wires=list(args[1:] if params else args)
            self.data.append(dict(name=name,params=params,qubits=wires,clbits=[]))
        return record


def validate_program(circuit,*,compiled,n):
    C.require(circuit.num_qubits==n+1 and circuit.num_clbits==1 and not circuit.parameters,'FIXED_PROGRAM_DIMENSION')
    C.require(circuit.layout is None and not circuit.calibrations,'NO_LAYOUT_CALIBRATION')
    sig=api().signature(circuit)
    C.require(sig['instructions'][-1]==dict(name='measure',params=[],qubits=[n],clbits=[0]),'FINAL_ANCILLA_MEASUREMENT')
    allowed=('rz','sx','x','cx') if compiled else ('h','s','sdg','cx','cy','cz','crz','p')
    for item in circuit.data[:-1]:
        C.require(not item.clbits and getattr(item.operation,'condition',None) is None and item.operation.name in allowed,'UNSUPPORTED_GATE_OR_MID_MEASUREMENT')
    return sig


def validate_controlled(source,controlled,wrapped,sidecar):
    a=api();tokens,phase=a.tokens(source); n=source.num_qubits
    C.require(phase==sidecar['aggregate_source_global_phase'],'BODY_AGGREGATE_PHASE')
    rec=TokenRecorder();mapping=sidecar['instruction_mapping']
    C.require(len(tokens)==len(mapping)==sidecar['source_instruction_count'],'D2_TOKEN_COUNT')
    for i,(token,m) in enumerate(zip(tokens,mapping)):
        C.require(m['source_instruction']==i and C.digest(token)==C.digest(m['source_token']) and m['output_start']==len(rec.data),'SOURCE_TOKEN_MAPPING')
        if token['kind']=='evolution':a.rotation(rec,token['label'],token['central_rz_angle'],token['wires'],n)
        else:getattr(rec,'c'+token['axis'])(n,token['wires'][0])
        C.require(m['output_stop']==len(rec.data),'SOURCE_MAPPING_STOP')
    rec.p(phase,n)
    actual=a.signature(controlled)
    C.require(controlled.num_qubits==n+1 and controlled.num_clbits==0 and actual['global_phase']==0. and actual['instructions']==rec.data,'D2_CONTROLLED_TOKENS')
    ws=a.signature(wrapped)
    head=dict(name='h',params=[],qubits=[n],clbits=[])
    measure=dict(name='measure',params=[],qubits=[n],clbits=[0])
    C.require(ws['global_phase']==actual['global_phase'] and ws['instructions']==[head,*rec.data,head,measure],'COSINE_WRAPPER_PHASE_WIRES')
    return dict(source_tokens=len(tokens),aggregate_phase=phase,controlled_instructions=len(rec.data),
        control_branches='SOURCE diag(I,U) token correspondence; no new whole-operator/state check',
        controlled_global_phase=actual['global_phase'],phase_owner='saved body aggregate P(phi) on ancilla exactly once')


def preflight():
    a=api(); C.require(a.qiskit.__version__=='1.3.0','QISKIT_VERSION')
    targets=[t for t in C.read(P.BASE/'targets.json')['targets'] if t['key'] in KEYS]
    C.require(tuple(t['key'] for t in targets)==KEYS,'TWO_FIXED_COSINE_TARGETS')
    model=C.read(ROOT/P.MODEL)
    array_hash=extract('src/trotterlib/h4_ball_evidence.py','array_hash',dict(struct=struct,hashlib=hashlib,json=json))
    C.require(array_hash(model['state'],vector=True)==model['state_hash'] and C.digest({k:v for k,v in model.items() if k!='binding_hash'})==model['binding_hash'],'C_STATE_MODEL_BINDING')
    verify_join=extract('src/trotterlib/rte_event_audit.py','verify_join',dict(digest=C.digest))
    circuits={};records=[];joins={}
    inputs=C.read(BASE/'inputs.json')['files']
    for t in targets:
        loaded={}
        for role in ('source_body','controlled_body','input_qpy','compiled_qpy'):
            path=t[role];C.require(C.sha(ROOT/path)==inputs[path],'ACTUAL_QPY_SHA:'+path)
            with (ROOT/path).open('rb') as f:items=a.qpy.load(f)
            C.require(len(items)==1,'SINGLE_SAVED_PROGRAM');loaded[role]=items[0]
        folder=(ROOT/t['input_qpy']).parent
        inp=C.read(folder/'input.json');metrics=C.read(folder/'metrics.json')
        P.check_saved_tags(inp,metrics,P.D2)
        join=C.read(ROOT/t['price_join']);verify_join(join,expected_case='pauli_jw_'+t['method'])
        C.require(all(join[k]==t[k] for k in ('trajectory_id','trajectory_record_hash','event_record_hash')),'SAVED_EVENT_IDENTITY')
        expected,_=a.recipe(model,t['method'],join)
        C.require([a.token(x,loaded['source_body']) for x in loaded['source_body'].data]==expected,'SOURCE_BODY_RECIPE')
        checks=validate_controlled(loaded['source_body'],loaded['controlled_body'],loaded['input_qpy'],inp['sidecar'])
        for role,fp in (('input_qpy','input_fingerprint'),('compiled_qpy','output_fingerprint')):
            sig=validate_program(loaded[role],compiled=role=='compiled_qpy',n=8)
            actual=a.fingerprint(loaded[role]);C.require(actual==t[fp],'ACTUAL_QPY_FINGERPRINT:'+role)
            if role=='compiled_qpy':
                counts=dict(loaded[role].count_ops());C.require(counts==metrics['posttranspile_histogram'],'POST_BASIS_HISTOGRAM')
            checks[role]=dict(sha256=C.sha(ROOT/t[role]),fingerprint=actual,instructions=len(loaded[role].data),
                             global_phase=sig['global_phase'],layout=None,terminal_measurement='wire8 -> clbit0')
        records.append(dict(key=t['key'],trajectory_id=t['trajectory_id'],status='PASS_STRUCTURAL_SOURCE_BINDING',**checks))
        key='PF' if t['method']=='PF' else 'PR0'
        circuits[key]=dict(input=loaded['input_qpy'],output=loaded['compiled_qpy']);joins[key]=join
    C.require(all(e['taylor_order']==0 for e in joins['PR0']['events']),'FIXED_PR0_ORDER0')
    C.require(model['n_qubits']==8 and model['qubit_order']=='qiskit_little_endian','STATE_LAYOUT')
    return targets,model,joins,circuits,dict(status='PASS',QPY_decodes=8,state_actions=0,records=records,
        state_hash=model['state_hash'],old_PR_historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED')


def raw_norm(encoded):
    """Exact scalar norm of stored binary64 pairs; no state replacement."""
    return sum((F(x)**2+F(y)**2 for x,y in encoded),F(0))


def propagate(circuit,raw,nu):
    a=api();np=a.np;n=len(raw)
    initial=a.Statevector(np.concatenate((raw,np.zeros_like(raw))))
    final=initial.evolve(circuit.remove_final_measurements(inplace=False)).data
    p=np.sum(np.abs(final.reshape(2,n))**2,axis=1)
    m=float((p[0]-p[1])/float(nu))
    return final,dict(signal=m,p0_raw=float(p[0]),p1_raw=float(p[1]),raw_norm=float(nu),
        output_norm_observed=float(np.vdot(final,final).real),state_renormalized=False,output_norm_division=False,
        probability_API_used=False,status='EMPIRICAL_ONLY_NO_RHO_UPPER',numerical_error_upper=None)


def selected_C_action(model,join,raw,nu):
    """Minimal vector form of the frozen C/trajectory-reference schedule.

    Reuse Pauli matrices and event_unitary; no full H, expm, circuit or law mean.
    C's exact encoded coefficient*delta/2 and native scalar operands are mapped
    to binary64 trig. The conversion/trig/matvec errors are not enclosed.
    """
    a=api();np=a.np;case=model['cases']['PR'];cfg=case['config']
    C.require(cfg==dict(T=.8,q=8,r=4,K=4,method='PR') and model['representation']=='pauli_jw','SELECTED_C_SCOPE')
    C.require(join['method']=='PR' and len(join['events'])==cfg['q']*cfg['r'],'SELECTED_EVENT_SCOPE')
    I=np.eye(len(raw),dtype=complex);pm={};blocks=[];angles=[]
    def pauli(label):
        if label not in pm:pm[label]=a.pauli(label)
        return pm[label]
    for b in case['blocks']:
        angle=F(b['coefficient'])*F(model['step_time'])/2
        point=float(angle);blocks.append(math.cos(point)*I-1j*math.sin(point)*pauli(b['label']));angles.append(str(angle))
    operators={};events=[]
    for record in join['events']:
        for app in record['application_sequence']:
            cid=app['component_id'];comp=case['components'][int(cid)]
            C.require(comp['sign']==app['coefficient_sign'] and comp['coefficient_abs']==app['coefficient_abs'],'C_EVENT_SIGN_COMPONENT')
            operators[cid]=comp['sign']*pauli(comp['label'])
        C.require(F(record['rotation_angle'])==F(case['law']['angles'][(0,2,4).index(record['taylor_order'])]),'C_SAVED_ROTATION_ANGLE')
        e=SimpleNamespace(rotation_component_id=record['rotation_component_id'],product_component_ids=record['product_component_ids'],
                          rotation_angle=record['rotation_angle'],phase=complex(record['phase']['real'],record['phase']['imag']))
        events.append(a.event(e,operators))
    scalar_exact=sum(map(F,case['native_scalar_phases']),F(0));scalar=complex(math.cos(float(scalar_exact)),math.sin(float(scalar_exact)))
    out=raw.copy()
    # Identical chronological C forward/events/reverse/native-scalar schedule;
    # events are fixed, not the repeated expected tail used by C's law engine.
    for step in range(cfg['q']):
        for E in blocks:out=E@out
        for j in range(cfg['r']):
            record=join['events'][step*cfg['r']+j]
            C.require((record['outer_step_index'],record['inner_RTE_position'])==(step,j),'C_EVENT_ORDER')
            out=events[step*cfg['r']+j]@out
        for E in reversed(blocks):out=E@out
        out*=scalar
    z=np.vdot(raw,out)/float(nu)
    return out,dict(signal=float(z.real),complex_signal=dict(real=float(z.real),imag=float(z.imag)),
        selected_trajectory_id=join['trajectory_id'],PR_law_mean_used=False,exact_retained_angle_operands=angles,
        exact_native_scalar_operand=str(scalar_exact),normalization_B_applied_to_target=False,
        status='EMPIRICAL_ONLY_NO_RHO_UPPER',numerical_error_upper=None)


def saved_PF_target(cresult,nu):
    target=cresult['implemented_law']['PF']['corrected_encoded_mean']['real']
    norm=cresult['state_norm'];C.require(F(norm['lower'])<=nu<=F(norm['upper']),'SAVED_C_NORM_ENCLOSES_RAW_RAY')
    lo,hi=F(target['lower'])/nu,F(target['upper'])/nu
    representative=(lo+hi)/2
    return representative,dict(source='saved C PF corrected_encoded_mean.real / exact raw ray norm',
        saved_raw_interval=target,saved_norm_interval=norm,normalized_interval_exact=[str(lo),str(hi)],
        representative_exact=str(representative),representative_display=float(representative),
        operation='exact scalar conversion of existing interval only; no new Arb/state interval evaluation',
        compiled_signal_error_enclosed=False)


def compare_signals(mC,minput,moutput,B,guard):
    mc,mi,mo,b,g=map(F,(mC,minput,moutput,B,guard))
    delta,eta,d=b*abs(mo-mi),b*abs(mi-mc),b*abs(mo-mc)
    C.require(d<=delta+eta,'EXACT_SCALAR_TRIANGLE')
    return dict(m_C=float(mc),m_C_representative_exact=str(mc),m_input=float(mi),m_output=float(mo),
        delta=float(delta),delta_exact=str(delta),eta=float(eta),eta_exact=str(eta),d=float(d),d_exact=str(d),
        B_saved=float(b),B_saved_exact=str(b),rho_budget_exact=str(g),rho_budget=float(g),
        d_over_budget=float(d/g),d_over_budget_exact=str(d/g),budget_minus_d=float(g-d),budget_minus_d_exact=str(g-d),
        empirical_relation='BELOW_SAVED_BUDGET' if d<g else 'EQUAL_SAVED_BUDGET' if d==g else 'ABOVE_SAVED_BUDGET',
        triangle_verified=True,status='EMPIRICAL_ONLY_NO_RHO_UPPER',numerical_error_upper=None,rho_upper=None,
        PR_law_rho=None,formal_certificate_accepted=False,LOCAL_BUDGET_MET=False)


def run(ns,contract,budget):
    ns=Path(ns);budget.take('preflight')
    targets,model,joins,circuits,binding=preflight();C.save(ns/'preflight.json',binding)
    a=api();np=a.np;raw=np.array([complex(x,y) for x,y in model['state']],dtype=np.complex128)
    nu=raw_norm(model['state']);C.require(nu>0,'POSITIVE_RAW_NORM')
    C.save(ns/'state_identity.json',dict(state_hash=model['state_hash'],nu_exact=str(nu),nu_binary64=float(nu),raw_state_changed=False))
    cresult=C.read(ROOT/P.CRESULT)
    pf_target,pf_record=saved_PF_target(cresult,nu);C.save(ns/'PF_C_target_saved.json',pf_record)
    guard=C.read(P.BASE/'guard_inputs_quoted.json')['axes'];results=[]
    def action(name,function):
        budget.take(name)
        C.save(ns/(name+'_claim.json'),dict(UTC=C.utc(),call=name,reserved_before_call=True))
        out,record=function()
        record.update(call=name,output_state_pairs=[[float(z.real),float(z.imag)] for z in out],UTC=C.utc(),binary64=True)
        C.save(ns/(name+'.json'),record)
        return record
    for key,method in (('PF','PF'),('PR0','PR')):
        budget.take(key+'_diagnostic')
        before=action(key+'_input',lambda:propagate(circuits[key]['input'],raw,nu))
        after=action(key+'_output',lambda:propagate(circuits[key]['output'],raw,nu))
        target=pf_target if key=='PF' else F(action('PR0_C_target',lambda:selected_C_action(model,joins[key],raw,nu))['signal'])
        t=next(t for t in targets if t['method']==method)
        gr=next(r for r in guard if (r['case'],r['epsilon_sig'],r['axis'])==('pauli_jw_'+method,'0.0005','cosine'))
        result=compare_signals(target,before['signal'],after['signal'],t['B_saved'],gr['rho_guard_exact'])
        result.update(key=key,method=method,axis='cosine',trajectory_id=t['trajectory_id'],scope='FIXED_PF' if key=='PF' else 'SELECTED_PR_INDEX0_ONLY')
        C.save(ns/(key+'_result.json'),result);results.append(result)
    C.save(ns/'results.json',results)
    used=C.read(ns/'call_ledger.json')['used']
    C.save(ns/'result_summary.json',dict(status='R0_EMPIRICAL_COMPLETE_MANDATORY_STOP',logical_diagnostics=2,
        successful_state_actions=sum((ns/(name+'.json')).is_file() for name in ACTION_LIMITS),
        total_state_action_attempts=sum(used[name] for name in ACTION_LIMITS),binary64_only=True,Arb_calls=0,new_compile=0,
        sampling=0,N_G_reanalysis=0,rho_upper=None,PR_law_rho=None,population_cost_tail='UNKNOWN',
        historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED',original_native='UNKNOWN_UNCHANGED',original_N_G_null=True,
        C_L='COMBINATION_BLOCKED',old_compiler='MISSING_EVIDENCE',formal_certificate_accepted=False,
        forbidden_calls=contract['forbidden_calls'],mandatory_STOP=True))
