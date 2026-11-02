"""FUTURE B-only binding: exactly 2 INPUT + 4 ARMS + 5 CONTEXTS.

Never run/imported during B-PRE. No Qiskit, reference operator, event, trajectory,
price builder, interval score or F1 dispatcher is imported/called by this path.
"""
from __future__ import annotations
from dataclasses import asdict
from fractions import Fraction
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace
from .contract import canonical,sha,schema_validate
from trottertracks.pr_dictionary_validation.fixture import (
    p0_candidate,specification_record,digest)
from trottertracks.pr_dictionary_validation.algebra import materialize_alternatives
from trottertracks.pr_dictionary_validation.prototype import (
    make_fragment_alternatives,canonicalize_arm,atom_record)
from trottertracks.pr_dictionary_validation.cases import catalog_atoms,bind_motif_labels


def encode(value):
    """Fractions and binary64 carry explicit distinct encodings."""
    if isinstance(value,Fraction):return str(value)
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('NaN/Inf forbidden')
        return dict(encoding='ieee754-binary64-hex',value=value.hex())
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(x) for x in value]
    if value is None or type(value) in (str,int,bool):return value
    raise TypeError('Unsupported output type: '+type(value).__name__)


def load_frozen_rte(root,expected_sha):
    """One future scientific module import, inside the first CONTEXTS case.

    The standalone fixed file imports only NumPy and stdlib. Do not import the
    broad trotterlib/__init__.py. Serialized records, not alias class instances,
    cross the B/F1 boundary. No old module or sys.modules[trotterlib] is replaced.
    """
    path=root/'src/trotterlib/rte.py'
    if sha(path.read_bytes())!=expected_sha:raise ValueError('Frozen RTE mismatch')
    name='_pr_dictionary_binding_frozen_rte'
    if name in sys.modules:raise ValueError('Unexpected preloaded RTE alias')
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ValueError('Cannot load frozen standalone RTE')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def record_map(mapping):
    return [dict(word=list(word),coefficient=str(value)) for word,value in sorted(mapping.items())]


def exact_lambda(atoms):
    return sum((abs(a.coefficient) for a in atoms),Fraction(0))


def tail_bridge(rte,atoms,name):
    # Narrow bridge equivalent to I make_tail, without importing trotterlib init.
    lam=exact_lambda(atoms)
    if lam<=0:raise ValueError('Fixed fixture requires nonzero residual')
    components=[]
    for a in atoms:
        probability=float(abs(a.coefficient)/lam);weight=float(abs(a.coefficient))
        if probability<=0 or weight<=0 or not math.isfinite(probability+weight):
            raise ValueError('Coefficient underflow/overflow')
        components.append(rte.RTEComponent(component_id=a.atom_id,probability=probability,
            coefficient_abs=weight,coefficient_sign=1 if a.coefficient>0 else -1,
            df_fragment_id=None,basis_id=a.basis_key,basis_hash=a.basis_key,
            is_identity=a.kind=='IDENTITY',
            diagonal_pauli_support=a.support if a.kind=='FRAME_Z' else None))
    mass=math.fsum(c.probability for c in components)
    loss=math.fsum(abs(c.probability-float(abs(a.coefficient)/lam))
                   for c,a in zip(components,atoms,strict=True))
    if abs(mass-1)>1e-12 or loss>1e-12:raise ValueError('Probability conversion qualification failure')
    return SimpleNamespace(tail_id=name,tail_hash=digest([atom_record(a) for a in atoms]),
        lambda_r=float(lam),exact_lambda=lam,components=tuple(components))


def law_record(rte,atoms,name,*,catalog=False):
    tail=tail_bridge(rte,atoms,name)
    delta=Fraction(1,20) if catalog else Fraction(1,10)
    r=1 if catalog else 2
    tau=tail.lambda_r*(float(delta)/r)
    # API accommodation only. Fixed K=2, no accuracy/order search or bias claim.
    bound=rte.step_taylor_truncation_residual_bound(tau,2)
    if not math.isfinite(bound) or bound<0:raise ValueError('Residual metadata invalid')
    tolerance=math.nextafter(bound,math.inf)
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('API tolerance unavailable')
    config,distribution=rte.make_rte_config(tail,evolution_time=float(delta),rte_steps=r,
        finite_taylor_order=2,truncation_tolerance=tolerance,seed=0)
    if config.finite_taylor_order!=2 or tuple(distribution.orders)!=(0,2):
        raise ValueError('Fixed law/cutoff changed')
    components=[asdict(x) for x in tail.components]
    angles={str(n):math.atan(distribution.dimensionless_step_time/(n+1)) for n in (0,2)}
    b_total=distribution.exact_finite_distribution**(1 if catalog else 4)
    if not math.isfinite(b_total):raise ValueError('Total normalization overflow')
    return dict(exact_lambda=str(tail.exact_lambda),angles=angles,B_total=b_total,config=asdict(config),
        distribution=asdict(distribution),components=components,
        context=dict(T='1/20' if catalog else '1/5',q=1 if catalog else 2,
                     delta=str(delta),r=r,K=2),
        truncation_policy='nextafter(existing_residual_bound,+inf)_API_ONLY_NOT_BIAS_BUDGET',
        semantic_status='METADATA_BOUND_NOT_F1_VALIDATED',
        price_status='PRICE_PROVIDER_UNBOUND',interval_status='UNPRICED_FALLBACK',
        shot_status='SHOT_COUNT_UNRESOLVED',recipe_prices=None,score=None,shots=None,
        F2_numeric_law_status='DEFERRED_NOT_WITHIN_FIVE_B_CONTEXTS')


def write_record(output,relative,payload,case_id,schema):
    schema_validate(schema,payload)
    data=canonical(payload);path=output/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    return dict(path=relative,sha256=sha(data),bytes=len(data),producer_case=case_id)


def execute(verified):
    """Executed only after FUTURE B approval; no pilot/retry/automatic F1."""
    root=verified['root'];out=verified['output'];cache={};records={};rte=None
    plan=verified['contract']['call_plan'];done=set();attempts=[]
    log_path=out/'run.log'
    with log_path.open('x') as stream:stream.write('B BEGIN\n')
    schema=verified['schemas']['output_record'];start_source=verified['contract']['source_files']
    def emit(case_id,kind,payload,relative):
        record=dict(schema_version='pr-dictionary-binding-v1',kind=kind,producer_case=case_id,
                    payload=encode(payload))
        selected_schema=schema
        if kind=='f1_handoff':
            record=dict(schema_version='pr-dictionary-f1-input-v1',kind=kind,producer_case=case_id,**encode(payload))
            selected_schema=verified['schemas']['f1_handoff_record']
        result=write_record(out,relative,record,case_id,selected_schema);records[case_id]=result
        return result
    for case in plan:
        cid=case['id']
        if cid in done or not set(case['requires']).issubset(done):raise ValueError('Invalid case order/duplicate')
        for source,expected in start_source.items():
            if sha((root/source).read_bytes())!=expected:raise ValueError('SOURCE changed during B')
        # Register attempt before science; no success overwrites this record.
        attempt=out/('attempt-'+cid+'.json')
        with attempt.open('x') as stream:stream.write(json.dumps(dict(case_id=cid,status='ATTEMPTED'))+'\n')
        attempts.append(cid)
        with log_path.open('a') as stream:stream.write(cid+' ATTEMPTED\n')
        if cid=='B.INPUT.physical':
            spec=p0_candidate()
            materialized=materialize_alternatives(spec)
            alternatives=make_fragment_alternatives(materialized)
            cache['spec']=spec;cache['alternatives']=alternatives
            payload=dict(specification=specification_record(spec),
                rational_frames={f.fragment_id:[[str(x) for x in row] for row in v]
                                 for f,v,_s,_p in materialized},
                alternatives=[dict(fragment=f.fragment_id,S=record_map(s),P=record_map(p))
                              for f,_v,s,p in materialized],
                typed_alternatives=[dict(fragment=x.fragment_id,S=[atom_record(a) for a in x.frame],
                                        P=[atom_record(a) for a in x.pauli]) for x in alternatives],
                state_specification=dict(occupied=[0,1],particles=2,register_modes=4,
                                         ancilla=4,materialized_state_vector=False))
            cache['physical_payload']=payload
            emit(cid,'physical_fixture',payload,'inputs/physical.json')
        elif cid=='B.INPUT.catalog':
            frame_atom=next(a for a in cache['alternatives'][0].frame if a.kind=='FRAME_Z')
            atoms=catalog_atoms(frame_atom.basis_key)
            if len(atoms)!=4 or tuple(a.atom_id for a in atoms)!=('A','B','C','I'):
                raise ValueError('Catalog identity mismatch')
            cache['catalog']=atoms;cache['lambda:catalog']=exact_lambda(atoms)
            payload=dict(atoms=[atom_record(a) for a in atoms],scalar='-1/31',
                T='1/20',q=1,r=1,K=2,exact_lambda=str(cache['lambda:catalog']),
                frame_reference=records['B.INPUT.physical']['sha256'],fifth_physical_arm=False)
            emit(cid,'catalog',payload,'inputs/catalog.json')
        elif case['family']=='B.ARMS':
            arm=cid.rsplit('.',1)[-1]
            atoms=canonicalize_arm(cache['alternatives'],arm)
            labels=bind_motif_labels(atoms)
            if len({a.atom_id for a in atoms})!=len(atoms):raise ValueError('Duplicate canonical atom')
            cache[arm]=atoms;cache['lambda:'+arm]=exact_lambda(atoms)
            payload=dict(arm=arm,atoms=[atom_record(a) for a in atoms],L=len(atoms),
                exact_lambda=str(cache['lambda:'+arm]),coverage=dict(zip(('I','u','s','v'),labels,strict=True)),
                physical_fixture_sha256=records['B.INPUT.physical']['sha256'],
                scalar_policy='faithful_identity_in_tail',
                hypothetical_F1_E=len(atoms)+len(atoms)**3,
                hypothetical_F1_component_probes=3*len(atoms),hypothetical_F2_trajectories=len(atoms)**2)
            emit(cid,'arm',payload,'arms/'+arm+'.json')
        elif case['family']=='B.CONTEXTS':
            name=cid.rsplit('.',1)[-1]
            if rte is None:rte=load_frozen_rte(root,start_source['src/trotterlib/rte.py'])
            payload=law_record(rte,cache[name],name,catalog=name=='catalog')
            if payload['exact_lambda']!=str(cache['lambda:'+name]):
                raise ValueError('Arm/law lambda mismatch')
            payload['binding_input_sha256']=records[case['requires'][0]]['sha256']
            payload['recipe_policy_identity']=dict(status='SOURCE_POLICY_ONLY_NOT_REALIZED',
                files=verified['contract'].get('recipe_policy_sources',{}))
            emit(cid,'law_metadata',payload,'laws/'+name+'.json')
        else:raise ValueError('Unregistered B case')
        done.add(cid)
        with (out/('completed-'+cid+'.json')).open('x') as stream:
            stream.write(json.dumps(dict(case_id=cid,status='COMPLETED',output=records[cid]),sort_keys=True)+'\n')
        with log_path.open('a') as stream:stream.write(cid+' COMPLETED\n')
    lengths={a:len(cache[a]) for a in ('SS','SP','PS','PP')}
    counts=dict(L=lengths,sum_E=sum(n+n**3 for n in lengths.values()),
        component_probes=3*sum(lengths.values()),Q=2*sum(n*n for n in lengths.values()),
        U_F2=None,U_F2_status='UNRESOLVED_UNTIL_IR_BINDING',
        F1_motif_controlled_builds=96,F1_full_wrapper_compile_requests=64,
        F1_catalog_controlled_builds=136)
    # Descriptors only: never expand finite events, motifs, trajectories or IR.
    ranges={a:dict(id_pattern='F1.LAW.'+a+'.{ordinal}',start=0,
                  stop_exclusive=n+n**3,event_objects_materialized=False) for a,n in lengths.items()}
    source_payload=dict(source_commit=verified['contract']['source_commit'],source_files=start_source,
                        source_closure_sha256=verified['source_closure_sha256'])
    emit('B.FINALIZE.source','source_manifest',source_payload,'manifests/source.json')
    environment_payload=dict(python_patch=verified['environment']['python_patch'],numpy='1.26.4',
        private_environment_receipt_sha256=verified['environment_sha256'],
        runtime_api_status='B_SCALAR_METADATA_PATH_EXERCISED_NOT_F1_VALIDATED')
    emit('B.FINALIZE.environment','environment_summary',environment_payload,'manifests/environment_public.json')
    handoff=dict(specification=cache['physical_payload']['specification'],
        arms={a:[atom_record(x) for x in cache[a]] for a in ('SS','SP','PS','PP')},
        rational_frames=cache['physical_payload']['rational_frames'],
        catalog_atoms=[atom_record(x) for x in cache['catalog']],literal_prices={},score_intervals={},
        status='INPUT_ONLY_PRICE_AND_F1_ENVIRONMENT_UNBOUND')
    handoff_record=emit('B.FINALIZE.handoff','f1_handoff',handoff,'handoff/f1_bundle.json')
    summary=dict(status='B_INPUTS_BOUND_F1_REVIEW_REQUIRED',counts=counts,
        physical_input_file_sha256=records['B.INPUT.physical']['sha256'],
        catalog_input_file_sha256=records['B.INPUT.catalog']['sha256'],
        F1_bundle_file_sha256=handoff_record['sha256'],
        F1_case_ranges=ranges,F1_final_payload_inventory=None,
        price_status='PRICE_PROVIDER_UNBOUND',interval_status='UNPRICED_FALLBACK',
        shot_status='SHOT_COUNT_UNRESOLVED',F1_authorized=False,F2_authorized=False)
    emit('B.FINALIZE','binding_summary',summary,'binding_summary.json')
    # Finalize is IO/hash/count metadata; no extra logical science case.
    index=dict(schema_version='pr-dictionary-binding-manifest-v1',records=records,
        source_closure_sha256=verified['source_closure_sha256'],
        environment_sha256=verified['environment_sha256'],approved_run_id=verified['run_id'],
        executed_cases=attempts,scientific_case_counts=verified['contract']['logical_counts'],
        price_status='PRICE_PROVIDER_UNBOUND',mandatory_stop=True)
    with (out/'manifest.json').open('xb') as stream:stream.write(canonical(index))
    with log_path.open('a') as stream:stream.write('B COMPLETE MANDATORY_STOP\n')
    return summary
