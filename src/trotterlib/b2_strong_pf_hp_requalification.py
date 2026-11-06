"""Five fixed strong-PF arms, HP construction and P1 all-round qualification.

Old evidence remains immutable. No new model/state solves, circuits, shots,
prices or branch decoding. An input failure stops the whole run; the first
ineligible round stops that arm. Packed step matrices are in-memory IPC only.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

import mpmath
import numpy as np

from . import b2_strong_pf_precision as hp
from . import b2_p1_strong_pf_h4_diagnosis as diagnostic

ROOT=Path(__file__).resolve().parents[2]
ARMS=(('Full_S2',4),('Full_S2',8),('Yoshida4',2),('Yoshida4',4),('Yoshida4',8))
PRECISIONS=(80,120)
PRECISION_SOURCE='75cb38e686ad8b82d00d787531d767e816e1331d'


def input_gate(ham,indices):
    rows=[]
    for name,a in [('one_body',ham.one_body)]+[(f'F{i}',g) for i,g in enumerate(ham.g_matrices)]:
        a=np.asarray(a)
        record=dict(block=name,dtype=str(a.dtype),shape=list(a.shape),
            exact_Hermitian=bool(np.array_equal(a,a.conj().T)),
            Hermiticity_max_abs_entry=float(np.max(abs(a-a.conj().T))),
            exact_spin_lift=bool(np.array_equal(a,np.kron(a[::2,::2],np.eye(2)))))
        try:
            hp.validate_matrix(a);record['status']='PASS'
        except ValueError as error:
            record.update(status='STOP_INPUT_INELIGIBLE',reason=str(error))
        rows.append(record)
    hp.checked_basis(ham.n_qubits,indices)
    return dict(status='PASS' if all(r['status']=='PASS' for r in rows) else 'STOP_INPUT_INELIGIBLE',blocks=rows,
        no_symmetrization_projection_or_fallback=True)


def pack_matrix(ctx,matrix):
    return tuple(tuple(ctx.mpc(matrix[i,j])._mpc_ for j in range(matrix.cols)) for i in range(matrix.rows))


def unpack_matrix(ctx,packed):
    return ctx.matrix([[ctx.make_mpc(value) for value in row] for row in packed])


def build_step_job(one,weights,factors,constant,indices,formula,substeps,digits):
    started=time.perf_counter()
    preparation=hp.prepare_precision(one,weights,factors,constant,digits=digits,basis_indices=indices)
    pair=hp.precise_step(preparation,formula=formula,substeps=substeps,delta_time=hp.legacy.p.DELTA)
    ctx=preparation.ctx
    return dict(formula=formula,substeps=substeps,digits=digits,
        native=pack_matrix(ctx,pair.native),CAR=pack_matrix(ctx,pair.car),
        diagnostics={k:diagnostic.hp_text(ctx,v) for k,v in pair.diagnostics.items()},
        component_diagnostics=[{k:diagnostic.hp_text(ctx,v) for k,v in row.items()} for row in preparation.diagnostics],
        schedule=[dict(block_index=i,time_fraction=str(t),time=float(t)) for i,t in pair.schedule],
        elapsed_seconds=time.perf_counter()-started)


def qualify_hp(m,native,car,phys,df,precision_checks,ctx):
    """HP checks precede the unchanged legacy radius/phase diagnostics."""
    to_complex=lambda z:complex(float(z.real),float(z.imag))
    converted={k:to_complex(v) for k,v in (('PF',native),('CAR',car))}
    result=hp.legacy.qualify(m,phys,df,converted['PF'],converted['CAR'])
    fixed=lambda x:ctx.mpf(float(x))
    if not native or not car:
        result.update(feasible=False,status='STOP_ARM_INELIGIBLE',
            reasons=result['reasons']+['HP_undefined_phase'],
            HP_bounds=dict(model=None,PF=None,implementation=None,
                PF_radius_lower=diagnostic.hp_text(ctx,abs(native)-fixed(hp.legacy.p.U)),
                total_with_reserved_RTE_and_statistical=None))
        return result
    phys_hp=ctx.mpc(phys.real,phys.imag);df_hp=ctx.mpc(df.real,df.imag)
    comparison=diagnostic.hp_difference(ctx,native,car)
    model=abs(ctx.arg(df_hp*ctx.conj(phys_hp)))+fixed(hp.legacy.p.U)
    pf=abs(ctx.arg(native*ctx.conj(df_hp)))+fixed(hp.legacy.p.U)
    implementation=abs(ctx.arg(native*ctx.conj(car)))+fixed(hp.legacy.p.U)
    radius=abs(native)-fixed(hp.legacy.p.U)
    total=fixed(hp.legacy.p.ALLOCATION.beta_total)+model+implementation+fixed(hp.legacy.p.CAPS['reserve'])
    reasons=list(result['reasons'])
    tests=dict(model=model<=fixed(hp.legacy.p.CAPS['model']),PF=pf<=fixed(hp.legacy.p.CAPS['PF']),
        implementation=implementation<=fixed(hp.legacy.p.CAPS['implementation']),
        positive_radius=0<radius<=1,total=total<=fixed(.4),precision=all(precision_checks.values()),
        independent=all(ctx.mpf(comparison[k])<=fixed(hp.legacy.p.MATCH) for k in ('complex_absolute','radius_absolute','phase_rad')))
    for name,passed in tests.items():
        if not passed:reasons.append('HP_'+name)
    conversion_errors={k:abs(value-ctx.mpc(converted[k].real,converted[k].imag))
        for k,value in (('PF',native),('CAR',car))}
    if any(v>fixed(hp.legacy.p.MATCH) for v in conversion_errors.values()):reasons.append('HP_to_binary64_conversion')
    result.update(feasible=not reasons,reasons=reasons,status='CONDITIONAL_PASS' if not reasons else 'STOP_ARM_INELIGIBLE',
        HP_bounds={k:diagnostic.hp_text(ctx,v) for k,v in dict(model=model,PF=pf,implementation=implementation,
            PF_radius_lower=radius,total_with_reserved_RTE_and_statistical=total).items()},
        HP_tests={k:bool(v) for k,v in tests.items()},HP_independent_difference=comparison,
        conversion_error={k:diagnostic.hp_text(ctx,v) for k,v in conversion_errors.items()})
    return result


def evaluate_in_order(evaluate,save):
    rows=[]
    for m in range(18):
        row=evaluate(m);rows.append(row);save(rows)
        if not row['qualification']['feasible']:
            return rows,'STOP_ARM_INELIGIBLE'
    return rows,'P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS'


def evaluate_arm(builds,psi64,references,save):
    contexts={};powers={};states={}
    for digits in PRECISIONS:
        ctx=mpmath.mp.clone();ctx.dps=digits;contexts[digits]=ctx
        powers[digits]={k:unpack_matrix(ctx,builds[digits][key]) for k,key in (('N','native'),('C','CAR'))}
        states[digits]=hp.hp_helpers().mp_matrix(ctx,np.asarray(psi64)[:,None])
    ctx=contexts[120];match=ctx.mpf(hp.legacy.p.MATCH)
    def evaluate(m):
        if m:
            for digits in PRECISIONS:
                for name in ('N','C'):powers[digits][name]=powers[digits][name]**2
        signals={};precision={};checks={}
        for name in ('N','C'):
            high=powers[120][name];low=diagnostic.mp_copy(ctx,powers[80][name])
            z=diagnostic.hp_expectation(high,states[120])
            z80=diagnostic.hp_expectation(powers[80][name],states[80]);z80=ctx.mpc(z80.real,z80.imag)
            difference=diagnostic.hp_difference(ctx,z,z80);operator=hp.maxabs(high-low)
            checks[name]=bool(operator<=match and all(ctx.mpf(difference[k])<=match for k in ('complex_absolute','radius_absolute','phase_rad')))
            precision[name]=dict(operator_max_abs_80_vs_120=diagnostic.hp_text(ctx,operator),signal_difference=difference,passed=checks[name])
            signals[name]=z
        prior=references[m];phys=hp.legacy.saved.complex_signal(prior['signals']['phys']);df=hp.legacy.saved.complex_signal(prior['signals']['DF'])
        qualification=qualify_hp(m,signals['N'],signals['C'],phys,df,checks,ctx)
        row=dict(m=m,Q=1<<m,T_Ha_inverse=(1<<m)*hp.legacy.p.DELTA,
            signals=dict(phys=prior['signals']['phys'],DF=prior['signals']['DF'],
                PF=diagnostic.hp_signal_record(ctx,signals['N']),independent_PF=diagnostic.hp_signal_record(ctx,signals['C'])),
            reference_provenance={k:v for k,v in prior.items() if k in ('evidence','result_commit','execution_SOURCE')},
            precision_check=precision,qualification=qualification,status=qualification['status'],
            normalization=1.,attenuation=1.,finite_RTE_disk=0.)
        print('round m='+str(m)+' '+row['status'],flush=True)
        return row
    return evaluate_in_order(evaluate,save)


def save_result(output,result):
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(private_dir,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source=hp.legacy.p.git('rev-parse','HEAD').decode().strip()
    paths=['src/trotterlib/b2_strong_pf_hp_requalification.py','scripts/run_b2_strong_pf_hp_requalification.py','tests/test_b2_strong_pf_hp_requalification.py']
    for path in paths:hp.legacy.p.require((ROOT/path).read_bytes()==hp.legacy.p.git('show',source+':'+path),'COMMITTED_SOURCE')
    result=dict(schema_version='b2_strong_pf_fixed_five_arm_HP_requalification_v1',execution_SOURCE=source,
        SOURCE_SHA256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        started_UTC=datetime.now(timezone.utc).isoformat(),arms=[],maximum_points=90,
        arm_set=[dict(formula=f,substeps=s) for f,s in ARMS],precision_dps=list(PRECISIONS),
        historical_results_overwritten=False,mandatory_stop=True,guarantee_status='not_certified',
        not_assessed=['Branch/energy reconstruction','shots/95% measurement success','compiled/all-support correctness','prices/total cost'])
    try:
        for path in ('src/trotterlib/b2_strong_pf_precision.py','scripts/validate_b2_strong_pf_precision.py','tests/test_b2_strong_pf_precision.py',
            'src/trotterlib/b2_p1_strong_pf_h4_diagnosis.py','scripts/diagnose_b2_p1_strong_pf_numerics.py'):
            hp.legacy.p.require((ROOT/path).read_bytes()==hp.legacy.p.git('show',PRECISION_SOURCE+':'+path),'FIXED_PRECISION_SOURCE')
        original,meta=hp.legacy.saved.fixed_json(diagnostic.ORIGINAL_COMMIT,diagnostic.ORIGINAL_PATH)
        for path,sha in original['SOURCE_SHA256'].items():
            hp.legacy.p.require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,'FIXED_ORIGINAL_SOURCE')
        references,provenance,identity,conditions,refmeta=hp.legacy.checked_references()
        hp.legacy.p.require(original['execution_SOURCE']==diagnostic.ORIGINAL_SOURCE and
            original['scientific_context']==identity and original['P1_conditions']==conditions,'ORIGINAL_P1_CONTEXT')
        inputs=hp.legacy.p.load_private_inputs(private_dir)
        hp.legacy.p.require(all(inputs['public'][k]==v for k,v in identity.items()),'PRIVATE_CONTEXT')
        result.update(input_identity=inputs['public'],scientific_context=identity,P1_conditions=conditions,
            original_result=meta,saved_references=refmeta,precision_SOURCE=PRECISION_SOURCE,
            reused_SOURCE_SHA256=original['SOURCE_SHA256'],historical_qualified_arms=original['qualified_arms'])
        ham=inputs['ham'];gate=input_gate(ham,inputs['indices']);result['input_gate']=gate
        hp.legacy.p.require(gate['status']=='PASS','STOP_INPUT_INELIGIBLE')
        result['environment']=dict(host=os.uname().nodename,CPU_affinity=len(os.sched_getaffinity(0)),
            load_average=os.getloadavg(),numpy=np.__version__,mpmath=mpmath.__version__,
            scipy=importlib.metadata.version('scipy'),worker_processes=len(ARMS)*len(PRECISIONS),
            resource_policy='one task per independent arm/precision; no CPU/RAM/BLAS caps')
        for formula,s in ARMS:
            result['arms'].append(dict(arm_id=formula+'_s'+str(s),formula=formula,substeps=s,rounds=[],status='BUILDING_HP_STEPS'))
        save_result(output,result);builds={i:{} for i in range(len(ARMS))}
        with ProcessPoolExecutor(max_workers=len(ARMS)*len(PRECISIONS),mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(build_step_job,ham.one_body,ham.lambdas,ham.g_matrices,ham.constant,inputs['indices'],f,s,d):(i,d)
                for i,(f,s) in enumerate(ARMS) for d in PRECISIONS}
            for future in as_completed(futures):
                i,d=futures[future];build=future.result();builds[i][d]=build;arm=result['arms'][i]
                print(arm['arm_id']+' step construction '+str(d)+' digits complete',flush=True)
                if len(builds[i])!=len(PRECISIONS):continue
                prior=next(a for a in original['arms'] if a['arm_id']==arm['arm_id'])
                hp.legacy.p.require(builds[i][120]['schedule']==prior['base_delta_schedule'],'FIXED_SCHEDULE')
                arm['step_diagnostics']={str(d):{k:v for k,v in b.items() if k not in ('native','CAR','schedule')} for d,b in builds[i].items()}
                def save(rows):
                    arm['rounds']=rows;save_result(output,result)
                rows,status=evaluate_arm(builds[i],inputs['sector_state'],references,save)
                arm.update(rounds=rows,status=status,first_ineligible_round=None if status=='P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS' else rows[-1]['m'])
                builds[i].clear();save_result(output,result)
        result['new_qualified_arms']=[a['arm_id'] for a in result['arms'] if a['status']=='P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS']
        result['evaluated_points']=sum(len(a['rounds']) for a in result['arms'])
        result['status']='FIXED_FIVE_ARM_HP_REQUALIFICATION_COMPLETE'
        for name,path in inputs['files'].items():
            hp.legacy.p.require(hashlib.sha256(path.read_bytes()).hexdigest()==inputs['public']['file_SHA256'][name],'PRIVATE_INPUT_CHANGED')
    except Exception as error:
        result.update(status='STOP_SHARED_INPUT_SOURCE_OR_NUMERICAL_FAILURE',error_type=type(error).__name__,error=str(error))
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save_result(output,result)
        with (output/'rounds.csv').open('w',newline='') as stream:
            fields=['arm_id','m','Q','T_Ha_inverse','PF_real','PF_imag','PF_radius','PF_phase_rad','PF_bound','model_bound','implementation_bound','radius_lower','status','reasons']
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for arm in result['arms']:
                for row in arm['rounds']:
                    sig=row['signals']['PF'];q=row['qualification'];b=q['HP_bounds']
                    writer.writerow(dict(arm_id=arm['arm_id'],**{k:row[k] for k in ('m','Q','T_Ha_inverse','status')},
                        **{'PF_'+k:v for k,v in sig.items()},PF_bound=b['PF'],model_bound=b['model'],
                        implementation_bound=b['implementation'],radius_lower=b['PF_radius_lower'],reasons=';'.join(q['reasons'])))
    return result
