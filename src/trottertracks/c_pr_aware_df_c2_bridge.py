"""Opt-in C2 Reference Bridge. Imports/passive preflight do no molecular work.

Saved CAR generators and RTE results are reused. The new scientific endpoint
is a numerical sector-reference correspondence pilot, never a P0 all-law proof.
"""
from fractions import Fraction
import importlib.metadata
import math
from pathlib import Path

from . import c1_stage1_execution as io
from . import c_pr_aware_df_stage_c2 as table

VERSION='C2_bridge_saved_CAR_K0_mean_v1'
SCOPE='C_PR_AWARE_DF_C2_REFERENCE_BRIDGE_V1'
FAMILY='artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27'
FRESH=FAMILY+'/Bridge-run01'
ARMS=('baseline','proposed')
POLICY=dict(table.POLICY, h='1/40',spin_order='interleaved_2i_plus_sigma',
            reference_coverage='complete sector36 factors;not native full512')
ALLOWANCES=dict(operator_component_abs=1e-10,signal_complex=2e-10,
    normalization=1e-12,mass_relative=1e-12,mass_absolute=1e-14,
    rtol=0,global_phase_fit=False,rigorous_enclosure=None)
LIMITS=dict(saved_json_load=1,provenance=1,input_load=1,sector_context=1,
    arm_arrays_load=2,arm_replay=2,atom_context=2,atom_audit=2,event_mean=2,
    head_reference=2,kernel_bridge=2,round_bridge=26,arm_summary=2,summary=1)
MAX_CALLS=sum(LIMITS.values())


def compare(actual,expected,label):
    from .c1_stage3_molecular import component_check
    return component_check(actual,expected,ALLOWANCES['operator_component_abs'],label=label)


def replay(packet,data,saved,source,input_identity):
    """Reuse actual general-K/split/make_tail path;never build a circuit."""
    from . import c_pr_aware_df_stage_b as b
    from .pr_dictionary_validation.prototype import atom_record
    arm=b.prepare_arm(**packet,**data,role=source['role'],
        source_identity=source['export_sha256'],input_identity=input_identity,ld=2)
    tail,config,dist=b.configure(arm,Fraction(1,40),16,0)
    if config is None or dist.orders!=(0,):raise ValueError('REGISTERED_K0_TAIL')
    if [atom_record(a) for a in arm.plan.atoms]!=[
            dict(x,origins=tuple(x['origins']),word=tuple(x['word'])) for x in saved['atoms_catalog']]:
        raise ValueError('ATOM_CATALOG_CHANGED')
    table.replay_saved_table(saved)
    if len(tail.components)!=len(saved['components']):raise ValueError('COMPONENT_COUNT')
    for c,old in zip(tail.components,saved['components'],strict=True):
        if any(getattr(c,k)!=old[k] for k in ('component_id','probability','coefficient_abs','coefficient_sign','basis_id')):
            raise ValueError('COMPONENT_OR_ORDER_CHANGED')
    for a,z in [(tail.lambda_r,saved['lambda_R']),
                (config.dimensionless_step_time,saved['configuration']['tau']),
                (config.distribution_normalization,saved['finite_distribution']['normalization'])]:
        if not math.isclose(a,z,rel_tol=ALLOWANCES['mass_relative'],abs_tol=ALLOWANCES['mass_absolute']):
            raise ValueError('LAMBDA_TAU_NORMALIZATION_CHANGED')
    if arm.plan.extracted_identity!=saved['identity']:raise ValueError('TAIL_IDENTITY_CHANGED')
    return arm,tail,config,dist


def atom_audit(arm,tail,operators,saved_generators,table_result):
    """Grouped all finite atoms, with existing CAR and component diagnostics."""
    import numpy as np
    if len(operators)!=len(tail.components) or len(operators)!=216:raise ValueError('FINITE_ATOM_COVERAGE')
    I=np.eye(len(saved_generators[0]),dtype=complex);records=[]
    for a,c in zip(arm.plan.atoms,tail.components,strict=True):
        P=operators[a.atom_id]
        records.append(dict(atom_id=a.atom_id,frame_id=a.frame_id,support=list(a.support),
            coefficient=str(a.coefficient),signed=True,
            hermitian=compare(P,P.conj().T,'atom Hermitian:'+a.atom_id),
            involution=compare(P@P,I,'atom involution:'+a.atom_id)))
    coeffs=[abs(a.coefficient) for a in arm.plan.atoms];total=sum(coeffs,Fraction())
    exact_prob=[str(x/total) for x in coeffs]
    if (table_result['status']!='EXACT_CATEGORICAL_TABLE_DISTANCE_UNDER_IID_GRID_MODEL'
            or exact_prob!=table_result['ideal_probabilities']):raise ValueError('C2B_IDEAL_TARGET_CHANGED')
    weighted=sum((float(abs(a.coefficient))*operators[a.atom_id] for a in arm.plan.atoms),np.zeros_like(I))
    raw=sum(saved_generators[3:],np.zeros_like(I))-arm.plan.extracted_identity*I
    residual=compare(weighted,raw,'saved CAR tail/nonidentity atom sum')
    return dict(status='FINITE_SECTOR_ATOM_NUMERICAL_PASS',atoms=records,
        raw_tail=residual,exact_mass=str(total),float_mass=tail.lambda_r,
        coefficient_identity_rounding_enclosed=False,reference_uncertainty=None)


def event_mean(arm,tail,dist,operators,table_result):
    """All K0 atoms once, not all16-event words;no sampling or compiler."""
    import numpy as np
    from trotterlib.rte import _make_event,event_unitary
    I=np.eye(len(next(iter(operators.values()))),dtype=complex)
    mean=np.zeros_like(I);average_P=np.zeros_like(I);unitarity=[]
    for i,(atom,prob) in enumerate(zip(arm.plan.atoms,table_result['ideal_probabilities'],strict=True)):
        p=float(Fraction(prob));event=_make_event((i,),tail.components,dist,0)
        expected=math.atan(dist.dimensionless_step_time)*(1 if atom.coefficient>0 else -1)
        if (event.taylor_order!=0 or event.phase!=1 or event.product_component_ids
                or event.unsigned_rotation_angle!=expected):raise ValueError('SIGNED_ANGLE_OR_K0_CHANGED')
        E=event_unitary(event,operators)
        unitarity.append(dict(atom_id=atom.atom_id,report=compare(E.conj().T@E,I,'event unitary:'+atom.atom_id)))
        mean+=p*E;average_P+=p*operators[atom.atom_id]
    angle=math.atan(dist.dimensionless_step_time)
    formula=math.cos(angle)*I-1j*math.sin(angle)*average_P
    report=compare(mean,formula,'weighted K0 event identity')
    return dict(matrix=mean,report=dict(status='K0_FINITE_MEAN_NUMERICAL_PASS',
        unitary_events=unitarity,mean_identity=report,angle=angle,
        normalization=dist.exact_finite_distribution,scalar_phase_excluded=True,
        reference_uncertainty=None,rigorous_operator_bound=None))


def heads(arm,generators):
    from .c1_stage3_reference import _ordered_exponentials
    schedule=tuple((i,Fraction(1,80)) for i in range(3));cache={}
    return dict(left=_ordered_exponentials(generators,schedule,cache),
        right=_ordered_exponentials(generators,tuple(reversed(schedule)),cache))


def kernel_bridge(arm,config,mean,head,saved):
    """Compare averaged atom kernel to saved raw CAR finite-RTE reference.

    The saved corrected kernel is reused;PF/exact-H evaluation is absent.
    A second normalization path exposes b**r vs exp(r*log(b)) roundoff.
    """
    import numpy as np
    from trotterlib.rte import finite_rte_corrected_operator
    from .c1_stage3_accuracy import _legacy
    n=len(saved['generators'][0]);I=np.eye(n,dtype=complex)
    raw_tail=sum(saved['generators'][3:],np.zeros_like(I))-arm.plan.extracted_identity*I
    corrected_tail=finite_rte_corrected_operator(raw_tail/config.lambda_r,config)
    phase=np.exp(-1j*float(Fraction(1,40))*arm.plan.extracted_identity)
    corrected=phase*head['right']@corrected_tail@head['left']
    gate=compare(corrected,saved['corrected'],'saved B1 corrected reference replay')
    attenuation=_legacy('rte').finite_rte_attenuation(config,tail_evolutions=1)
    raw_kernel=attenuation*saved['corrected']
    atom_kernel=phase*head['right']@np.linalg.matrix_power(mean['matrix'],config.rte_steps)@head['left']
    residual=compare(atom_kernel,raw_kernel,'K0 atom mean/saved raw CAR kernel')
    return dict(atom=atom_kernel,raw_reference=raw_kernel,
        saved_corrected=saved['corrected'],config=config,
        report=dict(status='KERNEL_REFERENCE_BRIDGE_NUMERICAL_PASS',corrected_replay=gate,
            atom_vs_raw=residual,attenuation=attenuation,tail_identity_phase_parts=[phase.real,phase.imag],
            nuclear_scalar='retained once in saved G0 head;not reapplied',
            numerical_reference_uncertainty=None,rigorous_bridge_bound=None))


def round_bridge(context,m,saved_row,psi,law):
    """Diagnostic differences and signal replay only;no PF/RTE error recompute."""
    import numpy as np
    from .c1_stage3_accuracy import _legacy
    n=4*(1<<m)
    atom=np.linalg.matrix_power(context['atom'],n)
    raw_power=np.linalg.matrix_power(context['raw_reference'],n)
    A=_legacy('rte').finite_rte_attenuation(context['config'],tail_evolutions=n)
    corrected_raw=A*np.linalg.matrix_power(context['saved_corrected'],n)
    if law['m']!=m or law['status']!='CONDITIONAL_IID_TABLE_BOUND':raise ValueError('C2B_ROUND_MODEL')
    if abs(A-saved_row['reference']['attenuation'])>ALLOWANCES['normalization']:raise ValueError('SAVED_ATTENUATION_CHANGED')
    residual=compare(raw_power,corrected_raw,'normalization power vs exp-log replay')
    z=complex(np.vdot(psi,corrected_raw@psi));old=complex(*saved_row['reference']['raw_signal_parts'])
    signal_error=abs(z-old)
    if signal_error>ALLOWANCES['signal_complex']:raise ValueError('SAVED_RAW_SIGNAL_REPLAY')
    difference=atom-corrected_raw
    return dict(m=m,status='BRIDGE_NUMERICAL_CORRESPONDENCE_OBSERVED',occurrences=n,
        attenuation=A,operator_distance_observed=float(np.linalg.norm(difference,2)),
        frobenius_observed=float(np.linalg.norm(difference,'fro')),
        raw_signal_distance_observed=float(abs(np.vdot(psi,difference@psi))),
        saved_signal_replay_error=signal_error,normalization_path_residual=residual,
        conditional_categorical_law=law,
        saved_PF_operator_error=saved_row['reference']['pf_operator_error'],
        saved_corrected_RTE_operator_error=saved_row['reference']['rte_corrected_operator_error'],
        state_alias_qualification='saved own-DF ground/alias numerical PASS;no new eigensolve',
        bridge_raw_signal_bound=None,numerical_reference_uncertainty=None,
        synthesis_signal_bound=None,actual_implemented_law_error=None,
        guaranteed_shots=None,guaranteed_total=None,CA_achieved=None,
        qualification='binary64 sector36 diagnostic;not a rounding enclosure or all-law synthesis proof')


def signal_schema(*,m,attenuation,pf,rte,categorical,bridge,synthesis,reference,
                  state_alias,random_model):
    """Separate raw units, error coverage and conditional models;no shot calls.

    Bound inputs are proof *inputs*, never certified by a label. Float
    diagnostic, missing bridge/reference, or a different compiler path cannot
    produce a qualified error_evidence packet. Coefficient-rounding pieces of
    categorical TV and reference uncertainty are counted once, not duplicated.
    """
    if not 0<=attenuation<=1 or not state_alias:raise ValueError('STATE_ALIAS_OR_ATTENUATION')
    for x in (pf,rte):
        if not math.isfinite(x) or x<0:raise ValueError('SAVED_OPERATOR_ERROR')
    entries=dict(categorical=categorical,bridge=bridge,synthesis=synthesis,reference=reference)
    missing=[];values={}
    for name,row in entries.items():
        if row is None or row.get('bound') is None or row.get('kind') not in ('analytic_bound','enclosed_bound'):
            missing.append(name);values[name]=None;continue
        if row.get('norm')!='raw_complex_signal' or row.get('m')!=m or not row.get('provenance'):
            raise ValueError('RAW_BOUND_SCOPE')
        if name=='synthesis' and row.get('compiler_coverage')!='all_final_whole_kernel_P0_outputs':
            raise ValueError('COMPILER_PATH_COVERAGE')
        if (name=='bridge' and row.get('reference_uncertainty_included') is not False):
            raise ValueError('REFERENCE_DOUBLE_COUNT_SCOPE')
        value=table.upper_float(table.rational(row['bound']))
        if not math.isfinite(value) or value<0:raise ValueError('BOUND_VALUE')
        values[name]=value
    if random_model!=table.IID:missing.append('random_model')
    # These are schema previews, not a new actual-implementation C1 lane.
    return dict(m=m,status='UNRESOLVED_INPUTS' if missing else 'CONDITIONAL_MODEL_INPUTS_COMPLETE',
        random_model=random_model,state_alias=state_alias,
        raw_components=dict(PF=attenuation*pf,corrected_RTE=attenuation*rte,**values),
        missing=missing,conditional_total_raw=None if missing else table.upper_float(sum(
            (table.rational(v) for v in [attenuation*pf,attenuation*rte,*values.values()]),Fraction())),
        source_qualification='B1/DF numerical pilot retained',proof_inputs_verified_by_schema=False,
        error_evidence=None,actual_implemented_law_error=None,
        guaranteed_shots=None,guaranteed_total=None,actual_CA_achieved=None)


def categorical_input(law,source):
    """Translate a pinned C2B conditional bound;not complete implemented law."""
    if (law['status']!='CONDITIONAL_IID_TABLE_BOUND' or law['random_model']!=table.IID
            or law['norm']!='raw_complex_signal' or not source):raise ValueError('C2B_BOUND_SCOPE')
    value=table.rational(law['conditional_raw_complex_bound_exact'])
    if not 0<=value<=2:raise ValueError('C2B_BOUND_RANGE')
    return dict(m=law['m'],kind='analytic_bound',norm='raw_complex_signal',
        bound=str(value),provenance=source,random_model=table.IID,
        conditional_on='same exact unitary trajectory map;absolute complex matrix element<=1',
        reference_bridge_included=False,PCG64_IID_proven=False,
        actual_implemented_law_error=None)


def uniform_synthesis_preview(factors,shells,m):
    """Use existing proof-input propagator. Never infer finite-word coverage."""
    if type(m) is not int or not 0<=m<=12:raise ValueError('REGISTERED_ROUND')
    result=table.synthesis_preview(factors,path='whole_kernel_P0',occurrences=4*(1<<m),shells=shells)
    return dict(result,m=m,whole_law_synthesis=None,guaranteed_shots=None,guaranteed_total=None)


def primitive_requirements():
    """Coverage inventory;counts from fixed structure,never inspect H4 arrays."""
    return dict(status='STRUCTURAL_REQUIREMENTS_NOT_A_BOUND',per_arm=dict(
        Gaussian_bases=9,head_relative_transitions=4,tail_frames=6,
        supports_per_frame=36,signed_atoms=216,events_per_kernel=16,
        tail_basis_runs='1..16;sharing exact basis keys only',
        head_basis_transitions='two independent3-factor halves;no head-tail fusion',
        scalar_phase='nuclear scalar once plus extracted tail identity',
        shells=['ancilla H','X H','Y Sdg then H'],actual_inverse='reverse sequence and actual gate inverses'),
        finite_inputs=['9 basis U matrices and4 directed relative products per arm',
          '216 atom supports/signs and K0 common unsigned angle per arm',
          '2 deterministic half-heads,scalar/identity phase and3 common shells',
          'all generated primitive gate definitions and full P0 numerical transformations'],
        raw_route='local operator error enclosures plus unitary telescoping could cover all words',
        P0_obstacle='finite primitive alphabet does not bound numeric1q run resynthesis across variable event boundaries',
        needed_proofs=['uniform Gaussian/relative phase/operator enclosures',
          'whole-P0 translation/resynthesis/mod2pi/declared-phase arithmetic coverage',
          'reference errors and shell error in same raw complex norm',
          'actual-adjoint proof with identical shared-state handling'],
        all_law_synthesis_bound=None,cost_support=None,
        sampled_residual_is_bound=False,event_P0_sum_is_whole_P0_bound=False)


def provenance(records):
    expected={'B1':('B1_MECHANISM_EVALUATION_COMPLETE',141,'7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'),
        'C2B':('C2B_CONDITIONAL_IID_TABLE_ANALYSIS_COMPLETE',35,'3c1bb2d9ce46c29205594a42b090c0811fbafd27')}
    for k,(status,count,commit) in expected.items():
        stop=records[k+'_STOP']
        if (records[k+'_result']['status']!=status or stop['scientific_units']!=count or stop['failed']
                or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
                or records[k+'_SOURCE']['execution_commit']!=commit):raise ValueError('SAVED_PROVENANCE:'+k)
    if any(records['C2B_result'][k] is not None for k in ('actual_implemented_law_error','whole_law_synthesis','guaranteed_shots','guaranteed_total')):
        raise ValueError('FALSE_PRIOR_QUALIFICATION')
    for arm in ARMS:
        if (records['B1_result']['arms'][arm]['classification']['status']!='POTENTIALLY_FEASIBLE'
                or records[arm+'_ground']['status']!='PASS'
                or [x['m'] for x in records['B1_result']['arms'][arm]['rounds']]!=list(range(13))):
            raise ValueError('SAVED_ARM_OR_GROUND_ALIAS')
    return dict(status='PASS',old_STOP_reused_as_permission=False)


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    if (c['scope']!=SCOPE or c['implementation']!=VERSION or c['status']!='UNAPPROVED_SCIENCE_CONTRACT_DRAFT'
            or c['policy']!=POLICY or c['allowances']!=ALLOWANCES or c['call_limits']!=LIMITS
            or c['maximum_logical_units']!=MAX_CALLS or c['fresh_output']!=FRESH or c['retry'] or c['resume']):
        raise ValueError('BRIDGE_SCOPE')
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    required={c['stage_A_input'],*c['records'].values(),*c['exports'].values(),*c['references'].values(),*c['grounds'].values()}
    if not required<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT')
    for k,want in c['packages'].items():
        if importlib.metadata.version(k)!=want:raise ValueError('PACKAGE:'+k)
    provenance({k:io.read(root/p) for k,p in c['records'].items()})
    return c


def execute(root,c,out,call):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from . import c_pr_aware_df_stage_b as b
    from . import c_pr_aware_df_stage_b_runner as old
    from .c1_molecule_extension import SectorCAR
    from .c1_stage3_integrated import resource_plan
    records=call('saved_json_load','fixed B1/C2B JSON',lambda:{k:io.read(Path(root)/p) for k,p in c['records'].items()})
    io.save(out/'provenance.json',call('provenance','prior SOURCE and STOP',lambda:provenance(records)))
    packet=call('input_load','saved original h/g/constant snapshot;no physical construction',lambda:old.load(Path(root)/c['stage_A_input'],('h','g','constant')))
    ctx=call('sector_context','complete spin-preserving CAR sector36',lambda:SectorCAR(4))
    prepared={}
    for arm in ARMS:
        def arrays(arm=arm):
            data=old.load(Path(root)/c['exports'][arm],('J','U'))
            saved=old.load(Path(root)/c['references'][arm],('generators','corrected','full_input_indices'))
            psi=old.load(Path(root)/c['grounds'][arm],('sector_state',))['sector_state']
            b.stage1.finite_arrays(*data.values(),*saved.values(),psi)
            if data['J'].shape!=(8,4,4) or data['U'].shape!=(8,4,4) or saved['generators'].shape!=(9,36,36) or saved['corrected'].shape!=(36,36) or psi.shape!=(36,):raise ValueError('ARRAY_SHAPE')
            if abs(np.vdot(psi,psi)-1)>ALLOWANCES['normalization']:raise ValueError('GROUND_NORMALIZATION')
            if not np.array_equal(saved['full_input_indices'],b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices]):
                raise ValueError('SPIN_SECTOR_BIT_ORDER_CHANGED')
            return data,saved,psi
        data,saved,psi=call('arm_arrays_load',arm,arrays)
        a,tail,config,dist=call('arm_replay',arm,lambda:replay(packet,data,records[arm+'_tail'],dict(role=arm,export_sha256=c['saved_inputs'][c['exports'][arm]]),c['original_input_sha256']))
        operators=call('atom_context',arm,lambda:b.atom_operators(ctx,a))
        tab=records['C2B_result']['arms'][arm]['table']
        io.save(out/(arm+'_atoms.json'),call('atom_audit',arm,lambda:atom_audit(a,tail,operators,saved['generators'],tab)))
        mean=call('event_mean',arm,lambda:event_mean(a,tail,dist,operators,tab));io.save(out/(arm+'_events.json'),mean['report'])
        head=call('head_reference',arm,lambda:heads(a,saved['generators']))
        context=call('kernel_bridge',arm,lambda:kernel_bridge(a,config,mean,head,saved));io.save(out/(arm+'_kernel.json'),context['report'])
        old.save_arrays(out/(arm+'_mean_reference.npz'),atom_kernel=context['atom'],raw_reference=context['raw_reference'],event_mean=mean['matrix'])
        prepared[arm]=(context,psi)
    plan=resource_plan(task_count=2,full_dimension=36);io.save(out/'resource_plan.json',plan)
    def task(arm):
        context,psi=prepared[arm];rounds=[]
        for m in range(13):
            row=call('round_bridge',arm+':m'+str(m),lambda m=m:round_bridge(context,m,records['B1_result']['arms'][arm]['rounds'][m],psi,records['C2B_result']['arms'][arm]['preview']['conditional_law_rows'][m]))
            io.save(out/(arm+f'_m{m:02d}.json'),row);rounds.append(row)
        def summarize():
            return dict(status='REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE',rounds=rounds,
                bridge_qualified=False,reference_uncertainty=None,whole_law_synthesis=None,
                actual_implemented_law_error=None,guaranteed_shots=None,guaranteed_total=None)
        value=call('arm_summary',arm,summarize);io.save(out/(arm+'_summary.json'),value);return arm,value
    results={}
    with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            for f in as_completed([pool.submit(task,a) for a in ARMS]):
                key,val=f.result();results[key]=val
    def summary():return dict(status='C2_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE',arms=results,
        scientific_units=sum(call.counts.values()),rigorous_bridge_bound=None,whole_law_synthesis=None,
        actual_implemented_law_error=None,CA_achieved=None,guaranteed_shots=None,guaranteed_total=None,
        sampling=0,fit=0,build_compile=0,C1_reanalysis=False,mandatory_stop=True)
    value=call('summary','registered bridge only',summary);io.save(out/'result.json',value);return value


def run(root,path,output):
    import resource,subprocess,time
    from .c1_stage3_integrated import Calls
    root=Path(root).resolve();c=preflight(root,path);out=Path(output).resolve()
    if out!=root/FRESH or out.exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    out.mkdir(parents=True);call=Calls(out,LIMITS,MAX_CALLS);wall=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',dict(io.read(root/c['source_manifest']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();io.save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        io.save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-wall,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
