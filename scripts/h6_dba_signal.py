"""H6 PR boundary signals using archived native blocks and frozen reference."""
from copy import copy,deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import h6_dba_contract as c
import h6_dba_native as native
from h6_dba_science import checked_results
import h8_key_compile_parallel as pool


class Progress:
    def phase(self,*a,**kw):pass
    def update(self,*a,**kw):pass


def prefix_view(whole,ham,prefix,m):
    """New declared prefix; no orbital/SCF/DF/state or Gaussian regeneration."""
    from trotterlib.df_rte_tail import extract_df_diagonal_tail,prepare_df_rte_event_inputs
    from trotterlib.df_partial_randomized_pf import df_fragment_weight
    from trotterlib.df_partial_s2 import _block_hash_payload
    rank=len(ham.lambdas);tail_blocks=whole.deterministic_blocks[prefix+1:]
    c.require(tuple(b.original_fragment_index for b in tail_blocks)==tuple(range(prefix,rank)), 'FROZEN_TAIL_ORDER')
    c.require(all(float(ham.lambdas[b.original_fragment_index]).hex()==b.lam.hex() for b in tail_blocks),
              'FROZEN_TAIL_COEFFICIENTS')
    proxy=math.fsum(df_fragment_weight(ham,i,weight_rule='lambda_frobenius_squared') for i in range(prefix,rank))
    tail=extract_df_diagonal_tail('H6_DBA_frozen_tail_p'+str(prefix),
        [SimpleNamespace(eta=m.np.asarray(b.diagonal_eta),lam=b.lam,U_ops=b.runtime_basis_operations) for b in tail_blocks],
        fragment_ids=[f'df-fragment-{i}' for i in range(prefix,rank)],identity_policy=whole.identity_policy,
        coefficient_atol=0.,ranking_proxy_lambda_r=proxy)
    blocks=whole.deterministic_blocks[:prefix+1]
    payload=dict(policy='H6_DBA_ARCHIVED_BLOCK_PREFIX_VIEW',full_preparation=whole.preparation_hash,
                 prefix=prefix,blocks=[_block_hash_payload(b) for b in blocks],tail_hash=tail.tail_hash)
    return replace(whole,ld=prefix,deterministic_blocks=blocks,
        deterministic_block_order=tuple(b.block_id for b in blocks),deterministic_fragment_indices=tuple(range(prefix)),
        randomized_block_indices=tuple(range(prefix,rank)),partition_hash=c.digest([whole.hamiltonian_hash,prefix,'generation_order']),
        preparation_hash=c.digest(payload),tail_extraction=tail,rte_preparation=prepare_df_rte_event_inputs(tail),
        ranking_proxy_lambda_r=proxy,exact_rte_lambda_r=tail.rte_lambda_r,
        extracted_identity_coefficient=tail.deterministic_identity_coefficient,
        threshold_dropped_component_count=tail.extraction_metadata.threshold_dropped_component_count,
        threshold_dropped_coefficient_l1=tail.extraction_metadata.threshold_dropped_coefficient_l1,
        threshold_operator_error_bound=tail.extraction_metadata.threshold_operator_error_bound)


def forward_finite(state,actions,tail,cell,scalar,lambda_r,np):
    """Independent forward polynomial; native uses the frozen Horner recurrence."""
    tau=lambda_r*.8/cell['R']
    b=math.fsum(abs(tau)**k/math.factorial(k)*math.hypot(1.,tau/(k+1)) for k in range(0,cell['K']+1,2))
    signals={};vectors={};maximum=1.;tail_count=0;det_count=0
    def checked(v):
        nonlocal maximum
        c.require(np.isfinite(v).all(),'NONFINITE_FORWARD_STAGE')
        maximum=max(maximum,float(np.linalg.norm(v)))
        return v
    for path in ('corrected','raw'):
        v=state.copy()
        for _ in range(cell['q']):
            for action in actions:v=checked(action(v,.8/cell['q']/2));det_count+=1
            for _ in range(cell['r']):
                power=v.copy();total=v.copy()
                for degree in range(1,cell['K']+2):
                    power=checked((-1j*tau/degree)*(tail@power));total=checked(total+power);tail_count+=1
                v=checked(total/b if path=='raw' else total)
            for action in reversed(actions):v=checked(action(v,.8/cell['q']/2));det_count+=1
            v=checked(np.exp(-1j*scalar*.8/cell['q'])*v)
        signals[path]=complex(np.vdot(state,v));vectors[path]=v
    return dict(signals=signals,vectors=vectors,b=b,log_B=cell['R']*math.log(b),
                intermediate_norm_max=maximum,counts=dict(tail=tail_count,deterministic=det_count))


def signal(config,grant,output):
    m,n,ham,sector,state,qstate,preps,compiler=native.load(config)
    from trottertracks.resource_applicability.h6_matched_contract_v1 import plan as original
    from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
    cells=c.plan()['A']['cells'];p=original()
    p.update(cells=cells,epsilons=c.plan()['A']['epsilons'],matrix_free=dict(p['matrix_free'],num_threads=1))
    p['numerical']['sensitivity_factors']=[1.]
    p['caps_proposed'].update(compile=0,trajectory=0,occurrence=0,control_probe=0)
    p['caps_proposed']['tail_matvecs_corrected_and_raw']['B2']=max(2*x['R']*(x['K']+1) for x in cells)
    writer=c.Writer(output)
    port=m.MatchedPort(Path(config['provider_root']),dict(kind='H6_TECHNICAL',plan=p),writer,Progress())
    port.ham,port.sector=ham,sector;port.qstate=qstate
    # Recover the exact bridge index convention through the unchanged loader API.
    mapping=config['mapping']
    _,_,full,loaded_state,_,_=m.load_h6_snapshot(Path(mapping['h6_input_snapshot.npz']),
        c.read(mapping['snapshot_receipt.json']),c.read(mapping['df_receipt.json']))
    port.qindices,_=m.checked_basis_bridge(ham,sector,full,loaded_state,1e-12)
    port.state=state.copy();port.saved_state=state.copy()
    port.spectra={};port.native_actions={};port.tails={};port.oracle_tails={};port.truncated_references={}
    whole=preps['faithful:p19']
    views={pfx:prefix_view(whole,ham,pfx,m) for pfx in (3,4)};views[5]=preps['faithful:p5']
    port.preps={cell['id']:views[cell['prefix']] for cell in cells}
    reference=c.read(mapping['A_saved_reference'])
    c.require(reference['metadata']['hamiltonian_hash']==config['binding']['hamiltonian']
              and reference['metadata']['state_hash']==config['binding']['state'],'SAVED_REFERENCE_BINDING')
    port.reference=complex(reference['reference']['real'],reference['reference']['imag'])
    port.reference_error=reference['occupation_matrix_vs_saved_DF_action_error']+reference['pair']['signal_difference']
    outer={}
    for row in c.read(mapping['A_saved_outer'])['rows']:
        outer[(row['prefix'],row['q'])]=dict(signal=row['signal'],kind='SAVED_PREFIX3_4_EXACT_TAIL_DIAGNOSTIC',
            error=row['independent_path_discrepancy'])
    origin=c.read(mapping['A_origin_signal'])['signals']
    for q in (4,8):
        row=next(r for r in origin if r['cell']['id']==f'B2_p5_q{q}_2nd_r4_K2')
        old_ref=complex(row['reference']['real'],row['reference']['imag'])
        reference_bridge_error=abs(old_ref-port.reference)
        c.require(reference_bridge_error<=1e-12,'ORIGIN_FULL_REFERENCE_ROUNDOFF_POLICY')
        outer[(5,q)]=dict(signal=row['signals']['exact_tail'],kind='SAVED_ORIGIN_EXACT_TAIL',
            error=max(row['signal_discrepancies']+row['state_discrepancies'])+reference_bridge_error,
            reference_bridge_error=reference_bridge_error,reference_bridge_policy='saved absolute roundoff allowance 1e-12')
    writer.write('saved_reference_and_outer_reuse.json',dict(reference=reference['reference'],
        saved_full_reference_recomputed=False,old_outer_diagnostics_recomputed=False,
        reused_outer=[dict(prefix=key[0],q=key[1],**value) for key,value in outer.items()]))
    # Warm the small primitive/spectral caches once; inherit by copy-on-write.
    for prep in views.values():port.actions(prep);port.actions(prep,oracle=True)
    # Tail matrices are cached per prefix and shared across all r/K at fixed p.
    zero=m.np.zeros_like(ham.one_body)
    for pfx,prep in views.items():
        port.oracle_tails[pfx]=m.np.asarray(m.occupation_df_matrix(-prep.extracted_identity_coefficient,
            zero,ham.lambdas[pfx:],ham.g_matrices[pfx:],sector.basis_indices),dtype=complex)/prep.exact_rte_lambda_r
    tasks=[dict(id='primitive_'+str(i),block=i) for i in range(6)]
    def primitive(task,directory):
        i=task['block'];prep=views[5]
        actual=port.actions(prep)[i];independent=port.actions(prep,oracle=True)[i]
        probes=[state,m.np.eye(1,len(state),0,dtype=complex)[0],m.np.eye(1,len(state),len(state)-1,dtype=complex)[0]]
        errors=[]
        for time in (.025,-.025):
            exact=m.expm(-1j*time*port.primitive_matrix('one' if i==0 else i-1))
            errors.extend(float(m.np.linalg.norm(action(v,time)-exact@v)) for v in probes for action in (actual,independent))
        maximum=max(errors);m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=maximum)
        return dict(block=i,times=[.025,-.025],probes_per_time=3,max_error=maximum,status='PASS')
    receipt=pool.map_owned(tasks,primitive,cpus=grant['worker_cpus'],limits=dict(max_workers=len(grant['worker_cpus'])),
        directory=output/'new_time_primitive_workers',deadline=math.inf,font_seed=config['font_seed'])
    writer.write('primitive_gate.json',dict(status='PASS',new_time=checked_results(receipt),
        old_times_reused_from_frozen_archive_validation=True,old_receipt_sha256=c.sha(mapping['A_saved_primitive'])))
    # Only q16 adds an outer-PF diagnostic; other p/q diagnostics are reused.
    prep=views[5];v=state.copy();actions=port.actions(prep,oracle=True)
    tail=m.occupation_df_matrix(0.,zero,ham.lambdas[5:],ham.g_matrices[5:],sector.basis_indices)
    step=m.expm(-1j*(.8/16)*tail)
    for _ in range(16):
        for action in actions:v=action(v,.8/16/2)
        v=step@v
        for action in reversed(actions):v=action(v,.8/16/2)
        v *= m.np.exp(-1j*whole.constant_coefficient*.8/16)
    z=complex(m.np.vdot(state,v));outer[(5,16)]=dict(signal=m.complex_record(z),kind='NEW_Q16_EXACT_TAIL',error=0.)
    writer.write('new_q16_outer.json',dict(prefix=5,q=16,signal=m.complex_record(z),certified=False))
    def callback(task,directory):
        local=copy(port);local.cells=[task['cell']];local.plan=dict(p,cells=local.cells)
        local.writer=c.Writer(directory);local.progress=Progress();local.completed=0
        local.calls=deepcopy(port.calls);local.tails={};local.truncated_references={}
        def oracle(cell):
            prep=local.preps[cell['id']]
            result=forward_finite(state,local.actions(prep,oracle=True),local.oracle_tails[cell['prefix']],cell,
                prep.constant_coefficient+prep.extracted_identity_coefficient,prep.exact_rte_lambda_r,m.np)
            result['signals']['exact_tail']=complex(
                outer[(cell['prefix'],cell['q'])]['signal']['real'],outer[(cell['prefix'],cell['q'])]['signal']['imag'])
            return result
        local.oracle_signal=oracle;local.correctness()
        record=c.read(Path(directory)/'signal_summary.json')['signals'][0]
        extra=outer[(task['cell']['prefix'],task['cell']['q'])]
        record['outer_diagnostic_binding']=extra
        record['allowance']['real']+=extra['error'];record['allowance']['imag']+=extra['error']
        record['precision_rows']=m.precision_rows(record,local.plan)
        noncancel=deepcopy(record)
        parts=record['error_decomposition']['signed_parts']
        noncancel['error_decomposition']['total_signed']={axis:math.fsum(abs(v[axis]) for v in parts.values())
                                                       for axis in ('real','imag')}
        record['empirical_noncancellation_precision_rows']=m.precision_rows(noncancel,local.plan)
        record['strict_bias_certificate']=None
        local.writer.write('boundary_signal.json',record)
        return record
    tasks=[dict(id=x['id'],cell=x) for x in cells]
    receipt=pool.map_owned(tasks,callback,cpus=grant['worker_cpus'],limits=dict(max_workers=len(grant['worker_cpus'])),
        directory=output/'signal_workers',deadline=math.inf,font_seed=config['font_seed'])
    rows=checked_results(receipt);c.require(len(rows)==16,'A_SIGNAL_COMPLETENESS')
    writer.write('worker_receipts.json',dict(primitive_workers=6,signal=receipt))
    return dict(stage='A',status='COMPLETE_MANDATORY_STOP',signals=rows,signal_count=16,
        new_native_compile=0,new_sampling=0,new_full_reference=0,new_outer_diagnostic=1,
        formal_accuracy_certified=False,global_optimum_claim=False,cost_acquisition_auto_started=False)
