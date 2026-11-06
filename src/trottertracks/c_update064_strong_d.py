"""Same-update064 strong S2 adapter; native body reused from frozen same-J/U D.

The old orchestration loads update005 ground and requires old PR evidence. This
adapter instead accepts the freshly qualified shared Stage-A candidate/eigenpair.
The kernel, P0, full512/actual-adjoint/XY, propagation and cost engines are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from . import c_same_ju_strong_d as reused

io, m2, accuracy, integrated = reused.io, reused.m2, reused.accuracy, reused.integrated


def evaluate(root, c, out, call, a, *, prepared_arm=None, full_generators=None):
    import numpy as np
    from qiskit import qpy
    from threadpoolctl import threadpool_limits
    from . import c_pr_aware_df_stage_b as b, c_pr_aware_df_stage_b_runner as loader, c_pr_aware_df_stage_b2 as b2
    from . import c1_h4_native_fair as fair, c1_h4_full_synthesis as synthesis, c1_molecule_extension as molecule
    from .c1_h4_strong_baseline import StrongBuilder
    from .c1_h6_ca_resources import serial_round_cost
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    root, out = Path(root), Path(out)
    for leaf in ('circuits', 'operators'): (out/leaf).mkdir()
    p, d, sector = a['p'], a['d'], a['sector']
    identity = a['arm'].model.source_identity
    candidate_label=a.get('candidate_label','update064')
    # Same in-memory J/U, input and freshly solved ground as PR. No old candidate
    # ground/PF/error/price path is accepted by this interface.
    if (prepared_arm is None) != (full_generators is None):
        raise ValueError('PREPARED_D_MODEL_AND_CAR_REQUIRED_TOGETHER')
    arm = call('native_model', 'same '+candidate_label+' full-frame D', lambda: prepared_arm if prepared_arm is not None else b.prepare_arm(**p, **d,
        role='proposed', source_identity=identity, input_identity=a['arm'].model.input_identity, ld=8))
    if prepared_arm is None:
        car = call('car_context', 'independent full256 CAR', lambda: b.stage1.car_bilinears(8))
        G = [call('raw_factor', str(i), lambda i=i: b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']), p['constant'], car)
            if i == 0 else b.stage1.raw_frame(d['J'][i-1], d['U'][i-1], car)[0]) for i in range(9)]
    else:
        if (arm.model != a['arm'].model or arm.plan.ld != 8
                or not np.array_equal(arm.J,d['J']) or not np.array_equal(arm.U,d['U'])):
            raise ValueError('SAME_PREPARED_D_REPRESENTATION_REQUIRED')
        G = full_generators
    ix, checks = call('sector_gate', 'all9 same-candidate full/sector factors', lambda: m2.sector_match(G, sector))
    io.save(out/'sector.json', checks)
    ground = dict(energy=a['pair'][0][0], sector_state=a['ground'])
    io.save(out/'identity.json', dict(candidate=identity, ground_origin='../A/ground_candidate.npz',
        new_DF_error=0, new_ground_eigensolve=0, same_JU_as_PR=True))
    sk = {}
    def pf_call(kind, label, fn):
        value = call('PF_kernel' if kind == 'sector_kernel_reference' else 'PF_round', label, fn)
        if kind == 'sector_kernel_reference': sk['pf'] = value[0]['pf']
        return value
    pf = molecule.PF_rounds(sector['generators'], 1, pf_call, candidate_label, eigenpair=a['pair'])
    io.save(out/'PF.json', pf)
    df = a['saved']['physical']
    saved = dict(shell_reports=a['saved']['shells'])
    packet = dict(id='D_'+candidate_label+':'+identity, arm=candidate_label, s=1, aliases=['D_'+candidate_label],
        df_error_with_guard=df['sector_error']+df['numerical_guard'],
        state_model='own_DF_ground_state_oracle', alias_verified=True)
    ideal = reused.ideal_plans(pf, packet, call)
    numerical=None;metrics=None;costs=[]
    if ideal['energy'].get('same_accuracy'):
        ref=call('full_reference','independent full-frame CAR S2',lambda:b.reference.reference_kernels(G,Fraction(1,10),ld=8)['pf'])
        io.save(out/'full_sector_kernel.json',call('reference_sector_gate','full/sector S2',lambda:fair.component_check(
            ref[np.ix_(ix,ix)],sk['pf'],1e-10,label='full vs sector S2')))
        shells={};shell_reports={};shell_metrics={}
        for label in ('head','X','Y'):
            qs={};matrices={}
            for kind in ('raw','compiled'):
                def load(label=label,kind=kind):
                    with (root/c['shells'][label][kind]).open('rb') as f:items=qpy.load(f)
                    if len(items)!=1 or items[0].num_qubits!=9:raise ValueError('SHELL_SCHEMA')
                    return items[0]
                qs[kind]=call('shell_load',label+kind,load)
                matrices[kind]=call('shell_decode',label+kind,lambda kind=kind:fair.decode(qs[kind]))
            call('shell_structure',label,lambda label=label:b2.shell_assessment(qs['raw'],qs['compiled'],label))
            H=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
            target=np.asarray(np.kron(H if label!='Y' else H@np.diag([1.,-1j]),np.eye(256)),complex)
            report=call('shell_segment',label,lambda label=label:synthesis.segment(matrices['raw'],matrices['compiled'],target,'shell:'+label))
            report['metrics']=saved['shell_reports'][label]['metrics'];shells[label]=matrices;shell_reports[label]=report
            shell_metrics[label]={k:report['metrics'][k] for k in accuracy.METRICS}
        io.save(out/'shells.json',shell_reports)
        builder=StrongBuilder(relative_fusion=True,call=call)
        for block in arm.model.blocks:call('gaussian_build',block.block_id,lambda block=block:fair.basis_request(builder,arm.model,block))
        raw=call('kernel_build','one deterministic h=.1 kernel',lambda:builder.ordered(arm.model,Fraction(1,10),optimized=True))
        io.save(out/'fusion.json',call('fusion_gate','M2 order/direction/shared rules',lambda:m2.fusion_gate(builder,arm.model)))
        profile=CompilerProfile();profile.validate()
        compiled=call('compile','fixed P0 once',lambda:fair.compile_checked(raw,profile));qs=dict(raw=raw,compiled=compiled)
        for kind,q in list(qs.items()):qs[kind+'_adjoint']=call('adjoint_build',kind,lambda q=q:b.native.actual_adjoint(q))
        for kind,q in qs.items():
            with (out/'circuits'/(kind+'.qpy')).open('xb') as f:qpy.dump(q,f)
        plan=integrated.resource_plan(task_count=4,full_dimension=256)
        plan['policy']='four independent raw/compiled/actual-adjoint full512 decodes;build/compile/inverse creation sequential'
        io.save(out/'resource_plan.json',plan)
        def decode(kind):return kind,call('adjoint_decode' if kind.endswith('adjoint') else 'operator_decode',kind,lambda:fair.decode(qs[kind]))
        with threadpool_limits(limits=plan['BLAS_threads_per_worker']):
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:mats=dict(pool.map(decode,qs))
        for kind,matrix in mats.items():loader.save_arrays(out/'operators'/(kind+'.npz'),matrix=matrix)
        target=fair.controlled(ref);segment=call('segment_assessment','full512 raw/compiled/reference',lambda:synthesis.segment(mats['raw'],mats['compiled'],target,'selected'))
        io.save(out/'segment.json',segment)
        adj={}
        for kind in ('raw','compiled'):
            def gate(kind=kind):
                x=mats[kind+'_adjoint'];return dict(dagger=fair.component_check(x,mats[kind].conj().T,1e-10,label='actual dagger'),
                    reference=fair.component_check(x,target.conj().T,1e-10,label='adjoint/reference'),
                    recovery=fair.component_check(x@mats[kind],np.eye(512),1e-10,label='adjoint return'))
            adj[kind]=call('adjoint_gate',kind,gate)
        io.save(out/'adjoint.json',adj);psi=np.zeros(256,complex);psi[ix]=ground['sector_state']
        io.save(out/'XY.json',call('XY_gate','raw/compiled complex X/Y',lambda:b2.XY_gate(mats['raw'],mats['compiled'],ref,psi,shells)))
        metrics=call('metrics','one P0 kernel',lambda:circuit_metrics(compiled));io.save(out/'metrics.json',metrics)
        packet.update(kernel_metrics={k:metrics[k] for k in accuracy.METRICS},shell_metrics=shell_metrics)
        numerical=m2.evaluate_accuracy(packet,pf,shell_reports,segment,out,call)
        costs=numerical['per_shot_costs']
        if not costs and ideal['energy'].get('same_accuracy'):
            for row in ideal['rounds']:costs.extend(call('round_serial_cost',str(row['m']),lambda row=row:serial_round_cost(packet,row['m'])))
    absent=dict(status='NO_TOTAL_UNQUALIFIED_ACCURACY',totals=None,cost_scope='serial_native_kernel_composition_without_state_preparation')
    ideal['cost']=call('ideal_aggregate','matched counterfactual totals',lambda:accuracy.aggregate_totals(
        ideal['energy'],ideal['rounds'],costs,cost_scope=absent['cost_scope']))
    numcost=absent if numerical is None else numerical['cost']
    result = dict(status='SAME_'+candidate_label.upper()+'_STRONG_D_COMPLETE' if numerical is not None else 'STOP_D_PF_UNQUALIFIED',
        candidate=identity, native_admitted=numerical is not None, saved_DF=df,
        PF_rounds=pf, numerical=numerical, ideal_exact_native=ideal, metrics=metrics,
        actual_CA_achieved=None, strict_DF_enclosure=None, guaranteed_resource_superiority=None)
    io.save(out/'result.json', result)
    return result
