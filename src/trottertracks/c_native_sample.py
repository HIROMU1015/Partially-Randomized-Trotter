"""One native price sample, shared by thread and process schedulers."""
from fractions import Fraction
from . import c_pr_aware_df_stage_b as b
from . import c_pr_aware_df_stage_b2 as b2
from . import c_pr_aware_df_stage_b_runner as loader
from . import c1_stage1_execution as io


def run_sample(i,c,out,call,a,policy,ctx,psi,builder,shells,shared=None):
    from qiskit import qpy
    from .c1_h4_native_fair import compile_checked
    from .pr_dictionary_validation.fixture import CompilerProfile
    from .pr_dictionary_validation.f2_execution import circuit_metrics
    arm=a['arm'];h=Fraction(1,10);steps=policy['events_per_sample']
    profile=CompilerProfile();profile.validate()
    if shared is None:shared=call
    def qsave(label,kind,q):
        with (out/'circuits'/(label+'_'+kind+'.qpy')).open('xb') as f:qpy.dump(q,f)
    label=f'sample{i:02d}'
    def sample():
        master=(policy['master'],policy['cohort'],c['source_manifest_sha256'])
        events=b.independent_occurrence(arm,a['tail'],a['dist'],a['config'],master=master,
            round_index=-1,axis=policy['axis'],shot=i,occurrence=0)
        if len(events)!=steps or any(x.taylor_order!=0 for x in events):raise ValueError('NATIVE_K0_EVENT_COUNT')
        return events
    events=call('trajectory_sample',label,sample)
    io.save(out/'samples'/(label+'_events.json'),dict(sample=i,sampling=policy,
        SOURCE=c['source_manifest_sha256'],JU=arm.model.source_identity,events=[x.to_dict() for x in events],
        coherent_occurrences_independent=True,quantum_shots=0,old_cohort_pooled=False))
    raw=shared('kernel_build',label,lambda:b.build_kernel(arm,h,events,a['config'],builder=builder));qsave(label,'raw',raw)
    ref=call('trajectory_reference',label,lambda:b2.trajectory_reference(arm,ctx,h,events))
    loader.save_arrays(out/'references'/(label+'.npz'),system_reference=ref)
    R,target,report=call('raw_operator_gate',label,lambda:b2.raw_gate(raw,ref));io.save(out/'samples'/(label+'_raw.json'),report)
    compiled=shared('compile',label,lambda:compile_checked(raw,profile));qsave(label,'compiled',compiled)
    C,report=call('compiled_operator_gate',label,lambda:b2.compiled_gate(compiled,R,target));io.save(out/'samples'/(label+'_compiled.json'),report)
    adjoints,report=call('adjoint_pair_gate',label,lambda:b2.adjoint_pair_gate(raw,compiled,R,C,target))
    for kind,q in adjoints.items():qsave(label,kind+'_adjoint',q)
    io.save(out/'samples'/(label+'_adjoint.json'),report)
    io.save(out/'samples'/(label+'_XY.json'),call('XY_gate',label,lambda:b2.XY_gate(R,C,ref,psi,shells)))
    metrics=call('kernel_metrics',label,lambda:circuit_metrics(compiled));io.save(out/'samples'/(label+'_metrics.json'),metrics)
    return i,metrics
