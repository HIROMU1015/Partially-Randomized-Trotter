"""Future H6 N batch adapter. Import performs no saved-input/scientific IO.

An explicitly invoked batch uses the existing D2 action/geometry, sharing the
context, references and half-step cache across only the supplied candidate menu.
Old D1/D2/m6/m7 execution entry points and fixed guards are never modified.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

from . import pf_pr_rpe_pipeline as pipeline


def load_saved_context(root, reuse_plan):
    """Only called inside a future permitted scientific batch, never planning."""
    import run_h6_df19_rpe_d2 as io
    from trottertracks import h6_df19_rpe_d2 as backend
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
    tail=reuse_plan['saved_tail_context']
    index=json.loads(io.blob(tail['source_index_commit'],tail['source_index_path']))
    paths=index['saved_inputs']['H6']['inputs']
    context_path=Path(index['T20_reference_files']['H6']['path'])
    if any(io.sha(v['path'])!=v['sha256'] for v in paths.values()) or io.sha(context_path)!=tail['sha256']:
        raise ValueError('Saved H6 input/context changed')
    context=json.loads(context_path.read_bytes())
    if context['binding']!=tail['binding'] or context['prefixes']['7']!=tail['prefixes']['7']:
        raise ValueError('H6 tail binding changed')
    ham,sector,full,state,metadata,_=load_h6_snapshot(Path(paths['H6_snapshot.npz']['path']),
        json.loads(Path(paths['H6_snapshot_receipt.json']['path']).read_bytes()),
        json.loads(Path(paths['H6_df_receipt.json']['path']).read_bytes()))
    if metadata['hamiltonian_hash']!=backend.BINDING['hamiltonian'] or metadata['state_hash']!=backend.BINDING['state']:
        raise ValueError('H6 H/state mismatch')
    return backend.make_context(ham,sector,state,tail['prefixes']),metadata


class H6ActionBatch:
    def __init__(self, context_factory, reuse_plan, source):
        self.context_factory=context_factory
        self.reuse_plan=deepcopy(reuse_plan)
        self.source=deepcopy(source)
        self.last_run=None

    def __call__(self, task, rows):
        import run_h6_df19_rpe_d2 as io
        from trottertracks import h6_df19_rpe_d2 as backend
        pipeline.validate_task(task)
        if (not pipeline.same(task['configuration'],self.reuse_plan['configuration_reuse'])
                or any(task['binding'].get(k)!=v for k,v in backend.BINDING.items())
                or not pipeline.same(task['binding'],self.reuse_plan['saved_tail_context']['binding'])):
            raise ValueError('Fixed saved H6 input/policy required')
        if len({pipeline.key(r) for r in rows})!=len(rows) or not rows:
            raise ValueError('Nonempty unique batch menu required')
        for row in rows:
            if row not in task['candidates'] or row['candidate']['method'] not in ('FULL','PR'):
                raise ValueError('Batch outside declared task')
            if not pipeline.same(row['structure'],pipeline.structure(row['candidate'])):
                raise ValueError('Batch signed recipe differs from saved SOURCE')
            if row['candidate']['method']=='PR' and str(row['candidate']['prefix']) not in self.reuse_plan['saved_tail_context']['prefixes']:
                raise ValueError('Saved tail unavailable; do not regenerate')
        self.last_run=dict(status='RUNNING',scientific_source=self.source,reference_times=[],references={},automatic_retry=False)
        ctx,metadata=self.context_factory(task)
        times=sorted({r['candidate']['T'] for r in rows});references=self.last_run['references']
        for T in times:
            self.last_run['reference_times'].append(T)
            try:references[T]=backend.reference_pair(ctx,T)
            except Exception as exc:
                self.last_run.update(status='STOP_REFERENCE_GATE',reason=str(exc));raise
        keys=backend.half_step_keys(rows);resources=io.worker_resources(ctx,len(keys))
        with ThreadPoolExecutor(max_workers=resources['half_step_workers']) as pool:
            for k,value in pool.map(lambda k:backend.half_step(ctx,k),keys):ctx.half_steps[k]=value
        plan=dict(self.reuse_plan,configuration_reuse=task['configuration'],rpe_task_id=task['task_id'])
        def job(row):
            ref=references[row['candidate']['T']]
            try:
                record=backend.evolve(ctx,row['candidate'],ref['reference'])
                record['origin']='EXPLICIT_FIXED_MENU_BATCH'
                saved=backend.analyze(record,row,plan,state_energy=metadata['solver_energy'],
                    state_residual=metadata['solver_residual'],reference_evidence=ref)
                saved['price_coverage']='NOT_EVALUATED_SEPARATE_PRICE_TASK'
                return pipeline.shot_record(task,row,saved,self.source)
            except Exception as exc:
                failed=dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],candidate=deepcopy(row['candidate']),
                    binding=deepcopy(task['binding']),target_id=task['task_id'],base_time=task['configuration']['base_time'],
                    alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'],
                    status='NUMERICAL_GATE_FAILURE',N_X=None,N_Y=None)
                return dict(identity=pipeline.identity(task,row),plan=failed,
                    actual=dict(status='UNRESOLVED_REFERENCE_OR_ERROR',N_X=None,N_Y=None),
                    diagnostics=dict(reason=str(exc)),source=self.source,origin='FAILED_FIXED_MENU_BATCH')
        with ThreadPoolExecutor(max_workers=min(len(rows),resources['half_step_workers'])) as pool:
            result=list(pool.map(job,rows))
        self.last_run=dict(status='STOP_UNQUALIFIED_N' if any(r['plan']['status']!='CONDITIONAL_ROUND_PASS' for r in result) else 'BATCH_EVALUATED',reference_times=times,reference_state_actions=2*len(times),
            finite_mean_candidates=len(rows),internal_PR_exact_tail_controls=sum(pipeline.arm(r['candidate'])=='PR' for r in rows),
            resources=resources,references=references,scientific_source=self.source,
            automatic_retry=False)
        return result
