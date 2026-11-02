"""Seal H2 saved input + common runtime + all four explicitly approved stages."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import h2_outer32_contract as c

def prepare(source,template_path,h2_path,exclude_paths,other_controller_finished,private):
    private=Path(private).resolve();private.mkdir(parents=True,exist_ok=False)
    template=c.read(template_path);old=c.read(h2_path)
    c.require(sys.executable==template['runtime']['python'],'H2_USE_CURRENT_COMMON_RUNTIME')
    c.require(old['binding']['rank']==3 and old['binding']['rank_policy']=='tol_only_1e-8','H2_FIXED_SAVED_DF3')
    # Only actually used provider/runtime bytes; no unrelated historical runs,
    # active worktree statuses or scientific outputs are frozen by this run.
    files={r['path']:r for r in template['sealed_files'] if 'runtime' in r['role'] or 'package_identity' in r['role'] or 'distribution_identity' in r['role']}
    for key,r in old['inputs'].items():
        c.require(c.sha(r['path'])==r['sha256'],'EXISTING_H2_INPUT_SHA:'+key);files[r['path']]=r
    # Retain the existing H2 implementation correction, byte-identically.
    guards=[r for r in old['sealed_files'] if r['path'].endswith('/scripts/h2_m5_compiler_phase_guard.py')]
    c.require(len(guards)==1 and c.sha(c.ROOT/'scripts/h2_m5_compiler_phase_guard.py')==guards[0]['sha256'],'ORIGINAL_H2_PHASE_GUARD_BYTES')
    # The saved F/FP input identity is independent of a later m5 run STOP.
    # Reuse no trajectory or completion flag from that stopped run.
    prior_status=c.read(Path(old['run_root'])/'completion_witness.json')['status']
    import concurrent.futures.process,concurrent.futures.thread,queue,_queue,threading
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).resolve().is_relative_to('/usr/lib/python3.12'):
            item=c.seal(Path(p).resolve(),'H2_stdlib_runtime');files[item['path']]=item
    for rel,blob in c.source_blobs(source).items():
        p=c.ROOT/rel;c.require(p.read_bytes()==blob,'COMMITTED_H2_SOURCE:'+rel)
        item=c.seal(p,'H2_SOURCE:'+rel);files[item['path']]=item
    excluded=set();sources=[]
    for path in [old['excluded_path'],old['registry_path'],*exclude_paths]:
        p=Path(path).resolve();v=c.read(p)
        values=v['values'] if 'values' in v else [x for row in v.values() for x in (row.values() if isinstance(row,dict) else row)]
        c.require(all(type(n) is int for n in values),'H2_INTEGER_SEED_EXCLUSIONS');excluded.update(values)
        sources.append(dict(sha256=c.sha(p),bytes=p.stat().st_size))
    ep=private/'initial_exclusions_PRIVATE.json';c.save(ep,dict(values=sorted(excluded),sources=sources,finite_known_registry_coverage=True))
    item=c.seal(ep,'H2_known_seed_exclusions');files[item['path']]=item
    approval=private/'approval_APPROVED_PRIVATE.json'
    c.save(approval,dict(SOURCE=source,authorized_stages=list(c.STAGES),no_intermediate_reapproval=True,
        user_instruction='私の承認は飛ばしてよいので現在H4とH6がやっている検証地点と同じところまで進めてほしい。',
        pilot_n_each=33,design_rule=dict(relative_MCSE_target=.02,variance_multiplier=4,allocation='GROUP',minimum_n=33,
            zero_observed_variance='FRESH33_MINIMUM_MCSE_DESIGN_UNRESOLVED'),
        endpoint='EXPLORATION_ACQUISITION_AND_TECHNICAL_END_AUDIT',
        post_exploration_statistics=False,shot_accounting=False,ranking=False,independent_confirmation=False,
        new_SCF_DF_state=False,AD03='DEFERRED',retry=False))
    item=c.seal(approval,'H2_explicit_four_stage_approval');files[item['path']]=item
    # Reuse the published H4 pair/RSS proxies conservatively for H2 admission;
    # these are planning values, never bounds or new runtime caps.
    cap=c.read(c.ROOT/'artifacts/resource_applicability/track_a_h46_pr_exploration_readiness/2026-10-20/capacity_proposal.json')
    resources=dict(worker_RSS_proxy_bytes=cap['worker_RSS_proxy_bytes']['H4'],raw_pair_proxy_bytes=cap['raw_pair_proxy_bytes']['H4'],
        per_trajectory_growth_proxy_bytes=4*cap['compressed_pair_proxy_bytes']['H4']+16*1024*1024,
        other_fixed_growth_reserve_bytes=cap['concurrent_peak_growth_proxy_bytes'],other_controller_finished=str(Path(other_controller_finished).resolve()),
        process_start_method='fork',runtime_caps=False,host_PSI_kill=False,inode_proxy=1000,disk_proxy_bytes=0,
        prewarm_readonly_contexts=True,worker_input_cache=True,proxy_is_upper_bound=False,existing_files_deleted=0,
        scientific_worker_policy='FRESH_IDLE_CPU_RAM_AND_SCRATCH_NO_HISTORICAL_CAP')
    parent=dict(prior_H2_m5_stage_status=prior_status,prior_H2_trajectories_reused=0,SOURCE=source,authorized=True,authorized_stages=list(c.STAGES),runtime=template['runtime'],provider_root=template['provider_root'],
        inputs=old['inputs'],binding=old['binding'],resources=resources,storage=c.storage.storage_policy(),
        compiler_phase_policy='H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1',approval_path=str(approval),
        private=str(private),initial_exclusions=str(ep),sealed_files=list(files.values()),run_root=str(private/'pipeline_run01'))
    plan=copy.deepcopy(parent);pp=private/'plan_PIPELINE_APPROVED_PRIVATE.json';c.save(pp,plan)
    parent.update(plan=plan,plan_path=str(pp),plan_sha256=c.sha(pp))
    from h2_outer32_pipeline import require_parent
    require_parent(parent)
    p=private/'config_PIPELINE_APPROVED_PRIVATE.json';c.save(p,parent)
    c.save(private/'preflight_PRIVATE.json',dict(status='STATIC_PASS_FRESH_GATES_STILL_REQUIRED',SOURCE=source,
        config_sha256=c.sha(p),sealed_files=len(files),excluded_seed_count=len(excluded),scientific_calls_started=0))
    return dict(config=str(p),SOURCE=source,sealed_files=len(files),excluded_seed_count=len(excluded),pilot_trajectories=396,pilot_compile_axes=792,
        main_counts='H2_PILOT_DEPENDENT_2PCT_F4_GROUP_MIN33',new_science_started=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--template',required=True);p.add_argument('--h2-existing',required=True)
    p.add_argument('--exclude',action='append',default=[]);p.add_argument('--other-controller-finished',required=True);p.add_argument('--private',required=True);a=p.parse_args()
    print(__import__('json').dumps(prepare(a.source,a.template,a.h2_existing,a.exclude,a.other_controller_finished,a.private)))

if __name__=='__main__':main()
