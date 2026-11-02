"""Summarize already saved correctness evidence. No science calls or bit generation."""
from pathlib import Path
import sys,json,hashlib,collections,subprocess,tarfile,os,shutil
sys.dont_write_bytecode=True
PACKET=Path(__file__).resolve().parent;ROOT=PACKET.parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g6_w2c_b3_final_live_v1 import guard as G,offline as O
from trottertracks.algorithm_codesign.g6_w2c_b3_live_preflight_v1 import spec as S
from trottertracks.algorithm_codesign import g6_w2c_b3_offline_result as V


def save(name,value):
    with (PACKET/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rowsum(rows,kind):
    out={}
    for pair in rows:
        for k,v in pair[kind].items():
            x=out.setdefault(k,dict(calls=0,cpu_ns=0,wall_ns=0))
            for n in x:x[n]+=v[n]
    return out

def main():
    c=O.read(PACKET/'candidate.json');m=O.read(PACKET/'approved_manifest.json');check=O.read(PACKET/'saved_evidence_check.json');out=Path(m['storage_destination']);root=out.parent
    report=O.read(out/'final.json');outer=O.read(out/'host_outer.json');simple=O.read(out/'simple_run_state.json')
    assert report['complete'] and check['full_documents']==512 and simple['execution_state']=='COMPLETED'
    coarse=rowsum(report['pairs'],'coarse');fine=rowsum(report['pairs'],'fine')
    calls=lambda suffix:sum(v['calls'] for k,v in fine.items() if k.endswith(':'+suffix))
    scientific_function_counts={k:v['calls'] for k,v in fine.items() if k.split(':',1)[0] in ('g7_generator.py','g7_provider.py','return_aggregation.py','g6_f3_contract.py','g6_f3_precision.py','g6_f3_generator.py') and '<locals>' not in k}
    bits=collections.Counter();per_arm=collections.defaultdict(collections.Counter);angle_count=0;worker_pids=set();core_sets=set();documents=0;by_mode=collections.Counter();evidence_bytes=0
    for task in m['tasks']:
        path=out/(task['key']+'.jsonl');evidence_bytes+=path.stat().st_size
        for line in path.read_bytes().splitlines():
            env=json.loads(line);p=V.decode(env['payload']);ev=p['evidence'];cons=ev['consumption'];draws=len(cons['draws']);documents+=1;by_mode[task['condition']]+=1
            assert ev['artificial_only'] is False and cons['artificial_only'] is False and ev['runtime_origin']=='PINNED_G7_F3'
            bits[draws]+=1;per_arm[task['arm']][draws]+=1;angle_count+=p['angle'] is not None
            worker_pids.add(ev['guard']['pid']);core_sets.add(tuple(ev['guard']['core_loaded']))
    assert documents==512 and sum(n*k for n,k in bits.items())==check['boundary_counts']['bit_draw_requests']==1288
    assert len(worker_pids)==16 and len(core_sets)==1
    core=next(iter(core_sets));assert {n.rsplit('.',1)[-1] for n in core}==set(G.CORE)
    driver_witness=coarse['correctness.event_invariants']['calls'];nested=check['boundary_counts']['nested_B_event_witness'];assert calls('event_witness')==driver_witness+nested
    canonical=calls('CanonicalGenerator.sample');full=calls('FullReturnGenerator.sample');session=calls('GeneratorSession.sample')
    assert canonical+full==512 and session==64 and canonical==384 and full==128
    assert calls('make_generator')==14 and calls('prepare')==2
    assert angle_count==check['boundary_counts']['B_angle']==250
    io=collections.Counter()
    for pair in report['pairs']:io.update(pair['io'])
    memory=dict(max_worker_VmHWM_bytes=max(p['memory']['VmHWM_bytes'] for p in report['pairs']),
        max_worker_final_VmRSS_bytes=max(p['memory']['VmRSS_bytes'] for p in report['pairs']),
        coordinator_peak_sampled_rss_bytes=report['peak_sampled_rss_bytes'],shared_page_sum_not_unique=report['rss_is_shared_page_sum_not_unique_memory'])
    statuses={label:O.read(PACKET/name) for label,name in [('begin','execution_invocation.json'),('end','resources_after.json')]}
    parent={}
    for label,rs in statuses.items():
        if label=='begin':rs=rs['begin']
        status=dict(line.split(':',1) for line in rs['/proc/self/status'].splitlines() if ':' in line)
        parent[label]={k:status.get(k,'').strip() for k in ('VmRSS','VmHWM')}
    resource=dict(host_outer=outer,worker_coarse_exclusive=coarse,worker_fine_exclusive=fine,
        worker_io_totals=dict(io),parent_memory_observations=parent,memory=memory,
        worker_io_window='worker before setup through normal final reporting; process bootstrap excluded',
        parent_io_observations='execution_invocation.json and resources_after.json /proc/self/io; includes wrapper and postprocessing, not the exact host_outer CPU window',
        stop_to_join_ns=report['stop_to_join_ns'],post_stop_parent_cpu_ns=report['post_stop_parent_cpu_ns'],
        post_stop_worker_cpu_estimate_ns=report['post_stop_worker_cpu_estimate_ns'],
        parent_stage_observations=report['parent_messages'],coarse_fine_added=False,
        observer_overhead='not isolated: included in measured CPU and residual; no separate instrumented pass',
        actual_power_loss_durability='NOT_ESTABLISHED')
    save('resource_summary.json',resource)
    log_names={'control.jsonl','operations.jsonl','trial_states.bin'}
    physical={kind:sum(p.stat().st_size for p in out.iterdir() if p.is_file() and ((p.name in log_names)==(kind=='log'))) for kind in ('result','log')}
    assert physical['result']<=1<<30 and physical['log']<=64<<20
    issues=[dict(path='output/final.json',field='artificial_only',observed=report['artificial_only'],meaning='inherited Coordinator report default is True although all 512 per-trial evidence and bits have artificial_only=False and real core imports. Frozen raw is preserved; this metadata defect is not repaired or rerun.'),
        dict(path='per-trial evidence / final report',field='live_API_compatibility and real_variable_consumption',observed='NOT_ESTABLISHED',meaning='preparation conservative labels remain in frozen source. Observed finite API/consumption checks are recorded separately; mathematical/general compatibility is not certified.')]
    save('observed_metadata_issues.json',dict(issues=issues,source_or_raw_modified=False,retry=0,next_action='USER/GPT decides future metadata-only correction/new scope; no current rerun'))
    save('result_summary.json',dict(classification=check['status'],technical_completed=512,by_mode=dict(by_mode),
        coverage_all_cells=True,inconclusive_coverage=False,failed=0,unknown=0,cells=check['cells'],
        actual_boundary_calls=check['boundary_counts'],actual_science_sampling=dict(CanonicalGenerator_sample=canonical,FullReturnGenerator_sample=full,
            GeneratorSession_sample=session,session_calls_nested_inside_FullReturnGenerator_sample=True,total_scheduled=512),
        driver_correctness_event_witness=driver_witness,nested_B_event_witness=nested,total_helper_event_witness=calls('event_witness'),
        scientific_function_counts=scientific_function_counts,draw_count_histogram=dict(bits),draw_histogram_by_arm={k:dict(v) for k,v in per_arm.items()},
        bit_draw_requests=1288,delivered_probability_bits=1288*160,discarded_hash_bits=1288*96,
        accepted_angles=angle_count,worker_pids=sorted(worker_pids),worker_core_loaded=list(core),parent_core_loaded=G.observation()['core_loaded'],
        receipt_caller_join=True,mandatory_STOP=True,execution_state=simple['execution_state'],claim_status=simple['claim'][3],claim_spent=simple['claim'][4],retry=0,resume=False,
        full_pair_evidence_bytes=evidence_bytes,physical_output_bytes=physical,control_store_outside_budget=True,
        failures=[],F01_real_failure_triggered=False,F01_failure_path_scope='retained unchanged + artificial failure cases only; no real failure occurred',
        observed_compatibility='ALL_ADOPTED_512_CALL_PATHS_COMPLETED_AND_SAVED_CHECKED',
        source_metadata_issues='observed_metadata_issues.json',scientific_correctness_proved=False,IID_proved=False,
        general_angle_theorem_independently_proved=False,performance_advantage_claimed=False,separate_reviewer=False))
    # Verify source remains exactly the source used for the run.
    for p,h in {**c['source_pins'],**c['input_pins']}.items():assert digest(ROOT/p)==h,p
    before=O.read(PACKET/'preservation_before.json');other=[]
    for row in before['other_worktrees_read_only']:
        p=ROOT.parent/row['worktree'];head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)
        other.append(dict(worktree=row['worktree'],HEAD=head,status=status,unchanged_observed=(head==row['HEAD'] and status==row['status']),policy='observation only; unrelated chat changes would not invalidate science'))
    save('preservation_after.json',dict(source_pins_unchanged=len(c['source_pins']),input_pins_unchanged=len(c['input_pins']),other_worktrees_read_only=other,old_source_raw_claim_STOP_modified=False,guard=G.observation()))
    inventory=[]
    for p in sorted(root.rglob('*')):
        if p.is_file():inventory.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
    archive=PACKET/'raw_scientific.tar.gz'
    with tarfile.open(archive,'x:gz') as tf:
        for row in inventory:tf.add(root/row['path'],arcname='run/'+row['path'],recursive=False)
    # Archive roundtrip byte hashes, without recreating or executing the run.
    with tarfile.open(archive,'r:gz') as tf:
        assert len(tf.getmembers())==len(inventory)
        for row in inventory:assert hashlib.sha256(tf.extractfile('run/'+row['path']).read()).hexdigest()==row['sha256']
    for row in inventory:assert digest(root/row['path'])==row['sha256']
    save('raw_inventory.json',dict(original_state_root=str(root),files=inventory,archive=dict(path=archive.name,bytes=archive.stat().st_size,sha256=digest(archive)),
        original_preserved=True,archive_roundtrip_SHA256=True,postprocessing_only=True))
    # Small standalone receipts are convenient for review; original bytes retained.
    dest=PACKET/'receipts';dest.mkdir(exist_ok=False)
    for name in ('receipt.json','delivery.json','caller.json','STOP_marker.json','host_outer.json','simple_run_state.json','pair_validation.json','S_receipt.json','P1_receipt.json'):
        shutil.copyfile(out/name,dest/name)
    assert not G.observation()['core_loaded'] and not G.ATTEMPTS
    print(json.dumps(dict(status=check['status'],sampling=documents,draws=1288,driver_witness=driver_witness,nested_witness=nested,archive_bytes=archive.stat().st_size,physical=physical,metadata_issues=len(issues)),indent=2))


if __name__=='__main__':main()
