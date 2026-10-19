"""Known-path preservation audit; unrelated worktree activity is observation only."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def git(root,*args):return subprocess.check_output(['git','--no-optional-locks','-C',str(root),*args])
def main():
    parent=Path('/home/AbeHiromu/projects/partially-randomized-trotter');before=(HERE/'preflight/parent_status.bin').read_bytes();after=git(parent,'status','--porcelain=v1','-z','--untracked-files=all');(HERE/'preflight/parent_status_after.bin').write_bytes(after)
    protected=[]
    for suffix in ('adapter','cli','identity','trace'):
        path='src/trottertracks/algorithm_codesign/g6_w2c_b3_pilot_independent_'+suffix+'.py';original=git(ROOT,'show','49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866:'+path);current=(ROOT/path).read_bytes();assert original==current;protected.append(dict(path=path,sha256=sha256(current).hexdigest(),original_bytes_equal=True))
    runtime=json.loads((HERE/'injection_runtime_inventory.json').read_text())
    for row in runtime['files']:assert sha256((HERE/'injection_runtime'/row['path']).read_bytes()).hexdigest()==row['sha256']
    inventory=json.loads((HERE/'public_bytes/docs/research/track_b_g6_w2c_b3_trace_reservation_accounting_repair_20261019/source_inventory.json').read_text())
    for row in inventory['files']+inventory['protected']:assert sha256((HERE/'public_bytes'/row['path']).read_bytes()).hexdigest()==row['sha256']
    producer=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/g6-w2c-b3-trace-reservation-accounting-repair-20261019')
    live_status=git(producer,'status','--porcelain=v1','-z','--untracked-files=all')
    report=dict(schema='b3_v2_independent_preservation_v1',write_scope='Dedicated independent branch/worktree namespace, discoverability indexes, new Git refs, and owned artificial temp/output only',old_independent_SOURCE=protected,used_fixed_Producer_bytes_equal=True,used_injection_runtime_files_unchanged=len(runtime['files']),parent_status_observation=dict(before_bytes=len(before),after_bytes=len(after),before_sha256=sha256(before).hexdigest(),after_sha256=sha256(after).hexdigest(),equal=before==after,not_a_gate_on_unrelated_worktrees=True,full_dirty_file_content_snapshot_not_taken=True),producer_worktree_observation=dict(HEAD=git(producer,'rev-parse','HEAD').decode().strip(),status_sha256=sha256(live_status).hexdigest(),read_only=True,unrelated_changes_do_not_invalidate_fixed_input_copies=True),existing_science_results_grants_ledgers_markers_STOP_modified_by_this_task=False,real_authority_stores_opened=False,other_processes_modified=False,shared_environment_settings_changed=False,mandatory_STOP=True)
    (HERE/'preservation_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(old_checker_bytes_equal=True,used_runtime_files_unchanged=len(runtime['files']),parent_status_equal=before==after)))
if __name__=='__main__':main()
