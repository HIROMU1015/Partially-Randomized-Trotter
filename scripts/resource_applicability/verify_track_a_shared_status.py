"""Read-only verification of the provisional server research inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
from urllib.parse import quote, unquote
from urllib.request import urlopen

REPO=Path(__file__).resolve().parents[2]
BUNDLE=REPO/'artifacts/resource_applicability/track_a_cross_environment_sync/2026-10-14'


def load(path):return json.loads(path.read_text())


def ident(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def public_bytes(commit,path):
    url='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'+commit+'/'+quote(path,safe='/')
    with urlopen(url,timeout=15) as response:return response.read(8*2**20)


def verify(public_evidence=False):
    manifest=load(BUNDLE/'asset_manifest.json')
    for path,expected in manifest['files'].items():
        dest=(REPO/path).resolve();assert dest.is_relative_to(REPO)
        assert ident(dest.read_bytes())==expected,path
    evidence=load(BUNDLE/'fixed_evidence_index.json')
    assert len(evidence)==19
    for e in evidence:
        raw=public_bytes(e['commit'],e['path']) if public_evidence else subprocess.check_output(['git','-C',str(REPO),'show',e['commit']+':'+e['path']])
        assert ident(raw)=={k:e[k] for k in ('bytes','sha256')},e['label']
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['git_blob_SHA'],e['label']
        assert e['GitHub_commit_reachability_confirmed'] is True
    worktrees=load(BUNDLE/'worktree_inventory.json')['worktrees']
    assert len(worktrees)==67 and sum(r['dirty'] for r in worktrees)==12
    assert sum(r['tracked_changes'] for r in worktrees)==969
    assert sum(r['untracked_files'] for r in worktrees)==19981
    assert len({r['worktree'] for r in worktrees})==67
    assert sum(r['commit_reachability']=='REACHABLE_FROM_FRESH_REMOTE_HEAD' for r in worktrees)==66
    external=load(BUNDLE/'external_review_inventory.json')
    assert len(external)==4 and all(r['publication']=='GITHUB_BYTE_IDENTICAL_COPY_CONFIRMED' for r in external)
    h8=load(BUNDLE/'h8_readonly_snapshot.json')
    assert h8['accepted_signals']==92 and h8['accepted_signal_groups']==4
    assert h8['total_COMPLETE_exploration']==h8['inherited_COMPLETE_exploration']+h8['new_COMPLETE_exploration']==203
    assert h8['remaining_exploration']+203==h8['planned_exploration']==592
    assert h8['active_slots']==8 and h8['coordinator_identity_match'] is True
    assert h8['source_closure_current_byte_match'] is True and h8['source_closure_hashes_count']==265
    assert h8['final_result']=='NOT_OBSERVED_NOT_CERTIFIED'
    assert h8['runtime_or_grant_writes']==h8['H8_operations']==h8['trajectory_reads']==0
    assert h8['latest_completed_group_ledger']['budget_reset'] is False
    assert h8['latest_completed_group_ledger']['failed_calls_refunded'] is False
    checked=0
    for path in manifest['authored_markdown']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',(REPO/path).read_text()):
            if target.startswith(('http://','https://','#','mailto:')):continue
            target=unquote(target.split('#')[0].strip('<>'))
            dest=((REPO/path).parent/target).resolve()
            assert dest.is_relative_to(REPO) and dest.exists(),(path,target)
            checked+=1
    # Only new inventory text is checked; inherited historical evidence is not rewritten.
    for path in manifest['new_public_snapshot_files']:
        text=(REPO/path).read_text()
        assert '/home/' not in text and '/tmp/' not in text,path
        assert not re.search(r'\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+',text),path
        assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',text),path
    return dict(status='PASS_PROVISIONAL_SERVER_INDEX_BYTES_LINKS',fixed_evidence_files=19,
        evidence_read_method='pinned GitHub public bytes' if public_evidence else 'fixed local Git blobs',
        worktrees=67,existing_dirty_worktrees=12,remote_reachable_worktree_HEADs=66,
        external_review_public_copies=4,H8_signal_candidates=92,H8_exploration_complete_at_snapshot=203,
        H8_final_result_certified=False,local_environment_reconciliation='NOT_PERFORMED',
        authored_internal_links=checked,sanitized_snapshot_files=len(manifest['new_public_snapshot_files']),
        new_science_compile_sampling=0,H8_runtime_writes=0,shared_venv_git_configuration_changes=0)


def verify_remote_assets():
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','--heads','origin','refs/heads/'+branch],text=True).strip()
    assert remote and remote.split()[0]==head,'remote HEAD mismatch or unpublished branch'
    files=dict(load(BUNDLE/'asset_manifest.json')['files'])
    for name in ('asset_manifest.json','verification.json'):
        path=str((BUNDLE/name).relative_to(REPO));files[path]=ident((REPO/path).read_bytes())
    for path,expected in files.items():assert ident(public_bytes(head,path))==expected,path
    return dict(status='PASS_REMOTE_HEAD_DOCS_ASSET_BYTES',branch=branch,remote_SHA=head,files=len(files))


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(45,45));signal.alarm(180)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-evidence',action='store_true',help='Read fixed primary evidence bytes from GitHub instead of Git objects')
    parser.add_argument('--remote',action='store_true',help='After push, verify this new branch and all published assets')
    args=parser.parse_args();result=verify(args.public_evidence)
    if args.remote:result['remote']=verify_remote_assets()
    print(json.dumps(result))
