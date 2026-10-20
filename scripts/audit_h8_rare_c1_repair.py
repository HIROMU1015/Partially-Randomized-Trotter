"""Post-execution byte/accounting/structure audit; no sampling or numeric reanalysis."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
BASE='d31ac1080e139eaad717301ffaa8d80160bbd378'
OUT=ROOT/'artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21'
PRIVATE=ROOT/'.c1_repair_private'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
         'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def paths():return sorted({x.decode() for x in (git('diff','--name-only','-z',BASE)+git('ls-files','--others','--exclude-standard','-z')).split(b'\0') if x})


def checks():
    source=read(OUT/'SOURCE_manifest.json')
    for r in source['files']:
        assert sha(ROOT/r['path'])==r['sha256']
        assert hashlib.sha256(git('show',source['SOURCE']+':'+r['path'])).hexdigest()==r['sha256']
    closure=read(PRIVATE/'closed_byte_rows.json')
    for r in closure:assert sha(r['path'])==r['sha256']
    baseline_manifest=json.loads(git('show',BASE+':artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/SHA256_manifest.json'))
    for r in baseline_manifest['files']:
        if r['path'].startswith('artifacts/'):
            assert sha(ROOT/r['path'])==r['sha256']
    result=read(OUT/'condition_results.json');successes=[]
    for c in result['conditions']:
        label=c['condition'];root=PRIVATE/label
        if root.exists():
            native=read(PRIVATE/(label+'_native_exit.json'));assert native['reaped']
            assert (root/'one_shot_consumed.json').exists()
            ledger=read(root/'budget/ledger.json')
            assert c['reserved']==ledger['used'] and ledger['nonrefundable']
            grant=read(PRIVATE/('grant_'+label+'.json'))
            marker=read(root/'one_shot_consumed.json')
            contract=OUT/('contract_'+label+'.json')
            approval=OUT/('execution_approval_'+label+'.json')
            assert marker['consumed'] and marker['nonrefundable']
            assert marker['grant_sha256']==sha(PRIVATE/('grant_'+label+'.json'))==native['grant_sha256']
            assert grant['contract_sha256']==sha(contract) and grant['approval_receipt_sha256']==sha(approval)
            assert grant['SOURCE']==source['SOURCE'] and grant['budget']==c['planned']==ledger['limits']
            reserved={}
            for reservation in (root/'budget').glob('*.reserved.json'):
                row=read(reservation);assert row['nonrefundable']
                reserved[row['kind']]=reserved.get(row['kind'],0)+row['count']
            assert reserved==ledger['used']
        if c['status']=='COMPLETE':
            completion=read(root/'completion.json')
            assert native['exit_code']==0 and completion['n']==43 and completion['compile']==86
            assert c['reserved']==c['planned'] and completion['calls']==c['planned']
            assert c['summary']==read(root/'pilot_summary.json')
            assert c['numeric_audit']==read(root/'independent_numeric_audit.json')
            assert c['summary']['groups']['S2_SINGLE']['n']==43
            assert c['summary']['groups']['S2_MULTI']['mean'] is None
            assert c['summary']['groups']['S4_PRESENT']['variance'] is None
            receipts=read(root/'batch_receipts.json')
            assert [len(r['workers']) for r in receipts]==[1,42]
            for receipt in receipts:
                assert receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['adopted_descendants']
                for h in receipt['workers']:
                    assert h['exit_code']==0 and h['reaped']
                    a=read(Path(h['directory'])/'axes/native_receipt.json')
                    assert a['status']=='PASS' and a['all_reaped'] and len(a['workers'])==2
                    assert all(w['exit_code']==0 and w['reaped'] for w in a['workers'])
            assert read(root/'law_binding_gate.json')['status']=='PASS'
            assert read(root/'postload_runtime_gate.json')['threads']==1
            assert read(root/'postload_fork_probe/native_receipt.json')['all_reaped']
            successes.append(label)
    if 'H8_0.0005' in successes:assert 'H8_0.05' in successes
    for row in read(PRIVATE/'private_result_inventory.json'):
        assert sha(row['path'])==row['sha256']
    tests=read(OUT/'synthetic_validation.json');assert tests['total']==67
    for log,n in [('driver_tests.txt',12),('c05_tests.txt',55)]:
        t=(OUT/log).read_text();assert 'Ran '+str(n)+' tests' in t and t.rstrip().endswith('OK')
    assert read(OUT/'runtime_repair.json')['toy_compiled_circuit_and_PCG64_exact']
    assert not git('ls-files','--','.c1_repair_private').strip()
    assert git('check-ignore','.c1_repair_private/closed_byte_rows.json').strip()
    old_modified={'scripts/h8_rare_c05/adapter.py','scripts/h8_rare_c05/provider.py','scripts/h8_rare_c1_run.py','tests/test_h8_rare_c1_run.py'}
    for raw in git('diff','--name-only','--diff-filter=MDRT','-z',BASE).split(b'\0'):
        if raw:
            rel=raw.decode();assert rel in INDEXES or rel in old_modified
            if rel in INDEXES:assert (ROOT/rel).read_bytes().endswith(git('show',BASE+':'+rel))
    allowed={'scripts/audit_h8_rare_c1_repair.py','docs/research/h8_rare_rte_c1_pilot_repair_20261021.md','docs/research/研究ノート/2026-10-21_h8_rare_rte_c1_pilot_repair.md'}
    links=0
    for rel in paths():
        assert rel in INDEXES or rel in old_modified or rel in allowed or rel.startswith('artifacts/resource_applicability/h8_rare_rte_c1_pilot_repair/2026-10-21/'),rel
        text=(ROOT/rel).read_text()
        if rel in INDEXES:text=text[:-len(git('show',BASE+':'+rel).decode())]
        assert not re.search(r'/(?:home|tmp|root|Users)/|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}',text),rel
        if rel.endswith('.json'):json.loads(text)
        if rel.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'):continue
                assert ((ROOT/rel).parent/link.split('#')[0]).exists(),rel+' '+link
                links+=1
    return dict(status='PASS_COMPLETION_PRESERVATION_ACCOUNTING_STRUCTURE',SOURCE=source['SOURCE'],baseline=BASE,
        audit_recorded_UTC=datetime.now(timezone.utc).isoformat(),grant_marker_ledger_reservation_hash_binding=True,
        completed_conditions=successes,closed_byte_rows=len(closure),old_artifact_manifest_preserved=True,
        synthetic_tests=67,all_accepted_workers_axes_reaped=True,private_result_files=len(read(PRIVATE/'private_result_inventory.json')),
        scoped_private_pattern_hits=0,links_checked=links,security_certificate=False,
        old_SOURCE_raw_grant_ledger_marker_STOP_changed=False,shared_environment_other_research_changed=False,
        reranking=False,production=False,push=False,detailed_science_review=False,final_freeze=False,mandatory_STOP=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--record',action='store_true');g.add_argument('--verify-committed',action='store_true')
    args=p.parse_args();status=checks()
    if args.record:
        save('completion_audit.json',status)
        save('SHA256_manifest.json',dict(baseline=BASE,self_excluded=True,files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in paths() if x!=str((OUT/'SHA256_manifest.json').relative_to(ROOT))]))
        print('PASS_C1_COMPLETION_AUDIT_MANDATORY_STOP')
    else:
        assert not git('status','--porcelain').strip()
        rows=read(OUT/'SHA256_manifest.json')['files']
        assert {x['path'] for x in rows}==set(paths())-{str((OUT/'SHA256_manifest.json').relative_to(ROOT))}
        for row in rows:assert sha(ROOT/row['path'])==row['sha256'] and git('show','HEAD:'+row['path'])==(ROOT/row['path']).read_bytes()
        print('PASS_COMMITTED_C1_SOURCE_RESULTS_AND_PRESERVATION_SHA')
        print(git('rev-parse','HEAD').decode().strip());print('MANDATORY_STOP_NO_PRODUCTION_NO_MORE_STRATA')


if __name__=='__main__':main()
