"""Post-STOP preservation/structure audit only. Cannot launch or retry science."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='77bf6472f901feab8cdc4871369bc8ddfc04be7b'
OUT=ROOT/'artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21'
PRIVATE=ROOT/'.c1_run_private'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
    'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def paths():return sorted({x.decode() for x in (git('diff','--name-only','-z',BASE)+git('ls-files','--others','--exclude-standard','-z')).split(b'\0') if x})


def check():
    source=read(OUT/'SOURCE_manifest.json')
    for row in source['files']:
        assert sha(ROOT/row['path'])==row['sha256']
        assert hashlib.sha256(git('show',source['SOURCE']+':'+row['path'])).hexdigest()==row['sha256']
    protected=read(PRIVATE/'post_STOP_preservation.json')
    assert len({x['path'] for x in protected})==35304
    for row in protected:assert sha(Path(row['path']))==row['sha256']
    tests=read(OUT/'synthetic_validation.json');assert tests['total']==63
    for log,n in [('driver_tests.txt',8),('inherited_tests.txt',55)]:
        text=(OUT/log).read_text();assert 'Ran '+str(n)+' tests' in text and text.rstrip().endswith('OK')
    for r in read(OUT/'mock_native_receipts.json')['records']:
        assert r['mock_only'] and r['receipt']['all_reaped']
        assert all(h['reaped'] for h in r['receipt']['workers']+r['receipt']['adopted_descendants'])
        for a in r['receipt']['axes']:assert a['all_reaped'] and all(h['reaped'] for h in a['workers'])
    runtime=read(OUT/'runtime_preflight.json');assert runtime['reason']=='ValueError:UNSAFE_MULTITHREADED_FORK_PARENT'
    assert runtime['observed_threads']==64 and runtime['native_exit']==2 and runtime['owned_probe_reaped']
    assert read(PRIVATE/'runtime_probe_native_exit.json')['native_exit']==2
    assert not (PRIVATE/'runtime_probe/fork_probe').exists()
    grants=read(OUT/'grant_status.json');assert grants['new_valid_grants_issued']==grants['consumed']==0
    for row in read(OUT/'condition_results.json')['conditions']:
        assert row['completed_n']==row['compiled_axes']==0 and row['requested_n']==43
        assert row['conditional_mean'] is row['unbiased_sample_variance'] is row['SE'] is None
        assert not any(row['reserved'].values()) and not any(row['actual'].values())
        assert not row['grant_issued'] and not row['grant_consumed'] and not row['marker_created'] and not row['ledger_created']
        old=read(ROOT/('artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/contract_'+row['condition']+'.json'))
        assert row['planned']==old['frozen_scientific_budget']
    assert not git('ls-files','--','.c1_run_private').strip()
    assert git('check-ignore','.c1_run_private/technical_runtime_probe_config.json').strip()
    assert not (PRIVATE/'H8_0.05').exists() and not (PRIVATE/'H8_0.0005').exists()
    for raw in git('diff','--name-only','--diff-filter=MDRT','-z',BASE).split(b'\0'):
        if raw:
            rel=raw.decode();assert rel in INDEXES
            assert (ROOT/rel).read_bytes().endswith(git('show',BASE+':'+rel))
    allowed=['scripts/h8_rare_c1_run.py','scripts/audit_h8_rare_c1_stop.py','tests/test_h8_rare_c1_run.py',
        'docs/research/h8_rare_rte_c1_pilot_run_20261021.md','docs/research/研究ノート/2026-10-21_h8_rare_rte_c1_pilot_run.md']
    links=0
    for rel in paths():
        assert rel in INDEXES+allowed or rel.startswith('artifacts/resource_applicability/h8_rare_rte_c1_pilot_run/2026-10-21/')
        text=(ROOT/rel).read_text()
        if rel in INDEXES:text=text[:-len(git('show',BASE+':'+rel).decode())]
        assert not re.search(r'/(?:home|root|tmp|Users)/|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}',text),rel
        if rel.endswith('.json'):json.loads(text)
        if rel.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'):continue
                assert ((ROOT/rel).parent/link.split('#')[0]).exists(),rel+' '+link
                links+=1
    return dict(status='PASS_POST_STOP_PRESERVATION_STRUCTURE_NOT_SCIENTIFIC_SUCCESS',
        SOURCE=source['SOURCE'],baseline=BASE,protected_files=35304,synthetic_tests=63,
        mock_receipts=14,all_mock_children_descendants_reaped=True,technical_probe_exit=2,
        technical_probe_reaped=True,science_workers=0,science_calls=0,grant_issued_consumed=0,
        conditional_estimates=None,old_evidence_changes=0,other_research_shared_environment_changes=0,
        observed_import_threads=64,runtime_probe_retried=False,thread_environment_modified=False,
        private_pattern_hits=0,markdown_links=links,push=False,review=False,freeze=False,mandatory_STOP=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--record',action='store_true');g.add_argument('--verify-committed',action='store_true')
    args=p.parse_args();status=check()
    if args.record:
        save('completion_audit.json',status)
        save('SHA256_manifest.json',dict(baseline=BASE,self_excluded=True,
            files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in paths() if x!=str((OUT/'SHA256_manifest.json').relative_to(ROOT))]))
        print('PASS_POST_STOP_AUDIT_NO_SCIENTIFIC_EXECUTION')
    else:
        assert not git('status','--porcelain').strip()
        rows=read(OUT/'SHA256_manifest.json')['files']
        assert {x['path'] for x in rows}==set(paths())-{str((OUT/'SHA256_manifest.json').relative_to(ROOT))}
        for row in rows:
            assert sha(ROOT/row['path'])==row['sha256'] and git('show','HEAD:'+row['path'])==(ROOT/row['path']).read_bytes()
        print('PASS_COMMITTED_STOP_SOURCE_RESULTS_PRESERVATION_SHA')
        print(git('rev-parse','HEAD').decode().strip());print('MANDATORY_STOP_NO_RETRY_NO_PHASE2')


if __name__=='__main__':main()
