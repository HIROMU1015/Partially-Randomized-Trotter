"""Read-only contract/preflight audit. No execution, grant or scientific CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h8_rare_c1_contract import BASE, budget, check_contract

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21'
PRIVATE=ROOT/'.c1_contract_private'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
         'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def paths():return sorted({p.decode() for p in (git('diff','--name-only','-z',BASE)+git('ls-files','--others','--exclude-standard','-z')).split(b'\0') if p})
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checks():
    for row in read(PRIVATE/'protected.json'):
        assert sha(row['path'])==row['sha256'],'ORIGINAL_BYTES_CHANGED'
    source=read(OUT/'SOURCE_manifest.json')
    assert git('merge-base','--is-ancestor',BASE,source['SOURCE'])==b''
    for row in source['files']:
        assert sha(ROOT/row['path'])==row['sha256']
        assert hashlib.sha256(git('show',source['SOURCE']+':'+row['path'])).hexdigest()==row['sha256']
    test=read(OUT/'synthetic_validation.json')
    assert test['new_contract_tests']==11 and test['inherited_C05_tests']==55
    assert 'Ran 11 tests' in (OUT/'contract_tests.txt').read_text()
    assert 'Ran 55 tests' in (OUT/'inherited_C05_tests.txt').read_text()
    for filename in ('contract_tests.txt','inherited_C05_tests.txt'):
        assert (OUT/filename).read_text().rstrip().endswith('OK')
    for row in test['tested_source_sha256']:
        assert sha(ROOT/row['path'])==row['sha256']
    plans=read(PRIVATE/'task_registries.json')
    occupied=set(read(PRIVATE/'old_seed_exclusion.json'));fresh=[]
    total={}
    for label in ('H8_0.05','H8_0.0005'):
        c=read(OUT/('contract_'+label+'.json'));assert check_contract(c,43)
        assert c['new_execution_SOURCE']==source['SOURCE']
        assert c['contract_status']=='FIXED_PENDING_SEPARATE_EXECUTION_APPROVAL_AND_FRESH_GATES'
        plan=plans[label]
        assert digest(dict(tasks=plan['tasks'],streams=plan['streams']))==c['private_task_registry_digest']
        seeds=[v for r in plan['streams'].values() for v in r.values()]+[plan['law_root_seed']]
        assert len(seeds)==len(set(seeds)) and not occupied.intersection(seeds)
        fresh.extend(seeds)
        assert len(plan['tasks'])==43 and len(plan['streams'])==43
        old=read(ROOT/('artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21/contract_'+label+'_PROPOSED.json'))
        for key in ('candidate','q','r','K','shots','fixed_PF','fixed_PF_G_exact','original_binding',
                    'target_preparation_sha256','original_law_sha256','canonical_order_probability_reference'):
            assert c[key]==old[key],'FIXED_METADATA_CHANGED'
        assert c['law_compatibility_analysis'].startswith('NOT_RUN')
        draft=read(OUT/('execution_approval_'+label+'_DRAFT.json'))
        assert not draft['explicit_C1_scientific_execution_authorized']
        assert draft['contract_sha256']==sha(OUT/('contract_'+label+'.json'))
        assert draft['budget']==c['frozen_scientific_budget'] and draft['SOURCE']==source['SOURCE']
        for key,v in c['frozen_scientific_budget'].items():total[key]=total.get(key,0)+v
    assert len(fresh)==len(set(fresh)) and len(fresh)==862
    assert total==read(OUT/'science_call_budget.json')['total_upper_budget']
    assert not git('ls-files','--','.c1_contract_private').strip()
    assert git('check-ignore','.c1_contract_private/task_registries.json').strip()
    assert not list(PRIVATE.glob('*grant*')) and not list(PRIVATE.glob('*marker*')) and not list(PRIVATE.glob('*ledger*'))
    for rel in git('diff','--name-only','--diff-filter=MDRT',BASE).decode().splitlines():
        assert rel in INDEXES,'OLD_FILE_MUTATION '+rel
        assert (ROOT/rel).read_bytes().endswith(git('show',BASE+':'+rel))
    allowed=['scripts/h8_rare_c1_contract.py','scripts/preflight_h8_rare_c1_contract.py','tests/test_h8_rare_c1_contract.py',
             'docs/research/h8_rare_rte_c1_pilot_contract_20261021.md','docs/research/研究ノート/2026-10-21_h8_rare_rte_c1_contract.md']
    hits=links=0
    for rel in paths():
        assert rel in INDEXES+allowed or rel.startswith('artifacts/resource_applicability/h8_rare_rte_c1_contract/2026-10-21/'),rel
        text=(ROOT/rel).read_text()
        if rel in INDEXES:text=text[:-len(git('show',BASE+':'+rel).decode())]
        # Scoped disclosure check only; not a security certificate.
        assert not re.search(r'/(?:home|tmp|Users|root)/|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}',text),rel
        if rel.endswith('.json'):json.loads(text)
        if rel.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'):continue
                assert ((ROOT/rel).parent/link.split('#')[0]).exists(),'BROKEN_LINK '+rel
                links+=1
    return dict(status='PASS_CONTRACT_TECHNICAL_PREFLIGHT_SCIENCE_BLOCKED',baseline=BASE,
        SOURCE=source['SOURCE'],n_per_condition=43,total_trajectories_planned=86,
        protected_physical_files_rehashed=len(read(PRIVATE/'protected.json')),
        tests=66,planning_numeric_paths_agree=True,new_private_seed_metadata_integers=862,
        seed_metadata_not_RNG_draws=True,new_H8_sampling=0,new_H8_law_or_saved_scientific_reanalysis=0,
        new_H8_simulation_circuit_compile=0,valid_grants_issued_consumed=0,scientific_workers=0,
        private_pattern_hits=hits,markdown_links=links,old_SOURCE_raw_grant_ledger_marker_STOP_changes=0,
        shared_environment_or_other_research_changes=0,push=False,detailed_scientific_review=False,
        final_evidence_freeze=False,mandatory_STOP=True,
        pending_execution_gates=['separate_explicit_user_approval','canonical_law_engineering_comparison',
            'live_import_and_SessionEquivalenceLibrary_identity','post_loader_fork_safety',
            'measured_or_bound_private_memory_thread_IO_admission','fresh_closed_byte_manifest_and_new_grant'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record',action='store_true');group.add_argument('--verify-committed',action='store_true')
    args=parser.parse_args()
    status=checks()
    if args.record:
        save('preflight.json',status)
        save('completion_audit.json',dict(**status,native_contract_driver_exit=0,
            science_native_exit=None,science_native_exit_reason='NOT_STARTED',
            mock_workers_reaped=True,old_public_bytes_preserved=True))
        save('STOP.json',dict(status='MANDATORY_STOP_CONTRACT_COMPLETE_C1_NOT_AUTHORIZED',
            execution=False,valid_grant=False,retry=False,resume=False,production=False,push=False))
        manifest=OUT/'SHA256_manifest.json'
        save(manifest.name,dict(baseline=BASE,self_excluded=True,
            files=[dict(path=x,sha256=sha(ROOT/x),bytes=(ROOT/x).stat().st_size) for x in paths() if ROOT/x!=manifest]))
        print('PASS_CONTRACT_PREFLIGHT_AND_COMPLETION_MANDATORY_STOP')
    else:
        assert not git('status','--porcelain').strip(),'WORKTREE_NOT_CLEAN'
        manifest=read(OUT/'SHA256_manifest.json')
        assert {r['path'] for r in manifest['files']}==set(paths())-{str((OUT/'SHA256_manifest.json').relative_to(ROOT))}
        for row in manifest['files']:
            assert sha(ROOT/row['path'])==row['sha256']
            assert git('show','HEAD:'+row['path'])==(ROOT/row['path']).read_bytes()
        print('PASS_COMMITTED_CONTRACT_SOURCE_RESULT_PRESERVATION_SHA')
        print(git('rev-parse','HEAD').decode().strip())
        print('MANDATORY_STOP_NO_C1_GRANT_OR_SCIENCE')


if __name__=='__main__':main()
