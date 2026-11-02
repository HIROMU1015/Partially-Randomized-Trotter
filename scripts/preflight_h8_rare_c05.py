"""C0.5 read-only metadata, preservation and structural audit; NO execute command."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/resource_applicability/h8_rare_rte_stage_c05/2026-10-21'
PRIVATE=ROOT/'.c05_private'
BASE='136bd2730e49938bae697ed32674dfcc53e63f07'
C0='artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21/'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
         'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def hash_bytes(data):return hashlib.sha256(data).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def changed_paths():
    return sorted({x.decode() for x in (git('diff','--name-only','-z',BASE)+git('ls-files','--others','--exclude-standard','-z')).split(b'\0') if x})


def technical_checks():
    tests=read(OUT/'synthetic_validation.json');assert tests['tests']==55
    assert 'Ran 55 tests' in (OUT/'synthetic_tests.txt').read_text()
    assert (OUT/'synthetic_tests.txt').read_text().rstrip().endswith('OK')
    for path,digest in tests['tested_source_sha256'].items():assert sha(ROOT/path)==digest
    for name in tests['tested_source_sha256']:
        tree=ast.parse((ROOT/name).read_text())
        for statement in tree.body:
            modules=[]
            if isinstance(statement,ast.Import):modules=[x.name for x in statement.names]
            if isinstance(statement,ast.ImportFrom):modules=[statement.module or '']
            assert not any(x.startswith(('numpy','qiskit','trotterlib','trottertracks')) for x in modules)
    for row in read(OUT/'source_acquisition.json')['fixed_blob_rechecks']:
        assert hash_bytes(git('show',row['commit']+':'+row['path']))==row['sha256']
    old_manifest=json.loads(git('show',BASE+':'+C0+'SHA256_manifest.json'))
    for row in old_manifest['files']:assert hash_bytes(git('show',BASE+':'+row['path']))==row['sha256']
    protected=read(PRIVATE/'preservation.json')
    helpers=read(PRIVATE/'resolved_live_helper_rows.json')
    closure=read(PRIVATE/'original_P_current_provider_rows.json')
    protected.extend(dict(path=x['path'],sha256=x['sha256']) for x in helpers)
    protected.extend(dict(path=x['path'],sha256=x['expected_sha256']) for x in closure)
    done={}
    for row in protected:
        path=row['path'];expected=row['sha256']
        if path not in done:assert sha(Path(path))==expected;done[path]=expected
        assert done[path]==expected
    runtime=read(OUT/'runtime_readonly_audit.json')
    for group in runtime['maps']:
        assert group['counts']['matched']==group['counts']['expected']
        assert group['counts']['missing']==group['counts']['mismatched']==0
    for scope in runtime['original_P_931_provider_closure_checks']:
        assert sum(x['counts']['matched'] for x in scope['groups'])==931
        assert all(x['counts']['missing']==x['counts']['mismatched']==0 for x in scope['groups'])
    assert runtime['source_live_locator_resolution']['matched']==21
    assert not runtime['session_equivalence_library_live_state_verified']
    previous=json.loads(git('show',BASE+':'+C0+'conditions_reference.json'))['conditions']
    for ref in previous:
        draft=read(OUT/('contract_'+ref['condition']+'_PROPOSED.json'))
        assert draft['approved'] is draft['runtime_authorization'] is draft['numeric_model_approved'] is False
        assert draft['status']=='PROPOSED' and draft['frozen_scientific_budget'] is None
        assert draft['new_execution_SOURCE'] is None and not draft['valid_grant_issued']
        assert draft['n_by_stratum']==dict(S2_SINGLE=None,S2_MULTI=None,S4_PRESENT=None)
        for key in ['candidate','q','r','K','shots','fixed_PF','fixed_PF_G_exact','original_binding','target_preparation_sha256']:
            assert draft[key]==ref[key]
        assert not draft['production_authorized'] and draft['unknown_multi_S4_means_variances_remain_null']
    records=read(OUT/'mock_native_receipts.json')['records']
    for record in records:
        assert record['mock_only']
        r=record['receipt'];assert r['all_reaped'] and not r['retry'] and not r['resume']
        assert all(h['reaped'] for h in r['workers']+r['adopted_descendants'])
        for child in r['axes']:assert child['all_reaped'] and all(h['reaped'] for h in child['workers'])
    seed=read(OUT/'seed_exclusion_receipt.json')
    assert seed['new_C1_seeds_created']==seed['real_RTE_event_sampling']==0
    assert sha(PRIVATE/'old_seed_exclusion.json')==seed['old_exclusion_manifest_sha256']
    assert not git('ls-files','--','.c05_private').strip()
    assert git('check-ignore','.c05_private/preservation.json').strip()
    assert not list(PRIVATE.glob('*grant*.json')) and not list(PRIVATE.glob('*marker*.json'))
    changed_old=git('diff','--name-only','--diff-filter=MDRT','-z',BASE).split(b'\0')
    for row in changed_old:
        if row:
            rel=row.decode();assert rel in INDEXES
            assert (ROOT/rel).read_bytes().endswith(git('show',BASE+':'+rel))
    return dict(status='PASS_C05_TECHNICAL_PREFLIGHT_NOT_SCIENTIFIC_EXECUTION',baseline=BASE,synthetic_tests=55,
        tested_source_hashes_match=True,original_C0_manifest_files=len(old_manifest['files']),
        original_operational_input_helper_provider_unique_files_rechecked=len(done),
        historical_runtime_files_34273_match_at_readonly_snapshot=True,
        runtime_34273_rehash_repeated_by_final_checker=False,
        original_P_provider_931_match_both_scopes=True,helper_source_21_resolved=True,
        live_session_library_import_and_postload_fork_safety='PENDING_C1_FRESH_GATE',
        mock_native_receipts=len(records),all_mock_children_and_adopted_descendants_reaped=True,
        new_C1_seed_or_RTE_sampling=0,real_H8_law_count_probability_calculations=0,
        real_H8_circuit_compile_signal_simulation_SCF_DF_state_generation=0,
        real_saved_value_scientific_numeric_analysis=0,valid_grants_issued_consumed=0,
        scientific_workers_created=0,old_SOURCE_raw_results_grant_ledger_marker_STOP_mutations=0,
        other_research_or_shared_environment_changes=0,push=False,detailed_scientific_review=False,
        final_evidence_freeze=False,C1_authorized=False,numerical_model_adopted=False,mandatory_STOP=True)


def structure_checks():
    patterns=[re.compile(re.escape('/'+s+'/')) for s in ['home','tmp','root','Users']]
    patterns += [re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
                 re.compile(r'AKIA[0-9A-Z]{16}'),re.compile('-----BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
    pending={OUT/n for n in ['preflight.json','completion_audit.json','resource_observation.json','status_manifest.json','STOP.json','SHA256_manifest.json']}
    links=0
    for rel in changed_paths():
        assert rel in INDEXES or rel.startswith(('scripts/h8_rare_c05/','artifacts/resource_applicability/h8_rare_rte_stage_c05/','docs/research/h8_rare_rte_stage_c05_')) or rel in ['scripts/preflight_h8_rare_c05.py','tests/test_h8_rare_c05.py','docs/research/研究ノート/2026-10-21_h8_rare_rte_stage_c05.md'],rel
        p=ROOT/rel;data=p.read_bytes()
        if rel in INDEXES:
            old=git('show',BASE+':'+rel);assert data.endswith(old);data=data[:-len(old)]
        text=data.decode();assert not any(pattern.search(text) for pattern in patterns),'PRIVATE_PATTERN: '+rel
        if rel.endswith('.json'):read(p)
        if rel.endswith('.md'):
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'):continue
                dest=(p.parent/link.split('#')[0]).resolve()
                assert dest.exists() or (args.record and dest in pending),'BROKEN_LINK '+rel+' '+link
                links+=1
    return dict(scoped_private_pattern_hits=0,private_scanner_is_complete_security_certificate=False,
                local_markdown_links_checked=links,old_index_bytes_retained_as_suffix=True)


def record():
    status=technical_checks();status.update(structure_checks())
    save('preflight.json',status)
    # Reuse existing C0 read-only resource observer; no loader or scientific import.
    spec=importlib.util.spec_from_file_location('c0_readonly_resource',ROOT/'scripts/preflight_h8_rare_rte_c0.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    resource=module.resources();resource.update(C05_worker_count_selected=None,C05_H8_performance_benchmark=False)
    topology={}
    for cpu in resource['affinity']:
        p=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
        topology.setdefault(key,[]).append(cpu)
    resource['physical_core_groups_in_affinity']=list(topology.values());resource['physical_core_group_count']=len(topology)
    save('resource_observation.json',resource)
    save('completion_audit.json',dict(status='PASS_C05_PRESERVATION_STRUCTURE_MANDATORY_STOP',**{k:v for k,v in status.items() if k!='status'}))
    save('STOP.json',dict(status='MANDATORY_STOP_C05_COMPLETE_C1_NOT_AUTHORIZED',C1_sampling_compile_numeric_analysis=False,
        valid_grant_issued_consumed=False,pilot_or_production_started=False,push=False,retry=False,resume=False))
    save('status_manifest.json',dict(schema='H8_RARE_C05_STATUS_V1',status='IMPLEMENTED_MOCK_VERIFIED_SCIENCE_UNAPPROVED',
        baseline=BASE,production_adapter_code_present=True,real_provider_executed=False,synthetic_tests=55,
        numeric_policy='PROPOSED_NOT_ADOPTED',sample_size_finalized=False,scientific_grants=0,
        scientific_execution=0,publication_authorized=False,mandatory_STOP=True))
    manifest=OUT/'SHA256_manifest.json'
    rows=[dict(path=rel,bytes=(ROOT/rel).stat().st_size,sha256=sha(ROOT/rel)) for rel in changed_paths() if ROOT/rel!=manifest]
    save(manifest.name,dict(schema='H8_RARE_C05_SOURCE_ARTIFACT_SHA256_V1',baseline=BASE,self_excluded=True,files=rows))
    print('PASS_C05_PREFLIGHT_AND_COMPLETION_NO_SCIENTIFIC_EXECUTION');print('manifest_files',len(rows))


def verify_committed():
    assert not git('status','--porcelain').strip(),'WORKTREE_NOT_CLEAN'
    technical_checks();structure_checks()
    manifest=read(OUT/'SHA256_manifest.json')
    expected=set(changed_paths())-{str((OUT/'SHA256_manifest.json').relative_to(ROOT))}
    assert {r['path'] for r in manifest['files']}==expected
    for row in manifest['files']:
        p=ROOT/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        assert git('show','HEAD:'+row['path'])==p.read_bytes()
    print('PASS_COMMITTED_C05_SOURCE_RESULTS_AND_PRESERVATION_SHA')
    print('commit',git('rev-parse','HEAD').decode().strip())
    print('MANDATORY_STOP_NO_C1_SAMPLING_COMPILE_ANALYSIS_GRANT_OR_PUSH')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--record',action='store_true');m.add_argument('--verify-committed',action='store_true')
    args=p.parse_args()
    if args.record:record()
    else:verify_committed()
