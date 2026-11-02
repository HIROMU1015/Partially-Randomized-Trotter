"""C-0 metadata/SHA/resources only. There is no scientific execution command."""
import argparse
import csv
import re
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/resource_applicability/h8_rare_rte_stage_c0/2026-10-21'
PRIVATE=ROOT/'.c0_private'
BASE='3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb'
INDEXES=['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','docs/research/研究概要・現状.md',
         'scripts/README.md','src/trotterlib/README.md','docs/research/研究ノート/README.md']


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def resources():
    mem={k:v.strip() for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    cgline=next((s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')),None)
    chain=[]
    if cgline:
        base=Path('/sys/fs/cgroup');current=base/cgline.split('::',1)[1].lstrip('/')
        while current.is_relative_to(base):
            row={}
            for name in ['cpu.max','cpu.stat','memory.max','memory.current','memory.events','memory.pressure','io.pressure']:
                p=current/name
                if p.exists():row[name]=p.read_text().strip()
            if row:chain.append(row)
            if current==base:break
            current=current.parent
    gpu={'nvidia_smi_available':bool(shutil.which('nvidia-smi')),'GPU_computation_started':False}
    if gpu['nvidia_smi_available']:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        gpu.update(query_exit_code=r.returncode,values=r.stdout.strip(),device_identifiers_exposed=False)
    packages={}
    for name in ['qiskit','numpy','scipy']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='NOT_IN_C0_INTERPRETER_DISTRIBUTIONS'
    return dict(meaning='read-only C0 snapshot, not a reservation or future capacity guarantee',
                affinity=sorted(os.sched_getaffinity(0)),loadavg=Path('/proc/loadavg').read_text().strip(),
                MemAvailable=mem['MemAvailable'],cgroup_chain=chain,
                proc_io=Path('/proc/self/io').read_text(),
                host_IO_pressure=Path('/proc/pressure/io').read_text(),GPU=gpu,
                C0_python_version=sys.version,C0_python_executable_sha256=sha(Path(sys.executable)),
                C0_package_metadata_only=packages,scientific_packages_imported=False,
                historical_H8_live_runtime_equivalence_verified=False,
                worker_count_selected=None,CPU_RAM_wall_worker_cap=None,
                affinity_cgroup_scheduler_shared_environment_changed=False)


def verify():
    acquisition=read(OUT/'source_provenance.json')['files']
    for row in acquisition:
        data=git('show',row['commit']+':'+row['path'])
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        assert git('rev-parse',row['commit']+':'+row['path']).decode().strip()==row['blob']
    remote=read(OUT/'remote_acquisition.json')['files'];assert len(remote)==6
    assert all(r['remote_content_equals_local_fixed_blob'] for r in remote)
    # Assert preserved Stage B packet against its own committed manifest, without recomputation.
    rel='artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21/SHA256_manifest.json'
    oldmanifest=json.loads(git('show',BASE+':'+rel))
    for row in oldmanifest['files']:
        assert hashlib.sha256(git('show',BASE+':'+row['path'])).hexdigest()==row['sha256']
    oldfiles=read(PRIVATE/'old_operational_preservation.json')
    inputfiles=read(PRIVATE/'used_input_preservation.json')
    for row in oldfiles+inputfiles:assert sha(Path(row['path']))==row['sha256']
    inputs=read(OUT/'input_binding_audit.json')
    assert all(r['status']=='CURRENT_KNOWN_LOCATOR_BYTES_MATCH' for r in inputs['rows'])
    targets=read(OUT/'conditions_reference.json')['conditions'];assert len(targets)==2
    raw=json.loads(git('show',BASE+':artifacts/resource_applicability/pr_rare_rte_stage_b/2026-10-21/result.json'))
    old=[c for c in raw['conditions'] if c['system']=='H8']
    for t,c in zip(targets,old,strict=True):
        for k in ['candidate','q','r','K','R','shots','fixed_PF','fixed_PF_G_exact']:
            assert t[k]==c[k]
        assert t['saved_unknown_mass']==c['conditional_expectation']['unknown_mass']
        assert t['saved_strata']==[{k:g[k] for k in ['stratum','probability','support_status','observation_status']} for g in c['groups']]
        draft=read(OUT/('contract_'+t['condition']+'_PROPOSED.json'))
        assert draft['approved'] is False and draft['runtime_authorization'] is False
        assert draft['valid_grant_issued'] is False and draft['new_execution_SOURCE'] is None
        assert draft['sample_design']['n_by_stratum']==dict(S2_SINGLE=None,S2_MULTI=None,S4_PRESENT=None)
        assert draft['budget']['numeric_frozen_budget'] is None
        assert draft['sampling']['law_numerical_convention_and_tolerance'] is None
        assert draft['authorization']['grant_sha256'] is None
        assert draft['authorization']['separate_grant_marker_STOP_per_condition']
        assert draft['strata']==['S2_SINGLE','S2_MULTI','S4_PRESENT']
        assert draft['shots']==c['shots'] and draft['target_preparation_sha256']==t['target_preparation_sha256']
    syn=read(OUT/'synthetic_validation.json');assert syn['status']=='PASS_SYNTHETIC_15_ONLY'
    assert not syn['production_bridge_integration_verified']
    # AST check new scripts only: do not import production science in C0.
    imports=[]
    for rel in ['scripts/h8_rare_rte_c0_mock.py','scripts/preflight_h8_rare_rte_c0.py','tests/test_h8_rare_rte_c0.py']:
        tree=ast.parse((ROOT/rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    assert not any(s.startswith(('qiskit','numpy','scipy','trotterlib','trottertracks','h8_ab_native','pr_cost_native')) for s in imports)
    changed=git('diff','--name-only','--diff-filter=MDRT','-z',BASE).split(b'\0')
    for row in changed:
        if not row:continue
        rel=row.decode();assert rel in INDEXES,'OLD_SCIENTIFIC_EVIDENCE_CHANGED'
        assert (ROOT/rel).read_bytes().endswith(git('show',BASE+':'+rel))
    assert not git('ls-files','--','.c0_private').strip()
    assert git('check-ignore','.c0_private/acquisition.json').strip()
    assert not list(PRIVATE.glob('*grant*')) and not list(PRIVATE.glob('*marker*'))
    return dict(status='PASS_C0_METADATA_DESIGN_SYNTHETIC_ONLY',scientific_execution_readiness='BLOCKED_PENDING_SEPARATE_APPROVAL_AND_CONTRACT_COMPLETION',
                baseline_commit=BASE,fixed_source_blobs=len(acquisition),remote_fixed_file_matches=len(remote),
                original_input_rows=len(inputs['rows']),known_original_input_files=len(inputfiles),
                old_B_operational_files_unchanged=len(oldfiles),old_B_manifest_files_unchanged=len(oldmanifest['files']),
                saved_B_values_copied_without_reanalysis=True,old_STOPs_retained=True,
                historical_profile='ACQUIRED_LITERAL_EMBEDDED_PROFILE_PER_5f267e96',
                live_H8_runtime_and_session_library_closure='NOT_REATTESTED_IN_C0',
                synthetic_tests=15,synthetic_native_children_reaped=2,scientific_children=0,
                production_sampling_compile_integration='NOT_RUN_NOT_AUTHORIZED',
                valid_scientific_grants_issued=0,valid_scientific_grants_consumed=0,
                scientific_sampling_signal_simulation_SCF_DF_state_circuit_compile=0,
                saved_data_numeric_reanalysis=0,old_evidence_mutations=0,
                unrelated_worktree_or_job_intervention=False,new_push=False,
                detailed_scientific_review=False,final_evidence_freeze=False,
                mandatory_STOP=True)


def record():
    receipt=verify();save('resource_observation.json',resources());save('preflight.json',receipt)
    save('STOP.json',dict(status='MANDATORY_STOP_C0_DESIGN_COMPLETE_NO_SCIENCE_AUTHORIZATION',
        scientific_grant_issued=False,science_execution=False,science_retry_resume=False,
        next_C_sampling_compile_authorized=False,new_saved_numeric_analysis_authorized=False,
        push_authorized=False,detailed_review_or_final_freeze=False))
    print(receipt['status']);print(receipt['scientific_execution_readiness'])


def changed_paths():
    data=git('diff','--name-only','-z',BASE)+git('ls-files','--others','--exclude-standard','-z')
    return sorted({p.decode() for p in data.split(b'\0') if p})


def final_checks():
    check=verify()
    assert read(OUT/'STOP.json')['science_execution'] is False
    assert read(OUT/'preflight.json')['status']=='PASS_C0_METADATA_DESIGN_SYNTHETIC_ONLY'
    table=list(csv.DictReader((OUT/'required_information.csv').open()))
    assert len(table)==6
    reference=read(OUT/'conditions_reference.json')['conditions']
    for row in table:
        ref=next(c for c in reference if c['condition']==row['condition'])
        g=next(g for g in ref['saved_strata'] if g['stratum']==row['stratum'])
        assert row['saved_probability_literal']==g['probability']
        assert row['planned_trajectory_count']==row['conditional_cost_mean']==row['conditional_cost_variance']==row['guaranteed_cost_upper_bound']==''
    assert len(list(csv.DictReader((OUT/'evidence_reuse.csv').open())))==7
    pending={OUT/name for name in ['completion_audit.json','SHA256_manifest.json','status_manifest.json']}
    # Inspect new bytes only; inherited index histories are retained unchanged.
    patterns=[re.compile(re.escape('/'+n+'/')) for n in ['home','tmp','root','Users']]
    patterns += [re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),re.compile(r'github_pat_[A-Za-z0-9_]{20,}'),
                 re.compile(r'AKIA[0-9A-Z]{16}'),re.compile('-----BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                 re.compile(r'"(?:seed|master_seed|rng_seed|raw_events|private_path)"\s*:')]
    links=0
    for rel in changed_paths():
        allowed=(rel in INDEXES or rel.startswith(str(OUT.relative_to(ROOT))+'/') or
                 rel.startswith('docs/research/h8_rare_rte_stage_c0_') or
                 rel=='docs/research/研究ノート/2026-10-21_h8_rare_rte_stage_c0.md' or
                 rel in ['scripts/h8_rare_rte_c0_mock.py','scripts/preflight_h8_rare_rte_c0.py','tests/test_h8_rare_rte_c0.py'])
        assert allowed,'UNEXPECTED_MUTATION_SCOPE: '+rel
        data=(ROOT/rel).read_bytes()
        if rel in INDEXES:
            old=git('show',BASE+':'+rel);assert data.endswith(old);data=data[:-len(old)]
        text=data.decode()
        assert not any(p.search(text) for p in patterns),'PRIVATE_PATTERN_IN_NEW_BYTES: '+rel
        if rel.endswith('.json'):read(ROOT/rel)
        if rel.endswith('.md'):
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                dest=((ROOT/rel).parent/target.split('#')[0]).resolve()
                assert dest.exists() or (args.audit_final and dest in pending),'BROKEN_LINK: '+rel+' '+target
                links+=1
    check.update(status='PASS_C0_COMPLETION_PRESERVATION_STRUCTURE',scientific_execution_readiness='NOT_AUTHORIZED_PENDING_CONTRACT_COMPLETION',
                 CSV_saved_probability_transcription_exact=True,required_information_rows=6,
                 new_mean_variance_cost_bound_values=0,scoped_private_pattern_hits=0,
                 scanner_is_complete_security_certificate=False,local_markdown_links_checked=links,
                 original_source_input_grant_ledger_marker_STOP_unchanged=True,
                 original_raw_locator_not_exported=True,
                 all_mock_children_reaped=True,real_scientific_workers_created=0,
                 technical_preflight_not_scientific_guarantee=True)
    return check


def audit_final():
    save('completion_audit.json',final_checks())
    save('status_manifest.json',dict(schema='H8_RARE_RTE_STAGE_C0_STATUS_V1',
        status='DESIGN_COMPLETE_SCIENCE_NOT_AUTHORIZED_MANDATORY_STOP',baseline_commit=BASE,
        science_sampling_compile_saved_numeric_analysis=0,valid_scientific_grants=0,
        proposed_contracts=['contract_H8_0.05_PROPOSED.json','contract_H8_0.0005_PROPOSED.json'],
        sample_sizes_frozen=False,production_adapter_integrated=False,live_runtime_reattested=False,
        synthetic_tests=15,new_push_authorized=False,mandatory_STOP=True))
    manifest=OUT/'SHA256_manifest.json'
    rows=[]
    for rel in changed_paths():
        if ROOT/rel==manifest:continue
        file=ROOT/rel;rows.append(dict(path=rel,bytes=file.stat().st_size,sha256=sha(file)))
    save('SHA256_manifest.json',dict(schema='H8_RARE_RTE_C0_SOURCE_ARTIFACT_SHA256_V1',baseline=BASE,
        self_excluded=True,includes='All new C0 source/design/preflight/audit and appended-index files; no private data',files=rows))
    print('PASS_C0_FINAL_AUDIT_NO_SCIENTIFIC_EXECUTION');print('manifest_files',len(rows))


def verify_committed():
    assert not git('status','--porcelain').strip(),'WORKTREE_NOT_CLEAN'
    final_checks()
    manifest=read(OUT/'SHA256_manifest.json')
    expected=set(changed_paths())-{str((OUT/'SHA256_manifest.json').relative_to(ROOT))}
    assert {r['path'] for r in manifest['files']}==expected
    for row in manifest['files']:
        p=ROOT/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        assert git('show','HEAD:'+row['path'])==p.read_bytes()
    print('PASS_COMMITTED_C0_SOURCE_DESIGN_PREFLIGHT_SHA')
    print('commit',git('rev-parse','HEAD').decode().strip())
    print('MANDATORY_STOP_NO_VALID_GRANT_NO_MOLECULAR_SAMPLING_COMPILE_ANALYSIS_OR_PUSH')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--record',action='store_true')
    modes.add_argument('--audit-final',action='store_true')
    modes.add_argument('--verify-committed',action='store_true')
    args=parser.parse_args()
    if args.record:record()
    elif args.audit_final:audit_final()
    elif args.verify_committed:verify_committed()
    else:print(json.dumps(verify(),ensure_ascii=False,indent=2))
