"""Final A-only metadata/byte audit; no science imports or primitive evaluation."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import h2_pr_identity_recovery as r
import h2_pr_diagonal_a as a
import audit_h2_pr_diagonal_a as audit
import audit_h2_pr_equivalence as prior
DOCS={'PROJECT_MAP.md','VALIDATION_STATUS.md','artifacts/validation_manifest.json','docs/README.md','scripts/README.md',
 'src/trotterlib/README.md','docs/research/研究概要・現状.md','docs/research/track_a_h2_pr_diagonal_reuse_a_20261023.md',
 'docs/research/研究ノート/2026-10-23_h2_pr_diagonal_reuse_a.md'}


def read(p):return r.loads(Path(p).read_bytes())


def verify(config):
    manifest=read(OUT/'SOURCE.json');source=manifest['audit_SOURCE']
    for item in manifest['files']:
        raw=(ROOT/item['path']).read_bytes()
        r.require(hashlib.sha256(raw).hexdigest()==item['sha256'] and raw==r.git_bytes(source,item['path']),'SOURCE_BYTES')
    plan=read(OUT/'execution_contract_DRAFT.json');binding=read(OUT/'input_binding.json')
    r.require(a.preflight(plan,binding['binding'],binding['producer_sources'])==read(OUT/'metadata_preflight.json'),'METADATA_PREFLIGHT')
    r.require(plan['budget']==a.BUDGET and not plan['scientific_execution_authorized'],'UNAPPROVED_EXACT_BUDGET')
    tests=read(OUT/'test_results.json');r.require(tests['status']=='PASS' and tests['tests']==sum(x['tests'] for x in tests['suites'])==284,'TEST_TOTAL')
    for row in tests['suites']:
        r.require(row['passed']==row['tests'] and row['exit_code']==0 and r.file_digest(ROOT/row['path'])==row['sha256'],'TEST_FILE_BINDING')
    scope=read(OUT/'axis_parameter_scope_134.json');rows=scope['rows']
    r.require(len(rows)==134 and len({(x['task'],x['axis']) for x in rows})==134,'AXIS_COUNT')
    r.require(Counter(x['candidate'] for x in rows)==Counter(H2O32_c7ec590a32a859bc=66,H2O32_8a2ecdd0294ffb74=68),'FIXED_COHORT')
    for row in rows:
        r.require(row['cohort']=='EXPLORATION' and row['absolute_deterministic_diagonal_vs_common_preparation']=='UNRESOLVED' and
                  not row['alternative_adopted'] and row['independent_existing_receipt'] is None,'NO_SCOPE_PROMOTION')
        r.require(tuple(row['signed_piece_times'])==a.expected_piece_times(row['task']),'ORIGINAL_SCHEDULE')
        for op in row['parameter_occurrences']:
            a.time_pair(op['piece_time_hex'],op['primitive_argument_hex']);r.require(op['independent'] is False,'QPY_NOT_INDEPENDENT')
    search=read(OUT/'existing_receipt_search.json');r.require(search['independent_same_target_receipts_found']==0,'CASE_A2')
    catalogue=r.Catalogue();old=catalogue.verify_config(config);root=config.parents[2]
    compare=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    paths=sorted(prior.watch_set(catalogue,old,root,compare));observed=prior.observe(paths)
    preserve=read(OUT/'preservation_audit.json')
    r.require(len(paths)==16804 and r.digest(observed)==preserve['before_sha256']==preserve['after_sha256'],'ORIGINAL_PRESERVATION')
    r.require(compare['source_commit']==manifest['strict_COMPARE_SOURCE_unchanged'] and
              r.digest(compare)==manifest['strict_contract_digest_unchanged'],'STRICT_CONTRACT_UNCHANGED')
    alt='artifacts/resource_applicability/track_a_h2_pr_equivalence_contract/2026-10-23/alternative_contract_DRAFT.json'
    r.require((ROOT/alt).read_bytes()==r.git_bytes(a.BASE,alt) and r.file_digest(ROOT/alt)==manifest['alternative_contract_sha256'],'ALTERNATIVE_CONTRACT_UNCHANGED')
    prior_manifest=r.loads(r.git_bytes(a.BASE,'artifacts/validation_manifest.json'));current=read(ROOT/'artifacts/validation_manifest.json')
    r.require(all(current[k]==v for k,v in prior_manifest.items()) and
              set(current)-set(prior_manifest)=={'track_a_h2_pr_diagonal_reuse_a_20261023'},'OLD_MANIFEST_PRESERVED')
    old_files=set(subprocess.check_output(['git','ls-tree','-r','--name-only',a.BASE],cwd=ROOT).decode().splitlines())
    added={p for p in audit.FILES if p not in old_files}
    changed=subprocess.check_output(['git','diff','--name-only','-z',a.BASE],cwd=ROOT).decode().split('\0')
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    changed=sorted(set(filter(None,changed)));namespace=str(OUT.relative_to(ROOT))+'/'
    r.require(all(p in DOCS or p in added or p.startswith(namespace) for p in changed),'ONLY_A_TECHNICAL_SCOPE_CHANGED')
    forbidden={'application_sequence','registry_seed_PRIVATE','science_master_seed','seed_bytes','seed_hex'}
    def inspect(v):
        if isinstance(v,dict):
            r.require(not forbidden.intersection(v),'NO_PRIVATE_EVENT_SEED_PAYLOAD')
            for value in v.values():inspect(value)
        elif isinstance(v,list):
            for value in v:inspect(value)
    for p in OUT.rglob('*'):
        if p.is_file():
            r.require(p.suffix in ('.md','.py','.json','.txt'),'NO_RAW_CONFIG_OR_ARCHIVE_COPIES')
            raw=p.read_text();r.require('/'.join(('','home','AbeHiromu')) not in raw and '/'.join(('','usr','lib','')) not in raw,'NO_NEW_INTERNAL_ABSOLUTE_PATHS')
            if p.suffix=='.json':inspect(read(p))
    for p in (OUT/'REPORT.md',ROOT/'docs/research/track_a_h2_pr_diagonal_reuse_a_20261023.md',
              ROOT/'docs/research/研究ノート/2026-10-23_h2_pr_diagonal_reuse_a.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            resolved=(p.parent/target).resolve()
            r.require(resolved.is_file() or resolved in (OUT/'final_verification.json',OUT/'file_inventory.json'),'REPORT_LINK:'+target)
    stop=read(OUT/'STOP.json')
    r.require(stop['mandatory_STOP'] and not stop['A_condition_resolved'] and not stop['alternative_adopted'] and
              stop['COMPARE']=='BLOCKED' and not stop['scientific_authorized'],'MANDATORY_STOP')
    r.require(all(stop[k]==0 for k in ('real_primitive_evaluations','sampling','H2_circuit_build','lowering','compile','transpile',
      'simulation','u_N_G','allowance','ranking','scientific_imports','new_grants','old_grants_reused')) and not stop['push'],'NO_NEW_SCIENCE_OR_PUBLICATION')
    r.require(not any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules),'NO_SCIENTIFIC_IMPORTS')
    return dict(status='PASS',performed_at_UTC=datetime.now(timezone.utc).isoformat(),base_commit=a.BASE,audit_SOURCE=source,
      verifier_sha256=r.file_digest(__file__),protected_files_rechecked=16804,protected_changes=0,case='A2',
      independent_real_receipts=0,A_verified_axes=0,A_unresolved_axes=134,static_parameter_coverage_axes=134,
      exact_primitive_budget=6,actual_primitive_calls=0,single_half_time_source_and_user_confirmed=True,
      prior_manifest_and_science_SOURCE_contracts_results_grants_ledgers_markers_STOP_unchanged=True,
      changed_paths=changed,tests=284,new_scientific_calls=0,new_live_grants=0,private_raw_copies=0,
      internal_absolute_paths_in_new_artifacts=False,alternative_adopted=False,COMPARE='BLOCKED',push=False,mandatory_STOP=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    target=OUT/'final_verification.json';r.require(not target.exists(),'NO_FINAL_OVERWRITE')
    value=verify(args.config.resolve(strict=True))
    with target.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:value[k] for k in ('status','case','protected_files_rechecked','protected_changes','tests','new_scientific_calls','mandatory_STOP')}))
