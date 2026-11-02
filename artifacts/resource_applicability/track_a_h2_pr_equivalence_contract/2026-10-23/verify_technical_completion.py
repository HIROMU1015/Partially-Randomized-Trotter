"""Final static completion audit; no scientific imports, execution or grants."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import h2_pr_identity_recovery as r
import h2_pr_equivalence_contract as contract
import audit_h2_pr_equivalence as audit
BASE=contract.BASE
NAMESPACE=str(OUT.relative_to(ROOT))+'/'
DOCS={'PROJECT_MAP.md','VALIDATION_STATUS.md','artifacts/validation_manifest.json','docs/README.md',
 'scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md',
 'docs/research/track_a_h2_pr_equivalence_contract_20261023.md',
 'docs/research/研究ノート/2026-10-23_h2_pr_equivalence_contract.md'}


def read(path):return r.loads(Path(path).read_bytes())


def verify(config):
    source=read(OUT/'SOURCE.json');commit=source['audit_SOURCE']
    for item in source['files']:
        raw=(ROOT/item['path']).read_bytes()
        r.require(hashlib.sha256(raw).hexdigest()==item['sha256'] and raw==r.git_bytes(commit,item['path']), 'SOURCE_FILE_IDENTITY')
    catalogue=r.Catalogue();old=catalogue.verify_config(config);execution=config.parents[2]
    compare=r.loads(r.git_bytes(r.BASE,r.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    r.require(compare['source_commit']==contract.OLD_SOURCE and r.digest(compare)==contract.OLD_DIGEST, 'OLD_STRICT_SOURCE_CONTRACT')
    paths=sorted(audit.watch_set(catalogue,old,execution,compare))
    observed=audit.observe(paths)
    previous=r.loads(r.git_bytes(BASE,audit.PRIOR+'preservation_audit.json'))
    r.require(len(paths)==previous['checked_files']==16804, 'PROTECTED_FILE_COUNT')
    for path,item in zip(paths,previous['files']):r.require(observed[path]==item['before']==item['after'], 'PROTECTED_ORIGINAL_CHANGED')
    preserve=read(OUT/'preservation_audit.json')
    r.require(r.digest(observed)==preserve['before_observations_sha256']==preserve['after_observations_sha256'], 'CURRENT_AND_AUDIT_OBSERVATIONS')
    rows=read(OUT/'axis_evidence_134.json')['rows']
    packets=read(OUT/'operation_stream_evidence_134.json')['rows']
    r.require(len(rows)==len(packets)==134, 'AXIS_COUNT')
    for row,packet in zip(rows,packets):
        r.require((row['task'],row['axis'])==(packet['task'],packet['axis']), 'AXIS_PACKET_ORDER')
        r.require(all(x['evidence_sha256']==r.digest(packet) for x in row['conditions'].values()), 'AXIS_EVIDENCE_PACKET_DIGEST')
        r.require(packet['structural']['status']=='VERIFIED' and packet['structural']['global_phase_relation_exact'] is True, 'STATIC_STRUCTURE_AND_PHASE')
        r.require(packet['strict_contract_status']=='BLOCKED' and packet['alternative_policy_adopted'] is False and
                  packet['science_authorized'] is False, 'NO_TECHNICAL_SCOPE_PROMOTION')
    actual=contract.preflight(read(OUT/'alternative_contract_DRAFT.json'),rows)
    r.require(actual==read(OUT/'alternative_preflight.json'), 'ALTERNATIVE_PREFLIGHT_REPRODUCED')
    r.require(actual['technical_axis_counts']['A']['UNRESOLVED']==134 and
              actual['technical_axis_counts']['B']['UNRESOLVED']==134 and not actual['scientific_GO'], 'A_B_SEPARATE_UNRESOLVED')
    for name in contract.A+contract.B+contract.OUTSIDE:
        expected=('NOT_APPLICABLE' if name in contract.OUTSIDE else 'UNRESOLVED' if name in
            ('absolute_deterministic_diagonal_vs_common_preparation','compiler_relevant_runtime_output_equivalence') else 'VERIFIED')
        r.require(Counter(row['conditions'][name]['status'] for row in rows)==Counter({expected:134}), 'CONDITION_COUNTS')
    tests=read(OUT/'test_results.json')
    r.require(tests['status']=='PASS' and tests['tests']==sum(x['tests'] for x in tests['suites'])==248 and tests['new']==48, 'TEST_TOTAL')
    for item in tests['suites']:
        r.require(item['passed']==item['tests'] and item['exit_code']==0 and
                  r.file_digest(ROOT/item['path'])==item['sha256'], 'ARTIFICIAL_TEST_SOURCE')
    original=r.loads(r.git_bytes(BASE,'artifacts/validation_manifest.json'))
    current=read(ROOT/'artifacts/validation_manifest.json')
    r.require(all(current[k]==v for k,v in original.items()), 'OLD_MANIFEST_ITEMS_CHANGED')
    r.require(set(current)-set(original)=={'track_a_h2_pr_equivalence_contract_20261023'}, 'ONLY_NEW_MANIFEST_KEY')
    changed=subprocess.check_output(['git','diff','--name-only','-z',BASE],cwd=ROOT).decode().split('\0')
    other=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    changed=sorted(set(filter(None,changed+other)))
    base_files=set(subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT).decode().splitlines())
    new_sources={p for p in audit.FILES if p not in base_files}
    r.require(all(p in DOCS or p in new_sources or p.startswith(NAMESPACE) for p in changed), 'ONLY_NEW_TECHNICAL_NAMESPACE_AND_INDEXES')
    for p in (OUT/'REPORT.md',ROOT/'docs/research/track_a_h2_pr_equivalence_contract_20261023.md',
              ROOT/'docs/research/研究ノート/2026-10-23_h2_pr_equivalence_contract.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            result=(p.parent/target).resolve()
            r.require(result.is_file() or result in {OUT/'final_verification.json',OUT/'file_inventory.json'}, 'REPORT_LINK:'+target)
    forbidden={'application_sequence','product_component_ids','rotation_component_id','basis_change_operations',
               'science_master_seed','seed_hex','seed_bytes','registry_seed_PRIVATE'}
    def inspect(value):
        if isinstance(value,dict):
            r.require(not forbidden.intersection(value), 'PRIVATE_RAW_EVENT_OR_SEED_KEY')
            for x in value.values():inspect(x)
        elif isinstance(value,list):
            for x in value:inspect(x)
    for p in OUT.rglob('*'):
        if p.is_file():
            r.require(p.suffix in ('.json','.txt','.csv','.md','.py'),'NO_RAW_ARCHIVE_OR_CONFIG_FILE_COPIES')
            if p.suffix=='.json':inspect(read(p))
    stop=read(OUT/'STOP.json')
    r.require(stop['mandatory_STOP'] is True and stop['alternative_adopted'] is False and
              stop['strict_contract_changed'] is False and not stop['execution_ready'] and not stop['science_authorized'], 'STOP_AUTHORITY')
    r.require(all(stop[k]==0 for k in ('new_sampling','new_signal','new_simulation','new_H2_circuit_build','new_native_lowering',
                                     'compile','transpile','new_u_N_G','new_cost_or_ranking','scientific_imports','live_grants')), 'ZERO_SCIENTIFIC_CALLS')
    r.require(not stop['push'] and not stop['detailed_scientific_review'] and not stop['final_scientific_evidence_freeze'], 'NO_PUBLICATION_REVIEW_FREEZE')
    r.require(not Path(compare['run_root']).exists() and not Path(compare['runtime']['cache_prefix']).exists(), 'NO_COMPARE_RUN_OR_BYTECODE_CACHE')
    r.require(not any(k.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks') for k in sys.modules),'NO_SCIENTIFIC_IMPORTS')
    return dict(status='PASS',performed_at_UTC=datetime.now(timezone.utc).isoformat(),base_commit=BASE,
        audit_SOURCE=commit,verifier_sha256=r.file_digest(__file__),protected_files_rechecked=16804,protected_changes=0,
        old_scientific_SOURCE_contracts_results_grants_ledgers_markers_STOP_preserved=True,
        previous_manifest_items_unchanged=True,changed_paths=changed,
        original_config_sha256=r.file_digest(config),DIAGNOSTICS_completion_sha256=r.file_digest(Path(compare['accepted_diagnostics']['run_root'])/'completion.json'),
        static_instruction_grammar_verified_axes=134,strict_reuse_accepted_axes=0,alternative_formally_accepted_axes=0,
        A_all_conditions_fulfilled_axes=0,B_all_conditions_fulfilled_axes=0,alternative_adopted=False,
        tests=248,scientific_calls=0,new_grants=0,new_raw_payload_copies=0,science_authorized=False,
        execution_ready=False,private_event_or_seed_payloads_in_new_JSON=False,push=False,mandatory_STOP=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    target=OUT/'final_verification.json'
    r.require(not target.exists(),'NO_FINAL_REPORT_OVERWRITE')
    value=verify(args.config.resolve(strict=True))
    with target.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:value[k] for k in ('status','protected_files_rechecked','protected_changes',
                        'static_instruction_grammar_verified_axes','tests','scientific_calls','mandatory_STOP')}))
