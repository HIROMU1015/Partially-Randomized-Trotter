"""Document/metadata checks only. No scientific imports, calculations or dispatch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BUNDLE='artifacts/resource_applicability/track_a_protocol_v1_draft/2026-10-19'
DOCS=('docs/research/track_a_pr_resource_evaluation_protocol_v1_DRAFT.md',
      'track_a_protocol_gap_and_exception_report.md','track_a_protocol_adoption_decisions.md',
      'docs/research/track_a_pr_resource_protocol_draft_preparation_20261019.md')
FIXED={'4e8ad1936363266d1b55bb689c36846e6c3dfcdb','8a18b277663bf2f8998f79e9b8c0a9e61d2b184c',
       '97d25cc431516d720ce91c7cb4f148e061d8ea34','64c85f7a833b5950ef2ccae623708b73081edcd6',
       'e6eea5387a6e84d1e39a9adb3b61975acc74e200','2bf5d83293b3a47f672805fae956ff2e50df2878',
       '0edd62403c35c5ee18b2a2a152158faca5c2ebd8','bf8c19057fd82c900c6234152e152e79cb52e707',
       '0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce'}
STATUSES={'COMPLETE','PARTIAL','PENDING','NOT_VERIFIED','NOT_APPLICABLE'}


def require(value,reason):
    if not value:raise ValueError(reason)


def validate(root,*,verify_git=True):
    root=Path(root);bundle=root/BUNDLE
    for name in DOCS:
        text=(root/name).read_text()
        require('DRAFT' in text and 'ユーザー' in text,'DRAFT_AND_NEXT_USER_REQUIRED:'+name)
    protocol=(root/DOCS[0]).read_text()
    for term in ('Core','Extended','Guarantee','symmetric-directional','MODEL_ESTIMATE','UNDETERMINED',
                 'POINT_NOT_REPRODUCED','R_G=Σ_a','自動進行しない','母平均','正式採択'):
        require(term in protocol,'PROTOCOL_TERM:'+term)
    catalog=json.loads((bundle/'evidence_catalog.json').read_text())
    require(catalog['protocol_status']=='DRAFT_NOT_ADOPTED','CATALOG_NOT_ADOPTED')
    ev={x['evidence_id']:x for x in catalog['entries']}
    require(len(ev)==len(catalog['entries']),'DUPLICATE_EVIDENCE_ID')
    require(FIXED<={x['commit'] for x in ev.values()},'NINE_REQUESTED_COMMITS_REQUIRED')
    for x in ev.values():
        require(re.fullmatch('[0-9a-f]{40}',x['commit']) and re.fullmatch('[0-9a-f]{64}',x['sha256']),'FIXED_EVIDENCE_HASH')
        p=PurePosixPath(x['path'])
        require(not p.is_absolute() and '..' not in p.parts,'EVIDENCE_REPOSITORY_PATH')
        require(x['retrieved'] is True and x['scientific_module_executed'] is False,'RETRIEVAL_IS_NOT_EXECUTION')
        if verify_git:
            value=subprocess.check_output(['git','show',x['commit']+':'+x['path']],cwd=root)
            require(hashlib.sha256(value).hexdigest()==x['sha256'] and len(value)==x['bytes'],'FIXED_GIT_OBJECT_BYTES')
    spec=json.loads((bundle/'requirements.json').read_text())
    ids={x['id']:x['tier'] for x in spec['requirements']};series=set(spec['series'])
    require(spec['protocol_status']=='DRAFT_NOT_ADOPTED' and spec['scientific_status_aggregation_performed'] is False,'NO_AUTOMATIC_SCIENTIFIC_ADOPTION')
    with (root/'track_a_protocol_compliance_matrix.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    keys={(r['series_id'],r['requirement_id']) for r in rows}
    require(len(keys)==len(rows) and keys=={(s,q) for s in series for q in ids},'MATRIX_CARTESIAN_METADATA_COVERAGE')
    for r in rows:
        require(r['protocol_status']=='DRAFT_NOT_ADOPTED' and r['scientific_adoption']=='PENDING_USER_DECISION'
                and r['new_science_authorized']=='FALSE','DRAFT_NO_SCIENCE_AUTHORIZATION')
        require(r['status'] in STATUSES and r['tier']==ids[r['requirement_id']],'STATUS_TIER')
        refs=r['evidence_ids'].split(';')
        require(refs and all(x in ev for x in refs),'UNRESOLVED_EVIDENCE_REFERENCE')
        require(set(r['evidence_commits'].split(';'))=={ev[x]['commit'] for x in refs},'MATRIX_COMMIT_BINDING')
        require(r['assessed_scope'] and r['cohort'] and r['unresolved_or_limit'] and r['minimal_next_action'],'SCOPE_COHORT_LIMIT_REQUIRED')
        if r['status']=='COMPLETE':require('NOT_VERIFIED_FOR_PENDING' not in r['reported_SOURCES'],'PENDING_SOURCE_CANNOT_BE_COMPLETE')
        if r['requirement_id']=='P_FIXED' and r['status']=='COMPLETE':
            require('FIXED_INDEPENDENT' in r['cohort'],'INDEPENDENT_CONFIRMATION_NOT_UNION_SELECTION')
            require(any(x in refs for x in ('E_H4P','E_H8P')),'FIXED_P_RESULT_REQUIRED')
        if r['series_id'] in ('H6_TOL_ONLY_DF19','H8_TOL_ONLY_DF28'):
            require('MODEL' in r['cost_evidence'],'PF_MODEL_CANNOT_BE_RELABELLED_DIRECT')
        if r['requirement_id']=='C_DF' and r['series_id']=='H4_LEGACY_DF12':
            require(r['target_relation']=='DIFFERS' and r['DF_policy']=='LEGACY_FIXED_RANK','LEGACY_CANNOT_BECOME_TOL_ONLY')
    return dict(status='PASS',validation='DOCUMENT_METADATA_STRUCTURE_ONLY',series=len(series),
                requirement_rows=len(rows),fixed_evidence_files=len(ev),
                real_scientific_values_recomputed=False,scientific_adoption=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(validate(a.root),ensure_ascii=False))


if __name__=='__main__':main()
