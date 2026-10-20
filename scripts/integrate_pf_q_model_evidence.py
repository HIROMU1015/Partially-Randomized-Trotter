"""Documentary integration only: saved literals, hashes, public-scope and structure."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import zlib

ROOT=Path(__file__).resolve().parents[1]
BASE='3983fe4f1ede003953735e757d2c098c12b4644f'
SOURCE='c0d8ca8ecd3fc29528ca8424f8b8d953d06eff91'
B='artifacts/resource_applicability/'
RESULT=B+'track_a_h6_noncancellation_stage_b_rerun/2026-10-20/'
OLD70=B+'track_a_pf_selection_cost_comparison/2026-10-20/'
STAGE_A=B+'track_a_h6_noncancellation_stage_a/2026-10-20/'
PAPER=B+'track_a_h468_paper_evidence_integration/2026-10-19/'
DISPLAY=B+'track_a_h468_paper_display_methods/2026-10-19/'
OUT=ROOT/(B+'track_a_pf_q_model_evidence_integration/2026-10-20')
PROTOCOL='docs/research/track_a_pr_resource_evaluation_protocol_v2.md'
SUPPLEMENT='docs/research/track_a_protocol_v2_pf_cost_supplement_20261020.md'
REPORT='docs/research/track_a_pf_q_model_evidence_integration_20261020.md'
DOCUMENT_UPDATES={SUPPLEMENT,'PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md',
                  'docs/README.md','scripts/README.md','src/trotterlib/README.md','artifacts/validation_manifest.json'}
CONDITIONS='linear H6/H8;1 Angstrom;STO-3G;tol-only DF19/28;T=.8;P=0;registered prefix L_D/q;delta=.8/q;five epsilon;factors1/10;H6 domains32/96/136;H8 domains32/184'
LIMITS='CONDITIONAL_MIXED_TO_DIRECT_POLICY;OVERLAPPING_SCOPES_NOT_INDEPENDENT;H6_PROFILE_DIFFERENCE;H8_MODEL_RUNTIME_IDENTITY_UNPROVEN;S4_COMPOSITION_HYPOTHESIS;INITIAL572_FINAL_GATE_MISSING;OLD48_PENDING;FINITE_PR_N32_POINT;NO_POPULATION_OR_UNSEEN_EVENT_GUARANTEE;NO_INTRINSIC_MODEL_ONLY_ERROR;NOT_INDEPENDENT_P;NOT_FINAL_FREEZE'


def read(name):return json.loads((ROOT/name).read_bytes())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(name):
    raw=(ROOT/name).read_bytes()
    return dict(path=name,sha256=sha(raw),bytes=len(raw),fixed_commit=BASE)
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT)
def save(name,value):
    raw=value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    (OUT/name).write_bytes(raw)
def require(value,reason):
    if not value:raise ValueError(reason)
def pointer(data,path):
    for key in path.strip('/').split('/'):data=data[key]
    return data


METRICS=(
 ('scope_conditions',RESULT+'summary.json','/all100/scope_conditions'),
 ('unique_endpoints',RESULT+'summary.json','/unique_endpoints'),
 ('best_candidate_changed',RESULT+'summary.json','/all100/best_candidate_changed'),
 ('q_changed',RESULT+'summary.json','/all100/q_changed'),
 ('PF_changed',RESULT+'summary.json','/all100/PF_changed'),
 ('prefix_changed',RESULT+'summary.json','/all100/prefix_changed'),
 ('winner_total_changed',RESULT+'summary.json','/all100/winner_total_changed'),
 ('ordering_changed',RESULT+'summary.json','/all100/ordering_changed'),
 ('PR_PF_relation_changed',RESULT+'summary.json','/all100/PR_PF_relation_changed'),
 ('PR_LOWER',RESULT+'summary.json','/all100/relation_after/PR_LOWER'),
 ('PF_LOWER',RESULT+'summary.json','/all100/relation_after/PF_LOWER'),
 ('strict_pairwise_inversions',RESULT+'summary.json','/all100/strict_pairwise_inversions'),
 ('pairwise_tie_changes',RESULT+'summary.json','/all100/pairwise_tie_changes'),
 ('known_baseline_costs_replayed',RESULT+'summary.json','/new30_replay_gate/known_costs_replayed'),
 ('new_eligible_baseline_costs',RESULT+'summary.json','/new30_replay_gate/newly_constructed_eligible_baseline_costs'),
 ('previous70_scopes',RESULT+'summary.json','/previous70/scope_conditions'),
 ('new30_scopes',RESULT+'summary.json','/new30/scope_conditions'),
 ('native_exit_code',RESULT+'native_exit_evidence.json','/native_exit_code'),
 ('completion_status',RESULT+'completion_audit.json','/status'),
 ('stage_A_eligible',STAGE_A+'summary.json','/eligibility_counts/ELIGIBLE'),
 ('stage_A_ineligible',STAGE_A+'summary.json','/eligibility_counts/INELIGIBLE'),
 ('stage_A_unresolved',STAGE_A+'summary.json','/eligibility_counts/UNRESOLVED'))


def verify_saved():
    summary=read(RESULT+'summary.json')
    expected=dict(scope_conditions=100,best_candidate_changed=2,q_changed=0,PF_changed=0,
        winner_total_changed=50,ordering_changed=50,PR_PF_relation_changed=0)
    require(all(summary['all100'][k]==v for k,v in expected.items()),'ACCEPTED_SUMMARY_LITERAL_MISMATCH')
    require(summary['unique_endpoints']==40 and summary['all100']['relation_after']==dict(PR_LOWER=90,PF_LOWER=10),'ACCEPTED_DIRECTIONS')
    require(summary['previous70']==read(OLD70+'summary.json')['all_scopes'],'OLD70_SUMMARY_LITERAL')
    combined=read(RESULT+'combined_selection100.json')['rows']
    old=read(OLD70+'selection_comparison.json')['rows']
    require(len(combined)==100 and len(old)==70,'SAVED_ROW_COUNTS')
    require([{k:v for k,v in r.items() if k!='numeric_origin'} for r in combined[:70]]==old,'OLD70_VERBATIM')
    require(combined[70:]==read(RESULT+'selection_comparison30.json')['rows'],'NEW30_VERBATIM')
    require(read(STAGE_A+'summary.json')['eligibility_counts']==dict(ELIGIBLE=34,INELIGIBLE=86,UNRESOLVED=0),'ACCEPTED_STAGE_A')
    require(read(RESULT+'completion_audit.json')['status']=='PASS_STAGE_B30_WITH_VERBATIM70_COMPLETE','COMPLETION_RECEIPT')
    native=read(RESULT+'native_exit_evidence.json')
    require(native['native_exit_code']==0 and native['reaped'] and native['live_owned_workers']==0,'NATIVE_RECEIPT')
    require(read(RESULT+'STOP.json')['mandatory_STOP'] is True,'ORIGINAL_STOP')
    for name,value in read(RESULT+'numeric_completion.json')['result_sha256'].items():
        require(ref(RESULT+name)['sha256']==value,'NUMERIC_FILE_SHA:'+name)
    for name,value in read(RESULT+'sealed_contract.json')['source_hashes'].items():
        require(sha(git('show',SOURCE+':'+name))==value and ref(name)['sha256']==value,'FIXED_SOURCE:'+name)
    return summary


SECRET_PATTERNS=(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'\bghp_[A-Za-z0-9]{30,}\b',rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
    rb'\bAKIA[0-9A-Z]{16}\b',rb'\bsk-[A-Za-z0-9_-]{30,}\b',
    rb'https?://[^\s/:]+:[^\s/@]+@')
PATH_PATTERN=re.compile(rb'/(?:home|Users|tmp)/[^\s\x00<>"\x27`;,\)\}\]]+')


def scan_layers(raw,suffix):
    layers=[('raw',raw)]
    if suffix=='.pdf':
        for i,match in enumerate(re.finditer(rb'stream\r?\n(.*?)\r?\nendstream',raw,re.S)):
            try:decoded=zlib.decompress(match[1])
            except zlib.error:continue
            layers.append(('pdf_FlateDecode_stream_'+str(i),decoded))
    if suffix=='.png':
        pos=8
        while pos+12<=len(raw):
            size=struct.unpack('>I',raw[pos:pos+4])[0];kind=raw[pos+4:pos+8];payload=raw[pos+8:pos+8+size];pos+=size+12
            if kind in (b'tEXt',b'iTXt'):layers.append(('png_'+kind.decode(),payload))
            elif kind==b'zTXt':
                try:layers.append(('png_zTXt',zlib.decompress(payload.split(b'\0',1)[1][1:])))
                except (IndexError,zlib.error):pass
    return layers


def scan_content(raw,suffix):
    paths=[];credentials=[];layers=scan_layers(raw,suffix)
    for layer,data in layers:
        for match in PATH_PATTERN.finditer(data):
            paths.append(dict(layer=layer,line=data[:match.start()].count(b'\n')+1,
                category='USER_HOME_ABSOLUTE' if match[0].startswith((b'/home/',b'/Users/')) else 'TEMP_ABSOLUTE',
                literal_sha256=sha(match[0])))
        for i,pattern in enumerate(SECRET_PATTERNS):
            for match in re.finditer(pattern,data):
                credentials.append(dict(layer=layer,pattern_index=i,line=data[:match.start()].count(b'\n')+1,literal_sha256=sha(match[0])))
    return dict(internal_paths=paths,credential_patterns=credentials,inspected_layers=[x[0] for x in layers])


def publication_audit():
    scope=read(RESULT+'publication_scope.json');remote=read(str(OUT.relative_to(ROOT))+'/github_observation.json')
    require(len(scope['paths'])==58 and set(scope['paths'])==set(remote['scope_file_blobs']),'EXACT58_REMOTE_SCOPE')
    require(remote['visibility']=='public' and remote['observed_branch_commit']==BASE==remote['result_commit'],'FIXED_PUBLIC_COMMIT')
    rows=[];findings=[]
    for name in scope['paths']:
        raw=git('show',BASE+':'+name)
        if name not in DOCUMENT_UPDATES:require(raw==(ROOT/name).read_bytes(),'BASE_FILE_CHANGED:'+name)
        entry=remote['scope_file_blobs'][name]
        require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['blob_sha'],'PUBLIC_BLOB_MISMATCH:'+name)
        require(len(raw)==entry['bytes'],'PUBLIC_SIZE:'+name)
        if name in scope['payload_sha256']:require(sha(raw)==scope['payload_sha256'][name],'SAVED_SCOPE_SHA:'+name)
        scan=scan_content(raw,Path(name).suffix)
        rows.append(dict(path=name,public_blob_sha=entry['blob_sha'],sha256=sha(raw),bytes=len(raw),
            public_fixed_bytes_match=True,internal_path_occurrences=len(scan['internal_paths']),
            credential_pattern_occurrences=len(scan['credential_patterns']),inspected_layers=scan['inspected_layers']))
        findings.extend(dict(path=name,kind='INTERNAL_ABSOLUTE_PATH',**f) for f in scan['internal_paths'])
        findings.extend(dict(path=name,kind='CREDENTIAL_PATTERN',**f) for f in scan['credential_patterns'])
    for name,entry in remote['extra_source_blobs'].items():
        raw=(ROOT/name).read_bytes();require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['blob_sha'],'EXTRA_PUBLIC_SOURCE:'+name)
    for d in remote['direct_files']:require(d['content'].encode()==(ROOT/d['path']).read_bytes(),'DIRECT_RAW_MISMATCH:'+d['path'])
    secret_hits=[f for f in findings if f['kind']=='CREDENTIAL_PATTERN']
    save('publication_findings58.json',dict(findings=findings,matched_literals_not_republished=True))
    audit=dict(status='PUBLIC58_BYTES_MATCH_APPROVAL_RECORD_UNVERIFIED_INTERNAL_PATH_REVIEW',
        fixed_result_commit=BASE,public=True,public_paths=58,public_blob_matches=58,extra_public_source_matches=len(remote['extra_source_blobs']),direct_raw_matches=3,
        original_scope_receipt_authorized=scope['publication_authorized'],
        explicit_58_path_publication_approval_found_in_available_records=False,
        available_record_scope='Current chat supplied instructions/command request plus fixed58 files; no broad chat-history or private-account search',
        push_command_request_observed=True,push_actor_or_separate_manual_approval_not_verified=True,
        no_inference_of_unauthorized_user_action=True,
        scientific_execution_approval_is_not_publication_approval=True,
        internal_path_files=sum(r['internal_path_occurrences']>0 for r in rows),
        internal_path_occurrences=sum(r['internal_path_occurrences'] for r in rows),
        credential_pattern_findings=len(secret_hits),
        inspection='All58 raw bytes/text; PDF successfully decoded Flate streams; PNG text chunks; no rendering or scientific calculations',
        inspection_limit='Pattern audit, not an exhaustive credential/security guarantee; font-encoded PDF text and raster pixels are not OCR-extracted',
        sensitive_literal_values_republished=False,history_changed=False,deleted=False,force_push=False,re_push=False,
        new_publication_authorized=False,
        required_user_decision='Confirm existing58 publication provenance and whether internal path disclosure may remain; decide any corrective action separately',
        rows=rows)
    save('publication_audit58.json',audit)
    require(not secret_hits,'CREDENTIAL_PATTERN_FOUND_STOP_NO_PUBLICATION')
    return audit


def append_csv(old_path,name,records):
    raw=(ROOT/old_path).read_bytes();fields=next(csv.reader(io.StringIO(raw.decode())))
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writerows(records)
    save(name,raw+stream.getvalue().encode())


CLAIMS=(
 ('SCOPE','100 overlapping scopes represent40 endpoints; fixed domains only','scope_conditions'),
 ('WINNERS','Best PF changes2/100; q and PF order change0/100; prefix changes2/100','best_candidate_changed'),
 ('COST','Minimum PF total changes50/100 under saved unit-price substitution','winner_total_changed'),
 ('RANKS','Full candidate ordering changes50/100; strict pair inversions180; pair tie changes16','ordering_changed'),
 ('DIRECTION','No PR/PF point-cost direction reversal; PR lower90, PF lower10 under both policies','PR_PF_relation_changed'),
 ('LINEAGE','Old70 copied; new30 use new34 eligible stageA costs plus exact replay of712 known costs','new_eligible_baseline_costs'),
 ('LIMITS','Mixed-model/measured to direct-price conditional diagnostic; not intrinsic model-only error or population superiority','completion_status'))


def build():
    summary=verify_saved();audit=publication_audit()
    adoption=dict(status='FORMALLY_ADOPTED_CONDITIONAL_RESEARCH_SUPPLEMENT',result_commit=BASE,
        scientific_SOURCE=SOURCE,scope_conditions=100,unique_endpoints=40,
        user_authorization='Current user explicitly accepts100 overlapping scopes/40 endpoints as conditional research results and requests documentary evidence integration only',
        scientific_conditions_changed=False,new_scientific_calls=0,new_numeric_analysis=0,new_ranking=0,
        old70_reanalysis=False,stage_A_recalculated=False,new_publication_authorized=False,
        detailed_scientific_review=False,final_evidence_freeze=False,mandatory_STOP=True)
    save('adoption_record.json',adoption)
    metrics=[dict(metric=k,value_literal=str(pointer(read(path),ptr)),evidence=ref(path),json_pointer=ptr)
             for k,path,ptr in METRICS]
    save('adopted_metrics.json',dict(status='SAVED_LITERAL_TRANSCRIPTION_ONLY',rows=metrics))
    source_map={'previous70':'19f2c5a9409db69fd8bd6500eb4228fefa00145c','stage_A':'85e250302d15b427731a50ddbe321fc280ec5f1a','new30':SOURCE}
    claims=[]
    for suffix,statement,metric in CLAIMS:
        m=next(r for r in metrics if r['metric']==metric)
        claims.append(dict(id='CE_PF_PRICE100_'+suffix,claim_candidate=statement,fixed_commits=BASE,
            result_files=m['evidence']['path'],SOURCE=';'.join(source_map.values()),conditions=CONDITIONS,
            existing_value_quote='saved '+m['json_pointer']+'='+m['value_literal'],
            measured_or_model='HISTORICAL_MIXED_POLICY_TO_SAVED_DIRECT_PRICES',search_or_confirmation='SAVED_UNION8_UNION6_N32;NOT_INDEPENDENT_P',
            unresolved=LIMITS,evidence_strength='FORMALLY_ADOPTED_CONDITIONAL_SAVED_RESULTS_ONLY',
            audit_links=RESULT+'completion_audit.json;'+RESULT+'numeric_validation.json;'+RESULT+'native_exit_evidence.json'))
    append_csv(PAPER+'claim_evidence_matrix.csv','claim_evidence_matrix.csv',claims)
    ledger=[]
    for i,m in enumerate(metrics,1):
        path=m['evidence']['path'];src=source_map['stage_A'] if path.startswith(STAGE_A) else SOURCE
        ledger.append(dict(evidence_row_id='PFC100_'+str(i).zfill(2),study_id='PF_PRICE100_CONDITIONAL',metric=m['metric'],
            value_literal=m['value_literal'],unit='saved_literal',comparison_scope=CONDITIONS,
            baseline_denominator='fixed scope set; new34 baseline costs have new stageA lineage',cost_evidence='SAVED_MIXED_TO_DIRECT_POLICY',
            scientific_guarantee=LIMITS,evidence_id='PF_PRICE100',source_line='',source_excerpt='JSON pointer literal; not recomputed',
            source_field=m['json_pointer'],SOURCE_commit=src,result_commit=BASE,evidence_path=path,
            evidence_file_sha256=m['evidence']['sha256'],reported_precision='exact saved literal; no new arithmetic'))
    append_csv(PAPER+'evidence_ledger_v3.csv','evidence_ledger_v4.csv',ledger)
    save('claim_evidence_current_overlay.json',dict(status='ADOPTED_CONDITIONAL_SCOPE_UPDATE_NOT_REVIEW_OR_FREEZE',
        inherited_matrix=ref(PAPER+'claim_evidence_matrix.csv'),inherited_overlay=ref(PAPER+'claim_evidence_matrix_current_overlay.json'),
        inherited_rows_preserved=14,new_claim_ids=[r['id'] for r in claims],
        historical_CE5='The old missing H6 noncancellation120 endpoint/30scope status is historical; stageA and new stageB resolve it with new lineage. Do not retroactively validate old D costs or winners.',
        other_old_claims_not_promoted=True,remaining_limits=LIMITS))
    data_files=[RESULT+n for n in ('summary.json','selection_comparison30.json','selection_comparison30.csv',
        'combined_selection100.json','combined_selection100.csv','candidate_totals.json','candidate_totals.csv',
        'candidate_ranks.json','candidate_ranks.csv','rank_relation_changes.json','rank_relation_changes.csv',
        'conditions100.md','sealed_contract.json','numeric_validation.json','completion_audit.json','native_exit_evidence.json',
        'ledger.json','STOP.json','result_manifest.json','figure_data.json','rendering_completion.json')]
    figures=[RESULT+n+'.'+ext for n in ('PF_cost_change100','PR_PF_ratios100') for ext in ('png','svg','pdf')]
    manifest=dict(schema='PF_PRICE100_PAPER_SUPPLEMENT_V1',status=adoption['status'],base_commit=BASE,
        previous_paper_manifest=ref(PAPER+'paper_manifest.json'),previous_display_manifest=ref(DISPLAY+'publication_manifest.json'),
        inherited_paper_data_manifest=ref(PAPER+'paper_data_manifest.csv'),paper_data_rows_unchanged=1576,
        inherited_CEM_rows=14,CEM_rows=21,inherited_ledger_rows=316,evidence_ledger_rows=338,
        CEM='claim_evidence_matrix.csv',evidence_ledger='evidence_ledger_v4.csv',current_claim_overlay='claim_evidence_current_overlay.json',
        accepted_summary=summary,scientific_SOURCE_by_origin=source_map,
        new30_SOURCE_file_hashes=read(RESULT+'sealed_contract.json')['source_hashes'],
        result_commit_by_origin=dict(previous70='85433c48fd4c37d4cb4594391da422558ac3ce7f',
            stage_A='c53dd16ae16413182c9d3080f82f80237231a248',new30=BASE),
        stage_A_34_costs='new baseline provenance; historical full-domain winner null; not a saved-winner replay',
        inherited_main_table=read(DISPLAY+'publication_manifest.json')['main_table'],
        inherited_Figure2=read(DISPLAY+'publication_manifest.json')['figure2'],
        protocol=ref(PROTOCOL),protocol_scientific_conditions_unchanged=True,supplement=SUPPLEMENT,
        current_supplement_binding=dict(path=SUPPLEMENT,sha256=sha((ROOT/SUPPLEMENT).read_bytes()),commit_role='CURRENT_DOCUMENT_INTEGRATION'),
        prior_supplement_snapshot=dict(path=SUPPLEMENT,commit=BASE,sha256=sha(git('show',BASE+':'+SUPPLEMENT))),
        evidence=[ref(p) for p in data_files]+[ref(STAGE_A+'summary.json'),ref(STAGE_A+'endpoints.json'),ref(OLD70+'summary.json'),ref(OLD70+'selection_comparison.json')],
        figures=[ref(p) for p in figures],limitations=LIMITS,
        original_science_packets_unchanged=True,main_table_and_Figure2_changed=False,
        new_science=0,new_numerical_analysis=0,independent_confirmation_added=False,
        publication_audit='publication_audit58.json',publication_approval_verified=False,
        new_publication_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False)
    save('paper_manifest.json',manifest)
    save('protocol_v2_supplement_manifest.json',dict(status=adoption['status'],protocol=ref(PROTOCOL),
        supplement=SUPPLEMENT,supplement_sha256=sha((ROOT/SUPPLEMENT).read_bytes()),accepted_result_commit=BASE,source_by_origin=source_map,
        scope_conditions=100,unique_endpoints=40,new34_baseline_lineage_explicit=True,
        supersedes_scope70_document_at_commit=BASE,prior_science_and_protocol_unchanged=True,
        paper_manifest='paper_manifest.json',new_numeric_analysis=False,new_publication_authorized=False))
    print(json.dumps(dict(status='BUILT_DOCUMENTARY_INTEGRATION',CEM_rows=21,ledger_rows=338,public58_matched=True,
                         internal_path_files=audit['internal_path_files'],internal_path_occurrences=audit['internal_path_occurrences'],credential_patterns=audit['credential_pattern_findings'])))


def check():
    verify_saved();manifest=read(str(OUT.relative_to(ROOT))+'/paper_manifest.json')
    for item in manifest['evidence']+manifest['figures']+[manifest['protocol']]:
        require(ref(item['path'])['sha256']==item['sha256'],'EVIDENCE_SHA:'+item['path'])
    for previous,current,expected in ((PAPER+'claim_evidence_matrix.csv','claim_evidence_matrix.csv',21),
                                      (PAPER+'evidence_ledger_v3.csv','evidence_ledger_v4.csv',338)):
        raw=(OUT/current).read_bytes();require(raw.startswith((ROOT/previous).read_bytes()),'HISTORICAL_CSV_PREFIX')
        require(len(list(csv.DictReader(io.StringIO(raw.decode()))))==expected,'CSV_ROW_COUNT')
    metrics=read(str(OUT.relative_to(ROOT))+'/adopted_metrics.json')['rows']
    for m in metrics:require(m['value_literal']==str(pointer(read(m['evidence']['path']),m['json_pointer'])),'TRANSCRIPTION_ONLY')
    supplement=(ROOT/SUPPLEMENT).read_text()
    require('100 scope' in supplement and '34' in supplement and '50/100' in supplement,'SUPPLEMENT_UPDATED')
    require('未解決のまま' not in supplement and '30 unresolved' not in supplement,'NO_STALE_CURRENT_EXCLUSION')
    require((ROOT/PROTOCOL).read_bytes()==git('show',BASE+':'+PROTOCOL),'FORMAL_PROTOCOL_CHANGED')
    for f in (OUT/'README.md',ROOT/SUPPLEMENT,ROOT/REPORT):
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if link.startswith(('http:','https:','#')):continue
            require((f.parent/link.split('#')[0]).exists(),'DOCUMENT_LINK:'+link)
    require(not any(p.name=='__pycache__' for p in OUT.iterdir()),'UNEXPECTED_CACHE')
    receipt=dict(status='PASS_DOCUMENTARY_LITERAL_HASH_STRUCTURE_ONLY',base_commit=BASE,
        source_results_protocol_preserved=True,old_CEM14_ledger316_prefix_preserved=True,
        CEM_rows=21,evidence_ledger_rows=338,new_scientific_calls=0,new_numeric_analysis=0,new_ranking=0,
        old70_reanalysis=False,stage_A_recalculated=False,new_figures=False,
        independent_reviewer=False,new_publication_authorized=False,detailed_scientific_review=False,
        final_evidence_freeze=False,mandatory_STOP=True)
    save('structure_validation.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('build','check'));args=parser.parse_args()
    if args.action=='build':build()
    else:check()
