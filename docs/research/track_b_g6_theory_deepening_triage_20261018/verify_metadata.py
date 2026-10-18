"""Document/Git/JSON/hash checks only. Never import or execute science source.

This is packet metadata validation, not verification code for a research candidate.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE='5f17a7548b7eef1976f45f5d09bdc979e2e0a81b'
PREFIX=HERE.relative_to(ROOT).as_posix()+'/'
DOCS=['README.md','theory_gap_matrix.md','proof_and_contract_risks.md',
      'candidate_priority_decision.md','minimal_next_work_proposals.md','handoff.md']

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])

def verify():
    checks=[]
    def check(name,value):checks.append({'check':name,'pass':bool(value)})
    source=json.loads((HERE/'source_inventory.json').read_text())
    evidence=json.loads((HERE/'evidence_index.json').read_text())
    receipt=json.loads((HERE/'publication_receipt.json').read_text())
    check('base_identity',source['worktree_base_commit']==evidence['worktree_base_commit']==receipt['worktree_base_commit']==BASE)
    check('base_ancestor',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',BASE,'HEAD'],check=False).returncode==0)
    check('dedicated_branch',git('branch','--show-current').decode().strip()==evidence['branch']=='track-b-g6-theory-deepening-triage-20261018')
    check('authorized_origin',git('remote','get-url','origin').decode().strip()=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git')
    entries=source['entries'];ids={r['id'] for r in entries}
    check('27_unique_sources',len(ids)==len(entries)==27)
    for row in entries:
        ident=row['id'];blob=git('show',row['source_commit']+':'+row['path'])
        check(ident+'_fixed_commit',bool(re.fullmatch('[0-9a-f]{40}',row['source_commit'])))
        check(ident+'_git_blob',git('rev-parse',row['source_commit']+':'+row['path']).decode().strip()==row['git_blob'])
        check(ident+'_bytes_sha256',len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256'])
        check(ident+'_public_bytes_record',row['remote_bytes_equal_git_blob'] and row['remote_sha256']==row['sha256'])
        check(ident+'_reading_scope',bool(row['reading_scope']) and not row['independent_scientific_reproof_or_execution'])
        if ident.startswith('A'):
            check(ident+'_alignment_packet_ref',row['source_commit']==BASE)
        if ident.startswith('C') and ident!='CP':
            check(ident+'_construction_packet_ref',row['source_commit']=='d4163acd7b577afaa7d6db8c29db64f76ec03ae7')
        if ident.startswith('N'):
            check(ident+'_nonpauli_readonly_ref',row['source_commit']=='128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6')
        if row['path'].endswith('.py'):
            tree=ast.parse(blob.decode());decl={}
            for node in tree.body:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                    decl[node.name]=(node.lineno,node.end_lineno)
                    if isinstance(node,ast.ClassDef):
                        for child in node.body:
                            if isinstance(child,ast.FunctionDef):decl[node.name+'.'+child.name]=(child.lineno,child.end_lineno)
            for item in row['declarations']:
                check(ident+'_static_locator_'+item['name'],decl.get(item['name'])==(item['start_line'],item['end_line']))
    reuse=source['primary_literature_reuse']
    check('no_repeated_literature_search',not reuse['new_literature_search'] and reuse['new_primary_body_downloads']==0 and len(reuse['papers'])==7)
    prior=json.loads(git('show',BASE+':docs/research/track_b_g6_full_generation_prior_art_audit_20261018/literature_scope.json'))
    by_id={p['id']:p for p in prior['papers']}
    for p in reuse['papers']:
        check(p['id']+'_prior_version_provenance',all(p[k]==by_id[p['id']][k] for k in ['version','url','sha256','bytes']))
    review=source['latest_private_review_availability']
    check('missing_review_not_inferred',review==evidence['latest_review_availability'] and review['status']=='NOT_RETRIEVED' and not review['unretrieved_content_inferred'] and not review['private_originals_published'])
    gaps=evidence['gaps'];risks=evidence['risks'];candidates=evidence['candidates']
    check('12_gap_records',len(gaps)==12 and len({r['id'] for r in gaps})==12)
    check('all_T1_T6_covered',{re.match('T[1-6]',r['classification']).group() for r in gaps}=={'T1','T2','T3','T4','T5','T6'})
    check('13_risk_records',len(risks)==13 and len({r['id'] for r in risks})==13)
    for record in gaps+risks:
        check(record['id']+'_full_documented_fields',all(record['cells']) and len(record['cells'])==(5 if record['id'].startswith('TG') else 4))
    check('eight_candidates',len(candidates)==8 and {c['id'] for c in candidates}=={'C'+str(i) for i in range(1,9)})
    for candidate in candidates:
        check(candidate['id']+'_ten_decision_fields',all(candidate.get(k) for k in evidence['candidate_required_fields']))
        check(candidate['id']+'_valid_source_ids',all(i in ids for i in candidate['source_ids']))
    check('two_priority_proposals',[c['id'] for c in candidates if c['priority'] is not None]==evidence['priority_candidate_ids']==['C1','C2'])
    check('first_work_W1',evidence['first_next_work_proposal']=='W1')
    proposals=evidence['minimal_next_work_proposals']
    check('two_minimal_work_proposals',len(proposals)==2 and {p['id'] for p in proposals}=={'W1','W2'})
    for proposal in proposals:
        check(proposal['id']+'_input_owner_output_boundary',all(proposal.get(k) for k in ['candidate','owner','input','source_ids','output','go','stop','authorization']))
        check(proposal['id']+'_only_proposal',proposal['authorization'].startswith('PROPOSAL_ONLY_NO_CONTRACT_OR_GRANT'))
    check('zero_scientific_execution_proof_implementation',all(v==0 for v in evidence['execution_accounting'].values()))
    check('not_scientific_review_or_adoption',not evidence['independent_scientific_review_performed'] and not evidence['formal_research_adoption'] and evidence['technical_priority_proposal_only'])
    check('mandatory_STOP',evidence['mandatory_stop_after_publication'])
    permitted={r['url'] for r in entries}
    for name in DOCS:
        path=HERE/name;text=path.read_text()
        check(name+'_nonempty',bool(text.strip()))
        definitions=dict(re.findall(r'^\[([^\]]+)\]:\s+(https?://\S+)',text,re.M))
        for ident,url in definitions.items():check(name+'_fixed_ref_'+ident,ident in ids and url in permitted)
        for ident in set(re.findall(r'\[([A-Z][A-Z0-9]*)\]',text)) & ids:
            check(name+'_defined_source_'+ident,ident in definitions)
        for target in re.findall(r'\[[^\]]*\]\(([^)\s]+)\)',text):
            if target.startswith(('http://','https://','#')):continue
            check(name+'_local_link_'+target,(path.parent/target.split('#',1)[0]).resolve().is_file())
        check(name+'_no_missing_refs','[MISSING' not in text and 'TODO' not in text)
    changed=git('diff','--name-only',BASE).decode().splitlines()
    untracked=git('ls-files','--others','--exclude-standard').decode().splitlines()
    check('changes_only_dedicated_directory',all(p.startswith(PREFIX) for p in changed+untracked))
    protected=git('diff','--name-only',BASE,'--','src','scripts','tests','artifacts','AGENTS.md','PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/track_b_g6_full_generation_prior_art_audit_20261018','docs/research/track_b_g6_construction_paper_evidence_20261018').decode()
    check('science_old_packets_and_indexes_preserved',protected=='')
    payload=[]
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name not in ['verification.json','publication_receipt.json']:
            blob=path.read_bytes();payload.append(dict(path=path.name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    return {'schema_version':1,'kind':'NON_SCIENTIFIC_DOCUMENTARY_METADATA_CHECK','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':all(c['pass'] for c in checks),'checks':checks,'payload_manifest':payload,'self_hash_exclusions':['verification.json','publication_receipt.json'],'scientific_source_or_test_executed':False,'candidate_proof_or_validation_code':False,'mathematical_or_novelty_certificate':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-report',action='store_true');args=parser.parse_args()
    report=verify()
    if args.write_report:(HERE/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),'failed':[r['check'] for r in report['checks'] if not r['pass']]}))
    raise SystemExit(0 if report['passed'] else 1)
