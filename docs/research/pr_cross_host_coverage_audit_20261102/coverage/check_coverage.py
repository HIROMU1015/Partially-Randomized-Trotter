"""Validate fixed coverage metadata and saved literal references, without science calls.

Usage: python3 check_coverage.py --repo REPOSITORY --phase2-dir FIXED_INTEGRATION_DIR
The Phase 2 metadata helpers must be the files at PHASE2 below; no scientific
runner, numerical package, raw archive, QPY decoder or compiler is imported.
"""
import argparse,csv,json,pathlib,sys,hashlib,subprocess,re,collections,os
PHASE2='604f0235840feb18d368f99ea0828f370ce70fc8'
PREFIX='docs/research/pr_cross_host_coverage_audit_20261102/'
MATRIX='claim_id claim_ref claim_status claimed_domain required_evidence_link applicable_axes condition_refs canonical_evidence_groups independent_cohorts comparison_alignment coverage_state assurance_scope preservation_access historical_conflicts unresolved_lookup candidate_gap_id assessment_basis_refs'.split()
GAPS='gap_id related_claim_ref present_evidence_refs missing_link effect_on_conclusion resolvable_with_existing_evidence requires_new_science discriminating_question category rationale priority_and_reason decision_status'.split()
STATES=set('SUPPORTED_WITHIN_RECORDED_SCOPE PARTIAL_EVIDENCE NOT_EXECUTED_DOCUMENTED EVIDENCE_UNLOCATED NOT_ACCESSIBLE INVALIDATED_ONLY CONFLICTING_EVIDENCE NOT_APPLICABLE NOT_ASSESSED'.split())
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],env=ENV,stderr=subprocess.PIPE)
def blob(path):
 h=hashlib.sha1();h.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def table(path,header=None):
 with path.open(encoding='utf-8-sig',newline='') as f:
  r=csv.DictReader(f)
  if header is not None:assert r.fieldnames==header,(path.name,'column order')
  assert len(set(r.fieldnames))==len(r.fieldnames)
  out=list(r)
 assert all(None not in r and all(v!='' for v in r.values()) for r in out),path.name
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--phase2-dir',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);a=p.parse_args();o=a.out
 # Only audited stdlib metadata helpers are imported. Require exact fixed source bytes.
 for name in ['metadata_integration.py','integrate_catalog.py']:
  assert blob(a.phase2_dir/name)==git(a.repo,'rev-parse',PHASE2+':'+PREFIX+'integration/'+name).decode().strip()
 sys.path.insert(0,str(a.phase2_dir));from metadata_integration import GitLookup,PRIVATE
 from integrate_catalog import pointer_value
 m=table(o/'COVERAGE_MATRIX.csv',MATRIX);g=table(o/'GAP_CANDIDATES.csv',GAPS);scope=table(o/'CLAIM_SCOPE.csv');e=table(o/'EVIDENCE_INDEX.csv');eg=table(o/'EVIDENCE_GROUPS.csv')
 primary={r['claim_id'] for r in scope};expected={'A'+str(n).zfill(2) for n in range(1,14)}|{'R01','R02','F01'}|{'P'+str(n).zfill(2) for n in range(1,6)};assert primary==expected
 assert len(scope)==len(primary);assert {r['claim_id'] for r in m}==primary|{'HIST-UWC','HIST-DFSCREEN'}
 assert len({(r['claim_id'],r['claimed_domain'],r['required_evidence_link']) for r in m})==len(m)
 ids={r['evidence_id'] for r in e};gids={r['gap_id'] for r in g};assert len(ids)==len(e) and len(gids)==len(g)
 lookup=GitLookup(a.repo);registry={r['evidence_id']:r for r in e}
 for r in e:
  assert re.fullmatch('[0-9a-f]{40}',r['commit']) and lookup.get(r['commit'],r['repo_path'])==r['git_blob']
  assert r['reference']=='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+r['commit']+'/'+r['repo_path']
 for rows,cols in [(m,['condition_refs','assessment_basis_refs']),(g,['present_evidence_refs']),(scope,['required_evidence_refs'])]:
  for r in rows:
   for col in cols:
    for ref in json.loads(r[col]):assert ref['evidence_id'] in ids and ref['reference']==registry[ref['evidence_id']]['reference']
 for r in m:
  assert r['coverage_state'] in STATES
  for gap in json.loads(r['candidate_gap_id']):assert gap in gids or gap=='NOT_APPLICABLE'
  selection=json.loads(r['canonical_evidence_groups'])
  for eid in selection['filter_evidence_ids']:assert eid in ids and any(x['evidence_id']==eid for x in eg)
 for r in g:
  assert r['category'] in set('ABCDE')
  assert all(z['claim_id'] in primary|{'HIST-UWC','HIST-DFSCREEN'} for z in json.loads(r['related_claim_ref']))
  d=json.loads(r['decision_status']);assert d['assessment']=='PROVISIONAL_CODEX_NOT_ADOPTED' and d['detailed_review']=='NOT_STARTED'
 # Revisit fixed Phase2 metadata for selected groups only; all catalog parts are streamed.
 selected={r['canonical_id'] for r in eg};catalog={};checked_phase2_files=0
 used=['CANONICAL_CATALOG.csv','CANONICAL_CATALOG.part002.csv','CANONICAL_CATALOG.part003.csv','RECORD_LINKS.csv','RECORD_LINKS.part002.csv','RELATIONSHIPS.csv','UNRESOLVED.csv','UNRESOLVED.part002.csv','SERIES_INDEX.csv','REFERENCE_INDEX.csv','SOURCE_COMPONENTS.csv','INPUT_FILES.csv','VALIDATION.json']
 for name in used:
  assert blob(a.phase2_dir/name)==git(a.repo,'rev-parse',PHASE2+':'+PREFIX+'integration/'+name).decode().strip();checked_phase2_files+=1
 for name in used[:3]:
  with (a.phase2_dir/name).open(newline='') as f:
   for n,r in enumerate(csv.DictReader(f),1):
    if r['canonical_id'] in selected:catalog[r['canonical_id']]=(name,str(n),r)
 assert set(catalog)==selected
 keys=set();counts=collections.Counter()
 for r in eg:
  k=(r['evidence_id'],r['canonical_id']);assert k not in keys;keys.add(k);counts[r['evidence_id']]+=1
  name,n,original=catalog[r['canonical_id']];er=registry[r['evidence_id']]
  assert (name,n)==(r['phase2_catalog_file'],r['phase2_data_row'])
  assert original['result_git_blob']==er['git_blob'] and original['result_selector']==r['result_selector']
  assert original['identity_status']==r['identity_status'] and original['anchor_record_id']==r['anchor_record_id']
  assert original['result_reference'].split('/blob/',1)[1].split('/',1)[1]==er['repo_path']
  assert json.loads(r['original_anchor_states'])['validity_status']=='UNKNOWN'
  assert json.loads(r['phase2_open_metadata_categories'])
 assert all(str(counts[r['evidence_id']])==r['phase2_group_lookup_rows'] for r in e)
 anchor_tags={(r['anchor_environment'],r['anchor_record_id']) for r in eg};original_states={};original_issues={}
 statecols='run_status scientific_status technical_status acceptance_status validity_status preservation_status access_status independence_status'.split()
 for name in ['RECORD_LINKS.csv','RECORD_LINKS.part002.csv']:
  with (a.phase2_dir/name).open(newline='') as f:
   for r in csv.DictReader(f):
    tag=(r['environment'],r['record_id'])
    if tag in anchor_tags:original_states[tag]={k:r[k] for k in statecols}
 for name in ['UNRESOLVED.csv','UNRESOLVED.part002.csv']:
  with (a.phase2_dir/name).open(newline='') as f:
   for r in csv.DictReader(f):
    if r['canonical_id'] in selected:original_issues[r['canonical_id']]=json.loads(r['categories'])
 assert set(original_states)==anchor_tags and set(original_issues)==selected
 for r in eg:
  assert json.loads(r['original_anchor_states'])==original_states[(r['anchor_environment'],r['anchor_record_id'])]
  assert json.loads(r['phase2_open_metadata_categories'])==original_issues[r['canonical_id']]
 # Literal checks of saved conclusions and scope flags: no recomputation of any scientific metric.
 cache={}
 def doc(k):
  if k not in cache:
   r=registry[k];b=git(a.repo,'show',r['commit']+':'+r['repo_path']);cache[k]=json.loads(b,parse_int=str,parse_float=str)
  return cache[k]
 checks=[('COEFFJSON','/summary/h12_direct_validation_performed',False),('COEFFJSON','/summary/all_single_dominant_phase_validation_pass',False),('COEFFJSON','/summary/all_paper_d6_estimator_validation_pass',True),('M1JSON','/research_decision',None),('M1VALID','/status','M1_B1_RESULT_VALIDATED_RESEARCH_REVIEW_COMPLETE'),('M2JSON','/status','TRANSFER_SUPPORTED'),('LONGACCOUNT','/priced_candidates','642'),('LONGACCOUNT','/registered_candidates','11632'),('LONGACCOUNT','/minima_are_priced_subset_only',True),('LONGEXT','/other_formulas_q32_not_evaluated',True),('LONGSEEDS','/complete_historical_coverage_claimed',False),('LONGSEEDS','/separate_cohort',True),('B3BIND','/old_PR_vs_PF_envelope_cells','33'),('B3SUM','/B3_below_B2','0'),('B3SUM','/cells','40'),('B3DONE','/formal_paper_claim_adopted',False),('HIGHJSON','/complete_historical_seed_noncollision_guaranteed',False),('H8RARESUM','/full_population_certificate',False),('R2STATUS','/status','PROOF_INCOMPLETE'),('R2STATUS','/whole_law_rho_upper',None),('R2STATUS','/formal_certificate_accepted',False),('RPEJSON','/scope/final_total_cost_evaluation_performed',False),('RPEJSON','/scope/q_above_32_directly_validated',False),('RPEJSON','/coherent_reoptimization/comparison/scenarios/local_5_percent/intervals_overlap',True),('PXINPUT','/same_L_D_is_same_partition',False),('MANIFEST','/result_sets/1/status','invalidated_pending_regeneration')]
 for k in ['FINITEJSON','FINITE-LD0','FINITE-LD1','FINITE-LD6','FINITE-DT025','FINITE-DT05','FINITE-DT1','FINITE-DT2','FINITE-DT4']:checks.append((k,'/summary/all_signal_error_bounds_pass',True))
 for k,sel,expected_value in checks:assert pointer_value(doc(k),sel)[0]==expected_value,(k,sel)
 csvchecks=[('PA-E4',{'candidate_id':'PR_p0000_q1_r4_K2','epsilon_sig':'0.05','cohort':'confirmation'},{'G_exact':'27036763/48','N_cos':'14906','N_sin':'14903','C_cos_exact':'907/48','price_n':'384'}),('EVENTCSV',{'candidate_id':'p000_q8_r8_K4','epsilon_sig':'0.0005','cohort':'confirmation','group':'RTE'},{'G_exact':'8066682748313/64','N_cos':'193753833','N_sin':'193753845','price_n':'384'}),('H6CSV',{'candidate_id':'PR_p0078_q8_r32_K2','epsilon_sig':'0.0005','cohort':'confirmation'},{'G_exact':'63691899506611/32','N_cos':'177727546','N_sin':'247950732','C_cos_exact':'299249/64','price_n':'64'})]
 import io
 for k,match,expected_values in csvchecks:
  r=registry[k];raw=git(a.repo,'show',r['commit']+':'+r['repo_path']).decode();found=[z for z in csv.DictReader(io.StringIO(raw)) if all(z[a]==v for a,v in match.items())];assert len(found)==1
  assert all(found[0][key]==value for key,value in expected_values.items()),k
 # Publication privacy scan and relative-document links; original payloads are not copied.
 for path in o.iterdir():
  if path.is_file() and path.name!='VALIDATION.json':
   text=path.read_text();assert not PRIVATE.search(text),('private pattern',path.name)
 report=(o/'REPORT.md').read_text()
 for name in re.findall(r'\]\(([^):#]+\.(?:csv|py|json))\)',report):
  if name!='VALIDATION.json':assert (o/name).is_file(),name
 lookup.close()
 result=dict(status='PASS_PHASE3_METADATA_AND_LITERAL_CHECKS',scope='Provisional coverage metadata; no scientific reanalysis or independent scientific review',primary_claims=len(primary),coverage_rows=len(m),gap_candidates=len(g),coverage_states=dict(collections.Counter(r['coverage_state'] for r in m)),gap_categories=dict(collections.Counter(r['category'] for r in g)),fixed_public_evidence_references=len(e),phase2_fixed_files_checked=checked_phase2_files,selected_canonical_groups=len(selected),group_lookup_links=len(eg),saved_json_literal_checks=len(checks),saved_csv_literal_row_checks=len(csvchecks),original_state_tokens_not_promoted=True,scientific_calls=0,new_N_G_SE_ratios_rankings=0,formal_claim_adoption='NOT_PERFORMED',independent_scientific_review='NOT_STARTED',mandatory_stop='AFTER_PUBLICATION_AND_VERIFICATION')
 (o/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
