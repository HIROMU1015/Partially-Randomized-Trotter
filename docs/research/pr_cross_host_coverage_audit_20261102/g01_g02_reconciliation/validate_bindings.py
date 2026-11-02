"""Validate G01/G02 identities and saved literals; no scientific runners or RNG calls.

Run with --repo REPOSITORY --phase2-dir FIXED_INTEGRATION_DIRECTORY.
Only exact Phase2 stdlib metadata helpers are imported. No means, N/G/SE,
scientific rankings, seed values, arrays, QPY or raw events are evaluated.
"""
import argparse,csv,json,gzip,io,pathlib,subprocess,hashlib,os,re,collections,sys
P2='604f0235840feb18d368f99ea0828f370ce70fc8'
BASE='docs/research/pr_cross_host_coverage_audit_20261102/'
OLD='155529e20a8fac39e00786c40b57830b71203c4f'
NEW='bb97a93d65992fdb53e642dde1584dd7d5b811d7'
SOURCE='cab73373593ba7c113c1af018b1dc3077c8e78fa'
SEVEN='analysis/h46_seven_cost_readiness_20261027/'
LONG='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
IDS=['d748ee85d02193720b34','205f25b8cb63949b4b18','53acfe65c49018ff2506','6eb81b93d048deeeb63f','54510520e99d85af4690','cf094355c78d50ae45f7','45f9fc0a53c1a37c7345']
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],env=ENV,stderr=subprocess.PIPE)
def blob(path):
 h=hashlib.sha1(b'blob '+str(path.stat().st_size).encode()+b'\0')
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def table(path):
 with path.open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f);rows=list(reader);assert len(set(reader.fieldnames))==len(reader.fieldnames)
 assert all(None not in r and all(v!='' for v in r.values()) for r in rows)
 for r in rows:
  for k,v in list(r.items()):
   if v.startswith(('{','[')):r[k]=json.loads(v)
 return rows
def walk(value):
 yield value
 if isinstance(value,dict):
  for v in value.values():yield from walk(v)
 elif isinstance(value,list):
  for v in value:yield from walk(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--phase2-dir',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent);a=p.parse_args();o=a.out
 for f in ['metadata_integration.py','integrate_catalog.py']:
  assert blob(a.phase2_dir/f)==git(a.repo,'rev-parse',P2+':'+BASE+'integration/'+f).decode().strip()
 sys.path.insert(0,str(a.phase2_dir));from metadata_integration import GitLookup,PRIVATE,numeric_metadata
 from integrate_catalog import pointer_value
 g1=table(o/'G01_CANDIDATE_BINDINGS.csv');g2=table(o/'G02_COHORT_BINDINGS.csv');assert [r['candidate_id'] for r in g1]==IDS and len({r['binding_id'] for r in g2})==len(g2)
 assert all(r['match_status']=='MATCHED' for r in g1)
 lookup=GitLookup(a.repo);cache={};jcache={};ccache={};refs={};literal_checks=0;pointer_checks=0
 def raw(s,p):
  key=(s,p)
  if key not in cache:cache[key]=git(a.repo,'show',s+':'+p)
  return cache[key]
 def obj(s,p):
  key=(s,p)
  if key not in jcache:
   b=raw(s,p);b=gzip.decompress(b) if p.endswith('.gz') else b;jcache[key]=json.loads(b,parse_int=str,parse_float=str)
  return jcache[key]
 def csvs(s,p):
  key=(s,p)
  if key not in ccache:ccache[key]=list(csv.DictReader(io.StringIO(raw(s,p).decode('utf-8-sig'))))
  return ccache[key]
 def parse_ref(url):
  m=re.fullmatch(r'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/([0-9a-f]{40})/(.+)',url);assert m,url
  return m.groups()
 for v in walk(g1+g2):
  if isinstance(v,dict) and 'reference' in v and 'git_blob' in v:
   s,p=parse_ref(v['reference']);assert lookup.get(s,p)==v['git_blob'];refs[(s,p)]=v['git_blob']
   # Resolve literal RFC6901 pointers and concrete CSV data rows. Human filter selectors stay explicit.
   selector=v.get('selector','')
   if '.json' in p:
    for sel in selector.split(';'):
     if sel.startswith('/') and not re.search(r'[\[\]*()= ]',sel):pointer_value(obj(s,p),sel);pointer_checks+=1
   if p.endswith('.csv'):
    m=re.match(r'CSV_DATA_ROW=([1-9][0-9]*)',selector)
    if m:assert int(m[1])<=len(csvs(s,p));pointer_checks+=1
 plan=obj(NEW,SEVEN+'PLAN.json');provenance=obj(NEW,SEVEN+'results/execution_provenance.json');audit=obj(NEW,SEVEN+'results/END_AUDIT.json');summary=obj(NEW,SEVEN+'results/extended_point_summary.json')
 assert raw(NEW,SEVEN+'PLAN.json')==raw(SOURCE,SEVEN+'PLAN.json')
 assert hashlib.sha256(raw(NEW,SEVEN+'PLAN.json')).hexdigest()==provenance['plan_sha256']
 assert provenance['SOURCE']==SOURCE and summary['original_saved_count']=='33' and summary['extended_exploration_point_count']=='33' and summary['total_cells']=='40' and summary['independent_confirmation_pooled'] is False
 assert audit['controller_exit_code'] is None and plan['seed_audit']['complete_historical_seed_coverage'] is False
 assert audit['seed_sequence_sha256']==plan['seed_audit']['new_seed_sequence_sha256'];literal_checks+=11
 samples=csvs(NEW,SEVEN+'sample_plan.csv');costs=csvs(NEW,SEVEN+'results/candidate_costs.csv');assert len(samples)==7 and len(costs)==38
 ex=obj(NEW,SEVEN+'results/SEVEN_20261027_EXPLORATION_costs.json.gz');cf=obj(NEW,SEVEN+'results/SEVEN_20261027_CONFIRMATION_costs.json.gz');original=csvs(OLD,LONG+'candidate_precision_accounting.csv');orig_by_id=collections.defaultdict(list)
 for n,r in enumerate(original,1):
  if r['id'] in IDS:orig_by_id[r['id']].append((n,r))
 endpoints=0
 for row in g1:
  cid=row['candidate_id'];item=next(v for v in plan['candidates'] if v['candidate']['id']==cid);candidate=item['candidate']
  for k in ['system','T','formula','prefix','q','R_outer','K','delta']:assert row[k]==candidate[k]
  assert len(orig_by_id[cid])==5 and all(r['native_cost_status']=='NOT_ACQUIRED' and r['G'] in ['','None','null'] for _,r in orig_by_id[cid])
  assert row['measurement_source']==SOURCE and row['result_commit']==NEW
  assert row['input_binding']['binding']==plan['systems'][row['system']]['binding']==obj(OLD,LONG+'scientific_provenance.json')['inputs'][row['system']]['binding'];literal_checks+=1
  assert row['phase2_catalog_binding']['new_measurement_registration_state'].startswith('NOT_REGISTERED_IN_FIXED_PHASE2')
  for ep in row['endpoint_price_G_SE_refs']:
   endpoints+=1;orig=next(r for _,r in orig_by_id[cid] if numeric_metadata(r['epsilon'])==numeric_metadata(ep['epsilon']))
   assert ep['N_cos']==orig['N_cos'] and ep['N_sin']==orig['N_sin'];literal_checks+=2
   saved_ep=next(v for v in item['saved_endpoints'] if numeric_metadata(v['epsilon'])==numeric_metadata(ep['epsilon']));assert saved_ep['log_B']==orig['log_B'] and saved_ep['u']==orig['u'];literal_checks+=2
   for ref in ep['measurement_rows']:
    n=int(re.match(r'CSV_DATA_ROW=(\d+)',ref['selector'])[1]);saved=costs[n-1]
    assert saved['id']==cid and saved['N_cos']==ep['N_cos'] and saved['N_sin']==ep['N_sin']
    assert numeric_metadata(saved['epsilon'])==numeric_metadata(ep['epsilon']);literal_checks+=3
  for role,groups in [('exploration_cohort',ex),('confirmation_cohort',cf)]:
   group=groups[cid];m=row[role];indices=[];names=set()
   for path in group['trial_files']:
    parts=pathlib.PurePosixPath(path).parts;assert parts[-3]==cid;indices.append(int(parts[-2]));names.add(parts[-4])
   assert sorted(indices)==list(range(int(group['n']))) and names=={group['cohort']}
   assert m['declared_cohort']==group['cohort'] and m['declared_n']==group['n'];literal_checks+=2
 assert endpoints==19
 allowed={'DOCUMENTED_DISJOINT','RECORDED_SEPARATE','REUSED_OR_OVERLAPPING','UNKNOWN','CONFLICTING'}
 for r in g2:
  assert r['claim_id'] in ['A08','A09','A11','A12'] and r['independence_class'] in allowed
  if r['research_series'] in ['LONG_TIME_B2','DF_NATIVE_B3','SEVEN_ADDITIONAL_PR_CANDIDATES']:
   e,c=r['exploration_cohort'],r['confirmation_cohort'];assert e['indices_match_zero_based_plan'] and c['indices_match_zero_based_plan'] and e['declared_cohort']!=c['declared_cohort']
  if r['relationship_role']=='COPY_OR_CITATION_ALIAS':assert r['independence_class']=='REUSED_OR_OVERLAPPING'
 # Catalog/index and original observation links remain exact fixed metadata, not new experiment IDs.
 entries={}
 for r in g1:
  for e in r['phase2_catalog_binding']['original_condition_or_result_links']+r['phase2_catalog_binding']['new_measurement_links']:entries[e['canonical_id']]=e
 for r in g2:
  for e in r['phase2_catalog_binding']['links']:entries[e['canonical_id']]=e
 verified=set();membercheck=collections.defaultdict(set)
 names=['CANONICAL_CATALOG.csv','CANONICAL_CATALOG.part002.csv','CANONICAL_CATALOG.part003.csv','RECORD_LINKS.csv','RECORD_LINKS.part002.csv']
 for name in names:assert blob(a.phase2_dir/name)==git(a.repo,'rev-parse',P2+':'+BASE+'integration/'+name).decode().strip()
 for name in names[:3]:
  with (a.phase2_dir/name).open(newline='') as f:
   for n,r in enumerate(csv.DictReader(f),1):
    if r['canonical_id'] in entries:
     e=entries[r['canonical_id']];assert (e['catalog_file'],e['catalog_data_row'])==(name,str(n));assert (r['result_reference'],r['result_selector'],r['identity_status'])==(e['result_reference'],e['result_selector'],e['identity_status']);verified.add(r['canonical_id'])
 for name in names[3:]:
  with (a.phase2_dir/name).open(newline='') as f:
   for r in csv.DictReader(f):
    if r['canonical_id'] in entries:membercheck[r['canonical_id']].add((r['environment'],r['record_id'],r['input_file'],r['input_data_row']))
 assert verified==set(entries)
 for cid,e in entries.items():assert membercheck[cid]=={(r['environment'],r['record_id'],r['input_file'],r['input_data_row']) for r in e['phase2_members']}
 for f in o.iterdir():
  if f.is_file() and f.name!='VALIDATION.json':assert not PRIVATE.search(f.read_text()),('private pattern',f.name)
 for v in walk(g1+g2):
  if isinstance(v,dict):
   for k in ['seed_values','actual_seed_values']:
    if k in v:assert v[k]=='REDACTED'
 lookup.close()
 result=dict(status='PASS_G01_G02_METADATA_LITERAL_RECONCILIATION',G01_candidate_rows=len(g1),G01_matched_rows=7,eligible_endpoint_links=endpoints,saved_accounting_rows=38,G02_binding_rows=len(g2),G02_classes=dict(collections.Counter(r['independence_class'] for r in g2)),fixed_reference_blobs_checked=len(refs),literal_pointer_or_row_checks=pointer_checks,saved_literal_identity_checks=literal_checks,phase2_catalog_groups_linked=len(entries),phase2_original_members_verified=True,new_canonical_experiments_created=0,new_scientific_metric_or_ranking_calculations=0,seed_replay_calls=0,scientific_calls=0,private_raw_rehash_or_decode=0,original_protocol_independence_status_changed=False,formal_claim_adoption='NOT_PERFORMED',detailed_scientific_review='NOT_STARTED',mandatory_stop='AFTER_PUBLICATION_VERIFICATION')
 (o/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
