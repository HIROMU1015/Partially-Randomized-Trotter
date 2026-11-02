"""Integrate fixed Phase 1 observations, preserving unresolved scientific identity."""
from metadata_integration import *
from concurrent.futures import ProcessPoolExecutor,as_completed

ORIGIN_FIELDS='id candidate_id cell_id template_id comparison_id molecule system geometry distance basis rank DF_rank df_rank_actual method family formula pf_order L_D prefix ld q q_m r r_m R_outer K K_m T t_m delta delta_time time_step epsilon epsilon_sig factor bias_kind cohort phase n price_n sample_count seed seed_namespace seed_cohort state_label input_sha256 candidate_fingerprint signal_record_fingerprint result_fingerprint run_id attempt_id status'.split()
def scalar_metadata(d):return {k:v for k,v in d.items() if k in ORIGIN_FIELDS and isinstance(v,(str,bool))}
def pointer_value(doc,sel):
    if sel=='FILE':return doc,[]
    cur=doc;parents=[doc]
    for token in sel.lstrip('/').split('/'):
        token=token.replace('~1','/').replace('~0','~');cur=cur[int(token)] if isinstance(cur,list) else cur[token];parents.append(cur)
    return cur,parents
def origin_metadata(task):
    repo,commit,path,blob,selectors=task;out={};error='UNKNOWN'
    try:
        if path.endswith('.csv'):
            wanted={int(x.split('=')[1]):x for x in selectors};proc=subprocess.Popen(['git','-C',str(repo),'show',commit+':'+path],env=ENV,stdout=subprocess.PIPE)
            reader=csv.DictReader(io.TextIOWrapper(proc.stdout,encoding='utf-8-sig',newline=''))
            for n,row in enumerate(reader,1):
                if n in wanted:out[wanted[n]]=safe(scalar_metadata(row))
            assert proc.wait()==0
        elif path.endswith('.json'):
            doc=json.loads(git(repo,'show',commit+':'+path),parse_int=str,parse_float=str)
            context=scalar_metadata(doc) if isinstance(doc,dict) else {}
            if isinstance(doc,dict):
                for k in ['request','configuration','candidate','input','task']:
                    if isinstance(doc.get(k),dict):context.update(scalar_metadata(doc[k]))
                ham=doc.get('hamiltonian',{})
                if isinstance(ham,dict) and isinstance(ham.get('metadata'),dict):context.update(scalar_metadata(ham['metadata']))
            for sel in selectors:
                try:
                    leaf,parents=pointer_value(doc,sel);meta=dict(context)
                    for x in parents:
                        if isinstance(x,dict):
                            meta.update(scalar_metadata(x))
                            if isinstance(x.get('candidate'),dict):meta.update(scalar_metadata(x['candidate']))
                    if isinstance(leaf,dict):meta.update(scalar_metadata(leaf))
                    out[sel]=safe(meta)
                except (KeyError,IndexError,ValueError,TypeError):out[sel]={'_selector_state':'NOT_RESOLVED'}
        else:error='NOT_DECODED_NON_METADATA_FORMAT'
    except (subprocess.CalledProcessError,UnicodeDecodeError,json.JSONDecodeError) as exc:error=type(exc).__name__
    return dict(commit=commit,path=path,blob=blob,metadata=out,error=error,selectors=len(selectors))
class SplitCSV:
    """Split at complete CSV records for normal Git hosting; no scientific thinning."""
    def __init__(self,out,name,columns,limit=48*1024*1024):self.out=out;self.name=name;self.columns=columns;self.limit=limit;self.parts=[];self.file=None;self.open()
    def open(self):
        n=len(self.parts)+1;stem,suffix=self.name.rsplit('.',1);name=self.name if n==1 else stem+'.part'+str(n).zfill(3)+'.'+suffix
        self.file=(self.out/name).open('w',encoding='utf-8',newline='');self.writer=csv.DictWriter(self.file,fieldnames=self.columns,lineterminator='\r\n');self.writer.writeheader();self.parts.append(dict(name=name,rows=0))
    def row(self,d):
        if self.file.tell()>self.limit:self.file.close();self.open()
        self.writer.writerow({k:jdump(v) if isinstance(v,(dict,list)) else v for k,v in d.items()});self.parts[-1]['rows']+=1
    def close(self):
        self.file.close()
        for p in self.parts:
            b=(self.out/p['name']).read_bytes();p.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        return self.parts
def group_records(records,source_roles=None):
    source_roles=source_roles or {}
    candidates=collections.defaultdict(list)
    for r in records:
        k=same_result_key(r);candidates[k if k else ('OPAQUE',r['environment'],r['record_id'])].append(r)
    groups=[];conflicts=[]
    for key,rs in candidates.items():
        bad=[]
        for i,a in enumerate(rs):
            for b in rs[i+1:]:
                c=conflict(a['condition'],b['condition'])
                sa,sb=a.get('source_commit','UNKNOWN'),b.get('source_commit','UNKNOWN')
                if known(sa) and known(sb) and sa!=sb:
                    roles=source_roles.get(a['result'].get('path',''),{})
                    if sa not in roles or sb not in roles:c.append('source_commit_component_binding')
                ra,rb=a.get('run_id','UNKNOWN'),b.get('run_id','UNKNOWN')
                if known(ra) and known(rb) and ra!=rb:c.append('run_id')
                if c:bad.append(dict(left=a['environment']+':'+a['record_id'],right=b['environment']+':'+b['record_id'],fields=c))
        if bad:
            conflicts.append(dict(key=list(key),conflicts=bad))
            for r in rs:groups.append((key+[r['record_id']] if isinstance(key,list) else (*key,r['record_id']),[r],'PROVISIONAL_METADATA_CONFLICT'))
        else:groups.append((key,rs,'VERIFIED_SAVED_RESULT_IDENTITY' if len(rs)>1 and key[0]!='OPAQUE' else 'PROVISIONAL_SINGLE_OBSERVATION'))
    return groups,conflicts
def source_component_bindings(repo,sourcefiles,out):
    roles_by_path={};rows=[];lookup=GitLookup(repo)
    for sha,prefix in [('155529e20a8fac39e00786c40b57830b71203c4f','artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'),('ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed','artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/')]:
        path=prefix+'scientific_provenance.json';doc=json.loads(git(repo,'show',sha+':'+path));items=[]
        for k,v in doc.items():
            if k.lower().endswith('source') and isinstance(v,str):items.append((k,v,'/'+k))
        for k,v in doc.get('source_commits',{}).items():items.append((k,v,'/source_commits/'+k))
        rolemap={}
        for role,literal,selector in items:
            if not re.fullmatch('[0-9a-f]{7,40}',literal):continue
            try:full=git(repo,'rev-parse',literal+'^{commit}').decode().strip()
            except subprocess.CalledProcessError:full='NOT_FOUND'
            if full!='NOT_FOUND':rolemap.setdefault(full,[]).append(role)
            rows.append(dict(binding_id='SRC-'+str(len(rows)+1).zfill(3),result_scope=prefix,component_role=role,recorded_commit=literal,resolved_commit=full,binding_reference=url(sha,path),selector=selector,match_state='VERIFIED_METADATA_BINDING' if full!='NOT_FOUND' else 'UNKNOWN',limitation='Component SOURCE binding only; not a new scientific source audit or assertion of full historical bitwise identity'))
        for _,p,_ in sourcefiles:
            if p.startswith(prefix):roles_by_path[p]=rolemap
    lookup.close();w=SplitCSV(out,'SOURCE_COMPONENTS.csv',list(rows[0]));
    for row in rows:w.row(row)
    return roles_by_path,w.close()
def build(repo,cache,out):
    out.mkdir(parents=True,exist_ok=True);records=[];observations=set();sourcefiles={};counts=collections.Counter();env_order={x:i for i,x in enumerate(COMMITS)}
    for envname in COMMITS:
        for f in sorted(cache.glob(envname+'-*.index.jsonl')):
            for line in f.open():
                r=json.loads(line);tag=r['environment']+':'+r['record_id'];assert tag not in observations;observations.add(tag)
                x=r['result'];sel=normalize_selector(x.get('original_selector',x.get('selector')))
                if x.get('blob') not in [None,'NOT_FOUND'] and sel:
                    x.update(selector=sel,state='VERIFIED_GIT_BLOB')
                    k=(x['commit'],x['path'],x['blob']);sourcefiles.setdefault(k,set()).add(sel)
                records.append(r)
    assert len(records)==119304
    # Check every public source result selector, without importing any scientific runner or decoding raw archives.
    tasks=[(str(repo),s,p,b,sorted(v)) for (s,p,b),v in sourcefiles.items()];origin={};workers=min(len(tasks),len(os.sched_getaffinity(0)))
    print('Underlying result metadata sources',len(tasks),'parser workers',workers,flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(origin_metadata,t) for t in tasks]):
            z=f.result()
            if z['blob'] in origin:origin[z['blob']]['metadata'].update(z['metadata'])
            else:origin[z['blob']]=z
    (cache/'origin-metadata.json').write_text(jdump(origin))
    for r in records:
        x=r['result'];z=origin.get(x.get('blob'),{});meta=z.get('metadata',{}).get(x.get('selector'))
        if meta is None or meta.get('_selector_state')=='NOT_RESOLVED':
            if x.get('state')=='VERIFIED_GIT_BLOB':x['state']='UNRESOLVED_ORIGINAL_SELECTOR';counts['unresolved_original_selector']+=1
            r['origin_meta']={}
        else:r['origin_meta']=meta;counts['original_selectors_resolved']+=1
    source_roles,source_outputs=source_component_bindings(repo,sourcefiles,out)
    groups,conflicts=group_records(records,source_roles);groups.sort(key=lambda a:min((env_order[r['environment']],r['part'],r['input_row']) for r in a[1]))
    (cache/'condition-conflicts.json').write_text(jdump(conflicts));counts['metadata_conflict_sets']=len(conflicts)
    rec_to_group={};groupmap={};original_by_tag={r['environment']+':'+r['record_id']:r for r in records}
    for i,(key,rs,status) in enumerate(groups,1):
        cid=('CAN' if status.startswith('VERIFIED') else 'TMP')+'-'+str(i).zfill(6)
        groupmap[cid]=dict(records=rs,status=status,key=key)
        for r in rs:rec_to_group[r['environment']+':'+r['record_id']]=cid
        counts[status]+=1
    link_columns=['environment','record_id','canonical_id','link_status','input_commit','input_file','input_data_row','research_series','scope_class','original_experiment_id','original_condition_id','original_run_id','original_analysis_id',*STATES,'original_reliability_ref','source_result_blob','source_selector','source_reference','source_scientific_commit','missing_identity_components']
    links=SplitCSV(out,'RECORD_LINKS.csv',link_columns)
    cat_columns=['canonical_id','identity_status','anchor_environment','anchor_record_id','environments','member_observations','research_series','molecule_as_reported','condition_metadata_as_reported','original_run_ids','original_source_commits','source_component_roles','result_git_blob','result_selector','result_reference','origin_metadata','original_states_reference','identity_basis','limitations']
    cats=SplitCSV(out,'CANONICAL_CATALOG.csv',cat_columns)
    relcols=['relationship_id','type','source_id','target_id','source_record','target_record_or_reference','metric_scope','match_state','basis_refs','independence_meaning','origin']
    rels=SplitCSV(out,'RELATIONSHIPS.csv',relcols)
    issuecols=['unresolved_id','canonical_id','categories','record_links_selector','basis_refs','required_metadata_check','scientific_coverage_assessed']
    issues=SplitCSV(out,'UNRESOLVED.csv',issuecols)
    rel_count=0;rel_keys=set()
    def relationship(kind,source,target,source_record,target_ref,scope,state,basis,meaning,origin_label):
        nonlocal rel_count
        key=(kind,source,target,source_record,jdump(target_ref),scope,origin_label)
        if key in rel_keys:return
        rel_keys.add(key);rel_count+=1
        rels.row(dict(relationship_id='REL-'+str(rel_count).zfill(7),type=kind,source_id=source,target_id=target,source_record=source_record,target_record_or_reference=safe(target_ref),metric_scope=safe(scope),match_state=state,basis_refs=safe(basis),independence_meaning=meaning,origin=origin_label));counts['relationship_'+kind]+=1
    lookup=GitLookup(repo);locator_map=json.loads((cache/'s1-locators.json').read_text())
    for cid,g in groupmap.items():
        rs=g['records'];a=rs[0];x=a['result'];basis=[dict(phase1_commit=r['input_commit'],record=r['record_id'],environment=r['environment']) for r in rs]
        public_ref=url(x['commit'],x['path']) if 'commit' in x else x.get('opaque','UNKNOWN')
        status=g['status'];counts['group_members']+=len(rs)
        for n,r in enumerate(rs):
            tag=r['environment']+':'+r['record_id']
            links.row(dict(environment=r['environment'],record_id=r['record_id'],canonical_id=cid,link_status=status if n==0 else 'COPY_OF_ANCHOR_SAVED_RESULT',input_commit=r['input_commit'],input_file=r['input_file'],input_data_row=r['input_row'],research_series=r['series'],scope_class=r['scope'],original_experiment_id=r['experiment_id'],original_condition_id=r['condition_id'],original_run_id=r['run_id'],original_analysis_id=r['analysis_id'],**r['states'],original_reliability_ref='PHASE1:'+r['environment']+':'+r['record_id']+'/reliability',source_result_blob=r['result'].get('blob','UNKNOWN'),source_selector=r['result'].get('selector','UNKNOWN'),source_reference=url(r['result']['commit'],r['result']['path']) if 'commit' in r['result'] else r['result'].get('opaque','UNKNOWN'),source_scientific_commit=r['source_commit'],missing_identity_components=r['signature_fields_unknown']))
            if n:
                relationship('COPY_OF',tag,a['environment']+':'+a['record_id'],tag,a['environment']+':'+a['record_id'],'Same saved result blob + exact validated selector; compatible recorded numeric physical/method metadata','VERIFIED',basis,'Duplicate observation of the same saved result, not a new independent run','PHASE2_EXACT_RESULT_IDENTITY')
            if isinstance(r['relationships'],list):
                for rel in r['relationships']:
                    target=rel.get('target','UNKNOWN');resolved='UNKNOWN';original_target=target
                    if isinstance(target,str) and target in locator_map:target=locator_map[target]['url'];resolved='ORIGINAL_FIXED_ARTIFACT_REFERENCE'
                    elif isinstance(target,str) and target in rec_to_group:resolved=rec_to_group[target]
                    elif isinstance(target,str) and re.match(r'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/[0-9a-f]{40}/',target):resolved='ORIGINAL_FIXED_ARTIFACT_REFERENCE'
                    elif isinstance(target,str) and re.match(r'^[0-9a-f]{40}:',target):
                        sh,path=target.split(':',1);target=url(sh,path);resolved='ORIGINAL_FIXED_ARTIFACT_REFERENCE'
                    relationship(rel['type'],cid,resolved,tag,target,rel.get('scope','UNKNOWN'),'RECORDED_ONLY' if rel.get('match_state')!='UNKNOWN' else 'UNKNOWN',[dict(environment=r['environment'],record_id=r['record_id'],phase1_commit=r['input_commit'],column='relationships',original_basis_refs=rel.get('basis_refs','UNKNOWN'))],'Original reported relationship retained; no promotion to independent cohort certification','PHASE1_RECORDED_RELATIONSHIP')
        condition_variants=[]
        for r in rs:
            if r['condition'] not in condition_variants:condition_variants.append(r['condition'])
        component_roles={s:source_roles.get(x.get('path',''),{}).get(s,'UNKNOWN') for s in dict.fromkeys(r['source_commit'] for r in rs)}
        cats.row(dict(canonical_id=cid,identity_status=status,anchor_environment=a['environment'],anchor_record_id=a['record_id'],environments=list(dict.fromkeys(r['environment'] for r in rs)),member_observations=len(rs),research_series=list(dict.fromkeys(r['series'] for r in rs)),molecule_as_reported=list(dict.fromkeys(r['molecule'] for r in rs)),condition_metadata_as_reported=condition_variants,original_run_ids=list(dict.fromkeys(r['run_id'] for r in rs)),original_source_commits=list(dict.fromkeys(r['source_commit'] for r in rs)),source_component_roles=component_roles,result_git_blob=x.get('blob','UNKNOWN'),result_selector=x.get('selector','UNKNOWN'),result_reference=public_ref,origin_metadata=a['origin_meta'],original_states_reference='RECORD_LINKS:canonical_id='+cid+'; all original states and reliability remain separately attributable',identity_basis='RESULT_BLOB_PATH_AND_VALIDATED_ORIGINAL_SELECTOR_WITH_COMPATIBLE_RECORDED_CONDITIONS_AND_SOURCE_ROLES' if status.startswith('VERIFIED') else 'PROVISIONAL; NOT A COUNT OF INDEPENDENT EXPERIMENTS',limitations='Same saved-result identity is not an independent-run/cohort assertion. Source, law, state, compiler, acceptance, raw custody and scope retain Phase1 uncertainty. No new numerical evaluation.'))
        categories=[]
        if not status.startswith('VERIFIED'):categories.append('EXPERIMENT_IDENTITY_NOT_CONFIRMED')
        if status.endswith('CONFLICT'):categories.append('CONDITION_METADATA_CONFLICT')
        if x.get('state')!='VERIFIED_GIT_BLOB':categories.append('RESULT_REFERENCE_NOT_ACCESSIBLE_OR_UNRESOLVED')
        if any(not known(r['source_commit']) or 'physical_input' in r['signature_fields_unknown'] or 'state_reference' in r['signature_fields_unknown'] for r in rs):categories.append('SOURCE_INPUT_STATE_BINDING_NOT_CONFIRMED')
        if any(not known(r['run_id']) or 'seed_cohort' in r['signature_fields_unknown'] for r in rs):categories.append('RUN_COHORT_INDEPENDENCE_NOT_CONFIRMED')
        if any(r['states']['validity_status'] in ['UNKNOWN','CONFLICTING','MIXED'] for r in rs):categories.append('VALIDITY_SUCCESSION_NOT_FULLY_RESOLVED')
        if any(r['states']['acceptance_status']=='UNKNOWN' for r in rs):categories.append('FORMAL_ACCEPTANCE_NOT_RECONCILED')
        categories.append('RAW_ARCHIVE_CUSTODY_NOT_NEWLY_VERIFIED')
        issues.row(dict(unresolved_id='UNR-'+cid,canonical_id=cid,categories=categories,record_links_selector='canonical_id='+cid,basis_refs=basis,required_metadata_check='Consult original Phase1 fields, fixed source binding and saved cohort/raw index; do not regenerate or infer absence. Preserve per-record states and any STOP.',scientific_coverage_assessed='NO'))
    # Explicitly documented price sharing across precision, using the same result table/candidate/cohort.
    price_sets=collections.defaultdict(list)
    for cid,g in groupmap.items():
        a=g['records'][0];x=a['result'];meta=a['origin_meta'];path=x.get('path','')
        allowed=('candidate_precision_accounting.csv','candidate_precision_table.csv','CANDIDATE_RESULTS.csv','fixed_nominee_confirmation.csv','new_candidate_accounting.csv')
        candidate=next((meta[k] for k in ['candidate_id','id','cell_id','template_id'] if known(meta.get(k))),None);cohort=meta.get('cohort')
        physics=jdump({k:meta[k] for k in ['molecule','system','T','q','L_D','prefix','r','R_outer','K','formula'] if k in meta})
        if path.endswith(allowed) and candidate and known(cohort) and cohort!='DETERMINISTIC' and ('long_time' in path or 'pauli' in path.lower() or 'hybrid_fairness' in path):price_sets[(x['blob'],candidate,cohort,path,physics)].append(cid)
    for key,cids in price_sets.items():
        if len(cids)<2:continue
        anchor=cids[0];a=groupmap[anchor]['records'][0]
        for c in cids[1:]:relationship('REUSES',c,anchor,'CANONICAL_GROUP',dict(result_blob=key[0],candidate=key[1],cohort=key[2]),'PRICE_COHORT_ONLY across saved precision rows; paired cosine/sine; signal identity separately retained','RECORDED_ONLY',[dict(result_reference=url(a['result']['commit'],a['result']['path']),selector=a['result']['selector']),dict(phase1_report=url(a['input_commit'],BASE+'inventory/'+a['environment']+'/REPORT.md'))],'Shared price samples do not add independent trials; no pooling or N/G arithmetic','PHASE2_DOCUMENTED_SHARED_PRICE')
    # Series/document-level history is explicitly typed; these nodes are not invented experiment records.
    b3sha='ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed';b3path='artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json';b3=json.loads(git(repo,'show',b3sha+':'+b3path));b3ref=url(b3sha,b3path)
    for kind,selector,target,scope in [
      ('RECOVERS','/recovery','NOT_RECORDED','Unfinished B3 run01 tasks; original pool STOP retained; 106 prior PASS trials reused, not new independent draws'),
      ('CONTINUES','/native_SOURCE_history','NOT_RECORDED','Documented continuation/recovery SOURCE history only; same candidate/n/cohort/seed constraints; no blanket run identity assertion'),
      ('RERUN_OF','/external_interruption_recovery','NOT_RECORDED','One documented event-only fixed-seed reconstruction of an unfinished event; not a fresh independent trajectory/cohort')]:
        field=selector.lstrip('/');assert field in b3
        relationship(kind,'DOCUMENT:'+b3ref+'#'+selector,target,'NOT_APPLICABLE',target,scope,'RECORDED_ONLY',[dict(locator=b3ref,selector=selector)],'Documented history; original per-event target identifier remains NOT_RECORDED; no new scientific execution','DOCUMENTED_HISTORY_METADATA')
    new=url('96e69a01d1fae7a402aac2c4efa570a59f8cc7a0','docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md')
    for sh,p in [('6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086','docs/research/track_a_h8_phase_c_run01_results_20261017.md'),('23d46e489af054735d9d328aad0342ecdbddf305','docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md')]:
        target=url(sh,p);lookup.get(sh,p)
        relationship('SUPERSEDES','DOCUMENT:'+new,'DOCUMENT:'+target,'NOT_APPLICABLE',target,'Selected six-axis compiled unit-price replacement and saved accounting version only; original signal, PR samples, STOP and unrelated metrics remain','RECORDED_ONLY',[dict(locator=new,selector='six-axis replacement / saved_output_audit'),dict(locator=target,selector='original saved analysis')],'Updated accounting is not an independent cohort or blanket invalidation of the old raw','DOCUMENTED_HISTORY_METADATA')
    manifest_sha='9a0bc7f17f47823e46c3945734127e9f0d09154c';mp='artifacts/validation_manifest.json';md=json.loads(git(repo,'show',manifest_sha+':'+mp));ix=next(i for i,x in enumerate(md['result_sets']) if x['id']=='df_partial_randomized_screening');assert md['result_sets'][ix]['status']=='invalidated_pending_regeneration'
    relationship('INVALIDATES','DOCUMENT:'+url(manifest_sha,mp)+'#/result_sets/'+str(ix),'RESEARCH_SERIES:df_partial_randomized_screening','NOT_APPLICABLE','df_partial_randomized_screening','Explicit old DF reduced screening invalidation only. Does not invalidate later PF coefficient or Track A results. No new scientific validity judgment.','RECORDED_ONLY',[dict(locator=url(manifest_sha,mp),selector='/result_sets/'+str(ix)+'/status')],'Original invalidation status is retained; corrected canonical input and raw-run join remain unresolved','DOCUMENTED_HISTORY_METADATA')
    lookup.close()
    # Environment-level unresolveds preserve unexpanded ranges and old aliases without inventing missing experiments.
    for key,category,question in [('UNEXPANDED','SOURCE_DATA_UNEXPANDED','Some Phase1 reports leave large grids and baseline joins unexpanded; this integration includes every registered observation, not absent Cartesian-product conditions.'),('HOST_ALIAS','OLD_HOST_ALIAS_UNCONFIRMED','SERVER_A to selected SERVER_1 roots has documented limited correspondence; do not infer a host-wide SERVER_A/B to SERVER_1/2 mapping.'),('MISSING','EVIDENCE_NOT_FOUND_IN_RECORDED_SEARCH_SCOPE','Preserve GAP-001 UWC and corrected canonical-input unresolved history; not an assertion that the experiment was never executed.'),('PRIVATE','PERMISSION_OR_PUBLICATION_LIMIT','LOCAL private V4 and custody locators remain opaque. No other-host filesystem/raw was accessed.')]:
        issues.row(dict(unresolved_id='UNR-GLOBAL-'+key,canonical_id='NOT_APPLICABLE',categories=[category],record_links_selector='ALL_REGISTERED_RECORDS',basis_refs=[url(s,BASE+'inventory/'+e+'/REPORT.md') for e,s in COMMITS.items()],required_metadata_check=question,scientific_coverage_assessed='NO'))
    outputs=list(source_outputs)
    for w in [cats,links,rels,issues]:outputs+=w.close()
    counts.update(input_observations=len(records),catalog_groups=len(groups),exact_copy_observations=counts['relationship_COPY_OF'],unresolved_groups=len(groups),declared_verified_independent_cohorts=0)
    (cache/'integration-stats.json').write_text(jdump(dict(counts=dict(counts),output_files=outputs,input_commits=COMMITS)))
    # Private technical cache supports further reference checks, without publishing duplicate 43-column payloads.
    (cache/'record-groups.json').write_text(jdump(rec_to_group));print(jdump(dict(counts=counts,files=[(x['name'],x['rows'],x['bytes']) for x in outputs])),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();build(a.repo,a.cache,a.out)
if __name__=='__main__':main()
