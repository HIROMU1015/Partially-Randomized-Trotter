"""Metadata input/output and reference checks; no scientific calculations."""
from metadata_integration import *
from integrate_catalog import SplitCSV

def reference_checks(repo,cache,out):
    public={};opaque={}
    pattern=re.compile(r'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/([0-9a-f]{40})/([^\s)\]"<>\\]+)')
    for envname in COMMITS:
        t=(cache/(envname+'-REPORT.md')).read_text()
        for m in pattern.finditer(t):
            sha,p=m.groups();p=urllib.parse.unquote(p.split('#')[0]);public[sha,p]=url(sha,p)
        if envname=='SERVER_1':
            for line in t.splitlines():
                m=re.match(r'^\|\s*(S1-[A-Z0-9-]+)\s*\|',line)
                if m:opaque[m[1]]=[(x[0],urllib.parse.unquote(x[1].split('#')[0])) for x in pattern.findall(line)]
    # Full fixed locators inside Phase1 references/identity metadata; no raw content is opened.
    for f in cache.glob('*.index.jsonl'):
        for line in f.open():
            for m in pattern.finditer(line):
                sha,p=m.groups();p=urllib.parse.unquote(p.split('#')[0]);public[sha,p]=url(sha,p)
    lookup=GitLookup(repo);columns=['reference_id','locator_kind','original_locator','resolved_public_locator','fixed_commit','repository_path','git_blob','access_result','verification_scope']
    writer=SplitCSV(out,'REFERENCE_INDEX.csv',columns);rows=[];n=0
    for (sha,p),u in sorted(public.items()):
        blob=lookup.get(sha,p);n+=1
        row=dict(reference_id='REF-'+str(n).zfill(5),locator_kind='FIXED_GITHUB_ARTIFACT',original_locator=u,resolved_public_locator=u,fixed_commit=sha,repository_path=p,git_blob=blob or 'NOT_FOUND',access_result='ACCESSIBLE_FIXED_GIT_OBJECT' if blob else 'NOT_FOUND_IN_LOCAL_OBJECT_SCOPE',verification_scope='Git object metadata existence only; no raw decode, complete raw hash, scientific review or other-host existence check')
        rows.append(row);writer.row(row)
    for original,targets in sorted(opaque.items()):
        for sha,p in targets:
            n+=1;blob=lookup.get(sha,p);writer.row(dict(reference_id='REF-'+str(n).zfill(5),locator_kind='SERVER_1_REPORT_LOCATOR',original_locator=original,resolved_public_locator=url(sha,p),fixed_commit=sha,repository_path=p,git_blob=blob or 'NOT_FOUND',access_result='ACCESSIBLE_FIXED_GIT_OBJECT' if blob else 'NOT_FOUND_IN_LOCAL_OBJECT_SCOPE',verification_scope='Scope limited to the original REPORT locator/member; no host alias inference'))
    lookup.close();parts=writer.close();(cache/'reference-validation.json').write_text(jdump(dict(references=n,public_artifacts=len(public),unresolved=[r for r in rows if r['git_blob']=='NOT_FOUND'],output_files=parts)));print('Reference metadata checked',n,'unresolved public objects',sum(x['git_blob']=='NOT_FOUND' for x in rows),flush=True)
    return parts
def input_index(repo,cache,out):
    validation=json.loads((cache/'input-validation.json').read_text());checks={(x['environment'],x['path']):x for x in validation['summaries']};remote=json.loads((cache/'remote-input-metadata.json').read_text());detail=json.loads((cache/'details-validation.json').read_text());rows=[]
    for envname,items in remote.items():
        for item in items:
            objid=git(repo,'rev-parse',COMMITS[envname]+':'+item['path']).decode().strip();assert objid==item['blob']
            check=checks.get((envname,item['path']),{})
            if item['name']=='DETAILS.jsonl':check=dict(sha256=detail['sha256'],rows=detail['entries'],first='NOT_APPLICABLE',last='NOT_APPLICABLE')
            rows.append(dict(environment=envname,input_commit=COMMITS[envname],file=item['name'],input_reference=url(COMMITS[envname],item['path']),git_blob=objid,bytes=item['bytes'],data_rows_or_detail_entries=check.get('rows','NOT_APPLICABLE'),first_record_id=check.get('first','NOT_APPLICABLE'),last_record_id=check.get('last','NOT_APPLICABLE'),sha256=check.get('sha256','NOT_RECORDED_IN_INPUT_REPORT'),remote_state='PUBLISHED_FIXED_DIRECTORY_BLOB_SIZE_MATCH',validation='SCHEMA_AND_RANGE_VERIFIED' if item['name'].endswith('.csv') else 'DETAIL_REFS_RESOLVED' if item['name']=='DETAILS.jsonl' else 'REPORT_GITHUB_BLOB_MATCH'))
    w=SplitCSV(out,'INPUT_FILES.csv',list(rows[0]));
    for r in rows:w.row(r)
    return w.close()
def validate_tables(repo,cache,out):
    stats=json.loads((cache/'integration-stats.json').read_text());files=stats['output_files'];groups={};links={};linked_states={};linked_ids={};states=collections.Counter();counts=collections.Counter();seen_input=set();part_headers={}
    for part in files:
        p=out/part['name'];b=p.read_bytes();assert len(b)==part['bytes'] and hashlib.sha256(b).hexdigest()==part['sha256']
        assert not PRIVATE.search(b.decode()),'Private pattern in output '+p.name
        with p.open(newline='') as source:
            reader=csv.DictReader(source);root=re.sub(r'\.part\d{3}', '',p.name);h=reader.fieldnames
            if root in part_headers:assert h==part_headers[root]
            part_headers[root]=h;nr=0
            for r in reader:
                nr+=1;assert None not in r and all(v!='' for v in r.values())
                if root=='CANONICAL_CATALOG.csv':assert r['canonical_id'] not in groups;groups[r['canonical_id']]=int(r['member_observations']);counts['catalog_rows']+=1
                elif root=='RECORD_LINKS.csv':
                    tag=r['environment']+':'+r['record_id'];assert tag not in links;links[tag]=r['canonical_id'];counts['record_links']+=1
                    linked_states[tag]={k:r[k] for k in STATES}
                    linked_ids[tag]=tuple(r[k] for k in ['original_experiment_id','original_condition_id','original_run_id','original_analysis_id','source_scientific_commit'])
                    seen_input.add((r['environment'],r['input_file'],r['input_data_row']));counts['links_'+r['environment']]+=1
                    states.update((k,r[k]) for k in STATES)
        assert nr==part['rows'],p.name
    assert len(links)==119304 and len(seen_input)==119304
    assert set(links.values())==set(groups)
    assert dict(collections.Counter(links.values()))==groups
    # Revisit the reduced public input metadata, retaining every original state token exactly.
    expected_states=collections.Counter();expected_ids={}
    for f in cache.glob('*.index.jsonl'):
        for line in f.open():
            r=json.loads(line);tag=r['environment']+':'+r['record_id'];assert tag in links;expected_ids[tag]=r['states'];expected_states.update(r['states'].items())
            assert linked_states[tag]==r['states'],'Original state changed for '+tag
            assert linked_ids[tag]==tuple(r[k] for k in ['experiment_id','condition_id','run_id','analysis_id','source_commit']),'Original identity field changed for '+tag
    assert states==expected_states,'Original state tokens changed'
    relation_ids=set();reltypes=collections.Counter();copy_keys=set()
    for part in files:
        if not part['name'].startswith('RELATIONSHIPS'):continue
        for r in csv.DictReader((out/part['name']).open(newline='')):
            assert r['relationship_id'] not in relation_ids;relation_ids.add(r['relationship_id']);assert r['type'] in RELTYPES;reltypes[r['type']]+=1
            if r['type']=='COPY_OF':
                assert r['source_id'] in links and r['target_id'] in links and links[r['source_id']]==links[r['target_id']]
                assert r['match_state']=='VERIFIED';assert r['source_id'] not in copy_keys;copy_keys.add(r['source_id'])
            if r['type']=='INDEPENDENT_CONFIRMATION_OF':assert r['match_state'] in ['RECORDED_ONLY','UNKNOWN'],'Unreviewed independence promoted'
    assert len(copy_keys)==119304-len(groups)
    issueids=set();issue_groups=set();categories=collections.Counter()
    for part in files:
        if not part['name'].startswith('UNRESOLVED'):continue
        for r in csv.DictReader((out/part['name']).open(newline='')):
            assert r['unresolved_id'] not in issueids;issueids.add(r['unresolved_id']);assert r['scientific_coverage_assessed']=='NO';categories.update(json.loads(r['categories']))
            if r['canonical_id']!='NOT_APPLICABLE':assert r['canonical_id'] in groups;issue_groups.add(r['canonical_id'])
    assert issue_groups==set(groups)
    # Evidence tree is read-only; concurrent unrelated additions/refs are recorded, never rolled back.
    assert git(repo,'rev-parse','HEAD')==(cache/'baseline-HEAD.txt').read_bytes()
    before=(cache/'baseline-status.txt').read_text().splitlines();after=git(repo,'status','--porcelain=v1','-uall').decode().splitlines();assert [x for x in before if not x.startswith('?? ')]==[x for x in after if not x.startswith('?? ')]
    untouched=all(git(repo,'cat-file','-t',s).strip()==b'commit' for s in COMMITS.values())
    result=dict(status='PASS_METADATA_INTEGRATION',input_rows=119304,record_links=len(links),catalog_groups=len(groups),copy_observations=len(copy_keys),relationships=dict(reltypes),unresolved_groups=len(issue_groups),unresolved_categories=dict(categories),original_state_tokens_preserved=True,original_active_HEAD_and_tracked_status_preserved=True,fixed_input_commits_unchanged=untouched,scientific_calls=0,new_independent_cohort_certification='NOT_PERFORMED',phase3_started=False,validation_scope='Schema, streaming, references, identity rules and provenance only; no scientific performance or coverage judgment')
    (cache/'output-validation.json').write_text(jdump(result));print(jdump(result),flush=True);return result
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['references','validate']);p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    if a.mode=='references':
        parts=reference_checks(a.repo,a.cache,a.out)+input_index(a.repo,a.cache,a.out);s=json.loads((a.cache/'integration-stats.json').read_text());s['output_files']+=parts;(a.cache/'integration-stats.json').write_text(jdump(s))
    else:validate_tables(a.repo,a.cache,a.out)
if __name__=='__main__':main()
