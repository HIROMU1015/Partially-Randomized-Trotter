"""Read-only PR-COVERAGE-1 metadata integration; standard library, no science imports.

Inputs are fixed Git objects. Data values are retained in the Phase 1 artifacts;
this adapter indexes identities, original states and explicit relationships only.
"""
import argparse,collections,csv,datetime,hashlib,io,json,os,pathlib,re,subprocess,sys,urllib.parse
from concurrent.futures import ProcessPoolExecutor,as_completed
from decimal import Decimal,InvalidOperation

COMMITS={'LOCAL':'17aacf67ee854657652ade8035c5fdfad1eb077a','SERVER_1':'2de84ed7af00e73339886429aabab0bf8180e9f0','SERVER_2':'8834de0cf938b13490c08e05006e3c566fed7c7a'}
BASE='docs/research/pr_cross_host_coverage_audit_20261102/'
PROTOCOL='968218c2b0c6280af6c1c02822603e3312aa410c'
TOKENS={'UNKNOWN','NOT_RECORDED','NOT_ACCESSIBLE','NOT_FOUND','NOT_APPLICABLE','REDACTED'}
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
JSONCOLS='claim_refs code_refs input_refs contract_refs geometry basis electronic_state hamiltonian method comparison precision implementation sampling evaluation result_refs stop_refs relationships identity_basis evidence_refs reliability missing_fields'.split()
OBJECT_KEYS={
'geometry':'id configuration coordinates_ref distance unit',
'basis':'name active_space orbital_basis mapping qubits',
'electronic_state':'electrons charge spin sector input_state_ref reference_ref',
'hamiltonian':'representation snapshot_ref rank_policy rank fragment_order_ref thresholds identity_policy one_body_policy decomposition_ref',
'method':'family pf_order L_D q r K T time_step law_ref normalization_ref round_id',
'comparison':'baseline_refs candidate_id candidate_set_ref selection_rule_ref matched_fields differing_fields comparison_scope',
'precision':'target error_budget_ref failure_probability_ref arithmetic tolerance_ref guarantee_scope',
'implementation':'compiler backend control measurement_boundary cost_units state_preparation_policy runtime_ref',
'sampling':'execution_environment role seed_spec rng cohort_id sample_count sample_ids_ref price_cohort_id signal_cohort_id draw_law_ref dependence_ref selection_data_refs confirmation_plan_ref'}
STATES='run_status scientific_status technical_status acceptance_status validity_status preservation_status access_status independence_status'.split()
RELTYPES=set('COPY_OF CITES REUSES RERUN_OF INDEPENDENT_CONFIRMATION_OF CONDITION_VARIANT_OF CONTINUES RECOVERS SUPERSEDES INVALIDATES'.split())
PRIVATE=re.compile(r'/(?:home|tmp|mnt|srv|scratch|media)/|(?:\d{1,3}\.){3}\d{1,3}|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY')

def jdump(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],env=ENV,stderr=subprocess.PIPE)
def url(sha,path):return 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+sha+'/'+path
def known(x):return isinstance(x,str) and x not in TOKENS and x not in ('','null')
def safe(x):
    if isinstance(x,str):return 'REDACTED' if PRIVATE.search(x) else x
    if isinstance(x,list):return [safe(v) for v in x]
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    return x
def normalize_selector(s):
    if not isinstance(s,str):return None
    m=re.fullmatch(r'CSV(?:_DATA_ROW=|:data_row=| data row )([1-9][0-9]*)',s,re.I)
    if m:return 'CSV_DATA_ROW='+m[1]
    if s.startswith('/') and not any(c in s for c in ['\n',';',' -> ']):return s
    if s in ['','FILE','WHOLE_DOCUMENT']:return 'FILE'
    return None
class HashReader(io.RawIOBase):
    def __init__(self,source):self.source=source;self.digest=hashlib.sha256();self.size=0
    def readable(self):return True
    def readinto(self,b):
        v=self.source.read(len(b));n=len(v);b[:n]=v;self.digest.update(v);self.size+=n;return n
class HashWriter:
    def __init__(self):self.digest=hashlib.sha256()
    def write(self,t):self.digest.update(t.encode('utf-8'));return len(t)
class GitLookup:
    def __init__(self,repo):
        self.proc=subprocess.Popen(['git','-C',str(repo),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],env=ENV,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);self.cache={}
    def get(self,s,p):
        key=s+':'+p
        if key not in self.cache:
            if '\n' in key or '\r' in key:return None
            self.proc.stdin.write((key+'\n').encode());self.proc.stdin.flush();a=self.proc.stdout.readline().decode().strip().split()
            self.cache[key]=a[0] if len(a)==3 and a[1]=='blob' else None
        return self.cache[key]
    def close(self):self.proc.stdin.close();self.proc.wait()
def result_ref(x,locators,lookup):
    locator=x.get('locator','UNKNOWN');version=x.get('version','UNKNOWN')
    if locator in locators:
        z=locators[locator];commit,path=z['commit'],urllib.parse.unquote(z['path'])
    else:
        m=re.match(r'https://github.com/HIROMU1015/Partially-Randomized-Trotter/(?:blob|tree)/([0-9a-f]{40})/([^#]+)',locator)
        if not m:return dict(original=safe(x),opaque=locator,selector=normalize_selector(x.get('selector')),state='NOT_RESOLVED')
        commit,path=m.groups();path=urllib.parse.unquote(path)
    blob=lookup.get(commit,path);sel=normalize_selector(x.get('selector'))
    return dict(commit=commit,path=path,blob=blob or 'NOT_FOUND',selector=sel or x.get('selector','UNKNOWN'),state='VERIFIED_GIT_BLOB' if blob and sel else 'UNRESOLVED_SELECTOR_OR_OBJECT',original_selector=x.get('selector'),original_ref=safe(x))
def numeric_metadata(s):
    if not isinstance(s,str) or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',s):return None
    try:return str(Decimal(s).normalize())
    except InvalidOperation:return None
def condition_signature(d):
    # Formatting normalization is for metadata comparisons only. Original fields remain unmodified.
    out={}
    molecule=d['molecule']
    if re.fullmatch(r'H?\d+',molecule,re.I):out['molecule']='H'+molecule.lstrip('Hh')
    for field in ['L_D','q','r','K','T','time_step']:
        v=numeric_metadata(d['method'].get(field))
        if v is not None:out['method.'+field]=v
    for col,k in [('geometry','distance'),('hamiltonian','rank'),('precision','target')]:
        v=numeric_metadata(d[col].get(k))
        if v is not None:out[col+'.'+k]=v
    for col,k in [('basis','name'),('basis','mapping'),('geometry','unit')]:
        v=d[col].get(k)
        if known(v):
            v=v.lower().replace(' ','');v={'ångström':'angstrom','å':'angstrom'}.get(v,v);out[col+'.'+k]=v
    return out
def conflict(a,b):return [k for k in a.keys()&b.keys() if a[k]!=b[k]]
def same_result_key(r):
    x=r.get('result',{})
    return (x['blob'],x['path'],x['selector']) if x.get('state')=='VERIFIED_GIT_BLOB' else None
def classify_attempt(same_result=False,same_conditions=False,distinct_attempt=False,same_seed=False,disjoint_documented=False,fixed_nominee=False):
    if same_result:return 'COPY_OF'
    if same_conditions and distinct_attempt:
        if disjoint_documented and fixed_nominee and not same_seed:return 'INDEPENDENT_CONFIRMATION_OF'
        return 'RERUN_OF'
    return 'UNKNOWN'
def load_details(repo,cache):
    target=cache/'details-index.json';proc=subprocess.Popen(['git','-C',str(repo),'show',COMMITS['SERVER_2']+':'+BASE+'inventory/SERVER_2/DETAILS.jsonl'],env=ENV,stdout=subprocess.PIPE)
    hr=HashReader(proc.stdout);stream=io.TextIOWrapper(io.BufferedReader(hr),encoding='utf-8',newline='');idx={}
    for line in stream:
        d=json.loads(line);key=d['record_id']+'/'+d['column'];assert key not in idx and set(['record_id','column','value'])<=d.keys();idx[key]=d['value']
    assert proc.wait()==0
    # Safe cache of the already-public metadata, not scientific raw or an approval database.
    target.write_text(jdump(idx));(cache/'details-validation.json').write_text(jdump(dict(entries=len(idx),bytes=hr.size,sha256=hr.digest.hexdigest())))
    return len(idx)
def index_part(task):
    repo,cache,envname,commit,path,part,header=task;cache=pathlib.Path(cache)
    loc=json.loads((cache/'s1-locators.json').read_text()) if envname=='SERVER_1' else {}
    details=json.loads((cache/'details-index.json').read_text()) if envname=='SERVER_2' else {}
    used=set();counts=collections.Counter();types=collections.Counter();series=collections.Counter();look=GitLookup(repo)
    proc=subprocess.Popen(['git','-C',str(repo),'show',commit+':'+path],env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    hr=HashReader(proc.stdout);reader=csv.DictReader(io.TextIOWrapper(io.BufferedReader(hr),encoding='utf-8',newline=''))
    assert reader.fieldnames==header and len(header)==43
    canonical=HashWriter();writer=csv.writer(canonical,lineterminator='\r\n');writer.writerow(header)
    first=last=None;seen=set();resultfile=cache/(envname+'-'+str(part).zfill(3)+'.index.jsonl');refissues=[]
    with resultfile.open('w') as output:
        for n,raw in enumerate(reader,1):
            assert None not in raw and all(x!='' for x in raw.values());writer.writerow([raw[k] for k in header]);counts['rows']+=1
            assert raw['schema_version']=='PR-COVERAGE-1' and raw['environment']==envname
            rid=raw['record_id'];assert rid not in seen;seen.add(rid);first=first or rid;last=rid
            d=dict(raw)
            for k in JSONCOLS:
                v=json.loads(raw[k],parse_int=str,parse_float=str)
                if isinstance(v,dict) and 'detail_ref' in v:
                    assert set(v)=={'detail_ref'} and v['detail_ref']=='DETAILS.jsonl#'+rid+'/'+k
                    key=rid+'/'+k;v=details[key];used.add(key);counts['detail_refs']+=1
                    assert not (isinstance(v,dict) and 'detail_ref' in v)
                d[k]=v
            for col,keys in OBJECT_KEYS.items():assert isinstance(d[col],dict) and set(keys.split())<=d[col].keys(),(rid,col)
            assert isinstance(d['evaluation'],list) and isinstance(d['result_refs'],list) and d['result_refs']
            references={}
            for col in ['claim_refs','code_refs','input_refs','contract_refs','result_refs','stop_refs','evidence_refs']:
                values=d[col]
                assert isinstance(values,list) or values in TOKENS,(rid,col)
                if isinstance(values,list):
                    for x in values:
                        assert set('ref_id role locator version selector preservation access identity'.split())<=x.keys(),(rid,col)
                        references[x['ref_id']]=x
            origrel=d['relationships'];assert isinstance(origrel,list) or origrel in TOKENS
            if isinstance(origrel,list):
                for rel in origrel:
                    assert rel['type'] in RELTYPES;types[rel['type']]+=1
            privacy=any(PRIVATE.search(x) for x in raw.values());counts['private_pattern_rows']+=bool(privacy)
            rr=result_ref(d['result_refs'][0],loc,look)
            if rr['state']!='VERIFIED_GIT_BLOB':counts['unresolved_result_locator']+=1
            field_refs={k:[safe(x) for x in d[k]] if isinstance(d[k],list) else d[k] for k in ['input_refs','contract_refs','code_refs','stop_refs']}
            candidate=d['comparison'].get('candidate_id','UNKNOWN')
            identity=d['identity_basis'].get('candidate_condition',{})
            if not known(candidate):
                v=identity.get('value',identity.get('value/ref','UNKNOWN'))
                candidate=v.get('recorded_id','UNKNOWN') if isinstance(v,dict) else v
            rec=dict(environment=envname,record_id=rid,input_commit=commit,input_file=path.rsplit('/',1)[1],input_row=n,part=part,series=raw['research_series'],scope=raw['scope_class'],observed_at=raw['observed_at'],experiment_id=raw['experiment_id'],condition_id=raw['condition_id'],run_id=raw['run_id'],analysis_id=raw['analysis_id'],source_commit=raw['source_commit'],molecule=raw['molecule'],condition=condition_signature(d),method=safe(d['method']),candidate=safe(candidate),sampling=safe(d['sampling']),result=rr,states={k:raw[k] for k in STATES},reliability=safe(d['reliability']),relationships=safe(origrel),binding_refs=field_refs,privacy_pattern=privacy,identity_basis=safe(d['identity_basis']))
            rec['signature_fields_unknown']=[k for k in ['physical_input','source_contract','state_reference','law_normalization','compiler_boundary','seed_cohort','continuation'] if d['identity_basis'].get(k,{}).get('match_state')=='UNKNOWN']
            output.write(jdump(safe(rec))+'\n');series[raw['research_series']]+=1
    assert proc.wait()==0
    if envname=='SERVER_1':assert canonical.digest.hexdigest()==hr.digest.hexdigest(),'Part not canonical RFC4180 serialization'
    summary=dict(environment=envname,commit=commit,path=path,part=part,rows=counts['rows'],first=first,last=last,bytes=hr.size,sha256=hr.digest.hexdigest(),canonical_sha256=canonical.digest.hexdigest(),counts=dict(counts),series=dict(series),relationships=dict(types),git_refs=look.cache,detail_keys=sorted(used))
    look.close();(cache/(envname+'-'+str(part).zfill(3)+'.validation.json')).write_text(jdump(summary));return {k:v for k,v in summary.items() if k not in ['git_refs','detail_keys']}
def prepare(repo,cache):
    cache.mkdir(parents=True,exist_ok=True)
    for envname,sha in COMMITS.items():
        destination=cache/(envname+'-REPORT.md')
        if not destination.exists():destination.write_bytes(git(repo,'show',sha+':'+BASE+'inventory/'+envname+'/REPORT.md'))
    locators={}
    for line in (cache/'SERVER_1-REPORT.md').read_text().splitlines():
        m=re.match(r'^\|\s*(S1-T[0-9]+|S1-META-[A-Z0-9-]+)\s*\|',line)
        u=re.search(r'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/([0-9a-f]{40})/([^\s)]+)',line)
        if m and u:locators[m[1]]=dict(commit=u[1],path=u[2],url=u[0])
    (cache/'s1-locators.json').write_text(jdump(locators))
    header=next(csv.reader(io.StringIO(git(repo,'show',PROTOCOL+':'+BASE+'INVENTORY_TEMPLATE.csv').decode())))
    load_details(repo,cache);tasks=[]
    for envname,commit in COMMITS.items():
        ps=[BASE+'inventory/'+envname+'/INVENTORY.csv']
        if envname=='SERVER_1':ps += [BASE+'inventory/'+envname+'/INVENTORY.part'+str(i).zfill(3)+'.csv' for i in range(2,31)]
        for i,p in enumerate(ps,1):tasks.append((str(repo),str(cache),envname,commit,p,i,header))
    workers=min(len(tasks),len(os.sched_getaffinity(0)));print('Metadata parser workers',workers,'independent CSV parts',len(tasks),flush=True)
    summaries=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(index_part,t) for t in tasks]
        for f in as_completed(fs):
            s=f.result();summaries.append(s);print(s['environment'],s['part'],'rows',s['rows'],'metadata complete',flush=True)
    summaries.sort(key=lambda x:(list(COMMITS).index(x['environment']),x['part']))
    for envname,expected in [('LOCAL',9665),('SERVER_1',100878),('SERVER_2',8761)]:
        ss=[x for x in summaries if x['environment']==envname];assert sum(x['rows'] for x in ss)==expected
        if envname=='SERVER_1':
            previous=0;report=(cache/'SERVER_1-REPORT.md').read_text()
            for s in ss:
                start=int(s['first'].split('-')[1]);end=int(s['last'].split('-')[1]);assert start==previous+1 and end-start+1==s['rows'];previous=end
                name=s['path'].split('/')[-1];line=next(x for x in report.splitlines() if x.startswith('| ['+name+']'))
                assert '`'+s['sha256']+'`' in line and str(s['bytes']) in line and '| '+str(s['rows'])+' |' in line
    assert set(json.loads((cache/'details-index.json').read_text()))==set(json.loads((cache/'SERVER_2-001.validation.json').read_text())['detail_keys'])
    (cache/'input-validation.json').write_text(jdump(dict(status='PASS_INPUT_SCHEMA_AND_PARTS',workers=workers,summaries=summaries)))
    print('ALL INPUT ROWS',sum(x['rows'] for x in summaries),'SERVER2 detail references',next(x['counts'].get('detail_refs') for x in summaries if x['environment']=='SERVER_2'),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare']);p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);a=p.parse_args();prepare(a.repo,a.cache)
if __name__=='__main__':main()
