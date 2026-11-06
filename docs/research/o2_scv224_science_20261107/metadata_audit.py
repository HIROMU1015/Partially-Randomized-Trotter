"""Saved metadata only. No engine, RNG, circuit, fit, probability or variance call."""
from pathlib import Path
from collections import Counter
import hashlib,json,subprocess

root=Path('/tmp/o2-scv224-science-20261107');out=root/'artifacts/o2_scv224/2026-11-07/campaign_run01';ctl=out.with_name('manager_run01');doc=root/'docs/research/o2_scv224_science_20261107'
source_sha='996e434e63ce9b1d14c6ae191649c9abe213cf47'
def read(p):return json.loads(p.read_text())
def digest(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert (ctl/'DISPATCH_EXIT.json').exists(),'manager has not completed; audit must wait'
assert (out/'PARENT_EXIT.json').exists(),'parent exit not persisted'
assert (out/'STOP.json').exists(),'STOP not yet persisted'
partial=[];start={};bits={};prices={};calls=Counter();stages={};max_rss={};signals=[]
for p in sorted(out.glob('worker_*.jsonl')):
 for i,line in enumerate(p.read_text().splitlines()):
  try:r=json.loads(line)
  except json.JSONDecodeError:partial.append(dict(file=p.name,line=i+1));continue
  t=r['ticket'];event=r['event']
  if event=='CALL':calls[r['name']]+=1
  if event=='WORKER_STARTED':assert t not in start,'duplicate start';start[t]=r
  elif event=='BIT_SAVED':assert t not in bits,'duplicate bit';bits[t]=r
  elif event=='PRICE_SAVED':
   assert t in bits and t not in prices,'price before bit or duplicate price'
   assert bits[t]['record']==r['record'],'bit/price record mismatch';prices[t]=r
  if 'worker_peak_RSS_bytes' in r:max_rss[p.name]=max(max_rss.get(p.name,0),r['worker_peak_RSS_bytes'])
for t,r in bits.items():
 d=r['domain'];key=':'.join(d[k] for k in ('axis','stage','cell'));s=stages.setdefault(key,dict(bits=0,priced=0,RZ=0,cache_hits=0,engine_worker_seconds=0.,compile_worker_seconds=0.,price_worker_seconds=0.))
 s['bits']+=1;s['engine_worker_seconds']+=r['engine_seconds']
 if t in prices:
  p=prices[t];s['priced']+=1;s['RZ']+=p['metrics']['rz_count'];s['cache_hits']+=p['cache_hit']
  s['compile_worker_seconds']+=p['compile_seconds'];s['price_worker_seconds']+=p['price_seconds']
words={};word_count=0
p=out/'engine_internal_words.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  r=json.loads(line);t=r['ticket'];assert t not in words,'duplicate word ticket';words[t]=r
  assert len(r['path_words'])<=64,'word cap'
  for x in r['path_words']:assert type(x)is int and 0<=x<2**53
  assert type(r['measurement_word'])is int and 0<=r['measurement_word']<2**53
  word_count+=len(r['path_words'])+1
planned={};p=out/'JOURNAL.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r['event']=='TRIAL_PLANNED':
   assert r['ticket'] not in planned;planned[r['ticket']]=r
assert len(planned)<=224 and len(start)<=224 and len(bits)<=224 and len(prices)<=224
assert word_count<=14560
assert all(calls[k]<=v for k,v in dict(exact_engine=224,Born=224,selected_full_build=448,compile=224).items())
setup=read(out/'SCIENCE_SETUP_CALL_COUNTS.json')
assert all(v['started']==1 and v['returned']<=1 for v in setup['calls'].values()),'setup repeated'
assert setup['science_retry']==0
streams=read(out/'STREAMS.json') if (out/'STREAMS.json').exists() else None
if streams:
 ds=streams['domains'];assert len(ds)==12 and streams['children']==24 and streams['seed']==20261109
 tokens=[json.dumps([r['domain'][k] for k in ('method','axis','stage','cell')],separators=(',',':')) for r in ds]
 assert tokens==sorted(tokens) and len(set(tokens))==12
 assert [r['path_spawn_key'] for r in ds]==[[2*i] for i in range(12)]
 assert [r['bit_spawn_key'] for r in ds]==[[2*i+1] for i in range(12)]
result=read(out/'SCV_RESULT.json') if (out/'SCV_RESULT.json').exists() else {'status':'STOPPED'}
if result['status']=='COMPLETE':
 assert len(bits)==len(prices)==224
 for axis in ('X','Y'):
  assert stages[axis+':train:1']['bits']==32
  for h,n in enumerate((4,64,4,4,4)):assert stages[axis+':prod:'+str(h)]['bits']==n
  fit=read(out/(axis+'_FIT.json'));assert fit['rank']==2
  from fractions import Fraction as F
  assert F(fit['width'])<=4 and F(fit['width'])==2+abs(F(fit['coefficient']))
  assert all(F(c)==0 for i,c in enumerate(fit['coefficients_by_cell']) if i!=1)
 assert calls['exact_engine']==calls['Born']==224 and calls['selected_full_build']==448
ledger=read(out/'ATTEMPTS.json')
assert set(ledger['trials'])==set(start)
for t in start:
 expected='BIT_AND_PRICE_DURABLE' if t in prices else 'BIT_DURABLE_PRICE_MISSING' if t in bits else 'ATTEMPT_NO_DURABLE_BIT'
 assert ledger['trials'][t]==expected
source=read(out/'SOURCE.json') if (out/'SOURCE.json').exists() else read(out/'EXECUTION_SOURCE.json')
assert source['source_commit']==source_sha
lock=read(doc/'PROFILE_LOCK.json')
for p,h in lock['source_sha256'].items():assert digest((root/p).read_bytes())==h,p
for p,h in read(doc/'INPUT_BINDING.json')['file_sha256'].items():assert digest((root/p).read_bytes())==h,p
worker_exit=read(out/'WORKER_EXIT_CODES.json') if (out/'WORKER_EXIT_CODES.json').exists() else {}
for p in out.glob('OBS_*.jsonl'):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r.get('event')=='SIGNAL_RECEIVED':signals.append(dict(file=p.name,signal=r.get('signal')))
x=dict(scope='record/provenance/word/cost audit only; no science reevaluated',science_SOURCE=source_sha,status=result['status'],
 planned=len(planned),started=len(start),bits=len(bits),prices=len(prices),uniform53_words=word_count,calls=dict(calls),
 setup_calls=setup['calls'],stages=stages,partial_worker_tails=partial,bit_before_price_verified=True,unique_tickets_verified=True,
 full_positive_strata_preserved=(result['status']=='COMPLETE'),stream_layout_verified=streams is not None,
 PRNG_independence='MODEL_ASSUMPTION_NOT_PROVED',source_input_bytes_unchanged=True,
 parent_exit=read(out/'PARENT_EXIT.json'),dispatch_exit=read(ctl/'DISPATCH_EXIT.json'),
 worker_exit_count=len(worker_exit),worker_exit_code_distribution=dict(Counter(str(v.get('exit_code')) for v in worker_exit.values())),signals=signals,
 max_single_worker_RSS_bytes=max(max_rss.values()) if max_rss else None,scientific_recomputation=0,C1='frozen')
write(doc/'FINAL_METADATA_AUDIT.json',x)
print(json.dumps({k:v for k,v in x.items() if k not in ('stages','dispatch_exit','setup_calls')},ensure_ascii=False))
