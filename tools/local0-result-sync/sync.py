"""Windows-owned, read-only public result mirror and per-run monitor."""
import argparse,base64,contextlib,hashlib,io,json,os,pathlib,subprocess,sys,tempfile,time,uuid,zipfile
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
P=pathlib.Path
sha=lambda b:hashlib.sha256(b).hexdigest()
blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def load(p):return json.loads(P(p).read_text(encoding='utf-8-sig'))
def save(p,v):atomic(P(p),(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def atomic(p,b):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    with t.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(t,p)

@contextlib.contextmanager
def lock(path):
    path=P(path);path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+b');f.seek(0);f.write(b'0');f.flush();deadline=time.monotonic()+90
    while True:
        try:
            f.seek(0)
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic()>deadline:f.close();raise TimeoutError('Sync lock is busy')
            time.sleep(.1)
    try:yield
    finally:
        if os.name=='nt':f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(f,fcntl.LOCK_UN)
        f.close()

def remote(config,payload):
    encoded=base64.b64encode(json.dumps(payload).encode()).decode()
    ssh=config.get('ssh_executable')or(str(P(os.environ['SystemRoot'])/'System32/OpenSSH/ssh.exe')if os.name=='nt'else'ssh')
    # Nested Windows OpenSSH can retain a redirected stdin pipe after Close.
    # The observer is public code; transport it as one quoted argument instead.
    quote=lambda value:"'"+value.replace("'","'\"'\"'")+"'"
    if not config.get('remote_observer'):raise ValueError('Installed read-only remote_observer path is required')
    remote_command='set -o pipefail; '+quote(config['remote_python'])+' -B '+quote(config['remote_observer'])+' --payload-base64 '+encoded+' | base64 -w 1024'
    command=[ssh,'-T','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',config['ssh_alias'],remote_command]
    if os.name=='nt':
        # Git for Windows SSH can retain Python's capture pipes when launched
        # inside Windows OpenSSH. The same command exits with file-backed output.
        with tempfile.TemporaryFile() as output:
            result=subprocess.run(command,stdout=output,timeout=120,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            output.seek(0)
            code,raw,error=result.returncode,output.read(),b''
    else:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,timeout=120)
        code,raw,error=result.returncode,result.stdout,result.stderr
    if code:raise RuntimeError('SSH observation failed: '+error.decode(errors='replace')[-2000:]+raw.decode(errors='replace')[-1500:])
    response=json.loads(base64.b64decode(raw))
    if not response['ok']:raise RuntimeError(response.get('error'))
    return response['result']

def observe(config,request):
    return remote(config,dict(action='run_status',run_id=request['run_id'],run_directory=request['run_directory']))

def fixture_publication(config,known,target):
    if not config.get('artificial_only'):raise ValueError('Fixture transport requires artificial_only')
    pointer=load(config['fixture_publication']);root=P(pointer['root']);manifest={};data=io.BytesIO();changed=[]
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED)as z:
        for n,s in config['files'].items():
            raw=(root/n).read_bytes();manifest[n]=dict(source_path=s,git_blob=blob(raw),sha256=sha(raw))
            if known.get(n)!=blob(raw):z.writestr(n,raw);changed.append(n)
    index=load(root/'INDEX.json');final=dict(status='NOT_REQUESTED')
    if target:
        final=dict(status='FINAL_PUBLICATION_PENDING',message='最終結果は公開待ち',run_id=target['run_id'])
        for r in index.get('runs',[]):
            if r.get('label')==target['run_id']and r.get('final_result_sha256')in[v['sha256']for v in target.get('artifacts',{}).values()]:
                final=dict(status='FINAL_RESULT_PUBLISHED',message='最終結果を公開版で確認',run_id=target['run_id'])
    archive=data.getvalue()
    return dict(commit=pointer['commit'],manifest=manifest,changed=changed,archive_base64=base64.b64encode(archive).decode(),archive_sha256=sha(archive),transfer_bytes=len(archive)if changed else 0,final=final)

def clean_transaction(tx):
    for p in tx.iterdir():
        if p.is_file():p.unlink()
    tx.rmdir()

def recover(destination,allowed):
    journal=destination/'.SYNC_TRANSACTION.json'
    if not journal.exists():return
    data=load(journal);tx=P(data['transaction']).resolve()
    if tx.parent!=(destination/'.sync-staging').resolve():raise ValueError('Unsafe recovery directory')
    for item in data['before']:
        name=item['name']
        if name not in allowed|{'LOCAL_SYNC.json'}:raise ValueError('Unsafe recovery member')
        path=destination/name
        if item['exists']:atomic(path,(tx/(name+'.before')).read_bytes())
        elif path.exists():path.unlink()
    journal.unlink();clean_transaction(tx)

def sync_once(config,target=None,reason='MANUAL'):
    destination=P(config['destination']).resolve();destination.mkdir(parents=True,exist_ok=True)
    allowed=set(config['files'])
    if any(P(n).name!=n for n in allowed):raise ValueError('Only flat owned mirror filenames are supported')
    with lock(destination/'.result-sync.lock'):
        recover(destination,allowed)
        known={n:blob((destination/n).read_bytes())for n in allowed if(destination/n).exists()}
        payload=dict(action='publication',repository=config['repository'],branch=config['branch'],files=config['files'],known=known,target=target)
        publication=fixture_publication(config,known,target)if config.get('provider')=='fixture'else remote(config,payload)
        if set(publication['manifest'])!=allowed:raise ValueError('Publication member set differs')
        archive=base64.b64decode(publication['archive_base64'])
        if sha(archive)!=publication['archive_sha256']:raise ValueError('Publication archive hash differs')
        changed=publication['changed'];contents={}
        with zipfile.ZipFile(io.BytesIO(archive))as z:
            if len(z.namelist())!=len(set(z.namelist()))or set(z.namelist())!=set(changed):raise ValueError('Publication archive members differ')
            for n in changed:
                if n not in allowed:raise ValueError('Unsafe publication member')
                raw=z.read(n);row=publication['manifest'][n]
                if blob(raw)!=row['git_blob']or sha(raw)!=row['sha256']:raise ValueError('Publication member hash differs: '+n)
                contents[n]=raw
        for n in allowed-set(changed):
            if known.get(n)!=publication['manifest'][n]['git_blob']:raise ValueError('Untransferred member differs: '+n)
            raw=(destination/n).read_bytes()
            publication['manifest'][n].update(sha256=sha(raw),bytes=len(raw))
        # Stage and validate before replacing any current file. The standalone
        # HTML is installed last; interrupted transactions are rolled back.
        receipt=dict(commit=publication['commit'],branch=config['branch'],files=publication['manifest'],changed=changed,
            transferred_files=len(changed),transfer_bytes=publication['transfer_bytes'],reason=reason,checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),final=publication['final'])
        tx=destination/'.sync-staging'/uuid.uuid4().hex;tx.mkdir(parents=True)
        order=sorted(changed,key=lambda n:n=='index.html')+['LOCAL_SYNC.json'];before=[]
        for n in order:
            path=destination/n;before.append(dict(name=n,exists=path.exists()))
            if path.exists():(tx/(n+'.before')).write_bytes(path.read_bytes())
        save(destination/'.SYNC_TRANSACTION.json',dict(transaction=str(tx),before=before))
        try:
            for n in order:
                if n=='LOCAL_SYNC.json':save(destination/n,receipt)
                else:atomic(destination/n,contents[n])
            for n in allowed:
                if blob((destination/n).read_bytes())!=publication['manifest'][n]['git_blob']:raise ValueError('Installed member differs: '+n)
            (destination/'.SYNC_TRANSACTION.json').unlink();clean_transaction(tx)
        except BaseException:
            recover(destination,allowed);raise
        # Pending records survive an idle manual sync, and are cleared only by
        # an exact final controller result in the C-produced public map.
        pending_path=destination/'FINAL_PUBLICATION_PENDING.json'
        pending=load(pending_path)if pending_path.exists()else{}
        if target:
            if publication['final']['status']=='FINAL_RESULT_PUBLISHED':pending.pop(target['run_id'],None)
            else:pending[target['run_id']]=target
        save(pending_path,pending)
        note='最終結果は公開待ち: '+', '.join(pending)if pending else'公開版を同期済み'
        atomic(destination/'同期状況.txt',(note+'\n公開commit: '+publication['commit']+'\n').encode('utf-8-sig'))
        with(destination/'SYNC_EVENTS.jsonl').open('a',encoding='utf-8')as events:events.write(json.dumps(receipt,ensure_ascii=False)+'\n')
        return receipt

def sync_manual(config,reason='MANUAL'):
    path=P(config['destination'])/'FINAL_PUBLICATION_PENDING.json'
    pending=load(path)if path.exists()else{}
    if not pending:return sync_once(config,reason=reason)
    results=[]
    for target in pending.values():
        if config.get('ssh_alias'):
            try:
                current=observe(config,target)
                if current['state']=='TERMINAL'and current['invocation_id']==target.get('invocation_id'):target=current
            except Exception:pass
        results.append(sync_once(config,target=target,reason=reason+'_PENDING_RECHECK'))
    return results[-1]

def prepare(config,config_path,result):
    if result['status']not in('STARTED','ALREADY_RUNNING','REUSED_COMPLETED')or not result.get('run_directory'):raise ValueError('Run start/resume was not confirmed')
    if result.get('artificial_only')and not config.get('artificial_only'):raise ValueError('Artificial run cannot activate production sync')
    identity='|'.join([config['ssh_alias'],result['run_id'],result['run_directory']]);key=sha(identity.encode())[:20]
    directory=P(config['state_directory'])/'runs'/key;directory.mkdir(parents=True,exist_ok=True)
    request_path=directory/'WATCH_REQUEST.json'
    with lock(directory/'prepare.lock'):
        old=load(request_path)if request_path.exists()else{}
        request=dict(config=str(P(config_path).resolve()),run_id=result['run_id'],run_directory=result['run_directory'],key=key)
        try:observation=observe(config,request)
        except Exception as error:observation=dict(state='UNCONFIRMED',invocation_id=old.get('invocation_id'),error=str(error))
        request.update(invocation_id=observation['invocation_id'],generation=old.get('generation')if old.get('invocation_id')==observation['invocation_id']else uuid.uuid4().hex)
        if not request['generation']:request['generation']=uuid.uuid4().hex
        request['preparing']=True;save(request_path,request)
        # Clear/recheck older pending finals as part of this initial sync too.
        try:
            if observation['state']=='TERMINAL':sync_once(config,target=observation,reason='RUN_FINAL_ALREADY_TERMINAL')
            else:sync_manual(config,reason='RUN_CONFIRMED:'+request['run_id'])
            request['initial_sync_complete']=True
        except Exception as error:
            request.update(initial_sync_complete=False,initial_sync_error=str(error))
        request['preparing']=False;save(request_path,request)
        task_name=config.get('task_prefix','Local0-H4-PublicSync-')+key
        return dict(task_name=task_name,request=str(request_path),config=str(P(config_path).resolve()),state=observation['state'],run_id=request['run_id'],index=str(P(config['destination'])/'index.html'),initial_sync_complete=request['initial_sync_complete'],initial_sync_error=request.get('initial_sync_error'))

def watch(request_path):
    directory=P(request_path).parent
    # Cross-process lock is a second guard in addition to Task Scheduler's
    # IgnoreNew policy. No controller or worker is launched by this monitor.
    with lock(directory/'monitor.lock'):
        observed_generation=None;next_public_check=0
        while True:
            request=load(request_path);config=load(request['config']);generation=request['generation']
            if request.get('preparing'):time.sleep(.2);continue
            if observed_generation!=generation:
                # prepare already performed the immediate sync.
                next_public_check=time.monotonic()+(config['interval_seconds']if request.get('initial_sync_complete')else 0);observed_generation=generation
            try:
                observation=observe(config,request)
                if observation['state']=='TERMINAL':
                    try:final=sync_once(config,target=observation,reason='RUN_FINAL')['final']
                    except Exception as error:
                        final=dict(status='FINAL_PUBLICATION_UNCONFIRMED',message='最終結果の公開確認待ち（最終同期失敗）',error=str(error))
                        destination=P(config['destination']);destination.mkdir(parents=True,exist_ok=True)
                        with lock(destination/'.result-sync.lock'):
                            path=destination/'FINAL_PUBLICATION_PENDING.json';pending=load(path)if path.exists()else{};pending[request['run_id']]=observation;save(path,pending)
                            atomic(destination/'同期状況.txt',(final['message']+'\n').encode('utf-8-sig'))
                    if load(request_path)['generation']!=generation:continue
                    save(directory/'MONITOR_STATE.json',dict(state='STOPPED',run_id=request['run_id'],observation=observation,final=final,periodic_enabled=False))
                    return
                if time.monotonic()>=next_public_check:
                    sync_once(config,reason='RUNNING_PERIODIC')
                    next_public_check=time.monotonic()+config['interval_seconds']
                save(directory/'MONITOR_STATE.json',dict(state='RUNNING',run_id=request['run_id'],periodic_enabled=True,interval_seconds=config['interval_seconds'],invocation_id=observation['invocation_id']))
                interval=config.get('status_poll_seconds',60)
            except Exception as error:
                save(directory/'MONITOR_STATE.json',dict(state='OBSERVATION_OR_SYNC_FAILED',run_id=request['run_id'],error=str(error),periodic_enabled=True))
                interval=min(60,config['interval_seconds'])
            until=time.monotonic()+interval
            while time.monotonic()<until:
                time.sleep(min(2,max(.01,until-time.monotonic())))
                if load(request_path)['generation']!=generation:break

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sync','prepare','watch','status']);p.add_argument('--config');p.add_argument('--result');p.add_argument('--request');a=p.parse_args()
    if a.action=='watch':watch(a.request);return
    config=load(a.config)
    if config.get('interval_seconds')!=600 and not config.get('artificial_only'):raise ValueError('Production interval must be 600 seconds')
    if a.action=='sync':result=sync_manual(config)
    elif a.action=='prepare':result=prepare(config,a.config,load(a.result))
    else:result=[load(x)for x in P(config['state_directory']).glob('runs/*/MONITOR_STATE.json')]
    print(json.dumps(result))

if __name__=='__main__':main()
