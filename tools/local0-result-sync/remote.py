"""Read-only publication and controller observation over the existing SSH route."""
import base64,hashlib,io,json,os,pathlib,subprocess,sys,zipfile
if 'CONFIG' not in globals():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-base64',required=True)
    CONFIG=json.loads(base64.b64decode(parser.parse_args().payload_base64))
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
def read(path):return json.loads(pathlib.Path(path).read_text())
def observe(config):
    indexed_run=pathlib.Path(config['run_directory'])
    run=indexed_run.resolve()
    applied=read(run/'APPLIED_SCOPE.json')
    if applied.get('run_id')!=config['run_id']:raise ValueError('Controller run identity differs')
    launch=read(run/'LAUNCH.json')if(run/'LAUNCH.json').exists()else{}
    exit_=read(run/'EXIT.json')if(run/'EXIT.json').exists()else{}
    invocation=launch.get('invocation_id')
    artifacts={}
    for name in ['ROUND_RESULT.json','ROUND_RESULT_INTERRUPTED.json','EVALUATIONS.json','STATUS.json']:
        path=run/name
        if path.exists()and (not(run/'LAUNCH.json').exists()or path.stat().st_mtime_ns>=(run/'LAUNCH.json').stat().st_mtime_ns):
            raw=path.read_bytes();artifacts[name]=dict(sha256=sha(raw),json_sha256=canonical(json.loads(raw)))
    if invocation and exit_.get('invocation_id')==invocation:
        state='TERMINAL';reason='EXIT_RECEIPT';code=exit_.get('returncode')
    else:
        pid=launch.get('pid');cmd=pathlib.Path('/proc')/str(pid)/'cmdline'
        try:arguments=cmd.read_bytes().split(b'\0') if pid else ()
        except FileNotFoundError:arguments=()
        live=any(os.fsencode(path) in arguments for path in (indexed_run,run))
        state='RUNNING'if live else'TERMINAL'
        reason='OWNED_PROCESS_ALIVE'if live else'STOPPED_WITHOUT_EXIT_RECEIPT'
        code=None
    spec=read(run/'ROUND.json')if(run/'ROUND.json').exists()else{}
    return dict(run_id=config['run_id'],run_directory=str(run),state=state,reason=reason,
        invocation_id=invocation,returncode=code,round_id=spec.get('round_id',config['run_id']),artifacts=artifacts)

def published(config):
    repo=pathlib.Path(config['repository'])
    def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
    origin=git('remote','get-url','origin').decode().strip()
    if origin.removesuffix('.git') not in ['https://github.com/HIROMU1015/Partially-Randomized-Trotter','git@github.com:HIROMU1015/Partially-Randomized-Trotter']:
        raise ValueError('Unexpected publication repository')
    ref='refs/heads/'+config['branch']
    head=git('ls-remote','origin',ref).decode().split()[0]
    if subprocess.run(['git','cat-file','-e',head+'^{commit}'],cwd=repo,capture_output=True).returncode:
        git('fetch','--no-tags','origin',head)
    use_sftp = config.get('transfer_mode') == 'sftp'
    manifest={};buffer=io.BytesIO();changed=[]
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)as archive:
        for name,source in config['files'].items():
            oid=git('rev-parse',head+':'+source).decode().strip()
            manifest[name]=dict(source_path=source,git_blob=oid)
            if config.get('known',{}).get(name)!=oid:
                raw=git('cat-file','blob',oid)
                if not use_sftp:
                    archive.writestr(name,raw)
                changed.append(name)
                manifest[name].update(sha256=sha(raw),bytes=len(raw))
    target=config.get('target');final=dict(status='NOT_REQUESTED')
    if target:
        final=dict(status='FINAL_PUBLICATION_PENDING',message='最終結果は公開待ち',run_id=target['run_id'])
        index=json.loads(git('show',head+':'+config['files']['INDEX.json']))
        for row in index.get('runs',[]):
            if row.get('label')not in(target['run_id'],target.get('round_id'))and row.get('id')!=target['run_id']and not row.get('directory','').endswith('/'+target['run_id']):continue
            # Match final controller bytes/content, rather than accepting a
            # stale RUNNING row or just the presence of the same run label.
            refs=[row.get('source',{}),*row.get('scope_sources',[])]
            for locator in refs:
                if pathlib.PurePosixPath(locator.get('path','')).name not in ('ROUND_RESULT.json','ROUND_RESULT_INTERRUPTED.json'):continue
                name=pathlib.PurePosixPath(locator['path']).name
                expected=target.get('artifacts',{}).get(name)
                if not expected:continue
                try:
                    commit=locator['commit'];path=pathlib.PurePosixPath(locator['path'])
                    if len(commit)!=40 or any(c not in '0123456789abcdef'for c in commit)or not str(path).startswith('docs/research/')or '..'in path.parts:continue
                    if subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=repo,capture_output=True).returncode:
                        git('fetch','--no-tags','origin',commit)
                    raw=git('show',locator['commit']+':'+locator['path'])
                    value=json.loads(raw)
                    for key in locator.get('selector',[]):value=value[key]
                    if sha(raw)!=expected['sha256']and canonical(value)!=expected['json_sha256']:continue
                    final=dict(status='FINAL_RESULT_PUBLISHED',message='最終結果を公開版で確認',run_id=target['run_id'],source=locator,issues=row.get('issues',[]),missing=row.get('missing',[]))
                except (KeyError,ValueError,subprocess.CalledProcessError):continue
    if use_sftp:
        return dict(commit=head,branch=config['branch'],manifest=manifest,changed=changed,
                    transfer_mode='sftp',transfer_bytes=sum(manifest[n]['bytes'] for n in changed),final=final)
    data=buffer.getvalue()
    return dict(commit=head,branch=config['branch'],manifest=manifest,changed=changed,
        archive_base64=base64.b64encode(data).decode(),archive_sha256=sha(data),transfer_bytes=len(data)if changed else 0,
        final=final)

try:
    result=observe(CONFIG)if CONFIG['action']=='run_status'else published(CONFIG)
    print(json.dumps(dict(ok=True,result=result),ensure_ascii=False))
except Exception as error:
    print(json.dumps(dict(ok=False,error=str(error)),ensure_ascii=False))
    raise SystemExit(1)
