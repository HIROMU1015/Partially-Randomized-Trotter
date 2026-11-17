"""Common operation transport to the existing central Gateway.

This module owns no scientific queue, optimizer, release registry or results.
Remote clients forward the same envelope as the Windows operation client.
"""
from pathlib import Path
import base64
import json
import os
import shlex
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ACTIONS=('preview','planonly','prepare','start','status','stop','resume')
REMOTE_SOCKET_PREPARE='''from pathlib import Path
import os,socket,stat,sys
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
directory=p.parent.stat()
if directory.st_uid!=os.getuid() or directory.st_mode&0o077:
 raise RuntimeError('Operation socket directory must be owned and private')
try:kind=p.lstat().st_mode
except FileNotFoundError:kind=None
if kind is not None:
 if not stat.S_ISSOCK(kind):raise RuntimeError('Refusing to replace a non-socket file')
 s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.settimeout(.5)
 try:s.connect(str(p))
 except (ConnectionRefusedError,FileNotFoundError):p.unlink(missing_ok=True)
 else:raise RuntimeError('Existing operation socket is active; preserve its owner')
 finally:s.close()
'''


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def envelope(config,action,request=b'{}',*,run_id=None,release=None,instruction=None):
    action=action.lower()
    if action not in ACTIONS:raise ValueError('Unsupported operation action: '+action)
    value=dict(server_config=config['server_config'],request_base64=base64.b64encode(request).decode(),action=action)
    for key,item in [('run_id',run_id),('release',release),('instruction',instruction)]:
        if item is not None:value[key]=item
    return value


def _checked_response(raw):
    response=json.loads(raw)
    if response.get('ok') is not True:raise RuntimeError(response.get('error','Central Gateway rejected the request'))
    return response['result']


def invoke_endpoint(config,value):
    """Use a fresh process so a registry switch cannot retain old imports."""
    encoded=base64.b64encode(json.dumps(value,ensure_ascii=False).encode()).decode()
    command=[config['python'],'-B',config['remote_entry'],'--transport-base64',encoded]
    process=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,
                           timeout=config.get('request_timeout_seconds',120))
    if process.returncode:
        raise RuntimeError('Central bootstrap failed: '+process.stderr.decode(errors='replace')[-1500:]+
                           process.stdout.decode(errors='replace')[-1500:])
    return json.loads(process.stdout)


def invoke(config,value):
    transport=config.get('transport','ssh')
    if transport=='local':return _checked_response(json.dumps(invoke_endpoint(config,value)))
    if transport=='unix':
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)as client:
            client.settimeout(config.get('request_timeout_seconds',120));client.connect(config['socket_path'])
            client.sendall(json.dumps(value,ensure_ascii=False).encode()+b'\n')
            with client.makefile('rb')as stream:raw=stream.readline()
        return _checked_response(raw)
    if transport not in ('ssh','windows-ssh'):raise ValueError('Unsupported client transport: '+transport)
    encoded=base64.b64encode(json.dumps(value,ensure_ascii=False).encode()).decode()
    remote='set -o pipefail; '+shlex.join([config['python'],'-B',config['remote_entry'],
        '--transport-base64',encoded])+' | base64 -w 1024'
    ssh=list(config.get('ssh_command')or[config.get('ssh_executable','ssh')])
    arguments=['-n','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
               '-o','ServerAliveCountMax=2',config['ssh_alias'],remote]
    if transport=='windows-ssh':
        # WSL uses the existing Windows SSH identity/config through interop.
        # No private key is copied to Linux or to a deployment artifact.
        quote=lambda item:"'"+str(item).replace("'","''")+"'"
        script="$ErrorActionPreference='Stop';$env:HOME=$env:USERPROFILE;"+\
            "$a=@("+','.join(quote(x)for x in ssh[1:]+arguments)+");$r=& "+quote(ssh[0])+\
            " @a;if($LASTEXITCODE -ne 0){exit $LASTEXITCODE};$r"
        command=[config.get('powershell_executable','powershell.exe'),'-NoProfile','-NonInteractive',
                 '-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()]
    else:command=ssh+arguments
    process=subprocess.run(command,stdin=subprocess.DEVNULL,capture_output=True,
                           timeout=config.get('request_timeout_seconds',120))
    if process.returncode:raise RuntimeError('Central operation route unavailable: '+process.stderr.decode(errors='replace')[-1500:])
    return _checked_response(base64.b64decode(process.stdout))


def operate(config,action,request=b'{}',*,run_id=None,release=None,instruction=None,install=True):
    """Resolve centrally, check missing local code, then pin that same version."""
    value=envelope(config,action,request,run_id=run_id,release=release,instruction=instruction)
    query=dict(value,operation='resolve',action='preview')
    if config.get('host_id'):query['host_id']=config['host_id']
    selected=invoke(config,query)  # Legacy bootstraps also perform only Preview.
    manifest=selected.get('components')or selected.get('component_manifest')
    requested=json.loads(request)
    deployment=None
    if action.lower()in('preview','planonly','prepare','start'):
        layout=selected.get('host_layout')
        if layout is None and config.get('deployment_layout'):layout=read(config['deployment_layout'])
        if manifest and layout is not None:
            from .rpe_release_deployment import ensure_components
            deployment=ensure_components(manifest,layout,install=install,components=['optimizer'])
            if deployment['status']!='PASS':raise RuntimeError('Selected release code is unavailable: '+json.dumps(deployment['holds']))
        elif config.get('require_components')and requested.get('scope'):
            raise RuntimeError('Selected release has no public component manifest; new-run code placement is HOLD')
    resolved=selected.get('release')or{}
    if not value.get('release')and requested.get('scope')and resolved.get('id'):value['release']=resolved['id']
    if action.lower()in('start','resume'):
        value['invoke_result_sync']=True
        if not value.get('instruction'):value['instruction']='Explicit user '+action.lower()+' command'
    result=invoke(config,value)
    if deployment is not None:result['client_deployment']=deployment
    result['operation_controller']=selected.get('controller_host','RomuTail')
    return result


class OperationGateway:
    """Owned UNIX socket and SSH reverse forwards; all decisions stay central."""
    def __init__(self,configuration):
        self.configuration=Path(configuration).resolve();self.config=read(self.configuration)
        self.directory=Path(self.config['state_directory']).resolve()
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.socket=Path(self.config['socket_path']).resolve()
        if not self.socket.is_relative_to(self.directory):raise ValueError('Gateway socket must be inside its owned state directory')
        self.children={};self.forward_status={};self.pending_forwards={}
        self.forward_pool=ThreadPoolExecutor(max_workers=max(1,len(self.config.get('forwards',[]))))

    def endpoint(self,value):
        # A client cannot redirect the central server to another bootstrap.
        value=dict(value,server_config=self.config['server_config'])
        return invoke_endpoint(self.config,value)

    def _start_forward(self,item,log):
        remote=item['remote_socket']
        ssh=list(item.get('ssh_command',['ssh']))
        setup=ssh+['-n','-o','BatchMode=yes','-o','ConnectTimeout=10',item['ssh_alias'],
            shlex.join([item.get('setup_python','python3'),'-c',REMOTE_SOCKET_PREPARE,remote])]
        subprocess.run(setup,stdin=subprocess.DEVNULL,stdout=log,stderr=log,timeout=15,check=True)
        command=ssh+['-N','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
            '-o','ExitOnForwardFailure=yes','-o','StreamLocalBindUnlink=yes',
            '-R',remote+':'+str(self.socket),item['ssh_alias']]
        return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log),command

    def _forwards(self,log):
        now=time.time()
        for item in self.config.get('forwards',[]):
            host=item['host_id'];state=self.forward_status.setdefault(host,{})
            if not item.get('enabled',True):state.update(status='DISABLED');continue
            pending=self.pending_forwards.get(host)
            if pending is not None:
                if not pending.done():continue
                del self.pending_forwards[host]
                try:
                    process,command=pending.result();self.children[host]=process
                    state.update(status='FORWARD_STARTING',pid=process.pid,command=command)
                except Exception as error:state.update(status='ROUTE_PENDING',error=str(error))
                state['retry_epoch']=now+self.config.get('retry_seconds',5)
            process=self.children.get(host)
            if process is not None and process.poll()is None:
                state.update(status='FORWARD_PROCESS_RUNNING',pid=process.pid);continue
            if now<state.get('retry_epoch',0):continue
            if process is not None:state['last_exit']=process.returncode
            state.update(status='FORWARD_CONNECTING')
            self.pending_forwards[host]=self.forward_pool.submit(self._start_forward,item,log)

    def serve(self):
        from run_rpe_q3_relay import acquire_lock
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        with(self.directory/'LOCK').open('a+')as lock:
            acquire_lock(lock)
            if self.socket.exists():
                if not stat.S_ISSOCK(self.socket.stat().st_mode):raise ValueError('Refusing to replace a non-socket file')
                self.socket.unlink()
            owner=self
            class Handler(socketserver.StreamRequestHandler):
                def handle(self):
                    try:response=owner.endpoint(json.loads(self.rfile.readline()))
                    except Exception as error:response=dict(ok=False,error=type(error).__name__+': '+str(error))
                    self.wfile.write(json.dumps(response,ensure_ascii=False).encode()+b'\n')
            class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
            server=Server(str(self.socket),Handler);server.timeout=.5;os.chmod(self.socket,0o600)
            try:
                with(self.directory/'GATEWAY.log').open('ab')as log:
                    while not(self.directory/'STOP').exists():
                        self._forwards(log)
                        write(self.directory/'STATUS.json',dict(pid=os.getpid(),configuration=str(self.configuration),
                            socket_path=str(self.socket),stopped=False,heartbeat_epoch=time.time(),forwards=self.forward_status))
                        server.handle_request()
            finally:
                server.server_close()
                self.forward_pool.shutdown(wait=True,cancel_futures=True)
                for host,pending in self.pending_forwards.items():
                    if pending.cancelled():continue
                    try:self.children[host]=pending.result()[0]
                    except Exception:pass
                for process in self.children.values():
                    if process.poll()is None:
                        process.terminate()
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
                self.socket.unlink(missing_ok=True)
                write(self.directory/'STATUS.json',dict(pid=os.getpid(),configuration=str(self.configuration),
                    stopped=True,heartbeat_epoch=time.time(),forwards=self.forward_status))


def gateway_status(configuration):
    config=read(configuration);path=Path(config['state_directory'])/'STATUS.json'
    value=read(path)if path.exists()else dict(stopped=True,status='NOT_STARTED')
    pid=value.get('pid')
    try:value['process_alive']=str(value.get('configuration',Path(configuration).resolve())).encode()in(Path('/proc')/str(pid)/'cmdline').read_bytes()
    except OSError:value['process_alive']=False
    return value


def start_gateway(configuration,script):
    from run_rpe_q3_relay import background_options
    previous=gateway_status(configuration)
    if previous.get('process_alive')and not previous.get('stopped'):
        command=(Path('/proc')/str(previous['pid'])/'cmdline').read_bytes()
        if previous.get('configuration')!=str(Path(configuration).resolve())or str(Path(script).resolve()).encode()not in command:
            raise ValueError('An existing operation Gateway owns this directory; stop it before changing its fixed entry/configuration')
        return dict(status='ALREADY_RUNNING',**previous)
    config=read(configuration);directory=Path(config['state_directory']);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    (directory/'STOP').unlink(missing_ok=True)
    with(directory/'GATEWAY.log').open('ab')as log:
        process=subprocess.Popen([sys.executable,'-B',str(script),'--serve','--gateway-config',str(Path(configuration).resolve())],
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,**background_options())
    return dict(status='STARTING',pid=process.pid,configuration=str(Path(configuration).resolve()))
