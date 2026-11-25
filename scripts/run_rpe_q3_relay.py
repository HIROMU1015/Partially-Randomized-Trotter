#!/usr/bin/env python3
"""Recover and dispatch fixed worker mailboxes without shared storage.

The controller owns candidate selection and result adoption. It may run on the
same host as this relay; SSH mailboxes can be Linux or Windows-hosted WSL.
Stopping this relay leaves all received jobs running independently.
"""
import argparse
import ast
import base64
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing, nullcontext
import threading
try:
    import fcntl
except ImportError:  # Native Windows local0; numerical workers remain Linux.
    fcntl = None
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.rpe_q3_coordinator import atomic_json
from trottertracks.rpe_q3_timing import stamp


def acquire_lock(stream):
    """One local relay process, using the platform's ordinary file lock."""
    if fcntl is not None:
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return
    import msvcrt
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write('\0'); stream.flush()
    stream.seek(0)
    try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error: raise BlockingIOError('Another relay owns this lock') from error


def background_options(platform_name=None):
    if (platform_name or os.name) == 'nt':
        return dict(creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0x8) |
                    getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200))
    return dict(start_new_session=True)


def child_options(platform_name=None):
    """SSH must stay invisible when its parent relay has no Windows console."""
    if (platform_name or os.name) == 'nt':
        return dict(creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))
    return {}


def ssh_wsl_command(worker, command, action, *, input_file=None, inline_input=None, timeout_seconds=60):
    """Windows SSH -> existing WSL mailbox, preserving JSON stdin/stdout.

    Encoded PowerShell avoids CMD quoting Linux paths or the Japanese Windows
    participation path. File-input requests use a temporary file in the SSH
    user's home and remove it on return. Authentication settings are reused.
    """
    argv = [worker.get('wsl_executable', 'wsl.exe'), '-d', worker['distribution'], '--exec'] + command
    literals = ','.join("'" + str(value).replace("'", "''") + "'" for value in argv[1:])
    executable = "'" + argv[0].replace("'", "''") + "'"
    script = (
        "$ErrorActionPreference='Stop';"
        "$OutputEncoding=New-Object System.Text.UTF8Encoding($false);"
        "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);"
    )
    if worker.get('require_running_wsl'):
        distro = "'" + worker['distribution'].replace("'", "''") + "'"
        script += (
            "$running=@(& " + executable + " --list --running --quiet);"
            "if($LASTEXITCODE -ne 0){exit $LASTEXITCODE};"
            "$running=@($running|ForEach-Object{($_ -replace '\x00','').Trim()});"
            "if(" + distro + " -notin $running){throw 'WSL distribution is stopped'};"
        )
    if input_file is None:
        script += (
            "$raw=[Console]::In.ReadToEnd();"
            "$mailboxArgs=@(" + literals + ");"
            "$lines=$raw | & " + executable + " @mailboxArgs;"
            "if($LASTEXITCODE -ne 0){exit $LASTEXITCODE};"
            "$value=($lines -join [Environment]::NewLine)|ConvertFrom-Json;"
        )
    else:
        # SCP already placed the payload on a mounted Windows drive. Open it
        # inside WSL, avoiding a second bounded Windows -> WSL stdin pipe.
        filename="'"+input_file.replace("'","''")+"'"
        launch=['/bin/sh','-c','path=$(wslpath -u "$0") || exit; exec "$@" < "$path"',
                '__RPE_INPUT_FILE__',*command]
        arguments="'"+subprocess.list2cmdline(argv[1:4]+launch).replace("'","''")+"'"
        write_input = ""
        if inline_input is not None:
            payload = base64.b64encode(inline_input).decode('ascii')
            write_input = "[IO.File]::WriteAllBytes($inputPath,[Convert]::FromBase64String('" + payload + "'));"
        script += (
            # An SSH timeout on RomuTail does not terminate Windows children.
            # Keep the deadline here so the launcher and temporary input are
            # cleaned up while an already accepted attempt remains in its spool.
            "$limit="+str(max(1, int((float(timeout_seconds) - 2) * 1000)))+";"
            "$clock=[Diagnostics.Stopwatch]::StartNew();"
            "function LeftMs{[int][Math]::Max(0,$limit-$clock.ElapsedMilliseconds)};"
            "function WslRun($a,[switch]$optional){"
            "$si=New-Object Diagnostics.ProcessStartInfo;"
            "$si.FileName="+executable+";$si.Arguments=$a;"
            "$si.UseShellExecute=$false;$si.CreateNoWindow=$true;"
            "$si.RedirectStandardOutput=$true;$si.RedirectStandardError=$true;"
            "$utf8=New-Object System.Text.UTF8Encoding($false);"
            "$si.StandardOutputEncoding=$utf8;$si.StandardErrorEncoding=$utf8;"
            "$proc=[Diagnostics.Process]::Start($si);"
            "$stdout=$proc.StandardOutput.ReadToEndAsync();$stderr=$proc.StandardError.ReadToEndAsync();"
            "if(!$proc.WaitForExit((LeftMs))){try{$proc.Kill()}catch{};throw 'WSL launcher timed out'};"
            "if(!$stdout.Wait((LeftMs)) -or !$stderr.Wait((LeftMs))){throw 'WSL output timed out'};"
            "$text=$stdout.GetAwaiter().GetResult();$errorText=$stderr.GetAwaiter().GetResult();"
            "if($proc.ExitCode -ne 0){if($optional){return ''};"
            "[Console]::Out.Write($text);[Console]::Error.Write($errorText);exit $proc.ExitCode};"
            "return $text};"
            "$inputPath=Join-Path $env:USERPROFILE "+filename+";try{"
            + write_input +
            "$a=("+arguments+").Replace('__RPE_INPUT_FILE__',('\"'+$inputPath+'\"'));"
            "$value=(WslRun $a)|ConvertFrom-Json;"
            "}finally{Remove-Item -LiteralPath $inputPath -Force -ErrorAction SilentlyContinue};"
        )
    if action == 'status' and (worker.get('participation_file') or worker.get('participation_wsl_file')):
        if worker.get('participation_wsl_file'):
            flag_args = ['-d', worker['distribution'], '--exec', 'cat', worker['participation_wsl_file']]
            if input_file is not None:
                flag = subprocess.list2cmdline(flag_args).replace("'", "''")
                script += ("$enabled=$false;try{$flag=(WslRun '"+flag+"' -Optional)|ConvertFrom-Json;"
                           "$enabled=($flag.accepting_new_jobs -eq $true)}catch{};")
            else:
                flag_literals = ','.join("'" + str(value).replace("'", "''") + "'" for value in flag_args)
                script += (
                    "$enabled=$false;$flagArgs=@(" + flag_literals + ");"
                    "$flagLines=& " + executable + " @flagArgs 2>$null;"
                    "if($LASTEXITCODE -eq 0){"
                    "$flag=($flagLines -join [Environment]::NewLine)|ConvertFrom-Json;"
                    "$enabled=($flag.accepting_new_jobs -eq $true)};"
                )
        else:
            path = "'" + worker['participation_file'].replace("'", "''") + "'"
            script += (
                "$enabled=$false;"
                "if(Test-Path -LiteralPath " + path + "){"
                "$flag=Get-Content -LiteralPath " + path + " -Raw -Encoding UTF8|ConvertFrom-Json;"
                "$enabled=($flag.accepting_new_jobs -eq $true)};"
            )
        script += (
            "if($value.accepting_new_jobs -eq $false){$enabled=$false};"
            "$value|Add-Member -Force -NotePropertyName accepting_new_jobs -NotePropertyValue $enabled;"
            "if(-not $enabled){$value.slots=0};"
        )
    script += "$value|ConvertTo-Json -Depth 100 -Compress;"
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    return 'powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + encoded


def scp_prefix(worker, ssh):
    """Reuse the SSH identity/options; SCP uses -P and user@host instead."""
    if worker.get('scp_command'):
        return list(worker['scp_command']), worker.get('scp_alias',worker['ssh_alias'])
    executable=Path(ssh[0])
    command=[str(executable.with_name('scp.exe'if executable.name.lower().endswith('.exe')else'scp'))]
    alias=worker.get('scp_alias',worker['ssh_alias']);index=1
    while index<len(ssh):
        value=ssh[index]
        if value=='-l':alias=ssh[index+1]+'@'+alias;index+=2;continue
        if value=='-n':index+=1;continue
        command.append('-P'if value=='-p'else value)
        index+=1
    return command,alias


# Use the existing remote worker validation and durable spool for each request.
# One SSH call carries the list while each receipt keeps its own job and attempt.
MAILBOX_SUBMIT_BATCH = """import hashlib, json, runpy, sys
from pathlib import Path
worker_script, profile, *compatibility = sys.argv[1:]
api = runpy.run_path(worker_script)
selected = api['read'](profile)
assignments = json.load(sys.stdin)
module = sys.modules[api['mailbox'].__module__]
submit_batch = getattr(module, 'mailbox_submit_batch', None)
batch_mode = 'NATIVE'
if submit_batch is None and compatibility:
    expected, source = compatibility
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected:
        namespace = vars(module).copy()
        exec(compile(source, '<fixed-mailbox-batch>', 'exec'), namespace)
        candidate = namespace.get('mailbox_submit_batch')
        if callable(candidate) and callable(namespace.get('_mailbox_submit')):
            submit_batch = candidate
            batch_mode = 'FIXED_COMPATIBILITY'
if submit_batch is not None:
    reply = submit_batch(selected, assignments)
else:
    batch_mode = 'LEGACY'
    reply = []
    for assignment in assignments:
        try:
            result = api['mailbox'](selected, 'submit', assignment)
        except Exception as error:
            result = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
        reply.append(dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], result=result))
for item in reply:
    item['batch_mode'] = batch_mode
print(json.dumps(reply, separators=(',', ':'), allow_nan=False))
"""

# Windows OpenSSH launches PowerShell through a command line with a practical
# length limit. Keep the batch program and fixed compatibility text in the
# same staged input as its assignments.
WSL_BATCH_BOOTSTRAP = ("import io,json,sys,contextlib; envelope=json.load(sys.stdin); output=io.StringIO(); "
                       "sys.argv=envelope['argv']; sys.stdin=io.StringIO(json.dumps(envelope['assignments'])); "
                       "exec(\"with contextlib.redirect_stdout(output): exec(envelope['program'])\"); "
                       "print(json.dumps({'rows':json.loads(output.getvalue())}))")

class Relay:
    def __init__(self, configuration):
        self.configuration = Path(configuration).resolve()
        self.config = json.loads(self.configuration.read_text(encoding='utf-8-sig'))
        self.directory = Path(self.config['state_directory']).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ssh = self.config.get('ssh_command', ['ssh'])
        self.ssh_socket_directory = None
        self.readiness_connections = None
        self.controller = self.config['controller']
        self.poll_seconds = float(self.config.get('poll_seconds', 1))
        maximum = float(self.config.get('retry_backoff_max_seconds', 30))
        if self.poll_seconds <= 0 or maximum <= 0:
            raise ValueError('Positive relay poll and retry intervals required')
        self.retry_max_seconds = max(self.poll_seconds, maximum)
        self.url = 'http://127.0.0.1:'+str(self.controller.get('local_port', 18763))
        self.tunnel = None
        self.discovery_pool = None
        self.pending_discovery = {}
        self.pending_ticks = {}
        self.recovery_pool = None
        self.pending_recoveries = {worker['host_id']: {} for worker in self.config['workers']}
        self.recovery_width = 2
        windows_workers = sum(item.get('transport') == 'ssh-wsl' for item in self.config['workers'])
        self.bundle_pool = (ThreadPoolExecutor(max_workers=windows_workers, thread_name_prefix='rpe-wsl-bundles')
                            if windows_workers else None)
        self.pending_bundles = {}
        self.log_lock = threading.Lock()
        self.mailbox_locks = {item['host_id']: threading.Lock() for item in self.config['workers']}
        self.deployed_workers={}
        self.reported_unowned = set()
        self.readiness_probe_counts = {}
        self.bundle_prefetch_counts = {}
        started = time.time()
        self.state = dict(pid=os.getpid(), configuration=str(self.configuration), workers={},
                          started_epoch=started, heartbeat_epoch=started, stopped=False,
                          transport='controller-local or SSH forwarding + independent worker mailboxes')

    def log(self, event, **value):
        row = dict(stamp(), event=event, **value)
        with self.log_lock:
            with (self.directory/'EVENTS.jsonl').open('a') as output: output.write(json.dumps(row)+'\n')

    def post(self, path, value):
        payload = json.dumps(value).encode()
        request = urllib.request.Request(self.url+path, payload, {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.config.get('request_timeout_seconds', 60)) as response:
                return json.load(response)
        except TimeoutError as error:
            raise TimeoutError('Controller RPC '+path+' timed out') from error

    def failed(self, state, error, event, **detail):
        failures = state.get('failure_count', 0)+1
        delay = self.poll_seconds
        for _ in range(failures-1):
            delay = min(self.retry_max_seconds, delay*2)
            if delay >= self.retry_max_seconds: break
        state.update(error=str(error), failure_count=failures, last_error_epoch=time.time(),
                     next_retry_epoch=time.time()+delay, retry_seconds=delay)
        self.log(event, error=str(error), failure_count=failures, next_retry_epoch=state['next_retry_epoch'],
                 retry_seconds=delay, **detail)

    def succeeded(self, state):
        state.update(failure_count=0, next_retry_epoch=0, retry_seconds=0)
        state.pop('error', None)

    def remote(self, worker, action, value=None):
        if worker.get('deployment')and(action=='status'or action=='submit'):
            self.ensure_worker_deployment(worker,value if action=='submit'else None)
        if action=='readiness' and (worker.get('deployment') or worker.get('task_deployments')):
            deployment={key:worker[key] for key in ('deployment','task_deployments') if key in worker}
            value=dict(value, deployment=deployment)
        command = [worker.get('python', 'python3'), '-B', worker['root'].rstrip('/')+'/scripts/run_rpe_q3_worker.py',
                   '--profile', worker['profile'], '--mailbox', action]
        return self.remote_command(worker,command,action,value)

    def ensure_worker_deployment(self,worker,assignment=None):
        """Install only missing selected code, then run existing task guards."""
        declaration=worker['deployment'];host=worker['host_id']
        if assignment is not None:
            release = (assignment.get('required_release') or {}).get('id')
            declaration = worker.get('task_deployments', {}).get(release, declaration)
        if assignment is None and self.deployed_workers.get(host)==declaration:return
        command=[worker.get('python','python3'),'-B',declaration['entry'],
            '--manifest',declaration['manifest'],'--layout',declaration['layout'],
            '--install','--components','optimizer']
        if assignment is not None:command+=['--assignment-stdin','--profile',worker['profile']]
        result,_=self.remote_command(worker,command,'deployment',assignment or{})
        if result.get('status')!='PASS':raise ValueError('Worker release deployment HOLD: '+json.dumps(result))
        if assignment is None:
            self.deployed_workers[host]=dict(declaration)
        return result

    def submit_batch(self, worker, assignments):
        if worker.get('deployment'):
            for assignment in assignments:
                self.ensure_worker_deployment(worker, assignment)
        command = [worker.get('python', 'python3'), '-B', '-c', MAILBOX_SUBMIT_BATCH,
                   worker['root'].rstrip('/')+'/scripts/run_rpe_q3_worker.py', worker['profile']]
        if worker_sha := self.config.get('batch_worker_sources', {}).get(worker['host_id']):
            if not hasattr(self, 'batch_compatibility_source'):
                path = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_q3_worker.py'
                source = path.read_text()
                functions = {'_mailbox_submit', 'mailbox_submit_batch'}
                definitions = [node for node in ast.parse(source).body
                               if isinstance(node, ast.FunctionDef) and node.name in functions]
                self.batch_compatibility_source = ('\n\n'.join(ast.get_source_segment(source, node)
                                                   for node in definitions)
                                                   if {node.name for node in definitions} == functions else '')
            if self.batch_compatibility_source:
                command.extend((worker_sha, self.batch_compatibility_source))
        payload = assignments
        if worker.get('transport') == 'ssh-wsl':
            payload = dict(program=command[3], argv=command[3:], assignments=assignments)
            command = command[:3] + [WSL_BATCH_BOOTSTRAP]
        rows, seconds = self.remote_command(worker, command, 'submit-batch', payload)
        if worker.get('transport') == 'ssh-wsl':
            rows = rows['rows']
        expected = [(a['job_id'], a['attempt_id']) for a in assignments]
        if [(r['job_id'], r['attempt_id']) for r in rows] != expected:
            raise ValueError('Worker batch response differs from the assigned attempts')
        self.log('MAILBOX_BATCH_MODE', host_id=worker['host_id'], count=len(rows),
                 mode=rows[0].get('batch_mode', 'LEGACY') if rows else 'EMPTY')
        return [r['result'] for r in rows], seconds

    def remote_command(self,worker,command,action,value=None):
        # Only spool-owner operations serialize. Result bytes and other hosts
        # can continue while a submission or acknowledgement owns this spool.
        lock = (self.mailbox_locks.get(worker.get('host_id'))
                if action in ('ack', 'status', 'resources', 'submit', 'submit-batch') else None)
        started = time.monotonic()
        with lock if lock is not None else nullcontext():
            result = self._remote_command(worker, command, action, value)
        return result, time.monotonic() - started

    def _remote_command(self,worker,command,action,value=None):
        # Reuse local0's published WSL mailbox transport (6e70f5d13061).
        # Scientific execution still resolves the route-specific saved profile.
        if worker.get('transport') == 'wsl':
            if worker.get('require_running_wsl'):
                running = subprocess.check_output(
                    [worker.get('wsl_executable', 'wsl.exe'), '--list', '--running', '--quiet'],
                    timeout=10, **child_options())
                names = running.decode('utf-8', errors='replace').replace('\x00', '').splitlines()
                if worker['distribution'].casefold() not in [n.strip('\ufeff ').casefold() for n in names]:
                    raise RuntimeError('WSL distribution is stopped')
            command = [worker.get('wsl_executable', 'wsl.exe'), '-d', worker['distribution'], '--exec'] + command
        elif worker.get('transport') == 'ssh-wsl':
            try:
                process=self._remote_wsl_file(worker,command,action,value or{})
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f'WSL mailbox {action} timed out after {error.timeout:g} seconds') from None
            command=None
        else:
            alias = worker['ssh_alias']
            if not alias or alias.startswith('-'): raise ValueError('Invalid existing SSH alias')
            limit = worker.get('cpu_limit')
            if limit is not None:
                if type(limit) is not int or limit < 1:
                    raise ValueError('Expected a positive CPU limit')
                # All mailbox calls, including their detached scientific
                # children, inherit the same subset of the allowed CPUs.
                command = [worker.get('python', 'python3'), '-B', '-c',
                    'import os,sys; '
                    'os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:int(sys.argv[1])]); '
                    'os.execvp(sys.argv[2],sys.argv[2:])',
                    str(limit), *command]
            remote_command = shlex.join(command)
            ssh = worker.get('ssh_command', self.ssh)
            reuse = []
            if self.readiness_connections is not None:
                reuse = self.readiness_connections.options(ssh, alias)
            elif self.ssh_socket_directory is not None and Path(ssh[0]).name == 'ssh':
                reuse = ['-o', 'ControlMaster=auto', '-o', 'ControlPersist=15',
                         '-o', 'ControlPath='+self.ssh_socket_directory.name+'/%C']
            command = [*ssh, *reuse, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', alias, remote_command]
        if command is not None:
            process = subprocess.run(command,
                input=json.dumps(value or {}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding='utf-8', errors='replace',
                timeout=self.config.get('request_timeout_seconds', 60), **child_options())
        if process.returncode:
            # Prefer the worker error to incidental Windows SSH diagnostics.
            # Never echo arbitrary stdout (it may contain an input bundle).
            detail = process.stderr.strip()[-1500:]
            if not detail or (worker.get('transport') == 'ssh-wsl' and process.returncode == 2):
                try:
                    response = json.loads(process.stdout)
                    error = response.get('error') if isinstance(response, dict) else None
                except (ValueError, TypeError):
                    error = None
                if isinstance(error, str) and error.strip():
                    detail = error[:1500]
            detail = ''.join(c if c.isprintable() else ' ' for c in detail)
            raise RuntimeError('Worker mailbox '+action+' failed ('+str(process.returncode)+'): '+detail)
        result = json.loads(process.stdout)
        if action == 'status' and worker.get('participation_file') and worker.get('transport') != 'ssh-wsl':
            participation = Path(worker['participation_file'])
            enabled = (result.get('accepting_new_jobs', True) and participation.is_file()
                       and json.loads(participation.read_text(encoding='utf-8-sig')).get('accepting_new_jobs') is True)
            result['accepting_new_jobs'] = enabled
            if not enabled:
                # Completion transfer and reconciliation above the poll continue.
                result['slots'] = 0
        return result

    def _remote_wsl_file(self,worker,command,action,value):
        ssh=list(worker.get('ssh_command',self.ssh));alias=worker['ssh_alias']
        if not alias or alias.startswith('-'):raise ValueError('Invalid existing SSH alias')
        scp,target=scp_prefix(worker,ssh)
        name='.rpe-mailbox-'+uuid.uuid4().hex+'.json'
        timeout=self.config.get('request_timeout_seconds',60)
        payload_bytes = json.dumps(value).encode('utf-8')
        ssh += ['-n','-o','BatchMode=yes','-o','ConnectTimeout=10',alias]
        # Small poll and receipt messages fit in a single Windows SSH call.
        # Keep larger requests below the Windows command-line length limit
        # by using the existing file transfer.
        if action in ('status', 'result', 'ack', 'resources') and len(payload_bytes) <= 600:
            remote = ssh_wsl_command(worker, command, action, input_file=name, inline_input=payload_bytes,
                                     timeout_seconds=timeout)
            if len(remote) < 7000:
                return subprocess.run(ssh+[remote], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, encoding='utf-8', errors='replace', timeout=timeout, **child_options())
        with tempfile.TemporaryDirectory(prefix='rpe-mailbox-')as temporary:
            payload=Path(temporary)/'request.json'
            payload.write_bytes(payload_bytes)
            transfer=subprocess.run(scp+['-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(payload),target+':'+name],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                encoding='utf-8',errors='replace',timeout=timeout,**child_options())
            if transfer.returncode:raise RuntimeError('WSL mailbox input transfer failed: '+transfer.stderr[-1500:])
            remote=ssh_wsl_command(worker,command,action,input_file=name,timeout_seconds=timeout)
            return subprocess.run(ssh+[remote],
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                encoding='utf-8',errors='replace',timeout=timeout,**child_options())

    def ensure_tunnel(self):
        if self.config.get('external_tunnel'): return True
        state = self.state.setdefault('tunnel', dict(failure_count=0, next_retry_epoch=0))
        if self.tunnel is not None:
            code = self.tunnel.poll()
            if code is None:
                # A process that survives several normal polls is healthy;
                # an immediate exit must not reset its retry delay each time.
                if time.time()-state['started_epoch'] >= max(5, 2*self.poll_seconds):
                    self.succeeded(state)
                return True
            self.tunnel = None
            self.failed(state, 'SSH tunnel exited ('+str(code)+')', 'TUNNEL_PENDING', returncode=code)
        if time.time() < state.get('next_retry_epoch', 0): return False
        alias = self.controller.get('ssh_alias', 'qg2')
        if not alias or alias.startswith('-'): raise ValueError('Invalid controller SSH alias')
        local_port = int(self.controller.get('local_port', 18763))
        remote_port = int(self.controller.get('remote_port', 8763))
        try:
            with (self.directory/'TUNNEL.log').open('a') as log:
                self.tunnel = subprocess.Popen(self.ssh+['-N', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                    '-L', '127.0.0.1:%d:127.0.0.1:%d' % (local_port, remote_port), alias],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **child_options())
        except OSError as error:
            self.failed(state, error, 'TUNNEL_PENDING')
            return False
        state.update(pid=self.tunnel.pid, started_epoch=time.time())
        self.log('TUNNEL_START', pid=self.tunnel.pid, ssh_alias=alias)
        return True

    def _poll_value(self, status):
        value = dict(host_id=status['host_id'], slots=status['slots'], capacity=status.get('capacity'),
                     source_sha=status['source_sha'], cached_prefixes=status['cached_prefixes'],
                     cached_artifact_sha256=status.get('cached_artifact_sha256', []),
                     cached_metadata_bundle_sha256=status.get('cached_metadata_bundle_sha256', []))
        if 'accepting_new_jobs' in status:
            value['accepting_new_jobs'] = status['accepting_new_jobs']
        if status.get('resource_report') is not None:
            value['resource_report'] = status['resource_report']
        return value

    @staticmethod
    def _readiness_bundle_hashes(challenges):
        hashes = set()
        for assignment in challenges:
            request=assignment.get('request',{})
            if(request.get('route')or{}).get('kind')!='preparation':continue
            spec=request.get('preparation', {})
            hashes.update(row['sha256']for row in spec.get('input_git_bundles',[]))
            for workspace in spec.get('workspaces',{}).values():
                hashes.update(row['sha256']for row in workspace.get('input_git_bundles',[]))
        return hashes

    @staticmethod
    def _metadata_requirements(assignments):
        """Only send the fixed SOURCE and bundle selectors, not full requests."""
        rows = []
        for assignment in assignments:
            request = assignment['request']
            spec = request['preparation']
            fields = {key: spec[key] for key in ('stage', 'provider', 'artificial', 'input_git_bundles') if key in spec}
            source = spec.get('source')
            if source:
                fields['source'] = {'commit':source['commit']}
            fields['workspaces'] = {alias:{'input_git_bundles':workspace.get('input_git_bundles', [])}
                                    for alias,workspace in spec.get('workspaces', {}).items()
                                    if workspace.get('input_git_bundles')}
            rows.append({'preparation':fields})
        return rows

    def _deliver_challenge_artifacts(self, worker, challenges):
        hashes = self._readiness_bundle_hashes(challenges)
        if not hashes:return
        preparation = [row for row in challenges if row.get('request', {}).get('route', {}).get('kind') == 'preparation']
        response,_=self.remote(worker,'artifacts-missing',dict(hashes=sorted(hashes),
                    direct_hashes=[],preparation_challenges=self._metadata_requirements(preparation)))
        for sha in response['missing']:
            try:
                result=self.post('/v1/artifacts/get',dict(host_id=worker['host_id'],hashes=[sha]))
                self.remote(worker,'artifacts-put',dict(blobs=result['blobs']))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.log('METADATA_GIT_BUNDLE_UNAVAILABLE',host_id=worker['host_id'],sha256=sha,error=str(error))
                continue
            self.log('METADATA_GIT_BUNDLE_DELIVERED',host_id=worker['host_id'],sha256=sha)

    def _readiness_batch(self, worker, challenges, cached=None, *, slots, prefetch=False):
        # Fill free slots with requests that need no extra bundle delivery.
        # Rotate the first request even when earlier checks keep returning HOLD.
        preparation = [(item, self._readiness_bundle_hashes([item])) for item in challenges
                       if item.get('request', {}).get('route', {}).get('kind') == 'preparation']
        if not preparation:
            return challenges
        history = self.bundle_prefetch_counts if prefetch else self.readiness_probe_counts
        counts = history.setdefault(worker['host_id'], {})
        known = cached if cached is not None else set()
        preparation.sort(key=lambda pair: (counts.get(pair[0]['job_id'], 0),
            len(pair[1] - known) if cached is not None else 0))
        delivered = known | preparation[0][1]
        selected = [item['job_id'] for item, hashes in preparation if hashes <= delivered][:slots]
        for job in selected:
            counts[job] = counts.get(job, 0) + 1
        return [item for item in challenges if
                item.get('request', {}).get('route', {}).get('kind') != 'preparation'
                or item['job_id'] in selected]

    def _wsl_ready_challenges(self, worker, challenges, cached, slots):
        """Copy one preparation batch in the background of WSL dispatch."""
        host = worker['host_id']
        pending = self.pending_bundles.get(host)
        if pending is not None and pending.done():
            del self.pending_bundles[host]
            try:
                pending.result()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.log('WSL_BUNDLE_PREFETCH_DEFERRED', host_id=host, error=str(error))
        ready, missing = [], []
        for challenge in challenges:
            (ready if self._readiness_bundle_hashes([challenge]) <= cached else missing).append(challenge)
        if missing and host not in self.pending_bundles:
            batch = self._readiness_batch(worker, missing, cached, slots=slots, prefetch=True)
            self.pending_bundles[host] = self.bundle_pool.submit(self._deliver_challenge_artifacts, worker, batch)
        return self._readiness_batch(worker, ready, cached, slots=slots)

    def _discover(self, worker):
        # Status and current-request checks may run concurrently for independent
        # hosts; no thread claims a job or delivers an assignment here.
        started = stamp()
        status, seconds = self.remote(worker, 'status', dict(metadata_only=True))
        if status.get('host_id') != worker['host_id']:
            raise ValueError('Mailbox host differs from relay profile')
        poll = self._poll_value(status)
        observed = self.post('/v1/poll', dict(poll, observe_only=True))
        challenges = observed.get('eligibility_challenges') or []
        # The controller may offer one preparation input check while this
        # mailbox is busy. The real slot count still governs every claim.
        if (challenges and status.get('accepting_new_jobs', True)
                and (observed.get('host_admission') or {}).get('accepting', True)):
            cached = (None if status.get('cached_artifact_sha256') is None else
                      set(status['cached_artifact_sha256']) | set(status.get('cached_metadata_bundle_sha256', [])))
            cached = set(cached) if cached is not None else None
            slots = max(1, status['slots'])
            if worker.get('transport') == 'ssh-wsl' and cached is not None:
                challenges = self._wsl_ready_challenges(worker, challenges, cached, slots)
            else:
                challenges = self._readiness_batch(worker, challenges, cached, slots=slots)
            if not challenges:
                return status, seconds, started, poll, observed
            # A fresh worker inventory avoids an SSH round trip. The worker
            # still verifies and binds all metadata in its readiness check.
            try:
                # Imported hints are not a proof for this particular fixed
                # SOURCE view. Ask the worker's exact metadata store if the raw
                # file is absent, even when a different view advertises it.
                raw_cached = status.get('cached_artifact_sha256')
                if raw_cached is None or self._readiness_bundle_hashes(challenges) - set(raw_cached):
                    self._deliver_challenge_artifacts(worker,challenges)
                facts, _ = self.remote(worker, 'readiness', dict(challenges=challenges))
            except (TimeoutError, subprocess.TimeoutExpired, RuntimeError) as error:
                # Completed and already-owned attempts still need this tick.
                # Reject no input and retry the failed check on the next poll.
                self.log('MAILBOX_READINESS_DEFERRED', host_id=worker['host_id'],
                         jobs=[item['job_id'] for item in challenges], error=str(error))
                return status, seconds, started, poll, dict(observed, readiness_deferred=True)
            if facts.get('host_id') != status['host_id']:
                raise ValueError('Readiness response host differs')
            poll.update(host_fingerprints=facts.get('host_fingerprints', {}),
                        readiness_errors=facts.get('readiness_errors', {}))
            observed = self.post('/v1/poll', dict(poll, observe_only=True))
        return status, seconds, started, poll, observed

    def _recover_artifacts(self, worker, attempt):
        hashes = list(dict.fromkeys(attempt.get('artifact_manifest', {}).values()))
        if not hashes:return
        missing = self.post('/v1/artifacts/missing', dict(host_id=worker['host_id'], hashes=hashes))['missing']
        for sha in missing:
            response, _ = self.remote(worker, 'artifacts-get', dict(hashes=[sha]))
            self.post('/v1/artifacts/put', dict(host_id=worker['host_id'], blobs=response['blobs']))
            self.log('ARTIFACT_RECOVERED', host_id=worker['host_id'], sha256=sha)

    def _deliver_artifacts(self, worker, assignment):
        return self._deliver_batch_artifacts(worker, [assignment])

    def _deliver_batch_artifacts(self, worker, assignments):
        hashes=[]; direct=[]; preparation=[]
        for assignment in assignments:
            request=assignment.get('request', {})
            if (request.get('route') or {}).get('kind') != 'preparation':continue
            spec=request['preparation'];inputs=assignment.get('artifact_manifest', spec.get('inputs', {}))
            need=list(inputs.values() if isinstance(inputs,dict) else [row['sha256'] for row in inputs])
            direct.extend(need); hashes.extend(need); preparation.append(assignment)
            hashes.extend(row['sha256'] for row in spec.get('input_git_bundles', []))
            for workspace in spec.get('workspaces', {}).values():
                hashes.extend(row['sha256'] for row in workspace.get('input_git_bundles', []))
        if not hashes:return
        response,_=self.remote(worker,'artifacts-missing',dict(hashes=list(dict.fromkeys(hashes)),
                    direct_hashes=list(dict.fromkeys(direct)),preparation_challenges=self._metadata_requirements(preparation)))
        for sha in response['missing']:
            result=self.post('/v1/artifacts/get',dict(host_id=worker['host_id'],hashes=[sha]))
            self.remote(worker,'artifacts-put',dict(blobs=result['blobs']))
            self.log('ARTIFACT_DELIVERED',host_id=worker['host_id'],sha256=sha)

    def _owned_attempts(self, host):
        """Read the controller's run ledger when the relay shares that host."""
        distributed = self.directory.parent
        if not self.config.get('external_tunnel') or (distributed.name, self.directory.name) != ('distributed', 'relay'):
            return None
        ledger = distributed/'QUEUE.sqlite3'
        endpoint = distributed/'ENDPOINT.json'
        if not ledger.is_file() or not endpoint.is_file() or json.loads(endpoint.read_text())['url'] != self.url:
            return None
        with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True, timeout=2)) as db:
            return set(db.execute('SELECT job_id,attempt_id FROM jobs WHERE host_id=? AND attempt_id IS NOT NULL',
                                  (host,)))

    def _log_unowned_attempt(self, owner):
        identity = (owner['host_id'], owner['job_id'], owner['attempt_id'])
        if identity not in self.reported_unowned:
            self.reported_unowned.add(identity)
            self.log('MAILBOX_ATTEMPT_OUTSIDE_RUN', **owner)

    def _report_attempt(self, owner, state, detail):
        try:
            response = self.post('/v1/state', dict(owner, state=state, detail=detail))
        except HTTPError as error:
            # A relay on a different host has no local ledger to consult.
            # Only an explicit ownership rejection can be skipped.
            with error:
                if error.code != 409:
                    raise
                try: reason = json.load(error)['error']
                except (ValueError, KeyError, TypeError): raise error
            if reason not in ('ValueError: Different execution host/attempt',
                              'ValueError: Unknown job or different execution host/attempt'):
                raise ValueError(reason) from error
            self._log_unowned_attempt(owner)
            return False
        if response.get('disposition') == 'STATE_OUTSIDE_CURRENT_RUN':
            self._log_unowned_attempt(owner)
            return False
        return True

    def _recover_attempts(self, worker, info, status, owned, attempts, download_seconds):
        host = status['host_id']
        recovered = False
        for attempt in attempts:
            state = attempt['state']
            if state == 'TRANSFERRED': continue
            owner = dict(host_id=host, job_id=attempt['job_id'], attempt_id=attempt['attempt_id'])
            if owned is not None and (owner['job_id'], owner['attempt_id']) not in owned:
                self._log_unowned_attempt(owner)
                continue
            try:
                info['phase'] = 'CONTROLLER_STATE'
                if not self._report_attempt(owner, state, attempt.get('detail', {})):
                    continue
                if state == 'COMPLETED_PENDING_TRANSFER':
                    started = time.monotonic()
                    result_download_seconds = None
                    if 'files' not in attempt:
                        result, result_download_seconds = self.remote(
                            worker, 'result', dict(owner, request_sha256=attempt['request_sha256']))
                        attempt = dict(attempt, **result)
                    info['phase'] = 'CONTROLLER_ARTIFACTS'
                    self._recover_artifacts(worker, attempt)
                    info['phase'] = 'CONTROLLER_COMPLETE'
                    response = self.post('/v1/complete', dict(owner, request_sha256=attempt['request_sha256'], files=attempt['files']))
                    if response.get('ok') is not True: raise ValueError('Controller refused completion')
                    upload_seconds = time.monotonic()-started
                    info['phase'] = 'MAILBOX_ACK'
                    acknowledgement, ack_seconds = self.remote(worker, 'ack', owner)
                    if acknowledgement.get('ok') is not True: raise ValueError('Worker refused completion acknowledgement')
                    recovered = True
                    timing = dict(upload_roundtrip_seconds=upload_seconds, ssh_status_download_seconds=download_seconds,
                                  ssh_ack_roundtrip_seconds=ack_seconds,
                                  artificial_transport_scope=('SSH metadata-only status' if status.get('metadata_only')
                                                              else 'SSH status includes all pending attempts for this host'))
                    if result_download_seconds is not None:
                        timing['ssh_result_download_seconds'] = result_download_seconds
                    self.log('RESULT_RECOVERED', **owner, duplicate=response.get('duplicate', False), **timing)
                    try: self.post('/v1/transfer-metric', dict(owner, timing=timing))
                    except Exception as error: self.log('TELEMETRY_PENDING', **owner, error=str(error))
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                self.log('MAILBOX_RESULT_PENDING', **owner, phase=info['phase'], error=str(error))
        return recovered

    def _schedule_recovery(self, worker, info, status, owned, download_seconds):
        host = status['host_id']
        pending = self.pending_recoveries[host]
        finished = set()
        for identity, future in tuple(pending.items()):
            if not future.done():
                continue
            del pending[identity]
            try:
                if future.result():
                    finished.add(identity)
            except Exception as error:
                job_id, attempt_id = identity
                self.log('MAILBOX_RESULT_PENDING', host_id=host, job_id=job_id,
                         attempt_id=attempt_id, phase='RECOVERY', error=str(error))
        # Offers can be sampled just before an ACK. Refresh those offers and
        # keep ownership of every in-flight attempt until its recovery ends.
        active = []
        for attempt in status['attempts']:
            identity = attempt['job_id'], attempt['attempt_id']
            if attempt['state'] != 'COMPLETED_PENDING_TRANSFER':
                active.append(attempt)
                continue
            if owned is not None and identity not in owned:
                self._log_unowned_attempt(dict(host_id=host, job_id=identity[0], attempt_id=identity[1]))
                continue
            if identity in finished or identity in pending or len(pending) >= self.recovery_width:
                continue
            pending[identity] = self.recovery_pool.submit(
                self._recover_attempts, worker, info.copy(), status, owned, [attempt], download_seconds)
        self._recover_attempts(worker, info, status, owned, active, download_seconds)
        return bool(finished)

    def tick(self, worker, observed=None):
        info = self.state['workers'].setdefault(worker['host_id'], dict(failure_count=0, next_retry_epoch=0))
        info['phase'] = 'MAILBOX_STATUS'
        if observed is None:
            status_start=stamp()
            status, download_seconds = self.remote(worker, 'status', dict(metadata_only=True))
            poll = self._poll_value(status)
            if status.get('metadata_only') is True:
                self.post('/v1/poll', dict(poll, observe_only=True))
        else:
            status, download_seconds, status_start, poll, _ = observed
        self.log('MAILBOX_STATUS_RETURN',host_id=worker['host_id'],started=status_start,seconds=download_seconds)
        host = status['host_id']
        if host != worker['host_id']: raise ValueError('Mailbox host differs from relay profile')
        owned = self._owned_attempts(host) if any(a['state'] != 'TRANSFERRED' for a in status['attempts']) else None
        if self.recovery_pool is not None:
            recovered = self._schedule_recovery(worker, info, status, owned, download_seconds)
            self._claim_and_deliver(worker, info, observed, status, poll, recovered)
        else:
            self._recover_and_deliver_once(worker, info, observed, status, poll, owned, download_seconds)
        info.update(last_success_epoch=time.time(), slots=status['slots'], phase='READY',
            states={name:sum(a['state']==name for a in status['attempts']) for name in sorted({a['state'] for a in status['attempts']})})

    def _recover_and_deliver_once(self, worker, info, observed, status, poll, owned, download_seconds):
        pending = []
        # Use the already observed free slots while returning completions.
        # No credit is taken for a slot that still requires acknowledgement.
        if status['slots']:
            pending = [attempt for attempt in status['attempts']
                       if attempt['state'] == 'COMPLETED_PENDING_TRANSFER']
        immediate = [attempt for attempt in status['attempts'] if attempt not in pending]
        recovered = self._recover_attempts(worker, info, status, owned, immediate, download_seconds)
        if pending:
            result_info = info.copy()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='rpe-mailbox-recovery') as pool:
                result = pool.submit(self._recover_attempts, worker, result_info, status, owned, pending, download_seconds)
                self._claim_and_deliver(worker, info, observed, status, poll, recovered)
                result.result()
        else:
            self._claim_and_deliver(worker, info, observed, status, poll, recovered)

    def _claim_and_deliver(self, worker, info, observed, status, poll, recovered):
        host = status['host_id']
        if (self.directory/'STOP').exists():
            return
        if recovered and status.get('metadata_only') is True:
            # A completed attempt held a slot in the original offer. Sample
            # the released capacity now, applying the same participation flag.
            status, _ = self.remote(worker, 'status', dict(metadata_only=True))
            if status['host_id'] != host:
                raise ValueError('Refreshed mailbox host differs from relay profile')
            fresh = self._poll_value(status)
            if worker.get('transport') == 'ssh-wsl' and fresh['source_sha'] == poll['source_sha']:
                fresh.update({key: poll[key] for key in ('host_fingerprints', 'readiness_errors') if key in poll})
            poll = fresh
            self.post('/v1/poll', dict(poll, observe_only=True))
        info['phase'] = 'CONTROLLER_POLL'
        # The worker's fresh offer still governs every claim, including Pause.
        response = self.post('/v1/poll', poll)
        if (not response['assignments'] and response.get('eligibility_challenges')
                and not (observed is not None and observed[-1].get('readiness_deferred'))):
            challenges = response['eligibility_challenges']
            if observed is not None:
                # cycle() already checked one group of preparation requests.
                # Preserve the existing inexpensive Q3 checks on this fallback.
                challenges = [item for item in challenges if
                              item.get('request', {}).get('route', {}).get('kind') != 'preparation']
            else:
                cached = (None if status.get('cached_artifact_sha256') is None else
                      set(status['cached_artifact_sha256']) | set(status.get('cached_metadata_bundle_sha256', [])))
                challenges = self._readiness_batch(worker, challenges,
                    set(cached) if cached is not None else None, slots=status['slots'])
            if challenges:
                self._deliver_challenge_artifacts(worker,challenges)
                current, _ = self.remote(worker, 'readiness', dict(challenges=challenges))
                if current.get('host_id') != host:
                    raise ValueError('Readiness response host differs')
                poll.update(host_fingerprints=current.get('host_fingerprints', {}),
                            readiness_errors=current.get('readiness_errors', {}))
                response = self.post('/v1/poll', poll)
        feedback = response.get('capacity_feedback')
        assignments = response['assignments']
        response_time=time.monotonic()
        for assignment in assignments:
            # Store delivery intent before SSH. A lost submit response is
            # reconciled with the same server2-owned job and attempt next tick.
            pending = self.directory/'deliveries'/assignment['job_id']
            pending.mkdir(parents=True, exist_ok=True)
            atomic_json(pending/'ASSIGNMENT.json', assignment)
        batches = []
        for assignment in assignments:
            preparation = assignment['request'].get('route', {}).get('kind') == 'preparation'
            # The mailbox batch endpoint validates each exact assignment,
            # including preparation. Group only within the same offered wave,
            # without delaying a Q3 behind a later preparation transfer.
            grouped = worker['host_id'] in self.config.get(
                'q3_batch_hosts', ('server1', 'server2', 'local0-wsl'))
            if (grouped and batches and batches[-1][0]
                    and (batches[-1][1][0]['request'].get('route', {}).get('kind') == 'preparation') == preparation):
                batches[-1][1].append(assignment)
            else:
                batches.append((grouped, [assignment]))
        for grouped, batch in batches:
            info['phase'] = 'MAILBOX_ARTIFACTS'
            if len(batch)>1:self._deliver_batch_artifacts(worker,batch)
            else:self._deliver_artifacts(worker,batch[0])
            info['phase'] = 'MAILBOX_SUBMIT'
            relay_wait = time.monotonic() - response_time
            owners = [dict(host_id=host, job_id=a['job_id'], attempt_id=a['attempt_id']) for a in batch]
            for owner in owners:
                self.log('REQUEST_DELIVERY_START', **owner)
            if grouped and len(batch) > 1:
                results, seconds = self.submit_batch(worker, batch)
                self.log('PREPARATION_BATCH_DELIVERED' if batch[0]['request'].get('route',{}).get('kind')=='preparation' else 'Q3_BATCH_DELIVERED', host_id=host, count=len(batch), seconds=seconds)
            else:
                individual, seconds = self.remote(worker, 'submit', batch[0])
                results = [individual]
            refused = []
            for owner, reply in zip(owners, results):
                if reply.get('ok') is not True:
                    refused.append(str(reply.get('error', 'Worker refused assignment')))
                    continue
                self.log('REQUEST_DELIVERED', **owner, seconds=seconds, duplicate=reply.get('duplicate', False))
                try:
                    self.post('/v1/transfer-metric', dict(owner, timing=dict(
                        ssh_delivery_roundtrip_seconds=seconds, relay_response_to_submit_seconds=relay_wait)))
                except Exception as error:
                    self.log('TELEMETRY_PENDING', **owner, error=str(error))
            if refused:
                raise ValueError('; '.join(refused))
        if feedback is not None and (status.get('resource_report') or {}).get('mode') == 'AUTO':
            try:
                self.remote(worker, 'resources', dict(capacity_feedback=feedback))
            except Exception as error:
                self.log('RESOURCE_FEEDBACK_PENDING', host_id=host, error=str(error))

    def _collect_ticks(self):
        for host,(worker,future) in list(self.pending_ticks.items()):
            if not future.done(): continue
            del self.pending_ticks[host]
            info=self.state['workers'][host]
            try:future.result()
            except Exception as error:
                self.failed(info,error,'TRANSPORT_PENDING',host_id=host,phase=info.get('phase'))
            else:self.succeeded(info)

    def _state_snapshot(self):
        # Per-host tick tasks mutate only their own existing info dictionary.
        # dict.copy is atomic under Python's GIL; serialize detached snapshots.
        return dict(self.state,workers={host:info.copy() for host,info in list(self.state['workers'].items())})

    def cycle(self):
        """Observe and deliver independent hosts without serial transport waits.

        Each host has at most one discovery or delivery in flight. A blocking
        submit retains that host's future and cannot strand another host.
        Completed attempts are returned by a separate, bounded pool.
        The cycle waits at most one poll interval, not every SSH timeout.
        """
        self._collect_ticks()
        budget=max(0,min(self.poll_seconds,float(self.config.get('discovery_wait_seconds',self.poll_seconds))))
        deadline=time.monotonic()+budget
        if self.ensure_tunnel():
            workers=[]
            for worker in self.config['workers']:
                if worker.get('enabled', True) is False:
                    self.state['workers'][worker['host_id']] = dict(
                        phase='DISABLED', accepting_new_jobs=False,
                        reason=worker.get('disabled_reason', 'Host disabled by controller configuration'))
                    continue
                workers.append(worker)
            if self.discovery_pool is None:
                self.discovery_pool=ThreadPoolExecutor(max_workers=max(1,len(workers)),thread_name_prefix='rpe-mailbox')
                self.recovery_pool=ThreadPoolExecutor(
                    max_workers=max(1, len(workers)*self.recovery_width), thread_name_prefix='rpe-mailbox-recovery')
            for worker in workers:
                host=worker['host_id']
                info=self.state['workers'].setdefault(host,dict(failure_count=0,next_retry_epoch=0))
                if host in self.pending_ticks or host in self.pending_discovery or time.time()<info.get('next_retry_epoch',0):
                    continue
                info['phase']='MAILBOX_DISCOVERY'
                self.pending_discovery[host]=(worker,self.discovery_pool.submit(self._discover,worker))
            if self.pending_discovery:
                wait([future for _,future in self.pending_discovery.values()],timeout=max(0,deadline-time.monotonic()))
            ready=[]
            for host,(worker,future) in list(self.pending_discovery.items()):
                if not future.done():continue
                del self.pending_discovery[host]
                info=self.state['workers'][host]
                try:ready.append((worker,future.result()))
                except Exception as error:
                    self.failed(info,error,'TRANSPORT_PENDING',host_id=host,phase=info.get('phase'))
            # All ready hosts have published metadata-only offers already. The
            # central matcher enforces per-REQUEST priority and available slots.
            ready.sort(key=lambda row:row[0].get('priority_group',1 if row[0].get('transport')=='wsl' else 0))
            for worker,observed in ready:
                if (self.directory/'STOP').exists():break
                host=worker['host_id']
                self.pending_ticks[host]=(worker,self.discovery_pool.submit(self.tick,worker,observed))
            if self.pending_ticks:
                wait([future for _,future in self.pending_ticks.values()],timeout=max(0,deadline-time.monotonic()))
            self._collect_ticks()
        self.state['heartbeat_epoch']=time.time()
        atomic_json(self.directory/'STATUS.json',self._state_snapshot())

    def run(self):
        lock = (self.directory/'RELAY.lock').open('a')
        try: acquire_lock(lock)
        except BlockingIOError: raise ValueError('This relay is already running')
        self.log('RELAY_START', pid=os.getpid())
        if os.name == 'posix':
            self.ssh_socket_directory = tempfile.TemporaryDirectory(prefix='rpe-ssh-', dir='/tmp')
        # Reuse RomuTail's startup-state fix (4058b48d5): a previous daemon's
        # stopped=True must not satisfy this new daemon's status/stop checks.
        atomic_json(self.directory/'STATUS.json', self._state_snapshot())
        try:
            while not (self.directory/'STOP').exists():
                self.cycle()
                time.sleep(self.poll_seconds)
        finally:
            if self.tunnel is not None and self.tunnel.poll() is None:
                self.tunnel.terminate(); self.tunnel.wait(timeout=10)
            self.state.update(stopped_epoch=time.time(), stopped=True)
            atomic_json(self.directory/'STATUS.json', self._state_snapshot())
            self.log('RELAY_STOP')
            if self.discovery_pool is not None:
                self.discovery_pool.shutdown(wait=False, cancel_futures=True)
            if self.recovery_pool is not None:
                self.recovery_pool.shutdown(wait=False, cancel_futures=True)
            if self.bundle_pool is not None:
                self.bundle_pool.shutdown(wait=False, cancel_futures=True)
            if self.ssh_socket_directory is not None:
                self.ssh_socket_directory.cleanup()
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--start', action='store_true')
    command.add_argument('--run', action='store_true', help='Run in foreground')
    command.add_argument('--status', action='store_true')
    command.add_argument('--stop', action='store_true', help='Stop only relay/tunnel; received workers continue')
    args = parser.parse_args(argv)
    relay = Relay(args.config)
    if args.status:
        path = relay.directory/'STATUS.json'
        result = json.loads(path.read_text()) if path.exists() else dict(status='NOT_STARTED')
        if 'heartbeat_epoch' in result: result['heartbeat_age_seconds'] = time.time()-result['heartbeat_epoch']
        print(json.dumps(result, indent=2)); return
    if args.stop:
        (relay.directory/'STOP').touch()
        print('Stop requested; received Q3 jobs keep running. Check --status for stopped_epoch.'); return
    if args.start:
        lock = (relay.directory/'RELAY.lock').open('a')
        try: acquire_lock(lock)
        except BlockingIOError: raise ValueError('This relay is already running')
        (relay.directory/'STOP').unlink(missing_ok=True)
        lock.close()
        with (relay.directory/'RELAY.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--config', str(relay.configuration), '--run'],
                **background_options(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(pid=process.pid, log=str(relay.directory/'RELAY.log')))); return
    relay.run()


if __name__ == '__main__': main()
