"""Reuse the existing Windows OnConfirmedRun hook after a central operation."""
from pathlib import Path
import argparse
import base64
import json
import subprocess
import sys
import tempfile
import uuid

from run_rpe_q3_relay import scp_prefix


def confirmed_run(config,result):
    if result.get('status')not in('STARTED','ALREADY_RUNNING','REUSED_COMPLETED'):
        return dict(status='NOT_A_CONFIRMED_RUN')
    if result.get('artificial_only')and not config.get('artificial_only'):
        return dict(status='ARTIFICIAL_SKIPPED')
    ssh=list(config.get('ssh_command',['ssh']));scp,target=scp_prefix(config,ssh)
    name='.rpe-confirmed-run-'+uuid.uuid4().hex+'.json'
    timeout=config.get('request_timeout_seconds',120)
    with tempfile.TemporaryDirectory(prefix='rpe-confirmed-run-')as directory:
        payload=Path(directory)/'result.json';payload.write_text(json.dumps(result),encoding='utf-8')
        transfer=subprocess.run(scp+['-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(payload),target+':'+name],
            stdin=subprocess.DEVNULL,capture_output=True,timeout=timeout)
        if transfer.returncode:raise RuntimeError('Confirmed-run transfer failed: '+transfer.stderr.decode(errors='replace')[-1000:])
        quote=lambda item:"'"+str(item).replace("'","''")+"'"
        script="$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);$p=Join-Path $env:USERPROFILE "+quote(name)+";try{"
        script+="$v=& "+quote(config['result_sync_script'])+" -Action OnConfirmedRun -ResultFile $p -Config "+quote(config['result_sync_config'])+";"
        script+="$v|ConvertTo-Json -Depth 60 -Compress}finally{Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue}"
        remote='powershell.exe -NoProfile -NonInteractive -EncodedCommand '+base64.b64encode(script.encode('utf-16-le')).decode()
        process=subprocess.run(ssh+['-n','-o','BatchMode=yes','-o','ConnectTimeout=10',config['ssh_alias'],remote],
            stdin=subprocess.DEVNULL,capture_output=True,timeout=timeout)
        if process.returncode:raise RuntimeError('Confirmed-run synchronization failed: '+process.stderr.decode(errors='replace')[-1500:])
        return dict(status='ON_CONFIRMED_RUN',result=json.loads(process.stdout))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=confirmed_run(json.loads(args.config.read_text()),json.load(sys.stdin))
        print(json.dumps(result,ensure_ascii=False));return 0
    except Exception as error:
        print(json.dumps(dict(status='RESULT_SYNC_PENDING',error=str(error)),ensure_ascii=False));return 1


if __name__=='__main__':raise SystemExit(main())
