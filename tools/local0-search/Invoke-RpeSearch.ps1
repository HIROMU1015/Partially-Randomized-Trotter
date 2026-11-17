[CmdletBinding()]
param(
    [ValidateSet('Preview','PlanOnly','Prepare','Start','Resume','Status','Stop')][string]$Action='Preview',
    [string]$RequestFile,
    [string]$Instruction,
    [string]$RunId,
    [string]$Release,
    [string]$Config=(Join-Path $PSScriptRoot 'CLIENT.json')
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../local0-dry-check/Common.ps1')
$client=Read-RpeJson $Config
$bytes=if($RequestFile){[IO.File]::ReadAllBytes([IO.Path]::GetFullPath($RequestFile))}else{[Text.Encoding]::UTF8.GetBytes('{}')}
if($Action -in @('Start','Resume') -and -not $Instruction){$Instruction='Explicit user '+$Action+' command'+$(if($RequestFile){': '+[IO.Path]::GetFullPath($RequestFile)}else{': confirmed target'})}
$helper=if($client.remote_helper){Join-Path $PSScriptRoot $client.remote_helper}else{Join-Path $PSScriptRoot 'remote.py'}
$transport=@{server_config=$client.server_config;request_base64=[Convert]::ToBase64String($bytes);action=$Action.ToLowerInvariant();instruction=$Instruction;run_id=$RunId;release=$Release}
# Windows OpenSSH used from an incoming SSH session may keep a nested stdin
# pipe open. Use the installed public bootstrap and a small data argument.
$transport64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($transport|ConvertTo-Json -Compress -Depth 80)))
$remote='set -o pipefail; '+(Quote-RpeShellArgument $client.python)+' -B '+(Quote-RpeShellArgument $client.remote_entry)+' --transport-base64 '+$transport64+' | base64 -w 1024'
$sshPath=Join-Path $env:SystemRoot 'System32/OpenSSH/ssh.exe'
$native=@('-n','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',$client.ssh_alias,$remote)
# A child with file-backed handles also works when this command itself is
# running inside Windows OpenSSH. Avoid Python/PowerShell capture-pipe hangs.
$temporary=Join-Path ([IO.Path]::GetTempPath()) ('rpe-search-'+[guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($temporary)
try{
    $answer=Join-Path $temporary 'response.txt';$errors=Join-Path $temporary 'errors.txt';$stdout=Join-Path $temporary 'stdout.txt'
    $psquote={param($v) "'"+([string]$v).Replace("'","''")+"'"}
    $arguments=($native|ForEach-Object {& $psquote $_}) -join ','
    $bridge='$ErrorActionPreference=''Stop'';$ProgressPreference=''SilentlyContinue'';$a=@('+$arguments+');$r=& '+(& $psquote $sshPath)+' @a;$code=$LASTEXITCODE;[IO.File]::WriteAllText('+(& $psquote $answer)+',($r -join [Environment]::NewLine),[Text.Encoding]::ASCII);exit $code'
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($bridge))
    $process=Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -PassThru -Wait -RedirectStandardOutput $stdout -RedirectStandardError $errors
    if($process.ExitCode -ne 0){throw ('RomuTail client transport failed, exit='+$process.ExitCode+': '+[IO.File]::ReadAllText($errors))}
    $raw=[IO.File]::ReadAllText($answer)
}finally{
    if(Test-Path $temporary){Remove-Item -Recurse -Force $temporary}
}
$responseText=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($raw))
$response=$responseText|ConvertFrom-Json
if(-not $response.ok){throw $response.error}
$result=$response.result
$out=Join-Path $PSScriptRoot 'responses';[void][IO.Directory]::CreateDirectory($out)
$file=Join-Path $out ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8)+'.json')
Write-RpeJson $file $result
if($result.errors){$result|ConvertTo-Json -Depth 30;throw ('Requested fields were not applied; response: '+$file)}
if($Action -in @('Start','Resume') -and $result.status -in @('STARTED','ALREADY_RUNNING','REUSED_COMPLETED') -and $client.result_sync_enabled -ne $false){
    $syncConfig=if($client.result_sync_config){$client.result_sync_config}else{Join-Path $PSScriptRoot '../local0-result-sync/CONFIG.json'}
    try{
        $sync=& (Join-Path $PSScriptRoot '../local0-result-sync/Invoke-H4ResultSync.ps1') -Action OnConfirmedRun -Config $syncConfig -ResultFile $file
        $result|Add-Member -NotePropertyName result_sync -NotePropertyValue $sync -Force
        Write-Host ('Public map: '+$sync.index+'; monitor='+$sync.task_name)
    }catch{
        $result|Add-Member -NotePropertyName result_sync_error -NotePropertyValue $_.Exception.Message -Force
        Write-Warning ('Controller start/resume confirmed; result-sync setup failed: '+$_.Exception.Message)
    }
    Write-RpeJson $file $result
}
Write-Host ('Run '+$result.run_id+' : '+$result.status)
if($result.applied){Write-Host ('Range: '+($result.applied.requested|ConvertTo-Json -Compress -Depth 30))}
Write-Host ('Release: '+($result.release|ConvertTo-Json -Compress -Depth 10))
Write-Host ('Scientific SOURCE: '+$result.computation_source+'; controller SOURCE: '+$result.controller_source)
if($result.run_directory){Write-Host ('RomuTail: '+$result.run_directory)}
$result
