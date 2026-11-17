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
. (Join-Path $PSScriptRoot 'Search-Progress.ps1')
$client=Read-RpeJson $Config
$bytes=if($RequestFile){[IO.File]::ReadAllBytes([IO.Path]::GetFullPath($RequestFile))}else{[Text.Encoding]::UTF8.GetBytes('{}')}
if($RequestFile){
    $document=(New-Object Text.UTF8Encoding($false,$true)).GetString($bytes).TrimStart([char]0xfeff).Trim()
    if(-not $document.StartsWith('{')){throw 'RequestFile must contain a JSON object.'}
    $request=$document | ConvertFrom-Json
    if($request -isnot [pscustomobject] -or $request.scope -isnot [pscustomobject] -or
        -not @($request.scope.PSObject.Properties).Count){
        throw 'RequestFile must contain a nonempty scope object.'
    }
    if($request.PSObject.Properties['run_id']){throw 'Use -RunId to name an existing run.'}
}
if($RunId -and $RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$'){throw 'Invalid run ID.'}
if($Action -in @('Start','Resume')){
    if([bool]$RunId -eq [bool]$RequestFile){throw 'Specify one explicit target with -RunId or -RequestFile.'}
    if(-not $Instruction){
        $Instruction=if($RunId){'Explicit user '+$Action+' for '+$RunId}else{'Explicit user '+$Action+' for '+[IO.Path]::GetFullPath($RequestFile)}
    }
}
if(-not $client.ssh_alias -or $client.ssh_alias.StartsWith('-')){throw 'Invalid RomuTail SSH alias.'}
# Keep the original JSON bytes, including exact fractions, binary64 spelling,
# and the difference between absent preparation budgets and explicit zero.
$transport=@{server_config=$client.server_config;request_base64=[Convert]::ToBase64String($bytes);action=$Action.ToLowerInvariant();instruction=$Instruction;run_id=$RunId;release=$Release}
# Windows OpenSSH used from an incoming SSH session may keep a nested stdin
# pipe open. Use the installed public bootstrap and a small data argument.
$transport64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($transport|ConvertTo-Json -Compress -Depth 80)))
$remote='set -o pipefail; '+(Quote-RpeShellArgument $client.python)+' -B '+(Quote-RpeShellArgument $client.remote_entry)+' --transport-base64 '+$transport64+' | base64 -w 1024'
$sshPath=if($client.ssh_executable){$client.ssh_executable}else{Join-Path $env:SystemRoot 'System32/OpenSSH/ssh.exe'}
if(-not (Test-Path -LiteralPath $sshPath)){throw ('Configured SSH executable is missing: '+$sshPath)}
$native=@('-n','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',$client.ssh_alias,$remote)
$raw=& $sshPath @native
$transportExit=$LASTEXITCODE
$response=$null
if($raw){
    try {
        $responseText=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String(($raw -join '')))
        $response=$responseText | ConvertFrom-Json -ErrorAction Stop
    } catch {
        if($transportExit -eq 0){throw 'RomuTail returned a malformed response.'}
    }
}
if($response.ok -eq $false -and $response.error){throw ('RomuTail rejected the request: '+$response.error)}
if($transportExit -ne 0){throw ('RomuTail client transport failed, exit='+$transportExit)}
if($response.ok -ne $true -or -not $response.result){throw 'RomuTail did not return a confirmed result.'}
$result=$response.result
$result|Add-Member -NotePropertyName display_status -NotePropertyValue (Get-RpeSearchDisplayStatus $result) -Force
$out=Join-Path $PSScriptRoot 'responses';[void][IO.Directory]::CreateDirectory($out)
$file=Join-Path $out ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfff')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8)+'.json')
Write-RpeJson $file $result
if($result.errors){$result|ConvertTo-Json -Depth 30;throw ('Requested fields were not applied; response: '+$file)}
if($RunId -and $result.run_id -ne $RunId -and
    -not ($Action -eq 'Resume' -and $result.continuation.parent_run_id -eq $RunId)){
    throw ('RomuTail returned a different run without a matching continuation reference. Response: '+$file)
}
if($Action -in @('Start','Resume') -and $result.display_status -ne 'TECHNICAL_FAILURE' -and
    $result.status -in @('STARTED','ALREADY_RUNNING','REUSED_COMPLETED') -and $client.result_sync_enabled -ne $false){
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
Write-Host ('Run '+$result.run_id+' : '+$result.display_status)
if($result.continuation.parent_run_id){Write-Host ('Continuation of '+$result.continuation.parent_run_id)}
if($result.applied){Write-Host ('Range: '+($result.applied.requested|ConvertTo-Json -Compress -Depth 30))}
Write-Host ('Release: '+($result.release|ConvertTo-Json -Compress -Depth 10))
Write-Host ('Scientific SOURCE: '+$result.computation_source+'; controller SOURCE: '+$result.controller_source)
if($result.runtime){Write-Host ('Runtime: '+($result.runtime|ConvertTo-Json -Compress -Depth 12))}
Write-RpeSearchProgress $result
if($result.run_directory){Write-Host ('RomuTail: '+$result.run_directory)}
$result
