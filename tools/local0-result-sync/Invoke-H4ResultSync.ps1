[CmdletBinding()]
param(
    [ValidateSet('Sync','OnConfirmedRun','Status')][string]$Action='Sync',
    [string]$Config=(Join-Path $PSScriptRoot 'CONFIG.json'),
    [string]$ResultFile
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../local0-dry-check/Common.ps1')
$configPath=[IO.Path]::GetFullPath($Config);$settings=Read-RpeJson $configPath
$engine=Join-Path $PSScriptRoot 'sync.py'
$nativeAction=if($Action -eq 'OnConfirmedRun'){'prepare'}else{$Action.ToLowerInvariant()}
$arguments=@('-B',$engine,$nativeAction,'--config',$configPath)
if($Action -eq 'OnConfirmedRun'){
    if(-not $ResultFile){throw 'Confirmed server2 response is required'}
    $arguments+=@('--result',[IO.Path]::GetFullPath($ResultFile))
}
$raw=& $settings.python @arguments
if($LASTEXITCODE -ne 0){throw ('Result synchronization failed, action='+$Action)}
$result=($raw -join "`n")|ConvertFrom-Json
if($Action -eq 'OnConfirmedRun' -and $result.state -ne 'TERMINAL'){
    $existing=Get-ScheduledTask -TaskName $result.task_name -ErrorAction SilentlyContinue
    if(-not $existing){
        $taskArgs=@('-B',$engine,'watch','--request',$result.request)|ForEach-Object {Quote-RpeWindowsArgument $_}
        $taskAction=New-ScheduledTaskAction -Execute $settings.pythonw -Argument ($taskArgs -join ' ') -WorkingDirectory $PSScriptRoot
        $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
        $taskSettings=New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
        $task=New-ScheduledTask -Action $taskAction -Principal $principal -Settings $taskSettings -Description 'Run-scoped H4 public-result sync. No logon trigger; no scientific execution.'
        try {Register-ScheduledTask -TaskName $result.task_name -InputObject $task -ErrorAction Stop|Out-Null}
        catch {if(-not(Get-ScheduledTask -TaskName $result.task_name -ErrorAction SilentlyContinue)){throw}}
    }
    $task=Get-ScheduledTask -TaskName $result.task_name
    $monitorState=Join-Path (Split-Path -Parent $result.request) 'MONITOR_STATE.json'
    if($task.State -eq 'Running' -and (Test-Path -LiteralPath $monitorState) -and (Read-RpeJson $monitorState).state -eq 'STOPPED'){
        for($i=0;$i -lt 25 -and $task.State -eq 'Running';$i++){Start-Sleep -Milliseconds 200;$task=Get-ScheduledTask -TaskName $result.task_name}
    }
    $alreadyRunning=$task.State -eq 'Running'
    if(-not $alreadyRunning){Start-ScheduledTask -TaskName $result.task_name}
    $result|Add-Member -NotePropertyName monitor_already_running -NotePropertyValue $alreadyRunning -Force
    $result|Add-Member -NotePropertyName periodic_interval_seconds -NotePropertyValue $settings.interval_seconds -Force
}
$result
