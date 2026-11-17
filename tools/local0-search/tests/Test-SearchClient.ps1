[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$FixtureConfig)

$ErrorActionPreference='Stop'
$search=Split-Path -Parent $PSScriptRoot
. (Join-Path $search '../local0-dry-check/Common.ps1')
$config=Read-RpeJson $FixtureConfig
if($config.server_config -notmatch '/artificial-client-progress-[^/]+/ARTIFICIAL_STATE.json$' -or
    $config.result_sync_enabled -ne $false){
    throw 'The test requires its isolated artificial SSH fixture.'
}
$directory=Split-Path -Parent ([IO.Path]::GetFullPath($FixtureConfig))
$request=Join-Path $directory 'REQUEST.json'
$zero=Join-Path $directory 'REQUEST.zero.json'
$body='{"scope":{"methods":["PR_PAULI"],"per_method":{"PR_PAULI":{"L_D":[4,5,6,7],"kappa":{"mode":"absolute","min":"2","max":"64"},"delta_pf":[{"kind":"binary64","value":"0.2"},{"kind":"exact","value":"1/5"}]}},"shots":"auto"}}'
[IO.File]::WriteAllText($request,$body)
[IO.File]::WriteAllText($zero,$body.Insert($body.Length-2,',"preparation_budgets":{"signal":0,"compile":0,"audit":0}'))
$command=Join-Path $search 'Invoke-RpeSearch.ps1'
$invalid=Join-Path $directory 'REQUEST.invalid.json'
[IO.File]::WriteAllText($invalid,'{"scope":')
try { & $command -Action Preview -Config $FixtureConfig -RequestFile $invalid; throw 'Expected invalid JSON rejection' }
catch { if($_.Exception.Message -eq 'Expected invalid JSON rejection'){throw} }
try { & $command -Action Start -Config $FixtureConfig; throw 'Expected explicit target rejection' }
catch { if($_.Exception.Message -notmatch 'Specify one explicit target'){throw} }

$ordinary=& $command -Action Preview -Config $FixtureConfig -RequestFile $request
$allZero=& $command -Action Preview -Config $FixtureConfig -RequestFile $zero
if($ordinary.applied.requested.PSObject.Properties['preparation_budgets'] -or
    $ordinary.progress.effective.preparation_budgets.signal -ne 3 -or
    $allZero.progress.effective.preparation_budgets.signal -ne 0){
    throw 'Omitted and explicit zero budgets were not distinguished.'
}
$rawHash=(Get-FileHash -Algorithm SHA256 -LiteralPath $request).Hash.ToLowerInvariant()
if($rawHash -ne $ordinary.received_request_sha256){throw 'Request bytes changed in transport.'}
$run=$ordinary.run_id
$powershell=Join-Path $PSHOME 'powershell.exe'
& $powershell -NoProfile -File $command -Action Start -Config $FixtureConfig -RequestFile $request -Instruction 'Start artificial client interface acceptance.' | Out-Null
if($LASTEXITCODE -ne 0){throw 'The separate Windows client failed.'}
$again=& $command -Action Start -Config $FixtureConfig -RequestFile $request -Instruction 'Repeat artificial request.'
$resume=& $command -Action Resume -Config $FixtureConfig -RunId $run -Instruction 'Resume the artificial run.'
if($again.run_id -ne $run -or $resume.run_id -ne $run -or
    $again.synthetic_launch_count -ne 1 -or $resume.synthetic_launch_count -ne 1){
    throw 'Resend or resume changed the run or duplicated the job.'
}
$ended=$resume
for($attempt=0;$attempt -lt 40 -and $ended.runtime.phase -ne 'COMPLETE';$attempt++){
    Start-Sleep -Milliseconds 400
    $ended=& $command -Action Status -Config $FixtureConfig -RunId $run
}
if($ended.runtime.phase -ne 'COMPLETE' -or $ended.synthetic_launch_count -ne 1 -or
    'PREPARATION_WAITING' -notin $ended.progress.candidates.state -or
    'EVALUATION_COMPLETED' -notin $ended.progress.candidates.state){
    throw 'The artificial job did not complete independently of the client.'
}
$compile=$ended.progress.stages | Where-Object stage -eq 'compile'
if($compile.queue_accepted -ne 8 -or $compile.execution_started -ne 0){throw 'Stage accounting differs.'}
$prepared=$ended.progress.candidates | Where-Object candidate_id -eq 'new-inputs-prepared'
$receipts=$ended.synthetic_receipts
$signal=$ended.progress.stages | Where-Object stage -eq 'signal'
$q3=$ended.progress.stages | Where-Object stage -eq 'q3'
if($prepared.state -ne 'EVALUATION_COMPLETED' -or $signal.execution_completed -ne 1 -or
    $q3.execution_completed -ne 2 -or
    $receipts.ready_evaluation.sequence -ge $receipts.preparation_intake.sequence -or
    $receipts.preparation_intake.sequence -ge $receipts.prepared_evaluation.sequence -or
    $receipts.preparation_intake.input_sha256 -notmatch '^[0-9a-f]{64}$' -or
    $receipts.preparation_intake.input_sha256 -ne $receipts.prepared_evaluation.input_sha256){
    throw 'Independent evaluation or preparation-result intake and subsequent evaluation differed.'
}
$failed=& $command -Action Status -Config $FixtureConfig -RunId 'fixture-failed'
if($failed.status -ne 'REUSED_COMPLETED' -or $failed.display_status -ne 'TECHNICAL_FAILURE'){
    throw 'The client treated a failed controller as complete.'
}
$continuation=& $command -Action Resume -Config $FixtureConfig -RunId 'fixture-failed'
if($continuation.run_id -ne 'fixture-continuation' -or
    $continuation.continuation.parent_run_id -ne 'fixture-failed'){
    throw 'The client did not accept the continuation linked by the central response.'
}
$evidence=[ordered]@{
    status='PASS'; run_id=$run; exact_request_sha256=$rawHash
    omitted_budget_and_explicit_zero=$true; explicit_target_required=$true
    artificial_launches=1; independent_client_exit=$true; preparation_wait_and_evaluation_completed=$true
    preparation_result_intake_and_evaluation=$true; central_continuation_reference=$true
    queue_compile_accepted=8; actual_compile_started=0; stale_completion_display='TECHNICAL_FAILURE'
    real_scientific_calls=0
}
Write-RpeJson (Join-Path $directory 'RESULT.json') $evidence
[pscustomobject]$evidence
