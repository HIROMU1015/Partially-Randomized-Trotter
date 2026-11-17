$ErrorActionPreference = 'Stop'

function Write-RpeJson([string]$Path, $Value) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 80) + "`n", $encoding)
}

function Get-RpeSha([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Read-RpeJson([string]$Path) {
    return ([IO.File]::ReadAllText($Path) | ConvertFrom-Json)
}

function Get-RpePublications([string]$CatalogPath) {
    $catalog = Read-RpeJson $CatalogPath
    $directory = Split-Path -Parent ([IO.Path]::GetFullPath($CatalogPath))
    foreach ($publication in $catalog.publications) {
        $files = [ordered]@{}
        foreach ($entry in $publication.files.PSObject.Properties) {
            $path = Join-Path $directory $entry.Value.file
            if ((Get-RpeSha $path) -ne $entry.Value.sha256) { throw "Published file hash differs: $path" }
            $files[$entry.Name] = @{sha256=$entry.Value.sha256;git_blob=$entry.Value.git_blob;data=[Convert]::ToBase64String([IO.File]::ReadAllBytes($path))}
        }
        $value = [ordered]@{commit=$publication.commit;files=$files}
        foreach ($role in @('mode','selector','adapter_root','adapter_commit','adapter_sha256','checker','profile_helper','handoff','metadata','profile','readiness','resolved_request','adapter_log','adapter_result')) { if($null -ne $publication.$role){$value[$role]=$publication.$role} }
        [pscustomobject]$value
    }
}

function Quote-RpeWindowsArgument([string]$Value) {
    # CommandLineToArgvW quoting; no cmd.exe or PowerShell string evaluation.
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Quote-RpeShellArgument([string]$Value) {
    return "'" + $Value.Replace("'", "'" + '"' + "'" + '"' + "'") + "'"
}

function Invoke-RpeRemote([string]$TargetAlias, [string]$Interpreter, $Config, [string]$HelperPath) {
    $sshPath = Join-Path $env:SystemRoot 'System32\OpenSSH\ssh.exe'
    if (-not (Test-Path -LiteralPath $sshPath)) { throw 'Windows OpenSSH ssh.exe is required' }
    $configJson = $Config | ConvertTo-Json -Compress -Depth 80
    $config64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($configJson))
    $source = "import base64,json`nCONFIG=json.loads(base64.b64decode('$config64'))`n" + [IO.File]::ReadAllText($HelperPath)
    $remoteCommand = (Quote-RpeShellArgument $Interpreter) + ' -B -'
    $nativeArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', $TargetAlias, $remoteCommand)
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $sshPath
    $info.Arguments = ($nativeArgs | ForEach-Object { Quote-RpeWindowsArgument $_ }) -join ' '
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $info.StandardErrorEncoding = New-Object System.Text.UTF8Encoding($false)
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $inputFailure = ''
        try { $process.StandardInput.Write($source.Replace("`r`n", "`n")) }
        catch { $inputFailure = $_.Exception.Message }
        try { $process.StandardInput.Close() } catch { $inputFailure = $_.Exception.Message }
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if (-not $stdout.Trim()) { throw "SSH $TargetAlias exit=$($process.ExitCode): $stderr $inputFailure" }
        try { $response = $stdout | ConvertFrom-Json } catch { throw "SSH $TargetAlias invalid JSON: $stderr $stdout" }
        if ($process.ExitCode -ne 0 -or -not $response.ok) {
            $detail = $response.error
            if (-not $detail) { $detail = $response.result.log }
            if (-not $detail) { $detail = $stdout }
            throw "SSH $TargetAlias exit=$($process.ExitCode): $detail $stderr"
        }
        return $response.result
    } finally { $process.Dispose() }
}

function Invoke-RpeHostJob([string]$ContextPath, [string]$TargetPath, [string]$HelperPath) {
    $context = Read-RpeJson $ContextPath
    $target = Read-RpeJson $TargetPath
    $config = @{
        action = 'host'; bundle_base64 = $context.bundle_base64; bundle_sha256 = $context.bundle_sha256
        profile_path = $target.profile_path; run_id = $context.run_id
        helper_sha256 = $context.helper_sha256; resume = [bool]$context.resume
        publications = @($context.publications); publication_sha256 = $context.publication_sha256
        refresh_readiness = [bool]$context.refresh_readiness
        assignment_paths = @($context.assignment_paths)
    }
    if($context.deployment){$config.action='host_deployment'}
    try {
        if($context.deployment -and $target.worker.transport -eq 'wsl'){
            . (Join-Path (Split-Path -Parent $HelperPath) '../local0-wsl/Wsl.Common.ps1')
            $response=(Invoke-WslCode ([IO.File]::ReadAllText($HelperPath)) $config $target.worker.distribution $target.worker.python)|ConvertFrom-Json
            if(-not $response.ok){throw ('WSL deployment: '+$response.error)}
            $result=$response.result
        }else{
            $interpreter=if($context.deployment){$target.worker.python}else{$target.profile.interpreter}
            $result = Invoke-RpeRemote $target.ssh_alias $interpreter $config $HelperPath
        }
        $archivePath = Join-Path $target.local_directory 'evidence.zip'
        $bytes = [Convert]::FromBase64String($result.archive_base64)
        if (Test-Path -LiteralPath $archivePath) {
            if ((Get-RpeSha $archivePath) -ne $result.archive_sha256) { throw 'Existing local archive differs' }
        } else { [IO.File]::WriteAllBytes($archivePath, $bytes) }
        if ((Get-RpeSha $archivePath) -ne $result.archive_sha256) { throw 'Host archive transfer hash differs' }
        $result.PSObject.Properties.Remove('archive_base64')
        $result | Add-Member -NotePropertyName ssh_alias -NotePropertyValue $target.ssh_alias
        $result | Add-Member -NotePropertyName transport_ok -NotePropertyValue $true
        Write-RpeJson (Join-Path $target.local_directory 'HOST_RESULT.json') $result
        return $result
    } catch {
        $result = [pscustomobject]@{host_id=$target.profile.host_id; ssh_alias=$target.ssh_alias; transport_ok=$false; error=$_.Exception.Message; rows=@()}
        Write-RpeJson (Join-Path $target.local_directory 'HOST_ERROR.json') $result
        return $result
    }
}

function Send-RpeEvidenceJob([string]$ContextPath, [string]$TargetPath, [string]$HelperPath, [string]$SummaryPath) {
    $context = Read-RpeJson $ContextPath
    $target = Read-RpeJson $TargetPath
    $result = Read-RpeJson (Join-Path $target.local_directory 'HOST_RESULT.json')
    $config = @{
        action='receipt'; host_id=$result.host_id; archive_sha256=$result.archive_sha256
        return_directory=$context.return_directory
        summary_name=[IO.Path]::GetFileName($SummaryPath)
        summary_base64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($SummaryPath))
    }
    try {
        if ($context.resume) {
            $probe = Invoke-RpeRemote $context.coordinator_alias $context.coordinator_python $config $HelperPath
            if ($probe.receipt -and $probe.receipt.received) {
                $receipt = $probe.receipt
                $receipt | Add-Member -NotePropertyName sent_archive_bytes -NotePropertyValue 0
                $receipt | Add-Member -NotePropertyName reused_verified_receipt -NotePropertyValue $true
                Write-RpeJson (Join-Path $target.local_directory 'RETURN_RECEIPT.json') $receipt
                return $receipt
            }
        }
        $config.action = 'receive'
        $config.archive_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $target.local_directory 'evidence.zip')))
        $receipt = Invoke-RpeRemote $context.coordinator_alias $context.coordinator_python $config $HelperPath
        $receipt | Add-Member -NotePropertyName sent_archive_bytes -NotePropertyValue $result.archive_bytes
        $receipt | Add-Member -NotePropertyName reused_verified_receipt -NotePropertyValue $false
        Write-RpeJson (Join-Path $target.local_directory 'RETURN_RECEIPT.json') $receipt
        return $receipt
    } catch {
        $receipt = [pscustomobject]@{host_id=$result.host_id; received=$false; error=$_.Exception.Message}
        Write-RpeJson (Join-Path $target.local_directory 'RETURN_ERROR.json') $receipt
        return $receipt
    }
}
