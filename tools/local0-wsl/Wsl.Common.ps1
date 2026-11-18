$ErrorActionPreference = 'Stop'
function Invoke-WslCode([string]$Code, $Config=@{}, [string]$Distribution='Ubuntu', [string]$Python='/usr/bin/python3') {
    if ($Distribution -notmatch '^[A-Za-z0-9_.-]+$' -or $Python -notmatch '^/[^\s"'']+$') { throw 'Unsupported WSL distribution/interpreter argument' }
    $encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($Config | ConvertTo-Json -Compress -Depth 80)))
    $source="import base64,json`nCONFIG=json.loads(base64.b64decode('$encoded'))`n"+$Code.Replace("`r`n","`n")
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=Join-Path $env:SystemRoot 'System32\wsl.exe'
    $info.Arguments="-d $Distribution --exec $Python -B -"
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true
    $info.RedirectStandardInput=$true; $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    $info.StandardOutputEncoding=New-Object Text.UTF8Encoding($false)
    $info.StandardErrorEncoding=New-Object Text.UTF8Encoding($false)
    $process=New-Object Diagnostics.Process; $process.StartInfo=$info
    try {
        [void]$process.Start(); $out=$process.StandardOutput.ReadToEndAsync(); $err=$process.StandardError.ReadToEndAsync()
        $process.StandardInput.Write($source); $process.StandardInput.Close(); $process.WaitForExit()
        $stdout=$out.GetAwaiter().GetResult(); $stderr=$err.GetAwaiter().GetResult()
        if($process.ExitCode -ne 0){throw "WSL exit=$($process.ExitCode): $stderr $stdout"}
        return $stdout
    } finally {$process.Dispose()}
}
