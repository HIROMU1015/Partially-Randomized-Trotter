$RpeWslHome = Join-Path $env:USERPROFILE 'rpe-local0-wsl-runtime-delta-v1-20261117'
$RpeWslZip = Join-Path $RpeWslHome 'WSL_RUNTIME_DELTA_V1.zip'
$RpeWslOut = Join-Path $RpeWslHome 'results'
if (Test-Path -LiteralPath $RpeWslOut) { throw 'results already exists; do not repeat dry-check' }
New-Item -ItemType Directory -Path $RpeWslHome -Force | Out-Null
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility/docs/research/rpe_q3_host_eligibility/optional_wsl/local0_tools/WSL_RUNTIME_DELTA_V1.zip' $RpeWslZip
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
if ((Get-FileHash -LiteralPath $RpeWslZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '7c3f96c23659706290cb39aafb4ebae95f7982d50498b66e946ce31a12c023ef') { throw 'ZIP SHA256 differs' }
Expand-Archive -LiteralPath $RpeWslZip -DestinationPath $RpeWslHome -Force
py -3 -B (Join-Path $RpeWslHome 'run_wsl_delta.py') --bundle $RpeWslZip --bundle-sha256 7c3f96c23659706290cb39aafb4ebae95f7982d50498b66e946ce31a12c023ef --output $RpeWslOut
Get-Content -LiteralPath (Join-Path $RpeWslOut 'SUMMARY.json')
Get-Content -LiteralPath (Join-Path $RpeWslOut 'RETURN_RECEIPT.json')
