# V3 runtime-guard PF remote 4-cell dry-check only.
$ErrorActionPreference = 'Stop'
$RpePfHome = Join-Path $env:USERPROFILE 'rpe-pf-runtime-cbc471c91-delta-20261117-v3'
$RpePfZip = Join-Path $RpePfHome 'PF_RUNTIME_DELTA_V3.zip'
$RpePfOut = Join-Path $RpePfHome 'results'
if (Test-Path -LiteralPath $RpePfOut) { throw 'results already exists. Do not repeat dry-check; use --return-only if only return failed.' }
New-Item -ItemType Directory -Path $RpePfHome -Force | Out-Null
scp qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility/docs/research/rpe_q3_host_eligibility/pauli_route_update/local0_tools/PF_RUNTIME_DELTA_V3.zip $RpePfZip
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
if ((Get-FileHash -LiteralPath $RpePfZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'bf808423ba8f4f98bf4dacbe4ef9ead1f0e09a7ed56d5620100d2683b7943747') { throw 'ZIP SHA256 differs' }
Expand-Archive -LiteralPath $RpePfZip -DestinationPath $RpePfHome -Force
py -3 -B (Join-Path $RpePfHome 'run_delta.py') --bundle $RpePfZip --bundle-sha256 bf808423ba8f4f98bf4dacbe4ef9ead1f0e09a7ed56d5620100d2683b7943747 --output $RpePfOut
if ($LASTEXITCODE -ne 0) { throw 'A dry-check or transfer did not PASS. Local results are preserved in results.' }
Get-Content -LiteralPath (Join-Path $RpePfOut 'SUMMARY.json')
Get-Content -LiteralPath (Join-Path $RpePfOut 'RETURN_RECEIPT.json')
