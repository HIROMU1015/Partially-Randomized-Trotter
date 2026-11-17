$ErrorActionPreference = 'Stop'
$RpePauliHome = Join-Path $env:USERPROFILE 'rpe-pauli-route-33e324b-delta-20261117'
$RpePauliZip = Join-Path $RpePauliHome 'PAULI_DELTA.zip'
$RpePauliOut = Join-Path $RpePauliHome 'results'
if (Test-Path -LiteralPath $RpePauliOut) { throw 'results already exists. Do not repeat dry-check; use --return-only if only return failed.' }
New-Item -ItemType Directory -Path $RpePauliHome -Force | Out-Null
scp qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility/docs/research/rpe_q3_host_eligibility/pauli_route_update/local0_tools/PAULI_DELTA.zip $RpePauliZip
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
if ((Get-FileHash -LiteralPath $RpePauliZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '5c5273436d2fafd65c2be5294cd67313a8dc7fcafee3839398b49601d66c1a70') { throw 'ZIP SHA256 differs' }
Expand-Archive -LiteralPath $RpePauliZip -DestinationPath $RpePauliHome -Force
py -3 -B (Join-Path $RpePauliHome 'run_delta.py') --bundle $RpePauliZip --bundle-sha256 5c5273436d2fafd65c2be5294cd67313a8dc7fcafee3839398b49601d66c1a70 --output $RpePauliOut
if ($LASTEXITCODE -ne 0) { throw 'A dry-check or transfer did not PASS. Local results are preserved in results.' }
Get-Content -LiteralPath (Join-Path $RpePauliOut 'SUMMARY.json')
Get-Content -LiteralPath (Join-Path $RpePauliOut 'RETURN_RECEIPT.json')
