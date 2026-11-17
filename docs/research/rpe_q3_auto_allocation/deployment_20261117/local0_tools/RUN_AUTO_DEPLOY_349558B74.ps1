$RpeAutoDeployHome = Join-Path $env:USERPROFILE 'rpe-q3-auto-deploy-349558b74-20261117'
$RpeAutoDeployZip = Join-Path $RpeAutoDeployHome 'AUTO_DEPLOY_349558B74.zip'
$RpeAutoDeployOut = Join-Path $RpeAutoDeployHome 'results'
if (Test-Path -LiteralPath $RpeAutoDeployOut) { throw 'results already exists; do not repeat dry-check; use --return-only if needed' }
New-Item -ItemType Directory -Path $RpeAutoDeployHome -Force | Out-Null
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-allocation/docs/research/rpe_q3_auto_allocation/deployment_20261117/local0_tools/AUTO_DEPLOY_349558B74.zip' $RpeAutoDeployZip
if ($LASTEXITCODE -ne 0) { throw 'ZIP download failed' }
if ((Get-FileHash -LiteralPath $RpeAutoDeployZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '8f172d8bdeeb68d81eb906270cf0a939780419bcc07fa1daab0053c1c71905b8') { throw 'ZIP SHA256 differs' }
Expand-Archive -LiteralPath $RpeAutoDeployZip -DestinationPath $RpeAutoDeployHome -Force
py -3 -B (Join-Path $RpeAutoDeployHome 'run_auto_deployment.py') --bundle $RpeAutoDeployZip --bundle-sha256 8f172d8bdeeb68d81eb906270cf0a939780419bcc07fa1daab0053c1c71905b8 --output $RpeAutoDeployOut
$RpeAutoDeployExit = $LASTEXITCODE
Get-Content -LiteralPath (Join-Path $RpeAutoDeployOut 'SUMMARY.json')
Get-Content -LiteralPath (Join-Path $RpeAutoDeployOut 'RETURN_RECEIPT.json')
if ($RpeAutoDeployExit -ne 0) { throw 'One or more cells require preparation; results were retained' }
