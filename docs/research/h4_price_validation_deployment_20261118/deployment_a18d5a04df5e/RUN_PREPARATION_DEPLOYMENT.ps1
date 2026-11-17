$ErrorActionPreference = 'Stop'
$D = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-a18d5a04df5e'
New-Item -ItemType Directory -Force -Path $D | Out-Null
$Z = Join-Path $D 'PREPARATION_DEPLOYMENT.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-price-validation-normal-entry-20261117/docs/research/h4_price_validation_deployment_20261118/deployment_a18d5a04df5e/PREPARATION_DEPLOYMENT.zip' $Z
if ($LASTEXITCODE -ne 0) { throw 'Package download failed' }
if ((Get-FileHash -LiteralPath $Z -Algorithm SHA256).Hash.ToLowerInvariant() -ne '32e23e839acbf57799ff1aa6218fd4f16096483c08167b75af4d1b5df06ec643') { throw 'Package hash differs' }
Expand-Archive -LiteralPath $Z -DestinationPath $D -Force
py -3 -B (Join-Path $D 'run_rpe_preparation_deployment.py') --bundle $Z --bundle-sha256 32e23e839acbf57799ff1aa6218fd4f16096483c08167b75af4d1b5df06ec643 --output (Join-Path $D 'results')
