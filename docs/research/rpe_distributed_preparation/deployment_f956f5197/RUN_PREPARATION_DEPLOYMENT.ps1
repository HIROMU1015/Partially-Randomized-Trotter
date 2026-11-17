$ErrorActionPreference = 'Stop'
$D = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-f956f5197a5a'
New-Item -ItemType Directory -Force -Path $D | Out-Null
$Z = Join-Path $D 'PREPARATION_DEPLOYMENT.zip'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-preparation/docs/research/rpe_distributed_preparation/deployment_f956f5197/PREPARATION_DEPLOYMENT.zip' $Z
if ($LASTEXITCODE -ne 0) { throw 'Package download failed' }
if ((Get-FileHash -LiteralPath $Z -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'cc3f1dafee18045736d813101d79a0bd86e29be20724c106abe4e1df56fe04f6') { throw 'Package hash differs' }
Expand-Archive -LiteralPath $Z -DestinationPath $D -Force
py -3 -B (Join-Path $D 'run_rpe_preparation_deployment.py') --bundle $Z --bundle-sha256 cc3f1dafee18045736d813101d79a0bd86e29be20724c106abe4e1df56fe04f6 --output (Join-Path $D 'results')
