$ErrorActionPreference = 'Stop'
$P = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-f956f5197a5a'
$R = Join-Path $P 'collect_saved_results_v1.py'
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-preparation/docs/research/rpe_distributed_preparation/recovery_transport_v1/collect_saved_results_v1.py' $R
if ($LASTEXITCODE -ne 0) { throw 'Recovery script download failed' }
if ((Get-FileHash -LiteralPath $R -Algorithm SHA256).Hash.ToLowerInvariant() -ne '393e648a0aade498223db143df6d32ce9fe95a372fdfd669b2ff258afb6b7c48') { throw 'Recovery script hash differs' }
py -3 -B $R --bundle (Join-Path $P 'PREPARATION_DEPLOYMENT.zip') --bundle-sha256 cc3f1dafee18045736d813101d79a0bd86e29be20724c106abe4e1df56fe04f6 --collect-saved (Join-Path $P 'results')
if ($LASTEXITCODE -ne 0) { throw 'Saved result return failed; do not repeat inspection' }
