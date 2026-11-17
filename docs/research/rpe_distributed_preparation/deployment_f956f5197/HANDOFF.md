# 準備工程の固定配置と保存実入力の検査

実装SOURCEは `f956f5197a5acfe113fbb2ed3bfa78564e7f4eec`。
この配布物は、新しいcontroller/worker/CAS対応relay用のファイル・profileを配置し、
保存実入力のSOURCE・blob hash・Python・compiler metadataだけを検査する。
新規の実H4信号・実compile・実Q3、worker/relay常駐、本番queue投入は行わない。

## 今回確認した範囲

server2で4方式×signal/compile/auditの12セルが `SAVED_INPUT_RUNTIME_INSPECTION_PASS`。
[公式記録](server2/dry/READINESS.json)、[工程×ホスト表](HOST_STAGE_READINESS.json)、[配置検査](CHECKS.json)を参照。
対象は固定保存locatorの内容と、固定SOURCE・入力hash・実Python・compiler packageである。
compile/auditでは保存契約から取り出したQiskit 1.3.0を照合した。

**これは、次に生成される実行REQUESTの全入力・CLI・依存成果物の検収ではない。**
そのため実行可否は `UNCONFIRMED` と記録した。
同梱の12例は `dry_check_only=true` で、科学実行へ投入できない。
通常探索が組み立てる実際のREQUESTは、個々のSOURCE・入力・stage profile・実Python/compilerを
通常workerのmetadata challengeで自動照合し、一致したホストだけが取得する。
旧Q3の16 PASSを新しい準備工程のPASSとして流用しない。

RomuTailとWSLの24セルは返送済み、各9 PASS・PR-DF各3 HOLDを中央照合した。
server1はSSH経路の`No route to host`により未接続で、今回の12セルは未確認/HOLD。
[現在の対応表](HOST_STAGE_READINESS.json)と[中央照合](../checks/REMOTE_IMPORT.json)を参照する。
初回配置は繰り返さず、回収も完了している。server1へ追加接続・自動再試行しない。
WSLの参加flagは初回開始時・回収時に受付OFFを確認。旧runner例外で初回の前後bytes比較は未完了のため、完全な前後一致の証拠とはしない。
今回のserver2には専用profileと固定viewだけを置き、運用spoolを作成していない。

## local0の1コマンド検査

PowerShellで次を実行する。既存のqg1・RomuTail・Ubuntu WSLの経路を使い、
3台へ同じ固定版を並列に配置して各12セルのmetadataを検査する。
以下は実行済みの初回手順の記録であり、現在の状態では再実行しない。
既存checkoutの切替や上書き、Python環境の変更、パッケージ導入はない。

```powershell
$D = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-f956f5197a5a'
New-Item -ItemType Directory -Force -Path $D | Out-Null
scp 'qg2:/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-preparation/docs/research/rpe_distributed_preparation/deployment_f956f5197/RUN_PREPARATION_DEPLOYMENT.ps1' (Join-Path $D 'RUN.ps1')
if ($LASTEXITCODE -ne 0) { throw 'script download failed' }
& (Join-Path $D 'RUN.ps1')
```

ZIP SHA256：`cc3f1dafee18045736d813101d79a0bd86e29be20724c106abe4e1df56fe04f6`。
返送先は新しい専用領域である。

```
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-preparation/docs/research/rpe_distributed_preparation/remote_results
```

返送だけ失敗したときは、検査を再実行せず次を使う。

```powershell
py -3 -B (Join-Path $D 'run_rpe_preparation_deployment.py') --bundle (Join-Path $D 'PREPARATION_DEPLOYMENT.zip') --bundle-sha256 cc3f1dafee18045736d813101d79a0bd86e29be20724c106abe4e1df56fe04f6 --return-only (Join-Path $D 'results')
```

## 次の採択済みrun用の設定

[profiles](profiles/)は4台の工程profileを持ち、17の既存provider phaseを扱う。
PFの固定providerは36ad、元のnative acquisition SOURCE d878も別の固定viewとして置く。
PRの共通providerと新しいadapterはf956。Q3の科学SOURCE28d・計時adapter349・元routeは保持した。
科学SOURCEとtransport/metadata adapterを混同しない。

[DISTRIBUTED_OPTIONS.json](DISTRIBUTED_OPTIONS.json)には4台のpreparation_profiles、
server2 local_worker、全体auto・host別auto、優先3台と任意WSLの設定を保存した。
これは設定テンプレートであり、採択済みroundの科学予算・明示上限や、Q3のexact REQUEST policyを代替しない。
`host_eligibility` のplaceholderはその新roundで通常入口が生成したpolicyへ置き換える。

配布ZIPの `relay_payload/` に、同じ固定SOURCEのCAS対応Windows relayと依存3ファイルがある。
`RELAY_TEMPLATE.json` はその3 remote workerの新transport/profileを指す。
これを新run専用の設定へコピーし、**採択済み新run**の `distributed/ENDPOINT.json` にあるportと、
新しいlocal0 state directory/local forward portを設定する。旧人工controllerのport・relay・spoolを使わない。
この配布物の配置・検査コマンド自体はrelayを起動しない。

科学開始が別途採択された後のrelay操作は従来どおりである。

```powershell
$R = Join-Path $D 'relay_payload\scripts\run_rpe_q3_relay.py'
py -3 -B $R --config $AdoptedRunRelayConfig --start
py -3 -B $R --config $AdoptedRunRelayConfig --status
py -3 -B $R --config $AdoptedRunRelayConfig --stop
```

再接続は同じconfigで `--start`。受領済みtaskは停止中も元ホストで継続し、
復帰後は同じjob/attemptの成果を回収する。WSLの明示Pauseは解除しない。

## controllerの作業領域と準備SOURCEを固定する

採択済み新roundでは `candidate_completion.source` を
`f956f5197a5acfe113fbb2ed3bfa78564e7f4eec` に明示する。
初期化時のHEADへ暗黙追従させない。SOURCEを変えるrunは、新しい契約とprofileで準備する。

controllerは、この固定commitから作る**別の書込み可能な専用worktree**で通常入口を実行する。
準備providerが入力・証拠をcommitするため、controllerのHEADはデータ公開で進む。
`preparation_adapter_root` や固定 `preparation_sources` のcheckoutをcontroller作業領域にしない。
配布物の `transport_root` とviewは固定・不変のまま保持する。
生成設定の `worker_script` は固定transportを指し、`local_worker` は同梱server2 profileを使う。
`preparation_profiles` は同梱の4 profileを中央に配置したlocatorへ向ける。

追加のreadonly runtimeとして、保存DF compile/auditの `provider_root` と `input_config` を
元のsealed provider/runtime manifestおよびconfig hashに照合する。
server2は元のreadonly入力へ明示的に解決済み。remote profileに対応がなければ、
そのセルは実行前に理由付きHOLDとなる。大きな `old_cache` は必須入力にせず、全体を配布しない。

同梱12例はnested Git入力、`signal_view`/`d2_view`、必要なreadonly pathを含む検査例である。
private入力を勝手にコピーしたり、環境を導入したりしない。実行REQUESTの可否は通常入口が
個別に判定し、今回の保存入力inspectionを一律の実行許可にはしない。
