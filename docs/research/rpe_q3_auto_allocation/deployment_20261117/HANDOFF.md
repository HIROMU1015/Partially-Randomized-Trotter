# 固定349版の配置と実入力readiness

実装は`349558b748a82f384a4589434a63ef9b5572032c`、公開成果基準は407fd460aab8bf3f30ec23472f216f2d3fa74104。
科学SOURCEは元28dの8file manifest、計時adapterは別に349版の3file manifestを使う。
今回の実H4 Q3・信号生成・価格較正・回路compile・本番queue投入は0件。

## 配置状況

server2のcontroller/worker/timed adapterは次へdetached固定checkoutとして配置した。

```
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-source-349558b74
/home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/server2.json
```

scripts/srcの固定hash・HEADを確認し、profileは[profiles/server2.json](profiles/server2.json)とbyte一致。
既存の科学checkout、run、UNKNOWN、STOP、Python環境を変更していない。
配置はfiles/profileまで。controller/worker/relayのdaemonは起動せず、運用spoolも作成していない。

server2の新profileでFull PF・B0I・PR-Pauli・PR-DFの実入力dry-checkは全4件PASS。
[公式証拠](server2/dry/)と[実runtime/fingerprint](server2/DRY_CACHE.json)を参照。
Pauliは33e、DFは内側27eをprofileで解決し、実adapterは3.11.0rc1を使用した。
同じ固定349のworkerが生成する現在fingerprintと保存証拠の一致も確認した。

## local0配布物と返送先

[1操作のPowerShell手順](local0_tools/RUN_AUTO_DEPLOY_349558B74.ps1)と
[配布物説明](local0_tools/README.md)を使う。ZIP SHA256は
`8f172d8bdeeb68d81eb906270cf0a939780419bcc07fa1daab0053c1c71905b8`。

local0の既存qg1、RomuTail、wsl.exe Ubuntu経路で、同じ349固定checkout/profileを新しい専用領域へ置く。
3台それぞれ元の4 REQUESTをdry-checkする。元のruntime・route view・科学SOURCE・spoolは保全し、環境installは行わない。
WSLの参加flagはPAUSEDのまま保持する。workerやrelayは起動しない。

返送先（qg2）：

```
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-allocation/docs/research/rpe_q3_auto_allocation/deployment_20261117/remote_results
```

公式READINESS、選択profile、実runtime、source/input/route/adapterの結合、現在fingerprint、raw DRY_CACHEを返す。
helperは固定SOURCE外のmetadata収集用で、worker/adapter349を変更しない。
checker上位のPythonではなく、command[0]とadapter_result.runtime、実launcher/バイナリhashを照合する。
新profile用のraw runtime indexを既存の照合処理が受理するため、新しい承認証明やguard緩和はない。
返送だけ失敗した場合は--return-onlyを使い、dry-checkを繰り返さない。

## 通常入口での取込み

[EVIDENCE.json](EVIDENCE.json)は新profileと上記の受領先を指す。
返送済みmanifest/receiptと一致する証拠だけを取り込む。
旧adapterの16 PASSを新profileへ転用しない。未受領・不一致セルは理由付きUNCHECKEDとなる。

```bash
python3 -B /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-source-349558b74/scripts/run_rpe_schedule_search.py \
 --problem /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-allocation/docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-allocation/docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir /home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/normal_after_remote \
 --plan-only --prepare-readiness \
 --preparation-cache /home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/preparation-cache \
 --host-readiness /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-allocation/docs/research/rpe_q3_auto_allocation/deployment_20261117/EVIDENCE.json
```

通常入口を返送前にも確認し、server2の新dry-check4件をcache再利用、新しいdry-check0・科学Q3/queue0で
server2の4 PASS、遠隔12 UNCHECKEDを生成した。初回READYまで33.20秒（差分履歴の取込みを含む）。
実運用の科学予算・SOURCE・queueを今回起動・変更する手順ではない。
新しい科学runの開始は別途採択されたroundから行う。
