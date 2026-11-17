# 固定349版の配置と実入力readiness

実装は`349558b748a82f384a4589434a63ef9b5572032c`、公開成果基準は407fd460aab8bf3f30ec23472f216f2d3fa74104。
科学SOURCEは元28dの8file manifest、計時adapterは別に349版の3file manifestを使う。
今回の実H4 Q3・信号生成・価格較正・回路compile・本番queue投入は0件。

返送後の取込みは完了した。今回の4 REQUEST×4台は全16セルPASS、readiness保留0件。
[最終REPORT](REPORT.md)、[能力と参加状態](ALLOCATION_STATUS.json)、[照合結果](FINAL_CHECK.json)を参照。
WSLの明示Pauseは維持し、controller/worker/relayのdaemonは全台とも今回起動していない。

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

3台への配置と12件の実入力dry-checkは完了し、[返送記録](remote_results/RETURN_RECEIPT.json)の
157ファイルを全件照合した。各PROFILEは配布物とbyte一致、全12セルの現在worker fingerprintも一致した。
WSL Pauseは検査の開始時と終了時に同一であることを配布runnerが確認した。サーバー2からWindowsの現在状態を直接観測したものではない。
この配布手順を再実行する必要はない。

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

返送後も上記コマンドを実行済み。[通常入口の結果](normal_after_remote/READY.json)は全16 PASS、
server2のdry-check4件再利用・新dry-check0件、READYまで7.302秒、報告完了まで7.821秒。
同じ検査を将来繰り返す場合は、既存結果を保全して`--run-dir`だけ新しい専用directoryにする。
生成した[policy](normal_after_remote/HOST_ELIGIBILITY.json)は各ホストの配置済みprofileと一致した。

server2の新run用設定は次に配置した。

```
/home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/CONTROLLER_OPTIONS.json
```

[同じ設定の保存版](CONTROLLER_OPTIONS.json)は全体auto・host別slot_policyと検査済みlocal_workerを含む。
これは設定のみで起動しない。別途採択された新roundに使用し、科学予算・明示上限はそのroundの条件を優先する。
Windows relayを次runで使うときは、各PLACEMENTの新transport_rootとworker_profileを指定する。
旧人工runのrelay・spoolを再開しない。WSLのrequired待ちはなく、Windowsの明示Pauseは解除しない。
