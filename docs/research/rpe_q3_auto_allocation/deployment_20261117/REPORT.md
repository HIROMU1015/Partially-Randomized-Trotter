# 自動割当・自動枠・計時349版の配置完了

server2のcontroller/workerと、server1・local（RomuTail）・local0-wslのworker用固定checkout/profileを配置した。
今回の4 REQUESTは、全16セルが新しい計時adapterで実入力dry-check PASSとなり、通常入口で割当表へ取り込めた。
実装は`349558b748a82f384a4589434a63ef9b5572032c`、配布の基準は`407fd460aab8bf3f30ec23472f216f2d3fa74104`。
科学SOURCEは`28d976b99cdf8329adcc1b79327e329e947ef774`の元8file manifestを維持する。

## 今回のREQUESTの可否

| 方式 | REQUEST ID | server2 | server1 | local / RomuTail | local0-wsl |
|---|---|---|---|---|---|
| Full PF | 9b9b4b5132ff89768c5c | PASS | PASS | PASS | PASS |
| B0I / TRUNCATED_PF | 72864f6cf968d3b514b1 | PASS | PASS | PASS | PASS |
| PR-Pauli | 805877c2ab6c3e03f7f1 | PASS | PASS | PASS | PASS |
| PR-DF | 48494ae31276a64c46ce | PASS | PASS | PASS | PASS |

readiness上の保留理由は0件。WSLは実行能力を持つが、Windowsの明示Pauseを維持している。
今回の配置では全ホストのworker/controller/relayを起動しておらず、運用spoolも作成していない。
[能力と参加状態](ALLOCATION_STATUS.json)を分けて記録した。
通常入口の[生の割当表](normal_after_remote/HOST_ALLOCATION.json)にある`accepting_new_jobs=true`は、
新runの中央設定が参加を許すという意味であり、現在のworker接続やWindowsの受付状態の観測ではない。
WSLはrequired_hostsに含まれず、次のrunでも明示Pauseは優先される。

## 照合した範囲

- server2の4件は先に実機dry-check済み。local0操作者が既存のqg1・RomuTail・Ubuntu WSL経路で残り12件を実行し、qg2へ返送した。
- 返送manifest/receiptの157ファイルを全件照合し、3ホストのPROFILEは配布した新profileとbyte一致した。
- REQUEST・入力hash・科学SOURCE・route・実行Python・新adapterの3file manifestを、固定349の既存照合処理で確認した。
  PR-Pauliは33e経路、PR-DFの内側SOURCEは27e。元のassignmentと実入力は変更していない。
- checker最上位のPythonと実adapterを区別した。実adapterは全台3.11.0rc1。
  server1とWSLは専用launcher/実バイナリ、RomuTailは`.venv311`、server2は既存native環境を使う。
  適用するrouteのAST guardも保存証拠と一致している。
- 通常入口で生成した16の実行profileは、物理配置したprofileのroute選択結果と一致した。
  全16セルのworker用fingerprintも一致し、server2は今回の時点のmetadataから再確認した。
  遠隔12セルは実機が返したfingerprintと一致した。実起動時には従来どおり現在fingerprintを再照合する。
- server2の固定349コード16ファイル、科学28dの8ファイルのhashを確認した。固定349のscripts/srcにtracked変更はない。

[FINAL_CHECK.json](FINAL_CHECK.json)、[公式遠隔READINESSと返送記録](remote_results/)、
[server2の公式証拠](server2/dry/)、[通常入口のpolicy](normal_after_remote/HOST_ELIGIBILITY.json)を参照。
担当agentによる返送metadataの別途再照合も行った。そのagentはmetadata収集helperの作者でもあり、外部独立reviewとは位置づけない。

## 通常入口の取込みと配置先

固定349の`run_rpe_schedule_search.py --plan-only --prepare-readiness`を、既存の実problem/roundで実行した。
保存履歴と未解決状態を再確認し、今回の4 REQUEST・配分・費用・入力hashが同一であることを確認した。
server2の4 dry-checkと遠隔12証拠を再利用し、取込み中の新dry-checkは0件。
READYまで7.302秒、同コマンド内の報告完了まで7.821秒、親processの実時間8.007秒だった。
これは保存履歴/cacheを使った準備処理の時間であり、実H4のQ3性能を表さない。

server2の配置先：

```
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-auto-source-349558b74
/home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/server2.json
/home/AbeHiromu/projects/rpe-q3-auto-deployment/next-h4-auto/CONTROLLER_OPTIONS.json
```

最後のファイルは生成された設定へ、検査済みserver2 profile全体を`executor.distributed.local_worker`として設定したもの。
科学roundやqueueの起動ファイルではない。遠隔の実配置先は各[PLACEMENT](remote_results/)と[profiles](profiles/)に保存した。
[HANDOFF](HANDOFF.md)に通常入口の取込みコマンドを示す。返送前の結果は`normal_before_remote`に保存し、上書きしていない。

次の科学runは別途採択されたroundと新しいrun directoryから開始する。今回の準備roundを本番実行しない。
利用者がhost/Pythonを毎回指定する必要はなく、REQUESTのrouteから今回検査したprofileを選ぶ。
新しいREQUESTには今回のPASSを一律転用せず、通常の照合・readiness生成を使う。

今回の実H4 Q3・信号生成・価格較正・回路compile・本番queue投入は全て0件。
既存の科学run・原結果・UNKNOWN・STOPは変更していない。実H4の完了実績・最適同時枠・速度倍率は未測定。
