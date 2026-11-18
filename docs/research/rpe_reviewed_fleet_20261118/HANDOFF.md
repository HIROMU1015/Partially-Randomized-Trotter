# PRレビュー・容量対策を統合したRomuTail版

初回の採用版は `romutail-reviewed-pr-3b38e4af6`。統合実装は `3b38e4af67f6b673303cff9f963ac1381627c4d0`、取り込んだPR実装は `e433938dd39ac4c2cb6b7ab8865e41c0b996d232`。直前の `romutail-startup-e97575eb0` から、起動時の一括mapperと既存の配送、準備bundle・Gitオブジェクト共有、中央受領ファイルの共有を継承した。[検収記録](ACCEPTANCE.json) · [探索設定とアルゴリズム](../../PERFORMANCE_METHODS.md)

## PR-DF保存信号照合の追加採用

新規runの `current` は `romutail-reviewed-pr-9187e39b5`。optimizer SOURCEは `9187e39b5af87bfca4132875a59ce40a756e4cb6`。保存信号照合の上流は `f79708a7126ecbd827747bd261289d93496d9fa4`。同一物理キーで数値・検査・参照入力が一致した保存信号を使う。差があればラウンド・出所・相違項目を残し、該当候補の信号・価格準備を投入前に保留する。

RomuTail、server2、local0-wsl、PBSゲートウェイへoptimizerを配置した。worker・adapter・runtime/profileの固定版と工程別の受付条件は前の検収を継承する。Windows/CLIは明示指定とrelease無指定の両方でPreview通過。回帰32件中30件が通過し、ルート登録の残り2件は旧採用版でも同じ失敗を確認した。server2の旧R08はユーザー停止で終了しており、旧runの固定版とSTOPを保全した。新規信号・compile・fit・Q3は各0件。[追加検収](PR_DF_RECONCILIATION.json)

## 通常入口

- Romulaptopでは既存の `Invoke-RpeSearch.ps1 -Action Preview -RequestFile SCOPE.json` を使う。RomuTailのCLIは `scripts/run_rpe_search.py --config .rpe-central/CLIENT.json --action preview --request-file SCOPE.json`。
- 承認済みの範囲・準備予算・Q3予算を指定して `Start`。既存runには同じ入口で `-Action Status|Stop|Resume -RunId RUN_ID`、CLIでは `--action status|stop|resume --run-id RUN_ID`。既存runはrunに保存されたreleaseを解決する。
- 正本はRomuTailの `.rpe-central/state`。登録は `.rpe-central/RELEASES.json`、版ごとの設定は `.rpe-central/deployment/RELEASE_ID`、manifest・配置・profileは `.rpe-central/fleet/RELEASE_ID`。新releaseは既存のregistryに登録し、明示Preview後に `--activate --expected-current OLD_RELEASE` で採用する。
- 停止中のserver1は、復帰時に同じmanifestと専用host layoutを使い `manage_rpe_release_deployment.py --install` で配置する。固定入力・runtimeを照合し、人工ジョブの配送と回収を確認してから既存の受付設定で参加させる。

## 配分と保存

新規runのPR-Pauli／PR-DFでは `self_price_floor`、`review_search_order`、`common_pauli_price_floor`、`allocation_exchange.enabled` を有効化した。同一モデルで根拠を確認できたPauli範囲でκ方向の除外を使う。PR-DFの高価な候補は準備順を調整する。RomuTailが候補選択・既存予算・採用を一元化し、準備とQ3は従来の同じキューを使う。

`pauli_b_common_entry` と `prdf_d2_common_entry` ではREQUESTに圧縮証跡と厳密残差並列を記録し、適合する固定adapterのあるhostを選ぶ。cacheは各hostのprofileが指定する `-reviewed-pr` のパスで、各ディレクトリの設定上限は40 GiB。科学Q3 SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。旧形式のPF・PR-DF split用adapterは `349558b748a82f384a4589434a63ef9b5572032c`。

PR-DFの価格較正・holdoutと対応する監査を新runnerに固定し、X/Yをhost profileのCPU予約内で並列化する。元の共有イベントdigest、個別receipt、較正・holdout区分、stage別予算を保持する。実行する価格providerは `ea9e66092465d39dfcd4aeb6fea8e9f09808cabb`。PR信号の準備SOURCEは `d887abdcadf47a969aa2f69c1dd3bdc88c20f1c2`。

圧縮済みの原証跡は、runに記録したadapterで次のように復元できる。

```sh
"$FIXED_PYTHON" -B "$FIXED_ADAPTER/scripts/run_rpe_optimizer_q3_job.py" \
  --mode export-evidence --request "$JOB/REQUEST.json" --output "$JOB" \
  --evidence-directory "$NEW_DIRECTORY"
```

## 検収の範囲

RomuTailとserver2の人工runで、先行Q3の実行中に別候補の信号・価格準備が揃い、同じ候補IDがserver2の空き枠へ戻った。割当・採用は各1回、同一run再開後の追加起動は0件。別の人工検査で、圧縮証跡を中央の回収経路に流し、exact整数を保持したまま原サイドカーをバイト単位で復元した。合成ケースではworker側のattempt保存量が約50%、中央のjobと転送を合わせた重複除外後の保存量も約50%減った。server2の既報約54%はR08の完了42件の結果ファイルに対する値であり、今回の中央全体使用量の測定値ではない。

新規実H4信号、実compile、fit、実Q3は各0件。価格準備は保存入力を使ったREQUESTの構築と人工の軸・receipt検査を実施した。外部runnerの `prepare` 呼び出しは自動審査で拒否され、コード内の非実行検査に切り替えた。既存中央run `search-be59f46fceee86257816` は `romutail-b0i-hold-b89818df0-r08` を保持し、server2のR08は元のtmux、固定SOURCE・出力先で継続した。新規runの実計算は承認済み範囲の通常Startから行う。[R08の記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/882e66644ee08034a63aa92924fee2501b4b34a1/docs/research/pr_r08_storage_review_resume_20261118/HANDOFF.md)

保存REQUESTのdry-checkはRomuTail・server2・local0-wslで両PR方式がPASS。WSLは新release専用のPython 3.11.0rc1 launcherにSQLite拡張を追加し、PR-DFの不足Gitオブジェクト2件を補った。PBSゲートウェイではPR-Pauliの保存入力がPASS。PR-DFの保存fixtureでは固定providerの旧 `INPUTS.json` が不足し、該当routeはHOLD。PBSの実Q3には既存の残りwalltime照合と工程別見積りが必要。server1は停止状態。

人工回帰126件と、固定DP実装を使ったprefix cache・並列残差の人工回帰15件がPASS。
