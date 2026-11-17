# local0-wsl：任意の第4計算ホスト

中央controllerはserver2、計算hostはserver2・server1・local（RomuTail）・local0-wsl。
Windows local0は中継を担当する。WSLは受付停止で引き渡す。

## 割当表（全16セルPASS）

PASSはこのREQUESTの実入力・SOURCE・route・実行Pythonを照合した実行可否を表す。
受付設定とは別であり、実Q3の完了を表さない。

| 方式 / request ID | server2 | server1 | local / RomuTail | local0-wsl |
|---|---|---|---|---|
| Full PF / 9b9b4b5132ff89768c5c | PASS | PASS | PASS | PASS：専用3.11・固定AST guard |
| B0I / 72864f6cf968d3b514b1 | PASS | PASS | PASS | PASS：専用3.11・固定AST guard |
| PR-Pauli / 805877c2ab6c3e03f7f1 | PASS / route33e | PASS / route33e | PASS / route33e | PASS / route33e・専用3.11 |
| PR-DF / 48494ae31276a64c46ce | PASS | PASS | PASS | PASS：専用3.11.0rc1 |
| 中央の新規受付設定 | ON（接続時） | ON（接続時） | ON（接続時） | **OFF、上限1、任意参加** |

[HOST_ALLOCATION.json](HOST_ALLOCATION.json)は全16セルの実行可否と受付設定を別fieldで出力する。
既存3台のONは通常の中央受付設定であり、今回workerが稼働中という観測ではない。
今回の新しい検査用に本番controller・worker・relayは起動していない。
WSLのPARTICIPATIONは追加検査toolがPAUSEDを読み取り、検査前後のbyte一致を確認した。
サーバー2からWindowsへの直接操作は行っていない。

## 既存成果の照合と差分

基準は`3cd8b1769ffc4df71d85bdaefce0e337becf028e`。
[IMPORT_VERIFICATION](IMPORT_VERIFICATION.json)に返送bundle・SERVER2_IMPORTのhashと固定Git証拠の検査結果がある。
既存12セルは実行profileとfingerprintが基準と完全一致し、既存検収記録を再利用した。
科学SOURCEは元の`28d976b99cdf8329adcc1b79327e329e947ef774`、計画は元339のまま。
4つのrequest全体hash・入力hash・評価契約は中央の対象と一致する。
Pauliは元bd42の記録を保全し、採用済み33e経路のREQUESTを照合した。

- `6e70f5d130618722ba44e8b79ed14e37fa7a1360`：任意WSL worker・実機人工検収の公開成果。
  元bundle SHA256は810066c9b6d0df5b84241662238510770888aac3de4836739110d867649602b9。
  44原証拠を固定manifestと照合。
- `de02a717f0bf2aa4ae4424b4386f89682457cf32`：WSL PR-DF専用profile・公式実入力PASS。
  元bundle SHA256は707e07bb76132ecab51a5a34a1aa51271517a92828041b01d9e86bd5a5d04355。
  50原証拠を固定manifestと照合。
- WSL PR-DFは公式READINESS.command[0]、HOST_METHOD_READINESS、adapter_result.runtimeを結合した。
  checkerの3.12を実adapterのPythonと取り違えていない。実バイナリSHAは5374e2d1d711fd9cc01fb15c48e3915718765b8200fb315f719d9d0017fa6331。
  portable adapter e37、内側27eは元profile.route_viewsのまま。元assignmentのSOURCEを変更していない。
- 残るWSL3方式は旧3.12のmetadata-only PASSであり、実行時AST guardの証拠がない。
  基準3cd8は既にこの問題を修正しているため、そのPASSを実行可能として流用しない。
  既存の専用3.11 launcherと同じ固定guardを選ぶroute profileを追加し、3セルだけの検査を公開した。
  初期policyではこの3セルをUNCHECKEDにした。追加3セルの実機PASS受領後に全16セルへ更新した。元PASS記録と初期13 PASS policyは上書きしていない。
  返送34fileをmanifest SHA6074c8949c540508b3979684556778e42bd123ceb44ab2909ff758c911d5d11aで照合。
  実測はFull PF 1.78秒・B0I 1.73秒・Pauli 2.66秒（各cellの全準備/dry時間、並列実行）。
  既存12セルとWSL PR-DFは再dry-checkしていない。

## 通常の入口と任意参加

既存の通常準備コマンドが4ホストを出力する。新しいdirectoryを使う。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir /home/AbeHiromu/projects/rpe-h4-ready-optional-wsl-NEXT \
 --plan-only --prepare-readiness \
 --preparation-cache /home/AbeHiromu/projects/rpe-q3-host-eligibility-checks/cache
```

`HOST_ELIGIBILITY.json`はREQUESTごとの能力、`HOST_ADMISSION.json`は現在の受付設定、
`HOST_ALLOCATION.json`は両方の表、`worker_profiles/local0-wsl.json`は同じ検査profileを出力する。
`DISTRIBUTED_OPTIONS.json`の通常executor設定は両policyを読み、WSLをrequired_hostsへ追加しない。
これは設定断片なので、科学round側のmax_concurrent_jobsを参加hostの枠に合わせて指定する。
今回の人工通常入口はmax_concurrent_jobs=4で検査した。指定を省略すると既存CLI既定1となる。
新REQUESTは元の完全照合を行う。今回のPASSを別REQUESTへ一律転用しない。

中央の設定は次の通り。JSON fileのatomic置換が次のpollで反映される。

```json
{"schema":"RPE_Q3_HOST_ADMISSION_V1","hosts":{"local0-wsl":{"accepting":false,"max_slots":1,"optional":true}}}
```

新しい科学runを別途採択した後に参加させる場合は、そのrunのHOST_ADMISSIONでacceptingをtrueにし、
local0側の既存参加操作をJoin/Resumeにする。停止は中央accepting=falseとlocal0のPause。
片側でもOFFなら新規割当を止め、受領済みattemptの照合・結果転送は続ける。
中央の枠1はRUNNING・UNKNOWN・完了転送待ちを含む未解決attemptを数える。

中継のWSL項目は公開SERVER2_IMPORTのtransport=wsl/distribution=Ubuntuを継承する。
通常の`scripts/run_rpe_q3_relay.py`へwsl.exe mailbox配送を統合したので旧runtime-c847 wrapperは不要。
次回の新runではcontroller・中継・WSL transportを共通版の固定checkoutへ配置し、
worker.rootは新transport、profile.rootは固定科学28d、profileのroute_views/adapter/runtimeは検査済み値を使う。
旧de02のtransportへ戻すと新runtime検査を使えないため、新規REQUESTの受付条件を満たさない。
旧spool・旧run・既存PROFILE.currentは書き換えず、新しいprofile fileとspoolを指定する。
今回、実機の新transport常駐workerは起動していない。

## 検査

[ARTIFICIAL_NORMAL_ENTRY](ARTIFICIAL_NORMAL_ENTRY.json)：通常schedule入口から、WSLが一度もpollしない状態で
既存3ホストが各1件取得。WSL停止時0件、参加後に8空き枠を申告しても1件だけ取得。
再停止後の完了回収と同run再開を確認し、追加LAUNCH・二重採用は0。
人工fixtureのみで、実worker/adapterプロセス起動は0。

[CHECKS](CHECKS.json)：新規の受付5検査、準備・中継3検査、差分tool4検査、WSL4候補profile検査1件がPASS。
[WSL profile人工割当](WSL_PROFILE_ASSIGNMENTS.json)でWSLの全4候補について、
4方式の独立人工queueで、実workerクラスから専用311・方式別route・DFのe37 adapter/内側27e route_viewsを使う引数まで確認した。
実際のPopenはmockで、実計算は行っていない。
元の12セル・43検査suite全体・実2/3台人工検収・切断復帰検収は固定記録を再利用した。
新規受付検査の初回に、既存queue unit 6件もdiscoveryされたが、既存12セルの実機検査は再実行していない。
新しいWSL profileの実worker引数確認は、対応する保存証拠がPASSした全4セルについて人工queueで検査した。
4件のmock supervisor/adapter引数、再開時追加起動0、OUTCOME/採用0を確認（4.20秒）。

## 通常準備の実測

[NORMAL_ENTRY](NORMAL_ENTRY/READY.json)は全16 PASS反映後、通常schedule入口から実行した。
READYまで7.54秒、報告生成まで8.07秒。初回の13 PASS段階は74.44秒で、変更した準備codeのcontext cache初回生成68.86秒を含む（NORMAL_ENTRY_INITIAL_13へ保全）。
サーバー2の4 dry-checkは全件cache再利用、新しいdry-checkは0件。
4対象REQUESTのID・hashは維持され、全16 PASS、WSL受付OFFを生成した。
公開3台基準の12セルのprofile・fingerprintは維持している。

## 完了した差分検査

[local0実行手順](local0_tools/RUN_WSL_RUNTIME_DELTA_V1.ps1)はWSLの3セルだけを検査する。
既存3.11環境を使用し、current profile・spool・cache・参加状態は変更しない。
失敗した返送だけを再開する場合は[tool README](local0_tools/README.md)の--return-onlyを使う。
返送先は`evidence/runtime_delta_v1`。34fileを受領・照合済みで、policyと割当表へ反映した。
この手順の再実行は不要。

今回の実H4 Q3・signal生成・価格較正・回路compile・本番queue投入は0。
実Q3完了や科学速度向上の主張はない。既存科学run・UNKNOWN・STOPは変更していない。

## 次回のWSL配置（今回未実行、科学起動なし）

共通transport codeの固定版は[CODE_SOURCE](CODE_SOURCE.json)の1c4710384f11d4b262d120a62c6a00161d622105。
生成profileは[WORKER_PROFILE](WORKER_PROFILE.json)、中継のworker設定は[RELAY_WORKER](RELAY_WORKER.json)。
WSLの現在のPROFILE.currentは保全したため、PF/B0I/Pauliの旧3.12設定が残る。
**Join前に新しい生成profileを指定する。** 古いprofileを使ったpollは現在fingerprint不一致で割当を受けない。

Ubuntu WSLで[SETUP_NEXT_WORKER.sh](SETUP_NEXT_WORKER.sh)を実行すると、固定共通transportと
検査済みprofileのcopyを新しい専用directoryへ置く。OS・共有環境・現PROFILE/spoolを変更せず、
worker・中継・controller・Q3を起動しない。既存directoryが異なる内容なら上書きしない。
この配置scriptは構文確認済み、WSL実機での配置は次回runの作業として未実施。

Windowsでは同じ共通固定版のscripts/run_rpe_q3_relay.pyを使い、新しいrunのrelay設定に
RELAY_WORKER.jsonのworker項目を追加する。既存qg1/RomuTailのworker項目は維持する。
新runのHOST_ADMISSIONで許可し、既存tools/local0-wsl/Invoke-Local0Wsl.ps1のJoin/Resumeで
Windows参加flagを有効にしたときだけ受付する。Pauseはこのflagをfalseに戻す。
今回の引渡しは中央accepting=false、Windows参加flagも差分検査前後でPAUSEDを確認した。
