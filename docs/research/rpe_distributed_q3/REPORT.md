# 分散Q3実装と固定SOURCE人工検収

2026-11-17 JST。固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb`、
計算SOURCE（controller/worker） **`4ae47cf4844638647c83ea898ee53588675b0d75`**。
local0 relay・配置helperの補正SOURCEは
**`25575ce0f3b43396144c768e9f7dab7dc193beab`**。
公開branchは `codex/rpe-distributed-q3-coordinator`。
起動・停止・再開・接続設定は [HANDOFF](HANDOFF.md)、
数値の小記録は [PUBLIC_ACCEPTANCE.json](checks/PUBLIC_ACCEPTANCE.json)。

4aeを計算SOURCEとし、local0 relayのWindows補正SOURCEを別に管理する。
旧716のLinux人工検収は [716参考記録](checks/PUBLIC_ACCEPTANCE_71667549e.json)
として保全した。以下は固定4aeのfresh runで取り直した採用記録。

## 到達点と未完了範囲

通常schedule入口から使う分散Q3を実装し、固定4aeで同一OS上の別spoolを使う
模擬1/2/3ホスト人工検収を完了した。実行終了codeは0。
サーバー2で実機2台用の専用人工controllerを起動し、server1参加を待っている。
**実サーバー間2台検収、計算用ローカルを加えた3台検収は未完了。**
local0は計算用ローカルと別ホストで、サーバー2環境からlocal0へのアクセスはない。
待機中controllerの生存は実機接続や受入PASSの証拠ではない。
実H4科学計算は0件で、稼働中のH4探索には途中導入していない。

実装と人工検収はサーバー2主担当およびsub-agentの作業であり、独立した科学レビューではない。
探索fixture/模擬1〜3host検収は `integration_fixture`、worker/relayと分散DP検収は
`q3_protocol`、store実装とその自己検査、および主担当作のcontrollerに対する独立した
技術検査は `persistent_store` が担当した。主担当は統合実装、最終unit/通常入口回帰、
ZIP配置検査と公開内容を確認した。store作者の自己検査を独立レビューとは数えていない。
今回の数値は人工系の実装・運用検査で、H4の科学的最安や最終総コストの結論ではない。

## 既存経路への実装

`run_rpe_schedule_search.py` → 共通 `make_executor()` →
`rpe_optimizer_concurrent.DistributedExecutor` の経路で、既存の候補順・方式別最安・
科学予算・採用順をサーバー2に保持する。
slotを申告した各hostは同じ共通queueから独立Q3を得る。
方式、L_D、幅をhostに固定しない。

既存のQ3 adapterのdry-check/execute/collect、Q3内部DP並列化、prefix context、
準備後のREADY再投入を使い、単一ホストの既存起動・再開経路を維持した。
新規moduleは `rpe_q3_store.py`（割当・予算・採用の永続化）、
`rpe_q3_worker.py`（host別spool・supervisor・回収）、
`rpe_q3_coordinator.py`（loopback API・artifact transfer）。
worker/relay CLIは `scripts/run_rpe_q3_worker.py` と `scripts/run_rpe_q3_relay.py`。

科学job IDとhost/attemptを分離し、SQLite transactionで二重取得を防ぐ。
通信断だけで割当をexpireしない。未開始・起動状態不明・完了未転送を区別し、
同じ保存attemptを照合する。完了重複通知は冪等に受理し、採用と予算は各1回。
SOURCE/入力blob hash/route/評価契約と終了code/signalを確認してから結果を公開する。
選択されたconfigurationの固定blob hashをcacheし、無関係な候補入力の繰返し読込を避ける。
DP cacheはhost内に置き、先頭候補順と待ち時間を保って局所性を考慮する。
承認を証明する認証・証書・承認台帳は追加していない。

実機接続方式はlocal0→qg2のSSH forwardingとlocal0→qg1のmailbox。
server1→server2直接SSH、共有filesystem、計算用ローカルの未確認aliasには依存しない。

## 固定4aeの人工受入

通常schedule CLIの `main()` に既存人工producerを渡す検収runnerを使用した。
2方式PR_PAULI/PR_DF、L_D=1,2、複数幅、最終roundの2段shot、
exact有理数MSE、各job sleep=0.12秒。分子geometry・basis・DF rankは非該当。
初回Q3完了後に人工価格が届き、待機幅がREADYになって探索が続く。

単一legacy参照と模擬1/2/3hostはすべて同じ18科学job ID集合を評価した。
全18件のMSE上下界と判定がexact一致し、両方式の最安人工費用26も一致した。
各hostの完了結果を同じ探索で採用し、通常再開の追加起動は各case0件。

| 模擬構成（実OSはすべて1台） | controller実時間 | 分散1host比 | legacy単一参照比 | queue待ち総計 |
|---|---:|---:|---:|---:|
| legacy単一参照 | 3.458秒 | — | 1.000倍 | — |
| 分散1host | 10.328秒 | 1.000倍 | 0.335倍 | 18.945秒 |
| 分散2host | 4.935秒 | 2.093倍 | 0.701倍 | 6.816秒 |
| 分散3host | 4.075秒 | 2.535倍 | 0.849倍 | 2.954秒 |

人工leafが短いため、分散版はこの測定でlegacy単一参照より遅い。
分散1host比の倍率は模擬hostを増やした人工測定であり、実ネットワークやH4実DPの
速度倍率ではない。起動・SOURCE照合・dry-check・転送・server終了の費用を含む。

| 構成 | host | 起動=完了=採用 | 保存結果再利用イベント |
|---|---|---:|---:|
| 1host | simulated-server2 | 18 | 26 |
| 2host | simulated-server1 | 11 | 16 |
| 2host | simulated-server2 | 7 | 10 |
| 3host | simulated-local | 6 | 8 |
| 3host | simulated-server1 | 6 | 9 |
| 3host | simulated-server2 | 6 | 9 |

再利用数は `RESULT_REUSE` イベントの累積で、同じexecutor invocation内では
同じ科学jobを1回だけ数える。幅の継続と最終再開を含むため、distinct job数とは異なる。
このexact有理数leafにDP cache HITはなく、保存結果再利用とDP prefix HITを区別する。

| 模擬構成 | controller受信後の検証・保存時間総計 | workerのupload/ACK往復総計 |
|---|---:|---:|
| 1host | 0.372秒 | 0.540秒 |
| 2host | 0.391秒 | 0.581秒 |
| 3host | 0.379秒 | 0.590秒 |

queue待ちは18jobの割当時刻−予約時刻の和で、同時待機を含む。
受信後検証・保存時間はnetwork往復を含まず、workerの往復時間と加算して
全体実時間の内訳にする値ではない。

## 切断・再開・重複通知

別の専用人工caseで、Q3開始後に通信を遮断しcontrollerをSIGKILLした。
controller終了codeは−9。detached workerは完了し、
`COMPLETED_PENDING_TRANSFER` を保存した。
同じSOURCEとrunでcontrollerを再開し、同じhost/job/attemptから回収した。
対象jobの人工評価開始は1回、保存数値と既存LAUNCH markerは不変。
同じ完了を1回追加通知し、`ok=true, duplicate=true` を確認した。

復旧後は新しくREADYになった別科学IDの10件を続行した。
これは既完了jobの再起動と区別する。収束後の再開は追加起動0件で、
方式別最安も参照とexact一致した。recovery caseの実時間は13.863秒。

## DPと検証範囲

固定4aeの人工unitは29件PASS（22＋7、[UNIT_TESTS](checks/UNIT_TESTS_4ae47cf48.log)）。
通常入口の回帰29件もPASS（[回帰log](checks/REGRESSION_TESTS_4ae47cf48.log)）。初版4ae時点のWindows固有分岐の人工検査3件とLinux relay検査2件は
[ZIP記録](checks/LOCAL0_ZIP.json) に記載した。Windows native実行はlocal0待ちで、
これらの人工分岐検査をWindows実機検収と扱わない。
既存prefix-cache回帰10件と通常CLI人工DPの2件もPASS
（[PREFIX_SUMMARY](checks/PREFIX_SUMMARY.json)、
[CLI_SUMMARY](checks/PREFIX_CLI_SUMMARY.json)、
[IDENTITY_SUMMARY](checks/PREFIX_IDENTITY_SUMMARY.json)）。
通常CLIの元DPを使う既存人工検査では4 HIT、各12round再利用、
cache無効の逐次参照とのMSE上下界・判定exact一致、再開追加起動0を確認した。
固定4aeの分散worker経路で人工表の元DPを使い、分散8件とlegacy8件の
MSE上下界・判定・方式別最安exact一致、4 HIT各12round、2模擬hostそれぞれ2 HIT、
同hostの完了donor再利用、host別cache分離を確認した。
再開追加Q3・完了準備stage呼出しは0件、worker終了codeはすべて0。
[分散DP要約](checks/DISTRIBUTED_PREFIX_SUMMARY.json) と
[DP prefixイベント](checks/DISTRIBUTED_PREFIX_DP_PREFIX_EVENTS.jsonl)、
[固定SOURCE blob照合](checks/DISTRIBUTED_PREFIX_SOURCE_CHECK.json) を参照する。
分散実時間4.149秒、legacy参照3.391秒、参照/分散=0.817倍で、
この小人工DPも分散版の方が遅い。queue待ち総0.373秒、worker転送往復総0.203秒。
18jobのexact有理数leaf検収のHIT=0を、DP cache機能の否定や実H4での効果測定とは解釈しない。
分散版の再現入口は `scripts/check_rpe_distributed_prefix.py`。
既存fixtureは `tests/test_pr_width_prefix_execution.py`、host内の再現入口は
`scripts/check_pr_width_prefix_integration.py`。
分散scriptの既定SOURCEは旧716なので、4aeを検査する際は次のように両方を明示する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48
RPE_DISTRIBUTED_SOURCE_ROOT=/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-distributed-q3-source-4ae47cf48 \
RPE_DISTRIBUTED_SOURCE_SHA=4ae47cf4844638647c83ea898ee53588675b0d75 \
PYTHONDONTWRITEBYTECODE=1 \
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python -B \
  scripts/check_rpe_distributed_prefix.py --output /tmp/rpe-distributed-4ae-prefix-NEW
```

採用記録は固定4aeのfresh runから作成した。
作業中の先行人工runではworker SOURCE更新をguardが検出しUNKNOWNで停止した。
そのrun、中間89f51a4feと716の検収記録は保全し、固定4aeの受入値に混ぜていない。
この人工開発時の停止と、実機参加がまだない状態を区別する。

## Windows実機でのterminal多重発生とrelay補正

local0で初版ZIPを起動すると大量のterminalが開き、利用者がSTOP fileによる
停止を確認した。親relayのdaemon起動を隠す設定に加え、SSH mailbox用の
`subprocess.run` とtunnel用の `Popen` にも `CREATE_NO_WINDOW` を渡す必要があった。
補正版は両経路へ適用し、起動直後の `stopped=false` と初期heartbeatも保存する。
後者はlocal担当4058の再開時表示修正を再利用した。
固定25575ce0fのrelay人工検査7件とhelperのfake SSH/一時Git repository検査3件がPASS
（[補正検査log](checks/RELAY_CORRECTION_TESTS.log)）。
補正ZIPのstatus/help smokeもPASSした。科学worker/controllerのSOURCE4aeは変更していない。
Windows実機で余分なterminalが出ないことは、補正版の再開時に確認する。

提供logはrelay PID34652の `stopped=true`、server1の固定4ae worker rootへの
`No such file` を示した。qg1のPythonには到達したが専用checkoutが未配置である。
TUNNEL.logが空であることから、tunnel接続成功を認定しない。
計算SOURCE4ae、controller PID97835、人工runのSOURCE/入力/DBは変更せず、
新規Q3起動0件を維持した。稼働中の実H4探索にも影響を与えていない。

補正ZIPはSOURCE.txt=25575ce0f、COMPUTATION_SOURCE.txt=4aeとし、
server1 profileと初回配置helperを同梱した。
[補正ZIP記録](checks/LOCAL0_ZIP.json) と [再開手順](HANDOFF.md) を参照する。
利用者がlocal0からhelperを明示実行すると、qg1の既存repositoryをfetchし、
固定4aeの専用detached worktreeとprofileを設置してmailbox statusを確認する。
既存root branchや旧結果を変更せず、relayを再開する前に配置不足を解消する。

## 実機統合の現在地

716の待機controller PID73114は科学Q3起動0件のまま停止し、SOURCE・入力・DBを保全した。
4aeの新専用SOURCE checkoutと人工runを分離し、controller PID97835、
loopback8763、local slot1/total2、required_hosts server2/server1で起動した。
記載時はQUEUED4件・割当0件で、server1のpoll登録待ち。
server1用固定worktree/profileとlocal0用relay設定・4操作を公開した。
[待機状態の記録](checks/REAL_TWO_HOST_STATUS.json) に起動0件を残した。
server1担当の `d8ee4efe4` は旧89 SOURCEのmailbox statusと単体人工jobの準備記録で、
実機参加前に4aeの専用checkout/profileへ更新する必要がある。
local0のrelayは一度実行され、terminal多重発生によりSTOP fileで停止した。
server1専用checkoutの配置不足を確認しており、補正版helperでの配置とrelay再開は未確認。

計算用ローカル担当の `4058b48d` は、利用者確認のlocal0→RomuTail aliasと
既存Linuxのworker配置を記録した。実際のSSH接続計測と3台目参加は未完了。
local固有のhost DP上書き拡張は今回統合せず、controller指定DPを使う。
実2台で両hostの結果採用→次候補のserver1再配送→切断回収を確認し、
次に新しい専用runで計算用ローカルを加える。
実際に通った構成・host別件数・待ち/転送/全体時間は、その受入後に追記する。

固定SOURCE・入力・結果・attemptの保全と既存STOPを維持する。
新科学実行、過去のH4 runへの導入、実科学の再解析は今回の成果に含めない。
