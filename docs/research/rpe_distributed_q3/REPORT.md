# 分散Q3実装と固定SOURCE人工検収

2026-11-17 JST。固定基点 `5829a7288d28395d2b3e89bfb676c23c690b9ddb`、
**次回公開版の固定SHAは `28d976b99cdf8329adcc1b79327e329e947ef774`**。
通常実3台と専用faultの計算SOURCEは `1ffdf610a682a23e2a0a388b570d33d0f353fedc` を保持する。
完了した実2台runは計算SOURCE `4ae47cf4844638647c83ea898ee53588675b0d75`、
Windows relay/helper SOURCE `25575ce0f3b43396144c768e9f7dab7dc193beab` を使用した。
公開branchは `codex/rpe-distributed-q3-coordinator`。
[HANDOFF](HANDOFF.md)／[模擬人工検収](checks/PUBLIC_ACCEPTANCE.json)／
[実2台公開要約](checks/REAL_TWO_HOST_PUBLIC_SUMMARY.json)。

4aeの模擬1/2/3host受入、通常実2台、1ffの通常実3台runを完了した。
**通常実2台・3台と専用実機faultはすべてPASS。fault relayも停止済みで、追加利用者操作は不要。**
新1ffでは巨大exact整数の保存・再読込とhost別DP設定を追加し、新しい専用3台runへ固定する。
完了した4ae runへ途中導入せず、そのSOURCE・結果・attemptを保持する。
旧716の人工根拠は [716参考記録](checks/PUBLIC_ACCEPTANCE_71667549e.json) として保全した。
次の表の模擬測定と実2台測定は、SOURCEと実行条件を分けて記録する。

## 到達点と適用範囲

通常schedule入口から使う分散Q3を実装し、固定4aeで同一OS上の別spoolを使う
模擬1/2/3host人工検収と、Windows local0を中継したserver2+server1の通常探索を完了した。
実2台の全18件が単一参照とexact一致し、両hostの結果採用、server1への後続配送、
完了runの再開追加起動0件を確認した。
実機の通信切断・復帰とcontroller中断後の回収は、別の専用fault runでPASSした。
通常run、模擬fault、実機faultの条件・証拠を分けて保持する。
local0は計算用ローカルRomuTailと別ホストである。
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
この人工開発時の停止と、後に別runで完了した実2台・3台・実機faultの受入を区別する。

## Windows実機でのterminal多重発生とrelay補正

local0で初版ZIPを起動すると大量のterminalが開き、利用者がSTOP fileによる
停止を確認した。親relayのdaemon起動を隠す設定に加え、SSH mailbox用の
`subprocess.run` とtunnel用の `Popen` にも `CREATE_NO_WINDOW` を渡す必要があった。
補正版は両経路へ適用し、起動直後の `stopped=false` と初期heartbeatも保存する。
後者はlocal担当4058の再開時表示修正を再利用した。
固定25575ce0fのrelay人工検査7件とhelperのfake SSH/一時Git repository検査3件がPASS
（[補正検査log](checks/RELAY_CORRECTION_TESTS.log)）。
補正ZIPのstatus/help smokeもPASSした。科学worker/controllerのSOURCE4aeは変更していない。
補正版のWindows local0経由で実2台の配送・回収は通った。
余分なterminal抑止に関する利用者の観察と、通常配送の受入は区別して記録する。

提供logはrelay PID34652の `stopped=true`、server1の固定4ae worker rootへの
`No such file` を示した。当時はqg1のPythonには到達したが専用checkoutが未配置だった。
TUNNEL.logが空であることから、tunnel接続成功を認定しない。
計算SOURCE4ae、controller PID97835、人工runのSOURCE/入力/DBは変更せず、
補正準備中は新規Q3起動0件を維持した。その後同じ4ae runで人工18件を完了した。
稼働中の実H4探索にも影響を与えていない。

補正ZIPはSOURCE.txt=25575ce0f、COMPUTATION_SOURCE.txt=4aeとし、
server1 profileと初回配置helperを同梱した。
[当時の補正ZIP記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c1119524d3d449a67c18271adae8ce3a6cb73dd/docs/research/rpe_distributed_q3/checks/LOCAL0_ZIP.json) と [再開手順](HANDOFF.md) を参照する。
利用者がlocal0からhelperを明示実行すると、qg1の既存repositoryをfetchし、
固定4aeの専用detached worktreeとprofileを設置してmailbox statusを確認する。
既存root branchや旧結果を変更せず、relayを再開する前に配置不足を解消する。

## 実2台受入：server2+server1とWindows local0

根拠は [REAL_TWO_HOST_ACCEPTANCE](checks/REAL_TWO_HOST_ACCEPTANCE.json)、
[timeline](checks/REAL_TWO_HOST_TIMELINE.jsonl)、
[公開要約](checks/REAL_TWO_HOST_PUBLIC_SUMMARY.json)。元記録は書き換えていない。
計算SOURCE4ae、relay25575、local0→qg2転送＋local0→qg1 mailboxを使った。
server1の固定checkout・profile設置後に両hostが同じ探索へ参加し、
全18jobのMSE上下界・判定と方式別最安が単一参照とexact一致した。
server1への次候補割当は5回記録され、ホストへ方式・幅を固定していない。
完了runのcontroller再開は終了code0、追加起動0件、再開実時間0.925秒だった。

| host | 起動 | 完了 | 採用 | 結果再利用イベント | distinct再利用job |
|---|---:|---:|---:|---:|---:|
| server1 | 5 | 5 | 5 | 7 | 5 |
| server2 | 13 | 13 | 13 | 19 | 13 |

再利用イベントは3つのexecutor invocationにわたるPROGRESSから数えた。
同じjobはinvocation内で1回とし、全18jobが少なくとも一度再利用された。

| 時間・観測 | 値 | 定義・欠測 |
|---|---:|---|
| 最初のLAUNCH〜最後のEXIT | 21.262秒 | job開始後の処理・転送区間 |
| controller開始〜最後のEXIT | 1107.508秒 | local0準備とrequired-host barrier待ちを含む |
| queue待ち総計 | 4375.960秒 | 18jobの予約〜割当の和。最大1090.476秒。並行待機を加算 |
| controller受信後検証・保存総計 | 0.342867秒 | 18件。network往復を含まない |
| worker upload/ACK往復総計 | 1.229827秒 | 17/18件の通知。最後の1件は未回収で欠測 |

controllerの待機時間込みのwallと、最初のjob起動後の区間を分けて示した。
queue待ちの総和は並行待機を含み、wall timeとは一致しない。
ACK往復は欠測1件を含まない部分総計で、18件の総時間として補完していない。
この実機runから速度比は算出していない。

実機切断後の回収は `NOT_YET_TESTED`。通常完了のPASSを切断検収へ拡張しない。

## 新SOURCE1ffの固定人工再検収

[full要約](checks/final_1ffdf610a/FULL_COMPACT_SUMMARY.json) と
[SOURCE照合](checks/final_1ffdf610a/FULL_SOURCE_CHECK.json) はPASS。
模擬1/2/3hostすべて18job集合・MSE上下界・判定・方式別最安が参照exact一致し、
通常再開追加0件。controller kill/同attempt回収/重複完了の模擬検収もPASSした。
実OSは1台で、次の実3台受入とは分ける。

| 模擬host数 | 実時間 | 分散1host比 | legacy参照比 |
|---|---:|---:|---:|
| 1 | 11.386秒 | 1.000倍 | 0.326倍 |
| 2 | 5.432秒 | 2.096倍 | 0.684倍 |
| 3 | 3.937秒 | 2.892倍 | 0.944倍 |

この小人工負荷もlegacyより遅く、H4性能へ外挿しない。
1ffの [unit log](checks/final_1ffdf610a/UNIT_TESTS.log) は29＋7＝36件PASS。
[分散DP](checks/final_1ffdf610a/PREFIX_SUMMARY.json) も8＋8件のexact一致、
4 HIT各12round・各host2 HIT・host別cache・同host donor再利用・再開追加0件を確認した。
分散4.028秒、参照3.327秒で参照/分散=0.826倍。worker終了codeはすべて0。

## 通常実3台受入：server2+server1+RomuTail

根拠は [受入記録](checks/REAL_THREE_HOST_ACCEPTANCE.json)、
[timeline](checks/REAL_THREE_HOST_TIMELINE.jsonl)、
[公開要約](checks/REAL_THREE_HOST_PUBLIC_SUMMARY.json)。元記録は変更していない。
SOURCE1ff、local0→qg2/qg1/RomuTailのmailbox relayで18件を完了した。
全MSE上下界・判定・方式別最安は単一参照とexact一致し、controllerはreturncode0・signalなし。
完了run再開もreturncode0、追加起動0件だった。

| host | 起動=完了=採用 | 結果再利用イベント | distinct再利用job | requested DP | resolved DP |
|---|---:|---:|---:|---:|---:|
| server2 | 11 | 16 | 11 | 1 | 1 |
| server1 | 2 | 2 | 2 | 1 | 1 |
| local (RomuTail) | 5 | 8 | 5 | 1 | 2 |

元REQUESTのDP数1を保ち、host profileの実行時解決を1/1/2へ適用した。
計算SOURCE・科学job ID・元REQUEST/hashを変更せず、全18件の同等性を確認した。

| 時間・観測 | 値 | 定義・欠測 |
|---|---:|---|
| 最初のLAUNCH〜最後のEXIT | 22.384秒 | job開始後の処理・転送区間 |
| controller開始〜最後のEXIT | 257.311秒 | host配置・required-host待機を含む |
| controller return wall | 258.180秒 | 最後のEXIT後の終了処理も含む、終了code0 |
| queue待ち総計 | 953.511秒 | 18件の和。最大236.329秒、並行待機を加算 |
| controller受信後検証・保存総計 | 0.340113秒 | 18件。network往復を含まない |
| worker upload/ACK往復総計 | 1.165449秒 | 17/18件。最後の通知1件は欠測 |

待機込みwallとjob開始後の区間を分け、ACK欠測を補完していない。
この実機runから速度比は算出していない。

巨大exact整数の4500桁制限修正とlocal4058のhost DP拡張を1ffへ統合し、
完了4aeへ途中導入せず、別の専用3台runで確認した。
実3台の初回controller/supervisorは142163/142162で、これは完了runの履歴値である。

## 実機fault受入：relay停止・controller中断・同attempt回収

根拠は [REAL_FAULT_ACCEPTANCE](checks/REAL_FAULT_ACCEPTANCE.json)、
[RUNNING観測](checks/REAL_FAULT_RUNNING.json)、
[転送前](checks/REAL_FAULT_BEFORE.json)、[転送後](checks/REAL_FAULT_AFTER.json)、
[実行stage](checks/REAL_FAULT_STAGES.jsonl)。全記録を保全した。
計算SOURCE1ff、transport/読み取り専用inspector SOURCE28dを使い、
通常3台とは別directory `/home/AbeHiromu/projects/rpe-distributed-q3-integration-1ffdf610a/fault_two_host` で実施した。

条件はserver2とserver1-fault、方式ごと1件・計2件、人工sleep12秒。
通常18件探索の最安26と比べる検査ではなく、固定2件の回収・同等性・冪等性を検査した。
2件とも参照の同じjobのMSE上下界・判定とexact一致し、人工executeは各1回だった。

server1-faultのjob `e87dc9349021da5eabf8`、
attempt `f547446f3d384b34b92d2cc0fc2c93b4` はrelay停止中に
RUNNING→COMPLETED_PENDING_TRANSFERへ進み、再接続後TRANSFERREDになった。
adapterの終了codeは0、signalなし、再executeは行っていない。
転送前・転送後・controller保存OUTCOMEは同じbyteで、SHA-256は
`15417a5286b42f1198341d8f513307943af5d4b139161b5df3653bdfa2184b02`。

初回controllerを意図的SIGKILL 9で中断し、同じSOURCE・同じrunから再開した。
再開はreturncode0・signalなし、完了後に通常入口でも追加起動0件を確認した。
その後、科学実行を無効にした回収専用RPCへ、保存stageの原本とdigestが一致する
完了payloadを再送し、`ok=true, duplicate=true` を確認した。
重複再送前後でACCEPT件数・ADOPT件数・予算予約・LAUNCHは変化しなかった。

全体のSTORE ACCEPTは2件、直接ADOPTイベントはserver2の1件、
予算予約はPR_DF=1/PR_PAULI=1。
remoteの1件はcontroller再開後にPRIOR_ROUND_JOB/RESULT_REUSE経由で
PR_DF方式別bestへ取り込まれた。remoteの `adopted_epoch` はNoneのままであり、
2件とも直接ADOPTされたとは記載しない。両方式のbestは保存結果にあり、
科学的な結果採用と計測イベントの経路を分けて報告する。既存DBを後から補正していない。

| 時間・観測 | 値 | 意味 |
|---|---:|---|
| 初回controller wall | 577.518秒 | 準備・待機から意図的SIGKILLまで。人工計算時間ではない |
| server1-fault adapter実行 | 12.066秒 | PROCESS_RETURNのwall、人工sleep12秒を含む |
| relay停止後の明示待機 | 15秒 | 手順中の待機設定。計算時間とは別 |
| 最初のLAUNCH〜最後のEXIT | 33.853秒 | 通信停止・回収・再開を含む区間 |
| 同SOURCEからのcontroller再開 | 7.907秒 | returncode0、signalなし |

速度比は算出していない。最後に利用者がfault relayの `--stop` を実行し、
この検収のための追加操作は不要となった。

## 次回利用する固定SHAと保全

次回公開版は **`28d976b99cdf8329adcc1b79327e329e947ef774`**。
retry backoff、完了jobの保存log維持、ASSIGNED仕事も開始しない読み取り専用inspectorを含む。
計算に使用する8 SOURCEファイルは実3台・faultの1ffとbyte一致し、
固定計算SOURCE1ffの稼働中差替えは行っていない。
実2台の計算4ae/relay25575、実3台の計算1ff、faultの計算1ff/transport28dを
それぞれ実際に通った構成として保持する。

開始・停止・再開コマンド、ホスト別path、ZIPとprofileは[HANDOFF](HANDOFF.md)に記載した。
次の科学runは新しい固定SOURCE/入力/評価契約と許可範囲で行い、今回の人工結果を
実H4性能や科学的最安へ外挿しない。稼働中の実H4へ途中導入せず、
SOURCE・入力・結果・attemptと既存STOPを保全した。
