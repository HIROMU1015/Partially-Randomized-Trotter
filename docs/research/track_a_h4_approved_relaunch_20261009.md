# H4再実行認可・13GiB固定・起動入口（2026-10-09）

利用者の「これについては問題ないので再実行して」を、提示済み累積charge13GiB案と一度のmap再実行への認可として反映した。
今回の変更は累積output charge上限10→13GiBと、それをplan/auth/schema/runtime保存へ結ぶ認可処理だけ。
**この資料の認可artifact時点では本体未起動。独立reviewとartifact固定後、fresh gateに合格した場合だけ一度起動する。**
実際のrun/PID/起動状態はhomeのruntime evidenceと最終報告で追跡する。

## 固定値と認可範囲

branch `track-a-h4-approved-relaunch-20261009`、SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`。
起点REVIEW `5bb241125717d06e0dabd244247f2a0dc9124228`、旧SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8` を保持した独立worktree。
[source closure39・旧/new blob/hash](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/source_freeze_v1.json)を固定した。
SOURCE commitはlaunch_binding.py/execution.py/prelaunch_audit.py/schema/test/new限定runnerの6files。
[利用者認可原文・適用根拠](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/USER_AUTHORIZATION_v2.md)を保存する。
科学条件・compiler options・12worker queue・距離内prepare再利用・ledger差分保存・monitor1秒/制限5秒は維持する。

新host認可に`output_budget_amendment`を追加し、10→13GiBのapproved=trueとauthorityを要求する。
許容するcapは10/13GiBのintだけ。13GiBには明示改定が必須、10GiBと改定approved=trueの不一致を拒否する。
launchのprepared budget、OwnedRun fallback、fresh budget検査へplan capを渡す。
共通OutputBudgetのlegacy default10GiBは変えず、予約前の累積cap検査・fsync順・no refundを保持する。
storage modelのmarginも今回の13GiBを基準とし、値の誤解を避ける。

| 条件 | 今回の固定値 |
|---|---:|
| 累積charge cap | 13,958,643,712 bytes（13GiB） |
| 累積worst charge | 13,165,893,832 bytes（12.261694GiB） |
| worst見積り後の余裕 | 792,749,880 bytes |
| carry actual | 20 |
| carry charged bytes | 4,428,938,712 |
| carry conservative wall | 5,472.345380863175秒 |
| actual cap / 新actual残枠 | 74,804 / 74,784 |
| 累積wall cap | 72h |
| driver/worker AS/RSS各上限 | 8GiB |
| headroom | 16GiB |
| observer AS/RSS | 256MiB / 64MiB |
| observer込みadmission | 120.25GiB |
| 新output物理容量案 | 5GiB / 301,000 inodes |

carryは返却・resetせず、全74,784 logical wrappersの最悪新actual74,784件を保持する。
再実行の実科学cache節約は保証しない。旧partial/random/cacheは使わない。
SOURCE変更後のtrajectory seedは新SHAへ再結合し、旧random結果を混合しない。

以前承認済みworker12 CPUs [2,4,5,6,8,9,10,11,12,13,14,15]、
driver16/observer18、own-process affinity・内部thread1と既存private environment/compilerを維持する。
候補環境はvenv変更なし。旧compiler output完全同一性は未検証のまま、18version/45raw RECORD差の履歴を保持する。
13GiBは実ディスク必要量ではなく、全予約とtemp/finalを含む累積charge上限である。

## 限定検証とbinding

[今回22 pure gate tests](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/limited_tests_v1.json)はPASS。
単一process・内部thread1、AS2GiB/RSS256MiB、wall60秒/output4MiB、child process/実worker/observer/affinity/transpile/scientific array/GPUは0。
13GiB未承認拒否、cap/type/authority/digest不整合拒否、実OutputBudgetに13GiBが渡ること、
carry込み10GiB超のreservation許可と13GiB超STOP、legacy10GiB保持、false flags/one-shot/fresh/source/own-write検査を確認する。
前の22件PASSの後、margin profile引数の変更に必要な同じ限定回帰だけを実施した。
旧48speedup・library/cleanup/native proof・128benchmark/旧synthetic campaignは再実行しない。
テストは人工fixture/科学stubであり、H4本体成功・性能・旧compiler同等性の証明ではない。

新SOURCEと[認可binding](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/authorization_binding_v11.json)を固定した。
既存環境/compiler/library cache profile、受領済み6凍結NPZ/freeze/native83controlと新host14停止identity/carryへbyteとmetadataで結合した。
旧hostの未記録exit code/正確な終了時刻はnullのまま保持する。
plan sealed=true、auth/review approved=true/runtime_authorization=trueを今回の明示指示に基づき設定した。
[plan](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/plan_authorized_v11.json)、[auth](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/authorization_v11.json)、[review](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/review_v11.json)は最終digestで結合する。
独立担当による[最終review](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/independent_relaunch_review_v1.json)はPASS_READY_FOR_FRESH_ONE_SHOT_LAUNCH、blocking findingなし。
closure39・三digest・profile/cache/byte receipts/carry・22人工結果原本とargvを独立照合した。担当による本体・回帰再実行は0。
限定[runner](../../scripts/resource_applicability/run_h4_output_amendment_tests.py)と[tests](../../tests/tracks/resource_applicability/test_h4_prelaunch.py)で改定条件を追跡する。

## 一度の起動と停止

候補run IDは `h4-newhost-signal-compile-20261009-run02`。
output：`/home/AbeHiromu/projects/h4-handoff-evidence/20261009/approved-relaunch/h4-newhost-signal-compile-20261009-run02`
control/log：`/home/AbeHiromu/projects/h4-handoff-evidence/20261009/approved-relaunch/control/h4-newhost-signal-compile-20261009-run02/runner.log`
observerはoutputの`observer.jsonl`、終了判定は`launch-stop.json`または`map-complete.json`で追跡する。

[runner argv](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/runner_argv_v11.json)を固定する。実行commandはそこから一度だけ使う。
SOURCE/profile/cache/input/carry、CPU許可/physical core/受動負荷、memory/PSI/OOM、
filesystem/block/inode/user-group-project quotaをartifact固定後にfresh確認する。
output/controlは未使用で、old one-shotと失敗証拠を保持する。exclusive one-shotは繰り返し実行を拒否する。
runner内でも同じsource/input/fresh gateを再検査し、合格後だけclaim/own affinity/observer/workersを起動する。

完了またはfail-closed STOP後にown childrenをcleanupし停止する。自動retry、入力再生成、次stage、GPU query/useは行わない。
共有環境・venv・他jobを変更しない。公開は軽量資料のみでNPZ/raw runtime/checkpoint/cache/log/credential/内部SSHはGit外。
認証失敗時は設定を変えずmanual push commandを報告し、local immutable artifactと独立reviewに基づく一度の起動条件を維持する。

## 実起動結果：worker/IPC処理でfail-closed STOP

固定artifact `1a9e920b13f29ee0cb7e38c1f79ae16f775e78f3`と独立reviewの後、
fresh gateはPASS（memory465.986GiB、filesystem411.341GiB、inodes220457784、quota KNOWN、選択core最大busy4.984%）。
runner自身でも再検査し、driver2865403/observer2865493/12 workersを一度起動した。

最初のcosine wrapperのreservation後、driverがsineのnumerical serialization中に、
observerは「Stop: owned process exited」を記録し、own driver/childrenをfail-closedで停止した。
driver実exit code143。直前の監視interval約1.007秒、観測所要約0.0056秒で、容量超過や5秒監視違反としては記録されていない。
新reserved1（science-000021）、完了wrapper0、signal record0、MAP未完了。
actual compiler呼出開始の独立証拠はないため、実transpile成功/開始を推測せず、reservation1を消費として保持する。

worker/IPCの元例外は停止前の永続資料に残っていない。worker stderrはDEVNULLで、
親I/O futureのfailureはメモリ内のままdriverが停止した可能性があり、根本原因を断定しない。
今回の一度起動は終了しており、自動retry・追加science campaignや次stageへ進まない。
次の修正では元例外を停止前に永続保存できる経路が必要である。このSTOP後にsourceを変更して再実行していない。

[停止・予約・native identity監査](../../artifacts/resource_applicability/track_a_h4_approved_relaunch/2026-10-09/runtime_stop_summary_v2.json)で全14own identitiesの残存0を2回確認した。
one-shotと全失敗費用を保持。累積消費/予約21、残actual74783、charged8692723164 bytes。
正確なdriver終了時刻・compiler終了時刻は未記録。
今回のwallは、process開始clock tickから2回目の残存0監査までを加えた
保守的upper5766.582514658794秒として記録し、post-stop監査待ち時間を含むと明記する。
実行時間とこの保守的upperを同一視せず、過去wallを返却しない。

共有環境・venv・他job・GPUは変更なし。raw runtime/observer trace/NPZ/科学cacheはGit外。
この追記は結果commit用の軽量STOP summaryで、上記認可artifactの起動前snapshotは保持する。
