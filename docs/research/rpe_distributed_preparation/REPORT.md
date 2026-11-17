# H4準備工程とQ3の共通分散実行

実装固定SHA：`f956f5197a5acfe113fbb2ed3bfa78564e7f4eec`。
branch：`codex/rpe-distributed-preparation`。
基点は分散実装349558b74、配置・検収91fe799dd。
通常の[開始・再開・local0一括配置](HANDOFF.md)、[既存providerとの接続](LEAF_PATHS.md)、
[SOURCEと検査locator](checks/SOURCE.json)を参照。

## 実装した動作

4方式の重い葉をsignal・compile・auditに分け、Q3と同じDistributedExecutor、JobStore、
worker、Windows relay、自動枠調整へ接続した。候補選択・証拠の照合と採用・探索状態はserver2に残る。
signalと価格枝は独立して進み、compile成果がそろえば別ホストでauditできる。
その候補の証拠がREADYになれば、他候補の準備完了を待たず、既存のshot探索・Q3へ戻す。
Full PF/B0Iの通常roundとPRの幅探索・通常roundに同じ接続を適用した。

各新工程REQUESTはstage/provider/SOURCEに合うprofileを使う。
固定入力のnested Git locator、WORKING成果物hash、readonly source view、native provider/configのpath、
実Pythonとcompiler packageを投入前に照合する。既存Q3のPASSは準備工程へ転用しない。
不足したホストだけをHOLDにし、別の実行可能ホストは進める。
Q3と準備が同じhost枠を重ねて予約することはない。優先3台の余剰を参加中のWSLへ渡し、
任意WSLのPause・未接続を開始待ちに数えない。

元のsignal/価格registry・cell/group key、既知の元ownerのSTARTED/STOPを照合する。
同じ物理契約の準備はsingle-flight、同じcompileは候補間で共有する。
回路・receiptはSHA256のCASへ置き、後続hostに不足したblobだけ移す。
新しいGit locatorはbundleで移し、元commit・原receiptを保つ。
切断は別hostへの科学retryにせず、元job/attemptの完了成果を回収する。

signal・compile・auditのqueue上限はそれぞれ明示設定し、Q3件数と分離した。
各providerのcell数、signal範囲、holdout、one-shot、失敗時STOPも維持する。
今回の配布設定の準備予算は0であり、実科学runを開始する設定ではない。

## 検査結果

固定実装で[80件の技術・回帰検査](checks/UNIT.json)がPASSした。
対象は共有枠、工程別readiness、artifact転送、Completion/既存providerとの接続、予算、候補選択互換、
通常と再開、保存証拠再利用、SOURCE/runtime変更の拒否、UNKNOWN/STOPの保全である。
新しい準備の正常経路と再開経路は別途読取確認も行った。
作者の検査を独立した科学レビューとは呼ばない。

通常入口から人工subprocessを使った[2構成](checks/ACCEPTANCE.json)は、ともに次を満たした。

| 項目 | first | swapped |
|---|---:|---:|
| signal完了 | 4 | 4 |
| 2候補で共有したcompile | 1 | 1 |
| compileと異なるhostのaudit | 1 | 1 |
| Q3完了・採用 | 8 | 8 |
| 再開時の新規起動 | 0 | 0 |
| 二重採用 | 0 | 0 |
| 人工run実時間 | 9.242秒 | 9.159秒 |

両構成でsignalとcompileのprocess実行が重なり、遅い候補のsignal完了前に別候補のQ3が始まった。
compile完了後・転送前の切断から同じattemptを回収し、WSLのPauseと途中参加も確認した。
compile担当・audit担当を入れ替えても、[exactなMSE上下界と判定](checks/first/EXACT_OUTCOMES.json)は一致した。
両構成のJobStore events、通常入口のPROGRESS、round/problem、OUTCOMEをchecks配下に保存した。
4つのhost名は**同じ1台のOS上の模擬host**であり、実4台の科学計算や速度倍率の測定ではない。

確認中に見つかった、共有artifactの同時採用競合、参加host変更中のmetrics走査、
fatal Q3後の候補継続、通常再開で準備中候補がattempt上限に隠れる問題は修正し、回帰検査へ加えた。
PF native compileの最初の38 AST文は[元D実装と一致](checks/PF_COMPILER_REUSE.json)し、
科学的な候補・shot選択と元のMSE判定は変更していない。

同じrun directoryへ既存の`summarize_rpe_q3_timing.py`を適用し、
[first](checks/first/TIMING.json)・[swapped](checks/swapped/TIMING.json)ともQ3 8件と準備6件を別々に集計できた。
body計時の欠測は0件。異なるhost時計の絶対時刻を引いて片道時間を推定しない。

## 工程×ホストの実入力検査

工程profileは4台とも設定済み。表のIは保存実入力とSOURCE/runtimeの実機inspection PASS、
未確認は今回の遠隔返送がまだないことを示す。新しい完全な実行REQUESTの起動可否は、
その都度通常workerのmetadata challengeで照合する。

| 方式 | 工程 | server2 | server1 | RomuTail | local0-wsl |
|---|---|---|---|---|---|
| Full PF | signal | I | 未確認 | 未確認 | 未確認 |
| Full PF | compile | I | 未確認 | 未確認 | 未確認 |
| Full PF | audit | I | 未確認 | 未確認 | 未確認 |
| B0I | signal | I | 未確認 | 未確認 | 未確認 |
| B0I | compile | I | 未確認 | 未確認 | 未確認 |
| B0I | audit | I | 未確認 | 未確認 | 未確認 |
| PR-Pauli | signal | I | 未確認 | 未確認 | 未確認 |
| PR-Pauli | compile | I | 未確認 | 未確認 | 未確認 |
| PR-Pauli | audit | I | 未確認 | 未確認 | 未確認 |
| PR-DF | signal | I | 未確認 | 未確認 | 未確認 |
| PR-DF | compile | I | 未確認 | 未確認 | 未確認 |
| PR-DF | audit | I | 未確認 | 未確認 | 未確認 |

機械可読の[対応表と不足理由](deployment_f956f5197/HOST_STAGE_READINESS.json)、
[server2の12件](deployment_f956f5197/server2/dry/READINESS.json)、
[配布物と配置検査](deployment_f956f5197/CHECKS.json)を公開した。
保存入力例は`dry_check_only=true`で、queueへの投入を拒否する。
`SAVED_INPUT_RUNTIME_INSPECTION_PASS`は、その候補の新科学REQUEST全体の実行PASSではない。
DF compile/auditのnative provider/configが遠隔hostにない場合は具体的path/hashの不足をHOLDに残す。
新工程の実4台での科学計算完了は未確認である。

既存4方式×4台のQ3については[全16セルの科学profileが同一](checks/Q3_PROFILE_REUSE.json)と照合した。
科学SOURCE28d・Q3計時adapter349・route・実Pythonを保持し、Q3 dry-checkを重複実行していない。
新しい準備transport/CAS relayはf956へ配置する。新stage用のsource view・profile・出力directoryを追加し、
既存run・spool・科学checkout・Python環境は変更しない。

## 次の実行に残ること

local0の初回実行ではRomuTailとWSLのrunnerが終了し、server1へのSSHは`No route to host`で失敗した。
利用者はserver1停止中と申告している。この接続ではserver1上の配置programは開始していない。
旧collectorの未処理例外で2台分の返送まで中断したため、hostごとの例外保存と、
既存のlocal0結果だけをqg2へ返す`--collect-saved`を追加した。
[修正の人工検査3件](recovery_transport_v1/TESTS.json)では、1台の接続失敗でも2台分を保持・返送し、
保存結果の回収時はqg2以外へ接続しないことを通常CLIで確認した。
旧bundle・工程実装・既存の実入力検査を再実行せず、[回収専用手順](HANDOFF.md)から返送する。
返送後はSOURCE・入力・profile・hashと照合して上表を更新する。server1は復旧まで未確認/HOLDを維持する。
WSLは受付停止で引き渡し、検査時もPauseを解除しない。
新しい科学roundは、準備範囲・工程別予算・Q3予算・exact readiness policyを採択してから通常入口で開始する。
controllerは可変の専用worktree、worker adapterとsource viewは固定の別領域とする。

実H4で今後測る指標は、方式別最安費用の推移、照合・採用済みQ3結果/時、準備工程結果/時、
次の探索判断までの待ち、計算・queue・転送・採用時間、DP/CAS cache再利用、host枠とCPU/RSSである。
今回の人工秒数を実H4の高速化率として報告しない。

新規実H4信号・価格較正・実compile・実Q3・本番queue投入はすべて0件。
着手時に確認した他担当の実行中run・原結果・UNKNOWN・STOPは保全した。
