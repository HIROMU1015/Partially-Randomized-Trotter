# H8 B0 anchor run01：fresh CPU gateで科学処理前STOP

**利用者の「次の作業に進んで大丈夫」を、公開済み契約内の4構成・8軸取得一回の承認として起動を試みた。launcherのfresh CPU idle gateが `PHYSICAL_CORE_BUSY` で拒否し、driver・compile worker・科学処理は開始していない。自動retry・CPU再割当・追加compileを行わず、run01をSTOPとして閉じた。**

実行SOURCE：`572a337d0dc3fa78b2fd6b25ace6a37845db384e`。
実行準備commit：`847266323ea3d7bd29ac42322b7819b0b2a6c43f`。
結果branch：`track-a-h8-b0-anchor-run01-stop-20261016`。
[原証拠のhash・STOP状態・保全監査](../../artifacts/resource_applicability/track_a_h8_b0_anchor_run01_stop/2026-10-16/README.md)。

## 試行したscope

linear H8、隣接1 Å、STO-3G、16 modes、Nα=Nβ=4、sector4900、保存tol-only DF rank28（tol1e-8/cutoff0、signed generation order、weighted Hermitian projection）、T=.8。B0 S2のTRAIN p14/q1・p21/q1・p14/q8、HOLDOUT p21/q8、各cosine/sine。q1/8、δ=.8/.1、最大4 worker・2 wave。B0+Iの取得、fit、残差、S4、再会計、PR sampling、新SCF/DF/stateは承認範囲に含めていない。

[公開済み実行契約](track_a_h8_b0_anchor_execution_readiness_20261016.md)のSOURCE・preparation・compiler・資源上限は変更していない。6物理core・実効空きRAM176.25 GiB以上を含むfresh gate通過が条件であり、失敗時に条件を緩めて起動しない。

## 起動前確認と停止点

1. 公開済みprivate configのSHA、SOURCE、未使用control/output/lock/fresh-gate pathと旧保護状態を確認した。
2. 事前の資源観測では実効空きRAM465.07297 GiB、host/cgroup full PSI avg10=0、既存OOM counterを記録した。filesystemはext4、mount options `rw,relatime`、空きdisk/inodeは契約条件以上だった。これは本番fresh gate通過の証拠ではない。
3. 2秒間の事前CPU観測でidleと判定された6物理coreを選択した。driver0、observer2、worker5/6/7/10。全選択coreの事前観測busy fractionは0だった。
4. 新scopeの承認済みgrantを新規作成し、launcherを一回起動した。既存draft/grantは変更していない。
5. launcher内のSOURCE/input/runtime照合後、1秒間のfresh CPU idle checkが `PHYSICAL_CORE_BUSY` を送出した。fresh gateの保存、supervisor、driver forkより前で停止した。

事前の2秒観測と起動直前の1秒観測は別である。原コードは失敗CPU番号・各counter差分を保存していないため、**どのcoreがどれだけbusyだったか、外部jobが原因だったかは確定できない**。現在のCPU状態を測り直して当時の値として補完していない。

launcherのtracebackと原stderr SHAを保存した。launcher PIDは現存せず、当該driver/launcherの生存processもない。detached launcherがsupervisor生成前に終了したため、launcherのnative exit codeは取得できずnull/unknownのまま残す。exit1やwait4確認済みと断定しない。

## 科学処理・予算・原証拠

|項目|確認した状態|
|---|---|
|fresh gate|未発行、未通過|
|control/output directory・science one-shot lock|すべて未作成|
|driver / compile worker|未起動 / 0|
|snapshot decode / preparation復元 / native build / control probe / compile|すべて0|
|科学予約・fit・残差・再会計|すべて0|
|再起動・CPU再割当・追加compile|すべて0|

承認済みgrantは発行後のbytes/SHAが一致し、STOP後も改変していない。原stderrとprivate bindingも保全した。source/provider/runtime/inputの読取専用再照合は実施したが、fresh gateを再試行して起動を目指す処理は行っていない。

旧676 grant/ledger/carry/one-shot関連ファイル、115旧worktreeのHEAD/status、旧V2未追跡scriptの不変を確認した。新しい結果worktreeだけに公開可能なSTOP記録を作り、実行worktreeはHEAD `84726632…` のまま保持する。SOURCE14/provider265/runtime34,273/input8のbyte照合もPASS。V1・V2 NO_GO・H5/H7・旧B0/新B0I結果は変更していない。

## 今回の結論と次の境界

新B0 anchorは未取得で、旧B1の来歴不足STOPは未解消。科学的な一致・費用・model精度について新しい結論はない。B1 fit、B2 compile、C再会計への承認もない。

**run01の承認試行はSTOPとして終了し、このgrantや実行域を自動再利用しない。再実行は別途明示承認後、新grant・未使用実行域を用い、元契約のfresh条件を満たした場合に限る。今回のSTOP成果物をcommit/push・remote SHA/tree確認後、mandatory STOPする。**
