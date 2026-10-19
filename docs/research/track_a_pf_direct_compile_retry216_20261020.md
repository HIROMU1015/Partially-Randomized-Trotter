# PF直接compile：loader修正・残り216軸の承認済み再実行

2026-10-20 JST。利用者の「その方針で進めて、再実行まで行って」を、残り216軸の修正・再実行承認として記録した。基点はa80f89b435340898f8cc46940e9080c9bcd3b403、旧科学SOURCEは220b9644df6e114116737fd041973973fcf86cf5。当初は封印・起動前の準備段階であった。以下の実行記録を現状として優先する。

[packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20/README.md)。対象はlinear H2/H4/H8、隣接1 Å、STO-3G、tol-only DF3/10/28、T=.8、P=0、symmetric_directional。H2 72軸、H4 DF10 122軸、H8 DF28 22軸、計216軸＝108 cosine/sine pair。保存受入572軸（H6/H8/legacy DF12）は旧task来歴を保持して再利用し、再コンパイルしない。旧profile等の48軸は対象外。

## 修正と検査

H2/H4の配列receiptは元実装どおりascontiguousarray後のshape・dtype・bytesを照合する。H8は保存policy REUSED_SAVED_P21_FULL_FOCK_SCALARの入れ子だけを展開し、元scalarへ保存Cを一度だけ加える。元receiptを変更しない。

分離sessionのobserverがcoordinatorを起動し、実wait4でcoordinatorを回収する。coordinatorは各workerを実wait4で回収し、PID/start ticks・exit・CPU・RSS・結果SHAを保存する。observer自身の終了は自己報告であり、外部からのnative終了証明と呼ばない。起動元終了後の継続と成功・失敗の終了監査を人工processで確認した。新13件＋既存境界13件PASS。実分子の追加試験compileは行わず、本契約内のcontrolで検証する。

SOURCE・入力・provider・compiler/runtime・profile・旧受入結果をbyte照合して新sealへ固定する。旧実行のglobal final gate欠測は補わず、572軸のtask単位受入という限界を継承する。旧SOURCE/raw/grant/ledger/marker/STOPは変更しない。

## 実行と資源

新216軸だけのgrant・ledger・marker・STOPを独立作成する。予約はnative108、control432、wrapper build/compile各216。新sampling/fit/bootstrap/shot再計算/費用再ランキングは0。技術異常は該当task・契約でSTOPし、本attempt後の自動retryを許可しない。

並列数は実際のaffinity/cgroup・空きメモリとtask予約量から決める。H8は同一n16/q8/S4/profileの確認済み旧worker peak RSS 5,336,395,776 bytesを基準に構造比で見積もる。見積もりはadmission用で、プロセスのRAM上限ではない。CPU/RAM/worker/qの独自上限、PSIの一時上昇による強制終了、無関係worktreeの一括凍結を設けない。共有設定・他利用者を変更しない。

## 次の担当と停止条件

Codexがseal、1回の分離起動、保存結果・終了監査を行う。新SOURCEと結果の公開範囲は完成後にユーザーが判断する。詳細科学レビュー・最終evidence freeze・モデルfit・費用再会計は行わない。完了または実行不能時に理由・成功/STOP/未実施・時間/資源を保存しmandatory STOPする。

## 実起動記録（2026-10-20 05:18 JSTの暫定値）

科学SOURCEは`0c5fcc83ef55045c2c64746aadfaf3128667b058`。依存38,686件のseal gate・起動直前gate PASS。observer PID199948はPPID1/session199948、coordinator PID200066は別session200066で、起動元の終了後も実workerが継続している。実CPU affinity 64に応じて初回64 workerを投入した。固定worker上限ではなく実資源admissionによる値である。

保存されたnative終了確認時点（2026-10-19T20:18:27.474399+00:00）は成功86軸、control STOP106軸、未終了24軸。STOPはすべてCONTROL_MISMATCHで、native/controlの後、wrapper build/compile前。H2/H4 B0IとH2 B0 prefix0/S4に分かれる。STOPを成功へ繰り上げず再試行しない。残りH8等の正常workerは継続する。この時点で全体完了・全worker回収済みとは認定しない。

保存SOURCEの静的確認では、B0Iのloaderが元providerと異なり_prepare_discardを選ぶ不一致を確認した。事前の静的照合で見落としていた点である。数値寄与やH2 prefix0/S4の原因は未検証。失敗したcontrolの個別errors/vectorは保存されておらず、保存記録だけでは原因を確定できない。sealed SOURCEは変更しない。

[RUNNING_HANDOFF.json](../../artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20/RUNNING_HANDOFF.json)に実PID/start ticks・進捗・終了receipt経路を保存した。observerが全worker終了後にmetadata auditorを一度起動し、成功/STOP・ゲート数・旧572との来歴をresults01へ保存する。216未達ならpartial結果を保存してexit2となり、observerもSTOPを記録する。これは再実行承認ではない。最終報告・結果commitは終了確認後にCodexが行う。

本source/起動記録の新しい公開範囲は未承認でpushしていない。旧40ファイルの公開承認を今回へ拡張しない。最終evidence freeze・詳細科学レビューは未開始。

05:20 JST追記：running_structure_validation01.json時点でH2/H4は終了し、成功88軸・control STOP106軸・H8実行中22軸。source16file不変、observer/coordinatorのPID/start ticks一致。最終結果は未生成。
