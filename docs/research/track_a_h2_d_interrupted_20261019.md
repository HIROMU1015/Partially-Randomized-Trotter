# H2 D_run01：native完了未確認によるSTOP（2026-10-19）

**Dは未完了。状態は `STOP_NATIVE_COMPLETION_UNCONFIRMED`。**
FP完了を正式受入し、固定D70を利用者承認により1回起動したが、
チャット中断後にdriver/native/既知workerが全て不在であることを確認した。
native exit・同一親wait4・terminal・最終resultが残っておらず、終了コードと終了原因は確定できない。
中断が先行した事実と、プロセス終了の原因の認定は区別する。
再起動・自動retry・A以降の起動は行わず、部分結果と全予約を保全してmandatory STOPした。

## 承認と固定範囲

利用者は「Dのみの実行承認を判断」の注釈を選択し、
「疎の作業に進んで大丈夫」と回答した。
FP完了commit `af724928689be3592b391fab3a14538bf4752b76` の受入と、Dだけの承認として記録。
後続の「続きを行って」は既存attemptの状態確認・保全として処理し、
消失したプロセスの新規再実行を承認したとは扱っていない。

- 科学SOURCE：`de72e5d4e48177734c0ddd79dffac14222410658`、23 filesを変更していない。
- linear H2、1.0 Å、STO-3G、tol-only DF1e-8の自動決定rank3、4 qubits。
- Nalpha=Nbeta=1のspin sector次元4、保存状態の再生成・再正規化なし。
- T=.8、P=0、ε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1。
- full Hadamard wrapper、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17、symmetric directional。
- B0/B0I prefix0/1/2、full PF prefix3、q1/2/4/8/16、S2/S4の事前登録70条件。
- 保存FP result SHA256：`0b43326c03b1e6786841920d0f38ad5db01fe5c22d9e1172e2354b87987bf80f`。

[元のSOURCE・入力・call予算](track_a_h2_fp_d_a_preparation_20261019.md)を維持し、
H2所有 `D_activation01` でFP result/completion audit/acceptance SHAとfresh resource/grantを結合した。
実行前のSOURCE・F入力・runtime照合とFP保全監査はPASSした。
観測された62低負荷CPU・約469 GiBの空きメモリから候補間並列化し、
CPU/RAM/worker/wallの恣意的上限、共有環境変更、他job操作は追加していない。

## 保存状態

| 対象 | 保存・確認できた状態 |
| --- | --- |
| one-shot marker | D_run01を消費済み。元SOURCE/spec/grant digest一致 |
| 70条件のsignal | 全条件のJSONを保存。FP6条件を同一bytesで再利用 |
| 新規base signal | 36 worker全てexit0・同一親回収・result hash一致を保存receiptで確認 |
| B0I analytic overlay | 新28条件をaggregate signals内に保存 |
| primitive | 137時間×3状態＝411 probeの検証記録を保存 |
| native compile | 28候補の開始記録、12候補のraw task result、27軸のraw measured記録 |
| control | 18構成のcontrolファイルを保存。全compile workerの終了確認は不可 |
| native/compile終了 | driver/nativeと28既知compile workerは不在。exit code・全体reap receiptは不明 |
| 最終会計 | 0 pass、最良PF未取得、`result.json`なし |

signal poolの36 workerに対しては、親が作成したPASS receiptに終了・回収証拠がある。
compile poolにはそのreceiptがない。12 raw task resultについて両axisのcopy/fingerprint形式は検査したが、
**native終了未確認のため正式な費用として受理していない**。
プロセス不在やtask resultの存在を、exit0・同一親回収の証明へ置き換えていない。
結果欠測を費用0としていない。D70の最良PFやPRとの優劣は認定しない。

## 非返還の予約

| category | 予約 | 元の上限 |
| --- | ---: | ---: |
| preparation / primitive spectrum | 4 / 4 | 4 / 4 |
| primitive probe | 411 | 411 |
| truncated reference pair | 2 | 2 |
| base signal / independent forward | 36 / 36 | 36 / 36 |
| B0I overlay | 28 | 28 |
| new signed shot endpoint | 320 | 320 |
| native evolution | 28 | 64 |
| wrapper compile | 56 | 128 |
| control probe | 112 | 256 |
| endpoint会計pass | 0 | 1 |

SCF・積分生成・DF・保存状態solver・新full-H reference・sampling・実shotsは予約0。
compileは全task開始前に56軸を予約済みであり、raw measuredが27軸でも予約は56のまま保全する。
未確認の実消費を0や27と確定せず、返還0・再試行0とする。
native wall/exit codeの確定値は取得できず、推定値で埋めない。

## 監査・保全

観測codeは `scripts/audit_h2_d_interrupted.py`。
Python `-S` のstdlibのみで、SOURCE/runtime・grant/依存入力・原activation・F295レコード・
FP204 private files・元準備sealのhashを確認した。
既知PIDと記録済みstart ticksに限定して生存を調べ、他ユーザーや他worktreeのプロセスを変更していない。
新しい科学計算、NPZ decode、sampling、compile、最良PFの再計算は行わない。
これは主担当Codexの保全監査であり、独立科学レビューではない。

原D attempt/activationの541 filesを、scratchを除く保全manifestに固定。
元のSOURCE・grant・ledger・marker・raw logsを変更せず、観測者のSTOPを別領域
`D_interruption_audit01/observer_STOP.json` に作成した。
存在しないnative_exit・terminal・worker reap・科学resultを後から作っていない。

公開用記録は
[`D_interrupted_run01`](../../artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/D_interrupted_run01/)。
元記録のcopy、非返還予約、raw未受理compile、FP受入、保全証拠を16 filesに分けた。
新たな公開push承認はないためpushしない。

## 再開の草案と次の作業

元の実行契約は「atomic claim後は再使用不可」「自動retryしない」「異常時はSTOP」を要求する。
従ってD_run01のmarkerを削除したり、同じgrantで再起動したりしない。
保存signalの再利用を前提にした `recovery_contract_DRAFT.json` を保存したが、
**SOURCE未固定・runtime_authorization=false・launch commandなしの実行不可草案**である。

草案では保存70 signalとFP対照を入力として扱い、未受理compile12候補を費用へ混合せず、
新規28候補/56軸・control112・会計1 passを提案する。新signal/DF/state solver/samplingは0案。
必要なnative preparationは2案。元の56軸予約はこれと別に保持する。
最終的な科学call予算は新SOURCEの接続検査後に固定する。
チャット切断に依存しないプロジェクト専用driver起動を人工テストで検証してから、新attemptへ結合する必要がある。

| 担当 | 次の作業 |
| --- | --- |
| Codex | 保存partialの入力受入規則・新SOURCE・再開契約と、切断に耐える起動方式を技術準備する。元D_run01は保全する。 |
| ユーザー | その固定SOURCE・入力・追加予算が提示された後、新D attemptの科学実行を別途明示承認するか判断する。 |

現在はmandatory STOP。A/S_STATS/P・詳細科学レビューには進まない。
