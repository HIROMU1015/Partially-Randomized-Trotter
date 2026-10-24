# H4 Pauli PR0 cosine Arb包含評価（2026-10-25）

## 対象と承認

基点 `f817c492c527d4e436b81abda9e9341defa1c49d`、専用branch `codex/non-df-h4-central-arb-pr0-cosine-20261025`。
ユーザーの「その方針で、次の作業に進んで大丈夫」を、前段で提示したPR0 cosine・64→96→128bit・各target1＋compiled1・最大6状態作用の実行承認として適用する。
[承認記録](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/approval.md)。

固定linear H4 / 隣接1.00 Å / STO-3G、Pauli-JW184非identity項、retained prefix27、S2 / T=.8 / q8 / r4 / K4、system8＋ancilla8。
PR index0、cosine一件だけ。保存32eventはすべてorder0。
保存raw state/ray・source/control/input/compiled QPY・元C SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d` のexact operandを使用する。

## 接続と包含性

前段の `h4_arb_pr0_preparation.py` からselected C-target数値部分を共通private kernelへ分離した。
人工用public/private入口のH4拒否は維持し、新しいproduction controllerがSOURCE/入力/runtime/契約/未使用grantを検査してdispatchする。
旧worktreeと固定SOURCE blobは変更しない。新worktreeの進化したcodeは別科学SOURCEへ固定する。

retained角はbinary64係数×binary64 delta/2を**exact有理積**として評価し、event角は保存binary64値、Pauli sign/product/order phase/作用順、各stepのexact scalar operandを保持する。
selected C-targetはraw braとの内積の実部/共通exact ν。compiledは既存native `rz/sx/x/cx`・global/relative phase・qubit配置を保持し、最終ancilla-Z/同じνを評価する。
両者とも既存Arb/FLINTのexact初期値・unitary L2 recenterを使い、C-targetはsqrt(ν)×e、compiledは2||m||e+e²を最終期待値へ転送する。
raw stateの再正規化、出力normによる除算、確率clippingはしない。

各precisionで2つの独立作用をprocess-local Arb contextに分け、別call予約後に並行する。
各childのnative join/exit/reapを記録し、既存C native witnessが親workerのexitと残留する自分の子孫だけを監査する。
precision間は早期STOP判定への逐次依存がある。CPU/RAM/affinity/worker数への恣意的な制限、共有設定変更は0。
Pauli permutation・角のtrig cache・raw入力共有を再利用する。非有限の場合も、既に開始した独立作用は不要に強制中断しない。

区間差は保存する両信号のwhole enclosureを使う。compiled端点からの再unionは外向きで、midpointによる置換はしない。
`ρ_local = B_saved × |m_compiled_cos − m_C_selected_cos|`、
`B_saved = 1449450801590761/1125899906842624` を一度だけ適用する。targetは非scale。
Arb ballの下端が負ならraw ballを保持し、数学的非負性との共通部分を別fieldで表示する。
PR全law mean・R0経験的点値を保証targetとして使わない。

## 実行契約と人工検証

保存予算は `19792741170690161541/7631344236317346665332736`（約2.593611368819805e-6）。
64bitから開始し、上端≤予算または下端>予算が決まればその時点でSTOP。
straddle/非有限の場合だけ96→128へ進み、128でも未決ならINCONCLUSIVE。
各precisionのC-target1＋compiled1、最大6状態作用・scalar縮約最大3。call予約は非返還。
幅1e-8は参考値で、決定後の不要な昇精度はしない。

新production制御13＋既存selected C-target13＋compiled Arb10＋QPY6＝**42人工テストPASS**。
2〜3qubitの独立dense参照包含・位相/符号/順序・exact operand・B尺度・測定を含み、production追加では並行2childのnative回収、共通ν、grant排他、早期STOP/非有限/精度枯渇を検証した。
保存実QPY4個を静的decodeして前段のexact plan/descriptorと同一性を確認した。ここまで実H4状態作用0。
確認は主agent本人によるもので、別reviewerによる独立reviewではない。

## 実行結果と監査

科学SOURCE **`f1f3049670234cfc9c65430bd5016b9c6315a375`**、新grant **`ac99c3df12320ee5c27a2cdc01d8f8bf4fa90c10e93438eb9b40065832063def`** を一回発行・消費。
[封印](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/seal.json)、[実行契約](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/active_contract.json)、[結果](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/result_summary.json)。

**64bitで上端≤保存予算が確定した。状態作用試行/成功2、scalar縮約1。96/128bitは未実行。**
計算callはC-target1＋compiled1だけで、失敗call/technical retryは0。

| 保存量 | 包含区間（十進端点は外向き24桁表示） | exact区間幅 |
|---|---|---|
| selected C-target cosine | [-0.106488293230470611846399, -0.106488293230470428412943] | 13535/73786976294838206464 ≈1.83433e-16 |
| compiled cosine | [-0.106488293230470956290653, -0.106488293230470248415054] | 6529/9223372036854775808 ≈7.07876e-16 |
| raw Arb ρ ball | [-4.67870948374473227725180e-16, 6.79574416339626407633468e-16] | 23831507850181476353/20769187434139310514121985316880384 ≈1.14745e-15 |

ρ≥0との共通部分は **`[0,14114208428403524923/20769187434139310514121985316880384]`**。
したがって **ρ_local ≤6.79574416339626407633468e-16 < 保存予算2.593611368819805e-6**。
非負表示の0は新ballの数学的交差であり、残差exact0の認定や旧nullの読み替えではない。
幅1e-8も満たすが、停止理由は保存予算との包含比較が確定したこと。
[raw ρと別fieldの非負交差](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/rho_64.json)。

C-target作用は約0.62744秒、compiled作用は約2.72173秒で、別processとして重なって実行した。
state hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`、同じexact νを使用。
selected retained432 rotation＋32event＋8 scalar、compiled3139 native gate。
source/control/QPY fingerprint・binary64値・global/relative phase・配置を保持する。

親worker PID666324、C-target childとcompiled childはいずれもnative exit0・回収。
親のnative witnessはexit0、残留子孫の強制終了0、all_owned_reaped=true。
[call ledger](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/call_ledger.json)、
[target native](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/target_64_native_exit.json)、
[compiled native](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/compiled_64_native_exit.json)、
[親native](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/native/worker_native_exit.json)、
[最終STOP](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/run01/STOP.json)。

科学code23・runtime206（FLINT数値package103を含む）・実入力17を封印前後でSHA照合した。
FLINT0.9.0/FLINT3.6.0と既存private環境を再利用し、package変更0。
旧C snapshotとの差は既知libz.so.1.3 SHA差だけで、数値packageは同一。原因UNKNOWNの履歴として記録し、今回の実runtimeを新規固定した。
[失敗によるものではないruntime観測](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/runtime.json)。
古いruntime全体と同一と主張せず、共有libraryを変更していない。

保存有理端点のwhole区間差とB一回・予算比較、SOURCE/runtime/実入力、旧証拠515 SHA・旧manifest188行、grant/claim/native exit/STOPを科学再計算なしで検証する。
[stdlibのみの検証](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/verify_saved_results.py)、
[検証結果](../artifacts/non_df_h4_central_arb_pr0_cosine/2026-10-25/verification.json)。
元の準備用DRAFT_DISABLED契約・STOPと旧SOURCE/results/grants/ledger/markersは保全する。

## 範囲と終了

正式certificate受理、LOCAL_BUDGET_METへの昇格、PR全lawρ、費用tail、C/L重複、旧N/G=null、元H4 native UNKNOWNは解決済みにしない。
新compile/sampling/Hamiltonian/state生成、PR1/sine/DF/R2/N_G再解析は0。
旧証拠515file・旧manifest188行を保全し、workerの取消gateは実際に使うSOURCE/入力/runtime/契約だけへ限定する。
提供された2026-10-23 policyに§10.3がない版不一致を記録し、実在する§8.4/10/10.1/10.2を適用する。

結果保存とcommit/push/remote確認後mandatory STOP。
次はユーザーが追加対象・軸・call範囲を判断し、Codexは別承認後だけ指定範囲を実装/実行する。
