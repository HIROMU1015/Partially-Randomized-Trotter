# M1-S 保存γ境界とπ/3：限定照合完了・STOP

2026-10-17。ユーザーの「計算を開始してよい　中断された場合は修正して計算完了まで行って」を受け、採択済み契約Cの限定照合を一回実行し、初回で正常完了した。statusは`PI3_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT`、failure=null。技術中断・SOURCE修正・追加試行は0。科学的GO、研究方針の新規採択、次期数値実験へ自動移行しない。

## 対象・条件・保存結果

linear H4 1.00 Å、STO-3G、8 modes、保存DF rank12、G3/G6/G9、保存δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1。同じ保存cellのshared ±対ordinary／independent ±という48対照を全件保持した。G3/G6の32対照だけを数値照合し、G9の16対照は未解像性を保持して照合しなかった。元108方式行・36zero control、state、Hamiltonian、複素信号を再生成していない。

| 分割 | 成立 SATISFIED | 不成立 NOT_SATISFIED | 閾値照合の未解像 UNRESOLVED | G9絶対差未解像・対象外 |
|---|---:|---:|---:|---:|
| G3 | 16 | 0 | 0 | 0 |
| G6 | 8 | 8 | 0 | 0 |
| G9 | 0 | 0 | 0 | 16 |
| 合計 | 24 | 8 | 0 | 16 |

G3はT=1,4 Ha^-1で各8対照が成立。G6はT=1 Ha^-1の8対照が不成立、T=4 Ha^-1の8対照が成立した。各行は2つのδ、2つのR、2つの対照方式を含む既存gridであり、都合の良いcellを除外していない。全48行と有理数区間の正確な文字列表現は[CSV](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/pi3_comparisons.csv)と[raw result](../../../artifacts/track_b_m1s_gamma_pi3_result/2026-10-17/attempt_01/result.json)を参照する。

## 判定の意味と限界

保存γ*区間[L,U]と保存π区間[p_lower,p_upper]を用い、`3*U<p_lower`を成立、`3*L>=p_upper`を不成立、それ以外を未解像とする採択済み規則を適用した。π/3区間の表示は保存π区間を正確な有理数で3分の1にしただけである。新しいπ／atan／γ境界は計算せず、精度増加・区間細分・許容閾値変更・γ grid変更もしていない。

成立は、保存された条件付き連続十分shot設計において、shared有利側とγ<π/3の開領域が交差するという**限定gate**の結果である。実際のRPEの残り位相誤差予算、state contamination、synthesis／SPAM、全round、energy prior等を満たしたという意味ではない。不成立も、この保存cell・保守的route・十分設計についてのものであり、M1-S一般のNO-GOではない。

G9は`G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED`および`ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE`を保持する。形式的margin順位や相対比によって絶対改善を認定しない。全分割で厳密な整数shot削減、native回路費用、総資源改善、RPE成功、科学的GOは未確立であり、対応するraw出力はnull／falseのままである。新しいGPT詳細科学レビューや研究方針の正式採択は行っていない。

## 実行承認・固定来歴

- SOURCE S：`2f800f179c7b29589b2f9a7e532408b16b8fe6d2`、実行対象8ファイルは不変。
- 正式契約 C：`bf97fcb2607904f8b96a625c5b64440a78bbbfdc`、契約SHA256 `56923b8ce762e4aa592f91a548657c8c0d0a343a48c4606b49778e68b620a570`。
- 試行01専用authority-only子 A：`a1dd87a1f81467bea0ce27cacf9d21e49cff7908`。
- 入力commit：`b34572770e7c2105e16da003af050d789f490662`、入力SHA256 `bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`、146764 bytes。
- raw result SHA256：`693ce941bf7793a6104af28443746f43e05d4903118c0ebbb6f09729f389fadf`、37054 bytes。

今回の別途実施承認は新規に保存し、旧採択記録や過去V1のauthorization／消費済markerは変更・流用していない。markerは今回1件を消費、production起動1回、再利用・retry0。ユーザーの技術修正完了承認は必要時に失敗を保全して別SOURCE／契約binding／authorityで続行する文脈として記録したが、今回は不要だった。科学的negative／unresolvedを修正によってpositiveへ変える承認として扱わない。

## 検査と資源

事前にSOURCE／契約／保存入力／interpreterのhash、AST、clean treeと専用authorityの公開SHAを照合した。32論理CPUと空きmemoryを確認したが、採択どおりCPU1 worker・GPUなし・全thread環境1を維持。既存の厳密有理数比較、保存π区間一回のparse、固定48件の処理を使い、transcendental計算や別backendを追加していない。速度向上率は主張しない。

採択済みmemory2GiB、input1MiB、result/log各16MiB、最大48対照、fraction field1024、rational65536 bits等を変更していない。この契約はCPU／wall capなしであり、旧V1のCPU160秒／wall320秒は保全した。worker CPUは約0.024秒、worker RSS15052 KiB、supervisor wall約0.0605秒、外部wall0.13秒・最大RSS22528 KiB。実測値はこの実行の記録であり一般的runtime保証ではない。

実行後監査はproduction helperやFractionを使わず、保存された整数分子・分母の交差積により32件の判定を独立に照合する。検査範囲はこの限定比較の符号、schema、全grid、provenance、resource、SOURCE／input保全である。元H4信号／π／γ境界の独立再生成ではなく、入力coverageを認証しない。証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`を維持する。保存結果・独立整数照合・provenance等の監査は275件PASS、保全監査は12件PASS。旧非index7752ファイル、親checkoutの追跡1284ファイル・index・HEAD、既存211worktreeのHEADを保持した。監査JSONを正本とし、ローカル検査をCIや外部独立H4再現と表現しない。

## 証拠と次の判断

[正式契約](m1s_gamma_pi3_formally_adopted_contract_20261017.md)、[新実行承認](../../../artifacts/track_b_m1s_gamma_pi3_execution_authorization/2026-10-17/attempt_01/authorization.json)、[実行前確認](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/preflight.json)、[保存結果・独立整数監査](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/saved_result_audit.json)、[保全監査](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/preservation_audit.json)、[evidence index](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/evidence_index.json)、[publication receipt](../../../artifacts/track_b_m1s_gamma_pi3_execution/2026-10-17/attempt_01/publication_receipt.json)を参照する。

残るのは、この限定gateを踏まえ、実RPEの位相予算・整数shot・native費用をどの範囲で次に調べるかという研究判断である。今回はそれを代行せず、追加科学計算・source拡張・新規科学レビューを開始しない。旧SOURCE／結果／契約／markerと親・他worktreeを保全し、公開・監査後に**mandatory STOP**する。
