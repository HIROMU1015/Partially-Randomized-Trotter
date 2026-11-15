# Full-H κ2 D7 CALIBRATION取得完了

全5 cell・9観測path・18軸の最終検査PASS、FAIL/UNKNOWN/価格未取得0。結果のcommit/pushと公開後のD1証拠照合を完了した。全較正をCへ引き渡し、mandatory STOP。

## 固定条件と来歴

H4 linear 1 Å/STO-3G、保存tol-only DF10、one-bodyを含むFull-H pure RTE、L_D=0、K=2・κ2。group時間0.1、元physical ID・D7 cohort/seed・scalarを保持。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、P0、測定付きfull Hadamard wrapper、新CONTROLLED_ZZ_CX_CRZ_CX_V1。X/Yは各々compileした。状態準備費は含めない。

D7固定plan/cohort/seedはa43050c786c741d267ddf322232d888780efc03c、A設計checker7a0c9468b2499fac9fa1cd279ae2980c9dab048eのrank3/domain PASSを維持。DF10と設計rank3は別の量である。

| 来歴 | 固定SOURCE/結果 |
|---|---|
| 元caller / 取得0の実行前STOP | 6069605d50270111d1510478b08e11b8e5a18d86 / df0e11fd1e8d0f09166db71646d3df101ab13233 |
| 親event直接連結・実装証拠修正、初回16 compile | d323fee1ca8cd6893c4b705f990cdc931bdb5c87 |
| process並列化・checkpoint引継ぎ、残2 compile | 887c3b9d04d15be96637b6634265424e7dd50339 |
| 監視dump付近で異常終了した旧0 compile検証 | 0dc1eeaee3d9d6465f18edc33d4e17ed312bd6f4 |
| 監視修正・保存結果だけの最終検証 | 8cce540645b99e515d72888b817782a1b9f97334 |
| 先行4 cellの固定公開結果 | 7d0c39066d4ed900cfe7f5de6f4c7c5a3b2ec41f |
| 旧STOPの部分証拠 | 1fdb5dd42f65d6286848dc76e686f62c5b3accf8 |
| G36223完成結果 | f98cb96384b775bec2c2b9e1e63fe01a6ccdf2b1 |

元SOURCE・旧STOP・raw/QPYは保全した。[旧停止報告](../d_full_h_p2_completion_20261115/REPORT.md)、[限定修正](SOURCE_PREPARATION.md)、[監視修正・再現証拠](../d_full_h_p2_completion_20261115/RECOVERY.md)を参照。

## 件数と観測価格

予定18実compile、事前の受入可能な同等保存価格の再利用0、実取得18（d323で16・887で2）。最終検査PASS18/FAIL0/UNKNOWN0、価格・必要receiptの未取得0。今回の8c再開で再利用したG36223の4 compileは18件の内数で、新sampling/compile0。追加の実H4試運転0。

| cell | q/R | path0 X/Y RZ | path1 X/Y RZ | 最終状態 |
|---|---|---|---|---|
| G566 | 1/566 | 82785 / 82785 | 87375 / 87375 | 4 PASS |
| G4528 | 1/4528 | 695637 / 695637 | 691505 / 691505 | 4 PASS |
| G36223 | 1/36223 | 5552901 / 5552901 | 5572406 / 5572406 | 4 PASS |
| Q8R566 | 8/566 | 696593 / 696593 | 689039 / 689039 | 4 PASS |
| P2R566 | 2/566 | 169888 / 169888 | 該当なし | 2 PASS |

有限cohortの観測価格でありlaw E[C]ではない。全5 cellのX/Y別PRICE_CALIBRATION_RECORD_V1は10件。

## paired・correctness・重複照合

G566 path0/group0（seed11290319720649780064）→path1/group0（seed9014355981526777692）の取得親eventを直接復元・連結した。build前に親SOURCE/physical/cohort/index/seed・順序・phase・scalarを確認。pairedのsampler0、同seed再生成0、親価格copy0。子の測定付きX/Y wrapperは各々compileした。SAVED_PARENT_EVENTS_NO_DRAW / PAIRED_EVENTS_IDENTICAL。paired観測は親2本と相関し、独立な第3標本には数えない。

各cellのindex0は既存7数値gate（low/high control/leakage、independent occupation vector、Hadamard X/Y、compiled vector X/Y）PASS。index1は数値gateを追加せず、全pathのparity・新lowering・one-body・scalar/event phase・X/Y・fingerprint/元compile対応を検査した。元閾値とphase_alignment=falseを保持した。G36223 path0は元native7 gate/2 compileを再利用して残検査だけを実行し、path1の完了receiptをそのまま使用した。

§10.5はdf0の調査を引継ぎ、[DELTA_PREFLIGHT.json](DELTA_PREFLIGHT.json)と[再開時照合](../d_full_h_p2_completion_20261115/RECOVERY_PREFLIGHT.json)でSOURCE変更影響・関連job・保存価格を確認。異なるphysical/lowering/cohort/R/pathの旧結果は同等扱いせず、今回同scopeで取得済みの18件だけを再利用した。重複compileやseed差替えはない。

## 監視修正・資源実績

同じPython3.11.0rc1と人工3-qubit FastWriterで定期faulthandler dump有効時のSIGSEGVを再現、無効時は完了。旧0dc実workerのsignalは記録欠落のため原因を断定しない。worker内の非同期dumpを除去し、親processの観測と実終了code/signal保存へ変更した。既存人工検査を再利用・補強して13 PASS。これはD1本人のlocal検査であり独立review/CIではない。

修正後jobはwall21:39.15、peakRSS9,530,444 KiB、親/子exit0・signalなし。[元計時](../d_full_h_p2_completion_20261115/recovery01/G36223.time.txt)、[段階log](../d_full_h_p2_completion_20261115/recovery01/PROGRESS.jsonl)。初回取得の各cell/native trial資源は元receiptに保持する。process並列化・exact controlled-gate cache・canonical bytesを維持するfast fingerprintを使用し、共有設定は変更していない。全jobの速度倍率は未測定。

## 公開後照合とCへの引渡し

raw RESULTはlocal保全し、lossless gzip/必要なbyte partsで公開した。RAW_RECEIPTの元bytes/hash・圧縮hash・復号一致を確認。G36223 rawは2,336,436,443 bytes、gzip94,085,922 bytes、2 part。READ_RECEIPT.pyで固定公開SHAから復号し、元receiptからrecordsを再生成して公開C_INPUTS/PRICES/IMPLEMENTATION_EVIDENCE/sidecarと完全比較した。元compile JSONは各軸でbytes/hash/fingerprint/metrics/compilerへ照合した。

先行14軸の固定locatorは7d0のまま再利用し、G36223の4軸をf98の固定locatorへ結合。[PUBLICATION_CHECK](../d_full_h_p2_completion_20261115/recovery01/G36223/PUBLICATION_CHECK.json)、[HANDOFF_C.json](HANDOFF_C.json)に全18軸・10較正recordのlocatorとproducer SOURCE/検証SOURCE・親子依存を記録する。

D1がC c628e9e8のgeneric evidence readerで公開JSONを読んだことと、C担当の科学的受入は区別する。C科学的受入はNOT_PERFORMED。非configuration較正に架空round IDやcanonical-round専用guardを適用していない。

未確認：finite-cohort CI/confidence、sampler/ideal-law距離、rare寄与、境界加法性の一般化、長時間資格、モデル採用。fit/係数凍結はCの後続作業。D1はHOLDOUT6 compile、長時間外挿、E[C]・N・Gへ進まずmandatory STOP。

ロードマップへの影響：D7較正一式はCの後続fit/凍結に渡せる証拠として公開済み。Cの科学的受入・model凍結と別指示のHOLDOUTの依存順序を維持する。ロードマップ本体は統合担当が更新し、D1は編集していない。
