# P3 Stage 3：H4共通Gaussian basis preflightで停止

2026-11-02 JST。**`STOP_STAGE3_PREFLIGHT_OR_EXECUTION_FAILURE_NO_RETRY`。価格は未取得。**
共通T_eff用Gaussian basisのone-particle一致が固定数値基準でFAILとなったため、保持指数との照合、sampling、trajectory生成、価格compileへ進まず停止した。閾値変更・fallback・科学retryは行っていない。

## 固定条件・科学SOURCE・許可scope

Linear H4／隣接1.00 Å／STO-3G／中性singlet、system8/full256、保存DF tol-only1e-8/rank10。
案Aの共通T_eff/V_square、T=.8/S2/q8/r4/K4、M_B/M_C、P=0。
ユーザーの明示許可は公開設計の共通native preflightと、PASS時だけ各arm4 trajectory・36 wrapper compileを行う初期scopeに対応する。

- 科学SOURCE：`d3c258eb019e59234a874e1d8458e383f8317d68`。SOURCE公開・固定後に1回起動した。
- 契約：`be3de04e6fb501fecc4a07f38ceea7c24ba10472`の[価格設計][DESIGN]と[preflight政策][POLICY]。
- Stage 2結果：`8bcf5851f99f980dc08c5885ace54874b554c815`。共通T、保存辞書184/360、DF U/ηと来歴を引き継ぐ。
- 元H4入力：`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。新SCF/FCI/DF分解・state/reference生成なし。
- 固定ファイル22件のbytes/hash対応、SOURCE code 23件、runtimeは[source_and_inputs.json](source_and_inputs.json)。

## 停止点と数値結果

保存h_effを8×8で1回対角化し、既存`U_to_qiskit_ops_jw_givens`の`BogoliubovTransform(U.T)`経路で共通basisをlowerした。dense fallbackは使用していない。
actual gate列のforwardとinverseをそれぞれfull256次元で評価し、one-particle部分はvacuum phaseに対する作用をUへ照合した。
完成した保持block/controlled bodyの事後phase alignmentは行っていない。

| 検査 | spectral差 | PASS上端 | FAIL境界（これを超えるとFAIL） | 判定 |
|---|---:|---:|---:|---|
| one_particle | 2.3763416820048081e-13 | 7.1054273576010161e-15 | 1.1368683772161626e-13 | FAIL |
| actual_adjoint | 1.1443916996305601e-16 | 2.2737367544324514e-13 | 3.6379788070919222e-12 | PASS |
| inverse_composition | 2.4262497761429123e-16 | 2.2737367544324519e-13 | 3.637978807091923e-12 | PASS |
| unitarity | 2.4262497761429074e-16 | 2.2737367544324529e-13 | 3.6379788070919246e-12 | PASS |
| vacuum_leakage | 0 | 2.2737367544324498e-13 | 3.6379788070919197e-12 | PASS |

one-particle検査の比較次元d=8、scale≈1、`τ=16γ_8 scale`。保存されたFAILは差が`4τ`を超えたことによる。
他4検査はfull256次元の基準でPASS。これらからone-particle targetのFAILを解除したり、共通T指数の一致を推定したりしない。
差は約2.38e-13という小さい値だが、実行後にd=256の閾値へ置き換えたり、許容差を広げて継続したりしていない。
保持半指数参照、T-half/actual inverse/controlled-halfの3比較、DF残差10 basis、scalar作用検査は未到達。
失敗数値とtracebackは[execution_status.json](execution_status.json)と[STOP.json](STOP.json)の`failed_checks`に保持した。
`preflight.json`は返却前に停止したため空の途中記録であり、PASSや検査欠落の補完結果として書き換えていない。

## 実行件数・終了状況

```json
{
  "scientific_batches_started": 1,
  "T_eigendecompositions": 1,
  "DF_eigendecompositions": 0,
  "native_bases": 1,
  "reference_half_exponentials": 0,
  "sampling_calls": 0,
  "trajectories": 0,
  "event_positions": 0,
  "native_body_programs": 0,
  "wrapper_targets": 0,
  "compile_started": 0,
  "compile_completed": 0,
  "scientific_retries": 0,
  "new_signal_B_N_C_G": 0,
  "finite_bit_mean": 0,
  "SCF_FCI_DF": 0,
  "new_state_reference": 0
}
```
自分の明示的な主処理カウンタ：

```json
{
  "T_eigendecompositions": 1,
  "native_basis_lowerings": 1,
  "native_operator_evaluations": 2
}
```

wall `2.795977`秒、process exit1（preflight FAILによる停止）。生存科学workerなし、worker回収済み。
science batchは1回。8×8対角化1・native basis1・forward/inverse operator評価2だけを実施した。
sampling0、trajectory0、event位置0、price target0、compile開始/成功0。価格値が0という意味ではなく、全価格が未取得である。
共通half参照exponential0、新signal/B/N/G0、実効law平均0、新state/reference0、科学retry0。

## 人工検査・再利用

SOURCE固定前の最終人工検査は新規Stage 3の12件＋既存bit-copy回帰1件＝**13 PASS**。
小さい手指定系で複素Gaussian orientation、actual adjoint、負係数・order0/2/4、D2/DF central制御、同basis reuse、F/E/R/Empty、+Im、QPY gzip可逆性を検査した。
測定wrapperの人工compiler回帰4targetも含む。初回のDF full-unitary全体位相差と、測定observableでの検査への修正は[実装記録][IMPLEMENTATION]に保持している。
人工PASSは実H4 native loweringのPASSではない。今回H4の36compileへ進める根拠にはならなかった。

既存のGivens lowering、DF registry/event builder、Pauli event/D2、wrapper/compiler、native occupation/normを再利用する最小接続を実装した。
科学SOURCEを失敗後に変更していない。後続のDF residual・compile実装は人工検査のみで、今回の実H4科学結果ではない。

## 原因について分かること・未確定事項

失敗後は保存記録の読取とSOURCEの静的確認だけを行い、H4演算子の再評価をしていない。
qiskit-nature0.7.2のBogoliubov/Givens SOURCEには`np.isclose`による零判定が存在する。近零要素処理や角度丸めは調査候補だが、このH4差の原因として同定していない。
実入力のgauge/degeneracy、native lowering、数値誤差政策のどれを扱うべきか、また差がTの物理作用へどの程度伝わるかは今回未評価。
科学条件や数値政策を変えた再実行、追加精度、native familyの変更は別の明示許可を要する。

## 保全・公開・次の担当

固定SOURCE/入力とStage 1/P1/P2/Stage 2のraw/result/grant/ledger/marker/STOP、main/元研究branchは保持した。
Stage 2の`STOP_STAGE2_UNRESOLVED_COMPONENTS`とnominal平均/Bの既存結論は変更しない。
本結果はStage 3価格比較の未完了記録であり、Pauli/DFの価格・母平均・総N/G・実compiledρ・方式順位を支持しない。
静的な保全・hash・件数照合は[verification.json](verification.json)に記録する。作者自身の技術確認であり、独立した詳細科学レビューではない。

次の担当はユーザー：one-particle差の原因調査と、必要なら数値政策/実装変更・再実行をどの範囲で許可するか判断する。
Codexは新たな指示後だけ、その範囲の技術調査・実装・許可された計算を行う。
**結果公開・remote照合後mandatory STOP。科学retry、sampling、Stage 3価格取得、Stage 4へ自動移行しない。**

[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be3de04e6fb501fecc4a07f38ceea7c24ba10472/docs/research/h4_p3_stage3_native_cost_design_20261102.md
[POLICY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/be3de04e6fb501fecc4a07f38ceea7c24ba10472/docs/research/h4_p3_stage3_preflight_readiness_20261102.md
[IMPLEMENTATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3c258eb019e59234a874e1d8458e383f8317d68/docs/research/h4_p3_stage3_native_validation_20261102.md
