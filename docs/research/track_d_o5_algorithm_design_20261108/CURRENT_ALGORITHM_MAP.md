# O5の現在のアルゴリズム対応

2026-11-08。静的SOURCE読解と既存成果の統合だけを行った。新builder・optimizer・test・回路・価格計算は0。基点b1e7cc5b6270e6e62ab57b7c889f37484ce7d7b7の封印済みSOURCEを参照し、進行中q64の領域やprocessを操作していない。

## 目的と比較domain

規範資料は[研究目的](../研究目的・研究課題.md)、[方法](../研究方法・解析手順.md)、[先行研究](../先行研究と未解決点.md)、[概要](../研究概要・現状.md)。目的は固定DF下で

\[
G=\sum_{m,b}N_{m,b}\,\mathbb E_{\omega\sim Q_m}[C_{m,b}(\omega)]
\]

を同じ精度・成功率・full interrogation scopeで比較すること。今回新たにL_D、delta、r/K、law、shot数、誤差配分を採択しない。

| 条件 | Track Dの固定科学系列 | O2移管系列 |
|---|---|---|
| Model | Linear H4、隣接1.00 Å、STO-3G、Nalpha=Nbeta=2、元H、number-CDF8、元frame順L_D2 | 保存Linear H4、1.00 Å、STO-3G、legacy DF12、L_D3、N1 actual finite-grid Q |
| PR | ordered partial S2/s1/r32/K0、h=delta=.1、q1〜32の既存直接証拠 | m0/q1/delta=.02/r1/K2、coefficient_atol=0、selected full |
| 基準入力 | 元H厳密ground仮定。主checkerは全system入力列で測定演算子を検査 | 任意入力の代数と人工例。実保存H4の数値評価は未実行 |
| controlled対照 | strong symmetric-directional＋relative/head/tail/合法step融合 | diagonal_only selected circuit。Track Dのstrongと同一とはしない |
| compiler | P0: Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approx1、initially_zero=false、topology-free、宣言位相保持 | 0edc0fa5の人工はopt1。781e155dの新人工と未採択H4草案はopt2、line条件も別 |
| 幅 | strong9、Reference17 | n+1対2n+1、人工1〜3 system、8-system descriptor確認のみ |

双方ともtarget state準備を研究scopeとして除外する。この除外は、追加register/reset・CSWAP・校正を無償にする意味ではない。Track DのP0旧結果をreset/routing付き実機費用と読み替えず、将来その条件を変更する場合は両方式の新しい共通会計が必要。

## 回路構造の対応表

| 機構 | Strong symmetric-directional | Track D Reference | 既知SE-QPE / O2 |
|---|---|---|---|
| branch意味論 | W0†W1=Vというcoherence条件。単に全gateをcontrolled化する方式ではない | V=ABを一回のwhole-path gadgetで測定 | 同じ二lane干渉。名称の違いを新機構としない |
| Gaussian | uncontrolled basis、厳密同rotationのgroup化、異frameのrelative transition | 同じ既存emitterをcontrol1へ制限し物理laneに使用 | O2のselected basisは異policy。移植にはframe/phase/identity照合が必要 |
| head | StrongBuilder.transition/sequenceのanalytic gauge、directional_head/mixed_head | physical_segmentでもrelative/head融合を保持 | O2 prototypeにTrack Dの強いhead構造の代替実装はない |
| tail | full-frame same-basis run sharing、signed angle/support順を維持 | 同じtailを物理化。random eventを並べ替えない | actual Qやsupport basisの選択は各系列で別 |
| step境界 | q−1合法融合。directional半step＋uncontrolled半step→ordinary controlled full | A/B内部で計q−2、中央cutで一境界の融合を失う | O2はq1のsemantic block/application cutで、同じ境界数モデルではない |
| scalar | path scalar/identityを一回集約し相対phaseへ保持 | A/Bのglobal phaseと全path chiを保持、actual A.inverse | O2はscalar-free lane＋exact Fraction ledgerをPへ転送 |
| control置換 | directional中心と融合を維持する強い対照 | CSWAP、A†/Bを別registerへ、CSWAP、P(+chi) | Roweと操作的に同じ |
| 並列性 | 1 system lane | A†/Bの並列lane、CSWAPは固定追加 | fanoutも既知最適化。今回は新たに採択しない |
| readout | X:H、Y:Sdg→H、bit0:+1/bit1:−1 | 同じ | 同じ。postselection/再正規化は導入しない |

## 読んだSOURCEと再利用単位

次は静的確認した関数であり、この調査では呼び出していない。

| SOURCE / 関数 | 担当・次の再利用候補 |
|---|---|
| src/trottertracks/c1_h4_strong_baseline.py : StrongBuilder.transition / sequence / vacuum_phase | 相対Gaussian、basis cache、analytic gauge補正。同rotationをIDだけで近似合併しない |
| track_d_h4_reference.py : control_one / directional_head / reference_gadget | 制御枝の物理化、directional中心、二CSWAPとactual adjoint。未知ancilla作用は拒否 |
| track_d_c3_whole_path.py : head_recipe / mixed_head / physical_segment / prepare_steps | mixed control modeの合法境界、明示ordered stepsとcomplete-step cut。prepare_stepsは両方式を構築する |
| tests/track_d_h4_reference_checks.py : run_checks / check_compiled、tests/check_track_d_c5.py | AB=V、actual adjoint、chi、signed/phase、coherence/observable、全入力/leakage、P0 postcheck |
| track_d_c7_2_toy.py : weighted_depth / structure | raw/P0命令列、DAG/critical-path検査。compiled側特徴をcompile前oracleにしない |
| [track_d_c7_4_models.py : event_features（固定64072ba8）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/src/trottertracks/track_d_c7_4_models.py) | O(qr)のdescriptor-only run/support/sign/zero/lane特徴。ingest/fitを今回実行しない |
| track_d_bc_total.py / track_d_b6b_bc.py | 両方式を別々に会計する既存adapter。今回N/総額再計算は0 |
| [O2 vacuum_h4_adapter.py（781e155d）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/781e155d0907a82c962e41ab8e89f985735bc49b/src/trottertracks/algorithm_codesign/vacuum_h4_adapter.py) | state-free復元、scalar ledger、境界observerの設計を再利用可能。Track D loaderの置換ではない |
| [O2 vacuum_h4_numeric.py（781e155d）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/781e155d0907a82c962e41ab8e89f985735bc49b/src/trottertracks/algorithm_codesign/vacuum_h4_numeric.py) | 局所intervalと全入力POVMの丸め込み評価方式。既存Track Dの固定checker/閾値を変更する許可ではない |

基点にはC7-4とO2の一部pathが存在しないため、上記固定Git objectをgit showで確認した。別worktreeへコピー・mergeせず、固定commit linkで結ぶ。q64の後続進捗は照会せず、b1e7cc5bの固定PLANを境界として参照した。

## 同一機構と条件の違い

固定B6bの一般式は

\[
T_{\rm Ref}
=e^{i\chi}\overline{\langle v|V|v\rangle}\,V,\qquad V=AB .
\]

全入力で元coherenceを保存する条件はwhole-Vのreference-ray保存と正しいchi。実DF checkerはさらにA/B各halfの真空ray保存を要求し、最終referenceの真空復帰も検査する。O2の単回導出はこの強い十分条件を使う。half-ray条件を外したcutは、signalが同じになり得ても最終reference/reset/旧checkerの資格が別であり、今回緩和しない。

非可換A/Bで一回のwhole-path測定が成立することと、共有targetを使う複数coherent QPE blockの同値性は別。候補A/B/Cも一whole-path interrogationを保持し、短gadgetをq回連ねる変更は含めない。
