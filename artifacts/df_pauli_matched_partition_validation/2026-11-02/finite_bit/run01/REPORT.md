# P3：H4有限bit CDF・binary64角度の限定診断結果

2026-11-02 JST。**承認された6複素平均の評価を完了。2方式のbridgeと係数差closureはengineering整合、20軸の条件付きheadroom previewが正。**
CDF/角度差の点推定はRe/Imとも未解像。差を0と認定せず、係数摂動からの上側評価候補を取得した。
これはStage 2と同じencoded operatorモデル上の評価であり、literal native/compiled平均や実装精度の保証ではない。
[STOP](STOP.json)：`STOP_FINITE_BIT_ENGINEERING_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`。

## 固定条件・SOURCE・取得範囲

linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10。
案A：共通T_eff保持/V_square残差、M_B=Pauli184、M_C=DF rotated Z/ZZ360・10 basis。
T=exact binary64 .8、S2/q8/r4/K4、32 event位置、P=0、cosine=Re/+Im sine。
保存raw stateを再生成・再正規化せず、exact νで一度だけ除算。方式別B_savedをraw平均へ一度だけ乗算。

- 科学SOURCE `d22ca967955a664260a3a9857df243b3f151610f`。人工9 tests PASS後に公開・9変更ファイル/tree・18入力hashを照合して1 batchだけ実行。
- [実行条件](../../../../../docs/research/h4_p3_finite_bit_execution_20261102.md)、[設計](../../../../../docs/research/h4_p3_finite_bit_law_design_20261102.md) commit `c52bbf2a76708079075f01f57af76676a6d5e7ff`。
- Stage 2結果 `8bcf5851f99f980dc08c5885ace54874b554c815` / SOURCE `4bfbbaad1f8178077489bf06b085377256d6fb80`。
- Stage 3 run04結果 `e44e87a306d05c8549f1a044b7fd056c2e1055a3` / SOURCE `1cfce5e961e3d063741a1e0656052b10c7b27392`。
- run04はrun03完了結果 `90bc02d458299212d3a1aa7c4bd34a0b98d1e0b5`を経由し、run02 native SOURCE `9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`と保存law/basisを継承。
- Stage 4結果 `a1d3f32aba6416fd1e200cd66a97d1a825b5dd73`。旧N/G・全STOP・未解像finite Taylor biasは変更していない。
- 元H4 anchor `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。[SOURCE/18入力・state/ν・runtime](source_and_inputs.json)。

NumPy1.26.4/SciPy1.14.1/complex128。共通T_halfのexpm1＋同精度spectral照合1。DFは保存Uを再利用し、DF再対角化・Gaussian合成なし。
2 armをthread並列、実行processのBLAS32 thread/arm（affinity64 CPU）。共有設定や他worktreeへの変更なし。

## 比較平均と差の定義

choice入力binary64をexact実数として正規化した連続lawがbase。CDF有限bit化だけがcdf、さらに固定binary64角度を使うものがgrid。
全variantで共通のS_j/T_half/scalar/state/ν/Bを使い、次数0/2/4をmomentで積分する。標本・seed・trajectoryの再生成や巨大event列挙は行わない。
`F=Σ_k π_k (-1)^(k/2)(c_k I−i s_k W)W^k`、`W=Σ_j p_j S_j`。`Z=B_saved φ† e^(−i phase)(E F^r E)^q φ/ν`。
Δ_CDF=Z_cdf−Z_base、Δ_angle=Z_grid−Z_cdf、Δ_bridge=Z_base−Z_nom,saved、Δ_grid=Z_grid−Z_nom,saved。
bridgeにはλ/p/β/Bの数値規約とStage 2多項式経路との差を含み、CDFの因果寄与と同一視しない。

| 方式 | variant | Re Z | Im Z | u_num | u_bind |
|---|---|---:|---:|---:|---:|
| P | base | -0.1615111465025717 | 0.98687085023045484 | 6.810959e-10 | 0.000000e+00 |
| P | cdf | -0.16151114650257772 | 0.98687085023045829 | 6.810940e-10 | 0.000000e+00 |
| P | grid | -0.16151114650257775 | 0.98687085023045829 | 6.810902e-10 | 0.000000e+00 |
| DF | base | -0.16151114650257128 | 0.98687085023046939 | 7.154983e-10 | 1.946738e-10 |
| DF | cdf | -0.16151114650256285 | 0.98687085023047072 | 7.154961e-10 | 1.946738e-10 |
| DF | grid | -0.16151114650256285 | 0.98687085023047072 | 7.154972e-10 | 1.946738e-10 |

[signals.csv](signals.csv)、全未丸め値とCDF count/角度区間は[results.json](results.json)の`arms`。
数字上同じ平均・ゼロの差も分解能以下であり、数学的な0の確定ではない。

## 直接差の解像と係数摂動からの上側評価

| 方式 | 差 | Re差 | Im差 | 診断用不確かさU | Re/Im判定 |
|---|---|---:|---:|---:|---|
| P | CDF | -6.022960e-15 | 3.441691e-15 | 1.362190e-09 | UNRESOLVED/UNRESOLVED |
| P | angle | -2.775558e-17 | 0.000000e+00 | 1.362184e-09 | UNRESOLVED/UNRESOLVED |
| P | bridge | 3.330669e-16 | -7.660539e-15 | 1.431434e-09 | UNRESOLVED/UNRESOLVED |
| P | grid | -5.717649e-15 | -4.218847e-15 | 1.431428e-09 | UNRESOLVED/UNRESOLVED |
| DF | CDF | 8.437695e-15 | 1.332268e-15 | 1.820342e-09 | UNRESOLVED/UNRESOLVED |
| DF | angle | 0.000000e+00 | 0.000000e+00 | 1.820341e-09 | UNRESOLVED/UNRESOLVED |
| DF | bridge | -1.387779e-15 | 9.992007e-15 | 1.780697e-09 | UNRESOLVED/UNRESOLVED |
| DF | grid | 7.049916e-15 | 1.132427e-14 | 1.780695e-09 | UNRESOLVED/UNRESOLVED |

上記Uは差の表示・整合判定用。headroomへ再加算しない。bridgeは両方式ともCONSISTENT_ENGINEERING、CDF/angle closureは4件ともPASS_ENGINEERING。

| 方式 | CDFのみ上側候補 | 角度のみ上側候補 | law＋角度上側候補（base基準） | 保存nominal基準のgap上側候補（bridge込み） |
|---|---:|---:|---:|---:|
| P | 9.243711149e-14 | 1.024085708e-15 | 9.346119720e-14 | 1.431534804e-09 |
| DF | 1.795110530e-13 | 1.800534082e-16 | 1.796911064e-13 | 1.780886338e-09 |

law＋角度候補は完成信号の差し引きだけから求めず、exact確率差・rotation係数区間・encoded成分normから短step差を評価し、r/qの積の差へ転送した。
保存nominal基準の候補にはbridge差とその数値不確かさを含む。上の2つの基準を混同しない。
**全signal上側候補はengineering評価。** scalar/CDFのexact演算が、spectral norm・expm・全signalの包含を自動保証するわけではない。
`certified_gap_upper=null`、`implemented_accuracy_guaranteed=false`を維持する。

## CDF実効質量・角度binding

NumPy1.26.4のchoice入力正規化、sequential cumsum、末尾CDF除算、side=rightを再現。
CDF端点をexact Fractionへ変換し、M=2^53について`n_i=ceil(M F_i)−ceil(M F_(i−1))`、`p_hat_i=n_i/M`でcountを得た。
4law表すべてcount非負・総count=M・NumPy CDF一致を確認。IID uniform-wordモデルは明示仮定で、PCG64独立性や有限seed集合の平均の証明ではない。

| 方式 | 成分TV(base,grid) | 次数TV(base,grid) | 成分のゼロgrid質量 | 次数のゼロgrid質量 |
|---|---:|---:|---:|---:|
| P | 2.755116114e-15 | 5.930220971e-17 | 0 | 0 |
| DF | 5.667216411e-15 | 4.914594054e-17 | 35 | 0 |

DFの35成分はこの53-bit CDFでcount0になる。成分を手動削除したのではなく、保存順序と有限gridの帰結として記録し、その差を上側評価へ含めている。
全3次数の実効質量は正で、order4も平均評価に含む。Stage 3でorder4未観測だった価格tailの問題は残る。

角度はSOURCE通り`float(atan(a/(k+1)))`の6値を取得。倍角・符号反転のoperand対応も確認した。
baseのalgebraic cos/sinは既存sqrt96有理区間、fixed thetaのcos/sinは既存2×2 exponential degree24で包含。
これはscalar係数だけの区間評価で、full信号やfinite Taylor biasの高精度再計算ではない。
`results.json/arms/*/angles`にbinary64/hex/exact theta、割算operand、cos/sin区間と係数差を保存する。

## Bias/headroom preview：共有数値誤差の一回計上

grid点と保存physical reference区間[L,U]から直接、`b_grid_upper=max(|z_grid−L|,|z_grid−U|)+u_grid_num+u_grid_bind+reference_transfer`とした。
区間幅は端点距離に含まれ、display.axis_resolution・旧nominal/split誤差・Δ診断用Uを再加算していない。
新u_bindは保存Uからのcomponent再構成丸めの伝播であり、pointに既に反映した辞書物理差の再加算ではない。

| 方式 | cosine bias上側候補 | sine bias上側候補 | ε_sig=.0005のh_cos | 同h_sin |
|---|---:|---:|---:|---:|
| P | 9.071309411e-05 | 1.481318621e-05 | 2.628402965e-04 | 3.387402044e-04 |
| DF | 9.071332320e-05 | 1.481341530e-05 | 2.628400674e-04 | 3.387399753e-04 |

保存ε_sig=.05/.01/.005/.001/.0005、ε_axis=Stage 4保存operandを使用した20軸すべてPOSITIVE_CONDITIONAL。
[headroom.csv](headroom.csv)に全20軸を保存。**native bridge=0、compiled ρ=0はpreviewの仮定**であり、実測済みの事実ではない。
`actual_implemented_headroom=null`。新N/G、価格SE、資源順位は計算・更新していない。B engineering振幅候補の保証も今回の対象外。

## Native/compiled接続で残ること

`Z_compiled−Z_nom=(Z_grid−Z_nom)+(Z_native−Z_grid)+(Z_compiled−Z_native)`。
本診断は最初のencoded law/angle項を評価した。共通Gaussian保持、DF native basisとStage 2 occupation operator、actual adjoint/basis cancellation、
πのbinary64集約やcontrolled/scalar phaseの一様接続差は別に残る。DF丸め成分をexact involutionと仮定していない。
保存Stage 3の局所native PASSは引用できるが、全trajectory一様boundや実compiled ρへ格上げしない。
`native_bridge`の一様上端はnull。有限bitだけの結果で実装後全gap=0仮定を全面解除したとは主張しない。
価格tail・全law compiled保証・実機精度・RPE全round総費用・一般的Pauli優位性も未取得のまま。

## 検査・保全・終了

人工9 tests PASS：小bit CDF全列挙/tie/ゼロ、角度係数区間、非可換signed event全列挙対moment、非直交DF U、
nonunitary差の転送、+Im/ν/B/scalar、bias一回計上/null、6信号toyのRNG/circuit遮断、run03経由native継承schema。
これらはH4性能の証拠と区別する。最終SOURCE固定前に実数fixtureへの複素dtypeとscalar誤差上向き丸めを修正した。
既存CDF/誤差伝播/occupation/有理scalar prototype/端点biasを再利用し、共有環境や旧科学SOURCEを変更していない。
科学batch1、6複素信号・4law表・6角度・20 headroom、exit0、wall 6.488978秒。

```json
{
  "scientific_batches": 1,
  "dictionaries": 2,
  "law_tables": 4,
  "event_angles": 6,
  "complex_signals": 6,
  "headroom_axes": 20,
  "new_sampling": 0,
  "RNG_replay": 0,
  "native_circuit_generation": 0,
  "new_compile": 0,
  "Gaussian_synthesis": 0,
  "DF_eigendecompositions": 0,
  "physical_reference_generation": 0,
  "finite_Taylor_bias_refinement": 0,
  "new_N_G": 0,
  "automatic_retries": 0
}
```

[verification.json](verification.json)にSOURCE16/入力18保全と実評価件数、[publication_verification.json](publication_verification.json)に転記・旧証拠不変・公開対象hashを記録。
旧Stage 1/2/3/4全結果とSTOPを保全。有限Taylor biasのUNRESOLVEDも変更していない。
次の担当はユーザー：本結果の利用範囲を確認し、native接続差とcompiled ρの限定検証scopeを判断する。
Codexは新たな明示指示まで追加科学計算・N/G更新・sampling/compile・精度昇格を開始しない。
**今回の限定診断は完了。公開・remote照合後mandatory STOP。**
