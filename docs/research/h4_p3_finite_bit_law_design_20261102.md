# P3 Stage 4後：H4有限bit実効law・角度誤差の非実行設計

2026-11-02 JST。基点はStage 4結果`a1d3f32aba6416fd1e200cd66a97d1a825b5dd73`。
ユーザーがStage 4を完了と判断し、次の優先順位を有限bit law/角度、compiled誤差、価格tailとしたことを受けた技術設計である。
**今回行うのは保存資料・schema・SOURCEの静的読取と文書作成だけ。科学的数値検証の実行許可は未取得。**
`STOP_STAGE4_CONDITIONAL_ACCOUNTING_COMPLETE_NO_AUTOMATIC_FOLLOWUP`、Stage 2の未解像finite bias、Stage 3の全STOPを維持する。
本書は後続限定実行の提案であり、新しい実行済み結果、SOURCE固定、重要claimの採択を意味しない。

## 1. 目的と固定条件

固定H4・matched分割の条件付き資源比較を保全し、Stage 4で仮定したfinite-bit law/angle gap=0を検証可能な差と上側評価へ置き換える経路を用意する。
finite Taylor biasそのものの高精度化、方式探索、資源順位の再計算は今回の目的に含めない。

| 項目 | 後続限定実行の提案範囲 |
|---|---|
| 系 | linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10 |
| 分割 | 案A、共通T_eff保持/V_square残差、共通残差identity分離 |
| 辞書 | M_B=Pauli184、M_C=DF rotated Z/ZZ360・10 basis。保存順序・符号を維持 |
| tuple | T=exact binary64 .8、S2/q8/r4/K4、32 event位置、次数0/2/4 |
| 信号 | 保存raw state、exact νで一度だけ除算、cosine=Re、sine=+Im、P=0 |
| law | NumPy1.26.4 choiceの53-bit uniform-wordモデル。各drawのIIDは明示仮定 |
| normalization | Stage 2の方式別B_savedを実数として固定し、raw平均に一度だけ掛ける |
| 初期精度案 | matrix/vector complex128。演算別engineering uncertaintyと係数差からの上側評価候補を併記 |
| 新規科学処理 | 後述の2方式×3平均=6複素信号と、同じ2lawに対するCDF/角度/誤差転送のみ |
| 初期範囲外 | sampling/RNG replay、native回路生成・compile、Gaussian合成・再対角化、物理reference生成、B再生成、N/G再計算 |

同じseedやtrajectory番号による方式間pairingは行わない。今回はtrajectoryを用いない決定論的なlaw平均評価を提案する。
有限bit lawの全supportをmomentで積分することと、compiled回路全law保証や価格tail保証は別である。

## 2. 参照する固定証拠と入力

| 固定ref | 役割と読む保存物 |
|---|---|
| Stage 2結果 `8bcf5851f99f980dc08c5885ace54874b554c815` / SOURCE `4bfbbaad1f8178077489bf06b085377256d6fb80` | `stage2/run01/dictionaries.json`、`results.json`、`common_operators.npz`、`source_and_inputs.json`、`STOP.json` |
| Stage 3 run02結果 `2da49fa4e8578227c130fc6061a0b3061276f893` / stable Gaussian SOURCE `9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c` | `stage3/run02/law_bindings.json`、`native_bases.json`、`preflight.json`、`source_and_inputs.json`、`STOP.json`。native接続境界の来歴 |
| Stage 3 run04結果 `e44e87a306d05c8549f1a044b7fd056c2e1055a3` / SOURCE `1cfce5e961e3d063741a1e0656052b10c7b27392` | `stage3/run04/source_and_inputs.json`、`reference_half.json`、`STOP.json`。run02のlaw/native再利用とruntimeの対応 |
| Stage 4結果 `a1d3f32aba6416fd1e200cd66a97d1a825b5dd73` / SOURCE `2f0d31f75ab1ecacc73d9ee5c34ddfc997396102` | `stage4/run01/results.json`、`axes.csv`、`source_and_inputs.json`、`STOP.json`。既存bias/精度operand/headroomへの接続定義 |
| 元H4 anchor `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | Stage 2入力来歴が指すraw state/exact ν。新stateを生成・正規化しない |

上の相対pathのrootは`artifacts/df_pauli_matched_partition_validation/2026-11-02/`。
Stage 2 `common_operators.npz`にはT/V0が保存されるが、T_half行列は保存されていない。
したがって後続実行では同じ保存Tから共通半step exponentialを1個だけ再評価し、同精度spectral照合を1個行う案とする。
これは既存保持演算子の数値接続であり、新しいfull-H物理reference生成ではない。
DF basisは`dictionaries.json`のU_real/U_imagを再利用し、対角化やGivens loweringを再実行しない。

run02の`law_bindings.json`は成分probability・basis hash・Stage 2 normalizationを持つが、最終CDFや角度表を保存していない。
run04は同じlawを接続し、runtimeのNumPy1.26.4を記録している。
将来CDF/角度をSOURCEの丸めグラフから再評価する処理自体が今回未承認の科学処理である。
公開SOURCEと対象入力の必要なhash/bytesを実装時に固定する。新しい承認証明、grant、承認文保存の仕組みは作らない。

## 3. 確認できたsampling経路とCDFの定義

`matched_partition_stage3.component_inputs`は各保存係数c_jと保存λ_Rからbinary64 `p_j=abs(c_j)/lambda_R`を作る。
`rte.iter_sample_rte_events`はさらに`math.fsum(p)`で割ったbinary64 vectorをcomponent choiceへ渡す。
order choiceには保存`order_probabilities`を直接渡す。両者のCDF正規化を省略してはいけない。

NumPy1.26.4の[choice SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx#L840)は、replacementあり・p指定で、
sequential binary64 cumsum、末尾CDFによる除算、uniform draw、`searchsorted(..., side='right')`を使う。
[PCG64接続](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx#L41)と
[uint64変換](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd#L67)は上位53bitを2^53で割る。
このSOURCE確認は決定論的PRNGの独立性証明ではない。

choiceへ渡すbinary64 vectorをv、最終binary64 CDFをF_i、M=2^53とする。
uniform-wordモデルU=m/M、m=0,…,M−1の下で、成分iの区間は`[F_(i−1),F_i)`。

\[
 n_i=\lceil MF_i\rceil-\lceil MF_{i-1}\rceil,
 \qquad \widehat p_i=n_i/M,\quad F_{-1}=0.
\]

CDF端点はbinary64のexact Fractionへ変換してceilし、通常のfloatによる`endpoint*M`の丸めに依存しない。
重複CDF端点、ゼロ質量、F_last=1、非負整数count、総count=Mを確認する。
同じ処理を3次数へ適用して実効order質量pi_hatを得る。supportを再ソート・切り捨て・再DF化しない。

比較用の連続lawは、choice入力vectorの各binary64をexact実数として正規化したp_bar、pi_barと定義する。
これとStage 2の理想係数比の差は§5のbridgeへ残す。CDF正規化後のfloat差分を実効確率と呼ばない。
保存eventの`event_probability`は理想式metadataであり、有限bit実効確率を表さないため再利用しない。

`h4_ball_evidence.cdf_binary64/grid_masses`がこのCDF/整数区間処理の既存実装である。
旧`verify_probability_mapping`は旧export schemaを要求するため、旧controller/EnclosureEngineを呼ばず、必要な純関数だけ接続する。
NumPy実装との照合は人工vectorで行い、実分子samplingや保存seedのreplayへ広げない。

## 4. 角度の固定とmoment評価

`rte._make_event`の角度は、保存a=`dimensionless_step_time`について、binary64除算後に
`theta_hat_k=float(math.atan(a/(k+1)))`とする。k=0/2/4の6角度を方式別に評価する案である。
除算・atan・その返却binary64を分け、値/hex/exact Fractionを保存する。
ideal algebraic rotationは同じ保存aをexact実数とし、x_k=a/(k+1)、c_k=1/sqrt(1+x_k^2)、s_k=x_k*c_kで定義する。
比較は角度差だけでなく、cos/sin係数の差を直接使う。

nativeの角度接続はSOURCE上、Pauli D2がcentral RZへ`2*sign*theta_hat`、DFがRZ/RZZへ同じ倍角を渡す。
normalな有限binary64角度の2倍・符号反転はexactなスケーリングであることをoperand範囲と共に確認する。
retained Gaussian角度・basis lowering・制御gate分解の精度をこの主評価で測定したことにはしない。

Stage 2辞書と同じ符号付き成分S_j、平均W(p)=Σ_j p_j S_jを使う。
productは選択順に作用し、rotationが最後、order phaseは(−1)^(k/2)。平均は

\[
 F(p,\pi,c,s)=\sum_{k=0,2,4}\pi_k(-1)^{k/2}
       [c_k I-i s_k W(p)]W(p)^k.
\]

任意に大きいevent集合を列挙せず、W/W²/W⁴と3項のrotation平均から求める。
独立drawの積lawを仮定した代数であり、同一seedの32個の決定論的出力の平均ではない。
order4は標本未観測でもこの式で含む。これは未観測order4のcompiled価格を取得することではない。

DFの保存Uには丸めがあり、Stage 2のoccupation表から構成したS_jのHermiticity/involution残差を無視しない。
このFはStage 2に合わせたencoded operatorモデルのlaw/angle平均である。
`cos(theta)I−i sin(theta)S_j`を、丸めたS_jについて厳密な`exp(−i theta S_j)`と同一視しない。
保存Gaussian gate列から得る正確なnative作用との接続は§7の別項である。

## 5. 後続実行で取得する6複素信号と差の分解

共通E=exp(−i δT_eff/2)、保存scalar angleからの共通phase、保存raw φ、exact ν、方式別B_savedを用い、

\[
 Z_X(F)=\frac{B_{\rm saved}^X}{\nu}
  \phi^\dagger e^{-i\widehat\varphi}(E F^r E)^q\phi
\]

を評価する。Bはraw平均へ一度だけ、scalar phaseも全体へ一度だけ。補正済みStage 2 nominalへ再乗算しない。
同じS/E/phase/Bを3variantで共有し、保持回路誤差をCDF差へ混入させない。

| variant / 各方式1複素信号 | 成分/order law | rotation係数 | 用途 |
|---|---|---|---|
| base | p_bar/pi_bar | c_k/s_k algebraic | Stage 2保存nominalとの数値・係数丸めbridge |
| cdf | p_hat/pi_hat | baseと同じc_k/s_k | CDFだけの影響 |
| grid | p_hat/pi_hat | cos(theta_hat)/sin(theta_hat) | CDF＋binary64角度の実効lawモデル |

\[
 \Delta_{\rm CDF}=Z_{\rm cdf}-Z_{\rm base},\quad
 \Delta_{\rm angle}=Z_{\rm grid}-Z_{\rm cdf},\quad
 \Delta_{\rm bridge}=Z_{\rm base}-Z_{\rm nom,saved},
\]
\[
 \Delta_{\rm grid}=Z_{\rm grid}-Z_{\rm nom,saved}
 =\Delta_{\rm CDF}+\Delta_{\rm angle}+\Delta_{\rm bridge}.
\]

bridgeはp/λ/β/Bの数値規約、Stage 2の多項式評価との差を含み、CDF量子化の因果寄与と呼ばない。
Stage 2のraw信号とB closureを引用するが、比較baseの違いを隠して一致を強制しない。
各差のRe/Im、絶対値、数値不確かさ、解像状態を保存し、P−DFの差も科学標本のpaired推定とは扱わない。
nominal/full reference/outer/finite信号を再生成せず、Stage 2の未解像finite Taylor biasは変更しない。

## 6. 差が未解像の場合の上側評価

complex128差が小さいだけでgap=0とは判定しない。
信号ごとのu_numはStage 2のproduct_error/signal_propagationに沿い、実norm、非unitary増幅、matvec/内積、ν、scalar/B丸めを含める。
DF成分の再構成丸めは別のu_bindとして伝播する。保存Uを再直交化せず、norm=1を無条件で代入しない。
差の表示用不確かさには両側の必要項を含め、共有誤差の相殺を証明なしに仮定しない。

さらに、ほぼ等しい完成信号を引くだけでなく、CDF質量差・rotation係数差から短stepの差の上側評価候補を作る。
例えばdW≤Σ|p_hat−p_bar| ||S_j||、

\[
 \|W_1^k-W_0^k\|\le dW\sum_{l=0}^{k-1}
      \|W_1\|^l\|W_0\|^{k-1-l}
\]

と、rotation差`|dc|+|ds|*||W||`、order質量差の重み付きnormからF差を評価し、r/q反復へ積の差の公式で転送する。
law/角度の上側評価では共通演算子の誤差を重複して加えない。
理想unitary eventならTVによる簡略boundも可能だが、丸めDF operatorへunitary仮定を流用しない。

初期案はengineering診断。exact整数CDF countだけで、matrix norm・libm trig・全signalのrigorous保証へ格上げしない。
下側も0を超える差はRESOLVED_ENGINEERING、0を含む差はUNRESOLVEDとし、後者でも有限のgap_upper_candidateを報告できる。
必要なstrict上側評価が得られなければ`certified_gap_upper=null`を保持する。
任意の固定1e−10閾値やgap=0への丸めを導入しない。未解像を理由に自動高精度化しない。

## 7. Native・compiledとの接続境界

本主評価は「同じStage 2演算子に53-bit lawとbinary64 event角度を適用した平均」である。
Stage 3のliteral native Fullやcompiled Fullの実装平均とは区別する。
後続の誤差は

\[
 Z_{\rm compiled}-Z_{\rm nom}
 =(Z_{\rm grid}-Z_{\rm nom})
 +(Z_{\rm native}-Z_{\rm grid})
 +(Z_{\rm compiled}-Z_{\rm native})
\]

という境界で扱う。最後の項が今後のcompiled残差の対象である。
中間のnative bridgeには、共通Gaussian保持、DF保存基底のgate作用、actual adjoint、basis cancellation、
符号やorder phaseのbinary64 π加算・集約、scalar/controlled phaseの対応を含める。
Pauli D2とDF builderはphaseを集約するため、単純な数学的(−1)だけでliteral phaseの丸めまで検証したとしない。
共有保持の差も、異なる残差との積では同じ信号誤差になるとは限らない。

既存Stage 3のpreflightは接続の局所的な数値証拠として使えるが、64本のPASSを全trajectoryの一様boundへ置き換えない。
初期実行ではnative bridgeの既知/未取得を表にし、一様上側評価がない部分はnullを維持する。
新QPY生成・Gaussian合成・compile・native全law平均の追加評価を本6信号へ紛れ込ませない。
従って今回の限定実効law評価だけで「実装後の全gap仮定を解除した」「compiled ρを検証した」とは主張しない。

## 8. 既存bias/headroomへの接続：N/Gは更新しない

Stage 4のbias候補とgap診断を単純加算すると、保存nominalのu_num/u_bindを二度計上し得る。
後続実行では、まずgrid平均点と保存physical reference区間[L,U]から直接、

\[
 b^{\rm grid}_{X,a,upper}=\max(|z^{\rm grid}_{X,a}-L_a|,
 |z^{\rm grid}_{X,a}-U_a|)+u^{\rm grid}_{a,num}
 +u^{\rm grid}_{a,bind}+u_{a,ref\ transfer}
\]

という候補を作る。区間幅は端点距離に含まれ、display.axis_resolutionを再加算しない。
新u_bindは再構成/接続の不確かさに限定し、実効信号点に既に反映した辞書の物理差を再度加算しない。
旧nominal/splitの数値不確かさやouter＋finiteの和はこの直接経路へ足さない。
Δ_gridの診断用不確かさも、別の誤差として再加算しない。

Stage 4保存の5精度・両軸のepsilon operandを読み、2方式×5精度×2軸について、
`h_grid=epsilon_axis−b_grid_upper−u_native_remaining−u_compiled_remaining`の定義を接続する。
初期実行で数値previewを出す場合は、後二者を0とする**追加仮定の下のheadroom**としてのみ示し、未測定の実装headroomと区別する。
missing/nullや非正headroomを0で補完しない。
B_savedは補正乗数、Stage 4 B_upperはshot振幅候補という異なる役割を保つ。
新N、G、価格SE、資源順位は計算せず、Stage 4結果ファイルを上書きしない。
finite-bitのみの評価結果を使うか、native bridgeやcompiled評価後に会計を更新するかは別の作業判断とする。

## 9. 最小実装・人工検査・将来の終了条件

| 再利用する資産 | 接続する機能 / 持ち込まない機能 |
|---|---|
| `h4_ball_evidence` | cdf_binary64/grid_masses、有限平均のmoment代数。旧C export/controller/grant/旧PF・PR条件は流用しない |
| `matched_partition_stage2` | native occupation、演算誤差伝播、共通T exponentialとsignal定義。build/evaluate全体やDF再対角化は呼ばない |
| `matched_partition_stage3.DictionaryOracle` | 保存DF U/support・Pauli label・符号の構成規約。回路生成/trajectory評価は呼ばない |
| `rte.iter_sample_rte_events/_make_event` | SOURCE上の確率正規化・atan規則だけを接続。関数を呼んでsamplingしない |
| `matched_partition_stage4` | biasの端点評価、精度operandと一回計上の規約。account_row/evaluate/fixed_signal_shotsは呼ばない |
| 既存人工tests | `test_h4_ball_evidence`のCDF/tie/moment、Stage 2の演算誤差、Stage 4の区間biasの例を参考に最小限拡張 |

承認後に小さいadapter/runnerを追加する案とし、不要な一般frameworkは作らない。
人工小行列でCDF tie/ゼロ質量/小bit全列挙、非可換X/Zのorder0/2/4列挙対moment、sign/order/+Im、
CDFのみ/角度のみの分解、nonunitary norm増幅、bias不確かさ一回計上/nullを確認する。
人工PASSはH4性能の証拠ではない。今回はこれらの新testsも実行していない。

科学入力を読む前にSOURCEと科学入力を固定し、新namespace例
`artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/`をfreshに作る。
最小成果物はSOURCE/input来歴、方式別CDF count/角度表、6信号と差・不確かさ、headroom接続表、REPORT/STOP。
1固定tupleのbatchを一度評価し、部分結果を残す。自動retry・別solver/precision/tuple・新samplingへ切り替えない。
実装上の独立taskは空き資源に合わせ並列化・cacheし、CPU/RAMへ恣意的な小上限を設けない。
共有環境・他利用者・他worktreeを変更しない。

入力/来歴不一致、CDF非単調/質量不整合、angle非有限、scalar/νの所有不明、誤差伝播不能なら後続評価を停止し、原因を保存する。
bridgeの整合が既存誤差評価から説明できなければgapの「小さい」判定を行わずMODEL_UNRESOLVEDとする。
差の解像とheadroom内の上側評価を別のstatusにし、差が未解像でも評価可能な上端を報告する。
終了後は成功/未解像/失敗いずれもmandatory STOP。compiled検証、tail、N/G更新へ自動移行しない。

## 10. 今回の成果と次の担当

今回確認したのは保存schema、SOURCEのsampling/CDF/角度経路、既存評価資産、native接続の未取得部分である。
新CDF数値・実効確率・角度・信号・bias/headroom・N/Gの科学計算は0。
新sampling/RNG replay・回路生成/compile・Gaussian合成・reference生成・finite bias高精度化も0。
旧科学SOURCE/入力/raw/結果/STOPを変更していない。

次の担当は**ユーザー**：この固定条件・6複素信号・engineering精度政策・native接続境界を確認し、具体的数値検証を実行するか明示判断する。
許可された場合の担当は**Codex**：適合する既存関数の最小接続、人工検査、科学SOURCE固定、限定1batchの実行・報告までをまとめて行う。
現時点では科学実行を開始しない。

[Stage 4報告](../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md) /
[Stage 4 STOP](../../artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/STOP.json) /
[Stage 2名目契約](h4_p3_stage2_nominal_signal_contract_20261102.md) /
[Stage 3 native条件](h4_p3_stage3_native_validation_20261102.md)。
