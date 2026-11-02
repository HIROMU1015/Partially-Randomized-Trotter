# 研究C：PR内部改善のshots・native価格・PF/RTE寄与分析

2026-11-02 JST。**既存公開結果の読解と資源会計の記述的整理。新規科学計算・数値再解析・重要GPT科学レビューは実施していない。公開照合後mandatory STOP。**

証拠の固定点は結果commit [`aced1647b81e8d28091058cdc7b6e3e1ee54896a`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/aced1647b81e8d28091058cdc7b6e3e1ee54896a)。以下の出典リンクもこのcommitに固定する。「旧PR」はStage A γ=.1最終J/Uを使うM4 s1/r128＋Native128、「新PR」は統合実験の選定点`update005`であり、通常CDF γ0との比較ではない。

## 1. 保存結果から説明できること

同じ直鎖H4・1.00 Å・STO-3G、number CDF8・元順prefix `L_D=2`、partial ordered S2、δ=h=.1/s1/r128/K0、13 roundの範囲で、次を区別できる。[統合報告][I]／[旧M4][M]／[旧Native128][O]

1. **DF物理予算を維持し、exact tail λが約60.60%低下した。** 旧1.44242988558→新0.568349854686 Ha。新候補はJ/U同時一次収束を達成しておらず、最適解や一般的優位を示さない。
2. **全13 roundでattenuationと補正RTE誤差は改善し、PF誤差は増えた。** λからattenuationへの関係は固定K0/h/r/mの式で説明できる。PF・shots・総資源の変化全体をλだけの因果効果とは認定しない。
3. **全6 native指標で新cohortのkernel平均価格が増えた一方、shotsとWが減り、ideal条件付き総RZは約11.19%低下した。** RZ単価は約40.84%増、shots比は約.874938、W比は約.630559。異なる32 trajectory cohortの記述的な比較である。
4. **保存強決定論benchmarkに対する主RZ優位は成立していない。** 新PRの207,919,150,188 RZは保存強Dの101,825,929,455 RZの約2.04191倍。ただし強Dは旧J/Uであり、新J/Uによる公平比較の最終順位は未決定。

**会計式の補足：提示された総RZはshell込みである。** SOURCE上の正確な式は`C_total = W μ_kernel + S_shell`。依頼の`W μ_kernel`はkernel部分に対応する。小さいshell項も落とさず、§6で保存総RZとの整合を示す。[旧affine会計][AC]／[serial SOURCE][CS]

## 2. 比較条件・SOURCE・候補identity

| 項目 | 旧PR | 新PR |
|---|---|---|
| 候補 | Stage A γ=.1終了J/U、M4 Proposed | 統合実験で選定した`update005` |
| 候補SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| PF/RTE科学SOURCE | `7dbabc7866cc62a5460bf1579ce84c59493d8e1e`（M4） | `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`（統合） |
| Native科学SOURCE | `dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad` | 同統合SOURCE |
| exact tail λ [Ha] | 1.4424298855756243 | 0.5683498546862645 |
| DF sector norm＋engineering guard [Ha] | 1.229561498918692e-8 | 8.46076622093e-9（既報） |
| DF予算 | CA/8=.00019875 Ha、PASS | 同予算、PASS |
| Native cohort/master | `M4_r128_native_fixed32_v1` / 2026102805 | `integrated_best_fixed32_v1` / 2026110201 |

旧identity・SOURCEは[旧M4][M]／[旧Native128][O]、旧DF精度は[証拠統合§2.3・3.1][E]、新identity・精度・λは[親result][P]の`selected.{id,physical}`・`references`および[統合報告§2][I]に対応する。新nativeの[保存event metadata][NE]の`JU`も新候補SHAを指す。SHAは保存文字列の照合であり、本作業でNPZを再decodeして配列同一性を再評価した結果ではない。

固定ファイル集合は[旧M4 SOURCE manifest][SO1]、[旧native SOURCE manifest][SO2]、[新統合SOURCE manifest][SN]から辿れる。科学SOURCE commitと、結果を収録した上記aced1647のcommitは役割が異なる。

共通条件は、Nα=Nβ=2・sector36（8 system qubit/full256、ancilla込みfull512）、一般対称J/real orthogonal U、元g補正κ・核scalar一回・元frame順0..7、tail identity抽出。時間は`m=0..12`、`n_m=2^m`、`t_m=.1×2^m`。単一のh=.1条件であり、別delta窓の検証や系サイズ移送は含まない。各候補自身のDF ground-state oracleを使い、状態準備費用は研究範囲として除外する。物理ground stateや準備費用が候補間で相殺するという主張ではない。[M][I][E]

共通RPE条件はCA=.00159 Ha、DF reserve=1/8、βtotal=π/6、βstat=π/12、候補別failure=.01、軸別failure=.01/26。P0はQiskit1.3.0/opt1/seed17、basis `rz,sx,x,cx`、approximation_degree=1、qubits_initially_zero=false、MPFR160 declared phase、位相fitなし。会計は状態準備を除くfenced serial native kernel compositionでそろっている。[M][O][I][CT]

SOURCEと候補は意図的に異なる。新PF/RTE・ground・law・native価格は新候補固有であり、旧J/Uの数値を新J/Uへ移した比較ではない。旧M4のPFは、同じ旧J/U・h/s/時刻のM3保存PFをidentity確認後に再利用したもの。rだけが異なるRTE側の値との混同を避ける。[M] 本書でも旧M3 r64や旧s4/r16のshots・価格・寄与率を今回の比較へ流用しない。

## 3. λ・attenuation・PF/RTE・signal room

### 3.1 数式で説明できる関係

exact tail λは、元順LD2より後の非identity Z/ZZ atom係数の絶対和`Σ|c_a|`である。抽出したidentity/scalarは別の相対位相として保持する。smooth DF目的のmassとは区別する。[E] 固定K0ではpaired-order normalizationが`B_0(τ)=√(1+τ²)`なので、[既存RTE SOURCE][RT]／[M4の保存normalization式][TA]より、

\[
\tau=\lambda_R h/r,\qquad
A_m=B_0(\tau)^{-r2^m}
=(1+\tau^2)^{-r2^m/2}.
\]

したがって**h/r/m/K0を固定すればλ低下はA増加につながる**。これはnormalizationによるattenuationの関係であり、実raw信号の絶対値そのものの式ではない。J/U変更によるPFの非可換積、補正RTE誤差、回路価格の単調性はこの式から導けない。

### 3.2 全13 roundの保存PF/RTE/A

PFは`‖U_PF(h)^n−exp(−iH_DF t)‖₂`、補正RTEは`‖U_corrected(h)^n−U_PF(h)^n‖₂`。以下は無次元の保存値の転記で、表示を丸めた。旧は[M4 A各mXX.json][AR]の`reference`および[既報表][M]、新は[C各mXX.json][NR]の同keyおよび[既報表][I]による。旧も新も全13 roundが`POTENTIALLY_FEASIBLE`であり、CA達成の認定ではない。

| m | 旧PF | 新PF | 旧補正RTE | 新補正RTE | 旧A | 新A |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.21615922e-5 | 6.31574602e-5 | 4.67188595e-5 | 5.06218778e-6 | .999918729761 | .999987382051 |
| 1 | .000123986862 | .000125932746 | 9.33575683e-5 | 1.01147747e-5 | .999837466128 | .999974764262 |
| 2 | .000245310383 | .000248852021 | .000186068962 | 2.01531196e-5 | .999674958673 | .999949529160 |
| 3 | .000470161325 | .000474946274 | .000367052608 | 3.97093875e-5 | .999350022997 | .999899060868 |
| 4 | .000800736972 | .000804130306 | .000696792640 | 7.50808953e-5 | .998700468465 | .999798131925 |
| 5 | .000962476519 | .001083294468 | .001176041179 | .000125980312 | .997402625712 | .999596304600 |
| 6 | .001295195448 | .001557628146 | .001612116979 | .000221344034 | .994811997777 | .999192772170 |
| 7 | .002451266114 | .003020752509 | .002122382260 | .000402953735 | .989650910921 | .998386195957 |
| 8 | .004504607426 | .005939657638 | .003929821417 | .000754410454 | .979408925486 | .996774996277 |
| 9 | .008973753953 | .011836658210 | .007516671110 | .001503319724 | .959241843322 | .993560393204 |
| 10 | .017934339409 | .023658178996 | .015005467813 | .003005501035 | .920144913979 | .987162254944 |
| 11 | .035835492634 | .047291851139 | .030024709246 | .006014202299 | .846666662722 | .974489317585 |
| 12 | .071614997914 | .094546804594 | .060945308518 | .012063601599 | .716844437765 | .949629430088 |

全13 roundで新PFは大きく、新補正RTEは小さく、新Aは大きい。λだけを目的にすればPFも改善する、という観測ではない。

### 3.3 Signal roomからshotsへの接続

本書の「signal room」は、保存されたraw complex signalの**半径下界`radius_lower`**を指す。決定論的な位相予算の余裕とは区別する。既存[accuracy SOURCE][AS]は、候補自身のDF ground state・alias・IID仮定の下で、

\[
\Delta_m=A_m(\epsilon_{PF,m}+\epsilon_{RTE,m})+epsilon_{synth,m}+epsilon_{law,m},
\quad\rho_m=A_m-\Delta_m,
\quad\beta_{det,m}=\arcsin(\Delta_m/A_m),
\]
\[
\epsilon_{coordinate,m}=\rho_m\sin(\beta_{stat})/\sqrt2,
\qquad
N_{m,b}=\left\lceil\frac{2\log(2/\alpha_{axis})}{\epsilon_{coordinate,m}^2}\right\rceil
\]

を用いる。最後の式は[既存Hoeffding実装][HS]による整数十分数であり、最小必要shotsや実量子測定数ではない。DF誤差[Ha]はenergy側で一度計上され、上式のoperator/signal誤差へ直接加えない。

次表は**ideal_exact_native laneだけ**の保存値。旧は[B各ideal_exact_native_mXX.json][BR]の`radius_lower, shots`、新は[新C/result.json][N]の`lanes.ideal_exact_native.rounds[m]`に対応する。X/Y両軸で同じNを使う。

| m | 旧ρ | 新ρ | 旧N/axis | 新N/axis |
|---:|---:|---:|---:|---:|
| 0 | .999809858158 | .999919163264 | 512 | 512 |
| 1 | .999620157023 | .999838720175 | 512 | 512 |
| 2 | .999243719544 | .999680537597 | 512 | 512 |
| 3 | .998513353234 | .999384457156 | 513 | 512 |
| 4 | .997204884940 | .998919098208 | 514 | 513 |
| 5 | .995269662545 | .998387517998 | 516 | 513 |
| 6 | .991919768494 | .997415236026 | 520 | 514 |
| 7 | .985124595642 | .994968014904 | 527 | 517 |
| 8 | .971148170596 | .990102516580 | 542 | 522 |
| 9 | .943423537587 | .980306319483 | 575 | 532 |
| 10 | .889835517896 | .960840876439 | 646 | 554 |
| 11 | .790905025391 | .922543137948 | 817 | 601 |
| 12 | .621819319431 | .848389050813 | 1322 | 710 |

m12の保存`raw_signal_error`は旧.09502511833464428→新.10124037927499521と**増加**した。一方、Aが増えた条件でρは上表のように大きく、`beta_deterministic`は.1329516390033004→.10681339785244368、`epsilon_coordinate`は.11380083373778974→.15526597244503024となった。旧1322→新710 shots/axisは、この複合的な信号・誤差会計に対応する。raw誤差を含めたすべてのsignal誤差が改善した、と説明しない。[旧m12][B12]／[新C/result.json][N]

| lane | 誤差の扱い | 旧総shots | 新総shots |
|---|---|---:|---:|
| evidence | 全law合成・実装law誤差が未資格 | null | null |
| ideal_exact_native | 理想合成・ideal finite-RTE IIDの反実仮定 | 16056 | 14048 |
| assumed_raw_signal_budget | 合成・law誤差を各.001A以下と仮定する感度lane | 16120 | 14104 |

出典：[旧M4 B][M]／[新C/result.json][N]の`lanes`。Ideal laneの0は定義した反実モデル内の値であり、未知の実装誤差を0と認定したものではない。Evidence laneの未知値はnullのまま。有限native検査を全law誤差へ移送していない。

## 4. Shotsとkernel重みWは別の指標

保存会計では、

\[
N_{total}=\sum_{m,b}N_{m,b},\qquad
W=\sum_{m,b}N_{m,b}\,2^m\quad(s=1),
\qquad b\in\{X,Y\}.
\]

各shotはround mのcoherent kernelを`2^m`回使う。両候補は全roundでs1/r128/K0なので、この比較のW差はround別の必要shotsの違いに対応する。r128の128 micro-eventsはkernel価格に含まれ、Wにさらに128を掛けない。[旧affine][AC]／[M3 serial・共有校正SOURCE][MS]／[serial SOURCE][CS]

| ideal lane | 旧PR | 新PR | 新/旧 |
|---|---:|---:|---:|
| 総shots | 16056 | 14048 | .874937718（保存scalarの比） |
| kernel重みW | 16631888 | 10487392 | .630559321（保存scalarの比） |

旧Wは[AC]の`affine.ideal_exact_native.W`、新Wは[N]の`lanes.ideal_exact_native.weighted_SE.kernel_reuse_weight`。総shotsは[M][N]による。比は引用済みscalarの算術照合であり、新しいround再集計ではない。

総shotsには同じ重みで入る各測定も、Wでは高mほど大きく効く。例えば保存N/axisはm0..2で同じ512、m12では1322→710である。**総shotsの減少率だけをWへ一律に適用することはできない。** 今回のW減少を、全候補・全scheduleでshots総数だけから予測できるとはしない。

新候補についてm10..12などのgroup別W・費用寄与割合はこの範囲の保存集計にないため、新しく算出しなかった。旧M4の`high_m_groups6.csv`はM3 r64対M4 r128の比較であり、今回の新旧J/U比較の寄与率として引用しない。高roundの新しい寄与率が必要なら不足証拠として別途扱う。

## 5. Native kernel価格と正しさ

### 5.1 全6metricの保存mean・SE

各候補の固定32 trajectoryに対するcompiled complete h-step controlled PR kernelの標本平均である。出典は[旧native result][ON]と[新C/result][N]の`calibration.sample_mean`／`sample_mean_standard_error`。新/旧倍率は[統合報告§4][I]の既報値。SEは表示を丸めた。

| metric | 旧mean | 旧mean SE | 新mean | 新mean SE | 新/旧価格 |
|---|---:|---:|---:|---:|---:|
| RZ count | 14076.4375 | 255.139624 | 19825.625 | 214.461468 | 1.40842631525 |
| RZ depth | 4342.28125 | 67.447474 | 5632.1875 | 57.586338 | 1.29705727836 |
| CX count | 5579.4375 | 78.735072 | 7399.3125 | 66.627734 | 1.32617535370 |
| CX depth | 3424.875 | 35.665611 | 4152.59375 | 29.896962 | 1.21248038250 |
| Total depth | 9574.53125 | 146.428193 | 12393.1875 | 124.337392 | 1.29439104395 |
| Circuit size | 28014.4375 | 514.159145 | 39603.9375 | 433.579954 | 1.41369740156 |

RZ/CX countはgate数、RZ/CX depthは既存filtered depth、total depthとsizeはそれぞれQiskitの`depth()`と`size()`による。[metric SOURCE][MT] 総資源のdepthはfenceで区切ったserial段の和として会計し、長い全回路を再最適化した最小depthとは呼ばない。

旧新ともPCG64・32 samples・128 events/sample・非適応、master/cohort/候補が異なる**非paired標本**で、旧標本をpoolしていない。32は古典価格校正数で、量子shotsへ掛ける数ではない。[O][I][NE] `FINITE_BIT_POINT_ESTIMATE_ONLY`であり、標本mean/SEを厳密な母平均・保証CI・ideal-law価格・全law support上界とは扱わない。新旧差の検定やpaired SEは計算していない。

### 5.2 Gaussian/head/tailと価格定義

旧新とも[共通build_kernel][KB]が、決定論headのforward half sweep、signed K0 tail、reverse head half sweep、scalar/抽出identity相対位相を含むcomplete kernelを構築する。Head内は既存StrongBuilderの相対Gaussian融合、tailは隣接する厳密同一basis_keyのrunでGaussian往復を共有する。異basis間の一般相対融合やcross-head/tail追加融合を費用から差し引いていない。[tail sequence SOURCE][TS]／[新native orchestration][NS]

J/U変更は、head対角係数、Gaussian基底・回転列、tail atom係数とfragment選択確率、同一basis runの分布などへ反映され得る。λはこれらのnative構造やP0 compile後gate数を表す量ではない。**λ減少からkernel単価減少は直接には導けず、今回の標本平均では単価が上昇した。** どの構造が上昇の主因か、Jだけ/Uだけの効果かは、保存された今回の二候補比較では未分離である。

Kernel単価にancilla head/X/Y shellとsystem状態準備は含まれない。共通shellはinterrogationごとに一度だけ別計上する。[共通shell][SH]ではRZがhead=2、X=2、Y=1なので、X interrogationのshellは4 RZ、Yは3 RZである。「決定論head半sweep」と「ancilla head shell」を混同しない。

### 5.3 保存native correctness

旧[Native128報告][O]と新[統合報告§4][I]に基づく。両候補で32 raw/compiled full512、actual adjoint、complex X/Y・unitarity・相対位相の数値検査がPASS。128 QPY/cohortは既存保存物で、本作業ではdecodeしていない。

| 保存最大観測値 | 旧PR | 新PR |
|---|---:|---:|
| raw/reference全成分差 | 2.242727e-14 | 5.72875080707e-14 |
| compiled/reference全成分差 | 2.554363e-12 | 4.28789615561e-12 |
| compiled actual adjoint/reference差 | 2.560195e-12 | 4.29228552179e-12 |
| raw complex X/Y差 | 3.557768e-14 | 1.66278832933e-14 |
| compiled complex X/Y差 | 9.267111e-14 | 1.20642488545e-13 |

Component/adjoint allowance=1e-10、signal allowance=2e-10、rtol=0、位相fitなし。これらは有限cohortの数値検査であり、全trajectory・全occurrenceの合成誤差上界や厳密丸めenclosureではない。Frobenius/max-entryとoperator2 boundも区別する。実装law誤差とcost supportの未知値を埋めない。

## 6. 総RZの記述的寄与：W改善と単価悪化の積

### 6.1 SOURCE上の正確な会計範囲

旧[affine会計][AC]と新[serial/aggregate SOURCE][CS][AS][NS]は同じ範囲の式、

\[
C_{total,k}=W\bar C_{kernel,k}+S_k,\qquad
S_k=\sum_{m,b}N_{m,b}C_{shell,b,k}
\]

を用いる。以下は**ideal lane**。W・平均価格・総RZの保存値について積の整合だけを確認した。Kernel部分と新shell残差は、本書の整合確認値であり、新たなround集計・回路測定ではない。

| 会計項目 | 旧PR | 新PR | 出典・区分 |
|---|---:|---:|---|
| W | 16631888 | 10487392 | [AC]／[N]の保存値 |
| kernel平均RZ | 14076.4375 | 19825.625 | [ON]／[N]の保存値 |
| W×kernel平均RZ | 234117731939 | 207919101020 | 上の保存scalarの積の整合確認 |
| shell RZ | 56196 | 49168 | 旧[AC]に明示保存。新は保存総RZ−Wμの残差確認 |
| shell込み条件付き総RZ | 234117788135 | 207919150188 | [ON]の`lanes.ideal_exact_native.conditional_totals.rz_count`／[N]の`lanes.ideal_exact_native.conditional_totals.totals.rz_count` |
| 総RZ descriptive SE | 4243453649.48214 | 2249141484.14022 | [ON]の`weighted_calibration_SE`／[N]の`weighted_SE.weighted_SE` |

新S合計は独立の集計fieldとして保存されていないため、49,168を新しい保存fieldであるかのように引用しない。旧S=56,196と新残差=49,168はいずれも[SH]の別shell政策と整合する。総費用を単純なWμの値へ書き換えない。

### 6.2 二因子の倍率による説明

引用scalarの比の整合確認として、

\[
\frac{W_{new}}{W_{old}}=.630559320746,\qquad
\frac{\bar C_{new}}{\bar C_{old}}=1.40842631525,
\]
\[
\frac{(W\bar C)_{new}}{(W\bar C)_{old}}
=.630559320746\times1.40842631525
\simeq .888096340666.
\]

Wは約36.94%減、単価は約40.84%増であり、その積のkernel部分は低下している。Shell込みの**既報保存比**は`.888096337507285`、約11.19%減である。[I][N] これが「W改善が単価悪化を相殺しても総費用低下が残った」の会計的意味である。総shots比.874937718をこの式のW比へ置換すると説明が変わるため、両指標を分ける。

旧新とも同じ32標本平均を全roundで共有するため、校正SEの伝播は`SE_total = W SE_kernel`である。13個の独立校正として√13で縮めない。[MS] SEは既存のdescriptive値の引用にとどめ、新旧差のCI・統計的有意性・母平均順位を追加しない。

感度lane `assumed_raw_signal_budget`の保存総RZは旧235155559637.25、新208893540201.5、新/旧`.8883206526085895`。[ON][N] 統合報告の新値208893540202は丸め表示であり、本書ではJSONの小数を保持した。Evidence laneのshots/総RZは両候補ともnullで、反実laneの改善を実CA達成や保証付き総資源優位へ昇格しない。

この節は保存資源会計の記述である。新しい4-corner代入、対称寄与配賦、round群再集計、統計推定、再ランキングは実施しなかった。旧M1の別s4/r16比較で報告された寄与額も今回の候補差へ転用しない。

## 7. 保存強決定論benchmarkとの関係

[M2結果][DR]の`candidates.proposed.cost.totals.rz_count`は101825929455、新PRの既報比は2.041907707602962。[N] 保存主RZでは新PRが低費用ではない。一方、既報のCX depth比は.7513872490513744なので、全副指標で支配されたという結論でもない。新しい指標横断順位は作らない。

| 比較条件 | 保存M2強D | 今回の新PR |
|---|---|---|
| J/U | 旧Stage A γ=.1（f47a…cd227） | 新update005（9383…6e56e） |
| PF構造 | 全frame ordered S2、PR splitなし | 元順LD2のpartial ordered S2 |
| 時間 | δ=h=.1/s1/m0..12 | 同じ |
| 回路融合 | 合法な全sweep相対Gaussian融合 | head内relative、tail同一basis共有 |
| RTE/attenuation | RTE非適用、A=1 | r128/K0、候補固有A/RTE |
| 合成誤差の資格 | full512とnorm-growth伝播による数値pilot | 全law誤差は未資格、資源は反実仮定lane |
| 共通部分 | H4・basis・rank、CA/時間/failure配分、P0、own-DF oracle、状態準備除外、serial metrics | 同左 |

出典：[M2報告§2・4][D]／[統合報告§4][I]。DのRTE非適用は定義であり、Dの未知合成誤差を0にした意味ではない。M2報告にある旧PR s4との条件差は、その歴史的比較の記述であり、今回の新PRへs4を転記しない。

**Codex-Aの同一新J/U強決定論比較は本書に結果未収録。** 取得後は、candidate identity・κ/scalar/順序、同じCA/RPE/状態条件、PF/synthesis資格、合法融合・価格scopeを確認し、新候補のDとPRを対応づける。それにより、現在の約2.04倍という比較に含まれるJ/U不一致を除き、新J/UでのPF・必要shots・native価格・主RZの関係を評価できる。ただし、その結果も単独ではJだけ/Uだけの因果効果、PRの全law保証、一般的優位を証明しない。未取得のPF値・価格・勝敗は推定しない。

## 8. 主張の範囲と、次の判断を変える追加証拠

| 科学的に説明できる範囲 | 残る限界 |
|---|---|
| 保存候補間でDF予算を維持してλを低下できた | 新候補はU一次NOT_MET。収束済み最適化・大域最良・分子間汎化ではない |
| 固定K0式でλ低下とA増加が結び付く | PF・native単価・総費用のλ単独因果は不明 |
| 新A/RTE/ρ改善とPF悪化が同時に観測された | raw_signal_errorも増え得る。全誤差改善とは言えない |
| 保存Wと標本単価の積が条件付きRZ低下を説明する | shotsだけの寄与率、J/U別効果、価格上昇の主因は未分離 |
| 保存強Dより主RZが大きい | 新J/Uでの公平比較の最終順位は未決定。一般PRのNO-GOではない |

以下は**追加検証の候補**であり、正式研究方針の採択や本作業からの科学実行許可ではない。

| 担当 | 次の具体的な証拠・作業 | 変え得る研究判断 |
|---|---|---|
| Codex-A | 別担当の範囲で、新update005と同一J/Uを使う強D比較を準備。本書では実行・変更しない | 旧J/U benchmarkに依存しないD/PR比較が可能か |
| 統合担当Codex | Codex-Aの結果取得後、上記identity・誤差資格・費用scopeを照合し、両報告を接続する | PR内部改善と新候補でのDへの競争力の区別 |
| ユーザー／研究C | J固定/U変更、U固定/J変更等の許容範囲と物理精度を保つ比較を選ぶ。必要ならCodexが限定契約を準備 | Gaussian/fragment分布と価格上昇の寄与を分離する価値があるか |
| ユーザー／研究C | PF・attenuation・round別W・native価格をより直接扱う目的関数の必要性を判断 | λ中心の目的を変えるべきか。既存surrogate未資格を根拠なく復活させない |

同じ新J/UでのD比較が主RZ判断を変える最も直接的な不足証拠である。一方、Uのnative価格への影響を因果的に切り分けるには、何を固定できるか、DF物理予算とlawがどう変わるかを先に定める必要がある。今回の二候補観測だけでその機構を特定しない。[証拠統合の資格区分][E]

## 9. 作業範囲・保全・引渡し

作業は指定commitから独立worktree `/tmp/prt-c-pr-resource-contribution-20261102`、branch `codex/research-c-pr-resource-contribution-20261102`で実施。入口・研究概要全文・関連報告・status/manifest・既存sourceを読解し、独立資料確認を並列化した。正式共通運用指示`research_project_instructions_unified_2026-10-25.md`は依頼元workspaceにある版の§8.1/8.4/10/10.1/10.3を確認した。同文書はこの固定commitの収録物ではないため、存在しない公開リンクは付けない。

再利用は既存の報告・JSON・SOURCE定義に限り、計算pipelineを実行しなかった。検証は引用key・path・表示丸め・保存scalarの積/比/会計残差・Git差分・公開本文の技術照合。新fit、PF/RTE、sampling、native build/compile、shots/総資源の新計算、実H4 NPZ/QPY decode、round再集計、統計推定、重要GPT科学レビューは行っていない。新コード・テストも追加しない。

本書による新しい科学結果・採択・資格昇格はない。[VALIDATION_STATUS][VS]／[validation manifest][VM]の既存証拠水準を保持し、immutable CI・外部再現とは呼ばない。旧SOURCE/raw/契約/grant/ledger/marker/STOPおよびCodex-Aの比較契約・実装は変更しない。共有の研究概要・PROJECT_MAP・manifest・研究ノート・索引も本branchでは編集せず、必要な入口へのリンク統合は統合担当へ引き渡す。

報告のcommit/push後、remote SHA/tree、公開本文と引用元の整合を確認して完了する。**Mandatory STOP：新しい実H4科学計算・数値再解析・重要GPT科学レビューへ進まない。**

## 出典対応

各表の出典記号とJSON keyは本文に記載した。`mXX`はm00〜m12の既存13ファイル。SOURCEリンクは定義の静的参照で、実行記録とは区別する。

[I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/result.json
[N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/result.json
[NR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C
[CT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/contract.json
[SN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/technical/SOURCE.json
[NE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_integrated/2026-11-02/run01/C/samples/sample00_events.json
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_aware_df_M4_s1_r128_feasibility_breakeven_20261028.md
[SO1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/technical/SOURCE.json
[SO2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4_native/2026-10-28/technical/SOURCE.json
[AR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/A
[BR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/B
[B12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/B/ideal_exact_native_m12.json
[TA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/A/tau.json
[AC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/C/result.json
[O]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_aware_df_M4_native128_execution_20261028.md
[ON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01/result.json
[SH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_stage_b2/2026-10-27/run02/shells.json
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md
[DR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/research_c_pr_aware_df_M2/2026-10-28/run01/result.json
[E]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_evidence_synthesis_and_decision_readiness_20261031.md
[RT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trotterlib/rte.py#L1959-L1970
[HS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trotterlib/rpe_resource_accounting.py#L553-L561
[AS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/c1_stage3_accuracy.py
[CS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/c1_stage3_connection.py#L128-L153
[MS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/c_pr_aware_df_m3.py#L341-L354
[NS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/c_pr_integrated_performance.py
[KB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/c_pr_aware_df_stage_b.py#L118-L138
[TS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/pr_dictionary_validation/circuits.py#L116-L153
[MT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/src/trottertracks/pr_dictionary_validation/f2_execution.py#L32-L40
[VS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/VALIDATION_STATUS.md
[VM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/artifacts/validation_manifest.json
