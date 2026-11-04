# 研究C：fresh IID-grid期待bias校正・技術準備

2026-11-04 JST。**技術adapterと人工検証を追加した。科学実行契約は `NOT_READY_METHOD_AND_NUMERICAL_INPUTS`、非実行、call上限null。** Kernel期待operator差を単純Hoeffdingで校正する案は、m12への伝播で必要標本が巨大になる。現状のper-word enclosureとpopulation supportを未資格のまま、小標本実行を「有用な期待bias保証」として推奨しない。新実H4計算・sampling・compile・保存raw数値再解析・shots/Stage C・重要科学レビューは0。

## 固定出典・再利用

起点は結果commit `3649b75af191916daf44abb91e14d34c43b83e8f`。固定4-word、後続B1/B2統合、IID-grid/accuracy adapterが同じ履歴にあり、他branch全体のmergeやraw再取得が不要なため選んだ。独立worktree/branchで作業した。正式共通指示原本2026-10-25版の§8.1・§8.4・§10を確認し、承認証明・承認原文保存・新grantを作らない。

| 証拠 | 固定結果commit | 科学SOURCE / 状態 |
|---|---|---|
| 固定4-word | `3649b75af191916daf44abb91e14d34c43b83e8f` | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、実装`6e8475cc871916473d48fd38fb4b4a0297e84d15`、86/86 COMPLETE/STOP |
| Stage A | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` | `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、38/38 COMPLETE/STOP |
| B1 | `69c9a764aee66cf2ca79ab29bd3145a027385bcc` | `P0_UNIFORM_NOT_ESTABLISHED`、準備のみ |
| B2 | `b7a3db09e395d36965e5b9dc28ae4a299db3b2f0` | `01baf2b5e400773e00a5ae548941265f85b9882b`、27/27 COMPLETE/STOP |
| B1/B2後続統合 | `891aa78f47e66184eaf8a6d527ca8773ba6bbdd0` | 実装`c22d79cfd5c6a1202587c2211031626cb8023df3`、旧`90a2a7b...`を保存した追補、NOT_READY |

[固定4-word報告](research_c_fixed_word_pilot_execution_20261104.md)、[Stage A](research_c_iid_grid_stage_a_execution_20261103.md)、[B1](research_c_p0_uniform_method_readiness_20261103.md)、[B2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b7a3db09e395d36965e5b9dc28ae4a299db3b2f0/docs/research/research_c_reference_bound_execution_20261103.md)、[後続統合](research_c_b1_b2_reference_bound_integration_20261103.md)の公開5本文を取得し、固定Git blobと一致確認した。SOURCE/入力/旧STOPの確認はbytes・JSON metadataのみ。NPZ/QPYの科学的decode、旧runnerの再起動は行わない。

再利用対象は `c_pr_integrated_performance.native`、`c_pr_aware_df_stage_b2`、`c1_stage3_connection.readout_segments/serial_native_sequence`、`c1_h4_full_synthesis.propagate`、`c_pr_aware_df_fixed_certificate`、`c_pr_aware_df_guarantee_strategy.zero_failure_tail_limit`、`c_iid_grid_bridge`、`c_pr_aware_df_stage_c2.choice_grid`、`c_b1_b2_*`、`c_iid_grid_accuracy`、`c1_stage3_accuracy.RoundEvidence/assess_round/assess_energy`。旧SOURCEの関数・default・資格は無変更。

## 共通条件

Linear H4・隣接1.00 Å・STO-3G、Nα=Nβ2、sector36、number CDF rank8。PR endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元frame順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12、CA=.00159 Ha・DF reserve=CA/8。元g補正κ/scalar・frame順・own-DF ground-state oracle・状態準備費用除外を維持する。

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、MPFR160 declared phase、既存head relative Gaussian/full-frame tail/basis_key共有。Controlled full512、head/X/Y shell込みfenced serial、long-round全体の再compileなし。PF/RTE/DF/attenuation/費用の保存値を新しく集計していない。

## A. Estimandと実際の信号

`c_pr_aware_df_stage_b2.XY_gate` と `c1_stage3_connection` の定義を採用する。System d=256、ancilla最後、初期状態ρ₀=|0ψ⟩⟨0ψ|。ψは正規化されたown-DF ground oracleという研究仮定で、保存binary64 vectorの真のgroundとの距離を認定しない。

- P：ancilla Hのpreparation shell。System state preparationとは別。
- S_X：ancilla H。S_Y：ancilla H S†（S†を先に適用）。最後にancilla Zを測定。
- C(w)：word wの**complete compiled controlled kernelの数学的gate作用**。保存binary64角度をexact dyadicとし、理想gate関数を適用する。Qiskit Operatorで観測した行列そのものをexact作用としない。
- T(w)：既存独立referenceの数学的literal saved-atom作用。Tail identity phase、forward/reverse head、32個のeventを同じ順序で含む。D(w)=diag(I,T(w))。
- n=2^m、V=C(w_n)…C(w_1)。Compiled raw complex signalは z_C(w₁:ₙ)=Tr[Z S_X V Pρ₀P†V† S_X†]+i Tr[Z S_Y V Pρ₀P†V† S_Y†]。X/Yは実測では別shot・別word列だが、両期待値の定義に同じ分布を使う。校正時のclassical paired評価では同一word列を両軸に使ってよい。
- Ideal shellとDを使うalgebraic referenceは z_T(w₁:ₙ)=ψ†T(w_n)…T(w_1)ψ。これは非正規化の式であり、途中・終端でnormを割らない。

第一estimandは **Δ_m=|E_q z_C−E_q z_T|**。Raw complex単位で、attenuationを後から掛けたり除いたりしない。RTE attenuationは期待信号の大きさに現れる。PF/RTE referenceのA_m(epsPF+epsRTE)は別のtransport legで一度だけ扱う。

**重要なtarget gap**：既存`event_unitary`はcosθ I−i sinθ Aを使用する。Literal binary64 UのCAR atom Aを数学的に作っても、Uの小さな直交残差だけではA²=Iを保証しない。したがってT(w)をexact unitary、Dを物理的な量子channelだと仮定しない。上式は現referenceと一致する数学的対象を確定したもので、未資格unitarityの解消ではない。Polar U等のexact-unitary trajectoryへ変更する案は別target bridge・研究採択が必要で、今回採用しない。

Actual inverseはraw/compiled実回路から作る復帰検査の対象であり、現在のX/Y信号生成にV†を追加実行するものではない。密度行列式中のV†はBorn則のbra側で、独立なinverse wordを抽出しない。将来inverseを含む別の回路functionalを対象にする場合、同じwに依存するC(w)とC(w)†を一つの確率単位として扱う。

## B. 平均unitary・平均channelの区別

IID occurrenceについて数学的積の期待値は E[T_n…T_1]=(E T)^n。この恒等式にunitarityは不要だが独立性が要る。Ideal control=0 I/control=1 Tとideal shellsなら、E_q z_T=ψ†(E_q T)^nψ。

一方、任意のcompiled full512 Cでは、Φ_C(ρ)=E_q[CρC†]という**平均channel**を用い、X/Y期待値は各S軸に対するTr[Z S Φ_C^n(Pρ₀P†) S†]になる。これは一般に(E C)ρ(E C)†ではない。各Cがexact diag(I,V_w)という構造を保つと証明できた場合だけ、off-diagonal ancilla blockを使う平均unitaryの式へ縮約できる。Binary64でcontrol leakageが小さいという観測では代用しない。Full512 channelをdenseに保持すると512⁴ complex128、約1 TiBなので新しい既定経路にはしない。

同じwordをn回反復する E[C(w)^n] とfresh IIDの(E C)^nは別。E[C(w)†C(w)]=Iと、独立に引き直したE[C(w)†]E[C(w)]も別。人工分布{I,Z}で両差をテストした。Mean channelが必要な場合も、各sampleに対するvector/state-actionを順に適用すればdense superoperatorは不要。

## C. 三つの校正法

| 観点 | 1. Kernel operator差 | 2. Kernel signal差 | 3. Round paired signal差 |
|---|---|---|---|
| 量 | μ=E_q||C−D||₂、非負 | E_q[z_C,0−z_T,0]または絶対差 | 各mのE_q[z_C,m−z_T,m]、複素signed差 |
| Sample | 一つの32-event word、raw/P0/独立reference/full512 enclosure | 同じword＋指定state/X/Y | n個の独立word、fenced serial product、同じ理想word列、両軸 |
| Range | C exact unitary・||D||≤aなら0..1+a。双方exact unitaryの定理がある時だけ0..2 | 正規化状態と双方unitaryなら軸差−2..2 | Literal targetでは軸差±(1+a^n)。双方unitaryなら±2 |
| 数値資格 | 全512入力列/全出力を覆うoperator enclosure。観測Frobeniusだけでは不可 | 真のstateと全signal評価の不確かさ | 同左＋n反復・shell・word相関を含むenclosure |
| 反復 | 両normを含むtelescopingで全mへ一つのμ CIを伝播 | 指定stateの一step信号差から任意中間stateへは移送不可 | 直接評価しkernel→roundの緩い換算を回避 |
| X/Y | State距離→両observable→complex norm | 一stepだけ直接 | Signed二軸のconfidence rectangle→complex norm |
| Confidence | 一つのμ上界の成功eventが13roundを同時に覆う。δを13回使わない | m0のCIだけ。全m資格なし | 13round×2軸×2tailのunion、相関があってもunion可 |
| N依存 | Hoeffding O(M² log(1/δ)/εμ²)、propagationでεμ≈b/4096 | 小Nでも得られるがround用途未資格 | Hoeffding O(W² log(52/δ)/b²) per round |
| Build/compile | 原則各sampleに一件ずつ | 同じ | 原則ΣN_m·2^m件ずつ。未実施wordをcacheで捏造しない |
| 計算・並列 | Full512 reference/評価/certificateが主負荷。独立sample/列を並列化 | State-actionで軽くできるがestimandが不足 | Matrix-free vector2本/状態密度の作用。巨大long-round回路を一体compileしない |
| Stage C | μ CI・shell・population norm・全transportが揃った別期待値qualification | 単独では不可 | 各mのCIとtarget/state/transport資格が揃えば別期待値qualification |
| 破綻条件 | 支持範囲未資格、μ統計誤差の4096倍がroom超過 | State依存、off-diagonal/control/phase差が後続で顕在化 | 未資格per-round数値誤差、oracle距離、計算量過大 |

方法2のsigned差は相殺を保持できるが、m0だけのscalar情報から全operator/channelを支配しない。人工の「初期stateでは差0、中間stateでは差あり」で拒否条件を確認する。

## D. 両normと反復への伝播

`c1_h4_full_synthesis.propagate` の両map norm-growth・shell telescopeを再利用する。全kernelと全shellのnorm≤a≥1、per-occurrence E||C−D||≤μ、n=2^mなら

E||C_n…C_1−D_n…D_1|| ≤ a^n min(2,nμ/a).

これは両mapをaで縮小して積をtelescopingし、normalized product間の距離≤2を使う式。Literal targetをunitaryと見なしたcapではない。Head/X/Yの誤差e_H,e_X,e_Yをそれぞれ一回だけ足し、各axisのstate距離上界を

t_j=a^(n+1)e_H+a²·a^n min(2,nμ/a)+a^(n+1)e_j

とする。Complex signal差は 2a^(n+2)√(t_X²+t_Y²) 以下。a=1・shell誤差0なら **2√2 min(2,nμ)**。Compiled kernelのexact control構造が未証明のため、安易にnμだけをraw biasにしない。

旧関数はbinary64 numerical pilotを返す。新adapterのscalar companionは同じ式をFractionと上向きsqrtで評価し、旧関数を厳密上界へ改名しない。人工a=1/1.001で両式の対応を回帰確認した。Native実装のpopulation aとshell enclosureは今回未評価でnull。大きなaの指数増大・不明なaを1とするfallbackはない。

B2のB_m/U_mは既に各roundのboundなので2^mを再度掛けない。Categoricalについても、literal targetのnorm≤aを今後資格付ける場合、一般形2a^n·TV(product law)などの置換が必要。既存Stage Aのunitary条件付きrowを無条件boundへ格上げしない。

## E. Per-sample数値誤差の経路

新`sample_enclosure`は同じword・phase-sensitive full512の二legを要求する。

1. **Compiled→raw mathematical gates**：既存`c_pr_aware_df_fixed_certificate.record/compare`のArb192 gate evaluation、local block action、recenter radius、全入力列Frobeniusを再利用できる。Exact unitaryという資格は受理済み理想gate定義から得るもので、Qiskitのunitarity残差から得ない。実際のglobal phaseとinverse回路を保持しglobal phase fitなし。既存certificateのmin(2,...)はこの二つの理想gate回路同士に限定する。
2. **Raw mathematical gates→literal saved-atom trajectory**：別途、literal-U CAR atom構成、cos/sin、head exp、phase/積、Gaussian角度のtarget差を包囲する必要がある。既存fixed-CAR/K1のArb作用を参照できるが、旧h/r・polar U/κ target・精度は今対象へ自動移送しない。新しい適合したper-word bridge evaluatorは未完成、契約blockerとして残す。逐次factor作用・column streaming、immutable CAR/exp cacheで実装し、非unitary literal mapにはnorm-growth radiusを使う必要がある。

両legのenclosure・norm評価丸めを含んだupperの和がtrue dのsample上界になる。数値観測/感度推定は別kindで保持し、CIの真の誤差上界へ入れない。`c1_h6_state_action.StateActions`の小gate/Aer CPU作用と宣言phaseの復元も静的に確認した。Direct roundのbinary64 pilotを巨大unitaryなしに評価する再利用候補だが、その`state_gate`はoperator bound/enclosureをnullとする。有限probeを全operatorまたは真のgroundの保証へ流用しない。

B2はsector36の**平均**atom→数学CAR→binary64CARを囲う二項で、上記fresh per-word/full512に直接転用できない。AdapterはB2型scopeを拒否する。

Per-sample confidenceとcalibration confidenceは別。有限4-wordと旧native32をfresh IID標本へ再分類しない。保存ground vectorのscalar信号だけを囲ってもown-DF oracleとの差は未資格。方法1の全operator boundは任意の正規化入力へ有効なので、このcalibration legには数値ground誤差を入れずに済む。

## F. 統計confidenceとRPE failure政策

新`mean_upper`は0≤d≤Mの既知population bound、固定N IID標本、各true d_iを含む数値enclosureを要求し、mean(d_iの上端)+M√(log(1/δ)/(2N))で一側期待値上界を作る。上界を既知のMと交差することは正当だが、観測最大をMにすることは拒否する。Unknownはnull。

`paired_round_upper`はtrue D_X,D_Yのinterval上端/下端平均へHoeffding幅を加え、四tailにδ_m/4を割り当てる。13roundにはΣδ_m≤δ_calを明示する。同じword列でX/Yをpaired評価しても座標独立は不要。異なるroundでprefixを共有する場合もunion boundには独立を要しないが、各roundのN個のsampleはIIDでなければならない。Kernel CI一つを全roundへ伝播する案ではcalibration event IDを共有しδを一回だけ数える。

| 方法 | 強み | 限界・採択状態 |
|---|---|---|
| Bounded mean/Hoeffding | 単純・分布不問・符号付き差にも適用 | O(N^-1/2)、kernel反復で極めて大きなN。Scalar adapter実装済み、科学政策未採択 |
| Threshold tail calibration | d≤tのcertificate決定を使う。全Nでbad0ならp_bad≤1−δ^(1/N)、μ≤t+(M−t)p_bad | 既存`zero_failure_tail_limit`再利用候補。判定straddleをgood扱いしない。小Nでbad0でもp_badを0としない。Threshold/Nは事前固定・未採択 |
| Signed paired direct difference | E|差|ではなく|E差|の目標に合い、観測した相殺を統計式内で扱う | 数値enclosureとround計算が重い。13round同時confidenceが必要 |
| Empirical Bernstein | 真に小さいvarianceなら改善余地、range/N項は残る | 人工/実標本varianceを未評価。今回は公式比較のみ、未実装・未採択。観測極小varianceやmax値からsupportを推定しない |

参考一次資料：[Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830)、[Maurer–Pontil (2009), Theorem 1/4](https://arxiv.org/abs/0907.3740)。Scalar計算は[Python Decimal.lnのcorrect rounding](https://docs.python.org/3/library/decimal.html#decimal.Decimal.ln)と上向き1 successor、正確なFraction/isqrtを用い、統計式自体の下向き丸めを避ける。これは入力enclosureを自動的に証明する機構ではない。

RPE α_total=.01・α_axis=.01/26・β_total=π/6・β_stat=π/12・CAは無変更。Schemaは校正成功確率≥1−δ_cal、校正成功を条件とする測定failure≤.01、無条件failure≤min(1,.01+δ_cal)を別に返す。Calとmeasurementの独立性を仮定した積でなくunionを使用。δ=.05なら無条件上界.06であり.01ではない。新しいjoint .01への再配分は未採択。

## G. m12までの事前見積もり

[planning.json](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/planning.json)は**数学的な設計例**。実H4保存260/13セル・native32を再解析せず、真のunitary norm a=1、数値/shell誤差0、raw room b=.25<sin(π/12)を全部使用できるという楽観例に限定する。実H4にこのroomが残ることを認定しない。Attenuation<1、既存PF/RTE/transportの差引き、a>1は通常これをさらに厳しくする。

Kernel μの統計幅は約2.16e-5以下が必要。Hoeffding用width2、direct signed差は各axis width4、13round全52tail。Tail案はt=εμ/2、μ≤t+2p_badとlog上界を使うzero-bad例。

| 候補δ_cal（未採択） | Kernel Hoeffding N | Kernel zero-bad tail N | Direct N/round | Direct kernel occurrences（全round） |
|---|---:|---:|---:|---:|
| .10 | 9,889,527,671 | 426,816 | 1,601 | 13,113,791 |
| .05 | 12,866,572,143 | 555,301 | 1,779 | 14,571,789 |
| .01 | 19,779,055,342 | 853,632 | 2,191 | 17,946,481 |

これらは**選んだ十分統計式の設計上の標本数**で、情報理論的な普遍的下界ではない。Tail案の全bad0やthreshold達成も未観測。Directは標本数だけ減っても、Σ2^m=8191回のkernelを各sampleで必要とする。約1457万build＋同数compileという規模は、現full512 per-sample certificateを伴う初回案として推奨しない。

固定4wordの保存timingは4件全体wall約58.95秒/CPU約142.13秒/peakRSS約905MiB。これはsetup・mutex・4 workerを含む既存実測の引用で、fresh samplingやArb certificateの一件単価ではない。安易にこの値をN倍したwall保証・speedupは出さない。Full512 complex128行列1枚4MiB、system256は1MiB。複数matrix/cacheと各Python/Qiskit workerのbaselineを合算する。Arb全列・gate列の作用は通常binary64より重く、現在のArb per-word原価は未知。

**推奨**：Kernel法の数学的伝播は成立条件を明確化できたが、単純Hoeffding＋range2を採択しない。Direct paired方式は目標estimandに最も直接的な対照として残す。次に選ぶ最小stageは、①literal targetのpopulation normとper-word enclosure方式、②事前固定した十分小さいthreshold/rangeまたは数値誤差資格、③その原価とNを合わせた採否、を決める技術段階。Tight support/good-thresholdを正当化できた場合のみ、kernel tail/variance-sensitive校正を優先候補に戻す。現条件で全m有用なfresh期待bias保証を得られる実行案が完成したとは報告しない。

## H. IID-grid、乱数、再利用

数学モデルは`IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL`。各event/occurrence/calibration sample/measurement shotでIIDを仮定し、実PCG64列はその再現可能なsimulationという別資格。NumPy1.26.4のcomponent probability正規化、CDF/cumsum/再正規化、searchsorted_rightを保持する。新`grid_index`は与えられた53bit整数kをk/2^53へ写す境界処理だけで、乱数を作らない。CDFのzero-mass binと境界を人工テストした。

既存K0 samplerはorder choice（確率1）とatom choiceを各eventで呼ぶ。将来のstream replayはこの消費順も固定し、Stage A exact q tableと照合する。Fresh namespaceは新SOURCE/candidate/scope/sample/occurrence/axisのtask-bound keyを使う設計だが、master seed・backend・stream予約は未採択でnull。Unique seed/hashはIID証明ではない。

同じkernel校正標本を全roundの伝播に再利用してよいが、measurement用の独立wordをその有限標本集合から再抽出する運用にはしない。旧4word/native32・新calibration・実measurementは別。Compileの決定的重複は将来の固定cache政策でのみ再利用し、fresh抽出そのものを省略しない。

## I. Accuracy接続と誤差会計

新`c_iid_grid_expected_bias.py`は旧all-law gateにprobabilistic recordを通さない別qualification。次を検査する：共通candidate/SOURCE/model/P0、全13round、raw単位、phase/shell、state coverage、fresh固定N、calibration eventの同時confidence、connected/disjoint transport。Unknownや観測のみならaccuracy呼出し前にNOT_READY。

Reference chainは **compiled expectation→q atom expectation→p atom expectation→数学CAR→保存binary64 CAR**。第一estimandΔは最初のlegのみ。これだけでStage Cへ進まず、categoricalとB2二項を整合する資格で追加する。Literal targetのcategorical norm前提はまだ欠ける。Directにcompiled→binary64CARのtotal期待差を校正する案なら四legを既包含と宣言し、B2/categoricalを再加算しない。Primary Δとtotal transportは別quantity。

`assess_if_adopted`は政策採択前には何も算定しない。将来適格なら既存`RoundEvidence/assess_round/assess_energy`を直接呼ぶ薄いadapterで、新RPE/shot式を複製しない。旧engineのideal reference law＋raw transport slotを使用する意味は既存IID adapterと同じだが、結果は**校正eventとIIDモデルに条件付けた十分条件**とラベルし直す。Existing actual-law error_evidence、PCG64 qualification、rigorous/actual CAはnull/falseのまま。Existing PF/RTE/DF pilot、ground/alias仮定を厳密保証へ昇格しない。

## J. 未承認契約とcall式

[設計契約](research_c_iid_grid_expected_bias_contract_unapproved_20261104.json)。Method/N/δ_cal/統計式/per-sample資格/追加native coverageが未採択なので、**executable=false、maximum_logical_units=null**。数値上限を旧86/38/27/16から流用しない。Fresh output予定は新family/run01、現存しない。Retry/resume0、mandatory STOP。専用entryはpassive preflightだけでscience実行flagなし。科学permissionを渡すだけでも非実行設計は起動できない。

以下は**既存pipelineから導いた将来案のcall式**であり、実装済み科学runner・承認済み上限ではない。

- 共通setup48：保存input等11、catalog＋fresh policy gate2、fullCAR/native/basis/shell35（固定4wordと同じ処理区分を再利用、word gateはfresh policy gateに置換）。
- Kernel案：既存9処理/sample（draw、build、reference、raw check、compile、compiled check、actual inverse pair、XY、metrics）＋compiled/raw enclosure1＋raw/atom enclosure1。共通population support1、mean/tail calibration1、round propagation13、summary1。**64+11N**。安い既存metricsはreuse baseline内で保持するが新費用比較はしない。
- Direct案：各round sampleにn個の既存9処理＋round compiled作用/ideal作用のenclosure2＋paired signal reduction1＋word-list identity1。共通setup48、range1、13round calibration、summary1。**63+Σ_m N_m(9·2^m+4)**。Per-round enclosure内部の全算術・actionをこの定義へ固定する必要があり、細分化方式を変えるなら上限も実行前に更新する。
- Confidence確定後の既存accuracy最大16 callsは別の将来stage。今回案へ自動同梱しない。Population norm/enclosureの新しい詳細アルゴリズムでさらにcallが必要なら、未定の現在はnullのままにする。

資源はlaunch時のaffinity/quota/RAM/GPU実能力でsample/列/batch並列を選ぶ。Qiskit共有状態を伴うbuild/compile/actual inverseは既存`SharedQiskitCalls` mutexを維持し、必要ならprocess分離。Arb contextはprocess-local、FLINT cacheはimmutable入力またはworker-local、BLAS/Rayonとのoversubscriptionを避ける。Global lock/cache/venv/OS/他job変更なし。CPU/RAM/worker/wallの人工capなし。

Correctness/identity/入力/target不一致は依存task投入を止め、開始済みを回収して部分証拠とSTOPを保存する。数値enclosure不足やrange不明はUNRESOLVEDであり、0への代入・precision/threshold変更・sample追加・retry/resumeを行わない。Bad-tail threshold超過は登録統計法のbadとして数え、科学的correctness FAILと区別する。

## K. 人工・回帰確認、保全、次の担当

**新47件＋適合回帰152件=199 PASS**。旧固定SOURCEに依存する4件は除外した。変更前の起点3649b75で同じ4件を再現し、いずれも旧契約が指す`c_pr_integrated_performance.py`のdigest不一致であることを確認した（[起点の確認ログ](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/historical_parent_checks.txt)）。旧契約を現SOURCEへ付け替えず、fresh/STOP検査の意味を維持した。新passive preflightは科学importを禁止したsubprocessでもPASS。

新テストは既知unitary分布、IID product/shared word、mean channel、forward/inverse相関、phaseとactual inverse、X/Y符号、nonzero numerical interval、one-sided coverage、13round union、invalid range/IID/古いsample拒否、B2 scope拒否、unknown null、double count/source/model/round誤り、未採択accuracy/未承認science/fresh/retry/STOPを含む。詳細件数は[tests.json](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/tests.json)と[tests.txt](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/tests.txt)を正本とする。

[技術SOURCE](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/SOURCE.json)、[passive確認](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/passive_preflight.json)、[静的確認](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/static_verification.json)、[technical STOP](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/STOP.json)に記録する。旧SOURCE/raw/科学契約/grant/ledger/marker/STOPに差分を作らず、Evidence Mapの科学Claimは変更しない。公開後はremote SHA/treeと主要本文blobを照合する。

次担当は**ユーザー／研究C**：calibration法・target解釈・per-sample資格水準・δ_calと必要計算量を選ぶ。**Codex**は採択に対応する不足enclosure/rangeの技術準備と実行可能な新契約を具体化する。実H4 fresh sample/校正・native/certificate・confidence/accuracy計算には、完成した固定SOURCE/範囲/calls/STOPへの別途明示許可が必要。**今回の技術準備はmandatory STOP。**
