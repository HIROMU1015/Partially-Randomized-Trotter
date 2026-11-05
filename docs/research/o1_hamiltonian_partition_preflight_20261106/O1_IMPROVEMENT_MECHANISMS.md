# O1：全RPE費用へ接続し得る改善機構

2026-11-06 JST。静的preflight。以下は**技術的に検討可能な既知法の応用候補2件**であり、性能改善の確認、独立新規性の認定、正式採択、算法仕様の固定ではない。新しい最適化目的・sampler・selector・実行契約は作っていない。既存結果と条件は[比較表](O1_EXISTING_METHODS.md)、将来必要になる確認は[最小計画](O1_MINIMUM_VALIDATION_PLAN.md)を参照する。

## 1. 目的量と判断の単位

元Hの厳密基底状態からE₀を推定し、状態準備を除いたfull interrogationの費用を、資源aごとに

\[
G_{\mathrm{total}}^{(a)}=\sum_{m,b}N_{m,b}\,
\mathbb E[C_{m,b}^{\mathrm{full,no\text{-}prep},(a)}]
\]

で扱う。RZ、CX、T、depthは別の量で、交換率なしに合算しない。古典fit・係数collect・誤差情報取得・価格較正・online生成、追加qubitは別欄に残す。価格較正用の古典trajectory数を量子shotへ重ねて乗算しない。[既存費用接続][costmap]

同じ成功taskに対しては、局所λの減少や安い1 shotだけで採否を決められない。以下の関係は既存SOURCE・規範式の整理であり、新しい保証の導出ではない。

| 接続点 | 表現・splitが安くし得る部分 | 同時に増える費用／壊し得る条件 |
|---|---|---|
| 1. D側PF | 安いcore、少ないbasis移動、低い位相biasならstep数を抑えられる可能性 | HD/HR境界の非可換性、内部D積公式、順序変更、追加stage。λ_RからPF係数の単調性は出ない |
| 2. random norm/event | exact collect・別辞書でλ_Rが小さければdimensionless stepが小さくなる | 非局所Pauli、Gaussian往復、重いevent、希少高次数、生成費。support数だけではevent数は減らない |
| 3. finite-RTE | 同じHRなら補正meanとnormalizationを分離して比較できる | 打切り、符号・identity・child順、finite-bit law、合成誤差。channel一致だけではcontrolled平均振幅を保存しない |
| 4. attenuation/shot | positive normalization Bの減少はraw半径を改善し得る | 補正estimatorのrange、許容bias、半径下界が必要。phase不変をshot負担ゼロとしない |
| 5. native回路 | 同basis run・既知diagonal-only control・actual adjoint・合法fusion | basis recipe、D/tail境界、ancilla phase、synthesis precision、routing。primitive価格の和はfull価格ではない |
| 6. round成功 | 正しい半径・位相予算なら有限shotでbranchを保てる | 早いroundの誤branch、alias prior、長時間のPF位相混入、fit誤差。最終roundだけの費用評価では不足 |
| 7. 全round費用 | 各roundで必要なNと実価格を接続した総和 | 整数shot、共有較正の共分散、rare-law価格、未知の全law誤差。未取得欄はnull/UNKNOWNのまま |

### normalization・bias・shotの区別

既存二次partial-S2の記号を用い、PF step幅δ、roundのPF反復q_m、tail occurrence内のRTE microstep数r_m、偶数cutoff K_mを区別する。`t_m=q_m δ`の既存構成では、[rte.py][rte]・[会計SOURCE][accounting]が

\[
\tau_m=\lambda_R\delta/r_m,\quad
B_K(\tau)=\sum_{k=0,2,\ldots,K}\frac{|\tau|^k}{k!}
\sqrt{1+\left(\frac{\tau}{k+1}\right)^2},\quad
A_m=B_{K_m}(\tau_m)^{-r_m q_m}
\]

を扱う。全候補へこの同じPF構成を採択するという意味ではない。K=0でもpaired次数は1まで含み、単純に「Taylor次数ゼロのidentity」とはしない。

補正meanをQ_m、raw signalを`Z_raw=A_m〈E₀|Q_m|E₀〉`とする。正当化した基準半径下界ρ★,lbとfinite-mean信号誤差ε_Zがある場合、既存の保守的接続は

\[
\rho_{m,\mathrm{lb}}^{\mathrm{obs}}
=A_m(\rho_{\star,m,\mathrm{lb}}-\epsilon_{Z,m})>0,
\quad \epsilon_{\mathrm{coord},m}
=\rho_{m,\mathrm{lb}}^{\mathrm{obs}}\sin\bar\beta_{\mathrm{stat},m}/\sqrt2,
\]

\[
N_{m,b}\geq\left\lceil\frac{2}{\epsilon_{\mathrm{coord},m,b}^{2}}
\log\frac{2}{\alpha_{m,b}}\right\rceil
\]

である。これは各round・各軸内fresh IIDの±1測定に対する十分式。B補正後のrangeで数える場合とraw半径で数える場合を二重計上しない。新しいweighted/correlated/postselected lawに無変更で適用しない。

PF energy biasからround位相へは既存の`β_PF,m≤t_m ε_PF`、finite-RTEからは適用域内の`arcsin(ε_Z/ρ★,lb)`を使う。Hamiltonian fit、native synthesis、finite-bit lawの誤差源は別に接続する。異なる単位のenergy誤差とoperator/signal誤差を直接加算しない。各roundの位相条件、全軸の失敗配分、共通energyのbranch/aliasを満たす必要がある。[研究方法][method]

**厳密な入力ground stateでも、近似PFや近似DFの固有状態とは限らない。** unit-radius/single-phaseの仮定を自動付与せず、旧`C_use`をrigorous上界にしない。最終的なE₀推定に必要な全roundの成功条件を閉じるまでは費用優越を主張しない。

## 2. 候補M1：同じ物理tailの辞書を替える、normalizationと価格の交換

**操作と位置付け。** 元H、保持operator HD、残差operator HRを固定し、HRのglobal collected Pauli、rotated Z/ZZ、既存S/P混在という同値な辞書の違いを比較する既知の応用経路。新しいrouter/selectorやsingleton recipeを設計しない。既存[matched partition設計][matched]と[Phase 2C][mixedprior]の課題を全RPE taskの条件へ対応付ける。

**費用削減の作用経路。** 同じHR・同じmicrostep時間・r/Kなら、正確な辞書では補正済みtail meanは既存のpaired polynomial `P_(K+1)(−i HR δ/r)^r`に対応する。一方、辞書別λ_Rはnormalization・raw attenuation・event回路価格を変える。したがってPFと理想補正meanを保ちながら、raw半径と必要N、full単価の積を変える機構は残る。この条件付き関係は[既存静的設計§7][matched]にも記載されている。今回新たな同値性定理を証明したものではなく、丸め・実装lawまでの一致を保証していない。

**既知法との差。** 混在辞書、Pauli collect、basis共有、cost-aware選択は既知。UWCが係数を集中するのに対して、ここで隔離する変数は同じHRの実装辞書である。ただしbaselineへ同じS/P poolとcollect・価格情報を開放すれば、同じ選択が可能かもしれない。これは独立な新算法の根拠ではない。調べる価値が残るのは、同一補正meanの下でどのN×価格の交換条件が成立するかである。[既存方法差資料][mixedprior]・[先行研究比較][prior]

**必要情報・前処理。** 元の係数/fragment/frame/sector/JW順、厳密なoperator同定とscalar bookkeeping、実装可能recipe、canonical finite law、全wrapper価格と精度情報。真のE₀、gap、近似H ground、全状態ベクトルは選択oracleにしない。係数をglobal Pauliへ展開・collectする古典負担やcold cache価格取得は無料ではない。

**確認済みの部分。** IのMD-INPUT/CONTROL/F1/F2/STATとRTE-BASEが対応。F1/F2はexact rational代数、4arm、修正版controlled operator・X/Y signalと単回価格を限定検証済み。F2のN加重費用はnull。H2の4方式比較はsplitが異なるため、M1の純粋な辞書因果の性能実証にはならない。元版・SOURCE・条件は[比較表§3](O1_EXISTING_METHODS.md#3-保存済み結果負結果の条件とsource)へ。

**相殺要因・未検証条件。** 小さいλが重いGaussian/controlや非局所Pauliを要求すると利益は消える。F2でPP端点が低単価だった事実、core/pair強対照負結果を保持。実装law、native誤差、希少event、全round半径・N、価格母平均と取得費、分子への移送は未資格。literal DPで全native価格を代用しない。

**最小追加検証とSTOP。** 既存の同一split記述と元H/state対応を先に照合できる。将来の科学実行が必要なら、既存のPauli/DF両端点を最初の比較集合として、同じ補正mean・scalar・state条件とN×full価格を確認する。混在内点や新selectorを自動追加しない。同情報・同poolの既知対照で再現されれば独立新法claimはSTOP。全体価格またはNの増加で総費用利益が消える、共通targetを保てない、必要情報がoracle化する場合もその条件の候補をSTOP。PS48/singletonの正式一時終了を解除する計画ではない。

## 3. 候補M2：圧縮coreの残差補償で元Hを保ち、backboneを軽くする

**操作と位置付け。** 既知のDF/X-DF/RC-DF等による実装可能coreと、元Hとの差を残差として扱う既存PRの応用経路を検討対象に残す。記帳上は`H=H_core+(H−H_core)`であり、これは新しい分割規則・rank・optimizer・PF仕様の提案ではない。既知PRの残差保持とRC-DFの圧縮を接続したときの成立条件を問う。[PR §VII.B][prpaper]・[RC-DF][rcdf]・[O1既存課題][opportunity]

**費用削減の作用経路。** backboneを実装しやすい少数blockへ移し、差分を実装可能な小さいrandom残差へ置ける場合、D側のsweep/基底変換費と、random normalization/shot負担の交換が生じる。M1はHD/HR固定、M2はHD/HR自体を変える点で独立した作用経路である。compressed fitの誤差を捨てず残差に戻すことにより、近似H自身のground energyへtargetを置換する必要を避けられる可能性がある。

**相殺要因。** Frobenius residualや近似H energy差が小さくても、実装辞書の残差λ_Rが小さいとは限らない。圧縮で残差の係数絶対和や非可換PF biasが増えると、microstep/PF step/shotが増える。general signed J、basis往復、scalar・一体補正、残差Pauli展開・価格較正の古典負担も増え得る。元Hとの差をexactに記帳することと、そのfinite-bit実装・RTE誤差を小さくすることは別である。

**既知法との差。** 圧縮、rank選択、正則化、residual Pauli化、symmetry shift自体は既知。SCDFも圧縮とsymmetry shiftを組み合わせる。[SCDF原著][scdf] この候補はそれらを新原理と呼ばず、**元H stateで残差を支払い、強い既知表現・PFにも同じcore候補を与えた後に、backbone低下が全RPEで残るか**を問う。UWCをnorm-onlyや固定の弱いprefixへ制限しない。

**必要情報・前処理。** 元積分とscalar、一体収縮補正、既存coreのJ/UまたはDF factors、再構成差分、対象sector、残差の実装可能辞書。fit・collect・価格・PF誤差の情報取得費を双方で記録する。未知E₀/gap・近似ground・未知のbias補正scalarを無料にしない。近似Hをそのまま推定する枝を比較するなら、別途元Hへの誤差転送と元H state上の信号が必要で、単なるground-energy差の検査では足りない。

**確認済みの部分。** PR-2のDF prefix＋残差、RTE-BASEの一般involution、DF-CONTROL、研究Cの既存fit/codec/native/pipelineを部品として参照可能。研究CのJ/U→exact tail係数mapは実在する。一方、own-DF groundの005/064/固定wordを、M2の元H state・完全残差補償・全RPE結果として流用できない。既存libraryのDF preparationが一般のfitted-core＋任意残差をそのまま受けるとは確認していない。[再利用地図][reuse]・[SOURCE比較表](O1_EXISTING_METHODS.md#5-sourceから見つかった再利用上の制約)

**先行NO-GOとの区別。** 研究Cのnative-cost fit調査は新候補を選んでいない。本候補はswitch penalty、overlap目的、凍結event価格fit、rankだけの削減、PS48を改名した再開ではない。現時点で新しいfitや探索を準備する技術的GOはない。残差補償という既知経路に適合する既存coreがない／具体的な価格機構が出せないなら、そのまま保留する。[研究C既存NO-GO][cfeas]

**最小追加検証とSTOP。** まず既存core入力・元H・残差定義・一体/scalar対応・SOURCE版を静的に照合する。その後の残差係数生成、元H stateでのPF/RTE信号、full価格、shot/全round会計は新科学計算であり別許可が必要。最初から新optimizerを作らず、適合する既存core候補がある場合の限定比較だけを将来の選択肢とする。残差負担で利益消失、誤差転送不能、oracle依存、同candidate強D/同情報UWC・既知residual法に差なしなら、当該条件でSTOP。N1 exact fallbackや研究Cの限定負結果を消さない。

## 4. 候補に格上げしない論点

| 論点 | 今回の扱い |
|---|---|
| phase/commutator-awareな非prefix split | 原PR自身がsplit依存PF誤差を認識し、既知のcommutator最適化もある。実装可能な追加規則・同情報対照後の差がまだ提示されていない。独立した実行候補へ格上げしない |
| UWC＋RC-DF＋joint optimization | 既知要素の結合としてbaseline/応用の範囲。新目的・勾配・solverの作成を提案しない。M2にも自動採用しない |
| basis-switch集中、near-basis共有、weighted-price fit | 研究Cの既存技術NO-GO、exact keyと近いbasisの差、native非加法価格の未資格を保持。同じproxy探索を再提案しない |
| 反可換pair、core吸収、Gaussian singleton | 既知対応、強対照負結果、正式一時終了を保持。実装が存在しても再開許可ではない |
| 高次PF交換、native合成、round別表現 | 公平なbaseline機会またはO3/O4等の別論点。O1の新機構として数えず、新scheduleや新合成を作らない |
| 元H ground→近似DF groundの差し替え | 今回のtask外。無料state oracleを増やして得た費用差を候補利益と呼ばない |

## 5. 技術的結論

M1とM2には費用を変える異なる構造がある。しかし、強baselineと同じ情報・候補・最適化機会で追加利益が残ることは**未確認**。既知法の応用としての成立条件を調べる余地と、新規性・性能の実証を分ける。現状の証拠だけで「UWC超え」「PR＋RPE総費用削減」「新optimizer着手可」は結論しない。

この段階の成果は、既存証拠で閉じた部分と、元H/state・全law・半径・価格・全round成功の未接続部分を切り分けたこと。次の科学条件選択や重要レビューはユーザーの別判断とし、公開照合後mandatory STOPする。

[costmap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rte.py#L1100
[accounting]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py#L626
[method]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/研究方法・解析手順.md
[matched]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/h4_df_pauli_matched_partition_static_design_20261102.md
[mixedprior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md
[prior]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/PRIOR_ART_COMPARISON.md
[opportunity]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md
[reuse]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[cfeas]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[prpaper]: https://arxiv.org/pdf/2503.05647v2
[rcdf]: https://arxiv.org/html/2212.07957v3
[scdf]: https://arxiv.org/html/2403.03502v1
