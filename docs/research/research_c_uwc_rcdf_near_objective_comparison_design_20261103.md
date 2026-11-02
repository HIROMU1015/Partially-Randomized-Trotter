# 研究C：UWC／RC-DF／SCDFとの方法差と、近接entrywise目的による比較設計

2026-11-03 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
共通起点 `ba0f280c8be9e7038c5fa6b5fd390a235b113833`。

## 1. Executive Summary

**概念的重複は強い。** PR向けの係数集中、軌道・対称性変換、CDF係数正則化、inner/outer最適化、減衰・shots・回路価格の接続は既知の要素である。研究Cの式が先行研究と異なることは、独立した貢献の成立を意味しない。特にUWCを除いた位置付けでは、PR向けHamiltonian最適化という重複を過小評価する。

**主要な新規比較は二目的だけに絞る。** 現在の写像後PR-tail mass二乗と、同じtailのweighted entrywise L1二乗（符号相殺を無視したtriangle envelope）を比較する。既存uniform L2/RC2は保存済みの比較証拠として残し、新しい大規模ランキングのarmに増やさない。UWC全体・SCDF全体の再実装は行わない。

技術判定は **ORACLE_READY / SCIENTIFIC_CONTRACT_NOT_READY**。最小のentry目的oracleを既存ffsim loss/ADへ接続し、新15＋既存23＝**38人工・回帰テストPASS**。実H4の配列decode・fit・U最適化・保存数値再解析・PF/RTE・native build/compile・新科学callは0。科学的な追加利益、新規性、中心claim、論文化GOは判定していない。

[未承認設計JSON](research_c_near_objective_comparison_contract_unapproved_20261103.json)は**実行不能な設計案**である。強度政策、到達度・予算、ENTRYのstable reduced接続、下流選択と校正精度が未固定なので、実行entryと最大call数をnullにした。恣意的なγ/r/N/call数を埋めて実行可能とは表示しない。

## 2. 固定資料・SOURCE・一次文献の確認範囲

入口PROJECT_MAP、研究概要の最新研究C記録、[最新Evidence Map][MAP]、[既存の貢献整理][POSITION]、[初期設計][INITIAL]、[正則化比較][REG]を起点に、以下の関連節・式を確認した。旧記録の未実行表示は、その後の固定実行報告と区別した。論文全体の網羅的新規性レビューではなく、確認外の手法・式の不存在を主張しない。

| 一次資料・使用した版 | 確認した節・式・対象と指標 |
|---|---|
| Kanasugi et al., [UWC 2603.22778v2][UWC]（arXiv改訂2026-03-30） | §II.2.4 式17/18、§III.1–3 式20/22–37、Appendix C、D.1–3/Algorithm2、§IV/V。Pauli-LCU、OO/BLISS、soft gate cost、STAR系見積り。 |
| Oumarou et al., [RC-DF 2212.07957v3][RCDF] | 式5/8/17、weighted L1/L2本文、Appendix A 式18–30、B。一般対称Z/U、tensor fitting、測定variance・qubitization λ。 |
| Rocca et al., [SCDF 2403.03502v1][SCDF] | §2.1–2.4 式4/13/15/19/20/22/25–27、Appendix A/B。一般V前段、rank-one W、shift、λ/Toffoli。 |
| Günther et al., [PR/RTE 2503.05647v2][PR] | §VII.B 式42、Appendix C.3、E.3 式E21/E22。DF/基底、partial randomization、RTE/sample・gate cost。指定HTMLは404のため44頁PDFを確認。 |
| Cohn et al., [CDF/QFD 2104.08957v1][CDF] | §II.1.3 式14–23、§II.3。一般core/軌道回転、固定U係数fit、回路基底変換。 |

UWCのHTML内表示日付より、版の識別は指定v2と[arXiv submission history](https://arxiv.org/abs/2603.22778v2)を優先する。PR v2の改訂日は[2026-07-10](https://arxiv.org/abs/2503.05647v2)。取得した論文全文を成果物へ転載していない。

正式運用指示 `research_project_instructions_unified_2026-10-25.md` は依頼元workspaceの原本で§8/8.4/10（10.1/10.3を含む）を実際に確認した。起点tree内のファイルだとは扱わない。承認証明・承認原文保存・新grantは作成していない。

## 3. 現在のPR-aware DFを正確に定義する

通常number CDFで

\[
g_{DF,pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},
\quad f=\tfrac12\|g-g_{DF}\|_F^2,
\quad F_\gamma=f/f_*+\gamma(M_\epsilon/M_*)^2.
\]

一般signed実対称J、frame別の実直交U、rank8、元順LD2、γ=.1、ε=1e-8 Ha、固定f*/M*を用いる。元g補正κを固定し、核scalarは一回、各fragmentのidentityは正確な決定論位相へ保持する。近似gDFからκを再定義しない。[既存定義][REG]

Spin mode a=2i+σではK_ab=J_ij/2、c_Za=−Σ_j J_ij/2、c_ZaZb=K_ab/2（a<b）。Z_RCDF=Jであり、spin lift Kと同じ行列ではない。固定tailの係数map Tについて

\[
\lambda_R=\sum_{t\ge2,a\ne I}|(TJ)_{ta}|,
\quad M_\epsilon=\sum_{t\ge2,a\ne I}\sqrt{(TJ)_{ta}^2+\epsilon^2}.
\]

Zero slotsもsmooth massへ入る。Exact λをε付きmassで置き換えてsamplingしない。異なるUのfragment係数を、同じPauli座標のように無条件で相殺しない。Penaltyは固定JでUに直接依存せず、Uはtensor表現とJ再最適化を通じて作用する。

物理DF誤差は元Hamiltonianに対するsector operator norm＋既存engineering guard≤CA/8で独立判定する。Tensor lossと物理誤差は別であり、許容された近似誤差は、同一sector spectrumを厳密に保つ変換ではない。

## 4. UWCとの対応：同じ狙いと異なる自由度

UWCの式17/35–37を、s_l=σ((|c'_l|−w)/ε_soft)と略記すると

\[
G_m=\min_{L_D}\{g_m^{det}L_D+g_m^{rand}\lambda_R^2\},
\qquad
G_{soft}=g_M^{det}\sum_l s_l+g_M^{rand}\left[\sum_l |c'_l|(1-s_l)\right]^2,
\]
\[
w=(|c_{L_D^*}|+|c_{L_D^*+1}|)/2,
\quad L_D^*=\arg\min_{L_D}(g_M^{det}L_D+g_M^{rand}\lambda_R^2).
\]

σはsigmoid。OOはH′=𝒰ᵀH𝒰、BLISSはH′=H−T、T=μ₁(N̂−N_e)+μ₂(N̂²−N_e²)+Σ_pq ξ_pq F̂_pq(N̂−N_e)。前者は全spectrum、後者は対象粒子数sectorのspectrumを保存する。Spin拡張はAppendix C。Soft境界は反復更新される。[UWC §III・Appendix C/D][UWC]

| 比較項目 | UWC | 研究Cとそこからの比較上の含意 |
|---|---|---|
| Hamiltonian/係数 | Pauli-LCU c′(h′,g′) | Cは回転number fragmentごとのZ/ZZ atom辞書。二つの係数集合を直接同一視できない。 |
| 軌道変数 | 単一のglobal OO | Cは各frameのU_tと独立J_t。全h/gへ一つの共役回転を作用させる操作とは異なる。 |
| 保存量 | OO全spectrum、BLISS target block | CのfitはHamiltonianを近似変更し、物理誤差予算を別に払う。 |
| 対称性自由度 | 粒子数・spin BLISS | Cはshiftなし。Cのframe数が多いだけでUWC変換族を包含するとは言えない。 |
| 目的 | deterministic count＋soft tail mass² | Cはtensor loss＋固定frame-tail mass²。共通するtail集中原理は明確。 |
| Split | 係数weightとsoft threshold | Cは元frame順LD2固定、可変threshold/決定論個数は最適化しない。 |
| 係数操作 | OO/BLISSによる分布変化 | Cのsigned Jは行和相殺・frame間再配分を許すが、その自由度自体は一般CDF/RC-DFにもある。 |
| 回路・指標 | PR/RPE Pauli rotation、STAR | CはGaussian/P0 kernel＋shellのRZ等。STARの実時間や他分子の値とH4/P0 RZの数値勝敗は比較しない。 |
| 実証対象 | 鉄硫黄・P450・Ru触媒等 | Cは現在H4の限定実装。規模・architecture・状態準備の違いを残す。 |

上表の右列の包含・比較可能性は、記載された変換とCの固定SOURCEからの本書の推論である。UWCのAlgorithm2はOO→通常BLISS→spin BLISSと離散cost確認、D.2はSLSQP/JAX、D.3は全λ最小化との比較も扱う。従って「λを下げるだけではtask費用を決められない」こともC独自の原理ではない。[UWC Appendix D][UWC]

**三段階の整理：** (i)概念は重なる、(ii)許される変換族と近似誤差・辞書・maskは異なる、(iii)その差による追加資源利益は未識別。Cのr32価格結果だけで(iii)をUWCに対して認定しない。

## 5. RC-DF・SCDFとの対応

| 対象 | 一次資料上の構造・目的 | Cとの対応と比較の限界 |
|---|---|---|
| CDF/QFD | 一般対称core ZとU、tensor二乗loss、固定U係数解とouter軌道更新（式14–23） | J/U自由度、tensor loss、二段階最適化は既知。Cの新原理として数えない。[CDF][CDF] |
| RC-DF | Z=J、式17はf+ΣρZ²。本文には(Σρ abs(Z)^p)^(2/p)、p∈{1,2}。Appendix A式18は非二乗L1も別に扱う | 本文L1二乗は近接対照の出所。現在のRC2は全frame ordered uniform L2。式17と付録の係数/微分規約をそのままコードの係数へ移さず、自実装ADを照合する。[RC-DF][RCDF] |
| RC-DF solver/指標 | U更新と係数solveのnested手順、L2線形問題・matrix-free CG、L-BFGS/JAX。測定MSE/varianceとFT λ | 既存joint ffsimやCのreduced solverは原論文solver全体の再現ではない。初期寄与一致は原論文の強度tuningと同じではない。[RC-DF Appendix A/B][RCDF] |
| SCDF前段 | 式4/19/20の一般V。式26はf(U,V)+ρΣ abs(V^t_kl−α_t) | 「SCDFは一般Vを扱わない」は誤り。一般V＋shift段階はCに近いが、shiftの自由度と対象normが違う。[SCDF][SCDF] |
| SCDF採用式 | 式27はV_t=W_tW_tᵀ、f(U,WWᵀ)+ρΣ abs(W_kW_l−α_t)。Shift後はrank≤2 | 任意signed対称J全体とは異なる。α=0でもCの一般Jを全て表せず、shiftを加えた族との全面包含は主張しない。[SCDF §2.4/Appendix B][SCDF] |
| SCDF shift/solver | −a₁N̂−a₂N̂²は固定Nで既知scalar差。W→α median→U、λ/Toffoliを評価 | 補償scalarと近似fit誤差を分ける。BLISSのtargetで零となる演算子、CのDF誤差予算と同一扱いしない。[SCDF §2.3/Appendix A][SCDF] |

PR/RTEの§VII.B式42とAppendix C.3はDF・軌道基底・normの最適化を、E.3はgate費用を既に扱う。Cの候補固有PF/RTE→shots→native接続は具体的な検証の候補であり、この一般原理の新発見とはしない。[PR/RTE v2][PR]

## 6. 同値・非同値・包含を、ε差から分離する

[既存の数理導出と人工反例][REG]を再利用する。一frameのexact massは

\[
\lambda_t=\sum_i|\sum_jJ^t_{ij}|+\tfrac14\sum_i|J^t_{ii}|+\sum_{i<j}|J^t_{ij}|.
\]

Triangle inequalityから、同じtailのentrywise上界は

\[
B_0=\sum_{t\ge2}\left[\tfrac54\sum_i|J^t_{ii}|+3\sum_{i<j}|J^t_{ij}|\right],
\quad B_0-\lambda_R=\sum_{t\ge2,i}(\sum_j|J^t_{ij}|-|\sum_jJ^t_{ij}|)\ge0.
\]

この差は行和の符号相殺そのもので、εを0へ送っても残る。Row内で相殺がない符号領域では差0、したがって**ε=0のmass二乗はweighted entrywise L1二乗と厳密に同じ**になる。この領域で同値な二目的を別手法として数えない。

一般signed Jでは固定entry重みで全域を再現できない。既存family J(t)=[[1,t],[t,1]]のt<0内に、entry符号が変わらないt=−1で行和のkinkがある。今回の人工例でもJ=[[1,1],[1,1]]と[[1,−1],[−1,1]]は同じ|J|、B0=5.5だが、λ=5.5と1.5になる。実H4の観測値ではない。

Uniform L2はmass二乗の交差項・tail mask・行和を含まず、一般に同値でない。一方、**線形写像後のnorm正則化**を許す広いclassにはC自身が含まれる。一般signed Jやframe Uだけを新規性とする余地も、RC-DFとの対応上ない。特定の符号錐での同値性と、無制約全域の非同値性を区別する。

## 7. 既知部分・具体的方法差・追加利益の不足証拠

| 層 | 現在言えること | まだ言えないこと |
|---|---|---|
| 既知の構成要素 | 一般CDF、係数正則化、OO/shift、PRのtail集中・RTE、基底融合、shots×価格 | これらをC独自の新原理とすること |
| Cの具体化 | 元順LD2のnumber atom写像後smooth mass、元κ/identity規約、独立DF予算、候補固有下流 | 他のmask・norm・変換族でも同じ利益がある/ないこと |
| 追加利益候補 | 符号相殺を取り込む目的が、近接entry上界より小さいλや低い条件付き資源を得る可能性 | 未収束RC2や同J/U強Dへの勝敗だけから、その原因を写像へ帰属すること |

最新update064/r32は、直鎖H4/1Å/STO-3G、number CDF8/sector36、元順LD2、partial S2/s1、δ=h=.1/K0/m0..12、own-DF ground oracle・状態準備除外、P0/fenced serialの条件で、独立native256のPR/D点比.955183617395、nominal95%近似t比区間[.946594634483,.963772600308]だった。[固定結果][PRICE64]　これは同じ064 J/U強Dへの条件付き価格結果で、UWC/近接正則化への比較でも目的の因果効果でもない。新しい区間や候補間差を本書で計算していない。

## 8. 最も近い比較目的の選定

| 候補 | 技術判断 |
|---|---|
| A：既存全frame uniform L2/RC2 | 保存結果を再利用。Normとmaskの二要素が変わるため、今回の主識別対照として新たに増やさない。 |
| **B：tail限定weighted entrywise L1二乗** | **主対照。** 同じJ/U/rank/tensor loss/maskとmass全体の二乗を共有し、行和相殺を使わない。固定Uでは両目的とも凸。 |
| C：UWC soft costをDF辞書へ移植、またはSCDF shift | 今回追加しない。Deterministic count/threshold/shift/変数制約まで変わり、Bより直接的な識別にならない。原法との勝敗も認定できない。 |

実装するBのsmooth continuationはordered重みw_ii=5/4、w_ij=3/2を使い、

\[
B_\eta=\sum_{t\ge2,ij}w_{ij}\sqrt{J_{ij}^{t\,2}+\eta^2},
\quad F_B=f/f_*+\beta(B_\eta/B_*)^2,
\quad \eta=\epsilon\frac{2n+1}{1.5n-1/4}.
\]

B*=Bη(J0)は将来の許可済みsetupで一回だけ固定する。現在の実H4から評価していない。このηは**零Jでのmass floorだけを一致**させ、滑らかさやHessianの一致を保証しない。Exact B0とλの差も必ず記録する。

Tail slots T=(R−LD)n(2n+1)について0≤Mε−λ≤Tε、0≤Bη−B0≤Tε。Norb4/rank8/LD2なら各上限2.16e-6 Haという構造上の式になる。目的二乗への摂動は(2λTε+(Tε)²)等を固定scaleで伝播できる。これを丸め誤差certificateやoptimizer解の距離保証にはしない。差がsmoothing感度と同程度なら「相殺の利益」と呼ばず保留する。εを0にした非平滑fitをL-BFGSへそのまま流す仕様も採らない。

## 9. 結果によって変わる判断

| 将来の観測 | 得られる判断材料と限界 |
|---|---|
| 比較可能な到達度・共通物理予算でλや条件付き費用に再現する差 | 固定tail写像を使う追加価値の支持材料。単一H4だけで独立新規性・汎化は決まらない。 |
| Bが同程度以上の性能 | 一般entry正則化で十分という説明が強まる。Cの原理的無効性・全条件での同値性までは示さない。 |
| λは改善、native単価やPFが悪化して総費用差なし | Proxyの差とtask追加利益を分離。λ低下だけを中心資源claimへ使えない。 |
| 到達度・DF誤差範囲・強度探索が比較不能 | 比較Iの勝敗はUNRESOLVED。比較IIの有限手順結果として保持し、都合の悪いarmを削除しない。 |
| 相殺gapが小さい/差がsmoothing感度内 | その条件では写像の機構claimを支持しない。実H4でこの状態かは未評価。 |

Rawな自己目的F_PRとF_Bの大小を、手法間の性能順位にしない。共通指標は元physical誤差、exact λ、到達度、計算量、および資格を合わせた下流費用である。

## 10. 公平な最適化・物理誤差・下流政策

**比較I（目的の能力）**は、共通物理予算・表現自由度・到達度のもとで問う。元qualified J/U/x0を共通起点とし、両arm共通の有限強度政策を結果前に固定する。共通CA/8 admissionを優先し、可能なら重なる物理誤差帯でλのfrontierを示す。誤差完全一致や同じγだけを公平性の定義にしない。初期値分布・再始動数を片armだけ増やさない。

**比較II（実用手順）**は、同じ初期化と計算予算に対する候補性能を問う。既存の初期penalty寄与=.1一致は再現可能な一pilot政策だが、最終誤差・強度の最適性を一致させない。最大受理U数だけでなく、失敗を含むU点・J solve・nfev/njev・CPU/wallを記録する。固定予算で未収束なら、その有限手順の結果として比較できるが、比較Iの一般優位にはしない。

停止診断は共通のisometric J座標と物理U接空間を使い、既存のJ∞/U₂各1e-5＋marginを共通engineering候補にする。Raw solver flag、独立終端、refinement stability、数値整合、DF予算は別々に保持する。Latest PR-onlyの64更新を新Bのcold startへ対する公平な対照とみなさない。強度集合・予算・収束不足の扱いは、どちらの比較原理を主とするか決めた後に固定する。

下流の依存順は

`candidate identity → 物理DF誤差 → exact tail law/λ → own ground/alias → PF/RTE → attenuation/phase → RPE shots/W → native → shell込み総RZ → 同J/U強D`

とする。各新J/Uに固有の誤差・law・価格を得る必要があり、064の保存値をBへ転用しない。

| 下流の問い | 推奨する有限政策（今回の実行固定ではない） |
|---|---|
| DF目的だけの差 | 両arm共通r32/s1/h.1/K0/m0..12を一つの候補とする。r32はPR探索由来と開示し、最適rや手順全体の順位としない。 |
| 実用的な候補別retuning | 同じ事前有限r集合・選定基準・予算を両armへ与える。既存16/32/64/128政策の再利用は候補だが、ここで正式採択しない。 |
| Selectionと校正 | r/候補を探索用情報で凍結してから、新holdoutで価格を校正。旧32/256/A cohortをpoolせず、適切なseeds・estimand・精度・多重比較政策を先に固定。 |
| 強D | 各armと同じJ/Uの全frame ordered S2、同P0/精度/serial/状態準備政策。旧別候補Dだけを分母にしない。 |

共通rだけでは候補別最適政策を比較したことにならず、PRだけr32/Bだけr128という既定にもならない。価格N=256を別候補へ根拠なく転用しない。W×μ＋shell、SE=W×SE_kernelを保ち、rと校正Nを量子反復へ重ねて掛けない。全law signal・ideal-law移送・actual CA・validated coverageが未資格ならnullのまま扱う。

## 11. 再利用確認・実装差分・既存RC2証拠

| 資産・固定版 | 再利用/適合性と残る差分 |
|---|---|
| [c_pr_aware_df.py](../../src/trottertracks/c_pr_aware_df.py)、[c_rcdf_comparator.py](../../src/trottertracks/c_rcdf_comparator.py) | 既存ffsim normalized tensor oracle・AD・入力検査・係数規約を再利用。旧目的を編集しない。 |
| [新c_near_objective.py](../../src/trottertracks/c_near_objective.py) | Bηの短いmass関数とpenalty AD、exact cancellation診断だけ。初期scale固定、直接U penalty0。Optimizer・分解・PF・native実装なし。 |
| 修正J/U codec、ffsim0.0.83 loss/util | 保存qualified x0の既存形式、一般J、det± gaugeを使用。外部2 source hashは元契約と照合。 |
| c_pr_coefficient_point.py、fixed-U map、c_j_u_stable/reduced | 同一binary64点・拡張残差・係数adjoint/Hessian・refinementを再利用できる。ただし現在PR/RC2を型でdispatchし、PR専用Tを持つ。BをRC2として流さない。 |
| [c_u_step_control.py（025d495…）][USTEP] | 最新のrecentered Armijo/独立終端/物理admissionを静的確認。共通起点にこのcode全体は未収録。全SOURCE closureの輸入・B stable接続は後続準備であり、完成runnerとは呼ばない。 |
| General-K、SectorCAR、rank_assessment/representation_gate | 同じsigned number表現と独立sector判定の再利用先。今回は実H4で呼ばない。 |
| 既存PF/RTE・RPE・native/serial・Calls/finish | 既存pipelineを後続contractへ接続する。新価格式・新call/STOP管理frameworkは作らない。 |

固定した最小実装commit：`acc32878daa9591597a4677b459e5454dd96aeda`。
[技術SOURCE9件](../../artifacts/research_c_near_objective/2026-11-03/technical/SOURCE.json)は**oracle/人工検証用**で、未来の科学runnerの全依存closureではない。必要な差分だけを独立moduleへ置き、既存科学SOURCEへ新目的を注入していない。

既存RC2の証拠は次のように使える。

| 保存段階 | 保存する情報と限界 |
|---|---|
| [初期RC2 fit][RC2FIT]（科学SOURCE150f904…） | 同x0・初期寄与一致の全frame L2という限定対照。DF予算PASS、相対減少停止で一次未達。 |
| [一次診断][RC2FIRST]（SOURCE0996a5a…） | 初期寄与一致≠gradient一致。既知tensor-null方向にRC2 penalty下降が残る。新しい強度の自動選定根拠にはしない。 |
| [共通停止政策](research_c_common_convergence_execution_20261101.md)（SOURCEe2f7dfd…） | 両未収束の有限終点。旧初期fit終点と別candidateとして保全。 |
| [RC2/PRのStage B](research_c_rc2_pr_stage_b_execution_20261101.md)、[shots](research_c_rc2_pr_conditional_shots_execution_20261101.md)、[native][RC2NATIVE] | 候補固有PF/RTE・減衰・shots・有限価格の接続証拠。原論文最強RC-DF、最新064、Bの新結果へ読み替えない。 |
| [stable RC2](research_c_j_u_stable_execution_20261102.md)（SOURCE12e0079…） | RC2の独立trajectoryはadmissionを通ったがU一次未達。親のPR例外後、新終端DF評価/qualified比較は未実施。初期DF PASSを終端へ流用しない。 |
| [update064最適化][OPT64]・[下流][DOWN64]・[独立価格][PRICE64] | 最新のPR側支持証拠。ただしU一次未達、比較相手は近接目的ではない。元SOURCE・条件・STOPを保持する。 |

この表は保存本文の引用で、NPZ/QPY・trace配列を新しくdecode/再解析していない。

## 12. 人工・回帰検証とpreflight

[人工test](../../tests/management/test_research_c_near_objective.py)、[実行log](../../artifacts/research_c_near_objective/2026-11-03/technical/artificial_tests.txt)、[preflight](../../artifacts/research_c_near_objective/2026-11-03/technical/preflight.json)を参照。

| 今回の試験 | 件数・内容 |
|---|---|
| 新near-objective | 15 PASS。符号制約下のexact同値、同abs(J)で異なるλ、相殺gap、tail mask、zero floorとsmoothing bound、固定Uの凸性、ffsim tensor値/gradient・独立FD・解析J gradient、直接U penalty0、strength0退化、異常値拒否、passive import。 |
| 既存RC2 | 14 PASS。既存の符号/写像反例・実ffsim route/mock optimizer・hook復元等を無変更再利用。 |
| 既存PR目的 | 9 PASS。係数/AD/codec等を無変更再利用。 |
| 合計 | **38 unique PASS**。Rank8/norb4/LD2/128 parameter形状も乱数人工tensorだけで確認。 |

新試験ではnumpy.loadとSciPy optimizerを禁止し、実データI/Oや反復へ逸脱しない。既存試験のoptimizerはmock。ffsim/JAXを既存fit venvで読み取り利用し、x64・依存sourceを照合した。Task-local cache/tmpと自processのBLAS調整だけで共有環境を変更せず、CPU/RAM/worker/wallへの人工capを追加していない。微小な人工配列のため多数processで同じJAX compileを複製しない。将来重い実行では実空き資源と支配負荷を再確認する。

Preflightは技術SOURCE9件、将来入力候補3件のbytes identity、ffsim2 source、passive importを確認する。実H4評価・scientific unitは0。人工PASSはH4性能、reduced inner安定性、比較の科学的GOではない。

## 13. 未承認科学契約と必要calls

[設計JSON](research_c_near_objective_comparison_contract_unapproved_20261103.json)は`UNAPPROVED_DESIGN_NOT_EXECUTABLE`、entry=null、maximum calls=null。これは実行許可待ちだけの完成契約ではない。

| 閉じる必要がある項目 | 推奨と未固定の理由 |
|---|---|
| 主RQ | 比較IかIIを選ぶ。前者には到達度・強度frontier、後者には同予算の手順定義が必要。 |
| 固定入力・初期値 | 元H4 h/g/c・qualified J/U/x0・scalesの3 identityを再利用。PR専用warm startを中立な初期値としない。 |
| 強度・予算 | 初期寄与一致.1はIIの候補。Iの強度集合/選定政策や、更新/trial/評価上限は未固定。 |
| Bの数値接続 | Same-array拡張精度のJ gradient/Hessian、共通reduced admission・物理U診断への最小adapterと人工確認が必要。現在のJAX oracleだけで代替しない。 |
| 物理・候補進行 | CA/8とgeneral-K/CARのcorrectnessを維持。両armの到達度/物理資格を分離し、不適格armを除外して勝者だけ進めない。 |
| 下流 | 共通r対照と候補別retuningを区別し、r/候補選択と新価格holdoutを分離。精度Nと主効果閾値は未固定。 |
| SOURCE/runtime | 最終runnerのtransitive SOURCE・実行環境・input/endpoint identityを新契約で閉じる。旧grant/未使用callsを移さない。 |

Call数の算出骨格は具体化できる。旧stable instrumentationを無変更で流用できた場合、一arm・一強度・一起点で、distinct U点Pごとに3 inner J solve＋2 precision state＋1 admission＝6 units、受理更新Aごとに独立終端＋物理＝2 units、初期物理1。したがって

`C_shared + Σ_registered_runs(C_setup_arm + 6P + 2A + 1) + C_comparison`

が骨格になる。Trial上限B、更新上限UならP≤1+UB。新Bの独立整合や親driverが同じinstrumentationで閉じるかをまだ検証していないため、この式を最大call契約へ昇格しない。旧12430-callや064実績をそのまま二倍する設計ではない。

Nativeは一candidate/r cellで9N個別units＋固有context/reference/shell＋calibration/summaryが基本。候補数・r/Nとseed/interval政策が未定なので合計は未定。旧2374-call価格契約や256標本を機械的に転用しない。新scientific runはfresh、retry/resume0、correctness FAILで部分証拠を保全してSTOP、正常終了もmandatory STOPとする。CPU/RAM/wallの人工上限は新設しない。

## 14. 残る未資格・未解決事項

近接目的が実H4で良い/悪いこと、同時一次収束・大域最適、写像による資源差の因果、他初期値/分子の再現、UWC/SCDF原法との公平な資源勝敗、独立新規性・重要性・論文化は未認定。全law synthesis・implemented-law signal、ideal-law期待価格への移送、validated CI・actual CAも既存の未資格を維持する。

Codex-Aの保存XDF/CDF静的監査を繰り返していない。今回の二目的は同じ一般CDFで比較するためAの結果待ちで文献・oracle作業を止める必要はない。**将来保存XDFを対照へ追加する部分だけはAの条件監査との照合が必要**であり、本設計がその公平性を先取りして保証しない。

## 15. 引渡し・保全・mandatory STOP

ユーザー／研究Cには、(a)UWCとの強い概念重複、(b)変換族・誤差・辞書の方法差、(c)追加利益を識別するPR対tail-entryの二目的案、(d)比較I/IIと未固定事項を渡す。中心claimの採択・正式新規性レビューは本書の技術判定から独立である。

次の技術担当Codexは、ユーザー／研究Cが主比較原理と有限政策を選んだ後、Bのstable接続を人工検証し、call/SOURCE/runtimeを閉じた実行可能契約を作る。実H4 fit/再解析/下流実行は、その具体的範囲への別明示許可後だけ行う。**今回の設計JSONをそのまま実行するコマンドはない。**

独立worktree `/tmp/prt-c-uwc-rcdf-near-objective-comparison-design-20261103`、branch `codex/research-c-uwc-rcdf-near-objective-comparison-design-20261103`。共通起点からの変更は新module/test/本書/設計案/技術記録の追加だけ。元checkout・旧科学SOURCE/input/raw・契約/grant/ledger/marker/STOPを変更していない。共有索引・Evidence Map・研究概要・validation manifestは編集せず、A/B完了後の統合担当へ引き渡す。

Commit/push後、remote SHA/tree、公開本文・引用先の版/式、技術SOURCEと主要旧結果のblob一致を照合する。新科学runner・新backup・承認証明機構は作らない。**公開照合後mandatory STOP。**

## 固定参照

[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0f280c8be9e7038c5fa6b5fd390a235b113833/docs/research/research_c_evidence_map_updated_20261103.md
[POSITION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0f280c8be9e7038c5fa6b5fd390a235b113833/docs/research/research_c_contribution_positioning_20261102.md
[INITIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47e0d406226be365951bd8666d5acae61106f500/docs/research/research_c_pr_aware_df_optimization_design_20261026.md
[REG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/150f904f6975868abe09966dda8ab41722e2a7cc/docs/research/research_c_regularization_comparator_design_20261101.md
[UWC]: https://arxiv.org/html/2603.22778v2
[RCDF]: https://arxiv.org/html/2212.07957v3
[SCDF]: https://arxiv.org/html/2403.03502v1
[PR]: https://arxiv.org/pdf/2503.05647v2
[CDF]: https://arxiv.org/html/2104.08957v1
[RC2FIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/335a7e04951e009bd6910afaecd1a3107e0d61a4/docs/research/research_c_regularization_comparator_execution_20261101.md
[RC2FIRST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5fdcd81addf83e893d76c89e64f733433a4e76b3/docs/research/research_c_regularization_first_order_execution_20261101.md
[RC2NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/611a618674aba7146b0103cc34f156c9bc89bc16/docs/research/research_c_rc2_pr_native_execution_20261101.md
[USTEP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025d495166311a58ec9c24c43f07ed6e40f65f42/src/trottertracks/c_u_step_control.py
[OPT64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/docs/research/research_c_u_step_control_execution_20261103.md
[DOWN64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_execution_20261103.md
[PRICE64]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md
