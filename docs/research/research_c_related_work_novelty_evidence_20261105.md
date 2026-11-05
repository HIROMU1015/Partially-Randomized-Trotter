# 研究C：先行研究・新規性候補と固定証拠の対応

2026-11-05 JST。文献調査・静的SOURCE読解・保存値の出典照合。**独立新規性、研究方針、論文化GOの正式認定ではない。** 論文構成・図表・A〜D担当への引渡しは[別冊](research_c_paper_outline_figure_plan_20261105.md)に置く。

## 1. この資料で確かめたこと

研究Cに近い先行研究は二つの軸で分ける必要がある。**数学的な表現自由度・最適化はCDF／RC-DFが近く、PR向け係数集中・資源目的はUWCが近い。** PR原論文もDF、軌道最適化、head/tail、RPE資源を扱う。一般signed J、frame別U、正則化、inner/outer最適化、Pauli写像、shots×回路価格を個別に新規な原理とは扱えない。

具体的な差分候補は、元のnumber-CDF frame順を固定したhead/tailに対し、**実際にサンプルする非identity Z/ZZ係数への写像後tail massを正則化し、その有限最適化終点を候補固有のPF/RTE・shots・native価格まで接続すること**にある。これは原論文との差分を記述したもので、独立したアルゴリズム貢献が成立するとの判定ではない。

保存されたPR対ENTRY比較には、両armの物理DF予算を満たす有限終点と、条件付き総RZの改善を支持する結果がある。ただしENTRYはRC-DF原法やUWCの再現ではなく、共通tailのweighted entrywise L1二乗対照である。未収束・単一強度政策・未実施retuning・単一H4という制限を残す。

### 範囲と資料の扱い

- 入口の`PROJECT_MAP.md`、`研究概要・現状.md`全文、研究目的、統一結果索引、指定の研究C設計・Evidence Map・実行報告を確認した。mainに収録された概要は旧snapshotの記述を含むため、11月の研究Cについては固定実行報告を優先した。
- 正式共通運用指示2026-10-25版§8.1/8.4/10/10.1/10.3は依頼元workspaceの原本を確認した。11月5日GPTレビューは依頼文の要約を背景に用いた。レビュー原文を読んだとは扱わない。
- `VALIDATION_STATUS.md`の旧DF screening／prose-only UWC使用禁止を維持する。baseの中央manifestには`research_c`登録がなく、研究Cの状態は下記の固定SOURCE・結果・STOPに結び付けた。中央manifestへの収録や公開commitへの固定だけで外部再現・CIを認定しない。
- 新規fit、実分子配列の数値decode、PF/RTE、shots、build/compile、新CI・比率・再ランキング・科学的再集計は行っていない。以下の数値・式は既存文書と保存JSON／SOURCEの引用である。

## 2. 一次文献と実際の確認箇所

原論文の関連本文・付録を対象とした限定調査であり、網羅的な先行技術不存在証明ではない。各リンクは確認した版を固定する。原論文の全数値を研究Cと同じ条件へ換算していない。

| ID | 原論文・著者・年／識別子 | 今回の確認箇所と用途 |
|---|---|---|
| L1 | Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, Christian Gogolin, **Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization**。2022 preprint／Quantum 8, 1371 (2024)。[DOI](https://doi.org/10.22331/q-2024-06-13-1371)、[arXiv:2212.07957v3](https://arxiv.org/html/2212.07957v3) | 本文式5/8/13–17、weighted L1/L2記述、Appendix A（最適化）、B（強度tuning）、F式48–60（number→Pauli写像）。一般coreと軌道、正則化、λの辞書依存性の主要比較資料。 |
| L2 | Shota Kanasugi, Riki Toshio, Kazunori Maruyama, Hirotaka Oshima, **Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime**。2026。[arXiv:2603.22778v2](https://arxiv.org/html/2603.22778v2) | §II.2.4式17/18、§III.1–3式20/22–37、Appendix C/D、§IV/V。PR/RPE費用、OO/BLISS、soft head/tail目的、資源見積り。 |
| L3 | Jakob Günther, Freek Witteveen, Alexander Schmidhuber, Marek Miller, Matthias Christandl, Aram W. Harrow, **Phase estimation with partially randomized time evolution**。2025 preprint／PRX Quantum 7, 020332 (2026)。[DOI](https://doi.org/10.1103/ynxb-p2xq)、[arXiv:2503.05647v2](https://arxiv.org/pdf/2503.05647v2) | §V/VI、VII.A/B（PDF pp.12–14）式42/43、Appendix A.2/C.3/E.3。PR、DF回転frame、軌道基底／norm最適化、paired RTE、RPE・gate費用。 |
| L4 | Dario Rocca, Cristian L. Cortes, Jerome Gonthier, Pauline J. Ollitrault, Robert M. Parrish, Gian-Luca Anselmetti, Matthias Degroote, Nikolaj Moll, Raffaele Santagati, Michael Streif, **Reducing the Runtime of Fault-Tolerant Quantum Simulations in Chemistry through Symmetry-Compressed Double Factorization**。JCTC (2024)。[DOI](https://doi.org/10.1021/acs.jctc.4c00352)、[arXiv:2403.03502v1](https://arxiv.org/html/2403.03502v1) | §2.1–2.4式4/13/15/19/20/22/25–27、Appendix A。一般core前段と採用rank-one core、symmetry shift、λ／Toffoliを区別する。 |
| L5 | Jeffrey Cohn, Mario Motta, Robert M. Parrish, **Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians**。PRX Quantum 2, 040352 (2021)。[DOI](https://doi.org/10.1103/PRXQuantum.2.040352)、[arXiv:2104.08957v1](https://arxiv.org/html/2104.08957v1)（preprint題名はCompressedなし） | §II.1.3–4式14–31、§II.3。一般対称core・frame別回転、固定Uの係数solveとouter最適化、量子回路。 |
| L6 | Mario Motta, Erika Ye, Jarrod R. McClean, Zhendong Li, Austin J. Minnich, Ryan Babbush, Garnet Kin-Lic Chan, **Low rank representations for quantum simulation of electronic structure**。2018 preprint／npj Quantum Information 7, 83 (2021)。[DOI](https://doi.org/10.1038/s41534-021-00416-z)、[arXiv:1808.02625v2](https://arxiv.org/html/1808.02625v2) | 式3–8、SI I.2/I.3。二段階分解・truncation、回転number表現、basis融合／partial rotationの基盤。 |
| L7 | Luis A. Martínez-Martínez, Tzu-Ching Yen, Artur F. Izmaylov, **Assessment of various Hamiltonian partitionings for the electronic structure problem on a quantum computer using the Trotter approximation**。2022 preprint／Quantum 7, 1086 (2023)。[DOI](https://doi.org/10.22331/q-2023-08-16-1086)、[arXiv:2210.10189v2](https://arxiv.org/html/2210.10189v2) | §2.3式27/31–34、§2.5式46/47、§3.2 Fig.4/Table2。FRO/GFROの一般係数・frame回転joint変数、PF誤差と一step回転資源の競合。 |
| L8 | Matthew Hagan, Nathan Wiebe, **Composite Quantum Simulations**。2022 preprint／Quantum 7, 1181 (2023)。[DOI](https://doi.org/10.22331/q-2023-11-14-1181)、[arXiv:2206.06409v3](https://arxiv.org/html/2206.06409v3) | §2式1、§4.2/5/6。Trotter＋qDRIFTのpartitionと所定条件下のdiamond-distance／oracle数定理。RTE/RPEとは区別。 |
| L9 | Kianna Wan, Mario Berta, Earl T. Campbell, **A randomized quantum algorithm for statistical phase estimation**。2021 preprint／PRL 129, 030503 (2022)。[DOI](https://doi.org/10.1103/PhysRevLett.129.030503)、[arXiv:2110.12071v2](https://arxiv.org/html/2110.12071v2) | Appendix C式34–38、D式39–41、E.2。paired Taylor、sample/gate tradeoff、cutoffの基礎。 |
| L10 | Davide Cugini, Touheed Anwar Atif, Yiğit Subaşı, **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms**。2026。[arXiv:2603.13495v1](https://arxiv.org/html/2603.13495v1) | §II式4–10/Theorem 1、§III式21–26/Theorem 2。固定circuit familyの費用付きimportance reweighting。DF変更とは操作対象が異なるため補助的に扱う。 |

**UWCの名称を区別する。** L2での正式名は *Unitary Weight Concentration* であり、依頼文の「Unitarily Weighted Clustering」と同一の展開名ではない。本書のUWCはL2を指す。repositoryに残る旧grouped-UWC実装・prose-only表とは、出典・状態を分け、旧表の数値は用いない。L2 HTML本文の表示日付とarXiv版履歴には差があるため、引用版はv2と[投稿履歴](https://arxiv.org/abs/2603.22778v2)で指定する。

L3のHTMLは取得できず固定PDFの本文を確認した。RC-DFの[著者データ](https://zenodo.org/records/7866658)は原論文リンクを確認したが取得できず、SCDFの[著者データ](https://zenodo.org/records/10836682)はarchive公開までの確認で内部コードを読んでいない。UWCの実装archive内部も未確認。公開実装の同一性・再現実行・全定理の独立再証明を行ったとの主張はしない。

## 3. 方法・目的・資源を同じ軸で比較する

| 対象 | Hamiltonian表現・変数 | 目的関数／PRへの対応 | 研究Cとの重複・具体差分 |
|---|---|---|---|
| CDF（L5） | 一般対称core Zとframe別実直交U。CのJに対応 | tensor二乗残差。固定U係数fitとouter U更新 | signed core、独立frame U、分離／nested構造は既知。Cは既存表現へ固定PR-tail目的を付加。 |
| RC-DF（L1） | 同じ一般core Z／各frame U | 式17のL2、本文のweighted L1/L2族。主対象は測定分散／MSEとqubitizationのλ | 係数正則化、jointの難しさと二段階最適化、number→Z/ZZ写像は既知。Cは固定tail・実atom辞書の写像後norm。原法の全solverや最適強度を再現した比較ではない。 |
| SCDF（L4） | 前段一般V、採用式27はV=W Wᵀ。shift変数α、frame U | tensor fit＋shift後係数のL1。固定Nでのsymmetry shiftとλ／Toffoli削減 | 一般V段階まで含めると近い。採用式のrank-one coreと、制約なしsigned Jは異なる。Cはshiftを導入せず元κを固定。変換族全体の包含は主張しない。 |
| UWC（L2） | Pauli-LCUにglobal OO／粒子数・spin BLISS | soft deterministic count＋soft randomized-tail mass²。PR/RPE資源を直接目的化 | PR向けtail集中と資源接続の概念的重複が強い。Cはframe別J/Uを近似fitし、元順LD2を固定。UWCのglobal OO／block保存shiftや可変soft境界と異なる。 |
| PR/RTE（L3） | 一般LCU、factorized Pauli＋Gaussian／DF frame | 重要headを決定論、tailをRTEでランダム化。軌道／normとRPE費用も既知 | PR＋DF、λ削減、basis-change costを新発見としない。Cの近接ENTRY対照・候補固有の有限native会計が追加の検証対象。 |
| FRO/GFRO（L7） | 一般係数行列とframe回転をjoint変数にするfermionic fragments | 残差tensorのL1などを用い、PF誤差／gate負担を比較。PR-tail目的ではない | 「L1」の対象が残差tensor、J entry、写像後atomで異なる。一般係数・U joint最適化だけをCの差分にしない。 |
| 研究C | 実対称signed J、frame別実直交U、固定number CDF8・元順LD2 | normalized tensor residual＋smooth mapped tail mass²。物理sector予算は独立判定 | 真の総資源をoptimizer内で直接最小化していない。sampling lawの形式・split・rankを共同最適化した手法ではない。 |

### 物理精度、古典負担、量子指標と比較可能性

| 対象 | 精度・保証と古典負担 | 評価モデル・条件／比較してはいけないもの |
|---|---|---|
| CDF／RC-DF | 近似tensor fit、測定bias/variance、FTではenergy error heuristic等。RC-DFのL2係数solve／CGとouter回転更新。global optimum保証ではない | para-benzyne/naphthalene等の測定、Cpd IのFT λ等。測定shotsとRPE shots、qubitization λとCのtail λは別。 |
| SCDF | approximate fitと固定N scalar shift補償を分離。W→α median→Uのnested最適化 | λとFT Toffoli。CのH4 native RZと単一ランキングにしない。 |
| UWC | OOはspectrum、BLISSは対象blockを保存。soft surrogateの反復最適化・JAX/SLSQP。数値資源観測と理論式を区別 | 鉄硫黄／P450／Ru触媒等のactive-space、Pauli rotationとSTAR資源。分子・basis・誤差予算・失敗率・architectureがCと一致しない。 |
| PR/RTE | LCU/RTEの解析と所定仮定下の誤差・sampling関係。数値軌道最適化／gate評価は別層 | 論文内のPauli／DF・分子／模型ごとの条件。一般定理はCの浮動小数点native実装・全law合成・actual CAの保証を自動供給しない。 |
| 研究C | ffsimのtensor loss／ADを再利用。近接比較はrecentered U＋Newton-CG J/refinement。共通物理sector norm＋engineering guard≤CA/8、J一次MET／U未達 | 直鎖H4、1.00 Å、STO-3G、CDF8/LD2、13round、P0・fenced serial。条件付きRZ／CX／depth。native有限標本と名目的区間、全law未資格を分ける。 |

rankの「frame数」とsecond-factor rank、λの辞書、energy toleranceと失敗確率を揃えず論文間の優劣を付けない。外部文献の資源数値をH4へ移したり、独自ENTRYを「最強RC-DF」と呼んだりしない。

目的の関係を明示すると、UWCの式35は $G_{soft}=g_D\sum_\ell s_\ell+g_R[\sum_\ell|c'_\ell|(1-s_\ell)]^2$（$s_\ell$はsoft境界のsigmoid）。Cの固定frame maskとは異なり、soft head個数も目的に入る。RC-DF本文はentry係数のweighted L1/L2正則化を、SCDF採用式27は $f(U,WW^T)+\rho\sum_{tkl}|W^t_kW^t_l-\alpha_t|$ を扱う。したがって、tail²という形、signed係数、軌道更新をそれぞれ独立新規性として数えない。ここでの簡略記号は規約を統一した比較表示であり、各論文のλや係数の数値同一性を意味しない。

L6の低rank基盤はtruncation誤差を伴い、Cの新しい最適化目的そのものではない。L7はSTO-3GのH2/LiH/BeH2/H2O/NH3、一次PF・Adaptive QPE・1e-3 HaのT-count上界推定を扱う。L3のH-chain（STO-6G・隣接1.4 Bohr等）の条件や、L1/L4のCCSD(T)精度基準をCへ移植しない。L8のchannel定理をRTEの複素信号保証に、L9の無限lawの不偏性を有限cutoffへ自動適用しない。L10の固定law reweightingはshotsと価格の一般的tradeoffの先行例であり、今回は新samplerの提案へ範囲を広げない。

## 4. 固定SOURCEが定義する目的と、既知写像との差分

以下は[初期設計](research_c_pr_aware_df_optimization_design_20261026.md)、[既存の近接目的整理][NEAR-DESIGN]と、科学SOURCE `70a98adfb701a209db2c65ea60fb789ff62b1fe7` の [`c_pr_aware_df.py`][PR-SOURCE]／[`c_near_objective.py`][ENTRY-SOURCE]の静的照合である。新しい導出・数値実験ではない。

$$
g^\theta_{pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},\quad
f(\theta)=\tfrac12\|g-g^\theta\|_F^2,\quad
F_{PR}=f/f_*+\gamma(M_\epsilon/M_*)^2.
$$

number fragmentは $D_t=\tfrac12\sum_{ij\sigma\tau}J^t_{ij}n_{i\sigma}n_{j\tau}$。spin mode $a=2i+\sigma$ に対し、既存general-K規約は

$$
K_{ab}=J_{a//2,b//2}/2,\quad
c_{Z_a}=-\tfrac12\sum_bK_{ab},\quad
c_{Z_aZ_b}=K_{ab}/2\;(a<b),\quad
c_I=(\sum_{ab}K_{ab}+\mathrm{tr}K)/4.
$$

$$
\lambda_R=\sum_{t\ge L_D,s\ne I}|(TJ)_{ts}|,\qquad
M_\epsilon=\sum_{t\ge L_D,s\ne I}\sqrt{(TJ)_{ts}^2+\epsilon^2}.
$$

元のgから作るκを固定し、核scalarは一回、各fragmentのidentityは正確な決定論位相に置く。identity値そのものをJ更新中に固定するのではない。異なるUのatomを同一Pauliとして無条件に相殺しない。固定J・固定maskでpenaltyはUに直接依存せず、Uはtensor fitとJ再最適化、PF非可換性、Gaussian価格を通じて効く。

L1 Appendix F式48–60にもnumber→Pauli、signed行和、および行和寄与をone-body Fへまとめる操作がある。**写像や符号相殺そのものがC固有なのではない。** Cでは元κとframe別Zを保持した辞書のtailだけを罰するため、L1の一体化後LCUやBurg型block encodingのλとは目的が異なる。L1の確認した目的式にこの固定tail $\|TJ\|_1^2$ を見いださなかった、という限定的な確認であり、文献全体の不存在主張ではない。

### ENTRYとの関係と同値になる領域

既存の比較設計で示された一frameの式は

$$
\lambda_t=\sum_i|\sum_jJ^t_{ij}|+\tfrac14\sum_i|J^t_{ii}|+\sum_{i<j}|J^t_{ij}|,
$$
$$
B_0=\sum_{t\ge L_D}\left[\tfrac54\sum_i|J^t_{ii}|+3\sum_{i<j}|J^t_{ij}|\right],\quad
B_0-\lambda_R=\sum_{t\ge L_D,i}\left(\sum_j|J^t_{ij}|-|\sum_jJ^t_{ij}|\right)\ge0.
$$

ENTRYのordered-J重みは対角5/4、非対角3/2（対称対を二度数える）。smooth continuation、自己初期massでの固定scale、penalty全体の二乗を用いる。零点floorの一致はHessianや最適化難度の一致を意味しない。

- 行内相殺がない符号領域では、ε=0で二つのmass二乗は同値になる。どこでも別の原理が働くとは言えない。
- 一般signed Jでは $|\sum_jJ_{ij}|$ と $\sum_j|J_{ij}|$ は異なる。固定entry重みだけで全域の写像後normを表せないという既存の数学的差分がある。
- それでも、線形写像後normの正則化という広い既知classにCは含まれる。数学的非同値だけで学術的新規性や実用上の追加利益を認定しない。

固定近接比較はγ=β=.1の初期寄与一致という一政策である。自己目的Fの大小をarm間の順位にしない。J/U同時収束、公平な強度retuning、重なる物理誤差帯・計算予算での比較は、保存された有限手順結果と分ける。

## 5. 証拠系列を固定して引用する

E1–E5に共通する基本系は**直鎖H4、隣接1.00 Å、STO-3G、通常number CDF8、Nα=Nβ=2・sector36**。元g補正κ、元frame順、own-DF ground oracle、状態準備費用除外、CA=.00159 Ha／DF reserve=.00019875 Haを用いる。PRはLD2、partial ordered S2、K0、δ=h=.1の単一時間条件、m0..12。s/r、candidate、cohortは下表ごとに異なる。これはdelta掃引やH6/H12への一般化ではない。

RPE政策はβtotal=π/6、βstat=π/12、candidate failure=.01、axis=.01/26。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation1、initial-zero false、MPFR160 declared phase。既存の合法basis共有・相対Gaussian融合とfenced serial会計を使う。全長回路を再compileしたhardware実時間の比較ではない。

### 5.1 SOURCE／candidate台帳

| ID | 閲覧する固定結果commit・報告 | 科学SOURCE・candidate／条件 |
|---|---|---|
| E1 | **`beb8de5cd8b71f80bb69556f92299435ff431c3e`** [PR対ENTRY r32][E1]。前段fit結果は`2ceb0a8825785ebd4f21241d8ef0f20cb321bf69` [有限16更新][E1-FIT] | 下流SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、fit SOURCE `70a98adfb701a209db2c65ea60fb789ff62b1fe7`。PR endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、ENTRY `345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f`。s1/r32、新native各32。 |
| E2 | **`aced1647b81e8d28091058cdc7b6e3e1ee54896a`** [PR統合実行][E2] | SOURCE `980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。selected **update005** `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。s1/r128・新native32。旧candidateからのPR内部改善。 |
| E3 | **`38b8d0df39b8943178ecef611194819cff1c2169`** [同J/U強D][E3] | SOURCE `5f27539ae0ab88181d9cfcd45badb54daf855c28`。E2の同update005。全8frame＋κのordered S2/s1/h.1、PR/r128の保存値と比較。 |
| E4 | `3718d369b3db515b2dd13b2cdbb6c49d57880121` [update064/r32独立価格][E4]。候補固有下流は`cca6b487c8e1ed3362f8825e95c937ebbcc87755` [下流報告][E4-DOWN] | 価格SOURCE `9837eac59b94113bf057fcc311a27b35e60343f5`、下流SOURCE `928fda0a6df260b66eba1bbea63b51f2734e7b54`。**update064** `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3`。s1/r32、新独立native256。 |
| E5 | `459c8472d66689bf149a60fd6cc7311ff5237c65` [update005/r32独立価格][E5] | 価格SOURCE `62fa4c0144b9e36f9d96b7dff67628051946632d`。E2と同update005、s1/r32、新独立native640。E2のr128価格やE4の256標本とpoolしない。 |

元h/g/scalar packetは `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。各結果の`SOURCE.json`、入力binding、契約、`result.json`、`STOP.json`を一組で読む。名称に`unapproved`が残る旧契約のファイル名を改名したり、過去の状態を今回の依頼で上書きしたりしていない。

| 保存JSON（上表の各結果commit内） | 引用する既存欄 |
|---|---|
| E1 `artifacts/research_c_near_practical_r32/2026-11-03/run01/result.json` | `arms/{PR,ENTRY}/PR/lanes/ideal_exact_native`、`PR/calibration`、`bindings`、`D/saved_DF`、`comparison/primary`。前段のgapはE1-FIT本文と同系列の保存result。 |
| E2 `artifacts/research_c_pr_integrated/2026-11-02/run01/result.json` | `selected/physical`、`selected/state`、`performance/lanes/ideal_exact_native`、`performance/calibration`。 |
| E3 `artifacts/research_c_same_ju_strong_d/2026-11-02/run01/result.json` | `selected_export`、`saved_DF`、`numerical/cost`、`comparisons`。 |
| E4 `artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/primary.json` | `kernel_mean/SE`、`total_point`、`ratio_point/interval`、`decision`、`precision_met`、資格のnull/false。 |
| E5 `artifacts/research_c_r32_price_statistics/2026-11-03/run01/primary.json` | E4と同じ主統計欄。ただしcandidate・精度政策・分母を共有しない。 |

### 5.2 E1：PR対ENTRYの有限観測

| 保存指標 | PR | ENTRY |
|---|---:|---:|
| Exact tail λ / Ha | 0.500044243591 | 0.941312847393 |
| Exact entry envelope B0 / Ha | 1.5591348809 | 1.23977058606 |
| Row-cancellation gap / Ha | 1.05909063731 | 0.298457738667 |
| DF sector norm / Ha | 4.84367101908e-9 | 1.11037670289e-8 |
| engineering guard / Ha | 3.2165809758e-9 | 3.31972010087e-9 |
| RPE shots（X＋Y、ideal） | 14,738 | 19,612 |
| W（反復・shotsを含むkernel回数） | 12,539,808 | 28,556,976 |
| kernel mean RZ（N32） | 7,519.09375 | 6,681.25 |
| kernel mean SE | 107.321883191 | 115.509565307 |
| shell込み条件付き総RZ | 94,288,043,542 | 190,796,364,542 |

前段表と下流表の保存値をそのまま引用した。両armはJ一次MET、U一次NOT_MET、16受理U更新上限停止。両者が同じDF予算を通ることと、到達した物理誤差が同じであることは別である。

保存された主比較はPR/ENTRY点比 **0.494181551983**、**50.5818448017%低い点値**、PR−ENTRYの近似Welch95%差区間 **[−103702867354, −89313774645.6]**、`NOMINAL_MODEL_PR_LOWER`。比の区間を今回新たに作っていない。双方のPF/RTE・J/U・価格が異なるため、差の全てをλや符号相殺だけへ因果帰属できない。

ideal laneはexact native synthesisとideal finite-RTE IID信号を仮定し、有限bit実装の価格proxyを接続する。evidence laneは全law synthesis／implemented-law signal誤差未資格のためshots・総費用がnull。`CONDITIONAL_CA_PASS`をactual CAや厳密保証へ読み替えない。

### 5.3 同J/U比較の肯定・否定・未解決を並べる

| 保存条件 | 保存された主RZ結果 | 読み方 |
|---|---|---|
| E2→E3、update005/r128、native32 | PR 207,919,150,188、同J/U D 105,532,782,860、比1.97018542062 | PR内部は旧PR比約11.19%改善したが、主RZでは同J/U強Dより高費用。CX depthの順位は別。 |
| E5、update005/r32、独立native640 | PR/D .999580138349438、nominal95% [.9937486698850807, 1.0054116068137955]、`UNRESOLVED_PRECISION_TARGET` | 1を含み、登録0.5%半幅目標も未達。優位・同等・非劣性の証拠へ格上げしない。 |
| E4、update064/r32、独立native256 | PR 100,330,684,089.25、同J/U D 105,038,112,319、比 .955183617395431、nominal95% [.9465946344829994, .9637726003078625] | 約4.48164%の低い点値、登録1%精度を満たし `NOMINAL_MODEL_PR_LOWER`。固定候補・名目的価格モデル内の支持。 |
| E1のPR endpoint/r32、native32 | 同J/U D 104,383,256,064、PR/D .903287050982 | E4の4.48%とは別candidate・cohort。二つをpoolして一般効果を推定しない。 |
| E1のENTRY endpoint/r32、native32 | 同J/U D 112,333,995,286、PR/D 1.69847394866 | ENTRYでPR/r32が常に有利ではない。保存副比較は精度目標未達であり、主PR対ENTRYとの差区間と同時保証にはしない。 |

E2/E3の旧不利はE4により誤りになったのではない。candidateとrが違う有限結果として両方残す。005対064の区間を並べただけで候補間の統計的優位を検定したことにもならない。

### 5.4 他の負結果・未資格記録

詳細数値の再集計をせず、[10月31日総括](research_c_evidence_synthesis_and_decision_readiness_20261031.md)、[資源寄与整理](research_c_pr_resource_contribution_analysis_20261102.md)、[11月3日Evidence Map](research_c_evidence_map_updated_20261103.md)にある条件付きの記録を再利用する。

- tail-P4はsupportを減らしても価格・総費用を改善しなかった。通常CDF7でも主RZの点改善がなかった。低rank／疎化の一般利益を仮定しない。
- basis/native複合surrogate、第三DFへのswitch移送は支持不足／不成立。未取得の一般設計則を補って書かない。
- 旧Stage A γ=.1 J/U・r128のround混合は全Dを選んだ（[旧報告](research_c_roundwise_det_pr_execution_20261030.md)、candidate `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`）。update005/064の候補とは異なり、064/r32について同じ混合結果を得たとはしない。
- 初期全frame uniform L2/RC2対照、共通停止政策、stable RC2には未収束・終端物理資格不足等の制約がある。これを「十分retune・収束したRC-DFに勝った」という比較へ転用しない。
- 固定F2/F3 trajectoryのvalidated CARや個別native correctnessは、そのtrajectoryの証拠。現在の全law・13round・実CAへの資格移転ではない。

## 6. 学術的claim候補の台帳

位置付けを **文献既知／固定SOURCEの構成／保存有限観測／未検証仮説／追加検証後の新規性候補** に分ける。下表は採択やGO判定を行わない。

| ID・候補 | 内容と現在の位置付け | 直接証拠 | 反証・制限・今後変わり得る部分 |
|---|---|---|---|
| C1 写像後tailを用いるDF目的 | 一般CDF＋正則化は既知。固定tailの $\|TJ\|_1^2$ とentry上界との差はSOURCEで確認。独立新規性は候補 | §4の固定関数、E1のλ/B0/相殺gap | 相殺なしの符号領域では同値。写像自体はRC-DF既知。単一強度・有限未収束政策の差と目的の固有能力は未分離。 |
| C2 物理予算内の下流改善 | 共通DF admissionを通る有限endpointで、PF/RTE→attenuation→shots/W→native→shell込み費用へ接続した観測 | E1のDF/13round/native64・主差区間 | 達成物理誤差は不一致。全law・actual CA・validated coverageは未資格。公平retuningで差が縮小／消失し得る。 |
| C3 shotsとkernel単価の相反 | λやshotsが小さくてもnative単価は増え得るという具体的事例 | E1のkernel上昇とW低下、E2/E3の負結果 | tradeoffの原理はUWC/PR等で既知。機構の因果、普遍的予測則、他系での再現は未確立。 |
| C4 同じDF上でのPR/強Dの条件依存性 | 各candidateのJ/Uを共有して、PRをDより有利にする条件があるかという有限比較 | E3の不利、E5未解決、E4のnominal支持、E1のarm別対照 | r・candidate・価格政策が異なる。一般PR優位、最適D/PR間のglobal比較、全資源指標支配ではない。 |
| C5 頑健性・公平retuning・未使用geometryへの追加利益 | **未検証仮説**。A〜Dの結果を待つ新規性候補 | 今回は結果なし | 初期値や正則化に不安定、ENTRYの再調整で差消失、1.15 Åで非移送なら一般的claimは弱まる。結果を先取りしない。 |

科学的な価値として提出できるのは、特定の写像／辞書／精度・費用条件に結び付いた再現用SOURCEと有限観測、ならびにproxyだけでは費用を決められない肯定・負結果の対応である。「価値の候補」と「独立した新アルゴリズムとして十分」の判断を分ける。

## 7. 新規性判断の残論点とA〜Dへの接続

1. **文献上の残論点**：RC-DFの広い正則化class、PR原論文のfactorized実装、UWCの費用目的に対して、固定tail辞書への適用がどの程度非自明か。今回確認外の実装・関連節に同じtail目的がないとは断定できない。原法との公平な直接比較もない。
2. **方法差の追加利益**：ENTRYとの差が複数初期値・強度・有限予算で残るか。共通物理予算だけでなく、到達物理誤差帯と実計算負担を示し、目的能力と有限recipe性能を分ける。
3. **下流での意味**：λの順位がPF/RTEとnative価格を含めても維持されるか。ENTRYや強Dへも同等の下流retuningを与えた後に、どの条件付き資源claimが残るか。
4. **移送**：未使用H4 1.15 Åでの候補固有結果が必要。同geometryの成功でもbasis／分子／rank／LD一般の検証にはならない。
5. **保証・重要性**：名目的価格区間と実装law/actual CAの不足を、論文でどの強さのclaimに対応させるか。厳密保証完成を無条件の主経路にしないが、保証がないものを保証付きと書かない。

別冊§6に、A〜Dの各結果がC1–C5を支持／弱める条件と必要データを示す。現在のA〜D SOURCE・契約を改訂する提案や新実験の登録ではない。正確なA〜D成果物名・実行状態・SOURCEは担当の固定公開報告取得後に補う。

## 8. 再利用・検証・公開範囲

baseは`546bca7ed2cece59b714b986a264d7b0b0871a01`、独立branchは`codex/research-c-related-work-paper-readiness-20261105`。依頼元のdirty checkoutとA〜D worktreeへ書き込まず、本書・別冊と入口リンクだけを変更した。

再利用したものは、初期PR-aware設計、近接目的設計と既存数理、E1–E5のSOURCE／結果、10月31日総括、11月2日寄与整理、11月3日Evidence Map、既存目的関数・general-K・reduced/native価格会計の静的SOURCE、および§2の一次論文関連節である。optimizerや解析基盤の新実装はない。実行済み研究testの再実行もしていない。

今回の技術確認は、文書リンクと固定Git path、指定commit・treeとscientific SOURCEの区別、候補identity、保存JSON中の引用値、差分が文書のみであることに限る。旧raw全hashや依存全体の再監査は行わない。公開後はremote branch SHA、commit tree、公開された両本文とlocal bytesを照合し、その結果を最終チャットで報告する。

公開前の実施結果：2冊の相対リンク・参照定義・UTF-8・数式区切り、固定GitHub参照15 path、E1/E4/E5の主要保存scalarと資格の転記、必須3commitのpublic GitHub treeとlocal treeの一致を確認した。`git diff --check`はPASS。研究testやmanifest全体checkerのPASSを意味しない。

**公開照合後MANDATORY STOP。** 新DF fit／PF/RTE／RPE／native、保存値の新規科学再解析、A〜D条件変更、正式新規性・論文化GO認定、重要GPTレビュー、投稿判断へ自動進行しない。

## 固定参照

[NEAR-DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2459e018ea2276c359d6fe96253fbcad0bfd65bb/docs/research/research_c_uwc_rcdf_near_objective_comparison_design_20261103.md
[PR-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/70a98adfb701a209db2c65ea60fb789ff62b1fe7/src/trottertracks/c_pr_aware_df.py
[ENTRY-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/70a98adfb701a209db2c65ea60fb789ff62b1fe7/src/trottertracks/c_near_objective.py
[E1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md
[E1-FIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_near_objective_practical_comparison_execution_20261103.md
[E2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aced1647b81e8d28091058cdc7b6e3e1ee54896a/docs/research/research_c_pr_integrated_execution_20261102.md
[E3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[E4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md
[E4-DOWN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_execution_20261103.md
[E5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/docs/research/research_c_r32_price_statistics_execution_20261103.md
