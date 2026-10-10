# PRを含むHamiltonian simulationの新規研究方針レビュー
## Hamiltonian表現・分解・部分空間構造と乱択時間発展の協調設計

- 作成日：2026-10-10（Asia/Tokyo）
- 区分：ユーザーの明示的依頼に基づく科学的研究方針レビュー
- 対象：外部アルゴリズム単独の改善、および外部とPR内部の一体設計
- 対象外：固定入力に対するPR内部だけの改良、Track Bの既存結果の再審査
- 状態：初期方針レビュー完了。研究課題の最終採択、新規性の確定、数値的優位性の確認は未完了
- 実施内容：GitHub上の規約・対応sourceの取得、一次文献調査、数学的検討、候補比較
- 実施していない内容：研究コードの変更、数値実験、既存実験の再実行、大規模計算へのGO

---

## 0. 結論

現段階で優先して比較する価値が高いのは、次の二方向である。

**候補A：決定論部分と乱択残差を、既成の分解から選ぶのではなく、分解そのものと同時に構成する。**

具体的な入口は、Hamiltonianを厳密に保存する因子空間の回転と処理配分の同時設計である。そこから、共通軌道基底・安価な基底間遷移を持つ決定論部分と、実際に安価に乱択できる残差を共同生成する方向へ拡張できる。ただし、DF/PRの接続、係数ノルムの最適化、残差の乱択処理という大枠は既知である。新しい構成法、構造条件、または資源上の関係を提示する必要がある。

**候補B：補助空間を用いる圧縮表現について、漏れを時間刻みの縮小だけで抑えるのではなく、生成子の段階で除去し、その表現をRTEで扱う。**

これはGRADEのような拡大軌道空間を使う表現を動機とする。既知の物理部分空間の反射を用いれば、与えられたinvolution展開の係数1ノルムを増やさず、物理部分空間を保存するHamiltonianを構成できる。さらに、正しく構成した有限RTEの補正平均演算子は、その部分空間を厳密に保存する。この代数的事実は以下で導く。ただし、単一軌道の漏れが消えることでも、総資源が必ず下がることでもない。群平均とLCUは既知の道具であり、この短い代数だけでは独立した新規論文の主要貢献として不十分である。

候補Aは既存DF実装を利用しやすい一方、新規性が単なる目的関数変更に縮退しやすい。候補Bは狙う問題が明瞭で、外部表現とPRの接続に意味がある一方、補助qubit、反射操作、乱択側へ移す重量の費用が利益を消す可能性がある。

第三の対抗案は、近可積分法が必要とする混合propagatorを安価に実装できるHamiltonian分解の構成である。THRIFTやSPRINTに新しい乱択係数を付ける研究ではなく、それらが仮定するアクセス構造を量子化学で実現する外側の構成を対象とする。

現時点で勧めないのは、一般的な「軌道を回す」「DF rankを掃引する」「BLISSの係数を最適化する」「groupingを変える」を、そのまま中心的な新規性とする進め方である。これらは有用な部品や比較対象にはなるが、既知研究との差分を別に用意しなければならない。

---

## 1. 研究範囲の再確認

この研究の目的は、PR外部・内部の区分そのものを維持することではない。量子化学Hamiltonian simulationの資源を減らす、科学的に意味のあるアルゴリズムを考えることである。

外部のみの改良も、外部と内部の一体設計も採択候補とする。一体設計であること自体を加点しない。一方、同じHamiltonian、同じ分解、同じ有限演算子を与えたまま、Taylor係数、return、sampling lawだけを改善する研究はTrack Bに属し、このレビューでは主題にしない。

分子、水素鎖、基底、mapping、PF次数、資源指標、計算機環境は未固定である。以下に登場する実数因子、sum-of-squares、補助真空部分空間などは、それぞれの構成の適用条件であり、研究全体に追加した条件ではない。

また、位相推定・エネルギー推定を主な例に使うが、完全な状態時間発展と同じ課題とは扱わない。特にRTEの平均振幅による推定の保証を、単一回路の出力状態の保証に置き換えない。

## 2. 取得資料とレビューの根拠

### 2.1 GitHub資料の取得確認

Repository：`HIROMU1015/Partially-Randomized-Trotter`

基準sourceとして取得したmainのcommit：

`0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`

取得した主要資料：

| パス | このレビューでの用途 |
|---|---|
| `README.md` | 現行規約への入口と旧結果の利用上の注意 |
| `docs/rte_conventions.md` | 有限RTE、involution、正規化、信号、controlled circuitの意味論 |
| `docs/rte_source_versions.md` | 原論文v1/v2と実装規約の版管理 |
| `src/trotterlib/README.md` | DF、RTE、compiled cost、RPEの実装位置 |
| `src/trotterlib/df_partial_s2.py`（先頭の対応source） | 決定論DF block、tail、identity、rank proxyと実際のlambdaの区別 |

加えて、`track-b-g10-final-review-intake-20261010`の`docs/research/README.md`を、研究の重複を避けるための文書索引として参照した。このbranchの全source・実験・監査を独立検証したわけではない。また、mainを現在のすべての研究branchの最新状態と見なしていない。

このレビューは新テーマの設計であり、既存の数値的優位性を根拠に候補を採択していない。そのため、過去の全raw結果や全実行監査は不可欠な資料ではない。取得した規約と対応sourceは、既存PRとの意味論上の接続を検討するには利用可能だった。

READMEにある旧screening失効・旧UWC raw artifact欠落の注意は尊重し、該当する旧数値は候補の有効性の根拠として使用していない。この注意を、後続branchの別実験の無効性に一般化していない。

### 2.2 PR原論文の版に関する制限

リポジトリは実装の一次資料をarXiv:2503.05647v2（2026-07-10）とし、v2検証PDFは一時ファイルでリポジトリには保存していないと記録している。今回のWeb取得では、versionを指定しないPDF URLがv1表記の本文を返した。v2指定本文の取得は成功しなかった。

したがって、原論文の一般的な研究範囲は取得できた本文と公開抄録で確認し、現行実装の有限RTE意味論は上記のcommit固定規約で確認した。v2の全式を独立に再監査したとは主張しない。以下の代数的検討は規約に定義された有限RTEの恒等式から直接導いている。

この制限は新規研究候補の検討を不可能にするものではないが、原論文との差分を最終確定する段階では版をそろえた比較が必要である。

### 2.3 文献調査の範囲と限界

2026-10-10時点で取得できた一次資料を調査した。2026年のpreprintも含む。本文の該当部分を読めた資料と、抄録・掲載情報までの資料を参考文献欄で区別する。

検索で同じ題名・構成を見つけられなかったことを、新規性の証明とはしない。本記録の優先順位は、確認できた先行研究と数学的な機構に基づく判断であり、網羅的な特許・全論文調査に基づく「未発表」の認定ではない。

---

## 3. 先行研究から見た探索地図

### 3.1 特に近い先行研究

| 領域 | 確認した既存到達点 | 新研究で追加して問う必要があること |
|---|---|---|
| PR | 決定論部分と乱択tail、軌道・表現による重量削減、DFとの接続を既に検討 [R1] | 既成分解からの選択を超える新しい分解生成、または異なる物理的障害の解消 |
| CDF / RC-DF | factor fitting・圧縮・正則化と量子資源削減を既に検討 [R3,R4] | rankや一般のlambdaではなく、処理分担・誤差・primitive costを結ぶ構成上の差分 |
| 対称性とfactorization | SCDF、およびBLISSとtensor factorizationの同時最適化が既存 [R5,R6,R23] | 「両者を同時最適化する」以外の新しい自由度・保証・アルゴリズム |
| 軌道とPF誤差 | 軌道選択、回転、時間step間の基底変更を研究済み [R7] | 共役変換だけではない分解変更の機構、事前情報だけで使える構成法 |
| near-integrable PF | THRIFT、SPRINT、近可積分性を利用した乱択PFが存在 [R2,R8,R9] | 有利なアクセス構造を分子Hamiltonianから構成する方法 |
| 拡大軌道factorization | GRADEでstep当たりの圧縮と補助空間leakageの問題を検討 [R2] | 漏れを含む総costの障害を解消する構成・条件 |
| symmetry protection | step間の対称操作による誤差相殺、Zenoとの関係が既存 [R10] | 生成子の構成と有限RTE平均の保存性を使う場合の明確な差分 |
| grouping / compilation | groupingとgate cancellationの同時考慮、TSP型順序設計が既存 [R13] | 固定項の並べ替えではなく、安価な遷移を持つ項そのものの生成 |
| sparsification | SparSto、SQuISH、PSD局所Hamiltonianの近年のsparsification理論 [R15–R17] | 対象エネルギー精度、負係数・非局所mapping、補正費用を含む設計 |
| downfolding / TC | 軌道数削減と有効Hamiltonian、非Hermitian固有値推定の研究 [R18,R19] | 追加の高体相互作用・非正規性・モデル誤差まで含めた利益 |
| 誤差推定 | BCHに基づく実用的PF誤差推定が研究されている [R20] | 指標を導入するだけでなく、新しい構成法・保証に結び付ける |
| 低エネルギー構造 | SOSSAとDFTHCを含むspectral amplification系が発展 [R21,R22] | 単なるSOS・DF/THC融合という主張を避け、アクセスモデル差を明確化 |

### 3.2 SPRINT/GRADEをどう扱うか

SPRINTは乱択順序、近可積分性、processing、対称性保護、factorizationと回路資源を横断する近接研究である。ただし、そこでのrandomizationを、既存PRのRTE tail samplingと同一視しない。

GRADEについて重要なのは、1 stepの費用だけで優劣を決められない点である。著者らは補助軌道へのleakageが総費用を支配し得ると報告し、最終SPRINT比較ではCDFを採用している。従って「GRADEがあるからDF改良は解決済み」でも「GRADEをPRへ接続すれば必ず改善」でもない。[R2]

本レビューではこの観測を、未検証の数値優位性の根拠ではなく、候補Bが解こうとする具体的な障害を定める根拠として使う。

---

## 4. 研究を比較する共通の枠組み

### 4.1 設計対象

設計変数を

\[
\theta=(\text{物理問題を表す変換・encoding},\ \mathcal D,\ \text{処理配分},\ \text{許容するPR構成})
\]

と書く。分解\(\mathcal D\)には、因子、軌道frame、block、順序、正確に保持する残差の表現を含む。

位相・エネルギー推定を例にすると、量子資源gの期待総量は

\[
C_{\rm total}^{(g)}(\theta)
=\sum_m N_m(\theta)\,
\mathbb E_\omega\bigl[C_g(\mathcal C_{m,\omega}(\theta))\bigr].
\]

ここでmは推定round、N_mは必要測定回数、omegaは乱択回路である。C_gは必要なら状態準備、制御化、基底変換、合成を含む1 shot全回路の費用とする。

これは評価の枠組みであり、RZだけを最適化する契約ではない。総T/Toffoli、RZ、depth、logical qubit、許容並列性などは研究対象に合わせて選ぶ。古典前処理の秒数をgate countへ無断で足さず、別軸で報告する。

### 4.2 RTEで区別すべき量

既存の規約では

\[
H_R=\lambda_R\sum_jp_jP_j,\quad P_j^\dagger=P_j,\quad P_j^2=I,
\quad \lambda_R=\sum_j|h_j|.
\]

DF fragment全体は、一般にこのP_jではない。回転したI/Z/ZZなどの具体的involutionに展開して初めて既存RTEへの入力となる。[G2]

1 stepのtail時間をhとし、\(\tau=\lambda_R h\)、偶数cutoff Kについて、規約の有限RTEは

\[
B_K(\tau)\,\mathbb E[U]
=T_{K+1}(-ihH_R),\qquad
T_{K+1}(X)=\sum_{j=0}^{K+1}\frac{X^j}{j!}
\]

を満たす。[G2]

複数occurrenceでは正規化の積\(\Gamma=\prod_i B_{K_i}^{r_i}\)が現れる。信号の減衰は、有限Taylor打切りのbiasとは異なる。固定した信号精度を通常の有界分散sample meanで達成する状況では、正規化補正による分散増大は概ねGammaの二乗で現れるが、これをすべてのRPE roundの厳密shot公式とはしない。状態重なり、信号半径、推定器、失敗確率配分も必要である。

同じ正確なH_R、同じh、Kなら、補正平均のTaylor多項式自体は表現によらない。表現を変えてlambdaが変わる効果と、処理配分によってH_Rそのものが変わる効果を分離する。

### 4.3 誤差の種類

有限次元Hermitian問題での基礎的な比較として、

\[
|E_k(H)-E_k(\widetilde H)|\leq\|H-\widetilde H\|,
\qquad
\|e^{-itH}-e^{-it\widetilde H}\|\leq|t|\|H-\widetilde H\|
\]

を使える。これはモデル誤差の保守的評価であり、対象状態のエネルギー誤差の精密な予測とは別である。

Hamiltonian近似、PF、有限RTE、合成、測定・推定の不確かさは別に記録する。演算子normの誤差とHa単位の誤差を、そのまま足し合わせない。物理sectorだけで保証する場合は、そのsectorが保たれる条件も明示する。

### 4.4 平均振幅と平均channelの違い

\[
\mathbb E[U_\omega]
\quad\text{と}\quad
\mathbb E[U_\omega\rho U_\omega^\dagger]
\]

は別の対象である。Uと-Uは同じ非制御channelを与えるが、制御干渉で測る振幅の符号は逆になる。このため、非制御のrandomized channelに関する誤差保証を、そのままPRのHadamard信号保証として使ってはいけない。

候補Bの保存性は補正平均演算子についてのものであり、各sample trajectoryが物理部分空間に留まるという主張ではない。

---

## 5. 探索前に除くべき「実は変わらない」自由度

以下は比較のための基礎的な代数整理であり、新規定理と主張するものではない。

### 5.1 全fragmentの共通共役変換

\(H=\sum_lH_l\)とし、全項を同じVで

\[
H_l'=V^\dagger H_lV
\]

へ移す。分解の対応、係数、順序を変えずに作る積公式は

\[
S'(t)=V^\dagger S(t)V.
\]

従って、unitarily invariant normの誤差と対応する固有位相誤差は不変である。交換子も

\[
[H_i',H_j']=V^\dagger[H_i,H_j]V
\]

なので、同じnormなら小さくならない。

これは「軌道最適化は無効」という結論ではない。Pauli再展開、grouping、truncation、factorizationのやり直し、hardware依存コンパイル、状態準備・境界回路を変えると、前提が崩れ、利益が生じ得る。[R7]との見かけ上の矛盾は、変換後に何を分解単位とするかを区別すれば避けられる。

特に、native DFのすべての因子を共変に移すだけの実験で「commutator削減」を期待することは避ける。

### 5.2 sector上で各blockまで不変ならPFも不変

Pを物理部分空間の射影とし、元と変更後の各blockがPを保ち、さらに

\[
H_l'P=H_lP
\]

を満たすなら、各指数演算子とその積の作用もP上で同一である。全Hamiltonianのsector保存だけでなく、各blockの作用まで変わらない場合の結論である。

従ってBLISS等で対象sector外を書き換えたという事実だけから、sector内PF biasが小さくなるとは言えない。lambda・表現コスト改善の可能性と、PF bias改善の可能性は別である。

### 5.3 射影したHamiltonianの一致だけでは時間発展は一致しない

\(P\widetilde HP=H_{\rm phys}\)であっても、\(Q\widetilde HP\neq0\)、\(Q=I-P\)なら、一般に

\[
Pe^{-it\widetilde H}P\neq e^{-itH_{\rm phys}}P.
\]

物理空間から出て戻る過程も物理振幅を変える。さらに、全Hamiltonianではleakageが消えても、各Trotter blockがPを保たなければ、積公式で新しい漏れが起こり得る。この二つを区別する。

### 5.4 小さなresidual normと小さなRTE係数normは別

R=H-H_coreが小さなoperator normを持っても、その表現を巨大な二つの和の差として作れば、係数1ノルムが大きくなることがある。基底が異なるDF因子の差で特に注意する。

したがって、候補Aでは「良い近似を作る」だけでなく「残差の相殺を反映した、実装可能なinvolution表現を作る」ことが必須である。

---

## 6. 候補A：分解生成と決定論・乱択処理の協調設計

### 6.1 中心研究課題

> 与えられた分解の大きな因子を選ぶだけでなく、同じ物理Hamiltonianを表す分解から、実装費用・残差の乱択費用・PF誤差のバランスが良い決定論部分と乱択部分を構成できるか。

「H_DとH_Rを最適化する」という抽象的な目的だけは既知である。[R1]との違いは、固定辞書の選択ではなく、辞書・factor・frame自体を生成する点に求める。ただし、一般の因子最適化も既知[R3,R4,R12]なので、実際の構成と保証を比較する必要がある。

### 6.2 厳密にHamiltonianを保存する入口：因子空間の回転

適切なone-body補正と定数をH_1に含めて、対象の一部を

\[
H=H_1+\frac12\sum_{\mu=1}^{L}\widehat L_\mu^2,
\qquad
\widehat L_\mu=\sum_{pq}(L_\mu)_{pq}a_p^\dagger a_q
\]

と書ける実Hermitianな正のsum-of-squares表現を考える。正係数はL_muへ吸収する。これは条件付きの例であり、符号付きfactorizationすべてをこの形とみなさない。

実直交行列Oにより

\[
\widetilde L_a=\sum_\mu O_{a\mu}\widehat L_\mu
\]

とすると、

\[
\sum_a\widetilde L_a^2
=\sum_{\mu\nu}\left(\sum_aO_{a\mu}O_{a\nu}\right)
\widehat L_\mu\widehat L_\nu
=\sum_\mu\widehat L_\mu^2.
\]

L_mu同士が非可換でも成立する。これは電子軌道空間の共通回転ではなく、**因子のラベル空間の回転**である。

そのうえで、

\[
H_D=H_1+\tfrac12\sum_{a\in D}\widetilde L_a^2,
\qquad
H_R=\tfrac12\sum_{a\notin D}\widetilde L_a^2
\]

を作る。各因子の固有値、固有vector、tailの具体的なI/Z/ZZ展開、決定論sweepの交換子、基底変更費用は変化し得る。全Hamiltonianを変えずに、PRが見る処理構造を変えられる。

### 6.3 何を最適化すれば新しい内容になるか

候補は、単なるFrobenius normではなく、例えば次の組で評価する。

- 実際のinvolution展開でのlambda_Rと有限RTE正規化
- 決定論部分のnative costと乱択eventの期待compiled cost
- 物理sectorまたは対象信号についてのPF誤差指標
- factor回転・diagonalization・residual構築の古典費用

これらを無根拠な重み付き和にまとめる必要はない。まずPareto構造や、誤差制約下の量子費用として定義する方が明瞭である。

アルゴリズムの候補としては、因子対の直交回転、決定論部分空間の更新、frame費用を考慮した局所構成などが考えられる。ただし、ここでは具体的optimizerを採択していない。全O(L^2)変数を無差別に最適化するだけでは、古典費用と解釈可能性が問題になる。

### 6.4 この入口の明確な限界：rank削減そのものではない

直交混合は独立なfactor数をそのまま減らさない。さらに、

\[
\sum_{a\notin D}\|\widetilde L_a\|_F^2
\]

だけを最小化するなら、元のpair-index行列の固有分解で大きな成分を選ぶ方式は、適切な条件下で既に最適である。Gram行列\(G_{\mu\nu}=\operatorname{Tr}(L_\mu^\dagger L_\nu)\)を使うと、保持する重量はrank-k射影MについてTr(MG)であり、その最大は上位k固有値和になる。

従って、この既存proxyを目的に同じ回転を最適化しても、本質的な上積みがない場合がある。研究の狙いは、proxyを既知より上手に最適化することではなく、**実際のRTE費用・交換子・回路構造に関する目的が、このproxyとどこでずれるかを利用すること**である。

### 6.5 発展形：少数の安価なframeと正確な残差

より広く、

\[
H=\sum_b U_b^\dagger D_b U_b+R
\]

を構成する。D_bをnumber演算子の可換な多項式とするなど、e^{-itD_b}が実装可能なクラスを明示する。

ここでRは自動的に捨てない。近似factorizationを使ってcoreを作っても、Rを正確に保持し、PRで補正する選択肢がある。モデル誤差を消せる代わりに、RTE費用が生じる。

共有frameや安価な相対回転U_b U_c^daggerを持つ構成を作れれば、長い回路で基底変更を節約できる。ただし固定fragmentの順序最適化やsampling済みwordの集約だけでは、既知grouping研究[R13]またはTrack B/既存compiler研究との重複になる。

新しい方向は、訪問頻度も踏まえた遷移費用

\[
\sum_{bc}w_{bc}\,g(U_bU_c^\dagger)
\]

を意識して、fragment自体を構成することである。w_bcは実際のsequence分布に応じた期待遷移回数とし、根拠なく独立samplingの積に置き換えない。

### 6.6 科学的な着地点

強い成果は、PR向けの新しい分解構成アルゴリズムと、その改善条件・計算量・誤差または資源関係である。目的関数を変えて数値的に少し改善しただけなら、独立方法論としての新規性は弱い。

弱い／否定的な結果としては、共役不変性やFrobenius最適性によって改善できない範囲を整理し、改善可能な自由度を特定する成果が考えられる。ただし、上記の短い既知代数だけで論文になるとは考えない。幅広い表現クラスを覆う非自明な分類や限界が必要である。

---

## 7. 候補B：漏れのある圧縮表現を、生成子対称化とRTEで扱う

### 7.1 解く価値のある具体的な問題

補助軌道による表現圧縮では、物理空間への射影が正しくても、拡大Hamiltonianの時間発展が補助空間へ漏れることがある。GRADEでこの問題が総costの障害として報告されている。[R2]

問いたいのは、単に既知の保護回路を追加して改善を測ることではなく、次である。

> 圧縮表現が作る漏れをHamiltonian生成子の段階で除き、その生成子を平均演算子として実現することで、漏れ抑制の時間刻み費用を回避できるか。

以下は、その問いに対する具体的な構成候補である。代数的保存性と量子資源の優位性は分ける。

### 7.2 条件付き構成：物理部分空間へのblock対角化

拡大空間を物理部分空間Pと補空間Q=I-Pに分ける。Pは、例えば「補助軌道が真空」のような**既知のencoding部分空間**であり、未知の基底状態への射影を仮定しない。

正確な入力の場合、物理Hamiltonianは\(H_{\rm phys}=P\widetilde HP\)のP上への制限とする。GRADE等にfactorization誤差がある場合、この等式は近似になり、その誤差は残る。

反射

\[
S=P-Q=2P-I,\qquad S^\dagger=S,\quad S^2=I
\]

を用いて、

\[
\overline H=\tfrac12(\widetilde H+S\widetilde HS)
=P\widetilde HP+Q\widetilde HQ
\]

と定める。従って、

\[
[\overline H,P]=0,
\qquad
Pe^{-it\overline H}P=e^{-itH_{\rm phys}}P.
\]

これはblock非対角成分を消す基本的な群平均である。群平均自体の発明を主張しない。

### 7.3 involution展開の係数1ノルムを増やさない

\[
\widetilde H=\sum_j h_j Q_j,
\qquad Q_j^\dagger=Q_j,\quad Q_j^2=I
\]

という実装可能な展開があるとする。すると

\[
\overline H
=\sum_j\frac{h_j}{2}Q_j
+\sum_j\frac{h_j}{2}SQ_jS.
\]

SQ_jSもHermitian involutionである。展開した辞書の係数1ノルムは、同一項の統合前でも

\[
\sum_j(|h_j|/2+|h_j|/2)=\sum_j|h_j|
\]

となる。重複統合・相殺が可能なら小さくなる場合があるが、必ず小さくなるとは言わない。

従って、**この操作そのものは、与えられた辞書に対するlambdaを増やさない**。これは、元の圧縮表現のlambdaがCDFより小さいという主張とは全く別である。

### 7.4 有限RTEへの接続と保存性

新しい辞書の各generatorを独立に抽出し、既存の有限RTEを正しく構成すれば、

\[
B_K(\lambda h)\,\mathbb E[U]
=T_{K+1}(-ih\overline H).
\]

右辺はPと可換なので、

\[
Q\,T_{K+1}(-ih\overline H)P=0.
\]

したがって、**有限Taylor次数でも、補正平均演算子には物理部分空間から補空間への成分がない**。ただし、そのP内の時間発展にはTaylor打切り誤差が残る。実装合成誤差も別である。

各tail occurrenceがこの保存性を持ち、決定論側の各blockもPを保つなら、部分ランダム化全体の補正平均もPを保つ。各stageの乱択は、その平均積の意味論が成立するように生成する必要がある。

同じlambda,h,Kに対して、B_Kは元の辞書のときと同じである。ただし公平な比較対象は、無効な漏れあり時間発展ではなく、正しい物理信号を得るCDF、既存保護法、または他の妥当な方法である。

### 7.5 実装で間違えやすい点：step全体を一度だけtwirlしては駄目

一般に

\[
\tfrac12(e^{-it\widetilde H}+Se^{-it\widetilde H}S)
\neq e^{-it\overline H}.
\]

例えば、P=|0><0|、S=Z、\(\widetilde H=X\)なら、\(\overline H=0\)で右辺はIだが、左辺はcos(t)Iである。

そのため、1つの乱択bitをRTE word全体で共有して共役するだけでは、目的のTaylor多項式にならない。**生成子を取り出す各箇所で、対称化された辞書を正しい独立分布で使う**ことが必要である。既知のstep間symmetry protectionと区別すべき核心である。

### 7.6 この保存性が保証しないこと

1回のsample回路では、物理部分空間外へ状態が動くことがある。有限sampleから作った平均の統計誤差も残る。従ってこの構成だけから、単一shotでの漏れ確率ゼロ、完全な出力状態の正しさ、noise耐性、postselection不要の一般状態シミュレーターを主張しない。

平均振幅による位相・スペクトル推定には接続可能だが、任意の量子サブルーチンへそのままcoherent unitaryとして渡すためには追加の構成と費用が必要である。

### 7.7 回路費用と失敗し得る理由

SQ_jSは、Q_jの前後にSを入れて実装できる。回転についても

\[
e^{-i\phi SQ_jS}=Se^{-i\phi Q_j}S
\]

である。controlled中央操作に対し外側のSを非制御で適用する構成は、control=0の枝でS^2=Iとなることを利用できる。ただし実際の回路、global phase、identity処理まで確認する必要がある。

補助真空への反射は無料ではない。補助数、ancilla許容、multi-controlled操作の合成費用、隣接反射の相殺を含めて評価する。n個の積generatorと1個の回転を持つeventでは、単純実装の追加反射は最大2(n+1)回であり、実際の期待費用はsampleとコンパイルに依存する。

また、GRADEの決定論fragmentとしての圧縮利益が、involution単位のRTE費用でも維持されるとは限らない。小さなleaky tailだけを乱択へ回せるか、coreまで重く乱択化しなければならないかが重要である。

最後に、全factorization誤差、lambda、B_K、必要shot、Pを保存する決定論block、実際のreflection primitiveを含めて、

\[
\sum_mN_m^{\rm sym}\mathbb E[C_m^{\rm sym}]
<\sum_mN_m^{\rm baseline}\mathbb E[C_m^{\rm baseline}]
\]

が成立する条件を問う必要がある。係数normの非増大だけではこの不等式は証明できない。

### 7.8 新規性と論文化の条件

群平均、LCU、symmetry protectionは既知であり、SPRINTでもleakage保護を扱っている。[R2,R10] 上の構成を「完全に新しい対称性保護」と呼ぶのは不適切である。

独立した研究貢献として狙えるのは、例えば次の組である。

- 圧縮factorizationの特定クラスから、物理信号を保存する実装可能なgenerator辞書を構成するアルゴリズム
- 元のnorm・reflection費用・乱択tail重量から、既存の保護PFやCDFに対する利益条件を導く資源解析
- leakage bias、Taylor bias、統計費用を分離し、実分子で利益・失敗条件を示す検証

代数が既知手法の直接の言い換えで、圧縮利益も残らないなら主テーマにはしない。逆に、既知の失敗要因を避ける新しい実装クラスを構成できれば、PRとの組み合わせに科学的な意味が出る。

### 7.9 外部だけで進める変種

各native blockが最初からPを保つ圧縮factorizationを構成する方向も対象である。この場合、PRを必ず使う必要はない。

ただし「Pを保つ」を課すと表現が実質CDFへ戻り、圧縮の自由度が消える可能性がある。圧縮と不変部分空間の両立条件をまず数学的に問う。新しい存在構成や非自明なtrade-offが得られれば、それ自体が着地点になる。

---

## 8. 候補C：近可積分アルゴリズムが必要とするblock構造を構成する

### 8.1 問題設定

\[
H=A+\alpha\sum_lB_l
\]

と書き、alphaが小さい構造を利用する研究は既に進んでいる。[R8,R9] ここで重要なのは、alphaの小ささだけではなく、利用できる指数演算子の種類である。

e^{-itA}とe^{-itB_l}だけを安価に実装できる場合と、e^{-it(A+alpha B_l)}まで安価に実装できる場合は別のアクセスモデルである。THRIFT系の改善を利用するには、その必要なpropagatorを構成しなければならない。

### 8.2 新研究で狙う差分

新しいPF係数や順序だけを探索するのではなく、分子Hamiltonianから、混合propagatorを安価に実装できるA,B_lを生成する。

候補には、共有固有基底を持つ部分、低次元Lie代数で閉じる部分、対称性依存の可解block、局所小clusterなどがある。可解Hamiltonianの拡張自体も先行研究がある[R24]ので、その構成をPR/近可積分アクセスへ接続する実装・費用上の差分を問う。

古典的に対角化できること、測定できること、coherent controlled exponentialを安価に実装できることは同じではない。特にfull Hilbert spaceでのdense対角化は、この研究が欲しい安価なoracleではない。

### 8.3 評価

理論的な可能性は大きいが、SPRINTを含む近接研究が強く、具体的な分子クラスとprimitive構成がないまま着手すると抽象的な組み合わせになりやすい。

第三候補として保持し、A/Bが既知同値に縮退した場合や、安価な混合blockが見つかった場合に優先度を上げる。2026年のrandomized近可積分PFに関する結果は、取得した抄録上のアクセスモデルに基づく位置付けであり、全定理の独立監査はしていない。[R9]

---

## 9. その他の方向をどう位置付けるか

### 9.1 BLISS・sector-equivalent表現

BLISSとtensor factorizationの同時最適化は既存である。[R5,R6,R23] 「PRのcostでBLISSを最適化する」という目的変更だけでは新規性が弱い。

可能性があるのは、対象sectorで和を保ちつつ各fragmentの作用を再配分し、lambdaとPF誤差の双方を変える構成である。ただし第5節の不変性により、各fragmentまでsector上同一ならPF改善は起こらない。どこでこの前提を破るかを明示する必要がある。

候補Aの変数として組み込む余地はあるが、主題にする前に有効な自由度の分類が必要である。

### 9.2 流動的fragment・一体項再配分

Fluid fermionic fragmentsは測定最適化で既に使われている。[R11] number演算子の代数関係を使う再配分は、Hamiltonianの物理内容を保ったままfragmentを変える手段になり得る。

ただし、測定分散の改善をtime-evolution costの改善に直接置き換えない。新しい再配分法や誤差関係が得られれば候補Aの非直交な発展方向になる。既知測定法をそのまま適用して比較しただけの場合は、応用・適用条件研究として区別する。

### 9.3 sparsificationと局所性

SparStoやSQuISH等が存在するため、係数の小さい項を削る・確率的に間引くこと自体は新しくない。[R15,R17]

近年のsparsification理論[R16]での相対的二次形式保証を、そのまま量子化学の絶対エネルギー精度へ移すことはできない。PSD、局所性、符号、mapping後の非局所性などの前提を確認する。

候補Aの残差構成に組み込む場合は、捨てる部分の精度保証と、正確に補正する部分の乱択費用を分ける。exact ground stateを知っている前提で項の重要度を決めるなら、実用的な運用アルゴリズムとは区別する。

### 9.4 downfolding・transcorrelation・active space

軌道数を減らす価値はあるが、モデル誤差、induced higher-body terms、classical amplitude計算が必要になる。[R18]

非unitary similarity transformのtranscorrelated Hamiltonianでは、有限basisでの近似に加えて非Hermitian・非正規性が問題となり、通常のunitary time evolutionに基づくQPE/PRへそのまま接続できない。非Hermitian固有値推定の先行研究も存在する。[R19]

研究の価値を否定するのではなく、既存PR基盤の上で新しい中心貢献を作るまでに、解くべき異なる問題が多いという理由で初期優先度を下げる。

### 9.5 fermion mapping・mode ordering・実空間表現

mappingやmode orderingはPauli support、routing、並列性を変え得る。ただし、正確なunitary同値性を保ち、対応する分解をそのまま移すだけならoperator normや固有値は変わらない。

新しいmappingに量子化学の具体的構造を使い、実装費用の改善条件を示せるなら独立テーマになり得る。既存mappingのベンチマークだけなら主貢献は別に必要である。今回これらを固定条件として除外してはいない。

### 9.6 低エネルギー専用の表現とspectral amplification

SOSSAおよびDFTHCとspectral amplificationを結ぶ研究が既にある。[R21,R22] 低エネルギーだけを利用する新しい表現のアイデアは重要だが、これらのblock-encoding上の改善とPRの有限信号を同一視しない。

これらは、PRが常に最良であることを前提にしないための比較軸でもある。新しい外部構成が他手法にも役立つなら、それは欠点ではなく適用範囲の拡大である。

---

## 10. 候補比較と推奨順位

以下は数値scoreではなく、今回の文献・代数からの定性的判断である。

| 候補 | 科学的な核 | 新規性上の主要な危険 | 実現上の主要な危険 | 現時点の位置付け |
|---|---|---|---|---|
| A：分解生成とD/R配分 | 同じHを異なる実装・誤差・tail構造へ変える | 既知factor fittingの目的変更だけになる | surrogate改善が実costへ移らない | 着手しやすい主候補 |
| A発展：共有frame＋正確な残差 | 基底変更を節約する表現そのものを作る | CDF、grouping、既知residual処理との重複 | 小さな残差を安い辞書にできない | Aの発展案として比較 |
| B：生成子対称化＋圧縮表現 | leakage障害を平均演算子レベルで除く | 既知群平均＋LCUの直接帰結だけになる | 反射・aux・lambdaが圧縮利益を消す | Aと比較する高い情報価値の対抗案 |
| B外部型：不変部分空間を保つ分解 | 圧縮と物理空間保存を同時に成立させる | 既知CDFへ戻るだけ | 存在する構成が狭すぎる | 理論寄りの高リスク案 |
| C：実装可能な近可積分分解 | 良い理論が必要とするoracleを構成する | THRIFT/SPRINTの組み合わせだけになる | 混合propagatorが結局高価 | 第三候補 |
| BLISS/再配分 | sector同値性を使った費用と誤差の変更 | 既知cooptimizationとの重複 | 実際にはsector内PFが不変 | Aの部品、条件付き独立候補 |
| sparsification | 必要精度に不要な構造を削る | 既知間引き法の再適用 | 誤差保証と実際の精度の隔たり | 補助候補 |
| downfolding/TC | 必要軌道数・表現精度そのものを改善 | 関連手法の適用評価に留まる | 高体項・非Hermitian性・モデル誤差 | 長期的な代替方向 |
| 軌道/mapping/grouping単独 | 実装表現・局所性の改善 | パラメータ比較に留まる | 同値変換だけで目的が不変 | まずは比較対象・構成要素 |

現時点の推奨は、AとBの二つで「既知と何が違うか」「何を変えると実際に利益が生まれるか」を詰めることである。両方の大規模実装を同時に開始することではない。

Aは厳密H保存と既存実装の接続が容易であるため、機構を調べやすい。Bは既知のleakageという障害に対する明確な構成案があり、単なる数値最適化以外の着地点を検討しやすい。その反面、どちらにも論文化を阻む明確な縮退条件がある。

## 11. 「新アルゴリズム」として成立するための条件

最適化を研究にすること自体を否定しない。重要なのは、既存法のparameter調整から一段進んだ内容があるかである。

最低限明確にするのは、入力、許される変換、出力、物理問題の保存条件、計算手順、古典計算量、実装primitive、誤差・資源への作用である。

新しい目的関数だけでも、その目的が本質的に異なる最適構造を生み、効率的に解ける構成や理論があれば貢献になり得る。一方、既知optimizerへ数値評価関数を差し替えた事実だけでは、独立した新規性を強く主張しにくい。

候補Bも同様である。第7節の式は正しいが短い既知代数の帰結なので、それだけを新規定理として売るべきではない。圧縮Hamiltonianへの実装構成、既知方法より有利な領域を特定する資源解析、または圧縮と保存性の非自明な構造定理が必要である。

否定的結果が得られた場合、単に数分子で改善しなかっただけでは新規手法論文になりにくい。自由度の不変性、到達可能な資源削減の限界、既知proxyの破綻条件など、他研究にも使える知見へ一般化できるかを評価する。

---

## 12. 研究方針を決めるために次に必要な判断

### 12.1 まず論点を二つへ限定する

Aについては、「厳密因子回転＋D/R設計」が既知のDF/SCDF/PR最適化と同じ問題にならないかを、変数・目的・制約・oracleごとに照合する。さらに、標準の重量proxyでは不変／既に最適でも、実RTE費用またはPF誤差が変わる非自明な構造を探す。

Bについては、対称化辞書がGRADE型の実際のprimitiveへ接続できるか、反射が構成可能か、どのfragmentを乱択へ移す必要があるかを具体化する。既知symmetry protectionと同じ単位の誤差・costで比較できる形へ定式化する。

これらはGPTの科学的・数学的判断である。ここで既知同値や構造上の不利益が明らかなら、大量の分子計算を始めない。

### 12.2 その後のCodexの役割

候補の意味論と比較対象が選ばれた後、Codexへ最小限の機構確認を依頼する。具体的な分子、basis、grid、実行上限、許容誤差は、その段階で対象に応じて定める。現時点でH4/STO-3Gや固定percent閾値を研究条件にしない。

機構確認の科学的目的は、単にtestを通すことではなく、以下を分けることである。

- H/sector/平均信号の保存性が正しいか
- 目的の自由度が既存法と違う構造を本当に作るか
- 変換・reflection・basis費用を含めて改善方向が残るか
- optimizerが使ったproxy以外の評価でも効果が見えるか

候補Bでは、sample trajectoryの漏れと平均演算子の漏れを別々に調べる。step全体twirlという誤実装を対照に入れると、意味論の違いを確認できる。

### 12.3 公平な比較

外部-onlyの研究であれば、既存内部構成を固定した比較で貢献を示してよい。一体設計の相互作用を主張する場合には、原則として旧外部＋旧内部、新外部＋旧内部、旧外部＋新内部、新外部＋新内部を区別する。Track Bの成果を比較に使う場合は、その時点の正しいbranch・commit・契約を改めて取得する。

基準法も同じ精度・失敗確率・利用可能情報の下で調整する。新手法だけ最適化し、baselineは未調整という比較を避ける。共通古典計算budgetを置くか、最適化費用を別に報告する。

比較相手は研究claimに合わせて選ぶ。すべての方法に対して勝たなければ価値がないとはしないが、CDF/RC-DF、PR、関係するsymmetry protectionや近可積分法を無視して一般的な優位性を主張しない。qubitization系との比較にはlogical qubitとアクセスモデルの差も含める。

### 12.4 独立性とoracleの管理

exact ground stateや真の誤差は、事後評価の参照と、運用時の構成入力を区別する。参照状態を知っているから成り立つ最適化は、その情報がないときの実用アルゴリズムとは別に報告する。

開発した分子・geometryだけでなく、未使用条件、精度域、必要ならcompiler条件へ移したときの維持を調べる。全条件の機械的な総当たりではなく、提案した改善機構が変化する条件を選ぶ。

## 13. 確定事項・未確定事項

### 今回確定したこと

- PR外部のみと外部・内部一体設計を同じ探索対象として扱える。
- 既知研究はすでにDF、orbital、symmetry、randomized PFの接続に踏み込んでいる。
- 共通共役変換だけで対応PF誤差は変わらない。
- 正のsum-of-squaresにおける因子空間の直交混合はHを保存する。
- 与えられたinvolution辞書へのblock反射対称化は、その辞書の係数1ノルムを増やさない。
- その辞書から正しく構成した有限RTE補正平均は、既知の物理部分空間を保存する。

### まだ確定していないこと

- A/Bそれぞれの論文レベルの新規性。
- 実分子での実compiled cost・shot込みの優位性。
- GRADE等のどの具体的実装で候補Bが有利になるか。
- 候補Aの効率的で解釈可能な構成法と、その改善保証。
- 主資源指標、対象分子、basis、実行環境、最終的な研究課題。

### 最終判断

**現在の研究方針は、「既知のDF・軌道・BLISSを順番に試す」ではなく、「非自明な表現変更がPRの資源と誤差へ作用する機構を、二つの具体的候補A/Bで比較すること」とするのが適切である。**

これは研究テーマの最終採択や数値的GOではない。現段階の成果は、広い候補地図、避けるべき既知重複、不変性によるふるい分け、具体的な構成例、および優先して答える研究問いを明確にしたことである。

---

## 付録A. 因子回転が単なる座標変換と異なる最小代数例

量子化学上の効率性の証明ではない、2次元の代数例を示す。

\[
L_1=I+X,\qquad L_2=I+Z.
\]

元の二つの平方は

\[
L_1^2=2I+2X,\qquad L_2^2=2I+2Z
\]

であり、互いに非可換である。

因子空間の45度回転で

\[
L_+=(L_1+L_2)/\sqrt2,
\quad L_-=(L_1-L_2)/\sqrt2
\]

とすると、

\[
L_+^2=3I+2(X+Z),\qquad L_-^2=I.
\]

和は元と同じだが、新しい二つの平方は可換である。従って、因子混合は全blockへの共通unitary共役とは異なり、分解の交換関係を変え得る。

ただし、単一qubitが容易に解けることを利用した例であり、一般の電子二体Hamiltonianでこの改善が残ると主張しない。実分子へ繋がる構成・費用が必要である。

## 付録B. 共通共役変換の証明

積公式を\(S(t)=\prod_j e^{-ia_jtH_{\ell_j}}\)とすると、

\[
\prod_j e^{-ia_jtV^\dagger H_{\ell_j}V}
=\prod_j V^\dagger e^{-ia_jtH_{\ell_j}}V
=V^\dagger S(t)V.
\]

正確な時間発展にも同じ式が成り立つので、誤差演算子全体が共役される。スペクトルnormやFrobenius normは不変である。状態を対応するV^daggerで移せば状態依存の行列要素も一致する。

この証明は、変換後に異なるPauli辞書へ展開し直して別の積公式を作った場合には適用しない。

## 付録C. 因子重量proxyに対する最適性の条件

L_muをHilbert-Schmidt内積のvectorとして並べ、Gram行列をGとする。Oの上位k行をO_Dとすると、保持する平方Frobenius重量は

\[
\sum_{a=1}^k\|\widetilde L_a\|_F^2
=\operatorname{Tr}(O_D G O_D^T)
=\operatorname{Tr}(MG),\quad M=O_D^TO_D.
\]

Mはrank-k直交射影である。Gの固有基底で\(0\le M_{ii}\le1\)、\(\sum_iM_{ii}=k\)より、Tr(MG)は上位k固有値和以下となる。対応する固有方向を選べば達成する。

標準のpair-indexスペクトル分解では、この選択は既知の大きな因子を残す方式になる。この結論は、全lambda_R、固有valueに依存するPauli係数norm、交換子誤差、compiler費用の最適性を主張するものではない。

## 付録D. 群平均への拡張と注意

有限unitary群Gに対し

\[
\mathcal T_G(H)=\frac1{|G|}\sum_{g\in G}U_gHU_g^\dagger
\]

と定義すれば、各involutionを全共役へ均等に展開することで、与えられた係数normを増やさずに群可換な生成子を作れる。複数の物理制約にも形式上拡張できる。

ただし、群のsizeが大きい場合の古典sampling、U_gの合成、既知のencoding projectorの実装が問題となる。係数normの非増大と全リソースの非増大を混同しない。

また、群平均後のHamiltonianは一般に元のHamiltonianとは異なる。対象物理問題を保存するには、目的部分空間での作用が一致することを別途確認する必要がある。

---

## 参考文献・資料

表記「本文」は当該レビューに必要な箇所の確認を意味し、全定理・全数値結果の独立再現を意味しない。preprintは出版済み論文と区別する。

### GitHub

- [G1] Repository/main ref確認：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf>
- [G2] Finite RTE conventions：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf/docs/rte_conventions.md>
- [G3] RTE source versions：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf/docs/rte_source_versions.md>
- [G4] DF partial-S2 source：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf/src/trotterlib/df_partial_s2.py>
- [G5] Source module index：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf/src/trotterlib/README.md>

### 一次文献

- [R1] Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026). DOI: <https://doi.org/10.1103/ynxb-p2xq>; arXiv: <https://arxiv.org/abs/2503.05647>. 取得できたPDF本文はv1表記。掲載抄録および[G2,G3]も参照。v2全本文再監査ではない。
- [R2] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026-06-29). <https://arxiv.org/abs/2606.30741>. Preprint。本文のfactorization、SPRINT、leakageおよび関連appendixを参照。数値図の再解析は行っていない。
- [R3] Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024). <https://quantum-journal.org/papers/q-2024-06-13-1371/>; <https://arxiv.org/abs/2212.07957>. 抄録・掲載情報と手法概要。
- [R4] Cohn, Motta, Parrish, *Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians*, PRX Quantum 2, 040352 (2021). <https://doi.org/10.1103/PRXQuantum.2.040352>. 出版社抄録・概要。
- [R5] *Simultaneously Optimizing Symmetry Shifts and Tensor Factorizations for Cost-Efficient Fault-Tolerant Quantum Simulations of Electronic Hamiltonians*, JCTC (2025). <https://doi.org/10.1021/acs.jctc.4c01722>. 出版社検索抄録。本文の全構成は未監査。
- [R6] *Global Minimization of Electronic Hamiltonian 1-Norm via Linear Programming in the Block Invariant Symmetry Shift (BLISS) Method*, JCTC 21, 703–713 (2025). <https://doi.org/10.1021/acs.jctc.4c01390>. 出版社抄録。
- [R7] Kronenberger, Erakovic, Reiher, *Trotter Error and Orbital Transformations in Quantum Phase Estimation*, arXiv:2602.18913v1 (2026-02-21). <https://arxiv.org/abs/2602.18913>. Preprint。HTML本文。
- [R8] Bosse et al., *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nature Communications 16, 2673 (2025). <https://doi.org/10.1038/s41467-025-57580-5>. 出版社本文。
- [R9] Kim, García-Pintos, *Randomized product formulas beyond optimal deterministic scaling*, arXiv:2608.07720 (2026-08-07). <https://arxiv.org/abs/2608.07720>. Preprint。抄録、アクセスモデルの位置付けのみ。
- [R10] Tran, Su, Carney, Taylor, *Faster Digital Quantum Simulation by Symmetry Protection*, PRX Quantum 2, 010323 (2021). <https://doi.org/10.1103/PRXQuantum.2.010323>. 出版社本文・抄録。
- [R11] Choi, Loaiza, Izmaylov, *Fluid fermionic fragments for optimizing quantum measurements of electronic Hamiltonians in the variational quantum eigensolver*, Quantum 7, 889 (2023). <https://arxiv.org/abs/2208.14490>. 抄録・掲載情報。測定最適化と時間発展を区別。
- [R12] Rubin, Lee, Babbush, *Compressing Many-Body Fermion Operators Under Unitary Constraints*, arXiv:2109.05010. <https://arxiv.org/abs/2109.05010>. 抄録。
- [R13] Gui et al., *Term Grouping and Travelling Salesperson for Digital Quantum Simulation*, arXiv:2001.05983. <https://arxiv.org/abs/2001.05983>. 抄録。
- [R14] Stair et al., *A stochastic quantum Krylov protocol with double factorized Hamiltonians*, arXiv:2211.08274. <https://arxiv.org/abs/2211.08274>. 抄録。DFと確率的時間発展の近接例。
- [R15] Ouyang, White, Campbell, *Compilation by stochastic Hamiltonian sparsification*, Quantum 4, 235 (2020). <https://arxiv.org/abs/1910.06255>. 抄録・掲載情報。
- [R16] Basu, Brakensiek, Putterman, *Many Hamiltonians Are Sparsifiable*, arXiv:2605.02211 (2026). <https://arxiv.org/abs/2605.02211>. Preprint。抄録、PSD/相対保証という適用条件の確認。
- [R17] Chamaki et al., *Self-consistent Quantum Iteratively Sparsified Hamiltonian (SQuISH)*, arXiv:2211.16522. <https://arxiv.org/abs/2211.16522>. 抄録。
- [R18] *Qubit-Efficient Quantum Chemistry with ADAPT-VQE and Double Unitary Downfolding*, JCTC 21, 8799–8811 (2025). <https://doi.org/10.1021/acs.jctc.5c00896>. 出版社抄録。PRの検証論文ではない。
- [R19] *Accuracy and Resource Advantages of Quantum Eigenvalue Estimation with Non-Hermitian Transcorrelated Electronic Hamiltonians*, JCTC 22, 6431–6443 (2026). <https://doi.org/10.1021/acs.jctc.6c00274>. 出版社抄録。
- [R20] Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1 (2026-06-29). <https://arxiv.org/abs/2606.30738>. Preprint。HTML本文の該当部分。
- [R21] Low et al., *Fast Quantum Simulation of Electronic Structure by Spectral Amplification*, Physical Review X 15, 041016 (2025). <https://doi.org/10.1103/pb2g-j9cw>; <https://arxiv.org/abs/2502.15882>. 出版社抄録。
- [R22] King et al., *Quantum Simulation with Sum-of-Squares Spectral Amplification*, Physical Review Letters 136, 110601 (2026). <https://doi.org/10.1103/m3fj-m4rm>; <https://arxiv.org/abs/2505.01528>. 出版社抄録。
- [R23] *Reducing the Runtime of Fault-Tolerant Quantum Simulations in Chemistry through Symmetry-Compressed Double Factorization*, JCTC (2024). <https://doi.org/10.1021/acs.jctc.4c00352>. 出版社抄録。
- [R24] Patel, Yen, Izmaylov, *Extension of Exactly-Solvable Hamiltonians Using Symmetries of Lie Algebras*, Journal of Physical Chemistry A 128, 4150–4159 (2024). <https://arxiv.org/abs/2305.18251>. 抄録・掲載情報。
- [R25] Motta et al., *Low rank representations for quantum simulation of electronic structure*, npj Quantum Information (2021). <https://doi.org/10.1038/s41534-021-00416-z>; <https://arxiv.org/abs/1808.02625>. 掲載情報・概要。sum-of-squares DFの基礎。

---

本記録における提案、優先順位、条件付きの数式展開は、文献の実験結果と区別したGPTの研究上の判断である。実験で未確認の利益を、確認済み成果とは扱っていない。
