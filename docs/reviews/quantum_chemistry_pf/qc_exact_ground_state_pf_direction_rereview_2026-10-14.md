# 厳密基底状態を前提とする量子化学PF研究：方針再レビュー

- 日付：2026-10-14 JST
- 区分：ユーザーの明示開始承認を受けた、重要な科学的方針レビュー。
- 対象：量子化学Hamiltonianの基底エネルギー計算を低費用化するための、Hamiltonian表現・分解・product formula（PF）構成の研究。
- 更新前提：厳密基底状態を仮定してよい。block-encodingとqubitizationは対象外。
- 判定：回路構造を維持する一体項再配分と、その基底固有位相誤差に対する低body-rank更新則を、次の限定的な構成検証の優先課題にする。独立新規性・量子資源優位を認定したものではない。
- 実行境界：新しい分子生成・固有値計算・PF性能評価・数値最適化・保存結果の再集計・sampling・compile・T合成は実施していない。レビュー承認はこれらの実行認可を含まない。
- 旧研究の保全：C1現系列の終了、C2の比較対照としての保持、C3/C4現構成への投資見送り、N1 STOP/N2 HOLD/N3 CLOSED/現THRIFT template CLOSEDを維持。別Trackのsource・結果・認可・STOPは変更しない。

## 0. 結論

今回の第一候補は、**「厳密基底状態の応答を利用し、実装可能なfragmentの種類を変えずにHamiltonianの分け方を更新する方法」**である。まず、DFなどの回転density fragmentと一体fragmentとの間で一体項を移し替える自由度を用いる。

重要なのは、一体項を移す操作自体を新発明としないことである。この自由度はFluid Fermionic Fragments（FFF）および既存Trotter partitioning研究に明示され、FFF論文はTrotter誤差最小化への応用も述べている。今回の研究差分候補は、**基底固有位相の誤差へ直接結び付く更新を、低いbody rankの期待値と小さい代数問題から構成できるか**に置く。

本レビューでは、次の解析的な構成を整理した。

1. 一体項＋単一の可解なdensity fragmentという二分割では、対称二次PFの主要基底固有値誤差を、正半定値の二次形式として最小化できる。必要な係数は二重交換子の期待値で表され、元Hamiltonianが二体までなら1-/2-RDMの収縮で記述できる。
2. 一般の多fragment DFでは、一つのfragmentと一体項の間の移動に限れば、主要BCH誤差演算子の変化は移動係数について厳密に二次までとなる。線形応答は最大三体、二次応答は最大二体である。
3. この構造は、全候補で高body-rankのBCH演算子を作り直す代わりとなる可能性がある。ただし、既知のcompact BCH評価より実際に安いこと、有限stepで有利なこと、一般分子へ移せることは未証明である。

第二候補はDF因子の等価回転である。ただし、因子normや総交換子二乗normなどの不変な量を最小化しても何も変わらない。回路基底を変えずに済む一体項再配分を先に調べ、因子回転を同時に巨大な探索軸として追加しない。

補正PFと低エネルギー解析は予備・補助方向とする。純共役processorは固有値を変えない。真の補正を追加する場合は高body-rank・回路費用が問題となる。厳密基底状態を用意できても、低エネルギー射影や各fragmentの可解性が無料になるわけではない。

**次の担当はCodexによる、一体項再配分の応答式・既知法との差・実装会計・最小比較計画をまとめた技術準備である。** 同じ一般文献調査やC1の救済探索を繰り返す指示ではない。

---

## 1. 研究目的、今回の変更、別Trackとの境界

### 1.1 固定する目的

元の量子化学Hamiltonianの基底エネルギーを求める際に必要なHamiltonian simulationを、PF・Hamiltonian構造・回路生成の工夫によって低費用化する。新しいprimitiveの発明だけを研究貢献の条件にはしないが、既知の構成に名前を付け直すだけでは新算法としない。

今回、ユーザーが厳密基底状態を仮定してよいと明示した。この仮定を受け入れ、初期状態準備法の開発や近似状態のfidelity証明を本研究の進行阻害条件に戻さない。

### 1.2 正確に区別する情報

- 初期入力として厳密な量子状態を利用できること。
- その状態上の特定の交換子期待値を、理想的な設計情報として評価すること。
- 任意のRDMや全固有分解を、古典的な厳密値として無償で取得できること。

これらは同じアクセス仮定ではない。今回の第一段階では厳密状態を用いる理想条件で構成を比較してよい。後に総アルゴリズム費用を主張する際は、期待値・RDM収縮の取得または評価費用を別記する。状態準備費用の研究へ戻すことはしない。

古典的な完全1-/2-RDMを厳密値として無償提供するモデルなら、二体HamiltonianのE0自体も収縮から分かる。その場合の成果は「理想情報を使ったsimulation設計」の結果であり、未知E0を求める無条件の量子優位とは区別する。この論理上の区別を理由に、ユーザーの厳密状態仮定を拒否しない。

### 1.3 対象外と役割分担

block-encoding、qubitization、およびそれらを中心とするQSVT/量子walkの設計を今回の候補・必須対照から除外する。

Track Aの既存PR資源評価・RPE/QPE task接続を本チャットへ移管しない。Track Bの有限RTE生成・乱択・return aggregation・proposal開発を重複起動しない。この研究の出力はHamiltonian再分割またはPF回路の生成手順であり、後でPRに利用できることは妨げない。PRとの組合せを必須にもしない。

前回の全体レビューにあった「次の主課題をTrack Aのground-energy task接続に戻す」という将来方向は、ユーザーの指摘に従って撤回したままである。本レビューはその将来方向を置き換える。

---

## 2. 取得資料と、過去の結論の継承範囲

### 2.1 今回実際にGitHubから取得した主要資料

| ID | 固定commit・資料 | 今回参照した内容 |
|---|---|---|
| G1 | 5f35988d3145641739fb83da98567772183046ef / docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md | 研究全体の固定結果・候補・scope。主要表の該当範囲を読んだ。25原資料を今回すべて再監査したという意味ではない |
| G2 | c24dcede10ed9726e1e0b1030eca749ba7bc5cba / docs/research/representation_construction_comparison_results_20261010.md | 旧common diagonal core、factor-label回転の不変性、回路と残差の交換条件 |
| G3 | 5ff55e749ceb33e54da0b3c70ac6ac08599ebb82 / docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md | C1の固定catalog oracle比較、旧state/eligibility/価格条件、限界 |

G1 URL:
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f35988d3145641739fb83da98567772183046ef/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md

G2 URL:
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/docs/research/representation_construction_comparison_results_20261010.md

G3 URL:
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5ff55e749ceb33e54da0b3c70ac6ac08599ebb82/docs/research/qc_c1_saved_oracle_diagnostic_v1_results_20261014.md

前回までのレビュー本文も会話と既存Markdownで利用できる。今回の再レビューは原pilotの再計算・全raw監査ではない。GitHub本文の取得と、各本文が記載する過去のremote監査PASSを区別する。

### 2.2 過去研究が今回に与える制約

**C1**：固定catalog・旧入力state・価格・gridでは、通常16contextのmacro/baseline比は1.082687339〜5.331029315。oracleにも登録5%の利益がなかった。この事実は保持する。しかし、厳密基底状態へ初期状態を変えれば、PF branchへのoverlap、適格性、費用が変わり得るため、旧比率をそのまま新前提の数値結果として扱わない。真の主要誤差を診断したことと、厳密基底状態入力で全費用を再評価したことは別である。

現C1を自動再開しない理由は、旧比率が新条件でも不変だからではなく、固定macrocycleの段数増加・既知compositionとの重複・新しい構成機構の不足が残るためである。

**THRIFT現template**：理想PFのstep削減と、占有数依存関数回路の費用が回収できないことを分けた結果を保持する。次候補では、改善する前に高価な関数primitiveの新規実装を必須にしない。

**N1/N3**：Hamiltonianの近似やactive-space削除による誤差会計が最終task利益を失わせ得た。第一候補はHを厳密に保存するため、同種のHamiltonian近似誤差は追加しない。ただし再分割後のPF誤差・合成誤差は残る。

**旧DF common core**：固定線形射影Pを全factorへ同じように適用して二乗和を取るcoreは、直交factor-label回転で不変である。この旧無効方向を再提案しない。個別factorを順序付きで指数化するPFまで不変という結果ではない。

**Track A/B**：実装資産・費用会計・対照の設計を参考にできるが、別条件の倍率を新方法へ転記しない。このレビューから別Trackの運用を変更しない。

---

## 3. 外部一次資料の比較と、新規性の境界

今回の調査は2026-10-14時点に取得した一次本文・特定版に基づく関連箇所の比較であり、全分野の初出性を証明する網羅調査ではない。

| ID | 文献・取得版 | 読み取った事実 | 本候補へ課す制約 |
|---|---|---|---|
| P1 | Choi, Loaiza, Izmaylov, Fluid fermionic fragments…, Quantum 7, 889 (2023), 出版PDF | 一体成分のfragment間移動で測定費用を最適化。結論でTrotter誤差最小化への応用も明記 | 「FFFをTrotterへ応用する」ことを新規性としない |
| P2 | Martínez-Martínez, Yen, Izmaylov, Assessment of various Hamiltonian partitionings…, Quantum 7, 1086 (2023), arXiv:2210.10189v2 | §2.3式37–39に同じ一体項抽出・再配分。spectral-range依存の最適化の難しさを述べ、reflection/L1に基づくheuristicを採用 | 変換の自由度と回路可解性は既知。state-specific eigenphase更新の具体差を比較する |
| P3 | Mehendale et al., Estimating Trotter approximation errors to optimize Hamiltonian partitioning for lower eigenvalue errors, Digital Discovery 4, 3540–3551 (2025), 出版PDF | 対称PFのBCH主要期待値で固有値誤差とpartitionを評価 | 基底状態期待値を使うこと自体は既知 |
| P4 | Kronenberger, Erakovic, Reiher, Trotter Error and Orbital Transformations in Quantum Phase Estimation, arXiv:2602.18913v1 | 軌道・順序・構成の変更によるenergy誤差と相殺を扱う | orbital/orderを変えて誤差を小さくする一般案では差が足りない |
| P5 | Casares et al., Theory and practice of Trotter product formulas for quantum chemistry, arXiv:2606.30741v1, 本文/PDF | GRADE/SPRINT、回路構造に合わせた一体成分の配置、fragment・Gaussian変換の扱い | 一体項をどのfragmentへ置くかが費用に効くことも既知。固定基底だから無料とは扱わない |
| P6 | Maxwell et al., Practical Estimation of Trotter Error for Hamiltonian Simulation, arXiv:2606.30738v1 | diagonal BCHとcompactな交換子表現、主要期待値の評価費用の削減 | 全BCHを愚直に展開する弱い対照だけと比較しない |
| P7 | Bagherimehrab et al., Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas, arXiv:2409.08265v3 | 純共役correctorと、step内に挿入する真の補正を区別できる構成 | 全CPFを「固有値不変」とまとめて否定しない。真の補正の回路費用が必要 |
| P8 | Hamiltonian simulation in the low-energy subspace, npj Quantum Information (2021); Better bounds for low-energy product formulas, arXiv:2402.10362 | 低エネルギーPF解析と前提付きの改善 | exact ground inputだけで局所性・正性・射影accessが得られるとはしない |

P1: https://quantum-journal.org/papers/q-2023-01-03-889/pdf/
P2: https://arxiv.org/pdf/2210.10189
P3: https://pubs.rsc.org/en/content/articlepdf/2025/dd/d5dd00185d
P4: https://arxiv.org/html/2602.18913v1
P5: https://arxiv.org/abs/2606.30741
P6: https://arxiv.org/html/2606.30738v1
P7: https://arxiv.org/html/2409.08265v3
P8a: https://www.nature.com/articles/s41534-021-00451-w
P8b: https://arxiv.org/abs/2402.10362

### 取得上の限界

P1/P2/P3などのPDF本文はwebの解析テキストで取得した。web screenshotで該当ページの画像取得を複数回試したがInternal Errorとなり、独立なページ画像検査は完了していない。P2の別ローカル取得も失敗した。したがって、図の数値を独自に読み取ったり、PDFの全組版を確認したとは主張しない。今回の判断は読み取れた文章・式と、明記した独自の代数導出に基づく。

P8は目的比較に必要な概要・前提の確認に留め、その全証明を新たに監査していない。P1–P7も全論文の正しさの査読ではない。

---

## 4. 評価対象：固有位相、state error、overlapを分ける

以下の漸近式は、固定された対称二次PF、同じprincipal-log branch、孤立した非縮退ground branchを小さいhで追跡する場合の説明である。全研究へ非縮退を黙って追加するものではない。

\[
S_2(h)=\exp\{-ih[H+h^2D+h^4E+O(h^6)]\}.
\]

\[
E_{0,h}=E_0+h^2d_0+h^4\left[
\langle E\rangle_0+\sum_{n>0}\frac{|D_{n0}|^2}{E_0-E_n}
\right]+O(h^6),\qquad d_0=\langle D\rangle_0.
\]

\[
1-|\langle\psi_{0,h}|\psi_0\rangle|^2
=h^4\sum_{n>0}\frac{|D_{n0}|^2}{(E_n-E_0)^2}+O(h^6).
\]

これは摂動論に基づく解析的整理であり、固有state入力を用いるPF研究[P3,P4,P6]との関係を持つ。式中の励起スペクトルを無料の設計入力として要求しているのではない。

厳密な元groundを入力しても、その状態が近似PFの固有状態になるわけではない。固有位相の系統誤差、異なるPF固有枝へ出る確率、controlled相対位相を分けて扱う。

縮退groundでは、単一のd0だけで固有枝を記述できない。P0DP0のスペクトルと入力状態の対応を用いる必要がある。二分割の正半定値二次形式は任意groundベクトルに対して成立するが、その期待値だけを縮退したQPEの単一出力の誤差と同一視しない。

### 4.1 主要項と費用の関係

主要biasが|d0|h²で支配され、step単価gが同じ精度条件でほぼ固定なら、一定のbias予算に対する時間発展費用の比較量はg√|d0|となる。この比例関係は、総進化時間が共通で、必要step数がT/hであることから得られる。

ただし、d0=0を費用0と解釈しない。高次項・overlap・合成精度が支配する。gが角度や精度に依存する場合は単純式を修正する。主結果は有限hで同じenergy精度を得るcontrolled回路費用によって判断し、主要係数は機構を説明する量とする。

### 4.2 必須の安価な対照：主要エネルギー誤差のスカラー補正

d0を同じ情報条件で取得できるなら、

\[
\widehat E_{\mathrm{corr}}=\widehat E_{\mathrm{PF}}-h^2d_0
\]

という既知の摂動式から直ちに得られる後処理を対照に置ける。回路で表せばe^{+ih³d0}S2(h)であり、QPEではcontrolの相対位相として保持する。

この補正は独立した新算法と認定しない。全Dを除くわけではなく、off-diagonal成分・overlap・h4残差は変えない。またd0の取得誤差はh²倍して残る。だが「主要energy biasを消した」だけの複雑な再分割案に対する強い対照である。

C2外挿、既存S4、既知ordering/basis最適化も同じ厳密状態条件で比較する。再分割案だけに正確なstate期待値を与えない。

---

## 5. 第一候補：可解fragmentを維持する一体項再配分

### 5.1 Hを厳密に保つ変換

有限基底のHを、scalarを正確に保持して

\[
H=H_0+\sum_{\ell=1}^{L}F_\ell,
\quad F_\ell=\mathcal U_\ell^\dagger
\left(\sum_{pq}V^{(\ell)}_{pq}n_pn_q\right)\mathcal U_\ell
\]

とする。H0は一体、各U_lは粒子数保存Gaussian軌道変換である。各fragmentから

\[
X_\ell(\mathbf x_\ell)=\mathcal U_\ell^\dagger
\left(\sum_p x_{\ell p}n_p\right)\mathcal U_\ell
\]

を取り、

\[
H_0(\mathbf x)=H_0+\sum_\ell X_\ell,
\qquad F_\ell(\mathbf x)=F_\ell-X_\ell
\]

と再分割する。総和は元Hに厳密に等しい。[P1,P2]にある既知の自由度である。

各F_lは同じU_lで対角化でき、H0(x)は一体のままである。DF相互作用の正負weightに関係なく、Hermitianな回転density表現が保持されていれば、この線形の移し替えは可能である。正の平方和だけに限定する必要はない。

### 5.2 何を維持でき、何を維持したと主張してはいけないか

維持できるものは、Hそのもの、fragmentの数、各density fragmentの軌道frame、number-conservingな一体＋densityという可解な演算子の種類である。

無条件に維持できないものは、実gate数・depth・T数・全体のbasis融合である。H0が元々対角または疎なら、移動後にdenseなGaussian変換が必要になり得る。元fragmentに存在しなかった一体位相が追加されることもある。特殊な「一体演算子の平方」を可逆算術で実装する方式では、F_l−X_lの線形項を別途実装する必要がある。

従って「回路構造を維持する」とは、追加の高体補正や新しいresolvent primitiveを必須にしないという意味であり、「完全に無料で誤差を改善する」という主張ではない。

---

## 6. 本レビューの解析：二分割における構成可能な二次最小化

### 6.1 成立する範囲

H=A+B、Aは一体、Bは固定の軌道frameで占有数に対角な二体までの演算子とする。N_pをそのframeのoccupationとし、X=Σ_p x_pN_pを使う。

\[
A_x=A+X,\qquad B_x=B-X.
\]

A_xの進化はGaussian、B_xの進化は同じframeのdiagonal進化である。これは全量子化学Hが二つのこの種のfragmentで表せるという主張ではなく、構成原理を解析的に閉じる最初のclassである。

### 6.2 主要固有値誤差の正半定値表現

\[
S_2(h;x)=e^{-ihA_x/2}e^{-ihB_x}e^{-ihA_x/2}.
\]

物理規約S2=exp[−ih(H+h²D+…)]では、

\[
D_x=\frac{[A_x,[A_x,B_x]]}{24}
+\frac{[B_x,[A_x,B_x]]}{12}.
\]

H|ψ0⟩=E0|ψ0⟩から、任意Kに対し⟨[H,K]⟩0=0である。またHermitianYについて、

\[
\langle[Y,[Y,H]]\rangle_0
=-2\langle Y(H-E_0)Y\rangle_0.
\]

従って、

\[
\boxed{d(x)=\langle D_x\rangle_0
=\frac1{12}\langle A_x(H-E_0)A_x\rangle_0\ge0.}
\]

ここでは[H−E0]は全spaceまたは選んだ対称性sectorで非負であり、演算子とstateはそのsectorに整合させる。任意励起stateへ同じ非負性を転用しない。

### 6.3 線形方程式による更新

\[
c=\langle A(H-E_0)A\rangle_0,
\quad b_p=\operatorname{Re}\langle N_p(H-E_0)A\rangle_0,
\quad M_{pq}=\operatorname{Re}\langle N_p(H-E_0)N_q\rangle_0.
\]

すると、

\[
12d(x)=c+2b^{\mathsf T}x+x^{\mathsf T}Mx,
\qquad M\succeq0.
\]

Mはベクトル(H−E0)^(1/2)N_p|ψ0⟩の実Gram行列である。Gram構造よりbはMのrangeに入り、無制約の最小ノルム解と最小値は、

\[
x_*=-M^+b,
\qquad d_{\min}=\frac{c-b^{\mathsf T}M^+b}{12}.
\]

M^+はMoore–Penrose逆行列。総粒子数や保存量の方向などで特異になる場合がある。特異性をバグとしない。係数範囲・回路費用・有限hの安全性を制限する場合は、その範囲内の凸二次最小化へ変える。正則化した解を元の無制約厳密最小値と表示しない。

線形代数solveそのものはdenseなn変数ならO(n³)。これは期待値取得・RDM収縮・係数表生成・回路構成の全費用ではない。

### 6.4 E0も励起固有分解も入力しない係数表現

\[
M_{pq}=\frac12\langle[N_p,[H,N_q]]\rangle_0,
\quad b_p=\frac12\langle[N_p,[H,A]]\rangle_0,
\quad c=\frac12\langle[A,[H,A]]\rangle_0.
\]

この表現はE0を値として入力しない。AとN_pが一体、Hが二体までなら、各二重交換子は二体までである。さらに[N_p,B]=0なので、Mは一体の収縮へ下がる。

従って、二分割classの更新は1-/2-RDMの適切な収縮で記述でき、Hの全スペクトルや全BCH高体テンソルを必要としない。この解析は本レビューで導いた構成上の結果である。先行性・最適な計算量・実量子費用は未認定であり、Codexで符号・係数・normal orderingを別実装確認する。

### 6.5 意味と限界

異符号の複数PFを組み合わせて打ち消すC1とは異なり、**一つのPFを作るfragment自体を変えて、非負の主要誤差を下げる**。

b=0ならこの自由度での主要項改善はない。Mが悪条件なら係数が大きくなり、fragment normと高次誤差・合成費用を増やす可能性がある。主要項を下げる解が、有限hや総費用の最良解であるとは限らない。

非縮退groundでd(x)=0まで達した場合、A_x|ψ0⟩はgroundと平行になり、B_xも同じstateを固有stateとする。そのときS2はそのstate上で全hに対して正しい位相を与える。しかし一般の相互作用Hamiltonianで、そのような共通固有state分割を一体再配分だけで作れるとは限らない。**零誤差を成功の必要条件にも予測結果にもしない。** 縮退の場合は単一stateのd=0だけから同じ結論を導かない。

---

## 7. 一般の多fragment DFへ向けた、局所更新の構成

### 7.1 全体を凸二次と誤認しない

一般の分子は多数のF_lを要する。6節の二分割のPSD最小化を、そのまま全DF分割へ適用できるとはしない。

固定順序の対称二次PFの主要BCH演算子Dは、fragment演算子について三次のLie多項式である。そこで、一つのF_lとH0だけの間に、同じ一体X(x)を正負に移す。

三つとも更新分から取る項は[X,[X,X]]=0なので、

\[
\boxed{D(x)-D(0)=\sum_p x_pL_p+\sum_{pq}x_px_qQ_{pq}.}
\]

ここで係数Qの対称化・1/2規約は実装時に固定する。式は厳密な三次BCH係数の差についてであり、有限hの全logが二次になるという意味ではない。

複数fragmentを一度に異なるX_lで更新するなら、[X_l,[X_m,X_n]]が残り得るため、この「高々二次」は無条件には維持されない。一つのfragmentずつ更新することが構成上の要点である。

### 7.2 body rankの階層

number-conservingなk体・l体演算子の交換子は最大k+l−1体となる。一体演算子による交換子はbody rankを増やさない。このため、元fragmentが二体までの場合、

| 量 | 最大body rank | 必要な期待値情報の表現 |
|---|---:|---|
| 一般D(0) | 4 | 4-RDM収縮またはoperator-state計算 |
| 線形応答L_p | 3 | 3-RDMの必要収縮 |
| 二次応答Q_pq | 2 | 2-RDMの必要収縮 |

これにより、各更新の度にD全体の四体表現を作り直さず、更新に必要な収縮を計算する案が得られる。baseline d(0)の初回取得は別に必要である。compact BCH[P6]やstatevector/tensor表現との比較なしに、古典費用の改善を確定しない。

3-RDMはdenseに保存すれば軌道数nに対してO(n^6)要素を持つ。低body-rankというだけで安価とは呼べない。必要な方向の収縮だけを評価する、既存のfragment作用を再利用する、という実装が重要になる。

### 7.3 具体的な候補生成手順

一体方向Yを固定し、移動をX=tYに限れば、

\[
d(t)=d(0)+\ell t+q t^2.
\]

有限区間内で|d(t)|を最小化する候補は、区間端、二次式の実根、頂点、元t=0から得られる。候補ごとの新しい固有値計算を必要とせず、交換子期待値を用いて生成できる。

これは、任意の高次元最適化を「量子化学特化」と言い換えただけの案より、構成手順が明確である。一方、この一次元最小化自体も初等的であり、それ単独を新規定理とはしない。

一般の場合、qの符号は保証されず、全分割のglobal optimumは得られない。方向の選び方、係数範囲、複数更新時の再利用、finite-hの棄却、controlled費用との同時判断が未実装である。独立した研究差分は、これらを低費用の生成器として閉じたときに問う。

### 7.4 現時点の研究仮説

「state-specificな二次応答と低body-rank収縮を用いることで、既存のfragment再配分最適化より少ない古典評価費用で、同じPF回路classのより有利な分解を生成できる」という仮説を置く。

- 数学的自由度は既知。
- 主要係数の期待値利用も既知。
- 本レビューの局所二次構造とbody-rank階層は解析的な候補。
- 独立新規性、実装全費用、有限hの改善、分子での頑健性は未確定。

同じ最適分解へ到達しても、その生成が非自明に安くなればアルゴリズム上の寄与となり得る。逆に、同じ式を同じ費用で計算するだけなら寄与はない。「既知controlが同じ結果を作れる」だけで高速生成の可能性を機械的に否定しない。

---

## 8. 第二候補：DF因子の等価回転

### 8.1 正確な自由度

\[
H=K+\frac12\sum_a L_a^2,
\qquad L'_a=\sum_bO_{ab}L_b,\quad O^{\mathsf T}O=I
\]

ならΣ_a(L'_a)²=Σ_a L_a²。L同士の可換性は不要であり、Oの直交性で全交差係数が消える。

符号付き表現H=K+1/2 Σ_a s_a L_a²では、符号行列Jを使ってOᵀJO=Jが必要。正負をまたぐ通常の直交混合はHを変えてしまう。同符号群内の直交回転なら安全に利用できる。非コンパクトな変換を許すとnorm増大・悪条件化を別に制御する必要がある。

平方表示の一体補正・scalar・normal orderingを固定し、二電子tensorだけが同じこととfull Hamiltonianが同じことを混同しない。

### 8.2 不変な目的を最適化しない

全factorを保持する直交混合では、以下は不変である。

\[
\sum_a\|l'_a\|_F^2=\sum_a\|l_a\|_F^2,
\qquad \sum_a\operatorname{Var}_{\psi_0}(L'_a)=\sum_a\operatorname{Var}_{\psi_0}(L_a).
\]

またbilinear commutatorにO⊗Oが作用するため、

\[
\sum_{ab}\|[l'_a,l'_b]\|_F^2
=\sum_{ab}\|[l_a,l_b]\|_F^2.
\]

同様に、全組の[ L_a,L_b ]|ψ0⟩の二乗norm和も直交変換で不変となる。これらは本レビューの線形代数的整理であり、特定の実装性能を評価したものではない。

旧common-coreの不変性[G2]も同系統である。これらをobjectiveにして「非可換性を減らす」探索は行わない。

一方、平方fragment L_a²の順序付きnested commutator、個々のframeの疎性、実回路のbasis変換費用は一般に不変ではない。したがって因子回転全般を無効とはしない。

### 8.3 なぜ第二候補か

因子回転は個別L_aの固有basisを変え、多数のGaussian回路・角度・fusionを変える。回転自体はfactor数を減らさず、rank削減を無償に達成しない。

第一候補の一体項再配分ならdensity frameを保持したまま進められる。そのため、まずそちらで構成と費用の関係を閉じ、必要なら因子回転を別自由度として加えるのが合理的である。

第二候補を進める条件は、不変でないstate/PF目的を明示し、既知CDF/FRO/軌道最適化と区別できる低費用生成規則を具体化することである。任意Oを多数試すだけでは新算法としない。

---

## 9. 補正PF・低エネルギー方向の比較

### 9.1 本当に固有位相を変える補正か

P^{-1}S(h)PはSと同じ固有値を持つ。純processorが初期stateとのoverlapを変えることと、固有位相biasを減らすことは別である。

しかしCorrected Product Formulas[P7]には純共役だけでない挿入補正がある。全CPFを固有値不変と誤分類しない。真の補正ではnested commutatorを実装する費用が問題になる。二体Hamiltonianのnested commutatorは高体rankになり得るため、安い可解構造・低rank・限定支持を具体的に得ない限り、THRIFT現templateと同様に追加回路費用がstep削減を失わせる危険がある。

従って予備候補は「量子化学構造により安価な補正演算子を生成できる場合」であり、一般的なcorrected PFの適用だけを新しい研究として起動しない。

### 9.2 低エネルギーに制限することの意味

exact ground inputはstate-dependentなerror評価を使う強い前提である。しかし各fragmentが同じgroundを持つわけではなく、PF途中の作用が低energy部分空間内にとどまるとは限らない。

低energy PFの既知解析[P8]は利用できる前提の範囲で参照する。positive-fragment分解やlocalityを一般量子化学に自動移送しない。identity shiftでエネルギー原点を大きく変えても、非可換誤差自体を小さくしたことにはならない。

今回は、低energy解析を第一候補の評価・保証の補助に位置付ける。未知projectorを新しい量子primitiveとして実装する研究へ逸脱しない。

---

## 10. 公平な最小比較の設計

### 10.1 研究仮説を三つに分ける

**構成仮説**：一体再配分でHと可解classを維持し、主要固有位相誤差を低費用に更新できる。

**有限step仮説**：主要係数の変化が、使用する有限hでのenergy bias・PF ground weightを改善する。主要係数だけ改善しh4やoverlapで失う場合を区別する。

**資源仮説**：古典生成とcontrolled回路・合成・推定の共通条件を含めて、強い対照より有利になる。厳密state準備を前提とするconditional resultとして表示する。

一つの仮説が通ったことを、他の成功へ昇格しない。

### 10.2 比較対象

1. 元分解のS2と、同費用になる単純細分化。
2. 同分解および許可された再分割を使うS4等の強い既知PF。
3. 既知reflection/L1ベースのfragment再配分[P2]、FFFの既知選択思想[P1]、energy-error partition評価[P3]。
4. 同じ厳密状態情報からの主要energy補正、およびC2/Richardson/Chebyshev型の既知外挿。
5. 一般DFへ進む場合、compact BCH[P6]を用いて同じ目的を直接評価する実装。弱い全交換子列挙だけを対照にしない。

全対照に同じH、DF誤差/保持rank、state仮定、control/phase、精度条件、許容する一体再配分と合理的fusionを与える。入力によって効率的実装ができない対照を無理に含める必要はないが、その不適用を明示する。

block-encoding/qubitizationは今回の候補にも必須対照にも含めない。

### 10.3 結果前に固定するもの

対象H、basis、geometry、DF truncation規約、対称性sector、初期厳密state、候補更新方向/係数範囲、PF order/ordering、h範囲、energy/統計/合成budget、controlled費用指標、比較対象、数値margin、原結果との分離を固定する。

このレビューでは分子名・個数・距離・call数を新しく指定していない。二分割の原理確認だけで一般分子への貢献を示したことにしない。一般DFの最小非自明caseを準備時に選び、結果を見て有利な方向・入力だけを追加しない。

厳密stateをアルゴリズム設計に使うこと自体は今回許された前提であり、旧C1の「近似stateだけで選ぶ」制約を流用しない。一方、目的値E0をそのまま位相として埋め込んだり、全候補の有限h正解誤差を無料で最適化する方式に変わった場合は、ideal-information診断とoperationalな生成手順を明記する。

---

## 11. 次のCodex作業と、レビュー・実行の境界

次の一作業単位を推奨する。

**一体再配分の二次応答・低body-rank更新を、既知法に対する具体的生成手順として技術的に閉じ、最小比較を実行前状態まで準備する。**

必要事項：

- 本レビュー6節のBCH符号、二次形式、二重交換子、singular M、縮退時の射程を独立確認する。
- 7節のlocal updateが高々二次であること、linear3-body/quadratic2-bodyのrank、複数同時更新では条件が変わることを確認する。
- FFF/P2/P3/P5/P6の該当箇所と、新しい更新・収縮アルゴリズムの対応を表にする。一般的な文献探索を無制限に続けない。
- density frame、一体Gaussian対角化、scalar/number sector、controlled回路、特殊square実装との違いを具体化する。
- 期待値・RDMをどの形式で使うか、生成時間・memory・計測費用の何を別建てにするかを明示する。exact ground inputの仮定は維持する。
- 既存sourceを再利用できる範囲と、新しいcollector/update/checkerを分離する。登録外の形式・人工テスト、source・入力・比較契約・実行planをまとめる。
- 分子生成、期待値の新数値評価、パラメータ最適化、PF eig、compile等は別の明示実行承認まで起動しない。

この作業は「また一般的に新規性があるかだけを調べる」段階ではない。具体式が誤っている、既知手順と完全に同じ計算量でしか実装できない、または回路構造を保てないという明確な技術的結論が出た場合は、その事実を保存して停止する。

普通のsource修正・人工test・preflight・公開のたびに同じGPT科学レビューを要求しない。重要な新しい数学的不整合、中心比較の変更、または認可済みpilotの科学的結果が得られた段階でGPTへ戻す。

---

## 12. 研究成果として成立する条件と終了条件

### 12.1 成立し得る主張

- 特定の可解classで、energy-weightedな基底応答により最良再配分を構成できること。
- 多fragment DFのlocal variationを低body-rankの収縮へ落とし、既存の直接再評価より安い生成法を与えること。
- その生成法が、同じ厳密state条件・energy精度で、補正済みPFや高次PFを含む強い対照に対して資源差を示すこと。
- どの構造では更新が効かず、どの構造では主要誤差減少が有限step利益につながるかを非自明に分類すること。

いずれも本レビューだけで実証された主張ではない。二分割の初等的な最小化式だけで、独立論文が成立すると認定しない。数学的に既知の変換を使っても、非自明な低費用生成や精度条件の改善が示されれば研究寄与になり得る。

### 12.2 縮小・終了を考える条件

- 変換も更新も計算方法も既存手順の直接再記述で、生成費用または精度条件に差がない。
- 二分割toyでしか安い更新が成立せず、一般量子化学DFへの接続に高価な全space処理が不可欠である。
- 主要dの減少はあるが、scalar補正やS4/C2を含めると有限hの回路利益が消える。
- 係数増大、H0 frameのdense化、追加diagonal回転、合成精度によって費用が増える。
- 効率が厳密な全RDMの無料古典アクセスにしか依存し、条件付きsimulation設計以外の主張を作れない。

これらの可能性を現時点で結果と断定しない。性能改善が小さいという理由だけで自動的に価値ゼロとも判定しない。目的は、新primitiveの有無ではなく、科学的に意味がある明確な差が残るかである。

---

## 13. 最終方針と未完了事項

**優先順位**

1. 厳密基底状態のenergy-weighted応答を利用する、回路構造保持型の一体項再配分。
2. 非不変なPF・回路目的を使うDF因子等価回転。第一候補と同時に探索を拡大しない。
3. 分子構造で補正演算子が具体的に安くなる場合のcorrected PF。現時点は予備。
4. 低energy/state-dependent error解析は上記の補助として用い、それだけを新算法名にしない。

**今回の決定の強さ**

研究課題として、第一候補の構成可能性を最初に判別する価値があると判断した。独立新規性・実際の資源優位・論文化採択は未確定である。DF非一意性の一般論を第一候補にした以前の暫定提案を、具体的な一体応答更新に絞り直した。

**今回実施したこと**

固定GitHub資料の読取り、一次文献の関連箇所確認、解析的な数式導出、候補比較、研究方針の判断、Markdown記録の作成。

**今回実施していないこと**

新しい数値H、ground/励起eig、state expectationの数値取得、最適化、保存値の再集計、sampling、PF finite-h性能評価、compile、T合成、source test実行、全raw証拠の再認証、全先行論文の初出性証明。GitHubへのwriteも行っていない。

**新しい計算は未承認のまま、既存Mandatory STOPを維持する。** 次のCodex技術準備は今回決めた具体的な構成課題に集中させ、重要な科学結果が得られるまで通常の修正ごとに詳細レビューを反復しない。
