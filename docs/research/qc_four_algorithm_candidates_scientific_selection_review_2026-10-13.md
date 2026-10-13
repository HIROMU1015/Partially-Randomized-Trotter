# 量子化学ground-energy算法4候補：科学的比較・研究方針選定レビュー

- 日付：2026-10-13（日本時間）
- 開始承認：ユーザーの「レビューを開始して」。対象は直前に提示された4候補の比較・研究方針選定。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 主資料ref：`fc063696026b2500d9608abe447849fd76274ed5`
- 主資料：`docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md`
- 正式運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`。研究全体の俯瞰を含む更新本文を確認した。
- 今回行ったこと：固定GitHub文書・文献台帳・過去証拠索引・関連Track B文書の読取り、一次文献の追加確認、数式の解析的検討、候補比較、最初の判別作業の設計。
- 今回行っていないこと：新分子、SCF/FCI、固有値/SVD、BCHの数値評価・候補最適化、回路生成・compile・T合成、sampling、保存科学データの新しい数値集計、既存科学テストの再実行。
- 本文の区分：**資料の記載**、**外部一次資料の確認**、**本レビューの解析・判断・提案**を区別する。後二者をCodexの既存成果へ遡及して追加しない。

## 0. 結論

**C1を、最初の限定的な研究判別へ進める第一候補とする。C4は条件付きの次点、C2は重要な既存対照、C3は現段階では保留とする。**

ただし、独立した新規算法が既に確立したと認定する候補は0件である。「最初に検討する候補を決めた」ことと「主研究・論文の主要claimを採択した」ことは別である。

C1の問いを次に絞る。

> 取得可能な近似基底状態の情報から、基底固有位相の誤差を抑える固定macrocycleを構成し、同じ原Hamiltonian・energy精度・成功確率で、情報取得と実装費用を含めた利点を得られるか。

単にBCH係数を小さくする、回路を長くして時間刻みを細分化する、真の基底状態で最良の組合せを探す、という実験を成功としない。

次のCodex作業は、4候補の広い調査を反復することではなく、**C1の回路辞書・利用情報・比較・不採択条件を固定した、最小判別の実装と実行前準備**を一つにまとめること。新しい数値実行は別の明示承認が必要であり、本レビュー承認を流用しない。

C4には本レビューから重要な設計上の緩和提案がある。正確なcompiler費用を有限状態で表現することは不偏性の必要条件ではない。有限状態の代理費用を使っても、実際のproposal確率で重み付けすれば固定finite-RTEの平均は保持できる。ただし、これは現資料のexact-cost案への**追加提案**であり、実費用の最適性や改善率を自動的には保証しない。

## 1. レビューの目的と境界

研究目的は、元の量子化学Hamiltonianの基底エネルギーを、同じ絶対精度・信頼度で低コストに求めることである。PR、PF、DFを主要な資産とするが、特定手法を使用することは目的ではない。スペクトル強度解析への対象拡大は今回の主方向にしない。

ユーザーは、Codexで可能な技術調査を先に進め、GPTが重要な科学判断を行うことを希望している。今回の4候補には、数式、生成手順、既存手法との重複、費用、障害、最小検証案が既にある。したがって、同じ範囲の一般的な方法差監査をもう一度依頼するのではなく、この資料を使って優先順位と判別内容を決める。

全てを新規・有望と認定する必要はない。一方、既知要素を含むというだけで研究価値を否定しない。新しい構成、保証、取得計算量、同品質資源比較、適用条件の非自明な解明のどこに寄与を置くかが重要である。

「厳密certificateを得られない研究は全て不採択」とする新条件は追加しない。理論保証、仮定付き保証、結果前固定の経験的な設計と独立評価を区別する。経験的な情報を使った構成は研究し得るが、それを厳密な運用保証へ昇格させない。

## 2. 資料の取得と、確認の限界

### 2.1 GitHubから実際に取得した主要資料

基準refは上記`fc063696...`。

1. `docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md`（全体とC1–C4の該当範囲）
2. `artifacts/qc_algorithm_candidates/2026-10-13/literature_registry.json`
3. `artifacts/qc_algorithm_candidates/2026-10-13/existing_evidence_index.json`
4. 別ref `2dcd56737be35526f6436a52835f59d817adf30c` の `docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md`（C4と既存Track Bの境界確認）

研究全体の経緯は、上記主資料の保存結果整理、固定証拠索引、同一会話で取得済みのclosureと完了レビューに基づく。過去229ファイルを本レビューで全て再取得・再hashしたとは主張しない。主資料にある公開照合はCodexの報告として扱う。

このレビューで必要な4候補の数学構成と条件は取得できており、判断を止めるような必須GitHub資料欠落はない。

### 2.2 外部情報の利用

先行研究の比較は今回の承認範囲に含む。主資料の13件を唯一の全集合とせず、直接関連する一次論文を追加で確認した。

特に、C4に直接関係する **Resource-Optimal Importance Sampling for Randomized Quantum Algorithms, arXiv:2603.13495v1** は今回の13件台帳にはないが、過去レビューで触れていた重要文献である。今回はTheorem 1、一般回路費用、composite channelsの節を本文で確認し、比較に復帰させた。「今回新しく発見した算法」とは扱わない。

C2のChebyshev外挿論文はPDF本文を取得し、固有値外挿・gap条件・state preparationの関連箇所を読んだ。PDFページ22（0-based 21）のTheorem 15と回路図はスクリーンショットでも確認した。

先行性を全分野で決着させたレビューではない。C1とC4の研究上の差は未確定のまま残す。

## 3. 研究全体の振り返り

### 3.1 既存成果から得られること

主資料が整理した範囲では、N1は指定された合成入力・誤差込みselectorでexact fallbackが選ばれ、N2は限定小系で追加回路費用を正当化できなかった。N3は保証条件と厳密分離を保持しつつ現検証系列を終了した。THRIFT型H4は理想PFのstep減少を確認したが、登録templateと価格仮説ではknown費用を回収できず、現backend投資を見送った。

これらは分野全体の不能性ではない。一方、同じ構成を改名したり、少し大きい入力なら勝つかもしれないという理由だけで再開する根拠にもならない。

Track AのH4/H6の値は、各固定条件での有限時間complex signalと回路費用の結果である。今回、基底エネルギー全費用の既存優位を前提にしない。Track B D1も3-system-qubitのsynthetic finite-RTEであり、分子DFの実行成績ではない。

### 3.2 今回変えるべき研究の進め方

これまで繰り返し現れた問題は、途中の量の改善と最終taskの改善が一致しないことだった。

- 回路の一部分が安いが、誤差・測定費用を戻すと利益が消える。
- 保証幅は狭くなるが、固定taskでは安価な対照でも同じ決定ができる。
- 理想PFのstep数は減るが、一stepの実装費用が増える。
- 数学的な構成は正しいが、生成に必要な情報やoracleを安く取得できない。

今回の4候補も同じ障壁を持つ。したがって、実装の空欄を全て埋めるのではなく、**何が起これば資源改善が可能かを、先に判別する**。特にC1は、実行回路へ新しい非線形関数backendを必須とせず、既存PFを使って情報利用の価値を調べられる。この点を第一候補とする理由の一つにする。

## 4. 総合順位と、その意味

| 候補 | 判断 | 主な理由 | 未採択の部分 |
|---|---|---|---|
| C1 固定macrocycle | **限定的な最小判別へ進める第一候補** | 従来のPF/QPEに直結し、既存kernelで構成と費用を調べられる。真値なしの情報利用に解くべき問題がある | 新規性、近似情報の精度、finite-h利益、総費用 |
| C4 finite-RTE proposal | **条件付き次点・予備候補** | 平均を変えずproposalだけを変えるため、保証の意味を保ちやすい。有限状態生成の課題は具体的 | exact/代理費用と実回路の対応、range項、古典費用、Track Bとの差 |
| C2 固有値外挿 | **重要な対照。独立新主題には現在昇格させない** | 固有値外挿、gap、準備費用まで既存研究がある。C1への有力な対照になる | 新しいbranch証明、安い剰余取得、既知法を超える差 |
| C3 self-energy | **保留** | 数式は条件付きで成立するが、一般分子への接続、resolvent回路、繰返しground推定が未構成 | 明示的低費用constructor、一mode削減を上回る利益 |

C1の優先理由は、以前の「第二候補」を順番で繰り上げたからではない。新たな技術資料と外部文献を照合したうえで、対象の近さ、追加backend負担、検証の判別力を比較した結果である。

ただし、C1とC2の近接先行研究は強い。C1の成功可能性が確立したわけではなく、独立した方法差がない、または同費用対照への利益がないと分かれば区切る。

## 5. C1：固定macrocycleの評価

### 5.1 資料にある構成の支持範囲

主資料の定義を維持する。

\[
S_j(h)=\exp\{-ih[H+h^2D_j+O(h^4)]\},
\qquad
V_\theta(h)=\prod_{\ell=1}^{L}S_{j_\ell}(a_\ell h),
\]

\[
a_\ell\ge0,\quad\sum_\ell a_\ell=1,
\quad (j_\ell,a_\ell)\text{ はpalindrome}.
\]

同じ原Hamiltonianを一次生成子に持ち、basisを変える場合は共通frameへ戻す。この条件の下で、logの主要誤差は

\[
D_\theta=\sum_\ell a_\ell^3D_{j_\ell}
\]

となる。異なるkernel間の交換子は主要なh³項には新しく入らない。時間反転対称性によりlogの偶数次が消えるが、第五次以降の交差項まで消えるわけではない。

正しいground-connected logarithmの領域で、Hの非縮退groundを考えると、

\[
E_{0,\theta}(h)-E_0=h^2d_\theta+O(h^4),
\qquad d_\theta=\langle\psi_0|D_\theta|\psi_0\rangle.
\]

この整理は支持する。ただし、一般operatorとして四次PFになったという結果ではない。

### 5.2 直接の先行研究

- Kronenberger–Erakovic–Reiherは、異なるorbital basis/orderのpropagatorを時間列として組み合わせ、energy誤差の平均・相殺を調べている。本文§2.5には繰返し同一macro-patternを使うperiodic evolutionの解釈もある。したがって、「固定列として繰り返す」ことだけを差分にできない。
- Mehendaleらの出版版は、第二次Trotterのground-energy誤差を摂動的・近似状態情報で評価し、Hamiltonian partitioningと資源見積りへ使う。近似状態を使うことだけも差分ではない。
- Maxwellらは、diagonal BCH、compact commutator representation、近似固有状態による誤差推定を扱う。BCHの取得が全て新技術だとは扱わない。また、その長時間・小stepの漸近結果を、任意のfinite-h certificateとして引用しない。

C1に残り得る研究問題は、**必要な情報を安く取得し、有限精度・実回路費用・固定unitaryの位相を整合させた生成規則**である。しかし、この一文だけで独立した新規性が成立するわけではない。新しい寄与が何かを、最小判別で確かめる。

### 5.3 本レビューの解析：係数減少を資源改善と混同しない

以下は主資料の式から本レビューで導く評価である。数値データへの新しい採点ではない。

主要誤差が\(d h^2\)で、許容PF energy biasが\(\epsilon_b\)なら、剰余を無視できる領域では

\[
h\simeq\sqrt{\epsilon_b/|d|}.
\]

一macrocycleの費用がgなら、物理時間当たりの費用はg/hなので、共通の後段推定器・準備条件の下での主要比較尺度は

\[
\boxed{J=g\sqrt{|d|}}.
\]

従って、同じ近似の下でcandidateがbaselineを改善するには、

\[
\frac{|d_\theta|}{|d_B|}
<\left(\frac{g_B}{g_\theta}\right)^2
\]

が必要である。これは完全なQPE費用式ではなく、leading-orderの判別尺度である。係数取得・状態準備・confidence・合成を戻した比較は別途必要。

同じkernelをL回、それぞれh/Lで使えば、

\[
d_\theta=d/L^2,\quad g_\theta=Lg,
\quad g_\theta\sqrt{|d_\theta|}=g\sqrt{|d|}.
\]

つまり、\(1/L^2\)の誤差係数縮小は、単なる細分化なら資源改善ではない。C1の対照には、同gate予算のuniform subdivisionを必ず入れる。

### 5.4 本レビューの解析：同符号・加法費用ではleading-order利益が出ない条件

これは限定した仮定の下での解析的帰結であり、新規定理としての優先性は主張しない。

各\(d_j\)が同符号で非zero、各kernelの実行費用\(g_j\)が時間係数\(a_\ell\)に依存せず、総費用が加法的、fusionや準備変化を無視できるとする。\(x_j=|d_j|>0\)と置く。

\[
|d_\theta|=\sum_\ell x_{j_\ell}a_\ell^3.
\]

Hölder不等式から、

\[
1=(\sum_\ell a_\ell)^3
\le
(\sum_\ell x_{j_\ell}a_\ell^3)
(\sum_\ell x_{j_\ell}^{-1/2})^2.
\]

したがって、

\[
\boxed{
(\sum_\ell g_{j_\ell})\sqrt{|d_\theta|}
\ge
\frac{\sum_\ell g_{j_\ell}}
{\sum_\ell x_{j_\ell}^{-1/2}}
\ge \min_j g_j\sqrt{x_j}.
}
\]

正係数を非一様に最適化するだけでも、このモデルでは最良単一kernelの反復を超えない。palindrome制約は可行集合を狭めるので、この下側条件を無効にしない。

**限定**：逆符号のd、正確なd=0、finite-h剰余、gate fusion、時間係数に依存する合成費用、異なるstate preparation、非加法的費用では、この帰結をそのまま適用しない。全PFの不能性や実分子への普遍的結果ではない。

これにより、最初に調べる価値があるのは、(i) 実際の異符号による相殺、(ii) 共有・fusionの非自明な費用差、(iii)有限hを含む明確な別機構、のどれが存在するかである。

### 5.5 情報誤差は、符号相殺と別の問題

資料の

\[
b_\theta=
\left|\sum_\ell a_\ell^3\widetilde d_{j_\ell}\right|
+\sum_\ell a_\ell^3\eta_{j_\ell}
\]

は、各\(|d_j-\widetilde d_j|\le\eta_j\)を正当に取得した場合の十分boundとして支持する。

近似期待値の和をゼロにしても、第二項は残る。\(\eta_j\)が固定なら、一般に\(h^2\eta\)の誤差床が残り、無条件の四次収束ではない。ただし、四次にならなくても、目標精度で同費用対照より安ければ価値はあり得る。高次化の証明を研究成功の唯一の条件にしない。

また、同じ近似状態ρを使うなら、各kernelの推定誤差が任意に独立に動くとは限らない。本レビューの補足として、正当なtrace-distance上界\(d_\rho\ge\|\rho-|\psi_0\rangle\langle\psi_0|\|_1\)が既に得られる場合、

\[
|\langle\psi_0|D_\theta|\psi_0\rangle|
\le |\operatorname{Tr}(\rho D_\theta)|
+d_\rho\|D_\theta\|.
\]

これは\(\sum a^3\|D_j\|d_\rho\)より良い場合がある。ただし、\(d_\rho\)や\(\|D_\theta\|\)を無料で求められると仮定しない。単に独立box誤差で相殺を過度に失う可能性を指摘したもので、新しい保証取得算法を完成させたわけではない。

### 5.6 finite-hとground branch

本レビューの条件付き整理として、

\[
H_{\rm eff,\theta}(h)=H+h^2D_\theta+R_\theta(h),
\quad\|R_\theta(h)\|\le K_\theta h^4
\]

が有効なlog branch上で成り立ち、Hのground gapの正当な下界をγとし、\(h^2\|D_\theta\|\le\gamma/4\)とする。この場合、通常のground perturbationのSchur評価とWeyl評価から、

\[
|E_{0,\theta}(h)-E_0-h^2d_\theta|
\le h^4\left(K_\theta+\frac{2\|D_\theta\|^2}{\gamma}\right).
\]

概略：H+h²DのgroundをHのgroundと直交補空間へ分けると、補空間との分離は少なくともγ−2h²||D||である。off-diagonalによる二次shiftはh⁴||D||²/(γ−2h²||D||)以下。RはWeylでK h⁴以下を加える。

この式は、主要期待値を消した後も\(\|D_\theta\|\)・gap・剰余が残ることを示す。現在の候補について安いKやγを得た結果ではない。基底の非縮退・適切な対称性sector・phase unwrapping・有効logの前提が必要。

gapの認証がない初期の経験的検証を全面禁止するのではない。その場合は、数値referenceによる事後branch確認を運用入力と区別し、保証付きalgorithmとしては未閉鎖と報告する。

### 5.7 C1を進める最小判別

次の一まとまりで、機構と実用性の両方を判別する準備を行う。

1. 同じHに対する有限の実行可能kernel辞書を、ground truthを参照せず固定する。既知の純共役・巡回類似・二項Strangの自明なisospectral対を独立な異符号kernelとして数えない。
2. 使用する近似状態、その取得方法と費用、利用できる情報の精度を固定する。真E0・真ground・真gapはreference評価側に隔離する。
3. macrocycle生成規則とfallbackを固定する。正係数・palindromeを維持し、有限bitの係数誤差を含める。
4. 最良単一kernel、同費用uniform subdivision、既存の強い高次PF、既知のordering/basis混合と比較する。C2をzero-step外挿の重要な別経路として位置付ける。
5. 小規模な数理fixtureだけでなく、可能なら既存のidentityを確認できる分子入力に接続する設計を用意する。具体的な分子、geometry、basis、DF policy、精度、input数は今回未固定で、Codexの実行前planで確定する。
6. 近似状態が不十分な対照、同符号で相殺できない対照を用意する。都合のよい反対符号をtruthから探索した入力だけを成功例にしない。
7. 出力するのはleading係数だけでなく、有限hのenergy bias、branch/overlap、固定gate予算、情報取得、state-prepを含む費用の成立範囲。

C1に純粋な新規primitiveを要求しない。同じ出力品質での有用な生成原理・適用条件・費用上の寄与が示されれば研究となり得る。ただし、既知構成の微調整しか残らず、意味のある資源利益や理論差分がなければ区切る。

## 6. C2：PF固有値外挿の評価

### 6.1 数学の支持範囲

資料の二点外挿

\[
\widehat E_{\rm ext}=(4\widehat E(h/2)-\widehat E(h))/3
\]

は、同じground-connected branchで\(E(s)=E_0+c_2s^2+R(s)\)、\(|R(s)|\le Cs^4\)なら、

\[
|E_{\rm ext}-E_0|\le\frac{5}{12}Ch^4
\]

となる。数式は支持する。単一unitaryを負係数でcoherently足すことではなく、異なるunitaryから得たenergy推定値の古典処理である。

コストモデルが\(A_i/\epsilon_i\)の場合の配分式

\[
C_{\min}=\frac{(\sum_i\sqrt{A_i|w_i|})^2}{\epsilon_{\rm stat}}
\]

も条件付きで支持する。confidenceの配分、state準備、branch確認、離散的なquery数を含む場合は別途見直す。全ての推定器がこのコスト則を持つとはしない。

### 6.2 先行研究との重複は特に強い

Rendon–Watkins–Wiebe, Quantum 8, 1266 (2024) は、固有値と時間発展期待値のChebyshev外挿を扱い、有限step解析、gap、ground state preparationの費用まで取り扱う。

したがって、「外挿にbranchやstate-prep費用を足す」という一般的説明では既知研究を超えない。C2の差を主張するには、例えば既存gap仮定を置き換える安い情報取得、より弱い前提でのbranch証明、非自明なfinite-cost配置など、具体的な結果が必要である。現在のdossierにはその取得算法が未定義である。

### 6.3 観測した点だけではbranch/剰余を保証できない

本レビューの説明として、有限のstep集合\(h_1,\ldots,h_m\)の値だけを与えても、解析的で偶関数の補正

\[
A\prod_{i=1}^{m}(h^2-h_i^2)
\]

は全観測点でゼロだがh=0では一般にゼロでない。従って有限点の一致だけからzero-step値の誤差幅を認証できない。Hamiltonian構造、級数剰余、解析性領域などの追加情報が必要。

また、phase区間が近いことや小さい残差は、同じground branchであることを自動保証しない。取り違えや未観測の低い固有値を除外する情報が必要。この問題は外挿の式とは独立である。

### 6.4 判断

C2を捨てるのではなく、**C1に対する有力で必須級の既存対照として保持する**。独立主題への昇格は、新しいbranch/剰余取得が定義された場合に再検討する。C1採択と同時にC2全実験を別系列で開始しない。

## 7. C3：self-energyを保持するactive-space構成の評価

### 7.1 形式は正しいが、対象classを広げない

資料の初期設計は、N3に由来する一外部mode・固定粒子数・one-body＋非負densityである。一般二体量子化学Hamiltonianの効率的constructorまで完成していない。

\[
H=\begin{pmatrix}A&B\\ B^\dagger&D\end{pmatrix},
\quad K(E)=A-B(D-EI)^{-1}B^\dagger.
\]

\(E<c\le\lambda_{\min}(D)\)ならSchur inertiaにより、ground energyは

\[
f(E)=\lambda_{\min}(K(E))-E=0
\]

で特徴付けられる。最小固有値の縮退点で通常微分を使えないこともあるが、Loewner順序で

\[
E_2>E_1\Longrightarrow
f(E_2)\le f(E_1)-(E_2-E_1)
\]

が成り立ち、適切なbracket内でrootは一意である。これは既知Schur/Feshbach原理の帰結であり、candidateの新規定理としない。

### 7.2 resolvent次数の依存を明示する

\(g_{\rm ext}=c-U>0\)、\(W=d-c\ge0\)と置く。資料のNeumann近似は

\[
\|\Sigma(E)-\widetilde\Sigma_\ell(E)\|
\le \frac{\beta^2}{c-E}\rho_E^{\ell+1},
\quad \rho_E=\frac{d-c}{d+c-2E}.
\]

従ってE≤Uでは、

\[
\|\Sigma-\widetilde\Sigma_\ell\|
\le\frac{\beta^2}{g_{\rm ext}}
\left(\frac{W}{W+2g_{\rm ext}}\right)^{\ell+1}.
\]

W>0で所定の\(\epsilon_{\rm self}\)を満たすには、比が1未満の適切な場合、

\[
\ell+1\ge
\frac{\log[\beta^2/(g_{\rm ext}\epsilon_{\rm self})]}
{\log[1+2g_{\rm ext}/W]}
\]

が十分。右辺が非正なら次数0でもよい場合があり、W=0やβ=0は別に自明処理する。\(g_{\rm ext}\ll W\)では次数が外部分離の逆数に依存して増え得る。

これは資料の式を整理した条件で、今回の入力へ数値代入した結果ではない。外部分離\(c-U\)と物理的なground spectral gapを同じものとして扱わない。

### 7.3 一modeを消しても、外部計算が消えるわけではない

poly(D)、B、B†のアクセス、normalization、QSVT/LCU補助qubit、各bracketでのK(E)のground推定が必要になる。古典行列でresolventを作れたことを、低費用量子oracleの証拠としない。

さらに、単一のactive ground solverを呼ぶ問題ではなく、E依存Hamiltonianの最小固有値を繰り返し判定する問題となる。状態準備とoverlapの費用をその都度freeにはできない。

真のgroundのP成分がゼロでないことはE<cから分かり、\(\|Q\psi\|\le(\beta/g_{\rm ext})\|P\psi\|\)という解析的情報も得られる。しかし、それはactive groundを安く準備する手順の完成ではない。

### 7.4 先行研究と判断

DUCCによるactive-space phase estimationと外部相関回復は既存研究である。Exact downfoldingの研究もeffective interactionsとcontrolled approximationを扱うが、今回の有限粒子sectorのSchur rootと同じ算法とはしない。

**現在のC3は保留**。数学的形式は研究の入口として意味があるが、安いself-energyの表現や一般分子へ適用する具体的機構がないまま、一外部modeの新しいsynthetic例を増やしても、現在の研究目的に対する中心障壁は解けない。N3のclosureを自動解除しない。

## 8. C4：回路列費用を反映するfinite-RTE proposal

### 8.1 意味論を保つ点は支持する

固定finite approximationを

\[
F=\sum_\gamma a_\gamma U_\gamma
\]

とする。実際に使うproposal\(q_\gamma>0\)と、実際の確率に基づく重みで推定すれば、固定Fに対する平均を保持する。変更されないのはFの平均であって、原Hamiltonianに対する有限Taylor/PF biasがゼロになるわけではない。

complex phaseは対応するcontrolled回路・測定軸へ組み込む。以下のsecond-moment評価では\(|a_\gamma|\)を使う。random-unitary channelの誤差boundを、線形振幅推定へそのまま流用しない。

### 8.2 最適確率原理は既知である

資料の

\[
M(q)=\sum_\gamma |a_\gamma|^2/q_\gamma,
\qquad G(q)=\sum_\gamma q_\gamma g_\gamma
\]

に対する

\[
q_\gamma^*\propto |a_\gamma|/\sqrt{g_\gamma}
\]

は、一般的な費用・second-moment最適化である。

今回再確認したCugini–Atif–Subaşı, arXiv:2603.13495v1 は、任意の回路実行パラメータに対する一般費用関数を扱い、Theorem 1に同じ平方根則がある。whole-pathを一つのパラメータとすることもその一般則の外ではない。従って「項ではなく回路列に重要度を付ける」という説明だけでは差分不足である。

既存2023年のcost-aware qDRIFTも関連するが、それだけを比較して2026年の直接の一般化を落とさない。

### 8.3 generatorの残る問題

資料の有限状態費用、positive quadrature、backward DPという構成は条件付きで理解できる。ただし、状態数Sがcompilerのinverse stackやGaussian全履歴を表すために指数増大するなら、\(O(JrS^2)\)という演算回数は効率性を示さない。これは算術演算回数で、分数bit長や正規化費用も別に必要。

さらに、qのnormalizerをDPで得られても、M(q)の和は一般に1/qを含み、同じDPで安く正確に求まるとは限らない。sample生成、q評価、moment/rangeのcertificationはそれぞれ区別する。

### 8.4 本レビューの設計上の緩和提案：代理費用を許せる

**これは現資料のexact-cost条件を変更する追加提案であり、既に実装・検証済みと扱わない。**

proposal生成に使う費用が実際のcompile費用そのものである必要はない。安く計算できる正の\(\widehat g_\gamma\)を使っても、実際のqで重み付けすれば平均恒等式は保てる。

その場合の問題は、不偏性ではなく、実費用での性能予測である。もし全supportで

\[
0<m g_\gamma\le\widehat g_\gamma\le M g_\gamma
\]

が正当に得られ、\(\widehat q\propto |a|/\sqrt{\widehat g}\)を正確に生成できるなら、Cauchy–Schwarzの最適値を\(\mathcal B^*=(\sum|a|\sqrt g)^2\)として、

\[
\begin{aligned}
M(\widehat q)G(\widehat q)
&=(\sum_\gamma |a_\gamma|\sqrt{\widehat g_\gamma})
(\sum_\gamma |a_\gamma|g_\gamma/\sqrt{\widehat g_\gamma})\\
&\le\sqrt{M/m}\,\mathcal B^*.
\end{aligned}
\]

これは既知最適式からの単純な条件付き導出で、新規性を認定しない。実際のratio boundが分からなければ、この性能保証を使えない。finite quadrature・dyadic lawが理想\(\widehat q\)と違う場合には、その違いも評価する。

この緩和でC4を救済済みとは言わないが、「正確なcompiler全履歴を有限状態で表せなければ、proposal研究そのものが不可能」という強すぎる条件は避けるべきである。

### 8.5 改善余地とconfidenceの壁

\(\Gamma=\sum|a_\gamma|\)、\(q_0=|a|/\Gamma\)とすると、second-moment×期待費用について

\[
\frac{\mathcal B(q^*)}{\mathcal B(q_0)}
=\frac{(\mathbb E_{q_0}\sqrt g)^2}{\mathbb E_{q_0}g}
=1-\frac{\operatorname{Var}_{q_0}(\sqrt g)}{\mathbb E_{q_0}g}.
\]

この原理と費用集中時の利点減少は既存文献にある。費用がほぼ一定、共通state preparationが支配的、または多数の独立加法費用が集中する場合、理想proposalでも大きい改善は期待できない。ただし、希少tailや強い依存のある現PRへ集中の仮定を無断で適用しない。

Track B D1はBernstein十分shot予算を使用している。second momentだけでなくrange\(R=\max|a/q|\)の項、failure配分、accepted-tailの費用もある。M(q)G(q)が小さくなっても、同じ信頼度での総資源が減るとは限らない。

whole-path defensive mixtureはsupport/range保護に使えるが、\(q\ge\zeta q_0\)から導く粗いboundだけでは利益を失う可能性がある。また、量子計算を始める前のzero-fill判定は、通常のquantum shotと費用が違う。既存D1の意味論を保つ必要がある。

### 8.6 判断

C4は、**有限状態または代理費用を使う生成器に、実際のPR回路と信頼度付き会計を接続する課題**として残す。ただし、一般最適確率則だけを主張する方向は採らない。

C1と並列に新しい大きい開発系列を起動しない。C4を進める場合は、既存Track B D1/D2との担当・source・契約の境界を先に確認する。今回の別候補比較は、他Trackの新しい認可を上書きしない。

## 9. 共通する最終評価

全候補で、元H、粒子数sector、絶対energy精度\(\epsilon_E\)、全failure\(\alpha\)を共通にする。対象変更、active Hamiltonianへのすり替え、oracle情報の無料利用を避ける。

ただし、初期の最小判別から全アルゴリズムの実機実証を要求しない。次の層を分ける。

1. 数学的機構：何を保ち、何が減るか。
2. 情報取得：候補を生成するための情報、費用、誤差。
3. 実装費用：固定回路・制御・basis・合成・workspace。
4. 基底energy task：branch、alias、overlap、推定器、confidence。

各層がどこまで実証されているかを明記する。後の層が未評価なら、前の層の結果は保持するが、総資源優位へ昇格しない。

C1/C2では固定unitaryのpowersという意味を維持する。C4では有限振幅推定の平均と推定器を維持する。C3ではE依存有効Hamiltonianの最小固有値推定を明示する。四候補の異なる出力を一つのPF-error表へ無理に並べない。

## 10. 次の担当と、作業単位

**次の担当：Codex。C1の一つの限定判別を、設計・実装・登録外テスト・固定planまでまとめて準備する。**

含めるもの：

- 本レビューの順位・新規性未確定・既存closure保全を追補。
- C1の実kernel辞書、同H/frame、palindrome、aの有限精度、利用可能近似状態を具体化。
- 単なる細分化、同符号・加法費用の条件、同値kernelを除く解析を独立に確認。
- oracle評価側と運用生成側を分離した実装。
- 同費用uniform subdivision・高次PF・既知ordering/basis混合・C2対照の役割を定義。
- 数値誤差、PF剰余、branch、情報不確かさ、費用のどこまでを最小pilotで扱うか固定。
- 入力、precision、calls、memory/output、source・checker・one-shotの境界を決めて公開。

Codexが裁量で決めてよいもの：ファイル構成、cache、source再利用、テストの細部、並列化、効率的なBCH取得、保存形式。科学対象・主対照・保証の意味・運用に利用できる情報を変更する場合は、変更理由をまとめてGPTへ戻す。

**新しい科学的数値計算、FCI/reference eigensolve、PF性能評価、sampling、compile/T合成は本レビューでは実行認可していない。** 既存one-shotを再利用せず、別の固定scopeとユーザー実行承認が必要。

実装・off-domain技術検査・公開が完了しただけで同じ科学レビューを反復しない。新しい科学的結果、重大な同値性/不整合、または主要契約を変える必要が生じたときに戻る。

## 11. C1の採択・再設計・終了の判断条件

### 継続を支持する場合

- 真値を運用入力へ混ぜずに有用なmacrocycleを生成できる。
- 同gate予算の単純細分化・最良固定kernel・強い高次PF・既知相殺構成との比較で、誤差と費用に意味のある差が残る。
- 有限hで利益が残り、branchや状態overlapの変化が無視されていない。
- 情報取得を含めても研究目的へ結び付くか、独立した非自明な理論結果が得られる。

### 区切る、または再設計する場合

- 観測された改善がuniform subdivisionやknown compositionで説明され尽くす。
- ground truthで符号や係数を選ぶ場合だけ成功する。
- 小さいleading coefficientが大きいfinite-h剰余・情報誤差・gate overheadに打ち消される。
- 独立した構成/保証/計算量/実用的結果のどこにも研究上の差を置けない。

成功例が出るまで入力、kernel辞書、近似状態品質、精度を追加し続けない。失敗条件は固定有限範囲の結果として保存し、全PF・全分子の不能性に拡張しない。

## 12. 先行研究確認の範囲と未解決点

### 本文・関連数式を確認したもの

- C1 dossier（GitHub）：数式、情報仮定、費用、失敗条件。
- Kronenberger et al. arXiv:2602.18913v1：§2.5、§4.3、結論。出版社のMolecular Physics公開本文も確認。版本を同じ本文として自動同一視しない。
- Maxwell et al. arXiv:2606.30738v1：diagonal BCH・漸近結果・compact展開の関連節。
- Rendon et al. arXiv:2212.14144v4：PDFの固有値外挿、gap、state preparation。Theorem 15のページは画像でも確認。
- Cugini et al. arXiv:2603.13495v1：Theorem 1、net-cost、composite channels、bias保存。
- Kiss et al. arXiv:2212.05952：PDF取得。今回の主なC4差分判断は、より直接的な2026年論文とdossierを中心に行った。

### 検索取得の本文・abstract/書誌中心

- Mehendale et al. Digital Discovery 2025出版版：出版社検索抽出のabstract、方法・結果の該当部分。直接ページ再取得は失敗しており、全出版版の独立全文監査とはしない。
- Exact downfolding：arXiv v4 HTML直接取得は失敗。PR Research 8,023091 (2026) の出版社abstract/書誌を確認。同一のSchur-root算法と断定しない。
- DUCC active-space PE：AIP出版社abstract。全文の方法差監査は未完了。
- PR：PRX Quantum 7,020332の出版社abstract/書誌を確認。arXiv v2の全定理を再監査していない。
- DFTHC spectrum amplification：arXiv abstract。追加のresource比較は行っていない。
- 検索で見つかったcorrected PFやtime-dependent PFの出版情報は、近接する外部文献として記録するが、全文確認をしていないため本候補の包含・性能判定には使わない。

主資料のL12/L13のようなabstract中心の新しいrandomization文献について、今回、全理論の独立再確認までは行っていない。未読を新規性の証拠にしない。

書誌上、PRの出版社情報は2026-05-19であり、台帳の2026-07-10はarXiv v2の時点と区別すべきである。またExact downfoldingには2026-04-29の出版版がある。これらは版管理の追補で、科学結果を変更するものではない。

## 13. 最終決定

**C1を、元の量子化学ground-energy目的に最も直接的な第一候補として、一回の小さい判別へ進める準備対象にする。**

新規性の認定は保留する。C1の主要期待値が減ったという結果だけでは採択しない。同費用の細分化・既知相殺・強いPF/外挿対照、情報取得とfinite-hを含めて評価する。

**C4は次点。** 正確な費用状態の圧縮だけでなく、代理費用と正しい確率評価という可能性も残す。ただし既知の最適importance式を新規とせず、confidence・classical費用・実回路への接続を問う。

**C2は有力な対照、C3は保留。** これらを低価値と一般化せず、現構成における新しい寄与と実装上の隔たりに基づいて優先順位を付ける。

N1 STOP、N2 HOLD、N3 CLOSED、mixed-access現templateのclosure、別Trackの既存状態と認可は維持する。既存global manifestのdescription欠落FAILを、本レビューで修正・無視・PASS化しない。

---

## 参考資料（固定リンク）

### GitHub資料

- [4候補技術比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md)
- [文献台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/artifacts/qc_algorithm_candidates/2026-10-13/literature_registry.json)
- [固定過去証拠索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/artifacts/qc_algorithm_candidates/2026-10-13/existing_evidence_index.json)
- [Track B D1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md)

### 外部一次文献

- Kronenberger, Erakovic, Reiher, *Trotter Error and Orbital Transformations in Quantum Phase Estimation*: [arXiv v1](https://arxiv.org/html/2602.18913v1), [出版社](https://www.tandfonline.com/doi/full/10.1080/00268976.2026.2681062).
- Mehendale et al., *Estimating Trotter approximation errors to optimize Hamiltonian partitioning for lower eigenvalue errors*, Digital Discovery 4,3540–3551 (2025): [DOI](https://doi.org/10.1039/d5dd00185d).
- Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*: [arXiv v1](https://arxiv.org/html/2606.30738v1).
- Rendon, Watkins, Wiebe, *Improved Accuracy for Trotter Simulations Using Chebyshev Interpolation*, Quantum 8,1266 (2024): [arXiv v4 PDF](https://arxiv.org/pdf/2212.14144v4), [DOI](https://doi.org/10.22331/q-2024-02-26-1266).
- Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*: [arXiv v1](https://arxiv.org/html/2603.13495v1).
- Kiss, Grossi, Roggero, *Importance sampling for stochastic quantum simulations*, Quantum 7,977 (2023): [arXiv](https://arxiv.org/abs/2212.05952), [PDF](https://arxiv.org/pdf/2212.05952).
- Profe et al., *Exact downfolding and its perturbative approximation*, Physical Review Research 8,023091 (2026): [出版社](https://journals.aps.org/prresearch/abstract/10.1103/npv1-w4gt).
- Bauman, Low, Kowalski, *Quantum simulations of excited states with active-space downfolded Hamiltonians*, JCP 151,234114 (2019): [出版社](https://pubs.aip.org/aip/jcp/article-abstract/151/23/234114/315571/Quantum-simulations-of-excited-states-with-active).
- Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7,020332 (2026): [出版社](https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq).
- Low et al., *Fast quantum simulation of electronic structure by spectrum amplification*: [arXiv abstract](https://arxiv.org/abs/2502.15882).

このMarkdownはGPTの比較・判断記録であり、Codexの新実験結果・新規性認定・計算実行認可ではない。
