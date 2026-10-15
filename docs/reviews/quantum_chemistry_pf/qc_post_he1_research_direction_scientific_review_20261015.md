# H-E1 Stage 0後の研究方針と固定回路係数修正PF（M案）の科学的レビュー

**量子化学Hamiltonian simulation：第一候補・H-E1の整理と次の限定技術監査の判断**

| 項目 | 内容 |
|---|---|
| 日付 | 2026-10-15 JST |
| 区分 | ユーザーの明示的な開始承認を受けて実施したGPT重要科学レビュー |
| 研究の目的 | 電子構造Hamiltonianの基底エネルギー推定に用いるproduct formula（PF）時間発展を、実装可能性と資源費用を踏まえて改善する独立したアルゴリズムを検討する |
| 今回の直接対象 | Claudeの最新レビュー（2026-10-15）、H-E1 Stage 0技術監査、第一候補の確定済み有限step評価、M案のK0実施価値、および **A：簡潔な成果整理 → B：M案のK0** という作業順序 |
| 最新研究リポジトリ | `HIROMU1015/Partially-Randomized-Trotter`、固定commit `0e923f73754d1ebd168bf21a1b9e73971946e603` |
| 外部レビュー | アップロードされた `レビュー_outer(1).md`（Claude、716行程度。UTF-8原本SHA256 `6c3b5d523cfddb87eece62c923096af40e8e9421707fe6e1e0114afedf478e4b`） |
| 実施境界 | 静的な数学的考察・GitHub本文取得・公開一次文献の確認。**新しい科学数値計算、保存値の新しい科学的再集計、compiler/T合成、repository変更を実施していない** |
| 科学的結論 | **Aを簡潔に行い、その後BのK0を一回限定で行うことを推奨。M案自体は未採択、科学計算は未承認** |

---

## 0. 結論の要旨

### 0.1 科学判断

**【GPT判断】Claudeの前回レビューの基本的な結論を採用する。** 第一候補の現在の固定構成には今後の研究投資を推奨しない。H-E1についても、現定義のまま独立した新アルゴリズムの研究として続ける根拠はない。実験結果・数式・STOP履歴は否定せず、既存資料への参照で再利用可能な形に保持する。

**【GPT判断】次の研究候補M案は、具体的な検証対象としてはH-E1より一歩進んでいる**。M案はfragmentの指数演算で使う係数をstep幅の二乗に依存させ、PFの実効Hamiltonianに現れる誤差演算子を、既に指数化可能な演算子成分によって部分的に打ち消そうとする。

しかし、**$h^2$の係数修正、modified potential、BCH誤差の打消し、PFパラメータの古典的最適化という発想は強い先行研究を持つ**。また、DFの回路に一般の密度二体係数を追加しても「ゲート数が不変」とは一般には言えない。M案は正式採択せず、これらを一度の**計算を伴わないK0監査**で判断する。

### 0.2 今回決めること／決めないこと

| 対象 | 今回の判断 | 確信度・境界 |
|---|---|---|
| 第一候補（固定first pair・一方向・$d$最適化） | **現構成への投資を区切ることを支持** | 高。限定した構成の意思決定であり一般no-goではない |
| H-E1（独立なPF・補正情報取得共同設計） | **現定義の独立研究として進めない** | 高。K0で固有生成規則が未特定。普遍的な不存在証明ではない |
| A：成果整理 | **実施を推奨** | 高。既存資料の索引と状態を簡潔にまとめるだけで十分。論文化は不要 |
| B：M案K0 | **Aの後に実施を推奨** | 中。新規性の蓋然性が高いという意味ではなく、一回の静的比較で識別できる問いがある |
| M案の研究採択 | **未採択** | 高。K0で数学・先行研究・native回路条件を解決する前にGOを出さない |
| M案のK1、数値pilot、追加分子、compile | **未承認・実施しない** | 明確。別途科学判断と計算実行承認を要する |
| 第一候補のGitHub正式 `CLOSED` 変更 | **ユーザーの採用判断待ち** | 現時点の科学レビュー推奨を、GitHubでの正式status変更承認へ読み替えない |
| DF因子等価回転 | 保留を維持 | M案のK0を行うことを理由に並行して再開しない |

**推薦する一連の作業**：CodexがAの簡潔な研究引継ぎ記録を作る → 同じ依頼でM案K0を静的に実施 → 元資料と監査結果をGitHubへ保存・再取得照合 → Mandatory STOP。K0で明白な既知法への還元が確認された場合は、その結果を記録して終了する。K0が通ってもpilotやK1を自動的に起動しない。

---

## 1. 使用した資料と、資料から判断できる範囲

### 1.1 今回実際に使用した主要資料

| ID | 資料 | 確認範囲 |
|---|---|---|
| C1 | 添付 `レビュー_outer(1).md`（Claudeによる最終レビュー） | 本文、M案§7.4、H-E1§5、研究着地点§8、次の作業§10。中断後に再開して最終ファイルを作成した記録を含む |
| G1 | [H-E1 Stage 0技術監査、commit 0e923f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/research/qc_he1_stage0_technical_prevalidation_20261015.md) | Github connectorで本文取得。K0の直接比較、既知B*、第1・第2研究との境界、K1–K3 STOPを確認 |
| G2 | [H-E1 GPT引継ぎ、同commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/research/qc_he1_stage0_gpt_handoff_20261015.md) | Github connectorで本文取得。六つの問い・承認境界を確認 |
| G3 | [H-E1 method status、同commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/artifacts/qc_he1_stage0_prevalidation/2026-10-15/method_status.json) | Github connectorでJSON本文取得。K0 `gate_pass=false`、K1–K3未実施を確認 |
| G4 | [第一候補の統合完成版科学レビュー、同commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/reviews/quantum_chemistry_pf/qc_onebody_repartition_h_boundary_final_scientific_review_20261015.md) | Github connectorで本文取得し主要判断・境界を確認。ローカル同名原文は100,617bytes |
| G5 | [固定H4 h境界checker v2報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md) | Github connectorで本文取得。比較・境界・監査結果を確認 |
| G6 | [H-E1固定の前の方法差監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/docs/research/qc_ground_pf_final_method_difference_audit_v1_20261014.md) | 過去のGitHub取得・Claude最終レビューを参照。今回全検査を再実行したわけではない |
| G7 | [H-E1文献取得・固定参照台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/artifacts/qc_he1_stage0_prevalidation/2026-10-15/literature_registry.json) | G1・C1に記録された固定文献取得状態を照合するための索引。全論文を今回再証明していない |

第1・第2研究、Track A/B、C1–C4などの詳細なhistoryはG1・G4・C1および各固定GitHub参照を根拠とする。今回それらの全個別実験を新規再現・全件再監査したものではない。

### 1.2 今回追加で確認した外部研究（本文・要旨の範囲）

- **[Mansuroglu, Fischer, Hartmann, *Problem-specific classical optimization of Hamiltonian simulation*, Physical Review Research 5, 043035 (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043035)**。出版社の全文表示で、パラメータ化したPF系列を量子最適化なしに古典的な摂動的費用関数で最適化し、固定ゲート数で通常のPFより良い条件があることを確認。**M案との同値証明ではないが、同じgate grammarの係数最適化一般は既知**。
- **[Bagherimehrabほか、*Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, arXiv:2409.08265v4／PRX Quantum (2026)](https://arxiv.org/html/2409.08265v4)**。2026年7月版のHTML本文を確認。演算子補正、symplectic／symmetric／composite corrector、実装費用・分解誤差は既存の強対照。ただしM案の「元fragment角度だけに吸収する」条件が既報そのものと断定しない。
- **[Blanes, Casas, González, Thalhammer, *Efficient Splitting Methods Based on Modified Potentials*, Commun. Comput. Phys. 33 (2023)](https://www.global-sci.com/cicp/article/view/7160)**。出版社の説明で、double commutatorとmodified processorを用いて高次分割公式を作ることを確認。個別式と化学DFの実装条件まで今回再証明していない。
- **[Blanes, Casas, Murua, *Splitting methods for differential equations*, Acta Numerica 33 (2024)](https://www.cambridge.org/core/journals/acta-numerica/article/splitting-methods-for-differential-equations/2C1BD434F8D2052E593A755351B3ACA3)**。公開総説の書誌・改良splitting研究の広さを確認。M案のK0ではmodified potential／force-gradient／shadow-Hamiltonianの先行式を本文と対応させる必要がある。
- **[Casaresほか、*Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 (2026)](https://arxiv.org/abs/2606.30741)**。化学向けnear-integrable PF、DF表現・回路コストの強対照。今回 arXiv HTML v1 は取得できなかったため、細部はC1・G1の固定本文照合記録に依拠した。
- **[Carenaほか、*Improved Hamiltonians for Quantum Simulations of Gauge Theories*, arXiv:2203.02823 (2022)](https://arxiv.org/abs/2203.02823)**。別の対象／誤差源に対してHamiltonian補正を使う関連研究としてC1・G1に記載。直接同一方法と扱わない。

**資料取得と判断の強さの区別**：本文を今回直接読んだ論文、出版社要旨・総説の索引だけ確認した論文、Codex／Claudeの監査を経由している論文を混同しない。M案の初出性を網羅的に否定または肯定したレビューではなく、**K0を実施する価値と評価条件についての科学判断**である。

---

## 2. 第一候補（一体項再配分PF）の科学的評価

### 2.1 成果として残ること

固定対象はH4、1.00 Å、STO-3G、DF rank12、全density fragmentの決定論的PF。最外側一体fragmentと最初のdensity fragmentの間で、元Hを保ったまま一体項を移し、厳密基底状態上の主要固有位相誤差の変化分を1-RDMから作る。

既存資料では、leading係数が元構成の `0.0133665` から応答選択後の `0.00820068` へ低下した。未補正S2の許容hも主要項則と整合した。これは**数学構成と小分子の有限step挙動の整合**を示している。1-RDM応答式やHamiltonian保存の正しさは有用な技術的記録である。

ただし、同じground恒等式・CAR・cacheを許したcompact BCH対照に対して、応答生成の独占的な演算量改善はない（G6）。この限定的な還元を、閉式の歴史的既出性や再配分一般の不可能性へ拡張しない。

### 2.2 なぜ現構成を区切るか

比較窓は最大 $h=1\ \mathrm{Ha}^{-1}$ まで検証され、候補・対照の局所許容h境界を識別できた。保存したh境界の費用比は次のとおり（**元資料の転記**）。

| target | static price | response／strong control | 事前登録の基準 |
|---|---|---:|---|
| $10^{-4}$ Ha | P1 | 0.9709028656 | 10%改善未達 |
| $10^{-4}$ Ha | P2 | 0.9715851442 | 同上 |
| $10^{-5}$ Ha | P1 | 0.9742728054 | 同上 |
| $10^{-5}$ Ha | P2 | 0.9749574508 | 同上 |

最強対照は `CS2:known_linear_extraction`、primary候補は `CS2:response_selected` だった。全4,572セルのchecker v2がPASSしているという**保存監査記録**があり、旧の停止・修復履歴も残る。今回は新たな数値監査を実施していない。

この比較が意味するのは、**一件のdevelopment H4・登録h grid・事後oracle・静的controlled-primitive／Heisenberg query proxyで、理想情報のもとでも事前基準の10%削減に達しなかった**ということである。未計算のT合成、情報取得費用、実際のQPE/RPE資源の優位も認定できない。

構造上、選択規則が小さくする量はS2の $h^2 d_X$ だが、比較を支配したCS2は主要項を差し引き、残りの $h^4$ 誤差係数と回路費用が効く。$d_X$ の最小化はCS2残差を直接最適化しない。この**設計目標と最良対照のずれ**が、追加の同型pilotを推奨しない主因である。

**【判断】現first-pair・一方向・leading $d$ 選択の構成への追加研究投資を区切る。** 必要な議論は記録化するが、新しい分子への事後救済や回路compileを開始しない。これは一般の一体項再配分や別ordering・多方向改良の数学的不能性ではない。

---

## 3. H-E1 Stage 0の評価

### 3.1 技術監査が解決したこと

CodexのK0では、PF選択・grouped BCH・補正係数の測定／古典縮約・scalar correction・外挿・誤差予算配分という具体化可能な処理を、既存のアルゴリズム・文献・第1・第2研究に対応付けた。固定statusは次である（G3）。

| 対象 | 保存判定 |
|---|---|
| 具体化できる既知ワークフロー | `REDUCES_TO_KNOWN` |
| H-E1全体の新規性 | `UNDETERMINED` |
| 独立した実行可能な生成規則 | `NOT_IDENTIFIED` |
| K0進行ゲート | `gate_pass=false` |
| K1〜K3 | `NOT_EXECUTED_K0_NOT_PASSED` |

**【GPT判断】これを支持する。**「PFの設計と補正情報の取得費用を共同最適化する」は研究目的であり、それだけでは入力からPFと測定器を作る新しい構成規則にならない。未定義の生成器を含む全可能性が既知法へ還元されたわけでも、一般的なno-goが証明されたわけでもない。

### 3.2 Claudeの情報価格・精度スケーリングの補足の評価

Claudeは厳密基底状態の提供方式を、コピーごとに価格を課す場合、準備unitaryまで利用できる場合、コピーが実質無料の場合に分けた。**これはH-E1の仮定を曖昧なまま費用比較しないために重要で、採用する。** コピーが十分安ければ、元のHの期待値を直接評価する方法自体が対照になり得るため、「厳密状態の利用可能性」と「QPEが当然最安」を同一視しない。

また、CS2の絶対補正係数精度は、$h$ に対して概ね $\eta_d\lesssim\epsilon_{\rm info}/h^2$ を要する。$h\asymp\epsilon^{1/4}$ の4次残差支配という**追加仮定のもと**で、$\eta_d\asymp\epsilon^{1/2}$ となる。固定分散・独立試料・状態コピー単価が精度に依存しない通常サンプリングならコピー数は $O(\epsilon^{-1})$、一方登録query proxyでは $O(\epsilon^{-5/4})$ と見積もられる。しかし、これを**情報取得が一般に律速にならないという定理**と扱わない。分散、演算子分解、成功確率、sector・状態準備・合成費用は系サイズ・精度に依存し得る。

Chebyshev等の多点補間は、必要な解析領域・スペクトル情報・branch同定・状態準備条件を満たす場合に強い競合となる。そこで理想的な$d$の**追加価値が一定倍程度しか残らない、または高精度で競争力を失う可能性**はあるが、未指定の全化学系に対する普遍的な価値上界は導かれていない。$O(\epsilon^{-5/4})$ の単独CS2と $\widetilde O(\epsilon^{-1})$ の補間を比較すると、むしろ一定の補間定理の条件下では漸近次数自体が異なる。単純に「常に定数倍の差」と断定しない。

情報価値プレミアムを $C^*_{\rm free}-C^*_{\rm ideal\ access}$ と置くなら、**同じ精度・失敗率・回路primitive・許可演算の下で最適化された費用**を使う必要がある。固定H4のCS2対S4比率を全研究の普遍的上限にしない。加えて同じ情報を必要とするcandidate/controlの取得費用を両側へ課す。Claudeによるこれらの区別は実務的に有益だが、今すぐH-E1に追加実験費用を投じる根拠にはならない。

### 3.3 H-E1についての研究結論

**【判断】H-E1は独立した新アルゴリズム研究として現定義のまま追究しない。** 研究仮説自体の不可能性を証明したわけではない。K0記録・情報アクセスの条件分け・限定的な解析枠組みを、Aの引継ぎ資料に短く残す。未承認のK1〜K3や、既存第1・第2研究への自動的な移管を行わない。

---

## 4. M案：数学的構成と真正な研究差分

### 4.1 形式的な構成

固定した対称二次PFを

\[
S_2(h;\mathbf F)=\exp[-ih\{H+h^2D(\mathbf F)+h^4E(\mathbf F)+O(h^6)\}],\quad H=\sum_\ell F_\ell
\]

と表し、元のHamiltonianを**修正するのではなく、各stepの近似回路生成に使う演算子を**

\[
F_\ell(h)=F_\ell+h^2W_\ell,\qquad W_\ell=W_\ell^\dagger,\qquad W=\sum_\ell W_\ell
\]

とする。$W_\ell$ が$h$に依存しない固定演算子なら、$D$のFréchet微分を $D'_{\mathbf F}[\mathbf W]$ と書いて

\[
\boxed{H_{\rm eff}^{\rm M}(h)=H+h^2(D+W)+h^4\{E+D'_{\mathbf F}[\mathbf W]\}+O(h^6)}
\]

が形式的に得られる。**これは今回のレビューでの解析的整理であり、数値による新規検証ではない。** すべての元Hamiltonian fragmentをそのまま用いるPFではない。**各stepに使用する和が$H+h^2W$となり、一般には元のHとは異なる**。実効Hamiltonian全体がHへ近づくように意図的に補正する、modified Hamiltonian／modified potential型の構成である。

Symmetricなステップ反転は、$F_\ell(h)=F_\ell(-h)$ と同じ係数規則を採用する場合に維持できる。$h$依存のfragment化を導入しても、符号・時間反転・全体の誤差展開と、$r$ step反復時の有効時間を確認する必要がある。

孤立した非縮退ground branchなら、$V=D+W$ として

\[
\begin{aligned}
E_{\rm M}(h)=E_0&+h^2\langle V\rangle_0\\
 &+h^4\left(\langle E+D'_{\mathbf F}[\mathbf W]\rangle_0
 -\sum_{n>0}\frac{|\langle n|V|0\rangle|^2}{E_n-E_0}\right)+O(h^6).
\end{aligned}
\]

この式も、$E_0$を候補生成へ無償に渡してよいことを意味しない。上式の$E_0$は**科学的な誤差解析側の参照量**である。

### 4.2 CS2を含むことは利点か、基準か

$W=-d_0 I$なら、全PF回路は標準S2に既知の位相を掛けたものとなり、固有位相を$h^2d_0$ずらすだけである。これは**scalar-corrected S2と同じ情報を使う基準**に過ぎない。物理的controlled回路へその位相を埋め込めばcontrolの相対位相の合成費用が生じ得るが、エネルギー値の古典後処理として補正する基準には、その量子費用は課されない。したがって**M案がCS2を表現できること自体を新規性・回路費用優位に数えない**。

M案の固有の可能性は、**スカラー以外の$W$が、元のPFの主要誤差演算子の非対角成分や補正後の4次残差・overlapを有意に変えられるか**にある。これは設計上は考えられるが、実現できる$W$のspaceと情報取得費用が未確定である。

### 4.3 M-aの射影案を何として比較すべきか

Claudeは状態に依存しない例として、$W=-\Pi_{\mathcal V}(D)$を提案した。固定sector上でHilbert–Schmidt直交射影を定義でき、$\mathcal V$が合法なHermitian線形部分空間なら、

\[
\|D-\Pi_{\mathcal V}(D)\|_{\rm HS}^2
=\|D\|_{\rm HS}^2-\|\Pi_{\mathcal V}(D)\|_{\rm HS}^2
\]

が成り立つ。**しかしこれは単なる射影の恒等式である。** $\|D+W\|_{\rm HS}$の減少だけでは、$E_{\rm M}(h)$のground固有位相誤差、CS2後の高次残差、overlap、回路費用が下がることを保証しない。

さらに、本問題の$D$はsectorを保つ演算子の交換子の和であるため、各不変sectorに制限したとき$\operatorname{Tr}_{\rm sector}D=0$である。したがって**sector Hilbert–Schmidt射影における$D$のscalar成分は元々ゼロ**であり、これをground-stateの$\langle D\rangle_0$を消すscalar-correctionと同一視できない。M-aがoperator normの一部を除去しても、groundでは先頭の$h^2$ biasが残り得る。必要ならM-a後にも同情報・同価格のscalar補正を許した対照比較が要る。

また$\mathcal V$は、**演算子の集合**と、**各fragmentに割り当てたラベル付き実装パラメータ空間**を区別する必要がある。同一の合計$W$でもfragment間の配分が異なると$D'_{\mathbf F}[\mathbf W]$とnative回路費用は異なり得る。単なる演算子射影だけで全生成規則は定まらない。

**【K0課題】** 同じ回路grammarでの作用可能な射影$\Pi$を、実際にどの古典費用で求めるのか、元のHilbert空間の次元が指数的でも効率的なfermionic収縮へ落とせるのか、状態別ground固有値の評価とどのように結び付くのかを特定する。

### 4.4 最重要：回路grammarによって合法な修正空間が変わる

M案で最も慎重に扱うべきなのは、Claude本文の「Givens網と位相の個数は変わらず、角度だけを修正できる」という条件付きの記述である。一般には**未証明**である。少なくとも以下の二つの実装契約を分ける必要がある。

**G1：任意の対角density polynomialを、固定の全対位相スロットとして実装できるgrammar。** 各frameで$\sum_pu_pn_p+\sum_{p<q}v_{pq}n_pn_q$を許す。このとき係数が変わっても形式上のslot集合は固定可能だが、元回路でゼロだったslotが非ゼロになると実際の位相回転数が増え得る。制御下の位相、合成精度、隣接基底のfusionも変わり得る。**元から全slotが実装されている場合と、必要な非零slotだけを実装している場合を区別**しなければならない。

**G2：density fragmentをrank-one平方型$\lambda_\ell(\sum_p\eta_{\ell p}n_p)^2$として実装するgrammar。** 任意の二体係数$w_{pq}$は、一般にこのrank-one構造を保たない。$W_\ell$を任意の$\mathcal U_\ell^\dagger n_pn_q\mathcal U_\ell$の線形結合まで許すと、同じsquare primitiveだけでは実装できない場合がある。

仮に$\eta_{\ell p}$を$\eta_{\ell p}+h^2\delta\eta_{\ell p}$に変える**rank-one構造保存**を選ぶなら、$L_\ell=\sum_p\eta_{\ell p}n_p$、$\delta L_\ell=\sum_p\delta\eta_{\ell p}n_p$として

\[
\lambda_\ell(L_\ell+h^2\delta L_\ell)^2
=\lambda_\ell L_\ell^2
+h^2\lambda_\ell\{L_\ell,\delta L_\ell\}
+h^4\lambda_\ell(\delta L_\ell)^2.
\]

すなわち合法な$h^2$方向は平方型の**接空間**に制限され、しかも$h^4$の追加項が実効Hamiltonianの4次係数に寄与する。前掲の$E+D'[W]$に、その$h^4$追加項を含めてはじめて費用・誤差の比較が公平になる。

**【K0の最優先事項】** G1/G2のいずれを対象とするか、どの$W_\ell$が真に回路費用を増やさず実装できるかを、native DF sourceと回路コスト算定規則から**静的に**確定する。技術的に未定義の「同一gate数」を新方法の保証として使わない。

### 4.5 M-b（状態依存4次残差の最適化）の位置付け

ClaudeのM-bは4次残差やground-state overlapを直接狙う。しかし$E+D'[W]$の期待値、$D+W$の励起状態との非対角成分、resolventやgap、必要精度が関わり、1-RDMの簡単な更新表だけでは閉じない。単に「最適化すれば改善可能」と表現することはできない。**M-bは現段階で実行可能な低費用算法と認定しない**。M案の最初のK0では、費用が定義できる可能性のある**M-a／構造保存型**を優先し、M-bを逃げ道にしない。

### 4.6 先行研究のどこまでが既知か

| 先行研究 | M案と重なる部分 | K0でまだ示すべき差 |
|---|---|---|
| Blanes–Casas系のmodified potential／force-gradient splitting、2023論文・2024総説 | commutatorを使うstep幅依存の演算子修正と高次分割という着想 | 量子化学DFの**合法native span**にだけ投影できること、固定価格で新しい適用条件が成立するか |
| Mansuroglu–Fischer–Hartmann 2023、同ゲート数のPF系列の古典最適化 | 固定回路grammar／係数自由度を使った事前最適化 | $h^2$という特殊化や投影規則が既知の係数最適化へそのまま還元されない具体的理由 |
| CPF 2026 v4 | BCH誤差をcorrectorで抑え、実装費用を戻す強い比較対象 | 追加の指数関数なしで**同情報・同価格**の優位があり得るか。共役processorと真の補正を混同しない |
| SPRINT/GRADE 2026 | 化学DFのnorm階層、分解、near-integrable構成、native回路価格の強い既知実装 | M-aの補正自由度がこの既知最適化・補正後残差に対し独立した寄与を持つか |
| Carenaほか 2022 | 誤差を打ち消すmodified Hamiltonian一般の関連例 | 格子ゲージ離散化とPF BCH誤差の違いを保ち、過剰に同一視しない |

**【判断】現時点でM案の新規性は`UNDETERMINED`であり、既知法への還元リスクは高い。** K0の科学的意義は、新算法を保証することではなく、**具体的な合法回路生成規則の可能性が残るかを少ない労力で判定できる**点にある。

---

## 5. A→Bという作業順序の研究上の判断

### 5.1 A：研究結果の整理は軽く実施する

ユーザーが希望しているのは**論文化ではなく、後からCodex/GPTが研究の進捗と判断を再構成できる引継ぎ記録**である。ここに合わせ、Aを大規模なreview再執筆・全raw再監査・論文草稿へ広げない。

**推奨する成果物は原則として一つの短いMarkdown**（たとえば`docs/research/qc_onebody_he1_archival_handoff_20261015.md`）。内容は以下に限る。

1. 現第一候補：設定、数学的に支持された成果、H4費用比較、現構成を区切る科学レビュー推奨（**正式statusは未変更**）。
2. H-E1：仮説、K0 `gate_pass=false`、具体化できる既知処理系、未特定の生成則、K1–K3未実施、独立テーマにしない推奨。
3. Claudeレビューと今回のGPTレビュー：著者・日付・原文SHA・固定参照。**ClaudeレビューをGPTの独立計算成果と表示しない。**
4. 結論から主張できないこと：一般的no-go、情報取得込みの資源優位、他分子への移送、全体GO。
5. 後続のM案K0へのリンク予定と、旧STOP・認可・manifest FAILを保持する条件。

既存の詳細表・数式・原文証拠は**リンク先に残す**。同じ数値表や全論文比較を重複複写しない。既にGitHub保存済みの科学レビューは改稿しない。必要に応じて索引に参照を一本追加するだけで足りる。

### 5.2 B：M案のK0監査を一回だけ実施する

Aが完了したら**同じCodex依頼の中で**次のK0へ進めてよい。ただし今回のGPTレビューは科学的な目的と境界を選定しただけで、数値実行を許可したものではない。

K0の最小要件は、次の5点である。

| 検査軸 | 技術的に求める成果 | STOP条件 |
|---|---|---|
| K0-1：先行研究対応 | modified potential、classical PF係数最適化、CPF、SPRINT等の**式・演算列・入力情報**を対応させる | 同じ合法演算列／同じ価格条件へ還元され、固有の生成規則が残らない |
| K0-2：実装grammar | G1密度対位相、G2 DF平方型を分け、各$W_\ell$の合法性・gate count・新非零角の発生を明示する | 「gate不変」を保証できず、その仮定を外すと改善機構が消える |
| K0-3：数学的機構 | $H_{\rm eff}$の$h^2$／$h^4$項、投影条件、scalar補正後の残差とoverlapへの関係を静的に確認 | $\|D-\Pi D\|$低下だけに帰着し、強い対照に差が出る機構がない |
| K0-4：情報・資源 | $D$または射影の古典取得費用、sector trace、状態アクセス、必要なcontrolled位相を分離 | 投影・補正を求める費用が未定義で、実装可能な入力→操作写像も作れない |
| K0-5：独立性 | 元CS2、M-a+CS2、適用可能なSPRINT/CPF/既知係数最適化の**公平な対照**を列挙 | 同等の自由度を強い対照に与えると固有差が消える |

**K0は文献本文・固定sourceのread-only読解、紙上の式、必要最小限の静的費用会計に限定する**。新H/eig/RDM/PF、数値射影・比率計算、compile、保存結果再集計、pilot用runner追加は行わない。実装可能性が曖昧な場合は自動で大きな変分探索やM-bの別候補へ進まない。

### 5.3 K0が成功した場合の扱い

K0の合格は、**少なくとも以下を一つの具体的な規則として記述できた場合に限る**。

\[
(H,\ \text{native grammar},\ h,\ \text{allowed information})\
\longmapsto\
\{W_\ell(h)\}_{\ell},\ S_2^{\rm M}(h),\
\text{checkable error/cost conditions}
\]

さらに、既知法に還元されない候補差が少なくとも一つ必要である。**「未知だから継続」は不可。** とはいえK0で性能の厳密優位や初出性を証明する必要はない。十分な差が見つからなければ`REDUCES_TO_KNOWN`または`UNDETERMINED_NOT_READY`を記録して停止する。差が残っても、**数値pilot・K1・正式採択はGPTの別開始承認と計算実行の別承認を待つ**。

### 5.4 作業管理・停止

- 一回のCodex依頼で**A→Bをまとめてよい**。ただしBはK0のみ。Aを実施したことをM案の科学的採択へ読み替えない。
- 独立branch使用、関連するREADME/索引への必要最小限の追記、sourceとresultの保全、GitHub公開後の独立再取得照合を行う。
- 第一候補の正式`CLOSED`変更、H-E1を既存第1・第2研究へ自動編入、M案の採択、N1–N3/THRIFT/C1–C4/Track A/Bの再開は禁止する。
- 消費済みのauthorization、one-shot、retry=0、既存manifest FAILを維持し、**Mandatory STOP**する。

---

## 6. 候補の採否、論文、および別方針について

### 6.1 既存成果を論文化する必要はない

ユーザーの意向は、今回の系列を直ちに論文化しないこと。現在の独立算法としての効果・新規性も、独立論文の中心主張を確立するほど強くない。したがって、**Aでは論文草稿、図表の再構築、網羅的な文献レビュー、新規数値実験を要求しない**。将来の別研究の付録や内部参考に使える程度の記録が目的である。

### 6.2 なぜM案を正式採択しないのか

1. **先行研究の重複リスクが高い**。modified potential、corrector、同gate数の古典係数最適化はいずれも既知の強い考え方である。
2. **合法native修正spaceが未確定**。DF平方型の係数自由度と、一般密度対位相は同じではない。
3. **Frobenius射影とground固有位相の有用性を直接結べない**。射影量の減少は、最強対照に対する有限step性能保証ではない。
4. **回路費用・情報取得が定義されていない**。新しい非零回転やstate情報の価格を無料にすると、従来の候補と同じ問題を繰り返す。

### 6.3 なぜ、それでもK0だけは実施してよいのか

M案には$h^2$係数修正と明示的な生成パラメータspaceがあり、H-E1の「共同設計」という目的の宣言より構成が具体的である。**K0だけなら、新しい科学計算なしに、既知法への還元・回路実装の不可能条件・数学的な新規性候補を整理できる。** 技術的に閉じられる問いを一回だけ閉じるという**情報価値**がある。

反対に、K0を通しても定量的に有望とはならない。次の研究方向が採択できないままでも、それは妥当な研究判断であり、次々と新しい候補に乗り換える義務はない。

### 6.4 DF因子回転・別テーマの扱い

DF因子等価回転については、直交因子混合のいくつかのnorm・commutator集約不変量と、回路基底の変化・既存分解最適化との重複が既に懸念されている。これをM案K0と同時に大規模探索することはしない。新しい時間発展の構成を探すなら、**最強対照の補正後誤差・情報取得・回路費用のどれへ作用するか**を最初に指定する。名称だけの候補探索を繰り返さない。

---

## 7. 本レビューの確証と反論

### 7.1 確証できること

- Claudeの最終レビューは、添付本文を実際に確認し、第一候補の終了推奨・H-E1非採択・M案K0の限定提案を区別している。
- 固定GitHubで、第一候補の結果・監査と、H-E1 Stage 0の`gate_pass=false`、K1–K3未実施を確認した。
- 独立新算法の主張が成立していない、という**現状の証拠状態**に対して、Aで記録化する価値とBの限定調査の価値を判断した。
- 既存modified potential／PF係数最適化／CPFの先行文献が存在することを出版・arXivの本文または要旨で確認した。

### 7.2 確証していないこと

- 第一候補の一般的no-go、全分子への移送性、全QPE/RPE総費用の有利不利。
- H-E1すべての可能な生成器の不在、情報価値の普遍的な上限。
- M-aの射影が効率的に求まること、同じDF平方primitiveで$W$を実装できること、CS2後の高次残差が実際に下がること。
- M案がmodified potential、CPF、classical optimizationに対して真に新しいこと。
- K0が通った場合の定量的な回路費用・科学的GO。

### 7.3 この判断への最も強い反論

**反論1：M案も既知法と同じだからK0を行う価値がない。**\
応答：確かに一般原理は既知である。しかし化学DFの実装grammarへ閉じた射影と費用・固有位相誤差の接続には未検証の狭い問いがある。**その問いを一度に閉じ、差がなければ即STOPする**程度の調査コストは許容できる。ただしここから長期研究投資を推奨するものではない。

**反論2：M案は高次誤差を減らせそうなので、直ちに数値実験すべきだ。**\
応答：現時点では$W$の合法な実装spaceさえ未確定であり、projection→精度改善の保証もなく、強い対照との分離も未確認である。実験を先行すると既知法への還元や無償回路仮定を見落とすため、K0を先行する。

**反論3：第一候補は単一H4なので終了は早い。**\
応答：一般不能性ではなく、事前の10%基準未達、強いCS2対照の存在、既知生成法との重複、補正後残差を直接狙えない現設計という理由で**現在の固定構成への投資を区切る**判断である。

**反論4：H-E1の量子状態コピーが無料なら大きな情報価値がありそうだ。**\
応答：状態コピーが極端に安ければ元Hの直接期待値推定が強い対照になり、もはやQPE/PF時間発展の最適化だけの問題ではない。コピー価格を固定しないH-E1を今すぐ独立アルゴリズムとして採択する根拠にはならない。

---

## 8. 次の担当・明示承認・Mandatory STOP

| 段階 | 担当 | 作業範囲 | 承認とSTOP |
|---|---|---|---|
| A | Codex | 第一候補・H-E1の引継ぎMarkdownを原則一つ作る。既存資料へリンク | 論文化・追加計算なし。研究statusを勝手にCLOSEDにしない |
| B | Codex | M案のK0文献・数理・native grammar・費用条件の静的監査 | K0で既知へ還元・固有生成則なしならSTOP。数値は実行しない |
| 公開・照合 | Codex | 独立branchへ報告・索引・証拠を保存しremote再取得確認 | **Mandatory STOP** |
| K0後の研究方針 | GPT | M案採否、別テーマ継続の価値、新しい数値pilotの要否を重要な科学判断として検討 | **ユーザーの別の明示レビュー開始承認が必要** |
| K1／pilot／compile | Codex | 正式に選定されたscopeで限定実行 | **別途の明示計算実行承認が必要** |

今回のユーザーの依頼は**GPTによる重要科学レビューの開始**である。GPT自身は新しいsource・GitHubの正式status・数値結果を変更していない。レビュー推奨と、実際のrepository更新・数値実行認可を区別する。

**K0結果の引継ぎに必須の情報**：元文献の正確な版と式、回路grammarと合法$W$の定義、同情報の強対照、計算不要の数学的支持範囲、技術的`REDUCES_TO_KNOWN`／`UNDETERMINED`の限定性、実行0回、remote取得SHA、Mandatory STOP。単なる「文献に見つからない」を研究GOとしない。

---

## 9. 固定参照・一次文献

### 9.1 GitHub研究証拠

1. [H-E1技術監査（固定commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/research/qc_he1_stage0_technical_prevalidation_20261015.md)
2. [H-E1 GPT引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/research/qc_he1_stage0_gpt_handoff_20261015.md)
3. [H-E1 method status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/artifacts/qc_he1_stage0_prevalidation/2026-10-15/method_status.json)
4. [第一候補・h境界の統合完成版科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/docs/reviews/quantum_chemistry_pf/qc_onebody_repartition_h_boundary_final_scientific_review_20261015.md)
5. [第一候補・H4 h境界数値監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)
6. [第一候補・方法差監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9e288089ca22e2729976ae75d096c4df00e49a7d/docs/research/qc_ground_pf_final_method_difference_audit_v1_20261014.md)
7. [第一候補・1-RDM応答の技術事前監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/docs/research/qc_exact_ground_pf_repartition_technical_preaudit_v1_20261014.md)
8. [第1研究の固定synthesis（別repository）](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/cd81018932b9557b7a55e2e1cd8c5e3988245409/PF_first_study_final_synthesis_20260925.md)
9. [第2研究の固定方針（別repository）](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/1d01e77247364b49dbce2e9ce2fe7c32af0fb937/docs/second_study_v2/budget_safety_mechanism_20261005/final_direction_and_claim_scope.md)

### 9.2 関連一次研究

- Mansuroglu, Fischer, Hartmann, *Problem-specific classical optimization of Hamiltonian simulation*, Phys. Rev. Research 5, 043035 (2023): https://doi.org/10.1103/PhysRevResearch.5.043035
- Bagherimehrabほか、*Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, PRX Quantum 7, 033018 (2026), arXiv:2409.08265v4: https://arxiv.org/html/2409.08265v4
- Blanes, Casas, González, Thalhammer, *Efficient Splitting Methods Based on Modified Potentials*, Commun. Comput. Phys. 33, 937–961 (2023): https://doi.org/10.4208/cicp.OA-2022-0247
- Blanes, Casas, Murua, *Splitting methods for differential equations*, Acta Numerica 33, 1–161 (2024): https://doi.org/10.1017/S0962492923000077
- Casaresほか、*Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 (2026): https://arxiv.org/abs/2606.30741
- Carenaほか、*Improved Hamiltonians for Quantum Simulations of Gauge Theories*, arXiv:2203.02823: https://arxiv.org/abs/2203.02823
- Rendon, Watkins, Wiebe, *Improved Accuracy for Trotter Simulations Using Chebyshev Interpolation*, Quantum 8, 1266 (2024): https://arxiv.org/abs/2212.14144

---

## 10. 最終的な研究判断

**A（第一候補・H-E1の必要最小限の保存整理）→ B（M案のK0文献・数学・回路grammar静的監査）という順序を採用することを推奨する。**

Claudeの最新レビューが示す「現在の既存案から、独立した新アルゴリズムを正式採択できない」という結論は、その適用範囲を守る限り支持する。M案も採択済みとはせず、まず合法回路の定義と先行研究差分を確認する。**K0で具体的な差が残らなければ、既知法への還元を記録して終了する。**

ユーザーが希望しているのは論文化ではなく、後続のCodexが再利用できる研究結果の保存である。そのためAに大きな労力を使わず、Bを実施する場合も**計算なし・一回限定・mandatory STOP**とする。K0が失敗しても次の新候補を自動生成せず、次の重要な研究意思決定が必要になった時点で別途の明示レビュー開始承認を得る。

今回のレビューはこれで完了する。**新しい科学計算・保存数値の再集計・source変更・数値pilot・GitHubへの書込みは実施していない。**
