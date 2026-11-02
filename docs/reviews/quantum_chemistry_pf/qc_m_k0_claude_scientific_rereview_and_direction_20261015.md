# M案K0停止後：ClaudeレビューのGPT独立科学再評価と研究方針

**量子化学Hamiltonian simulation／基底エネルギー推定用product formula — 2026-10-15（JST）**

| 項目 | 内容 |
|---|---|
| 文書区分 | ユーザーの「Claudeのレビューに対してもう一度GPT側でもレビューしてほしい」に基づく、承認済みの重要科学レビュー |
| 対象 | Claudeの完成レビュー `レビューoutter.md`（590行、89,305 bytes）、M案K0技術監査、以前のGPTレビュー、関連一次文献 |
| 添付原本の同一性 | ローカルでSHA256 `bf696b31f2511362d1819e8bc7ff3c89252c6eba91d34c35d0340dc49b838e2c` を確認。別名のファイルと同一とは推定しない |
| 研究repo／固定commit | `HIROMU1015/Partially-Randomized-Trotter`、`155336b07ad0d5e6e9c6cad35736e9412060a299` |
| レビュー状態 | **完了**。原資料と文献との照合、紙上の数理検討、Claudeの主張の適用範囲の評価を実施 |
| 実施しなかったこと | 新規Hamiltonian／RDM／期待値／固有値／PFの数値計算、保存科学数値の再集計、最適化、compile・T合成、科学的test、GitHubへの変更、正式CLOSED／新研究採択／K1承認の代行 |
| 基本結論 | **Claudeの主要な研究判断を採用。ただし一部の理論的推論・情報費用・費用比較の表現は限定して解釈する。** |

## 0. 要旨：採用する判断と修正する範囲

**【GPT判断】M案の現構成への追加研究投資は、現時点では見送ることを推奨する。** M-a（sector-trace HS射影）、現在提案されているM-b（基底状態依存の高次残差最適化）、および単なる合法slot・Givens角の増加について、独立した生成・保証・費用上の改善規則が特定されていない。新規性の明確な差がないままK1を行う情報価値は低い。これはM型counterterm一般の不可能性ではない。

**【GPT判断】Claudeの「現在のPF小改変系列をいったん区切る」という研究資源の配分提案は妥当**であり、基本的に支持する。第一候補とH-E1の既存終了推奨も維持する。既にAとして成果索引を作成したので、論文化目的の追加整理も不要である。ただし、研究系列のpauseと、量子化学Hamiltonian simulationのアルゴリズム研究そのものの終了は異なる。

**【限定修正】Claudeが新たに加えた以下の判断を、一般的no-goに読み替えてはならない。**

1. 「合法パラメータ数が全固有値数より少ない」という次元評価は、**全固有値を同時に補正するための一般位置での障害**であり、本来の目的である**基底エネルギーひとつ**に対する改良を否定する定理ではない。さらにBCH誤差 $D(H)$ は任意の行列ではなく、対象Hamiltonianに強く依存する。
2. 有限次元で `tr(Δ[F,X])=0` が成り立つことは正しいが、これが禁止するのは**外側がFの交換子の、同じFに可換なslotへの直接吸収**である。他fragmentへの投影、非可換なGivens方向、異なる構成規則の可能性をすべて排除しない。
3. H4の小sectorで肯定的なpilotが得られても規模への移送を主張できない、という警告は正しい。しかし「H4のK1は反証にも支持にも情報を持たない」は強すぎる。K1は実現可能性の反証・数式の反例探索には使える。ただし**現時点でK1を推奨しない結論自体は維持**する。
4. 同じbody-rankのleading補正演算子を測る場合でも、測定分散・grouping・基底回転・状態コピー数は同一とは限らない。M案が情報取得で必ず得しない／必ず同額とも断定できない。
5. 保存H4の `S2:response_selected` は1-RDMを使ってXを選ぶ方式である。それを「状態情報なしの新M案」の費用との比較に使う場合、**楽観的な条件付き参照**として扱い、公平な情報なしの定量的下界とは呼ばない。
6. Claudeのcompress-and-correct（R2）で示した変分的な二次残差式は**H'の厳密固有値を使用した場合**には正しいが、H'の有限step PFでQPEを行う実際の提案には追加のPF位相・branch・測定誤差が残る。

これらの補正は、Claudeの「M案を現時点で採択しない」「既存のSTOPを維持」「次のCodex作業を自動起動しない」という主な判断を変更しない。

### 0.1 判断表

| 対象 | GPT結論 | 根拠の確かさ | 正式状態との関係 |
|---|---|---|---|
| M-aのK0 | **不通過を支持** | 高（GitHub原資料、標準BCH・線形代数、一次文献との対応） | `gate_pass=false` を変更しない |
| M-aのK1 | **今は実施しない** | 中〜高（生成の独立性・価値の根拠がない） | 未承認を維持 |
| M-bとM型一般 | **現定義のまま継続投資しない** | 中（具体的生成則・情報取得費用が未特定） | 一般no-goや正式CLOSEDではない |
| 第一候補 | **現固定構成を区切る以前の推奨を維持** | 高（既存H4固定結果に限定） | `AWAITING_USER_ADOPTION_DECISION` |
| H-E1 | **独立算法として現定義を追究しない推奨を維持** | 高（現在具体化された部分に限定） | 自動採択／新計算なし |
| 研究系列全体 | **既知に近い小改変の反復探索はpauseを推奨** | 中（研究資源の判断であり不能性定理ではない） | 研究全体の廃止を意味しない |
| 新研究仮説R1〜R3 | **現時点では採択しない** | 中（各案で異なる未充足条件） | 将来の再審査条件を保存 |

## 1. 資料取得・根拠・独立性

### 1.1 GitHub固定本文を直接取得

本レビューは以下をGitHubから指定commitで実際に取得し、重要箇所を読み取った。`PASS`の文書記述を科学的な新規再計算と混同しない。

| ID | 固定資料・GitHub | 今回確認した範囲 |
|---|---|---|
| G1 | [M案K0技術監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_m_fixed_circuit_k0_technical_audit_20261015.md)（blob SHA `35104778f2fd6b39985b1918b06b9d54b9b5ac8c`） | 全文。形式展開、G1/G2 native grammar、HS/Gram、文献対応、K0判定 |
| G2 | [K0 GPT引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_m_fixed_circuit_k0_gpt_handoff_20261015.md)（blob SHA `0e14f98afd6b32526f78ddafe96bc2d31cf8e808`） | 全文。停止理由と承認境界 |
| G3 | [技術status JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/artifacts/qc_m_fixed_circuit_k0/2026-10-15/method_status.json)（blob SHA `06c0b06274762c83a3aefee596793709d79a5b50`） | 全文。`REDUCES_TO_KNOWN`、`UNDETERMINED_NOT_READY`、K1未実施、正式採択なし |
| G4 | [第一候補・H-E1簡潔引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_onebody_he1_archival_handoff_20261015.md)（blob SHA `8899e660b594b1d2b2b11475544ed08b339699ba`） | 全文。既存成果・終了推奨・制限 |
| G5 | [直前のGPT科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/reviews/quantum_chemistry_pf/qc_post_he1_research_direction_scientific_review_20261015.md)（blob SHA `70b793bd25c2e8260857553350b2cf2cff887aeb`） | 主要部分。K0以前のM案に対する条件付き推薦の範囲 |
| G6 | [第一候補のh境界独立監査記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md) | 過去の文脈および保存値。今回4,572セルの検証を再実行していない |

また、[native会計source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_resources_v1.py)をGitHubから取得し、特に `repeats`、`cost_scope`、query・state preparationの数え方を確認した。実行はしていない。

原資料の取得・内部データ保存のprovenanceはG1/G3に明記されている。本レビューは固定GitHub文書の実ファイル取得を行ったが、原raw数値・5,177旧保護files・旧manifestの全byte監査を独立に再実施していない。

### 1.2 公開文献の確認範囲

- [Blanes, Casas, González, Thalhammer (2023), modified potentials](https://www.global-sci.com/cicp/article/view/7160)：出版情報・研究対象・要旨を出版社ページで確認。詳細式2.7–2.13等についてはClaudeの関連節報告とG1の一次文献台帳に依拠し、今回原PDFを改めて完全に数式照合したとは記さない。
- [Mansuroglu, Fischer, Hartmann (2023), Phys. Rev. Research 5, 043035](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043035)：公開された論文本文情報・固定指数列の古典最適化という先行研究を確認。G1・Claudeの読み取った特定の次数の式については同報告を参照。
- [Bagherimehrab et al. (2026), Corrected Product Formulas, arXiv:2409.08265v4](https://arxiv.org/html/2409.08265v4)：HTML本文のEq.(4)・(5)・(24)〜(28)を確認。symplectic共役とsymmetric sandwichの差は本文で検証できる。Eq.(28)の右端符号とEq.(5)の不一致に関する従前の留保は維持する。
- [Casares et al. (2026), SPRINT/GRADE, arXiv:2606.30741](https://arxiv.org/abs/2606.30741)：公表版・研究主題・強対照としての位置付けを確認。Prop.4などの詳細な式はClaude・G1の固定本文確認に依拠し、今回その全証明を再現していない。
- [Cohn, Motta, Parrish (2021), compressed DF + QFD](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040352)：公開本文で、時間発展に圧縮DF、観測量の期待値により収束したDFを使う既知の分離を確認。ただしClaudeのR2のスカラー補正を同じ方法として認定しない。

以上は**今回の取得状況**であり、Claudeが詳細に取得した六つの文献本文や、Codexの文献SHA台帳をGPTがすべて独立に再取得・照合したという主張ではない。新規性の網羅的初出性調査ではない。

### 1.3 前提を固定

厳密な**量子**基底状態 $|0\rangle$ へのアクセスは可能とするが、基底エネルギー $E_0$、古典波動関数、RDM、期待値、gap、resolventが無償の設計入力になるわけではない。新規計算・保存値の新集計・GitHub変更を開始しない。block-encoding／qubitization／それらを中心とするQSVT・量子walkを対象とせず、Track A/BおよびPF第1・第2研究との独立性を維持する。旧source/result/STOP・認可・manifestの既知FAILを変更しない。

## 2. M案の形式数学：G1/K0とClaudeの再導出を評価

固定した対称S2について

\[
S_2(h;\mathbf F)=\exp\{-ih[H+h^2D(\mathbf F)+h^4E(\mathbf F)+O(h^6)]\},\qquad H=\sum_\ell F_\ell.
\]

偶な係数修正 $F_\ell(h)=F_\ell+h^2W_\ell+h^4R_\ell+O(h^6)$ を同じpalindromeの各stepに適用すると、形式的には

\[
H^{M}_{\mathrm{eff}}=H+h^2V+h^4T+O(h^6),\quad
V=D+\sum_\ell W_\ell,\quad
T=E+D'_{\mathbf F}[\mathbf W]+\sum_\ell R_\ell.
\]

これらは対称BCHの既知次数構造から直接従う。Claude §5.1およびG1 §2との整合に大きな問題は見当たらない。**同じ合計 $W$ でもfragmentへの配分が異なれば $D'[\mathbf W]$ と実装費用は異なり得る**点は重要であり、単なる元Hamiltonianの一体項再配分とは違う。またG2平方型では $R_\ell$ を落とせない。

孤立した非縮退ground branchについては

\[
E_M(h)=E_0+h^2\langle V\rangle_0+h^4\left(\langle T\rangle_0-\langle0|VR_0V|0\rangle\right)+O(h^6),
\quad R_0=Q(H-E_0)^{-1}Q.
\]

したがって、$V$ のHSノルムを小さくしたことは、**スカラー補正後の $h^4$ 残差、branch overlap、有限step時刻での総費用を改善した証明ではない**。この区別に基づくClaudeの停止判断を支持する。

### 2.1 漸近式とfinite-hを混同しない

上式は $h\to0$ の形式展開である。既存H4で観測された有限 $h$ の局所境界では、branch同定、位相alias、synthesis、予算配分が別条件となる。Claudeが報告する実効次数が4に近いことは、有限範囲での経験的整合性であって、他分子や全許容領域への次数保証ではない。

**レビュー判断**：形式数学の成立はM案の独立新規性・性能を自動的に支持しない、というCodex・Claudeの結論を採用する。

## 3. Claude独自の三つの理論論点

### 3.1 可換子環への射影：正しいが、それ自体が安価なalgorithmではない

有限次元sectorの自己共役 $H$ をスペクトル分解 $H=\sum_\lambda\lambda P_\lambda$ とする。HS直交射影は

\[
\Pi_{\mathcal C(H)}X=\sum_\lambda P_\lambda XP_\lambda,
\qquad \mathcal B=\mathcal C(H)\oplus\operatorname{Ran}([H,\cdot]).
\]

非縮退固有状態 $|k\rangle$ では $\langle k|[Y,H]|k\rangle=0$ なので、$V\mapsto V+[Y,H]$ による変更は一次（実効Hamiltonianの $h^2$）の固有値摂動を変えない。退化部分空間では $P_\lambda V P_\lambda$ の固有値が一次分裂を決める。Claude §5.3の数学的主張は標準的なスペクトル摂動論と一致する。

**ただし** $\Pi_{\mathcal C(H)}D$ を設計時に数値的に求めるにはHのスペクトル情報を要し、現在の「$E_0$未知・RDM無償ではない」という入力契約から無料では得られない。$Y$ を低body-rankの既知演算子族へ制限した射影は形式上の最小二乗だが、Hの真の可換子環への完全射影ではない。

さらに、$\lVert D+W\rVert_\tau$を最小化するM-aは、基底状態の $\langle D+W\rangle_0$ ではなくsector一様な全演算子ノルムを最適化する。固有値に寄与しない交換子方向まで評価するため、目的とタスクのずれがある。**この指摘をClaudeの重要な追加貢献として採用する**。ただし「目的がずれている」から直ちに全ての代理量最適化が無価値と証明されたわけではない。

### 3.2 有限次元での交換子・可換slotの直交性：補題は正しい

$F$ と可換な $\Delta$ に対して、正規化traceの巡回性から

\[
\tau_\nu\bigl(\Delta[F,X]\bigr)
=\tau_\nu\bigl([\Delta,F]X\bigr)=0.
\]

ゆえに $[F,[F,G]]$ のように**最外側の交換子がF**である成分は、同じFの可換子環とHS直交する。DF density frameで対角なG1 slotはFと可換なので、**その項をF自身のframe対角slotへHS射影して直接吸収することはできない**。

これは有限次元の厳密な性質で、連続系の $[V,[T,V]]$ が乗算演算子として扱える場合の議論を、そのまま有限次元DFへ写してはいけないことを示す。Blanesらのmodified potentialを強対照に置く際の重要な限定である。

ただし、この補題の結論は**他fragmentのslotへの吸収**や**Fと可換でないGivens／frame方向への再構成**を排除しない。Claude自身も他fragmentに言及しているが、要旨§2.3の「有限次元では古典的modified potentialの機構が働かない」を一般的な吸収不能定理と読むのは過度である。正確な結論は「**自己fragmentの可換slotへの吸収は不可能**」に限る。

### 3.3 パラメータ数対全固有値条件数：重要だが基底エネルギーへのno-goではない

非縮退なsector次元を $d$ とする。補正後のすべての固有値の $O(h^2)$ 誤差を同一の既知scalar位相を除いて打ち消すには、一般に $d-1$ 個の独立した条件を満たす必要がある。合法方向の次元 $P$ が $d-1$ より小さいとき、**任意の独立した対角誤差パターンを**修正することはできない。$P$ が多項式で$d$が指数的に増える通常の成長則では、全スペクトルの普遍的な補正能力に強い制約がある。

Claude §5.5はこの点を明確化した。数学的なランク障害の着眼は妥当である。ただし次の三点は必ず留保すべきである。

**(a) 本来のタスクは全固有値ではなくground energyである。** 非縮退groundの一次条件 $\langle0|D+W|0\rangle=0$ は、$d-1$ 条件ではなく**一つのスカラー条件**である。もちろんその条件を、未知の $E_0$ や無料RDMなしで安く選ぶことは別の難題だが、全スペクトル次元評価だけで不能とは言えない。

**(b) 物理的なBCH誤差は任意の対角ベクトルではない。** $D(H)$ は同じHのfragment・交換子から構築される。独立条件が一般位置を占める、という仮定は、化学DFの構造に対して別途証明されていない。特定の対称性・可換性・near-integrable構造で達成可能な特殊族も否定されない。

**(c) 完全相殺を目標にしない改善も論理的には可能である。** 固有値残差の部分的な減少、overlap向上、情報取得費用減、step単価減を同時に評価する方式があり得る。ただし、M-a/M-bの**現在の定義**にそのような競争力ある生成規則があるとは確認できていない。

従って「一般位置での全固有値一括相殺は困難」という判断は採用するが、「状態情報なしのground改善は分子規模で不可能」「H4 pilotは情報をまったく持たない」には拡張しない。小sectorのpilotはスケール保証にはならない一方、候補を早期に**反証**する役割はある。**K1を今実施しない理由は、次元的不能性の証明ではなく、K0不通過と独立した優位規則未特定である。**

## 4. 回路・DF grammar・情報費用の再評価

### 4.1 G1/G2/G3の区別を維持する

- **G1**：固定orbital frameのdensity one-/two-body pair-phaseの係数を動かす広い方向。lower-level RZ/RZZで表現できても、高位 `DFBlock` が任意係数を入力できるとは限らない。
- **G2**：`λL(h)^2`というrank-one平方形を維持する方向。`L(h)=L+h²δL` のとき、$h^2$接方向に加えて$h^4\lambda(\delta L)^2$が現れ、補正後の高次誤差評価で必須となる。
- **G3**：Givens/frame角を変更する方向。一般には同じ回路gate種・個数のまま変更できる条件があるが、すべてのframeで同一の共役はisospectralである。fragmentごとに異なる回転は固有値を変え得る。G3を追加しただけでは新規な生成規則にはならない。

native sourceはゼロ角の位相gateを省略する。特にtraceless factorの一部条件では元のRZ slotが消え、係数修正によって新しいRZが生じ得る。従って「固定frame／固定slot」だけではnative gate数、controlled位相gate、Clifford+T synthesis costを無条件に一定とできない。この点はClaude §5.6とG1 §3の結論を採用する。

### 4.2 補正情報の費用は双方に必要

$\hat E_M(h)=E_M(h)-h^2\hat d_M$ 型の補正には、$d_M=\langle D+W\rangle_0$ の推定・誤差・失敗確率が要る。元CS2の$d_0=\langle D\rangle_0$も同様である。同じbody-rank（一般に4体まで）の演算子だからといって、分散、可換grouping、測定basis数、実コピー費用が同じとは限らない。

Claude §5.2の「情報取得の面で優位は生じない」は、**独立した情報費用優位が今のM-aで示されていない**と読み替える。M案が情報費用で必ず不利／必ず同額という定理ではない。exact-groundの量子stateへのアクセス価格を固定してから同情報のCS2対照と比較する必要がある。state preparationは無料の古典状態ベクトルへのアクセスではない。

### 4.3 overlapと状態準備の価格

G1/G5が参照する静的会計は、`repeats(overlap,failure)` として

`max(1, ceil(log(failure) / log1p(-overlap)))`

を用いている（overlapが1未満の場合）。したがってoverlapは、成功確率を通して**反復回数に乗法的に作用する**。対数的なのはfailure目標への依存であって、overlapが悪化しても費用への影響が一般に小さいという意味ではない。実際の旧登録gateではoverlapが高く保たれる候補だけを採用しており、その限定下では主費用を決める量ではなかった、と解釈するのが正確である。

## 5. 費用比較と情報条件：Claudeの結論をどう読むか

### 5.1 $1/p$乗根則は有用なscreening、一般的な上限定理ではない

同一価格モデル・同一実効次数p・同一予算配分で、決定論的な残差が $|c_p|h^p$、query費用が$\propto1/(h\varepsilon_{\rm stat})$であるなら、

\[
C_{\rm QPE}\propto C_{\rm step}\,A\,|c_p|^{1/p}\varepsilon^{-1-1/p}\,r,
\quad +C_{\rm info} \text{ を別途会計},
\]

となる。これは、残差係数の変化がstep費用・情報費用と異なり$p$乗根で効くことを意味する。Claudeの「p=4で10%費用減には残差係数0.9^4≈0.656以下、2倍減には1/16以下」という関係は、**この模型の中では正しい**。

しかし、これはfinite-hで実際に同じp則・同じhの選択可能域・branch・overlap・synthesis・費用モデルが保たれる場合の尺度である。**小さい係数改善を一般的に無価値とする証明ではなく、事前判定に便利な損益分岐規則**とする。とくに実効次数そのもの、step単価、統計係数A、情報価格、状態準備回数を変える方式には直接適用できない。

### 5.2 H4数値を情報なしの保証と読まない

既存結果では第一候補 `CS2:response_selected`／`CS2:known_linear_extraction` は理想static controlled primitive価格で約0.9709〜0.9750、登録10%改善は未達である。ClaudeはさらにS2:response、S4:known等の保存値から倍率を読んでいる。これ自体は保存値に基づく条件付き算術だが、解釈に次の留保が必要。

1. `S2:response_selected` の構成選択には、**1-RDM情報（またはそのproxy）が使われる**。それを「状態情報なしS2」の比較基準と呼ぶと、情報モデルが一致しない。Claudeの5.3倍・16倍等は、選択済みのresponse回路を**無償で継承した場合の楽観的な参照値**であり、新M-aの情報なし性能下界ではない。
2. `S4:known`、`EXT2:known` と `S2:response` は、回路構造、step当たり価格、必要な情報、有限h selectorで違い得る。新候補のgateを受理する**登録閾値**ではなく、従前の開発用会計に基づく相対規模の説明にとどめる。
3. 旧H4は一分子・同じ保存cellに対する異なる価格仮説であり、4個の独立した汎化実験ではない。必要なら将来の研究設計でstate情報と実装価格をそろえた対照を固定する。

この限定後も、現M-aにCS2・S4・外挿を上回る具体的な改善規則がないことは変わらない。

### 5.3 R2（compress-and-correct）の理論式：厳密groundとPF出力を分離

Claude §7.4 R2は $H'=H-\Delta H$ として、理想的に$E'_0$（**H'の厳密基底固有値**）と$\langle\Delta H\rangle_{\psi_0}$を得るなら

\[
E'_0+\langle\psi_0|\Delta H|\psi_0\rangle-E_0
=E'_0-\langle\psi_0|H'|\psi_0\rangle\le0
\]

と導く。これは変分原理から正しい。$H'$が$H$の小さな摂動でground branchが孤立している場合、先頭の一次ずれが相殺され、残差が二次になる説明も成立する。

**ただし**R2の実際の操作は、$H'$をS2/CS2で時間発展させ、固有位相からエネルギーを推定するというものだった。その場合、例えば

\[
\widehat E_{\rm R2}-E_0=
\underbrace{(E'_0-\langle\psi_0|H'|\psi_0\rangle)}_{\text{変分的な二次残差}}
+\underbrace{\delta_{\rm PF}(h)+\delta_{\rm phase}}_{\text{有限step・QPE誤差}}
+\underbrace{\eta_{\Delta H}}_{\text{補正測定誤差}}
\]

となる。加えて $|\psi_0\rangle$ と $H'$ のground branchのoverlap・alias・正しいbranchの選別も必要である。

従ってR2の変分的二次残差だけを**完成したPF+QPE方式の総誤差や厳密な符号保証**として扱うことはできない。この注意はClaudeのR2を再採択する理由にはならない。元の圧縮DF時間発展と収束DFでの観測の使い分けが既知であることも、出版社の一次資料から確認できる。

## 6. 新規性の評価：Claudeの主判断を採用

重要なことは、**各既知部品の存在**と**新しい全体構成が不可能**という主張を分けることである。

| 構成 | 既知と確認できる内容 | なお主張できないこと |
|---|---|---|
| Modified potential／counterterm | $h$依存の補正項を入れる一般思想、特定の可解交換子を吸収するsplittingは既知 | 一般化学DFの任意countertermを追加gateなしで吸収できること |
| 固定指数列の角度最適化 | perturbativeな古典費用関数で角度系列を最適化する既存研究がある | 本M-aの3次S2目的が一つの論文に完全一致すること、または独立した優位があること |
| BCHのHS／Gram射影 | 線形least-squaresと固定sector CAR traceは既知の数学的手順 | ground-energyに効く成分が安く得られること、新しい計算量差 |
| Symplectic processing | 厳密な共役は固有値を変えない。effective orderの演算子誤差と固有値誤差は別 | processingだけで固有位相biasが消えるという一般保証 |
| G1/G2/G3合法係数・frame変化 | 数式上の合法方向が記述できる | native FT単価不変、合成の最適性、競争力ある新生成則 |

M-aで確認できた核心は、標準的なBCH係数を線形最小二乗で合法spanへ写すことに近い。従来法に同じ恒等式、同じ合法slot、同じ入力情報、同じcacheを与えれば同じ系統の係数表を得られ、現時点ではアルゴリズム上の独占的な生成・計算量優位がない。

そのため `REDUCES_TO_KNOWN` は**具体化されたcore**について妥当だが、**M型全体が文献中の単一手順と完全同値だという証明ではない**。M-bについてもresolvent・高体rank・測定・変分探索の実施可能な生成規則が特定されない。`NOT_IDENTIFIED` を `IMPOSSIBLE` と置き換えずに区切るのが適切である。

### 6.1 CPFの特殊な表記について

arXiv:2409.08265v4のHTMLで、Eq.(4)は共役、Eq.(5)は同符号sandwichと区別される。Eq.(28)に表示された $e^{C_{\rm sym}}S_2e^{-C_{\rm sym}}$ について、同時に記されている $C_{\rm sym}=O(\lambda^3)$ の共役では一般の $\lambda^3$ BCH項が消えないため、Eq.(5)との表示上の不整合という前回からの留保は維持する。この一点からCPFの全構成を否定しない。今回のM研究判断の結論はこの論文の当該表示式の正誤には依存しない。

## 7. 研究全体の方向についての独立判断

### 7.1 Claudeのpause推奨は妥当

これまでの第一候補、H-E1、M-a、関連の過去候補は、それぞれ異なる理由で強い対照に対する主張が限定された。M-aの停止には**登録10%基準の未達**といった実測は存在せず、K0の**先行性・生成規則・目的関数・費用機構の不足**が理由である。異なる停止理由を同じnegative experimentと扱わない。

「現在の固定PFに小さな自由度を追加し、誤差係数の代理量を下げ、数値結果を見て研究を救済する」を繰り返す方針は、独立した新規性を確認する効率が低い。そこで、**この探索系列を一時停止し、無制限の新候補生成や自動的なK0→K1を始めない**というClaudeの推奨を支持する。

ただし、このpauseは、既に成果のある第1・第2研究、Track A/B、他のHamiltonian simulation構成の研究を終了する科学的主張ではない。ユーザーがアルゴリズム研究を続けると決めた場合に、まず科学的に意味のある問いと実装可能な新生成規則を提示することは可能である。

### 7.2 R1〜R3の判断

**R1：固有値と整合したM型**。従前のHS代理目的より物理的に適合する一方、いま示された $\min_{W,Y}\|D+W+[Y,H]\|$ は既存のeffective-order/processingと通常の係数fitに接近し、ground energyに関する保証・安価な情報取得・強対照との差を確立していない。現定義の採択は推奨しない。全スペクトル相殺の次元障害を、基底状態相殺の不可能性と取り違えない。

**R2：圧縮＋補正**。step単価に直接作用する点は探索方向として合理的だが、圧縮DFによる時間発展とより正確なHamiltonianの観測を使い分ける構成は先行研究にあり、今回の式はPF位相・有限step・測定費用を除いた理想関係である。独立算法としては未採択、別の科学的差が示されれば再審査は可能。

**R3：基底状態分散を使うfragment／ordering選択**。既存の状態依存PF誤差予測・fragment最適化と重なる。情報取得費用とCS2後の残差・native回路価格まで含む新生成則は未提示。現定義は未採択。

**旧D案**（DF因子等価回転）は、今回のM-aのK0不通過だけを理由に自動的な次案へ繰り上げない。保留を維持する。ただしDF回転全体の科学的no-goは本レビューで再審査・証明していない。

### 7.3 次の新候補を判定するための事前条件

具体的な候補が再び提案される場合、計算前の限定K0で以下を明確化する。

1. **改善レバー**：step単価、実効次数、有限step残差、overlap、情報取得費用のどれをどう動かすか。残差だけを対象とするときは強い補正後の$1/p$則を確認。
2. **具体的生成写像**：入力（H、state access）、fragment／frame／native gate生成規則、保証／条件、出力。単なる「共同最適化」「測りやすくする」を方法と呼ばない。
3. **同じ合法情報・価格の強対照**：既存CS2・S4・外挿／Chebyshev・SPRINT／CDF／GRADE・modified potential・coefficient fittingのうち該当する方法を選び、双方に適用可能な条件で比較する。強対照が成立しない系にはその前提を明示する。
4. **情報モデルの明示**：量子groundコピー価格、control準備unitaryの有無、測定variance・precision・failureの扱い。E0や古典RDMを無料にしない。
5. **スケールへの機構**：小sectorの過剰自由度に依存しないか。完全な全固有値相殺を前提とせず、主張する対象（groundのみか、複数固有値か）に合った条件数を数える。
6. **native／FT費用の経路**：RZ/RZZ、Givens、controlled phase、fusion、合成精度と特殊角による価格変動。表現上のslot数だけをT数保証としない。
7. **明示的なSTOP条件**：既知法の単純再表現、実装のclosure不成立、強対照に対する十分な天井不在ならpilotを自動開始しない。

上記は今回新規候補を採択したり、新しい技術探索をCodexへ包括委任したりする承認ではない。

## 8. 推奨する研究運用と次の担当

### 8.1 いま必要なこと

**【推奨】新しい科学計算・K1・M-b探索・別候補の自動K0は実施しない。** M-aのK0 technical stopを保ち、第一候補とH-E1の既存整理Aを再作成しない。今回のClaudeレビューと本GPTレビューを後から参照できる形に保管することは任意の事務作業であり、論文化や新しい実験を伴わない。

| 担当 | 状態・役割 |
|---|---|
| ユーザー | 第一候補の現構成の正式CLOSEDを採用するか、H-E1／M案を現定義で区切るか、探索系列のpauseを採るかを決定。判断を先送りしても旧STOPは継続 |
| Codex | **必須の新しい科学技術作業なし**。希望するなら受領レビュー・本レビューの原文保存と索引一行の追記のみ。無断でGitHubを編集・pushしない |
| GPT | 本書で一回の重要科学レビューは完了。将来、具体的に新しい生成・保証・費用の差を示す科学的仮説が提示された場合のみ、別の開始承認を受けて再レビュー |
| 科学計算 | 一件も新たに認可せず。K1等を行うには独立した固定scope・実行承認が必要 |

既存の `one-shot`、`retry=0`、消費済み認可、`Mandatory STOP`、旧結果・sourceの保護、既知global manifest FAIL、N1 STOP/N2 HOLD/N3 CLOSED、THRIFT現template CLOSED、Track A/BとC1〜C4の独立性、旧D案保留を維持する。第一候補の正式状態はユーザーの意思決定を待つ。

### 8.2 研究判断を変更し得る具体的証拠

- M型について、物理的に意味のあるchemical Hamiltonian familyに対して、**ground taskに直接効く**低費用な合法span補正生成則・保証を与え、既知coefficient fitとの差を示した場合。
- 同じ情報・価格条件でCS2/S4/SPRINT/最適化外挿より安いことを事前に見積もれる上界または構造的改善機構が得られた場合。
- 情報取得費用・branch overlap・synthesisを含む総資源評価に必要な前提を先に固定し、数学的に区別できる新方向が現れた場合。
- 特定のrepartition／compression／factorization classについて、実装の追加gate条件、実効次数、有限hのerrorを一体に分析した新しい構成が定義された場合。

現時点でこれらは**まだ得られていない**。過去のSTOPが多いこと自体を、将来の独立算法の不可能性の証拠としては使わない。

## 9. 最終結論

**Claudeレビューの主要結論は十分に支持でき、現M-a/M-bの追加投資を推奨しない、第一候補・H-E1を現定義で区切る、現在の小改変探索をいったんpauseする、という判断を採用する。**

独立に確認した数学的主張（可換子環の射影、有限次元での自己slot直交、全固有値の一般位置での次元障害）は妥当であり、K0の既知法との重複を補強する。しかし、これらの定理が示す範囲を超えて**基底状態特化の改善不能**を導くことはできない。Claudeの研究方針の提言を保持しつつ、数学的no-goと投資判断の根拠を分けて記録する。

**科学的STOP・研究採否・リポジトリ状態の正式変更はすべて別概念**である。今回のレビューで、数値計算、GitHub変更、正式CLOSED、別候補の採択を行っていない。

## 10. 参照文献・固定原資料

**ユーザー提供**：`レビューoutter.md`（Claude重要科学レビュー、2026-10-15、上記SHA256）。細目の参照は原文の§2〜§9。

**GitHub**：

1. [M案K0技術監査、固定155336b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_m_fixed_circuit_k0_technical_audit_20261015.md)
2. [M案K0 GPT引継ぎ、固定155336b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_m_fixed_circuit_k0_gpt_handoff_20261015.md)
3. [M案 method_status、固定155336b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/artifacts/qc_m_fixed_circuit_k0/2026-10-15/method_status.json)
4. [第一候補・H-E1 archival、固定155336b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/research/qc_onebody_he1_archival_handoff_20261015.md)
5. [前回GPT科学レビュー、固定155336b0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155336b07ad0d5e6e9c6cad35736e9412060a299/docs/reviews/quantum_chemistry_pf/qc_post_he1_research_direction_scientific_review_20261015.md)
6. [元のH4 h境界結果、固定500ea6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/500ea6ece429d6951f710cd99ac7e46337fcf017/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)
7. [native価格計算source、固定0e923f](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e923f73754d1ebd168bf21a1b9e73971946e603/src/trottertracks/ground_pf_repartition/qc_onebody_repartition_resources_v1.py)

**外部一次文献**（確認範囲は§1.2に記載）：

- Blanes, Casas, González, Thalhammer, *Efficient Splitting Methods Based on Modified Potentials*, Commun. Comput. Phys. 33 (2023), 937–961. https://doi.org/10.4208/cicp.OA-2022-0247
- Mansuroglu, Fischer, Hartmann, *Problem-specific Classical Optimization of Hamiltonian Simulation*, Phys. Rev. Research 5, 043035 (2023). https://doi.org/10.1103/PhysRevResearch.5.043035
- Bagherimehrab et al., *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, arXiv:2409.08265v4. https://arxiv.org/html/2409.08265v4
- Casares et al., *Theory and Practice of Trotter Product Formulas for Quantum Chemistry*, arXiv:2606.30741v1. https://arxiv.org/abs/2606.30741
- Cohn, Motta, Parrish, *Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians*, PRX Quantum 2, 040352 (2021). https://doi.org/10.1103/PRXQuantum.2.040352

---

**レビューの適用範囲**：既存固定資料、Claudeのレビュー、限られた原著文献の確認、静的数式の考察に基づく科学的判断である。初出性・一般的不能性・物理的性能差・全FT/QPE費用を新規に証明したものではない。新しい実験・計算の実行承認を含まない。
