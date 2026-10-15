# Track B／M1-S G0-T1研究契約：Claude完成レビューに対するGPT独立科学レビュー

- **レビュー日**：2026-10-16 JST
- **担当**：GPT-6
- **対象**：ユーザー提供 `レビューinner(1).md`（Claude、2026-10-15、全740行）。対象テーマはM1-SのG0-T研究契約の採択範囲・検証順序・受理／STOP条件。
- **直接確認した固定資料**：GitHub `HIROMU1015/Partially-Randomized-Trotter` の G0事前契約 `0f0952614dcf5c06d3d63c21b5d49091b8ecce37`、前回GPTレビュー `4f5114e137563bb871c9f881f9eadab0c180bdd4`、G0-pre `8b7666d0270f7fac01e430e195bef7b58ac0116c`、P0.5 `ed3ddeacc88e998e7bc1d8c0db8163e106312791`、および事前契約索引。本文をGitHub接続で実際に取得した。原資料の取得状況と外部文献の確認範囲は§2。
- **権限と実施状況**：ユーザーの明示的なGPT重要科学レビュー開始承認を受けて実施。紙上の代数・極限・適用条件の検算、一次文献との照合、研究方針の比較まで。**新規の科学数値計算、シミュレーション、回路生成・compile、保存データの再集計、G0-T1の新しい定理の研究実行、GitHubの変更は行っていない。M1-S／研究契約は未採択、Gate 0未承認、mandatory STOP維持。**
- **本書の位置付け**：研究方針についての**科学的推奨**。ユーザーの正式採択および別途の理論・数値実行承認を代行しない。

---

## 0．総合判断

### 0.1 結論

**Claudeの「現行Q1–Q5を縮小し、G0-T1（T1a：完全乱択の平均化保証、T1b：部分乱択への先導次数の移送）の二段階に再定義する」という方向性は、科学的研究計画として条件付きで支持する。** とくに、Q1だけでは研究としての差分が小さく、部分乱択への移送判定を早める必要があること、およびGate 0を未認証proxyのまま先行実行しないという判断には合理性がある。

ただし、**今回のClaudeレビューの数学的草案を無修正でG0-T1の固定契約・定理の受理基準へ採用することは推奨しない**。理由は、研究順序ではなく、中心statementと部分乱択への移送の厳密性に関わる未閉鎖事項があるためである。

最も重要な判定は以下のとおり。

| 論点 | GPT独立判定 | 採択前に必要な修正・留保 |
|---|---|---|
| Claudeの第三案 **A′（T1a→判定→T1b→判定→Gate 0）** | **支持：中〜高** | T1aで形式的な定理完成を目的化せず、T1bの移送に資する最小限の信号保証に限定する。各checkpointで研究上の判断を独立に行う |
| **Q1のsecular平均化という理論方針** | **有望：中** | 既知平均化法と整合。ただし誤差式には**各stepの局所高次項の累積誤差**を明示する必要がある。現行§8.1候補式は誤差項が不十分となる例がある（§4.3） |
| **部分乱択の因数分解** `Ψ(X)=D′Φ^r(DXD′)D` | **指定された順序・独立性の下で正確：高** | これは平均演算の厳密な恒等式であり、減衰率・固有mode・RPEの保証ではない（§5） |
| **dressingの規模** `ασ_k(H_R)` | **状態の近さの指標として妥当：高** | **相対的な減衰率補正**がその大きさで抑えられるとは、別途分母・非正規mode・摂動の証明が必要（§6） |
| 一般Hermitian eventの二次展開 | **紙上の演算子等式として正しい：高** | 非involution eventの正規化・制御価格・対称性・対象modeはPauli参照モデルとは別の契約（§7） |
| 完全乱択の `ρ∞=(V+v)/(2V)` | **先導係数の代数として支持：高** | 有限 `N,θ`・同RPE精度・総資源での実現保証ではない |
| 有限時間の `ρ(T)` とsinc補正 | **一次近似上の示唆として支持：中** | Duhamel一次を `Nθ²=O(1)` へ無条件に持ち込んだ厳密区間ではない（§8） |
| 部分乱択の `ρ∞^(H)` | **未証明の移送仮説** | joint scaling、effective Hamiltonianの基底、dressing、pair-sum/alias条件をT1bで閉じるまでGate 0の認証済み指標にできない |
| `ρ*=1−S*/f_R` | **固定コストモデルの代数条件として正確** | `f_R`は選んだ**評価対象全体**に対する割合を同一座標で定義。shot・bias・再最適化を含む実資源のGO条件ではない |
| 「DF eventでは共鳴が消える」 | **条件付きの可能性** | 保存する対称sectorと、その**到達可能空間で目的状態が端点**である条件を明示。粒子数保存だけでは十分でない（§9） |
| M1-Sの独立新規性 | **未認定、限定的な保証研究として候補を維持** | 方向制御、qDRIFT、回文、量子分岐間の共有乱数自体は既知。新しいのはtaskに有効な保証・費用改善が示せた場合だけ |

**総合分類：`G0-T1 / A′ の科学的方針を修正条件付きで推奨、現行statementの無修正採択は不可、正式採択はユーザー判断待ち`。確信度：方針＝中〜高、一般定理の成立＝中、量子化学で有意な資源改善＝低〜中。**

### 0.2 重要な新しい訂正：数式の「不足」と研究方針の「否定」は異なる

本書で見つかった最重要の数学的問題は、Claude §8.1の提案するQ1平均化誤差式が、非共鳴項と低周波項だけでは一般に完結しないことである。これは**平均化理論の方向性そのものの反証ではない**。適切な局所打切り・位相補正を追加する、または一stepの正確な対数を参照モデルに使うといった修正の余地がある。ただし、修正が実際に一様保証として閉じるかは**G0-T1の未実施の証明義務**であり、本レビューで完成させていない。

そのため、ClaudeのA′案を全体として不採用にする理由はない。一方で、S0の時点で数学的に足りないものを特定せず理論実施へ移ることは避ける。

---

## 1．対象・既定の研究方針・今回の境界

M1-Sは、Hadamard testの2分岐を同じ乱択ラベルで進め、符号だけ反転した方向制御gateを用いる候補である。完全乱択での平均は通常のCPTP channelではなく、\(X\mapsto\mathbb E[A_jXA_j]\)の反復として書ける。このコヒーレントな**信号**に生じる減衰と位相biasの制御が研究の核になる。

すでに保持されている判断は以下である。

- Parent allocationのD＋Eは採択済み、追加数値探索は区切られている。
- 元のM1について、完全qDRIFT/RTEの既知の係数差だけを新アルゴリズムとする方針は採らない。
- M1-Aは普通のqDRIFTと独立±についての理論補助候補。単独の大型テーマとしての新規性は限定的。
- M1-Sは条件付きの研究候補。現状において研究としての独立性、長時間保証、部分乱択への移送、RPE・総資源は未認証。
- G0-preとG0事前契約整理は完了しているが、**M1-S／G0-T1はユーザーにより正式採択されていない**。

本レビューは「次にどこまで研究契約として採択する価値があるか」を判断するもの。過去の全Track B候補を一から順位付けすることや、G0の数値実験を実施することではない。

---

## 2．一次資料・固定GitHub資料・確認範囲

### 2.1 今回実際に確認したGitHub資料

| ID | 固定原本 | 対象 | 取得・確認状況 |
|---|---|---|---|
| G0-CONTRACT | [固定commit `0f0952614dcf5c06d3d63c21b5d49091b8ecce37`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md) | Q1–Q5、protocol候補、Gate 0、A/B、STOP | GitHub接続で本文を直接取得。Git blob `3e81aadff95257f4a180f68319f05f8ddab1cf81` |
| CONTRACT-IDX | [事前契約evidence index](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/evidence_index.json) | 原本同一性、未承認、STOP | GitHub接続で取得。Git blob `7c8591248a3b0cc52f08f28daaac21517c8c775b` |
| GPT-PREV | [固定commit `4f5114e137563bb871c9f881f9eadab0c180bdd4`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4f5114e137563bb871c9f881f9eadab0c180bdd4/docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md) | Claudeの前回B推奨・長時間一様性・資源留保 | GitHub接続で本文を直接取得。Git blob `91fb06e6de2a2c00289173c41be8f34aa0137016` |
| G0-PRE | [固定commit `8b7666d0270f7fac01e430e195bef7b58ac0116c`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md) | 8近接文献、共有law、protocol、費用の先行監査 | GitHub接続で本文を直接取得。Git blob `e08a6e03119c8e286d2dad32e68c3ae800559732` |
| P0.5 | [固定commit `ed3ddeacc88e998e7bc1d8c0db8163e106312791`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md) | 既存定理の対象差、H6 source、D/R費用未確立 | GitHub接続で本文を直接取得。Git blob `fc15ed3af4f72e35b93d2b1270f1bc1c6ea07ba4` |

Claude原文は今回のユーザー添付`レビューinner(1).md`を直接読み、ファイル内容を参照した。ファイルのSHA256（この環境で確認）は `17cee6834f7ee0f731142279b15ac1aa96026f5e976377b227b89abe68221b69`。これは今回の**Claude G0-T契約レビュー**であって、以前の`レビューinner.md`などとは混同しない。

GitHub公開索引にあるファイルのbytes・hashおよびClaudeの受領報告はsource provenanceの補助証拠である。本レビューではGit blobと固定commitの取得を確認したが、全資料の独立SHA256再照合や8論文の全証明の再現は実施していない。

### 2.2 公開一次論文で直接確認した内容

1. [Günther et al., *Phase estimation with partially randomized time evolution*（2026）, arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)：公開PDFのAppendix A.4、Eqs. (A42)–(A46)相当部分。対称PFの厳密なhalvingと、部分乱択RTEの`U`・`U†`を独立乱数で実装する区別、および「randomized PFに同じ厳密なtrickをそのまま適用しない」という記述を確認した。**この記述は無効性の一般no-goではなく、サンプルごとの厳密な時間反転対称性に関するもの**である。
2. [Kanasugi et al., *Enabling chemically accurate quantum phase estimation in the early fault-tolerant regime*, arXiv:2603.22778v2](https://arxiv.org/abs/2603.22778v2)：公開PDF Appendix B、Eqs. (B3)–(B5)。`γ=1` qDRIFT、`γ=2` RTE／TE-PAI、`C_gate=1`はSTARの設定値であり、**M1-Sの新しい方向制御回路とordinaryの同一実費用を認証した数値ではない**と確認した。
3. [Campbell, *A random compiler for fast Hamiltonian simulation*（2019）](https://arxiv.org/abs/1811.08017)：著者公開受理稿（[PDF](https://eprints.whiterose.ac.uk/id/eprint/150026/1/1811.08017.pdf)）Appendix Dのcontrolled evolution（`|1><1|⊗H`）とFig.3の制御gate価格を確認した。`Z_c⊗H`へ同じqDRIFT発想を使うことは自然だが、**今回のM1-Sのcoherent signal保証までそのまま載っているわけではない**。

これらは**関連節の確認**であり、論文全文の新規性ネットワークを網羅したものではない。COS、KGW、MLMC、SCU、KIM、CHEN等はGitHub G0-preの取得済み文献対応表を参照したが、今回それぞれの本文全体を独立精読したとは記録しない。世界初・同等先行結果の不存在は認定しない。

---

## 3．確認済みの数学的核：平均の意味論と分散

### 3.1 完全乱択での写像

次を固定する：\(P_j=P_j^\dagger\)、\(P_j^2=I\)、\(p_j\ge0\)、\(\sum_jp_j=1\)、\(h=\sum_jp_jP_j\)、\(A_j=e^{i\theta P_j}\)。両枝で同位置のlabelを共有し、位置間ではIIDとする。`B_1=A_{j_N}···A_{j_1}`、`B_0=A_{j_N}†···A_{j_1}†`のとき、右端先作用の規約で

\[
B_0^\dagger B_1=A_{j_1}\cdots A_{j_N}A_{j_N}\cdots A_{j_1},
\qquad \mathbb E[B_0^\dagger B_1]=\Phi_\theta^N(I),\quad \Phi_\theta(X)=\sum_jp_jA_jXA_j.
\]

さらに `M=cosθ I+i sinθ h`、`C(X)=E[(P_j−h)X(P_j−h)]`として

\[
\Phi_\theta(X)=MXM-\sin^2\theta\,\mathcal C(X)
\]

が厳密に成立する。通常の\(\mathbb E[A_jXA_j^\dagger]\)のCPTP channelとは異なる。\(\|\Phi(X)\|\le\|X\|\)（operator normでの収縮性）は各\(A_j\)がunitaryだから成立する。

### 3.2 分散分解と限界

\(h|k\rangle=e_k|k\rangle\)として

\[
V_k:=1-e_k^2,\qquad
v_{km}:=\mathbb E_j|\langle m|(P_j-h)|k\rangle|^2,\qquad
v_{kk}=\operatorname{Var}_j\langle k|P_j|k\rangle.
\]

\(P_j^2=I\)により

\[
\sum_mv_{km}=V_k,\qquad
V_k=v_{kk}+w_k,\quad w_k:=\sum_{m\ne k}v_{km}\ge0,
\quad 0\le v_{kk}\le V_k.
\]

この恒等式はClaudeと以前のGPTレビューに整合する。\(w_k\)は各項の状態上の量子分散平均、\(v_{kk}\)は項選択による期待値の古典分散と解釈できる。ただし、**\(V_k\)や\(v_{kk}\)の式が正しいことと、それに基づく有限時間のイベント比・RPE改善が保証されることは別**である。

### 3.3 分岐の独立性・freshness

同位置labelのbranch間共有が\(\Phi\)を定義する。**位置間独立性**は単一\(\Phi^N\)の反復に必要。一方、**毎shotのfresh drawは\(\mathbb E_\omega\)という無条件平均恒等式の必要条件ではない**。列を再利用する場合、条件付きmeasurement noiseと列間ばらつきの分離が必要で、ショット数による標準的な独立平均の信頼保証は変わる。Claudeが前回のこの誤読を訂正したことは適切である。

---

## 4．Q1：平均化定理の候補式の検証

### 4.1 二次の演算子展開は整合する

\(\mathcal L_1(X)=hX+Xh\)とし、\(\mathcal C\)を§3.1の写像とする。Pauli/involutionでは\(A_j=e^{i\theta P_j}\)だから、\(|\theta|\ll1\)で

\[
\Phi_\theta(X)=X+i\theta\mathcal L_1(X)
-\frac{\theta^2}{2}\{\mathbb E_j[P_j^2],X\}
-\theta^2\mathbb E_j[P_jXP_j]+O(\theta^3).
\]

同じ自由発展\(\mathcal U_\theta(X)=e^{i\theta h}Xe^{i\theta h}=e^{i\theta\mathcal L_1}(X)\)との差は

\[
\Phi_\theta(X)=\mathcal U_\theta(X)+\theta^2\mathcal D(X)+O(\theta^3),
\quad\boxed{\mathcal D(X)=-\tfrac12\{I-h^2,X\}-\mathcal C(X)}.
\]

この\(\mathcal D\)と\(\mathcal L_1\)の周波数ブロックへの射影（secular部分）を使う方向性は、有限減衰予算\(N\theta^2\le D\)の平均化に自然である。現時点では**新しい証明ではなく研究契約の候補**として扱う。

### 4.2 Claudeの近似モデルと、その適用条件

Claude §8.1は、概念上

\[
z_N(\psi)=\langle\psi|\Phi_\theta^N(I)|\psi\rangle,
\qquad
z_N^{\mathrm{sec}}(\psi)=\langle\psi|e^{iN\theta\mathcal L_1}e^{N\theta^2\mathcal D_{\mathrm{sec}}}(I)|\psi\rangle
\]

と書き、\(|z_N-z_N^{\rm sec}|\le C(D)[\theta\mathcal E_{\rm off}+\mathcal E_{\rm LF}]\)という形を提案する。ただし\(\mathcal E_{\rm off}\)・\(\mathcal E_{\rm LF}\)のnorm、coupling measure、modeの選定、\(C(D)\)の依存性は未定義であり、現段階では**定理の完成形として扱えない**。

さらに、非共鳴mode間の移送だけでは捉えられない**局所打切り誤差の累積**がある。以下に、これが実際に消えないことを紙上で確認する。

### 4.3 可換なPauli模型でも残る局所高次誤差：候補式への具体的な留保

二つの可換Pauli項\(P_1=Z_1\)、\(P_2=Z_2\)を確率\(p\)、\(1-p\)で選び、計算基底\(|k\rangle=|01\rangle\)に作用させる。\(p\ne1/2\)、\(0<p<1\)とし\(e=2p-1\)とおく。\(|k\rangle\)は\(h=pZ_1+(1-p)Z_2\)の固有状態であり、Pauli項はすべて対角であるから、当該modeから別のmodeへの結合はない。とくに\(p=1/4\)なら\(e=-1/2\)で、当該\(|k\rangle\)は非縮退である。

このとき共有±信号は**厳密に**

\[
z_N(k)=\left(p e^{2i\theta}+(1-p)e^{-2i\theta}\right)^N
       =\left(\cos2\theta+i e\sin2\theta\right)^N.
\]

一方、\(V_k=v_{kk}=1-e^2\)であり、上の二次secular modelは

\[
z_N^{\rm sec}(k)=\exp\{2iN\theta e-2N\theta^2(1-e^2)\}.
\]

\(\theta\to0\)での**紙上Taylor展開**は

\[
\log(\cos2\theta+i e\sin2\theta)
=2i\theta e-2\theta^2(1-e^2)
+\frac{8i}{3}e(1-e^2)\theta^3+O(\theta^4).
\]

従って\(D=N\theta^2\)を固定すると

\[
z_N(k)=z_N^{\rm sec}(k)\,
\exp\{\tfrac{8i}{3}e(1-e^2)D\theta+O(D\theta^2)\}.
\]

これは\(e\ne0,\pm1\)で一般に**\(O(D|\theta|)\)の位相誤差を持つ**。しかしこの模型では別modeへのoff-resonant couplingとnear-resonant leakageはゼロである。ゆえに\(\mathcal E_{\rm off}\)と\(\mathcal E_{\rm LF}\)を、Claudeが意図した**secular部分に既に含めたmode以外の漏れ**だけと定義すると、その右辺はゼロでも誤差は残る。

これは新しい数値実験ではなく、**提示された候補式の適用可否の代数的検算**である。また、平均化の研究可能性を否定する例ではない。誤差に例えば局所展開の累積\(O(D|\theta|)\)を加えるか、各stepの正確な位相補正を取り込んだ比較対象を採用すれば、修正可能な種類の不足である。

**採択前修正事項：Q1の受理statementには、非共鳴・近共鳴の汎関数と別に、局所高次誤差／有限角度の位相歪み／logarithmic-generator correctionを明示する。** それらを\(\mathcal E_{\rm off}\)に含めるなら、数学的な定義と\(O(D|\theta|)\)相当の導出を示す必要がある。何も補正せず\(\theta\mathcal E_{\rm off}+\mathcal E_{\rm LF}\)だけを厳密な一般定理として受理してはならない。

### 4.4 T1aに課すべき最小の数学的受理条件

次の事項を同じstatementの範囲で確認することを推奨する。

1. **比較対象**：secular平均写像、または正確な一step平均\(\Phi_\theta\)のlogに基づく有効生成子。どちらを取るか明記する。
2. **漸近窓**：\(\theta\to0\)、\(N\theta^2\le D\)、比較する\(T_{\rm rel}\)の対応。\(N\)を固定して\(\theta\to0\)する場合と混同しない。
3. **誤差会計**：局所高次項の累積、非共鳴mode、低周波／pair-sum resonance、初期状態・重み、対象信号の小ささを別々に扱う。
4. **一様性**：すべての対象\(N\)について成立するbound。高い非正規の射影条件数や次元因子を無断で無視しない。
5. **有用性**：少なくとも非可換の例で、粗いchannel diamond normより有用な減衰と位相の区別が実現できる候補であること。単純な可換極限だけで独立claimを付けない。
6. **証拠分類**：完全な定理に至らなくても、解析的な十分条件・反例・不能性を`INCONCLUSIVE`とは分けて記録する。

全次元に一様な強い条件を最初から必須とするより、まず「固定モデル・明示されたgap／重み／normで成立し、後に依存性を吟味できる」受理基準が妥当である。**次元非依存が未達でも直ちに理論失敗と宣言しない**。ただし、dimension-explicit boundが化学系で実質的に空なら、独立な論文claimの評価を下げる。

---

## 5．部分乱択での厳密な写像\(\Psi\)：Claude付録A.1の確認

### 5.1 因数分解そのものは正確

決定論half-sweepのforward/reverse wordを\(D,D'\)とし、乱択gateを\(R_1,\dots,R_r\)、\(R=R_r\cdots R_1\)、\(R^{\rm rev}=R_1\cdots R_r\)とする。1 stepの正枝を\(G_+=D'RD\)と定義し、負枝を**各gateの符号を同じ時間順に反転**したものとする。このとき\(G_-^\dagger=D'R^{\rm rev}D\)となり、干渉の1-step mapは

\[
\Psi(X)=\mathbb E[G_-^\dagger XG_+]
        =D'\,\mathbb E[R^{\rm rev}(DXD')R]\,D.
\]

位置間で独立な同一確率lawであり、\(\Phi(X)=\mathbb E[A_jXA_j]\)なら

\[
\boxed{\Psi(X)=D'\,\Phi^r(DXD')\,D}.
\]

Step間にも同一lawでの独立性を仮定すれば\(\mathbb E[B_0^\dagger B_1]=\Psi^s(I)\)。この代数関係は、Claude付録A.1・G0事前契約§7に整合し、**「部分乱択には平均写像が存在しない」という見方を退ける正しい技術的出発点**である。

\(r=0\)なら\(\Psi(X)=(D'D)X(D'D)\)、\(H_D=0\)なら\(D=D'=I\)で\(\Psi=\Phi^r\)。どちらも期待される極限と整合する。

### 5.2 この恒等式がまだ証明していないこと

- \(\Psi\)の固有modeは一般に\(\Phi\)の固有modeではない。\(H_D\)と\(H_R\)が非可換なら、有効生成子を**全Hamiltonianの固有基底**で扱う追加の条件が必要。
- 1 stepの\(\Psi\)が一定でも、各RPE roundの\(\delta,\theta,r\)などが変われば、round間で同じ固有basis・weightを維持できない可能性がある。
- \(\Psi^s(I)\)という形式から、\(\rho_\infty^{(H)}\)の簡単な式、近共鳴の消失、shot数、RMS energy errorは直ちに導けない。
- 正確なbranch relative phase、identity項の処理、実際のH6 native sourceと提案回路の同値性は、依然として別の実装・資源契約が必要。

### 5.3 適用するjoint scalingはS0で固定すべき

T1bで「\(\theta\)が小さい」と言うだけでは足りない。決定論halfの角度\(\alpha\)、相対step幅\(\delta_{\rm rel}\)、一stepの乱択gate数\(r\)、step数\(s\)、全相対時間\(T_{\rm rel}=s\delta_{\rm rel}\)、全減衰予算\(D_{\rm damp}=sr\theta^2\)の**同時の取り方**が必要である。

線形角の参照辞書\(\theta=\lambda_R\delta_{\rm rel}/(2r)\)、\(\alpha=\delta_{\rm rel}/4\)を採用するなら、\(\theta\to0\)は\(r\)を増やすことでも達成できるが、その場合\(\alpha\)は一定に留まり得る。したがって\(\theta\to0\)だけからdressingの小ささは得られない。\(\delta_{\rm rel}\to0\)を含む小step極限を別途固定するか、有限\(\alpha\)に対する補正を正面から扱う必要がある。

また\(\delta_{\rm rel}\to0\)では、有効PFの固有位相の分離も\(O(\delta_{\rm rel}\Delta E)\)まで縮む。したがって**摂動の大きさ／spectral gapの比**が同時に小さいかを確かめる。\(\theta,\alpha,\delta_{\rm rel},r,s\)を独立に小さいと呼ぶだけの説明は受理しない。

---

## 6．dressing（状態の着替え）と先導係数\(\rho_\infty^{(H)}\)

### 6.1 状態の近さ：簡単な強い不等式はある

\(H=H_D+H_R\)、\(H|k\rangle=E_k|k\rangle\)、\(a=\langle k|H_D|k\rangle=E_k-\langle H_R\rangle_k\)、\(D=e^{i\alpha H_D}\)とする。Duhamel恒等式により

\[
 e^{i\alpha H_D}|k\rangle-e^{i\alpha a}|k\rangle
 =i\int_0^\alpha e^{i(\alpha-u)H_D}(H_D-a)e^{iua}|k\rangle\,du.
\]

従って有限次元で**厳密に**

\[
\boxed{\big\|D|k\rangle-e^{i\alpha a}|k\rangle\big\|
\le |\alpha|\,\sigma_k(H_D)
=|\alpha|\,\sigma_k(H_R)}.
\]

これはClaude付録A.2の\(O(\alpha\sigma_k(H_R))\)という直観を支える、追加の数値を必要としない初等的な見積りである。\(\sigma(H_D)=\sigma(H_R)\)は\((H_D-a)|k\rangle=-(H_R-\langle H_R\rangle)|k\rangle\)から従う。交換子の大きさを仮定しなくても**状態ベクトルの近さ**自体は評価できる。

### 6.2 ただし減衰率の**相対誤差**の保証にはならない

Claudeはこの状態の近さから、\(\rho_\infty^{(H)}\)や1イベントの先導減衰係数の**相対**補正が\(O(\alpha\sigma+\alpha^2\|[H,H_R]\|)\)であると述べる。しかし状態の近さは通常、散逸汎関数の差に対する**絶対**的なLipschitz型制御を与えるにすぎない。元の散逸係数が小さい（特に\(1-\langle h_R^2\rangle\to0\)）場合、絶対誤差が小さくても相対誤差は大きくなり得る。

**したがって、\(\rho_\infty^{(H)}\)を受理するには**少なくとも（i）比較する両モデルの係数の正確な定義、（ii）誤差の絶対上界、（iii）分母の下界または「ratio undefined」領域、（iv）有効mode（\(|\tilde k\rangle\)）と元の\(H\)固有状態（\(|k\rangle\)）の差、（v）非正規な左右modeの重み、（vi）step内交差項の扱いを要する。

もう一つの注意として、\(\Psi\)は\(DXD'\)という**同符号・非随伴の二側作用**を含む。Exact-Dで\(D'=D\)としても\(D|k\rangle\langle k|D\)は、密度演算子\(D|k\rangle\langle k|D^\dagger\)と等しくない。従ってdressingを単に「変換された量子状態の分散」と読むのは一般に誤りであり、coherent-superoperatorの左右の作用として扱う必要がある。この境界はClaude付録A.2の双線形記法を正式な定理へ昇格させる際に重要である。

### 6.3 実行前の診断としての意味

\(\alpha\sigma_k(H_R)\)は「deterministic halfで対象状態がどれだけ回転するか」の**必要十分条件ではなく、上からの尺度**として有用である。\(\alpha^2\|[H,H_R]\|\)も非可換性の粗い尺度として記録できる。ただしこの二量だけから\(\Psi\)のスペクトルまたは\(\rho\)の有用性を十分条件として判定しない。両者は**候補診断量**であり、実際の数値取得には別途科学計算の実行承認が必要である。

---

## 7．一般のHermitian event・対称性・量子化学への適用

### 7.1 一般のHermitian生成子に対する二次展開は正しい

\(A_j=e^{i\theta G_j}\)、\(G_j=G_j^\dagger\)有界、\(\bar G=\mathbb E G_j\)、\(C_G(X)=\mathbb E[(G_j-\bar G)X(G_j-\bar G)]\)、\(V_G=\mathbb E G_j^2-\bar G^2\)とすると、紙上Taylor展開から

\[
\boxed{\mathbb E[A_jXA_j]
=e^{i\theta\bar G}Xe^{i\theta\bar G}
-\frac{\theta^2}{2}\{V_G,X\}-\theta^2C_G(X)+O(\theta^3)}.
\]

これはClaude付録A.3と一致する。\(\bar G|k\rangle=e_k|k\rangle\)の場合、候補となる対角secular係数は\(V_k^G+v_{kk}^G\)、\(V_k^G=\langle k|V_G|k\rangle\)、\(v_{kk}^G=\mathrm{Var}_j\langle k|G_j|k\rangle\)となる。ただし、**その係数だけで長時間・近共鳴・RPEの一様保証になるわけではない**。

ここで\(V_G\)は非負演算子（\(\mathbb E(G_j-\bar G)^2\)）だが、単に\(I-h^2\)へ置き換えられるのは\(G_j^2=I\)の場合に限られる。

### 7.2 DF fragmentへの拡張は「追加負担が小さい」と即断しない

General-Hermitian展開を補題として書くことと、実際のDF native blockのsampling distribution、normalizer、gateコスト、制御phase、有限角度発展、sector保存をすべて閉じることは別である。したがってClaudeが提案する「一般のHermite生成子補題をT1aに含める」方針は**任意の将来拡張**なら妥当だが、初期契約の必須完了条件にするとscopeを膨らませる。

初期scopeではPauli／involutionでのT1aと、Pauli-tailでのT1bを先行させ、一般Hermitian event・DF fragmentはT1b後の科学的判断事項とすることを推奨する。ただし、DFのsector保存によって理論仮定が実際に緩むなら、別の新規性候補として再評価してよい。

### 7.3 対称性がpair-sum共鳴を抑えるための正確な条件

乱択eventと決定論進行の全てがある不変部分空間\(\mathcal K\)を保ち、対象状態\(|k\rangle\)が\(H|_{\mathcal K}\)（または正確に用いる有効基準Hamiltonian）の**非縮退な端点固有状態**であるなら、\(E_a+E_b=2E_k\)というpair-sum共鳴は、\(E_a,E_b\)を\(\mathcal K\)内に限定したとき自明な\((k,k)\)以外には生じない。

一方、**粒子数保存だけでは、さらに細かなspin／空間対称sectorに置いた目的状態が\(\mathcal K\)全体の端点であると保証できない**。DF eventが粒子数を保存しても、spinや他の保存量と固有空間をどの単位で閉じるか、また有効PFのaliasingがないかは別条件である。さらに非involution eventの正規化と一イベントの費用はPauliとは異なる。

したがって、Claude§8.2(6)の「DF fragmentではsector外のpair-sum共鳴が構造的に消える」は、**共通に保存される到達部分空間と端点性を確認した場合**に限定すべきである。粒子数保存それ自体を、M1-S成立の十分条件として扱わない。

---

## 8．有限時間の\(\rho(T)\)と費用閾値：何がどこまで言えるか

### 8.1 \(\rho_\infty\)の正しい意味

完全乱択のisolated modeに対し、\(V_k=1-e_k^2>0\)、\(v_{kk}\)を§3.2で定義する。\(N\theta^2=D\)を固定し、十分な非共鳴・small-angle極限を取ると、共有±の先導減衰係数が\(V_k+v_{kk}\)、ordinary／独立±の比較係数が\(2V_k\)となる場合、**同じ相対時間・同じ振幅減衰予算・同じ一イベント価格**の下で

\[
\rho_\infty=\frac{V_k+v_{kk}}{2V_k}=\frac12+\frac{v_{kk}}{2V_k},\qquad 1/2\le\rho_\infty\le1.
\]

分散の恒等式自体は厳密。ここで\(\rho\)を**必要イベント数の比**として読むためには、全次数の近似・比較コスト・別のbias条件を要する。\(V_k=0\)では比は未定義。部分乱択の\(\rho_\infty^{(H)}\)については§6の留保を維持する。

### 8.2 Claude付録A.4のsinc-kernel

Duhamelの\(\mathcal C\)一次項を非摂動の\(M\)基底へ展開すると、\(\sum_{n<N}\cos(n x)\)の振動核が現れる。\(x\)が小さく、\(N\)が大きく、他のmodeのradial差・上位の共分散項・有限角度補正を無視する**一次近似**では、\(F_N(x)=(1/N)\sum_{n=0}^{N-1}\cos(nx)\)が\(\sin X/X\)型に近づく。この関数は負になる範囲があるので、**中間時間の近似比が長時間極限の\(\rho_\infty\)を少し下回る可能性**という指摘は数学的に筋が通る。

しかし、これは次を証明していない。

- 実際の有限\(N,\theta,D\)で\(\rho(T)\)が常に一つの確定した区間に入ること。
- \(\rho_\infty-0.109\,w/V\)が実際の平均信号・要求イベント数の普遍的な下界になること。
- \(N\theta^2=O(1)\)の全次数、複素モード干渉、低周波結合に対して同じ最小値が保たれること。

従って付録A.4の区間は**「一次kernelの示唆」へラベルを限定**し、G0-T1における正式なfinite-time ratio guaranteeにはしない。以前の強い\([\rho_\infty,1]\)という表示を取り消すこと自体は正しい。

### 8.3 \(\rho^*=1-S^*/f_R\)は代数式、採択済み閾値ではない

固定した分割、等しい単価・同じ減衰やshot・追加biasを無視した加法コストモデルで、総費用に占める乱択部割合を\(f_R>0\)、乱択部イベント比を\(\rho\)とすれば

\[
S\simeq f_R(1-\rho),\qquad \rho\le1-\frac{S^*}{f_R}
\]

は代数的に正しい。Claudeが固定の0.7／0.85を撤回し、目標効果\(S^*\)と費用内訳に依存させたことは改善である。

ただし\(f_R\)は、目的が**全RPEの期待総費用**ならround/shot/実行回数で重み付けした全体の費用割合として定義する必要がある。「後半roundの乱択部割合」は、後半roundが総費用を支配することを同条件で確認した場合の近似である。\(f_R\)自体は現在の対象protocolで`NOT_ESTABLISHED`。D/R再最適化、HWP batching、合成後の融合、state-prep、online classical費用などを含むと、上の等式はそのまま使えない。

**この式から直接、Gate 0の科学的GO/STOP閾値を設定しない。** 研究上のGOには、\(\rho\)の正当性・誤差・signal floor・RPE accuracy・同条件価格が必要。\(f_R=0\)または\(S^*>f_R\)では式の使途を別途明記する。

---

## 9．G0-T1研究契約・科学的新規性・論文化の着地点

### 9.1 最優先対象の比較

| 選択肢 | 主要な内容 | GPTの独立評価 |
|---|---|---|
| 現行Q1–Q5をまとめて採択 | 完全乱択、一般スペクトル、部分乱択、RPE、native費用 | **推奨しない**。論点と受理条件が広すぎて失敗理由を切り分けにくい |
| Q1だけを採択 | 完全乱択のsecular平均化 | **補題級**。既知の平均化・摂動論の適用として新規性が限定的になりやすい |
| Q1＋Q2を一般理論として完遂 | あらゆる縮退cluster・Jordan・uniform dimension | **初期scopeでは推奨しない**。研究価値に比べ負担が過大になり得る |
| **Claude A′：T1a（Q1-core＋Q2-min）→T1b（Q3-core）** | 完全乱択の信号保証を使って、部分乱択の先導次数の移送が可能か早期に判断 | **修正条件付きで最も合理的**。初期研究の情報価値とSTOPの節目が明確 |
| 探索的Gate 0をまず実行 | 実分子で未認証proxyの傾向を見る | 独立の科学計算承認を要し、proxyと実際のDF/PRが一致せず誤った判断の危険。**現時点では優先しない** |
| M1-Sを今すぐ正式採択・G0一括開始 | 全範囲を研究実行へ移す | **推奨しない**。理論statementと比較contractに未閉鎖事項がある |

### 9.2 最初の論理的な研究価値判定は「T1bの移送」

既知のdirectional qDRIFT、回文構造、共有lawという外形を超えて論文の中心claimを作るためには、**部分乱択までの有用なcoherent signal保証**が重要である。T1aで完全乱択に関する摂動補題を得たとしても、それだけなら既知数学の再編成やtechnical noteの可能性が高い。T1bで次の点が示せるなら、研究価値は上がる。

- \(H_D\ne0\)の正しいcontrolled ordered lawについて\(\Psi\)の厳密な構造がある（§5で既に成立）。
- 同じ物理時間・同じHamiltonian・同じ情報・比較する減衰予算の下で、完全乱択の先導改善機構が**自然なdressing／gap／alias条件**の下でも存続する。
- その保証が、qDRIFTの通常channel誤差評価や既存PR/RTE guaranteeを単に書き換えただけでない価値を持つ。
- 実分子で条件が極端に強くなく、評価できる余地がある。

ただしこれらは**T1bが成功した場合の将来の研究価値**であり、今回既に確認済みの結果ではない。次元非依存の重み付き理論ができたとしても、その数値的な条件・状態・分子の実用性はなお未証明となる。

### 9.3 研究成果の段階的な着地点

- **T1aのみ成立**：既存平均化理論を特定のcoherent signalへ適用したtechnical note・内部技術資料。論文化の独立性は限定的。失敗しても基準・反例・適用限界の分析を残す。
- **T1a＋T1bの有用な移送が成立**：**「分岐相関を用いたpartial randomized simulationに対する条件付きcoherent signal減衰・位相保証」**という限定理論claim候補。既存法との同値性・定量的改善幅・量子化学への適用可能性をなお検証する。
- **後日Gate 0で有用な領域が確認**：H4/H6等の特定条件で、oracle-stateに依存したイベント効率の参考証拠。oracle診断をオンラインpredictorの利点と同一視しない。
- **さらにQ4／Q5とsame-accuracy・strong baseline費用比較を通過**：部分乱択RPEの実資源改善という本来の最も強い論文claim候補。
- **T1b不能または条件が非現実的**：M1-SのPR資源改善claimを一度区切り、T1aをtechnical note化するか縮小／STOPを検討する。M2やTrack B全体の扱いは別の科学判断。

### 9.4 先行研究の境界

PR原論文Appendix A.4は、対称PFの**サンプルごとの厳密な随伴性**に基づくhalvingについて正しく記述している。したがってM1-Sを「PRの誤りを覆す」と表現しない。Campbellの拡張Hamiltonianを用いるcontrolled qDRIFT、Kato/Davies型の摂動・平均化、KGWの状態分散といった近接結果は強いbaselineである。

今回調べた固定資料では、M1-Sの**特定law＋coherent signal＋部分乱択移送＋RPE資源保証**の全てを含む既存同等定理は特定されていない。しかし、その不存在は証明されておらず、同じ構成の単純な言い換えが見つからないこと自体は新規性を認定する根拠ではない。**成果の独立性は得られた定理と既知保証を仮定・出力・費用で比較してから判断する。**

---

## 10．G0-T1開始前のS0：修正すべき科学契約

### 10.1 S0で科学的に固定すべき最小範囲

S0では、Codexが技術的な契約文書を作れるが、**研究目的や主要statementをCodex単独で採択しない**。GPTによるこの推奨とユーザーの正式採択を分ける。次の範囲を固定することを**ユーザーが採択する場合**に推奨する。

1. **研究仮説**：「同じlabelのdirectional qDRIFTは、完全乱択の非共鳴領域ではcoherent signalの先導減衰を変え、その一部が対称stepの部分乱択へ移せる」。既知の構成を新しいアルゴリズムの発明と表現しない。
2. **T1aのlaw**：Pauli/involution、\(p_j\ge0\)、IID、同位置2枝のlabel共有、線形角、phase/identity台帳、初期状態・対象固有mode、無条件平均とshotモデルの区別。
3. **T1bのlaw**：Pauli-tail、\(H=H_D+H_R\)、中央r個の乱択gate、exact deterministic half（PFは後段）、\(D,D'\)の順序、相対時間\(T_{\rm rel}\)、\(\alpha,\theta,r,s,\delta_{\rm rel}\)のjoint scaling、全Hの対象modeと対称sectorの関係。
4. **T1a受理statement**：\(\Phi^N(I)\)の信号水準近似と厳密な誤差。**局所高次項累積を含む**、nonresonant/low-frequency/pair-sum contribution、signal floorが必要なときの条件、\(N\theta^2\le D\)の一様性を区別。
5. **T1b受理statement**：\(\Psi\)の厳密mapから、\(H_D\ne0\)での有効mode、dressing・commutator・alias・低周波／pair-sum・outer PF biasの区分を定め、\(\rho_\infty^{(H)}\)の式が成立する／修正が必要／使えない、のいずれかを判定する。
6. **比較条件**：ordinary/独立±の強いqDRIFT、既知PR-RTE、同じ相対時間・物理target・状態・実行精度・情報アクセス・費用座標。減衰だけの効率比をRPE改善と読まない。
7. **Checkpoint**：T1aの評価後にT1bへ進める価値を確認（必要ならGPTが科学判断）、T1bの移送判定後には**必ず重要GPTレビュー**。各段階で新規数値計算はしない。

**S0は「新しい理論を完遂したこと」ではなく、研究実施前に何を証明・反証し、何をもって未解決として止めるかを定める契約である。** 本レビューの内容をS0文書に移すことは可能だが、ユーザー承認前に研究契約・authorization/markerを変更しない。

### 10.2 進行権限の区別

- **今回のレビュー開始承認**：G0-T1を実施する承認ではない。
- **研究方針の正式採択**：ユーザーがG0-T1、A′、対象lawとcheckpointを採用する行為。
- **S0の技術的固定・公開**：採択されたscopeに従い、Codexが文書化・整合性確認して公開。科学的なstatementの承認はユーザー／GPTの別階層。
- **T1a/T1bの新しい理論作業**：承認された対象・仮定・STOPで実施する。既存の別研究で認められた紙上検算を無条件に転用しない。
- **Gate 0／E0／行列・期待値・数値シミュレーション**：独立した具体的な科学計算実行承認が必要。
- **Q4（RPE）／Q5（native資源）**：T1b以降の結果を踏まえる追加の科学契約・実行承認が必要。

---

## 11．リスク、反論、研究選択の公平性

### 11.1 主要な研究リスク

**① 方法の新規性の弱さ**：qDRIFT＋directional controlの外形自体は自然で既知。coherent averagingの結果も一般摂動論の直接系に終わる可能性がある。これを新規アルゴリズムと呼ばない。**T1bまで到達しても直ちに独立論文としての主張を許可しない**。

**② 数学的なスケーリングの難しさ**：小さい\(\theta\)では演算子の自由周波数間隔も縮み、\(N\)反復での定数は\(\theta\)、gap、低周波重み、次元・結合の形に依存し得る。\(N\theta^2\le D\)の一様性は簡単ではない。強すぎる仮定で有限分子しか対象にできない場合、一般スケーリングclaimを縮小する。

**③ 分子系での成立性**：Pauli-tailは電子数／spinの目的sectorから出る可能性がある。H4/H6の既存データはDF表現であり、Pauli-tail化は別のlaw／normalizer／gate費用。保存state・ハミルトニアン・spectral dataの情報アクセスもoracle診断と実用predictorを区別する。FeMoco等の近縮退領域へ無条件に一般化しない。

**④ 誤った早期STOP**：未認証proxyが\(\rho\approx1\)でも、dressing補正後やDF eventで異なる可能性がある。T1aに厳しい次元非依存を課して失敗しただけで、有限系での科学的可能性まで否定しない。

**⑤ 実用資源の過大評価**：イベントの数、1イベントのnative価格、Hadamard測定shot、RPEの初期overlap、state preparation、古典online費用の区別。\(S\approx f_R(1-\rho)\)を保証済み総費用削減と誤読しない。

### 11.2 代替方針

- **案A（フル理論先行）**：厳密性は目指しやすいが、最初から一般定理を大きくし過ぎるリスク。
- **案B（探索数値先行）**：早く化学の傾向を見られるが、現状の\(\rho^{(H)}\)は未認証proxy。追加数値承認が必要で、Gate 0のGO/STOPを代行できない。
- **案A′**：T1aは短く、T1bの移送可能性を早めに検証。**研究価値が必要とする情報を最小の理論投資で得る**方針として支持。
- **M2／Track A統合／Track B休止**：M1-Sの理論・化学系・nativeに失敗した場合の合理的な代替。M1-Sの限定的な失敗を全Track Bのgeneral no-goにしない。

案A′の情報価値は高いが、**T1aの実施にどれだけ時間がかかるか・T1bの証明がどれほど難しいかは実測されていない**。無条件に「最も安い研究」とは言えない。S0で少数の必要十分な成果物と明確なcheckpointを固定する意義がある。

---

## 12．最終的な採択推奨・次段階・STOP条件

### 12.1 推奨

**`RECOMMEND_G0_T1_A_PRIME_AFTER_S0_REPAIR_NOT_ADOPTED`。**

- **研究契約**：Claude提案のG0-T1を、**Q1の局所高次誤差とT1bのjoint scaling／dressing caveatを反映して修正**することを条件に推奨。
- **実施順序**：**A′（T1a→Checkpoint 1→T1b→Checkpoint 2→Gate 0候補）**。
- **初期law**：Pauli/involution＋方向制御共有±。部分乱択T1bはexact deterministic halfの中央Pauli-tailに限定。一般Hermitian/DF event、Q4/Q5は初期scope外。
- **開始前の要件**：S0で比較時間・角度・振幅減衰・局所誤差・低周波weight・有効mode・phaseを文書化し、ユーザーが採択すること。
- **科学的価値の見込み**：条件付き理論研究として中、総資源改善の確率・大きさは未確立。
- **結論の変更強制なし**：Claudeの狭い理論先行という中心的結論は維持する。ただし数学的な受理statementの精密化は必須。

### 12.2 Checkpointごとの受理／方向転換の枠組み

| 節目 | 何を確認するか | 判断候補 |
|---|---|---|
| **S0完了** | 原本・law・statement・error accounting・時間／角度／価格・STOP条件が固定される | 必要な実行承認のもとでT1a開始、または契約差し戻し |
| **Checkpoint 1（T1aの結果）** | 局所高次誤差を含む一様信号評価、非共鳴・近共鳴の意味、強い既知対照との差、T1bに使えるか | T1bへ限定GO／さらに範囲縮小／technical note／STOPを推奨。**これは自動実行許可ではない** |
| **Checkpoint 2（T1bの結果）** | 正しい\(\Psi\)、全H-mode、joint scaling、dressing/alias、改善因子の正当化、適用不可能例 | Gate 0の評価対象選定、理論縮小、M2等への転換、STOPを**重要GPTレビューで判断** |
| **後日のGate 0** | 科学的に正当化された指標ベクトル、状態・law・費用モデル・数値誤差・oracleの範囲 | Q4／Q5へ進む科学的判断の材料。実行は別途承認 |

**T1aが未完了でも“先へ進めば何とかなる”としてT1bへ進まず、T1bが難しいからと言って未承認の数値計算を先行しない。** 一方、十分に意味のある部分結果が出た場合、一般定理が未完成でも、その内容・限界をレビューして縮小contractへ切り替える道を残す。

### 12.3 担当・権限

- **ユーザー**：G0-T1/A′の暫定方針、S0の修正項目、研究契約の正式採択、T1a/T1bの新規理論作業承認、Gate 0の別途計算承認を個別に判断。
- **Codex**：承認された科学方針の範囲でS0契約の技術的記述・出典/implementation semanticsの照合、既存GitHubの保全と文書公開。**新規の理論証明や科学計算を指示されない限り開始しない。**
- **GPT／Claude**：S0での重要な科学的statementの採択判断、Checkpoint 1の価値判定（必要に応じて）、Checkpoint 2の重要科学レビュー、結果が出た場合の新規性と論文claimの精査。
- **数値研究の担当**：Gate 0など、範囲・対象・回数・資源・STOPを別途ユーザー承認したうえでCodex。

### 12.4 Mandatory STOP

**現在の状態は `GPT_REVIEW_COMPLETE_G0_T1_REPAIR_RECOMMENDED_NOT_ADOPTED_STOP`。** 新しい数学定理・数値simulation・行列演算・回路compile・Gate 0・科学的classification変更・research GOは実施していない。M1-S／G0-T1の正式採択とG0-Tの実行承認はなく、既存の`mandatory STOP`・parent-worktree・過去のauthorization/markerは維持する。

---

## 13．出典・判断の追跡先

### 13.1 GitHub固定資料

- [G0事前契約 Decision Packet（`0f0952614dcf5c06d3d63c21b5d49091b8ecce37`）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md)
- [G0事前契約 evidence index（同commit）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/evidence_index.json)
- [前回GPT独立科学レビュー（`4f5114e137563bb871c9f881f9eadab0c180bdd4`）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4f5114e137563bb871c9f881f9eadab0c180bdd4/docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md)
- [M1-S G0-pre Decision Packet（`8b7666d0270f7fac01e430e195bef7b58ac0116c`）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md)
- [M1 P0.5技術報告（`ed3ddeacc88e998e7bc1d8c0db8163e106312791`）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md)

### 13.2 外部一次論文

1. Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum **7**, 020332 (2026), [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2). Appendix A.4（対称PFの半減、A42–A46）、Appendix B（RPE）、Appendix E（費用）。
2. Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, [arXiv:2603.22778v2](https://arxiv.org/abs/2603.22778v2). Appendix B（B3–B5、γ、C_gate）。
3. Campbell, *A random compiler for fast Hamiltonian simulation*, Phys. Rev. Lett. **123**, 070503 (2019), [arXiv:1811.08017](https://arxiv.org/abs/1811.08017). Appendix D、[著者受理稿](https://eprints.whiterose.ac.uk/id/eprint/150026/1/1811.08017.pdf).
4. Childs–Ostrander–Su, *Faster quantum simulation by randomization*, [arXiv:1805.08385v2](https://arxiv.org/abs/1805.08385v2)（G0-preの確認範囲）。
5. Kivlichan–Granade–Wiebe, *Phase estimation with randomized Hamiltonians*, [arXiv:1907.10070v1](https://arxiv.org/abs/1907.10070v1)（G0-preの確認範囲）。
6. Chen–Huang–Kueng–Tropp, *Concentration for random product formulas*, [arXiv:2008.11751v4](https://arxiv.org/abs/2008.11751v4)（G0-preの確認範囲）。
7. Mohammadipour–Li, *MLMC-qDRIFT*, [arXiv:2604.26865v2](https://arxiv.org/abs/2604.26865v2)（G0-preの確認範囲）。
8. Gu et al., *Noise-resilient phase estimation with randomized compiling*, [arXiv:2208.04100v2](https://arxiv.org/abs/2208.04100v2)（P0.5の比較対象）。
9. Kato, *Perturbation Theory for Linear Operators*（一般的手法の背景。今回、M1-Sへの完全な適用定理は確認していない）。
10. Davies, *Markovian master equations*, Commun. Math. Phys. **39**, 91–110 (1974)（secular／弱結合極限の背景。今回の離散写像への適用を証明していない）。

### 13.3 数学的に新しく確認した内容の分類

- **既存レビューと一致する厳密恒等式を今回照合**：\(\Phi\)の共分散分解、項間／項内分散、\(\Psi=D'\Phi^r(D\cdot D')D\)の因数分解。
- **Claudeの紙上式を代数的に検算**：一般Hermitian eventの二次展開、固定コストモデルの\(\rho^*\)、局所的なDuhamelの状態ノルム評価。
- **重要な科学的留保を追加**：Q1のsecular候補式に\(O(D|\theta|)\)級の局所高次誤差が欠ける可能性（可換模型で明示）、dressingの相対値への飛躍、joint scalingの未固定、有限時間の比の未保証、sector端点条件。
- **未実施**：新しいT1a/T1bの一般定理、近共鳴／Jordan clusterの完成解析、分子の科学数値評価、RPE・資源forecast、既存データの再集計、GitHub書込み、M1-S採択。

---

## 14．最終自己監査

- [x] 最新Claude原本を添付から直接確認した。
- [x] G0-CONTRACT・GPT-PREV・G0-PRE・P0.5・evidence indexを固定GitHub refから取得し、引用先と本文を照合した。
- [x] PR Appendix A.4、富士通 Appendix B、Campbell Appendix Dの一次公開資料の該当箇所を確認した。
- [x] 代数恒等式と、未証明の長時間保証・部分乱択移送を分離した。
- [x] 回路の方向・順序・共有確率law・relative phase・shot freshnessの意味論を区別した。
- [x] Q1候補の欠けた誤差項を可換な例で検算した（数値simulationなし）。
- [x] Q1–Q5の初期scope、A′、Checkpoint、Gate 0実施順序と新規性を独立判断した。
- [x] 研究の失敗・縮小・STOPとM2等の代替を公平に扱った。
- [x] ユーザーの正式採択、新規理論実施、Gate 0の新規科学計算承認を代行していない。
- [x] mandatory STOP、Parent allocation D＋E、既存のauthorization/marker/source/result保護を維持した。
- [x] 完成時はMarkdownリンクのみで回答する。
