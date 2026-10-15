# Track B／M1-S：Claude完成レビューに対するGPT独立科学レビュー

**作成日：2026-10-15 JST**  
**レビュー担当：GPT-6**  
**レビュー対象：** ユーザー提出のClaude完成版 `レビューinner.md`（全767行、2026-10-15）。2026-10-15版のP0／P0.5／G0-preおよび既存GPTレビューと照合。  
**状態：** `GPT_INDEPENDENT_SCIENTIFIC_REVIEW_COMPLETED_M1S_CONDITIONAL_THEORY_CANDIDATE_NOT_ADOPTED`  
**権限：** ユーザーが本GPT科学レビューの開始を明示承認。**M1-S研究の正式採択、G0の実行、Gate 0を含む新規科学計算、実装、repository変更は未承認。mandatory STOP維持。**  
**主旨：** Claudeの結論を変更するためのレビューではない。中心恒等式、長時間漸近、先行研究との差分、実task・資源への接続をそれぞれ独立に判定する。

---

## 0. エグゼクティブサマリー

### 0.1 最終科学判断

**Claudeの `B：M1-Sを条件付きで推奨` という方向性を、対象を絞った理論検証候補として条件付きで支持する。** ただし、これは**M1-Sの方法GOでも、G0／Gate 0の自動実行GOでもない**。

前回GPTの `C：元のM1を暫定再定義し、G0-preを先行` という判断は、当時の証拠に対して適切だった。今回、G0-preの8論文調査とClaudeの新しい**固定超演算子の固有値に着目した議論**、**2準位での記号的な厳密例**が追加されたことにより、狭い理論検証へ投資する根拠は以前より明確になった。したがって、**Cから条件付きBへ移行すること自体に合理性がある**。

ただし、以下を揃えた**実用的なPR/RPEの資源改善は未成立**である。

1. 同じ物理時間・状態・精度・失敗確率に対する比較。
2. 非可換な決定論部を挟む部分乱択での厳密な平均写像と摂動上界。
3. 密なスペクトル／pair-sum共鳴／対称sector外遷移の扱い。
4. 多round RPEのsignal floor・overlap・aliasing・bias予算。
5. Native controlled loweringと統計費用を含めた公平な総資源会計。

このため、**最初に着手すべきなのは、代表分子に対する無条件の数値Gate 0ではなく、先に理論上の比較契約と部分乱択での鍵量の意味を確定する、小さく限定した理論G0-Tの準備**である。数値Gate 0を必要とする場合は、その後に対象・情報アクセス・閾値を固定し、ユーザーの別途承認を得る。

### 0.2 Claudeの重要な主張別判定

| Claudeの主張 | GPTの評価 | 根拠・留保 |
|---|---|---|
| M1-Sは `Z_c\otimes H` にqDRIFTをかけた構成である | **支持・高** | 符号と相対時間を揃えれば、方向制御Pauli回転のIID列そのもの。回路の外形は新発明ではない |
| `\mathbb E[B_0^\dagger B_1]=\Phi^N(I)` | **支持・高** | 位置間IID・位置ごとのlabel共有・同順sign reversalで成り立つ。ただし**fresh drawのshotごとの要件は「平均の等式」には不要** |
| `\Phi(X)=MXM-\sin^2\theta\,\mathcal C(X)` | **支持・高** | 有効な共通角Pauli-involutionの厳密恒等式。通常のCPTP channelの写像ではない |
| `V_k=1-e_k^2=v_{kk}+w_k` | **支持・高** | 項選択に伴う期待値の分散と、各項の状態内量子分散の恒等的分解 |
| 孤立スペクトルでは固有値摂動を先に求めてからN乗できる | **妥当な証明方針・中** | 固定有限次元・スペクトル分離・射影の条件数・非共鳴が必要。**一様剰余証明は未完成** |
| `\rho_\infty=(V_k+v_{kk})/(2V_k)` | **限定的に支持・中〜高** | **完全乱択・hの対象固有state・V_k>0・非共鳴・small-angle／長時間の先導漸近**として整合。有限時間・部分乱択・総費用の普遍的な比ではない |
| 2準位例で `|\lambda_\pm|=|\cos\theta|` | **small-angle域で支持・高** | 表示根号が実数になる領域では正しい。**任意のθで無条件には成立しない**ため明示が必要 |
| 部分乱択でも `\Psi^s(I)` の形にできる | **形式的には支持・中〜高** | 完全なordered full-step lawが固定され、step間IIDなら成立。完全乱択の固有値係数・\(\rho_\infty\)をそのまま移送できない |
| 部分乱択の `\rho_\infty^{(H)}` をGate 0で利用する | **未確立・低〜中** | 全Hの固有状態で定義した候補式は、新たな摂動展開を要する。現時点では**候補診断量**であり科学的GO/STOP指標としての正当化が未了 |
| Gate 0の `0.7`／`0.85` を閾値にする | **未採択** | レビュアーの便宜的目安。数理保証・多候補探索・native費用との接続がない |
| 結論B：段階的G0に進む価値がある | **条件付きで支持・中** | 研究課題を「既知方向制御のcoherent信号効率保証」に限定し、既知法との比較契約を先に固定するなら妥当 |

### 0.3 重要な修正点

Claudeの中心となる厳密恒等式や2準位模型を否定する根拠は見つからなかった。一方、重要度の高い限定・修正は次の五点である。

1. **「毎shot新しい乱数列がなければ `\Phi^N(I)` の平均等式が成り立たない」は正しくない。** 新しいdrawが必要になるのは主にshots間の独立性・集中保証であり、ensembleに関する無条件期待値とは区別する。
2. **固定有限次元の非共鳴極限**での固有値解析と、分子の大きなHilbert空間での**スペクトル分離が小さくなる極限**は別。`O(\theta/\Delta)`は条件付きであり、粒子数が増えても一様という結論はない。
3. **PR原論文のhalving trickの「不適用」を、誤りの反証と表現しない。** 元の主張は標本ごとの厳密な対称性を要求する恒等式についてである。M1-Sは**別の乱択channelを採用して近似的な位相信号を得る候補**である。
4. **部分乱択の `\rho_\infty^{(H)}` は未証明であり、これを用いた数値Gate 0の閾値を先に正式採択しない。** Pauli-tailに限定した場合でも、outer PF・全Hと\(h_R\)の固有basis差・pair-sum共鳴・同情報比較が必要。
5. **正確な基底状態に依存する `v_{kk}`・gap・\(\rho\)を新アルゴリズムの無料predictorに流用しない。** オフラインの研究診断に利用する場合は`exact-state/oracle diagnostic`と明示し、実行可能な選択手順や本番資源と分離する。

---

## 1. 証拠、独立性、実施範囲

### 1.1 固定GitHub資料：取得を確認したもの

Repository：`HIROMU1015/Partially-Randomized-Trotter`。以下の固定refはGitHub接続を通して内容を取得した。

| ID | 固定commit・資料 | 今回の利用範囲 |
|---|---|---|
| [G0-PRE] | `8b7666d0270f7fac01e430e195bef7b58ac0116c` — `docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md` | 全体、8論文との対応、protocol、費用契約、未解決事項 |
| [G0-INDEX] | 同commit — `artifacts/track_b_m1s_g0_pre/2026-10-15/v1/evidence_index.json` | 取得索引・元レビューのprovenance確認 |
| [G0-LIT] | 同commit — `artifacts/track_b_m1s_g0_pre/2026-10-15/v1/primary_literature_locator.csv` | 8論文の版と関連箇所の索引 |
| [GPT-1] | `41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03` — `docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md` | 前回GPTの科学判断・恒等式／長時間展開の留保 |
| [P05] | `ed3ddeacc88e998e7bc1d8c0db8163e106312791` — `docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md` | 既知定理、H6 v5の実装境界、D/R費用不足 |
| [P0] | `fa91ffb3d4880d767e55f0f661e8ade294b4c9fe` — `docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md` | PR原論文、Campbell等の既知性・signal意味論 |

主たるClaude資料は今回のユーザー添付 `レビューinner.md`（全767行）を直接読解した。Claudeの添付原本と、Codexで以前に公開した「GPTレビュー原本」は**異なる資料**であり、執筆・検証の責任を混同しない。

### 1.2 外部一次文献の確認範囲

- Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026), [arXiv:2503.05647v2](https://arxiv.org/abs/2503.05647v2)：Appendix A.4の「対称PFの厳密なhalving」、部分乱択でRTEを利用する部分を出版社版の公開テキストとP0／G0-preの原文locatorにより確認。arXivのPDF/HTML直接取得は今回のWeb経路では成功しておらず、**PDFの全44頁を再読したとは主張しない**。
- Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, [arXiv:2603.22778v2](https://arxiv.org/abs/2603.22778v2)：公開PDFのテキストからAppendix B式(B3)–(B5)、`γ=1`（qDRIFT）、`γ=2`（RTE/TE-PAI）、`C_gate=1`（STARのrotationsに関するモデル設定）を確認。PDF画像取得機能はcache missで失敗したため、**紙面画像を独立に検証したとは主張しない**。
- Campbell, *A random compiler for fast Hamiltonian simulation*, [arXiv:1811.08017](https://arxiv.org/abs/1811.08017)：controlled qDRIFTについて原論文の公開要約・図説明とP0の本文監査を照合。今回Appendix Dの全証明を独立には再検証していない。
- COS、KGW、MLMC、SCU、CHEN等：G0-preの該当原文locatorとCodex監査を参照した。**今回全8本を独立に再読したわけではない。** 同一保証の不在・世界初の新規性は認定しない。

外部論文の読了深度とGitHubファイルの取得・内容確認は区別する。一次文献の記述は、該当する項目についてのみ根拠に使い、既存監査で未取得の章を読んだことにしない。

### 1.3 今回実施したこと・実施しなかったこと

**実施：** ユーザー添付Claudeレビューの数式・論理の独立検算、GitHub固定資料の内容確認、限られた一次文献の該当記述の確認、研究価値・反論・先行研究の差分評価、Markdown研究記録作成。代数的な不等式・極限・総分散の恒等式を紙上で検討した。

**実施していない：** 新しい数値行列・固有値の計算、simulation／sampling／trajectory、科学的な資源forecast、保存結果の科学的再集計・再ランキング、compile／synthesis、operator guard、G0の新定理構築・全次数証明、登録実験の変更、GitHub書込み、正式なM1-S採択。

---

## 2. 研究問題：何が新しいのか

### 2.1 構成の外形は既知の方法の自然な変形

`H=\lambda\sum_j p_jP_j`、`P_j^2=I`、`p_j\ge0`、`\sum p_j=1` とする。制御ancillaの`Z_c`を導入し、`H_dir=Z_c\otimes H`を考える。`Z_c\otimes P_j`もinvolutionで、係数の正の総和は元の\(\lambda\)と同じである。各項へqDRIFTを適用すると、各乱択位置で共通のlabelを選ぶ`e^{\pm i\theta Z_c\otimes P_j}`が得られる。これは**gate共有±の提案回路**に一致する（符号・相対時間の規約は合わせる）。

理想的には、枝の相対位相は`e^{\pm i HT}`を用いた`e^{\pm i H(T/2)}`の干渉に対応する。ただし実際の乱択標本では、**同じ順序で符号だけを反転した回路は、全wordの厳密adjointと一致しない**。したがって、一つの標本で対称PFの恒等式を満たしているとは主張しない。

**判断：** 「方向制御」「qDRIFT」「乱択gateを2枝へ載せる方法」それ自体のアルゴリズム的独立新規性は低い。研究に残る候補は、**その既知／自然な構成が、信号の複素一次moment・減衰・位相・RPE成功条件にどのような定量的効果を与えるか**について、既知のchannel norm保証よりtaskに合った新しい保証を示すことである。

### 2.2 PR原論文への「反論」という表現を限定する

PR原論文 Appendix A.4の「halving trickはrandomized product formulasには適用できない」という記述は、**対称積公式のサンプルごとの随伴・時間反転恒等式**がランダムwordにはない、という意味で正しい。M1-Sが示そうとするのは、その厳密同値を回復することではない。

M1-Sは**分岐間に共有乱数を持つ別のensemble**を実行し、理想の相対時間発展へ近づくために新しいsignal biasを引き受ける。したがって、研究仮説として「従来は不適用とされた操作の、別の乱択実装による効率的代替」を問うことには意味があるが、**元論文の数学的な記述を誤りとして覆すとは表現しない**。

### 2.3 最も強い対照

独立した研究claimが成立するには、最低限次の強い対照を固定する必要がある。

1. **PR型RTE／独立±：** same task・同じ誤差保証・最適化機会を与えた比較。
2. **部分乱択qDRIFT／ordinaryまたは独立±：** `γ=1`という既知のランダム部費用係数を利用する強い対照。**RTEだけをbaselineにして既知のfactor-twoを再主張しない。**
3. **方向制御の既知native lowering／HWP等：** primitiveの数、制御枝の位相、ancilla等のoverheadを公平に扱う。
4. **最小の対照問題：** `H_D=0`（完全乱択）の解析はproof-of-mechanismであり、**部分乱択RPEへの有効性を示す本命のbaselineではない**。

同じ情報アクセス条件・初期状態条件・失敗確率・物理時間を基準にしなければ、独立した方法貢献や実用資源優位を認定しない。

---

## 3. 数学的核心：厳密に支持できること

### 3.1 枝の演算順序と写像の恒等式

`A_j=e^{i\theta P_j}`、`B_1=A_{j_N}\cdots A_{j_1}`、`B_0=A_{j_N}^{\dagger}\cdots A_{j_1}^{\dagger}`とする。両枝は**同位置でlabelを共有**し、異なる位置ではIIDとする。干渉wordは

\[
 B_0^{\dagger}B_1=A_{j_1}\cdots A_{j_N}A_{j_N}\cdots A_{j_1}.
\]

`\Phi_\theta(X)=\sum_jp_jA_jXA_j`を導入すると、内側から乱数期待値を順に取って

\[
 \boxed{\mathbb E_{\omega}[B_0^\dagger B_1]=\Phi_\theta^N(I)}.
\]

が成り立つ。非IIDの場合は位置依存写像の合成となり、単一\(\Phi\)のN乗とは限らない。この式に、**決定論部を含むpartial PRで同じ係数がそのまま残るとは書かれていない**。

### 3.2 共分散写像の分解

`c=\cos\theta`、`s=\sin\theta`、`h=\sum_jp_jP_j`、`M=\sum_jp_jA_j=cI+is h`とする。`A_j=cI+isP_j`だから

\[
 \Phi_\theta(X)=c^2 X+ics(hX+Xh)-s^2\mathbb E_j[P_jXP_j],
\]

\[
 \boxed{\Phi_\theta(X)=MXM-s^2\mathcal C(X)},\qquad
 \mathcal C(X)=\mathbb E_j[(P_j-h)X(P_j-h)].
\]

は厳密。一般の乱択unitary\(A_j\)に対しても、\(\Delta A_j=A_j-\mathbb EA_j\)とすれば

\[
 \mathbb E[A_jXA_j]=M X M+\mathbb E[\Delta A_j\,X\,\Delta A_j]
\]

という単純な共分散分解が成立する。なお、\(\Phi\)は**`A_jXA_j^\dagger`ではない**から、\(\Phi\)自身をCPTPと呼ばない。\(\|\Phi(X)\|\le\|X\|\)という作用素ノルム収縮性は、各\(A_j\)がunitaryであることから成立する。

この写像が拡張系\(Z_c\otimes H\)に作用する通常のqDRIFT channelの**off-diagonal coherence block**に対応する、というClaudeの解釈も、アンシラのブロック演算として正しい。ただしchannel全体のdiamond boundをそのままcoherent signalの**鋭い減衰係数**と同一視しない。

### 3.3 項間・項内の分散分解

`h|k\rangle=e_k|k\rangle`で、`V_k=1-e_k^2`、

\[
 v_{km}=\mathbb E_j|\langle m|(P_j-h)|k\rangle|^2
\]

と定義すると、\(P_j^2=I\)から

\[
 \sum_m v_{km}=V_k,
 \qquad v_{kk}=\mathbb E_j\bigl(\langle k|P_j|k\rangle-e_k\bigr)^2,
\]

\[
 w_k=V_k-v_{kk}=\mathbb E_j\bigl[1-\langle k|P_j|k\rangle^2\bigr]\ge0.
\]

従って\(0\le v_{kk}\le V_k\)。\(v_{kk}\)は**どのPauli項を選ぶかに伴う期待値のばらつき**、\(w_k\)は**各Pauli項の状態上の測定分散の平均**を表す。

Claudeの分散分解は正しい。しかし\(v_{kk}\)という量自体は既知の状態依存ランダムHamiltonian解析と近く、**この恒等式が新定理だという意味ではない**。

### 3.4 Shotのfreshnessに関する必要な修正

Claude §6.3は「Hadamard shotごとに新しい列をdrawしないと、\(\Phi^N(I)\)という平均等式が成り立たない」と読める。しかしここには**平均の意味論の混同**がある。

乱数列\(\omega\)を一度sampleし、その同じ回路を\(K\) shot繰り返しても、**乱数列を選ぶ操作を含む無条件期待値**については

\[
 \mathbb E_{\omega,\mathrm{shots}}[\widehat z_K]
 =\mathbb E_\omega[z(\omega)]
 =\langle\psi|\Phi^N(I)|\psi\rangle
\]

が成立する。ここで\(z(\omega)=\langle\psi|B_0(\omega)^\dagger B_1(\omega)|\psi\rangle\)。一方、\(\omega\)を固定した後に増やしたshotsの平均は\(z(\omega)\)へ収束するため、単一回路の偏差は消えない。

実部等の実数推定量\(Y\)について、全分散公式から

\[
 \mathrm{Var}(\widehat Y_K)
 =\mathrm{Var}_{\omega}\bigl(\mathbb E[Y\mid\omega]\bigr)
 +\frac{1}{K}\,\mathbb E_\omega\bigl[\mathrm{Var}(Y\mid\omega)\bigr]
\]

となる（1回路をsampleしてK shot繰り返す条件付き独立測定）。**初項はKを増やしても残る**。独立な乱数列を各shotでdrawする場合とはshot複雑度・集中保証が違う。

したがって、**fresh IIDは簡単な独立shot会計の十分条件であって、\(\Phi^N(I)\)という無条件平均恒等式の必要条件ではない。** PR原論文 Appendix A.3にも、深い乱択回路について単一回路を複数shot再利用し得るという議論がある。ただしその集中理論をM1-Sの共有label構造に無条件に流用しない。これはClaudeの主要理論を否定しないが、**実験・shot契約を固定するときは修正が必要**である。

---

## 4. 長時間極限の数理検証

### 4.1 一次展開を単純にN倍できないという旧GPTの指摘

`\Phi_0(X)=MXM`、`\Phi=\Phi_0-s^2\mathcal C`とする。`\mathcal C`について一次で展開すると

\[
 \Phi^N=\Phi_0^N-s^2\sum_{n=0}^{N-1}\Phi_0^{N-1-n}\mathcal C\Phi_0^n
 +\text{higher-order terms}.
\]

固定の\(N\)に対するこれは代数的に正しい。しかし重要な資源極限では\(N\theta^2\sim D=O(1)\)であり、\(N\theta^2v_{kk}\)は定数の大きさとなる。**一次の有限N展開だけから `1−Nθ²v` を最後まで用いることはできない。** 以前のGPTの懸念はこの点に限って正しい。

### 4.2 Claudeの修正：生成子／超演算子の固有値を先に評価する

`h|a\rangle=e_a|a\rangle`、`\mu_a=\cos\theta+i\sin\theta\,e_a`とすると、\(\Phi_0\)の固有operator\(|a\rangle\langle b|\)の固有値は\(\mu_a\mu_b\)。対象の対角モード\(|k\rangle\langle k|\)の固有値\(\mu_k^2\)が**結合する他のモードから孤立**していれば、通常のスペクトル摂動論では

\[
 \lambda_k(\Phi)
 =\mu_k^2-\sin^2\theta\,v_{kk}
 +O\!\left(\frac{\sin^4\theta\,\|\mathcal C\|^2}{g_k}\right),
\]

が期待できる。\(g_k=\min_{(a,b)\ne(k,k)}|\mu_k^2-\mu_a\mu_b|\)のうち**実際に結合するモード**が重要。小角で\(g_k\sim\theta\Delta_{\Sigma,k}\)なら残差は概ね\(O(\theta^3/\Delta_{\Sigma,k})\)。これを\(N\theta^2\le D\)の範囲でN乗した差へ戻すと\(O(D\theta/\Delta_{\Sigma,k})\)になる、というClaudeの論理は**筋が通っている**。

ただし、これを正しい**一様定理**として主張するには次が必要。

1. 近傍固有値の分離と連続分岐の定義。特に\(\theta\to0\)ですべての\(\Phi_0\)固有値が1へ集まるため、定数を\(\theta\)に対して管理すること。
2. スペクトル射影の摂動と条件数。\(\Phi\)は一般に非正規で、収縮性だけでは**各モードのspectral weightの良好な条件数**は出ない。
3. pair-sum共鳴、すなわち\(2e_k=e_a+e_b\)に関係するモードが結合すると\(g_k\)の一次係数が消えること。
4. 次元・入力分布への依存。**固定有限次元でのsmall-angleの正しさは、分子サイズに一様な保証を意味しない**。
5. 有限\(N\)／有限\(\theta\)の修正、初期状態が複数modeを含む場合のsignal floor、およびbranch phase。

**独立判断：** 以前の「一次近似の反復だから危険」という留保に対して、Claudeは合理的な数学的解決方針を提示した。これは**Bへ進む情報価値を高める**。一方、**全次数評価が証明されたとは認定しない**。

### 4.3 先導減衰係数とイベント数比\(\rho_\infty\)

上記の条件下で、対象モードの先導減衰は

\[
 |\lambda_k|^N\simeq \exp[-N\theta^2(V_k+v_{kk})],
 \quad V_k=1-e_k^2.
\]

物理的な相対時間を揃えたordinary型qDRIFTでは、1イベントを相対角\(2\theta\)として使うので、先導減衰は\(\exp[-2N\theta^2V_k]\)となる。**同じ対象時間・同じ減衰予算を比較**し、\(V_k>0\)なら、必要イベント数の候補比は

\[
 \boxed{\rho_\infty=\frac{V_k+v_{kk}}{2V_k}}
 =\frac12+\frac{v_{kk}}{2V_k}
 =1-\frac{w_k}{2V_k}\in[1/2,1].
\]

この代数式は\(0\le v_{kk}\le V_k\)によって支持できる。**ただし区間は「先導漸近の\(\rho_\infty\)」に対するものであり、すべての有限時間の費用比\(\rho(T)\)が常にこの区間に入るという定理ではない。** Claude §11.1の有限\(T\)に対する区間表示には追加の単調性・非負性の証明が必要である。

さらに\(V_k=0\)では比は\(0/0\)になる場合があり、式を適用できない。すでに基準lawの減衰がないモードでは、「必要イベント数を半分にする」という比較自体を別に定義し直す必要がある。\(v_{kk}=0\)なら先導漸近の最良ケース1/2、\(w_k=0\)なら先導漸近の利益0となる。

**判断：** Claudeの\(\rho_\infty\)は、完全乱択に対する**適切な条件付きの研究指標**である。**「常に乱択イベント数が半減する」や「総資源が半減する」という主張ではない。**

### 4.4 二準位の記号的な厳密例

Claudeは\(P_j\in\{Z,X\}\)、\(p_j=1/2\)に対する\(h=(X+Z)/2\)を用い、\(h\)固有basisの対角成分について

\[
 \begin{pmatrix}x'\\y'\end{pmatrix}
 =\begin{pmatrix}\mu_+^2&-\sin^2\theta/2\\-\sin^2\theta/2&\mu_-^2\end{pmatrix}
 \begin{pmatrix}x\\y\end{pmatrix}
\]

を得ている。固有値は

\[
 \lambda_\pm=\cos^2\theta-\frac{\sin^2\theta}{2}
 \pm i\sin\theta\sqrt{2\cos^2\theta-\frac{\sin^2\theta}{4}}.
\]

この式は対角成分の閉じた漸化式と整合する。**根号が実数のsmall-angle範囲**では固有値は複素共役で、行列式が\(\cos^2\theta\)となるため

\[
 |\lambda_\pm|=|\cos\theta|
\]

が厳密に成立する。よって**非可換なPauliを含む少なくとも一つのモデルに、長時間減衰改善の実在する機構がある**というClaudeの見方を支持する。

**精度のための限定：** \(2\cos^2\theta-\sin^2\theta/4\ge0\)という条件を外して\(|\lambda_\pm|=|\cos\theta|\)を任意角で宣言しない。Claude本文§7.3はsmall-angleの議論だが、一部「厳密に」とだけ書かれた表現は領域を添えるべきである。

**この例の到達点：** `H_D=0`、非共鳴、二次元、\(v_{kk}=0\)の**最も有利な条件**での存在証拠。分子PR、他sector、複数round RPE、未知stateでの最適化、whole-circuit費用の証明ではない。

### 4.5 低周波汎関数とpair-sum共鳴

Claudeが示した\(\mathcal L_k\)は、\(v_{km}\)で重み付けしたエネルギー差の逆数を考える点で有益である。**全Fock空間の最小gapが小さいだけで、M1-Sが必ず失敗するわけではない**。近縮退への結合weightが重要である。

ただし\(\Phi_0\)の対象モードの共鳴は、\(e_m-e_k\)だけではなく

\[
 2e_k-(e_a+e_b)=0
\]

という**二つのエネルギーの和に関するpair-sum条件**も含む。そこへの結合\(\mathbb E[Q_{ak}Q_{kb}]\)は、単なる\(v_{km}\)の表とは別の情報である。従って、**鍵となる判定量を\(\rho_\infty\)という単一スカラーだけで完全に代表させることはできない。**

\(k\)が対象\(h\)の**大域的な非縮退最低固有値**なら、他の固有値の和で\(2e_k\)を実現する非自明な組はない。しかし対象が全\(H\)のsector内groundであって、\(h_R\)の大域的groundとは限らない部分乱択では、その自動的な保護は期待できない。さらにfinite-step aliasingは別条件である。

---

## 5. 部分乱択・多round RPEへの移送

### 5.1 固定stepの超演算子反復は自然だが、係数移送は未証明

各乱択stepについて正branch\(G_+(\omega)\)、負branch\(G_-(\omega)\)が**完全に定義**され、step間のlabel列がIIDであれば、

\[
 \Psi(X)=\mathbb E_\omega[G_-(\omega)^\dagger X G_+(\omega)]
\]

という写像を導入し、符号・順序の規約を揃えて**\(\Psi\)の反復として枝干渉を整理する方針**は妥当である。Claude§9.1の\(G^{rev}XG\)という表記は、\(G_-^\dagger=G^{rev}\)と定義したときに限る。`rev`が文字どおりのadjointか単なる逆順かを明示すべきである。

しかし\(H_D\ne0\)の場合、非摂動の固有状態は\(h_R\)と一致せず、outer PFも作用する。\(\Phi\)で導いた分散分解だけから\(\Psi\)の固有mode、減衰係数、pair-sum共鳴、RPE biasが自動的に出るわけではない。

Claudeの\(\rho_\infty^{(H)}\)は

\[
 \rho_\infty^{(H)}\stackrel{\mathrm{candidate}}=\frac12+
 \frac{v^{(H)}_{kk}}{2\bigl(1-\langle k|h_R^2|k\rangle\bigr)}
\]

と記述される。しかしこれは**全\(H\)の固有state**での候補量であり、今回の資料には、実際のpartial controlled protocolの\(\Psi\)の目的固有値から、この式を導く証明がない。特に、\(\delta\)・乱択事象数・決定論blockの交差項と、physical time\(T\)の対応を同時に固定する必要がある。**本式を既に認証された資源削減倍率として扱わない。**

### 5.2 RPEの測定対象は超演算子固有位相だけではない

PR型single-ancilla RPEは、\(z_m=\langle\psi|\Psi_m^{s_m}(I)|\psi\rangle\)を実際に測定する。\(\Psi_m\)の固有値の位相\(\arg\lambda_a\)と、固有射影から来る**複素weight**、近縮退モード、鏡像周波数は、最終信号では足し合わされる。

位相推定保証へ進むには、少なくとも以下を別々に制御する。

- ターゲットmode以外の総重み、複素weightとsignal floor。
- 相対時間\(T_m\)の倍化と、\(\theta_m,\delta_m,r_m\)がround間でどのように動くか。
- 位相のaliasing、branch alignment、X/Y測定の独立性、shot confidence。
- Outer PFの有効エネルギーbias、\(h_R\)の歪み、finite-bit/native controlled phase。
- RPEで必要な**RMS errorか高確率errorか**の明示、基底overlapの適切な下界。

**超演算子固有値の先導減衰が小さくなることと、エネルギー推定総費用が小さくなることは別の研究段階**である。

### 5.3 化学系への適用性

JW表現の個々のPauli項は、一般に粒子数や\(S_z\)を保存しないことがある。このため、全Hamiltonianの化学的groundを与えたとしても、乱択項は**他のFock sectorへ状態を移し得る**。sector内のgapだけで不十分というClaudeの指摘は支持できる。

一方、**スペクトルのどこに低周波重みが存在するか**が重要なので、「全Fock空間に小gapがある」という事実だけで一般的STOPは出ない。実際の分子で\(v_{km}\)・pair-sum coupling・alias距離がどの程度かは未測定。平衡構造で必ず有利という予測も証拠不足である。

### 5.4 代表系の数値Gate 0の前に解消すべき問題

ClaudeはH4/H6等で\(v^{(H)}\)、\(\rho^{(H)}_\infty\)、\(\mathcal L\)を取得し、閾値\(0.7\)／\(0.85\)で早期STOPを提案する。しかし以下を先に扱う方が研究判断として適切である。

1. **数式の対象の固定：** 実際にPauli-involution tailを使うのか、DF fragment／Gaussian事象なのか。後者へ単一Pauliモデルの式を直接移すことはできない。
2. **\(\rho^{(H)}\)の理論上の意味：** full H-stateで評価した候補量が、同じ\(H_D/H_R\)のpartial\(\Psi\)の減衰を支配する必要条件なのか、十分条件なのか、単なるheuristicなのかを識別する。
3. **同一情報条件：** exact-ground stateと本当のgap・重みを事前に知ることは、実際のQPEアルゴリズムでは通常利用できない。Gate 0でのFCI/eigsh等は、**oracle的な研究診断**として用いる場合に限り、predictorや新規アルゴリズムの入力へ混入させない。
4. **候補の事前固定：** D/R分割を結果を見て複数回変えるならdevelopment適合・selection biasとなる。分割候補、accuracy、baseline、STOP閾値、結果の扱いを実行前に固定する。
5. **\(\rho\)が小さくても残る条件：** 非共鳴、angle/alias、重み、RPE success、実装単価、乱択部費用割合。\(\rho\)単独で科学的GOは出ない。

**新しい数値Gate 0はまだ承認されていない。** P0/P0.5/G0-preまでの既存資料を使った数式・protocol契約の検証を先行するべきである。

---

## 6. Native資源費用の監査

### 6.1 三つの異なる改善因子

以下を混同してはならない。

1. **既知のqDRIFT対RTEの費用係数：** 富士通2026の式(B5)で`γ=1`対`γ=2`という**既知モデル上の差**。
2. **M1-Sのgate共有±に伴う追加のevents削減：** 完全乱択の先導\(\rho_\infty\)と、部分乱択への未証明の候補\(\rho_\infty^{(H)}\)。
3. **総計算資源：** controlled回路の単価、state preparation、readout、ancilla、shots、X/Y・round数、準備・online古典費用、bias保証を含む。

三つの差を自由に乗算して「2倍／4倍のend-to-end改善」とすることは認めない。Claudeもその誤用を避けており適切である。

### 6.2 `C_gate=1`の意味を限定する

富士通2026 Appendix Bは、`C_gate`を**Pauli rotationsをそのアーキテクチャのelementary gate数へ換算する因子**と定義し、STARではsingle-qubit Pauli rotationをnative elementary gateとみなすため`C_gate=1`とする。これはその**資源計算モデルの採用値**であって、M1-Sの全ての`Z_c\otimes P`とordinary controlled-Pが同じphysical/native単価であることを**直接測定・認証したものではない**。

PR原論文のHWP＋catalysisも等角・可換群・ancilla処理に条件がある。従って、**強いbaselineで方向制御の追加2倍が消える可能性は高いが、「消えることが厳密に実証済み」とまでは書かない。** 最終的な比較では同じcompilation opportunitiesを与える。

### 6.3 費用の違いは実装契約で確認する

Pauli\(P\)について、符号規約を揃えれば

\[
 \operatorname{diag}(I,e^{-i2\theta P})
 =\bigl(I_c\otimes e^{-i\theta P}\bigr)
 \exp(+i\theta Z_c\otimes P).
\]

この分解はordinary-controlled側で**uncontrolled Pauli rotationがもう一つ現れ得る**ことを示す。しかし全回路合成やHWPにより最終価格は変わり、`1 event=1 rotation`という抽象単位だけでは比較不能である。

**暫定判断：** 同一のFT/HWP会計では\(\rho\)によるevent数利益を主要候補とし、hardware-nativeの追加利益は別項目として未認定とする。D/R費用比率はP0.5の`NOT_ESTABLISHED`を維持する。

---

## 7. 先行研究・研究価値・論文化可能性

### 7.1 既知の部分と、新しい候補となる部分

| 論点 | 現状 |
|---|---|
| qDRIFT、`Z_c\otimes H`による方向制御 | 回路の外形は既知または自然な構成。独立新アルゴリズムの強いclaimにはしない |
| 対称PFの回文、mirror／echo、branch共有乱数 | 構成部品・近接方法は既知。G0-preの8文献には関連する具体例がある |
| `\Phi^N(I)`と共分散分解 | 厳密な恒等式。ただし標準的な期待値・代数操作が中心で、それ単独では新規性が弱い |
| 2準位の厳密可解例、\(v/w\)の分解 | **条件付きの性能機構の存在証拠**。一般・分子への性能保証ではない |
| 孤立スペクトルでのNに一様な信号漸近 | 有望な**未証明の定理候補**。既知のKato型摂動等を適用して得られる強さ・条件緩和を確認する必要あり |
| 非可換な部分乱択\(\Psi\)とRPE保証 | **主要な新規性・科学的難所の候補**。未証明であり方法差・仮定の実用性も未確立 |
| 化学的有用性 | \(\rho^{(H)}\)、低周波重み・pair-sum共鳴・D/R費用が未測定。肯定／否定どちらにも未確立 |

**新規性判断：** 「世界初の新アルゴリズム」ではなく、**既知の方向制御randomized evolutionに対するtask-specificな複素位相・減衰・RPE保証**の研究候補と位置付ける。その保証が単なる既知channel boundの弱い系より厳密に有用であり、かつ自然な条件で成り立つなら独立貢献となり得る。

PRの「trick不適用」を覆したこと自体を論文claimとしない。代わりに「厳密な対称性なしに、branch-correlated乱択がどの誤差予算で有用になるかを示す」と定式化することを推奨する。

### 7.2 本研究の価値を決める検証義務

高価値の着地点には以下が必要。

1. **正確な対象lawと情報アクセス**：same Hamiltonian／relative time／state／confidence／branch phaseを揃える。
2. **現実のpartial PRでの意味のある理論保証**：完全乱択の精密解析だけに留まらず、\(H_D\ne0\)において自然な条件を示す。
3. **RPE成功の証明**：signal floor・complex weights・near-degeneracy・phase unwrappingとbias budgetに整合する。
4. **quantum chemistryにおける有効な領域**：単なる「例がある」でなく、Pauli-tail・sector外遷移・pair-sum coupling・D/R内訳の受理範囲を示す。
5. **既知強baselineへの実費用優位**：qDRIFT/RTE・HWP/STAR・shotが公平な条件で比較される。

**独立論文化可能性は現時点では中以下の条件付き可能性**と判断する。理論補題のみ成立し、化学・RPEの改善が示せない場合には、**技術ノートや既知手法の条件整理として閉じる**ことを推奨する。

### 7.3 ClaudeのB/C再評価

Claudeは今回、C（再定義）からB（条件付き推奨）へ判断を進めた。これは資料が増えたから自動的に正しいのではなく、**研究上の証明対象が具体化され、固定超演算子からの導出経路と2準位での存在証拠が提示された**点に評価の根拠がある。

- **維持できるところ：** `B`は条件付きで合理的。解くべき問題が単なる方法の命名でなく、証明義務・STOP条件に分かれている。M1-Aは補助的理論として保留、M2は代替として保留するのが適切。
- **制限するところ：** 「大部分は固有値摂動の標準適用で閉じる」「化学系では後半roundで自然に成立」「\(\rho^{(H)}\)が安価で単一の鍵量」「STAR/HWPでは各event同価格」といった表現は、いずれも未証明／モデル依存。
- **最終的な扱い：** **B（限定された理論検証候補）を支持。ただし新研究の本採択・無条件Gate 0数値評価は支持しない。**

---

## 8. 次の科学的判断・実務計画

### 8.1 推奨分類と担当

**科学的推奨：`B-conditional — narrow theory-first feasibility`**。

- **M1-S：** 完全乱択の共分散superoperatorに関する理論課題として、**研究価値がある候補**に残す。部分乱択・RPE保証へ接続する見込みを低costに判定する小さな理論前段階を優先する。
- **M1-A：** 既知qDRIFT／FRとの対応を保持し、baseline側のphase/radius・bias評価の補助として用いる。**単独の独立新アルゴリズムGOは出さない**。
- **M2：** HWP・基底共有の条件付き代替として保持。M1-Sの理論・資源の両面でGOできなければ検討する。
- **Track B：** 他候補まで一般的にno-goとしない。研究資源を限定して失敗時に縮小・休止を検討する。

### 8.2 推奨する作業順（科学レビュー後・別途承認を前提）

| 段階 | 担当候補 | 作業 | 実行権限 |
|---|---|---|---|
| **S0：科学contract固定** | GPT（判断）＋Codex（技術整理） | `B0/B1` ordered protocol、時間・角度・step・Pauli-tail、same-information、主要baseline、RMS/高確率、受理基準を具体化。\(\Phi\)と部分\(\Psi\)を区別 | **研究契約の採択はユーザーの明示承認が必要** |
| **G0-T：限定的理論可行性** | Codex（既知数理・技術整理）、GPT/Claude（重要科学判断） | 完全乱択の固有mode摂動の一様性を明示し、部分\(\Psi\)について\(\rho^{(H)}\)の成立範囲または反例を検討。必要なスペクトル条件、評価項目を絞る | **新規理論研究・G0の開始は別途明示承認後** |
| **Gate 0：登録された数値診断** | Codex | G0-Tで正当化した鍵量のみ、Pauli-tailか明示した一般化モデルで、登録された分子・分割を評価。exact groundはoracle診断と明示。benchmark条件・計算予算・停止閾値を事前固定 | **科学計算の独立承認が必要** |
| **G0本段階：RPE保証** | GPT中心、Codex補助 | 部分\(\Psi\)のsignal、gap/alias、complex weights、shot・RMS、定数の明示とbaseline対応 | 重要科学レビュー／研究実行の範囲を別途承認 |
| **V1：実装費用評価** | Codex | 同一task・同精度・同情報のcontrolled native会計と必要ならnative synthesis／simulation。先行baselineの最適化機会を一致 | **別途科学実行承認** |

**Gate 0を理論より先に無条件で走らせることは推奨しない。** 数値診断が検証したい科学claimと結び付いていなければ、新たなデータを作っても研究判断が変わらない可能性がある。

### 8.3 GO／縮小／STOP条件（数値閾値は未採択）

| 条件 | 推奨判断 |
|---|---|
| 理論G0-Tで\(\rho_\infty\)の適用範囲・一様誤差が自然な条件下で閉じる | 事前登録した数値Gate 0が必要か評価する。**自動GOではない** |
| 部分\(\Psi\)で物理時間・phase・PF biasを整合できない | M1-Sを完全乱択の限定理論ノートへ縮小、またはSTOP |
| \(\rho^{(H)}\)が未定義／非正当化のまま | それを使うGate 0のGO/STOP判定を実施しない |
| 共鳴・低周波weightが強く、signal floorやRPE保証が自然な条件で維持できない | M1-SのPR/RPE資源claimを縮小・終了候補とする |
| \(\rho^{(H)}\)の効果が小さい、または同条件のcompiled/shot費用で消える | アルゴリズム的な資源改善claimを採択しない。理論的結果のみの価値を再評価 |
| 既知の同等のcoherent-signal保証が見つかる | 独立新規性claimを縮小。方法の再利用や技術ノートとして扱う |
| 主要な科学的障害が残る場合 | M2等へ方向転換、またはTrack B新規探索の休止を検討。全候補の一般no-goは宣言しない |

**Claudeの `\rho\le0.7`／`\rho\ge0.85`、end-to-end 1割といった値は今回採択しない。** 計算を行う場合には、研究上の価値・不確かさ・費用model・境界例を反映した閾値を別途決定する。

### 8.4 本レビューの科学判断として確定する範囲

- **支持：** `Z_c\otimes H`としての自然なqDRIFT解釈、\(\Phi\)／\(\mathcal C\)の恒等式、分散分解、限定された2準位例。
- **条件付き支持：** 孤立スペクトルに対する固有値perturbationを先に解き、N乗した後の有限残差を管理する理論方針、\(\rho_\infty\)の完全乱択での先導漸近。
- **未認定：** 分子部分乱択の\(\rho^{(H)}\)、化学での実用的な大きさ、多round RPEへの厳密保証、実native/shotを含む総費用改善、M1-Sの独立新規性の最終証明。
- **権限：** **研究案Bは推奨に留まり、M1-S採択／G0-T／Gate 0／G0本段階／V1はすべて未承認。mandatory STOPを維持。**

---

## 9. 参考資料・固定出典

### 9.1 ユーザー添付

- Claude（2026-10-15）、`レビューinner.md`：本レビューの主対象。§0、§5–§9、§11、§15–§17、付録A–E。Claudeの初回中断・再開の経緯は付録Eに記録されている。

### 9.2 GitHub（すべて固定commit）

- **[G0-PRE]** [M1-S G0-pre Decision Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md)
- **[G0-INDEX]** [同 evidence index](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/track_b_m1s_g0_pre/2026-10-15/v1/evidence_index.json)
- **[G0-LIT]** [同 primary literature locator](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/track_b_m1s_g0_pre/2026-10-15/v1/primary_literature_locator.csv)
- **[GPT-1]** [直前GPT独立科学レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03/docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md)
- **[P05]** [M1 P0.5技術報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md)
- **[P0]** [M1 P0 Decision Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md)

### 9.3 関連一次論文

- J. Günther et al., [*Phase estimation with partially randomized time evolution*](https://arxiv.org/abs/2503.05647v2), PRX Quantum **7**, 020332 (2026). 特にAppendix A.4、A.3、B、E。
- S. Kanasugi et al., [*Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*](https://arxiv.org/abs/2603.22778v2), Appendix B 式(B3)–(B5)；`γ`／`C_gate`。
- E. Campbell, [*A random compiler for fast Hamiltonian simulation*](https://arxiv.org/abs/1811.08017), Appendix D（controlled qDRIFT）。
- A. Childs, A. Ostrander, Y. Su, [*Faster quantum simulation by randomization*](https://arxiv.org/abs/1805.08385), 回文型のrandomized symmetric PFとの対照。
- I. D. Kivlichan, C. Granade, N. Wiebe, [*Phase estimation with randomized Hamiltonians*](https://arxiv.org/abs/1907.10070), gapとstate-dependentな乱択phase estimation。
- P. Mohammadipour, X. Li, [*MLMC-qDRIFT*](https://arxiv.org/abs/2604.26865v2), index coupling／coherent branchとの対照。
- J. Peetz, S. Smart, P. Narang, [*Quantum Simulation via Stochastic Combination of Unitaries*](https://arxiv.org/abs/2407.21095v2), stochastic cross termとの対照。
- C.-F. Chen, H.-Y. Huang, R. Kueng, J. Tropp, [*Concentration for random product formulas*](https://arxiv.org/abs/2008.11751v4), 単一乱択回路再利用と平均の意味論を区別する参照。

### 9.4 今回独立に検討した論点と未検証事項

- **紙上確認した数式：** gate共有干渉word、\(\Phi\)・共分散恒等式、\(V=v+w\)、\(\rho_\infty\)の条件付き代数、二準位例の固有値式と成立角領域、shotの無条件期待値／全分散の区別。
- **導出の妥当な道筋を確認したが定理として未完成：** \(N\theta^2=O(1)\)での固有値・固有射影の一様摂動、pair-sum共鳴の制御。
- **未検証：** 部分\(\Psi\)の先導減衰係数、化学系での\(\rho^{(H)}\)／低周波重み、対称sector外のスペクトル、native単価、RPEのRMS／高確率保証。これらを今回確立したとは主張しない。

---

## 10. 最終分類・STOP

**最終分類：`B（条件付き推奨）— 理論契約先行・科学実行は未承認`。**  
**意味：** 新規アルゴリズムとしての即採択ではない。限定したcoherent-signal理論検証を行う科学的な情報価値があるという判断。  
**次に必要な意思決定：** ユーザーによるS0／G0-Tの研究契約・理論実施の承認。Gate 0の数値計算は独立の実行承認が必要。  
**保持：** Parent allocation D＋E、M1-Aのtechnical-note位置付け、M2の代替候補、既存分類、one-shot・retry制約、固定結果、**mandatory STOP**。  
**今回許可していないもの：** M1-Sの正式研究採択、G0-Tの新理論実施、分子Gate 0数値取得、native回路実行、資源予測、GitHubの研究状態変更。
