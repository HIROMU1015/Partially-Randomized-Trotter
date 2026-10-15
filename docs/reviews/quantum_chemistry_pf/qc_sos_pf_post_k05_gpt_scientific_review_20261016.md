# SOS-PF K0・K0.5後：Claude科学レビューのGPT独立再評価と研究方針

**対象分野**：量子化学Hamiltonian simulation／product formula（PF）／基底エネルギー推定  
**作成日**：2026年10月16日（JST）  
**文書区分**：ユーザーによる明示的開始承認に基づく**重要科学レビュー・Level 1（高）**  
**レビュー状態**：**COMPLETED**。研究方針についての推奨を示す文書。**ユーザーによる正式採択・新規科学計算・K1実施・repository変更の承認ではない。**

---

## 0. 独立レビューの結論

### 0.1 結論：現在のA1を区切ることを推奨する。ただしA2の評価を単純化しない

**【GPT独立科学判断】** Claudeの最優先案 **A3――SOS-PFを「独立した新しいPF-native分割生成アルゴリズム」候補として一区切りとし、K0・K0.5の結果を保存する――を支持する。確度：中～高。**

これはSOS-PF一般の不可能性、SOSに基づくground-energy誤差保証の数学的無価値、あるいは量子化学Hamiltonian simulation研究全体の終了を意味しない。今回のA1構成について**独立した生成規則・有利な費用機構が特定されていない**という研究資源配分上の判断である。

一方、Claudeによる以下の一般化には**重要な留保・修正**を付ける。

1. **「実際の量子資源を変えるレバーは実biasとnative step単価の二つだけ」は一般には成り立たない。** PF固有branchへのoverlap・反復回数、制御／合成／情報取得費用、保証に従って選択できるstep数も資源を変える。外部から任意の精度で真の最小step数を知る *oracle型の事後比較* に限定すると、Claudeの二つへの帰着は近似的に理解できるが、実際の保証付きアルゴリズムの価格とは別である。
2. **A2（誤差保証・certified step選択）には実際のアルゴリズム運用上の意義があり得る。** 同じ回路の真のbiasを変えなくても、同品質で保証されたstep幅が大きくなれば、実際に実行する制御付きPF総数は減る。「同じ回路だから実費用を減らさない」は保証付き利用設定では不適切。とはいえ、現段階で**既知の強い保証を実用的に超える数値定数・安価な証拠・科学的新規性は未確立**であり、A2を主研究テーマに据える根拠も不足している。
3. **gap不要の状態依存誤差保証は、Claudeの§3.8で提案された形より粗いものなら有限stepで明示的に閉じる。** 本レビュー§3.6で非負fragmentと二次PFの演算子Taylor係数だけから、
   \[
   \|(S_2(h)-e^{-ihH'})|0\rangle\|\le |h|^3\Lambda^{5/2}\sqrt{E_g}\,e^{|h|\Lambda}
   \tag{R1}
   \]
   を**紙上で導出**する。励起gapの下界を必要とせず、実数の任意の \(h\) に対して成り立つ**保守的な十分上界**である。証明を§3.6に明記する。これは新規性や実用性を認定する定理ではなく、既知の解析的手法から得られる粗い比較点である。特に \(e^{|h|\Lambda}\) が大きく、有用な分子でのstep削減は示さない。
4. Claudeの「SOS gapの利益は励起gapの有無によって平方根則／四分の一乗則に分かれる」という説明は**同種の粗い上界・情報条件・成功確率を固定した場合の目安**であり、普遍的な費用倍率ではない。K0の漸近係数boundと§3.8の出力分布boundは**保証対象が異なる**。
5. **ClaudeのK1-Fにある試験的成功・STOP基準は、総費用最小化の主目的と厳密には整合していない。** たとえばS2の同じ費用モデルで実bias係数比が \(1/2\)、step単価比が \(1.5\) なら、漸近的な合計費用比は \(1.5\sqrt{1/2}>1\) である。「係数半減かつ単価1.5倍以内」は総費用改善の十分条件ではない。また \(G_{\rm PF-native}\ge G_{\rm DF}\) だけでSTOPすると、\(K_{\rm SOS}\)、実bias、step単価の改善を見落とす。**いずれも未認可草案であり、そのまま実行条件にしてはならない。**

### 0.2 推奨と正式状態の区別

| 論点 | GPTの推奨 | 確度 | 既存状態への影響 |
|---|---|---|---|
| **A1：SOS-PF新規生成アルゴリズム** | **現構成への追加投資を区切る（Claude A3を支持）** | 中～高 | **まだ正式CLOSEDにしない**。採用はユーザーが判断 |
| **A2：有限step保証の理論** | 主テーマへの自動転換は**非推奨**。有効な研究問題として別に保持 | 中 | 研究テーマ変更の採択なし |
| **K1-F：固定 \(R_0\) 反証pilot** | **現時点では実施を推奨しない**。例外的な情報目的が明確なときのみ検討 | 中～高 | K1・SDP・数値は未承認 |
| Kang–Su関連 | 「SOS×PF」の広いアイデアを新規性根拠にしない。アクセス・保証の差は限定して評価 | 高（広い重複）／中（完全包含関係） | 先行研究の追加把握のみ |
| Claudeの新たな低energy scaling | 係数比・成功確率・finite-stepを分離し、**未証明の部分を定理化しない** | 高 | 未検証事項として保持 |
| 次の研究方向 | 研究目的・合法な情報モデルと比較方法を明示したうえで、**高い費用レバーから研究課題を再選定** | 中 | 新候補は未採択 |
| 既存の第1・第2研究、Track A/B、N1～N3等 | **変更しない**。別の成果・価値と本A1判定を混同しない | 高 | 既存状態維持 |

**この文書の結論は、Claudeの見解を追認するためのものではない。** 以下で独立に調べた事実、再検討した数学、結論を修正した箇所を分離する。

---

## 1. レビューの対象、資料取得、実行権限

### 1.1 直接の対象

- **Claude完成レビュー原本**：ユーザー添付 `レビューoutter(1).md`。2026-10-16 JST付、SOS-PF K0・K0.5後の独立科学評価。今回の直接の批判的レビュー対象。
- **K0（固定commit）**：`HIROMU1015/Partially-Randomized-Trotter`、`58d8672702284d73321cc67a26c4640adaddde5f`。
- **K0.5（固定commit）**：同repository、`01bdc56a2970dcb749390b0540846143f775d6f9`。
- **研究状態の採用記録（固定commit）**：`78b50c515d9df9556f261e74ca7c57af461b7199`。

直接取得して確認した中核GitHub資料：

| 区分 | 資料 | 用途・今回の確認の範囲 |
|---|---|---|
| K0 | [技術監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md) | 二分割・多fragment・Schur＋Cauchy・PSD化・費用式・STOP条件。全文の主要数式と結論を確認 |
| K0 | [method status](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/method_status.json) | `UNDETERMINED_DATA_REQUIRED`、未認可境界を確認 |
| K0 | [GPT向け引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/gpt_handoff.md) | 新規性・保証・費用・K1未認可の要約を確認 |
| K0 | [K1草案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/k1_design_draft.md) | 未固定 \(R\)、二軸比較、実行契約未完成を確認 |
| K0.5 | [追加技術調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md) | Kang–Su、gap-free保証、Childs上界、有限剰余、生成規則の欠落。全文確認 |
| K0.5 | [追加文献対応表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/literature_correspondence.md) | 検索・取得した版、ACM出版版未取得、比較上の留保を確認 |
| K0.5 | [GPT向け判断資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/gpt_decision_handoff.md) | B1～B6、研究採否判断と実行境界を確認 |
| K0.5 | [README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/README.md)・[provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/provenance.json) | 公開scope・元K0への参照・固定証拠の存在を確認 |
| 研究状態 | [正式decision](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/artifacts/qc_pf_exploration_state_adoption/2026-10-15/decision.json) | 第一候補等の既存状態に対する保護境界 |
| 研究俯瞰 | [C1～C4・Track A/Bの統合引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/qc_research_integrated_gpt_decision_handoff_v1_20261014.md) | 既存研究の到達点と費用指標が異なることを参照 |
| 過去GPTレビュー | [SOS-PF K0開始前の独立レビュー原文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/supplied_gpt_review.md) | 何が既存の科学判断で、K0・K0.5後に何が変わったかを照合 |

K0・K0.5のremoteで公開された文書は必要範囲で確認したが、**全repositoryのsource・全保存数値・全Git保護対象を本レビューで再監査したという意味ではない**。出版版原論文の完全取得や全引⽤文献の網羅的初出性調査も行っていない。添付Claude文書そのものについては、今回のユーザー提出物を直接読んだため、K0側での「Claude原本未取得」という履歴とは矛盾しない。

### 1.2 一次文献・確認した部分

本レビューで独立に確認した一次文献は次の通り（すべて完全な全証明の再現ではない）。

- [Kang–Su, *Quantum Matrix Arithmetics with Hamiltonian Evolution*, arXiv:2510.06316](https://arxiv.org/abs/2510.06316)：要旨と公開論文本体のうちSOS Hamiltonian simulation、Hamiltonian squaring、化学DFTHC／sector norm、K0.5で示された定理番号・アクセスモデルの対応。ACM [DOI:10.1145/3816436](https://doi.org/10.1145/3816436)は書誌・検索経由の本文抜粋を参照したが、**出版版本文の全式・全付録とarXiv v3との差分は未照合**。
- [Hejazi et al., *Better product formulas for quantum phase estimation*, arXiv:2412.16811v1](https://arxiv.org/html/2412.16811v1)：§I（固有値摂動・Magnus）、§II、§IV（low-energy PF）。
- [Hejazi et al., *Better bounds for low-energy product formulas*, arXiv:2402.10362v1](https://arxiv.org/html/2402.10362v1)：低エネルギー射影・漏洩の仮定と主張。
- [Mizuta–Kuwahara, *Trotterization Is Substantially Efficient for Low-Energy States*, arXiv:2504.20746](https://arxiv.org/html/2504.20746)：本文・補足§S2（Theorem S1）、局所性・PSD・energy依存の保証条件。
- [Yi–Crosson, *Spectral analysis of product formulas for quantum simulation*, npj QI (2022)](https://www.nature.com/articles/s41534-022-00548-w)：QPEでの固有位相bias・固有vector overlap・gap仮定。
- [Childs et al., *Theory of Trotter Error with Commutator Scaling*, PRX 11, 011020 (2021)](https://arxiv.org/abs/1912.08854)：S2の交換子に基づく有限時間誤差上界を、K0.5が明示した式と比較。
- [Low et al., *Fast Quantum Simulation of Electronic Structure by Spectral Amplification*, PRX 15 (2025)](https://arxiv.org/abs/2502.15882)、[King et al., *Quantum Simulation with SOS Spectral Amplification*](https://arxiv.org/abs/2505.01528)、[Rubin–Low–DePrince, *Near-Frustration-Free Electronic Structure Hamiltonian Representations and Lower-Bound Certificates*](https://arxiv.org/abs/2602.05069)：化学SOS、weighted SOS/v2RDM双対、spectral amplificationの既知成果。今回確認範囲は公表要旨・一部本文、K0文献台帳の明記する部分。
- [Casares et al., *Theory and Practice of Trotter Product Formulas for Quantum Chemistry*, arXiv:2606.30741](https://arxiv.org/abs/2606.30741)：SPRINT・GRADEとnative価格の強い比較候補（要旨とK0台帳）。
- [Bay-Smidt et al., *Quantum simulation of nanographenes and Trotter error cancellation*, arXiv:2605.00745v2](https://arxiv.org/abs/2605.00745)：差エネルギーという別taskでの誤差相殺（要旨確認）。
- [Dutkiewicz et al., *Spectral amplification for ground-state energy estimation of electronic structure in first quantization*, arXiv:2607.15358](https://arxiv.org/abs/2607.15358)：第一量子化へのSOSSA応用、本文ではなく要旨確認。
- [Somma–de Wolf, *Optimal Lower Bound for Ground-State Energy Estimation with a Guiding State*, arXiv:2608.24493](https://arxiv.org/abs/2608.24493)：guiding state・位相推定の問い合わせ下界、要旨確認。**特定のPF新規性の否定定理としては用いない。**

### 1.3 計算・公開・正式採択の権限

本レビューで実施したのは固定資料の閲覧、先行研究の確認、**紙上の数理的推論**、研究方針の比較、詳細Markdownの作成だけである。分子Hamiltonian生成、SDP求解、行列の固有値計算、シミュレーション、保存科学数値の再集計、native compile、T合成、K1実行は**すべて実施していない**。K0の `UNDETERMINED_DATA_REQUIRED`・旧 `mandatory STOP`・`one-shot`・`retry=0`・科学SOURCE/結果/認可・既知manifest FAILは維持する。

---

## 2. 研究全体から見た本件の位置付け

### 2.1 研究目的の再確認

本研究が求めているのは、元の有限基底電子Hamiltonian \(H\) の**絶対ground energy**を、固定された精度 \(\varepsilon\)、成功率 \(1-\alpha\)、合法な入力・状態アクセスで**より低費用**に得ることである。量子ground stateへのアクセスを仮定することと、その古典的な全振幅・RDM・正確な \(E_0\)・励起gapを無料の設計入力として使うことは異なる。

本来の学術的評価軸を、少なくとも次の三つに分ける。

1. **アルゴリズム的構成差**：入力（Hamiltonian係数、sector、許す前処理・アクセス）から実際の回路／推定器を生成するルールが、既知の強い方法に対して何を新たに実現するか。
2. **科学的保証差**：返すground-energy推定値のbias、成功率、branch／aliasをどの情報とコストで保証するか。
3. **総費用差**：実行するcontrolled-step、T／Toffoli、論理qubit、準備回数・測定、古典証明書の費用などのうち、対象として定義した資源で何が減るか。

一つだけ満たす研究にも価値はあるが、**それを「新しい低費用アルゴリズムが完成した」と読み替えてはいけない**。

### 2.2 これまでの研究系列の教訓

研究状態の固定資料によると、第一候補（一体項再配分PF）の現固定構成はCLOSED、H-E1の現定義探索は終了、M-a/M-bは追加探索STOP、PF小改変系列はPAUSED、旧D案はHOLDである。Track A/B、第1・第2研究、N1～N3、THRIFT、C1～C4等は**それぞれの独立状態を維持**している。

横断的な教訓は、(i) 漸近的な誤差係数または代理量の改善だけでは総費用に転化しにくい、(ii) 強い補正済みPF・前処理・資源会計との同情報比較が必要、(iii) 既存のDF/一体移動/分割/回路最適化と新規性が重なりやすい、(iv) 実際の設計規則がないままpilotから逆向きに生成則を探すと研究を惰性的に拡大しやすい、というものである。

ただし**Track A/Bの限定的な正の局所結果や、第1・第2研究の既存価値をSOS-PFの否定と一緒に消去してはならない。** 全QPE/RPE資源に対する未完成部分が残ることは、その研究の別の科学的課題である。SOS-PFの結論だけから、独立した研究間の資源配分の厳密な順位を捏造しない。

### 2.3 K0・K0.5が確立した境界

**【固定資料で確認したこと】**

- PSD fragmentの分散の総和は \(E_g=E_0-E_{\rm SOS}\) で抑えられ、二分割・多fragmentのS2主要bias係数への具体的な上界を作れる。
- Hermitian一体演算子の平方は軌道回転と対角位相で指数化できる。
- 複素Gramの実化により実fragmentのrankは増え得る。残余一体項のPSD化はSOS下界を悪化させ得る。
- **DF fragmentに定数shiftを加えただけなら、既知scalar位相の復元後の実PF回路・固有位相bias・overlapは同一**である。
- 有限step保証としてglobal normのCauchy＋Schur式はあるが、化学で競争力のあるstep幅・certificate価格を示していない。
- K0.5はgap不要の既存operator／QPE出力安定性と、個別branchのgap依存を区別した。
- **独立した有利な分割生成規則 \(R\)、分子での実bias改善、native総費用改善は未確認**である。
- K0正式結果は `UNDETERMINED_DATA_REQUIRED`、 `K0_PROTOTYPE_CANDIDATE=false`。一般的な不可能性の定理は示されていない。

よって、次に必要な問いは「何をもう一回計算するか」ではなく、**どの科学的主張に対して研究資源を投入する価値があるか**である。

---

## 3. Claudeの数学・保証・新しいスケーリングの独立評価

### 3.1 PSD分散補題と二分割係数：支持

対象sector上で \(H'=H-E_{\mathrm{SOS}}I=\sum_jF_j\)、\(F_j\succeq0\)、\(f_j\ge\|F_j\|\)、\(E_g=E_0-E_{\mathrm{SOS}}\ge0\) と置く。厳密ground \(|0\rangle\) について

\[
\phi_j=\langle0|F_j|0\rangle,\quad
\sum_j\phi_j=E_g,\quad
\sigma_j^2=\langle F_j^2\rangle-\phi_j^2\le f_j\phi_j.
\]

これは \(F_j^2\preceq f_jF_j\) の標準的帰結である。\(E_g=0\) なら各 \(F_j|0\rangle=0\) なので、任意の順序のPFはgroundに対して厳密である。ただし**この端点での厳密性自体は既知・初等的**。

二分割の対称S2について、

\[
S_{ABA}(h)=e^{-ihA/2}e^{-ihB}e^{-ihA/2},\quad
\log S_{ABA}(h)=-ih[H'+h^2D+O(h^4)],
\]
\[
D=\frac1{24}[A,[A,B]]-\frac1{12}[B,[B,A]].
\]

基底状態の固有方程式から \((B-\langle B\rangle)|0\rangle=-(A-\langle A\rangle)|0\rangle\) となるため、

\[
\boxed{d=\langle0|D|0\rangle
=\frac1{12}\langle0|A(H'-E_gI)A|0\rangle\ge0.}
\]

これはK0・Claudeの簡約と一致する。\(d\ge0\) はPSD**分割**を必要とせず、\(H'-E_gI\succeq0\) と非縮退ground branchを用いる。PSDは \(d\) を \(E_g\) に結びつけるために必要。

**限定**：これは非縮退のground eigenvalueの**漸近的な \(h^2\) 係数**であって、任意の \(h\) におけるbiasの符号・値ではない。一般の多fragmentに同符号性を移すこともできない。縮退なら \(P_0DP_0\) の縮退摂動を扱う。

### 3.2 多fragmentの辞書・行和上界：支持。一様上界も紙上で確認

K0は指定順序のS2について

\[
D=\sum_{j=1}^{M-1}\left[\frac1{24}[F_j,[F_j,R_j]]-\frac1{12}[R_j,[R_j,F_j]]\right],
\quad R_j=\sum_{k>j}F_k
\]

と展開し、PSD分散とCauchy–Schwarzによって

\[
|\langle D\rangle_0|\le K_{\rm SOS}E_g,
\qquad K_{\rm SOS}\text{：非負対称行列の最大行和で計算できる上界}
\]

を得た。K0のraw係数辞書と行和評価の構造は整合する。

**【今回の独立した紙上照合】** K0 Eq.(10)の \(f_j\le f\) に対する一様上界

\[
K_{\rm SOS}\le\frac{M^2-1}{8}f^2
\]

について、Claudeが「後半の行和式は未検算」と明記した部分も、K0が定義した**相殺前のraw係数辞書を前提に**以下で照合できる。先頭 \(1/24\) の \([F_j,[F_j,F_k]]\) の行 \(m\) への寄与は

\[
\frac{f^2}{24}\bigl[3(M-m)+(m-1)\bigr]
\le\frac{M-1}{8}f^2.
\]

後半 \(-1/12\) の \([F_k,[F_l,F_j]]\) の行 \(m\) への寄与は

\[
\frac{f^2}{12}\left[(M-m)^2+3\sum_{j<m}(M-j)\right]
\le\frac{M(M-1)}8f^2.
\]

括弧内は \(m\) を1増やすと \((M-m)+1\) 増え、最大は \(m=M\) である。二つの安全な上界を足してEq.(10)となる。**これは固定されたraw辞書に対する代数的照合**であり、係数相殺後の最適 \(K\) や化学分子での実際の値を求めたものではない。

### 3.3 Claudeの \(\frac98 E_g/\Lambda\) 比：数式の適用範囲を限定する

同一の一様norm \(f_j=f\)、\(\Lambda=Mf\) を仮定し、弱い状態非依存の三重交換子上界

\[
C_{\rm raw}=4\sum|\kappa_{abc}|f^3
=\frac{M(M-1)(4M+1)}{36}f^3
\]

を比較対象に選べば、

\[
\frac{K_{\rm SOS}E_g}{C_{\rm raw}}
\le\frac{9(M+1)}{2M(4M+1)}\frac{E_g}{f}
\sim\frac98\frac{E_g}{\Lambda}\quad(M\to\infty)
\]

は代数的に整合する。ただし次の4点が重要である。

1. **弱い上界と別の上界の比**であり、実際の基底固有位相bias、認証済みstep数、量子ゲート数の比ではない。
2. 一様上界 \(f_j\le f\) だけでは一般に \(\Lambda=Mf\) としてよいわけではない。均一norm族を仮定する目安である。
3. 公平な対照には、同じsector・同じfragmentを用いた実際の交換子norm、[Childs et al.](https://arxiv.org/abs/1912.08854)の有限時間上界、適用可能なlow-energy bound、古典的な上界改善を含める。\(C_{\rm raw}\) のみと比較して「大幅な理論的speedup」を確定できない。
4. 単なるSOS lower bound \(L\) ではなく、**実際に回路へ採用するPSD分割**に対応する厳密 \(L\) と認証 \(U\) による \(G=U-L\) が必要である。

したがってClaudeの「spectral amplificationと平行する」は**発想を説明する類比**として妥当だが、SOSSAの実際のquery削減と同等の資源改善がPFでも成立したという意味ではない。

### 3.4 励起gapが必要な問題と、必要でない問題

ここでは二つのgapを混同しない。

- **SOS offset**：\(E_g=E_0-E_{\rm SOS}\)。基底状態がPSD和をどの程度frustration-freeに近いかの指標。
- **励起gap**：\(g_0=E_1-E_0\)。単一のPF固有branchの連続追跡、perturbative resolvent、個別overlap保証に関わる。

**認証済みの正の \(g\le g_0\)** がなくても、operator-normでの有限時間PF誤差、QPE回路のtelescoping出力安定性、次節のspectral-weightを使う確率保証は成立し得る。ただしoperator-norm保証は通常、ground-energy特化の資源優位を十分に表せない。

一方、個別のground固有位相 \(E(h)\) とその固有vectorの連続追跡にSchur補元を使うK0の式は、認証済み励起gapなどを必要とする。\(E_g\) が小さいだけで \(g_0>0\) の下界を安価に取得できるわけではない。

Claudeがこの二者を区別した点は重要で、支持する。特定branchに固執しない**出力の確率的成功保証**は、今後の保証理論を定義するときの有力な意味論となる。

### 3.5 Spectral-weight補題：正しいが、状態残差の取得が肝心

真の \(H'\) の基底状態 \(|0\rangle\) に \(U=S_2(h)\) を作用させる。

\[
\eta(h)=\|(U-e^{-ihE_g})|0\rangle\|.
\]

\(U\) の固有位相を \(\theta_j\)、\(|0\rangle\) の各PF固有vectorへの重みを \(p_j\) とすると、

\[
\eta(h)^2=4\sum_j p_j\sin^2\!\left(\frac{\theta_j+hE_g}{2}\right).
\]

したがって対象位相から**円周距離** \(\omega\in(0,\pi]\) 以上離れた重みは

\[
\boxed{P_{\rm far}\le\frac{\eta(h)^2}{4\sin^2(\omega/2)}.}
\tag{S}
\]

これは固有branchの孤立を仮定しない、正しいスペクトル定理の直接帰結である。一方、有限精度QPEの読出し失敗率、エネルギー窓によるalias排除、制御回路の合成誤差は**別に追加**する必要がある。\(\omega=|h|\varepsilon_b\) と置くには、少なくとも \(|h|\varepsilon_b\le\pi\) と適切な円周上のenergy復号規約が要る。

\(\eta(h)\) に**真のground vectorそのものを無料で古典読出しして評価した値**を入れるのでは研究上のcertificateにならない。入力係数・PSD証拠・norm・認証upper/lowerから導出できる上界が必要である。

### 3.6 【追加の紙上導出】粗いが閉じたgap不要・有限stepのSOS状態誤差上界

Claude §3.8(b)は \(\eta(h)\le h^3\Gamma_3+h^4\Gamma_4+\cdots\)、\(\Gamma_3\propto\sqrt{E_g}\) を有望な方向として述べるが、**剰余定数は導出していない**。一方、少なくとも次の保守的な上界は、複素対数・励起gap・任意の有限半径を使わずに紙上で示せる。

**前提**：有限次元sector、\(F_j\succeq0\)、\(f_j=\|F_j\|\) または正の認証上界、\(H'=\sum_jF_j\)、\(H'|0\rangle=E_g|0\rangle\) とする。二次Strang PFを、非負係数 \(c_\ell\in\{1/2,1\}\) による積

\[
S_2(h)=\prod_{\ell=1}^{q}\exp(-ihc_\ell F_{j_\ell}),\qquad
\sum_{\ell:j_\ell=j}c_\ell=1
\]

と表す（順序は固定）。\(\Lambda:=\sum_\ell c_\ell f_{j_\ell}=\sum_j f_j\)、\(\phi_j=\langle F_j\rangle_0\) と置く。\(\Lambda=0\) は両unitaryがground上で一致する自明な場合。

**ステップ1（係数別のベクトルnorm）**：\(S_2(z)|0\rangle\) を \(z\) のTaylor級数に展開する。次数 \(n\ge1\) の各非零積の**右端で最初に現れる**非零のfragment冪を \(F_j^k\) とすると、\(k\ge1\) で

\[
\|F_j^k|0\rangle\|
\le f_j^{k-1}\|F_j|0\rangle\|
\le f_j^{k-1}\sqrt{f_j\phi_j}.
\]

その右端を残して他の冪をoperator normで評価し、積のTaylor係数を非負の生成関数で上から抑える。次数 \(n\) の係数ベクトルを \(u_n\) とすると、

\[
\|u_n\|
\le\frac{\Lambda^{n-1}}{(n-1)!}
\sum_\ell c_\ell\sqrt{f_{j_\ell}\phi_{j_\ell}}
\le\frac{\Lambda^{n-1}}{(n-1)!}\sqrt{\Lambda E_g}.
\tag{T}
\]

最後はCauchy–Schwarzと \(\sum_\ell c_\ell\phi_{j_\ell}=E_g\) による。右端の選択を各Taylor項に一度だけ行うため、重複計数はない。

**ステップ2（厳密発展との差）**：厳密発展はground上で \(e^{-izH'}|0\rangle=e^{-izE_g}|0\rangle\) であり、その次数 \(n\) の係数normは \(E_g^n/n!\)。\(0\le E_g\le\Lambda\) なので、これも \(\sqrt{\Lambda E_g}\,\Lambda^{n-1}/(n-1)!\) 以下。Strangは**operatorとして次数0・1・2まで一致**するから、差は \(n\ge3\) から始まる。ゆえに実数 \(h\) について

\[
\begin{aligned}
\eta(h)&\le2\sqrt{\Lambda E_g}
\sum_{n\ge3}\frac{|h|^n\Lambda^{n-1}}{(n-1)!}\\
&\le |h|^3\Lambda^{5/2}\sqrt{E_g}\,e^{|h|\Lambda}.
\end{aligned}
\tag{R1 revisited}
\]

\(E_g\) は利用可能な認証値 \(G=U-L\ge E_g\) に置き換えられる。

**評価の限定**：これは**新規な科学計算ではなく、既知の演算子Taylor評価を組み合わせた紙上の保守的十分不等式**である。原文献に未掲載の新定理だという初出性・最適性は主張しない。\(e^{|h|\Lambda}\) と \(\Lambda^{5/2}\) が粗いので、分子で実用的な保証付きstep幅が拡大するという証拠ではない。詳細な定数と適用範囲は、もし研究主題に採用するならCodexによる独立した技術的検証・厳密な書き下しが望ましい（**実行・数学検証作業は今回承認していない**）。

**意味**：gap不要のfinite-\(h\)保証の**存在**は未解決ではない。**実用的な定数、鋭い資源比較、他の既知state-dependent boundsとの差**が本質的な未解決点である。

### 3.7 四分の一乗則をどこまで主張できるか

上記の状態誤差上界と(S)から、\(\omega=|h|\varepsilon_b\) が小さい場合の理論的な失敗重みは、おおまかに

\[
P_{\rm far}\lesssim
\frac{h^4\Lambda^5 G\,e^{2|h|\Lambda}}{\varepsilon_b^2}
\]

と制限できる。このレベルで、同じ失敗確率と粗い状態非依存の \(O(h^3\Lambda^3)\) boundを比べるなら、必要step数の**形式的な**比が \((G/\Lambda)^{1/4}\) に現れる。これはClaudeの直感と整合する。

しかしこれは、(i) 固定成功率、(ii) \(h\Lambda\) が小さいため指数項が支配しない範囲、(iii) 同等の精度・読み出し・alias条件、(iv) 同程度に粗い比較上界、を満たす場合だけの**保証上のscreening**である。より鋭い既知のsector／交換子boundや実際のPF biasを対照にすれば比は変わる。

励起gapを使う \(O(E_g)\) 型の**個別固有値bias評価**と、ここでの \(O(\sqrt{E_g})\) 型の**円周上の失敗重み**は、同じ保証対象ではない。この違いを「gapがないだけで普遍的に一段階の資源損」と定式化すべきではない。

### 3.8 有用なエネルギー窓についてのClaudeの主張

認証済みの \(L\le E_0\le U\) があり \(G=U-L\le2\varepsilon\) なら、同じ誤差基準の下で中点を出すだけで加法誤差 \(\varepsilon\) が保証される。**この意味で、化学Hの厳密値を量子計算で新たに推定する必要はない**というClaudeの注意は妥当。ただしその下界・上界を取得する**古典的な費用はゼロではない**。

一方、\(G\gtrsim\Lambda\) ならSOS上界の小ささからの**保証レバー**は期待しにくいが、これは「SOS-PFを使うあらゆる分割・回路で実際のbiasやtotal costが下がらない」という必要十分条件ではない。\(2\varepsilon<G\ll\Lambda\) は**有望性を調べるための目安**であり、全研究の形式的可否判定ではない。

---

## 4. 先行研究の包含関係と新規性の評価

### 4.1 Kang–Suが否定する「広い新規性」

Kang–Suは、SOSを二重の演算子構成から時間発展で実装し、外側のproduct formulaで和をシミュレートする。化学系についてHermitian一体平方・DFTHC・固定粒子数sector normを明示的に扱い、QPEへの応用に言及する。

したがって「SOSで表現した電子HamiltonianをPFでシミュレートする」「平方fragmentを使う」「particle sectorでcommutatorを締める」のみを**新しいアルゴリズムの中心claimとして提出できない**。この点はK0・K0.5・Claudeの結論を支持する。

ただし、**実装アクセスは同一ではない**。Kang–SuはHamiltonian-evolution型block encodingとHamiltonian squaring/QSVT型の算術を含む。提案SOS-PFはHermitian一体平方を直接軌道回転＋対角位相で指数化する。直接実装の具体的価格が異なる可能性まで「完全同一」と断じない。またKang–SuのSOS計算で \(E_g\) 依存の**ground-energy専用finite-step certificate**が証明済みだという資料は確認できていない。

**結論**：「広い構想」は既知、**全構成の完全同値性**と**狭い保証命題の既知性**は未確定、という三段階を守る。

### 4.2 低エネルギーPF理論が既に示していること

Hejazi et al.はMagnus＋固有値摂動でQPEのbranch biasを扱い、低energyでの改善を局所・PSD・可解群数等の仮定の下で考察した。Mizuta–KuwaharaはPSDかつ局所性・extensiveness等の条件から有限時間の低energy発展誤差を改善した。Yi–Crossonは固有値・固有vectorとbranchへの影響を区別した。

したがって**「低エネルギー×PSDならPF誤差が抑えられる」一般原理は既知**。電子化学での非局所な一体平方へ**有用で検証可能な定数**で接続し、直接エネルギー推定に役立つ保証を出す点にのみ、独立成果の候補が残る。

局所定理を非局所系へ粗く特殊化できることと、化学において競争力のあるnorm依存・step価格を得ることは異なる。単なる「非局所だから空白」は誤りであり、**適用条件・定数・保証対象・情報価格まで違うか**を論文の主張単位で比較すべきである。

### 4.3 PF-native分割生成 \(R\) の科学的新規性は現状不足

K0・K0.5が整理した

- DFの定数shift、
- 一体平方完成と残余への再配分、
- SOS Gram SDP（v2RDM／BLISS等）、
- 複素Gramの実化、
- Hermitian平方の指数化、
- Gram因子の直交回転、

はいずれも**存在や基本的な生成処理が既知**である。より狭い実対称Gramの \(R_0\) も、現状では**既知のSOS SDPの制約付きサブクラス**である。

このことだけで「既知の部品の組合せだから絶対に論文にならない」とは言えない。たとえば**有意で反証可能な資源スケーリングの構造的利得**が明示されれば、既知構成を使う新しいトレードオフ定理・アルゴリズムはあり得る。しかし現資料は、\(R_0\) が一般の分子群で許容可能なrank・幅でよい \(G\) を返すこと、既知の強い手法より少ないstep・総費用を実現することを示していない。

**科学判断**：A1の**独立性と期待利益を同時に立証する入口条件が未充足**である。K0・K0.5が終わった時点で惰性的に同族の \(R\) を追加探索することは推奨しない。

### 4.4 ACM出版版未取得の位置付け

Claude・CodexはACM全文取得のHTTP 403を記録し、arXiv v3と出版版の全文同一性を推定していない。今回もその差分を完全には認証していない。この欠落は**定理番号・有限係数の引用、最終論文での初出確認**には影響する。

しかし、arXiv v3のSOS＋PF＋化学応用の広い構成が既知であるという認定は、出版版との差分の未確認を理由に撤回する必要はない。逆に「v3を読んだので出版版も読了」と言ってはならない。**今回のA1区切り判断を妨げる決定的な資料不足とは判断しない**。

---

## 5. Claudeの費用評価・K1-F提案への批判的検討

### 5.1 「実biasとnative価格だけ」という整理の成立範囲

予測済みのground bias係数 \(c_p\) とstep価格 \(C_{\rm step}\) があり、同一PF次数・同じ成功モデル・同じ情報費用で漸近費用を比較するなら、

\[
C_{\rm quantum}\propto C_{\rm step}\,|c_p|^{1/p}\,\varepsilon^{-1-1/p}
\]

のようなscreening式が便利である（期間・round・定数等は省略）。その狭い模型で「bias係数とstep単価」が主要レバーになるのは正しい。

しかし、実際のPF-QPE/RPEの総計画費用は概念的に

\[
C_{\rm task}\approx
N_{\rm prep}C_{\rm prep}
+\sum_{q=1}^{N_{\rm query}}C_{\rm controlled\,PF}(q,h,\epsilon_{\rm synth})
+C_{\rm info}+C_{\rm classical},
\]

と、準備回数・成功確率・読出し・branch・制御・合成・情報取得等に依存する。**外部に状態準備方法の開発を置いたとしても、反復回数やコピー数を恣意的に無償化しない。**

さらに真の実biasが同じでも、未知のbiasを保証付きに抑える手続きが改善し、選択可能な \(h\) が大きくなれば、**実際に実行される**量子回路数・深さは変わる。したがってClaudeの「保証だけでは実際の量子資源は変わらない」は、**事後oracle最適stepと保証に従う実装stepを混同**している。A2の研究価値を過小評価する理由にはならない。

### 5.2 それでもA2を主テーマとして推奨しない理由

A2に現実的な資源価値があり得ることと、**現時点で独立した有用な保証・新規性があること**は別である。現在の具体的な障害は、(i) 実用的に小さな \(G=U-L\) の確保、(ii) nonlocal化学PSD fragmentのnorm・高次剰余、(iii) 必要な励起gap・あるいはgap不要なQPE失敗率の正確な契約、(iv) 既存Childs／sector／low-energy boundと同じ合法情報での改善幅、(v) 古典証明書の取得・検証費用、である。

本レビュー§3.6の粗い(R1)も、**有限step保証が存在する**ことしか示さず、これらを解決しない。\(\varepsilon\) や許容失敗率に対して、どの分子クラスで既存法より実際に大きい \(h\) を安全に認証できるかは未確定。

したがってA2は「理論として無意味だから却下」ではなく、**研究目的を保証理論へ変更してまで投資する正の証拠が今は不足**という理由で主テーマへの自動転換を勧めない。短い理論ノートという可能性の有無も、差分の証明と既存の鋭い対照を閉じてから判断する。

### 5.3 K1-Fの科学的情報価値：存在するが優先順位は低い

Claudeの提案する、固定 \(R_0\) に対して最大2つの保存H4入力・SDP各1回・S2＋DF対照で反証するK1-Fは、**実装可能性の反証、PF-native \(G\) の劣化、fragment数・回路費用増、有限step biasの悪化**を知る手段にはなる。

ただし、その結果が肯定的でも \(R_0\) の生成規則は既知SOS SDPの制限であり、**独立的新規性が自動的に生まれるわけではない**。H4の二幾何だけでは異分子・サイズスケーリング・精度窓への一般化もできない。一方、否定的でも**PF-native SOSの一般的不可能性**にはならない。

#### Claude草案の科学的に問題のあるgate

1. **実bias半減、単価1.5倍以内**というAND条件は、S2の費用根拠 \(C\propto C_{\rm step}\sqrt{|d|}\) で**費用改善の十分条件ではない**。境界値なら \(1.5\sqrt{0.5}>1\)。さらに別の実効次数・有限grid・異なるoverlapを扱う場合、単純な係数条件は不十分。
2. **\(G_{R_0}\ge G_{DF}\) なら即STOP**は過度に強い。\(G\) だけでなく \(K_{\rm SOS}\)、実bias、回路価格、相殺等が変わるので、\(G\) の単独劣化からすべての費用レバーを否定できない。
3. **\(G/\Lambda\le10^{-2}\)** は理論的な有望性の例示であって、ユーザーが事前登録した正式基準ではない。同じ分子・元H・情報費用の下でどの程度の差が意味を持つかを別途決める必要がある。
4. K0の未認可K1草案ではSDP0回だったが、ClaudeのK1-FではSDP最大2回へ変更する。**計算承認のscopeが明確に変わる**ため、過去の承認・保護markerを流用できない。

**結論**：新規算法という主目的を維持する限り、K1-Fを現段階で推奨しない。唯一、ユーザーが「PF-SOSの実現可能性という物理的事実を、後の研究方向選定のため一度だけ判別したい」と明示した場合には、**研究採択・K1科学計算承認とは別に**主指標・total費用のpass条件・独立入力・実装契約を設計したうえで限定検証を提案できる。

### 5.4 結果が研究判断を変えるかという基準

同じK0型の情報収集をさらに繰り返しても、有利な \(R\) と必要な分子の実費用は自動的に得られない。よってこの時点での選択肢は「未解決だから数値を続ける」ではない。**K1-Fが成功したら次に何を新たに主張できるか**を先に定義できないなら、K1-Fの優先順位は低い。

---

## 6. A1・A2・A3の比較と研究全体の選択

### 6.1 三方向の比較

| 評価軸 | A1：分割生成算法として継続 | A2：保証理論への転換 | **A3：現候補を区切り保存** |
|---|---|---|---|
| 現在の具体的な新規性 | **未特定**の \(R\)。既知制約SDPと近い | 既知の断片の統合。鋭い非局所化学保証なら残り得る | 新規性を追認しない |
| 元の低費用アルゴリズム目標 | 直接的。ただし利益の機構は未実証 | 保証下のstep選択を通じて貢献し得る | 直接の算法成果を追加しない |
| 現在の厳密な結果 | PSD gap型の漸近係数上界、実装可能性の一部 | 条件付きfinite-step式と粗いgap不要保証（§3.6） | 証拠を保護・蓄積 |
| 研究リスク | 高：生成差・rank・実bias・価格 | 高：既存理論との重複・実用定数・certificate費用 | 小：研究資源を再配置できる |
| 次に必要な作業 | 新しい差分を先に明確化、または認可済みの限定反証 | 独立定理の仮定と強対照・実用領域の確定 | 正式状態の最小更新だけ |
| 主要claimの確度 | 低 | 低～中（粗い保証の存在と価値を分ける） | **中～高**：投資中止の判断として妥当 |
| GPT推奨 | 現構成で継続しない | 自動的には主テーマにしない | **第一推奨** |

### 6.2 理論保証研究A2を否定しすぎない

本レビューはClaudeよりA2の**運用上の価値**を高く認めるが、主テーマの推薦順位は変えない。理由は、**価値が存在する可能性と、独立した論文的貢献が今成立していることは異なる**ためである。A2が再浮上する条件は、例えば

- **非局所化学の意味のあるクラス**で \(K\)、\(G\)、認証可能な剰余が全て係数・多項式コストの証拠から得られる、
- 同じS2、同じ情報モデルで既知の強いfinite-time/sector/low-energy boundsより有意に大きい保証stepを得る、
- その意味で**保証つきの量子実行総費用**が減ることを示す、
- 既知結果からの単純な特殊化ではない定理・構成の差分を明確にできる、

といった、**具体的な反証可能な主張**を構築できた場合である。今の資料にその証拠はない。

### 6.3 今後の研究スコープに関する二つの判断

Claudeが挙げたD1（block-encoding/qubitizationの除外）とD2（絶対ground energyだけかenergy differencesも対象か）は、**別の研究方針を選ぶうえでは重要**である。ただし、これを決めなければ**現A1を区切る判断すらできない**わけではない。

- **D1：方法範囲**。現時点では従来の「新規構成としてblock-encoding/qubitization中心の方法は対象外」を**無断変更しない**。ただし**既知の強い対照として利用すること**と、**新算法の構成要素に採用すること**は別判断である。後者の解禁はユーザーの明示選択が必要。方法範囲を広げればSOSSAを含む既報との新規性競争が一段と厳しくなる。
- **D2：タスク範囲**。絶対ground energyを維持するなら、差エネルギーで起こる誤差相殺を同じtaskの成果に流用しない。energy gapや反応エネルギーを独立テーマにすることは可能性があるが、その場合は入力状態・複数branch・誤差相関・正しいbaselineを**新しく定義**する必要がある。Bay-Smidt et al.のPPP模型での大きな改善を、一般分子の無条件の利益としない。

したがって**D1/D2の即時変更を前提としてE1/E2を新テーマへ自動昇格させることは推奨しない**。

### 6.4 次の研究として科学的に価値が高い進め方

現在の局面で求めるべきは「SOS-PFの後継となるもう一つの小さい分割案」ではない。次の方向選定の前に

1. **研究目標を一文に固定**：絶対ground energy・許容精度・成功率・合法アクセスと、比較する量子資源（T/Toffoli、RZ層、論理qubit、制御回路回数など）を整理する。異なる資源座標を一つの倍率に捏造しない。
2. **有力な費用レバーを先に特定**：幅を大きくしたときの偏りだけでなく、branch/overlap、実効次数、制御回路・合成、情報取得価格、同情報PF・SOSSA等との距離を把握する。既存のTrack A/B台帳で既に確立している内容は再計算せず参照する。
3. **そのレバーに効く具体的な構成規則だけを候補にする**：何の入力を何へ写し、どのcostを何倍程度動かせる機構か、最強baselineとSTOPを事前に書けないならK0/K1を開始しない。
4. **既存の研究資産との優先順位を別途比較**：まだ最終資源評価に達していないTrack A/B、第1・第2研究等について、少なくとも今後の投入資源に対する科学的情報価値を、何も得られていない新テーマと比較する。別trackの既存statusは勝手に変更しない。

これは「新しい研究方向を既に採択した」という判断ではなく、**次の科学的意思決定のための方法論的推奨**である。Codexに同型のK0静的監査を延々と要求するより、研究の問いと資源レバーが固まった節目で、新たな科学レビュー（ユーザー承認が必要）へ戻す。

---

## 7. 推奨する次の担当・具体的作業・承認事項

### 7.1 直近の作業

**第一推奨（ユーザーによる採用が前提）**：A1の現定義（「新しいPF-native SOS分割生成算法」としてのSOS-PF）への追加研究投資を停止・保存する。A2は**未採択の理論的可能性として索引だけ残す**。\(R_0\) のK1-Fや候補B/Cの自動開始はしない。

これを採用する場合の担当は次のとおり。

| 順 | 担当 | 作業 | 承認境界 |
|---|---|---|---|
| 1 | **ユーザー** | A3を**現A1の研究投資・状態判断**として採用するか決める。A2・K1-Fを未採択にするか明示 | GPTは正式CLOSEDを代行しない |
| 2 | **Codex** | ユーザー採用後、K0・K0.5・Claude・本GPTレビューへの参照を付けて、正式状態を必要最小限で記録。既存結果は書き直さない | GitHub変更・commit/pushは対象範囲の別承認に従う |
| 3 | **ユーザー／GPT** | 次の独立研究方向についてD1/D2・費用目標・他trackとの優先順位を検討。必要なら別の重要科学レビューを明示承認 | 今回のレビューの結果だけで次テーマを採択しない |
| 4 | **Codex** | 採択された新しい科学的問いに限り、必要な文献・技術K0をまとめて検証 | 新数値は別途明示承認。既存STOPを流用しない |

### 7.2 A2やK1-Fへ進む場合だけ必要な別の判断

- **A2（保証理論）**：新規アルゴリズム研究から保証・認証研究への目的変更を正式に受け入れるか、何を主claimとし、どの強い既知boundを超える必要があるかを先に決定する。本レビューは**自動移管を承認しない**。
- **K1-F（SDPと小系数値）**：固定 \(R_0\) の反証という情報価値を認めた場合に限り、比較task、入力SHA、許可情報、判定式、最低限のscience source、SDP回数、数値精度・保護・retry規則を確定し、**科学レビューとは別の明示的な計算承認**を取得する。ここまで許可なし。
- **B/C・旧D・PAUSED系列**：いずれも別判断なしに自動再開しない。

### 7.3 Mandatory STOPと確定していない事実

本レビュー終了時点で

- **SOS-PFの正式研究状態**：K0 `UNDETERMINED_DATA_REQUIRED`、K0.5完了、研究テーマとして未採択、K1未認可。
- **既存研究の正式状態**：第一候補の現固定構成CLOSED、H-E1現定義終了、M-a/M-b追加探索STOP、PF小改変系列PAUSED、旧D案HOLD、その他独立研究変更なし。
- **新しい科学計算・SDP・再集計・PF pilot・compile/T合成**：未実施。
- **GitHub commit/push・method statusの変更**：未実施。
- **新研究方向の正式採択**：なし。

**Mandatory STOPを維持する。** 本文の新しい紙上不等式(R1)は、重要科学レビューの範囲内での数式的考察であり、K0/K1の新規科学実験を開始したものではない。

---

## 8. 最も強い反論、弱点、未解決事項

### 8.1 A3への最大の反論：「まだ最小pilotすら行っていない」

**反論**：PSD化した分割から \(G\ll\Lambda\) を得られれば、PFでも保証・実biasの大きな改善がある可能性があり、A3は有望な方向を早く捨てすぎている。

**評価**：この反論は物理的には合理的である。K1-Fは、固定された \(R_0\) に対してPF-native化の利益が失われるかを限定的に判別できる。しかし**肯定的でも新規アルゴリズムの生成原理が成立するわけではなく**、小系二幾何から規模への外挿もできない。今回の中心目標が「研究として独立した新しい低費用アルゴリズム」である以上、追加投資の情報価値は相対的に低い。A3は**一般的不可能性**ではなく、**今の研究資源配分の推奨**なので矛盾しない。

### 8.2 A2への最大の反論：「保守的でも保証を改善すれば本当に費用を減らせる」

**評価**：正しい。特に厳密な許容失敗率・精度を要求する実務では、保証付きstep選択の改善が実量子回路を節約し得る。本レビューはClaudeよりこの価値を明確に評価する。しかし今は既存の最強保証との厳密な比較、正しいcertificateを取得する費用、化学の有用な数値範囲が閉じていない。**A2を切り捨てるのではなく、主テーマとして未採択のまま保存**することが科学的に妥当である。

### 8.3 理論的な注意

- §3.6の導出は独立した**紙上の保守的数学評価**であり、他文献に未記載の数学的新規性を認定しない。必要なら正式な証明書としての査読や別の独立検算が必要。
- Claude §3.8(d)にある「Rayleigh位相＋Kato–Temple型」の \(O(h^3E_g)\)、\(O(h^4E_g/g)\) の詳細定数は**未導出**であり、今回も完成定理とは扱わない。
- K0 Cauchy＋Schurの式は数学的な十分条件として整合するが、\(1/\Lambda\) の狭い半径、\(E_g\) と無関係なglobal remainder、認証gapの取得価格を持つ。
- 弱い上界比や2つの小分子の数値比較から、SOS-PF全般の優位・不可能性・論文化可能性を確定しない。
- ACM出版版の全数式・版差分は確認できていない。ただし広い先行構成の重複を認定するためのarXiv v3は公開されている。
- block-encoding/qubitizationの競合研究が高度化していることは事実でも、**PF研究の理論的価値や論理qubit・実装特性が消滅した証拠にはならない**。SPRINTはPFが別資源で競争できる例を報告しているが、タスクや入力条件が異なるため同一倍率の比較は不可。

### 8.4 判断の確度を分ける

| 判断 | 確度 | 根拠と主な留保 |
|---|---|---|
| A1の現構成に独立した有利な生成規則がない | **高（現資料の範囲）** | K0・K0.5と先行研究対応。ただし新規則一般の不存在を証明していない |
| 現A1を区切る投資判断 | **中～高** | 研究目的・候補成熟度・追加pilotの期待情報価値からの判断 |
| A2の運用上の価値はゼロではない | **高（原理的）** | 同回路でも証明付きstep変更は実行資源へ影響する |
| A2が独立した有用な新論文となる | **低～中** | 最良既知boundとの差・化学での定数・証明費用が未閉鎖 |
| gap不要の粗いfinite-step SOS状態誤差上界（R1） | **高（示した紙上不等式の範囲）** | 全実hで成立する保守的Taylor評価。新規性・実用性は未認定 |
| 四分の一乗・平方根の「実際の費用倍率」 | **低** | 特定の粗い上界・成功率・有限h範囲に依存 |
| K1-FがA1新規性を立証できる | **低** | \(R_0\) は既存SDPの制限。小系pilotは外挿に弱い |
| PF全般を止めるべき | **判断していない** | 現A1のnegativeと他研究の正の局所結果は区別する |

---

## 9. 引用資料と改訂の位置付け

**一次文献**：§1.2に各論文の公開URLと確認対象を列挙した。特にKang–Su [arXiv:2510.06316](https://arxiv.org/abs/2510.06316)、Hejazi [arXiv:2412.16811](https://arxiv.org/abs/2412.16811)、Mizuta–Kuwahara [arXiv:2504.20746](https://arxiv.org/abs/2504.20746)、Childs [arXiv:1912.08854](https://arxiv.org/abs/1912.08854)、Yi–Crosson [npj QI 8, 37](https://www.nature.com/articles/s41534-022-00548-w)、Rubin–Low–DePrince [arXiv:2602.05069](https://arxiv.org/abs/2602.05069)を中心とする。

**研究資料**：[K0固定資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/README.md)、[K0.5固定資料索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/README.md)、[研究状態正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78b50c515d9df9556f261e74ca7c57af461b7199/docs/research/%E7%A0%94%E7%A9%B6%E3%83%8E%E3%83%BC%E3%83%88/2026-10-15_qc_pf_exploration_state_adoption.md)。固定commitは本文で明示した。

**レビュー原本**：ユーザー添付 `レビューoutter(1).md`（Claude完成版、2026-10-16）。K0の入力として以前保存されたGPTレビューは[固定GitHub](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/supplied_gpt_review.md)。本レビューは後者の繰返しではなく、**K0・K0.5後の新証拠、ClaudeのA3判断・T1/T2案・K1-F・スコープを評価**したものである。

**このレビューで新たに加えた主な点**：

1. Claudeが提示した資源レバーの二分法を、保証付き運用と事後oracle比較の違いに基づいて限定した。
2. 多fragment一様norm行和上界の未検算部分を、raw辞書の紙上係数計数から確認した。
3. gap不要・有限stepの粗いSOS状態誤差上界(R1)を、Taylor係数とPSD分散から独立に構成した（新規性・実用性の主張なし）。
4. ClaudeのK1-F success／STOP例が総費用目的の必要十分条件ではないことを明示し、採用を推奨しなかった。
5. A3支持の理由を「既知だから科学的に不可能」ではなく、「独立生成規則・十分な情報価値が未確立な段階での資源配分判断」と明確化した。

---

## 10. 最終科学判断

**【独立レビューとして確定する推奨】**

1. **現SOS-PF（A1）は新しいPF-native分割生成アルゴリズム候補として区切ることを推奨する。** K0・K0.5を超えて生成規則を探し続けること、未認可K1-Fを「次は数値だから」と自動起動することは勧めない。
2. **SOS gapを用いる保証理論（A2）の数学的可能性を否定しない。** gap不要の粗いfinite-step状態誤差は紙上で上界できる一方、化学で鋭いcertificateを合法・安価に得ることは未確立。したがって現時点で主テーマに変更しない。
3. **研究全体では、先に強い比較対照・タスク・量子資源のレバーを選ぶ。** 既存Track A/B等の成果を別研究として尊重し、未採択のE1/E2やSOS-PF後継候補を自動開始しない。block-encoding法を**対照として参照すること**と**新算法として採用すること**を分離してユーザーが範囲を判断する。
4. **正式状態の変更はユーザー承認後に限る。** 既存の研究判断・SOURCE・計算認可・mandatory STOPは変更していない。新規科学計算は別途承認を要する。

**レビュー状態：COMPLETED。** 以上は科学的判断の推奨であり、研究方針の正式採用・計算・GitHubの変更を代行したものではない。
