# Track B：G1後の独立研究レビューと研究方針の再設計

**作成日：2026年10月9日（JST）**  
**版：1.0**  
**対象：PR内部のアルゴリズム改良／RA-RTE。研究A、Hamiltonian前処理、PR外のアルゴリズム研究とは分離する。**  
**証拠の最終基点：G1結果 commit `e62407b3c51e9318f700673b6cf403d54e344c76`。**

本書は、添付チャット履歴、固定commitの研究資料・保存値、一次文献を照合したGPTによる研究レビューである。ユーザーから、添付履歴の最終時点以降に追加のCodex作業・commit・研究方針変更はないと確認されている。

旧実行の再開、登録LPの求解、新しい回転角の合成、量子回路・分子シミュレーション、リポジトリの変更は行っていない。本レビューでは、新たな数式整理と、その記号的・算術的な自己検算を行った。その部分は既存の登録実験結果とは区別する。

---

## 要旨：今回の研究判断

**Track Bを終了する根拠はない。一方、RA-D0 v4の全面実装を直ちに進める根拠も不十分である。**

推奨するのは、RA-RTEを現在の有力候補として残しつつ、研究の中心を次へ具体化することである。

> **有限Taylor平均を保つ乱択表現において、次数間の係数配分を変える自由度は、既知のsampling最適化・低次数への吸収を考慮した後にも、有限精度のcoherent-signal推定資源を減らすか。その条件を、利用可能な情報から評価して実行列を構成できるか。**

「B3を最適化できた」「exact LP backendが動いた」「6頂点がある」だけを研究の完成条件にはしない。目標は、**どの構成を、なぜ、どの情報から選ぶべきかを明らかにするアルゴリズムと、その有効範囲**である。

今回の重要な判断は四つある。

1. **既存の数学的成果は残す。** R0の限定最適性とG1の理想P₃構造は、研究資産として明確に意味がある。ただし、任意LCUでの最適性や新規性の確定ではない。
2. **B2/B3比較を、主研究の全体ではなく機構検査として位置付ける。** 同一保存表で自由度の追加価値を調べる設計としては適切だが、それだけで既知手法全般を上回ることは示せない。
3. **本格実装前に、比較の強さを確かめる。** 既知のimportance samplingと、R0にも保存されたidentity-return構成が重要である。今回、それらを無視できない具体的な解析上の理由を確認した。
4. **次のCodex作業を一つにまとめる。** 保存表による構造・資源診断と、本書の数式の独立監査を行う。新しい大規模solver基盤や登録one-shotには自動進行せず、その結果でGPTが次の研究判断を行う。

| 判断事項 | 今回の結論 |
|---|---|
| GPT研究レビュー | 必須。本書で研究内容のレビューを実施 |
| Track B全体 | 継続候補を残す |
| RA-RTE | 有力候補。ただし実資源上の追加価値・独立新規性は未確定 |
| RA-D0 v4全面実装 | 引き続き保留 |
| 既存のR0・G1・v4数学監査 | 棄却しない。限定範囲を維持して利用 |
| 次の主担当 | Codex：限定した保存値診断・独立反証を一括実施 |
| 次のGPT判断点 | G2：追加自由度の研究価値と、必要な最小実装・実験の決定 |
| 新規合成・分子計算・旧run再実行 | 今回は行わない。別途の研究判断・実行指示が必要 |

---

## 1. レビューの根拠と、確認の深さ

### 1.1 三種類の根拠を区別する

本書では、以下を混同しない。

- **既存の研究証拠**：固定commitの報告、証明文書、保存された資源値。
- **先行研究**：今回確認した一次論文の具体的な構成・保証。
- **今回のレビューでの導出**：既存の式・保存値を出発点にした新しい整理、条件式、算術再計算。

「今回の導出」は、この回答内で自己検算した結果である。Codex等による独立監査済みの定理、先行研究に対する優先性、登録された資源改善結果として扱わない。

### 1.2 直接確認した主要資料

| 資料 | 固定commit | 今回の用途 |
|---|---|---|
| G1 one-shot result validation | `e62407b3c51e9318f700673b6cf403d54e344c76` | 6頂点、B2 embedding、人工LPの到達点と限界 |
| R0 independent proof | `672d6bc667eaa7b9ca4979b012f1530499d701b8` | 有限平均、全奇数次数の限定最適性、identity return、multiblock上界 |
| R0.5 equivalence/novelty audit・handoff | `61dd534567fda5c7348fdc688814089eb26a3561` | PTSC/CTSとの違い、I0/I1、既知部分 |
| R1.5 saved-value attribution | `af3d014d0a0cfcbbd25bb544f6544652fec92942` | 合成精度、bias、shots、native costの関係 |
| RA-D0 preparation handoff・manifest | `0ddf67756516e08f85fed1b987459a5e862676b7` | 保存表の由来、数値baseline、巨大gridの経緯 |
| RA-D0 candidate tableの該当部分 | 同上 | A0のword別確率・native T費用の確認 |
| v4 mathematical-audit handoff | `beb82427d202f479cc2ba954480d73a51941e322` | inner/outer、丸め、認証可能性の保証範囲 |
| 添付 `Rpartially_trackB.md` | ユーザー提供の履歴 | BF/BM/SP/BSからの判断履歴、最新GO撤回、研究範囲 |

候補表については、manifestと関連する保存recordを確認したが、全21 columns/xの全recordを今回独立再監査したわけではない。したがって、本書はB3の最良点・完全な資源frontierを新たに報告しない。

R0の証明文書は内容を数学的に検討したが、旧checkerや全実験を再実行してはいない。G1・v4のPASS件数は原報告の内容であり、今回新しく増やした成功件数ではない。

---

## 2. 研究は何を目指し、どこまで到達したか

### 2.1 Track AとTrack Bを混ぜない

Track Aは、既存PR構成のmatched-accuracy資源競争と適用条件を扱う。Track Bは、**時間発展・乱択表現・回路生成の方法自体を改善する**ことを目指している。[H1]

Track Bの成功をTrack Aの完成条件にしない。既に使用したH4 1.30 Å等を、新しい独立検証と呼ばない。

また、**Track AのB0/B1/B2/B3という方式名と、RA-D0のB0/B1/B2/B3という比較class名は別物**である。本書のB2/B3は、特記しない限りRA-D0のrepresentation classを指す。

### 2.2 過去のSTOPが意味すること

| 系列 | 確認されたこと | そこから言えないこと |
|---|---|---|
| B-F | 固定5-stage family・条件・探索予算では、F/Lの最良finite構成が一致。原runはINCONCLUSIVE、read-only復元がBF-A | 高次PFの全最適化が無意味、という一般的不可能性 |
| B-M | 現adapterの三次BCH係数・同情報での評価はcompact再帰と同値 | multirate構成全般に資源価値がないこと |
| SP | primitiveの合成／測定交換、toy wrapperでの累積crossover。登録条件では選択的D/R配置のmaterial gainなし | 真のfinite-RTEや全角度・全配置で改善不能ということ |
| BS | 提案された具体的なblock-LCU手順について、同条件generic LCUとの差が未定義 | 汎用LCUの部分集合である新構成はすべて無価値、ということ |
| R0/R0.5 | 有限平均を保つfamilyと、限定されたnormalization最適構成A | 全dictionary・Pauli相殺・sampling法を含む最適性 |
| R1/R1.5 | 固定toy・合成器・保存列におけるprecision依存の資源trade-off | 大系、独立条件、全PR、化学精度エネルギー推定での優位性 |
| RA-D0 v3〜G1 | 技術障害の分離、条件付き数理保証、理想構造、人工backendの確認 | B3対B2の登録資源改善 |

過去の反証は候補選別に有用である。しかし「前の案が失敗したから次の案が有望」という推論は成立しない。次の案にも、固有の成立理由と反証可能な問いが必要である。[H1][R1–R5]

### 2.3 現在の未解決事項は一つではない

現在は、少なくとも三つの問いを分ける必要がある。

**数学の問い**：同じ有限平均を保つ新しい構成を作れるか。  
→ 限定classで明確な成果がある。

**資源の問い**：新しい自由度は、適切に調整した対照より有利か。  
→ R1では一部trade-offがあるが、B3対B2は未判定。

**研究価値の問い**：その差は既知手法の直接適用を超え、他者が使える方法・知見になるか。  
→ まだ確定していない。

G1のPASSで第一の構造と技術基盤が進んでも、第二・第三の問いへの答えは自動的には得られない。[R1]

---

## 3. 現在の数学的成果の評価

### 3.1 R0の中心的な成果

\[
\widehat R=\sum_\ell p_\ell Q_\ell,\qquad
p_\ell\ge0,\quad\sum_\ell p_\ell=1,\quad
Q_\ell^\dagger=Q_\ell,\quad Q_\ell^2=I
\]

とし、有限targetを

\[
M=P_m(-i\sigma x\widehat R),\qquad
m=2d+1,\quad t_k=\frac{x^k}{k!}
\]

に固定する。

非負係数が

\[
a_k+b_{k-1}=t_k,\qquad b_{-1}=b_m=0
\]

を満たすとき、

\[
c_k=\sqrt{a_k^2+b_k^2},\qquad
U_k=(-i\sigma)^k e^{-i\sigma\phi_kQ_0}Q_k\cdots Q_1,
\quad\phi_k=\operatorname{atan2}(b_k,a_k)
\]

によって

\[
\sum_k c_k\,\mathbb E[U_k]=M
\]

を保てる。各indexは元の分布から独立に生成する。非可換word中でrotationを勝手に移動することや、複数occurrenceへ同じsampleを再利用することは、この証明に含まれない。[R2]

偶数・奇数のTaylor係数総量をE、Oとすると、非負adjacent-degree classでは

\[
B=\sum_k c_k\ge\sqrt{E^2+O^2}
\]

であり、全奇数次数に対する明示的達成構成Aがある。これは単なる有限gridでの観測より強い成果である。[R2]

### 3.2 ただし、限定最適性の外側が重要

この定理は、符号付きdictionary、異なるwordの相殺、Pauli collection、任意のLCU構成、異なる有限targetまで最適化していない。

原証明自身に、単一involution・x=1で、P₃を一つのscaled rotationとして表したnormalization²が17/18である一方、adjacent非負classの最適値は65/18となる例がある。[R2]

したがって、論文で主張できるのは「定義したclass内の最適性」であり、「有限Taylor simulation一般の最適性」ではない。

### 3.3 R0だけから新しい漸近スケーリングは出ない

P₃について、既存証明には

\[
B_{\rm pair}-B_A=\frac{x^4}{9}-\frac{13x^6}{81}+O(x^8)
\]

がある。総無次元時間τをs等分しx=τ/sとした場合、

\[
s\log\frac{B_{\rm pair}}{B_A}
=\frac{\tau^4}{9s^3}+O\!\left(\frac{\tau^6}{s^5}\right)
\]

である。[R2]

**この差だけを根拠に、PR全体の新しい漸近計算量改善を期待するのは弱い。** 現在最も妥当な目標は、有限のstep・実装精度・回路費用における改善と、その成立条件である。

高次数へ進むなら、「別の次数なら勝つか」を探すためではなく、構成・必要情報・計算量・価格条件の一般化を証明するために進むべきである。

---

## 4. R1/R1.5の結果は、何を支持するか

R1の主条件は、2-qubit、P₃、x∈{1/8,1/4}、p=(3/4,1/4)、distinct-basis controlled実装である。保存costは、共通のexact cancellation後の**合成済みprimitiveの加法的native counts**であり、whole-circuit optimizerで得た最終最適costではない。[R4]

主taskのaxis精度は0.005、complex精度は0.01、familywise失敗確率は0.05である。ここでの資源は実測値ではなく、保存された十分shot数と1-shot費用から計算したforecastである。[R4]

x=1/4、合成精度10⁻³のA/PTSC-K0は、

\[
\frac{B_A^2}{B_P^2}=0.9945715353,
\quad
\frac{N_A}{N_P}=0.9055130347,
\quad
\frac{\mathbb E C_{T,A}}{\mathbb E C_{T,P}}=0.9094176195,
\]

\[
\frac{G_{T,A}}{G_{T,P}}=0.8234895084
\]

となっている。二次モーメントの約0.54%差だけでは、総T費用の約17.65%差を説明できない。保存合成列のT費用とbias上界が、ともに影響している。[R4]

一方、x=1/8、10⁻⁴ではAのshot数は少し多く、T費用の低下がそれを上回る。Aの10⁻⁶は、両xでPTSC-K0にdominateされる。x=1/4のA・10⁻⁴は、1Q座標により三資源Paretoに残るが、T/CXでは有利ではない。[R4]

### 研究上の解釈

これはRA-RTEを考える十分な動機である。**normalization最小と、有限精度下の資源最小は同じではない**からである。

しかし、少数のangle・precision・固定synthesizerの離散的な費用に依存するため、普遍的な利得でもない。二符号の一致は実装controlであり、独立replicationではない。既存二つのxと三つのprecisionは、これ以降の設計に使うdevelopment情報として扱う。

---

## 5. 新規性の監査：残るものと、既知であるもの

### 5.1 近接する一次研究

| 先行研究 | 本文で確認した既知部分 | Track Bで追加的に必要なもの |
|---|---|---|
| Güntherほか、PR論文 [L1] | 高次PF＋RTE、adjacent Taylor pairing、single-ancilla phase estimation、factorized構成、合成と残差の調整 | 同じtask・実装前提で、具体的にどのrepresentationを改善したか |
| Zengほか、Trotter-LCU/PTSC [L2] | Taylor／Trotter補償、Euler化、ゼロ次specialization、sampling overhead | 同target・同accessの強いspecializationを超える差 |
| Peetzほか、SCU/CTS [L3] | Taylor由来operatorのPauli collection、相殺、stochastic-unitary表現、Markov layering | collection可否・取得費用・native実装をそろえた差 |
| Koczor、Sparse Probabilistic Synthesis [L4] | dictionaryの線形制約、凸最適化、bias／測定負担のtrade-off、高精度数値処理 | 「LPにした」ではない構造・入力情報・生成法・実行費用上の改善 |
| Cuginiほか、Resource-Optimal IS [L5] | 固定protocolの回路費用と二次モーメントの共同最適化 | samplingだけで得られる利益を除いたrepresentation変更の追加価値 |
| Zhao–Yuan [L6] | Hamiltonianの代数的相殺、高次項の低次数への吸収、modified Taylor/LCU | 単純な既知吸収を超える構成、またはその後に残る再配分の価値 |
| SPRINT [L7] | factorization・群ごとのPF・random残差・実装費用の統合 | 広い共同設計思想ではなく、限定した問題への具体的成果 |

PR論文は公開版に加え、arXivの版履歴も確認した。公開論文と後続arXiv版の存在を区別し、最新版全文の全箇所に対する完全な差分監査を行ったとは主張しない。

### 5.2 「channelとoperator meanが違う」だけでは対照を外せない

Sparse Probabilistic Synthesisの本文はprocess matrix／channelを主要な対象としている。system channelの分解へ後からcontrolを付けても、coherent first momentが保存されるとは限らない。[L4]

しかし、これを理由にsampling最適化や凸最適化の先行研究を除外することも誤りである。ancillaを含む実際の測定回路を標本単位にすれば、通常のimportance samplingの期待値恒等式は適用できる。

**比較時の意味論を合わせる必要があることと、既知原理が利用不能であることは別**である。

### 5.3 現在認められる新規性候補

現時点で最も筋が通る貢献候補は、次のまとまりである。

> 一般Hermitian-involutionのsampling／実装accessに対し、有限Taylor平均を保つ具体的なdegree-local表現を構成し、その限定最適性と、実装費用・誤差・測定負担を含む選択条件を明らかにする。

R0の全奇数次数の達成構成、G1の構造、実行可能な設計法、強い対照後に残る効果が、一つの方法としてつながればよい。

ただし、R0.5で直接corollaryとは確認されなかったことは、文献全体に対する新規性の証明ではない。また、**ordinary／ゼロ次PTSCも一般involutionへ拡張できるため、Aだけがそのaccessを利用できるという説明は不可**である。[R3]

### 5.4 現在の新規性評価

- **数学的な構成差：候補として残る。**
- **LP・最適化という一般原理：既知。**
- **独立論文の主貢献：未確定。**
- **量子化学での優位性：未確認。**

これは全面STOPの判断ではない。構成を具体化できている点は評価できるが、研究価値を支える比較がまだ不足しているという判断である。

---

## 6. 重要な再評価①：既知のidentity returnは弱い対照ではない

### 6.1 R0にも保存されている構成

\[
\chi=\sum_i p_i^2,\qquad
D=\widehat R^2-\chi I
=\sum_{i\ne j}p_ip_jQ_iQ_j
\]

とすると、R0原証明には、

\[
\begin{aligned}
P_3(-i\sigma x\widehat R)
={}&\left(1-\frac{\chi x^2}{2}\right)I
-i\sigma\left(x-\frac{\chi x^3}{6}\right)\widehat R\\
&-\frac{x^2}{2}
\left(I-i\sigma\frac{x}{3}\widehat R\right)D
\end{aligned}
\]

が保存されている。[R2]

同一labelの二つのinvolutionがIへ戻ることを、Taylor係数の段階で吸収する。Pauliの完全なcollectionは必要としない。ただしχや条件付き分布を作るためのpへのaccess・古典費用は明示する必要がある。

この再構成のnormalizationは、

\[
B_{\rm ret}=
\sqrt{\left(1-\frac{\chi x^2}{2}\right)^2+
\left(x-\frac{\chi x^3}{6}\right)^2}
+(1-\chi)\frac{x^2}{2}\sqrt{1+\frac{x^2}{9}}.
\]

低次数への吸収という原理は既知であり、R0でもknown-equivalentとして扱われている。Zhao–Yuanは、Iや低次operatorへ戻る高次項の利用を明示している。[R2][L6]

### 6.2 今回のレビューでの算術比較

R1のp=(3/4,1/4)ではχ=5/8である。既存の式を代入すると、次になる。

| x | ordinaryのB | AのB | returnのB | \(B_{\rm ret}^2/B_A^2\) |
|---|---:|---:|---:|---:|
| 1/8 | 1.0156014973 | 1.0155749708 | 1.0058441876 | 0.9809287044 |
| 1/4 | 1.0621347256 | 1.0617369860 | 1.0231978586 | 0.9287211847 |

**これは今回の数学的な代入計算であり、登録されたnative resource比較ではない。**

新しいreturned angleの合成費用は、今回取得していない。したがって、この表から「returnの総T費用がAより1.9%／7.1%低い」とは言えない。示しているのはcanonical weightの二次モーメント比である。

それでも、重要な含意がある。**Aがnormalization最小となる限定classの外には、現在のtoyでより小さいnormalizationを持つ、既知原理に基づく構成がある。** この対照を無視して「実用上最良の再配分」と主張することはできない。

### 6.3 今回導出した比較条件

以下は、本レビューで既存の二式から導いた条件である。先行文献上の新規性や独立監査済みを主張しない。

\[
u=(1,x),\qquad v=(x^2/2,x^3/6),
\]

\[
a^2=\|u\|^2,\quad b=\|v\|,\quad c=u\cdot v,\quad
A=\|u+v\|=B_A
\]

とすると、

\[
B_{\rm ret}=\|u-\chi v\|+(1-\chi)b.
\]

x>0、0≤χ≤1について、Aとの交点は

\[
\boxed{
\chi_*(x)=\frac{b(A-b)-c}{b(A-b)+c}
}
\]

であり、

\[
\boxed{B_{\rm ret}<B_A\quad\Longleftrightarrow\quad\chi>\chi_*(x)}
\]

となる。導出は付録Aに示す。

\[
\chi_*(x)=\frac{x^2}{9}-\frac{25x^4}{162}+O(x^6).
\]

x=1/8でχ*≈0.0016991270、x=1/4でχ*≈0.0063835666である。R1の5/8は、どちらの閾値からも離れている。

### 6.4 この条件が示す研究上の方向

\[
B_A=1+x^2+O(x^4),\qquad
B_{\rm ret}=1+(1-\chi)x^2+O(x^4).
\]

ordinary→Aの差がO(x⁴)である一方、ここでのreturn利用はO(χx²)の項へ効く。したがって、**確率集中度χ、step x、native costを含む、どの表現を選ぶべきかという問題**が自然に現れる。

これは「今すぐreturn-aware法へ全面転換する」根拠ではない。単純returnは既知であり、新しい方法上の貢献は別途必要である。しかし、RA-RTEの有用性を考えるとき、この比較を後回しにする理由は弱い。

---

## 7. 重要な再評価②：現在の保存表ではISを無視できない

### 7.1 既知の結果

固定された回路分布pと正の費用Cに対して、二次モーメントに基づくnet-costは

\[
\mathbb E_q[C]\;\mathbb E_q[(p/q)^2]
\ge(\mathbb E_p\sqrt C)^2
\]

であり、q∝p/√Cで達成される。[L5]

これは固定ensembleのsampling最適化であり、RA-RTEの係数matching問題そのものではない。しかし、**既存表現にもこの最適化機会を与えた後で、representationの追加価値が残るか**を調べる必要がある。

### 7.2 保存済みのA0だけでも、ゼロとは言えない

RA-D0保存表のx=1/8、A0では、内部二eventの確率が3/4、1/4である。実際に確認したT費用から、次の局所診断を計算できる。[R7]

| precision | 二eventのT費用 | 元の平均T | \((\mathbb E\sqrt T)^2/\mathbb ET\) |
|---|---|---:|---:|
| 10⁻³ | 80、156 | 99 | 0.9761890922 |
| 10⁻⁴ | 92、192 | 117 | 0.9708525058 |
| 10⁻⁶ | 140、276 | 174 | 0.9753676681 |

約2.4〜2.9%の差がある。**この値はA0内部だけの二次モーメント目的の改善余地であり、A全体・PR全体・有限confidenceでの削減率ではない。**

旧B-Sの小さいheadroomは、別の保存分布・回路costに対する診断だった。今回の値は旧診断の否定ではなく、**その陰性結果を別の分布へ一般化してはいけない**ことを示している。

### 7.3 finite-confidenceではrangeも必要

回路iの非負係数をcᵢ、proposalをπᵢ、測定結果をYᵢ∈{−1,+1}とすると、

\[
Z=\frac{c_i}{\pi_i}Y_i,
\qquad m_2=\sum_i\frac{c_i^2}{\pi_i},
\qquad L=\max_{i:c_i>0}\frac{c_i}{\pi_i}.
\]

実装誤差のaxis期待値上界をeᵢと定義すれば、weighted bias上界は

\[
b\le\sum_i c_i e_i
\]

であり、同じ実装と正確なweightを使う限りproposalには依存しない。

残りのaxis統計予算をs>0とすると、|Z−EZ|≤2Lとm₂を使うBernstein型の十分条件は

\[
\boxed{
n\ge\log(2/\alpha)
\left(\frac{2m_2}{s^2}+\frac{4L}{3s}\right)
}
\]

である。整数化、両axis、数値誤差、sampler lawの誤差は別途戻す。

**q∝p/√Cは二次モーメントnet-costに最適であって、この有限confidenceの式全体の最適proposalと自動的に一致するわけではない。** また、T最適proposalがCXや1Qにも最適とは限らない。

### 7.4 T=0のeventを安易に処理しない

pure-word／Clifford／identityのeventではT費用が0になり得る。この場合、c/√Tはそのまま使えない。

任意の小さなT費用を足して問題を変えたり、T=0なら測定も状態準備も無料と扱ったりしてはいけない。二次モーメント目的の下限、達成可能なproposal、shot cap、CX／1Q費用を分離する。既知のidentity寄与を古典的に処理するなら、対象operatorと平均の会計を明示した別構成として扱う。

---

## 8. 今回の数理的な前進：追加自由度の利益を説明する条件

以下は本レビューの導出である。旧G1の実験結果として追加しない。

### 8.1 G1の3変数表示

\[
\mu=\frac{x^2+2}{x^2+6}
\]

とし、prototype順をO0,O2,P2,P3,A0,A1,A2とする。G1で確認された倍率は

\[
\gamma=(s,b,\mu+(1-\mu)s-\mu r-b,1-r-b,1-s,1-s,r).
\]

B2はr=1−s、0≤b≤sという断面であり、B3はこの連動を外す。[R1]

各prototypeへの、固定した線形mass-priceをℓgとする。例えば後述する二次モーメント診断では、prototypeの基準係数normと、条件付き√cost期待値の積を使う。

\[
F(\gamma)=\sum_g\ell_g\gamma_g
=F_0+\alpha s+\beta r+\zeta b,
\]

\[
\begin{aligned}
F_0&=\ell_{A0}+\ell_{A1}+\mu\ell_{P2}+\ell_{P3},\\
\alpha&=\ell_{O0}-\ell_{A0}-\ell_{A1}+(1-\mu)\ell_{P2},\\
\beta&=\ell_{A2}-\mu\ell_{P2}-\ell_{P3},\\
\zeta&=\ell_{O2}-\ell_{P2}-\ell_{P3}.
\end{aligned}
\]

### 8.2 追加3頂点の改善条件

B2の三頂点の価格は、F₀を除いて

\[
\alpha+\zeta,\quad\alpha,\quad\beta.
\]

B3の追加J1/J2/J3では

\[
0,\quad\mu\zeta,\quad\alpha+\beta.
\]

したがって、固定した線形価格に対して追加自由度のstrictな価値がある条件は、

\[
\boxed{
\min\{0,\mu\zeta,\alpha+\beta\}
<\min\{\alpha+\zeta,\alpha,\beta\}
}
\]

である。

これにより、「B3は自由だから有利になるかもしれない」から、**低次数側・高次数側・pairing側の費用差がどの符号関係なら有利になるか**へ進める。

ただし、この式が直接評価しているのは線形価格であり、元のRA-D0の丸め・confidence・他資源capsを含む目的ではない。6頂点だけで元問題全体を解けるという主張はしない。

### 8.3 known ISとdegree matchingを接続すると、価格が具体化する

実装候補jの内部eventをωとし、

\[
c_{j\omega}=w_jp(\omega\mid j),\qquad
h_j=\sum_\omega p(\omega\mid j)\sqrt{C_{j\omega}}.
\]

すると、fixed-representationのIS最適化後の二次モーメントnet-costは、正のcostと理想proposalの条件下で

\[
\boxed{K(w)^2=(h^\mathsf T w)^2}
\]

になる。

ここで**√(平均cost)ではなく、平均(√cost)**が必要である。平均costだけの表では十分でない場合がある。今回R1表にword別記録があることは、その診断に役立つ。

合成biasの線形上界をdᵀw、固定したaxis予算をεとすると、二次モーメント項の設計量は

\[
\left(\frac{h^\mathsf T w}{\epsilon-d^\mathsf T w}\right)^2,
\qquad Dw=t,\quad w\ge0,\quad\epsilon-d^\mathsf T w>0.
\]

これは**診断用の線形分数計画**にできる。v=w/(ε−dᵀw)、u=1/(ε−dᵀw)と置けば、

\[
Dv=ut,\qquad\epsilon u-d^\mathsf T v=1,
\qquad v\ge0,\quad u>0,
\]

の下でhᵀvを最小化する問題になる。

この変数変換、Cauchy–Schwarz、凸最適化の原理は標準的である。本研究の新規性は、それらの使用そのものには置かない。

### 8.4 dualによる説明も可能

biasやcapsを除いた最も単純な診断では、

\[
\min_{w\ge0,Dw=t}h^\mathsf T w
\]

のdualは

\[
\max_\lambda t^\mathsf T\lambda,
\qquad D^\mathsf T\lambda\le h.
\]

λは各Taylor degreeの価格として解釈できる。あるcolumnが有効になるかを、そのcolumnが供給するdegree成分と実装価格の関係として説明できる。

単にoptimizerのwinnerを報告するのではなく、**どのdegree制約が費用を支配し、何を変更すると選択が変わるか**を研究成果にする方向である。

### 8.5 finite-confidenceとの接続と限界

Cᵢ>0のとき、特定のproposal πᵢ∝cᵢ/√Cᵢについて

\[
S=\sum_i\frac{c_i}{\sqrt{C_i}},\qquad
K=\sum_i c_i\sqrt{C_i}
\]

とすると、

\[
m_2=SK,\quad\mathbb E C=K/S,\quad
L=S\sqrt{C_{\max}}.
\]

従って、整数shotの切上げを除いた上記Bernstein十分条件の総costは

\[
\log(2/\alpha)
\left[2(K/s)^2+\frac43(K/s)\sqrt{C_{\max}}\right]
\]

で評価できる。

これはこの特定proposalの式であり、全proposalでの最適性ではない。切上げ後には1-shot cost相当の追加があり、dyadic sampler、数値residual、複数資源制約、zero-cost eventも戻す必要がある。

重要なのは、**既知ISを含む公平な診断を、必ずしも巨大な登録LP gridから始める必要はない**ことである。

---

## 9. 6頂点とprecisionを使った、小さい完全診断の可能性

### 9.1 6固定点だけではなく、pure-precision profileを考える

G1で確認された各頂点のactive prototype数は、ordinary/PTSC-K0/A/J1/J2/J3について2,3,3,4,4,3である。[R1]

各active prototypeに保存済み三precisionのいずれかを割り当てると、候補数の上限は

\[
3^2+3^3+3^3+3^4+3^4+3^3=252
\]

となる。既存二つのxなら504 profileである。新しいangleやsynthesisは必要ない。

### 9.2 完全性を主張できる限定された目的

本レビューで次の補題を得た。

> **理想degree matching、有限precision variants、他資源capsなし、非負線形価格と線形biasを用いたK/(ε−dᵀw)という診断目的では、正の統計予算を持つpure-precision vertex profileのいずれかが最小値を与える。**

理由は二段階である。

第一に、任意のgroup配分を6頂点の凸混合へ分解し、各group内のprecision shareを戻すと、pure-precision profileの凸混合で表せる。

第二に、Kとs=ε−dᵀwはいずれもその混合に関してaffineである。s>0の混合が、すべてのs>0成分より小さいK/sを持つことはできない。s≤0の成分は、非負Kを加えながら分母を減らすため、その結論を覆さない。詳細は付録Cに示す。

### 9.3 何に対して完全ではないか

この252/xという整理は、以下にはそのまま適用しない。

- 元のRA-D0の固定shot・複数資源cap付き最適化。
- Bernsteinのrange、整数shot、dyadic lawを全て含む最適化。
- 数値K3全体と保守的inner generatorの差。
- 新しいangleや新dictionary。
- arbitrary signed／cancellation-aware ensemble。
- zero-costで最適IS分布が達成されない場合の実行資源。

従って、504 profileを調べて差がなくても、B3全体の不可能性とはしない。一方、**この限定診断については、任意の細かいη gridを増やす必要がない**。構造と計算量を明示した、次の判断材料にできる。

---

## 10. 研究方針の候補比較

### 方針A：現在のcanonical B2/B3比較を、そのまま主研究にする

利点は、数理契約と実装基盤が既に準備されていること、追加自由度を隔離して比較できることである。

弱点は、固定dictionary・固定sampling・一blockの差だけでは、既知sampling・相殺を含む実用上の優位性を説明できないことである。

**判断：機構検査として残す。これ単独を最終的な研究の主張にはしない。**

### 方針B：構造を使ったresource-aware representation設計

R0の一般構成とG1の価格条件を基点に、representation、precision、samplingを分離して評価し、追加自由度が有利になる条件を示す。

求める成果は、大きいoptimizerではなく、入力情報から候補を構成・選択する手順と、その費用・保証・限界である。

**判断：現時点の第一候補。** 既存資産と接続し、具体的な数学診断が作れたことが理由である。性能や新規性が実証済みだから選ぶのではない。

### 方針C：return／cancellationを利用したPR残差表現

χに依存する低次数吸収には、normalizationのleading項を変える機構がある。PR内部の改良として検討する理由は明確である。

ただし、単純なidentity return、Pauli collection、既知のmodified Taylorを実装するだけなら新規methodではない。全word列挙を避けた生成法や、return後の再配分に固有の追加価値など、具体的な未解決点が必要になる。

**判断：最重要の強い対照、および第二候補。現在の新主線として自動採択しない。**

### 方針D：多block、高次数、別PF・別randomized法へ広げる

最終的なPR利用に接続するため、必要になる可能性は高い。しかし、現在の一blockで利益の原因と強い対照後の余地が分からないまま広げると、自由度だけが増える。

**判断：現段階の第一選択にはしない。** ただし一般次数の理論解析は、分子grid拡張とは別であり、具体的な定理・計算量改善を狙うなら先に行ってよい。

### 比較からの結論

**方針Bを暫定主線にし、方針Aを機構検査、方針Cを強い対照・次候補、方針Dを接続段階として配置する。**

これにより、既存のRA-RTEを惰性で続けることも、まだ中心仮説を判別していないのに別手法へ乗り換えることも避けられる。

---

## 11. 推奨する主RQと、論文として狙う主張

### 11.1 主RQ

> **一般involutionからなる有限Taylor平均の乱択実現について、次数ごとの係数配分を利用した構成は、同じ情報・native実装・精度条件を与えた既知対照より有利な資源点を生成できるか。その有効域と不利域を、係数構造・実装費用・合成誤差から説明できるか。**

### 11.2 副RQ

**構成の問い**：どの自由度が、既存representationの混合では表現できないか。一般次数でどの情報と計算量が必要か。

**利益の問い**：normalization、合成費用、biasによるshot増加、既知IS、return利用のどれが支配するか。

**実用性の問い**：一blockの利益が、PR wrapperと古典取得費用を戻しても残るか。既知条件で作った規則が、未使用条件へ移るか。

### 11.3 最も強い完成形

仮題：

> **Coherent randomized time evolutionにおける構造化された有限平均表現の資源設計**

主成果の組は、

\[
\boxed{
\text{構成familyと正しさ}
+\text{条件付き最適性／選択条件}
+\text{実行可能な生成手順}
+\text{強い対照後の資源的価値}
}
\]

である。

R0の定理だけでも、nativeの一点勝利だけでもなく、構成と有用性をつなぐことを目標にする。

---

## 12. 必要なbaselineと、公平な比較階層

### 12.1 既存RA-D0を改ざんしない

旧B2へJ1–J3やreturnを追加して、過去の比較条件を書き換えない。旧B0_saved、B0_ideal、B1_num、B2_num、B3_numの区別も保つ。[R5][R6]

### 12.2 比較を三層に分ける

| 層 | 比較 | 答える問い |
|---|---|---|
| 機構比較 | 旧canonical B2対B3、同一表 | 次数の連動解除自体に価値があるか |
| samplingをそろえた比較 | 既存表現＋IS対degree-local表現＋同等のIS自由度 | samplingだけで説明されない追加価値があるか |
| 実用比較 | ordinary、PTSC-K0、適用可能なreturn／CTS等との比較 | 現実に採用する理由があるか |

高次PTSCの補償targetは固定P₃とは異なる。従って、同targetの比較と、最終taskをそろえたalgorithm比較を分ける。

CTSがPauli情報を利用する場合、その情報・取得費用を別に記録する。ただし、現在の2-qubit toyでその情報を取得できる以上、「一般involutionを想定する」という理由だけで実用比較から外さない。[R3][L2][L3]

### 12.3 取得費用も成果の一部

R0のO(m)は係数生成の算術量であり、全native-cost tableの取得費用ではない。

L個のinvolutionから長さmまでのwordを全列挙すれば、一般にL^m規模の記録が問題になる。word列挙なしに条件付き費用・誤差を評価できるのか、近似評価ならどの誤差を許すのかを明示する。

既存表を無料のoracleとして与えたbenchmarkは有用だが、それだけを「大系でも使える設計アルゴリズム」と呼ばない。

---

## 13. 次にCodexへ任せる作業は、一つの研究判断パケットにする

### 13.1 目的

**全面v4実装の前に、追加自由度の価値と、比較不足の大きさを、保存値と限定した数学から判定できるようにする。**

この作業は、新しい分子実験でも、旧RA-D0 one-shotの再開でもない。新規のpost-hoc／development診断として記録する。

### 13.2 一括して行う内容

**A. 本書の数学の独立監査**

identity-return閾値、G1の線形価格条件、ISと線形分数目的の接続、pure-precision profileの完全性が、記載した仮定の下で正しいかを独立に確認する。反例があれば、その適用範囲と研究への影響を報告する。一般定理を有限点のテストだけで代用しない。

**B. 保存表からの小さい診断**

固定の二つのx、7 prototypes、3 precisionを使う。全eventの確率・phase・費用・誤差identityを読み取り専用で照合し、最大252 profile/xの範囲で、線形価格と線形分数診断を評価する。元ensembleを含む対照へ同じprecision選択を与える。

複数資源を任意の重みで混ぜない。T、CX、1Qをそれぞれの診断として扱い、他資源の変化も保存する。zero-cost、bias exhaustion、range、shot cap、数値認証の有無を区別する。

これらの値は旧RA-D0のU₃<L₂ witnessや新しいheld-out証拠に転記しない。

**C. 強い対照を含めた説明**

既知ISについて、理想二次モーメント診断と有限confidenceで実行可能なlawを区別する。return構成については、既存の数式でnormalizationと必要なeventを整理し、未取得のangle cost・errorをMISSINGとして残す。勝つと仮定して埋めない。

**D. 次に必要な最小範囲を返す**

追加自由度の改善を示す候補があるのか、既知samplingだけで説明されるのか、範囲の狭い診断では判別できないのかを分ける。必要な実装・証明・新規取得を、どの科学的問いに答えるかと対応付けて返す。

### 13.3 実行範囲の境界

今回の次作業案では、旧runのretry、新angle/synthesis、新dictionary、分子・DF入力、quantum trajectory、新registered B2/B3 LP grid、full v4 productionは行わない。

必要な読み取り、数式検算、保存値の算術処理、結果整理、単体テストは一つの作業としてCodexに任せ、テストごとにGPT承認を要求しない。

数学的target、主要評価指標、baselineの意味、独立性、研究仮説を変更する必要が生じた場合だけ、研究判断へ戻す。

### 13.4 成果物

一つのhandoffから、次を確認できる状態にする。

- 仮定別の数式判定と、反例・未証明部分。
- 元の21-column表へのprovenanceと、pure-profile診断の再現結果。
- priceの符号、J1–J3が有効／不要になる理由。
- 既知ISを考慮したときの変化と、finite-confidenceで残る未確定幅。
- return／CTS等の比較に必要な追加取得の具体的な最小集合。
- 本格実装が必要か、限定実装で足りるかを判断できる情報。

---

## 14. 次のGPT分析点：G2

G2を、単なるsource完成確認ではなく、**研究価値・実験設計・必要実装範囲の決定点**として使う。

### G2で必ず判断すること

1. **構造が実際の選択に効いているか。** J1–J3の利用理由を費用・誤差から説明できるか。
2. **強い対照を戻しても意味があるか。** degree-localの差と、precision／sampling／known returnの差を分離できるか。
3. **何がまだ未判定か。** 全classの改善余地、finite-confidence、保守的inner、outer gapのどれが不足か。
4. **必要な次実装はどれだけか。** 少数のsource・law・certificateで済むか。旧55,275 main LP規模を復活させる情報価値があるか。
5. **論文の主張は何になるか。** 方法、理論的限界、適用条件、または内部研究記録のどれとして閉じるか。

### 結果別の分岐

**追加自由度の候補が、同等sampling対照後にも残る**  
→ その候補と対照に限定したfinite-confidence／native実装の検査へ進む。必要な実装だけをまとめてCodexへ渡す。

**簡単な診断では差がないが、未判定幅が大きい**  
→ 陰性結果とはしない。原classの下界、precision混合、range／capsのどれを追加すれば判断が変わるかを評価する。

**適切な上下界から、改善しても小さいと分かる**  
→ B3の実用改善を主線として追う優先順位を下げる。R0/G1を限定理論結果として残すか、return後の別の具体的問題へ移るかを判断する。

**差が既知IS・低次数吸収だけで説明される**  
→ それを新しいRA-RTE固有の方法差としない。新しい独立差がないなら現在の主claimを縮小する。

### witness不在と非改善を分ける

B2/B3の同一query・同一classで最小値をG₂*、G₃*とすれば、包含関係からG₃*≤G₂*である。

利益を証明するには、有効なboundsで

\[
U_3<L_2
\]

を使える。

一方、残る改善幅を小さく押さえるには、

\[
0\le G_2^*-G_3^*\le U_2-L_3
\]

のような上界が必要になる。**B3の保守的inner minimumをL₃にしてはいけない。** また、これらは定義したconfidence／resource classの最適値に関する比較であり、全量子推定法の情報論的最小費用ではない。[R6]

5%・10%を過去から自動転用しない。strictな数学的改善と実用上の効果量は分け、正式な次実行では不確かさ・資源trade-offに見合う判断基準を結果前に定める。

---

## 15. その後に必要になる検証と、今は不要な検証

### 15.1 最初は「同じtarget」に対する構成の検査

P₃等の同じ有限平均を保ち、回路・phase・bias・shot・資源をそろえる。ここで新表現だけに良いangleやprecisionを与えない。

### 15.2 次に一つの実PR wrapperへ接続

一blockの利益が確認されたら、決定論部分・状態準備・測定を戻す。全体資源は

\[
G=N(C_D+C_R)
\]

であり、変更前にランダム部分が占める割合をfとすると、

\[
\frac{G_{\rm new}}{G_{\rm old}}
=\frac{N_{\rm new}}{N_{\rm old}}
\left[(1-f)+f\frac{C_{R,\rm new}}{C_{R,\rm old}}\right].
\]

ランダム回路だけの削減は、周辺費用で薄まる。一方、shotsが減れば決定論部分を繰り返す費用にも効果がある。

### 15.3 多blockでは平均と確率weightを別に伝播

独立なoccurrenceで平均を保存すること、同一sampleの再利用で平均が変わらないことは別である。

理想平均がcontractiveなら、局所実装誤差eⱼの積への伝播は

\[
\left\|\prod_j\widetilde M_j-\prod_jM_j\right\|
\le\prod_j(1+e_j)-1
\]

で扱える。しかしcanonical normalizationの積や二次モーメントの増大が消えるわけではない。[R2]

### 15.4 独立検証は、主張に必要な一軸から

固定した規則が、atom確率の集中度、basis遷移費用、atom数、未使用実装条件のどれに対して再利用できると主張するかを決める。その一軸を事前に選び、予測と結果を照合する。

既存のx・precision・H4条件で規則を作った後、それらを再評価してfresh validationとは呼ばない。

### 15.5 今すぐは不要

- 別分子・別geometryを大量に増やすこと。
- 7-stage／8次PF等を、旧BFの結果後に差を探す目的だけで追加すること。
- 既に限定適合性が分かったbackendへ、研究判断と無関係な人工LPを際限なく追加すること。
- 全RPE/QPEの化学精度資源を、有限P₃の検査の完了条件にすること。
- 新しい名前の準備段階を重ねるだけで、資源価値の問いを延期すること。

---

## 16. 論文としての着地点

### 16.1 方法論文としての目標

必要な中心成果は、次の連鎖である。

> **利用可能な入力情報 → 有限平均を保つ構成 → 誤差・資源の評価 → 強い対照を超える、説明可能な設計 → 再利用範囲**

R0の限定最適性とG1の構造は理論部分を支える。今回のprice条件やgeneral-degreeの設計が、既知汎用最適化の単なる実装を超えた利点へつながるかが次の焦点になる。

### 16.2 定量的な設計・適用研究としての成立

新しい一般アルゴリズム差が小さくても、確率集中度・native費用・合成誤差によって、既知構成と再配分構成の使い分けを定量的に説明できれば研究になり得る。

ただし、Track Aと同じH4のwinner表を増やすだけでは弱い。**こちらでは表現の構成を変更し、その変更が有効な理由・不利な理由を示す**必要がある。

### 16.3 限定理論ノート／内部研究記録

R0とG1の限定定理、同値性、過大なclaimへの反例は、正確に残す価値がある。しかし、これらだけで独立論文に十分とは現時点では断言しない。

資源上の追加価値がなければ、あるように記述しない。陰性結果を論文化する場合にも、他者が利用できる非自明な限界や条件が必要である。

### 16.4 現時点で書いてはいけない主張

- RA-RTEは既存PRより一般に優れる。
- Aは任意のfinite-Taylor LCUでnormalization最小である。
- 6頂点の存在が、新規性や資源改善を証明する。
- I0の係数生成がO(m)だから、native-cost取得もO(m)である。
- G1の8LP PASSによりproductionが保証された。
- B3のstrict witnessが未取得なので、B3に価値はない。
- R1の保存資源forecastを、化学精度の基底エネルギー推定実測へ外挿できる。

---

## 17. 最終判断と、前回の進行判断からの変更点

前のチャットでは、G1の技術PASSからv4 source実装へ進める提案が一度出され、その後、研究上の必要性を評価し切っていなかったとして保留された。[H1]

今回、その保留を単に延長するのではなく、次の根拠を追加した。

**第一に、現在の数学的自由度は線形価格で明確に説明できる。** したがって、まず構造に基づく小さい診断を行う価値がある。

**第二に、既知returnのnormalizationはR1条件でAより小さく、既知ISにも現在の保存recordで無視できない局所的な余地がある。** これらを考慮せずproductionへ進むと、比較を終えてから研究上の価値が崩れる危険がある。

**第三に、R0だけでは漸近改善の主張が弱い。** 目指すべき成果は、有限資源における再現可能な構成・選択条件・有効範囲である。

従って、現時点の推奨は次である。

\[
\boxed{
\text{RA-RTEを候補として維持}
\rightarrow
\text{保存表・構造・強い対照を含む限定診断}
\rightarrow
\text{GPT G2：研究価値と必要最小実装を決定}
}
\]

**次の担当はCodex。ただし担当するのは、上記の限定診断と数式の独立反証であって、全面v4実装ではない。**

新しい科学的な問いを伴わないコード修正・検査・記録整理は、Codexにまとめて任せる。次にGPTへ戻すのは、追加自由度に資源価値があるか、対照を戻すと価値が消えるか、または何が未判定かが分かった時点である。

---

# 付録A：identity-returnとAの交点

u=(1,x)、v=(x²/2,x³/6)、a²=||u||²、b=||v||、c=u·v、A=||u+v||と置く。

\[
B_{\rm ret}(\chi)=\sqrt{a^2-2\chi c+\chi^2b^2}+(1-\chi)b.
\]

Aとの等号を移項して平方すると、A−b>0より余分な符号解を導入せず、

\[
a^2-2\chi c+\chi^2b^2=(A-b+\chi b)^2.
\]

χ²項を消し、A²=a²+2c+b²を使うと、

\[
\chi=\frac{b(A-b)-c}{b(A-b)+c}.
\]

x>0ではuとvは正の成分を持ち、平行ではない。したがって0<χ*<1であり、Bret(χ)はこの区間でstrictに減少する。よってχ>χ*とBret<BAが同値になる。

この条件が比較するのは、二つの明示したrepresentationのnormalizationである。angle合成、word費用、Bernsteinのrange、全PR費用を比較したものではない。

本レビューでは、一般式の記号検算、x=1/8と1/4における交点代入・前後の符号、small-x展開を確認した。

# 付録B：G1価格表示の確認

\[
\begin{aligned}
F={}&s\ell_{O0}+b\ell_{O2}
+[\mu+(1-\mu)s-\mu r-b]\ell_{P2}\\
&+(1-r-b)\ell_{P3}
+(1-s)\ell_{A0}+(1-s)\ell_{A1}+r\ell_{A2}.
\end{aligned}
\]

定数・s・r・bを集めるだけで本文のF₀、α、β、ζを得る。

B2はordinary=(1,0,1)、PTSC=(1,0,0)、A=(0,1,0)の凸包である。
B3はこれにJ1=(0,0,0)、J2=(0,0,μ)、J3=(1,1,0)を加えた凸包である。

固定した線形目的は凸包の頂点のいずれかで最小になるため、本文のstrict条件を得る。これは非線形な元の資源目的へ無条件に適用する定理ではない。

# 付録C：pure-precision profileと線形分数目的

任意の理想group配分γを、G1の頂点γ^(v)の凸混合

\[
\gamma=\sum_v\theta_v\gamma^{(v)},\qquad\theta_v\ge0,\quad\sum_v\theta_v=1
\]

とする。positive groupでprecision shareπgp=γgp/γgを定義し、zero groupでは正の寄与がないことを利用する。

各頂点内で同じπgpを用い、そのprecision shareをpure choiceの積分布へ展開すれば、元の(γgp)はpure-precision vertex profileの凸混合となる。新しいangleやprototypeは作らない。

各profileの非負価格をKᵥ≥0、統計予算をsᵥ=ε−dᵀwᵥとする。任意の混合のs=Σθᵥsᵥが正であるなら、sᵥ>0の成分が少なくとも一つある。

もし全ての正のsᵥについてKᵥ/sᵥ>K/sなら、正の成分からのKはその比より大きく、sᵥ≤0の成分は非負Kを追加しながらsを増やさないため矛盾する。従って少なくとも一つの正の成分がKᵥ/sᵥ≤K/sを満たす。

この補題は本文の線形分数診断に対するものである。他資源caps、実装lawへの射影、zero-costでのIS達成性、整数shot等を含めた実資源の最適性を証明しない。

# 付録D：今回行った計算と、行っていない計算

**行ったこと**

- R0の既存formulaから、二つのx・χ=5/8でnormalizationと二次モーメント比を計算。
- 同formulaからreturn/A閾値を導出し、記号・数値自己検算。
- 保存済みA0の二event費用から、三precisionの局所IS net-cost比を計算。
- G1の倍率表示から線形価格条件を導出し、記号恒等式を確認。
- ISと線形分数目的、pure-profile補題の数学的整理。

**行っていないこと**

- 登録RA-D0 LP、B2/B3最適値、元classのdual/Farkasの新取得。
- 新しい回転angleの生成・合成、pygridsynth再実行。
- 分子／DF／Hamiltonian行列・statevector計算。
- 新しいtrajectory、回路構築・compile、量子測定。
- 旧runのretry、marker変更、repository/branch/commitの変更。
- 全文献にわたる優先性の証明、全candidate tableの独立再監査。

---

# 資料・参考文献

## ユーザー資料と固定repository evidence

**[H1]** ユーザー添付 `Rpartially_trackB.md`。2026年10月4日〜9日の議論。とくに終盤のG1結果、v4実装GOの保留、研究方針の本格レビュー依頼。添付履歴の旧URLや旧生成物は、実在する最新artifactと混同しない。

**[R1]** G1結果・監査。commit `e62407b3c51e9318f700673b6cf403d54e344c76`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e62407b3c51e9318f700673b6cf403d54e344c76/docs/tracks/algorithm_codesign/g1_one_shot_result_validation_20261009.md

**[R2]** R0 independent mathematical and semantic audit。commit `672d6bc667eaa7b9ca4979b012f1530499d701b8`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/672d6bc667eaa7b9ca4979b012f1530499d701b8/docs/tracks/algorithm_codesign/rte_reallocation_r0_independent_proof_v1.md

**[R3]** R0.5 equivalence/novelty auditおよびhandoff。commit `61dd534567fda5c7348fdc688814089eb26a3561`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_equivalence_novelty_audit_v1.md  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/61dd534567fda5c7348fdc688814089eb26a3561/docs/tracks/algorithm_codesign/rte_reallocation_r05_gpt_handoff_20261006.md

**[R4]** R1.5 saved-value attribution。commit `af3d014d0a0cfcbbd25bb544f6544652fec92942`。入力R1 commit `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af3d014d0a0cfcbbd25bb544f6544652fec92942/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md

**[R5]** RA-D0 preparation handoffとmanifest。commit `0ddf67756516e08f85fed1b987459a5e862676b7`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ddf67756516e08f85fed1b987459a5e862676b7/docs/tracks/algorithm_codesign/ra_d0_gpt_handoff_20261006.md  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ddf67756516e08f85fed1b987459a5e862676b7/artifacts/track_b_ra_d0_preparation/2026-10-06/evidence_manifest_v1.json

**[R6]** v4 mathematical audit handoff。commit `beb82427d202f479cc2ba954480d73a51941e322`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb82427d202f479cc2ba954480d73a51941e322/docs/tracks/algorithm_codesign/ra_d0_v4_gpt_handoff_20261009.md

**[R7]** RA-D0 saved candidate table。commit `0ddf67756516e08f85fed1b987459a5e862676b7`。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ddf67756516e08f85fed1b987459a5e862676b7/artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json  
本レビューで局所IS診断に使用したのは、x=1/8のA0三precision・二eventの保存recordである。

## 一次文献

**[L1]** Jakob Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**, 020332 (2026), DOI:10.1103/ynxb-p2xq. arXiv:2503.05647。公開論文、とくにRTEのAppendix Aと実装に関する箇所を確認。arXiv v2の存在も確認。  
https://link.aps.org/pdf/10.1103/ynxb-p2xq  
https://arxiv.org/abs/2503.05647

**[L2]** Pei Zeng, Jinzhao Sun, Liang Jiang, Qi Zhao, *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, PRX Quantum **6**, 010359 (2025). arXiv:2212.04566。ゼロ次specializationと高次補償targetを区別して参照。  
https://arxiv.org/html/2212.04566v2

**[L3]** Joseph Peetz, Scott E. Smart, Prineha Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information **12**, 52 (2026), DOI:10.1038/s41534-025-01168-w。CTSのPauli表現・normalization・Markov layeringを参照。  
https://www.nature.com/articles/s41534-025-01168-w

**[L4]** Bálint Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum **5**, 040352 (2024). arXiv:2402.15550v2。Section IIのprocess-matrix表示、dictionary、凸最適化を参照。  
https://arxiv.org/html/2402.15550v2

**[L5]** Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (2026)。Theorem 1、Eq. (7)–(13)、biasに関するSection III。二次モーメント目的と有限confidenceの最適性を区別して参照。  
https://arxiv.org/html/2603.13495v1

**[L6]** Qi Zhao, Xiao Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum **5**, 534 (2021). arXiv:2103.07988v2。Section 4.2の低次数operatorへの吸収を参照。本文の同label-return式はR0の明示specializationを用いる。  
https://arxiv.org/html/2103.07988v2

**[L7]** *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026)。SPRINTの構成と資源設計の既知範囲を参照。  
https://arxiv.org/html/2606.30741v1

---

**文献調査の限界**：上記の近接する一次研究と追加検索を確認したが、引用ネットワーク全体を網羅した体系的レビューではない。本書の新しい数式整理を既存文献に見つけなかったことだけで、新規性・優先性を確定していない。
