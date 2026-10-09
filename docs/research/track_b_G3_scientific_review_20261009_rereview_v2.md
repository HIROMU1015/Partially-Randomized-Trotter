# Track B G3 科学的研究レビュー・再評価 v2
## 有限J1構成、B2下界の意味、実用性、強い対照、および次の研究判断

- 作成日：2026-10-09（JST）
- 対象：Partially Randomized Trotter / Track B — RA-RTE
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- G3 branch：`track-b-g3-finite-law-diagnostic-20261009`
- G3固定結果commit：`3b0fa70b47848e72ef9a9c9e13afc3164962f7cd`
- G3 Phase A source：`381449daefc7310e22729725f69656dd651b3d20`
- G3 Phase B source：`34fdf1c21becd96a77122a9c468b4191b9dbf2ad`
- G2固定結果commit：`b260189b020ab7dfabb16bf424f49a6efff40d75`
- 対象前版：`track_b_G3_scientific_review_20261009.md`
- 依頼：前回のG3レビューを丁寧にやり直す。結論の変更を目的にしない。
- 証拠時点：上記G3まで。G4完了や新しい実験結果の存在を仮定しない。
- 最終判断：**限定継続。G4-Aの条件付き分離の独立認証、成功時のG4-Bのmatched CTS比較、G4-Cの次検証設計を推奨。全面v4・大規模計算・未使用条件の本実行は認めない。**

本書は前版の再要約ではなく、証拠・数学・解釈・研究計画を再点検した改訂レビューである。前版は保存し、本書では判断を維持した部分と、説明・条件を補正した部分を分ける。G3の原結果・分類・markerを書き換えるものではない。

---

## 1. 結論と前版からの差分

**前版の「限定継続」という結論を維持する。** 理由は、G3の有限構成が、無限重みの極限に頼らず、保存された費用・誤差・confidence会計の下で既存の有限対照集合を上回り、さらに限定されたB2全混合に対する分離命題を証明する経路が残るためである。

ただし、**「有望な条件付き構成例がある」ことと、「PR全体の実用的な新手法として採択できる」ことの間には、依然として大きな隔たりがある。** 前版はこの点を留保していたが、数値分離の説明が目立つ一方、共通回路費用、toyの特殊構造、比較class間の対応の検討が十分でなかった。

今回の再評価では次を明確にした。

| 論点 | 再評価 |
|---|---|
| G3で有限lawが構成されたか | 支持する。実sampling・量子測定済みとはしない。 |
| J1が固定poolのT・1Qで有利か | 支持する。各軸最良値と同一lawの値を区別する。 |
| B2下界の不等式 | 明示したBernstein予算方式内では妥当。情報理論的・物理的最小資源の下界ではない。 |
| x=1/4の分離の数値的余裕 | 保存G2区間を使う別の保守的導出でも残る。新規の全63 profile独立再計算をしたとはしない。 |
| 理想B2とデジタルJ1の比較 | 対応を証明せず「全B2実装に勝つ」と言わない。L1誤差を支払う丸めに対する条件付き補題を提示する。 |
| 実用性 | 選択J1はAより約8.93%多いshotを要する。共通費用に対する損益分岐を算出し、全体への移送リスクを具体化する。 |
| CTS比較 | 維持。ただし同一operator mean・位相・会計・情報モデルを固定した比較に限る。CTSの勝敗は研究全体の単純なGO/STOPではない。 |
| 論文の中心 | optimizerの完成やJ1の一点勝利ではなく、構造的自由度の条件付き価値と取得可能な情報からの設計原理を候補にする。 |

この改訂は、結論を変えるための新条件追加ではない。既存の研究目的に照らして、前版の主張範囲と判断根拠を明確にしたものである。

## 2. 確認した資料と、今回実施した作業

### 2.1 原資料

G3 handoff [R1]、Phase A保存結果 [R2]、保存比較auditの関連部分 [R3]、G3の候補生成・certificateコード [R4]、保存出力auditコード [R5]、G2の一般数学監査 [R6]、G2の保存目的値区間 [R7]、添付された前回G3レビュー [R8]を確認した。G3原報告が挙げるPhase B結果・設計も、同じ固定commitの資料として識別した。

論文については、resource-optimal IS [L1]、CTS/SCU [L2]、Sparse Probabilistic Synthesis [L3]、元PR論文の版・出版情報 [L4]を一次資料で照合した。本書は関連文献の不存在や世界初を証明する網羅的優先性調査ではない。

### 2.2 今回の作業範囲

- 前版の主張を、G3の原報告・保存値・コードと対応付けた。
- 下界を数式から再導出し、G2保存区間の有理数下端とG3保存費用を用いて再照合した。
- 対数の下界を有理Taylor級数で自己検算した。
- 固定されたA/J1の共通追加費用に対する損益分岐を算出した。
- 固定toyのPauli積を記号代数で整理した。量子行列の生成・対角化・signal計算はしていない。
- リポジトリ変更、旧runner実行、登録LP、追加合成、分子・DF計算、sampling、量子測定は行っていない。

**今回の算術再照合は、保存されたG2の最小値区間を前提にしたレビュー用の検算である。元candidate tableの全eventをコンテナへ取り込み、63 profileを別実装で一から認証し直したものではない。** 独立認証として必要な作業はG4-Aへ明記する。この点で、過去のGPT検算と今回の検算を別々の独立科学実証として数えない。

## 3. G3で何が進み、何が進んでいないか

### 3.1 固定された対象

\[
R=\frac34Q_0+\frac14Q_1,\qquad Q_0=ZI,\qquad Q_1=V^\dagger IZV,
\qquad V=e^{-i\pi XX/16}.
\]

対象は2-qubit distinct-basis、controlled finite

\[
M=P_3(-ixR)=I-ixR-\frac{x^2}{2}R^2+i\frac{x^3}{6}R^3,
\qquad x\in\{1/8,1/4\}.
\]

入力は報告上 \(|00\rangle\)、追加workspaceは1 qubit。axis精度は \(\epsilon=0.005\)、\(\alpha=1/5280\)。ここでの精度はfinite \(P_3\) に対するもの。指数関数、PF全体、QPEの最終精度・総費用ではない。[R1]

Phase Aはordinary/PTSC-K0/Aの63 pure precision profiles/xとJ1の81/x、計288 profiles、6,912有限law。Phase Bは固定12 synthesis keys・12 event条件を取得し、returnの18 profiles・162 lawsを評価した。各phaseは一回、retry 0。これらは同じdevelopment入力に対する診断であり、独立held-outではない。[R1–R3]

### 3.2 有限lawの意味

G3では、有限係数 \(a_i\)、正のdyadic確率

\[
q_i=k_i/2^{60}>0,\quad\sum_iq_i=1,
\]

と有理補正重み \(w_i=a_i/q_i\) を持つ。

\[
\sum_iq_iw_iU_i=\sum_i a_iU_i
\]

の取消しはexactである。理想係数とのずれと合成誤差はbiasへ課金する。したがって、「理想 \(P_3\) を無誤差で物理実装した」ではなく、「有限の法を明示し、固定誤差上界から使用可能な予算を算出した」である。[R1,R4]

G2で未達成だったzero-Tのinfimumそのものを達成したのではない。**infimumとは別の、正確に指定された有限点でも差が残った**ことがG3の前進である。

### 3.3 証拠の独立性

候補生成関数とcertificate関数は分かれており、certificateは出力event・q・weightから量を再計算している。一方、保存出力auditは同じ `g3_finite_law.py` のcertificate関数を呼び、G2の区間算術helperも共有している。[R4,R5]

これは実用的な内部整合性・再計算チェックとして有用であるが、**数学・算術・sourceを完全に独立に実装した外部再現ではない。** 27 focused testsも、この限定実装の検査であり、新規性・一般性・実用性の証拠へ足し合わせない。今回のコード確認では、示された主要式に直ちに結果を覆す不整合は見つからなかったが、全source・全法・全合成列の再実行監査を完了したとはしない。

## 4. 数値結果を正確に読み直す

### 4.1 固定poolの座標別最小

単位は共通のfinite-confidence方式で算出した期待総ゲート数。別々の座標最小であって、単一lawが全てを達成した表ではない。[R1–R3]

| x | 座標 | ordinary/PTSC/A pool最小 | J1 pool最小 | return pool最小 |
|---|---|---:|---:|---:|
| 1/8 | T | 179,987,284.56 | 179,160,991.73 | 187,808,511.12 |
| 1/8 | CX | 4,281,301.64 | 4,310,150.09 | 4,245,968.98 |
| 1/8 | 1Q | 465,290,176.32 | 463,389,609.27 | 481,297,026.22 |
| 1/4 | T | 177,548,958.29 | 174,720,368.17 | 189,825,055.94 |
| 1/4 | CX | 4,610,549.46 | 4,723,948.73 | 4,448,986.39 |
| 1/4 | 1Q | 468,173,605.23 | 461,712,911.04 | 495,013,282.20 |

主poolはA側864・return側54のaxis-winnerを共通比較したもの。同一optimized-axisだけの別集計でも差の符号が一致する。一方、G3の6,912法全てに対する一般的な全資源Pareto完全性や、任意B2混合・任意sampling lawの最適性は主張されていない。

### 4.2 同一lawを見ると何が重要か

x=1/4に絞ると：

| 選択されたlaw | T | CX | 1Q |
|---|---:|---:|---:|
| Aの1Q選択 | 177,548,958.29 | 5,718,904.22 | 468,173,605.23 |
| J1のT選択 | 174,720,368.17 | 6,662,825.87 | 483,946,285.80 |
| J1の1Q選択 | 174,820,322.72 | 5,661,819.95 | 461,712,911.04 |

T最小のJ1は他資源を増加させる。対してJ1の1Q選択lawは、選択されたA点の全3座標を改善する。前版が分離候補に使ったのは後者である。

ただし「1Qを選んだので全B2をPareto支配する」とは言えない。CXでは別のPTSCやreturnが安い。Tと1Qも、Tゲートが1Q countの一部である以上、独立した二つの実験的成功ではない。二座標での同一lawの改善として記述する。

1Qで選ばれた点を理論的な存在例に利用することは可能だが、**事前登録されたT-primaryの独立実験として遡及的に扱わない。** G3はdevelopment/post-hocという区分を維持する。

## 5. J1はアルゴリズムとして何を変えているのか

G1/G2の構成を用いると、J1はAと比べて低次数側A0・A1を保持し、高次数側A2を

\[
\mu P2+P3,\qquad \mu=\frac{x^2+2}{x^2+6}
\]

へ置き換える。[R6,R9]

より具体的にはA2の寄与

\[
a_2F_2+b_2F_3,
\qquad a_2=\mu x^2/2,\quad b_2=x^3/6,
\quad F_k=(-i)^kR^k
\]

を、一つのrotationを伴うeventとしてまとめず、二つのpure-word側の寄与として実現する。

そのためnormalizationには

\[
\Delta B=a_2+b_2-\sqrt{a_2^2+b_2^2}>0
\]

の増加が生じる一方、高次数側のrotation・basis・合成誤差の負担を軽くできる可能性がある。G2のprice条件はこの交換を表している。ただし実際の総資源は、bias予算・sampling・range・shotsも含めて決まる。

**研究上の中身は、単なる「より良い確率」ではなく、同じ有限平均を保って、どの次数でrotationへのまとめ方を使うかを変えることにある。** ただし、この説明だけで世界初とは言わない。構成のどの部分が既知のEuler化・PTSC・一般LCUの直接的な選択に含まれ、どこに追加の定理・生成法があるかは文献比較を要する。

## 6. 前版のB2下界を、最初から点検する

### 6.1 比較classを明示する

本節の \(\mathcal B_{2,\mathrm{ideal}}\) は、ordinary/PTSC-K0/Aの非負凸混合、保存済み三precisionへの配分、固定された内部label lawからなる**理想係数class**である。固定dictionaryの下でformal degree matchingを正確に満たす。各正係数eventへの抽出確率qには、任意のfull-support分布を許す。

これに以下を勝手に含めない。

- 旧RA-D0の許容幅付き数値class全体。
- Pauli相殺、新angle、新dictionary、別のcoherent synthesis。
- 決定論的な既知寄与の古典除去、別estimator、stratification、adaptive measurement。
- 真の状態依存分散、より鋭いerror analysis、別confidence式を使う全ての方法。
- Re/Imごとに別の自由度を持つ、未定義の測定契約。

これらが無価値という意味ではなく、今回の下界が証明している範囲ではない。

### 6.2 十分shot条件から下界を出してよいか

\[
\ell=\log(10560),\quad
s=\epsilon-b(c)>0,\quad b(c)=\sum_i c_i d_i,
\quad m_2=\sum_i c_i^2/q_i,
\quad L=\max_i c_i/q_i.
\]

G3が採用したbudget policyは

\[
n(c,q)=\left\lceil\ell\left(\frac{2m_2}{s^2}+\frac{4L}{3s}\right)\right\rceil.
\]

**このpolicyの出力を比較する限り**、

\[
n(c,q)\ge2\ell m_2/s^2
\]

は正しい。range項と切上げを落とすことは、下から評価する方向である。

しかし、「Bernstein条件は十分条件だから、任意の正しい測定もそのn以上が必要」と言ったら誤りである。物理的最小shot数はこの式未満でもよい。したがって、本節の下界は**固定された予算設計方式の費用下界**であり、情報理論的下界ではない。この区別を前版より強く強調する。

### 6.3 抽出確率を最適化しなくてもよい理由

Cauchy–Schwarzより、zero-cost eventを含めても

\[
\left(\sum_iq_iC_i\right)\left(\sum_i\frac{c_i^2}{q_i}\right)
\ge\left(\sum_i c_i\sqrt{C_i}\right)^2.
\]

したがって

\[
G_C(c,q)=2n(c,q)\sum_iq_iC_i
\ge4\ell\left(\frac{K_C(c)}{s(c)}\right)^2,
\qquad K_C(c)=\sum_i c_i\sqrt{C_i}.
\tag{1}
\]

最適ISがzero-costのため達成されなくても、不等式は有効である。下界の達成性は、分離証明の必要条件ではない。

このISのnet-cost原理は既知である [L1]。本研究で狙う差は、固定representationに対してISを使ったことではなく、構造的に許すrepresentationを広げた時の追加価値である。

### 6.4 63 profileへの還元は、何を完全にするか

B2の任意の係数・precision混合は63 pure profilesの凸混合で書ける。各profile vに対して \(K_v\ge0\)、\(s_v=\epsilon-b_v\) とする。\(s_v>0\) のprofileの比の最小値をrとすると、正のsでは \(K_v\ge r s_v\)、非正のsでも \(K_v\ge0\ge r s_v\) である。

従って任意の凸混合の \(s>0\) について

\[
K\ge r s.
\]

これがG2のpure-profile補題の核心である。[R6]

**完全なのは式(1)の緩和目的の最小化であり、G3の整数shot・rangeを含む目的の最適化そのものではない。** ただし、B2の有効な下界がJ1の実行可能な上側費用を超えれば、B2の最適値を厳密に求めなくても分離はできる。

### 6.5 1Q readoutを落とさない

G3では

\[
G_{1Q}=n(2\mathbb E_qC_{1Q}+5)
=2n\mathbb E_q(C_{1Q}+5/2).
\]

従って最も直接的には、\(\sqrt{C_{1Q}+5/2}\) をpriceとして式(1)と63-profile還元を適用する。前版の約466,052,534という1Q下界は、この計算を用いていた。

今回の再レビューでは、別の保守的な導出でも分離を確認する。三角不等式により

\[
\left(\sum_i c_i\sqrt{C_i+h}\right)^2
\ge\left(\sum_i c_i\sqrt{C_i}\right)^2+h\left(\sum_i c_i\right)^2.
\tag{2}
\]

B2ではformal identity-degree係数が1であり、非負係数と各normalized columnの成分が高々1であることから \(B=\sum c_i\ge1\)。また \(s\le\epsilon\) である。G2のnative-1Q目的の最小値を \(\Phi^*_{1Q}\) とすると

\[
G_{1Q}\ge4\ell\left(\Phi^*_{1Q}+\frac{5/2}{\epsilon^2}\right)
=4\ell(\Phi^*_{1Q}+100000).
\tag{3}
\]

式(3)は直接の63-profile再計算より弱いが、新しいevent別価格の列挙をしなくても、保存されたnative-1Q最小値区間を利用できる。

### 6.6 保存された厳密区間による再照合

x=1/4におけるG2のB2最小値は、表示値で

\[
\Phi_T^*\simeq4,769,242.109228741,
\qquad \Phi_{1Q}^*\simeq12,438,508.97212956.
\]

今回の算術には表示値ではなく、保存JSONの有理数下端を使った。[R7]

\(\ell>37/4=9.25\) を有理Taylor上界で検算すると、

\[
G_T(B2)\ge37\Phi_{T,\mathrm{lo}}^*,
\]

\[
G_{1Q}(B2)\ge37(\Phi_{1Q,\mathrm{lo}}^*+100000).
\]

従って、さらに整数部分まで弱めても次の比較が得られる。

| x=1/4 | 今回の保守的B2下界 | 同一J1 finite lawの保存費用 | 正の差（概算） |
|---|---:|---:|---:|
| T | 176,461,958以上 | 174,820,322.7203 | 1,641,635以上 |
| 1Q | 463,924,831以上 | 461,712,911.0410 | 2,211,919以上 |

J1の費用は同一law

`1/4/J1/P2=1e-4,P3=1e-3,A0=1e-3,A1=1e-3/1Q/zero_mass=0;mix=1`

のもの。exact値は

\[
G_T=\frac{6298565922079926703836975}{36028797018963968},
\qquad
G_{1Q}=\frac{33269921505865913553690245}{72057594037927936}.
\]

前版の下界176,744,842／466,052,534より今回の下界は弱い。**前版の数値を誤りとして訂正したのではなく、別の保守的経路でも差が残ることを確認した。** 全63 profileの基礎区間の正しさはG2の保存記録を前提とする。G4-Aではその前提自体を別実装で監査する。

### 6.7 x=1/8について

前版の下界とG3有限点では、同じ方法による分離は確認できていない。今回x=1/8を新しい最適化で救済してはいない。

「この下界では分離できない」は「J1が負ける」と同義ではない。一方、x=1/4の分離候補をx=1/8、任意x、独立入力へ一般化してもいけない。

## 7. 理想B2とデジタルJ1の公平性

### 7.1 前版に残っていた論点

G3 J1は理想係数そのものではなく有限係数 \(\widetilde c\) を使う。一方、上のB2補題は理想係数 \(c\) を使う。J1側の係数誤差が非常に小さいことは、直観上は分離を支持するが、**小さいから自動的にclassが同一になるわけではない。**

ここは、次の二つの主張を分ける。

1. 理想B2のbudget-policy費用に対して、誤差を支払った一つのデジタルJ1費用が下回る、という比較。
2. B2にも同様の数値実現を認めた、対称な実装class間での分離。

論文で2を主張するなら、対応の証明が必要である。

### 7.2 本レビューでの条件付き補題：L1誤差を支払うデジタル化

以下は今回の数学的補足。新しい登録実験・旧K2全体の証明ではない。

ある理想classで

\[
h^Tc\ge r(\epsilon-d^Tc),\qquad h_i=\sqrt{C_i}\quad\text{又は}\quad\sqrt{C_{1Q,i}+5/2}
\]

が全ての理想係数cについて成立しているとする。\(s\le0\) のcでも右辺は非正なので、この不等式を成立させられる。

同じevent label集合上のデジタル係数 \(\widetilde c\ge0\) について、

\[
e\ge\|\widetilde c-c\|_1,
\qquad \widetilde s=\epsilon-e-d^T\widetilde c>0
\]

とする。さらに

\[
\max_i(h_i+r d_i)\le r
\tag{4}
\]

を満たすなら、

\[
\begin{aligned}
h^T\widetilde c-r\widetilde s
&=[h^Tc-r(\epsilon-d^Tc)]
 +(h+rd)^T(\widetilde c-c)+re\\
&\ge-\max_i(h_i+rd_i)\|\widetilde c-c\|_1+re\ge0.
\end{aligned}
\]

従って同じ比の下界rが、L1係数誤差を正しく予算へ課金するデジタル化にも延長される。

この補題は、単に「丸め幅が小さい」と言うより強く、**丸めによる費用減を、支払う誤差予算が上回る条件**を明示する。源泉は線形不等式とL1双対性であり、その一般原理を独立新規性とはしない。

### 7.3 何を未確認として残すか

式(4)を全source eventで認証したわけではない。旧数値classの全点が、ある理想B2からのL1係数摂動として、budgetに課金された同じeで表せることも未証明である。

G3のような「理想profileを丸め、係数L1誤差を支払う」実装には接続しやすい。一方、degree residualだけを許す広い数値classでは、近い理想係数とのL1対応が別途必要となる。**この補題を使って、未確認の旧K2/K3全体を証明済みにしない。**

G4-Aの目標は、必要なこの対応を閉じるか、閉じられたclassを厳密に限定して定理を記録することである。巨大な旧LPを再開することではない。

## 8. 実用的な意味：小さい差をどう評価するか

### 8.1 「1%だから無意味」とも「厳密に正だから重要」とも言わない

理論的な反例・分離例は、小さい効果でも意味を持つ。今回、既存構成を混ぜるだけでは表現できない自由度が、同じbudget policyの下で必要になり得ると示せれば、構造的な知見である。

一方、実用手法の主要貢献には、少なくとも対象taskで意味のある効果、費用取得可能性、対照後の価値、再利用可能な範囲が必要になる。数値の符号が丸め誤差より大きいことは、hardware・compiler・合成errorモデルを変えても残る頑健性とは異なる。

### 8.2 G3のJ1はshotを増やして一回路を安くしている

保存されたx=1/4、A/J1の1Q選択lawでは

\[
n_A=1,022,850,\qquad n_J=1,114,220.
\]

J1のshot数は約8.93%多い。[R3]

同一精度の総T・1Qが低くなるのは、このshot増を一回路当たりの費用低下が上回るからである。したがって、周囲の共通回路費用に対して敏感になる。

### 8.3 共通追加費用の損益分岐

**固定された二つのlawとshot予測をそのまま使い、追加費用が誤差を増やさない**という感度解析を考える。一回の測定ごとに共通費用h_Cを追加すると

\[
G_A(h_C)=G_A(0)+2n_Ah_C,
\quad
G_J(h_C)=G_J(0)+2n_Jh_C.
\]

J1の利益が残る条件は

\[
h_C<\frac{G_A(0)-G_J(0)}{2(n_J-n_A)}.
\]

G3のexact保存値から算出すると：

| 資源 | 共通追加費用の損益分岐／shot |
|---|---:|
| T | 約14.9318 |
| 1Q | 約35.3546 |
| CX | 約0.31238 |

例えばこの固定law比較では、一shot当たり15 T程度の共通費用を戻すとTの優位は消える。

**これはPR全体でJ1が負けるという新実験ではない。** 共通費用を入れてsampling・precision・構成を再最適化すれば結果は変わり得るし、複数blockではmoment・biasの合成も変わる。しかし、「一blockで1.5%得したのでPR全体でも同程度得する」とは言えないことを具体的に示している。

### 8.4 今後の情報価値

次の独立条件を設計する際は、xを少し増やすだけよりも、**共通回路費用を戻した後に採用すべき表現が変わるか**を問う方が、実用性に直接答える可能性がある。

これは新しい主指標や費用の恣意的加重和を追加する指示ではない。T、CX、1Qを分けたまま、実際のtaskで避けられない費用を何に含めるかを科学的に定義する課題である。

## 9. 固定toyの特殊性とCTS比較の意味

### 9.1 今回の記号的整理

G3が固定したQ0、Q1から、\(c=\cos(\pi/8)\)、\(s=\sin(\pi/8)\) とすると

\[
Q_1=c\,IZ+s\,XY,
\qquad R=\frac34ZI+\frac c4IZ+\frac s4XY.
\]

Pauli積から

\[
\boxed{R^2=\frac58I+\frac{3c}{8}ZZ}
\]

を得る。さらに

\[
R^3=\frac{15+3c^2}{32}ZI+\frac{7c}{16}IZ+\frac{5s}{32}XY+\frac{3cs}{32}YX.
\]

従ってfull operator \(P_3(-ixR)\) は、\(I,ZZ,ZI,IZ,XY,YX\) の6つのPauli成分で記述できる。これはG3の既存targetからの記号的推論であり、CTSの資源計算結果ではない。式の詳細と自己検算は付録Bに示す。

### 9.2 何を意味し、何を意味しないか

「Q0とQ1のbasisが違う」ことは、toyのPauli collectionが大きい、又は不可能であることを意味しない。この固定toyではfull operatorのPauli記述は小さい。

また、\(ZZ\) はRと可換し、入力 \(|00\rangle\) はeven-parity側にある。その部分空間だけで問題を再定義するなら、さらに簡約できる。ただしそれはfull operator meanを保つ現在の比較と別のtarget/access条件になり得る。**J1にはfull operator実現を要求し、対照だけ状態限定の古典解で置き換える比較はしない。**

小さいtoyでアルゴリズムの構成を検証すること自体は妥当である。一方、このtoyでの結果を、一般involutionの大規模問題におけるI0アクセスの実用的優位の実証と呼べない。

### 9.3 CTSをどう比較するか

CTSはTaylor演算子をPauli basisへ展開し、実係数Pauliと虚係数から作るrotationを使う具体的な構成を持つ [L2]。従って「channelの論文だから今回と無関係」とは言えない。一方、公開論文の全設定をそのまま同じcoherent first momentのcostへ転記してもいけない。

G4-Bでは次を固定する必要がある。

- 同じfull \(P_3(-ixR)\) のfirst operator momentを目標にする。
- unitaryのglobal phaseがcontrol後にはrelative phaseになる点を保持する。
- 論文のどのCTS構成を有限化したかを明記し、独自に最適化した別methodへ黙って変更しない。
- Pauli係数取得、precision、native費用、合成誤差、proposal、整数shot、readoutを同じ比較会計へ接続する。
- I0/I1の情報差を表示する。ただし、このtoyではそのPauli情報が取得可能であることを認める。

CTSの方が安ければ、今回のtoyでJ1を採用する実用的理由は弱くなる。しかしB2限定classの分離定理は消えない。逆にJ1がCTSに勝っても、全既知法・任意LCUへの優位性や大規模移送は証明されない。

## 10. 新規性をもう一度評価する

### 10.1 既知の原理と、現在残る候補差

| 論点 | 評価 |
|---|---|
| 費用と二次モーメントを同時に扱うIS | 既知 [L1]。新規性を置かない。 |
| 辞書に対する平均保存・凸最適化 | 一般原理は既知 [L3]。演算子とchannelの違いを明記する。 |
| Taylor/LCUを構造的に整理する | CTS、PR、PTSC等に既知要素がある [L2,L4,R9]。 |
| 固定非負adjacent-degree classの構成・限定最適性 | 研究資産として保持。先行研究との差は具体的命題で示す。 |
| 既存3構成の全混合に対する有限lawの条件付き分離 | G4-Aで成立させる価値がある候補。既知不等式を使うこと自体は新規性ではない。 |
| 同じ規則が未使用の入力・実装・全体taskでも有効 | 未検証。 |
| 全般的に最良の新PRアルゴリズム | 現証拠では支持しない。 |

IS論文 [L1] は2026年3月13日のv1、Sparse PS [L3] は2024年の出版論文、CTS [L2] は2026年2月19日の出版本文を照合した。元PR [L4] はarXiv v2（2026年7月10日改訂）、PRX Quantum 7, 020332 (2026) が確認できる。版が違う文献を混ぜて「最新まで新規性確認済み」とは言わない。

### 10.2 小さい分離例だけでは弱い理由

一つの固定dictionaryで、新しい頂点が既存面の外にあり、適当な線形価格なら得になる、という幾何学的事実だけでは独立methodの主張として弱い。

研究的な差は、**その価格が実装可能な回路と妥当な誤差会計から生じ、必要情報を現実に取得でき、どの構成を使うべきかを事前に判断できること**に置くべきである。

G3は、この連鎖のうち「有限の回路tableとsampling lawに結び付ける」部分を前進させた。残る課題は、強い対照、条件の頑健性、取得費用、最終taskへの接続である。

### 10.3 I0を研究の中心にする場合の条件

一般involutionを扱うことを特徴として残すなら、単に「Pauli collectionを禁止したからJ1が勝つ」では不十分である。

何が入力として与えられるのか、controlled-Qや任意角rotationはどう実装するか、conditional cost/errorは列挙なしにどう取得するか、Pauli記述を作る費用との関係は何かを、具体的なaccess modelで定義する必要がある。R0の係数生成O(m)は、全native table取得O(m)ではない。[R9]

## 11. 研究として目指す着地点

### 11.1 現時点の第一候補

**構造化されたfinite-mean representationの、条件付き設計原理と有限実装例を示す研究。**

主張候補は次である。

> 一般involutionから作る限定された有限Taylor表現について、完成済み既存ensembleの混合では得られないdegree-local構成を与える。固定した実装・誤差・confidenceモデルでは、その追加自由度が実行可能な資源改善を生む具体例と条件を示す。必要情報と適用限界を明記する。

ただし、これを「独立論文として十分」と現時点で確定しない。理論の一般性と具体的なmethod delta、新規性の照合が必要である。

### 11.2 実用方法論文へ進むために必要なこと

小さいtoyの勝者を増やすことではなく、強い対照後の価値、共通wrapper費用を含む総資源、古典取得費用、未使用条件への移送のいずれが主要claimを支えるかを明確にする。

新しい分子系を大量に計算することが自動的に必要になるわけではない。例えば、少数の構造的に異なる入力と、結果前に固定した選択規則で機構を検証する方が、単なるgeometry sweepより情報価値が高い場合がある。

### 11.3 縮小する場合の着地点

CTSやcommon-overheadを戻すと実用的な差が消える一方、構成・限定最適性・分離例が正しければ、理論・mechanism noteとして整理する余地はある。ただし、これを独立論文に十分と約束しない。反証結果を正確に残し、当初のPR資源改善という目的にどこまで答えたかを説明する。

## 12. 代替案の比較と今回の採否

| 案 | 判断 | 根拠 |
|---|---|---|
| 同じG3有限proposal gridを細かくする | 優先しない | poolの局所最良を磨くより、B2下界と強い対照の方が研究判断へ直結する。 |
| 全面v4／旧巨大LP gridを再開 | NO-GO | 今回の限定分離にはもっと小さい証明経路がある。 |
| G4-Aで下界と数値実装の対応を独立認証 | 最優先GO | 正誤を限定した作業で判別でき、全混合未評価という弱点を埋める。 |
| A成功時のmatched CTS | 限定GO | 同toyの実用上の主張に不可欠な対照が未評価。 |
| 新x・多数分子へ直行 | NO-GO | 何を一般化するか、共通費用で利益が残るかが未確定。 |
| 直ちにreturn-aware新methodへ転換 | 保留 | returnのCX最良だけでは新しいmethod deltaがない。 |
| 直ちにTrack B全体を終了 | 採用しない | 有限分離候補を閉じる情報価値が残る。 |

**同じ証拠で結論を反転させる理由は見つからなかった。** 一方、前版の数値分離だけを根拠に研究上の成功へ昇格させることもしない。

## 13. 次の担当とG4の作業範囲

### 13.1 担当

この再レビューで科学的方針を判断した。**次の実施担当はCodex。** ただし、G3までの旧STOPを再利用して再開するのではなく、G4として固定された範囲の新しい作業を行う。GPTは作業終了後の科学的解釈・採否を担当する。

### 13.2 G4-A：条件付きB2分離の独立認証

目的は「前回GPTの計算が合っているか」だけではなく、**どのclassについて何が証明できるかを閉じること**。

- G2のpure-profile補題、63 profileへの還元、保存区間、G3同一lawの費用と誤差を、別実装で再構成する。
- 1Q readout、Bernstein policy、zero-cost、整数shot、抽出lawの範囲を明記する。
- 固定policy内の下界と物理的最小資源を区別する。
- 理想B2とデジタル係数の対応を、必要なら本書§7の補題などで認証する。成立した範囲だけを報告する。
- 旧許容幅付きK2/K3全体への拡張を、無理に目標にしない。適用範囲が閉じなければ不足を明示する。
- source修正・テスト・証明書形式などの技術事項はCodexの裁量。科学的なclass・誤差会計・主要比較を変更する必要が出たらGPTへ戻す。

### 13.3 G4-B：A成功時のmatched CTS比較

Aが成功したら、前版と同じくG4-Bへ進む。ただし、合成前に具体的なCTSの有限specialization、target、phase、必要な取得key、対照側のprecision/proposal、計算上限を固定する。

ここで行うのは、同じ既知toyの強い対照比較であり、CTSを新しく設計する無制限な探索ではない。channel同値だけでcoherent meanを一致したことにしない。sourceから同targetの構成が閉じない場合は、勝手に異なる比較へ置き換えず、問題点を返す。

G3のreturn比較は、指定されたreturn構成の比較として保持する。CTS取得のついでに別angle grid、別precision、別分子を追加しない。

### 13.4 G4-C：次の独立検証の設計のみ

A/Bの結果を踏まえ、次に検証すべき問いを具体化する。候補は、共通wrapper費用を戻した時の構成選択、異なるp・basis構造への選択規則の移送、合成器の離散費用依存、取得費用の実行可能性など。

全部を一度に追加するのではなく、**どの検証が現在の主claimを最もよく判別するか**を比較して設計を返す。実行は次のGPT判断後。

Tの位置付けを結果後に1Qへ置き換えない。1Qは補助的・探索的な座標として保持する。新しい成功率や5%/10%といった閾値を過去の結果へ遡及適用しない。将来のmaterialityは、対象task・共通費用・許されるtrade-offから結果前に定義する。

### 13.5 このG4に入れないもの

旧runのretry、元resultの再分類、全面v4、旧数万LP、分子・DFへの移送、追加の独立条件本実行、実量子sampling、広い辞書探索は含めない。技術作業ごとの細かなGPT承認を挟む必要はないが、重要な科学的意味論の変更は別である。

## 14. 次にGPTへ戻す結果と、研究判断の分岐

| G4結果 | GPTで判断すること |
|---|---|
| 分離が正しく、デジタル化対応も閉じる | 固定class内の成果を確定し、強い対照後の意味を評価する。 |
| 前提又は証明が成立しない | G3の有限pool結果と、失敗した全class claimを分ける。修正の情報価値を判断する。 |
| CTSがJ1より安い | 同toyの実用的優位claimを縮小。I0条件付き構成の研究価値は別途判断する。 |
| CTS後にもJ1の価値が残る | まだ一般GOではない。common-overhead、入力構造、取得費用、独立検証の具体的価値を審査する。 |
| 技術的に未判定 | 改善余地と証明不足の大きさを区別し、同種の最終pilotを自動で繰り返さない。 |

G4後のレビューは、数値的な勝敗だけでなく、「どの学術的貢献を完成させるのか」「追加計算の価値は何か」を判断する。理論的成果として区切ることも、より一般的な構成法へ進むことも選択肢に残す。

## 15. 現時点で書けるclaim／書けないclaim

### 15.1 現G3で支持される記述

> 固定された2-qubit controlled finite P3、保存されたnative実装と共通のBernstein予算方式において、有限dyadic samplingと有理補正重みを持つJ1構成は、評価した既存構成・returnの有限poolに対して、T/1Qの非支配な資源点を与えた。CXでは指定return構成が最良であった。

### 15.2 G4-A後に成立させたい、より強い記述

> 固定dictionary、内部label law、誤差会計、budget policyで定義したclassにおいて、x=1/4の一つの有限J1 lawのT・1Q費用が、既存3構成の任意混合と精度配分からなるB2 classに対する有効な下界を同時に下回る。デジタル化の適用範囲は別途定理の仮定に示す。

### 15.3 現時点で書かない記述

- 任意の物理的B2実装より必要shot数が少ない。
- 全ての合成器・hardware、全資源でJ1が最適。
- CTSや任意LCUより一般に強い。
- x=1/8の全B2分離も確立した。
- PR＋QPE、分子・DFの総費用改善が実証された。
- Tと1Qで二つの独立再現が得られた。
- 最新の全先行研究に対する独立新規性・世界初が確定した。

## 16. 総括

G3は技術的PASSだけではなく、有限の構成を示した科学的前進である。前版のB2分離の考え方も、**固定した予算設計方式に限定すれば**維持できる。保存されたG2最小値区間を使う別の保守的計算でも、x=1/4のT・1Q分離の数値的余裕は残った。

一方、実用性はまだ弱い。選択J1はshot数を増やして回路を安くする構成であり、共通費用への感度が高い。toyには小さいPauli代数があり、情報制約による実用優位をそのまま代表しない。新規性も、既知ISや凸最適化そのものには置けない。

したがって、**限定継続・G4-A→条件付きG4-B→GPT研究判断**という前版の方向を維持する。今回の改訂で加えたのは、進める理由と進め過ぎない理由の両方を、数学・費用・情報モデルの具体的な条件として残すことである。

---

## 付録A：今回の算術再照合を再現する最小コード

以下は保存されたG2最小値区間を入力にした再照合である。63 profilesの最小値を一から証明するコードではない。外部依存は不要。

```python
from fractions import Fraction as F

# G2 result_v1.json: x=1/4, same_IS_original_vertices のvalue.lo
phi_t = F(
    '476924210922874165616417536529682957981101682374891794563681693343/'
    '100000000000000000000000000000000000000000000000000000000000'
)
phi_q = F(
    '6219254486064779156222365665009782238108976342800045367708764200761/'
    '500000000000000000000000000000000000000000000000000000000000'
)

# G3 phase_A_result: x=1/4, axes.1Q.J1_totals。同一lawのT/1Q。
j_t = F('6298565922079926703836975/36028797018963968')
j_q = F('33269921505865913553690245/72057594037927936')

# exp(37/4) < 10560 の有理上界。k=0..60、tailは幾何級数で上から囲む。
z = F(37, 4)
term = F(1)
partial = term
for k in range(1, 61):
    term *= z / k
    partial += term
next_term = term * z / 61
exp_upper = partial + next_term / (1 - z / 62)
assert exp_upper < 10560

lb_t = 4 * z * phi_t
lb_q = 4 * z * (phi_q + F(5, 2) / F(1, 200)**2)
assert lb_t > j_t
assert lb_q > j_q

print(lb_t.numerator // lb_t.denominator)  # 176461958
print(lb_q.numerator // lb_q.denominator)  # 463924831
print(float(lb_t - j_t))  # 約1641635.3212（表示用）
print(float(lb_q - j_q))  # 約2211920.9277（表示用）
```

浮動小数点は最後の表示だけに使用する。対数下界、分離の符号は有理数比較で確認する。基礎G2区間とclass還元の正しさは別途G4-Aで認証する。

## 付録B：固定toyのfull Pauli係数

\(c=\cos(\pi/8)\)、\(s=\sin(\pi/8)\) とする。G3の符号規約で、

\[
\begin{aligned}
P_3(-ixR)= {}&\left(1-\frac{5x^2}{16}\right)I
-\frac{3cx^2}{16}ZZ\\
&+i\frac{x[x^2(c^2+5)-48]}{64}ZI
+i\frac{cx(7x^2-24)}{96}IZ\\
&+i\frac{sx(5x^2-48)}{192}XY
+i\frac{csx^3}{64}YX.
\end{aligned}
\]

自己検算では、Pauli文字列の積と \(s^2+c^2=1\) の記号的簡約だけを使った。4×4量子行列・Hamiltonian signal・CTS回路・新規合成は生成していない。

この式を、既知CTS又は別のsame-target構成へ変換した場合のnormalization・native費用は未評価である。6成分だから必ず安いとは結論しない。

## 付録C：共通費用の再現データ

x=1/4の同一A/J1 1Q選択lawについて、

- A shots/axis：1,022,850
- J1 shots/axis：1,114,220
- \(2(n_J-n_A)=182,740\)

保存値からの費用差A−J1：

- T：約2,728,635.5741
- CX：約57,084.26549
- 1Q：約6,460,694.18398

上記を182,740で割ると、§8の損益分岐を得る。これはfixed-law sensitivityであり、新しい最適化のoutcomeではない。

## 参考資料・出典

### Repository／添付資料

- [R1] [G3 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_handoff_20261009.md)
- [R2] [G3 Phase A result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/phase_A_result.json)
- [R3] [G3 saved comparison audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/saved_comparison_audit.json)
- [R4] [G3 finite-law source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/scripts/tracks/algorithm_codesign/g3_finite_law.py)
- [R5] [G3 saved-output audit source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/scripts/tracks/algorithm_codesign/audit_g3_saved_outputs.py)
- [R6] [G2 independent math audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/docs/tracks/algorithm_codesign/g2_independent_math_audit_20261009.md)
- [R7] [G2 result intervals](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b260189b020ab7dfabb16bf424f49a6efff40d75/artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json)
- [R8] 添付前版 `track_b_G3_scientific_review_20261009.md`。本書では前版を上書きしていない。
- [R9] [G1 scientific review（G3基点の保存版）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/research/track_b_G1_scientific_review_20261009.md)
- [R10] [G3 Phase B result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/artifacts/track_b_g3_finite_law/2026-10-09/phase_B_result.json)
- [R11] [G3 Phase A design](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_finite_law_design_20261009.md)
- [R12] [G3 return comparator design](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b0fa70b47848e72ef9a9c9e13afc3164962f7cd/docs/tracks/algorithm_codesign/g3_return_comparator_preparation_20261009.md)

### 外部一次文献

- [L1] D. Cugini, T. A. Atif, Y. Subasi, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1 (2026). [本文](https://arxiv.org/html/2603.13495v1). 特に§II、Theorem 1、Eq. (7)–(13)、bias不変性の§III。
- [L2] J. Peetz, S. E. Smart, P. Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). [出版本文](https://www.nature.com/articles/s41534-025-01168-w). 特にMethods「Convex Taylor sampling procedure」、operator decomposition、Pauli collection。
- [L3] B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024). [arXiv本文v2](https://arxiv.org/html/2402.15550v2), [出版情報](https://link.aps.org/doi/10.1103/PRXQuantum.5.040352). 特に§II.1–II.3。主たる表現対象がprocess matrixである点を区別する。
- [L4] J. Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2、PRX Quantum 7, 020332 (2026). [版・出版情報](https://arxiv.org/abs/2503.05647). 本再レビューでは版情報と研究目的を照合したもので、全44ページの再監査完了を主張しない。

外部文献は本書の研究結果の代替ではなく、既知の原理と新規性候補の境界を確認するために用いた。G3の数値はrepository資料、追加の下界・損益分岐・Pauli整理は本レビューの推論・算術として明示した。
