# Track B：全検証を踏まえたアルゴリズム改良・研究計画

作成日：2026-10-06  
位置付け：GPTによる研究設計・導出。リポジトリの実行契約・authorizationではない。

## 0. 結論

第一候補は **有限平均を保存するRTEの次数間係数再配分** とする。既存アルゴリズムを別の目的関数で選ぶだけでなく、ランダムeventの角度・長さ・分布を生成する構成そのものを変える。

第二候補は **恒等作用へ戻るeventの解析的回収と残余sampling** とする。ただし、恒等項・反交換関係をTaylor/LCUへ利用する原理は既知であり、最初から独立の新手法とは呼ばない。有限RTE・DF実装に即した非列挙samplerと、実装資源上の追加価値が残る場合だけ発展させる。

第三候補は **平均作用素の安定性を用いた合成誤差・精度配分**。これは全比較の共通基盤として優先するが、現段階では独立した主論文候補にしない。

主RQ案：

> 固定されたpartial-randomized time-evolutionの有限平均作用素を保ちながら、random eventの構成・生成分布・実装精度を変更し、測定負担とnative controlled回路費用を減らせるか。その改善を、利用可能な入力だけから構成・説明できるか。

第一候補は、今回、一般の奇数Taylor次数について限定family内のnormalization最小値と達成構成まで導出した。これは実装確認・新規性確定・分子での資源改善を意味しない。Codexに渡すべきものは、未具体化の「最適化してほしい」ではなく、本書の式・主張・反証対象である。

## 1. 証拠と今回の判断を分ける

本書では次を区別する。

- **既存証拠**：固定commitの文書・保存結果が示すこと。元のstatus、scope、dirty/local/held-out/post-hocの境界を変更しない。
- **一次文献の既知内容**：公開本文で確認した手法。完全な引用網監査や新規性の不存在証明ではない。
- **今回の導出**：GPTが数式から構成した候補・証明。独立なCodex検証は未実施。
- **実行提案**：後で利用者が採否・資源上限を決めるもの。既存one-shotの再開や新科学実行の許可ではない。

新しい分子入力、保存結果の科学的再計算、trajectory、回路合成、リポジトリ編集は今回行っていない。GPT作業環境では、次数1/3/5/7の係数一致等について小さい記号計算を補助的に行った。これは原repoのtests、科学pilot、独立再現に数えない。

## 2. 全ルートの証拠棚卸し

| 系列 | 根拠が支持する結果 | 支持しない一般化 | 次候補への含意 |
|---|---|---|---|
| Track A M1/M2/PM | 近接discardを加えた固定218構成でもB2の低いRZ-work点が残る。固定5構成transferあり。qを増すbias/shot減少と回路費用増加が競合 | 全方式・全合成法への優位、真の最適構成、T/物理総費用 | partial構造を残す理由はある。全wrapper・準備費用込みで評価する |
| P-A | interval分割は30/30で一区間対照と同じ。一方、先行run-level full/support-union選択のH5 −17.076%、H4 opt2 −6.598%は保持 | basis最適化全体が無効 | 有効だったrun単位basis処理を全対照へ共通適用。区間DPを再発明しない |
| P-B | 現H4 gridでenergy/signal選択不一致0/6、実用的signal failureなし | 全系でphase/weightが無関係 | 未観測failureを動機として過大に使わない |
| P-C | 局所のsigned-error相殺はあるが追跡prefixは全8点で同じ。stretch予測と診断が失敗 | geometry間相殺が一般にない | 外挿依存より、入力ごとの恒等式を保つ改良を優先 |
| P-D | 主対照の最良選択は一致、近傍誤差は小さい。ただし全域ではfalse acceptanceや大差がある。nested/native work差は大きい | finiteモデルは常に不要、一般にnestedが13倍悪い | 選択器だけを詳細化しない。内部実装・同じtargetを揃える |
| R3 | 広いselector/一般multifidelityとの独立差が未定義、数値pilotなし | あらゆる最適化・selectorの不可能性 | 新しい構成・取得手順・保証を具体化する |
| FR | 正scalar除去後も8件で厳密なphase-bound改善。固定予算の単独合格0 | 理論機構がない | 数学的改善と離散的resource改善を別段階にする |
| B-F | 限定familyでF/Lは同じSuzuki5。全対照最良はnative S2。原INCONCLUSIVE/R0 BF-Aの二層 | 高次PF係数最適化全般のno-go | PF係数ではなく、同じ有限平均の実現方法を変える |
| B-M | 現adapterの三次係数はcompact BCHと同値。同情報・同集約・同reuseでも独立差なし | multirate一般の無効性 | 式の書き換えと新しい構成を区別 |
| B-S | 三つの32標本経験分布のIS headroomは約0.03–0.12% | population上界、別representation/samplerの否定 | 固定supportのreweightだけより、representation変更を候補に |
| SP-0.5 | exact安価notchと高精度通常合成のprimitive trade-offを実装確認 | 新規性、DF上の優位 | 合成器・位相検査の部品は再利用可能 |
| SP-1 | 固定toyで短いDRが有利、長いと不利。D/R単独gainなし | actual RTE×PAIの失敗、selective全般のno-go | 外側はweight1 coin。実RTEの二層相互作用を実証したとは言わない |
| BS-0.5 | 現sparse多項式→operator LCU candidateはgenericと同処理 | 新構成がgeneric LCUで表現できたら無価値 | 同じ問題を同じ手順で解くことと、新しい構成familyを区別 |

### 2.1 研究運用上の修正

過去STOPは、各当時の問いを閉じる記録であって、その研究領域全体の禁止札ではない。ただし、閉じたdomainを再実行して好結果を探すことはしない。

「汎用LCU/最適化で表現可能だから新規性なし」は過剰な基準である。新しい公式・event familyも一般のLCU集合には属する。新規性は、具体的な構成、限定class内最適性、必要情報、計算量、実装、適用範囲のどこに差があるかで判断する。一方、現BS candidateはその具体処理まで同一だったため専用armを外した判断を維持する。

同じsplitでも成果になり得る。全資源で厳密dominanceしなくても、説明可能な有用なPareto trade-offは成果候補となる。逆に、数式上の差やnormalization差だけで論文成立とはしない。

## 3. 候補の比較と優先順位

| 候補 | 変えるアルゴリズムの部位 | 今回具体化したもの | 主なリスク | 優先順位 |
|---|---|---|---|---|
| A 次数間係数再配分 | Taylor項をunitary eventへ割り当てる規則 | 全奇数次数のexact family、限定normalization最適構成、K2閉形式、1D資源調整family | 既知文献との一致、小xで利得が微小、合成costが利益を消す | 第一主線 |
| B 恒等return回収 | 重複indexのpathをsample前に積分し、残余を直接sample | K2 exact identity、normalization式、棄却不要conditional sampler | 原理は既知圧縮と重なる、s2が小さいと実益小 | 第二候補／強い対照候補 |
| C 安定性・精度配分 | block合成誤差の伝播とprecision選択 | P3のcontractivity範囲、mean telescope、KKT配分の出発点 | 基本原理は既知、独立新規性が弱い | 共通基盤 |
| PF/rank/split選択の再探索 | 比較設定 | 新しい機構は未提示 | P-D/B-F/R3/SPRINTと重複 | 今は再開しない |
| multirate/compact評価の再整理 | leading error推定 | 現案は同値確認済み | B-Mと同じ問題 | 今は再開しない |
| PAI/block辞書のそのまま適用 | 合成層 | 既知の基準・toolは有用 | SP/BSの限定差しかない | 比較基盤として保持 |
| THRIFT/interaction picture | simulation構成 | 本repoに即した安価な必要primitiveは未具体化 | exact H_D oracleを仮定しやすい | 今回は主線にしない |

A/Bはfinite meanを保存するため、geometry外挿やexact ground stateへのfitを必要としない。これは過去P-B/P-C/R3の不確実性と独立に試しやすいという判断であり、資源改善の実測根拠ではない。

## 4. 共通数学と対象

Hermitian involutionの分布を

\[
\widehat R=\sum_{\ell=1}^L p_\ell Q_\ell,\qquad
p_\ell\ge0,\quad \sum_\ell p_\ell=1,\quad
Q_\ell^\dagger=Q_\ell,\quad Q_\ell^2=I
\]

とする。係数の符号はQへ含める。Q同士の可換性は仮定しない。

Pauliの場合だけでなく、`Q=V† P V`のような共役involutionも式の対象にできる。ただしVの実装は無料ではない。同じbasisのproductと異なるbasisのproductを同じ費用としない。

x>0は一microstepのdimensionless絶対時間、σ=±1は時間符号とする。元の有限目標は

\[
M_m=P_m(-i\sigma x\widehat R)
=\sum_{n=0}^m(-i\sigma)^n t_n\widehat R^n,
\qquad t_n=x^n/n!.
\]

paired cutoff Kがevenならm=K+1はodd。特にK=2はP3である。x=0は別のidentity branchとし、0/0を計算しない。

対象はcoherent first momentである。新旧でMを同じにしても、平均channel `E Ad(U)`が同じとは限らない。Hadamard信号を測る際のphaseと、各occurrenceの独立samplingを維持する。有限Mとexact evolutionの差は元のまま残る。

## 5. 候補A：隣接次数を重ねて割り当てるRTE

### 5.1 係数制約

k=0,…,mについてa_k,b_k≥0とし、

\[
b_{-1}=0,\quad b_m=0,\quad a_k+b_{k-1}=t_k\quad(k=0,\ldots,m)
\]

を課す。**b_m=0を必須**にし、意図しないm+1次数を作らない。

\[
c_k=\sqrt{a_k^2+b_k^2},\quad
\phi_k=\operatorname{atan2}(b_k,a_k).
\]

c_k=0のfamilyはsampleしない。独立にpからQ_0,…,Q_kを取り、

\[
U_k=(-i\sigma)^k e^{-i\sigma\phi_k Q_0}Q_k\cdots Q_1
\]

を実行する。右から作用するという演算規約と、chronological circuit列を区別して保存する。

### 5.2 有限平均保存の証明

Q_0²=Iと独立性から、

\[
c_k\mathbb E U_k
=(-i\sigma)^k(a_k I-i\sigma b_k\widehat R)\widehat R^k.
\]

各次数nの係数はa_n+b_(n−1)=t_n。terminal b_m=0により余分な次数がなく、

\[
\boxed{\sum_{k=0}^m c_k\mathbb E U_k=M_m.}
\]

Q同士の非可換性は妨げにならない。因子の期待値が同じRhatであるため、その順序で積が得られる。

\[
B=\sum_kc_k,\qquad q_k=c_k/B
\]

でsampleし、Hadamard ±1 outcome YにBを掛ける。補正後平均は各軸のM_m信号、二次モーメントはB²。

複数microstepとdeterministic因子を組み合わせても、各occurrenceを独立にsampleすれば元のfinite mean積を保つ。全時刻で同じrandom sampleを使い回した場合はこの証明を適用しない。

### 5.3 標準paired RTEの包含

標準構成は

\[
(a_{2j},b_{2j})=(t_{2j},t_{2j+1}),\qquad
(a_{2j+1},b_{2j+1})=(0,0)
\]

という特殊点である。

Wan–Berta–Campbell Appendix C Eq.(C4)–(C5)とGüntherらAppendix A Eq.(A3)は、この偶数次数から次の奇数次数をまとめる構成を示している。一般のTaylor/LCUやこの標準pairingを新規性にしない。

### 5.4 限定family内のnormalization下界

m=2d+1とし、

\[
E_d=\sum_{j=0}^d t_{2j},\qquad
O_d=\sum_{j=0}^d t_{2j+1}.
\]

偶数kにベクトル(a_k,b_k)、奇数kに(b_k,a_k)を対応させると、その総和は(E_d,O_d)。各ベクトルのEuclidean normはc_kなので、三角不等式から

\[
\boxed{B\ge\sqrt{E_d^2+O_d^2}.}
\]

これは **非負係数・隣接二次数familyの範囲の下界** である。任意LCU、任意Pauli恒等式、任意量子手法についての下界ではない。

### 5.5 全奇数次数で下界を達成する構成（今回の導出）

\[
\rho=O_d/E_d>0
\]

とおく。`a0=t0`から始め、各kで

\[
b_k=\begin{cases}\rho a_k&k\text{ even},\\a_k/\rho&k\text{ odd},\end{cases}
\qquad a_{k+1}=t_{k+1}-b_k
\]

とする。k=mではa_m=b_m=0となる。

全係数の非負性も示せる。E_j=Σ_(l=0)^j t_(2l)、O_j=Σ_(l=0)^j t_(2l+1)、O_(−1)=0と書くと、

\[
a_{2j}=E_j-O_{j-1}/\rho,\qquad
b_{2j}=\rho E_j-O_{j-1},
\]

\[
a_{2j+1}=O_j-\rho E_j,\qquad
b_{2j+1}=O_j/\rho-E_j\quad(j<d).
\]

ここでO_j/E_jは、t_(2l)をweightとするx/(2l+1)の加重平均であり、jとともに減少してtanh(x)へ収束する。
一方O_(j−1)/E_jは、同じweightを用いた2l/x（l=0では0）の加重平均であり、jとともに増加してtanh(x)へ収束する。
したがって

\[
O_{j-1}/E_j\le\tanh x\le\rho\le O_j/E_j\quad(j\le d),
\]

から全a,b≥0。最後のa_m=O_d−ρE_d=0でterminal条件も満たす。

さらに向きを揃えた全ベクトルは比率ρで平行となり、下界を達成する：

\[
\boxed{B_* =\sqrt{E_d^2+O_d^2}.}
\]

標準の各pairベクトルの傾きはx/(2j+1)。d≥1,x>0では少なくとも二つが非平行なので

\[
B_*<B_{\mathrm{pair}}.
\]

上記の再帰はreal arithmeticでの構成。特に小x・高次数では近い量の差し引きがあり、有限精度での非負性・係数一致は別に保証する。丸めで生じた負値を無断で0へclampしない。K2では以下の非負な有理式を使える。

これで示したのは **このclass内の最小normalization** であり、最小T数や最小総資源ではない。一般sizeの根拠はこの解析であり、有限fixture一致を一般証明の代用にしない。

### 5.6 K=2の閉形式

\[
\rho=\frac{x+x^3/6}{1+x^2/2}.
\]

| k | a_k | b_k |
|---|---|---|
| 0 | 1 | ρ |
| 1 | x−ρ | (x−ρ)/ρ |
| 2 | x³/(6ρ) | x³/6 |
| 3 | 0 | 0 |

同値な差し引きの少ない式：

\[
a_1=\frac{2x^3}{3(x^2+2)},\quad
b_1=\frac{2x^2}{x^2+6},\quad
 a_2=\frac{x^2(x^2+2)}{2(x^2+6)}.
\]

\[
B_{\rm pair}=\sqrt{1+x^2}+\frac{x^2}{2}\sqrt{1+x^2/9},
\]

\[
B_*=\sqrt{(1+x^2/2)^2+(x+x^3/6)^2}.
\]

差の符号は次から直接確認できる。

\[
B_{\rm pair}^2-B_*^2
=x^2\left[\sqrt{(1+x^2)(1+x^2/9)}-1-x^2/3\right]>0.
\]

括弧内の二つの正量を二乗した差は4x²/9。

### 5.7 小xでの改善限界

\[
B_{\rm pair}-B_*=x^4/9+O(x^6).
\]

L回、x=τ_total/Lの同じmicrostepへ使う単純化では、normalizationの対数利得はO(Lx⁴)=O(τ_total⁴/L³)。
**細かいstepでは数学的利得が実用上小さい可能性が高い。** 原H4で何%になるか、まだ計算していない。

大きいxでnormalization差があっても、P3のtruncation errorが許容されるとは限らない。same finite targetの性能とsame exact taskの性能を分ける。

### 5.8 角度・回路への接続

θ=atanρとすると、最適点のeven family角はθ、odd family角はπ/2−θ。
odd kについて一般に

\[
(-i\sigma)^k e^{-i\sigma(\pi/2-\theta)Q_0}Q_k\cdots Q_1
=(-i\sigma)^{k+1} Q_0 e^{i\sigma\theta Q_0}Q_k\cdots Q_1.
\]

K2のodd k=1なら前係数は−1。従って任意角rotationの絶対値を共通θへ寄せる実装が可能だが、追加Q_0、global/controlled phase、basis遷移の費用が必要。

Pauli productが一つのPauliへcollapseする場合、その簡約を新旧へ共通適用する。DFの異basis involution積は一般に同じ簡約では処理できない。raw word lengthの削減をT/RZ削減と同一視しない。

### 5.9 resource-awareな1変数family

最初から全a,bを最適化しない。標準点と上記最適点の間を

\[
(a(\eta),b(\eta))=(1-\eta)(a^{\rm pair},b^{\rm pair})+\eta(a^*,b^*),\quad0\le\eta\le1
\]

で補間する。線形制約と非負性は維持され、**全ηで同じfinite mean**。
convexityより

\[
B(\eta)\le(1-\eta)B_{\rm pair}+\eta B_*\le B_{\rm pair}.
\]

これにより、理想normalizationは悪化させず、角度とevent費用のtrade-offを一変数で試せる。ηごとのcompiled costは未取得。η=0,1/2,1などの少数案は、将来pilotの提案値であり現在の実行認可ではない。

実際の資源：

\[
\overline C_Q(\eta)=\sum_k\frac{c_k(\eta)}{B(\eta)}\,\mathbb E[C_{k,Q}(\phi_k(\eta))].
\]

canonical moment B²だけならleading proxyはB·Σc_k C_kだが、全体はfinite-confidence、前後のD回路、合成bias、初期化を戻す。
fixed cost係数を使うΣw_k sqrt(a_k²+b_k²)はconvex conic問題。一方、角度依存のactual synthesis countを入れた総資源がconvexまたはglobally optimizedとは主張しない。

### 5.10 新規性の境界

今回本文確認したWan/PRでは固定偶数pairingが明示されている。importance samplingは固定ensembleの抽出確率を変えるもの。本案はevent角・supportを変える。

ただし、Taylor/LCUの一般表現、係数norm最小化、凸最適化自体は既知。上記のoverlapping隣接family・全奇数次数の達成構成・特定最適性が既知の別定式化から直接得られていないかは、Codexのfocused prior-art照合でも反証対象にする。「今回見つからない」を世界初の根拠にしない。

generic solverに同じfamilyを与えて同じB_*を得ることは期待されるcorrectness対照。そこから提案の全価値を消さず、構成式と必要情報・実装利得の新しさを別途判断する。generic解法との差が標準実装の小さいconstant factorだけなら、その範囲を正直に記述する。

## 6. 候補B：identity-returnをsampling前に回収する

### 6.1 K2の完全な式

\[
s_2=\sum_i p_i^2,\qquad D=\sum_{i\ne j}p_i p_j Q_iQ_j,
\qquad \widehat R^2=s_2I+D.
\]

ここでは反交換性すら仮定しない。P3を整理すると

\[
M_3=(1-s_2x^2/2)I-i\sigma(x-s_2x^3/6)\widehat R
-\frac{x^2}{2}(I-i\sigma x\widehat R/3)D.
\]

第一項群を

\[
A=1-s_2x^2/2,\quad D_1=x-s_2x^3/6,\quad
c_{\rm ret}=\sqrt{A^2+D_1^2},\quad\phi_{\rm ret}=\operatorname{atan2}(D_1,A)
\]

として、`c_ret E exp(-i σ φ_ret Q0)`へまとめる。AやD1が負になる範囲でもatan2の符号と位相を保持する。そこを実用的Taylor精度域と主張はしない。

残りは(i,j)|i≠jを確率p_i p_j/(1−s2)で取り、独立Q0を使う通常のorder2型eventで実装できる。

\[
\boxed{
B_{\rm ret}=c_{\rm ret}+(1-s_2)\frac{x^2}{2}\sqrt{1+x^2/9}.
}
\]

ベクトルv0=(1,x)、v2=(x²/2,x³/6)を使えば、c_ret=||v0−s2 v2||。三角不等式からB_ret≤B_pair。
small-xでB_ret=1+(1−s2)x²+O(x⁴)。改善幅を支配するs2は、分布が拡散すると小さくなる。

s2=1なら残余samplingはない。identity-only、zero time、符号同値の重複項を先にどう統合するかも両armで固定する。

### 6.2 集中分布でも棄却待ちしない生成

独立にi,jを引いてi=jなら引き直す方法は、s2→1で古典費用が増える。
代わりに

\[
\Pr(i)=\frac{p_i(1-p_i)}{1-s_2},\qquad
\Pr(j\mid i)=\frac{p_j}{1-p_i}\quad(j\ne i)
\]

を使う。前処理O(L)のtable、prefix CDFでiのintervalを飛ばすsample等を用いれば、残余を直接生成できる。
これは古典的な条件付き抽出の構成であり、その技法自体を新規性としない。入力確率のbit precisionとsample生成costを数える。

### 6.3 反交換性への拡張と先行研究

正確に{Q_i,Q_j}=0と分かるunordered pairは、Q_iQ_j+Q_jQ_i=0で消せる。可換pairの合計massだけを残余へ戻す拡張が可能だが、group graphの取得費用は別途必要。

**この原理はZhao–Yuan 2021が直接扱う領域である。** 同研究の§4.2は高次のidentity/H_l寄与を係数へ戻し、§4.3は完全反交換系の簡約を示す。従ってidentity抽出・反交換相殺を新しい発明として扱わない。

Bを独立候補として残す条件は、既知圧縮との対比で、有限RTEのexact meanを保つ非列挙sampler、DF-native実装、取得costを含む実用的利益が具体化すること。そうでなければAの強いbaseline・既知前処理として用いる。

共通basis中のDF involutionは簡単な関係を持つ場合があるが、異basisのGaussian共役involution同士は一般のPauli反交換graphではない。近い反交換をexact0にしない。Q_i²=Iのreturn検出だけならこの問題を避けられる。

## 7. 候補C：平均の安定性を利用した合成誤差配分

### 7.1 P3のcontractivity

real yについて

\[
|P_3(-iy)|^2=1-y^4/12+y^6/36.
\]

従って|y|≤sqrt(3)で|P3|≤1。Hermitian Rhat、||Rhat||≤1なら|x|≤sqrt(3)において||P3(-ix Rhat)||≤1。
これはscalar polynomialとspectral mappingからの基本的導出。一般Kや全xには拡張しない。

exact mean blocks M_jがcontractiveで、implemented meanが||Mtilde_j−M_j||≤e_jなら、

\[
\|\widetilde M_L\cdots\widetilde M_1-M_L\cdots M_1\|
\le\prod_j(1+e_j)-1
\le e^{\sum_j e_j}-1.
\]

局所実装誤差には、そのblockのLCU weightを数える。例えばe_j≤Σ_l |a_(j,l)| δ_(j,l)。
他blockのsample-level normalizationを全て掛ける粗いboundしか使えないとは限らない。

**平均作用素がcontractiveでも、推定量のsecond momentは減らない。** B_total²やweight積は統計会計へ残す。この区別を破って「誤差もshotsも無料で改善」としない。

### 7.2 合成precisionの出発点

expected synthesis costモデルがΣ_j d_j κ_j log(1/η_j)、bias budgetがΣ_j w_j η_j≤δなら、内点の最適配分は

\[
\eta_j^*=\frac{\delta d_j\kappa_j}{w_j\sum_l d_l\kappa_l}.
\]

これは標準的なKKT計算であり、研究の新原理ではない。上限/下限precisionとactual gate countの段差を戻す。
d_jとw_jが同じ比率、κ_jも同じならuniform精度へ戻るので、「rare eventだから必ず粗いprecisionでよい」とはしない。

Cは全baselineへ適用できる比較整備・実装改善として扱う。tight boundの量子固有の有効域、取得cost、actual資源差まで残らなければ独立主線にはしない。

## 8. 先行研究との比較表

| 一次資料 | 今回確認した箇所・既知内容 | A/B/Cへの位置付け |
|---|---|---|
| Wan, Berta, Campbell; arXiv:2110.12071v2; PRL129030503 | Appendix C Eq.(C4)–(C5)：偶数Taylor項と次項のpairing、unitary sampling。本文Lemma2 | Aの直接の構成baseline。pairing自体は既知 |
| Güntherら; arXiv:2503.05647（取得PDFの表示v2） | Appendix A Eq.(A3)–(A5)：RTE、平均operator、独立step。Appendix E.3：rounding/circuit accounting | PR全体・実装・有限平均の直接baseline |
| Zhao, Yuan; arXiv:2103.07988v2; Quantum5,534 | §4.2 modified LCU、identity/H_l係数更新、§4.3反交換簡約 | Bの原理新規性は認めない。AもTaylor再編という広い比較を要する |
| Cugini, Atif, Subaşı; arXiv:2603.13495v1 | §II Theorem1、固定ensembleのq∝p/sqrt(C)、bias保存 | 全新旧armへ同等に使える強いsampling対照 |
| Koczor; arXiv:2402.15550v2 | §II/III：辞書型QPD、l1、残差、累積sampling | 汎用最適化・block分解という一般原理を新規性にしない |
| Kiumi, Koczor; arXiv:2410.16850v2 | 時間発展でのPAI、matrix/channel taskの違い | SPの既知機構、mean/channel比較の強い対照 |
| Casaresら; arXiv:2606.30741v1 | integrated PF/SPRINT、factorization/near-integrability/randomization | 広いsplit/co-designや高次化を新規claimにしない |
| Wadaら; arXiv:2512.06260v1（repo BS監査） | grouped LCU、KρK†型task、workspace | groupingと今回のlinear meanを混同しない。Aへ無料ancilla対照を押し付けない |
| Fujiwaraら; arXiv:2606.06070v1 | sign-flip groupingを使うcontrolled時間発展構成 | 単純なcontrol除去を別の新手法としない。control task規約を合わせる |

今回の検索は最接近資料の本文と関連検索に基づくscoped audit。検索が一致を返さないことは新規性証明ではない。Aの具体familyと達成式の優先性は未確定であり、著者の未確認supplement・別形式の既知結果を含め独立照合する。

## 9. 公平な比較の設計

### 第1層：同じfinite meanを保つ比較

Hamiltonian、DF表現、split、PF、microstep、K、入力state、精度をそろえる。Aは実現ensembleだけを変える。Bはsample前の代数的return回収だけを変える。

必要対照：

1. 標準paired finite-RTE、canonical sampling。
2. 標準paired＋既知cost-optimal IS（適用の仮定・C=0・range項を明示）。
3. A標準点、解析normalization最適点、少数のresource-aware補間点。
4. 同じA familyを直接扱う汎用conic/numerical solve：主に正しさ・探索到達性の対照。
5. 既知Pauli product簡約、identity/anticommutation圧縮が使える場合の強い対照。
6. P-Aで有効だったrun-level basis方針、controlled-phase、合成precisionを両armで共通化。

すべてを最初の数式テストへ入れるのではなく、claimに必要な段階で固定する。A/B/Cを最初から総当たりjoint最適化しない。

### 第2層：同じexact taskの比較

有限平均保存下で実益があった後に、q/r/Kや合成precisionの再調整を全armへ対称に認める。元finite Mのbiasが小さい/大きいことと、representationそのものの利益を分離する。

### 資源

primaryは用途を明記したresource vector：total synthesized T（取得可能な場合）、compiled RZ/CX/depth、sufficient shot数、workspace、古典生成・前処理cost。未知座標は0にしない。

厳密Pareto dominanceのみを研究継続の必須条件としない。trade-offなら条件を示し、その条件が実用上意味を持つかを評価する。逆に都合のよい後付けhardware weightでwinnerを作らない。

mean保存は理想primitiveでの主張。有限合成・係数丸め・確率誤差を別に含める。入力stateのoracleを使う評価と、運用で使えるpredictorを区別する。

## 10. 最小検証計画

### R0：今回の式を独立に反証する

Codexへ最初に渡す範囲はAの証明・実装意味論監査を中心とする。B/Cは別節として式と既知関係を検査する。

- Aの係数matching、末端b_m=0、zero support、x=0、負時間、phase±1/±i。
- 任意奇数次数の非負性証明、Euclidean下界、達成構成。
- K2閉形式の同値性、small-x次数、η補間のexactnessとB上界。
- 非可換involutionの自由wordでoperator恒等式。特殊な可換toyだけで確認しない。
- odd familyのcomplement angle変換とcontrol位相。rotation数・Pauli/DF action数を区別。
- 既知Wan/PR/modified Taylorの式と候補式を同じ規約へ変換して比較する。
- generic solverへ同じfamilyを与えたときの一致は正しさとして記録する。
- Bのreturn恒等式、conditional probabilities、s2=1の端点、既知圧縮との差の範囲。
- Cのcontractivity範囲、meanとvarianceの分離、KKTの退化条件。

ここでは新分子、元run再評価、合成、原科学pilotを実行しない。有限自由wordやexact Fraction等の記号検査を、科学的性能結果と混同しない。新しい小行列/solver性能評価を始める場合は、静的監査とは別scopeとして予算を先に採用する。

### R1：安価な実益判別

R0で式が残ったら、第一候補Aに限り、小さい登録domainでactual primitive/controlled実装費用まで確かめる。

最初の候補は標準η=0、解析端点η=1、必要なら中間η=1/2。K2を主にし、K4は初期にはcorrectness controlだけとする案。
step xの選び方は、物理入力からの取得可能範囲と有限Taylor誤差budgetを根拠に決める。normalization差が見えやすいxだけを後から足さない。

Qが共通Pauli basisのcase、非可換case、basis変換が費用を持つcaseを必要最小限用意する。詳細target、count、wall/RAM/storage、compile/call cap、materiality、数値guard、failure記録は結果前の別契約で固定する。

R1は分子規模の有利性ではなく「標準pairing＋強いIS/簡約後にもrepresentation差に情報価値があるか」を判定する。

### R2：実DF文脈への接続

有効性が残る場合だけ、既知developmentの一構成へ接続する。Q²=Iの実表現、basis/context、P-Aの既存run-level policy、準備費用、合成精度を戻す。A/Bの独立効果を混ぜない。

### R3：必要な独立条件と論文化

可搬な運用規則・分子への一般性を主張する場合だけ、未使用条件を確保する。証明の一般性は独立dataの代用を必要としないが、実装資源の転移・汎用性は別に検証する。M2の1.30Åは新held-outにしない。

探索的な数学/技術検証、source固定の確認実験、バグ修正を分ける。既存consumed markerや原結果を消さず、failure後の新契約を元の一回実行として扱わない。一方、全研究を一律「一度のtoyで不利なら分野全体STOP」にしない。

## 11. 成功・縮小・終了条件

| 判定層 | 良い結果 | 不十分な結果 | 次の扱い |
|---|---|---|---|
| 数学 | same finite mean、限定B最適性、phase正しい | terminal漏れ、非可換で崩れる、係数不正 | 実装前に修正/棄却 |
| 新規性 | 具体family/生成式/限定最適性/実装手順の差が残る | 同じ構成が既存論文に明記、差は名前だけ | 既知法として活用しnew-method claimは下げる |
| 理想資源 | B低下・event費用変化に再利用可能な機構 | 微小B差のみ、最良対照で消える | 理論note/機構記録へ縮小を検討 |
| 実装資源 | full contextとtask-tuned精度後も意味ある点/領域 | synthesis/basis/準備で利益消失 | 対象scopeを閉じる。新条件の追加は新機構がある場合だけ |
| 科学的成果 | 理論・構成・実装・有効/不利条件が一つの主張になる | 多数のtoy winner表だけ | 独立論文を無理に作らない |

相対normalizationが下がるだけ、stage/splitが変わるだけ、同じ解になるだけ、いずれも単独で全研究のGO/STOPにしない。

## 12. 論文の着地点

### 第一目標：有限平均保存型RTE representationの方法論

仮題：**Finite-mean-preserving redesign of randomized time-evolution ensembles**。

中核：標準pairingを含む構成family、限定normalization最適性、controlled/native実装、強い同条件対照、資源trade-offと実用範囲。

限定classの定理だけで査読採択や十分な新規性を保証するものではない。大系で巨大な改善を必須にはしないが、どの情報・構造・用途で役立つかを示す。

### 最小着地点

Aの構成と最適性が正しく、文献上独立だが実資源利得が小さい場合：理論・技術noteとして、何が最適化でき何がボトルネックに残るかを閉じる。原理自体が既知なら、再現可能な実装比較を残すだけでもよく、独立論文を必須にしない。

### Bの発展条件

既知のTaylor相殺を、有限RTEで実用的・非列挙・phase-safeなsamplerに落とすことで、取得とquantum資源の両方に意味が残る場合だけ、構造利用samplingの第二論文またはAの発展へ進める。

### Cの位置

基本恒等式と標準precision配分だけなら共通技術部品。FRの過去結論を変更しない。新しいcertified大系error情報や実装上の重要な知見が出た場合だけ独立研究を検討する。

Track Aは既存resource case studyとして進める。Aの原稿完成をBの成功待ちにしない。

## 13. Codexへの直近作業指示

本書の候補Aを主対象に、**独立の数学・意味論・先行研究反証**を行ってください。候補の方向をCodex側で別の最適化へ広げないでください。

### 入力

固定履歴はBS-0.5 `5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d`まで。Aは`4c23453c541700c6a41ba71fc5ec9323b53858d6`を別系列参照。本書は新しいGPT設計入力であり、既存科学sourceや実行authorizationではありません。

### 作業

1. §5の定義から、有限mean保存、一般奇数次数の非負性、下界達成、K2式、η補間を独立導出してください。
2. free-word/exact arithmeticの少数fixtureで、term order、negative time、odd global phase、terminal条件を検査してください。非可換性を消したtoyだけで合格にしないでください。
3. Wan Appendix C、PR Appendix A、Zhao–Yuan等の関連式を同規約へ戻し、同じfamily/達成構成/最適性が既知か、scope別に報告してください。検索不一致を新規性証明にしないでください。
4. Q productがglobal PauliとしてcollapseできるcaseとDFの異basis caseで、event実装の必要operationを静的に分けてください。既存P-Aの有効なrun-level policyは共通baselineへ含めてください。
5. §6のBと§7のCは主に式と重複の確認に留め、A/B/C同時最適化や新scienceを始めないでください。
6. 次に数値・合成pilotが必要なら、具体target/call/数値guard/予算を一つの提案として提示し、実行前にreviewへ戻してください。

### 禁止範囲

既存科学result・marker・authorizationの編集/再生成、原runの再実行、新分子・新Hamiltonian・NPZ操作、trajectory、actual circuit synthesis/compile、GPU、別辞書/別PF/別geometry探索、Track A・共通API変更は今回行わないでください。

### 成果物

本書の主張ごとの `PROVED / COUNTEREXAMPLE / KNOWN_EQUIVALENT / UNRESOLVED` 表、独立証明、fixtureとsource identity、文献の具体箇所、次pilotを提案するなら必要最小範囲を保存してください。静的/記号作業を「実resource改善の検証」と呼ばず、commit/pushしてSTOPしてください。

methodや論文化の最終判断はGPTへ返してください。汎用solverが同じ問題を解けるという理由だけでAの構成を無価値とせず、同時に既知LCUを名前変更して新手法と扱わないでください。

## 14. 固定資料一覧

GitHub repository: `HIROMU1015/Partially-Randomized-Trotter`。

- 全体handoff：`0da4d18acf3f5d32d1bc32c9661b667885bcf5f2` / `docs/tracks/algorithm_codesign/research_redesign_handoff_20261005.md`
- P-A：`6d2645a09440f50e5b869ef42a1b73a1b625a1af` / `docs/research_direction_joint_synthesis_mechanism_validation.md`
- P-B：同commit / `docs/research_direction_signal_weight_pilot.md`
- P-C：同commit / `docs/research_direction_geometry_tracking_breakdown.md`
- P-D：同commit / `docs/research_direction_pd_s1_posthoc.md`
- R3：同commit / `docs/research/r3_prior_art_and_minimal_contract.md`
- FR：同commit / `docs/fr_revision_nonuniform.md`
- B-F：同commit / `docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md`
- B-S：同commit / `docs/tracks/algorithm_codesign/bf0_prior_art_claim_matrix.md` §6
- B-M：`d55de044b8e956ba6292209a94bb081014dfdae2` / `docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md`
- SP-0.5：`e57c1fdd28589422e9c973e34e53f6c725b921e9` / `docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md`
- SP-1：`9d2bb1fa439748b02084bd9fbc9b10a705328f8a` / `docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md`
- BS-0.5：`5a4ae817ec8d833bb2929c0c0a85e2d4d3064e7d` / `docs/tracks/algorithm_codesign/bs05_method_target_design_audit_v1.md`
- Track A：`4c23453c541700c6a41ba71fc5ec9323b53858d6` / `docs/research/track_a_post_pm2_claim_evidence_map.md`

固定commitで参照し、可変branchの現在HEADと混同しない。本書では元artifactの全行再監査や実験の独立再現を行っていない。

## 15. 一次文献

1. K. Wan, M. Berta, E. T. Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071v2; Physical Review Letters 129,030503(2022). https://arxiv.org/abs/2110.12071v2
2. *Phase estimation with partially randomized time evolution*, arXiv:2503.05647. https://arxiv.org/abs/2503.05647 ; 本文確認は取得PDFのv2表示に基づく。byte-level immutable監査ではない。
3. Q. Zhao, X. Yuan, *Exploiting anticommutation in Hamiltonian simulation*, arXiv:2103.07988v2; Quantum5,534(2021). https://arxiv.org/abs/2103.07988v2
4. D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1. https://arxiv.org/html/2603.13495v1
5. B. Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, arXiv:2402.15550v2. https://arxiv.org/html/2402.15550v2
6. *TE-PAI: Exact Time Evolution by Sampling Random Circuits*, arXiv:2410.16850v2. https://arxiv.org/html/2410.16850v2
7. *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1. https://arxiv.org/abs/2606.30741v1
8. K. Wada et al., *Tradeoffs between quantum and classical resources in linear combination of unitaries*, arXiv:2512.06260v1. https://arxiv.org/abs/2512.06260v1 ; 今回のtask対応はrepo BS-0.5監査を根拠とし、全定理の再検証ではない。
9. *Efficient Quantum Circuit Construction of Controlled Time-Evolution for Arbitrary Pauli-Sum Hamiltonians*, arXiv:2606.06070v1. https://arxiv.org/abs/2606.06070v1
10. D. W. Berry et al., *Simulating Hamiltonian dynamics with a truncated Taylor series*, arXiv:1412.4687. https://arxiv.org/abs/1412.4687

## 16. 最終的な研究判断

現block-synthesis application pilotを直ちに走らせるのではなく、具体的なAの構成・最適性をまず独立に反証する。BとCは、過去の有効な構造・誤差機構を捨てずに使う第二候補と共通基盤として保持する。

第一の貢献候補は「PR＋別の既知最適化」ではなく、**同じfinite meanに対するRTE ensembleの構成法そのものを変え、限定classの最適性を与えること**である。既知性・実装費用・小xでの微小利得の三点は未解決であり、そこを次の反証と実装評価で閉じる。
