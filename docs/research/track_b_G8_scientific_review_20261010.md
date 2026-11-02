# Track B G8 科学的研究レビュー
## on-demand実装の評価、固定policy分離の射程、P5特殊化、次のmatched-native検証

- **作成日**：2026-10-10 JST
- **開始承認**：利用者の「レビューを開始して」。G8の結果、研究価値、新規性、今後の研究方針を対象とする。
- **対象repository**：`HIROMU1015/Partially-Randomized-Trotter`
- **G8 branch**：`track-b-g8-on-demand-provider-budget-20261010`
- **固定結果commit**：`e4b410746aadcf03c955b5e961c672c04b220de5`
- **実行source**：`cb60a4a1336f1803ad49d890f4e74aafd6ad7c61`
- **基点G7**：`a689694080f4b7600fe67cf77d841d1cbbf04503`
- **レビュー状態**：完了。新規性の優先性、一般native優位、主methodの最終採択は未確定。
- **本レビューの判断**：条件付きアルゴリズム候補として限定継続。次は短い同traceの反復ではなく、P5特殊化と、明示providerを持つ一つの同target比較をまとめる。

> **証拠区分**：G8原報告・契約・source・監査に記載された事実を「保存証拠」、今回導いた数式を「レビュー導出」、次段階の条件を「提案」として区別する。後二者をG8の登録結果へ転記しない。

## 1. 結論と前回からの変更

G7レビューの「限定継続」を維持する。ただし、継続理由と次の作業を更新する。

G8は、全event表・全angle表・旧合成列をproductionへ注入せず、局所生成からlive Rz取得、actual adjoint、条件付きprovider IRへ接続した。これはG7の実装上の不足に直接答える。一方、2048 interface trialsは8 rowのcold128/warm128であり、warmは同じbitstreamの再生である。百万規模の推定試行の完了、全supportの取得、大系でのcache費用を測った実験ではない。[R1][R2][R4]

P5では、保存価格をG8予算で再評価したfull-returnのRz費用が、3対照の固定policy下界より低い。固定したrepresentationのsamplingだけでは説明できない差がある。ただし、この下界は同rational係数・Rz価格・共通bias・Bernstein十分予算式の範囲に限る。[R1][R5]

今回の重要な追加整理は、**P5のfull-return representationもO(L²)の群へまとめられる**ことである。この特殊化では全L⁵ raw wordを列挙せずnormalizerを求め、canonical samplerを構成できる。したがって、G8の肯定例はrepresentationの価値を支持するが、一般Green-generatorやzero-fillがその入力で最良の実現方法だと示してはいない。これはG6の母関数から導いたレビュー段階の式で、独立認証前である。

次は、同じpipelineの形式的な確認を増やすのではなく、以下を一つの研究判断用作業にまとめる。

1. P5特殊化を独立確認し、一般法と同じ理想representationであることを確認する。
2. §14で提案する一つの明示providerに対し、direct native実装・既知CTSを含む同target比較を行う。
3. その結果から、条件付き構成研究としてまとめるか、一般入力での方法研究へ進むかを判断する。

旧G5の固定toy・固定dictionary主線は再開しない。物理providerを指定しないまま一般分子・DF・PR/QPEの総費用へ拡張もしない。

## 2. 使用した資料と確認の強さ

固定branchのremote refを読み、HEADがG8結果commitと一致することを確認した。主要handoff、結果前契約、予算source、on-demand source、oracle診断source、runner、tests、manifest、保存値監査、raw resultの取得可能性と関連箇所を確認した。source 54 hash・旧918 pathの照合は**保存監査の報告**であり、私が全pathを再取得・再hashした結果ではない。[R1–R10]

今回、旧runner、pytest、synthesizer、量子回路、量子測定、LP、分子計算を再実行していない。原結果320,161 bytesの全eventを別実装で再生成し、全oracle certificateを独立認証したとも主張しない。特にG8のoracle監査自体が、保存scalarとflagの照合であり、全eventの独立再構成ではないと記録している。[R1][R7]

レビューの新しいP5式は、同じP5 development入力を用いた小さい形式word oracleで自己検算した。経済値の割合・差分はhandoffの表示小数からの再計算であり、exact source certificateの代用品ではない。再現コードは末尾の付属資料に含める。

## 3. G8の証拠は三種類に分かれる

### 3.1 新しい取得・実装証拠

局所event生成、pre-quantum zero、signed tangent、live cache miss、strict Rz合成、actual-adjointを保持するprovider IRの経路を接続した。20 unique positive keysを取得し、全strict error guardsが通過した。production終了後にだけG7保存結果をreferenceへ読んでいる。runnerのimport/read順序とpipelineの引数も、この分離と整合する。[R3][R4]

ただし、物理controlled-Q providerは未実装であり、生成したのは条件付きnative descriptionである。20列がG7と同hashであることは、固定backend/seedの再現性・結び付けを支持するが、別実装・別機関の独立再現ではない。

### 3.2 保存価格の再予算化

G7のper-trial費用・provider係数を、G8で再設定したNへ掛けている。G8で全quantum shotsを実行した結果でも、全supportのnative費用を新たに取得した結果でもない。[R1][R3]

### 3.3 全表を使う補助oracle

production後にG7の全event係数・価格を使い、q∝alpha/sqrt(C_Rz)を一つずつ構成した。これはI2の参照診断で、実装した非列挙sampling lawではない。全表を使うoracleの費用を無料としたruntime比較でもない。[R5]

この三種類を混ぜて「G8で一般入力の総native優位を実証した」と書いてはならない。

## 4. G8の固定条件

| 項目 | 契約 |
|---|---|
| target | full first operator moment P_m(-ixR), R=Σp_iQ_i, Q_i²=I |
| P3 control | p=(3/7,4/7), x=2/5, m=3 |
| P5 | p=(1/5,3/10,1/2), x=5/7, m=5 |
| production arms | ordinary, partial_return_tail, closed_P3_tail, full_return |
| bit | probability H=160, root K=256, eta=rho=10⁻¹² |
| native Rz | strict error 10⁻⁶, 固定backend・seed・phase規約 |
| provider error | delta=10⁻⁶という仮想parameter。達成する物理providerは未取得 |
| estimator | Re/Im各epsilon=1/200、Bernstein sufficient-shot policy |
| failure | estimator16 axesで0.049、resource8 rowsで0.001 |
| execution | cold128＋同一trace warm128を各8 row、実測量子shots0 |

P3とP5はp、x、label数、次数が異なる。「mだけを変えると符号が変わった」と解釈しない。両者とも既知developmentであり、held-outとはしない。[R1][R2]

## 5. 保存されたP5経済結果の評価

| 方式 | Rz T / 2 axes | 期待accepted calls | N / axis |
|---|---:|---:|---:|
| ordinary | 327,368,694.046 | 2,390,744.000 | 1,195,372 |
| partial-return＋tail | 242,737,873.171 | 1,821,082.000 | 910,541 |
| P3 closed＋tail | 251,054,974.049 | 1,791,866.000 | 895,933 |
| full return | 241,470,947.527 | 1,770,981.710 | 1,032,322 |

T_Rzはprovider/preparationを含まない。sourceの総条件付きTは

\[
G_a(\boldsymbol\tau,\tau_{\rm prep})
=T_{\mathrm{Rz},a}+\sum_i A_{a,i}\tau_i+K_a\tau_{\rm prep}.
\]

P5 full-minus-partialの表示値は

\[
\Delta G=-1,266,925.644-13,539.434\tau_0-34,273.218\tau_1
-96,432.206\tau_2-50,100.290\tau_{\rm prep}.
\]

すべて負なので、同じerror上界とsamplingを固定したこの加算会計では、任意の非負provider/preparation価格に対して符号は変わらない。これは条件付きの有効な比較であり、providerを0と置いて作った勝利ではない。[R1]

ただし、providerごとのnative最適化、異なるerror allocation、総costに応じたproposalの再最適化まで含む結論ではない。

表示値からのRz削減率はordinary比26.23884%、partial比0.521932%、closed P3比3.817501%。最も近い対照との差を隠してordinary比だけを強調しない。percentageの大小だけで採択閾値を作らない。

## 6. Cauchy下界が支持する強い、しかし狭い主張

固定representationの非負rational係数をa_e、proposalをq_e、正の保存Rz価格をC_e、共通残余精度をsとする。

\[
m_2=\sum_e a_e^2/q_e,\quad L=\max_e a_e/q_e,\quad
N=\left\lceil\ell(2m_2/s^2+4L/(3s))\right\rceil .
\]

Cauchy–Schwarzより

\[
m_2\,\mathbb E_q C\ge\left(\sum_e a_e\sqrt{C_e}\right)^2.
\]

したがって

\[
G_{\rm Rz}=2N\mathbb E_q C\ge
4\ell\left(\sum_e a_e\sqrt{C_e}\right)^2/s^2.
\]

rangeとceilを落とす方向は下側であり、この使い方は妥当である。G8のsourceはrootとlogのlower endpointを用いている。[R5]

| P5 representation | q∝a/√Cの有限候補費用 | 固定policy下界 |
|---|---:|---:|
| ordinary | 327,407,968.637 | 326,612,794.345 |
| partial-return＋tail | 242,758,707.143 | 242,073,990.858 |
| closed P3＋tail | 251,094,683.335 | 250,398,296.435 |
| full return（I2 oracle） | 239,936,891.036 | 239,306,711.180 |

local fullの241,470,947.527は3対照の下界を下回る。最も近いpartial下界との差は603,043.331、下界比約0.249115%。これは「一つのIS候補を試しても追いつかなかった」より強く、**その固定係数・価格・policy内で、proposalだけを変えてlocal fullへ追いつくことはできない**という説明になる。[R1][R5][R7]

### 6.1 混合に関するレビュー導出

同じ残余精度sを使い、3つの固定representationをtag付きeventとして非負混合すると、K=Σa√CはK(θ)=Σ_j θ_jK_jである。したがってmin K(θ)=min K_j。この非常に限定されたconvex mixtureには、同じ最低下界を延長できる。

これは新precision、再合成、cross-representationのsigned collection、別dictionary、旧B2/K3の許容残差classを含まない。G8がこの混合定理を独立認証したという意味でもない。最終論文化時には前提と短い証明を付ければよく、これだけのために巨大LPを再開する必要はない。

### 6.2 二つの量化を混ぜない

G8の結果は、

- 固定production law同士なら任意の非負provider価格でfullが有利。
- providerを含まないRz endpointなら、3対照の任意proposalに対して下界分離。

の二つである。この二つを合わせて「任意のprovider価格・任意の総cost-aware proposalにも勝つ」とは言えない。

provider込みでは

\[
C_e(\tau)=C_{\mathrm{Rz},e}+\sum_i n_{e,i}\tau_i+\tau_{\rm prep}
\]

が平方根の中に入り、各対照の最適proposalが変わる。総費用版の下界・有限candidateを別に評価する必要がある。

### 6.3 fixed-policyと物理的下界の違い

この下界は、特定の十分shot数を使って予算を決める方式の下界であり、どんな正しい推定法にも必要なshot数の下界ではない。より鋭いbias、別confidence方式、stratification、既知寄与の除去、state-dependent varianceは含まれない。

表示値による感度として、最も近いlowerとlocal fullの比を二次モーメント項だけで説明すると、s=0.004952に対して約6.18×10⁻⁶の残余精度差で下界の差に相当する変化が生じる。これは実際に対照の誤差を改善できる証拠ではない。common-biasという比較契約を外した場合に、0.249%の分離を無条件な頑健性と扱えないことを示す。

## 7. on-demandの成果と、完了保証の不足

G7は全small-supportからkey inventoryを作っていた。G8はproduction中の要求だけからkeyを取得しており、この変更はsourceでも確認できる。[R3][R4]

ただし、row cacheは8 entries、共有acquisition memoは32 entriesである。共有memoはevict/resynthesizeせず、未知keyが上限を超えれば停止する。したがって、row LRUがboundedであることを、一般入力・任意試行数でのbounded-memory完遂の証明と呼べない。[R4]

今回のtraceではrow eviction0、warmは同じ128試行を再生、full P5で見たangleは6種類。G7の同supportには10種類がある。warm miss0はその再生traceでの再利用確認であって、定常的なmiss率・全support探索率ではない。[R1][R11]

一方、未観測4 keyを「存在しない」「未push」と扱わない。同じ入力のG7 inventory・合成列に記録がある。固定G7/G8の全方式key集合は24で、G8の共有memo上限32より小さい。この事実は、**この固定対象の容量見通し**を与える。ただし、G8で全keyをlive取得したことや、一般入力へ同じ上限を使えることは証明しない。

任意の固定lawでkey kの一試行出現確率をr_kとすれば、M回で少なくとも一度必要になる確率は1−(1−r_k)^Mである。無制限の再利用memoなら期待unique key数はその和になる。G8はこの全r_k分布や大きなMでの総取得費用を評価していない。この式は一般的な占有確率の説明であり、今回実測したmiss数ではない。

## 8. 推定誤差、量子call上限、技術abortの区別

G8のestimator failureは0.049、resource tailは0.001で、合計0.05。accepted countのBernstein境界は、独立uniform bitsと固定lawの下で成立する。[R2][R6]

P5 fullの比較は次のとおり。

| 指標 | full | closed P3＋tail |
|---|---:|---:|
| 期待量子calls | 1,770,981.710 | 1,791,866 |
| 確率付きaccepted上限 | 1,788,076 | 1,791,866 |
| 全試行の絶対上限 | 2,064,644 | 1,791,866 |

確率付きaccepted上限は約0.2115%小さいが、絶対上限は約15.2231%大きい。accepted count上限を総T費用上限と同一視しない。

合成timeout、未知key上限、memory cap、provider未取得は、上記0.05とは別のtechnical abortである。abort時にprefixを採点しないことは正しい。しかし「0.95以上で完全な数値回答が出る」と述べるには、技術abortの確率・排除条件も必要になる。推定誤差保証とアルゴリズムの完了保証は別である。

初期研究のprototypeに全入力での無条件termination証明を要求して直ちに失格とする必要はない。論文では、保証する入力・synthesis手続・成功条件を明示し、実証範囲と一般保証を区別する。

## 9. providerの有限誤差は「仮定を戻した」のであって「達成した」のではない

G8はunitaryなcontrolled-Q近似、strict joint error≤delta、actual adjoint、relative phaseの保持を仮定する。

\[
\delta_e\le2\epsilon_{\mathrm{Rz}}+\sum_i n_{e,i}\delta_i,
\]

\[
b_\delta=2\{3\rho+3(1+\rho)[2\epsilon_{\mathrm{Rz}}+(m+1)\delta]\}.
\]

これをtarget係数でbiasへ戻す設計は適切であり、proposal頻度をbiasの重みへ誤用していない。[R2][R6]

しかし、delta=10⁻⁶を達成する実providerはない。errorとnative priceを独立の変数として置いた条件付き分析から、実回路の費用・誤差・workspaceが同時に満たせるとはまだ言えない。

特に、近似providerでのnative-level相殺、basis変換の再利用、controlled化の方法を変える場合、G8の加算T vectorと同じ費用であるとは限らない。次に実providerを扱うなら、generic helper templateだけを全方式へ強制するのではなく、与えられたV†PV構造からのdirect実装も対称に認める。

## 10. レビュー導出：P5にもO(L²)の閉形式特殊化がある

以下は今回の新しい整理である。G8原結果でも、既知文献の新規性確定でもない。G6の既知Green再帰の有限次数展開から導く。[R12]

### 10.1 長さlのwordに対する一回returnの係数

\[
F_i(z)=p_i z\{1+(\chi-p_i^2)z^2+O(z^4)\},\qquad
G(z)=1+\chi z^2+O(z^4).
\]

reduced word u=(i_1,...,i_l)について

\[
P_{l+2}(u)=p(u)\left[(l+1)\chi-\sum_{t=1}^{l}p_{i_t}^{2}\right].
\]

これは係数のscalar積を展開した式であり、演算子Q(u)の順序は入れ替えない。

### 10.2 P5のparent群

`t_n=x^n/n!`、`mu_k=Σ_i p_i^k`とする。

empty parent：

\[
a_0=1-\chi t_2+(2\chi^2-\mu_4)t_4,
\]
\[
s_0=t_1-(2\chi-\mu_3)t_3+
(5\chi^2-4\chi\mu_3+2\mu_5-2\mu_4)t_5.
\]

これはG8の`p5_root_formula`とも一致する。[R6]

rootの各child係数も閉形式で、

\[
a_i=p_i\{t_1-(2\chi-p_i^2)t_3+
(5\chi^2-4\chi p_i^2+2p_i^4-2\mu_4)t_5\}.
\]

その和がs_0であり、rootのchild分布はa_i/s_0。ここでも全word表は不要である。

長さ2のparent u=(j,k), j≠k：

\[
a_{jk}=p_jp_k A_{jk},\quad
A_{jk}=t_2-t_4(3\chi-p_j^2-p_k^2),
\]
\[
s_{jk}=p_jp_k S_{jk},
\]
\[
S_{jk}=(1-p_j)t_3-t_5\{(1-p_j)(4\chi-p_j^2-p_k^2)-(\mu_3-p_j^3)\}.
\]

従ってd_jk=p_jp_k sqrt(A_jk²+S_jk²)、tan(phi_jk)=S_jk/A_jk。

長さ4のparent u、先頭j：

\[
a_u=t_4p(u),\quad s_u=t_5(1-p_j)p(u),\quad
\tan\phi_u=x(1-p_j)/5.
\]

### 10.3 normalizerを全parentなしに計算する

r_(4,j)を、長さ4のraw-reduced wordで先頭がjのIID質量とする。wordを反転してもIID積は不変なので、G8の末尾label再帰で得るv_(4,j)と一致する。

\[
\boxed{
B_5=\sqrt{a_0^2+s_0^2}
+\sum_{j\ne k}p_jp_k\sqrt{A_{jk}^2+S_{jk}^2}
+\sum_j r_{4,j}\sqrt{t_4^2+t_5^2(1-p_j)^2}.
}
\]

群の数は1+L(L−1)+L=L²+1以下。有限precision rootのbit費用を別にすればO(L²)の群係数・root計算で足り、L⁴ parentやL⁵ raw wordを列挙する必要はない。

### 10.4 同じ理想ensembleのcanonical生成

empty、各(j,k)群、長さ4の先頭j群をnormに比例して選ぶ。長さ2のchild i≠jは

\[
p_i\{t_3-t_5(4\chi-p_j^2-p_k^2)+t_5p_i^2\}
\]

へ比例する。0<x≤1なら括弧の定数部分は正なので、除外j付きp分布とp³分布の正混合として扱える。全j,k,i表を作る必要はない。

長さ4のwordは、残り長と直前labelに対する「隣接一致がない語のcompletion mass」の動的計画法で生成する。ここで扱うのはraw-reduced wordだけであり、一般stack reductionをlast labelだけで置き換えているのではない。

group normalizer、conditional probabilities、補正weightはfinite-bitで近似・認証する。**理想ensembleが同じことと、G8保存rational係数がbyte-exactで同じことは別**。実装時は同target・同精度で比較し、必要なら共通係数近似を固定する。

### 10.5 自己検算と研究上の意味

既存P5入力p=(1/5,3/10,1/2),x=5/7で、長さ0〜5のraw形式wordを参照に、6つの(j,k)群と31parentの係数をFractionで確認した。10種類のtangentはG7 inventoryと一致した。Bの表示小数はsqrtを使った表示であり、費用の認証結果ではない。

\[
B_5\simeq1.288444557674533,\quad U\simeq1.2967558748239574,
\quad U/B_5\simeq1.00645066.
\]

これは「G8が失敗」という意味ではない。**P5で示された利益の中心は全return representationであり、一般Green-query・zero-fillの採用だけが利益の必須原因ではない**ことを明確にする。

特殊化は同じ研究familyのfast pathとして採用できる。別の既知法が全methodを吸収したという結論ではない。一般mのアルゴリズム価値を示すには、一般次数の複雑性・取得費用を別に説明すべきであり、有限一例の勝敗だけに新規性を置かない。

## 11. 先行研究と新規性の評価

### 11.1 今回外部確認した範囲

- Cugini–Atif–Subaşı：arXiv v1のTheorem 1 Eq.(9)–(13)、ZeroFill/Discard関連節を確認。fixed-protocol cost×second moment最適化は既知。[L1]
- CTS：出版本文のTheorem 1、MethodsのPauli collectionとMarkov/layering記載を確認。全K^M展開だけを唯一の実装対照にしてはいけない。[L2]
- Wan–Berta–Campbell：出版社・著者一次記録とG6の本文監査を参照。今回はAlgorithm 2のPDF本文を改めて全読していない。[L3][R13]
- Aomoto–Kato：出版社/Numdamの一次書誌・抄録を確認。F_i/Gがその特殊化である詳細対応はG6監査に基づく。今回は1988年PDF本文を再精読していない。[L4][R12][R13]
- Zhao–Yuan：出版社の研究内容・書誌を確認。今回arXiv HTMLは取得できず、低次数吸収の具体式対応は既存G6監査と区別した。[L5][R13]
- Ross–Selinger：一次書誌・要旨のprobabilistic synthesisと仮定付き効率性を確認。特定pygridsynthの固定time cap内成功を保証する結果とは読まない。[L6]
- Koczor：一次arXiv書誌・要旨と既存監査を参照。一般的なdictionary/process合成の発想と、現在のphase-sensitive operator meanの違いを維持する。[L7][R13]

「全return」「free product」「Taylor」「Hamiltonian」等の検索も行ったが、完全な優先性調査ではない。同一名称が見つからないことを新規性の証拠にしない。

### 11.2 残る貢献候補

既知Green再帰、Euler pairing、rejection、Cauchy/IS、cache実装を個別の新規発見として数えない。

候補となるのは、**同finite P_mの全形式returnを保つrepresentationを、必要な局所情報から有限bitで生成し、未知normalizerを使わない認証予算とnative取得へ接続する構成**である。

ただし、同一構成が近接研究の容易なspecializationでないこと、物理providerまたは明確なoracle modelで追加価値があること、一般入力での費用説明があることが必要である。G8だけでは独立新規性・主methodの最終採択は確定しない。

## 12. 研究としての着地点

### 12.1 暫定的に残す主RQ

> 明示されたinvolution/controlled-rotation accessの下で、有限Taylorのreturn集約を、全word表に依存せず実行可能な乱択表現へ変換できるか。低次数特殊化・既知sampling・native実装費用を同等に扱った後、どの条件で追加価値が残るか。

これを条件付きアルゴリズム研究の候補として残す。G8の小さいT差を「実分子の一般的な改善率」にしない。

### 12.2 二種類の論文化経路

**条件付き構成・理論研究**：一般mの有限平均、非負性、local query、finite-bit、予算、明確なinput/oracle complexity、特殊化との関係を中心にする。実機実験がないだけで不成立とはしないが、既知構成の組合せを超える貢献を説明する必要がある。

**実用的なnative資源研究**：具体provider、direct構成、CTS等の強い対照、取得費用、最終taskへの接続が必要。現在はこちらの完成度に達していない。

G8までの結果から、独立論文として十分と保証しない。否定的な結果が出ても、R0–G5とG6–G8の限定成果を保存する。旧固定dictionaryの閉鎖を、新methodの都合で書き換えない。

## 13. 代替案の比較と採否

| 方針 | 今回の判断 | 理由 |
|---|---|---|
| もう一度短い同traceを動かす | 主研究作業にしない | path確認には前進があった。次は研究の採用理由を検査する |
| G8のI2 oracleへ全法を置き換える | 主線にはしない | 表accessが異なる。性能上限の補助診断として保持 |
| p/x/seed/precisionを増やして勝つ条件を探す | 採用しない | 現在の帰属・provider・低次数特殊化の不足を解消しない |
| P5特殊化を含む同target・明示provider比較 | **次の一作業として採用提案** | representation、生成器、native実装、強い対照を一つの比較へ接続できる |
| 分子・DF・PR/QPE全体へ直行 | 保留 | 現在はfinite P_m、一つのknown形式P5。targetと取得費用が未接続 |
| 理論・構成ノートとして今区切る | 代替案として保持 | nativeの追加価値が消えても正しい構成は残る。ただしG8の肯定証拠から最小native比較の情報価値はある |

## 14. 次のCodex作業案：G9を一束のmatched-native検証にする

以下は**今回の研究提案**で、G8の結果や既存科学条件ではない。実作業は別branchでsource・scopeを固定して行う。現在ここで科学runを開始してはいない。

### 14.1 科学的目的

1. P5の成功は、全集約representation、一般local generation、予算のどこに帰属するか。
2. Q本体を実装してdirect rotationとPauli情報を使える状況でも、採用する価値が残るか。

native実証は一つの小さい同target比較に限定し、同時に新p/x/m gridを追加しない。

### 14.2 先行する数学・source確認

§10のP5特殊化を、G6 evaluatorの期待値コピーではなく独立に確認する。任意Lの群分解とconditional samplerを確認し、既存P5で同じ理想ensemble・同じangle集合を再現する。有限bitではmean/bias/rangeを別に戻す。

§6の混合下界を利用する場合は前提を明示する。G8のoracle scalarは変更せず、必要な最小event bindingを同作業内で確認する。これだけのための新しい大規模最適化はしない。

### 14.3 一つの明示provider提案

抽象CQに任意のT価格を入れるだけではなく、**3 system qubits、三つのinvolutionを明示回路で与える**。

\[
R_P(\theta)=e^{-i\theta P/2},\qquad
Q_0=Z_0,
\]
\[
V_1=R_{X_0X_1}(\pi/4),\quad Q_1=V_1^\dagger Z_1V_1,
\]
\[
V_2=R_{X_1X_2}(\pi/4)R_{Z_0Z_1}(\pi/4),\quad
Q_2=V_2^\dagger Z_2V_2.
\]

演算子積は右側から作用する。各VはPauli rotationのparity ladderとT gateでexact Clifford+T記述を持つ（scalar位相はV†とVの対で厳密に相殺されるが、controlled targetの位相は別に保持する）。これは費用を取得済みという主張ではない。

選ぶ理由は、(a) 三つのlabelと既存P5を維持できる、(b) 異なるbasisと非Clifford費用がある、(c) providerを無料oracleにしない、(d) Pauli collectionも小さい計算で取得できるため、不公平にCTSを排除できない、の四点である。

このproviderは**科学的な反証・接続用の合成model**であり、実分子・DF代表性・I0取得優位を主張しない。Pauli情報が安く得られる厳しい対照contextである。今回、このproviderの行列・費用・勝敗は計算していない。

p=(1/5,3/10,1/2)、x=5/7、m=5を維持し、full operator P5をtargetにする。入力状態の簡単な古典解へtargetを置き換えない。新providerなのでforward validationだが、p/x/mやangleは既知developmentであり、完全なheld-outとは呼ばない。

### 14.4 比較集合と公平性

比較するのはordinary、partial-return＋tail、closed P3＋tail、G8型local full、P5 closed full特殊化、および同P5へfinite-specializeしたmatched CTS。

P5 closed fullは同研究familyのfast pathであり、独立した既知競合法として新規性を採点しない。G8型との違いは生成・normalizer・予算の会計として分解する。

全方式へ同じ公開provider情報を与え、以下を同等に適用する。

- V† controlled-P rotation Vというdirect実装を許す。generic helperを主対照へ強制しない。generic helperはG8からの接続診断として分離してよい。
- controlled relative phase、actual adjoint、workspace、strict error、state preparation/readoutを記録する。
- 同じcompiler/簡約規則、同じprimitive精度を使う。全回路最適化を一方だけへ適用しない。
- 既存合成列はidentityが一致する場合のみ再利用し、未取得CTS角度等だけを結果前inventoryへ固定する。結果を見てangle/precision/backendを追加しない。
- cost-aware samplingを比べる場合、実provider込みの同じ価格を使う。Rz-only最適ISを総T最適と呼ばない。列挙oracleは補助として費用/accessを分ける。
- CTSはchannel equalityだけでなく同じfull first operator momentを認証する。全return側だけがPauli相殺を使わないことは構成の性質であり、実用比較でCTSを排除する理由ではない。

### 14.5 予算・実行範囲

Re/Im各1/200、T-primaryを維持する。failure配分は新しい比較row数に対して結果前に固定し、G8の16 axes配分をそのまま違う集合へ流用しない。新しい5%/10%閾値は設定しない。

明示providerがexactならδ=0という認証済みの条件を使える。これをG8の仮想δ=10⁻⁶達成実験と呼ばない。実装に近似を導入するなら、そのstrict誤差を同一契約で戻す。

accepted-call capと総T capは分離する。技術abortを含む成功条件、cacheを超える際の処理、完了したprefixだけを採点しないことを結果前に固定する。現在の24/32 key関係を一般規模の保証へ転記しない。

小systemなので、正式targetのoperator/phaseの参照検証、全event reference会計を行ってよい。ただし、reference表をlocal productionへ注入しない。百万shotsを実機またはtrajectoryで実行することは、この最小資源比較の必須条件ではない。

実装script、テスト、有限bit選択、必要最小限の証拠形式、合理的なcall/time/memory capsはCodexへ委ねる。ただし、provider・target・比較集合・主要指標・評価精度を勝敗に応じて変更しない。scope超過・数学反例・source mismatchがあれば拡張せずSTOPして返す。

### 14.6 結果を見てからの判断

- direct native費用とCTS後にも候補の有効点が残る：一般次数・入力構造のvalidationを設計する価値をGPTで判断する。世界初・独立論文十分性を自動採択しない。
- P5 closed特殊化がG8型より有利：一般法の失敗とせずfast pathの選択原理として整理する。小mでは特殊化を使い、一般法はgeneral-m complexityの役割として評価する。
- CTSが指定native contextで有利：I1が利用可能なこのcontextでは既知法を採用する。G6の形式恒等式や一般I0全体を否定しないが、実用優位のclaimは縮小する。
- technical failure・非常に保守的なboundsで判別不能：数学的反証、取得失敗、未指定情報を分け、条件を増やして勝ちを探さない。

全作業を一束で完了・公開した時点でmandatory STOPし、GPTへ研究判断を戻す。正常な技術修正ごとに新しいGPTレビューを挟まない。

## 15. 今回の検算資料と再現性

付属`review_checks.py`はrepository moduleをimportしない。次を行う。

1. G8 handoffの表示値から削減率・affine差・cap差を計算する。入力値は丸め表示なので認証用exact sourceとは区別する。
2. 既存P5入力で0〜5次のraw wordを小さく列挙し、first-adjacent-pair deletionで参照係数を作る。
3. 今回のroot、長さ2、長さ4の閉形式と比較する。係数一致はFractionで確認する。
4. B5、Uの表示を算出する。平方根表示はfloatであり、これを新T優位の証明に使わない。

ファイルは`review_checks_result.json`、`README.md`、本レビュー、検算scriptとして同梱する。原repositoryの全result/sourceをzipに複製したものではない。元証拠は下の固定URLを正本とする。

## 16. 最終判断の要約

G8は、G7で残ったon-demandの接続不足を、限定した実装証拠として前進させた。P5のRz-only固定policy下界は、samplingだけでは説明できないrepresentationの追加価値を示す。

その一方、物理providerは未取得、現在のtraceは短く、全工程の汎用完了・native優位・独立新規性は未確定である。また、P5にはO(L²)の特殊化があり、一般生成器の実用上の役割を切り分ける必要がある。

**条件付き候補として継続する。次はP5特殊化と一つの明示provider/CTSを含む比較をまとめ、研究の採用理由を直接検査する。** 同じ短traceの反復や、勝つ入力を探すgridを主研究にはしない。

---

## 参考資料：repository（固定commit）

[R1] [G8 handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md)

[R2] [G8 proof/contract/scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md)

[R3] [G8 runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/scripts/tracks/algorithm_codesign/g8_on_demand_provider_budget.py)

[R4] [G8 pipeline/cache](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/src/trottertracks/algorithm_codesign/g8_pipeline.py)

[R5] [G8 reference oracle](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/src/trottertracks/algorithm_codesign/g8_reference_audit.py)

[R6] [G8 bounds](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/src/trottertracks/algorithm_codesign/g8_bounds.py)

[R7] [G8 saved audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/artifacts/track_b_g8_on_demand_result/2026-10-10/v1/saved_output_audit.json)

[R8] [G8 manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/artifacts/track_b_g8_on_demand_result/2026-10-10/v1/evidence_manifest_v1.json)

[R9] [G8 tests](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/tests/tracks/algorithm_codesign/test_g8_on_demand_and_bounds.py)

[R10] [G8 raw result](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4b410746aadcf03c955b5e961c672c04b220de5/artifacts/track_b_g8_on_demand_result/2026-10-10/v1/result_v1.json)

[R11] [G7 angle inventory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/artifacts/track_b_g7_budget_control_preparation/2026-10-10/synthesis_key_inventory_v1.json)

[R12] [G6 independent mathematical audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md)

[R13] [G6 prior-art audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md)

## 参考資料：外部一次文献

[L1] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1. [本文](https://arxiv.org/html/2603.13495v1)

[L2] Peetz, Smart, Narang, *Quantum simulation via stochastic combination of unitaries*, npj Quantum Information 12, 52 (2026). [出版本文](https://www.nature.com/articles/s41534-025-01168-w)

[L3] Wan, Berta, Campbell, *Randomized Quantum Algorithm for Statistical Phase Estimation*, Physical Review Letters 129, 030503 (2022). [出版社](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.030503)

[L4] Aomoto, Kato, *Green functions and spectra on free products of cyclic groups*, Annales de l'Institut Fourier 38(1), 59–85 (1988). [一次記録](https://www.numdam.org/articles/10.5802/aif.1123/)

[L5] Zhao, Yuan, *Exploiting anticommutation in Hamiltonian simulation*, Quantum 5, 534 (2021). [出版社](https://quantum-journal.org/papers/q-2021-08-31-534/)

[L6] Ross, Selinger, *Optimal ancilla-free Clifford+T approximation of z-rotations*, Quantum Information and Computation 16, 901–953 (2016). [一次arXiv記録](https://arxiv.org/abs/1403.2975)

[L7] Koczor, *Sparse Probabilistic Synthesis of Quantum Operations*, PRX Quantum 5, 040352 (2024). [一次arXiv記録](https://arxiv.org/abs/2402.15550)

[L8] Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum 7, 020332 (2026). [出版社](https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq)。PR/QPE全体が別の最終taskであることの背景資料。本レビューで44頁を再精読したとはしない。
