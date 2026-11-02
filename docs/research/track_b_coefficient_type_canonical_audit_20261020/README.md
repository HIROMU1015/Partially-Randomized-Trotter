# Track B 案B：exact係数型による一般次数canonical RAの事前監査

2026-10-20 JST。科学SOURCE基点 `243eea8e840f3c472753f7935e510391626fd325`、案A監査 `df90ed8c8e847f8ea1861d9761b9515e7a895916`。今回の利用者指示による一次エージェント自身の紙上監査であり、別AI・外部査読者のレビューではない。

**結論：B1・B2は前提付きACCEPT。B3の数式はparent-count領域の上界として正しいが、算法全体の状態数・費用と区別する修正を要するためREVISE。理想canonical構成に破綻は見つからなかった。数え上げ・completion抽出・係数型共有は既知手法の直接適用で説明でき、これら単体の独立新規性は支持しない。一般次数へのRA接続は具体的な方法上の差になり得るが、原著の中心とする十分性・実用性はUNRESOLVED。**

対象は**exactに等しい有理係数型のみ**。B4〜B6の近似型拡張を監査・採択していない。科学計算、保存値数値再解析、simulation、benchmark、科学SOURCEのimport/実行/変更、sampler/DP実装、科学test、回路build/compile/synthesis、新しい非自明な定理・算法の創作、Full-N/追加Pilotはすべて0。既存grant/claim/ledger/marker/STOPは変更していない。

## 1. 前提とB1〜B3の判定

有限のlabel集合を `I_t` に分割し、型数 `1≤K≤L`、`L_t=|I_t|≥1`、`p_i=w_t>0 (i∈I_t)`、異なる型のwはexactに異なり、`Σ_t L_t w_t=1` とする。`w_t,x` は明示有理数、`0<x≤1`、mは正の奇数、`σ=±1`、各 `Q_i` はHermitian involution。零係数labelは入口で除外し、対応表を保持する。[MATH] §§1–8、[F1] §1。

対象は `R=Σ_i p_iQ_i` の同じ有限 `M=P_m(−iσxR)`。語の同label隣接消去だけを使い、追加のPauli関係やQの交換則は仮定しない。`x=0` はidentity別経路。nは**reduced parentの型別出現回数**であり、raw Taylor語の出現回数ではない。

| 式 | 判定 | 根拠と限定 |
|---|---|---|
| B1：`D_s(n)=L_sΣ_(t≠s)D_t(n−e_s)+(L_s−1)D_s(n−e_s)` | **ACCEPT** | head labelを一つ外したsuffixのfirst型で場合分けできる。Dはlabel付き語の**整数個数**。base `D_s(e_s)=L_s`、負成分・不可能状態は0。空語は別に一個 |
| B2：`B_full=d_∅+Σ_(even 2≤|n|≤m−1)Σ_s D_s(n)d(n,s)` | **ACCEPT** | 各非空parentは一意の `(n,first型s)` に属し、同じ群ではa/s/dが一定。D=0群を除けば各parent質量を一回ずつ再構成する。空語をDの和へ重複させない |
| B3：`1+K binom(m−1+K,K)` | **REVISE** | `|n|≤m−1` のhistogram数へのstars-and-bars上界として正しい。だがchild係数は `|n|≤m`、各状態のtransition/系列演算・root・精度・出力は別。全前処理時間や全メモリをこの一式と同一視しない。§5に修正範囲を示す |

B3に数え上げの反例を発見したという判定ではない。REJECTすべき原式は見つからなかった。ACCEPTは未作成のgeneratorや一般原著価値への認証ではない。

## 2. 係数共有と数え上げの紙上確認

### Scalar圧縮の範囲

既存の次数順Green再帰は、同じpを持つlabelの置換に不変である。低次数係数からの帰納で `F_i=F_t (i∈I_t)`。全labelのweighted totalを `C(z)=Σ_t L_t w_tF_t(z)` とすれば、既存式は

\[
F_t=\frac{zw_t}{1-z(C-w_tF_t)},\qquad
G=\frac1{1-zC},\qquad
H_n=G\prod_t F_t^{n_t}\pmod{z^{m+1}}.
\]

これは既存 [KERNEL] 136–160行の置換・線形集計であり、新Green定理ではない。**K個のlabelへ確率 `L_tw_t` を与えて元のReturnKernelを呼ぶこととは違う**。同型の異なるlabel間のexcursionと非消去の遷移が失われるからである。

`a(n)` は長さ `|n|` 用の既存signed Taylor汎関数をH_nに作用させたもの。合法な語なら順序や同型labelの具体的選び方に依存しない。非空parentのfirst型sに対して

\[
S_{n,s}=\sum_t(L_t-\delta_{ts})a(n+e_t),\quad
d_{n,s}=\sqrt{a(n)^2+S_{n,s}^2}.
\]

除外するのは**firstの実label一個**で、型s全体ではない。D=0のhistogramの形式積を実parent係数として解釈せず、質量0群やmultiplicity 0の項で不要な比を評価しない。空parentでは除外なし、`S_∅=Σ_tL_t a(e_t)`。aは既にpの積とreturn寄与を含み、B1/B2へ別の `∏w_t^{n_t}` を重ねて掛けない。

Histogram＋first型で共有できるのはa/S/d/angle tangent等のscalarである。**`Q(u)`、firstの実label、childの実label、controlled相対位相と語順は共有・同一視できない**。同型のQが物理的に等しいことも、同じnative価格であることも要求・認定しない。[MATH] §§5–6、[PROVIDER] 2–22行。

### B1の重複・欠落

長さ2以上の語のhead型をs、残りのsuffixのhead型をtとする。`t≠s` ならs型のheadはL_s通り、`t=s` ならsuffixの最初の実labelを除きL_s−1通り。head削除と再付加が一対一なのでB1となる。Dはunweighted integer countであり、probabilityやweighted DPへ読み替えない。

| 境界・落とし穴 | 紙上確認 |
|---|---|
| base/空語 | `D_s(e_r)=δ_sr L_s`、`D_s(0)=0`。空parentは別の一個。`n_s=0` の非空s群は0 |
| 同型の異なるlabel | `D_s(2e_s)=L_s(L_s−1)`。型が同じという理由で隣接を排除するとこの項を失う |
| 再使用 | 同じ実labelも非隣接なら再使用可能。`n_t≤L_t` というinventory制約を追加してはいけない |
| K=1 | 長さl≥1の語数は `L(L−1)^(l−1)`。L≥2なら交互・反復語がある。L=1なら長さ2以上は不可能でcanonical parentは空語のみ |
| m=1 | 非空parent群はなくroot一個。`a_∅=1`、`a_i=xp_i` で既存Euler pairingへ戻る |
| x=0/zero群 | M=Iへ分岐しrandom draw・root比・angle・weight除算を行わない。D=0や不可能completionを抽出候補に入れない |

B2はこの群分割とscalar一定性を代入した有限和の組替えである。正値性と `B_full≤B_o<3` は既存short-step証明を用いる。[MATH] §§4,6–7。L/K/mの全域で新しい正値性定理を必要とする欠落は、今回の確認では見つからなかった。

## 3. Label付きordered canonical lawの保存

群 `g=(n,s)` の個数を `D_g`、一parentの質量をd_g、群質量を `c_g=D_gd_g` とする。rootは `D_∅=1`。群を `c_g/B_full` で選び、その群の**実label付きreduced語**を一様 `1/D_g` で選ぶことが必要である。

型列を一様に選んで実labelを後付けするだけでは一般に一様にならない。型列 `(t_1,…,t_l)` のlabel付け数は `L_(t_1)∏_(j=2)^l(L_(t_j)−δ_(t_j,t_(j−1)))` と変わる。既知recursive methodのcompletion比による抽出が必要である。[RECUR] §§2,4。

B1のDはfirst型別の集計であり、これを無条件に次labelの確率にしてはいけない。型sの特定first実labelに固定した語数は対称性から `D_s(n)/L_s`。suffixを伸ばすときは**remaining histogram＋境界の前label型**に対するcompletion countを使い、実label選択では前labelそのものを一個除外する。これらは有限状態・計数DPの通常の条件付けであり、新しいDPを実装・考案していない。prefixのcompletionをK型で共有できることと、実labelの除外情報を捨てられることは違う。

既知completion抽出の正しさは、各prefixで選ぶ次の実labelについて `残りcompletion数/現在completion数` を掛けると、最終値が `1/D_g` へtelescopingすることで確認できる。最初のs型実labelも一様 `1/L_s` とする。空群・0 completionは選ばない。

childは型tを `(L_t−δ_ts)a(n+e_t)/S_g` で選び、その型の合法実labelへ一様に割り当てれば、各childの確率は元の `a_iu/S_g`。したがって

\[
P_{\rm can}(u,i)=\frac{D_gd_g}{B_{full}}\frac1{D_g}\frac{a_{iu}}{S_g}
=\frac{\alpha_{u,i}}{B_{full}},\qquad \alpha_{u,i}=d_g a_{iu}/S_g.
\]

既存 `V_(u,i)=(−iσ)^|u|exp(−iσφ_gQ_i)Q(u)` とweight B_fullで同じMを再現する。operator左から右とcircuit時間逆順、偶数parent位相を保持する。[MATH] §6。

これは**Full Returnのaccepted eventを正規化したcanonical law**である。G6/G7の全trialはraw-zero/coin rejectを含み、event確率は `α/B_o`、受理率は `B_full/B_o`。両者の全trial lawは同じではない。canonicalの理想second momentは `B_full²`、zero-fill全trialの値は `B_oB_full`。比較の試行単位・normalizer取得費用・必要Nを揃えず量子総費用の勝敗を結論しない。[V11] §§3,5、[A] §§2–4。

## 4. 有限bitの成立条件と未閉鎖点

B2は平方根和を**特定・取得する表現**であり、一般にはexact rationalなBを与えない。理想的な実数samplingと、固定bit版を区別する。群CDFは一般にirrationalなので、案Aのexact rational CDF比較をそのまま無料で使えない。既存canonical/P5の外向きroot区間・正規化区間・dyadic実proposalという手段は利用できる。[P5] 40–47行、[GEN] 137–143行、[F1] §§3,5。

群質量の区間を `c_g∈[c_g^lo,c_g^hi]`、有理代表値をc̃_gとし、全存在群で `|c̃_g−c_g|≤ρc_g` を認証する。群内の**理想**語比率 `r_g(u)=1/D_g`、理想child比率qを使って

\[
\widetilde\alpha_{u,i}=\widetilde c_g\,r_g(u)q(i|u),\qquad
\widetilde W_{u,i}=\widetilde\alpha_{u,i}/\pi_{u,i}
\]

と置く。πは丸め後の群・全completion/type/label・childの**実proposal因子**の積で、ideal count比をπへ代入しない。この既存IS恒等式なら `πW̃=α̃` はexact rational、global L1 biasは `Σ_g|c̃_g−c_g|≤ρB_full`。exact Mへの一致ではなく、M̃への一致とMへのbias保証である。[BIT] §§3–6、[V11] §5.1。

**Multiplicityとprecision：** d_gの絶対幅Δをそのまま群へ移すと幅はD_gΔ、総幅は `Σ_gD_gΔ` へ増える。「群数だけ×Δ」とはならない。群質量を `√(D_g²(a_g²+S_g²))` として囲む場合も、D_gのbit長を課金する。群相対ρとnormalized group lawの相対誤差を別に認証し、既存P5のgroup-root保証を新入力へ自動認定しない。

各存在proposal因子がidealの `(1±η)`、`0<η<1` に入り、その因子数の一様上界をFと明記すれば、既存の因子積による直接会計から

\[
|\widetilde W|\le\frac{(1+\rho)B_{full}}{(1-\eta)^F},\quad
\|\widetilde M-M\|\le\rho B_{full},\quad
E[\widetilde X^2]\le\frac{(1+\rho)^2 B_{full}^2}{(1-\eta)^F}.
\]

認証予算には外向き `B^hi` を使う。angle/nativeのstrict誤差、測定task固有bias、正remaining、confidence・IID bits/測定・全N完了は別条件。F3や旧Pilotの成功を新版へ移さない。

Fは抽出の分け方による。群一回＋各実label一回＋child一回なら `F≤m+1` の数え方ができる。一方、型と実labelを別々に丸めてdrawするなら、一parent labelあたり二回・childも二回で保守的に `F≤2m+1`。**旧 `m+2` を無条件に流用しない**。固定bitでの一様1/Dや1/L_tも一般には非dyadicで、実確率補正または別の承認済み乱数仕様が必要である。

必要precisionの規模は既存下界から点検できる。`d_min=t_(m−1)p_min^(m−1)/2`、`t_l=x^l/l!` とすると、存在群の `c_g≥d_min`、`c_g/B_full≥d_min/3`。正の整数completion比は粗く `≥L^(−m)`、一様実label選択は `≥1/L`。存在childの下界は `p_min/2`、child型の確率も少なくとも一つのchildの質量比以上。これらのlog inverseは入力bit長と**数値m**に多項式で、固定Kでは群数も多項式である。F1のnormalized interval受入述語と既存relative-root条件を使う余地があり、直ちに超多項式bit精度が必要という障害は見つからなかった。

ただし、**案B全到達lawの固定precision preflight・実law binding・terminal failure・Evidence・実装は未作成**。本監査はH/K_rootを選ぶselectorや新しい一様定理を作っていない。群の区間とcompletion/labelの相対条件を具体化する前に「F3認証済み」としない。これ以上の非自明な保証が必要になればUNRESOLVEDとして別承認へ返す。

## 5. 状態数・計算量・memory

`N_p=binom(m−1+K,K)`、`N_c=binom(m+K,K)`、M_gをD>0のeven parent群数（root込み）とする。`M_g≤1+K N_p`。N_pは全次数≤m−1のhistogram個数で、oddな中間count状態も含む上界。D>0・even・`n_s>0` 等で実数は減る。**B3はこのcount/group領域に限定して記述する。**

| 処理 | 条件付きの算術・保持量と注意点 |
|---|---|
| exact型分類 | 明示p表readはΩ(L)要素。正規化した有理数の比較・分類と実label↔型の対応表を課金。通常sortならO(L log L)比較。小Kだから入力やoperator表をsublinearに読めるわけではない |
| Green | 型ごとの既存次数再帰とweighted total共有ならO(Km²)有理演算・O(Km)係数。現行ReturnKernelはL本のF_iを作るためそのままではO(Lm²)/O(Lm)。型化済みと見なさない |
| B1/count/completion | count表は高々O(KN_p)整数。原式の型sumを各stateで直接行えばO(K²N_p)算術。総和・completionの合法cacheを双方に認めれば改善余地がある。状態数だけを処理回数にしない |
| scalar a/S/d | child係数用に次数mまで必要。全a(n)を独立な既存畳込み積で評価する粗い上界はO(N_c m³)有理演算。Sは各存在群にK項でO(M_g K)。系列cache/共有積で改善可能だが、未実装の最適上界を取得済み扱いにしない |
| normalizer/group law | M_g個の群質量root enclosure、その区間和・normalized interval・dyadic lawが必要。largest-remainderならO(M_g log M_g)比較等を追加。Bを無料oracleにしない |
| 語・child抽出 | groupは保存済みlawの線形走査ならO(M_g)、通常selectorならO(log M_g)等。completionによるtype選択は素朴にO(mK)、実label選択には対応表とfirst/previous実label一個の除外が必要。全Lの配列を毎段再構築する実装ならO(mL)項が残る |
| root/angle/weight | 群rootはsetupで、angleを群ごとにcacheするなら全M_g回の取得・保持をsetupへ課金。実πとW̃、native provider/IR/strict誤差・T/CX/depthは別費用 |
| Evidence/保存 | 入力、型対応、D、係数、群質量区間・lawへの結合、prefix count比と実π、実ordered eventの証拠が必要。毎trialの全表再hashや全表保存を残せば費用も残る。全N出力・失敗prefixの保存を0にしない [A] §5 |

上表は標準有限和・既存畳込みの**粗い上界**であり、新DPや最適算法の設計ではない。B3から単一の実秒・量子費用上界へ飛躍しない。固定KでN_p/N_cはmに多項式だが、次数がKに依存する。これはKに関するFPTを示さず、FPT算法の不存在も示さない。K=Lで一般入力に効率的との主張は不可。mをbinaryで与える場合、数値mの多項式をlog mの多項式と呼ばない。

countのbit長は `b_D=O(1+m log(L+1))`（D≤L^m）。既存係数幅は `b_c=O(1+m(H_p+H_x+log(m+1)))`、H_pは明示p表の総bit長。群radicandのD²、count比、c̃/D、実proposal積にもこの長さを戻す。F2の保守的 `Q(b)=O((b+1)³)`、`R(b,K_root)=O((b+K_root+1)³)` を使うなら、有理/整数演算数に各幅のQを、M_g回のrootに `R(b_c+b_D,K_root)` を課金する。F個のH-bit因子を持つπの分母長はO(FH)、weight長には `b_c+b_D+K_root+FH` と精度パラメータ長が入る。[F2] §§1–3。

一つの保守的memory会計は、型Green O(Km b_c)、count/completion O(KN_p b_D)、scalar表O(N_c b_c)、group/root/law O(M_g(b_c+b_D+K_root+H))、実label表O(L log L) bits、局所語・一時系列・Evidenceの別項。全H_n系列までcacheする選択ならさらにO(N_c m b_c) bitsを要する。表を持つ物量、bit数、Pythonの実bytesは異なる。全trialのsetup償却とcache再訪率を測っていない。

## 6. 強い既存法・一次研究との対応

### SOURCE上の方法差

| 固定SOURCE | 既存の到達点と案Bへの含意 |
|---|---|
| [P5] 10–47行 / 49–132行、[V11] §5 | P5Closedはm=5限定で、root・ordered pair・first-label群、weighted completion DP、group root、実proposal補正を既に持つ。高々L²+1群で、全L^4 parent列挙を比較対照へ強制できない |
| [P5] 49–79行 / 81–132行 | pair scalarはp_j/p_k等、長さ4のroot因子とangleはfirst係数に依存。exact型が同じ既存群は通常のcache/群併合を許せる。pairを型対、長さ4をfirst型で共有するscalar群数はO(K²+K+1)にできる対応が式から読める。これは既存式の同値項共有であり、新samplerを作成した意味ではない |
| [GEN] 102–212行 | ordinary/partial/closed-P3＋tailのcanonical経路は既存。一般奇数mでGreen全returnのB_fullを型群から取得する案とは異なるが、非列挙性・canonicalという名称だけは差にならない |
| [KERNEL] 136–202行 / [GEN] 215–251行 | 一般次数Fullは局所全child取得＋量子前zero-fill。案Bは有限型状態を事前に集計しB_fullを取得する代わり、ideal accepted ensembleを直接正規化する。前処理・memory・試行単位の交換であり、全用途の改善ではない |
| [F3] prepare / [F1] §5 | 現行F3の非identity preflightはFull専用。canonicalの有限lawを扱う数理条件は既存だが、新しい型群・completion law・SOURCE/policy/schema結合への実装保証はない |

P5の長さ4群はhistogram全体を分けずweighted completionでまとめられるため、案Bのhistogram分割が常に最小とは言えない。m=5・同じexact型入力で、通常最適化したP5Closed以上の成果を自動認定しない。より高い一般次数への構成的拡張は候補差だが、既知の有限状態・係数query・completion DPを同条件で使う対照にも同じ型共有を認める。

### 一次文献の重点確認

| 一次資料・今回読んだ箇所 | 重複と限定 |
|---|---|
| Freiberg–Heuberger–Prodinger [SMIR] §2 Eq.(5) | 元の実label上のSmirnov wordと多変量出現数は既知。型alphabet上で「同型隣接禁止」にしてはいけない。B1はこの既知の語計数を同一重みのlabel群へ集計した場合分け |
| Denise–Ponty–Termier [RECUR] §§1–2,4 | exact frequenciesを固定した一様生成、数表を用いるrecursive sampling、multivariate countingが既知。Histogram制約＋completion比を新しい生成原理としない。掲載された一般費用を案Bへ無条件転用しない |
| Flajolet–Zimmermann–Van Cutsem [FZV] 1993 RR-1830、導入・計算モデル | count表に基づくrecursive generationと前処理/抽出の分離が既知。取得したのは1994出版版そのものではなく著者の研究報告版。arithmetic modelの大整数単位費用をbit/CPU費用としない |
| Mohri [WA] §2、§6.3 Eq.(35)–(40) | 有限状態のpath weight和、acyclic DP、completionに相当するsuffix weightによる確率正規化が既知。型＋histogram counterの有限状態への適用は通常のもの。K状態だけで非線形d(n,s)まで表せるとはしない |
| Wang–McArdle–Berta [WMB] Appendix Bのsample-access preprocessing | 少数のexact Pauli係数値を古典取得の追加構造として使う着想は明示済み。ただし案BのGreen/histogram/canonical RA全手順を同論文が与えるという証拠ではない |
| 既存固定対応 [KNOWN]、[V11] §§8–9、案A監査 [A] | Green、Euler、LCU/RTE非列挙生成、importance補正、区間精度は既知部品。既知法にも同じGreen・cache・前処理・selectorを認める。既存G6全構成のpriority未確定を案Bの新規性へ読み替えない |

**技術的評価：** 一般次数への移行には表サイズ増大、群root精度、実label復元とphase、有限bit因子数の処理が必要。しかし今回のexact型条件では、既存Green/正値性/Euler/有限law保証＋標準count/completion法からの通常の接続で説明でき、新しい非自明な組合せ論・確率論上の困難は特定されなかった。実装が未作成であること自体を理論的新規性の根拠にしない。

全著者コード・全証明・全priorityの網羅監査ではない。一次文献は関連本文の確認で、文献に同一の公開コードが見つからないことを新規性の証明としない。独立原著として十分かはUNRESOLVEDである。

## 7. 発展価値と停止条件・次の担当

小Kのexact入力は人工例だけとは限らない。例えば均一結合と均一磁場のIsing型Pauli Hamiltonianでは、符号をQへ吸収し正規化すれば、実label数が増えても係数値は少数に留まる。[ISING] §2のuniform-coupling/field構造はその具体例である。ただしrational/exact表現は本監査の仮定で、実験の近似値を勝手に同型化しない。Isingの特別な解法・Pauli関係等も強い対照に含める必要があり、「物理例がある」だけで案Bの必要性は確定しない。

分子DF等の既存実入力に小Kがある証拠は今回取得していない。丸めてKを減らすと対象p/Mが変わり、今回は禁止されたB4〜B6の別問題となる。

| 発展させる価値があり得る条件 | 追加投資を停止・縮小すべき条件 |
|---|---|
| exact小Kが対象taskで自然に与えられ、必要な一般次数でB_fullの取得・直接canonical生成に意味がある | 小Kを作るために未承認の係数丸めが必要、K≈Lでbinomial表が大きい、または一様K=1の既知特殊化だけを独立新理論とする |
| 強いP5/closed-tail/ordinary、同情報のGreen/型共有DPに対し、何を追加する成果かを限定して説明できる | m=5の通常最適化P5や既知の同じ型集約で方法差が尽きるのに、新原理・一般優位だけを目的に開発する |
| setup/root/表容量/実π/監査を含む費用と前処理償却の意味がtaskに合う | 状態数だけをCPU speedupへ換算し、root・bit・Evidence・provider費用を落とす |
| 新しい一般次数仕様や再利用可能な型制約付き実装成果として、貢献の水準をユーザー/GPTが受理する | 理想lawの正しさ・未読文献・非同一コードの不発見だけを原著GOにする |

一般原著の中心として決定的に不足しているのは、**同じtarget/access/精度/前処理を許した既知の組合せに対して、RAへのこの特殊化が何を新たに解決し、その解決がなぜ重要か**という根拠である。全既知法への勝利・分子実験・量子資源改善を一律の必要条件として追加していない。現時点では既知DPの直接応用として、方法ノート追補・実装改善に限定する判断も支持できる。

**次の担当：ユーザー・GPT。** ユーザーは、案Bを追加研究として継続するか、既知法の適用として限定するか、投資を止めるかを判断する。原著十分性・中心claimの詳細な科学判断をGPTへ依頼する場合は、別のレビュー開始承認が必要（共通指示上Level 1、GPT-6 Pro推奨）。これはモデル自動切替やレビュー開始ではない。

**Codex：** 本監査を引き渡して停止する。新しい証明、実装、追加検証、科学計算は、それぞれ具体的scopeの別承認後。B4〜B6・Full-N・Pilot・他系列へ自動進行しない。サブエージェントへの委譲も行っていない。

## 8. 公開・保全

専用branch `track-b-coefficient-type-canonical-audit-20261020`。案A監査HEADから新しい監査ディレクトリだけを追加し、科学SOURCE・v1.1・案A資料・raw・各種保全対象・私的原本を変更しない。原本は非公開のままhashだけを [source_inventory.json](source_inventory.json) に記録した。2026-10-17共通指示と親AGENTSを適用し、過去実行のresource capを新規作業の恒常上限にしていない。共有設定・他worktree・他ユーザーのプロセスへ介入していない。

公開payloadと保全照合の記録は [publication_receipt.json](publication_receipt.json)。自己参照を避けreceiptはpayloadの子commitに置き、最終HEAD・receiptを含む全追加fileの公開bytes/SHA256を別の最終照合記録とチャットで報告する。全rawの再hash・独立再現を行ったとは主張しない。

**公開・保全照合後にmandatory STOP。既存STOPを解除せず、研究方向の正式採択は行わない。**

## 固定参照

[V11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[PROVIDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_provider.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[KNOWN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/known_method_composition_audit.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df90ed8c8e847f8ea1861d9761b9515e7a895916/docs/research/track_b_partial_sum_query_sampler_audit_20261020/README.md
[SMIR]: https://arxiv.org/pdf/1503.08096
[RECUR]: https://arxiv.org/pdf/1006.0423
[FZV]: https://algo.inria.fr/flajolet/Publications/RR-1830.pdf
[WA]: https://cs.nyu.edu/~mohri/pub/hwa.pdf
[WMB]: https://publications.rwth-aachen.de/record/991007/files/991007.pdf
[ISING]: https://arxiv.org/pdf/1807.07112
