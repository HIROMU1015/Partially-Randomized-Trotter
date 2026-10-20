# Track B 案A：部分和query samplerの独立技術監査

2026-10-20 JST。基点 `243eea8e840f3c472753f7935e510391626fd325`（方法ノートv1.1）。
利用者の今回の指示による、一次エージェント自身の紙上検証・静的SOURCE監査である。別AI・外部査読者によるレビューではない。

**結論：部分和取得と有限bit補正は、下記の既存仮定の下で技術的に成立する。A1–A4・A6・A7はACCEPT、A5は計上範囲を明示してREVISE。新しい非自明な定理を必要とする欠落は、この式列の確認では見つからなかった。一方、抽出部分は標準的な線形集約・木探索・inverse-CDFの適用であり、その部分の独立新規性は支持しない。原著全体の十分性・実用上の利益はUNRESOLVED。**

科学計算・数値再解析・simulation・benchmark・科学SOURCEのimport/実行/変更・sampler実装・科学test・回路build/compile/synthesisはすべて0。新定理・別算法の創作、案Bへの移行、既存grant/claim/ledger/marker/STOPの更新も行っていない。Git/文書/hash/公開照合だけを実行した。

## 1. 固定対象と式別判定

仮定：明示された有限個の正有理数 `p_i`、`Σp_i=1`、`L≥1`、有理 `0<x≤1`、正の奇数 `m`、`σ=±1`、Hermitian involution `Q_i²=I`。`u` は長さ `l≤m−1` の偶数reduced語。`x=0` はidentity別経路。対象は有限多項式 `M=P_m(−iσxR)`、`R=Σp_iQ_i` のcoherent operator first momentである。[MATH] §§1–8、[V11] §§2–4。

記号：`t_n=x^n/n!`、`p(u)=∏_(j∈u)p_j`、`b_l=√(t_l²+t_(l+1)²)`、`B_o=Σ_(even l≤m−1)b_l`、`B_full=Σ_(even reduced u)d_u`。A4では合法nに対し `w_n=(−1)^((n−l−1)/2)t_n`、その他は0、`g_(u,b)=Σ_(a=0)^(m−b) w_(a+b)H_u[a]` とする。`H_u[a]=[z^a]H_u(z)`。

式番号は非公開の提供レビュー§§4–5に対応する。原本は転載せず、byte数/SHA256を [source_inventory.json](source_inventory.json) に記録した。

| 式 | 判定 | 独立に確認した理由・必要条件 |
|---|---|---|
| A1：切断係数への線形汎関数 | **ACCEPT** | `t_n=x^n/n!`、`n≥l+1`、`n−l−1` 偶数という有限和。領域外の重みは0。解析的収束・無限級数の極限交換は不要 |
| A2：`a_iu=L_(l+1)(H_u F_i)` | **ACCEPT** | `iu` がreducedなら既存Green式に `iu` を代入できる。`F_i(0)=0` のためm次切断後の積で必要係数を失わない。違法childへ同じ物理解釈を延長しない [MATH] §5 |
| A3：child部分和恒等式 | **ACCEPT** | 有限集合上の線形性。`C(∅)` は全label、非空なら `i≠first(u)`。非空parentでは集合が含むfirst labelの `F_i` を引く。引く項を非reduced語のreturn係数と呼ばない |
| A4：queryをdot productへ移す式 | **ACCEPT** | `L(HF)=Σ_n w_nΣ_(a+b=n)H[a]F[b]=Σ_(b=1)^m F[b]Σ_(a=0)^(m−b)w_(a+b)H[a]` と有限和の順を交換したもの。b=0が無いのは `F_i[0]=0`。符号付きgの途中項を確率として扱わない |
| A5：`O(m³+Lm²)→O(m³+m log L)` | **REVISE** | 前処理後の係数取得・抽出coreの有理演算上界として支持。現行digital/F3全処理の上界・bit費用・CPU speedup・最適法との差ではない。§4の別項、同等最適化、保全guardの条件をstatementへ組み込む |
| A6：実確率がCDF端点のceil差 | **ACCEPT** | 半開区間と一つの一様H-bit整数を使う場合の厳密な整数個数。§3で境界・誤差・supportを確認。木の各辺で新しい整数を引く方法には適用しない |
| A7：`2^(−H)≤η p_min/2` | **ACCEPT** | `0<η<1`、存在する合法child、正確な有理CDFが前提。既存 `q_i≥p_i/2` とA6から相対誤差・正supportが従う。child段階の十分条件であり、全P0–P5・root・angleの十分条件ではない [BIT] §5、[F1] §4 |

上のACCEPTは数式への判定であり、未作成SOURCEの認証・実装開始承認ではない。REJECTすべき式は見つからなかった。

## 2. 全childを作らずに保持できるもの

既存Green式は `H_u=G∏_(j∈u)F_j=Σ_n P_n(u)z^n`。[MATH] §5と [K] 170–202行を照合した。Aomoto–Katoのmultiplier構造自体は既知である [AK] §1、Eq.(1.6)、Lemma 1.1。

木の各nodeに保存するのは、**parentに依存しない** `Σ_(i∈S)F_i` のm次ベクトルである。parentごとに `H_u` と `g_u` を作れば、総量 `s_u=A_u(all labels)`、探索で使うnodeの質量、選択leafの `a_iu` を取得できる。`a_u` は同じ `H_u` に偶数長用の既存汎関数を作用させる。`d_u²=a_u²+s_u²` と `s_u/a_u` に全child配列は要らない。

したがって、式が要求する全childの再構築を別の場所へ隠しているわけではない。ただし、**既存の `parent()`・`digital_parent()`・`event()` をそのまま呼べば実際には再構築される**。§5の変更要件が必要である。

正確な実数一様変数による木下降の理想確率は、経路上の `mass(child node)/mass(parent node)` がtelescopingして `a_iu/s_u` になる。一つのUをCDF探索しても同じである。有限H-bit版の確率はA6であり、一般には理想qと異なる。

parent提案と受理を変えなければ、理想event確率は

\[
\Pr(e=(u,i))=\frac{b_l}{B_o}p(u)\frac{d_u}{b_lp(u)}\frac{a_{iu}}{s_u}
=\frac{d_u a_{iu}}{B_os_u}.
\]

既存Euler pairing `d_uΣ_i(a_iu/s_u)V_(u,i)` はparent係数と全child係数を再現する。`V_(u,i)=(−iσ)^l exp(−iσφ_u Q_i)Q(u)`、`tan φ_u=s_u/a_u`、operatorの語順、回路時間順の逆順、偶数parentの位相 `(−1)^(l/2)` を保つ。scalar積の可換性からQの並替えは導けない。[MATH] §§6–8、[PROVIDER] 2–22行。

この理想層では `Z=B_full/B_o`、`E[X²]=B_o B_full`、`|X|≤B_o` とideal event依存量子費用分布は同じ。有限bitの実分布・実受理率・second momentまで一致するとは言わない。

| 境界 | 確認結果・要求 |
|---|---|
| 空parent | first-label除外なし。`H=G`、全labelをchild候補にする |
| `L=1` | 正規化からp=1、`F_1=z`、`G=(1−z²)^(−1)`。到達する偶数reduced parentは空語だけ、childは一個・q=1。非空偶数rawはzero。存在しない長さ2以上のreduced parentを仮定しない |
| `m=1` | parentは空語のみ。`a_∅=1`、`a_i=xp_i`、`s=x`、q=pで直接整合。次数0の畳込みも有効 |
| `x=0` | M=Iとしてbits・root比・CDF・weight・angleを経由しない。legacyの低水準digital処理に投入せず、F3のidentity入口を維持する |
| zero-mass node | 空集合・first-labelだけの集合・padding等は0。0幅CDF区間は選ばない。許可域では存在する全合法childは正質量 |
| 総量 `s=0` | 許可域の到達Full parentでは発生しない。汎用helperなら純parent規約を別に定め、`a=s=0` や不正入力を正当zero trialへ変えない |

## 3. 有限bit・weight・biasの確認

`N_b=2^H` とし、累積確率を正確な有理数 `C_0=0≤…≤C_J=1` とする。選択条件 `C_(i−1)≤R/N_b<C_i` は整数で `N_b C_(i−1)≤R<N_b C_i`。よって個数は `ceil(N_b C_i)−ceil(N_b C_(i−1))` であり、**floor差へ変更してはいけない**。端点がgridと一致するとその点は右側leafへ属し、R=N_bは供給されない。

`δ_i=ceil(N_b C_i)−N_b C_i∈[0,1)` と置けば、`q̃_i−q_i=(δ_i−δ_(i−1))/N_b`。従って絶対誤差は厳密に `<2^(−H)`、総和はtelescopingして1、0幅leafは0のままである。CDF探索中も元のgridに対する比較・offsetを保持する。各nodeの比を新しくH-bit丸めする処理にはこの証明を流用しない。

既存正値下界から `a_iu≥t_(l+1)p(u)p_i/2`、上界から `s_u≤t_(l+1)p(u)Σ_(j∈C(u))p_j`。よって

\[
q_i\ge\frac{p_i}{2\sum_{j\in C(u)}p_j}\ge p_i/2.
\]

A7なら `|q̃_i−q_i|<ηq_i`、従って `q̃_i>0`。child用の十分bit長は `H≥ceil(log₂(2/(ηp_min)))`。これは必要最小値ではない。全supportのため `2^H≥J` も必要だが、A7はそれを満たす強い条件である。H-bit整数の保持量はO(H)であり、2^H個の表を作る意味ではない。

固定H-bit mappingは一般の有理qをexactには生成できない。紙上の最小例 `m=1, p=(1/3,2/3)` ではq=pで、分母3は2^Hを割らない。この点はA6–A7の誤りではなく、理想lawとdigital proposalの区別である。

実proposalはorder、l個のlabel、受理coin、**一つのchild draw**の実確率を掛けた `π_e` とする。childのq̃は選択leafの左右の累積質量からceil差で再取得でき、全q̃配列は不要。根号近似を従来と同じ `d̃_u` とすれば

\[
\widetilde\alpha_e=\widetilde d_u a_{iu}/s_u,\qquad
\widetilde W_e=\widetilde\alpha_e/\pi_e,\qquad
\pi_e\widetilde W_e=\widetilde\alpha_e.
\]

これは実proposal補正の既存恒等式で、確率丸めのbiasを相殺する。根号近似・角度・native誤差は残る。他のproposal因子も `(1±η)`、`|d̃−d|≤ρd` を保てば、因子数は `l+3≤m+2` のままで、[BIT] §§3–6、[V11] §4の既存保証を適用できる：

\[
|\widetilde W|\le\frac{(1+\rho)B_o}{(1-\eta)^{m+2}},\quad
\|\widetilde M-M\|\le\rho B_{full}\le\rho B_o,
\quad E[\widetilde X^2]\le\frac{(1+\rho)^2B_oU}{(1-\eta)^{m+2}}.
\]

最後は既存の認証上界 `U≥B_full`、同じ測定モデルを使う場合。角度誤差τとstrict native誤差は別途課金し、予算の正remaining・IID bits/測定・全N完了の条件を残す。[G] 266–284行。既存の固定α規約を任意confidenceへ拡張しない。F1のP1/P3/P4等をA7だけで置き換えず、P2のchild条件とP5のlaw bindingを新CDFへ接続する。旧largest-remainderのdigital event列・実πは一般に変わるため、旧Pilotを新版の実行証拠へ流用できない。

## 4. A5の正確な費用範囲

以下は単位有理演算モデル、固定p/x/mに対する前処理後の上界。最悪時間の下界・CPU実測ではない。[K]、[F2]。

| 費用 | 計上 |
|---|---|
| 入力・Green前処理 | 入力readはΩ(L)要素。既存GreenはO(Lm²)有理演算、O(Lm)係数 |
| 木の追加前処理・保持 | O(Lm)有理加算・O(Lm)係数。保存内容はF_iの部分和であり全parent表ではない。p/m変更時は作り直す |
| `H_u`・`a_u`・`g_u`・総量s | 単純畳込みでO(lm²)、gはO(m²)、aとroot-node dot productはO(m)。合計O(m³)に含む |
| child探索・実q̃ | 固定label順の平衡木を一度下降。訪問nodeごとのdot productはO(m)、合計O(m log(2L))。累積offsetとleaf両端を保持する。各段階でprefixを根から再queryして余分なlog Lを掛けない |
| raw/order選択 | orderと固定dyadic label lawのCDF等を一度検査・前処理し、各drawをO(log m)/O(log L)で行う条件。現行 `dyadic_index` の全law検査を毎draw残すならO(mL)項が残る |
| root・weight | valid parentごとのdとacceptanceのroot enclosure、実π、weight、exact比較を別課金。child law変更で不要にならない |
| angle/provider | 既存half-angle/atanの提供 `K_φ,J,τ` の費用とstrict phase保証、provider/IR/合成・量子資源は別。ratio取得とangle取得を同一視しない |
| 監査・Evidence | 全state再hash、request/preflightの全p処理、angle時event再query、全child packet保存が残ればL依存。削除していない保証を削除済みとして費用を減らせない |

したがって修正statementは **「固定入力の前処理後、全child配列を作らない係数・抽出coreはO(m³+m log(2L))有理演算。setup、bit算術、root/angle、整合性検査、保存、物理providerを別計上する」**。旧 `O(m³+Lm²)` は係数queryの上界で、現行digital packetにはさらにs再sumによるO(L²)、law sortのO(L log L)、各drawの全law検査がある。[F2] §§3–7。旧上界と新上界の比を実速比・普遍的改善率にしない。

bit費用：F2の `b_c=1+m(H_p+H_x+ℓ_m)` を基礎とし、F_i部分和には共通分母と加算個数、g/dotの有限積・和にも同じ係数bit補題を適用する。符号による相殺は浮動小数の安定性を仮定せずexactに扱う。保守的に `b_*=O(b_c+log(L+1)+log(m+1))` の算術幅で課金できるが、**H_p自体が全入力長でありL依存**である。coreのbit費用は標準モデルで `O((m³+m log(2L))Q(b_*+H+E))` 型に写し、setup、2回のroot `R(b_c,K_root)`、O(mH) bits供給、weightの `O(b_c+K_root+(m+2)H+E+ℓ_L+ℓ_m)` 長、angle、Codec/I/Oを追加する。`Q(b)=O((b+1)³)` はF2の保守的GCD込み課金で、tight boundでもCPython秒保証でもない。

保持量O(Lm)は**係数個数**。bitメモリはGreen/木についてO(Lm b_*)級、局所H/g・探索path・precision・Evidenceを別に加える。全N費用は `setup + Σ_trial 実行経路の費用 + 保存`。coin rejectもs/d/受理計算を払う。child探索だけはaccept後へ置ける。理想native費用分布は変わらず、digital側の期待量子費用は実πが変わるため同一と断言しない。

## 5. 静的SOURCE対応と最小変更要件

行番号はすべて固定v1.1 commitのもの。リンク先のbytesは目録で固定した。以下は変更仕様の必要事項であり、SOURCEを改変した記録ではない。

| 固定SOURCEの箇所 | 現在の処理 | 新方式に必要な最小差分 |
|---|---|---|
| [K] 136–160 / 170–202 | Green前処理、H共有後に全child畳込み・mass配列 | 新namespaceで部分和query用の前処理とparent queryを持ち、`a,s,H/g` を取得する。全child版parentは本番経路から呼ばない |
| [K] 114–133 | `Parent.s` は毎回sum。`child_law` 各要素でsを再sum | scalar sをpacketに保持。選択childのmassとq̃だけを渡し、全配列を必須としない |
| [K] 55–82 / 97–110 / 214–236 | largest-remainder sort、全law検査・走査、coin前に全child law | childはA6で一draw。order/labelは同等に許された事前検査とselector。root/acceptanceの認証は維持 |
| [K] 239–254、[G] 227–251 | 全packet取得、`children.index`、`masses[index]`、weight | selected label・exact mass・実count/2^Hからπとα̃を算出できるAPI。全child探索/indexを必須にしない |
| [G] 35–42 / 216–225 | U setupが空parentの全childを取得、raw-reduced DPはO(Lm) | U取得も部分和の空parent queryへ接続可能。DPとsetup費用は残す。旧経路をsetup限定で使うならそのO(Lm²)を明記 |
| [F3G] 29–36 / 113–128、[F3C] 26–92 / 154–175 | `_state` が全first_passage等を含み毎APIでcanonical化/hash。request/preflightにもp全体 | 現状のままなら毎APIに少なくともΩ(Lm)係数の走査・serialization。新しい固定stateの結合・整合性検査契約が必要。単に保存digestを返す処理では、現行の状態変化検知と同等にならない |
| [F3G] 239–256 / 261–287 | `parent_packet` はchildren/masses/child_lawをEvidence化。angleは `_generator.event()` を再呼出し | 選択経路の証拠を受け取るversion付きschemaとevent/angle結合へ変更。旧packet/eventを検証のために呼び、全childを再構築しない |
| [F3P] 62–142、[F3C] 16–23 / 228–239 | P0–P5と旧SOURCEのhard pin | 旧版を保全し、新SOURCE・policy/schema・tree/query・CDF丸め規約を識別する新identity。旧pinを破るmonkey patchでは接続できない |

必要なEvidenceは、setup一回のinput/source/policy/tree定義・固定label順・係数表への結合と、各trialのraw語/zero理由/消費bits、parentのa/s・root/acceptance、選択child、query nodeの集合範囲とfirst-label除外、exact累積両端・ceil count・実π・α̃・weight・ratio・phase/語順・angle結合である。nodeの質量と `left≤R/2^H<right`、`πW=α̃` を検査可能にし、0とtechnical failureを分ける。全係数表はsetup資料として一度保持でき、毎trialに全childを保存する必要はない。

**未解決の実装条件：** 現行F3と同じ入力/SOURCE/状態改変の検知保証を、どの脅威モデル・不変state契約で保つかはまだ仕様化されていない。毎callの全state走査をそのまま保つ選択も可能だが、その場合A5はcore限定となる。hashを一回にするだけで同じ保証と認定しない。軽量な証拠・整合性方式の設計に非自明な新保証が必要なら、別承認の理論課題として返す。この監査では考案・証明しない。

## 6. 既知法との重複と残る貢献

一次文献の関連箇所をWebで再確認した。全論文読了・網羅的priority調査・著者コード実行は主張しない。AK/DevroyeのPDF頁画像取得は失敗し、解析テキストと固定既存証明を用いた。

| 一次資料 | 直接重なる部分 | この監査から言えないこと |
|---|---|---|
| [AK] §1 Eq.(1.6)、Lemma 1.1 | free-product Green multiplierと局所係数取得の基盤 | 原論文に有限Taylor RA・A6の全手順が書かれているとは言わない |
| [DEV] Ch.III §2.3、pp.89–92 | 一つのUからCDF比較で二分木を下降するinverse-CDF | parent依存CDFをどう安価に取得するかは別。固定H-bitのA6は上記整数個数の直接計算 |
| [FEN] 著者論文の出版社要旨 | 累積頻度を部分和で保持する標準データ構造 | Fenwick木そのものの新規性なし。案Aのベクトル内積を元論文が扱うとはしない |
| [TAO] §§3.1–3.2、5 | alias、subtree質量によるtree sampling、全reportを避けるquery sampling | 論文の固定正重みやpredicate条件を、符号付きgでparentごとに変わる質量へ無条件転用できない |
| [BP] §§1–1.1、Thm.1.3 | 明示分布へのO(L)前処理とO(1)抽出、setup/queryの区別 | Real RAMの実数一様源・単位演算を、有限bit・有理演算の実費へ移さない |
| [WBC] Appendix C、Algorithm 2 | orderとIID labelでordinaryを非列挙生成 | 既存対照へ全語列挙を強制しない。一般involutionへの完全対応やRA同一lawまでの還元を同論文だけで確定しない |

**判定：** A3はGreen式の線形性、A4は転置した有限畳込み、木とA6は標準抽出法である。`g_u·Σ_(i∈S)f_i` というqueryへ既知の加法的データ構造を適用した、と説明できる。これらを独立の新しいGreen定理・sampling原理とする根拠はない。

通常最適化した旧local Fullにもs cache、一度のlaw検査、高速selector、同じGreen前処理、parent再訪cacheを認める。既に取得した全childのCDF/aliasは抽出だけを速くするため、初回parentの係数取得は別に残る。しかし**同じ部分和oracleまで対照へ認めれば、対照にも同じO(m log L)下降を適用でき、案Aだけの計算量差は残らない**。相手へ弱い配列生成を強制して一般分離を主張してはいけない。

再訪parentのcacheでは一度の取得・前処理を償却できるため、全workloadにおける優劣も未確定。[P5] 10–132行のclosed P5には既に有限group・completion DPがあり、Green前処理を必須とする一般Fullとは異なる強い対照である。低次数のp依存式にも通常の前処理を同等に認める。

残り得る成果は、**既存RAへの具体的なimplicit係数アクセスの特殊化、有限bit/weight/rangeまで含む再利用可能な仕様、および全child配列を要求する現行実装からの取得費用改善**。これは技術的に具体的だが、現時点の証拠では「標準手法の適用を超える独立原著貢献」とは認定しない。実装改善・方法ノートの追補に留める選択は合理的である。全既知算法に対する下界・非還元性・普遍的速度優位はUNRESOLVEDで、ここで新しい定理を作って補わない。[KNOWN]のG6全構成の既知性未確定とも区別する。

## 7. 実装へ進む条件・進めない条件・次の担当

| 進むために満たすこと | 進めるべきでない状態 |
|---|---|
| ユーザー/GPTが中心claimを、既知法の適用である点を含めて選ぶ。一般原著か実装改善かを区別 | A1–A7のACCEPTだけを新規性・投稿GOへ格上げする |
| coreと全処理の費用を分け、新namespaceのAPI/Evidence/整合性保証・SOURCE固定を具体化 | 本番またはverifier/angleで全childを再取得しながら全処理のL依存を除去したとする |
| 同じ理想target、実π補正、phase/語順、境界、P0–P5と失敗保存を保持する | float CDF、各辺の無断丸め、旧q̃での補正、state guardの黙った弱化、旧trace/結果の流用 |
| 強い対照にも同じcache/selector/前処理・監査保証を認める | 通常最適化や同じ部分和アクセスで差が消えるのに、独立新算法の優位だけを目的に開発する |
| 必要な実装・検証・科学計算を、それぞれ対象/範囲付きで別途明示承認する | 今回の監査承認から実装・追加test・benchmark・案Bを自動開始する |

**次：ユーザー**が研究として継続するか、実装改善に限定するか、追加投資を止めるかを判断する。独立原著としての価値を決める詳細レビューが必要なら、**GPT**が本資料を根拠に既知適用との差とclaimの十分性を判断する（共通指示上Level 1、GPT-6 Pro推奨、ユーザーによる別の開始承認が必要）。モデルの自動切替・レビュー自動開始はしない。

**Codex**の次の実装・追加検証・新科学計算は未承認であり実行しない。採択後もそれぞれのscopeに対する別承認が必要。後続担当としてClaude Code等への委譲を設定していない。

## 8. 公開・保全とSTOP

専用branchは `track-b-partial-sum-query-audit-20261020`。追加物は本監査と固定参照目録・公開照合記録だけ。基点との差分で既存ファイルの変更がないこと、参照・保全対象17件のbytes/SHA256、私的原本2件のhashを確認する。v1.0は保全照合だけで再監査していない。無関係なworktreeのHEAD/statusを凍結せず、他の計算・プロセス・共有設定へ介入しない。

公開照合の記録は [publication_receipt.json](publication_receipt.json)。自己参照を避け、同記録は先に公開した監査commitと本文/目録の実公開bytesを参照する。記録自身を含む最終HEADと各fileの照合はチャットおよびローカルの最終receiptへ報告する。変更していない全rawの再hashや全既存結果の独立再監査をしたとは表現しない。

**公開・保全照合後にmandatory STOP。技術的成立と科学的価値の最終採否を混同せず、次の判断はユーザー・GPTへ返す。**

## 固定参照

[V11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[PROVIDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_provider.py
[F3G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[F3C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_contract.py
[F3P]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_precision.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[KNOWN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/known_method_composition_audit.md
[AK]: https://www.numdam.org/item/AIF_1988__38_1_59_0.pdf
[DEV]: https://luc.devroye.org/chapter_three.pdf
[FEN]: https://onlinelibrary.wiley.com/doi/10.1002/spe.4380240306
[TAO]: https://www.cse.cuhk.edu.hk/~taoyf/paper/pods22-iqs.pdf
[BP]: https://people.mpi-inf.mpg.de/~kbringma/paper/J2017ALGO-2.pdf
[WBC]: https://arxiv.org/pdf/2110.12071v2
