# Hamiltonian simulation：新しい構成課題とアルゴリズム候補の設計

- 作成日：2026-10-11（日本時間）
- 区分：GPTによる新研究設計・先行研究比較・数学的検討。既存実験の再レビューではない。
- 開始依頼：「新しい研究方針・アルゴリズム候補の設計を開始して　できるだけ丁寧に考えてください。」
- 研究範囲：Hamiltonian表現・分解・近似・実装構造、およびPRとの共同設計。PR内部だけのsampling/return/RTE改善は別Track Bに残す。
- 状態：候補の数学的定義・構成手順・改善機構・比較対象・最小検証を設計した。中心テーマ、一般的優位、新規性、論文の成立は未確定。
- 実行範囲：一次文献の調査、既存結論の参照、手計算・小さい決定論的代数チェック。新しい分子実験、回路compile、量子shot、Codex production runner、repository変更は行っていない。

## 1. 結論

旧A-core/B′-pairを再び微修正するのでなく、資源を削減する構造を生成する問題として、三案を具体化する。

| 候補 | 生成するもの | 主な改善対象 | 第一の未解決問題 |
|---|---|---|---|
| N1：縮退を作るDF・基底変換削減 | 同じ固有値を持つ群と、その自由度を使った安価な軌道frame | 各fragmentの基底変換。場合により対角位相回路も削減 | 許容誤差内で、有意味な量のnative gateを本当に消せるか |
| N2：疎整数の集団占有数による係数圧縮 | 小さい整数行列S、係数K、残差Eと誤差上界 | 多数のdensity相互作用の一括実行 | 加算・uncompute・ancillaを含めても強い既知実装に勝てる構造を作れるか |
| N3：結合・外部エネルギー下界に基づくactive-space選択 | 小さいactive Hamiltonian、捨てる空間の影響上界、選択履歴 | system qubit数と相互作用数 | 全系の厳密基底状態を知らず、使えるmany-body下界を安く得られるか |

最初の実装検証の優先案はN1、対抗案はN2。N3は量子化学の基底エネルギーに対象を定める場合の別系統・高リスク候補とする。これは中心テーマ採択ではなく、次の情報取得の優先順位である。N1とN2は接続できるが、両方を使うことを義務にしない。三案が成立しない場合、既存案に条件を足して成功を作らず、別の構成原理へ戻る。

## 2. 出発点と既存研究の扱い

既存repositoryは `HIROMU1015/Partially-Randomized-Trotter`。参照した限定探索の結果commitは `f98050e9402e8c65bb0f3dd27c7bd30d12fe7069`、sourceは `3975650908c594ce6dddb4fc2086167caf5b6460` である。[R1]

前回レビューの判断は変更しない。現A-coreとB′-pairの同型探索は一区切り、旧B/Cは保留。whole-Pauli等の安い対照を基盤に残し、Track A/Bのsource、契約、結果、STOPへ変更を加えない。

今回、旧実験の全IR再照合・再採点は行わない。閉じた探索の各数値を新候補の実証と混同しない。未公開のローカル引継ぎ資料も、新設計を止める必須資料とはしていない。既知の結論と公開資料で設計できるためである。

重要なのは、前の失敗を「DFが駄目」「Pauliが常に最良」「近似は必須」と一般化しないこと。今回の三案に近似が多いのは、資源削減の新しい自由度として検討するためであり、研究全体を近似法に限定するものではない。

## 3. 今回確認した近接先行研究

### 3.1 単なるcost-aware groupingは新しい空白ではない

Mukhopadhyay–Wiebe–Zhangは、可換Pauli fragmentの合成、人工的な対称性の導入、1-normの処理量とgate費用の比を用いるgreedy分解、部分的・符号付き配分、Hamiltonian truncationを扱っている。[P1]

したがって、従来のD1の説明だった「安いblockを作って残差を乱択する」「gate費用を目的にgroupingする」だけでは差分にならない。論文のqDRIFT channel解析を、PRの補正first momentと同じ保証として移用することもできない。

### 3.2 DF・低rank・圧縮の主張の境界

CDFはHamiltonianの圧縮と短い時間発展回路を扱い、RC-DFは正則化したfactor fittingで誤差・係数norm等を改善する。[P2,P3] SCDFとBLISSを含む共同最適化にも先行研究がある。[P4,P5]

SPRINT/GRADEはfactorization、残差、randomized formula、近可積分構造、対称性保護、QROM回路を組み合わせている。[P6] よって「分解と実装を共同設計する」という一般論も新規性にはできない。

Low–Su–Tong–Tranは、相互作用構造と低rank性を利用したTrotter step実装を扱う。[P7] Qroneckerの要旨は、Pauli係数のKronecker圧縮とstate-independentなエネルギー誤差認証を報告する。[P8] 後者は今回要旨の確認に留め、本文の全保証・実装を再監査していない。圧縮の証明があっても、その指数演算が安価になるとは自動的に言えない。

### 3.3 誤差付き縮約も既知分野である

SQuISHは近似基底状態情報等を用いたHamiltonian縮約を提案している。[P9] SWの有効Hamiltonianとその誤差の理論、量子回路によるSW実装も存在する。[P10,P11] 「active-spaceを選ぶ」「gapに基づく誤差boundを書く」だけを独立の新規貢献にはしない。

以下では、既知の数学的部品と、今回提案する未検証の構成手順を分ける。今回の検索は、全世界の優先権や特許を完全に確認したものではない。

## 4. N1：縮退を生成して基底変換を削るDF構成

### 4.1 問いと入力

中心の問いは「近い固有値を意図的に一致させることで生まれる自由度を使い、Hamiltonianの許容誤差内で基底変換そのものを減らせるか」である。

具体的な入口として、spin-orbital数n、粒子数νのsectorで

\[
H=H_1+\sum_\ell w_\ell F_\ell^2,
\quad F_\ell=d\Gamma(g_\ell),
\quad g_\ell=U_\ell\operatorname{diag}(\eta_\ell)U_\ell^\dagger
\]

を扱う。wは実数で、負の場合は誤差和に絶対値を用いる。既存のone-body補正・scalarはH1へ正確に含める。この形が直接使えないfactorizationへ無断で適用しない。一般CDFのleaf行列はN2で別扱いする。

この固有値はn×nのone-body factorの固有値であり、指数次元Hamiltonianの基底エネルギーを入力として要求していない。

### 4.2 縮退で何が変わるか

factorの固有値を群I_gごとに中心c_gへ置き換え、\(\widetilde\eta_i=c_g\) とする。すると

\[
\widetilde F=\sum_g c_g N_g,
\qquad N_g=\sum_{i\in I_g}n_i,
\qquad \widetilde D=\widetilde F^2.
\]

同じ群の内部で回す \(W=\bigoplus_g W_g\) は、\(\operatorname{diag}(\widetilde\eta)\) と可換である。従って

\[
\Gamma(UW)\widetilde D\Gamma(UW)^\dagger
=\Gamma(U)\widetilde D\Gamma(U)^\dagger.
\]

近似後のfragmentを変えず、Uではなく安価な代表UWを選べる。近似前に存在しなかった縮退自由度を、誤差予算を使って生成することが狙いである。

ただし、縮退群ができたことと、native gate数が必ず減ることは別である。denseな部分空間を合わせる回転は残る。nモードが複数の同程度の大きさの群に分かれても、群の間の基底変換はO(n²)規模になり得る。

明確な十分例は \(\widetilde g=cI+V_kDV_k^\dagger\) の形でk本の例外軌道だけを持つ場合。補空間のframeは自由に選べるので、k本の軌道を合わせるpartial Givens構成を使える。ただしこれはshifted/truncated DF等に還元する場合があり、この特殊例だけを新規性にはしない。[P4,P5]

### 4.3 固定粒子数で安く計算できる誤差上界

実ベクトルxについて

\[
m_\nu(x)=\max\left\{\left|\sum_{i=1}^{\nu}x_{(i)}\right|,
\left|\sum_{i=n-\nu+1}^{n}x_{(i)}\right|\right\}
\]

とする。x_(i)は昇順で、ν=0なら0。これは\(\sum_i x_i n_i\)のν粒子sectorでのnormである。ν個の和の最大・最小を取ればよいから、指数次元行列を作らず計算できる。

\[
d_\ell=m_\nu(\eta_\ell-\widetilde\eta_\ell),\qquad
\epsilon_\ell=|w_\ell|d_\ell\{m_\nu(\eta_\ell)+m_\nu(\widetilde\eta_\ell)\}
\]

なら、同じframe内での平方差から

\[
\|(H-\widetilde H)|_\nu\|\le\sum_\ell\epsilon_\ell=\epsilon_H.
\]

既存factorization誤差、数値丸め、入力係数の不確かさは別に加える。両Hamiltonianがsectorを保つ場合、

\[
\|(e^{-itH}-e^{-it\widetilde H})|_\nu\|\le |t|\epsilon_H,
\quad |E_{0,\nu}(H)-E_{0,\nu}(\widetilde H)|\le\epsilon_H.
\]

これは標準的なnorm・変分の議論を具体的なfactor表現へ適用した十分条件であり、新規定理とは呼ばない。固有値を数値で取得した際、数学上の上界と丸めまで保証した計算結果は区別する。

基底の近似削除を別に行う場合も、n×nの差行列を計算し、その固有値からmνを評価できる。一つのGivensが対角固有値p,qを混ぜるケースでは

\[
\|g-G_{pq}(\theta)gG_{pq}(\theta)^\dagger\|_2
=|\eta_p-\eta_q||\sin\theta|.
\]

角度が大きくてもgapが小さければ影響が小さい。小角度だけを基準に消す方法とは異なる設計情報になる。一般の多段回転を消すときはこの局所式を無条件に使わず、差行列または正しくtelescopingした上界を使う。

### 4.4 提案する具体的な生成手順

1. 各factorのn×n固有分解を取得し、元の無変更表現を候補として保存する。
2. 固有値を並べ、隣接する群の併合・中心の変更を有限候補として生成する。群数最小だけを目的にせず、誤差と実装費用を残す。
3. 各群構成について、縮退群内のframe自由度を使うQR/Procrustes更新やpartial Givensを試し、近似後の同一fragmentを安く実装する代表を作る。
4. 中心位相の実装は、通常のZ/ZZ、既知の同角回転合成、必要なら集団占有数方式を公平に比較する。
5. 誤差、native費用、workspaceの非劣候補を保存する。比較対象にも同じcompiler改善を適用する。
6. 全factorの誤差和と最終taskの許容誤差の中で候補を選ぶ。有限候補集合なら多目的knapsack/動的計画等が使えるが、連続frameの大域最適解を多項式で保証したとはしない。
7. \(\widetilde H\)、frame、回路構成、誤差証拠、元Hとの差の係数表現を出力する。

PRとの接続には二つある。近似\(\widetilde H\)をそのままsimulationしmodel errorを計上する経路と、元Hとの差を正確な残差としてRTE等で戻す経路である。後者では残差の取得費用、展開norm、shot、basis切替を数え、安くなると仮定しない。最初から差を全てRTEへ戻すことは必須にしない。

### 4.5 新規性候補と最小検証

既知：圧縮DF、固有値切断、人工的対称性による安価な合成、可換群の最適化。[P1–P5]

提案差分：縮退のpartitionを離散変数、縮退群内frameを連続変数とし、sector誤差制約の下でnative基底変換を直接削る構成手順。rank/λの変更だけではなく、どの回転を消す対称性を作るかを出力する。

未確定：同じ構成全体の既報との差、一般的な費用改善、実分子での非自明な縮退群の存在。

最初の検証は三つの問いに答える。①同じ近似誤差のrank切断・shifted DFより安いframeが得られるか。②近い固有値があるだけでなく回路が実際に簡単になるか。③新旧に公平な中心位相実装を入れても利益が残るか。局所的な近縮退の設計例は機構確認に使えるが、同じ既知簡約で全Hamiltonianが可解になる例だけで性能を主張しない。少なくとも別の重なりを持つ非可換fragmentを含む比較へ接続する。

失敗条件：許容誤差内では群を作れない、できてもgroup間回転が支配、既存shift/truncationと同じ、approximation分だけPF/合成精度を締めて利益が消える、あるいは安いPauli対照が勝つ。こうした結果は、その条件での不支持として残す。

## 5. N2：疎整数の集団占有数へ係数を圧縮する

### 5.1 問い

可換density blockを、単に小rankの実行列へ分解するのでなく、「量子回路上で安価に計算できる少数の整数占有量」の二次式へ近似する。

\[
D=\sum_{ij}J_{ij}n_i n_j,\quad J=J^T\in\mathbb R^{n\times n}
\]

と書く。対角は\(n_i^2=n_i\)なのでone-bodyであり、i<jだけの規約を使う場合は係数2を正確に調整する。ここでは全ijの二次形式を定義として使う。

### 5.2 出力する表現

\[
J=SKS^T+E,\quad S_{ia}\in\{-1,0,1\},\quad
Q_a=\sum_iS_{ia}n_i.
\]

Sの各行に含まれる非zero数をd以下にする。すると

\[
D=\sum_{ab}K_{ab}Q_aQ_b+\sum_{ij}E_{ij}n_i n_j.
\]

通常の実係数low-rank分解ではなく、Sを疎な小整数へ制限するのは、Qを加算・減算で計算できるようにするためである。SVDが小さいFrobenius誤差を与えても、この条件は保証しない。

既に等係数の相互作用が与えられている場合、Hamming-weight等の集約は既知である。[P1,P7,P12] 狙う差分は、与えられたJから許容残差内でこのような構造を生成する手順であり、加算器の再実装ではない。

### 5.3 回路と費用の構造

\(|n\rangle|0\rangle\mapsto|n\rangle|Q(n)\rangle\)を計算し、\(e^{-itQ^TKQ}\)を作用させ、Qをuncomputeする。

Qはsigned integerであり、b=O(log n) bitあれば足りる。単純な逐次controlled addではcompute/uncomputeがO(nd b)程度のToffoli/CX操作、k個のregisterでO(kb) workspaceとなる。各整数のbitを用いてQ_aQ_bを展開する直接phase構成なら、dense KについてO(k²b²)個のphase項となる。Kが疎ならその非zero数で置き換わる。

この比較は回転の合成費用・制御数・加算・消去・qubit数を別々に数える前の構造上の見積もりである。単にn²回転から少数回転へ減るだけで総費用が下がるとはしない。controlled evolutionではQのcompute/uncomputeを無制御にし、phase側だけHadamard ancillaで制御する構成が可能だが、中心phaseの追加制御の費用も含める。

全候補の費用式は
\[
C_{\mathrm{block}}=C_{\mathrm{frame}}+2C_{\mathrm{charge}}+C_{\mathrm{phase}}
\]

を出発点とし、実装後は全回路の相殺・depthを評価する。Low等の構造利用型Trotter、Hamming-weight、直接Pauli、既知QROM方式を比較対象から外さない。[P6,P7,P12]

### 5.4 誤差

\[
\|D-\widetilde D\|\le
\sum_i|E_{ii}|+2\sum_{i<j}|E_{ij}|=\epsilon_D.
\]

占有数が0/1であることからの安全な上界である。固定粒子数では、ν個の対角項と\(\binom\nu2\)個のpairに限定した絶対値の緩和も使える。ただし一般のbinary quadraticの最大値を厳密かつ容易に計算できるとは主張しない。量子回路へ格納するKの丸め誤差もEに加算する。

近似modeではモデル誤差を計上する。正確modeではEを残差として保持し、PR等で処理する。後者の辞書費用・normalizationが重い場合、構造圧縮の利得が消える。

### 5.5 候補生成器

入力はJであり、真の低rank因子を与えられていることを前提にしない。

- 行・列の相互作用patternを用いて初期群を作り、0/±1の群indicator・signed differenceを候補にする。物理geometryが与えられる場合、近接・遠隔群の階層構造も候補生成に使える。
- 固定Sに対してKを係数フィットし、誤差制約とphase費用を確認する。weighted least-squares解だけを物理誤差保証としない。
- 群の分割・併合・重なり・符号変更を試すが、実装費用を減らさない更新は採用しない。探索予算の範囲内の非劣候補を返し、大域最適性を保証しない。
- 残差Eと誤差上界を計算し、元のJをそのまま実装する候補を必ず保持する。

N1で等しい固有値の群ができたとき、N_gがこのQの特別な場合になる。しかしN2は一般のJから始められ、N1を必須にしない。

### 5.6 最小検証とリスク

まず、構造が既知のJで回路恒等式・error bound・加算uncomputeを確認する。その後、既知Sをconstructorへ渡さずJだけから再構成する。小さい摂動、低rankだがdense実係数因子しか見つからない対照、疎だがこの集団表現では圧縮しない対照を分ける。

構造を植え込んだ例はmechanism witnessであり、化学Hamiltonianでの改善証拠とはしない。大きいnのmetadataだけで実費用scalingを証明したともしない。

主なリスクは、k/dが大きくなる、精度を保つにはEが大き過ぎる、整数制約でrankが膨れる、adders・ancillaがphase節約を上回る、既知の低rank/Hamming/QROM構成と同値になることである。

## 6. N3：外部空間との結合に基づく誤差付きactive-space選択

### 6.1 この候補のtask

最初の対象は、指定された粒子数・対称性sector内の基底状態エネルギーである。これはN3の保証対象の明示であって、プロジェクト全体をエネルギーだけに限定することではない。N1/N2は時間発展そのものを対象にできる。

固定占有を課したactive projector Pを、既知のorbitalとoccupationから構成する。Pは未知の真の基底状態への射影ではない。Q=I−Pとして

\[
H=\begin{pmatrix}A&B\\B^\dagger&C\end{pmatrix},
\quad A=PHP|_P.
\]

Aを小さいsystemとしてsimulationし、元Hのエネルギーとの差を、未知の真値を使わず評価する。

### 6.2 基本的な十分条件の導出

古典的に検証可能な量として

\[
C\succeq cI,\qquad \|B\|\le\beta,\qquad
\lambda_{\min}(A)\le U<c
\]

を取得できたとする。UはP内の明示試行状態のvariational energy上界でよく、厳密な基底エネルギーを要求しない。統計推定を使う場合は上側信頼限界とそのconfidenceを用いる。

a=λmin(A)、g=c−U>0とすると

\[
a-\delta\le E_0(H)\le a,
\quad
\delta=\frac{\sqrt{g^2+4\beta^2}-g}{2}
=\frac{2\beta^2}{\sqrt{g^2+4\beta^2}+g}
\le\frac{\beta^2}{g}.
\]

証明：任意の正規化状態(p,q)のエネルギーは
\(a\|p\|^2+c\|q\|^2-2\beta\|p\|\|q\|\)以上である。その2×2行列の最小固有値を評価し、c−a≥c−Uを用いる。上側はP内の変分原理。標準的なblock perturbation/Schurの議論であり、この式自体を新しい定理と呼ばない。[P10]

Aの基底エネルギーをアルゴリズムでe_A±ε_algまで求めたなら、同じ成功事象上で

\[
E_0(H)\in[e_A-\epsilon_{alg}-\delta,\ e_A+\epsilon_{alg}].
\]

この基底エネルギーの幅から、任意の状態の時間発展が|t|δ以内と推論してはいけない。状態準備・overlap・誤った固有値の選択の扱いも、Aのsolver側の条件として残る。

### 6.3 cとβをどう取得するかが研究の本体

未占有orbitalのone-body gapだけを、そのままc−Uとみなしてはならない。cはdiscarded空間全体のmany-body Hamiltonianの下界である。

最も単純な実装可能経路は、\(H=H_{ref}+V\)、\(H_{ref}=c_0+\sum_i e_i n_i\)という対角one-body基準を置き、Q上のH_refの最小値を占有制約付きsortingで取得し、\(\|V\|\)の係数上界を差し引く方法である。βはP↔Qを結ぶ励起項の係数とnormから上界を作る。global係数normは正しいが、非常に緩い可能性がある。

改善候補として、Qを直交した少数のclass Q_jに分ける。各対角blockの下界c_j、block間norm v_jk、Pとの結合norm β_jを取得する。\(d_j=c_j-\sum_{k\ne j}v_{jk}\)を用いれば、2ab≤a²+b²からCを\(\oplus_jd_jI\)で下から評価できる。全g_j=d_j−U>0なら、

\[
\delta_* = \sum_j\frac{\beta_j^2}{g_j+\delta_*}
\]

の非負解が同様の保守的shift上界になる。単一の最小gapと全結合normへまとめるより、結合の弱い高エネルギーclassを区別できる。式の右辺はδについて減少するので、上側bracketを保持する二分法で計算できる。

ただしQ_j間の結合を無視しない。各Q_jを全occupationごとに作って指数個のclassへ増やさない。外部occupationの最初の違反、励起数、reference-energy帯など、係数から分類できる有限の粗いclassから始める。classをどう作ると下界の質と古典費用が両立するかが、構成研究の対象である。

### 6.4 Active-space生成器

1. 既知のorbital情報から初期active setとfrozen occupationsを提案する。
2. Aをnormal orderingで構築する。射影Pがfrozen occupationsならAは既存のone/two-body active Hamiltonianとして得られ、任意のdense many-body unitaryを作らない。
3. 既知の試行状態からU、係数からc_j/β_j/v_jkを計算し、δを評価する。
4. δが許容誤差を超える場合、寄与\(\beta_j^2/(g_j+\delta)\)や結合構造を手掛かりに、関連orbitalをactiveへ戻す。これは選択heuristicであり、最小active sizeの定理ではない。
5. 全量を再計算する。入れ替え後の誤差が必ず単調に小さくなると仮定しない。
6. 小さいAと誤差上界を出力するか、削減不可/保証不能として元の空間を返す。

最終資源はsystem qubitの減少だけで決めず、Aの相互作用、state preparation、PR/PF誤差、必要精度、古典証明費用まで比較する。

### 6.5 既知との差分・最小検証

SQuISHやSW/DUCC等が存在するので、active-space化や二次摂動そのものは新規性にしない。[P9–P11] 狙う差分は、真の基底state・真のgapを使わず、係数から計算可能な結合別上界と量子費用を使って、小さいHamiltonianを生成する手順である。

第一の判別は、実分子の大規模benchmarkではなく「U/c/βを求める方が元の問題より難しくなっていないか」「正しい上界が緩過ぎて一軌道も除けないか」である。

小さいgapped結合模型はboundの検査に使える。相関を持つactive block、外部結合、外部block同士の結合を含め、gapが小さい/認証不能な対照を置く。厳密対角化は評価用であり、候補生成・認証量の入力へ混ぜない。

一般の強相関系や小gapで本手順が働くと主張しない。保証不能と物理的に削減不能は異なる。証明手法が緩いだけの場合、数値上削減できたことを保証済みへ読み替えない。

## 7. 三案の関係、期待成果、比較

| 項目 | N1 | N2 | N3 |
|---|---|---|---|
| 変更する対象 | factorスペクトルとframe | diagonal係数と回路上の算術構造 | active/discarded空間 |
| 量子上の利益 | basis回転をなくす | 多数のphaseをまとめる | qubit/項数を減らす |
| 新規性候補 | partition＋gauge＋誤差制約の構成 | 疎整数因子を回路費用から生成 | 係数由来の結合別認証とactive選択 |
| 強い既知対照 | CDF/RCDF/SCDF、同じ近似での最良compiler、whole-Pauli | HWP、QROM、構造化low-rank Trotter、direct Pauli | frozen/natural active space、SQuISH、SW系、元H |
| 主な危険 | 既知shift/truncationに還元・回転が減らない | ancilla/加算が高い・整数制約で圧縮できない | lower boundが緩い・取得費用が高い |
| 最初の作業 | spectra→gate reductionの機構検査 | J→S,K,Eと算術実装の検査 | 小さい入力でboundsを真値なしに取得 |
| 現在の状態 | 数学的構成・十分bound、未実装 | exact表現・safe bound・構成案、未実装 | conditional bound・selector案、実用性未検証 |

論文化の強い着地点は、新しい入力→出力の構成法、改善理由、適用条件、同じtask/精度での資源効果である。新しい定理の存在や全入力での勝利を必須条件にしない。一方、既知式を再実装しただけ、λだけが改善、植込みtoyだけが成功、という結果は独立の新規アルゴリズムの根拠にはならない。

N1/N2が独立に効果を持つならそのまま扱う。合わせた場合の相乗効果を主張するなら、N1のみ/N2のみ/両方/元baselineを切り分ける。N3は目的がenergy-specificなので、時間発展全体の候補と誤差指標を混ぜて単一ランキングにしない。

## 8. 次のCodex検証の科学的契約案

今回、新しいproduction実行を開始したわけではない。設計を次の指示に用いる場合、実装・検証の詳細とresource guardはCodexにまとめて任せる。

最初の目的は改善率の大規模集計ではなく、N1で「既知のcutoff以上のgate消去があるか」、N2で「加算を含む利益があるか」、N3で「認証量を安価に取得できるか」を判別すること。

推奨する一体作業は、N1の有限constructorとN2の必要最小限primitiveを優先し、N3は下界取得の小さいfeasibilityだけを別ラベルで確認する形である。三方向へ同じ計算量を配分しない。数学的同値性で閉じる案に大量のcompileを行わない。

固定するのは、元の物理target、許されるapproximationとそのerror budget、比較対象、true ground/gapを使える箇所と使えない箇所、費用項の範囲である。toyサイズ、数値許容、実装API、CPU配分等をこの設計書で過剰に固定しない。

共通品質事項：

- 元Hに対する誤差を評価する。近似Hに対するPF誤差だけで精度一致としない。
- 回転数だけでなくCX/Toffoli・ancilla・uncompute・制御・state preparation・sampling normalizationを必要に応じて数える。
- logical T評価では有限合成精度を含める。RZ countをそのままT countとしない。
- 既知の安いbaselineにも同じcompilerの改善を使う。
- Exact-data mechanismと利用可能情報だけのselectorを分ける。
- 近似を戻すRTE残差は無料ではない。raw平均、補正first moment、channelの保証を分ける。
- 現行のSTO-3G、水素鎖、q1、epsilon=.05等を新研究へ固定条件として移さない。
- 新しい候補を中心研究へ自動採択しない。重要結果を保存・公開したら研究判断をGPTへ戻す。

今回の設計を受けて、同じ旧結果のレビューを繰り返す必要はない。必要になる次の科学レビューは、新候補に関する新しい構成/費用/認証結果が得られたときである。

## 9. 今回実施した代数チェック

以下は式の実装時取り違えを避けるためのGPT側の小さい決定論的チェックである。方法の科学的有望性や分子での資源改善の実証ではない。

N1：4 modes、ν=2、η=(1,1.002,−.4,−.397)を(1.001,1.001,−.3985,−.3985)へ揃えた。異なる群を混ぜるframe Vと、群内だけを混ぜるWを用意し、U=VWで確認した。

- sector内の実際の平方誤差：0.003018750000000236
- 上記mνによるbound：0.010010000000000453
- 近似後に群内Wを消したoperator残差：1.1590e−15
- 一つのGivensに対するgap×sinθの式も一致。

このεは説明用の単位なし入力であり、化学精度を満たす設定と主張しない。消去した回路のnative countは測っていない。

N2：4占有bit、2つのsigned chargesについて16状態を有理数で全確認し、二次形式の恒等式はexact error0。対称残差E03=E30=1/1000では最大誤差1/500となり、係数上界と一致した。

N3：2×2のA=0,C=10,B=.1,U=0では実energy shiftとδが0.000999900019995…で一致。U=1という緩い上界ではδが0.00111097397…へ広がる。C=.1まで近づく例ではδが0.061803…となる。これはboundの式と緩さを示す例であり、many-body cの取得可能性を証明しない。

コードと結果を付録へ収録する。新規分子入力や既存repositoryの科学moduleはimportしていない。

## 10. 参考資料と確認範囲

[R1] Repository固定結果。今回は既存判断と次設計への引継ぎ範囲を参照。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f98050e9402e8c65bb0f3dd27c7bd30d12fe7069/docs/research/representation_attribution_pair_results_20261010.md

[P1] P. Mukhopadhyay, N. Wiebe, H. T. Zhang, “Synthesizing efficient circuits for Hamiltonian simulation”, npj Quantum Information (2023), DOI:10.1038/s41534-023-00697-6. 本文、特に人工対称性、Algorithm1、部分・符号付き配分、truncation、回路費用を確認。
https://www.nature.com/articles/s41534-023-00697-6

[P2] J. Cohn, M. Motta, R. M. Parrish, “Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians”, PRX Quantum 2,040352 (2021). 要旨と公開本文の関連記述を確認。
https://doi.org/10.1103/PRXQuantum.2.040352
https://arxiv.org/abs/2104.08957

[P3] O. Oumarou et al., “Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization”, Quantum 8,1371 (2024). HTML v3のfactor fitting・regularization・誤差指標の関連部分を確認。
https://arxiv.org/html/2212.07957v3
https://quantum-journal.org/papers/q-2024-06-13-1371/

[P4] D. Rocca et al., “Reducing the runtime of fault-tolerant quantum simulations in chemistry through symmetry-compressed double factorization” (2024), DOI:10.1021/acs.jctc.4c00352. 要旨・symmetry shiftの関係を確認。全source再実装はしていない。
https://arxiv.org/abs/2403.03502
https://pubs.acs.org/doi/10.1021/acs.jctc.4c00352

[P5] K. Deka, E. Zak, “Simultaneously optimizing symmetry shifts and tensor factorizations for cost-efficient Fault-Tolerant Quantum Simulations of electronic Hamiltonians”, arXiv:2412.01338. 要旨で同時最適化の近接性を確認。
https://arxiv.org/abs/2412.01338

[P6] P. A. M. Casares et al., “Theory and practice of Trotter product formulas for quantum chemistry”, arXiv:2606.30741v1 (2026). 公開PDF、Fig.1の画面、factorization・QROM等の関連記述を確認。全数学・実験の独立検証ではない。
https://arxiv.org/abs/2606.30741
https://arxiv.org/pdf/2606.30741

[P7] G. H. Low, Y. Su, Y. Tong, M. C. Tran, “Complexity of Implementing Trotter Steps”, PRX Quantum 4,020323 (2023), arXiv:2211.09133（arXiv題名 “On the complexity of implementing Trotter steps”）. 要旨と公開された本文抜粋で構造利用型実装を確認。
https://doi.org/10.1103/PRXQuantum.4.020323
https://arxiv.org/abs/2211.09133

[P8] “Qronecker: A Certifiable Kronecker Compression Primitive for Quantum-Chemistry Hamiltonians”, arXiv:2603.06963 (2026). 要旨の確認のみ。本文の全保証を確認したとはしない。
https://arxiv.org/abs/2603.06963

[P9] “Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression”, arXiv:2211.16522. 要旨を確認。試行したv2 HTMLは取得できず、後で確認した公開履歴にはv1のみが表示されている。本文の全文監査とはしない。
https://arxiv.org/abs/2211.16522

[P10] S. Bravyi, D. P. DiVincenzo, D. Loss, “Schrieffer-Wolff transformation for quantum many-body systems”, Annals of Physics (2011), arXiv:1105.0675. 要旨と出版社本文抜粋で誤差理論の存在を確認。
https://arxiv.org/abs/1105.0675
https://www.sciencedirect.com/science/article/abs/pii/S0003491611001059

[P11] Z. Zhang et al., “Quantum algorithms for Schrieffer-Wolff transformation”, arXiv:2201.13304. 要旨の確認。詳細なAPI/complexityの比較は未実施。
https://arxiv.org/abs/2201.13304

[P12] Quantinuum, “Trotter dynamics with Hamming-weight phasing”, 公式algorithm documentation。既知のcompute–phase–uncomputeと、回転削減・adder/ancilla trade-offの具体例を確認。新提案の優位性の証拠にはしない。
https://docs.quantinuum.com/guppy/algorithms/examples/hamiltonian_simulation/trotter_hamming_weight_phasing.html

[P13] “Quantum Simulation of the First-Quantized Pauli-Fierz Hamiltonian”, PRX Quantum5,010345 (2024), AppendixF。対角整数のbinary signature/LCU構成が既知であることの確認に使用。今回の主候補にはその再提案を採用しない。
https://doi.org/10.1103/PRXQuantum.5.010345

### 文献上の未確定事項

- N1の縮退partition＋frame gauge＋sector誤差＋native費用という全手順の優先権。
- N2の疎整数制約・classical constructorが既知階層low-rank/Hamming方式に還元される範囲。
- N3の結合別certificateとactive選択に最も近い既報の完全な比較、および実用的many-body下界。

文献が見つからなかったことを「未解決と証明した」とはしない。今後の小さい実装と並行して、具体化された手順単位で近接研究を比較する。

## 付録A：独立した数式チェックの再現コード

NumPyを使用。repository/moduleを読み込まない。実行は `python design_algebra_checks.py`。古い科学runの再実行ではない。

```python
"""Small deterministic checks of new-design formulas. No repository imports or quantum benchmarks."""
from __future__ import annotations
from fractions import Fraction as F
import itertools, json, math
from pathlib import Path
import numpy as np

def sector_norm(x, nu):
    x = sorted(float(v) for v in x)
    if nu == 0: return 0.0
    return max(abs(sum(x[:nu])), abs(sum(x[-nu:])))

def givens(n,i,j,theta):
    q=np.eye(n); c,s=math.cos(theta),math.sin(theta)
    q[i,i]=q[j,j]=c; q[i,j]=-s; q[j,i]=s
    return q

def fock(u):
    n=len(u); occ=[[i for i in range(n) if state>>i&1] for state in range(1<<n)]
    ans=np.zeros((1<<n,1<<n),dtype=complex)
    for i,a in enumerate(occ):
        for j,b in enumerate(occ):
            if len(a)==len(b): ans[i,j]=np.linalg.det(u[np.ix_(a,b)]) if a else 1
    return ans

eta=np.array([1,1.002,-.4,-.397]); et=np.array([1.001,1.001,-.3985,-.3985]); n=4; nu=2
W=givens(n,0,1,.6)@givens(n,2,3,-.4)
V=givens(n,0,2,.35)@givens(n,1,3,-.22)
U=V@W
occupation=np.array([[int(s>>i&1) for i in range(n)] for s in range(1<<n)])
D=np.diag((occupation@eta)**2); Dt=np.diag((occupation@et)**2)
HU=fock(U)@D@fock(U).conj().T
HUt=fock(U)@Dt@fock(U).conj().T
HVt=fock(V)@Dt@fock(V).conj().T
idx=np.flatnonzero(occupation.sum(axis=1)==nu)
actual=float(np.linalg.norm((HU-HUt)[np.ix_(idx,idx)],2))
d=sector_norm(eta-et,nu); bound=d*(sector_norm(eta,nu)+sector_norm(et,nu))
assert actual <= bound+1e-12
assert np.linalg.norm(HUt-HVt,2)<1e-12
G=givens(2,0,1,math.pi/4); one=np.diag([1,1.002])
gate_error=float(np.linalg.norm(one-G@one@G.T,2))
gate_formula=abs(1-1.002)*abs(math.sin(math.pi/4))
assert abs(gate_error-gate_formula)<1e-12
out={"scope":"deterministic formula checks only; not molecular validation, new science batch, compiler benchmark or novelty proof", "N1":{
"modes":n,"particles":nu,"eta":eta.tolist(),"clustered_eta":et.tolist(),
"sector_square_error":actual,"analytic_sector_error_bound":bound,
"same_cluster_frame_removal_residual":float(np.linalg.norm(HUt-HVt,2)),
"givens_error":gate_error,"givens_formula":gate_formula}}

S=[[1,0],[1,1],[0,1],[-1,0]]; K=[[F(3,10),F(-1,10)],[F(-1,10),F(1,5)]]
J=[[sum(F(S[i][a])*K[a][b]*F(S[j][b]) for a in range(2) for b in range(2)) for j in range(4)] for i in range(4)]
errs=[]; perrs=[]
for bits in itertools.product((0,1),repeat=4):
    q=[sum(S[i][a]*bits[i] for i in range(4)) for a in range(2)]
    orig=sum(bits[i]*J[i][j]*bits[j] for i in range(4) for j in range(4))
    charge=sum(q[a]*K[a][b]*q[b] for a in range(2) for b in range(2))
    errs.append(abs(orig-charge)); perrs.append(F(2,1000)*bits[0]*bits[3])
assert max(errs)==0 and max(perrs)==F(1,500)
out['N2']={'occupations_checked':16,'S':S,'K':[[str(v) for v in row] for row in K],
'exact_max_identity_error':str(max(errs)), 'symmetric_E03_E30':'1/1000',
'perturbation_exact_norm':str(max(perrs)), 'coefficient_error_bound':'1/500'}

examples=[]
for c,beta,Ubound in [(10.,.1,0.),(10.,.1,1.),(.1,.1,0.)]:
    H=np.array([[0,beta],[beta,c]])
    actual=-float(np.linalg.eigvalsh(H)[0]); g=c-Ubound
    delta=2*beta**2/(math.sqrt(g*g+4*beta**2)+g)
    assert actual<=delta+1e-13
    examples.append({'A_ground':0.,'C_lower':c,'coupling_upper':beta,'A_upper':Ubound,
    'gap_lower':g,'actual_ground_shift':actual,'certificate_shift':delta})
out['N3']={'two_by_two_examples':examples,
'no_positive_gap_policy':'unresolved_or_promote_active_space; never assert compression error certified'}
print(json.dumps(out,indent=2,ensure_ascii=False))
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
```

## 付録B：実際の出力

```json
{
  "scope": "deterministic formula checks only; not molecular validation, new science batch, compiler benchmark or novelty proof",
  "N1": {
    "modes": 4,
    "particles": 2,
    "eta": [
      1.0,
      1.002,
      -0.4,
      -0.397
    ],
    "clustered_eta": [
      1.001,
      1.001,
      -0.3985,
      -0.3985
    ],
    "sector_square_error": 0.003018750000000236,
    "analytic_sector_error_bound": 0.010010000000000453,
    "same_cluster_frame_removal_residual": 1.159032638560308e-15,
    "givens_error": 0.0014142135623731922,
    "givens_formula": 0.0014142135623730961
  },
  "N2": {
    "occupations_checked": 16,
    "S": [
      [
        1,
        0
      ],
      [
        1,
        1
      ],
      [
        0,
        1
      ],
      [
        -1,
        0
      ]
    ],
    "K": [
      [
        "3/10",
        "-1/10"
      ],
      [
        "-1/10",
        "1/5"
      ]
    ],
    "exact_max_identity_error": "0",
    "symmetric_E03_E30": "1/1000",
    "perturbation_exact_norm": "1/500",
    "coefficient_error_bound": "1/500"
  },
  "N3": {
    "two_by_two_examples": [
      {
        "A_ground": 0.0,
        "C_lower": 10.0,
        "coupling_upper": 0.1,
        "A_upper": 0.0,
        "gap_lower": 10.0,
        "actual_ground_shift": 0.0009999000199950015,
        "certificate_shift": 0.0009999000199950017
      },
      {
        "A_ground": 0.0,
        "C_lower": 10.0,
        "coupling_upper": 0.1,
        "A_upper": 1.0,
        "gap_lower": 9.0,
        "actual_ground_shift": 0.0009999000199950015,
        "certificate_shift": 0.0011109739707595883
      },
      {
        "A_ground": 0.0,
        "C_lower": 0.1,
        "coupling_upper": 0.1,
        "A_upper": 0.0,
        "gap_lower": 0.1,
        "actual_ground_shift": 0.06180339887498948,
        "certificate_shift": 0.061803398874989486
      }
    ],
    "no_positive_gap_policy": "unresolved_or_promote_active_space; never assert compression error certified"
  }
}
```
