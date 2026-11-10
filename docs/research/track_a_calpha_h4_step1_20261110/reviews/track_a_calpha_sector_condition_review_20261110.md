# C-α／α0-P：sector保存条件の限定科学レビュー

- 作成日：2026-11-10（日本時間）
- 対象：元Track A・Claude案 C-α／α0-P、H4 Step 1の `TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED`。
- 承認：上記STOPを受けた「レビューを開始して」に基づく。研究方向全体や新規性の再調査ではなく、sector条件と信号保証の接続に限定する。
- 状態：**理論レビュー完了。前回の一律の個別primitive保存要求を修正する。実入力の適格性・数値性能・科学実行の再開は別の判断。**
- 実施：前回理論レビュー・Codex報告・正式共通指示の該当箇所の読解、固定SOURCEの限定的な静的確認、原PR論文の平均信号に関する部分の確認、紙上の証明と反例、本書作成。
- 非実施：科学配列の読取・再解析、行列生成・対角化・simulation、moment／信号／gap計算、sampling、テスト・compile、価格評価、repository変更、旧runのretry／resume。
- 本書は前回 `track_a_calpha_theoretical_review_20261110.md` の**追補・訂正**である。元ファイルおよび既存の科学結果を上書きしていない。

## 0. 結論

**個々のsampled primitiveがNα・Nβを保存することは、今回のα0-Pのcoherent平均信号をsector36で評価し、その信号へsector内gapを用いるための必要条件ではない。**

前回レビュー§1.3の「各sampled primitiveがNα,Nβを保存することを確認せず、sector36のgapを使ってはならない」という一律の要求は強すぎた。各primitiveが保存するなら十分ではあるが、今回の平均信号保証に必要な条件と同一ではない。[S1]

今回の固定構成では、次の**十分条件**で置き換えられる。

1. full Fock空間で定義した、正しい確率・相対位相を持つunitary trajectoryを、指定どおり独立にsampleする。
2. idealなtail平均がsectorを保存する。α0-Pでは、同じ辞書が表すHermitian残差Rがsectorを保存すれば、これは厳密に従う。
3. sectorへ制限して順に乗算するheadの完成block／sweepがsectorを保存する。内部のGaussian gateや単独Z／ZZ gateに同じ条件を課す必要はない。
4. 信号定理の参照Hと減衰作用素Γが同じsectorを保存し、元stateがそのsector内にある。gapはその参照Hのsector内gapを使う。
5. finite-bit、数値再構成、実装などによる理想式からのずれは別に評価する。数値的な0や未取得の誤差を、無条件の厳密保存・誤差0へ置き換えない。

このとき、**generic DF frame、既存tail辞書、K0のsampling law、sector36、保存stateを変更せずに、平均信号の二点検証を続ける数学的な経路がある。** spin-aware辞書やsector70への変更は、今回のSTOPを解消するための必須条件ではない。

ただし、これは現runの自動再開許可ではない。旧STOPは当時の固定条件に従った正しい停止として保全する。条件訂正を採用し、停止後の残作業を実行するには、その範囲についてユーザーの許可が必要である。

### 今回確定することと、確定しないこと

| 論点 | 本レビューの結論 |
|---|---|
| 個別primitiveのspin非保存だけで平均信号のsector解析を禁止できるか | **できない**。以下で反例と一般証明を示す |
| sector36のgapを使えるか | **参照H・Γがsectorを保存する等の条件の下で使える**。個別trajectoryの保存は不要 |
| 現在の46件が数値的に正しく検出されたか | Codexの報告値として受領。local runner／JSONを独立再取得・再実行していない |
| 実際のgeneric辞書の平均・headが必要条件を満たすか | 報告は支持材料だが、同じlawからの再構成との対応を確認する必要がある |
| m0／m17の位相・半径・総費用 | 未計算・未判定のまま |
| α0-Pを別の新アルゴリズムとして採択するか | 今回の対象外 |

## 1. 根拠と今回の確認範囲

本書では、【資料記載】、【直接静的確認】、【紙上導出】、【推奨】を分ける。

| ID | 資料 | 今回の扱い |
|---|---|---|
| S0 | ユーザーが本文で提示した「C-α／α0-P H4 Step 1：sector適用確認での技術的STOP」 | 全文を読解。46件、和の保存、実行範囲、未取得量、local未commit状態は**報告の記載**として扱う |
| S1 | `track_a_calpha_theoretical_review_20261110.md` | 原本を取得。§1.2–1.3、§2、§4、§5の証明、§7、§8.3等を確認。訂正の直接対象 |
| S2 | `貼り付けたテキスト（1）(8).txt` | 接続設計、raw半径、誤差内訳、Step 1／2／3の区分を参照 |
| S3 | `貼り付けたMarkdown(3).md` | Phase 0、保存入力、sector／Hermitian／gapの留保、二点scopeを参照 |
| G1 | `2e79d80dc9cce2885580e50fe6f1705da8fa41af` の `src/trotterlib/rte.py`、1585–1670行 | `_make_event`のatan角、正規化確率、event位相を直接静的確認 |
| G2 | 同commitの `src/trotterlib/df_partial_s2.py`、1–230、650–970行 | head blockの構造、basis往復・対角primitive、scalar位相、forward／reverseの順序を直接静的確認 |
| W1 | Günther et al., arXiv:2503.05647 | §II・§IV・付録AのHadamard平均とarctan型の平均式を確認。一般的な信号意味論の補助根拠 |
| I0 | `research_project_instructions_unified_2026-11-09.md` | レビュー・科学実行の区別、結果保全、研究記録、技術作業の役割に関する該当節を確認 |

[S0]にある `applicability.json`、`execution.log`、新runnerはユーザー環境のlocalパスで、報告上未commitである。今回それらを直接読了・監査したとは主張しない。原本未取得でも、提示されたlawと前回証明から**必要条件の理論的訂正**は判断できる。一方、実runの適格性をそのまま認定することはできない。

外部文献は新規性調査のために読み直したのではない。[W1]の取得テキストと画像表示には式番号の相違があったため、番号の一致は根拠にせず、共通して確認できたHadamard平均・atan恒等式だけを参照する。以下のsector命題・反例は、本書で明示的に導出したものである。

## 2. 最初に区別する三つの保存性

full Hilbert空間を \(\mathcal F\)、対象sectorを \(\mathcal S\)、その直交射影を \(\Pi\)、補空間射影を \(Q=I-\Pi\) とする。H4の今回の \(\mathcal S\) は \(N_\alpha=N_\beta=2\) の36次元sectorである。

\(J:\mathcal S\to\mathcal F\) を等長埋め込みとし、\(JJ^\dagger=\Pi\)、\(J^\dagger J=I_{\mathcal S}\) と書く。作用素の圧縮は \(A_{\mathcal S}=J^\dagger A J\) とする。

### 2.1 個別trajectoryの保存

各sampled unitary \(U_\omega\) に

\[
Q U_\omega\Pi=0
\]

が成り立つなら、初期stateがsector内にある各trajectoryは、当該block適用後もsector内にある。unitaryの有限次元不変部分空間なので、この場合は \([\Pi,U_\omega]=0\) と同値である。

### 2.2 coherent first momentの保存

平均作用素

\[
M=\mathbb E U_\omega
\]

について \([\Pi,M]=0\) が成り立つこと。これは個別trajectoryの保存より弱い。

\[
\mathbb E[Q U_\omega\Pi]=0
\]

が成立しても、各 \(Q U_\omega\Pi\) が0である必要はない。

### 2.3 平均channelにおける漏れ確率

state \(|\psi\rangle\in\mathcal S\) に対し、平均channelのsector外確率は

\[
\ell=\mathbb E\|Q U_\omega|\psi\rangle\|^2.
\]

一般に \(\mathbb E[Q U_\omega|\psi\rangle]=0\) から \(\ell=0\) は従わない。

**今回のHadamard型RPE信号に直接必要なのは2.2であり、2.3の漏れ確率が0であることではない。** ただし、sector内encoding、trajectory simulation、対称性測定でのpostselectionなどを別に行う場合には、2.1／2.3が重要になる。今回の理論訂正はそれらの用途まで緩和するものではない。

## 3. 独立samplingの平均とsectorへの制限

### 3.1 順序付き平均の積

【紙上導出】\(U_1,\ldots,U_N\) が独立なrandom unitary、\(M_j=\mathbb E U_j\) なら、非可換でも

\[
\boxed{\mathbb E[U_N\cdots U_1]=M_N\cdots M_1.}
\tag{3.1}
\]

証明は、確率分布の積を使い、行列の順序を変えずに和を一つずつ取ればよい。同分布性はこの恒等式そのものには不要だが、今回の固定lawでは各round内をIIDとする。決定論blockはそのまま間に挿入できる。

これは平均channelの積ではない。また同じlabelや同じsampled circuitを複数箇所で再使用する場合には、独立性を失い、一般に \(\mathbb E[U^N]\ne(\mathbb E U)^N\) となる。

### 3.2 保存する平均blockで制限する

\(M_j\Pi=\Pi M_j\Pi\) が各平均blockで成り立てば、帰納的に

\[
M_N\cdots M_1J=J(M_{N,\mathcal S}\cdots M_{1,\mathcal S}).
\]

従って

\[
\boxed{
\langle\psi|\mathbb E[U_N\cdots U_1]|\psi\rangle
=
\langle\psi_{\mathcal S}|M_{N,\mathcal S}\cdots M_{1,\mathcal S}|\psi_{\mathcal S}\rangle.
}
\tag{3.2}
\]

ここで \(|\psi\rangle=J|\psi_{\mathcal S}\rangle\)。**この証明には各 \(U_j\) のsector保存を使っていない。**

さらに、microstepごとの保存を示せなくても、独立なmacrostepの平均 \(S_\delta\) がsectorを保存すれば、macrostep境界だけで

\[
\langle\psi|S_\delta^q|\psi\rangle
=\langle\psi_{\mathcal S}|(S_{\delta,\mathcal S})^q|\psi_{\mathcal S}\rangle
\tag{3.3}
\]

とできる。ただし、(3.3)だけでは、内部の全因子を個別にsectorへ圧縮してよいとは限らない。計算で射影を挿入する境界と、証明済みの保存境界を対応させる必要がある。

## 4. α0-Pでは、平均tailの保存がRの保存から従う

### 4.1 full空間でのlawを先に固定する

signを吸収したHermitian involutionを \(Q_l\) とし、

\[
Q_l^\dagger=Q_l,\qquad Q_l^2=I,\qquad
R=\lambda_R\sum_l p_lQ_l,
\quad p_l\ge0,\quad\sum_l p_l=1.
\]

\(\lambda_R\) は**既存のfull辞書でのmass**である。sectorへ制限した後に別のmassへ置き換えたり、sector内で消えるatomを勝手にdropしたりしない。

\[
\phi=\arctan(\lambda_R\theta),\qquad
U_l=\cos\phi\,I-i\sin\phi\,Q_l
\]

なら、full空間で厳密に

\[
\boxed{
M_P(\theta)=\sum_l p_lU_l
=\frac{I-i\theta R}{\sqrt{1+\lambda_R^2\theta^2}}.
}
\tag{4.1}
\]

[G1]はこのatan型角度を実装している。理想的な確率と、実装された有限bit確率・角度の差は§8で分ける。

### 4.2 個別atomの保存なしに平均が保存する

\([\Pi,R]=0\) なら、(4.1)から直ちに

\[
[\Pi,M_P]=0,\qquad
M_{P,\mathcal S}=
\frac{I_{\mathcal S}-i\theta R_{\mathcal S}}
{\sqrt{1+\lambda_R^2\theta^2}}.
\tag{4.2}
\]

**\([\Pi,Q_l]=0\) を要求する必要はない。** \(\sum_l p_l Q_l\) のsector外blockが相殺すればよい。

また、\(\|R\|\le\lambda_R\) はfull辞書の三角不等式から従い、\(M_P\) はunitary平均なのでcontractionである。補正平均 \(I-i\theta R\) は一般にcontractionでない。この二つを取り違えない。

### 4.3 減衰generatorも同じsectorを保存する

前回の厳密式

\[
R_\theta=\theta^{-1}\arctan(\theta R),
\]
\[
\Gamma_\theta=
\frac{\log(1+\lambda_R^2\theta^2)I-\log(I+\theta^2R^2)}{2\theta}
\succeq0
\tag{4.3}
\]

はRのspectral関数である。Hermitian Rがsectorを保存すれば、そのsectorはreducing subspaceであり、spectral関数も同じ直和に分かれる。

\[
[\Pi,R_\theta]=[\Pi,\Gamma_\theta]=0,
\]
\[
(\Gamma_\theta)_{\mathcal S}
=
\frac{\log(1+\lambda_R^2\theta^2)I_{\mathcal S}
-\log(I_{\mathcal S}+\theta^2R_{\mathcal S}^2)}{2\theta}.
\tag{4.4}
\]

このため、\(\gamma\) と \(v_\Gamma\) をsector内で計算しても、full演算子から得る値と一致する。個別trajectoryが他spin sectorへ動くことを理由に、参照generatorのgapまで自動的にsector70へ広げる必要はない。

### 4.4 headは完成blockで保存すればよい

各head Hamiltonian \(F_a\) がsectorを保存するなら、\(e^{-itF_a}\) も保存する。forward／reverse sweepがその積なら両sweepも保存する。

DF実装が \(G e^{-it f(\hat n)}G^\dagger\) の形を取る場合、**Gや内部の各対角gateが個別にspinを保存する必要はない**。完成したblockが元の保存する \(F_a\) の指数を表すことが重要である。

[G2]の `_append_block` はbasis変換・対角primitive・逆変換を一組として構成し、`_build_parts` はこれをforward／reverseの順に配置する。これは上の解析境界と対応する。ただし、SOURCEを読んだだけで、現在の保存inputについて数値再構成誤差・全native loweringが適格になったわけではない。

### 4.5 全PR信号のsector式

既知scalarをc、左右headを \(D_{\rm fwd},D_{\rm rev}\) とする。独立samplingでのmacrostep平均は

\[
S_\delta=e^{-ic\delta}D_{\rm rev}M_P(\theta)^rD_{\rm fwd},
\qquad\theta=\delta/r.
\]

Rと完成した左右headがsectorを保存すれば、

\[
S_{\delta,\mathcal S}
=e^{-ic\delta}D_{{\rm rev},\mathcal S}
\left[\frac{I_{\mathcal S}-i\theta R_{\mathcal S}}
{\sqrt{1+\lambda_R^2\theta^2}}\right]^r
D_{{\rm fwd},\mathcal S},
\]
\[
\boxed{
z_m=\langle0|S_\delta^{q_m}|0\rangle
=\langle0_{\mathcal S}|S_{\delta,\mathcal S}^{q_m}|0_{\mathcal S}\rangle.
}
\tag{4.5}
\]

この式は**full-space trajectoryの平均信号と同じ量**であり、sector外に出るtrajectoryを捨てたり、sectorを測定したりする操作を含まない。scalarは一回だけ計上する。

## 5. 前回の位相・半径定理は、どのsector保存を使っているか

### 5.1 定理の対象は参照発展である

前回§5の証明に現れるのはH、Γ、元state、gap、収縮semigroupであり、個々のrandom atomではない。[S1]

今回の十分条件として、Hermitian参照 \(H_{\rm ref}\) と \(\Gamma_\theta\) が \(\Pi\) と可換とする。すると

\[
J^\dagger e^{(-iH_{\rm ref}-\Gamma_\theta)T}J
=e^{(-iH_{\rm ref,\mathcal S}-\Gamma_{\theta,\mathcal S})T}.
\tag{5.1}
\]

\(|0_{\mathcal S}\rangle\) が \(H_{\rm ref,\mathcal S}\) の非縮退基底状態、energyがE0、gapが \(\Delta_{\mathcal S}>0\) とする。必要なのは**このsectorでの基底状態**であり、他sectorも含めた全Fock空間で最も低いenergyであることではない。

\[
\gamma=\langle0_{\mathcal S}|\Gamma_{\theta,\mathcal S}|0_{\mathcal S}\rangle,
\quad
v_\Gamma=\langle0_{\mathcal S}|\Gamma_{\theta,\mathcal S}^2|0_{\mathcal S}\rangle-\gamma^2.
\]

前回と同じ証明を \(\mathcal S\) 内で適用でき、

\[
\boxed{
\left|\langle0|e^{(-iH_{\rm ref}-\Gamma_\theta)T}|0\rangle
-e^{-iE_0T-\gamma T}\right|
\le v_\Gamma\min\left\{\frac{T^2}{2},\frac{2T}{\Delta_{\mathcal S}}\right\}.
}
\tag{5.2}
\]

他sectorへ出る**実trajectory**と、sector内に閉じる**参照減衰発展**を混同しない。この定理は前者一つ一つを近似する保証ではない。

### 5.2 証明中に必要な条件の再確認

\(P_0=|0_{\mathcal S}\rangle\langle0_{\mathcal S}|\)、\(Q_0=I_{\mathcal S}-P_0\)、\(g=Q_0\Gamma_{\theta,\mathcal S}|0_{\mathcal S}\rangle\) とする。これはsector射影 \(\Pi\) と別の射影である。

energyと主要減衰を外した参照stateの \(P_0,Q_0\) 成分を \(a(t),y(t)\) とすると、

\[
a'=-g^\dagger y,\quad y'=-iBy-ga,
\]
\[
B=Q_0(H_{\rm ref,\mathcal S}-E_0)Q_0
-i(Q_0\Gamma_{\theta,\mathcal S}Q_0-\gamma Q_0).
\]

\(\Gamma_{\theta,\mathcal S}\succeq0\) より \(|a(t)|\le e^{\gamma t}\) と \(\|e^{-iBt}\|\le e^{\gamma t}\)。積分kernel \(F(u)=\int_0^u e^{-iBv}dv\) を使うと

\[
a(T)-1=\int_0^T g^\dagger F(T-s)g\,a(s)\,ds.
\]

積分の絶対値評価で \(v_\Gamma T^2/2\) を得る。さらにsector内gapがあれば、\(\operatorname{Re}B\succeq\Delta_{\mathcal S}I\) から \(\|B^{-1}\|\le1/\Delta_{\mathcal S}\)。

\[
F(u)=iB^{-1}(e^{-iBu}-I)
\]

を使って、absoluteな信号差は

\[
\frac{v_\Gamma}{\Delta_{\mathcal S}}
\left[T+\frac{1-e^{-\gamma T}}{\gamma}\right]
\le\frac{2Tv_\Gamma}{\Delta_{\mathcal S}}
\]

で抑えられる。\(\gamma=0\) ならPSD性から \(g=0\) として扱う。この再確認のどこにも \([\Pi,Q_l]=0\) は登場しない。

### 5.3 実際のPR平均との誤差をつなぐ

(4.5)のsector平均と(5.1)の参照発展との差を、実際に正当化した \(\epsilon_{\rm dyn,\mathcal S}(T)\) で抑える。

\[
d(T)=v_\Gamma\min\{T^2/2,2T/\Delta_{\mathcal S}\}
+\epsilon_{\rm dyn,\mathcal S}(T).
\]

\(d(T)<e^{-\gamma T}\) のとき、前回のdisk幾何をそのまま適用できる。

\[
\boxed{
\rho_{\rm lb}=e^{-\gamma T}-d(T)>0,\qquad
\beta_{\rm sys}=\arcsin(e^{\gamma T}d(T)).
}
\tag{5.3}
\]

実装誤差等でfull平均と理想sector平均に差が残る場合は、§8の接続誤差もdへ**一度だけ**加える。

前回の \(C_{\rm split}\)、arctan補正、head誤差、model誤差はなくならない。sector内のnormで評価するなら、対象となる完成作用素の制限が正しいことを要する。完成headと理想head指数の誤差は、各完成blockが不変ならsector内で評価できる。一方、非保存の内部gateをすべてsectorへ射影して、その積からhead誤差を作ってはいけない。

### 5.4 保持する留保

- 保存gapの値があることと、認証済み下界 \(\Delta_-\) があることは別である。今回の条件訂正はgapの資格を引き上げない。
- gap下界を使えなければ、前回どおりgap非依存の \(T^2/2\) 側を使える。有用な強さになるとは限らない。
- raw `H_phys` の反Hermitian数値残差、明示した `H_solve`、元targetとDF modelの対応は、従前の留保を維持する。Hermitian参照への変更やstate置換を黙って行わない。
- 元stateを厳密固有状態とする研究promiseと、保存浮動小数点vectorへの厳密包含証明を分ける。
- 今回訂正するのはsector適用条件であり、(5.2)の独立新規性、実分子での鋭さ、全RPE成功、費用優位を新たに認定するものではない。

## 6. 射影の仕方を誤ると、同じlawではなくなる

### 6.1 平均と圧縮の線形性は使える

線形性から

\[
J^\dagger(\mathbb E U)J=\mathbb E[J^\dagger UJ]
\]

である。したがって「full行列を必ず全て数値生成してから射影する」ことが必須なのではない。full作用素として正しい恒等式を使って、sector内で直接その圧縮を計算してよい。

α0-Pなら

\[
J^\dagger U_lJ=\cos\phi I_{\mathcal S}-i\sin\phi\,J^\dagger Q_lJ
\]

は正しい。しかし、右辺は一般にsector内unitaryではない。

### 6.2 非線形操作は交換できない

一般に

\[
J^\dagger e^{-i\phi Q_l}J
\ne e^{-i\phi J^\dagger Q_lJ},
\tag{6.1}
\]

なぜなら \((J^\dagger Q_lJ)^2\ne I_{\mathcal S}\) となり得るからである。

また、個別trajectoryで

\[
\Pi U_2U_1\Pi
=\Pi U_2\Pi U_1\Pi+\Pi U_2Q U_1\Pi.
\tag{6.2}
\]

右辺第二項を削るのは、sector外へ出て戻る経路を削ることである。個別trajectoryのstatevectorをsector36内で更新・再正規化してはならない。

独立samplingと保存する平均がある場合、(6.2)の第二項の**平均**が0となり、(3.2)が使える。これは各trajectoryで項が消えるという意味ではない。解析的に圧縮eventの平均を使うことは可能だが、それをsector内の実unitary回路やtrajectory漏れ診断の代替として扱わない。

### 6.3 一般指数lawへはそのまま拡張しない

一般の \(M(\theta)=\mathbb E e^{-i\theta X}\) では、\([\Pi,\mathbb EX]=0\) だけから \([\Pi,M(\theta)]=0\) は従わない。二次以降のmomentがsectorを破る可能性がある。

その際、圧縮後の二次momentは

\[
J^\dagger X^2J
=(J^\dagger XJ)^2+J^\dagger XQXJ.
\tag{6.3}
\]

後半はPSDであり、圧縮したXだけを平方すると失われる。今回のα0-Pは、full空間で \(Q_l^2=I\) を使って平均を**厳密にRの一次関数**にできるため、この追加問題を回避している。

前回§8.3の「sector内のfragment端点を用いて半幅上界を作る」という別の議論では、各fragmentのsector保存、または(6.3)の追加項を扱うことが必要となる。今回の訂正を、α1／α2のsector端点や全てのsampling lawへ一括適用しない。

## 7. 記号的な反例：平均が保存しても、trajectoryは漏れる

【紙上導出。数値simulationではない。】二次元空間で、\(\Pi=|0\rangle\langle0|\)、Pauli X,Z、\(0<\chi<\pi/2\) とし、

\[
Q_\pm=\cos\chi\,Z\pm\sin\chi\,X.
\]

\(Q_\pm\) はHermitian involutionだが、どちらもsectorを保存しない。等確率で

\[
U_\pm=e^{-i\phi Q_\pm}
\]

を選べば、

\[
M=\cos\phi\,I-i\sin\phi\cos\chi\,Z,
\qquad[\Pi,M]=0.
\]

\(R=\lambda\cos\chi Z\)、\(\phi=\arctan(\lambda\theta)\) とすれば、これはα0-Pの正確な例である。

各trajectoryは

\[
U_\pm|0\rangle
=(\cos\phi-i\sin\phi\cos\chi)|0\rangle
\mp i\sin\phi\sin\chi|1\rangle
\]

となり、sector外確率は

\[
\ell=\sin^2\phi\sin^2\chi
\]

である。それでも漏れ**振幅**の平均は相殺し、独立なN eventのcoherent信号は

\[
\langle0|M^N|0\rangle
=(\cos\phi-i\sin\phi\cos\chi)^N
\]

とsector内だけで正確に計算できる。

一方、先にgeneratorを圧縮すると \(Q_{\pm,\mathcal S}=\cos\chi\) であり、その指数は \(e^{-i\phi\cos\chi}\)。絶対値1となり、実際の減衰を失う。これが禁止すべき射影計算である。

この例は、今回の46件の分子入力について漏れの大きさを推定するものではない。**「個別保存は平均保存の必要条件でない」「個別射影した指数は別operatorである」**という論理的な区別だけを示す。

## 8. finite-bit・近似的保存・実装誤差はどう扱うか

### 8.1 理想平均と実装平均を同一視しない

理想の確率とunitaryを \(p_l,U_l\)、実装のものを \(\widehat p_l,\widehat U_l\) とする。両確率が正規化されているとき、

\[
\widehat M=\sum_l\widehat p_l\widehat U_l,\qquad
\|\widehat M-M\|
\le\sum_l|\widehat p_l-p_l|
+\sum_l\widehat p_l\|\widehat U_l-U_l\|.
\tag{8.1}
\]

同じ正確なatan角と同じinvolutionを用い、確率だけが変わる場合は、\(\sum_l(\widehat p_l-p_l)=0\) によりidentity項が消え、

\[
\|\widehat M-M\|
\le |\sin\phi|\sum_l|\widehat p_l-p_l|.
\tag{8.2}
\]

これらは十分上界である。最良の誤差評価を要求する趣旨ではなく、**理想的な相殺が有限bitでも自動的に厳密だとは扱わない**ための一つの経路である。

回転のglobal phaseをcontrol前に捨てると、system-only channelが同じでもHadamard平均は変わり得る。既存のcontrolled相対位相規約を維持する。近似unitary生成・basis・scalarの誤差も必要な場所で会計する。

### 8.2 実装平均が厳密にsectorを保存しなくても、誤差評価で接続できる

理想の平均blockを \(A_j\)、実際の平均blockを \(\widehat A_j\) とする。全てcontractionで、理想blockはsectorを保存するとする。

\[
e_j=\|(\widehat A_j-A_j)\Pi\|
\]

を評価できれば、telescopingにより

\[
\left\|(\widehat A_N\cdots\widehat A_1-A_N\cdots A_1)\Pi\right\|
\le\sum_{j=1}^N e_j.
\tag{8.3}
\]

証明は、差をj番目のblockの差の和へ展開し、右側の理想積はsectorを保ち、左側の実装積は全空間でcontractionであることを使う。

ここで必要なのは、sectorから**外へ出る成分も含むcolumn norm**である。

\[
\|\Pi(\widehat A_j-A_j)\Pi\|
\]

だけでは、後続blockで外から戻る寄与を抑えられない。

(8.3)があれば、理想参照のgapはsector内のものを保ち、実装による対称性破れは \(\epsilon_{\rm impl}\) 等へ入れられる。実装の微小な非保存のたびに、全Fock gapや別sectorへ変更する必要はない。ただし誤差が長時間に累積して許容枠を超える場合は、保証は成立しない。

### 8.3 近似的なmacrostep保存に対する別の安全な上界

比較のため、actualなmean macrostep Sがcontractionだが厳密保存しない場合を考える。

\[
S_{\rm bd}=\Pi S\Pi+Q S Q,
\qquad\ell_S=\|S-S_{\rm bd}\|.
\]

\(S_{\rm bd}\) もcontractionなので、

\[
\left|\langle0|S^q|0\rangle
-\langle0_{\mathcal S}|(J^\dagger S J)^q|0_{\mathcal S}\rangle\right|
\le q\ell_S.
\tag{8.4}
\]

これは紙上の安全な代替評価であり、今回 \(\ell_S\) を算出したわけではない。相殺・往復を利用していないため保守的になり得る。

(8.3)や(8.4)のいずれかで既に計上した差を、model差・head差・数値差として重ねて加算しない。必要な追加項を本書で \(\epsilon_{\rm conn}(T)\) と呼ぶ場合も、新しい誤差予算枠を自動で新設する意味ではない。

### 8.4 科学的な診断と認証を分けて進める

今回の目的は最初の二点の成立性検証であり、理想lawの診断を行う前に、全hardware・全native回路の厳密認証を終えることを新たな条件にしない。

一方で、未知のfinite-bit誤差を0と置いて実装込みの保証PASSを出してもいけない。従前どおり、

- 理想演算子モデルに対する条件付きの理論上界、
- 保存float入力に対する数値diagnostic、
- 丸め・入力誤差まで囲んだ上界、
- 未取得／未資格の実装誤差

を別に記す。必要な資格が不足する場合、取得した信号やmomentの診断結果は保存できるが、認証済みの費用優位には昇格させない。

## 9. Hadamard測定・shot数・回路費用への意味

### 9.1 測定するのはtrajectoryの振幅平均

ordinary-controlledなUωを使い、初期stateを \(|\psi\rangle\) とすれば、control後の状態は

\[
\frac{|0\rangle_a|\psi\rangle+|1\rangle_aU_\omega|\psi\rangle}{\sqrt2}.
\]

systemをtraceしたancillaのcoherenceは \(\langle\psi|U_\omega|\psi\rangle\) で決まる。既存のX/Y符号規約に従えば、

\[
\mathbb E X=\operatorname{Re}\langle\psi|\mathbb E U_\omega|\psi\rangle,
\quad
\mathbb E Y=\operatorname{Im}\langle\psi|\mathbb E U_\omega|\psi\rangle.
\tag{9.1}
\]

この基本原理は原PR論文にも明示されている。[W1] 個々のtrajectoryでspinが変わっても、測定を途中で選別しない限り、上式は変わらない。

### 9.2 個別sector漏れに独立のshot倍率を足さない

各軸での一shot outcomeは±1である。fresh IID trajectoryと測定を繰り返すなら、回路乱数と測定乱数を合わせた全outcomeに、前回のHoeffding十分条件が使える。

\[
N_{m,b}=\left\lceil
\frac{4\log(2/\alpha_{m,b})}
{\rho_{{\rm lb},m}^2\sin^2\beta_{{\rm stat},m}}
\right\rceil.
\tag{9.2}
\]

無重みの±1 outcome \(Y_b\) の分散は、両方の乱数を平均した \(\mu_b\) に対して \(1-\mu_b^2\) である。trajectoryのばらつきだけを別に追加したり、sector内に残る確率の逆数を機械的に掛けたりしない。

これはsector漏れに物理的効果がないという意味ではない。その効果は実際のcoherent平均・半径に既に現れ得る。さらにsectorを測定して漏れたshotをdiscardするなら、測定protocol・分布・費用が変わるので、(9.1)–(9.2)を無変更では使えない。

同じtrajectoryを多数shot再使用する場合も、fresh IIDの十分条件とは区別する。今回のscopeへ自動追加しない。

### 9.3 36次元の古典計算と量子回路の縮小は別

平均信号を36次元で計算できても、実回路が36次元sector内に閉じることや、system qubitを削減できることは意味しない。

実際のcontrolled trajectoryは、元のGaussian変換・Z／ZZ回転・scalar・wrapperを含むfull-spaceの回路として扱う。費用 \(\mathbb E C\) はその実装を対象にする。sector内平均を一つのgateとして無料で実行できるとはしない。

本レビューにより、C-αの減衰・shot・総費用が数値的に改善したわけではない。今回は**平均信号の古典評価を止めていた、過剰に強い適用条件を取り除いた**ことが到達点である。

## 10. 今回のSTOP報告の解釈

### 10.1 支持されること

[S0]によれば、固定generic frameの非zero単一Z atomにspin混合が46件検出され、元H／DF fragment等のsector外blockは数値診断で0だった。これは、個別atomの保存と演算子和の保存が異なるという本レビューの整理と矛盾しない。

当時の指示は個別atom保存を要求していたため、Codexがそこで止まったことは適切だった。**停止を取り消して旧runをPASSへ書き換える理由にはならない。** 修正すべきなのは、私が前回レビューから指示へ持ち込んだ一律の必要条件である。

### 10.2 この報告だけでは、まだ認定しないこと

- 46件を独立再現したこと。
- 非保存atomのsampling頻度や、回路の漏れ確率の大きさ。
- 小係数だから無視してよいという結論。
- 辞書の実際のsigned atom和と、保存されたR／DF fragmentが厳密に一致すること。
- 有限bitの実装平均が厳密にsectorを保存すること。
- 完成したhead blockの数値・native再構成が正しいこと。
- Γのmoment、m0／m17の位相・半径、0.14 rad枠への適格性、費用優位。

特に、報告の「元fragmentが保存する」から、別の数値再構成で作ったtail平均が完全に同じRを使うことまで自動で断定しない。適用確認は、**実際の固定辞書・係数・identity抽出・平均式との結合**を対象にする。

### 10.3 既存K2結果を取消す根拠ではない

今回の個別primitive非保存は、既存K2の保存平均信号が誤っていることを直接示さない。finite polynomialの平均も、同じ保存するRの関数として正しく構成されていればsectorへ制限できる。

ただし、これは過去K2実装の全体再監査を完了したという主張でもない。既存固定結果・留保・STOPはそのまま保持する。

## 11. 前回レビュー・実行条件への具体的な訂正

### 11.1 §1.3の置換文案

> **sector36を使う条件**：元stateがsector36にあり、理論参照のH・Γがそのsectorを保存することを確認する。α0-Pでは、full空間の正しいinvolution辞書が表すRと、sectorへ制限して合成する完成head blockがsectorを保存し、独立sampling・controlled位相・既知scalarの規約が維持されれば、各sampled atomや各trajectoryがNα・Nβを保存しなくても、coherent平均信号をsector36で評価し、参照Hのsector内gapを使える。有限bit等の非保存・再構成差は別に評価する。個々のprimitiveをsector内unitaryとみなすこと、sector外に出たtrajectoryを除外することは、この条件からは許されない。

これを採用しても、sampling law、既存の係数、frame、λ、state、物理target、時間、r、systematic予算は変えない。変わるのは**理論の適用確認条件**である。

### 11.2 修正しない部分

- α0-Pとα0-Eの区別。
- arctan由来のHermitian補正。
- 元H／DF model／Hermitian参照／保存stateの区別。
- Γを含むsplit誤差、head誤差、有限bit・実装誤差。
- gap下界、moment上界の資格状態。
- 前回§5・§7の定理、disk→位相・半径→shotの写像。
- α1／α2におけるsector端点・momentの別条件。
- 既存mandatory STOP、他Trackの成果、科学実行の承認境界。

### 11.3 成立条件を無制限に緩めない

「平均がspinを保存するらしい」という説明だけでは足りない。Rが保存しても、一般指数lawの平均や、非保存のhead細分化の積まで自動保存とはならない。今回採用を推奨するのは、**α0-Pの厳密な一次平均と、完成blockの保存を使う、特定の証明経路**である。

## 12. 次の作業と、再開の判断

### 12.1 推奨

**spin-aware辞書やsector70へ切り替えるのではなく、固定generic辞書を維持し、上記の平均信号用のsector条件を採用して、残るStep 1の検証へ接続することを推奨する。**

ただし、現runは既にmandatory STOPしている。ユーザーが条件訂正と残作業の実行を許可するまでは、追加の科学計算を開始しない。新しい承認証明書やgrantの発行は不要である。

### 12.2 Codexへ戻す確認は限定する

採用・実行許可後に必要なのは、今回の証明をもう一度大規模に調査することではなく、次の対応付けである。

| 確認対象 | 必要な内容 | 不要な作業・避ける変更 |
|---|---|---|
| 同じlawのR | 既存signed atom和、identity抽出、λ、Rの対応。元fragmentとの差があれば誤差として区別 | 非保存46件をdropしない。λやpを再最適化しない |
| 平均tail・Γ | (4.1)–(4.4)の意味でsector制限できること。差は資格付きで扱う | 各atomをsector内involutionに偽装しない |
| head | sectorへ射影する境界が完成block／sweepであること。内部gateの射影をしない | 新しいspin-aware basisへ置換しない |
| 参照H・state・gap | 既存の保存対象とHermitian参照の対応を保持。gapの資格を維持 | 新SCF・DF fit・ground solve・別sectorへの移動をしない |
| 実装・数値差 | 理想lawの診断とfinite-bit保証を分離。既存の誤差台帳に一度だけ含める | 未知項を0にせず、同じ差を二重計上しない |
| 残る科学評価 | 固定θのoperator量とm0／m17だけ。前回未取得の出力を得る | 全18round・r/LD探索・compile・sampling・価格取得へ広げない |

既存のinput identity・46件の診断・原SOURCE確認を、理由なく最初から再実行する必要はない。使える保存結果を参照する。新runner／結果を公開・照合する場合も、科学入力の秘密保護と既存未commit作業の保全を守り、研究上必要な最小範囲にする。

現在の専用branch／worktreeを継続利用できるかは、既存の保護・並列作業と照合して判断する。新たな検証のたびにbranchを増やす必要も、別研究へmergeする必要もない。

### 12.3 今回変更しない実行範囲

[S0]に記載されたH4／1.00 Å／STO-3G、保存DF12、generation-prefix LD3、元state、sector36、正時間partial-S2、paired K0、δ=.02、r4、m0／m17、systematic .14 rad等を、後続scopeの基礎として維持する。

本書は新たなthreshold・倍率・追加roundを定めない。既存二点の数値が未取得であることも変わらない。

同じfixed inputを用いたmean-only接続に必要な差分確認と、残るmoment／信号／上界評価を一括して扱えばよい。新しい本質的な定義矛盾がなければ、形式的なGPTレビューをもう一段挟む必要はない。

旧runの技術的STOPはそのまま保存し、再開後の結果には、どの適用条件を訂正した後の実行かを記す。旧runを失敗していなかったことにしたり、承認済みの別runの結果を上書きしたりしない。

## 13. 確信度と残存リスク

| 判断 | 確信度 | 留保 |
|---|---|---|
| 独立samplingのfirst momentの積分解 | 高 | 独立性・順序・正しいlawが前提 |
| α0-PのR保存から平均・Γ保存が従う | 高 | full-space involution恒等式と同じλ・確率の下で厳密 |
| 個別primitive保存が今回の信号定理の必要条件ではない | 高 | 一般証明と記号的反例により確認 |
| sector内gapで前回信号定理を使える | 高 | 参照H・Γのsector保存、元stateとgapの適合性が条件 |
| 固定generic辞書で実runを進められる | 条件付き | 新runner／JSONは独立未取得。同じR・head・参照への接続確認が残る |
| 数値gap・moment・実装誤差の認証 | 未実施 | 今回の理論訂正からは得られない |
| 固定r4での位相・半径の良否、総費用改善 | 未判定 | 信号・momentが未計算 |

### 最終判断

**今回の停止を「spin保存する別辞書が必要」という結論へ直結させない。前回の必要条件の書き方を訂正し、同じgeneric辞書のcoherent平均に対するsector解析へ進む方針を支持する。**

これはC-αの成功を保証するものではなく、正しく検証できる適用条件を定め直したものである。旧STOPは保持し、条件変更後の科学実行は別途の明示許可後とする。

## 14. 参照資料

### 添付・会話資料

- **[S0]** 2026-11-10のユーザー提供本文「C-α／α0-P H4 Step 1：sector適用確認での技術的STOP」。status `TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED`。local branch `codex/track-a-calpha-h4-step1-20261110`、基点 `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。本書では報告本文のみ読解し、localの新runner・JSON・logを独立取得したとはしない。
- **[S1]** `track_a_calpha_theoretical_review_20261110.md`。特に§1.3、§2.1、§4.1–4.4、§5、§7、§8.3、§8.5。§1.3の一律要求を本書で訂正。
- **[S2]** `貼り付けたテキスト（1）(8).txt`。Codexの静的接続設計・最小検証計画。
- **[S3]** `貼り付けたMarkdown(3).md`。Phase 0実行前確認。
- **[I0]** `research_project_instructions_unified_2026-11-09.md`。§0、§7–10等。レビュー承認と科学実行承認、原本・STOP保全、詳細Markdown記録の規則。

### 固定GitHub資料

- **[G1]** `src/trotterlib/rte.py`、fixed commit `2e79d80dc9cce2885580e50fe6f1705da8fa41af`。今回直接確認は1585–1670行。blob `53601b7c524653f5e1966b3955b9f51373e54002`。atan角、event probability、phaseの定義を確認。実行していない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/rte.py
- **[G2]** `src/trotterlib/df_partial_s2.py`、同fixed commit。今回直接確認は1–230行、650–970行。blob `f467ae0a987595c63a02a927e5e9c95dfb7b2146`。完成blockのbasis往復、対角構成、scalar、forward／reverseを確認。全moduleや全native pipelineの監査ではない。
  https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/df_partial_s2.py

### 外部一次文献

- **[W1]** Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647。取得PDFの§II、§IV、付録AにおけるHadamard-testの±1 outcome、unitaryのcoherent平均、atan型の平均恒等式を確認。画像ページ3・6・19（0始まり）も取得した。テキスト抽出と画像で版／式番号の表示差があるため、本書では特定の式番号を根拠にせず、両方にある基本式だけを引用した。今回のsector命題をこの論文がそのまま証明していると主張していない。
  https://arxiv.org/abs/2503.05647
  https://arxiv.org/pdf/2503.05647

---

**レビューは完了。科学数値・新しい性能値・回路費用は生成していない。旧理論レビュー、既存SOURCE、raw結果、local成果、mandatory STOPを変更していない。**
