# 元Track A・C-α／α0-P：対称macrostep誤差とgap依存保証の限定科学レビュー

- **日付**：2026-11-10（日本時間）
- **対象**：C-α／α0-P、H4・sector36・LD3・δ=0.02・r=4・m0/m17の既存Step 1を踏まえた「headを含む誤差上界の改善」
- **レビュー開始承認**：ユーザーの「レビューを開始して」。先行のレビュー開始承認とは区別した今回の限定レビュー。
- **正式共通運用指示**：`research_project_instructions_unified_2026-11-09.md`を実ファイルから確認。既存科学契約・旧run・mandatory STOPを維持。
- **行ったこと**：CodexのStep 1報告と前回二つのGPT理論レビュー、公開SOURCEの構成資料、対称PF・収縮semigroupに関する外部文献の読解、紙上の数学的導出と科学的判断。
- **行っていないこと**：新しい行列・operator・数値の生成、保存数値の再集計・再ランキング、シミュレーション、回路build/compile、sampling、科学用コード実行、repository変更、旧runのretry/resume。新たな性能数値の生成なし。
- **状態**：レビュー完了。**理論的な置換上界と代替のresolvent型保証を提示するが、H4でそれらが有用になること・科学的GOを認定しない**。

## 0. 結論（重要）

**限定レビューの結論は「保証改善の研究を継続する価値がある。ただし次の科学計算へ自動進行しない」である。**

1. Codexの`q(η_forward+η_reverse)`は、**対称配置の左右の誤差を別々に評価することで、2次PFが持つ局所低次項の相殺を失っている**。固定α0-Pの物理的平均macrostepを、複数の決定論的head指数と中心の**減衰を含むtail指数**による**多項Strang分解**として扱えば、headとD/R splitを一括して、非漸近・交換子型の`T δ² C_multi`に置き換えられる。これは紙上で条件付きの厳密な上界として導出できる（§3–4）。
2. **ただし、`C_multi`の数値は未取得**であり、新上界がCodexの現状の評価より小さい保証はない。旧`Tδ²C_split`は**新`Tδ²C_multi`と別に足さない**。今回の改善は*上界を構成する方法*の改善であり、採用だけでm17の保証が成立するものではない。
3. **head項だけを解消しても十分とは限らない**。m17では、報告済みのsplit項とgap非依存のvariance項も、相対位相disk予算に対して無視できない。必要条件は`d(T) ≤ e^{-γT} sin(0.14)`である。headを削減しただけで自動的なGOにはできない（§5）。
4. gap数値診断`0.23263021512018667 Ha`は、**まだ認証済み下界ではない**。gap下界を科学的に認証する道筋と、必ずしも明示的なglobal gap下界を使わず、励起側の非Hermitian resolventを検証する代替道筋を区別する（§6–7）。後者について、既存のVolterra証明から有効な状態依存の紙上上界を導く。
5. 現時点の保持判断：**m0/m17の数値診断は完了、m17の十分条件は未成立／実装込み保証UNKNOWN、C-αの総費用優位未認定**。今回は追加の科学実行を許可しない。次の担当は原則Codexによる差分SOURCE調査と、別途ユーザー承認後の最小限の保証入力取得である（§9）。

## 1. 確認した科学的事実と適用範囲

### 1.1 Step 1の既存報告（引用のみ）

Codexの`貼り付けたMarkdown(4).md`（2026-11-10）に基づく。以下は**保存報告値**であり、本レビューでは再計算・外部再現していない。

| 量 | m0 | m17 |
|---|---:|---:|
| q、T (Ha⁻¹) | 1、0.02 | 131072、2621.44 |
| α0-P raw半径 | 0.999984230759 | **0.126574440422** |
| 元energyに対する数値位相差 (rad) | 1.0510543e-7 | **0.013819560800** |
| 既存K2 raw半径 | 0.999965980312 | 0.011572723492 |
| 減衰参照半径 | 0.999984230717 | 0.126573962610 |
| 既存`q(η_forward+η_reverse)` | 8.20091015e-5 | **10.74909696** |
| 既存`Tδ²C_split` | 3.34678235e-7 | **0.04386694557** |
| 既存arctan補正 | 2.23572240e-8 | 0.002930406067 |
| gap非依存variance項 | 5.14640862e-12 | **0.08841462679** |
| 数値gapを*仮使用*したvariance診断 | 5.14640862e-12 | 0.0005799333245 |

共通の保存値：`γ=0.0007884706336364447`、`vΓ=2.573204308076602e−8`、保存sector gapの**診断値**`0.23263021512018667 Ha`。model差・full辞書/sector接続差・raw Hermitian参照差は報告の別項であり、ここに含めて一括0とはしない。finite-bit/native誤差は**UNKNOWN**。

Codexはfull Fockでの同law接続とsector36への平均制限、完成head blockの保存性を数値診断として確認した。旧run01の個別primitive非保存46件とmandatory STOPは保全。run02のrunner、raw JSON、logは**local未commit**であり、本レビューでは原本を直接取得していない。報告のSOURCE・SHA・数値を独立再計算したとは書かない。

### 1.2 原則として維持する科学条件

H4、1.00 Å、STO-3G、保存H_phys／DF12、generation-prefix LD3、元H保存基底状態、Nα=Nβ=2の**sector36**、α0-P（paired K0）、正時間partial-S2、δ=0.02、r=4、m0/m17、C-α専用systematic上限0.14 rad、stat0.24、reserve0.02、β_RPE0.4。旧P1の各成分枠は不変更。異なるsector70やspin-aware辞書への置換・K2の再計算・branch変更を伴わない。

### 1.3 前回二つのGPTレビューとの接続

- `track_a_calpha_theoretical_review_20261110.md`：α0-Pのtail平均の厳密log、非unitary収縮Strang、元H基底状態とΓのVolterra誤差、disk→位相／半径→shotを接続。
- `track_a_calpha_sector_condition_review_20261110.md`：**個別sampled primitiveのspin保存を、coherent平均信号のsector制限の必要条件とする誤りを訂正**。個々のatomをsector内で指数化しない。平均tailと完成head、参照H・Γがsectorを保存することを要求。

今回の新しい主題は、これらの枠組みを維持したまま、**左右headを個別に理想head指数と比較する粗い項**を改善できるか、gap項を保証付きにできるか、という限定的な科学判断である。

## 2. なぜ旧head項は保守的なのか

### 2.1 既存の比較

決定論的headを`D=Σ_{j=1}^s D_j`とする。既存の半sweepは一般に

\[
F_\delta=\prod_j e^{-i\delta D_j/2},\qquad
R_\delta=\prod_j^{\leftarrow} e^{-i\delta D_j/2}
\]

であり、各半sweepは*単独では*`e^{-iδD/2}`との違いが`O(δ²)`になり得る。Codexの旧上界は、forwardとreverseをそれぞれ近似headへ比較して

\[
\|F_\delta-e^{-i\delta D/2}\|\le\eta_F,
\qquad\|R_\delta-e^{-i\delta D/2}\|\le\eta_R
\]

とし、全反復で`q(η_F+η_R)`を払うものだった。

**しかし、実際のmacrostepは`Fδ M_P(θ)^r Rδ`というパリンドローム構成である。** forwardとreverseを合わせると二次の奇対称誤差が相殺され、対象全体の局所誤差は一般に`O(δ³)`となる。片側ずつnormを取ると、その相殺を捨ててしまう。

### 2.2 相殺が成立するための確認事項

1. headの各**完成block**が、固定されたHermitianな`D_j`の指数`exp(-iδD_j/2)`として意味付けられること。
2. 反対側が**同じblockの逆順**であり、勝手なphase fit、独立の近似generatorへの変更、誤った逆時間発展を含まないこと。
3. 中央が実際のα0-P物理平均`M_P(θ)^r=e^{δB}`であること。`B=-iRθ−Γθ`であって、K2の平均でも、各atomをsector内で指数化したものでもない。
4. 上記の完成block境界で、同じsectorへ制限できること。raw model、finite-bit、generic/CAR再構成差は別の誤差台帳に保持すること。

`df_partial_s2.py`（固定`2e79d80d…`）はbasis change→diagonal primitives→逆basis changeのhead block、forward／reverseを示している。**コードの組合せだけで全実装の完全一致を証明したことにはならない**ため、Codexは既存macrostep SOURCEとの同一性を静的に確認する。

## 3. 対称macrostep全体の定数付き上界（紙上命題）

### 3.1 収縮二項Strang（基礎補題）

有限次元で`X,Y,X+Y`がそれぞれ正時間にcontraction semigroupを生成するとし、

\[
S_{X,Y}(t)=e^{tX/2}e^{tY}e^{tX/2},\qquad
C=[X,Y]
\]

とする。このとき

\[
\boxed{
\|S_{X,Y}(t)-e^{t(X+Y)}\|
\le t^3\left(
\frac{\|[X,[X,Y]]\|}{24}
+\frac{\|[Y,[X,Y]]\|}{12}
\right),\qquad t\ge0.}
\tag{3.1}
\]

**導出（符号と定数）**：`D(t)=S'(t)−(X+Y)S(t)`とすると、

\[
D(t)=[e^{tX/2},Y]e^{tY}e^{tX/2}
+e^{tX/2}[e^{tY},X/2]e^{tX/2}.
\]

ここで`C=[X,Y]`なら、

\[
[e^{tX/2},Y]=
+\int_0^{t/2}e^{(t/2-u)X}C e^{uX}\,du,
\]
\[
[e^{tY},X/2]=
-\frac12\int_0^t e^{(t-u)Y}C e^{uY}\,du.
\]

二項の主要な`(t/2)e^{tX/2}Ce^{tY}e^{tX/2}`が相殺する。残部は収縮性と交換子の積分式により

\[
\|D(t)\|\le t^2\left(
\frac{\|[X,C]\|}{8}+
\frac{\|[Y,C]\|}{4}\right).
\]

変分公式

\[
S(t)-e^{t(X+Y)}
=\int_0^t e^{(t-u)(X+Y)}D(u)\,du
\]

から(3.1)を得る。

**前回理論レビュー付録Bの訂正**：同付録の`[e^{tA/2},B]`を積分表示した行には先頭に**負号**が記載されていたが、`C=[A,B]`の規約では**正号**が正しい。付録の後続の主要項相殺および上界の結論は、この正号を用いることで整合する。本レビューで訂正を明示し、旧原本は変更しない。

先行：Childs et al., *Theory of Trotter Error with Commutator Scaling* (2021) はunitary PFの交換子型上界を扱う。Iserles–Kropielnicka, *An elementary approach to splittings of unbounded operators* (2024) はsemigroupのStrang積分表現・評価を扱う。式(3.1)はこれら既知の解析技法に沿うもので**新しいPF定理としての新規性を主張しない**。

### 3.2 複数headを一体として扱う帰納法

同じ不変sector内で、決定論的完成blockを`A_j=-iD_j`（`D_j=D_j†`）、tailを

\[
B=-iR_\theta-\Gamma_\theta,\quad \Gamma_\theta\succeq0
\]

とする。**中央から外側へ向かう、実際の回路演算順**に`A_1,…,A_s`を番号付けする。`G_0=B`、`G_j=B+Σ_{k=1}^j A_k`とし、

\[
F_0(t)=e^{tB},\qquad
F_j(t)=e^{tA_j/2}F_{j-1}(t)e^{tA_j/2}.
\]

実際の理想CAR macrostepがこの`F_s(δ)`に等しいことを確認する（phase scalarは別途一回計上）。`A_j`はskew-Hermitian、`B`はdissipativeなので、`G_j`の正時間指数はcontractionである。

(3.1)とtelescopingを帰納的に使えば、

\[
\boxed{
\|F_s(\delta)-e^{\delta G_s}\|
\le \delta^3\,C_{\rm multi},}
\tag{3.2}
\]
\[
\boxed{
C_{\rm multi}:=\sum_{j=1}^{s}
\left[
\frac{\|[A_j,[A_j,G_{j-1}]]\|}{24}
+\frac{\|[G_{j-1},[A_j,G_{j-1}]]\|}{12}
\right].}
\tag{3.3}
\]

有限時間`T=qδ`に対して両辺の作用素が収縮なので、

\[
\boxed{
\|F_s(\delta)^q-e^{TG_s}\|
\le q\delta^3 C_{\rm multi}
=T\delta^2 C_{\rm multi}.}
\tag{3.4}
\]

これは**局所3次・全時間2次の明示的な非漸近上界**であり、`q(η_F+η_R)`の2次局所誤差会計による不必要な保守性を避ける。`B`には減衰`Γθ`を含むので、単純なunitary head+Rの上界を使い回していない。

**厳密に区別すべきこと**：

- `C_multi`が小さいことは**証明されていない**。局所の次数が改善しても、交換子和が大きければm17には不十分。
- `C_multi`は**head内部誤差とhead/tail splitを両方含む**。旧`Tδ² C_split`と`q(η_F+η_R)`を**両方とも差し替える**。旧splitを再加算するのは二重計上。
- 定数付きの(3.2)は、理想CARの完成指数blockと正時間の収縮semigroupに関するもの。finite-bit・generic/CAR差・native実装は別に囲む。

### 3.3 さらに鋭い合法な代替：一macrostep defect

実際の理想平均macrostep`F_s(δ)`と、同じ参照generatorの指数`Eδ=e^{δG_s}`について、仮に認証付き

\[
\eta_{\rm macro}\ge\|F_s(\delta)-E_\delta\|
\]

が取得できれば、

\[
\boxed{\|F_s(\delta)^q-E_\delta^q\|
\le q\min\{\delta^3 C_{\rm multi},\eta_{\rm macro}\}.}
\tag{3.5}
\]

を使える。これは実装済みの既存平均macrostepを再利用し得るが、**`η_macro`の新規数値評価・認証は今回未実施であり、別途科学実行承認が必要**。既存のraw signal差や、m0の一状態の位相差を、作用素normの認証に読み替えない。

状態だけを使った補助上界として、`F_s(δ)`と`Eδ`がcontractionなら

\[
|\langle0|(F_s^q-E_\delta^q)|0\rangle|
\le\sum_{k=0}^{q-1}\|(F_s-E_\delta)E_\delta^k|0\rangle\|
\tag{3.6}
\]

も成り立つ。これはstateに適合した評価の候補だが、現時点で収縮性だけから小さくなることは保証されない。**既存のm17 raw信号を理論証明書に格上げする抜け道ではない**。

## 4. 元H基底信号へつなぐ際の新しい誤差台帳

### 4.1 二重計上を避けた構成

同じ数学的head`D=Σ_j D_j`、物理tail`R`、model`H_model=D+R+cI`（既知scalar`c`は一度だけ扱う）とし、Hermitianな参照`H_ref`、raw`H_phys`を区別する。sector保存および同じstateという仮定の下で、理想CAR macrostepの反復から、前回の参照発展`e^{T(-iH_ref−Γθ)}`への作用素差を例えば

\[
\begin{aligned}
\epsilon_{\rm dyn,new}(T)\ \le\ {}&
T\delta^2 C_{\rm multi}
+T\|R_\theta-R\| \\
&+T\|H_{\rm model}-H_{\rm ref}\|
+q\,e_{\rm conn}\\
&+\epsilon_{\rm raw}(T)+\epsilon_{\rm impl}(T)
\end{aligned}
\tag{4.1}
\]

と構成できる。`e_conn`は旧runのgeneric/fullとCAR/sectorの間の**sector外成分も含む**macrostep接続差に対応するが、その数値は現時点で認証された一様上界ではない。従って`q e_conn`は証拠水準を明示する。

`ε_raw(T)`はHermitian参照からraw H_physへの差。rawが非Hermitianなら、収縮Duhamelを無条件に適用せず、**原差のnormと必要な指数係数を確認した有効な上界**を使う（Codexが用いた`expm1`型は一候補）。

**重要：**(4.1)には旧`q(η_F+η_R)`も旧`Tδ² C_split`も含めない。これらを残すと二重計上になり、理論的な改善経路の評価にならない。

### 4.2 実装とsectorの注意

- 理想full辞書の平均がsector36を保存することと、各sampled atomが保存することは別。前回追補の訂正を維持する。
- `C_multi`の各交換子は、対象の**同じ理想CAR completed blocks**の生成子について評価する。数値的に微小なspin leakageを「厳密zero」とはしない。
- Gaussian基底変換、phase scalar、forward/reverseの順序、lawの正規化を変えない。
- full-spaceのgeneric/CAR接続差は別項で一回だけ払う。`J†Q_lJ`の指数をsamplerの代用にしない。

## 5. なぜheadだけではm17の保証にならないか

前回のdisk→位相／半径の十分条件は

\[
d(T)=d_{\Gamma}(T)+\epsilon_{\rm dyn}(T),\quad
a(T)=e^{-\gamma T},\quad
\rho_{\rm lb}=a-d,\quad \beta_{\rm sys}=\arcsin(d/a).
\]

**新C-αのsystematic上限0.14 radを保証する条件**は

\[
\boxed{d(T)\le a(T)\sin(0.14),}
\tag{5.1}
\]

である。上の条件は`d<a`も含意する（`0.14<π/2`）。

Codexの既存m17報告では、旧split項`0.04386694557`、gap非依存variance項`0.08841462679`、leading半径`a=0.126573962505`であった。**この既存値と(5.1)の関係から、headだけを0にしても、旧split上界を据え置く方法では0.14 radのdisk保証に届かない。gap非依存variance上界をそのまま使う方法でも届かない。** この定性的な大小判定は既存報告値の引用に基づき、新しい科学数値を算出したものではない。

よって次の作業は**三つの論点を別々に認識したうえで、保証の一体的な改善**を図る必要がある。

| 論点 | 必要な方向 | STOPの意味 |
|---|---|---|
| headとD/R split | (3.2)–(3.4)の`C_multi`、または証明付きmacrostep defectへ置換 | 改善が小さいとm17の位相diskは不成立のまま |
| 非可換な減衰と固有状態結合 | gap認証または§6のresolvent型の状態依存評価 | gap非依存の`T²`項が大きいままでは十分条件が弱い |
| raw/DF/finite-bit/実装差 | 元H、sector、full辞書、実装精度の資格を別に整理 | 未資格の実装項を0として包括PASSにしない |

**新しい`C_multi`で旧splitも置き換えること**が、本レビューの最重要の設計変更である。その数値取得・適格性は今後の判断に属する。

## 6. spectral gapの利用：保証と数値診断を区別

### 6.1 既存の分散項

元Hのsector内基底状態を`|0⟩`、`P=|0⟩⟨0|`、`Q=I-P`、`Γ=Γθ≥0`、`γ=⟨Γ⟩_0`、`g=QΓ|0⟩`、`b=‖g‖`とする。`b²=Var_0(Γ)`。

有限次元参照Hermitian Hがsector内で非縮退基底を持ち、**認証された**gap下界`Δ_- >0`があるなら、前回定理により

\[
\boxed{
|\langle0|e^{(-iH-\Gamma)T}|0\rangle
-e^{-iE_0T-\gamma T}|
\le b^2\min\{T^2/2,2T/\Delta_-\}.}
\tag{6.1}
\]

ここで`Δ_-`は**同じH、同じsector、同じstate/energy**のgap下界である。保存された数値gap診断`0.23263021512018667 Ha`を、そのまま`Δ_-`として利用しない。

### 6.2 どのようにgapを認証できるか（今回未実行）

対象Hermitian行列の二つの低固有値に対して、数学的に包含される区間

\[
E_0\in[\ell_0,u_0],\quad E_1\in[\ell_1,u_1]
\]

が得られ、両区間の順序と低い二固有値の**漏れのない同定**が認証できれば

\[
\Delta\ge \ell_1-u_0>0
\]

である。証明法の候補は、Hermitian行列の**誤差包含付き固有値区間**、またはshiftしたHermitian行列の**慣性（inertia）をverified LDL*等で証明する方法**である。浮動小数点で対角化しただけ、あるいは残差が小さいだけでは、低い二固有値を漏れなく挟んだとは限らない。

同じH_phys/H_ref、sector36、保存stateに対する理論promise・有限精度表現と、認証対象の「正確な行列」を明記する。新ground solveは自動的に要求しない。これらの数値認証は**別途科学実行承認を要する**。

### 6.3 global gapだけに依存しない代替（紙上のresolvent命題）

前回のVolterra証明で、Q部分空間上

\[
B_Q=Q(H-E_0)Q-i(Q\Gamma Q-\gamma Q)
\]

を定義する。`B_Q`が可逆なら、`w=B_Q^{-1}g`を考える。γ≥0、Γ≥0の収縮性から、前回と同じVolterra式により

\[
\boxed{
|\langle0|e^{(-iH-\Gamma)T}|0\rangle
-e^{-iE_0T-\gamma T}|
\le
b\|w\|\left[T+\frac{1-e^{-\gamma T}}{\gamma}\right].}
\tag{6.2}
\]

γ=0ならΓ≥0からΓ|0⟩=0、`g=0`となり左辺は0と扱う。`γ>0`における(6.2)は**全励起空間gapの数値を明示的に使わない、状態依存の上界**である。ただし、**`B_Q`可逆と`‖B_Q^{-1}g‖`の認証が必要**であり、gap仮定を無条件に不要にしたものではない。

**導出**：前回レビューの`a(t), y(t)`について

\[
a(t)-1=\int_0^t g^\dagger F(t-s)g\,a(s)\,ds,
\quad F(u)=\int_0^u e^{-iB_Qv}\,dv
=i(e^{-iB_Qu}-I)B_Q^{-1}.
\]

`‖e^{-iB_Qu}‖≤e^{γu}`、`|a(s)|≤e^{γs}`、`‖g‖=b`を用いると

\[
|g^\dagger F(u)g|\le b\|B_Q^{-1}g\|(1+e^{\gamma u}).
\]

積分後に全体の`e^{-γt}`を掛けて(6.2)を得る。global gapが使える場合、`‖B_Q^{-1}g‖≤b/Δ`を代入すると、前回の`2b²T/Δ`上界と整合する。

従って、実際に認証された入力が利用可能な場合は

\[
\boxed{d_\Gamma(T)\le\min\left\{\frac{b^2T^2}{2},\;b\|B_Q^{-1}g\|\left[T+\frac{1-e^{-\gamma T}}{\gamma}\right]\right\}}
\tag{6.3}
\]

を適用し得る。global gapの厳密下界を用いる(6.1)も、別の合法な候補である。

**注意：**(6.2)の`B_Q`にはH、Γ、γ、E0、射影Qが含まれる。ユーザーにとって未知のE0を無料の設計oracleに使うものではない。今回は数理的な**証明／診断**の候補であり、予め計算資源を確保するための情報アクセスや回路の一部として無償に扱えない。保存stateを使ったdiagnosticと、将来のスケールする保証法の区別が必要である。新しいresolvent計算も今回未実施。

## 7. 定理・観測・改善見通しを取り違えない

| 主張 | 今回の科学的状態 |
|---|---|
| 旧`q(η_F+η_R)`は対称性の相殺を無視する | **紙上・構造として確認**。改善可能な上界の選択が存在 |
| 多項Strangの(3.2)–(3.4) | **条件付き厳密な紙上命題**。SOURCEがそのfactorizationであることと、C_multiの認証取得が必要 |
| 旧付録Bの積分式の先頭符号 | **紙上で誤記を特定・訂正**。正号なら主項が相殺 |
| 新しい上界はm17で有用になる | **UNKNOWN**。数値未取得。次数だけで利得を主張できない |
| 保存gap診断を厳密gap下界とする | **不可**。そのまま認証へ昇格させない |
| resolvent型の(6.2) | **条件付き紙上命題**。invertibilityとノルムの認証が必要 |
| 既存K0の数値m17位相は0.14rad以内 | **Codexの保存診断**。保証ではない |
| 新しいα0-P法がK0既知法より物理的に安い | **未成立**。same-law・same-settingなら同じ物理回路 |
| RPE全18roundの成功／総費用勝利 | **未検証**。追加の承認が必要 |

### 独立性と先行研究

今回の主張は、**既知PFの多項Strang誤差評価を、固定α0-Pの正しい物理平均・sector解析・非Hermitian減衰の中へ安全に適用する方法**である。Strang／積公式／交換子評価や状態固有のresolvent評価の一般原理に対して新規性を主張しない。Track CのPF設計そのものを元Track Aへ移すのではなく、C-αの**保証資格**のため既存のPF解析部品を再利用する位置付けとする。

## 8. 研究上の判断

### 8.1 選択肢の比較

- **推奨**：現行α0-Pと辞書・sector36・δ/rを維持し、**多項Strangの`C_multi`または認証付きmacrostep defectの技術的取得可能性を確認**する。同時に、gap下界認証かresolvent型のどちらが十分条件を満たせるかを検討する。新しいPF familyやsampling lawは作らない。
- **非推奨**：head項だけを削除して旧splitを据え置く／数値gapを厳密gapに昇格する／m17の観測位相値だけでRPE成功を宣言する／旧runに上書きする。
- **後順位**：新しい状態依存phase/spectral保証（macrostepの非unitary固有分枝追跡、実効generatorによる完全な誤差構造化）。比較的複雑であり、まず多項Strangとresolventを評価した方が情報価値が高い。

### 8.2 継続・保留の判定

現時点で**保証の構成可能性はあるが、科学的適格性は未確認**。結果分類は旧Step 1の「保証改善が必要」を維持。今回のレビューはC-αの研究継続を正式採択するものでも、旧mandatory STOP解除でもない。

## 9. 次の担当・作業・別途承認

### 9.1 Codexが次に担当すること（科学実行前の技術作業）

1. 既存C-α専用branch/worktreeとSOURCE、旧run01/02・raw・STOPの状態を確認し、全て保全する。
2. `df_partial_s2.py`、`compose_step`、既存CAR headの**完成blockと正確なcentral-to-outer stage order**を静的に確認する。既存の交換子生成・norm評価・BCH/PF helperを探索し、`C_multi`または`η_macro`を取得する際の重複実装を避ける。
3. 既存gap生成記録とsector36のHermitian参照に対して、**どの数値資格／包含法が必要かを技術的に計画**する。resolventについては`B_Q`のinvertibilityと`B_Q^{-1}g`の認証を取得する際に必要な入力・コードを整理する。
4. `C_multi`とvarianceの両方を改善する前提の**最小科学実行plan**を作成する。旧rawにない数値・operatorを勝手に算出しない。新版の誤差台帳が旧split+旧headを二重加算しないこと、既存fixed H/state/sectorの意味を変えないことを確認する。

### 9.2 別途明示的な科学実行承認が必要な範囲

- `C_multi`に現れる交換子ノルム・多項Strang誤差の新規評価。
- 同条件での`η_macro`の新規上界評価と、有効な数値包含認証。
- Hermitian参照のsector36 gap下界の数値資格化、またはresolvent型認証。
- 新しい十分条件でのm0/m17位相・半径disk再計算、数値誤差の包含。

これらを一度に実施する場合も、**旧科学SOURCE・raw・STOPを変更せず、新scopeとして対象と停止点を指定してユーザーが実行許可する**。新しい承認証明書やgrantを必須にしない。

### 9.3 今回のmandatory STOP

今は科学実行をしていない。旧C-αのStep 1 run01/run02とそれぞれのSTOPを保全し、**上記技術照合の後、別途実行承認が必要**。良好な新上界が得られても、全18round・新r/LD・sampling・native価格・総費用へ自動拡張しない。

## 10. 資料台帳・参照元

### 10.1 実際に確認した私的・プロジェクト原本

- `research_project_instructions_unified_2026-11-09.md`（`/mnt/data`。正式版全文の該当節を確認）
- `track_a_calpha_theoretical_review_20261110.md`（`/mnt/data`。§1.3、§4.1–4.4、§5.1–5.2、§7、付録Bを確認）
- `track_a_calpha_sector_condition_review_20261110.md`（`/mnt/data`。§0、§3–6、§8、§11–12を確認）
- `貼り付けたMarkdown(4).md`（`/mnt/data`。Codex run02の数値報告全文、保全・STOP・取得状態を確認）
- `貼り付けたMarkdown(3).md`（`/mnt/data`。前段Phase 0、SOURCE・入力、先行結果との対応）
- `貼り付けたテキスト（1）(8).txt`（`/mnt/data`。事前技術設計、予算・K0とK2の区別）

`applicability.json`、`result.json`、新runner、local未commit SOURCEとprivate H/state/DF配列は、**報告内のpathは確認したが、当環境で原本を取得・検証していない**。報告値は資料記載に基づく。

### 10.2 固定Repository資料

- [原RTE・α0-P law]（`2e79d80dc9cce2885580e50fe6f1705da8fa41af`）：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/rte.py
- [head完成block・forward/reverse]（同commit）：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/df_partial_s2.py
- [保存P1 shot scope]（`0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`）：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md

固定SOURCEは公開アクセスの関連断片と先行の取得情報を静的に確認した。今回のlocal run02とは別成果であり、最新local SOURCEそのものを取得したとは扱わない。

### 10.3 外部一次文献

- Childs, Su, Tran, Wiebe, Zhu, **Theory of Trotter Error with Commutator Scaling**, *Physical Review X* **11**, 011020 (2021), DOI:10.1103/PhysRevX.11.011020. https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020 。PFの交換子依存の厳密誤差理論として確認。
- Iserles, Kropielnicka, **An elementary approach to splittings of unbounded operators**, *Computers & Mathematics with Applications* **176**, 29–34 (2024), DOI:10.1016/j.camwa.2024.08.031. https://arxiv.org/abs/2401.06635 。semigroupのStrang誤差・交換子に関する公開abstractおよび出版社ページの本文部分を確認。**論文PDFの全ページを読了済みとは扱わない**。
- Sylvesterの慣性法則・verified Hermitian eigenvalue countは一般数学の基盤。関連する現代の概説・説明（数値精度による認証上の問題を含む）：https://www.numerical-linear-algebra.com/essays/an-eigenvalue-count-that-cannot-be-slightly-wrong/ 。外部概説は具体的なH4のgapを認証したものではない。

## 11. 変更・結論の最終チェック

- α0-Pと既知qDRIFT型PRは**同じlaw**。今回新しいサンプリングアルゴリズムを発明したとは主張しない。
- 「headを含めた収縮多項Strang」「状態特異的resolvent」は**既知の数学的道具を対象PRへ適用した条件付き保証**であり、先行文献上の優先権を認定しない。
- 数値m17の観測位相`0.013819560800 rad`は、許容`0.14 rad`以下という**Codex診断**。証明付きの全時間phase保証ではない。
- **今回のm17保証はまだ成立していない**。head項を置き換える数学的方法と、gapの代替資格方法を示した段階。
- 新しい科学数値・実装・ショット／価格・全18roundの評価は行っていない。
- 固定結果・SOURCE・marker・STOP、元Track AとTrack B/C／新A／Track Dの分離は維持。

**最終結論：C-αの限定した理論保証改善は可能性があり、次に評価すべき対象を二点に明確化できた。`C_multi`とgap/resolvent側の認証入力が有用になることが確認されるまでは、科学的GOや費用優位を宣言せず、追加科学計算は別途ユーザー承認とする。**
