# 元Track A・Claude案 C-α：理論的成立性レビュー

- 作成日：2026-11-10（日本時間）
- 対象：Claude案 C-α。「元Hの基底状態入力・一次ランダムtail・D/R合成・有限時間のcoherent信号・RPE保証」の接続。
- 承認：本チャットの「レビューを開始して」に基づく詳細科学レビュー。
- 状態：**レビュー完了。条件付きの数理的十分条件を導出。研究採択・実分子での有用性・独立新規性は未認定。**
- 実施内容：添付Claude原文とCodex静的調査の読解、固定GitHub資料の限定的な再確認、外部一次資料の確認、紙上導出、本文作成。
- 非実施：科学moduleの実行、数値行列の生成、対角化、simulation、保存値の再集計、テスト、compile、費用計算、repository変更。以下の解析例も記号的な紙上例である。
- 正式運用指示：`research_project_instructions_unified_2026-11-09.md` の該当規則を原文で確認。
- GPT案のD/R最適化とは別の検討線。他Trackの成果・承認・STOPを移管しない。

## 0. 結論

**C-αを原文のまま採択することは推奨しない。一方、誤りを修正して「元Hの基底状態に対する、一次ランダムtailの有限時間信号保証」という限定した理論課題を継続する根拠はある。**

今回の主な到達点は、Claudeの非Hermitian固有値摂動の説明を追認することではない。**元Hの状態を別の実効Hamiltonianの固有状態へ置換せず、実際に測定する複素信号の位相と半径を同時に抑える、定数付きの十分条件を構成した。**

その中心は、任意の有限次元Hermitian HとPSDな減衰作用素 Γについて、非縮退基底状態 |0⟩、同じ不変部分空間内のgap Δ>0を用いる次の評価である。

\[
\gamma=\langle0|\Gamma|0\rangle,\qquad
v_\Gamma=\langle0|\Gamma^2|0\rangle-\gamma^2.
\]

\[
\boxed{
\left|\langle0|e^{(-iH-\Gamma)T}|0\rangle
-e^{-iE_0T-\gamma T}\right|
\le v_\Gamma\min\left\{\frac{T^2}{2},\frac{2T}{\Delta}\right\}.}
\tag{S}
\]

この式は本レビュー§5で証明する。gapを使わない T²/2 側は、非縮退性を要求せず、指定した固有ベクトルに対して使える。gapを使う側は**基底状態に対する**評価で、任意の内部励起状態にはそのまま移さない。

実際のPR平均とこの減衰発展とのoperator距離を ε_dyn(T) で抑えれば、

\[
 d(T)=v_\Gamma\min\{T^2/2,2T/\Delta\}+\epsilon_{\rm dyn}(T),
 \quad \eta(T)=e^{\gamma T}d(T)<1
\]

の下で、

\[
\boxed{|z(T)|\ge e^{-\gamma T}-d(T)>0,\qquad
|\arg z(T)+E_0T|_{2\pi}\le\arcsin\eta(T).}
\tag{R}
\]

これを既存のbounded-outcome shot式とbranch復号へ接続できる。

**重要な限定**：式(S),(R)は、C-αが安価になることの証明ではない。必要なgap・moment・operator上界を実際に取得できるか、十分条件が実際の条件で鋭いか、強い対照より費用が小さいかは未評価である。また、本レビュー内で導出した式であることと、世界初の定理であることは別であり、後者は主張しない。

### 推奨する扱い

| 対象 | 推奨 |
|---|---|
| α0の定義 | paired arctan型を **α0-P**、通常の指数回転を **α0-E** と明示的に分ける |
| 最初に仕様化する候補 | 既存SOURCEを使えるα0-P。これは検討順の推奨であって正式採択ではない |
| α0-E | 異なるlawとして保持。同じ二次減衰でも高次位相が異なる |
| α1／α2 | 後述の一般指数lawの十分条件を利用可能。ただし情報取得・希少event・費用は未評価 |
| 相関sampling（R6） | IIDの定理へ含めない。今回の最初の検証候補から外す |
| 新規性 | qDRIFT型PR、係数2、標準偏差samplingを新規性にしない。有限時間信号保証の追加性が候補 |
| 実験 | 今回は未実行・未承認。直ちに全18roundやH6へ進まない |
| 既存I／II／III、他Track | 既存の保留・固定結果・mandatory STOPを維持 |

---

## 1. 資料・研究範囲・比較条件

### 1.1 根拠の区分

本文では、**資料記載**、**直接確認した一次資料**、**本レビューの紙上導出**、**研究上の推奨**を区別する。Codexが読んだ文献を、このレビューでも全文読了したとは扱わない。未知の性能を既存の数値から補わない。

中心資料は以下である。

- [C1] `元TrackA_独立研究方向再探索_Claude科学レビュー_20261110.md`：特に§3.3、§5.2–5.9。候補と未完成の証明骨格。
- [C2] `貼り付けたテキスト（1）(7).txt`：Codexによる先行研究監査・静的照合、技術分類B。
- [C3] `track_a_new_algorithm_hypotheses_20261110(1).md`：旧I–IIIの定義。旧IIの第一候補順位は後続判断で撤回済み。
- [G1] `2e79d80…/src/trotterlib/rte.py`：今回1520–1665行、1720–1990行を直接取得。`_make_event`、物理平均／補正平均を確認。実行していない。
- [G2] `0d5ca357…/docs/research/track_a_b2_p1_shots_20261107/REPORT.md`：P1のsector、元H状態、既存shot式と資格境界を直接確認。
- 外部資料[P1]–[P9]：確認範囲・取得限界は§12。

### 1.2 このレビューで固定するものと、固定しないもの

数式を検討するために、**有限次元の不変部分空間、元Hの供給された基底状態、正時間のpartial-S2型配置、各round内で同一分布を使う独立sampling**を明示して扱う。これらはこの十分条件の適用条件であり、新しい分子実験の契約を勝手に追加・採択したものではない。round間でlawや刻みを変える場合は各round別に評価する。trajectory内でstateを測って分布を適応更新する方式は、今回のIID証明には含めない。

独立性については「他Trackがすべて同じtargetを保っていたから、targetを変えればTrack A」という説明を採らない。Codexが指摘したとおり、Track Bにも異なる平均・PSD収縮を比較する研究がある。[C2]

本チャットで残す問いは、**既存の一次tail lawが、非可換D/R合成を経たときに、元Hの基底energyをどの誤差・半径で読み出せるか**である。新しいreturn law、PF族、compiler、reference interferenceを独立開発する方向へ拡張しない。元Track Aの中心範囲との正式な位置付けは、この数学が成立しただけでは確定しない。

### 1.3 sector36／sector70の修正

[G2]のP1は、H4・8 spin orbitals、Nα=Nβ=2の**sector36**、元Hの保存基底状態を使う。全粒子数N=4だけを固定したsector70とは別である。[C1]の§2.4と§5.8はこの点で不一致だった。

P1に接続する候補検証ではsector36を基準にする。sector70を使う研究へ移る場合は別scopeであり、状態・gap・比較SOURCEを置換したことを明示する。**各sampled primitiveがNα,Nβを保存することを確認せず、sector36のgapを使ってはならない。**

以下のΔは、単なる一体orbital gapでも、別sectorのgapでも、PFの固有位相間隔でもなく、明記したHと不変部分空間のgapである。分子の実gapや保存stateの厳密性を今回再計算・認証していない。

---

## 2. α0の定義を二つに分ける

H_R=R=Σ_l h_l P_l、P_l=P_l†、P_l²=I、λ=Σ_l|h_l|>0とする。signはP_lへ吸収し、p_l=|h_l|/λ、θ>0、τ=λθとする。λ=0ならtailは不要。

### 2.1 α0-P：既存paired K=0（arctan型）

\[
U_l^{P}=e^{-i\arctan(\tau)P_l},\qquad
\boxed{M_P(\theta)=\mathbb E U_l^P
=\frac{I-i\theta R}{\sqrt{1+\theta^2\lambda^2}}.}
\tag{2.1}
\]

これは[G1]の`_make_event`のatan角と`event_unitary`の符号、補正平均とnormalizationの分離と一致する。RZ等のSDK gate angleとの因子2は別のlowering規約であり、数学的φをそのままRZ引数とみなさない。

**補正平均はI−iθR、物理的平均はM_Pである。** K=0でもnormalization式は存在するが、C-αの生信号に必要なのはM_Pであり、B⁻¹だけで半径を決められない。

原PR論文[P1]には、このarctan型の固有状態信号が既にある。UWC[P2]も同じ角度を用い、付録BにqDRIFTとRTEの費用係数の違いを記載する。「これをPRに入れた」「係数2が現れた」だけを新しい方法としない。

### 2.2 α0-E：通常のPauli指数回転

\[
U_l^{E}=e^{-i\tau P_l},\qquad
\boxed{M_E(\theta)=\cos\tau\,I-i\frac{\sin\tau}{\lambda}R.}
\tag{2.2}
\]

M_PとM_Eは一次・二次の展開が一致する。しかし**同じlawではない**。[C1]§5.2の一般指数lawを採るなら、このα0-Eであって既存paired K=0ではない。

### 2.3 共通の分散、異なる高次位相

\[
V=\lambda^2I-R^2\succeq0.
\]

小θで、

\[
\log M_P=-i\theta R-\frac{\theta^2}{2}V
+i\frac{\theta^3}{3}R^3+O(\theta^4),
\tag{2.3}
\]
\[
\log M_E=-i\theta R-\frac{\theta^2}{2}V
+i\frac{\theta^3}{3}(R^3-\lambda^2R)+O(\theta^4).
\tag{2.4}
\]

従って実効Hermitian補正は、それぞれ

\[
E_P=-\theta^2R^3/3+O(\theta^4),\qquad
E_E=\theta^2(\lambda^2R-R^3)/3+O(\theta^4).
\]

二次減衰が同じだから、位相保証や最適なstepが同じという推論は成立しない。

### 2.4 「可換なら偏りゼロ」ではない

D=0、Rの固有値Eの状態では、α0-PのN stepの位相は厳密に

\[
\phi_P=-N\arctan(\theta E)
\]

であり、目標−NθEとの差がある。[P1]の既知信号式の帰結である。

さらにR=λP、対象がPの固有状態ならΓ=0だが、このarctan位相誤差は残る。α0-Eでは同じ一項の指数が厳密に実装される。これは記号的な境界例であり、新しい数値simulationではない。

**[H,Γ]=0で消えるのは、純粋な−iΓによる非可換な二次固有値補正であり、lawに固有のHermitian補正ではない。** 完全乱択で可能なarctanの逆写像を、一般の非可換D/Rへ無変更で流用もしない。

---

## 3. L1を定数付きで書き直す

### 3.1 一般指数law

Hermitianな有限support確率変数Xについて、μ=E X、S₂=E X²、S₃=E X³、V=S₂−μ²と置く。

\[
V=\mathbb E[(X-\mu)^2]\succeq0.
\]

非可換でもE[Xμ]=μ²、E[μX]=μ²なので、この恒等式は正しい。

\[
M(\theta)=\mathbb E e^{-i\theta X}
=e^{-i\theta\mu}(I-\theta^2 V/2)+R_3.
\]

\[
\boxed{\|R_3\|\le\theta^3 C_3^{\rm abs},\qquad
C_3^{\rm abs}=\frac{\mathbb E\|X\|^3+\|\mu\|^3}{6}
+\frac{\|\mu\|\,\|V\|}{2}.}
\tag{3.1}
\]

**証明。** Hermitian exponentialの二次Taylor剰余は、積分形とunitarityから||e^{-iθX}−(I−iθX−θ²X²/2)||≤θ³||X||³/6。μにも同じ式を用い、残る−θ²(e^{-iθμ}−I)V/2を||e^{-iθμ}−I||≤θ||μ||で抑える。これは粗いが明示的な十分上界である。

### 3.2 中心三次momentだけでは不十分

[C1]L1のCθ³E||X−μ||³という表現には、非可換交差項への一様な依存が明示されていない。

紙上例としてX=μ±εAを等確率、A²=Iとする。μとAが非可換なら、上のfactorizationの三次差に

\[
\frac{i\theta^3\epsilon^2}{6}(A\mu A-\mu)
\]

がある。例えばAとμが反交換すれば−iθ³ε²μ/3であり、E||X−μ||³=ε³||A||³とはεの次数が違う。従って、ε→0でも一様な定数Cで中心三次momentだけを使うことはできない。

式(3.1)はその問題を避ける。より鋭い中心moment＋交換子上界へ改善する余地はあるが、本レビューの保証にその未証明の改善を仮定しない。

### 3.3 三次のHermitian補正

logの0近傍の解析的branchでは、

\[
\log M=-i\theta\mu-\theta^2V/2+i\theta^3 C_{\rm H}+O(\theta^4),
\]
\[
\boxed{C_{\rm H}=S_3/6-(\mu S_2+S_2\mu)/4+\mu^3/3=C_{\rm H}^{\dagger}.}
\tag{3.2}
\]

これはΘ(θ²)の実効Hamiltonian補正を生む。C-αの「位相偏りは二次」の説明には、Γの二次摂動だけでなく、この補正も必要である。

q_Fやc_Fを変えてE Xを保っても、S₃やC_Hは変わる。従って[C1]O7は、**一次の正しさは保持するが、有限θの位相偏りまで不変ではない**と修正する。

### 3.4 PSD減衰を持つ参照発展との距離

一般の非normalなMについて、−θ⁻¹log MのHermitian部分が自動的にPSDだとは仮定しない。その代わり、明示した参照generator

\[
B_\theta=-i\mu-\Gamma_\theta,\qquad \Gamma_\theta=\theta V/2\succeq0
\]

を使う。Duhamel公式と収縮性から、

\[
\boxed{\|M-e^{\theta B_\theta}\|
\le \eta_{\rm micro}:=
\theta^3\left(C_3^{\rm abs}+\frac{\|[\mu,V]\|}{4}\right)
+\frac{\theta^4\|V\|^2}{8}.}
\tag{3.3}
\]

導出は付録Aに示す。Mもe^{θBθ}もcontractionなので、r個の独立microstepでの差はrη_micro以下である。logのbranchを使わず、全時間へのtelescopingができる。

fragment lawではX_F=(A_F−c_FI)/p_Fであり、

\[
\mathbb E\|X\|^3=\sum_F\frac{\|A_F-c_FI\|^3}{p_F^2}.
\]

小さい確率は高次剰余を大きくし得る。小さい状態分散だけを見て希少eventを無条件に許さない。

---

## 4. L2：D/R合成をlogの曖昧さなしに扱う

### 4.1 α0-Pには厳密な減衰generatorがある

式(2.1)はRのnormalな関数であり、全θ>0について

\[
\boxed{M_P(\theta)=\exp[-i\theta R_P(\theta)-\theta\Gamma_P(\theta)]}
\tag{4.1}
\]

\[
R_P(\theta)=\theta^{-1}\arctan(\theta R),
\]
\[
\boxed{\Gamma_P(\theta)=\frac1{2\theta}
\left[\log(1+\theta^2\lambda^2)I-\log(I+\theta^2R^2)\right]\succeq0.}
\tag{4.2}
\]

R_PとΓ_Pは可換。M_Pの固有値の実部は正であり、このtail単独のlogのbranchは明確である。D/R合成全体のlogを取る必要はない。

\[
\|R_P-R\|\le \theta^2\|R\|^3/3.
\tag{4.3}
\]

また作用素順序で

\[
\frac{\theta V}{2(1+\theta^2\lambda^2)}\preceq\Gamma_P\preceq\frac\theta2V,
\qquad
0\preceq\frac\theta2V-\Gamma_P
\preceq\frac{\theta^3}{4}(\lambda^4I-R^4).
\tag{4.4}
\]

証明は、各固有値においてlog(1+a)−log(1+b)=∫_b^a(1+x)⁻¹dx、a≥b≥0を用いる。arctanの差は∫_0^x t²/(1+t²)dtで評価する。

### 4.2 α0-Eにも、限定した角度branchで厳密表現がある

0<τ=λθ<π/2なら、

\[
R_E(\theta)=\theta^{-1}\arctan\left(\frac{\tan\tau}{\lambda}R\right),
\]
\[
\Gamma_E(\theta)=-\frac1{2\theta}
\log\left(\cos^2\tau\,I+\frac{\sin^2\tau}{\lambda^2}R^2\right)\succeq0,
\tag{4.5}
\]

でM_E=exp(−iθR_E−θΓ_E)と書ける。十分上界の一つは

\[
\|R_E-R\|\le\frac{\tan\tau-\tau}{\theta}
\le\frac{\theta^2\lambda^3}{3\cos^2\tau}.
\tag{4.6}
\]

各固有値をr=αλ、|α|≤1と書き、d/dτ[atan(αtanτ)−ατ]=α(1−α²)tan²τ/(1+α²tan²τ)を積分すれば得られる。この角度条件は十分条件の適用範囲であり、既存科学契約を勝手に変更する条件ではない。

### 4.3 収縮semigroupのStrang上界

A=−iD、B=−iR_\star−Γ_\starとする（\star=PまたはE）。D、R_\starはHermitian、Γ_\star≥0なので、e^{tA}、e^{tB}、e^{t(A+B)}はt≥0でcontractionである。

\[
\mathcal S_\delta=e^{\delta A/2}e^{\delta B}e^{\delta A/2},
\quad
C_{\rm split}:=\frac{\|[A,[A,B]]\|}{24}
+\frac{\|[B,[A,B]]\|}{12}.
\]

\[
\boxed{\|\mathcal S_\delta-e^{\delta(A+B)}\|
\le\delta^3C_{\rm split}.}
\tag{4.7}
\]

これは既知のStrang型誤差評価の収縮作用素版であり、本レビューでは付録Bに直接のdefect証明を示す。指数的なe^{T||H||}を必要としない。

θ=δ/r、r正整数ならM_\star^r=e^{δB}であり、n=T/δ正整数の独立macrostepについて

\[
\|\mathcal S_\delta^n-e^{T(A+B)}\|
\le T\delta^2C_{\rm split}.
\tag{4.8}
\]

**Γを含む交換子もC_splitへ含める。** 無雑音PFの既存bias値を、そのままこの式のC_splitとみなさない。

### 4.4 元H・現実のhead実装への接続

数学的なD+RをH_model、入力状態の元HamiltonianをH_physとする。η_model=||H_model−H_phys||を別に残す。記号的にはH_phys=Hとしてよいが、実repositoryで同一と勝手に仮定しない。

headの左右の実装が理想e^{-iδD/2}からそれぞれη_L、η_Rだけずれ、finite-bit等の全時間operator誤差をε_impl(T)で抑えられるなら、α0-\starに対して

\[
\boxed{
\epsilon_{\rm dyn}^{\star}(T)=
T\delta^2C_{\rm split}
+T\|R_\star-R\|
+n(\eta_L+\eta_R)
+T\eta_{\rm model}+\epsilon_{\rm impl}(T)
}
\tag{4.9}
\]

は、実際の物理平均とe^{T(-iH_phys−Γ_\star)}の距離の十分上界になる。Duhamelとtelescopingで、比較する各semigroup／平均が収縮であることを用いる。

この式はunitary回路またはその確率平均のoperator距離を扱い、任意の実機雑音channelを自動的に保証するものではない。既存headが単一の厳密指数でないことを無視していない。既に別の鋭いPF保証があれば置換可能だが、その適用条件を明示する。

### 4.5 α1／α2への接続

一般のfragment指数lawでは、μ=R−cI、c=Σc_FをD側の既知scalarへ移してH_model=D+cI+μと書く。Γ=θV/2を使う§3.4の参照発展に対して

\[
\boxed{
\epsilon_{\rm dyn}^{\rm exp}(T)=
\frac{T}{\theta}\eta_{\rm micro}
+T\delta^2C_{\rm split}
+n(\eta_L+\eta_R)
+T\eta_{\rm model}+\epsilon_{\rm impl}(T)
}
\tag{4.10}
\]

を得る。ここでA=−i(D+cI)、B=−iμ−θV/2。小θでmicrostep誤差の累積はO(Tθ²)だが、その定数には希少eventの高次momentも入る。

この式は、一般のMのlogからPSD Γを断定せずにα1／α2を扱う方法である。**上界が鋭いことは保証していない。**

---

## 5. 元Hの基底状態から直接、有限時間信号を抑える

### 5.1 命題の仮定

有限次元の同じ不変部分空間でH=H†、H|0⟩=E₀|0⟩、P=|0⟩⟨0|、Q=I−Pとする。Γ=Γ†≥0、γ=⟨Γ⟩₀、g=QΓ|0⟩、b²=||g||²=Var₀(Γ)。

まず|0⟩が指定した固有ベクトルであるだけの場合、

\[
|z_\Gamma(T)-e^{-iE_0T-\gamma T}|\le b^2T^2/2.
\tag{5.1}
\]

さらに非縮退基底状態でQ(H−E₀)Q≥ΔQ、Δ>0なら、

\[
\boxed{|z_\Gamma(T)-e^{-iE_0T-\gamma T}|
\le b^2\min\{T^2/2,2T/\Delta\},}
\quad z_\Gamma(T)=\langle0|e^{(-iH-\Gamma)T}|0\rangle.
\tag{5.2}
\]

**||Γ||≤Δ/4は式(5.2)の必要仮定ではない。** ただし、これが有用な小さいrelative誤差になるかは別である。

### 5.2 証明

E₀の既知・未知とは無関係に、証明中の座標変換として

\[
a(t)=e^{(iE_0+\gamma)t}P e^{(-iH-\Gamma)t}|0\rangle
\]

のP成分をscalarで表し、対応するQ成分をy(t)とする。B=Q(H−E₀)Q−i(QΓQ−γQ)と置くと

\[
a'(t)=-g^\dagger y(t),\qquad
y'(t)=-iBy(t)-g a(t),\qquad a(0)=1,\ y(0)=0.
\]

従って

\[
a(t)-1=\int_0^t g^\dagger F(t-s)g\,a(s)\,ds,
\qquad F(u)=\int_0^u e^{-iBv}dv.
\tag{5.3}
\]

Γ≥0よりe^{(-iH−Γ)t}はcontractionなので|a(s)|≤e^{γs}。QΓQ≥0より||e^{-iBu}||≤e^{γu}。

したがってFの積分上界を使うだけで

\[
|a(t)-1|\le b^2 e^{\gamma t}t^2/2.
\]

e^{-γt}を掛けて(5.1)を得る。

基底gapがある場合、任意のQ空間のvについて

\[
\operatorname{Re}\langle v,Bv\rangle
=\langle v,(H-E_0)v\rangle\ge\Delta\|v\|^2
\]

であり、||B^{-1}||≤1/Δ。またF(u)=iB^{-1}(e^{-iBu}−I)だから

\[
\|F(u)\|\le (1+e^{\gamma u})/\Delta.
\]

(5.3)へ代入し、γ>0では

\[
|a(t)-1|\le\frac{b^2}{\Delta}
\left[te^{\gamma t}+\frac{e^{\gamma t}-1}{\gamma}\right].
\]

よって

\[
|z_\Gamma(t)-e^{-iE_0t-\gamma t}|
\le\frac{b^2}{\Delta}
\left[t+\frac{1-e^{-\gamma t}}{\gamma}\right]
\le\frac{2b^2t}{\Delta}.
\]

γ=0ではPSD性からΓ|0⟩=0、従ってb=0であり、信号は厳密にe^{-iE₀t}。極限としても整合する。□

この証明は、K=H−iΓの対角化可能性、全固有分枝の条件数、PF固有状態への置換を要求しない。元H状態からの複素信号そのものを評価している。

### 5.3 Claudeの主要次数をどこまで支持するか

Γ=θV/2なら

\[
\gamma=\theta\langle V\rangle_0/2,\qquad
b^2=\theta^2\operatorname{Var}_0(V)/4.
\]

従ってgapを使うabsolute誤差は

\[
|z_\Gamma(T)-e^{-iE_0T-T\theta\langle V\rangle_0/2}|
\le\frac{T\theta^2}{2\Delta}\operatorname{Var}_0(V).
\tag{5.4}
\]

**減衰の主要係数は⟨V⟩₀、純減衰による位相・半径のずれを抑える量はVar₀(V)でもある。** 小さい⟨V⟩₀だけでは保証を閉じられない。

c=Tθ⟨V⟩₀/2を一定にしたとき、relative誤差の上界に

\[
\frac{2e^c c^2}{T\Delta}
\frac{\operatorname{Var}_0(V)}{\langle V\rangle_0^2}
\tag{5.5}
\]

が現れる。単純なc²/(TΔ)だけではなく、状態momentの比も必要である。⟨V⟩₀=0の場合は別途b=0を使い、0除算しない。

さらに、実lawには式(4.9),(4.10)の誤差が残る。Tが大きいほどtailのθ²位相項が減るようθを選んでも、δを固定したPF誤差Tδ²まで自動的に減るわけではない。

### 5.4 「小さい平均減衰⇒小さいoperator摂動」の反例

記号的に

\[
\Gamma=\begin{pmatrix}\varepsilon&\sqrt{\varepsilon M}\\
\sqrt{\varepsilon M}&M\end{pmatrix}\succeq0,
\qquad |0\rangle=(1,0)^T
\]

なら、γ=εは小さくても||Γ||=M+ε、b²=εMは別の大きさを持つ。従ってTγ=O(1)だけから||Γ||≪Δは従わない。

固定したVでθを1/Tに下げる漸近議論は可能だが、その場合でも||V||/⟨V⟩₀等の定数を明示する必要がある。これは実分子の値を計算した反例ではなく、原文の一般的な含意の限界を示す紙上例である。

---

## 6. P3の固有値摂動にも、明示的な剰余を付ける

§5の信号定理だけでRPEへの接続はできるが、ClaudeのP3の中心式も別途確認する。

Hの非縮退基底gapをΔ、g_\infty=||Γ||≤Δ/4とする。b²=||QΓ|0⟩||²、γ=⟨Γ⟩₀と置く。

K=H−iΓのE₀へ連続につながる固有値κ₀について、

\[
\beta_2=\langle0|\Gamma Q[Q(H-E_0)Q]^{-1}Q\Gamma|0\rangle,
\quad 0\le\beta_2\le b^2/\Delta,
\]

\[
\boxed{\left|\kappa_0-(E_0-i\gamma+\beta_2)\right|
\le \frac{11}{7}\frac{g_\infty^3}{\Delta^2}.}
\tag{6.1}
\]

**証明の詳細。** A_Q=Q(H−E₀)Q、B=A_Q−i(QΓQ−γQ)、z=κ₀−E₀+iγと置く。Schur complementから

\[
z=g^\dagger(B-z)^{-1}g.
\]

半径r₀=2b²/Δ≤Δ/8のdisk内で、|| (B−z)^{-1} ||≤1/(Δ−r₀)。右辺はdiskを自身へ写し、Lipschitz定数はb²/(Δ−r₀)²≤4/49<1。従ってこのdisk内の固有分枝は一意である。b=0は直接κ₀=E₀−iγ。

resolvent identityから

\[
|z-\beta_2|
\le\frac{b^2\|Q\Gamma Q-\gamma Q\|}{\Delta^2}
+\frac{16b^4}{7\Delta^3}.
\]

PSD性より||QΓQ−γQ||≤g_\infty、b≤g_\infty、g_\infty/Δ≤1/4を用いて(6.1)を得る。□

純減衰モデルの右固有ベクトルuについては、Re κ=⟨u|H|u⟩/⟨u|u⟩≥E₀でもある。ただし元H状態の有限時間信号のphaseは、単一固有値だけでは決まらない。

従って、**純粋な−iΓが単独で加わるモデルでは、Claudeの二次係数の符号と次数は正しい。** ただし以下を限定する。

- 実lawのHermitian補正E_P／E_EやPF誤差が加われば、全energy偏りの符号はこのβ₂だけでは決まらない。
- κ₀が分かっただけでは、元H状態の有限時間信号の位相・半径保証は完成しない。ここは§5を用いる。
- ||Γ||≤Δ/4はこの固有値展開の十分条件。§5の基底信号定理には不要だが、信号のrelative誤差が小さいという条件は別途必要。
- 任意の励起状態、縮退部分空間、相関samplingへこの式を自動移送しない。

これは条件付きの解析命題であり、保存H4で条件が満たされたという科学結果ではない。

---

## 7. C4／RPE：物理的な信号の位相と半径を同時に保証する

### 7.1 diskから位相・半径への写像

実際に測定する、各shotのfreshな独立trajectoryを平均した複素信号をz(T)とする。§4のε_dyn(T)と§5を用い、

\[
d(T)=b^2\min\{T^2/2,2T/\Delta\}+\epsilon_{\rm dyn}(T),
\]

\[
|z(T)-e^{-iE_0T-\gamma T}|\le d(T).
\tag{7.1}
\]

a(T)=e^{-γT}、η(T)=d(T)/a(T)<1なら、半径aの点を中心とする半径dの円盤は原点を含まない。幾何から、

\[
\boxed{\rho_{\rm lb}(T)=e^{-\gamma T}-d(T)>0,
\qquad \beta_{\rm sys}(T)=\arcsin(e^{\gamma T}d(T))}
\tag{7.2}
\]

が生信号の半径下界と、元Hのphaseからのsystematic phase誤差上界になる。角度の距離はまず2πを法とする距離であり、energyの一意復号とは別である。

**E₀や正確なγを実行時に知る必要はない。** E₀は証明の参照点であり、decoderへ渡さない。γ≤γ₊、b²≤b²₊、Δ≥Δ₋>0、ε_dyn≤ε₊という取得可能な上界・下界があれば、

\[
d_+=b_+^2\min\{T^2/2,2T/\Delta_-\}+\epsilon_+,
\quad
\rho_{\rm lb}=e^{-\gamma_+T}-d_+,
\quad
\beta_{\rm sys}=\arcsin(e^{\gamma_+T}d_+)
\tag{7.3}
\]

を使える。gapの保証がなければ、minをT²/2だけに置き換える。これはgapがなくても必ず有用な費用になるという意味ではない。

γ₊・b²₊をoperator normで粗く取ることは可能だが、状態依存の改善を失うおそれがある。**保存ground stateを診断計算に使えること、計算規模に対して効率的な設計情報として使えること、数学的に認証されたmoment上界が得られることは別である。** 本レビューでは新しいmomentもgapも計算していない。

### 7.2 既存shot会計へ接続する十分条件

各軸のoutcomeが±1、同じround内でfresh IID trajectoryと測定を実行する条件を置く。0<β_stat,m<π/2、0<α_m,b<1とする。X/Y平均の各推定誤差をη_stat以下に抑えれば、複素平均の誤差は√2η_stat以下になる。

\[
\eta_{{\rm stat},m}=
\frac{\rho_{{\rm lb},m}\sin\beta_{{\rm stat},m}}{\sqrt2},
\]

\[
\boxed{N_{m,b}=\left\lceil
\frac{2}{\eta_{{\rm stat},m}^{\,2}}
\log\frac{2}{\alpha_{m,b}}
\right\rceil,\qquad b\in\{X,Y\}.}
\tag{7.4}
\]

Hoeffdingの十分条件とunion boundで全軸の統計失敗確率はΣ_{m,b}α_{m,b}以下になる。この式は[G2]の会計と同形であり、**新しい統計推定器ではない**。新しいのは、その入力となる物理信号のρ_lbとβ_sysの根拠である。

既存APIはcanonical RTEのB⁻¹と補正平均diskを前提にしている。C-αの生信号を、既存のPF半径へそのまま入れてさらにB⁻¹を掛けてはならない。必要なのは、(7.3)から得たraw半径・phase保証を明示的に受け取る接続である。新しいadapter自体は今回実装していない。

phase補正をshotごとのancilla回転・読出し軸として実行するなら、±1範囲は保てる。一方、別々のX/Y値を後処理の重み付き推定で組み替える場合は、同じ範囲・shot式が無条件に使えるとはしない。

### 7.3 branch・alias・energyの条件

全ての統計誤差が所定範囲に入った事象で、round mの元energyに対するphase誤差をβ_m=β_sys,m+β_stat,mで囲む。model／implementation誤差をε_dyn内に含めた場合、さらに同じ誤差をphase予算へ二重加算しない。別項として扱う設計ならその分離を明記する。

T_m=2T_{m-1}のnearest-branch型復号では、正しい初期aliasを選べるprior条件に加え、例えば

\[
\beta_m+2\beta_{m-1}<\pi
\tag{7.5}
\]

が、前roundから予測したphaseの近傍で正しい2π分枝を一意に選ぶ十分条件になる。最終roundで正しい分枝が維持されれば、energy誤差はβ_M/T_M以下である。実際の既存decoderが要求するより強い条件は別途維持する。

**元Hの既知energyを用いて事後的に分枝を選ぶ方法ではない。** 初期prior、energy origin、既知scalarの制御位相、符号規約を整合させる。

### 7.4 P1へそのまま数値を代入してよいわけではない

P1のr4/K2、保存shot値、.04 radのRTE枠、.24 radのstat枠等は既存契約の結果であり、本レビューはそれらを変更していない。C-αではlawとbiasの根拠が変わるため、新しいscopeで条件と誤差配分を明示する必要がある。

数理的には、**元H状態→物理平均→有限時間disk→phase/radius→shot→branch/energy**の接続を構成できた。実H4で全roundの不等式が満たされること、十分shot数が減ること、旧方式より安くなることは、今回何も算出していない。

---

## 8. P5・R6と情報アクセス：残すもの、修正するもの

### 8.1 最小化されるのは固定stateでの二次減衰係数

R=Σ_F A_F、p_F>0、Σp_F=1、c=Σc_Fとする。X_F=(A_F−c_F I)/p_FならE X=R−cI。

固定したstate |0⟩についてm_F=⟨A_F⟩₀、σ_F²=Var₀(A_F)、a_F=m_F−c_Fと置くと、

\[
\boxed{v(c,p):=\langle V\rangle_0
=\sum_F\frac{\sigma_F^2+a_F^2}{p_F}
-\operatorname{Var}_0(R)-\left(\sum_Fa_F\right)^2.}
\tag{8.1}
\]

Cauchy–SchwarzによりΣa_F²/p_F−(Σa_F)²≥0。c_F=m_Fはこの部分をゼロにする。全σ_F>0ならp_F=σ_F/Σσ_Fが残りを最小にし、

\[
\boxed{v_{\min}=\left(\sum_F\sigma_F\right)^2-\operatorname{Var}_0(R).}
\tag{8.2}
\]

さらに|0⟩がH=D+Rの固有状態なら、(R−⟨R⟩)|0⟩=−(D−⟨D⟩)|0⟩なのでVar₀(R)=Var₀(D)。これは理想的な同じHについての等式であり、H_physとH_modelが異なる場合の無条件な等式ではない。

**確率は分散比例ではなく標準偏差比例**である。これに近い固定stateの目的関数と確率はWu–Fan–Zhang[P6]に直接の先行があり、C-αの新規性としない。

なお中心化の最適値は一意とは限らない。固定pについてa_F=a p_Fという共通定数aもCSの等号を満たす。外側scalarを正しく戻すと、各trajectoryのunitaryは同じになり得る。この自由度を独立した改善パラメータとして二重計上しない。

### 8.2 zero varianceとzero supportを区別する

σ_F=0は(A_F−m_F I)|0⟩=0を意味するが、A_F−m_F Iが演算子としてゼロとは限らない。別eventで状態が|0⟩から変わった後に、その項を省略してよい保証はない。

従って、作用素としての一次平均を保存する仕様では、非zero operatorへp_F=0を割り当てない。確率floorを設ければ、(8.2)は一般に達成されず、実際の正規化確率を使って(8.1)と高次momentを評価する。state-specificな省略を扱うなら別の証明・比較条件が必要になる。

### 8.3 状態情報なしの半幅上界

全primitiveが保存するsectorで、A_Fのスペクトルを[c_F−b_F,c_F+b_F]に含められるとする。p_F=b_F/B、B=Σb_F（b_F=0の純scalarは別処理）なら、

\[
\sum_F\frac{(A_F-c_FI)^2}{p_F}\preceq B^2 I,
\qquad 0\preceq V\preceq B^2 I.
\tag{8.3}
\]

これは直接のoperator不等式であり、状態分散のPopoviciu不等式だけから推測したものではない。sector内のb_Fを使用するには、そのsectorの不変性が必要である。厳密端点の取得が高価なら安全な包含区間でよいが、その分だけ保証は弱くなる。

### 8.4 平均分散の最小化は総費用の最小化ではない

(8.2)が最小にするのは指定stateのvであり、次を同時に最小にするわけではない。

- Var₀(V)やVar₀(Γ)、gapを用いたrelative信号上界。
- Σ||A_F−c_F I||³/p_F²という高次剰余。
- Gaussian変換・fragment指数・control・合成精度・workspaceを含むevent単価。
- moment推定・古典前計算・取得誤差の費用。
- 全roundのshot加重費用。

StateDRIFT[P7]の公式abstractも、input stateとobservableに依存するsampling、bias–variance、情報取得を扱うことを確認できる。従って「state-aware確率を使う」というだけの差分は特に弱い。全文中のenergy eigenstateの扱い等はCodex報告[C2]に依拠する部分であり、このレビューで直接再確認したとはしない。

粗い推定値に基づくp_F,c_Fでも、実際の確率p_Fと回転時間1/p_Fが一致し、全supportを保てば一次の作用素平均は保たれる。しかし高次位相、減衰、認証可能な上界は変わる。**「情報が粗くても効率だけが変わり、正しさには影響しない」という原文の無限定な表現は修正が必要**である。

### 8.5 R6は今回のIID保証へ含めない

独立samplingなら、順序を保ってE[U_N⋯U_1]=E[U_N]⋯E[U_1]が成立する。相関があればこの等式は使えず、§4の平均macrostepの冪から構成する保証も自動的には成立しない。

定常性・可逆性による順序付き二点momentの対称性は参考になるが、有限時間の余剰を抑えるには初期分布、mixing、相関和、高次の順序付きmomentが必要である。また非可逆性は位相項を**許し得る**のであって、必ず非零の位相を生むわけではない。例えば、全生成子が同じ可換演算子である場合に、非可逆な古典label過程だけで非可換phase機構が生じるとはいえない。

R6は別仕様として保留し、α0-Pの最初の接続に含めない。これは相関方式一般の否定ではない。

---

## 9. 先行研究との対応と、今回主張できる追加性

### 9.1 直接再確認した内容と、Codexの報告に留まる内容

| 文献 | このレビューでの確認 | C-αに関係する位置付け |
|---|---|---|
| PR原論文[P1] | PDFのqDRIFT固有状態信号、RTE、関連付録の式を直接確認 | atan位相・energy依存減衰は既知。完全乱択の関数を部分乱択の全energyへ無条件に移せない |
| UWC[P2] | v2 PDFのqDRIFT定義と付録Bの係数差を直接確認 | arctan型qDRIFTをPRへ入れる構成、RTEとの係数2の差は直接先行 |
| SPRINT[P3] | v1 PDF付録F.1の本文・式を確認。該当頁の画像も確認 | 固有線shift、分散dephasing、gap分母・他分枝は強い部分先行。「減衰とgap」が新しいという説明は不可 |
| Kivlichanら[P4] | PDF定理1と関連付録の記述を確認 | sampled Hamiltonian群のgapと変化量による位相推定保証。同じΓモデルではない |
| Guら[P5] | PDF定理1・関連記述を確認 | Hermitian Kraus channelの位相中立性。operator first momentへの保証の直接移送ではない |
| Wu–Fan–Zhang[P6] | v1 HTMLの確率・fidelity・moment取得に関する関連節を確認 | 標準偏差samplingの直接先行。RPEのphase／radiusの目的とは区別 |
| StateDRIFT[P7] | **公式abstractのみ直接確認。全文取得は失敗** | state-aware bias–variance・observable最適化が既知。本文の厳密な重複判定は未完 |
| Morisaki–Fujii[P8] | **今回全文は取得できず、詳細は[C2]の照合報告** | channelだけではなくcoherent平均も扱うという訂正を保持。独立に全文読了したとはしない |
| その他[C2]のComposite、Jin–Li、Chen、qSWIFT等 | 今回の詳細再読は行っていない | 再監査済みという表示をしない。一次tail・composite・平均bias自体は既知という保守的整理に用いる |

先行研究の内容を、本レビューの紙上命題と同じ仮定・同じ定数・同じ費用保証であると推測しない。一方、同一タイトルの定理が見当たらないことも、新規性の証拠にはならない。

### 9.2 残る候補と、新規性を認定しない理由

今回構成したのは、既存の一次lawを使い、

> **元Hの基底状態を保持したまま、純減衰による状態間の結合と、law／PF／model／実装の誤差を分離し、有限時間信号の位相・半径から既存RPEへ接続する十分保証**

である。

これは本レビュー内で具体化できた有用な研究仕様である。しかし、§5のVolterra・coercive resolvent評価や§4のStrang評価は標準的な解析道具から導いている。**ここで導出したことを、新しい数学的原理や公表済み研究に存在しない定理と認定しない。** 特にSPRINTの摂動解析との差は、一般的な着想ではなく、仮定・有限時間範囲・剰余・保証入力・適用可能性で比べる必要がある。

StateDRIFTの全文を直接取得できていないため、その保証が同じHadamard観測量を含めて今回の結果を既に包含するかは、このレビューでは確定できない。Hadamard testをancilla込みのobservable推定として表すことは可能なので、「observableが目的だからRPEとは無関係」とは言えない。ただしその変換をしたときのinput access、control、phase復号、半径、全round費用まで同じかも確認が要る。

“Which terms to keep coherent? …”は、著者・一次識別子を今回も確定できなかった。**存在しないとは判断せず、確実に存在する必須論文とも扱わない。** 書誌候補が得られた時点で照合するUNKNOWNとし、未確定タイトル一件を理由に研究全体を無期限停止させない。

### 9.3 他Trackとの境界

Track Bの希薄補償には、指数平均と線形平均、PSD収縮とnormalizationの交換が既にある。「同じtarget制約を初めて外す」というC-αの説明は撤回する。[C2]

本レビューの中心はtailの既存lawに対する元H信号保証であり、Track Bの新しいreturn集約、Track Cの新PF族・compiler、GPT案の保証付きD/R optimizerは実施していない。

一方、**C-αの保証が成立しただけで、研究組織上の帰属が自動確定するわけではない。** 元Track Aとして継続するなら、この有限時間信号保証を独立した中心課題として明示し、既存FR・Track Bの収縮解析・他Trackの未解決事項との差を保持する。新しいsampling法を開発する必要が出れば、境界を再確認する。

---

## 10. 研究判断と次の進め方

### 10.1 今回の判定

**C-α全体に科学的GOは出さない。しかし「数理的骨格が未完成だから何も進められない」という状態からは前進した。**

本レビューで、α0-Pの物理平均から元H基底状態の有限時間信号、phase/radius、RPEの十分条件まで、適用条件を明示した解析経路を示した。一般指数lawにも、より保守的だが同じ接続を用意した。

一方、以下は未確定である。

- 対象分子におけるgap・moment・交換子上界の値と取得可能性。
- その十分条件で必要shot数とevent数が実際に減るか。
- fragment化、中心化、有限bit、native価格を含めた全費用。
- 先行研究に対する独立した論文上の増分。

CodexのB分類に対応する**未採択候補**として保持するが、その内訳は「保証の骨格が未完成」から「条件付き十分保証を提示し、その適用可能性と追加性が未確定」へ更新する。単にコード部品があることや、紙上の上界があることを、実分子の適格性・費用優位に読み替えない。

### 10.2 推奨する最初の対象

検討順としては、**α0-P／IID／正時間partial-S2／元Hの同一保存sector**を先にする。理由は、既存のpaired K=0 lawと直接対応し、Γ_P・R_Pの厳密なoperator式が得られ、一般fragment lawのlogや希少確率の問題を初回から同時に抱えなくてよいからである。

これはα0-Pが最良という性能順位ではない。α0-Eは境界例で異なる挙動を持ち、有用な対照として残す。α1／α2の状態分散最適化とR6の相関拡張を、最初の単一検証へ一括追加しない。

### 10.3 次の担当：Codexがまとめて扱える仕様・接続確認

今回のレビューの推奨を採用する場合、次はCodexへ、既存資産に対する限定的な仕様整備をまとめて依頼するのが合理的である。

1. 既存K=0のraw平均と式(2.1)の対応、角度・sign・identityの扱いを固定し、通常指数lawと混同しない。
2. SOURCE上のheadの積順序・sector保存性・finite-bitの実装確率・元HとDF modelの違いを、既存の固定資料に対応させる。
3. 式(4.9),(7.3)の各入力について、既存資料から得られるもの、静的に書けるもの、未取得の科学量を分ける。未取得値を0にしない。
4. raw radius／phase保証を既存会計へ接続する最小仕様と、対象を固定した科学的pilot案を作る。実装・計算の許可範囲は別途明示する。
5. StateDRIFT本文等、残っている重要一次資料が取得できるなら、その保証の対象だけを照合する。すでに終わった全分野の文献調査を一から繰り返さない。

軽微なSOURCE照合や数式の転記確認ごとにGPTの詳細科学レビューを要求する必要はない。新しい前提・主要claim・研究範囲に関わる問題が出た場合だけ戻す。

### 10.4 科学的pilotを選ぶ場合に判断したいこと

**次の計算はまだ承認されていない。** 以下は、必要なら後続の実行計画で扱う研究問いである。

- 実際のraw信号が、減衰のみの参照とどの程度ずれるか。
- ε_dyn、Var₀(Γ)、gapを使う上界が実際のphase/radiusを囲むか。上界が保守的で囲めない場合と、実信号自体が悪い場合を分ける。
- 既存RTE、α0-P、α0-Eの同じtarget Hに対する誤差と費用交換がどのように違うか。
- 追加情報を使うなら、その取得費・誤差と、強い対照へ同じ情報を与えた結果。

人工行列のsimulation、保存H4状態からのmoment算出、回路生成・compile、価格取得はいずれも新しい科学実行として扱う。実行対象・SOURCE・条件・STOPを定めたユーザーの許可があれば足り、新しい承認証明書・grant制度を要求しない。

最初から全18round・全変種・H6展開を必要条件にせず、最初の判別で研究判断が変わる最小範囲を選ぶ。一方、少数点の正結果を全roundの保証へ拡大もしない。

### 10.5 継続・縮小・停止の条件

- 同じinput accessと実行条件の先行結果が今回の構成・保証を包含するなら、新規性の主張を縮小し、適用・検証としての価値を別に判断する。
- gapや必要なmomentの取得が不可能な条件では、その条件での保証を主張しない。gapなしの評価等に切り替える場合はその弱化を明示する。
- 上界が保守的すぎるだけの場合は、アルゴリズム一般の不可能性とは判断しない。実際の信号が悪い場合とも区別する。
- 既存最良PRに対する総費用の追加利益がなければ、費用改善claimを出さない。Full／Truncatedに勝たないことが直ちに全ての理論的価値を否定するわけでもない。
- 原レビューの「3倍」等の例示的閾値は、今回新しいSTOP契約として採択しない。

---

## 11. 主張の強さ・未確認事項

| 項目 | 状態・確信度 | 限界 |
|---|---|---|
| paired K=0と通常指数lawの区別 | 高。SOURCE・一次文献・代数が一致 | SDK loweringの全実装監査ではない |
| L1の明示的上界・中心三次momentだけの限界 | 高。本文で直接導出 | 上界の鋭さは未評価 |
| 収縮参照発展へのL2接続 | 高。仮定付きでdefect／Duhamel証明 | 正時間、指定配置、有限次元、実装差の別会計 |
| 元H基底状態からの有限時間信号上界 | 高（紙上の導出として） | 第三者査読・形式証明・数値照合は未実施。世界初の主張ではない |
| 純減衰の二次固有値補正 | 高。§6で剰余付き | law・PFを含む全偏りの符号ではない |
| state依存samplingの新規性 | その一般原理は既知 | StateDRIFT等との詳細な包含関係は未確定 |
| 実分子での実用性・総費用 | UNKNOWN | 新規科学計算をしていない |
| 元Track Aでの継続価値 | 中。限定理論課題として継続推奨 | 正式採択・他Trackとの帰属確定ではない |

**補足：** 「高」は、明記した前提からの紙上の証明に対する評価である。非専門的な直感だけの保証ではないが、外部の独立検証を済ませた数学的certificateという表示もしない。

既存のscope別SOURCE、raw、REPORT、grant、ledger、marker、mandatory STOPを変更していない。旧I／II／IIIや他Trackの実行・保留状態も変更していない。

---

## 12. 参考資料と実際の確認範囲

### 12.1 添付・プロジェクト資料

**[C0]** `research_project_instructions_unified_2026-11-09.md`。原文のレビュー・科学実行・成果保全・研究階層の規則を確認。旧版およびClaude用テンプレートを運用正本にしていない。

**[C1]** `元TrackA_独立研究方向再探索_Claude科学レビュー_20261110.md`（2026-11-10、531行）。特に§3.3 O1–O7、§5.2–5.9、先行研究台帳。候補の原文であり、そこにある「数理」のラベルを厳密証明の完成と扱わない。

**[C2]** `貼り付けたテキスト（1）(7).txt`（Codex静的監査報告、本文136行の提供資料）。law不一致、sector36/70、先行研究の追加確認、Track Bのtargetに関する訂正、SOURCE再利用、L1–R6の未解決点。Codexが取得した全論文・全SOURCEを本レビューでも独立に再取得したとは主張しない。

**[C3]** `track_a_new_algorithm_hypotheses_20261110(1).md`。旧I–IIIの定義と、raw tailとは異なる誤差補償対象の区別。旧IIの優先順位は後続の判断で撤回されている。

### 12.2 固定GitHub資料（読取りのみ）

**[G1]** `src/trotterlib/rte.py`、固定commit `2e79d80dc9cce2885580e50fe6f1705da8fa41af`、blob `53601b7c524653f5e1966b3955b9f51373e54002`。今回直接確認した範囲は1520–1665行および1720–1990行。`_make_event`のatanとsign、event unitary、finite Taylor補正平均、normalizationとの関係。コードは実行していない。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/src/trotterlib/rte.py

**[G2]** `docs/research/track_a_b2_p1_shots_20261107/REPORT.md`、固定commit `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。P1の元H/state、sector36、固定予算、raw半径とHoeffding会計の式、条件付き／not_certifiedの留保を確認。科学実行SOURCEは文中に別途固定されており、閲覧commitと混同しない。保存数値の新規集計は行っていない。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md

### 12.3 外部一次文献

**[P1]** Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647。公式PDFでqDRIFTの固有状態信号、atan位相、RTEの定義と関連付録を確認。version付き取得が失敗し、取得できた無version URLのPDF本文を使用した。本文の取得表示は2026年7月版。PDF画像も一部取得したが、キャッシュで式番号の表示に差があったため、本文の内容とSOURCEの直接照合を優先した。全版本文の差分を監査していない。

https://arxiv.org/pdf/2503.05647

**[P2]** Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2。公式PDFのqDRIFT定義（§II.2.2、式(9)–(13)周辺）と付録BのqDRIFT/RTE係数を直接確認。該当PDF screenshot要求の一部はcache missで失敗し、未取得の図表を視覚確認済みとはしていない。

https://arxiv.org/pdf/2603.22778v2

**[P3]** Casares et al., SPRINT、arXiv:2606.30741v1。公式PDF付録F.1のspectral shift、cumulant／dephasing、固有ベクトルと他分枝の項を確認。PDFの該当2頁を画像でも確認した。C-αの全保証と同一であると認定したわけではない。

https://arxiv.org/pdf/2606.30741

**[P4]** Kivlichan, Granade, Wiebe, arXiv:1907.10070。公式PDFの定理1と関連付録を確認。変動するHamiltonianの差とその群のgapを用いる保証。screenshotは一部取得失敗、図に基づく数値読取りはしていない。

https://arxiv.org/pdf/1907.10070

**[P5]** Gu et al., arXiv:2208.04100。公式PDFのHermitian Kraus雑音の定理1と関連記述、該当頁画像を確認。補足証明全体の再証明は行っていない。

https://arxiv.org/pdf/2208.04100

**[P6]** Wu, Fan, Zhang, *Fluctuation-guided adaptive random compiler for Hamiltonian simulation*, arXiv:2509.10158v1。公式HTMLの§II、標準偏差sampling、fidelity目的、moment取得の関連箇所を確認。

https://arxiv.org/html/2509.10158v1

**[P7]** Coyle et al., *Problem-aware sampling for bias and variance reduction in randomised Hamiltonian simulation*（StateDRIFT）, arXiv:2610.31515。**公式abstractのみ直接確認**。PDF／HTML全文の取得はこのレビューでは成功しなかった。本文の定理・energy eigenstateの詳細については[C2]が報告する範囲と、今回の直接確認を区別する。

https://arxiv.org/abs/2610.31515

**[P8]** Morisaki–Fujii, arXiv:2609.36694。**全文の独立取得は成功せず**。[C2]が主文・付録の平均作用素・Haar平均coherent誤差を確認したと報告している。そのため「channelしか扱わない」とは整理しないが、このレビュー自身が同じ箇所を読了したとはしない。

https://arxiv.org/abs/2609.36694

**[P9] その他の先行研究の索引**：[C2]の文献表。Hagan–Wiebe（2206.06409）、Kiss–Grossi–Roggero（2212.05952）、Jin–Li（2109.07987）、Chenら（2008.11751）等。今回全てを再読したとは扱わない。名称だけで新規性を除外／認定しない。

---

## 付録A：式(3.3)の導出

U(t)=e^{-itμ}、E(t)=e^{tBθ}、Bθ=−iμ−θV/2、v=||V||と置く。V≥0よりEは正時間でcontractionである。Duhamel公式から

\[
E(\theta)=U(\theta)-\frac\theta2
\int_0^\theta E(\theta-s)V U(s)\,ds.
\]

また||E(t)−U(t)||≤tθv/2なので、積分中のE(θ−s)をU(θ−s)へ置換する誤差は

\[
\frac\theta2\int_0^\theta
\frac{(\theta-s)\theta v}{2}\,v\,ds
=\frac{\theta^4v^2}{8}.
\]

U(θ−s)VU(s)=U(θ)U(−s)VU(s)で、

\[
\|U(-s)VU(s)-V\|\le s\|[\mu,V]\|.
\]

従って

\[
\left\|E(\theta)-U(\theta)(I-\theta^2V/2)\right\|
\le\frac{\theta^3}{4}\|[\mu,V]\|
+\frac{\theta^4v^2}{8}.
\]

式(3.1)と三角不等式で(3.3)を得る。逆時間のUはunitaryであるため問題ない。減衰semigroup Eの逆時間を有界と仮定していない。

## 付録B：収縮Strangの定数24・12

A、B、A+Bが正時間でcontraction semigroupを生成するとする。有限次元なので全交換子は有界。S(t)=e^{tA/2}e^{tB}e^{tA/2}、C=[A,B]と置く。

\[
D(t):=S'(t)-(A+B)S(t)
=[e^{tA/2},B]e^{tB}e^{tA/2}
+e^{tA/2}[e^{tB},A/2]e^{tA/2}.
\]

交換子の積分形を使うと、最初の項は

\[
\left(\int_0^{t/2}e^{(t/2-s)A}C e^{sA}\,ds\right)e^{tB}e^{tA/2},
\]

二番目は

\[
-\frac12e^{tA/2}
\left(\int_0^t e^{(t-s)B}C e^{sB}\,ds\right)e^{tA/2}.
\]

それぞれから(t/2)e^{tA/2}C e^{tB}e^{tA/2}を引き、加えると主要項は相殺する。残りは、正時間の収縮性ともう一度交換子の積分形を用いて

\[
\|D(t)\|\le\frac{t^2}{8}\|[A,C]\|
+\frac{t^2}{4}\|[B,C]\|.
\]

変分公式

\[
S(\delta)-e^{\delta(A+B)}
=\int_0^\delta e^{(\delta-t)(A+B)}D(t)\,dt
\]

を積分して式(4.7)を得る。負の時間の減衰operatorを使わないことが、この単純な定数評価の条件である。

---

## 最終状態

**本レビューは、α0の定義修正と、条件付き有限時間信号保証の構成を完了した。** 元H基底状態を別stateへ置換せず、law・PF・model・実装の残差を明示する解析経路を得た。

**未完なのは新規性の完全判定、実分子への適用、実装・総費用の科学検証である。** それらを完了したと扱わない。

研究上は、C-α原文の全面採択ではなく、α0-Pからの限定的な仕様化・検証設計を推奨する。実行には各対象に必要な明示的な許可を得る。既存の保留・mandatory STOPは維持する。
