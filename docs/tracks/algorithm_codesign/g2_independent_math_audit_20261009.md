# G2：一般式の独立数学監査

2026-10-09。対象は[GPT G1レビュー](../../research/track_b_G1_scientific_review_20261009.md)§6–9,13。
判定は **PASS_FOR_STATED_IDEAL_DIAGNOSTIC_ONLY**。以下はCodexが式を再導出した一般証明であり、
有限点のtestsで一般命題を代用しない。優先性、新アルゴリズムの採択、旧RA-D0 witness、実装lawの認証ではない。
実装・cost/error boundsは既存保存表のものを条件として用いる。

## 1. identity return：恒等式・交点・単調性

R=Σp_i Q_i、p_i≥0、Σp_i=1、Q_i²=I、χ=Σp_i²、D=Σ_{i≠j}p_i p_j Q_i Q_jとする。
R²=χI+D、R³=χR+RD。従って

\[
P_3(-i\sigma xR)=(1-\chi x^2/2)I-i\sigma(x-\chi x^3/6)R
 -(x^2/2)(I-i\sigma xR/3)D.
\]

ここではRDの順序を保持し、DRへの交換は用いない。回路時間順序は右から左。
最後のrotation labelとDの二labelは独立、Dの二labelだけを不一致で条件付ける。
σ=±1、controlled相対phaseを保持する。χ=1ではD=0で、条件付け分布は作らない。
低次数係数が負になる大きいxでもatan2とsigned phaseを用いれば恒等式は保てるが、
現在の非負7-prototype classへ同一実装として組み込む命題ではない。
今回のdevelopment x={1/8,1/4}, χ=5/8では両係数は正。

u=(1,x)、v=(x²/2,x³/6)、a²=||u||²、b=||v||、c=u·v、A=||u+v||と置く。
Bret=||u−χv||+(1−χ)b。x>0ならc>0、A>b、
det(u,v)=−x³/3≠0。さらに

\[
(bA)^2-(b^2+c)^2=b^2a^2-c^2=x^6/9>0.
\]

全項が正なので b(A−b)>c>0。H=b(A−b) とすると χ*=(H−c)/(H+c) は(0,1)。
右辺 A−b+χb>0を確認してから平方比較すると

\[
\|u-\chi v\|^2-(A-b+\chi b)^2=2[H-c-\chi(H+c)].
\]

従って0≤χ≤1の全域で Bret<A iff χ>χ*、等号 iff χ=χ*。
微分も Bret'=(χb²−c)/||u−χv||−b<0。
最後のstrict性は u−χv がvと平行にならないことからCauchy–Schwarzで従う。
根の符号を落として平方するだけの導出ではない。

z=x²とおくと b=z/2+z²/36−z³/1296+O(z⁴)、
A=1+z−5z²/24+2z³/9+O(z⁴)、c=z/2+z²/6。
H−c=z²/9−17z³/162+O(z⁴)、H+c=z+4z²/9−17z³/162+O(z⁴)。
したがって χ*=z/9−25z²/162+O(z³) で、レビューの展開と一致する。
これらの級数係数は平方と商の係数比較から独立に得た。

この比較はweight normalizationのみ。新returned angleのnative cost/errorはMISSINGであり、
normalization比を総T/CX/1Q比へ置き換えない。

## 2. G1固定線形価格：必要十分条件

μ=(x²+2)/(x²+6)。7倍率は
(s,b,μ+(1−μ)s−μr−b,1−r−b,1−s,1−s,r)。
価格ℓ=(ℓO0,ℓO2,ℓP2,ℓP3,ℓA0,ℓA1,ℓA2)を掛け、各変数を集めると

\[
F=F_0+\alpha s+\beta r+\zeta b,\quad
F_0=\ell_{A0}+\ell_{A1}+\mu\ell_{P2}+\ell_{P3},
\]

α=ℓO0−ℓA0−ℓA1+(1−μ)ℓP2、β=ℓA2−μℓP2−ℓP3、ζ=ℓO2−ℓP2−ℓP3。
既に監査済みの6頂点凸包を仮定して、ordinary/PTSC/Aの増分は(α+ζ,α,β)、
J1/J2/J3の増分は(0,μζ,α+β)。線形目的の凸混合は成分価格の混合なので、
min(0,μζ,α+β)<min(α+ζ,α,β)が追加自由度のstrict利益の必要十分条件。
α等一つの符号だけでは全winnerを決めない。同precision固定価格として診断する。
正規化Bを再度掛けるcanonical net cost、bias分母、capsの非線形問題へは拡張しない。

## 3. 既知IS、zero-cost、confidence

非負event係数c_i、outcome Y_i∈{−1,+1}、full-support proposal π_i>0に対して
Z=(c_i/π_i)Y_i。m2=Σc_i²/π_i、L=max c_i/π_i。
Cauchy–Schwarzから

\[
(\sum\pi_i C_i)(\sum c_i^2/\pi_i)\ge(\sum c_i\sqrt{C_i})^2=K^2.
\]

C_i>0なら等号 iff π_i∝c_i/√C_i。
これは[Resource-Optimal Importance Sampling, Theorem 1, Eq.(7)–(13)](https://arxiv.org/html/2603.13495v1)
の既知net-cost原理と一致する。bias不変は同じ実装を再重み付けするという条件付き。
独立した新規性の証明ではない。

c_i>0のzero-cost eventとpositive-cost eventが併存するとfull-supportの等号proposalは存在しない。
positive-cost側を総mass t、内部は c_i/√C_i 比、zero側を1−tで正に割り振り、t→0とすると
net costはK²へ収束する。しかしm2は発散する。
従ってinfimumを達成可能な有限shot実装と呼べない。
全event C=0ならnet cost 0自体は達成可能だが他資源・measurementは無料ではない。
0を任意の小さい費用へ置換せず、今回zero-cost側の有限proposalはMISSING。

各eventのjoint observable bias上界e_i（保存表では2δ_i）を同じまま用いれば、
|bias|≤Σc_i e_i はπに依存しない。implementationやphaseを変えるとこの保証は継承しない。
残りs=ε−Σc_i e_i>0、Var≤m2、|Z−EZ|≤2Lから、通常のBernstein sufficient条件は

\[
n\ge\ln(2/\alpha)\{2m2/s^2+4L/(3s)\}.
\]

positive-cost ISでは S=Σc_i/√C_i、m2=SK、EC=K/S、L=S√Cmax。
連続shotの目的資源は ln(2/α){2(K/s)²+(4/3)(K/s)√Cmax}。
これはnet-cost最適proposalの予測であり、finite-confidence全proposalの最適性ではない。
整数shot、別資源費用、state preparation/readout、sampling acquisition、law丸めを別記する。
今回は旧R1のper-axis ε=1/200、α=1/5280、shot cap=10⁹を**記述的な各予測**へ使用する。
504 profile全体に対する新しいfamilywise実測保証は主張しない（測定実行0）。

## 4. 線形分数化とpure-profile完全性

各prototype/precision jの独立条件付きlabel法p(ω|j)を固定する。
w_j≥0、event c_jω=w_j p(ω|j)、h_j=Σp√C（sqrt(ΣpC)ではない）、
保存bias上界d_j=Σp e_jωとすればK=hᵀw、s=ε−dᵀwはaffine。
ideal degree matching Dw=tの下でK/sの最小化は、v=w/s、u=1/sにより
Dv=ut、εu−dᵀv=1、v≥0、u>0、目的hᵀvへ**双方向**に変換できる。
逆写像w=v/uではs=1/u>0。今回このLPは実行しない。

G1で証明されたγの6頂点分解 γ=Σθ_v γ^(v) を用いる。
positive groupはπ_gp=γ_gp/γ_gとし、各頂点でも同じπを与える。
zero groupは非負性から全positive θ_vのγ_g^(v)=0であり、divisionは不要。
頂点vのactive groupsについて、pure precision assignment p_gへ確率
λ_(v,p)=θ_v∏_(g active) π_(g,p_g) を与える。
Σλ=1、各group/precisionの係数は元γ_gpに一致する。
inactive groupのprecisionを任意に付けてprofile数を増やさない。

任意の混合でK≥0、s>0とする。q=min_(s_v>0) K_v/s_v≥0を取ると、
positive s_vについてK_v≥q s_v、nonpositive s_vについてもK_v≥0≥q s_v。
全成分を加えてK≥q s、従ってq≤K/s。
よってpositive-sのpure profileのいずれかが任意の混合以上に良い。
6頂点active数(2,3,3,4,4,3)、3 precisionから252/x。
元3頂点側も同じ証明で63/x。zero-costを含む場合はnet-cost **infimumモデル**の完全性。

仮定はideal degree equality、固定dictionary/internal law、非負価格、affine bias上界、
他資源capなしである。旧numeric K3、丸め後mean/membership、shot/range最適性、
precision-dependent internal law、新dictionary、signed cancellationまでの完全性ではない。
特にcanonical目的 B·Σw EC/s²はaffine分子でなく、252 profile最小をcanonical混合全体の最小と呼ばない。
保存された保守的bias上界を用いた本モデルの最小は、全物理実装のlower boundでもない。

## 5. 未解決の認証と停止境界

上記4命題に記載前提内の反例・blocking issueは見つからなかった。
focused synthetic/off-domain testsは算術・保存identity拒否・zero-cost・scopeを検査する補助である。
一般証明そのものはこの文書。元G1のsymbolic監査・backend・one-shotは再実行していない。
実装law/dyadic rounding、全finite-confidence最適proposal、元classのU₃<L₂、return/CTSの
同task費用、取得費用は未解決。保存値診断後はGPT G2へ戻し、追加実行は行わない。
