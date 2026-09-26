# FR-1後の研究方針改訂：共通振幅変化と状態依存の位相歪みを分ける

作成日：2026-09-26  
対象：`HIROMU1015/Partially-Randomized-Trotter`  
branch：`all-r-coherent-opt2-reoptimization`  
証拠基点：`ecb7f4c007ab4dd98666a53035bf2eefff17f0bb`

**文書の位置付け**：FR-1の結果を受けた提案。リポジトリの正式方針や既存gateを書き換えるものではない。今回、新しい研究用simulation、回路compile、全testの再実行は行っていない。資料・実装・一次文献を照合し、代数的な再分析とscalar式の記号的検算を行った。以下の新しい検証条件は実行前に別途承認・凍結する案である。

**採択状況**：本案の数式・情報層・比較baseline・停止規則は、[FR-R0正式契約](docs/research/fr_revision_scalar_structure_contract.md)へ採択した。FR-R0と数値事前登録は分離し、FR-R1a事後解析計画とFR-R1b事前登録は別文書として固定済みで、どちらも未実行である。本案と正式契約が衝突する場合は正式契約を優先する。

## 0. 推奨する決定

FR全体を直ちに放棄することも、旧計画のFR-2をそのまま開始することも勧めない。**一度だけ、比較対象と機構の定義を修正する短い検証単位を置く。**

主研究候補は、次の一文へ変更する。

> 有限RTEの打切り誤差のうち、全状態に共通する正の振幅変化を解析上分離した後、残るスペクトル依存の変形と入力状態の条件付けが、Hadamard信号の位相精度をどのように決めるか。それを利用可能な情報から評価し、必要な近似精度を判断できるか。

最初の実行単位は **FR-R0（定式化・強い対照）→ FR-R1a（既存toyの再解釈）→ FR-R1b（非一様スペクトルの4×4判別）→ 停止** とする。FR-R1aだけの改善では主題化しない。H4、FR-2、H12、旧P-D S2、長RPE、全分子、backend/noiseへ自動進行しない。

「高い参照信号半径を得る新手法」は最初の必須開発にしない。まず、従来の上界が、位相を変えない共通振幅まで位相誤差として課金していないかを修正する。一般系でなお状態情報が必要になったときだけ、その取得方法と費用を検討する。

## 1. FR-1から実際に言えること

### 1.1 確認された結果

FR-1の結果文書は、33条件・99状態、495個の適用可能method recordでbound違反0、G0/G1/G3/G4通過、主有用性G2不通過を報告している。主gridは `H_D=0.7Z`、`H_R=cos(theta)Z+sin(theta)X`、`theta=pi/3`、`T=0.8`、`K=2`、`r=1`、`q=1,2,4,8,16`。主判定の入力情報は `rho_lower=0.8`。[G1,G2]

| q | 提案/STRONG位相上界比 | 解釈 |
|---:|---:|---|
| 1 | 1.106794 | この点では旧提案の方が悪い |
| 2 | 0.855074 | 上界の縮小はある |
| 4 | 0.727864 | 上界の縮小はある |
| 8 | 0.663982 | 上界の縮小はある |
| 16 | 0.631998 | 約36.8%縮小だが、事前基準50%縮小には未到達 |

事前登録の50%基準と、提案だけが1e-3 radを認証する基準は、いずれも満たしていない。元の `GO_FR2_MECHANISM_ONLY` とG2不通過は保存する。

### 1.2 前の判断の訂正

「利用可能情報では利益が消えた」「高いrhoが得られないことが唯一の障害」とするのは強すぎる。正しくは、**今回の情報・旧上界・gridでは、設定した大きさの利益または設定変更につながる片側認証を示せなかった**。

数値testが全件通ったことは一般的な証明ではない。一方、50%基準を通らなかったことも、数学的な改善の不存在を証明しない。この二つを対称に扱う。

### 1.3 旧上界の漸近的な限界

FR-0の旧提案は、概略的に

\[
Y=b+R_2+\kappa(\underline\rho)(s+R_2),\qquad
L=1-a-R_2-\kappa(\underline\rho)(s+R_2),
\]

\[
\kappa(\underline\rho)=\frac{\sqrt{1-\underline\rho^2}}{\underline\rho}
\]

として `atan(Y/L)` を使う。[G3]

K=2、同種の小step、固定Tでstepを細分化する領域では、radial成分が先頭になり、`a~s`、`b/s→0`、`R_2/s→0` となる。STRONGの局所誤差と旧提案のeが同じ先頭係数を持つこのtoyでは、

\[
\frac{\beta_{\rm proposed}}{\beta_{\rm strong}}
\longrightarrow \kappa(\underline\rho)\underline\rho
=\sqrt{1-\underline\rho^2}=0.6
\quad(\underline\rho=0.8).
\]

これは今回の式に対する漸近診断であり、任意の有限stepで比が必ず0.6以上という定理ではない。ただし、同じ旧上界でqだけを増やし、0.5未満が出るまで探索する根拠は弱い。

## 2. 新たに特定できた重要な構造：FR-1の主toyは特殊である

### 2.1 非可換性はあるが、tailの二乗は単位行列

FR-1の

\[
h_\theta=\cos\theta Z+\sin\theta X
\]

は

\[
h_\theta^2=I
\]

を満たす。`[H_D,h_theta]` が非零でも、この恒等式は変わらない。[G2,G4]

ここで重要なのは、RTEを構成する**個々のinvolution**が二乗Iであることと、それらを足した**tail Hamiltonian全体**が二乗Iであることは別だという点である。一般のDF tailは後者を満たさない。[W1]

### 2.2 K=2の有限Taylorは正のscalar×unitaryへ正確に分解できる

\[
P_3(-i\tau h)
=\left(1-\frac{\tau^2}{2}\right)I
-i\left(\tau-\frac{\tau^3}{6}\right)h
\quad(h^2=I).
\]

\[
a(\tau)=1-\tau^2/2,\qquad b(\tau)=\tau-\tau^3/6,
\]

\[
\alpha(\tau)=\sqrt{a(\tau)^2+b(\tau)^2}
=\sqrt{1-\tau^4/12+\tau^6/36}>0,
\]

\[
\varphi(\tau)=\operatorname{atan2}(b(\tau),a(\tau))
\]

とすれば、原点から連続な角度規約の下で

\[
P_3(-i\tau h)=\alpha(\tau)e^{-i\varphi(\tau)h}.
\]

したがって、exact deterministic unitaryを任意に挟んでも、全corrected列は

\[
A_{\rm corr}=\left(\prod_j\alpha_j\right)\widetilde U
\]

という正scalar×unitaryである。全体の正因子は位相を変えず、信号振幅を変える。

**これは正確な代数的観察であり、新しい合成法や世界初の定理として主張しない。** FR-1のtoyが、状態に依存して振幅が変わる一般の非unitary平均を十分に試していないことを示す。

### 2.3 旧boundが見落としていたこと

同じtoyでは

\[
e^{i\tau h}P_3(-i\tau h)-I=f_K(\tau)I+i g_K(\tau)h.
\]

F成分はscalarそのものである。しかし旧上界は `||F||` を取り、`kappa s` にも含めるため、**位相を直接変えないscalar成分まで、状態ずれによって位相へ混ざり得るものとして数える**。[G3,G4]

従って次の順序が合理的である。

1. 共通振幅を解析上分離した強いnorm対照を作る。
2. 旧提案の改善が、単なるscalar処理だけで説明できるか確認する。
3. 二乗Iでない多準位tailで、なお独立した役割が残るか調べる。

高いrhoを得るための新しい認証法を先に開発する必要はない。

## 3. 改訂後の研究RQと主張の範囲

### 主RQ

> 有限RTE平均列の位相精度は、打切り誤差全体のnormではなく、共通振幅を除いたスペクトル非一様性、位相方向の誤差、入力信号の条件付けから、どこまで予測・保証できるか。

### 副RQ

- **構造と限界**：どの条件でradial誤差の先頭項が位相から除かれ、どの条件では入力状態によって同じ次数で位相へ現れるか。
- **実用性**：必要なスペクトル情報と信号半径下界を、full-system eigendecompositionなしに与えられるか。その費用まで含めてcutoff/substep選択に利益があるか。

### 最初に固定するタスク

固定された参照unitary列Uに対する、単一の複素Hadamard信号の位相推定を対象とする。Uはtailをexact exponentialに戻した同じPF列である。

\[
z_0=\langle\psi|U|\psi\rangle,\quad
z_{\rm corr}=\langle\psi|A_{\rm corr}|\psi\rangle,\quad
z_{\rm obs}=z_{\rm corr}/\mathcal B.
\]

物理Hamiltonianの基底エネルギーとUの位相の差は別のPF誤差であり、本研究のfinite-RTE誤差と混合しない。RPE全体の成功やenergy biasの主張は、別途の接続を行った場合に限る。

独立なRTE occurrenceの平均積を使う。相関samplingへの一般化はしない。チャネル平均 `E[U rho U†]` とcoherent平均 `E[U]` を区別する。[W1,W2]

## 4. 理論作業：共通振幅を除いた上界

以下はFR-0の補題を正scalarで再規格化して使う導出案である。独立の代数監査を行うが、この変換そのものを新規性とみなさない。

### 4.1 局所的な正scalar分離

\[
A_j=U_j(I+D_j),\quad D_j=F_j+iG_j,
\]

とし、F_jのスペクトルが既知の区間 `[f_j^-,f_j^+]` に入るとする。例えば

\[
c_j=(f_j^-+f_j^+)/2,\quad d_j=(f_j^+-f_j^-)/2,
\quad \gamma_j=1+c_j>0
\]

と置く。

\[
A_j=\gamma_j U_j(I+\widehat D_j),\qquad
\widehat D_j=\frac{F_j-c_jI+iG_j}{\gamma_j}.
\]

これにより

\[
\|\widehat F_j\|\le d_j/\gamma_j,\qquad
\|\widehat G_j\|\le\|G_j\|/\gamma_j.
\]

局所のF/Gが同じHermitian h_jの関数なら、`e_hat`は同じ保証スペクトル集合上で

\[
\widehat e_j\ge
\sup_{x\in\mathcal S_j}
\left|\frac{e^{ix}P_{K_j+1}(-ix)}{\gamma_j}-1\right|
\]

として計算できる。非可換なのは異なるoccurrenceを結んだ列であり、単一occurrenceのF/Gは可換である。

端点が分からない場合、保証区間を使う。保証区間上のsupremumを未検証grid最大値に置き換えない。`gamma_j<=0`ではこの正scalar版を採用せず、gamma=1の元のboundへ戻すか適用不能とする。

### 4.2 全列への合成

\[
\Gamma_c=\prod_j\gamma_j,\quad
\widehat A=A_{\rm corr}/\Gamma_c,
\]

\[
\widehat a=\sum_j\widehat a_j,\quad
\widehat b=\sum_j\widehat b_j,\quad
\widehat s=\sum_j\widehat e_j,
\]

\[
\widehat R_2=\prod_j(1+\widehat e_j)-1-\widehat s.
\]

与えられた保証 `0<rho_lower<=|z_0|` から

\[
\bar\kappa=\frac{\sqrt{1-\rho_{\rm lower}^2}}{\rho_{\rm lower}}
\]

を作り、

\[
\widehat L=1-\widehat a-\widehat R_2-
\bar\kappa(\widehat s+\widehat R_2),
\]

\[
\widehat Y=\widehat b+\widehat R_2+
\bar\kappa(\widehat s+\widehat R_2)
\]

とすれば、`L_hat>0`の範囲でFR-0と同じ証明から

\[
\Delta\phi\le\arctan(\widehat Y/\widehat L),\qquad
|z_{\rm obs}|\ge\frac{\Gamma_c\rho_{\rm lower}\widehat L}{\mathcal B}
\]

を得る。

証明の要点は、`A_corr/Gamma_c`に対するFR-0の仮定が再び成立し、positive scalarはargを変えないことである。Hermitian Fの中心を選ぶことは、その固有値幅を使うことに等しい。

### 4.3 必須baseline：正scalarを除いたnorm-only評価

同じgamma、同じ情報から

\[
\widehat E=\prod_j(1+\widehat e_j)-1
\]

を作り、`E_hat<rho_lower`なら

\[
\Delta\phi\le\arcsin(\widehat E/\rho_{\rm lower}),\qquad
|z_{\rm obs}|\ge
\Gamma_c(\rho_{\rm lower}-\widehat E)/\mathcal B
\]

を比較対照にする。

さらに強い対照として、同じ保証スペクトル情報を使い、各局所評価について正scalarを一次元最適化してよい。

\[
\inf_{\gamma>0}
\sup_{x\in\mathcal S_j}
\left|e^{ix}P_{K+1}(-ix)/\gamma-1\right|.
\]

中心選択だけが唯一のnorm対照だとしない。この最適化に保証付き数値手法を使う場合、前処理costも記録する。主張する最適性はこの有限/一次元の評価class内に限定する。

**共通振幅を除くだけで得られる改善を、位相専用の新しい方法の改善として二重計上しない。**

### 4.4 物理的な減衰は消えない

正scalar分離は解析上の操作であり、sampler、回路、shotの統計を変更しない。

\[
z_{\rm obs}=(\Gamma_c/\mathcal B)\langle\psi|\widehat A|\psi\rangle.
\]

位相から除ける係数でも、観測振幅とshot費用には残る。特に `diag(I,gamma U)` を `gamma diag(I,U)` とみなしてはいけない。control-|0> branchまでscalar倍したわけではない。正の振幅係数と、identity Hamiltonian項の既知だが非零の相対位相も別である。

### 4.5 一般系では何を期待し、何を期待しないか

K=2では

\[
F(\tau h)=-\tau^4h^4/24+O(\tau^6).
\]

先頭radial成分の非scalarな大きさは、例えば

\[
\inf_c\|F-cI\|
\simeq\frac{|\tau|^4}{48}
\left(\lambda_{\max}(h^4)-\lambda_{\min}(h^4)\right)
\]

で決まる（極値の同定と剰余の扱いが必要）。`h^2=I`ならこの先頭幅は0、一般の複数絶対固有値では非零である。

従って、一般の入力状態とtailで、位相誤差が必ず5次になるとは主張しない。非一様radial成分が残れば、状態混合によって4次で位相へ寄与し得る。

## 5. 必要な条件を見極める反例

### 5.1 rhoだけでは非scalarなradial成分を一律に除けない

一般的な相対誤差classの例として

\[
U=e^{-i\vartheta Z},\quad |\psi\rangle=|+\rangle,\quad
A=U(I+\epsilon Z),\quad 0<\vartheta<\pi/2
\]

を取る。`|epsilon|<1`ならradial因子は正定値である。

\[
z_0=\cos\vartheta=\rho,\qquad
z_A=\rho-i\epsilon\sin\vartheta.
\]

したがって

\[
|\arg z_A-\arg z_0|
=\arctan\left(|\epsilon|\frac{\sqrt{1-\rho^2}}{\rho}\right).
\]

Hermitianな相対誤差でも、非scalarなら一般状態で一次の位相誤差を起こす。この例は一般相対誤差classの限界であって、任意のepsilonを固定KのRTEが正確に生成するという主張ではない。

RTE固有の限界を論じる場合は、次節の複数絶対固有値を持つhに対し、`F~h^4`の非一様性から同じ先頭次数が出ることを別途示す。

### 5.2 研究として残すべき範囲

単に旧toyのscalar性を見つけることや、上記の一般反例だけでは独立主題としての着地点に足りない。必要なのは、finite-RTE列について

- どの利用可能情報が位相改善を保証するか、
- どの情報だけでは改善を保証できないか、
- その差が要求精度または設定選択へどう効くか、

の少なくとも一つを、適切な既知対照より具体化することである。

## 6. 入力情報を増やす前に、情報量をそろえる

| 情報層 | 利用してよいもの | 扱い |
|---|---|---|
| I0 | 明示されたnorm/LCU係数上界と粗いrho下界 | 最も汎用的。スペクトルは保証区間だけ |
| I1 | 代数的に既知のスペクトル集合・不変sector・involution関係 | 同じ構造情報を全baselineへ渡す |
| I2 | 小系full diagonalization、真のrho、正確な極値 | 機構診断専用。実用性の根拠にしない |

主toyの `spec(h)={-1,+1}` を提案だけが使い、baselineには `[-1,1]` しか渡さない比較は、方法差と情報差を混合する。両者を別のablationとして表示する。

### 6.1 高いrhoの認証は既存法を先に試す

入力がHの固有状態という条件で、既知のPF operator bound

\[
\|U-e^{-iHT}\|\le\varepsilon_U
\]

があるなら、

\[
|\langle\psi|U|\psi\rangle|\ge\max(0,1-\varepsilon_U)
\]

を使える。exact Hの対象固有状態への重みを `w>=w_lower>1/2` と保証できる場合は、単純な対照として

\[
\rho_{\rm lower}\ge\max(0,2w_{\rm lower}-1-\varepsilon_U)
\]

がある。これは標準的な三角不等式の利用であり、新規性ではない。より鋭いresidual/gap型の既存解析とも比較する。[W3,W4]

**energy bias、D6係数、過去に測った大きいrhoは、そのままoperator boundや別の入力のrho保証にはならない。**

今回のanalytic mixtureは参照Uの固有vectorを使って構成されている。`rho_lower=0.8`という数学的保証は正しいが、実用状態の準備・証明が安価だと確認したわけではない。[G2,G4]

### 6.2 スペクトル幅を安価に与えられるか

主な候補は、明示した不変sector上の保証区間、可換blockの解析的固有値範囲、`||h^2-sigma^2 I||`の上界である。

例えば

\[
\|h^2-\sigma^2I\|\le\varepsilon_2
\]

なら

\[
\|h^4-\sigma^4I\|
\le\varepsilon_2(\|h^2\|+\sigma^2).
\]

これ自体は初等的である。DF/LCU成分の構造からepsilon_2を安価に評価できるかが次の検討対象となる。`h=sum_l p_l P_l`のとき、h²にはpair anticommutatorが現れるため、全pair処理が必要なら少なくとも成分数に対し二次の費用となり得る。安価だと決めつけない。

状態に依存した狭いスペクトル集合を使う場合、入力がそこに存在することだけでなく、全参照prefixでそこから漏れないこと、または漏れの上界も必要である。

## 7. 先行研究との差分と修士研究・論文の着地点

### 7.1 既知として出発するもの

| 一次資料 | 既知として扱う内容 | 今回の比較で注意する点 |
|---|---|---|
| PR / RTE [W1,W2] | Taylor/LCU平均、normalization、Hadamard信号、sampling負担 | finite-RTEや信号の線形平均自体は新しくない |
| spectral PF [W3]、QPE error [W4] | 固有値/固有vector/入力状態/residual/gapを使う誤差解析 | 状態情報を使うことだけを新規性にしない |
| phase-lag/dissipation [W5,W6] | scalar近似の位相と振幅の次数分離 | involution toyのscalar計算だけで主題化しない |
| SPRINT [W7] | randomized PFのspectral shift、damping、peak/weight解析 | 非可換列の有限cutoffに対する具体的な仮定・出力量で比べる |
| Gu et al. [W8] | Hermitian Kraus noiseに対する位相の一次不変性 | チャネル固有位相の摂動とcoherent平均の有限時間信号を混同しない |

[W8]は現行FR prior-art表にない追加対照である。Theorem 1と補足の非可換noise積に関する議論が近い。ただし、CPTPチャネルの固有位相と、今回の `arg <psi|E[U_omega]|psi>` は違うため、既存定理をそのまま代入できるとは限らない。

このscoped調査は、新規性や優先権の不存在/存在の証明ではない。新しい定理が既に完成していないことだけを理由に探索を禁止もしない。先行研究にない明確な問いと、反証可能な予測を置いて、小さく確かめる。

### 7.2 最小の着地点

**同じ入力情報を使う強いbaselineを含め、有限RTEにおける共通振幅と非一様radial変形の違いを定量化し、位相認証の改善・不改善を説明する条件を示す。**

以下のいずれかが必要である。

- 既存結果より強い、または適用範囲/必要情報が異なる位相・半径評価。
- 非一様radial変形が改善を制限することに関する、finite-RTE固有の具体的な上界と対応する例。
- 原理から構成した認証規則が、強い対照より有用なK/r選択を未使用条件で行う。

正scalarを因数分解しただけ、旧toyで比が0.5を下回っただけ、数値testが全部通っただけでは不足する。

### 7.3 目標の着地点

**状態・スペクトル情報の取得負担を含め、同じ信号位相精度・成功確率の下でK/rを選び、より少ない実行資源を得るか、従来法で十分になる条件を事前に識別する。**

論文の主張は、固定された参照信号タスクと認証条件に限定してよい。Hamiltonianの全エネルギー推定や全RPEへ広げる場合は、別途PF誤差・branch・全round資源を接続する。

### 7.4 研究の完了を無制限に延ばさない

この主題で必要なのは、明確な機構/方法、適切な対照、未使用条件、正しいscopeである。H12、全PF family、実機noise、状態準備法全体を完成条件に追加しない。

改善が得られなくてもすべてが論文になるとはしない。既知の初等事実だけで説明できる結果なら、技術ノートまたは既存研究の検証節として閉じ、独立主題にはしない。

## 8. 検証計画：最初の一巡

旧FR-1は変更しない。新しい作業名をFR-Rとする。FR-R0の代数・比較契約を先に完了し、FR-R1aの事後解析計画とFR-R1bの数値事前登録は別文書へ分離して凍結する。FR-R0完了だけでは計算を開始しない。

### FR-R0：式と比較契約の確認

**目的**：高いrhoを新たに求める前に、scalar処理と強いbaselineを確立する。

作業：

1. h²=Iの因数分解と、旧bound比の0.6への漸近を独立検算する。
2. 第4節の正scalar版を、gamma>0、位相branch、半径、積順序、control branchについて確認する。
3. 正scalarを最適化できるnorm対照を定義する。
4. 第5節の限界例を、一般relative-error例とfinite-RTE固有例に分ける。
5. [W1]--[W8]との仮定・出力・計算負荷の対応を一枚にする。

終了：新規性を断定するのではなく、「既知scalar処理だけで何が解決し、その先に何を問うか」を固定する。明白な既存定理への代入だけなら、独立手法の主張を縮小する。

### FR-R1a：既存33条件を同じ情報で再解析

**目的**：FR-1の成否が、特殊なスペクトルと旧上界のscalar課金でどこまで説明できるかを確認する。

同じ演算子/状態について、OLD、STRONG、旧FR、正scalar-NORM、正scalar-FR、h²=Iのexact polar対照を比較する。rho_lower=0.8をそのまま使う主比較と、真rhoの診断を分ける。

数値演算子の再構成が必要なら行うが、それを新しいholdoutとは呼ばない。元のG2を再分類しない。

出力：旧/new上界、実位相、物理振幅、gamma、B、Fの非scalar幅、情報源、適用不能理由。正scalar処理だけの効果と、さらにHermitian分離を使う効果を分ける。

**ここで勝っても主研究GOにはしない。** h²=Iの解析的対照を再現した段階である。

### FR-R1b：非一様スペクトルの4×4最小判別

以下は具体的な事前登録案であり、結果前に採択して固定する。

\[
h_\nu=\frac{1+\nu}{2}Z\otimes I+
\frac{1-\nu}{2}I\otimes Z,
\qquad \nu\in\{0,1/2,1\}.
\]

\[
\operatorname{spec}(h_\nu)=\{1,\nu,-\nu,-1\},\qquad \|h_\nu\|=1.
\]

`nu=1`はinvolution対照、`nu=1/2`は複数絶対固有値、`nu=0`は零固有値を含む強い非一様性である。identity項を足しただけの2×2では、identity抽出後に再びinvolutionへ戻るので、主反例にしない。

非可換deterministic blockの案：

\[
H_D=0.7X\otimes I+0.3Z\otimes X.
\]

可換対照の案：`H_D=0.7 Z⊗I+0.3 I⊗Z`。

| 項目 | 初期固定案 |
|---|---|
| 主cutoff | K=2 |
| 共通物理時間 | T=0.8 |
| outer steps | q=2,4,8 |
| inner tail steps | r=1 |
| 列 | symmetric referenceと、そのtailをfinite Taylorに置換した列 |
| 主状態 | 同一$(\nu,H_D)$について一度だけ構成し、全$q$で再利用する固定superposition。構成法と全$q$共通の$\underline\rho$保証を事前登録で固定 |
| 主情報 | supplied certificate rho_lower=0.8。状態構成にdenseを使ったことを明記 |
| 追加状態 | 参照固有状態、physical ground stateの診断 |
| 有限演算子 | 4×4直接参照。trajectory列挙やMCは不要 |

主行列条件は3 nu×3 q×2構成の18条件で、三状態を評価すれば54状態行となる案である。ただし負時間、K=4、signal-zero付近のcontrolは、具体的な$\nu,q,r,K,H_D,state$を結果前に固定する。「代表を少数」とだけ書いて結果後に選ばない。全符号×全K×全状態の大gridにはしない。

主状態は混合状態でなく、例えば `sqrt(0.9)|u0> + sqrt(0.1/3) sum_{k=1}^3 |uk>` というpure superpositionである。同一$(\nu,H_D)$では一度だけ構成し、各$q$のeigenbasisから作り直さない。縮退部分のprojector、canonical basis、global phaseと、全$q$共通のrho下界の保証法を事前登録する。三角不等式によるrho保証が可能でも、cheap preparationを証明したことにはならない。

**主判定**：

正式なGOには[FR-R0契約](docs/research/fr_revision_scalar_structure_contract.md)のsoundness、非一様性、同情報利益、decision relevance、oracle非依存、機構整合の6条件を全て要求する。上界比の改善だけではGOにしない。

- 全適用点で、式から導かれた上界/下界が実値を数値許容差内で覆うか。
- nu=1だけでなく、nu=0または1/2でも有用な領域があるか。
- 正scalar-NORMを超える情報があるか、それとも改善が全て基本的なrescalingで説明されるか。
- radial非一様性と状態条件が、改善の有無や先頭次数を事前予測するか。
- exact spectrum入力の改善と、norm intervalだけの改善を分ける。

比較誤差許容値は、機械精度、積因子数、normスケールから事前に固定する。解析的に近零の量は高精度再計算またはinterval参照で検証する。単なるgrid最大値を数学的supremumにしない。

### FR-R1終了時の分岐

| 結果 | 判断 | 次の作業 |
|---|---|---|
| nu=1だけで改善し、正scalar-NORMで全て説明 | 特殊toyの監査として閉じる | 旧FRを一般実用法として拡大しない |
| 非一様系でも改善、ただしdense固有値/rhoが必須 | 条件付き機構は残るが実用法未成立 | 安価な保証情報の取得可能性を一件だけ机上評価 |
| 非一様系で、同情報の強い対照を超える認証/適用条件 | 主題候補を維持 | FR-R2の単一H4検証を新たに固定 |
| 一般反例が上界の次数を飽和し、新しいfinite-RTE固有の限界を与える | 理論寄りの着地点を検討 | 既知perturbation結果との差を再照合 |
| 数式のsoundness違反 | 実行拡張を停止 | 原因を修正し、旧版の失敗履歴を保存 |

**FR-R1b終了時に必ず停止する。** 同じtoyでrho/nuを動かし続け、GOが出るまで探索しない。修正を行う場合は、何が誤っていたかを明記した新しい仮説とする。

## 9. 条件付きの次段：実用的情報と一つのH4

FR-R1が有望な場合だけ、FR-R2へ進む。これは旧FR-2の自動再開ではなく、修正した問いと対照による新規契約である。

最初は既存のH4 snapshot、同じDF分割、同じPF、同じ物理時間で固定する。既参照H4はdevelopmentである。K/rを少数だけ変える。

検証することは次の三つに限定する。

1. actual tailのradial非一様性が、involution toyとどれだけ違うか。
2. dense参照ではなく、代数的/LCU/不変sectorの保証情報と既存PF boundから得るrho下界でも使えるか。
3. その情報取得costが、避けたい詳細評価より安いか。

各候補を `analytic_certified`、`norm_bound_certified`、`provided_state_certificate`、`dense_oracle_diagnostic` に区分する。最後の区分を実用的GOへ混ぜない。

half-widthやrhoが得られずinconclusiveになることは正しい出力である。そのとき、より都合のよいgeometryを次々に追加しない。情報不足を補う一つの具体的方法がないなら、この実用化経路は止める。

## 10. 設定選択・資源比較は、その次に行う

### 10.1 共通タスクと公平な選択

固定されたU、入力、T、要求位相精度、成功確率を全法でそろえる。各法が同じK/r候補から設定を選ぶ。exact-rho版はoracle benchmarkとして分ける。

既存のq/δを固定した局所機構試験と、設計ごとのK/r最適化を混同しない。物理Hamiltonianの位相へ拡張する場合はPF誤差予算も必要となる。

### 10.2 振幅とshotを戻す

同じ参照演算子について、各法は位相上界 `beta_sys` と観測半径下界 `rho_obs_lower` を出す。正scalar-NORMでも修正FRでも、分離したGamma_cを半径式に戻す。

例えばX/Yの±1測定を独立に行い、各軸N shot、統計位相予算beta_stat、総失敗確率alphaを指定する保守的設計では、

\[
\epsilon_{\rm coord}=\rho_{{\rm obs},lb}\sin\beta_{\rm stat}/\sqrt2
\]

とし、Hoeffdingと2軸union boundから

\[
N\ge\frac{4}{\rho_{{\rm obs},lb}^2\sin^2\beta_{\rm stat}}
\log\frac4\alpha
\]

を各軸へ割り当てる。これは一つの共通統計baselineであり、最適shot法と主張しない。

\[
G_{\rm model}=2N\,\mathbb E[C_{\rm 1shot}]
\]

で比較し、証明/較正用の追加計算や追加測定は別に計上する。cutoffによりeventの次数分布・期待action数は変わるため、1-shot costを固定値のまま扱わない。

### 10.3 閾値の固定

旧G2の50%基準を緩めて合格に変更しない。旧判定は保存する。

新段階では、理論的soundness、上界のtightness、設定選択、資源利得を分ける。新しい誤差予算を旧結果の間へ後から挟んで、one-sided successを作らない。

具体的な探索案は、固定TのもとでK={0,2,4}、r={1,2,4,8,16}を共通候補にし、要求位相予算は実際の用途または事前固定した小さい集合から選ぶ。用途が未確定なら「信号位相タスクの感度分析」と表記し、化学精度や最終QPE資源へ換算しない。

資源改善の採択基準は、計算前にcost指標と不確かさに合わせて固定する。上界を少し縮めただけではなく、同じ要求条件を満たす設定・保証費用・適用可能範囲が変わるかを重視する。

## 11. 新規性と妥当性を混同しない運用

各段階で判断を三つに分ける。

- **数学・意味論が正しいか**：正scalar、相対位相、平均演算子、半径、積順序、独立sampling。
- **同じ情報を使う既知法を超える知見があるか**：上界、適用条件、反例、計算費用、選択性能。
- **対象タスクで意味があるか**：実現可能な入力情報、shotと回路cost、未使用条件。

一つ目のPASSだけで研究を採択しない。一方、特定改善率のFAILだけで、機構の理解や別の有用な定理の可能性を否定しない。

baselineと予測を固定してから、本検証のholdoutを生成する。失敗例も保存する。旧H4、旧FR-1、今回の4×4開発点は、一度見た後は新しいholdoutと呼ばない。

## 12. 記録と成果物

新しい記録系列：

- `docs/research/fr_revision_scalar_structure_contract.md`（FR-R0として採択済み）
- `docs/research/fr_revision_fr1a_posthoc_plan.md`（固定済み・未実行）
- `docs/research/fr_revision_nonuniform_preregistration.md`（固定済み・未実行）
- `docs/fr_revision_nonuniform_results.md`
- 対応するlibrary、runner、test、artifact。

必須保存項目：

| 分類 | 項目 |
|---|---|
| 入力 | commit、source hash、h/H_D、T、q/r/K、identity処理、state規約 |
| 情報 | rho下界の根拠、spectral setの根拠、情報取得cost、oracle使用flag |
| 構造 | `||h²-sigma²I||`、Fの区間/幅、選択gamma、Gamma_c、B |
| 比較 | original norm、optimized rescaled norm、old FR、rescaled FR、exact involution対照 |
| 参照 | actual phase、corrected radius、physical observed radius、数値誤差 |
| 判定 | 適用可能性、上界違反、one-sided acceptance、設定差、限定scope |
| 再現 | expected仕様、結果fingerprint、実行command、version、失敗履歴 |

凍結ファイルは末尾を含む完全性を確認する。expected task数、section終端、設定digestを凍結後に検査し、FR-0で起きた途中切れを繰り返さない。[G1,G3]

## 13. 直ちに行うこと・行わないこと

### 次の一件

**FR-R0、FR-R1a事後解析計画、FR-R1b数値事前登録は固定済みである。次はFR-R1aだけを実装・実行し、旧判断を変えず説明監査を完了する。FR-R1b契約はその結果で変更せず、後続4×4結果まで進んだ時点で必ず止める。**

現在のFR-1を都合よく再分類しない。新規性が確定した主題としてではなく、「定義とbaselineを修正したときにも独立した価値があるか」を判別する。

### 今は行わない

- rhoを真値へ近づけるためだけの高価な全対角化を実用法に組み込む。
- qを増やし、旧50%gateが通るまで探索する。
- 同じinvolution toyに角度だけを増やす。
- P-D S2、H12、長RPE、full compiled総costを再開する。
- 初等的scalar処理だけを新アルゴリズムと称する。
- 全方向の再pilotを自動で開始する。

## 14. 最終整理

修正の中心は「より高いrhoを要求する」ことではなく、**位相を変えない共通振幅を除いた、残る非一様性に研究対象を絞る**ことである。

これにより、旧FR-1の特殊性、旧boundの保守性、一般tailに残る本当の難しさを切り分けられる。成功時には、有限RTEの位相認証がどの構造・情報で有用になるかという明確な着地点がある。失敗時には、involution対照の再現に留まることを確認して、独立主題としては閉じられる。

## 15. 出典・確認範囲

### リポジトリ：すべてecb7f4c固定

[G1] [FR-1結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecb7f4c007ab4dd98666a53035bf2eefff17f0bb/docs/finite_rte_phase_amplitude_validation.md)

[G2] [FR-1事前登録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecb7f4c007ab4dd98666a53035bf2eefff17f0bb/docs/research/finite_rte_phase_amplitude_fr1_preregistration.md)

[G3] [FR-0数式契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecb7f4c007ab4dd98666a53035bf2eefff17f0bb/docs/research/finite_rte_phase_amplitude_contract.md)

[G4] [FR実装](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecb7f4c007ab4dd98666a53035bf2eefff17f0bb/src/trotterlib/finite_rte_phase_amplitude.py)

[G5] [既存の先行研究対応表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecb7f4c007ab4dd98666a53035bf2eefff17f0bb/docs/research/finite_rte_phase_amplitude_prior_art.md)

### 公開一次資料

[W1] Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647 / PRX Quantum 7, 020332 (2026). coherent平均とチャネル平均の区別、RTE定義、normalizationを確認。現行公開PDFの式番号とrepository参照v2の式番号は同一と仮定しない。  
https://arxiv.org/abs/2503.05647

[W2] Wan, Berta, Campbell, *A randomized quantum algorithm for statistical phase estimation*, arXiv:2110.12071 / PRL 129, 030503 (2022). randomized phase estimationの基礎対照。  
https://arxiv.org/abs/2110.12071

[W3] Yi and Crosson, *Spectral analysis of product formulas for quantum simulation*, npj Quantum Information 8, 37 (2022). 固有値/固有vectorと状態依存PF解析。  
https://www.nature.com/articles/s41534-022-00548-w

[W4] Li, *Some Error Analysis for the Quantum Phase Estimation Algorithms*, arXiv:2111.10430 / J. Phys. A 55, 325303 (2022). 残差、gap、近似unitary、入力状態。  
https://arxiv.org/abs/2111.10430

[W5] Van der Houwen and Sommeijer, *Phase-Lag Analysis of Implicit Runge–Kutta Methods*, SIAM J. Numer. Anal. 26(1), 214–229 (1989). scalar位相誤差解析。  
https://doi.org/10.1137/0726012

[W6] Papakostas and Tsitouras, *High Phase-Lag-Order Runge–Kutta and Nyström Pairs*, SIAM J. Sci. Comput. 21(2), 747–763 (1999). phase-lag/dissipation条件。  
https://doi.org/10.1137/S1064827597315509

[W7] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741 (2026), Appendix F. spectral shiftとdamping、状態weight。  
https://arxiv.org/abs/2606.30741

[W8] Gu, Ma, Forcellini, Liu, *Noise-Resilient Phase Estimation with Randomized Compiling*, PRL 130, 250601 (2023), arXiv:2208.04100. Theorem 1、補足Iと非可換noise積の議論を確認。一次Hermitian perturbationの位相不変性は本案だけの着想ではない。  
https://arxiv.org/abs/2208.04100

今回の文献調査はscopedであり、検索で完全一致が見つからなかったことを新規性の証明としない。提示した代数は独立に検算すべき導出・対照の原案であり、一般DF系での実用改善は未確認である。
