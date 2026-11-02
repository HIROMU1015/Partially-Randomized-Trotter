# FR-R0：正scalar分離・構造比較契約

最終更新：2026-09-26 JST  
状態：`FR_R0_COMPLETE_FR_R1A_PLAN_AND_FR_R1B_PREREGISTERED_NOT_RUN`  
基点：`all-r-coherent-opt2-reoptimization`、`ecb7f4c007ab4dd98666a53035bf2eefff17f0bb`

本書は、完了済みFR-1の`GO_FR2_MECHANISM_ONLY`を受け、次の計算より前に行う
FR-R0の数式・情報・比較契約を固定する。研究案の由来は
[FR-1後の研究方針改訂案](../../fr1_revised_research_plan_20260926.md)である。

FR-R0は文書上の再設計であり、新しいHamiltonian、対角化、RTE sampling、回路compile、
test実行、数値artifact生成を含まない。後続の[FR-R1a事後解析計画](fr_revision_fr1a_posthoc_plan.md)と
[FR-R1b事前登録](fr_revision_nonuniform_preregistration.md)は別文書として固定済みだが、
実装・計算はまだ開始していない。

## 1. 保存する旧判断

完了済みFR-1の事前登録、数値、gate、判定は変更しない。

- 33条件・99状態、495適用recordでbound違反0。
- G0/G1/G3/G4は通過、利用可能な$\underline\rho=0.8$でのG2は不通過。
- 判定は`GO_FR2_MECHANISM_ONLY`であり、旧FR-2は開始しない。
- 真の$\rho$を使う改善はoracle診断であり、実用的なGOではない。
- H4/H12、compiled回路、Monte Carlo、長RPE、最終総costへの含意はない。

FR-Rは旧FR-1の再分類ではなく、旧toyの特殊性と比較baselineの不足を切り分ける新しい
再設計単位である。

## 2. 修正した主RQ

主RQを次に固定する。

> finite-RTEのcorrected相対誤差から位相を変えない正の共通scalarを除いた後、残る
> spectral nonuniformityとstate conditioningは、同じ利用可能情報を使う強いnorm対照では
> 得られない安全かつdecision-relevantなHadamard信号位相認証を与えるか。

ここで問う対象は「高い$\rho$を新たにoracleで求める方法」ではない。また正scalarの
因数分解自体、Hermitian/anti-Hermitian分解自体、scalar ODEのphase-lag/dissipation分離自体を
新規性として主張しない。

## 3. involution対照で説明できる部分

$h=h^\dagger$、$h^2=I$、$K=2$ではTaylor numeratorは

$$
P_3(-i\tau h)
=\left(1-\frac{\tau^2}{2}\right)I
-i\left(\tau-\frac{\tau^3}{6}\right)h
=\alpha(\tau)e^{-i\phi(\tau)h},
$$

$$
\alpha(\tau)
=\sqrt{\left(1-\frac{\tau^2}{2}\right)^2
+\left(\tau-\frac{\tau^3}{6}\right)^2}>0,
$$

$$
\phi(\tau)
=\operatorname{atan2}\!\left(
\tau-\frac{\tau^3}{6},\,1-\frac{\tau^2}{2}
\right)
$$

である。従って参照$e^{-i\tau h}$に対するHermitian誤差は共通scalarとなり、状態ごとの
radial nonuniformityは生じない。旧FR-1の主toyは$h^2=I$なので、この特殊構造を持つ。

旧FR境界へ$\underline\rho=0.8$を入れた小時間極限で、提案/STRONG比が
$\sqrt{1-\underline\rho^2}=0.6$へ近づくことも、この対照から説明できる。これは
旧G2の0.5基準を通らなかった結果と整合するが、旧判定を変更する理由にはしない。

## 4. 一般の正scalar再中心化

一つのcorrected occurrenceについて

$$
D=U^\dagger A-I=F+iG,
\qquad
F=\frac{D+D^\dagger}{2},
\qquad
G=\frac{D-D^\dagger}{2i}
$$

とする。実数$c$を選び

$$
\gamma=1+c>0,
\qquad
\widehat D=\frac{D-cI}{\gamma}
=\frac{F-cI+iG}{\gamma}
$$

と置けば、厳密に

$$
A=\gamma U(I+\widehat D)
$$

である。$\gamma>0$なのでcorrected複素信号の位相は変えない。複数occurrenceでは

$$
A_{\rm corr}=\Gamma_c\widehat A_{\rm corr},
\qquad
\Gamma_c=\prod_j\gamma_j>0,
$$

実際の平均信号は

$$
z_{\rm obs}=\frac{\Gamma_c}{\mathcal B}\widehat z_{\rm corr}
$$

となる。従って$\Gamma_c$は位相上界から除いてよいが、物理的な信号半径、shot数、
controlled branchのvisibilityから除いてはならない。`diag(I,γU)`では$\gamma$は全系の
global scalarでないため、control-arm間の相対振幅として残る。

### 4.1 固定scalar規則

Hermitian部の既知区間$[f_-,f_+]$を同じ情報層から得られる場合、主たる解析的規則を

$$
c_{\rm mid}=\frac{f_-+f_+}{2}
$$

とする。これは$\|F-cI\|$の区間上界を最小化する。$1+c_{\rm mid}\le0$なら適用不能とし、
符号を吸収して位相を変える規則へ自動的に切り替えない。

### 4.2 1次元最適化対照

正scalar処理だけの最大効果を測るため、同じ許可情報から構成した目的関数を
$c>-1$上で最小化する`OPT-SCALAR-NORM`を強い対照として置く。探索区間、目的関数、
数値許容差、境界処理、情報取得と探索の古典costはFR-R1事前登録で固定する。

## 5. 再中心化後の境界

各occurrenceで$\widehat D_j=\widehat F_j+i\widehat G_j$とし、

$$
\|\widehat F_j\|\le\widehat a_j,
\quad
\|\widehat G_j\|\le\widehat b_j,
\quad
\|\widehat D_j\|\le\widehat e_j
$$

を同じ情報層から構成する。既存FR-0命題の
$a_j,b_j,e_j$を$\widehat a_j,\widehat b_j,\widehat e_j$へ置換すれば、
積剰余

$$
\widehat R_2
=\prod_j(1+\widehat e_j)-1-\sum_j\widehat e_j
$$

と$\rho$または事前供給された$\underline\rho$から、再中心化後の位相上界
$\widehat\beta_{\rm FR}$と半径下界を得る。最終的な観測半径下界には必ず
$\Gamma_c/\mathcal B$を掛ける。

この置換がsoundであることと、強いnorm-only対照より有用であることは別判定とする。

## 6. 情報層

比較に使った情報の強さを次の三層へ固定する。

| 層 | 利用を許す情報 | 役割 |
|---|---|---|
| I0 | 係数、$\|h\|$、Taylor remainder、$B_K$、解析的norm区間 | 実運用の最小対照 |
| I1 | 問題定義から安価に得る代数的spectrum、involution、対称性、不変sector、証明済みinterval | 構造利用候補 |
| I2 | dense対角化で得る真のspectrum、真の$\rho$、exact extrema、結果を見た最適scalar | oracle診断のみ |

実用的GOはI0またはI1だけで判定する。I2による改善は機構診断として保存するが、
同情報比較や設定選択のGOへ数えない。状態構成にdense情報を使う場合も、その事実とcostを記録し、
状態構成用oracleと境界評価用oracleを別flagで保存する。

## 7. 比較契約

同じ参照$U$、finite numerator $A_{\rm corr}$、$\mathcal B$、入力状態、$\underline\rho$、
情報層、位相branchを使い、最低限次を比較する。

| 名称 | 内容 | 判定上の位置付け |
|---|---|---|
| `OLD-NORM` | 正項Taylor remainderの積連鎖 | 歴史的対照 |
| `STRONG-NORM` | 同じspectral interval上の局所誤差supremumの積連鎖 | 強いnorm対照 |
| `OLD-FR` | scalarを除かない既存Hermitian/anti-Hermitian境界 | 旧提案 |
| `SCALAR-NORM-COMMON` | 固定した同じ$\gamma_j$で再中心化したnorm-only境界 | 分解効果の主対照 |
| `SCALAR-FR-COMMON` | 上と同じ$\gamma_j$で再中心化したFR境界 | Hermitian分離の追加効果 |
| `OPT-SCALAR-NORM` | 同じ許可情報内でscalarを1次元最適化したnorm境界 | 最も強い実用対照 |
| `OPT-SCALAR-FR` | 許す場合に同じ規則で最適化したFR境界 | 方法固有最適化の診断 |
| `INVOLUTION-POLAR` | $h^2=I$でのexact polar factorization | 解析的対照のみ |
| `DENSE-ORACLE` | 真のspectrum、$\rho$、exact scalar/extrema | I2診断のみ |

比較を二層に分ける。

1. **共通$\gamma$比較**：`SCALAR-NORM-COMMON`対`SCALAR-FR-COMMON`で、同じ再中心化後に
   Hermitian/anti-Hermitian分離が追加で与える効果だけを測る。
2. **各法最適化比較**：`OPT-SCALAR-NORM`等で、各法が許可された範囲でscalarを最適化した
   最終性能と情報取得・探索costを測る。

異なる$\gamma$の結果だけを比較して「FR分離の利益」と呼ばない。位相上界だけでなく、
$\Gamma_c/\mathcal B$を含む物理半径、適用不能理由、情報取得costを併記する。

## 8. 先行研究境界

既存FR-0のscoped監査に加え、Gu et al. (PRL 130, 250601, 2023)のrandomized compilingにおける
一次Hermitian perturbationと位相不変性を近接対照とする。従って、一次Hermitian成分が位相へ
直接寄与しないという観察だけは本研究固有としない。

現時点で残す候補差分は、finite paired-Taylor RTEの非一様radial errorについて、正scalar、
finite normalization、非可換interleaving、state-conditioned Hadamard信号を同じ比較契約へ入れ、
利用可能情報ごとに「強いnorm対照で十分な条件」と「FR分離が設定判断を変える条件」を分けることに
限定する。これは新規性の確定ではなく、FR-R1で棄却可能にする仮説である。

## 9. FR-R1の順序と事前登録要件

FR-R0と数値事前登録を同時に凍結しない。順序は次である。

1. 本FR-R0で代数、情報層、baseline、主張境界を固定する。
2. 既存33条件のFR-R1aは`posthoc`と明記し、旧判定を変更しない解析規則を別文書へ固定する。
3. FR-R1aの結果をFR-R1bの閾値調整に使わない。
4. 非一様4×4 FR-R1bの入力、状態、sequence、control、grid、閾値、GO/STOPを別の
   source-hash付き事前登録へ完全に固定する。
5. expected task数、文書末尾、設定digestを検査した後にだけ実装・計算する。

### 9.1 FR-R1bで必ず明記するsequence

候補Hamiltonianを

$$
h_\nu=\frac{1+\nu}{2}Z\otimes I
+\frac{1-\nu}{2}I\otimes Z,
\qquad
\nu\in\{0,1/2,1\}
$$

とする案は維持する。spectrumは$\{1,\nu,-\nu,-1\}$である。ただし数値事前登録では、
少なくとも次を曖昧さなく固定する。

- 共通物理時間$T$、outer step数$q$、$\delta=T/q$。
- exact参照stepの積順序
  $S_\nu(\delta)=e^{-iH_D\delta/2}e^{-ih_\nu\delta}e^{-iH_D\delta/2}$と
  $U_{\nu,q}=S_\nu(\delta)^q$。
- 一つの中央tail exponentialを$r$個のshort stepへ分ける場合の$\tau=\delta/r$、
  numerator、normalization、fresh sampling、発生回数$q r$。
- 負時間、K=4、signal-near-zero controlの具体的$\nu,q,r,K,H_D,state$。これらを
  「代表を少数」だけとして結果後に選ばない。
- 可換/非可換$H_D$、積順序、control branch、relative-phase規約。

### 9.2 状態規約

主状態はsuperpositionであり「混合pure state」と呼ばない。同一$(\nu,H_D)$では状態vectorを
一度だけ構成し、全$q$で同じvectorを再利用する。各$q$の参照eigenbasisから状態を作り直しては
ならない。

事前登録は、固定状態の構成法、縮退部分のcanonical projector/basis規約、global phase規約、
全$q$に共通な$\underline\rho$の保証方法を固定する。dense参照で状態を作る場合は
`provided_state_certificate`またはI2診断と明記し、cheap preparationを示したとは扱わない。
全主条件で$\underline\rho$が保証できなければ、適用不能またはabstentionを正しい出力とする。

## 10. FR-R1bの定量的GO/STOP契約

具体的な位相予算、共通$K/r$候補、数値許容差は結果前のFR-R1b事前登録で固定する。
FR-R1aの既知値の間に予算を後付けしてone-sided successを作らない。

実用的GOには次をすべて要求する。

1. **soundness**：事前登録した全適用点で位相上界と物理半径下界の違反0。
2. **非一様性**：$\nu=0$または$1/2$の非一様条件で成立し、$\nu=1$だけの効果でない。
3. **同情報利益**：I0/I1の同じ情報、同じ$\gamma$で`SCALAR-FR-COMMON`が
   `SCALAR-NORM-COMMON`より厳密に有用。
4. **decision relevance**：事前固定した少なくとも一つの位相予算について片側認証を生むか、
   共通候補集合から選ぶ最小$K/r$を変える。上界比が小さいだけでは不足する。
5. **oracle非依存**：上のGOにI2の真の$\rho$、dense spectrum、結果後scalarを使わない。
6. **機構整合**：radial幅、state weight、非可換性から、改善または失敗の向きを結果前の
   予測と照合できる。

分岐を次とする。

| 結果 | 判断 |
|---|---|
| soundness違反 | `STOP_FR_R_SOUNDNESS`。拡張せず原因を監査 |
| 効果が$\nu=1$だけ、または`OPT-SCALAR-NORM`で全て説明 | `STOP_FR_R_INVOLUTION_OR_SCALAR_ONLY` |
| 非一様系で改善するがI2が必須、または設定判断が不変 | `MECHANISM_ONLY_NO_PRACTICAL_GO` |
| 上の6条件を全て満たす | `GO_FR_R2_CANDIDATE`。ただしFR-R1b後に停止して再設計 |

FR-R1b終了後は判断にかかわらず必ず停止する。同じtoyの$\nu,q,\rho$を増やしてGOを探索せず、
H4へ自動移行しない。

## 11. 現在の完了条件と次の一件

FR-R0の完了条件は次である。

- involutionのexact polar対照と旧比0.6の由来を明示した。
- $\gamma>0$、位相branch、半径、積順序、controlled branchを固定した。
- 共通$\gamma$と各法最適化の二層比較を固定した。
- I0/I1/I2とoracle禁止規則を固定した。
- FR-R1a posthocとFR-R1b preregisteredを分離し、定量的decision relevanceを要求した。

以上を満たすためFR-R0は`COMPLETE`とする。FR-R1aの事後解析計画とFR-R1bの数値事前登録は
別文書として固定し、共通の位相予算を結果前に確定した。次の一件はFR-R1aだけを実装・実行し、
旧FR-1を再分類せず説明監査を完了することである。FR-R1b契約はその結果で変更しない。

## 12. 今は行わないこと

- FR-R1a完了前のFR-R1b実装・4×4計算。
- 旧FR-1のG2失敗や`GO_FR2_MECHANISM_ONLY`の再分類。
- dense真値を実用的certificateとして採用すること。
- H4/H12、P-D S2、長RPE、full compiled総cost、noise/backend。
- 正scalar分離だけを新規アルゴリズムまたは資源優位性として主張すること。

## 13. 参照

- [旧FR-0/FR-1契約](finite_rte_phase_amplitude_contract.md)
- [旧FR-1事前登録](finite_rte_phase_amplitude_fr1_preregistration.md)
- [旧FR-1結果](../finite_rte_phase_amplitude_validation.md)
- [既存scoped先行研究監査](finite_rte_phase_amplitude_prior_art.md)
- [FR-1後の研究方針改訂案](../../fr1_revised_research_plan_20260926.md)
- Günther et al., arXiv:2503.05647 / PRX Quantum 7, 020332 (2026)
- Wan, Berta, Campbell, arXiv:2110.12071 / PRL 129, 030503 (2022)
- Yi and Crosson, npj Quantum Information 8, 37 (2022)
- Li, arXiv:2111.10430 / J. Phys. A 55, 325303 (2022)
- Van der Houwen and Sommeijer, SIAM J. Numer. Anal. 26, 214--229 (1989)
- Papakostas and Tsitouras, SIAM J. Sci. Comput. 21, 747--763 (1999)
- Casares et al., arXiv:2606.30741 (2026)
- Gu et al., PRL 130, 250601 (2023), arXiv:2208.04100
