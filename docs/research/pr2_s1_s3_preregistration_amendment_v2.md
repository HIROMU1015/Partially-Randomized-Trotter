# PR-2 S1--S3 結果前事前登録 amendment v2

日付: 2026-09-28  
状態: `PR2_PREREG_AMENDMENT_V2_FIXED_S0_IMPLEMENTATION_ONLY`  
数値実行: 0（本amendmentでは分子計算、signal評価、compile、trajectory sampling、量子shotを実行していない）

## 1. 文書の位置付け

本書は、次の凍結済みv1に対する**結果前amendment**である。

- v1: [PR-2 S1--S3結果前事前登録](pr2_s1_s3_preregistration.md)
- v1 SHA-256: `9cff2a38071c3779648ad6179679b425e7a2d63618ee28de8b3784898b91b97b`
- v1 dry-run manifest SHA-256:
  `07d1afa9fa9d2ee805573a9201ac482330b45c67a45f420d713bc28d6ea261f7`
- review対象commit: `d3e17239702b56e765ff0a2f8993135015332ea8`
- 外部レビュー: [PR-2 S0/S1前 独立批判レビュー](pr2_s0_s1_external_review_d3e1723.md)
- 外部レビューSHA-256:
  `748c3546fbacf06046fe5f0ee0457e96f7dfb9bb252ea4b20aa0afdc9e06942b`
- 外部判定: `AMEND_BEFORE_S0`

v1、親契約、v1 manifestは上書きしない。本書は、以下で明示する項目だけを置き換える。その他の
Hamiltonian、geometry、rank、時間、finite-RTE grid、compiler、seed、accuracy、禁止事項はv1を維持する。
本書とv1が衝突する場合は本書を優先する。

本amendment後に許可するのは**S0 runner/test/input-freezeの実装**だけである。S0、S1、S2、S3の一括実行、
S1の数値実行、S2/S3への自動進行は許可しない。

## 2. Codexによる外部レビューの再監査

| 指摘 | 判断 | 反映 |
|---|---|---|
| A01 normalization-corrected estimator | 採用 | primaryをcorrected signalへ変更し、既知normalizationをshot overheadへ移す |
| A02 B3名称 | 採用 | `L_D=0 two-body-random endpoint`へ変更し、one-bodyは決定論と明記 |
| A03 state-preparation sensitivity | 採用 | S2/S3のsecondaryとして共通準備costのbreak-evenを固定 |
| A04 cost uncertainty | 修正して採用 | axis別intervalをtotal workへ加算し、ratio intervalとsetting uncertaintyを定義 |
| A05 bugとscientific STOPの分離 | 採用 | implementation block、estimand stop、scientific negativeを分離 |
| A06 B2-G/B2-W decision | 採用 | intermediate partial family $\mathcal B_2$を判定単位とする |
| S1軽量化 | 修正して採用 | canonical 1 trajectory/cellに加え、全distinct primitiveの構造testを要求 |
| O01 order-only診断 | 採用し必須化 | dense $H_D/H_R$ equalityとPF-order差を分離記録 |
| O02 held-out表記 | 採用 | `pre-frozen held-out geometry transfer`へ統一 |
| O03 exact-state scope | 採用 | oracle/state-conditioned subroutine benchmarkと明記 |

A01は一次資料と現行実装の双方に整合する。finite-RTE実装は
`corrected_operator = normalization_product * attenuated_event_mean_operator`を明示している。Günther et al.
のRTEもHadamard raw signalへ既知normalizationを掛けてtarget signalを推定し、その二乗をsampling overheadへ
入れる。従ってv1の「raw meanを補正せずattenuationをsystematic biasに含める」規約はprimaryから外す。

A04の$\bar C\pm2SE$は厳密な95% confidence intervalではない。以後、**engineering materiality interval**
とのみ呼ぶ。adaptive samplingを含むformal coverageは主張しない。

## 3. 修正後の中心主張

> 固定したrank-12 DF H4 Hamiltonianのstate-conditioned coherent-signal benchmarkにおいて、DF二体
> fragmentの中間prefixをdeterministic backboneとして残し、exact residualをfinite-RTEで補完するpartial
> constructionについて、既知normalizationのsampling overhead、finite-cutoff bias、controlled Hadamard
> wrapper、compiled one-shot cost、必要shot数を同一scopeで戻す。discard、rank-12 deterministic、
> weight-ranked ordinary partial、$L_D=0$ two-body-random endpointと比較し、中間partial splitがresource
> frontierに残る条件と、one-step screeningの利得を消す費用項を特定する。

これはH4、固定精度、固定outer-$S_2$ scopeの限定resource studyである。圧縮法、residual randomization、
prefix orderingの新規発明は主張しない。

## 4. estimandとfinite-RTE normalizationの置換

### 4.1 exact targetとstate scope

v1のexact target

$$
z_{12}(T)=\langle\psi_{12}|e^{-iH_{12}T}|\psi_{12}\rangle
$$

を維持する。ここで$H_{12}$は**rank-12 DF Hamiltonian**であり、H12分子を意味しない。
$\psi_{12}$は同じ$H_{12}$のsector ground stateなので、targetは$e^{-iE_0T}$でunit modulusとなる。

これはHamiltonian-simulation subroutineを分離するoracle/state-conditioned benchmarkである。次を評価したとは
主張しない。

- end-to-end state preparation
- unknown-energy discovery
- imperfect guiding stateに対する頑健性
- full RPE reconstruction

### 4.2 raw signal、corrected signal、normalization multiplier

candidate $j$、axis $a\in\{\operatorname{Re},\operatorname{Im}\}$について、Hadamard raw outcomeを

$$
X_{j,a}\in\{-1,+1\},\qquad
\mu^{\rm raw}_{j,a}=\mathbb E[X_{j,a}]
$$

とする。全finite-RTE occurrenceの既知normalization積を

$$
\mathcal B_j
=\prod_o \mathcal B_{j,o}
=\frac{1}{A_j}\ge1
$$

とする。$A_j$は全occurrenceを通したattenuationである。primary estimatorは

$$
Y_{j,a}=\mathcal B_jX_{j,a},\qquad
\nu_{j,a}=\mathbb E[Y_{j,a}]
=\mathcal B_j\mu^{\rm raw}_{j,a}
$$

とする。deterministic候補は$\mathcal B_j=1$である。

有限Taylor cutoff、outer PF、partition、thresholdによるsystematic errorは

$$
b_{j,a}=|\nu_{j,a}-z_{12,a}|
$$

へ残す。既知normalization attenuation自体をprimary systematic biasへ数えない。

次を別fieldで保存する。

- raw mean $\mu^{\rm raw}$
- corrected mean $\nu$
- $\log\mathcal B_j$、$\mathcal B_j$、attenuation $A_j$
- raw mean対targetの差（diagnosticのみ）
- corrected mean対targetのaxis bias
- finite-cutoff、outer PF、discardの分解可能なbias

normalization積とshot数はlog-spaceでも計算し、overflowで候補を有利に丸めない。通常の数値fieldへ収まらない
場合はdecimal stringと$\log_{10}$値を保存し、比較もlog-spaceで行う。

## 5. 共通accuracyと修正後shot式

v1の

$$
\epsilon_{\mathbb C}=0.05,\qquad
\epsilon_{\rm axis}=\frac{0.05}{\sqrt2},\qquad
\alpha_{\rm total}=0.05,\qquad
\alpha_{\rm axis}=0.025
$$

を維持する。precision sweepは行わない。

$$
\epsilon^{\rm stat}_{j,a}=\epsilon_{\rm axis}-b_{j,a}
$$

とし、$\epsilon^{\rm stat}_{j,a}\le0$ならcandidateはaccuracy-ineligibleとする。
$Y_{j,a}\in[-\mathcal B_j,\mathcal B_j]$なので、primary shot countは

$$
N_{j,a}
=\left\lceil
\frac{2\mathcal B_j^2}{(\epsilon^{\rm stat}_{j,a})^2}
\log\frac{2}{\alpha_{\rm axis}}
\right\rceil
$$

とする。二軸のunion boundと$\ell_2$合成により、固定complex error 0.05へ接続する。randomized候補は
Hadamard shotごとにfresh IID trajectoryを引く。fixed-trajectory reuseをprimaryへ混ぜない。

このshot数はexact small-system referenceからbiasを知るoracle-informed retrospective resource accountingで
あり、online algorithmが未知biasを事前に知るという主張ではない。

## 6. baseline名称とintermediate partial family

baselineを次に置換する。

| ID | 内容 |
|---|---|
| B0 | generation-prefix rank-$r$でresidualを捨てるdeterministic $S_2$ |
| B1 | rank-12 deterministic $S_2$ endpoint |
| B2-G | generation-prefix deterministic backbone＋exact residual finite-RTE |
| B2-W | weight-ranked ordinary prefix partial randomization |
| B3 | $L_D=0$ two-body-random endpoint。one-body correctionはdeterministicに保持 |

B3をHamiltonian全体の`full-random`とは呼ばない。本当にone-bodyまでrandomizeする新baselineは今回追加しない。

rank 6の中間partial familyを

$$
\mathcal B_2=\{B2\text{-}G,B2\text{-}W\}
$$

と定義する。S0 identity gateで両者がordered indicesまで一致すれば一候補へ統合する。

## 7. prefix identity gateの追加必須record

v1 §3.2に加え、fragment集合が同じで順序だけ違う場合は、dense small-systemで

$$
\|H_D^G-H_D^W\|_2,
\qquad
\|H_R^G-H_R^W\|_2
$$

を保存し、$10^{-10}\max(1,\|H_{12}\|_2)$以下か判定する。次を別々に記録する。

- split Hamiltonianの同一性
- deterministic $S_2$ application orderの違い
- compiled circuit costの違い
- random-tail component集合・確率の同一性
- component enumeration、hash、seed mappingの実装上の違い

集合またはHamiltonianが異なる場合はB2-G/B2-Wを別candidateとして残す。順序だけの違いも自動的なmethod
deltaとは数えない。

## 8. compiled-cost engineering interval

S2/S3のrandom candidateについて、axis $a$のtrajectory compiled cost
$C_{j,a,1},\ldots,C_{j,a,n}$からmean $\bar C_{j,a}$とstandard error $SE_{j,a}$を計算する。

$$
I^C_{j,a}
=\left[
\max(0,\bar C_{j,a}-2SE_{j,a}),
\bar C_{j,a}+2SE_{j,a}
\right]
$$

をengineering materiality intervalとする。厳密な95% confidence intervalとは呼ばない。

total no-preparation workのpoint estimateとintervalは

$$
G_j^{\rm no\text{-}prep}
=\sum_aN_{j,a}\bar C_{j,a},
$$

$$
I^G_j
=\left[
\sum_aN_{j,a}I^{C,L}_{j,a},
\sum_aN_{j,a}I^{C,U}_{j,a}
\right]
$$

とする。候補A/Bのratio intervalは

$$
I_{A/B}
=\left[
\frac{I^{G,L}_A}{I^{G,U}_B},
\frac{I^{G,U}_A}{I^{G,L}_B}
\right]
$$

とする。分母下端が0の場合は上端を$+\infty$としてmaterialな優位性を認めない。

AがBより10%以上安いと判定する必要十分な本研究内規則は

$$
\sup I_{A/B}<0.9
$$

である。0.9を跨ぐ場合は`MATERIALITY_UNRESOLVED`とする。

各method内ではpoint estimate最小candidateを暫定選択する。暫定選択$s$に対し、別candidate $k$が

$$
\inf I_{k/s}<0.9
$$

となり、sampling uncertainty内で$k$がmaterialに安くなり得る場合は`SETTING_UNCERTAIN`とする。

initial sampleは32 trajectory/cellとする。次のいずれかを満たす関連cellだけ、事前固定した別seed列で
合計128へ一度だけ拡張する。

1. primary RZ meanのrelative standard errorが2%を超える。
2. 32 sample intervalがprimary 10% decisionまたはmethod内`SETTING_UNCERTAIN`を未解決にする。

128後も未解決なら追加sampleを行わない。formal coverage claimをしない。

## 9. state-preparation break-even sensitivity

candidate selectionのprimaryはv1どおりstate preparationなしとする。S2/S3ではsecondaryとして、1 shot当たり
共通state-preparation costをcompiled-RZ相当$P\ge0$で表し、

$$
N_j^{\rm total}=N_{j,\operatorname{Re}}+N_{j,\operatorname{Im}},
$$

$$
G_j(P)=G_j^{\rm no\text{-}prep}+PN_j^{\rm total}
$$

を報告する。hardware依存の$P$を一つ選ばない。

候補A/Bで$N_A^{\rm total}\ne N_B^{\rm total}$なら、ranking equalityの候補点

$$
P^*=
\frac{G_B^{\rm no\text{-}prep}-G_A^{\rm no\text{-}prep}}
{N_A^{\rm total}-N_B^{\rm total}}
$$

を計算し、有限かつ$P^*\ge0$の場合だけnonnegative break-evenとして報告する。shot数が等しい場合、または
$P^*<0$の場合は、その理由と$P\ge0$でrankingが変わるかを明記する。このsensitivityはprimary candidate
selectionを変更しないが、common preparation costが結論を縮める／反転する範囲を明示する。

## 10. S1のcorrectness-only軽量化

S1はresource勝敗を判定しない。次へ置換する。

- development 1.00 Å、$T=\delta=0.1$、$q=1$
- rank 6について固定全grid $r\in\{1,2,4,8,16,32\}$、$K\in\{2,4\}$のraw/corrected mean、
  normalization、bias式を検査
- random full-wrapper compileは各cell・axisで**frozen canonical 1 trajectory**だけを通す
- 全distinct `(basis_id, support, event role, Taylor order)` primitiveは構造testで別途網羅する
- rank 3/9はHamiltonian/residualのstructural correctnessと、固定sentinel $(r,K)=(4,2)$のB2-G/B2-W
  wrapperだけを検査する
- B0 rank 3/6/9、B1 rank 12のdeterministic wrapperを検査する
- 32/128 trajectory expected-cost estimation、resource winner、10%判定、state-prep break-evenはS1で行わない

B2-G/B2-Wを統合する前のfull-wrapper compile上限は、rank-6 random grid 72、rank-3/9 sentinel 8、
deterministic wrapper 8の計88とする。identity collapse後は減らしてよい。distinct primitive testはこの
full-wrapper countへ含めず、S0 manifestで件数を確定する。

S1終了時は次のいずれかだけを返す。

- `S1_CORRECTNESS_PASS_AWAITING_EXTERNAL_REVIEW`
- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

S1からpositive/negative resource conclusionを出さず、S2を自動実行しない。

## 11. implementation blockとscientific STOPの分離

### `BLOCKED_IMPLEMENTATION_INVALID`

code bug、adapter不備、wrapper mapping、normalization/probability test、snapshot serialization、compiler runner
の失敗。scientific negativeとは数えない。Hamiltonian、state、rank、grid、accuracy、compiler、decision ruleを
変更せず、bug修正commit/hash/testを記録してS0から再実行してよい。

### `STOP_ESTIMAND_OR_SCOPE_INVALID`

common estimand自体が比較不能、target/state/taskを変えないと成立しない、または候補間でphysical time、
control semantics、cost scopeを揃えられない場合。これは研究設計の停止とする。

### scientific negative

correctness通過後、固定条件で中間partial familyのresource crossoverが消えた場合。threshold、rank、precision、
state、grid、compilerを結果後に変更して救済しない。

## 12. S2/S3 decision ruleの置換

### 12.1 S2 development判定

rank-6 intermediate partial family $\mathcal B_2$について、corrected estimator、accuracy、engineering intervalを
用いる。

#### `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`

次のいずれかでS3へ進まない。

1. $\mathcal B_2$の全candidateがaccuracy-ineligible。
2. accuracyを満たすB0、B1、B3のいずれかが、$\mathcal B_2$の全eligible candidateよりmaterialに安い。
3. 128 sample後も必要settingが`SETTING_UNCERTAIN`で、development crossoverを固定できない。

#### `S2_CONDITIONAL_RESOURCE_MAP_AWAITING_REVIEW`

B2-GがB2-Wにmaterialに負けるが、B2-Wが中間partial crossoverを示す場合、または中間partialがfrontierへ
残るがB1比10%以上の条件を満たさない場合。generation-prefixの新規成功とは呼ばない。S2後レビューで
`COMPLETE_CONDITIONAL_RESOURCE_MAP`として終了するか、held-out transferへ進むかを明示決定する。

#### `S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`

次を全て満たす場合だけ返す。

1. $\mathcal B_2$の少なくとも一候補がaccuracy-eligible。
2. B1に対するcost ratio interval上端が0.9未満。
3. B0/B3にmaterialに支配されない。
4. method内settingが`SETTING_UNCERTAIN`でない。
5. deterministic saving、$\lambda_R$、normalization、finite bias、sample cost、shot、wrapper overheadの分解が
   整合する。

このstatusでもS3を自動実行しない。

### 12.2 S2後の人手・外部レビュー

S2 artifact、source/test hash、identity result、component breakdownを外部レビューへ戻し、次の一つを明示する。

- `PROCEED_S3_HELD_OUT_TRANSFER`
- `COMPLETE_CONDITIONAL_RESOURCE_MAP`
- `COMPLETE_NEGATIVE_FULL_SCOPE_ERASES_GAIN`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

### 12.3 S3 held-out geometry transfer

H4 1.30 Åは`independent validation`ではなく、**pre-frozen held-out geometry transfer**と呼ぶ。
S2でfreezeしたrank policy、$(r,K)$、accuracy、normalization-corrected estimator、compiler、seed、decision ruleを
変更しない。

- held-out geometryでも§12.1のtransfer-candidate条件を満たす:
  `COMPLETE_POSITIVE_RESOURCE_CROSSOVER`
- developmentでは残るがheld-out geometryで消える、またはmateriality unresolved:
  `COMPLETE_CONDITIONAL_RESOURCE_MAP`

S3後は追加geometry、H12、別分子、precision sweep、長RPE、最終total costへ自動拡張しない。

## 13. stage gate

```text
amendment v2 freeze
  -> STOP
S0 runner/test/input-freeze implementation
  -> S0 artifact + STOP
S1 correctness-only
  -> external review + STOP
explicit PROCEED_S2
  -> S2 development resource comparison + STOP
explicit PROCEED_S3_HELD_OUT_TRANSFER
  -> S3 held-out geometry + terminal decision + STOP
```

本amendmentの次に許される作業はS0 implementationである。S0実行結果を見る前に、runner、test、artifact
schema、non-overwrite guard、source hash、snapshot recipe、identity recordを固定する。S1はまだ未許可である。

## 14. 主張しないこと

- raw attenuated signalを通常RTE resource comparisonのprimaryとすること。
- $\bar C\pm2SE$を厳密な95% confidence intervalとすること。
- B3がone-bodyを含むHamiltonian全体のfull randomizationであること。
- exact-state benchmarkがend-to-end state preparationまたはunknown-energy discoveryを評価すること。
- H4 1.30 Åが別分子・別basis・別system sizeへの外部独立検証であること。
- B2-GがB2-Wに負けたことだけでpartial randomization全体をnegativeとすること。
- S1 correctness passがS2の自動許可またはresource advantageを意味すること。

## 15. 一次資料

1. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*, PRX Quantum **7**,
   020332 (2026), [DOI:10.1103/ynxb-p2xq](https://doi.org/10.1103/ynxb-p2xq)。特にRTEのLCU
   normalization、Hadamard signal、sampling overheadを定めるEqs. (27)--(29)、Appendix A Eqs. (A22)--(A31)。
2. v1および親契約に列挙したSPRINT/GRADE、RC-DF、partial randomizationの一次資料。
