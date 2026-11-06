# 固定finite signalからPR＋RPE全roundへの接続

2026-11-07 JST。**既存理論・SOURCEとの対応整理のみ。全RPE評価、energy保証、優位性を確立したものではない。**
現在のSCV対象は保存linear H4 / 1.00 Å / STO-3G / legacy DF12 / 8 system qubits /
全4電子sector / $L_D=3$ / $m0,q1,\delta=.02,r1,K2$ / coefficient_atol=0 /
N1 actual finite-grid $Q$・selected full / N2保証state・認証済みexact engine / X,Y。
新round、state、Hamiltonian、lawの評価は行っていない。

## 1. 再利用できる論理と、その前提

研究の[目的][purpose]・[方法][method]は、元Hamiltonianの厳密groundを理論基準に、
所定$\epsilon_E,\alpha_{\rm tot}$で状態準備を除いた全interrogation資源を比較する。
SCVが保存するのは**同じfinite law・同じlogical回路の平均**であり、そこからenergyへの写像には
追加の信号・位相条件が要る。

| 接続 | 再利用できる既存部分 | 追加確認 |
|---|---|---|
| residual→座標 | [calibrate/sufficient_shots][n3]、層別unionと加重誤差 | SCV全層のcal/plan、全range・bias・独立性 |
| X/Y座標→統計位相 | 正半径下界とarcsinの幾何 | actual targetの半径と数値allowance |
| 各round位相→分枝復元 | strict $\beta_{\rm RPE}<\pi/3$、nearest-branch decoder | 同じ基準energyの位相関係、alias-free interval、全round |
| 最終位相→energy | $\beta_{\rm RPE}/(2^M\delta_{\rm time})$ | 同じ精度目標、対象Hamiltonianへのbias |
| shots→compiled資源 | 同scopeの期待単価×N | 条件付き価格、全train/cal、長round価格、返却 |

固定signalの分散低下だけでは上記の残りの条件を満たさない。

## 2. Normalizationと平均保存

round $m$・axis $b$におけるfinite targetを
$z_m^Q=E_{Q_m}[B_{m,X}]+iE_{Q_m}[B_{m,Y}]$と定義する。
readout・時間発展の符号規約は保存SOURCEに従う。SCVは各座標の中心化期待値を0にし、
$E[\widehat z_m^{\rm SCV}\mid\text{frozen history}]=z_m^Q$を目指す。
X/Yの係数が異なっても各軸で正しく中心化すれば同じ複素targetを保存する。

finite-RTEの既知positive attenuationを$A_m^{\rm att}$とすると、
理想化した解析lawではnormalizationを戻した信号は$z_m^Q/A_m^{\rm att}$に対応する。
ただしactual finite-grid law・selected列とその解析lawとの対応は別に確認する必要がある。
SCVはattenuationそのものを取り除く操作ではない。
正の既知scaleは位相を変えないが、信号を割れば座標誤差、range、variance、biasも同じscaleで変換する。
attenuated信号へ既に設定したshot式に再度$1/A^2$を掛けない。

scalar Hamiltonian phase、identity処理、event $(-1)^{n/2}$、ordinary diag(I,U)、
Y readoutの符号、products→rotationの適用順を維持する。
異なるlawの$\nu_h$、nominal連続lawの平均、旧結果の係数を新roundへ代入しない。
平均保存から、PF・finite-RTE・selected-policy biasが0になるとはいえない。

## 3. 基準半径とphase budget

[規範の一般半径モデル][method]は

$$
z_{\star,m}=\rho_{\star,m}e^{-iq_m\theta_\star},\quad q_m=2^m,\quad
t_m=q_m\delta_{\rm time}
$$

という位相関係と正の$\rho_{\star,m}$を別の条件としている。
正当化された$\rho_{\star,m,\rm lb}$と、normalization補正後の誤差$\epsilon_{Z,m}$があれば

$$
\rho_{m,\rm lb}^{\rm obs}
=A_m^{\rm att}(\rho_{\star,m,\rm lb}-\epsilon_{Z,m})>0,\qquad
\beta_{{\rm RTE},m}\leq
\arcsin(\epsilon_{Z,m}/\rho_{\star,m,\rm lb})
$$

を利用できる。actual $Q_m$・selected列への差がこの$\epsilon_{Z,m}$に含まれていなければ、
差を同じnormalization conventionで追加するか、直接の基準信号差に対する上界を別に与える。
未取得biasを0にして二つのtargetを同一視しない。

各軸について、SCVの統計・中心化・engine・算術誤差を合成した最終座標誤差を
$E_{m,X},E_{m,Y}$と書く。両座標事象の成立時に

$$
\sqrt{E_{m,X}^2+E_{m,Y}^2}
\leq \rho_{m,\rm lb}^{\rm obs}\sin\overline\beta_{{\rm stat},m}
$$

を満たすようにshotを決めれば、既存の座標→位相の幾何を使える。
engine誤差を座標headroomで既に差し引いた場合、位相側で同じ誤差を二重加算しない。
PF、RTE、statへ含めたbiasの一覧を持ち、追加のselected/law/state誤差がどの枠にあるか明示する。

共通座標許容値なら既存の形は

$$
\epsilon_{{\rm coord},m}=
\rho_{m,\rm lb}^{\rm obs}\sin\overline\beta_{{\rm stat},m}/\sqrt2 .
$$

SCVではこの許容値に対して層別cal・十分Nを計算する。
複素推定値の偏角そのものは一般に不偏ではないため、平均保存ではなく上の高確率座標事象で制御する。

## 4. 現RPE実装は別の条件付きspecialization

固定560c6092の[rpe_resource_accounting.py][rpe]はversion
finite_rpe_resource_accounting_v3、guarantee scope
effective_partial_s2_eigenstate_unit_radius_alias_free_iid_shots_v2。
**有効deterministic partial-S2のexact eigenstate、非減衰半径1、alias-free energy branch**を仮定する。
元の保存DF Hamiltonianのexact groundを準備できるだけでは、このeigenstate・半径1条件は従わない。
N2のground-state認証もこの相違を解消しない。

evaluate_rpe_round_candidateは$\rho_{\star,\rm lb}=1$を固定し、
$\rho_{\rm obs,lb}=A(1-\epsilon_Z)$、$\beta_{\rm RTE}=\arcsin\epsilon_Z$を計算する。
一般半径を入力するAPI、SCVのcal上界・層別quota、additional finite-law/selected biasの入力は持たない。
現在の精度保証statusをSCVへ移すために、shot欄を手で置き換えるだけでは不十分。

内部のraw Hoeffding式は

$$
N_{m,b}=\left\lceil2\,\epsilon_{{\rm coord},m}^{-2}\log(2/\alpha_{m,b})\right\rceil .
$$

これは幅2のfresh bounded raw outcome用で、幅4まであるresidualや
非IIDな全層混合quotaにそのまま適用しない。
SCVは各層内のfresh IID測定を使い、層別保証を加重して**同じ座標event**を供給する別adapterが必要。
新しい集中不等式を開発する必要はないが、この意味論の接続は未実装である。

## 5. 訂正後の分枝条件・decoder・energy

[RPE会計文書][rpedoc]で確認した条件は次のとおり。

- 数値評価の入力域は$\beta_{\rm RPE}\leq\pi/2$。認証条件は厳格に$\beta_{\rm RPE}<\pi/3$。
  等号も認証不可であり、数値feasibleとcertifiedは異なる。
- 各roundの実際のPF/RTE上界が配分内、配分和が$\beta_{\rm RPE}$以内であることを確認する。
  certificationではfeasibility用$10^{-15}$緩和を証明余裕として使わない。
- empirical PF係数は条件を通ってもempirical_screening。rigorousと外部宣言した入力も、
  半径・state・freshness・threshold・branchの仮定まで実証した意味ではない。
- randomized tailでは各raw shotのcomplete trajectoryをfresh IIDとする。
  一trajectoryを複数shotで使う場合の追加集中誤差は現保証に含まれない。

[reconstruct_rpe_phase][decoder]は、$q=1,2,4,\ldots$が連続することを確認し、
前段推定に最も近いbranchを選ぶ。関数は一般の連続2冪列を受け取るが、
各段で$q_m$候補を列挙するため長round向けの効率は保証していない。
既存の4-round validationの局所検証範囲と、関数の入力可能範囲を区別する。
decoder自体はcalibration・SCV residual・成功確率を検査しない。

同じ基準位相への全round eventとstrict branch条件が成立すれば、
既存の条件付きenergy bound

$$
|\widehat E-E|\leq\frac{\beta_{\rm RPE}}{2^M\delta_{\rm time}}
$$

へ接続できる。alias-freeなenergy intervalに一意な代表があり、phase cutを跨がないことが必要。
現APIにtarget $\epsilon_E$の入力はないので、この上界が$\epsilon_E$以下かは呼出し側で確認する。
保存DFモデルから別Hamiltonianへ比較対象を広げる場合はmodel biasも別途必要。
経験的$C_{\rm use}$をrigorous PF上界へ昇格させない。

## 6. Round別に必要な入力

| 情報 | 固定signalから流用できる原則 | 流用できない実値・条件 |
|---|---|---|
| $Q_m,p_{m,h}$、CDF/grid、support、$r_m,K_m$ | actual lawのexact質量、全正層保持 | m0の一event・5層samplerはn0/n2固定。別r/K/qのlawを扱えない |
| control定義、$S_m$、$\nu_{m,h}$、$\Sigma_{m,h}$ | metadataのみ、実測前固定、条件付き中心化 | m0群や平均の再利用には同一性根拠が必要 |
| X/Y frozen係数・range | trainと評価を分離 | m0の相関や係数品質を長roundへ外挿しない |
| cal上界、誤差・失敗配分、quota | 既存bounded式の層別適用 | 224-shot sample varianceや旧Nの転用不可 |
| state・phase・normalization・bias | mean-preserving補正 | N2/engineのq1全support証明は別roundを覆わない |
| $c_{m,b,h}^{(a)}$とcompiler/scope | 全段階full no-prep価格、hitも課金 | SavedPrices/selected_full_pricesは固定m0/q1用 |
| STOP・返却・来歴 | 成功/失敗履歴を保持 | 旧manager/stream/cache/科学許可の移植不可 |

同じcontrolという記号を使っても、occurrence数やconditionが変わればconditional meanは変わり得る。
round間で係数やcalを共有する案は、同分布・保証範囲の根拠を必要とする別設計。
全roundへ無条件に224-shotずつの予備診断を並べる計画にはしない。

## 7. 全失敗確率と全費用

一方式のsignal familyには、全round・X/Y・正層のcalとprodの失敗予算を含める。
guard/word/worker等の不返却を$\eta_{\rm abort}$とすると、
要求するenergyサービスには

$$
\eta_{\rm abort}+
\sum_{m,b,h}(\delta^{\rm cal}_{m,b,h}+\delta^{\rm prod}_{m,b,h})
+\alpha_{\rm other}\leq\alpha_{\rm tot}
$$

という明示配分が必要。$\alpha_{\rm other}$は、確率的な追加認証がある場合に限る。
決定論的biasを失敗確率へ根拠なく置き換えない。
二方式の同時比較や費用CIには別familyの配分も記録する。
union bound自体は軸間・round間の独立性を要しないが、各cal・prodで使う独立性は必要。

元の資源指標$a$に対する式は、pre費を含めて読む必要がある。
全段階が完了するfrozen計画なら

$$
G^{(a)}_{\rm total}
=F_{\rm all\ rounds}^{(a)}
+\sum_{m,b,h}n_{m,b,h}\,c_{m,b,h}^{(a)}
=F_{\rm all\ rounds}^{(a)}
+\sum_{m,b}N_{m,b}\,\overline c_{m,b}^{(a)},
$$

$$
N_{m,b}=\sum_h n_{m,b,h},\qquad
\overline c_{m,b}^{(a)}
=\sum_h\frac{n_{m,b,h}}{N_{m,b}}c_{m,b,h}^{(a)} .
$$

層別quotaの重みは$p_h$とは一般に違うため、$N$へ無条件$E_{Q_m}C$を掛けない。
さらに$F$には全roundのtrain/probe/calの量子費を含め、
classical setup/compile/engine費は別vectorで記録する。
履歴依存quotaとabortを含むpolicyでは、この式を[総費用文書](SCV_TOTAL_COST_COMPARISON.md)の
条件付き期待値・無条件期待値へ拡張して扱う。

baselineとSCVは同じ$\epsilon_E,\alpha_{\rm tot}$、Hamiltonian/state/normalization、
alias-free window、full価格scope、返却条件で比較する。
baselineにも同じ校正・配分機会を与え、SCVの追加pre費を落とさない。
状態準備除外は研究scopeによるものであり、shot数が異なる方式間で数学的に相殺されるからではない。

build_rpe_resource_summaryは全連続round、同じpreparation/compiler/provider/fingerprint/scope、
軸別alpha和を検査する点を再利用できる。
しかし現dataclassはraw shot式に由来し、cal失敗やpre費・層別価格を持たない。
SCV保証eventと全費用を持つ薄い外側会計が必要であり、旧statusの単純転載はできない。
direct providerの既定$q\leq4$等の構築guardと、検証済みproxyのscopeも維持する。

## 8. 現在の未検証事項

SCVの実H4追加分散削減、独立cal上界、有限confidenceの整数N改善、
train/cal込み期待RZ改善はいずれも今回未検証。
全roundでのcontrolの適用可能性、一般半径、selected/finite-law bias、長round engine・価格、
SCV座標eventを受けるdecoder接続も未検証である。
固定signalで有利という結果が将来得られても、それだけでPR＋RPE全体の優位とは判断しない。

[purpose]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/研究目的・研究課題.md
[method]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/研究方法・解析手順.md
[n3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[rpe]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trotterlib/rpe_resource_accounting.py
[rpedoc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/rpe_resource_accounting.md
[decoder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trotterlib/rpe_four_round_phase_validation.py
