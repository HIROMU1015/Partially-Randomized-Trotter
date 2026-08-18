# Product Formula誤差surrogate・CPU摂動・単一固有位相近似のholdout検証

## 目的と範囲

H4の全非自明分割 $L_D=0,\ldots,11$ について、二次partial-$S_2$ の実PF誤差を
時間刻みgrid上で測定し、既存の`state_specific_phase_bias_surrogate`、
CPU Qiskitによる状態作用、摂動論的誤差、および理想QPE固有位相分布を比較する。

対象はPF演算子誤差、複素信号誤差、位相誤差、QPE分枝分布および経験surrogateの予測精度に
限定する。finite-RTE打切り、shot数、compiled cost、状態準備、ノイズおよび最終
コストは含まない。出力は `final_cost_evaluation_performed = false` に固定する。

追加検証では、元のfull Hamiltonian $H$ の物理基底状態
$|\psi_H\rangle$ に対して

$$
U_{\mathrm{partial}}(\delta)
=S_D^{\mathrm{rev}}(\delta/2)
e^{-iH_R\delta}
S_D(\delta/2)
$$

を作用させる。決定論側の二つの半ステップはQiskit
`Statevector`で実行し、中央の$e^{-iH_R\delta}$は粒子数sector内の
`scipy.linalg.expm`で直接作用させる。ここで$H_R$は`coefficient_atol=10^{-12}`を
超えて保持されたsymbolic tailであり、除外成分は別のthreshold誤差として記録する。
したがって、これは将来のGPU経路と同じ
演算モデルをCPU上で確認するhybrid referenceであり、finite-RTEは含まない。

## Qiskitと摂動エネルギー誤差の比較

Qiskitで得た最終状態から

$$
z(\delta)=e^{iE_H\delta}
\langle\psi_H|U_{\mathrm{partial}}(\delta)|\psi_H\rangle
$$

を作り、位相による誤差

$$
\Delta E_{\mathrm{phase}}=-\frac{\arg z(\delta)}{\delta}
$$

と、状態差を一次摂動として線形化した

$$
\Delta E_{\mathrm{pert}}
=\operatorname{Re}\left[
e^{iE_H\delta}
\frac{\langle\psi_H|(U_{\mathrm{partial}}-e^{-iE_H\delta})
|\psi_H\rangle}{-i\delta}
\right]
=-\frac{\operatorname{Im}z(\delta)}{\delta}
$$

を独立に計算する。後者は厳密な有限$\delta$公式ではなく、
$|z|\simeq1$かつ位相誤差が小さい領域の一次近似である。検証ではQiskitによる
statevectorとsector行列積の一致、両誤差の相対差、およびそれぞれの二次則を確認する。

## calibrationとholdout

surrogateは決定論部分 $H_D$ の基底状態に対するsmall-time survival phaseから

$$
\epsilon_{\mathrm{PF}}^{\mathrm{surr}}(\delta)
=
C_{D}\delta^2
$$

をfitする。calibrationには既存のversioned grid

$$
\{0.01,0.02,0.04,0.08\}
$$

を使う。partial-$S_2$ の検証には、これと重複しないholdout grid

$$
\{0.0125,0.025,0.05,0.1,0.2,0.4\}
$$

を使う。各時間刻みで $q_m\in\{1,2,4\}$ を評価する。予測するround位相誤差は

$$
\beta_{\mathrm{PF},m}^{\mathrm{surr}}
=q_m\delta\,C_D\delta^2
$$

である。

暫定受理条件は次のとおりである。

- 物理基底状態の実energy phase biasのlog-log傾きが1.5以上2.5以下
- surrogate点予測の実値に対する最大相対誤差が25%以下
- 数値的operator-norm位相上界がすべて適用可能で、実位相誤差を覆う
- 実PF位相誤差が暫定配分0.08 rad以下
- 支配固有位相clusterの重みが0.9995以上
- 非支配重みから導く信号位相汚染上界が$8\times10^{-4}$ rad以下
- $q=1,2,4$の信号energy biasが支配固有位相biasと相対2%以内で一致
- full-$H$基底状態の線形化摂動係数が支配固有位相係数と相対2%以内で一致

25%はscreening用点予測の暫定許容差であり、surrogateを厳密上界へ変更する条件
ではない。

## 高速化

H4 Hamiltonianのsector eigensystemとsymbolic tailを1回だけ構築し、6個の
$\delta$ で再利用する。各 $q_m$ は回路を再構築せず、1 step演算子の行列累乗で
評価する。代表runの検証本体は約4秒だった。

## 実行

```bash
.venv311/bin/python scripts/run_pf_delta_validation.py \
  --ld-values 0,1,2,3,4,5,6,7,8,9,10,11
```

このbatchは$L_D=0,1$でscreening用surrogateの条件をrejectするため、12成果物を保存した
後のprocess終了codeは1になる。一方、単一固有位相近似とその摂動係数推定は全12分割で
passする。終了code 1はQiskit・sector行列・摂動計算または単一位相近似の不整合を
表すものではない。

## 2026-08-18の結果

H4 linear chain、距離1.0 Å、STO-3G、DF rank 12、4電子sector、$L_D=3$、
`coefficient_atol=1e-12` を用いた。fit結果は

$$
C_D=0.01163949,
\qquad
p_{\mathrm{free\ fit}}=2.00042
$$

で、fit-window間の係数spreadは約 $3.06\times10^{-4}$ だった。

| $\delta$ | 実energy phase bias（$q=1$） | surrogate | 相対誤差 |
|---:|---:|---:|---:|
| 0.0125 | $2.0844\times10^{-6}$ | $1.8187\times10^{-6}$ | 12.75% |
| 0.025 | $8.3382\times10^{-6}$ | $7.2747\times10^{-6}$ | 12.75% |
| 0.05 | $3.3360\times10^{-5}$ | $2.9099\times10^{-5}$ | 12.77% |
| 0.1 | $1.3355\times10^{-4}$ | $1.1639\times10^{-4}$ | 12.84% |
| 0.2 | $5.3589\times10^{-4}$ | $4.6558\times10^{-4}$ | 13.12% |
| 0.4 | $2.1702\times10^{-3}$ | $1.8623\times10^{-3}$ | 14.19% |

holdout上の実energy phase biasの傾きは2.00397、全18個の物理状態比較における
最大相対誤差は14.19%で、暫定受理条件を満たした。最大実位相誤差は
$\delta=0.4,q=4$ の $3.45\times10^{-3}$ radであり、0.08 rad配分以内だった。

ただしsurrogateは18点すべてで実誤差を過小予測し、最大
`actual / prediction` は1.1653だった。したがって、検証範囲
$0.0125\leq\delta\leq0.4$ におけるscreening用点予測としては採用できるが、
厳密上界または`certified`入力には使用できない。観測結果から1.2倍の経験的margin
を候補として検討できるものの、別の分割・問題インスタンスでholdout検証するまでは
上界とは呼ばない。

## 理想QPE分枝分布

one-step演算子をsector内でSchur分解し、

$$
U_{\mathrm{partial}}(\delta)|\phi_j\rangle
=e^{-i\widetilde E_j\delta}|\phi_j\rangle,
\qquad
w_j=|\langle\phi_j|\psi_H\rangle|^2
$$

から、full-$H$基底エネルギーを基準とする固有位相のenergy shift
$\Delta E_j$を求めた。理想的な無限分解能QPEで得る分布について、

$$
\mu_{\mathrm{QPE}}=\sum_jw_j\Delta E_j,
\qquad
\mathrm{RMSE}_{\mathrm{QPE}}
=\sqrt{\sum_jw_j(\Delta E_j)^2}
$$

を計算した。これは有限ancillaや有限shotによるQPE離散化誤差を含まず、PF stepが
作る固有位相分枝だけを分離して調べる診断である。全固有位相から再構成したsurvival
amplitudeは直接計算と数値誤差内で一致した。

## 全$L_D$検証結果

全12分割、6個の$\delta$、各$\delta$で$q=1,2,4$を評価した。Qiskit statevectorと
sector行列基準の最大状態差は$6.6\times10^{-15}$、位相biasの最大絶対差は
$4.5\times10^{-15}$だった。線形化した摂動誤差と位相誤差の最大相対差は
$3.04\times10^{-6}$で、位相biasのfree-fit slopeは2.00396--2.00436だった。

| $L_D$ | $\lambda_R$ | $C_D$ | $C_{\mathrm{phase}}$ | $C_{\mathrm{peak}}$ | $C_{\mathrm{RMSE}}$ | $C_{p}\;(p_{\mathrm{off}}\simeq C_p\delta^4)$ | 1.2倍$C_D$ |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 13.7237 | 0 | 0.01152474 | 0.01149311 | 0.03878915 | 0.000729833 | 不通過 |
| 1 | 2.862957 | 0.00997435 | 0.01344040 | 0.01340735 | 0.04036042 | 0.000756257 | 不通過 |
| 2 | 1.603096 | 0.01032642 | 0.01340902 | 0.01337595 | 0.04035252 | 0.000757495 | 不通過 |
| 3 | 0.583270 | 0.01163949 | 0.01338993 | 0.01335687 | 0.04034470 | 0.000757307 | 通過 |
| 4 | 0.0871678 | 0.01348773 | 0.01341614 | 0.01338299 | 0.04040358 | 0.000758892 | 通過 |
| 5 | 0.0544047 | 0.01348263 | 0.01341640 | 0.01338325 | 0.04040364 | 0.000758893 | 通過 |
| 6 | 0.0220789 | 0.01309744 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758891 | 通過 |
| 7 | 0.000312954 | 0.01336928 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758891 | 通過 |
| 8 | $8.33\times10^{-5}$ | 0.01336901 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758892 | 通過 |
| 9 | $1.56\times10^{-7}$ | 0.01336958 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758892 | 通過 |
| 10 | 0 | 0.01336958 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758891 | 通過 |
| 11 | 0 | 0.01336958 | 0.01341641 | 0.01338327 | 0.04040362 | 0.000758891 | 通過 |

$L_D=0$の$C_D=0$は、$H_D$だけのcalibration biasが数値noise floor以下となった結果で
あり、全HamiltonianのPF誤差が0という意味ではない。$L_D=10,11$では
`coefficient_atol`後のtailが空であり、恒等tailとして評価した。

survival phase biasは$\mu_{\mathrm{QPE}}$と最大相対差
$8.25\times10^{-4}$で一致した。しかし$C_{\mathrm{RMSE}}/C_{\mathrm{phase}}$は
3.003--3.366であり、survival phaseをQPE RMSEの代理にする25%条件は全分割で
不成立だった。支配分枝の最小重みは0.99998032と大きいが、$\delta=0.4$では
$10^{-5}$程度の重みを持つ$O(1)$のenergy-shift分枝がRMSEへ寄与する。

非支配分枝の確率は全分割でfree-fit slope 4.00380--4.00389となり、
$p_{\mathrm{off}}\simeq C_p\delta^4$でscaleした。支配分枝のenergy biasを
$\Delta E_*$とすると、$q$反復後のtarget-centered信号は

$$
Z_q=e^{-iq\delta\Delta E_*}\{(1-p_{\mathrm{off}})+R_q\},
\qquad |R_q|\leq p_{\mathrm{off}}
$$

と書ける。このため$p_{\mathrm{off}}<1/2$なら

$$
|Z_q|\geq1-2p_{\mathrm{off}},
\qquad
|\arg Z_q+q\delta\Delta E_*|
\leq\arcsin\frac{p_{\mathrm{off}}}{1-p_{\mathrm{off}}}
$$

である。holdout上では、最小の解析的radius下界は0.99996065、最大位相汚染上界は
$1.97\times10^{-5}$ radだった。直接計算した$q=1,2,4$信号でも最小radiusは
0.99997449、支配分枝からの最大位相差は$1.55\times10^{-5}$ rad、energy-bias換算の
最大相対差は1.23%だった。これは暫定位相配分0.08 radの1%として置いた
$8\times10^{-4}$ radより十分小さい。

### 採用する近似評価方針

今回の主評価では単一の支配固有位相を仮定し、PF係数として

$$
C_{\mathrm{PF,eig}}=C_{\mathrm{peak}}\simeq0.01338
$$

を使う。大規模系での推定器にはfull-$H$基底状態に対する線形化摂動係数
$C_{\mathrm{partial}}\simeq0.01342$を使う。両係数の相対差は全分割で0.28%以下だった。
QPE/RPEの有限shot・有限分解能による統計誤差は$C_{\mathrm{PF,eig}}$へ混ぜず、別の
誤差項として扱う。

$C_{\mathrm{RMSE}}\simeq0.0404$と$p_{\mathrm{off}}$は主コスト入力および独立な
「PF失敗確率」から外し、上記受理条件を監視する診断量としてだけ残す。したがって
単一位相条件を満たす間は、非支配分枝用に失敗確率を別配分しない。条件を外れた
インスタンスではこの簡略化を停止し、multi-phase/RMSEモデルへ戻る。

$C_D$を広い$L_D$ screeningに使い、候補点では$C_{\mathrm{partial}}$を再計算する
二段階方針は維持する。この方針はH4 holdoutで支持された経験モデルであり、別問題と
より大きい$q$で再検証するまでは厳密保証ではない。

machine-readable artifactは
`artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld{0,...,11}_v4.json` に保存する。
dirty local worktreeで生成した証拠であり、immutable CI evidenceではない。

| 確認内容 | 結果 |
|---|---:|
| 既知3分枝の解析値、toy holdout、tamper検査 | `3 passed` |
| 12個のartifact schema・fingerprint | 全件成功 |
| 全local test（系サイズ検証追加後） | `442 passed, 4 warnings` |

4 warningは既存の`chemistry_hamiltonian.py`の`ComplexWarning`であり、今回の検証の
失敗ではない。
