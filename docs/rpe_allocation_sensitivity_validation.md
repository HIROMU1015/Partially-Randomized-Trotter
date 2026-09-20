# RPE位相誤差・失敗確率配分の感度検証

## 目的と範囲

直近の検証ゴールとして、固定した回路条件において

$$
\beta_{\rm PF}+\beta_{\rm RTE}+\beta_{\rm stat}=\beta_{\rm RPE}=0.40
$$

の配分と、各round・各測定軸への失敗確率$\alpha_{m,b}$の配分が、必要shot数と
RZ comparison costへ与える影響を調べた。この検証は配分感度と暫定配分の選択までを扱い、
$L_D$、$\delta$、$(r_m,K_m)$の探索、RPE位相復元、最終総コスト評価は行わない。

## 固定条件と入力

- H4 chain、1.0 Å、STO-3G、8 qubit、DF rank 12。
- 保存した同一Hamiltonian snapshot、$L_D=3$、$\delta=0.1$、$r=4$、$K=2$。
- Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし。
- $q=1,2,4$は既存の状態準備なしHadamard一体compile平均を固定入力として再利用。
- $q=8$は未使用holdoutを通過した固定affine proxyを使用。
- cost指標はRZ count、$\alpha_{\rm tot}=0.05$。

配分を変えても1 shot回路は変わらない。このため、sweep中に回路の再構築・再compileは行わず、
shot数だけをresource accounting式から再計算した。これにより、配分感度へ新たなMonte Carlo
compile誤差を混ぜていない。

## 比較した配分

| label | $\beta_{\rm PF}$ | $\beta_{\rm RTE}$ | $\beta_{\rm stat}$ |
|---|---:|---:|---:|
| more conservative | 0.12 | 0.12 | 0.16 |
| current provisional | 0.08 | 0.08 | 0.24 |
| moderate | 0.04 | 0.04 | 0.32 |
| guarded provisional | 0.02 | 0.02 | 0.36 |
| tight diagnostic | 0.01 | 0.01 | 0.38 |

$\alpha$は次の二方式を比較した。

1. 一様配分：8軸へ$0.05/8=0.00625$ずつ配る。
2. cost感度重み配分：Hoeffding shot式の連続近似におけるcost感度

$$
w_{m,b}=\frac{2g_{m,b}}{\epsilon_{\rm coord,m}^{2}},\qquad
\alpha_{m,b}=\alpha_{\rm tot}\frac{w_{m,b}}{\sum_{m,b}w_{m,b}}
$$

で配る。$g_{m,b}$は1 shot RZ countである。この式は整数shotを含む厳密な大域最適解ではなく、
高価な軸へ大きい失敗確率を許してshotを減らすための解析的な比較規則である。

## 結果

| $\beta$ label | 一様$\alpha$ comparison cost | cost感度重み comparison cost | 重み配分による減少 |
|---|---:|---:|---:|
| more conservative | $1.6659\times10^8$ | $1.5947\times10^8$ | 4.27% |
| current provisional | $7.4848\times10^7$ | $7.1679\times10^7$ | 4.23% |
| moderate | $4.2781\times10^7$ | $4.0990\times10^7$ | 4.18% |
| guarded provisional | $3.4159\times10^7$ | $3.2674\times10^7$ | 4.35% |
| tight diagnostic | $3.0834\times10^7$ | $2.9490\times10^7$ | 4.36% |

全10 scenarioで、$\beta$和、$\alpha$和、PF・finite-RTE実寄与、正の信号半径、shot式、
`round_cost = N_c g_c + N_s g_s`を検査し、全て通過した。

現行の一様配分に対し、guarded provisionalとcost感度重み配分のcomparison costは
56.35%小さい。内訳を見ると、同じ$\beta=(0.02,0.02,0.36)$のまま$\alpha$だけを変える効果は
4.35%であり、主な感度は$\beta_{\rm stat}$に由来する。

## 暫定採用配分

単一H4条件の小さい実寄与へ予算を過度に合わせないため、PFとRTEの配分が、それぞれ現在の
$q=1,2,4,8$で観測した最大実寄与の100倍以上という暫定guardを置いた。このguardを通る候補の
うちcomparison costが最小だった

$$
(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)
$$

とcost感度重み$\alpha$配分を次段階の暫定入力とする。100倍という値は理論から決まる保証係数ではなく、
一つの代表点へ過適合しないための明示的な運用条件である。tight diagnosticのPF headroomは
93.1倍だったため、より安価でも暫定採用から除外した。採用候補の最小headroomは186.2倍である。

| $q$ | 各軸$\alpha_{m,b}$ | 各軸shot | round RZ comparison cost |
|---:|---:|---:|---:|
| 1 | 0.0017131 | 229 | $2.8689\times10^6$ |
| 2 | 0.0033650 | 207 | $5.0853\times10^6$ |
| 4 | 0.0065999 | 186 | $8.9315\times10^6$ |
| 8 | 0.0133220 | 164 | $1.5788\times10^7$ |

重み配分では安価なroundのshotが増えるため、総shot数は一様配分の1502から1572へ増える。
一方、高価な$q=8$の各軸shotを189から164へ減らすため、RZ comparison costは小さくなる。

## 結論と留保

- 現条件では$\beta$配分がcostに強く影響し、$\alpha$の非一様化は二次的だが有効だった。
- 次段階では$(0.02,0.02,0.36)$と上記$\alpha$を、限定的な4 round集計の暫定入力にする。
- これはH4、固定snapshot、固定$(L_D,\delta,r,K)$、RZ指標だけのlocal choiceである。
- 56.35%は配分候補間のdiagnostic comparisonであり、最終総コスト削減率ではない。
- PF係数は経験的入力であり、rigorousな誤差保証ではない。
- 選択後の非一様$\alpha$について、物理信号を使った厳密二項失敗率または仮想測定を再実行していない。
  今回確認した失敗確率条件は、fresh IID bounded outcome仮定の下でのHoeffding shot式とunion boundである。
- $q=8$の物理信号・位相、RPE branch selection、実backend、noise、$q>8$は未評価である。

machine-readable resultは
`artifacts/rpe_allocation_sensitivity_validation/2026-09-01/`に保存した。
