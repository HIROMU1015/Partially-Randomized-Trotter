# finite-RTE 信号近似の小規模検証

## 目的と範囲

最終的な compiled 総コストへ接続する前に、有限 Taylor RTE について次を
独立に検証する。

- normalization 補正後の演算子誤差が Taylor 残差上界以下か
- 物理基底状態と partial-$S_2$ 固有状態で複素信号誤差が同じ上界以下か
- attenuation と保守的な信号半径下界が整合するか
- 複素信号誤差から換算した位相上界が、適用可能な場合に実位相誤差を覆うか
- 暫定位相予算と失敗確率配分の下で、どの $(L_D,r_m,K_m,q_m)$ が実行可能か

この検証は `final_cost_evaluation_performed = false` に固定される。compiled
cost、量子 shot 実行、ノイズ、RPE branch reconstruction、状態準備および
全 round 集計の妥当性を主張しない。

## 方法

小規模系について、Qiskit基準回路とDF basisの変換には8 qubit以下にguardした
full-system密行列を使い、物理比較は粒子数sectorへ制限して行う。独立な RTE
trajectory 列を列挙せず、finite-Taylor operator moment と行列累乗によって期待
演算子を解析的に計算する。したがって、検証時間は trajectory 総数に依存しない。

DF tail の I/Z/ZZ 成分は同じ basis ごとに対角成分を先に合算し、basis 変換の
密行列積を fragment ごとに1回だけ行う。`coefficient_atol` 以下の成分は除外し、
除外係数の $\ell_1$ 和を `threshold_operator_error_bound` として別に保存する。

出力 JSON は schema、生成条件、環境、source hash、Hamiltonian/partition hash、
全 grid 点、判定結果および SHA-256 fingerprint を持つ。loader は改変された
fingerprint を拒否する。

## 暫定条件

代表検証では H4 linear chain、距離 1.0 Å、STO-3G、DF rank 12、4電子 sector
（8 qubit、sector 次元70）を使う。DF rank truncation は0である。

| 項目 | 値 |
|---|---:|
| $\delta_{\mathrm{time}}$ | 0.1 |
| $q_m$ | 1, 2, 4 |
| $r_m$ | 1, 2, 4, 8 |
| $K_m$ | 0, 2, 4, 6 |
| $\beta_{\mathrm{RPE}}$ | 0.40 rad |
| $\overline\beta_{\mathrm{PF}}$ | 0.08 rad |
| $\overline\beta_{\mathrm{RTE}}$ | 0.08 rad |
| $\overline\beta_{\mathrm{stat}}$ | 0.24 rad |
| $\alpha_{\mathrm{tot}}$ | 0.05 |
| validated 6 axes への均等配分 | $\alpha_{m,b}=0.05/6$ |
| `coefficient_atol` | $10^{-12}$ |

## 実行

```bash
.venv311/bin/python scripts/run_finite_rte_signal_validation.py \
  --ld-values 0,1,3,6
```

複数の $L_D$ は同じプロセスで1回だけ生成した Hamiltonian を共有する。この
経路は分子積分と DF 分解の重複を避け、同一 batch 内で Hamiltonian hash が
異なった場合は失敗する。

$L_D=3$の時間刻み感度は次で再現する。

```bash
.venv311/bin/python scripts/run_finite_rte_signal_validation.py \
  --ld 3 \
  --delta-time-values 0.025,0.05,0.1,0.2,0.4 \
  --r-values 1,2,4,8,16 \
  --k-values 0,2,4,6,8
```

## 2026-08-18 の代表結果

各 $L_D$ で48 grid点、2状態、合計96信号を評価した。すべての演算子誤差上界、
信号誤差上界、保守的半径下界、および適用可能な位相誤差上界が成立した。

| $L_D$ | $\lambda_R$ | 最小 attenuation | 暫定位相予算を満たす点 | 最大回避 trajectory 数（$\log_{10}$） |
|---:|---:|---:|---:|---:|
| 0 | 13.7237 | 0.01421 | 32 / 48 | 560.5 |
| 1 | 2.86296 | 0.73062 | 45 / 48 | 548.9 |
| 3 | 0.583270 | 0.98651 | 48 / 48 | 523.8 |
| 6 | 0.0220789 | 0.999981 | 48 / 48 | 470.5 |

$L_D=3$ では最大演算子誤差 $5.21\times10^{-3}$ に対して上界は
$6.96\times10^{-3}$、物理基底状態の最大信号誤差は
$4.97\times10^{-4}$ だった。$L_D=3$ を基準にしたPF誤差surrogateの時間刻み
holdout検証は[別資料](pf_delta_validation.md)に記録する。$L_D=0,1,6$ は分割感度の
対照として保持する。

成果物は `artifacts/finite_rte_signal_validation/` に保存する。これは dirty
local worktree で生成した tamper-evident evidence であり、immutable CI evidence
ではない。また、別プロセスで再生成した DF 表現の生 hash は縮退部分の数値自由度
により変化し得る。$10^{-12}$ threshold 導入後は成分数と判定が一致し、H4 の
基底エネルギーは比較した再生成間で一致したが、canonical DF representation の
構築は今後の再現性課題として残る。

## $L_D=3$の時間刻み・RTE感度

direct $e^{-iH_R\delta}$を中央に置くpartial-$S_2$を基準として、
$\delta\in\{0.025,0.05,0.1,0.2,0.4\}$、$q\in\{1,2,4\}$、
$r\in\{1,2,4,8,16\}$、$K\in\{0,2,4,6,8\}$を評価した。合計375点・750状態で、
Taylor残差による演算子上界、信号上界、半径下界および適用可能な位相上界はすべて
数値許容差内で成立した。

| $\delta$ | 暫定配分内 | 最小attenuation | 最大RTE誤差上界 | 最大実RTE位相誤差（物理基底状態） |
|---:|---:|---:|---:|---:|
| 0.025 | 75 / 75 | 0.999150 | $4.27\times10^{-4}$ | $2.02\times10^{-7}$ |
| 0.05 | 75 / 75 | 0.996605 | $1.72\times10^{-3}$ | $1.62\times10^{-6}$ |
| 0.1 | 75 / 75 | 0.986509 | $6.96\times10^{-3}$ | $1.28\times10^{-5}$ |
| 0.2 | 75 / 75 | 0.947408 | $2.86\times10^{-2}$ | $9.89\times10^{-5}$ |
| 0.4 | 74 / 75 | 0.809430 | $1.23\times10^{-1}$ | $7.18\times10^{-4}$ |

$\delta=0.4,q=4,r=1,K=0$だけが、保守的位相上界0.1235 radのため暫定
$\overline\beta_{\mathrm{RTE}}=0.08$ radを超えた。実位相誤差は
$7.18\times10^{-4}$ radで予算内だが、保証判定には実測値ではなく上界を使う。
同じ$\delta,q$でも、$r=2,K=0$なら上界0.0581 rad、$r=1,K=2$なら
$5.18\times10^{-4}$ radとなり、どちらも暫定配分内に戻る。

この範囲では$K$を0から2へ上げる効果が大きく、$r$だけを増やすより少ない変更で
上界を縮められる。ただし回路コストはまだ評価していないため、$r=2,K=0$と
$r=1,K=2$のどちらが有利かは現段階では決めない。
