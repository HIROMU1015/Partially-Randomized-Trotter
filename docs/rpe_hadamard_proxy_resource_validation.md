# $q=8$ Hadamard 1 shot cost proxy・resource accounting接続検証

## 目的

短roundの直接Hadamard interrogation cost検証は$q=1,2,4$までだった。本検証では、
状態準備を除くHadamard interrogation全体について、短い$q$から求めた1 shot当たりの
compiled-cost proxyが未使用の$q=8$を再現するかを調べる。さらに、holdoutを通過した
固定proxyだけをRPE round accountingへ渡し、

$$
G_m=N_{m,c}g_{m,c}+N_{m,s}g_{m,s}
$$

が二重計上なしに構成されることを確認する。全roundの総和や最終総コストは求めない。

## 条件

- H4 linear chain、距離1.0 Å、STO-3G、8 qubit
- 保存済みDF Hamiltonian、rank 12、$L_D=3$
- $delta=0.1$、$r=4$、$K=2$
- 状態準備なし、ancilla測定を含むcosine/sine Hadamard interrogation
- Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし
- 各$q$で古典Monte Carlo 8 trajectory、cosine/sineは同じtrajectory集合を使用
- proxy較正点$q=1,2,4$、未使用holdout $q=8$
- axis・metric別の一時式 $\widehat C(q)=sq+b$、一様重み
- 受理条件は両軸・全6 cost指標のholdout相対点誤差5%以下、かつRZ入力平均の
  相対標準誤差2%以下

古典Monte Carlo標本数8は、量子測定shot数へ掛けない。8標本は精密な順位付けには小さいが、
今回の各RZ平均の相対標準誤差は最大0.732%で、事前の2%条件を満たした。

## holdout結果

| 指標 | $q=8$一体compile平均 | proxy予測 | 相対誤差 |
|---|---:|---:|---:|
| RZ count | 48462.25 | 48135.08 | 0.675% |
| CX count | 26323.50 | 26222.34 | 0.384% |
| RZ depth | 20438.38 | 20391.08 | 0.231% |
| CX depth | 21506.38 | 21480.07 | 0.122% |
| total depth | 44505.50 | 44391.89 | 0.255% |
| circuit size | 92702.75 | 92029.53 | 0.726% |

このcompiler条件ではcosine/sineの6指標が同じ値になった。最大相対誤差はcircuit sizeの
0.726%で、全12 entryが5%点基準を通過した。holdoutは係数fitに使用していない。

## resource accountingへの接続

providerは、次のすべてを満たす場合だけproxy値を返す。

1. holdout validation全体が通過している。
2. Hamiltonian、DF split、$L_D$、$delta,r,K$、compilerが較正時と一致する。
3. 要求$q$が実際にholdoutで確認した集合に含まれる。

今回は$q=8$だけを許可し、未検証$q=16$は拒否した。暫定配分
$(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.08,0.08,0.24)$ radと、
4 round・2軸を想定した$\alpha_{m,b}=0.05/8$では、解析下界から各軸414 shotとなった。
RZ costについて

$$
414\times48135.0804+414\times48135.0804
=3.9855847\times10^7
$$

となり、resource accountingのround costと一致した。providerの古典標本数は`None`として返し、
較正時の8 trajectoryをshot数へ再度掛けない。

## 判定と限界

このH4 snapshot・分割・時間幅・RTE条件・compilerでは、状態準備なしHadamard interrogationの
$q=8$ 1 shot costに、$q=1,2,4$からのaffine proxyを使用できるlocal evidenceを得た。
また、検証fingerprintに固定したproxyを1 round accountingへ接続できた。

ただし、次は未評価である。

- $q>8$、別$L_D$、別$delta,r,K$、別Hamiltonian snapshotまたは別compilerへの移送
- proxy係数の共分散を含む予測区間
- $q=8$での物理状態上の信号・位相・RPE branch selection
- 状態準備、backend実行、noise
- 複数roundの集計、配分最適化、最終総コスト

したがって、この結果は$q=8$以外への外挿保証でも、部分ランダム化の最終優位性でもない。

## 成果物

`artifacts/rpe_hadamard_proxy_resource_validation/2026-09-01/`に次を保存した。

- `*.dataset.json`: calibration/holdout一体compile dataset
- `*.proxy.json`: calibration-only affine係数
- `*.proxy_validation.json`: 未使用$q=8$の全entry判定
- `*.connection.json`: shot数・1 shot cost・round cost接続と未検証$q$拒否

生成scriptは`scripts/run_rpe_hadamard_proxy_resource_validation.py`、providerは
`src/trotterlib/rpe_hadamard_validated_proxy_provider.py`、専用検証は
`src/trotterlib/rpe_hadamard_proxy_resource_validation.py`である。成果物はlocal
dirty-worktree evidenceであり、immutable CI evidenceではない。

変更後のlocal全test suiteは`478 passed, 4 warnings`だった。4 warningは既存の
grouped-UWC test由来である。
