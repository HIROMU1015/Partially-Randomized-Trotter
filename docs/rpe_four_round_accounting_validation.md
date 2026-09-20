# RPE限定4段の集計・失敗確率検証

## 目的と範囲

配分感度検証で選んだ暫定条件を、$q=1,2,4,8$の4段にまとめて使った場合に、
各軸のshot数、1 shot当たりのRZ数、失敗確率予算が同じ資源集計式で整合するかを確かめた。
これは固定条件の**限定的な4段集計**であり、目標エネルギー精度に必要な全roundの選択、
RPE位相復元、分割・時間幅の最適化、最終総コスト評価ではない。

## 入力

- H4鎖、1.0 Å、STO-3G、8量子ビット、DFランク12、保存した同一Hamiltonian snapshot。
- $L_D=3$、$\delta=0.1$、各段$r=4,K=2$、$q=1,2,4,8$。
- $(\beta_{\mathrm{PF}},\beta_{\mathrm{RTE}},\beta_{\mathrm{stat}})=(0.02,0.02,0.36)$ rad、$\beta_{\mathrm{RPE}}=0.40$ rad。
- $\alpha_{\mathrm{tot}}=0.05$。8軸への配分は[配分感度検証](rpe_allocation_sensitivity_validation.md)で選んだコスト感度重み方式。
- $q=1,2,4$の1 shot costは直接Hadamard一体コンパイルの保存平均、$q=8$は未使用holdoutを通過した代理モデル。
- コスト指標はRZ数。Qiskit 1.3.0、`rz,sx,x,cx`、最適化レベル1、seed 17、結合制約なし。
- $q=1,2,4$の物理基底状態上の複素平均信号は[以前の失敗確率検証](rpe_hadamard_failure_validation.md)から固定入力として再利用。

各$q$のコスト推定方法は異なるが、出典を保持した一つの複合providerのfingerprintを構成し、
既存の`build_rpe_resource_summary`に4段すべてを通した。これにより、Hamiltonian、DF分割、
コンパイル条件、回路scope、fresh IID仮定が揃っていることも集計時に検査した。

## 検証した式と結果

$$
G_{4\mathrm{段}}=\sum_{q\in\{1,2,4,8\}}\sum_{b\in\{c,s\}}N_{q,b}g_{q,b},
\qquad \sum_{q,b}\alpha_{q,b}\leq0.05.
$$

| $q$ | 各軸の$\alpha$ | 各軸shot | 1 shot RZ数 | 段のRZ数 | cost出典 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.00171309 | 229 | 6263.875 | $2.8689\times10^6$ | 直接平均 |
| 2 | 0.00336503 | 207 | 12283.25 | $5.0853\times10^6$ | 直接平均 |
| 4 | 0.00659988 | 186 | 24009.5 | $8.9315\times10^6$ | 直接平均 |
| 8 | 0.01332200 | 164 | 48135.080357 | $1.5788\times10^7$ | 検証済み代理モデル |

4段・8軸の合計は**1,572ショット、RZ数32,673,960.607143**だった。
各軸の失敗確率予算の和は0.05で、既存の厳格なunion-bound判定を通過した。
各段および全体でshot数×1 shot RZ数の手計算と資源集計APIの結果が一致した。
古典的なコンパイル標本数を量子shot数へ追加乗算していない。

新しい$\beta_{\mathrm{stat}}=0.36$ radとshot数を使い、以前の$q=1,2,4$の物理信号から
厳密二項確率を再計算した。座標成功なら統計位相誤差が予算内という条件は各離散gridで成立した。

| $q$ | cosine座標失敗率 | sine座標失敗率 | 統計位相失敗率 |
|---:|---:|---:|---:|
| 1 | $5.61\times10^{-24}$ | $1.074\times10^{-4}$ | $3.04\times10^{-8}$ |
| 2 | $1.33\times10^{-11}$ | $1.010\times10^{-4}$ | $4.49\times10^{-8}$ |
| 4 | $1.28\times10^{-5}$ | $1.36\times10^{-6}$ | $2.19\times10^{-9}$ |

独立した3段のいずれかで座標誤差が許容量以上となる厳密確率は
$2.2246\times10^{-4}$、統計位相誤差が0.36 radを超える厳密確率は
$7.7444\times10^{-8}$だった。ともに3段へ割り当てた$\alpha$の合計0.023356以下であり、
各軸・各段でも割当額以下だった。

## 解釈と残る範囲

- **今回確認したこと**：固定した4段の資源集計、8軸の解析的union bound、$q=1,2,4$での新配分の厳密二項確率。
- **経験的な部分**：PF係数は厳密上界でなく、集計APIの保証statusも`empirical_screening`である。
- **未評価**：$q=8$の物理信号・厳密二項失敗率、4段を通したbranch selection／最終位相復元、新配分での明示的なfresh IID trajectoryの再実行、実backend・ノイズ。
- **コストの意味**：32,673,960.607143は固定H4条件の4段RZ集計値であり、最終的なRPE全round総コストでも、他条件に対する優位性でもない。

## 成果物

- 結果：`artifacts/rpe_four_round_accounting_validation/2026-09-18/`
- 実装：`src/trotterlib/rpe_four_round_accounting_validation.py`
- 実行：`scripts/run_rpe_four_round_accounting_validation.py`
- テスト：`tests/test_rpe_four_round_accounting_validation.py`

結果はlocal worktreeの証拠であり、clean checkoutまたはCIでの再生成証拠ではない。
