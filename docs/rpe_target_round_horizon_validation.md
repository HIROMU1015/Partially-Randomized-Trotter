# 目標精度からのRPE round範囲と固定設定の長$q$診断

## 目的

これまで検証した$q=1,2,4,8$の4段が、実際の目標エネルギー精度に十分かを判定した。
採用式は

$$
\frac{\beta_{\rm RPE}}{2^M\delta_{\rm time}}\leq\epsilon_E,
\qquad q_{\max}=2^M
$$

である。正本文書では$\epsilon_E$は外部入力であり、補足資料の$0.50$は架空の数値例である。
今回は既存設定`TARGET_ERROR = CA/10`を暫定主条件として使い、化学精度$CA$も感度比較した。
これは最終的な目標精度の採用決定ではない。

## 条件

- H4鎖、1.0 Å、STO-3G、8量子ビット、DFランク12。
- 保存Hamiltonian snapshot、$L_D=3$、$\delta=0.1$。
- 既存の固定設定$r=4,K=2$、
  $(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})=(0.02,0.02,0.36)$ rad。
- $\beta_{\rm RPE}=0.40$ rad。
- $CA=1.5936001019904\times10^{-3}$ Ha、暫定主条件$CA/10$。

## round範囲

| $\epsilon_E$ | 位置づけ | $M$ | round数 | $q_{\max}$ | 到達分解能 [Ha] |
|---:|---|---:|---:|---:|---:|
| $0.50$ | 補足資料の架空例 | 3 | 4 | 8 | $0.50$ |
| $CA$ | 感度比較 | 12 | 13 | 4096 | $9.765625\times10^{-4}$ |
| $CA/10$ | 暫定主条件 | 15 | 16 | 32768 | $1.220703125\times10^{-4}$ |

したがって、従来の4段検証は架空例のround範囲とは一致するが、化学精度または`CA/10`には
足りない。

## 固定$r=4,K=2$を長$q$へ延ばした診断

大きな一体回路は構築せず、sector内の小規模行列参照で$q=8,4096,32768$を評価した。

| $q$ | attenuation | PF位相誤差 [rad] | PF予算0.02 | finite-RTE位相上界 [rad] | 半径下界 |
|---:|---:|---:|:---:|---:|---:|
| 8 | 0.9932198656 | $1.06819\times10^{-4}$ | 通過 | $6.04570\times10^{-8}$ | 0.9932197667 |
| 4096 | 0.0307074300 | 0.0546607 | 不通過 | $3.09545\times10^{-5}$ | 0.0307064761 |
| 32768 | $7.90584\times10^{-13}$ | 0.437285 | 不通過 | $2.47662\times10^{-4}$ | $7.90388\times10^{-13}$ |

$q=32768$ではPF位相誤差が0.02 rad予算を超え、RTE attenuationもほぼゼロになる。
従って、$q=8$で使った固定$\delta=0.1,r=4,K=2$をそのまま最大roundへ外挿して、
回路コストだけを調べるのは適切でない。

## 判定と次工程

目標精度から必要round範囲を決める段階は完了した。ただし、`CA/10`は既存設定を流用した
暫定値であり、研究上の最終選択ではない。次は、候補$\delta_{\rm time}$ごとに必要$M$を再計算し、
PF予算を満たすかを先に判定する。その後、roundごとに$r_m,K_m$を選び、正の実用的な信号半径を
保つ候補だけにcompiled-cost proxyのholdoutを行う。

この検証では$q>8$回路のコンパイル、fresh-IID shot実行、cost proxyの検証、最終総コスト評価は
行っていない。結果はdirty local worktreeの小規模H4診断であり、immutable CI evidenceではない。

## 後続状況

上で予定した$\delta$とround別$(r_m,K_m)$の再探索は後続検証で実施した。
暫定`CA/10`に対し$\delta=0.01,0.0125,0.02$でH4行列検査を通るscheduleを構成し、
$\delta=0.01,0.02$をcompiled-cost比較のshortlistに残した。詳細は
[$\delta$/round schedule検証](rpe_delta_round_schedule_validation.md)を参照する。

## 成果物

- 結果：`artifacts/rpe_target_round_horizon_validation/2026-09-20/`
- 実装：`src/trotterlib/rpe_target_round_horizon_validation.py`
- 実行：`scripts/run_rpe_target_round_horizon_validation.py`
- テスト：`tests/test_rpe_target_round_horizon_validation.py`
