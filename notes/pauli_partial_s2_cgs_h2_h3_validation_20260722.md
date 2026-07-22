# H2/H3 Pauli 部分ランダム S2 の full-H Cgs 検証

作成日: 2026-07-22
基準 commit: `944ac2391b2fd01be3efbbe74aeedba4910af38b`（検証実装は未 commit の worktree 上）

## 結論

- H2 は 15 個の全 prefix `LD=0..14`、H3 は 62 個の全 prefix `LD=0..61` を評価した。
- `Cgs` は最適化変数ではない。各 `LD` で full-H 対象状態に対する係数 `Cgs(LD)` を先に fit し、その後で `LD, q, kappa` を最適化した。anchor モデルだけは `Cgs(LD)` を中央 prefix の値に固定する。
- `Cgs` がほぼ一定という結果ではない。二次則を満たす正の係数に限ると、`sqrt(Cmax/Cmin)` は H2 で `1.59966`（prepared-state QPE RMSE）または `3.90066`（論文型 ground-energy branch）、H3 で `115.769`（prepared-state QPE RMSE）だった。
- それでも、主 fit 窓内に時間刻みを制限した比較では、anchor 一定モデルと exact-`Cgs(LD)` モデルが選ぶ `LD` は、全ての applicable な `epsilon`、qDRIFT/RTE、誤差指標で一致した。したがって今回の decision regret は全て `0`。
- 標準点 `epsilon_total=1e-4` では、H2 は `LD=14`、H3 は `LD=61`、すなわち両方とも fully deterministic を選んだ。ただし同じ `LD` を選んでも、H2 の総コスト校正値は exact/anchor が `1.06577`（QPE RMSE）または `1.32206`（論文型）であり、anchor はコスト値まで正しくするものではない。
- H3 は対象エネルギーが full Hilbert space で3重縮退するため、単一の ground-energy branch を `Cgs` とする論文型判定は不成立だった。主指標には prepared-state QPE RMSE、保証側には3次元対象空間上の worst-case RMSE を使う必要がある。

## 定義と条件

各 prefix に対して

```text
S2(delta; LD)
  = exp(-i delta H1/2) ... exp(-i delta H_LD/2)
    exp(-i delta H_R)
    exp(-i delta H_LD/2) ... exp(-i delta H1/2)
H_R = identity + sum_{j>LD} H_j
```

を dense 行列で構成し、tail `H_R` は一つの exact block として指数化した。これは tail の有限回ランダム標本を模擬するものではなく、部分ランダム法の coherent S2 誤差係数を分離して測る exact-tail 検証である。

対象状態は full Hamiltonian を指定された粒子数・`Sz` sector 内で対角化して得た。S2 の時間発展と位相解析自体は full Hilbert space で行った。

- H2: 4 qubit、14 非自明 Pauli 項、`LD=0..14`、anchor `LD=7`
- H3: 6 qubit、61 非自明 Pauli 項、`LD=0..61`、anchor `LD=30`
- H2 対象エネルギー: `-1.1011503302326184`（global ground と一致）
- H3 対象エネルギー: `-1.035682613698176`（`N=2, Sz=1` sector）。global ground `-1.568351864512918` は別 sector なので使わない
- 主 fit 窓: `delta = 0.05, 0.075, 0.1, 0.15, 0.2`
- 感度 fit 窓: `delta = 0.025, 0.0375, 0.05, 0.075, 0.1`
- fixed order: 2、自由 fit slope の有効範囲: `2 +/- 0.2`
- cost sweep: `epsilon_total = 1e-2, 1e-3, 1e-4, 1e-5`、quadrature error budget、`q` と `kappa` を最適化
- random cost: `G_rand=1`、qDRIFT と RTE の簡略 prefactor を両方評価
- partial-S2 step cost: `0` (`LD=0`)、`2 LD` (tail あり)、`2 LD-1` (fully deterministic)

prepared-state QPE RMSE は、S2 の固有位相から得る擬エネルギー `E_tilde_j` と full-H 対象状態の重み `w_j` を使って

```text
e_RMSE(LD, delta)
  = sqrt(sum_j w_j (E_tilde_j - E0)^2)
```

とした。各誤差指標 `e` について

```text
Cgs(LD) = geometric_mean_delta [ e(LD, delta) / delta^2 ]
```

を用い、自由 fit slope は二次則の診断だけに使用した。

## Cgs の prefix 依存性

`LD=0` は中央の exact `H_R=H` だけを実行するため `Cgs=0` である。従って、全 prefix を文字どおり含む `sqrt(Cmax/Cmin)` は無限大／未定義になる。次表は、正、noise floor 超、かつ二次 slope を満たす prefix だけの有限 span である。

| 分子・Cgs 定義 | `Cmin` (LD) | `Cmax` (LD) | `sqrt(Cmax/Cmin)` |
|---|---:|---:|---:|
| H2 target-cluster ground energy（論文型） | `2.13109e-4` (5) | `3.24250e-3` (13) | `3.900663` |
| H2 prepared-state QPE RMSE | `4.55742e-3` (5) | `1.16620e-2` (13) | `1.599656` |
| H3 prepared-state QPE RMSE | `2.63821e-8` (2) | `3.53585e-4` (20) | `115.768994` |
| H3 target-energy space worst RMSE | `2.63827e-8` (2) | `3.85848e-3` (3) | `382.427148` |
| H3 target-cluster ground energy（論文型） | 未定義 | 未定義 | 未定義 |

H2 では `LD=1..4` も二次係数ゼロ／noise floor 未満だった。H3 の論文型 target-cluster ground branch は、二次則を示す `LD=20,22` でも最低枝の prepared-state weight が約 `0.5` で、判定閾値 `0.536` を下回る。anchor `LD=30` は slope 約 `4.20`、最低枝の weight は `2.82e-5` であり、二次 `Cgs` の anchor にできない。

## anchor 一定モデルと exact モデル

中央 prefix `L_anchor=floor(R/2)` の係数を全 prefix に使うモデルを `A`、prefix ごとの係数を使うモデルを `E` とした。

```text
G_A(LD) = min_(q,kappa) G(LD; Cgs(L_anchor))
G_E(LD) = min_(q,kappa) G(LD; Cgs(LD))
```

fit 窓より大きい `delta` への `Cgs delta^2` 外挿を避けるため、コスト評価にも `delta <= max(delta_values)` を課した。主窓では `delta_max=0.2` である。decision regret は、anchor が選んだ `LD_A` を exact モデルで再評価して

```text
[G_E(LD_A) - G_E(LD_E)] / G_E(LD_E)
```

とした。

### 標準点 epsilon_total = 1e-4

この点では最適解が fully deterministic なので qDRIFT/RTE は同じになる。

| 分子・Cgs 定義 | anchor/exact 最適 LD | `G_A` | `G_E` | `G_E-G_A` | `G_E/G_A` | regret |
|---|---:|---:|---:|---:|---:|---:|
| H2 target-cluster ground energy | 14 / 14 | 1,874,462.12 | 2,478,160.34 | +603,698.22 | 1.3220648 | 0 |
| H2 prepared-state QPE RMSE | 14 / 14 | 4,409,729.95 | 4,699,762.78 | +290,032.83 | 1.0657711 | 0 |
| H3 prepared-state QPE RMSE | 61 / 61 | 6,050,006.20 | 6,050,006.20 | 0.00 | 1.0000000 | 0 |
| H3 target-energy space worst RMSE | 61 / 61 | 6,110,130.81 | 6,050,006.20 | -60,124.61 | 0.9901598 | 0 |
| H3 target-cluster ground energy | 未定義 | 未定義 | 未定義 | 未定義 | 未定義 | 未定義 |

H3 の `delta=0.2` cap は、この標準点の anchor/exact 双方で active だった。この場合、決定論コストの主因は `A_2(LD)/(epsilon_QPE delta_max)` であり、`Cgs` の差は最適コストに直接は現れにくい。従って、H3 の係数 span が大きいことと、上表のコスト差が小さいことは矛盾しない。

### epsilon sweep の最適 LD

各セルは `anchor / exact`。H2 の二つの usable 指標、および H3 の prepared/worst RMSE で同じ `LD` になった。

| `epsilon_total` | H2 qDRIFT | H2 RTE | H3 qDRIFT | H3 RTE |
|---:|---:|---:|---:|---:|
| `1e-2` | 13 / 13 | 14 / 14 | 50 / 50 | 53 / 53 |
| `1e-3` | 14 / 14 | 14 / 14 | 58 / 58 | 59 / 59 |
| `1e-4` | 14 / 14 | 14 / 14 | 61 / 61 | 61 / 61 |
| `1e-5` | 14 / 14 | 14 / 14 | 61 / 61 | 61 / 61 |

全条件で `LD_A=LD_E` のため decision regret は `0`。つまり、この H2/H3 screening では中央 anchor は最適 `LD` の選択には十分だったが、H2 の絶対コスト校正には不十分だった。

### 決定論・ランダム費の内訳

exact QPE-RMSE モデルの最適点で、主窓の内訳は次の通り。

| 分子・method・epsilon | 最適 LD | `Gdet/Gtotal` | `Grand/Gtotal` |
|---|---:|---:|---:|
| H2 qDRIFT, `1e-2` | 13 | 96.98% | 3.02% |
| H2 RTE, `1e-2` | 14 | 100% | 0% |
| H3 qDRIFT, `1e-2` | 50 | 93.97% | 6.03% |
| H3 RTE, `1e-2` | 53 | 96.52% | 3.48% |
| H3 qDRIFT, `1e-3` | 58 | 97.79% | 2.21% |
| H3 RTE, `1e-3` | 59 | 98.07% | 1.93% |
| H2/H3, `epsilon <= 1e-4` | full | 100% | 0% |

ランダム側の1操作が軽くても、tail cost は `lambda_R(LD)^2 / epsilon_QPE^2` で増える。この簡略モデルでは、精度を厳しくすると tail を残す利点が急速に消え、最適点が full prefix に寄る。その結果、最適点で観測されるランダム費の比率も小さくなる。

## fit 窓感度と数値検査

fit 窓を半分にしたときの有限 span は次のとおりで、`Cgs` の prefix 依存性自体は安定だった。

- H2 論文型: `3.900663 -> 3.900773`
- H2 prepared-state RMSE: `1.599656 -> 1.599642`
- H3 prepared-state RMSE: `115.768994 -> 115.733375`
- H3 worst-case RMSE: `382.427148 -> 382.325994`

一方、`delta_max` も `0.2 -> 0.1` になるため、緩い精度での最適 prefix は動いた。

- `epsilon=1e-2`: H2 qDRIFT `13 -> 12`、H2 RTE `14 -> 13`、H3 qDRIFT `50 -> 45`、H3 RTE `53 -> 50`
- `epsilon=1e-3`: H3 qDRIFT `58 -> 54`、H3 RTE `59 -> 58`（H2 は full のまま）
- `epsilon<=1e-4`: H2/H3 とも full のまま

これは係数 fit の不安定性ではなく、許可した一ステップ時間の上限感度である。従って loose-error の絶対的な最適 `LD` を確定するには、より大きい `delta` まで二次近似が有効かを追加検証する必要がある。`epsilon=1e-4` で fully deterministic という選択は両窓で不変だった。

追加の数値検査:

- full-H 対象状態 residual: H2 `8.92e-16`、H3 `4.44e-16`
- 最大 unitary defect（両 fit 窓）: H2 `3.05e-15`、H3 `4.37e-13`
- 実際の centered-S2 spectrum の log branch-cut 最小余裕: H2 `2.815527145 rad`、H3 `2.618828230 rad`
- target cluster の実測最小 principal overlap: H2 `0.9999998453`、H3 `0.9999999176`
- 全 test suite: `39 passed, 4 warnings`。warning は既存 grouped-UWC 経路の `ComplexWarning`

## 解釈上の制限

- exact-tail phase は RTE の coherent/期待信号側と整合するが、保存 JSON の qDRIFT 結果は既存簡略コスト式の prefactor 感度であり、有限-step qDRIFT channel の bias/variance を直接シミュレーションした結果ではない。
- `G_rand=1` のみを使用した。ランダム操作を10倍・100倍にした感度分析と、fully deterministic に対する speedup は次段階で評価すべきである。
- H3 の単一最大-overlap branch は縮退空間内の基底選択に依存し得るため診断値に留め、主結論には使っていない。

## 再生成

```bash
TMPDIR=/tmp MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba PYTHONPATH=src \
.venv311/bin/python scripts/validate_pauli_partial_cgs_all_prefixes.py \
  --molecules 2,3 \
  --delta-values 0.05,0.075,0.1,0.15,0.2 \
  --epsilon-values 1e-2,1e-3,1e-4,1e-5 \
  --randomized-methods qdrift,rte \
  --noise-floor 1e-12 \
  --cgs-reporting-floor 1e-10 \
  --output artifacts/partial_randomized_pf/diagnostics/H2_H3_s2_full_h_partial_cgs_all_prefixes_20260722.json
```

Raw artifacts（`.gitignore` 対象）:

- `artifacts/partial_randomized_pf/diagnostics/H2_H3_s2_full_h_partial_cgs_all_prefixes_20260722.json`
  - SHA-256: `4b7614f287eb9e1fa51a85695bf853921652dbc4e9cb882970a0ca0b3dd0bc66`
- `artifacts/partial_randomized_pf/diagnostics/H2_H3_s2_full_h_partial_cgs_all_prefixes_half_window_20260722.json`
  - SHA-256: `6c73642a438c719e8354d98618b6f893fb292d0fb4cd2f9238d60156d1edf1cb`

環境: Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、OpenFermion 1.6.1、Qiskit 1.3.0。
