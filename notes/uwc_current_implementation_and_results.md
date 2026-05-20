# UWC 実装の現状と検証結果

作成日: 2026-05-20

## 要約

現状の UWC は、Hamiltonian 生成後、Pauli term のソートや `L_D` split に入る前の **Hamiltonian preprocessor** として実装されている。Trotter 展開や product formula の内部には UWC 固有の分岐を入れていない。

grouped Hamiltonian + PF + QPE との比較パイプラインも追加済みで、UWC 後 Hamiltonian に対して同じ grouping 方針を適用し直し、Trotter error coefficient `alpha`、1 step Pauli rotations、RZ layers、T-depth proxy、QPE iteration factor、total cost を計算する。

ただし、現在の BLISS 型 UWC は最小実装に近く、今回試した `theta=0.01`, quadratic shift では実質的な cost reduction はほぼ出ていない。H2/H4/H5/H6 は cost ratio がほぼ 1、H3 も sector 修正後は約 0.35% 改善に留まる。

## 実装ファイル

主な追加・変更箇所は以下。

| file | 役割 |
|---|---|
| `src/trotterlib/uwc.py` | UWC の設定、Hamiltonian preprocessing、BLISS shift、metadata/hash/metric 計算 |
| `src/trotterlib/partial_randomized_pf.py` | 通常の Pauli PR-PF pipeline に UWC preprocessor を接続 |
| `src/trotterlib/grouped_uwc_comparison.py` | grouped baseline と UWC grouped の PF+QPE 比較、GPU/CPU alpha fit |
| `scripts/run_partial_randomized_pf.py` | 通常 PR-PF 用 UWC CLI オプション |
| `scripts/run_grouped_uwc_pf_qpe.py` | 1 molecule の grouped baseline vs UWC grouped 比較 CLI |
| `scripts/run_uwc_grouped_alpha_h2_h6.py` | H2-H6 の UWC grouped alpha/cost batch 実行 CLI |
| `tests/test_uwc_preprocessor.py` | UWC preprocessor の単体テスト |
| `tests/test_grouped_uwc_comparison.py` | grouped UWC 比較、GPU runner 経路、sector 推定のテスト |

## UWCConfig

UWC 関連の設定は `UWCConfig` にまとめている。

| field | 意味 |
|---|---|
| `enabled` | UWC を使うかどうか。default は `False` |
| `method` | `none`, `simple_shift`, `test_shift`, `bliss`, `orbital_optimization`, `orbital_optimization_bliss` |
| `objective` | `l1_norm`, `lambda_r`, `estimated_total_cost` |
| `target_ld` | UWC 最適化や metric 計算に使う target `L_D` |
| `optimizer_settings` | optimizer 用の設定。例: theta bounds |
| `max_iterations` | optimizer の最大反復数 |
| `seed` | 乱数 seed |
| `use_cache` | UWC 関連 cache を使うかどうか |
| `parameters` | method 固有のパラメータ。例: `theta`, `power`, `target_particle_number` |
| `sector_energy_tolerance` | BLISS sector-preserving check の許容誤差 |
| `sector_energy_check` | `warn`, `error`, `off` |
| `max_sector_dimension_for_check` | 明示的 diagonalization で sector energy check する最大次元 |

## 実装済み UWC method

### `none`

入力 Hamiltonian をそのまま返す。metadata には `preprocessor = "none"` を保存する。

### `simple_shift` / `test_shift`

実験用の簡単な変換。

- `coefficient_scale` が指定されれば非 identity term の係数を一様 scale
- `shift` が指定されれば指定 qubit の `Z` 項を追加

これは pipeline 接続確認用であり、sector-preserving な物理的 UWC ではない。

### `bliss`

現在の主な UWC 実装。粒子数 sector でゼロになる shift を Hamiltonian に足す。

```text
H_UWC = H + K(theta)
K(theta) = theta * (N - N_target)^power
```

ここで `N` は Jordan-Wigner の粒子数演算子。

```text
N = sum_i (I - Z_i) / 2
```

`power` は `linear` または `quadratic`。今回の検証では主に `quadratic` を使った。

`theta` を明示しない場合、`max_iterations > 0` なら `l1_norm` または `lambda_r` objective で 1D scalar optimization を行う実装がある。ただし `estimated_total_cost` objective はまだ予約扱いで、現時点では未実装。

### `orbital_optimization` / `orbital_optimization_bliss`

interface 上は method として予約しているが、現時点では `NotImplementedError`。

## UWC metadata

`preprocess_qubit_hamiltonian` は、変換後 Hamiltonian とあわせて以下のような metadata を返す。

| key | 内容 |
|---|---|
| `preprocessor` | `"uwc"` or `"none"` |
| `uwc_method` | UWC method |
| `uwc_objective` | objective |
| `uwc_target_ld` / `target_ld` | metric 用 target `L_D` |
| `uwc_parameters` | 実際に使われた parameter |
| `uwc_config` | normalized `UWCConfig` |
| `original_hamiltonian_hash` | UWC 前 Hamiltonian hash |
| `uwc_hamiltonian_hash` | UWC 後 Hamiltonian hash |
| `original_l1_norm`, `uwc_l1_norm` | L1 norm |
| `original_num_terms`, `uwc_num_terms` | Pauli term 数 |
| `original_lambda_r_at_target_ld`, `uwc_lambda_r_at_target_ld` | target `L_D` での lambda_R |
| `sector_preservation_check` | BLISS sector check 結果 |

Hamiltonian hash は term list と係数を反映するため、UWC 後の C_gs fit や結果 cache が通常 Hamiltonian と混ざらないようになっている。

## 通常 Pauli PR-PF pipeline への接続

通常 PR-PF では次の流れにした。

```text
jw_hamiltonian_maker
  -> preprocess_qubit_hamiltonian
  -> Pauli term sort
  -> L_D split
  -> C_gs fit
  -> q/kappa optimization
  -> result save
```

UWC の挿入点は `build_preprocessed_sorted_pauli_hamiltonian` のみ。以降は通常の `SortedPauliHamiltonian` として処理するため、product formula や Trotter 展開には UWC 固有処理を入れていない。

`PartialRandomizedStudyResult` には UWC metadata を追加した。UWC なしでも `preprocessor = "none"` が入る。

## grouped Hamiltonian + PF + QPE 比較

ユーザー指定に合わせて、比較基準は non-grouped original Hamiltonian ではなく、既存の grouped Hamiltonian + PF + QPE とした。

比較は以下の 2 行を作る。

```text
1. grouped_baseline
2. uwc_grouped
```

`grouped_baseline` は既存の grouped artifacts と既存 cost table を基準にする。

`uwc_grouped` は次の処理を行う。

```text
baseline grouped Hamiltonian
  -> UWC preprocessing
  -> delta = H_UWC - H_original
  -> delta terms を既存 clique に再配置
  -> 可能なら同じ grouping rule を維持
  -> UWC grouped Trotter circuit で alpha fit
  -> UWC grouped cliques から step Pauli rotations を再計算
  -> UWC grouped cliques から step RZ layers を再計算
  -> QPE factor と total cost を計算
```

重要なのは、UWC 側の `A_step` は baseline の値をそのまま流用していないこと。`uwc_cliques` から `step_pauli_rotations` と `step_rz_layers` を再計算している。

ただし今回の BLISS quadratic shift では、結果として term set と group sizes が変わらなかったため、step cost は baseline と同じになった。

## alpha fit

grouped UWC の `alpha` は grouped Trotter 回路で time evolution し、保存済み/計算済みの参照 ground state と比較して perturbative error を出し、固定 order の係数として fit している。

2nd order の場合は概念的に次。

```text
error(t) ~= alpha * t^2
alpha = average_t error(t) / t^2
```

実装では既存の `loglog_average_coeff` を使い、PF order を固定して coefficient を取る。

GPU backend では:

- grouped cliques から `PauliEvolutionGate` を含む回路を作る
- GPU backend に渡せるよう標準ゲートへ分解
- time grid を GPU ids に round-robin 割り当て
- `df_gpu_statevector.simulate_statevector_gpu` を利用

`alpha_backend` は `cpu`, `gpu`, `auto` を選べる。

## H3 sector 問題と修正

一度、H3 だけ `alpha ratio = 4.36` という大きな悪化が出た。

原因は UWC そのものではなく、BLISS の `N_target` 推定ミスだった。

H3 は以下の条件。

```text
ham_name = H3_sto-3g_triplet_1+_distance_100_charge_1
charge = 1
multiplicity = 3
```

以前は `N_target = Hn - charge = 2` と推定していた。しかし grouped baseline の参照状態は粒子数 `N = 3` sector にある。

この状態で

```text
(N - N_target)^2
```

を足すと、対象状態上で shift がゼロにならず、別 Hamiltonian 的な評価になっていた。そのため H3 の alpha が不自然に大きくなった。

修正後は、BLISS の `target_particle_number` 未指定時に charge から推定せず、**grouped reference state から粒子数 sector を推定**する。

H3 の修正後診断:

| check | value |
|---|---:|
| inferred `N_target` | 3 |
| `<N>` | 3.0000000000000004 |
| `Var(N)` | 約 0 |
| max shift on sector | 1.73e-18 |
| ground energy difference | 8.88e-16 |

これにより H3 の大きな悪化は消えた。

## H2-H6 grouped UWC 結果

実行条件:

```text
method = bliss
power = quadratic
theta = 0.01
pf_label = 2nd
target_error = 1e-2
cost_metric = rz_layers
alpha_backend = gpu
gpu_ids = 0,1
baseline_alpha_source = artifact
```

結果ファイル:

```text
artifacts/grouped_uwc_pf_qpe/H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json
```

| molecule | method | alpha | qpe factor | step rot | total rot | step RZ | total RZ | cost ratio | alpha ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2 | grouped_baseline | 3.355907245e-03 | 1.806085e+02 | 24 | 4.334603e+03 | 9 | 1.625476e+03 | 1.000000000 | 1.000000000 |
| H2 | uwc_grouped | 3.355475852e-03 | 1.805969e+02 | 24 | 4.334325e+03 | 9 | 1.625372e+03 | 0.999935724 | 0.999871453 |
| H3 | grouped_baseline | 5.277914009e-03 | 2.264980e+02 | 118 | 2.672676e+04 | 39 | 8.833420e+03 | 1.000000000 | 1.000000000 |
| H3 | uwc_grouped | 5.241362013e-03 | 2.257123e+02 | 118 | 2.663405e+04 | 39 | 8.802779e+03 | 0.996531253 | 0.993074537 |
| H4 | grouped_baseline | 1.139543324e-02 | 3.328117e+02 | 396 | 1.317934e+05 | 99 | 3.294836e+04 | 1.000000000 | 1.000000000 |
| H4 | uwc_grouped | 1.139397694e-02 | 3.327904e+02 | 396 | 1.317850e+05 | 99 | 3.294625e+04 | 0.999936099 | 0.999872203 |
| H5 | grouped_baseline | 5.542025349e-03 | 2.320959e+02 | 998 | 2.316317e+05 | 341 | 7.914469e+04 | 1.000000000 | 1.000000000 |
| H5 | uwc_grouped | 5.541274891e-03 | 2.320801e+02 | 998 | 2.316160e+05 | 341 | 7.913933e+04 | 0.999932292 | 0.999864588 |
| H6 | grouped_baseline | 1.987151567e-02 | 4.394896e+02 | 2116 | 9.299601e+05 | 568 | 2.496301e+05 | 1.000000000 | 1.000000000 |
| H6 | uwc_grouped | 1.986983719e-02 | 4.394711e+02 | 2116 | 9.299208e+05 | 568 | 2.496196e+05 | 0.999957766 | 0.999915533 |

## sector check と group 構造

| molecule | N_target | source | sector energy diff | num groups base/UWC | num terms base/UWC |
|---|---:|---|---:|---:|---:|
| H2 | 2 | grouped_reference_state | 0.000e+00 | 2/2 | 14/14 |
| H3 | 3 | grouped_reference_state | 0.000e+00 | 7/7 | 61/61 |
| H4 | 4 | grouped_reference_state | 4.441e-16 | 13/13 | 184/184 |
| H5 | 4 | grouped_reference_state | 0.000e+00 | 43/43 | 443/443 |
| H6 | 6 | grouped_reference_state | explicit diagonalization skipped; algebraic zero shift | 57/57 | 918/918 |

H6 は target number sector dimension が大きいため、明示的 diagonalization ではなく、number-sector shift が対象 sector で algebraically zero になることを使った check になっている。

今回の BLISS quadratic shift では、H2-H6 すべてで group 数と Pauli term 数は baseline と同じだった。したがって step rotations と step RZ layers が同じなのは自然。

## H3 time-grid error 診断

H3 の修正後診断ファイル:

```text
artifacts/grouped_uwc_pf_qpe/H3_bliss_sector_scaling_diagnostics.json
```

CPU statevector 診断では、time grid ごとの error は以下。

| t | baseline error | UWC error | ratio |
|---:|---:|---:|---:|
| 0.750 | 2.940424083e-03 | 2.941835210e-03 | 1.000479906 |
| 0.752 | 2.957473768e-03 | 2.958916906e-03 | 1.000487963 |
| 0.754 | 2.974589756e-03 | 2.976065619e-03 | 1.000496157 |
| 0.756 | 2.991772374e-03 | 2.993281692e-03 | 1.000504490 |

修正前のような 4 倍悪化は消えている。

H3 の slope:

| fit | slope |
|---|---:|
| baseline | 2.1727 |
| UWC | 2.1757 |

2nd order として妥当。

## H3 theta sweep

H3 で `theta` を変えた診断。表の ratio は grouped artifact baseline を分母にしたもの。

| theta | alpha | alpha ratio | cost ratio | slope | coeff spread |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.262981616e-03 | 0.997170778 | 0.998584387 | 2.171287 | 1.364855e-03 |
| 0.001 | 5.308071558e-03 | 1.005713914 | 1.002852888 | 2.169683 | 1.352074e-03 |
| 0.005 | 5.286616702e-03 | 1.001648889 | 1.000824105 | 2.171846 | 1.369312e-03 |
| 0.01 | 5.299857551e-03 | 1.004157616 | 1.002076652 | 2.172881 | 1.377557e-03 |
| 0.05 | 5.246896582e-03 | 0.994123165 | 0.997057253 | 2.189495 | 1.509940e-03 |
| 0.1 | 5.271755943e-03 | 0.998833239 | 0.999416449 | 2.210723 | 1.679095e-03 |

同じ run 内で baseline alpha も fit した fit-to-fit 比較では、`theta=0` は完全に一致する。

```text
theta = 0:
baseline_alpha_fit = 0.005231009416394276
uwc_alpha_fit      = 0.005231009416394276
alpha_ratio        = 1.0
cost_ratio         = 1.0
```

artifact baseline を分母にした theta sweep では、artifact と再fit alpha の差があるため、`theta=0` でも ratio が厳密に 1 にはならない。

## 結果の解釈

### cost ratio の式は整合している

2nd order では、step cost が同じなら total cost ratio は概ね以下。

```text
sqrt(alpha_UWC / alpha_baseline)
```

H2/H4/H5/H6 で cost ratio が `0.99993` 程度になるのは、alpha ratio が `0.99987` 程度であることと整合する。

### 現状の UWC では step cost が下がっていない

今回の BLISS quadratic shift では、group 数、Pauli term 数、group sizes が変わらない。したがって:

```text
step_pauli_rotations: unchanged
step_rz_layers: unchanged
```

total cost の改善は alpha の微小変化だけから来ている。

### 現状の UWC は実質的にはほぼ cost reduction していない

H2/H4/H5/H6 は改善が 0.01% 未満。H3 は sector 修正後で約 0.35% 改善だが、まだ「有意な UWC cost reduction」と呼ぶには弱い。

現在の結論:

```text
UWC pipeline は動いている。
sector-preserving 性と alpha scaling は確認できた。
ただし現在の simple BLISS quadratic shift では、grouped PF+QPE cost はほぼ下がらない。
```

## simple_shift 結果について

実験用 `simple_shift=0.01` でも H2-H6 を流した。

結果ファイル:

```text
artifacts/grouped_uwc_pf_qpe/H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json
```

これは sector-preserving ではないため、物理的な UWC 比較としては採用しない。H4-H6 で alpha が大きく悪化しており、pipeline 接続確認用の結果と見るべき。

## テスト状況

現時点で以下を確認済み。

```text
.venv/bin/python -m pytest -q
26 passed
```

主なテスト内容:

- UWC なしで Hamiltonian term list が変わらない
- UWC metadata が JSON に保存される
- UWC 後 Hamiltonian hash が通常 Hamiltonian と分離される
- lambda_R が target `L_D` で計算できる
- BLISS sector-preserving shift で ground energy が許容誤差内で一致する
- grouped baseline が grouped table を使う
- UWC grouped で step cost を再計算する
- GPU alpha runner 経路が呼ばれる
- BLISS `target_particle_number` が grouped reference state から推定される

## 現状の制限

1. BLISS shift の自由度が小さい。
   現在は主に `theta * (N - N_target)^power` の 1 パラメータ。

2. grouped cost を直接最小化していない。
   `estimated_total_cost` objective は interface 上はあるが未実装。

3. UWC 後の grouping は完全な再最適化ではない。
   現在は UWC delta terms を既存 clique に可能な範囲で再配置している。grouping rule を本格的に再実行する実装はまだ。

4. orbital optimization は未実装。

5. GPU 環境依存がある。
   `qiskit-aer-gpu` の CUDA ライブラリ解決が環境によって不安定になることがある。alpha の数式診断は CPU statevector でも実行可能。

## 次にやるべきこと

優先度順:

1. grouped PF+QPE total cost を objective にした UWC theta optimization
2. `theta` の自動 sweep と結果比較 table の標準出力化
3. UWC 後 Hamiltonian に対する grouping の再最適化
4. BLISS shift の自由度追加
5. spin/Sz sector-preserving shift の追加
6. orbital optimization + BLISS
7. DF Hamiltonian 側への拡張

