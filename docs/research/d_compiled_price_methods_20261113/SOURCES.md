# SOURCEと人工検証の再現

2026-11-13 JST。調査基点は`39ddd59edf54a3a901e5302207b2765ddbd03fbb`。
同branchを再利用し、以下の既存科学SOURCEは変更していない。

## Project SOURCE

| 固定commit | 読解・再利用した主なpath / API |
|---|---|
| `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | `docs/research/d_p2_direct_pilot_20261113/REPORT.md`, `HANDOFF.md`、`src/trottertracks/p2_direct_pilot.py`、`scripts/run_p2_direct_pilot.py`。保存科学数値JSON/CSV/rawの新解析なし |
| `5a67cea7705e80ff295306606611d774724ae2e2` | `src/trottertracks/compiled_price_fast.py`: price_accounting / structural_pair / make_backend_class。元C1 compose・wrapper・fingerprint/compileを再利用 |
| `54b64ef72bf507be25970f2d3362d3be091321c3` | `src/trottertracks/p1_efficient_zz.py`、`efficient_controlled_zz.py`、`h4_df10_price_v1.py`: 元Accounting / tail / predict。P1 REPORTを引用、再fitなし |
| `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | `src/trottertracks/h4_structural_kernel.py`: Model / Wire / Band、`h4_structural_schedule.py`、`h4_pr_physical_closure.py`。Cの全26 E[S]の再利用条件をSOURCEと既存HANDOFFから確認 |
| `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` | `src/trottertracks/r_price_phase1_science.py`: Backend.price_requests。C1のS/C、scalar、wrapper、compiler/cache条件 |
| `155529e20a8fac39e00786c40b57830b71203c4f` | `scripts/h46_long_time_native.py`: trial_seed / sampler / PR compose |
| provider `c31ef424` | C1が固定する`src/trotterlib/df_rte_qiskit.py`、native PF/wrapper、`rte_compiled_cost.py`: transpile_and_measure_cost |
| `af6f27d7a6467bb355686421acd067544edcbbd2`由来、基点39ddd59edの保存版 | `src/trottertracks/compiled_residual.py`: paired_residuals / cohort_summary / identifiability / combine。`mean_compiled_price.py`: MeanKernel / full_wrapper_mean / combine_mean |
| 基点39ddd59edの保存版 | `src/trottertracks/track_d_native_emitter.py`、`track_d_p0_1q_bridge.py`、`track_d_p0_cx_bridge.py`、`track_d_stream_bridge_fast.py`、`track_d_bridge_run_reuse.py`。限定規則の再利用候補、一般compiler parityはP0_UNQUALIFIED |

sparse checkoutに現れないC等のSOURCEは`git show <commit>:<path>`で読んだ。
既存全26 E[S]はCの`docs/research/h4_structural_schedule_results_20261113/HANDOFF.md`を参照し、
今回新しくmean・分散・係数・価格は計算していない。旧Dの比較案と局所clusterの議論は
[`d_compiled_residual_c_minus_s_20261113/REPORT.md`](../d_compiled_residual_c_minus_s_20261113/REPORT.md)を再利用した。

## Qiskit 1.3.0の一次SOURCE

インストール環境：`/home/AbeHiromu/projects/h6-environments/20261011/venv/`、
Python3.11.0rc1 / Qiskit1.3.0 / NumPy1.26.4。共有環境は変更していない。

| 一次SOURCE | 確認点 |
|---|---|
| [level1.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/level1.py)・[builtin_plugins.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py) | 設定に応じたinit/translation/optimization stage、1q→inverse cancellationと固定点loop。インストールされた.pyを実際に読解 |
| [optimize_1q_decomposition.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py) | target=Noneのdecomposer選択とRust呼出し。インストールされた.pyを読解 |
| [euler_one_qubit_decomposer.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/euler_one_qubit_decomposer.rs) | 公式tagをGitHub connectorで取得。blob `eb53b8309b05ee05ded5fcdcfd38958e8a7f8514`。ANGLE_ZERO_EPSILON=1e−12、targetなしのgate数error、行列積、再合成・置換条件（約1055–1239行） |
| [inverse_cancellation.rs](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs) | 公式tagから取得。blob `9664f52089dcbbce048da06301cd4b84b2e30148`。parameter比較tolerance=1e−10、同gate/qargs runの偶数削除、逆pair消去 |
| [flow_controllers.py](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/passmanager/flow_controllers.py) | loopにはmax_iteration（既定1000）があり、混在mirrorの無限反復が実行されるとは主張しない |

同設定のpublic generate_preset_pass_managerを回路なしで生成して、target=None、ZSXX、
layout/routingなし、実stageを確認した。公式Rust SOURCEとインストール版・人工testの照合であり、
インストールbinaryの完全なbuild provenanceまで証明したものではない。

## 追加実装と再現手順

追加は[`outer_window_residual_design.py`](../../../src/trottertracks/outer_window_residual_design.py)と
[`test_outer_window_residual_design.py`](../../../tests/test_outer_window_residual_design.py)。
前者は与えられた数の差分・stencil・共分散・条件付きtail和だけを扱い、I/O・sampling・
compiler・fit・価格providerを呼ばない。既存MeanKernelを再実装しない。
全ての呼出し入力は人工値で、qualificationは常にfalse、未知tail/law価格はnullのまま。

worktree rootから、次の2つを**別process**で実行する（残差回帰のpassive-import検査のため）。

```bash
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/test_outer_window_residual_design.py tests/check_mean_compiled_price.py
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/check_compiled_residual.py
```

[29 PASS](TESTS.txt)＋[13 PASS](TEST_RESIDUAL_REGRESSION.txt)＝42項目。
新14項目は非局所costでの恒等式、共通cap、h=q、未知共分散、PSD拒否、条件付き幾何和、
既存MeanKernelのtoy reward、Qiskitの人工1q/CX反例を含む。
小回路のcompileに既存transpile_and_measure_costを用い、元compiler profileを維持した。
分子入力/H4 pathは使っていない。今回の42 PASSはH4/local-window性能・全law価格の科学証拠ではない。
