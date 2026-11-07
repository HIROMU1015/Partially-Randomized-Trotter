# O2-A2：最小adapterと新規人工correctness

2026-11-08 JST。**9 tests PASS、0.248秒**。この作業の人工結果で、CI・外部独立再現やPRでの分散/資源改善の証拠ではない。

## 実装と再利用

- 追加module：[a2_lcu_access.py](../../../src/trottertracks/algorithm_codesign/a2_lcu_access.py)。PREP、fine/coarseの二重control、block RY、PREP†、b/aの測定基底、flag付きdecoderを接続。
- 追加test：[test_a2_lcu_access.py](../../../tests/tracks/algorithm_codesign/test_a2_lcu_access.py)。入力は創作した1/2 system qubit回路と状態のみ。
- 既存 `RPE_HADAMARD_BIT_VALUE_MAPPING` を直接利用。RY/H/SdgとQiskitのcontrol/to_gate/inverse rotationを使い、sampler・state solver・RPE decoder・一般回路frameworkを新設しない。
- 既存 `pr_dictionary_validation.gates.matrix_gate` を人工のsuccess/failure blockとunitarity比較に再利用。rtol=0、phase fittingなし。
- 既存 `rte.py` のapplication順、`df_partial_s2.py` のcontrolled/uncontrolledアクセスと旧HadamardのX/Y規約を静的に確認。旧engine/schema/phase suiteは変更せず再実行していない。

## 今回だけ検査した差分

| test | 新規接口の確認 |
|---|---|
| success_and_failure_operator_blocks_on_every_input | toy 1/2system・ζ=.5/1/3のunitary全行列から、success/failure operator blockを独立式と照合 |
| success_probability_and_zero_fill_both_axes | p=S/β²、X/Yの全attempt mean、common signed Fraction、M2、四次momentのbounded包含 |
| identical_paths_retain_all_attempt_noise_floor | Δ=0でもM2>4を保持。success平均や小normだけを利益にしない |
| scalar_phase_is_coherent_and_Y_sign_is_preserved | system global phaseがcontrolled signalへ残り、負のsineの符号も一致 |
| gate_order_and_two_system_qubit_layout | 非可換gate順とsystem/a/b/hのwire順 |
| measurement_register_mapping_and_failure_accounting | classical bit順、failure zero-fill、全attemptにfine/coarse gateが存在、数値資格UNKNOWN |
| exact_toy_pushforward_marginal_without_sampling | dyadic toy lawの全有限組合せでcoarse marginal、input metadataの無変更 |
| nonphysical_or_unspecified_inputs_stop | reset/measurement/未知Gate/偽名Gate/未束縛parameter/不適切scale/曖昧weightを拒否 |
| adapter_has_no_probability_state_rng_or_science_loader | adapterにstatevector/確率oracle・科学bundle読込・RNG・compile経路がない |

testのOperator/Statevectorは独立な小規模検査参照のみ。実装経路はそれらに依存せず、直接χをstatevectorへ代入して準備したことにしない。固有値計算・eighも使っていない。toyの精度はblock比較3e−12、mean等の指定小数桁内の一致で、一様な数値認証ではない。

実行は `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests/tracks/algorithm_codesign -p test_a2_lcu_access.py -v`、既存read-only仮想環境Python3.12.3/Qiskit1.3.0/NumPy1.26.4。記録は[ARTIFICIAL_TESTS.json](ARTIFICIAL_TESTS.json)。Qiskitのcontrol内部の小回路loweringとdeprecation warning、既存Almost_optimal_groupingのSyntaxWarningを観測した。実分子build/compile/native synthesisは行っていない。

## 限界

実H4のfine/coarse binding、保存Qからのdraw、state/Born/shot、実価格、uniform gate/scale/grid誤差は未評価。旧認証engineの代替品として起動しない。実装に科学runnerや自動continueはない。旧SOURCE/STOPは変更していない。
