# 最小実装と人工correctness

2026-11-08 JST／SERVER1。**新接口の人工9件PASS。科学的性能・H4実装資格は未実証。**

## 実装範囲

新規は[module](../../../src/trottertracks/algorithm_codesign/same_position_residual.py)と[test](../../../tests/tracks/algorithm_codesign/test_same_position_residual.py)だけ。新しいsampler、Hamiltonian/DF solver、optimizer、回路builder、価格器、RPE会計framework、scientific runnerは作成していない。

| API | 役割 |
|---|---|
| RotationSite / Replacement | 同joint registerのPauli回転、slot/role、exact target/coarse/residualを保持 |
| plan_sequence | 同coarse表に対するS0/S1/S2。探索・fusion・周期化なし |
| ResidualWord | slot/generator/angle付きcoherent time word。channel/phase dropをschemaで拒否 |
| splice_realization | coarse→residualを元slotへ挿入。他site順序を保持。非zero全slotのwordを過不足なく要求 |

元moduleはすべて無変更。再利用基点は`62c450a6c22fc6cf228bd2fbaabda3c36a7cf169`。

| 再利用SOURCE | Git blob | 今回の使い方 |
|---|---|---|
| synthesis_placement/wrapper_sequence.py | dc117f7f3147e04ae860c1bbb3e1d040f17168ac | 既存Angleをimport。旧toy inventory/PAIを呼ばない |
| g7_generator.py | 70bd5eff7c5f84ddbfa50493aba7a6e55c999032 | ordinary degree3の人工1involution・2orderだけをtestで使用。sample禁止 |
| parent_allocation.py | d2fcabd99b26d37eb7b8a1e7743011e4260ca143 | event_statisticsの既存weight整合検査。allocation探索はしない |
| parent_allocation_native.py | b1edd1daf4d0305fd85952dd5e83293823ed65e6 | operator_targetへ人工matrix callbackを供給。strict合成guardや登録cacheは使わない |

## 実行した9test

| 新test | 確認した追加接口 |
|---|---|
| three_arms_common_table_and_fixed_sites | S0/S1/S2が同じcoarse表とFIXED境界を共有 |
| exact_symbolic_angle_sign_and_no_periodic_phase_loss | 負角・π/radian差、2πを落とすstrict位相誤りを検出 |
| noncommuting_neighbors_stay_in_place | 非可換隣接回転を保ち、residualを中央/末尾へ移すと異なること |
| zero_residual_needs_no_provider_and_no_extra_word | exact zeroはprovider不要、余分なword拒否 |
| realization_site_generator_angle_and_scope_reject | identity/binding不一致、channel、phase drop、欠測を拒否 |
| duplicate_register_role_and_incomplete_coarse_table_reject | duplicate site、register/role、table不足/余剰、float角度を拒否 |
| canonical_finite_operator_mean_inserted_without_exact_claim | 既存ordinary有限eventを負residualへ接続し、coarse×degree3 meanと一致。exact targetとは差が残ること |
| joint_control_and_identity_phase_preserved_in_splice | lower済joint回転でordinary diag(I,U)とcontrolled scalarの位相を保持 |
| no_synthesis_sampler_or_scientific_provider_called | plan/spliceはmetadataだけ。provider生成を遮断しても動作 |

testで使う生成子は人工I/X/Y/Z、2×2/4×4のみ。matrix評価は人工correctnessとして明示許可された範囲。実H4、旧bit/raw、state/evolution、Born、scientific sampling/RNG、pygridsynth、Qiskit circuit/compile、実価格、N/G評価は0。

実行command：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/partially-randomized-trotter-wp11/.venv-server-opt2/bin/python -m unittest discover -s tests/tracks/algorithm_codesign -p test_same_position_residual.py -v
```

結果：9件、失敗0、unittest計測0.064秒。[実行log](ARTIFICIAL_TESTS.log)。これはlocal人工結果で、CI/独立再現・実H4資源改善ではない。初回で全件PASS、人工debug再試験なし。既存SOURCEを変更していないため、旧suite全体の回帰実行はせず、既存coreを通る新しい接続テストに限定した。

## 保証しないことと次の不足

splice入力の`phase_retained=True`は宣言であり、提供wordのphase/meanを認証する仕組みではない。将来の科学providerはSOURCE固定と実演算子照合が必要。原本と同じH_D/H_R、eligible class、元state、original/residual joint law・独立性、coefficient/angle/finite/native誤差、full価格は別に照合する。

現在の`rte_reallocation.native.Angle`はatan/piに限定され、`parent_allocation_native`のcache/strict guardも登録規約を持つ。新coarse/residualの任意pi+radian角や別joint registerに無変更で使えるとはしない。SPのprojective guardもstrict coherent operatorへ流用しない。今回、そのprovider境界を黙って拡張・再合成しなかった。

最小接口が完成した時点で科学入力投入前STOP。新しいsource固定後も実行許可を自動付与せず、[有限科学契約草案](NEXT_MINIMAL_SCIENCE_CONTRACT_DRAFT.md)に残した条件の採択を待つ。
