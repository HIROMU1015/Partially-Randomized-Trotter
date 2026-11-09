# 出典と再利用判定

2026-11-09 JST。調査基点141fa7624d74aa5deacbd752d09df516cad9206f。現設計のコミットはGit履歴で確定する。下表のSOURCEは静的に確認し、既存実装の変更や科学runnerの起動はしていない。

## 必須固定資料

| 資料 | 固定commit／参照 |
|---|---|
| ユーザー指定統合成果 | 9d12d7c6cf3da8f739e0238f7abbbc7018bb6d53。作業基点141fa762はこの子孫。統合枝を進めず設計専用枝を使用 |
| 最新q64完了・戦略・固定PLAN・数値修正 | [fe361b20のdirectory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108)。COMPLETION_RESULT、LONG_Q_STRATEGY、COMPLETION_PLAN、NUMERICAL_REPAIR_RESULTを確認 |
| C7-2 | [709d6ed9 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709d6ed9c234bf6d5a8848d3a30be69d88fd6476/docs/research/track_d_c7_2_cx_structure_20261107/REPORT.md)、既存人工traceとdepth定義 |
| C7-4 | [64072ba8 MODEL_REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64072ba81bb70db1df1b34e94adb08e78926f24b/docs/research/track_d_c7_4_model_stability_20261107/MODEL_REPORT.md)。A/F/S・88path・P0未閉包 |
| B/C条件付き全13-round | [8828b298 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md)。保存会計を引用し、再算定しない |
| C1 H4追加融合 | [0ef53fa9 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9/docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md)。別のON policyでq64と混合しない |
| B6b理想測定同値性 | [7ffe8b28 MEASUREMENT_EQUIVALENCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ffe8b281b76cef3a6fd06f43413904d6766d0ec/docs/research/track_d_b6b_bc_integration_20261108/MEASUREMENT_EQUIVALENCE.md)。価格の厳密性／浮動小数点誤差とは別 |

q64のsampling/raw/P0初期SOURCEは6b1d613d、sample00の数値修正はf1e89590、限定完了はb1e7cc5b。最終成果fe361b20を実行SOURCEそのものと呼ばない。141fa762とfe361b20は共通祖先b1e7cc5bから分岐しているため、統合treeの旧q64報告だけで未完了と判定せず、固定最終報告を別参照した。

## 実装・テストの再利用

以下の相対linkは本設計枝の基点に含まれるSOURCEを指す。q64実行版との関係は上記の固定SOURCEに従う。

| SOURCE・関数 | 確認事項と次段階の再利用判断 |
|---|---|
| [track_d_c3_whole_path.py](../../../src/trottertracks/track_d_c3_whole_path.py)：head_recipe, mixed_head, physical_segment, prepare_steps | q−1/q−2融合、control modeと順序、scalar、actual inverse。prepare_stepsはCAR/dense検証も作るため価格だけのiteratorへは受け口分離が必要 |
| [c1_h4_strong_baseline.py](../../../src/trottertracks/c1_h4_strong_baseline.py)：vacuum_phase, StrongBuilder.transition/sequence | uncontrolled basis、actual rotation group、relative Gaussian、analytic gauge補正。既存対照を弱体化しない |
| [track_d_h4_reference.py](../../../src/trottertracks/track_d_h4_reference.py)：stored_events, control_one, directional_head, reference_gadget, raw_reference_columns | event順・符号、physical restriction、二CSWAP、A.inverse、χ、全256列。価格だけにraw列生成は不要 |
| [ax2b_native_df_v5.py](../../../src/trottertracks/resource_applicability/ax2b_native_df_v5.py)：append_directional_diagonal | RZ2CX/RZZ4CXという指定展開、ancilla phase。ordinary制御と区別 |
| [df_trotter/ops.py](../../../src/trotterlib/df_trotter/ops.py)：one_body_diagonal_primitives, df_squared_diagonal_primitives, append_diagonal_primitives | 対角中心・identity/global phase・zero prune・ordinary RZZ.controlの定義 |
| [pr_dictionary_validation/circuits.py](../../../src/trottertracks/pr_dictionary_validation/circuits.py)：BasisProvider, build_event_sequence | tailの既存basis共有・signed event。価格用に別物理emitterを再実装しない |
| [track_d_c7_2_toy.py](../../../src/trottertracks/track_d_c7_2_toy.py)：weighted_depth, structure | filtered同期・全wire counterとDAG longest path。再実装せずmockでも関数本体を再利用。structureは実回路を要求するので今回呼ばない |
| [track_d_c7_4_models.py](../../../src/trottertracks/track_d_c7_4_models.py)：event_features, design, fit | basis/lane-run・support work・zero/sign等の定義を確認。将来stream受け口の最小拡張。fitは今回は呼ばない |
| [track_d_c7_4_stability.py](../../../src/trottertracks/track_d_c7_4_stability.py)：NativeSmallUnitary, residual | ≤9qubit native unitary・精度、17全Operator拒否。価格counterへ持ち込まない |
| [track_d_head_tail_h4.py](../../../src/trottertracks/track_d_head_tail_h4.py)：splice_join, prepare | C1 opt-inの接口変換と位相。q64のOFF policyとは別cache/domain |
| [c1_h6_compiler_precision.py](../../../src/trottertracks/c1_h6_compiler_precision.py)：compiled_wrapper／[gates.py](../../../src/trottertracks/pr_dictionary_validation/gates.py)：compiled_wrapper, compile_cache_key | 宣言phase MPFR160 offset、custom展開、固定P0、full QPY/angle等を含むcache。新compilerを作らない |
| [track_d_whole_path_price.py](../../../src/trottertracks/track_d_whole_path_price.py)：OrderedRequest, _checked_price | finite-fixture/qualified価格とscopeのguard。新しい構造proxyを無断登録しない |
| [track_d_c7_3_acceleration.py](../../../src/trottertracks/track_d_c7_3_acceleration.py)／[track_d_state_action_acceleration.py](../../../src/trottertracks/track_d_state_action_acceleration.py) | 正確な局所gate cache、≤9全unitary／17qubit全列batch。検証負荷の因子であって価格生成器ではない |
| [C8-1 completion adapter](../../../src/trottertracks/track_d_c8_1_completion.py)／[runner](../../../scripts/run_track_d_c8_1_completion.py) | 既存raw/compile/post stage・全256列進捗、nofusion選択、保存wall timeとworker high-water。再起動・実行はしない |
| [C7-4 tests](../../../tests/check_track_d_c7_4.py)、[C5 tests](../../../tests/check_track_d_c5.py)、[C3 tests](../../../tests/check_track_d_c3.py)、[C1 H4 tests](../../../tests/check_track_d_head_tail_h4.py)、[C7-2 runner](../../../scripts/run_track_d_c7_2_toy.py) | feature/max・axis依存・境界control mode・gauge・構成・P0の試験入口を静的確認。既存一括testsにはnumerical/P0経路があるため今回は起動しない |

本成果は上表SOURCEを変更していない。mockだけを新しく置いた理由は、既存counterにない全wire transferの数学的合成仕様を、Qiskit importや科学データなしで確認するためである。

## Qiskit1.3一次資料

- [QuantumCircuit.depth](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/quantumcircuit.py#L3479)：filtered-out操作も同期点として時刻を伝播する実装。
- [preset builtin_plugins](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py#L535)：opt1の1q decomposition／inverse cancellationとdepth/size固定点loop。
- [Optimize1qGatesDecomposition](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py)：実際のnative再合成への入口。
- [InverseCancellation](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/inverse_cancellation.py)：指定inverse pair・self-inverse gateの処理入口。

固定環境の上記Python SOURCEをread-onlyで確認し、公式1.3.0 tagのdepth/builtin sourceも取得した。Rust内部の全閉包や数値tie-breakまで証明したわけではない。現行の別versionのcompiler仕様へ置換していない。

## 運用・公開の境界

親AGENTS、worktree AGENTS、PROJECT_MAP、研究概要・現状、正式共通指示2026-11-09を適用した。正式共通指示が追加された後に当該版を確認し、旧版で代用しなかった。ユーザーの許可はこの設計scopeの実行許可であり、承認証明機構は作らない。

q64 run04の既存private `local/q64_sample01` のcompile/check/stage/progress JSONを読んで処理時間と記録の意味を確認した。これらの原記録・CPU/RAM/worker詳細・内部pathを公開payloadへコピーしない。events/QPY/NPZを読み直す科学再解析や再計数は行わず、原本の再hash・再archiveも不要に実行しなかった。
