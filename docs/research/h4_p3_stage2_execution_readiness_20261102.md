# P3 Stage 2：実行準備・再利用対応表

2026-11-02 JST。**文書のみ。Stage 2科学実行は未承認。**
数学・scalar・reference・精度は[科学契約案](h4_p3_stage2_nominal_signal_contract_20261102.md)を正とする。
本資料の「将来実行」は今回の実施内容ではない。今回はSOURCE読取・保存値引用・hash/リンクの静的検査だけを行う。

## 1. 固定identityと証拠の境界

| 役割 | 固定commit | 用途・境界 |
|---|---|---|
| 本branch基点・Stage 1結果 | `5708e37080738c4ebf74590d799a3741246483b2` | run01の34項目PASS、旧preflight STOPを含む結果を保全 |
| Stage 1科学SOURCE | `da13170e593e6d845b014e3f110691be10e29d43` | bit-copy raw loader、独立JW/native構築。runner再実行は不要 |
| P3静的設計 | `eb20e81074aaae3cb365b348f6c3ab9d217d3697` | [分割・scalar・sector設計][STATIC]、[段階設計][OLDREADY]。今回の初期案がStage 2の未決定項目を具体化 |
| H4共通入力anchor | `4591b8d75a2a6a55c74a51225f0c61648023c8ee` | 共通raw state・snapshot・Cモデル・reference・exact ν |
| 元入力生成SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12` | 積分/JW/spin/DF定義、既存signal/RTE kernelの読取。SCF/FCIを再実行しない |
| C reference科学SOURCE | `e80d7510880f8db993b777c627bb8e95b05a070d` | 保存DF raw reference区間を共通基準へ。旧契約・grant・runnerは流用しない |
| P1 Pauli診断SOURCE | `a0a3aff915f98456733d6bf09ed641a1be423628` | paired recurrenceとsigned bias骨格。H6値/可換保持/許容差は移さない |
| matrix-only DF既存例 | `638e0a989b4471a075549ccb58617b1030e6748f` | `scripts/h46_new2_core.py::matrix_only_tail_coefficients`。旧B3のsnapshot/B/結果は使わない |

固定保存ファイルは以下。SHA256はファイルbytesのidentityであり、承認証明のhashではない。
パスprefix `I` / `C` / `N` / `S` はこの表の省略だけに使う。

```text
I = artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/
C = artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/
N = artifacts/non_df_h4_central_arb_pf_cosine/2026-10-25/
S = artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/
```

| 保存ファイル | 固定ref | SHA256 |
|---|---|---|
| `I/receipt.json` | H4 anchor | `6a2304db448845aef6f86da06a7bffd5d5c708273964e7e797c3d8ec71fcb6bf` |
| `I/snapshot.npz` | H4 anchor | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| `I/pauli.json` | H4 anchor | `49e2f0b86cba9933eee20a03a05934d950caafb5044401bca50d3aa79a518175` |
| `C/pauli_jw_model.json` | H4 anchor | `9ae8973b457cab26bb685a8fac9887dbc5e3be5db9059355822893d26cae2aa4` |
| `C/df_native_model.json` | H4 anchor | `cc60f6dc3de885974cfe74ef822e49380edb6b9ca44777bf12e9654256cd322a` |
| `C/pauli_jw_result.json` | H4 anchor | `b63d3da01269b2706a3fe185658188779d24f665f057b91367ac7c9254a83245` |
| `C/df_native_result.json` | H4 anchor | `612f646a233c6ecfa52d57f0c1bf969eb6cc940e2c60a691ca21a0c2276e7cb2` |
| `N/run01/signal_64.json` | H4 anchor | `a8156ba7bebf8ca9ed2639bbdc74d9ce8ec8ba33f8fd890735cf9ae15d55cea6` |
| `N/input_binding.json` | H4 anchor | `9458b317005cc17a7aacd508b6b3cd28eccd994b5b1f5bfb5d9fee2ff1639110` |
| `N/inputs.json` | H4 anchor | `da86c8df2fe64d298967a48e26013d2f2daa3d2bd2cc1b667f404953ea4b7d67` |
| `S/REPORT.md` | Stage 1結果 | `4e75f062cdcb66e65a08e1ba0e21e05f76d78c9874105cb7180a8a2d89bc1b9d` |
| `S/run01/operator_comparisons.json` | Stage 1結果 | `b7b158c6349659243298a7cac77ca2bf07e83e54788d5723746bc4f59b1f7c84` |
| `S/run01/construction_paths.json` | Stage 1結果 | `46b2fea7d631687dbbff430d72944a9a9e7633fabd20fec2b0f0d9847b45bd07` |
| `S/run01/source_and_inputs.json` | Stage 1結果 | `8dc388fdfd24eba358a26ec10553ae3b7877c438ebfb28f0c21f47702f92f53a` |
| `S/run01/STOP.json` | Stage 1結果 | `e44f5bfc12f51aa8ef03693c0e342d55431bd3f6d77af9d579ce60965f9e2e05` |

別のdescriptorを区別するため、Pauli Hamiltonian hash
`e0dbbe9a2b6e86f81004307cd337ba83ee1e25642f04a30b02365be4d2dae795`、DF hash
`156939eb75c1446d3a29d28702ecb7349f36e408cddee250aa4520d93cc967e3`も維持する。
両hashの文字列一致をHamiltonian一致条件にはしない。共通raw state hashとoperator一致の証拠は別に接続する。
旧H4 Pauli prefix27、DF prefix1、legacy DF12、別long-time H4、H6を新しい案Aの結果へ混ぜない。

## 2. SOURCE接続・再利用と最小拡張

以下はSOURCEの静的確認。科学関数を今回呼び出していない。

| 処理 | 読取した既存SOURCE・関数 | 再利用／最小変更 |
|---|---|---|
| 保存raw入力 | [Stage 1 runner][RUN1]、`h2_non_df_pilot`の保存入力処理 | 修正版のbit-copy decodeを再利用。`real+1j*imag`でsigned-zero bytesを変えない |
| Pauli residual | [matched_partition_stage1][ADAPT1]の`correction_from_two_body`, `jw_matrix`, `pauli_route` | Dと元Wの独立経路を再利用。`jw_matrix`はmatrix/identityしか返さないため、既存JW変換からresidual model/dictionaryも返す最小接続が必要 |
| 共通T/V・DF projector | 同SOURCEの`_native_one_body`, `df_route` | native occupation表から構築し`reverse_bits(8)`でQiskit順へ。canonical Vはfactor square、DF辞書和は別経路。全Stage 1診断34件を再実行しない |
| Pauli辞書・finite作用 | [P1 SOURCEのpauli_signal_adapter][PAULI]の`PauliHamiltonian`, `paired_polynomial_action` | cutoff0、label/sign/orderを再利用。`step_time=δ`（内部でr除算）、`r=4,K=4,normalization=1` |
| P1信号診断 | [pauli_mechanism_bias][P1] | explicit T/V/scalarと共通stateを渡す薄い接続。全H prefix split、trace(A)=0仮定、旧H6の`1e-10`を丸ごと使わない |
| DF I/Z/ZZ係数 | [df_rte_tail][TAIL]の`exact_df_diagonal_coefficients` | 全10fragment、係数cutoff0、identity抽出。fragment/basis別の係数情報を保存 |
| DF basis | [diag_hermitian][DIAG]、[matrix-only既存例][MATRIXONLY] | Hermiticity確認後`assume_hermitian=True`。保存G対角化とfull occupation projectorへの接続だけ追加 |
| finite corrected operator | [rte][RTE]の`finite_taylor_operator`, `finite_rte_corrected_operator` | DF辞書和のWとconfig `evolution_time=δ,rte_steps=4,dimensionless_step_time=λ_R h`を入力。Pauliの計算済み多項式を渡さない |
| λ/β/B | 同SOURCEの`_paired_order_weight`, `finite_rte_distribution`, `make_rte_config` | scalar関数を利用。旧tail/Bを再利用せず、新辞書ごとに計算。50桁decimalの3項scalar検算を小さく追加 |
| 既存Arb scalar評価 | [h4_ball_evidence][BALL]の`finite_normalization` | K4固定の3項とβ^(rq)を確認。旧runtimeパスが存在しないため、初期案ではbinary64既存関数＋標準decimalの小検算を選び、Arb環境再構築を要求しない |
| raw ideal mean | [h4_ball_evidence][BALL]の`finite_implemented_mean` | momentの代数だけ再利用し、ideal p・cos/sinを渡す小接続。CDF/保存角度検証関数へ進まない |
| corrected/raw会計の既存例 | `rte.finite_rte_operator_moments` | 同関数はcorrected/β^rでrawを返すため、独立raw検算としては直接使わず、会計規約の回帰参照に留める |
| 共通operator対照 | `finite_taylor_operator`の既存5次recurrence | `V_0*`から1回だけ。独立armの結果に代入しない |
| Reference/ν | [CandidateWorker/ signal][BALL]と保存`raw_norm_exact` | `signal`はraw内積。保存DF raw referenceのexact端点をνで一度だけ除算。旧referenceを再生成しない |
| 誤差伝播 | [h4_enclosure_prototype.product_error_bound][ERROR] | 積誤差の代数を再利用。engineering丸め項を足す少数の式で足りる。Arb全行列certificateは初期scopeに要求しない |
| Gaussian one-body | `df_trotter`一体basis/actionとStage 1 native action | 数学・indexの根拠。Stage 2はnumeric exponentialのみ、Gaussian circuit helpersは呼ばない |

既存`extract_df_tail_from_hamiltonian`、`prepare_df_partial_s2`、`build_df_blocks`、
`DFBasisRegistry`を含む一括経路はGaussian circuitを生成するため、初期signal-only処理で直接起動しない。
`evaluate_fixed_signal`、旧Cの`CandidateWorker`一式、Stage 1 runner全体も、
reference再計算・旧prefix・回路・旧承認契約等を含みscopeが合わない。
SOURCEの機能を再利用し、旧runnerの承認や旧実行を再利用しない。
最小の新規実装は「明示T/V/scalar＋2辞書＋既存finite kernel＋共通signal/誤差会計」を接続するP3専用処理で足りる。
新しい汎用framework、grant基盤、approval-text保存は不要。今回はこの実装も行わない。

## 3. Correctness testsの再利用計画

testsは今回実行していない。下表の既存testには回路生成を含むものもあるため、
将来は必要なmatrix-only人工fixture/代数を再利用し、分子や回路を作るsuiteの一括実行は避ける。

| 既存test SOURCE | 必要な検査内容 |
|---|---|
| [test_matched_partition_stage1][TEST1] | signed-zero decode、Wの1/2・D符号、独立構築。全H4 batchの再実行とは区別 |
| [test_fixed_signal_evaluation][TFIX]の`test_pauli_consolidation_order_identity_hash_and_no_cutoff`, `test_s2_sign_order_constant_and_time_reversal` | Pauli統合/identity・scalar一回・S2順序・±time |
| 同`test_all_pauli_events_match_generic_involution_definition_and_mean`, `test_artificial_df_exact_circuit_law_mean_matches_sparse_mean` | signed involutionのmoment/paired identity。回路を作らない小matrix oracleだけ再利用 |
| 同`test_pr_reconstruction_sparse_mean_repetition_seed_and_empty_tail`, `test_df_b0i_phase_matches_known_tail_identity` | q/r、identity phase、raw/corrected会計。sampling部分は初期science対象外 |
| [test_pauli_mechanism_bias][TP1]の`test_exact_tail_half_order_scalar_and_raw_norm`, `test_degree_r_q_and_no_B_multiplier`, `test_legacy_raw_mean_preserved_by_shared_recurrence` | explicit retained half・degree5・r/q・B/ν一回・回帰 |
| [test_finite_rte_signal_validation][TFIN]の`test_operator_moment_acceleration_matches_explicit_partial_s2_mean` | 独立moment平均と直接多項式の一致。元testのcircuit buildは新H4 scopeへ持ち込まない |
| [test_df_rte_tail][TTAIL]のI/Z/ZZ identity検査 | α/b/d式とbasis別辞書のsum。保存分子の再生成なし |

将来の最小追加人工検査は、非可換T/Vを使った上記接続、同じVの異なる辞書で
corrected meanが一致しBは変わり得ること、ν未補正referenceの一回除算、identity差の会計、+Im signである。
科学的H4適合検査は承認後の同tuple batchに含め、人工PASSを科学的性能証拠にしない。

## 4. 将来の最小科学実行範囲と件数

初期対象はH4/案A/T.8/S2/q8/r4/K4の1tuple、M_B/M_Cの2arm。
既存結果の新しい数値処理もこの将来承認scopeに含める。今回の件数は全て0。

| 科学処理の単位 | 必要件数 | 重複を避ける方法 |
|---|---:|---|
| 辞書構築 | 2 | Pauli元tensor→JW、DF保存factor→rotated Z/ZZを独立構築 |
| 共通operator bundle | 1 | native T、factor-square V、common scalarを共有。DF辞書和から作らない |
| DF保存Gの8×8 eigendecomposition | 10 | 保存factorごとに1回、固有basisをcache。新DF分解ではない |
| 共通256×256 exponential | 2 | T halfとV_0 full。q/r位置ごとに再計算しない |
| 上記exponentialの同一精度spectral照合 | 2 | 同じ2operatorだけ。別tuple・full-H reference生成なし |
| `Z_split` | 1 | 両armの共通baseline |
| `Z_nom^P`, `Z_nom^DF` | 2 | 独立辞書＋異なる既存finite kernelで評価 |
| `Z_nom*` | 1 | 共通operatorからの直接多項式対照 |
| `Z_raw^P`, `Z_raw^DF` | 2 | ideal moment式から評価。Bとのclosureを独立確認 |
| Normalization bundle | 2 | 各armのλ、3次数weight/probability、β、B、log B。binary64主値＋decimal50検算を同bundle内に含む |
| 保存referenceのphysical変換 | 1 | raw enclosure両端/保存exact ν。新しいfull-H指数は0 |

したがって**複素信号の新規伝播6件＋normalization bundle2件**が比較の最小構成。
cosine/+Imは同じ複素内積から保存し、12件の別信号runにしない。
共通operator対照を削るとcanonical residualへの独立照合を失い、raw2件を除算だけに置き換えるとB/順序/phaseの検査が自明になるため残す。
表の構築・対角化・exponential・norm/matrix積は同じscience batch内の下位処理であり、
「6 library callsで全処理が終わる」という意味ではない。内部matvecや誤差診断は実際の処理数を結果に記録する。
恣意的なCPU/RAM/worker上限や、同じoperatorの不要な再構築・全34項目再検査を追加しない。

trajectory sampling0、RNG replay0、implemented CDF/law評価0、M_A0、回路build/compile0、
SCF/FCI/DF factorization0、新state/reference生成0、N/C/G・shot感度0。
scientific workerの並列利用は可能だがAI sub-agentは使わず、共有環境や他jobを変更しない。
無関係worktreeのHEAD/status変化や一時的なCPU競合をSTOP条件へ追加しない。

## 5. 精度・scalar・referenceの判断と残る条件

| 項目 | 本契約案の推奨・固定内容 | 残る確認／科学的影響 |
|---|---|---|
| scalar | Stage 1 DF保存`v̄`を共通binding、T identityはT内 | 辞書抽出identityとの差は未計算。適合しない場合は別scalar比較へ自動変更せずSTOP |
| target/reference | DF canonical T/Vと保存DF C raw reference/ν | C operand↔保存matrix↔新operatorのtransferを計上。表現差を0と見なさない |
| nominal precision | complex128、full256、共通expm＋spectral照合 | biasの各Re/Imが解像できるか未測定。不足はUNRESOLVED、精度昇格は別許可 |
| scalar precision | 既存binary64 RTE関数＋標準decimal50の小さい検算 | 旧Cの記録済みflint runtimeパスは今回の読取時に存在せず。Arb環境復旧を必須工程にしない |
| sector | 初期はfull256を固定 | 36次元の高速化には対象operator不変性の追加根拠が必要。初期には追加しない |
| 独立性 | JW元tensor／DF factor辞書／native canonical対照 | T/state/scalar共有は可。residual dictionary・nominal出力共有は禁止 |
| SOURCE/runtime | 実装は未作成、科学SOURCE SHAは未定 | 承認後の最小実装・人工検査後に固定。未知の未来SHAを本書へ捏造しない |
| 実効law/native価格 | 初期scopeから除外 | grid/角度/compiledρ・費用の結論には進めない |

初期tuple、scalar方針、reference選択、full-space、nominal数理、最小件数は選択可能であり、
研究条件をユーザーへ毎段階で問い直す必要はない。
残る必須事項は**この提案scopeと精度政策へのユーザーの明示的科学実行許可**、
その後の実装SOURCE/runtime固定と実測preflightである。
「案の採択」「commit/push」「重要レビューの許可」を科学実行許可に読み替えない。
数値不整合や分解能不足が判明した場合、解決に必要な新しい科学条件・追加精度・新referenceは別scopeとして判断する。

## 6. 将来の出力・来歴・STOP

将来の新結果は例えば
`artifacts/df_pauli_matched_partition_validation/<実行日>/stage2/run01/`へ分離する。
Stage 1 namespaceを上書きしない。最低限の出力は以下を一つの結果bundleにまとめればよい。

- REPORT：固定条件、6信号、signed outer/finite差、λ/β/B、Re/Imの解像状態、支持できる主張と限界。
- JSON：共通operator/2辞書のdescriptor、scalar所有・identity defect、raw state/ν/referenceのbinding、
  6信号とnormalization、独立closure、数値/binding/reference-transfer uncertainty、PASS/UNRESOLVED/FAIL。
- 必要な再利用可能array/辞書：canonical T/V、各residual辞書・DF basisを、重複せず保存。
- SOURCE/input来歴と終了状態：実際のcommit、直接使う入力hash/runtime、実際の科学件数、失敗原因・STOP。

別々の承認台帳・証明書・多数のguard manifestを新設する必要はない。
入力・結果・科学SOURCEのprovenanceと、ユーザーの許可の証明手続きは区別する。
正式受理・全law保証・native価格のclaimは追加しない。

入力/hash/ν/scalar/辞書/basisが非PASSなら信号に進まず停止。
独立nominal/B closureが非PASSなら原因を記録して停止。
bias成分だけが分解能以下の場合は`STOP_STAGE2_UNRESOLVED_COMPONENTS`、整合し予定比較が解像できた場合も
`STOP_STAGE2_NOMINAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP`で終了する。
同一SOURCE・固定scope内の通常debugと、scientific conditionを変えた追加runは区別する。
失敗した科学attempt/出力は保持し、科学batchの自動retry・精度昇格・追加tupleは行わない。

## 7. 今回の静的検査・保全・公開手順

本branchは`codex/h4-p3-stage2-contract-20261102`、Stage 1結果commitを親とする独立branch。
作成対象は本資料と科学契約案のMarkdown2件だけ。
保存SHAは固定Git objectのbytesで照合し、JSONは通常の読取に限定する。
数値ライブラリimport、operator構築、norm再計算、信号/B/law評価、pytest、回路処理は起動しない。
参照パスの存在・文書リンク・必須scope項目・`git diff --check`・2ファイル以外に変更がないことを静的確認する。
固定commit上の参照先27件、保存ファイルSHA256 15件、νとreference区間の引用一致は静的検査でPASS。
この作者自身による確認を、独立した詳細科学レビューと呼ばない。

公開時はGitHubの新branch SHAを取得し、そのcommitでの両Markdown全文をローカルと照合する。
比較元の保存SOURCE/input/raw/result/grant/ledger/marker/STOPは更新しない。
Stage 1の`STOP_STAGE1_PASS_NO_AUTOMATIC_FOLLOWUP`と旧preflight STOP、P1/P2、main/元研究branchを保全する。
今回の文書追加は新科学結果・主要研究方針の採択ではないため、概要・manifest・indexの一括改訂は行わない。
正式共通運用指示`research_project_instructions_unified_2026-10-25.md`の§8.1/§8.4/§10、
共通/各worktreeのAGENTS.mdを適用する。

**文書公開・remote SHA・両全文照合後mandatory STOP。**
次の担当はユーザー：初期Stage 2科学scopeと精度政策を判断し、実行を希望する場合に明示許可する。
Codexはその許可後だけ実装・限定実行を行う。今回はStage 2科学計算、詳細科学レビュー、Stage 3/4、案B、P1N、論文改訂へ移行しない。

[STATIC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb20e81074aaae3cb365b348f6c3ab9d217d3697/docs/research/h4_df_pauli_matched_partition_static_design_20261102.md
[OLDREADY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb20e81074aaae3cb365b348f6c3ab9d217d3697/docs/research/h4_df_pauli_matched_partition_execution_readiness_20261102.md
[RUN1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da13170e593e6d845b014e3f110691be10e29d43/scripts/run_h4_p3_stage1.py
[ADAPT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da13170e593e6d845b014e3f110691be10e29d43/src/trotterlib/matched_partition_stage1.py
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_signal_adapter.py
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/src/trotterlib/pauli_mechanism_bias.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_rte_tail.py
[DIAG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/df_trotter/decompose.py
[MATRIXONLY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/638e0a989b4471a075549ccb58617b1030e6748f/scripts/h46_new2_core.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/src/trotterlib/rte.py
[BALL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80d7510880f8db993b777c627bb8e95b05a070d/src/trotterlib/h4_ball_evidence.py
[ERROR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80d7510880f8db993b777c627bb8e95b05a070d/src/trotterlib/h4_enclosure_prototype.py
[TEST1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da13170e593e6d845b014e3f110691be10e29d43/tests/test_matched_partition_stage1.py
[TFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_fixed_signal_evaluation.py
[TP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a0a3aff915f98456733d6bf09ed641a1be423628/tests/test_pauli_mechanism_bias.py
[TFIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_finite_rte_signal_validation.py
[TTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0c329b30a81accb35c212be8defa66fee0dd0f12/tests/test_df_rte_tail.py
