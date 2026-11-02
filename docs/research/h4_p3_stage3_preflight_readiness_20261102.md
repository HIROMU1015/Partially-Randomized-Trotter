# P3 Stage 3：技術preflight・再利用・実行準備の非実行計画

2026-11-02 JST。[科学・価格設計案](h4_p3_stage3_native_cost_design_20261102.md)の実装準備表。
**今回の実行はSOURCE読取、保存metadata/hash照合、Markdown作成だけ。**
科学ライブラリimport、数値再構成、回路build/compile、sampling、pytestは行っていない。
以下のpreflightは将来の科学実行許可に含める対象であり、今回のPASS一覧ではない。

## 1. 固定入力・保全対象

branch基点はStage 2結果`8bcf5851f99f980dc08c5885ace54874b554c815`。
Stage 2科学SOURCE `4bfbbaad1f8178077489bf06b085377256d6fb80`、Stage 1結果`5708e37080738c4ebf74590d799a3741246483b2`、
元H4共通入力`4591b8d75a2a6a55c74a51225f0c61648023c8ee`を固定refとして使う。
Stage 2の[入力来歴][INPUT2]に元receipt/snapshot/Cモデル/reference/νの15ファイルhashがある。
科学条件を再生成せず、その対応をStage 3 SOURCEへ接続する。

下表は基点commitのStage 2 namespace
`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/`の保存bytesのSHA256。

| ファイル | SHA256 | Stage 3での使い方 |
|---|---|---|
| `REPORT.md` | `94521efef94ae0a1450623aaddae15315cfa355f2264361e9ec898322d744d4e` | 結論と限界。finite biasを再解析しない |
| `STOP.json` | `ad3c238820cc2bcb5d620e223e5b273e37acd82236fc449aea95672cfc9267a2` | `STOP_STAGE2_UNRESOLVED_COMPONENTS`をそのまま保持 |
| `dictionaries.json` | `41594716f68e3d106d5be09b4746d506ab352c0e83fd09abac80e108b2fc27f6` | 係数・成分ID/順・DF U/η。basisを再対角化しない |
| `common_operators.npz` | `fd6e4f8d7d37ebab2cd64a55580e0c2b7d6ffb9480add583ea906b8d83dc7fc2` | 共通T/V0を参照。将来T半指数とのnative照合に使用 |
| `results.json` | `398e43cff0c1a314994c337a08601aa80753482acd72b30541d4bc6a413d2a92` | 保存λ・次数分布・B・旧不確かさ。新signal/B/費用会計なし |
| `source_and_inputs.json` | `9a0acbab136fb8102e6cf44ee1211590ac7d4dacc38e5bef8ece107c540ed4fc` | 実SOURCE・直接入力・runtimeの対応 |
| `operator_preflight.json` | `f18e587743d655a4c5b39361c2c4b482ea18bc548e3133eaeb7e67d643200354` | 旧operator比較とidentity defect。native実装のPASSとは別 |

hashは科学入力・証拠のidentityであり、ユーザー許可の認証ではない。
Stage 1/P1/P2/Stage 2のSOURCE、input、raw、result、grant、ledger、marker、STOP、main/元研究branchは変更しない。
旧prefix27/DF prefix1/legacy DF12/H6/別長時間H4の値を新価格として使わない。

## 2. 再利用箇所と最小拡張

下表のSOURCEは全て基点commit上で静的に確認した。

| 既存処理 | 再利用計画・適用上の注意 |
|---|---|
| [Stage 2 kernel][KERNEL2] | 保存辞書の構成意味、native occupation/JW順、共通scalar。`build/evaluate`を再実行しない |
| [df_trotter.ops][OPS] `U_to_qiskit_ops_jw_givens` | 保存U_lの10 basis＋共通T用1 basisをnative lowering。`BogoliubovTransform(U.T)`を固定しdense fallbackなし |
| 同`one_body_diagonal_primitives/append_diagonal_primitives` | TのRZとphase、controlled対角作用。前後のactual basis/adjointを共通half helperで接続 |
| 同`build_one_body_gaussian_block_givens` | T用の一度のHermitian対角化＋Givens。両armで生成し直さず同じobject/命令を再利用 |
| [df_rte_tail][TAIL] `DFBasisRegistry.register` | 保存basis IDとnative操作hashを対応づける。新しいgate hashは旧Stage 2のmatrix/basis IDと区別して保存 |
| 同`extraction_component_circuit_specs/prepare_df_rte_event_inputs` | typed接続の構造を再利用。既存一括extractは再対角化するため、保存component descriptorを接続する小adapterが必要 |
| [df_rte_qiskit][DFBUILD] | signed rotation/product、order phase、basis central制御、actual adjoint、adjacent same-basis reuse |
| [pauli_signal_adapter][PAULI] `append_pauli_event` | 保存Pauli labelとRTEEventの順・負product・rotation・order phase |
| [h4_d2_pauli_control][D2] `central_control_body/central_control_tokens` | Pauli残差だけを制御化。Gaussian命令の混在bodyは対象外 |
| [rte][RTE] `RTEComponent/finite_rte_distribution/sample_rte_events` | 方式別componentを保存順で渡す。flat32 event streamを両armで統一。今回呼び出さない |
| [df_partial_s2][S2] controlled deterministic block | 前後無制御basis＋controlled primitiveの実装を参照。旧preparationのidentity/再対角化を丸ごと使わない |
| [df_partial_s2_repeated][REPEAT] | outer/time/event順と保存形式を参照。別step-seed階層や一括samplingは新flat32案へ暗黙適用しない |
| [fixed_signal_evaluation][FIXED] `hadamard_wrapper/default_compiler` | 両軸同境界・+Im Sdg、P=0、rz/sx/x/cx/opt1/seed17 |
| [rte_compiled_cost][COST] `transpile_and_measure_cost` | 実compile・QPY・metrics・fingerprintを再利用。共有persistent cacheを変更せず、新namespaceかprocess内で再利用 |
| [pauli_mechanism_cost][PARTS] | F/R/E/Emptyの操作的定義、metrics、非加法性の代数。既存K2/Pauli-only Rを汎用Gaussianへ誤適用しない |

最小新規接続は、**保存辞書→typed component/native basis、共通Gaussian half helper、
outer-major合成、固定seed/event保存、両軸wrapperとF/E/R/Empty dispatch**で足りる。
旧固定runner、承認用grant/ledger、旧C reference生成・N/Gまで走る一括パイプラインは移植しない。
今回はadapter/runner/tests自体も新規実装しない。

## 3. 将来のpreflight順序

| 段階 | 確認対象・方法 | 非PASS時／後続 |
|---|---|---|
| A：入力・runtime | 固定commit/hash、Stage 2の辞書360/184・U/η・T/V、scalar所有、time。qiskit/natureのversionとGivens lowering可用性 | 不一致はH4回路生成前STOP。共有環境を修復名目で変更しない |
| B：人工correctness | 小規模手指定系で保持制御/phase/actual adjoint、signed order0/2/4、Y/非可換順、basis切替、+Im、parts削除、compiler回帰 | 分子・samplingへ拡張せず実装debug。PASS後SOURCE commit固定 |
| C：共通basis/保持 | T用1回の8×8対角化、DFは保存10 Uを使用。native basis、adjoint、one-particle作用とvacuum phaseを照合 | native変換に差があればSTOP。別loweringへ自動変更しない |
| D：H4保持作用 | system half／actual inverse／controlled halfを同じT半指数/adjoint/diag(I,E_T)へfull-space照合。外側scalarは別phaseとして検査 | **PASSしてから**trajectoryをsamplingする |
| E：event固定 | 8seedから各32 eventを一度取得、辞書ID/係数/sign/basis/順/角度/次数、q×r配置を保存 | 再抽出なし。law/入力不一致はSTOP |
| F：選定program | 8本それぞれF/Eのnative bodyを、同じ保存eventと辞書からの独立matrix oracleへ照合。R/Emptyも共有programとして照合 | 全対象PASS後だけ36 price compileへ進む |
| G：価格 | 固定wrapperをcompileし入力/出力QPY、RZ/CX/depth等を対応づける | 失敗や欠測は保持しSTOP。新条件・自動retryなし |

Gaussian basisについてはone-particle8×8のorientation/JW順とvacuum作用を検査し、
forward/actual inverseの合成をfull-spaceで確認する。保存Uを別gaugeへ回転し直して価格を変えない。
basis単体に全体phaseがある場合は、vacuum位相に対するone-particle作用をUへ照合し、
同じbasisとactual adjointの対でその位相が消えることを確認する。
これはbasisの表現規約の確認であり、完成した保持blockやcontrolled bodyの位相差を事後補正する処理ではない。
DF Z/ZZの意味はそのbasis/符号/supportで接続し、componentごとの巨大matrixを全て保存する必要はない。
共通halfはsystem256、controlled512。ancilla8が最上位bitなのでcontrolled targetは`diag(I,U)`。
ancilla0側でbasisが消えること、ancilla1側のscalar/一体identityが保たれることを別々に検査する。

F/Eの独立oracleはgate-listを再利用して自己照合せず、保存Pauli matrix／保存DF Uからのnative occupation作用を用い、
signed product・ideal involution rotation式を同じbinary64 event angleで時間順に合成する。
これは選定した**具体的native program**の数値診断であり、全law平均・新physical信号・compiledρの評価ではない。
full controlled bodyをphase alignmentせず比較し、wrapperは小規模人工例でcosine/+Imの符号を確認する。
compile後のQPYは科学的再現性のため保存するが、価格取得と全compiled operator保証は区別する。

## 4. 将来の数値・correctness政策案

complex128を主精度とし、basis/対角化残差、Hermiticity、bit順、actual adjointを検査する。
対称化・QR修復・phase alignment・sector射影・高精度solverへの自動変更は行わない。
基底や共通halfのoperator比較はStage 1/2と同じengineering式
`τ=16γ_d max(1,||A||₂,||B||₂)`、`γ_d=dε/(1−dε)`、`ε=2^-52`を第一案とする。
対象次元dは8/256/512を区別し、PASS `≤τ/4`、UNRESOLVED `τ/4超〜4τ`、FAIL `>4τ`。

長いF/E programにはこの1block閾値を無条件に適用しない。
構成block/eventの数値residualとroundoffを、
`e(AB)=||A||e_B+||B||e_A+e_Ae_B+γ_(8d)|| |A||B| ||_F`で順に伝播する。
basis reuse前後もactual adjoint残差を含め、丸められたmatrixのnormを1と仮定しない。
独立oracleとの差を、両経路のengineering不確かさの和Uと比較し、
PASS `≤U`、UNRESOLVED `U超〜4U`、FAIL `>4U`を提案する。
native preflightの非PASSは価格compile前STOP。数値分解能不足を大きな固定1e-10の採用だけで隠さない。
SciPy/Qiskitの丸め診断をrigorous certificateや全lawρ上界とは呼ばない。

Stage 2 finite bias未解像をこのpreflightの失敗に読み替えず、再高精度化を必須にしない。
逆に、Stage 2 nominal一致からGaussian native circuitの一致を省略しない。

## 5. 再利用するtestsと未実行の確認事項

下記testsはSOURCEを読んだだけで今回再実行していない。

| テスト | 将来再利用する検査 |
|---|---|
| [test_df_partial_s2.py][TS2] `test_shared_diagonal_primitives_match_existing_and_controlled_unitaries` | 一体RZ/RZZとglobal/relative phase、controlled branch |
| 同`test_identity_policy_one_body_and_constant_variants_match_dense` | T内identity・外側scalarの一回適用 |
| [test_df_rte_qiskit_builder.py][TDF] `test_every_small_event_matches_dense_reference_with_all_signs_and_phases` | 全偶数order、signed products/rotation |
| 同`test_sequence_order_and_cross_event_basis_reuse_preserve_unitary`、`test_different_fragment_bases_are_not_cancelled` | actual inverseと同basisだけのreuse |
| 同`test_controlled_circuit_controls_only_the_central_action` | 無制御Gaussian basis、central-only制御 |
| [test_h4_d2_pauli_control.py][TD2] `CentralTests` | X/Y/Z、負rotation、product/order/global phase、little-endian、非可換順 |
| [test_fixed_signal_evaluation.py][TFIX] `test_hadamard_wrapper_measures_complex_signal_including_identity` | cosine/+Im sine wrapperとidentity phase |

新規人工fixtureは共通Gaussian保持とD2/DF residualを混ぜた小さなouter S2、
同じ保存eventからのF/E、R/Empty共有、軸間event再利用、全partsのphase所有を対象とする。
旧end-to-end testsがN/Gや新state/reference/分子生成まで実行する場合は一括起動しない。

## 6. 将来の件数・保存物・承認境界

| 単位 | 初期案の件数・再利用 |
|---|---|
| 共通保持basis | Tの8×8対角化1回、native lowering1組、両arm共用 |
| DF residual basis | 保存Uのnative lowering10組、固有値/係数再生成なし |
| 共有basis検査 | 11組のforward/actual adjoint/one-particle＋vacuum確認。gate内部の全call監査を新設しない |
| 共通E_T参照 | 同一T/half時間のexponential1種をcache。full-H reference/finite信号再実行0 |
| 保持検査 | system half、actual inverse、controlled halfの3対象＋scalar位相検査 |
| sampling | 方式別4seed、計8 trajectory×32＝256 event位置。一度取得して保存 |
| native body確認 | F8＋E8＋R1＋Empty1＝18 program。軸ごとに再計算しない |
| 価格target | F16＋E16＋R2＋Empty2＝36 wrapper。全target固定後にcompile |
| 追加科学 | 高精度化、実効law平均、母平均費用、shot/N/G、新state/reference、別tupleは0 |

これは科学対象の数であり、matrix積・local gate作用・norm等の内部callが同数という意味ではない。
実装時に重複targetをcacheし、実際のbuild/compile/cache数と失敗対象を記録する。
独立taskは空き資源に応じて並列化し、共通basis/E_T/stateは共有・cacheする。
CPU/RAM/workerに旧runの上限を引き継がず、共有環境・他jobを変更しない。

将来の最小保存物は、新SOURCE/runtime/input来歴、native basis/共通halfの記述とpreflight、
方式別law recipeと8本の全event、F/E/R/Empty input/compiled QPYと対応metrics、
REPORT/終了状態/STOP。大きな同一入力や全局所matrixを重複保存しない。
新しいstage3 namespaceへ分離し、旧結果・grant・STOPを更新しない。

ユーザーにまだ必要な判断は、各arm4本・36compile・上記preflight/精度政策を含む初期scopeを実行するかどうか。
具体的なSOURCE SHAは承認後の最小実装・人工検査を経て初めて固定する。
現時点で新しい科学SOURCE、runner、law JSON、QPY、費用表、数値PASSは作っていない。
今回の文書静的検査はlink/固定hash/diffのみで、作者自身の確認を独立した科学レビューと呼ばない。
固定commit上の参照先21件と保存ファイルSHA256 7件を通常のGit/stdlib読取で照合した。
数値operatorや科学libraryを起動して得たPASSではない。

将来の非PASS/失敗は`STOP_STAGE3_PREFLIGHT_OR_EXECUTION_FAILURE_NO_RETRY`等で原因と出力を保持する案。
完了時も`STOP_STAGE3_FIXED_COHORT_COST_COMPLETE_NO_AUTOMATIC_FOLLOWUP`で終了する。
**今回の準備完了後はmandatory STOP。Stage 2 STOPを保全し、別の明示許可なしに回路生成・sampling・compileを起動しない。**

[INPUT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/source_and_inputs.json
[KERNEL2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4bfbbaad1f8178077489bf06b085377256d6fb80/src/trotterlib/matched_partition_stage2.py
[OPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/df_trotter/ops.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/df_rte_tail.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/df_rte_qiskit.py
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/pauli_signal_adapter.py
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/h4_d2_pauli_control.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/rte.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/df_partial_s2.py
[REPEAT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/df_partial_s2_repeated.py
[FIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/fixed_signal_evaluation.py
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/rte_compiled_cost.py
[PARTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/src/trotterlib/pauli_mechanism_cost.py
[TS2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/tests/test_df_partial_s2.py
[TDF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/tests/test_df_rte_qiskit_builder.py
[TD2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/tests/test_h4_d2_pauli_control.py
[TFIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/tests/test_fixed_signal_evaluation.py
