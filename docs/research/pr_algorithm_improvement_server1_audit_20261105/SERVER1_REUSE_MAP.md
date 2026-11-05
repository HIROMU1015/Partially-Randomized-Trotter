# SERVER 1：再利用可能資産と版・条件の対応

2026-11-05 JST。以下は再利用候補の所在表であり、実行許可や新しい科学PASSではない。全資産の固定path/blobと関数・class・静的importは[CODE_ASSETS](CODE_ASSETS.csv)、原結果は[証拠CSV](SERVER1_IMPROVEMENT_EVIDENCE.csv)と[参照索引](REFERENCE_INDEX.csv)から辿る。コードはimportせずASTだけを検査した。

## 中心実装・runner・test

| 系列 | 再利用対象・主要関数 | 条件・依存・利用境界 |
|---|---|---|
| G6 return aggregation | `algorithm_codesign/return_aggregation.py`：`reduce_word`, `convolution`, `ReturnKernel`, `root_interval`, `dyadic_distribution`, `rational_event_weight`。`tests/tracks/algorithm_codesign/test_g6_return_aggregation.py` | 正有理p・Hermitian involution・奇数m等の元domain。abstract kernel、synthetic provider、H4 C1を分離。現在mainの当該fileは固定`28cfabb1d47e0e1824bce1e154a9c289738fa2b9`と同blob。これはkernel版の同一性で、全run SOURCEの同一性ではない |
| G7/G8 | `g7_generator.py`：`CanonicalGenerator`, `FullReturnGenerator`, `make_generator`, `budget`。`g8_pipeline.py`：`AcquisitionCache`, `NativePipeline` | fixed known provider、登録short-step/degree/budget。G7 SOURCE`ab2549f41b3546fb3940342a2162dd9ee93699c4`、G8 SOURCE`cb60a4a1336f1803ad49d890f4e74aafd6ad7c61`。G8の20 sequence hashはG7と一致したとの記録。独立再現としてpoolしない |
| G9/G10・native比較 | `g9_p5.py`：`P5Closed`, `audit_p5`。`g9_native.py`, `g10_generator.py`：`generators`, `static_support_bounds`、対応runner/test | 3-system-qubit synthetic・同degree有限target内の比較。pygridsynth等の元native primitive精度/strict phase/caching条件を継承。G9 v2 SOURCE`0ef2b92738750a3c0d187743dd4b7c9b927db802`、G10 v3 SOURCE`b9ed01455351628c9073748f5ba5751aa794b789`。authorization-only HEADとは別 |
| Parent / IS / RTE再配分 | `parent_allocation.py`：`digital_proposal`, `ParentAllocation`, `RootAllocationGenerator`, `reweight_events`, `event_statistics`, `error_accounting`。`parent_allocation_native.py`、`rte_reallocation/`と対応test | 既知Euler/ISとparent z/angle/priceの交換。S4 SOURCE`4ccc1832f63cd323b23fc05623f7e0b58cf34f37`。旧proxy目的と訂正KKTを混ぜない。同じ保存信号・価格を使う派生表は独立標本ではない |
| 混在辞書のexact入力 | `pr_dictionary_validation/algebra.py`, `fixture.py`, `prototype.py`：`make_fragment_alternatives`, `canonicalize_arm`, `make_tail`, `ordered_event`, `finite_events` | Bの保存入力とserializer/hashを使う。4 mode・2粒子・occupied(0,1)・L_D1・残差density-square二fragmentのsynthetic fixture、分子geometry/AO basis/DF rankは非該当。SS/SP/PS/PPとfaithful identity tailを維持 |
| controlled回路・参照 | 同packageの`circuits.py`, `reference.py`, `gates.py`：`compiled_wrapper`, `wrapper_signal`, `component_derived_mean`, `compose_corrected_mean` | 独立90/128digits参照、rtol0、phase fitting禁止、JW/ancilla順序を保持。旧metadata失敗版、F1修正版、F2旧compile版、位相保持修正版の4版をCODE_ASSETSで分離 |
| F1保存入力検証 | `f1_saved_plan.py`, `f1_saved_execution.py`、`scripts/run_pr_dictionary_f1_saved.py`、対応test | SOURCE`8192530b17d6bc4af195c734ff43a6e78c506942`。T1/5,q2,delta1/10一点,r2,h1/20,K2。登録correctnessとliteral DP、未資格fallbackを保持。testファイル存在と過去runner実行は別 |
| F2限定actual費用 | `f2_execution.py`：`measure_f2_circuits`, `aggregate_metrics`。`f2_resources.py`：`inspect_packet`, `execute_resources`、専用saved runner | SOURCE`9468c7089727f5648d9f59cddbaebe0a42f5f0a0`。同synthetic model、T1/5,q1,delta1/5一点,r2,h1/10,K0。全ordered trajectoryの保存期待1-shot資源。F1のK2 law/Phiを流用しない |
| F2統計経路 | `f2_preflight.py`, `f2_statistics.py`：`require_adopted_statistics`, `qualified_shots`、`intervals.py` | 限定RESOURCE経路と分離。η/α/γ/interval未資格を解除しない。保存N=nullを0へ変換しない。人工scalarテストのPASSをblanket certificateへしない |
| 保存寄与・Gaussian | `f2_attribution.py`、`rectangular.py`：`rectangular_gaussian`, `RectangularBasisProvider`、`rectangular_execution.py` | 帰属結果はF2保存caseの記述的分解。PS48 SOURCE`770e53addda63a03fa8c6c5d5168f7062067c572`は選択48-wrapper、同F2 baseline/referenceを再利用。一律適用の負結果と局所改善例を保持。新しい全F2再解析/compileは今回行わない |

`algorithm_codesign/`は`src/trottertracks/algorithm_codesign/`、混在辞書packageは`src/trottertracks/pr_dictionary_validation/`の略記。正確なrunner/test名はCODE_ASSETSと参照索引を使う。表の関数が存在することだけで、未掲載の科学条件で動作・改善するとはしない。

## 混在辞書の重要な版差

| 固定SOURCE | 保存された位置付け | 現在mainの`gates.py`との関係 |
|---|---|---|
| `31e8391a56fbb1a0e64f477f78e242655329de99` | F1旧run。metadata=Noneのsetterエラー、transpiler未到達 | 異なるblob。旧失敗の再現用SOURCEとして保持 |
| `8192530b17d6bc4af195c734ff43a6e78c506942` | metadata修正版F1。登録correctness完了 | 異なるblob。F1結果へ後続F2 helperを付け替えない |
| `9b1e103102f918bb157dea90c7eaf3acb5d88b6c` | 限定F2旧run。compile後operator 3件FAIL | 異なるblob。失敗原本・STOPを保持 |
| `9468c7089727f5648d9f59cddbaebe0a42f5f0a0` | 公式定義loweringで位相を保持した限定F2修正版 | 当該fileは同blob。共有環境変更やphase fitでの救済ではないという原報告を保持 |

比較はfile blob単位。package全体・依存環境・SOURCE closure全体の互換を推定しない。main内の過去SOURCE収録と、過去runの実行SOURCE・現在の実行採択を区別する。

F1/F2の固定compilerはQiskit1.3.0、opt1、seed17、`rz/sx/x/cx`、free topology、approximation_degree1、qubits_initially_zero=False。専用環境のCPython3.11.9、NumPy1.26.4、SciPy1.14.1、mpmath1.3.0等は元lock/provenanceへ戻る。今回環境を構築・変更・importしていない。

## 再利用する保存結果と独立性

Bの保存入力→F1、F1後の別K0条件→F2、同F2保存case→寄与分析/PS48という依存を[INPUT_COHORT_BINDINGS](INPUT_COHORT_BINDINGS.csv)に保持した。公開input hashは転記し、lawや行列を再生成していない。run ID未記録の選択fieldには`NOT_RECORDED_IN_SELECTED_FIELDS`を残した。

Track BのG8/G7 sequence再利用、Parent旧目的/訂正結果、M1-Sの保存信号を使うgamma/pi3/Stage S、C1のP5 worker回復は、各元報告とCSVの`successor_update`・`independence`を参照する。同じ値・同じfileのコピー、技術的fresh rerun、異なるmethod PRNG key、独立標本による確認を一種類にまとめない。

H4 C1の分子適用域はlinear H4、1Å、STO-3G、DF12、R1 generation-prefix L_D3、13 basis/324 labels、T4/5,q1,r4,m5, microstep1/5、各方式1024 whole-path、TAIL_ONLY_PLUS_SCALAR。logical費用と古典生成観測であり、strict native・deterministic halves・全wrapper・共通認証Nの再利用根拠ではない。[原資源証拠と限界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)。

非prefix RQ-02、Gaussian選択、sampling調査、既知compact/IS等価の資料は、既存案の重複調査を避けるため再利用できる。具体的な新split/sampler、科学的実行、一般的資源改善の証拠としては使わない。

今回のmetadata処理は既存カード・SOURCE索引・manifest・保存receipt・固定Git照合方式を再利用した。研究runnerは科学処理へ接続するため呼ばず、stdlibの限定adapterでCSV/JSON/AST/Git metadataだけを扱った。既存コード・入力・結果・grant・STOPを変更していない。**公開照合後mandatory STOP。**
