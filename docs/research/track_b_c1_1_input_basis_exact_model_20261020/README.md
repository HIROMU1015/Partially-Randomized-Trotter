# C1-1 保存H4入力・basis・exact model 限定再解析

**結果：`REANALYSIS_INCONCLUSIVE`。1回実行・retry 0。保存NPZ/state/DF入力の同一性は確認したが、partition hashの不一致で停止した。basis成果物とG6 exact modelは未取得。公開・保全確認後mandatory STOP。**

これはCodex本人による承認済み限定再解析とその失敗記録であり、別reviewerによる独立再現ではない。G6の科学的GO、独立新規性、投稿GO、adapter実装GOを認定しない。

## 固定範囲と実行

| 項目 | 固定値 |
|---|---|
| Repository / branch | `HIROMU1015/Partially-Randomized-Trotter` / `track-b-c1-1-input-basis-exact-model-20261020` |
| C1-0基点 | `a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed` |
| 科学SOURCE参照 | `243eea8e840f3c472753f7935e510391626fd325` |
| 実行SOURCE commit | `d5f962b8b35f3912d45801f4d18998aa309feb0c` |
| 新authorization commit | `cb5dc85ca0386579dd79649ea0e3c72a751c7e6c`（SOURCE直後、receiptのみ追加） |
| Run ID | `track_b_c1_1_h4_saved_basis_exact_model_one_shot_v1` |
| 入力 | Linear H4、1.00 Å、STO-3G、DF rank 12、8 system qubits、生成順prefix 3 |
| snapshot | `artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz` |
| bytes / SHA256 | `8652` / `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| 保存結果参照 | `artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json` のV1 development / V2 rank 3 B2-G |
| 実行回数 / retry | `1 / 0`。旧grantを再利用せず、今回の明示承認に紐付けた |

[実行契約](execution_contract.json)、[新承認receipt](authorization_receipt.json)、[SOURCE manifest](source_manifest.json)、[環境identity](environment_identity.json)、[実行前静的確認](SOURCE_REVIEW.md)を実行前commitで固定した。`src/trotterlib`を含む156ファイルのbytes/hashを実行前後に照合。実行SOURCEと契約は失敗後も変更していない。

既存隔離環境 `h6-environments/20261012/native-dependency-v2/venv` を読取利用。Python 3.11.0rc1、NumPy 1.26.4、SciPy 1.14.1、Qiskit 1.3.0、qiskit-nature 0.7.2、OpenFermion 1.6.1。元S0に記録された主要7 package版と一致するが、当時のqiskit-nature版やBLAS実体まで同一と認定しない。Python実体および主要packageのMETADATA/RECORD hashを記録し、環境変更・package追加はしていない。

実行前はCPU affinity 64、MemAvailable `487008116 kB`、保存領域空き `431903055872 bytes`。CPU/RAM/workerに独自上限を設定せず、既存BLAS既定値を使用。逐次依存・最初の失敗でSTOPする経路を採用。後段の共通basis対角和cacheは未到達。所要wall約2.895秒、own CPU約16.017秒、peak RSS `363044 KiB`（Linux self、子processを含む総費用や性能benchmarkではない）。[実コマンド](result/launch.json)、[process exit](result/process_exit.json)、[資源観測](result/prelaunch_resources.json)。

## 到達点

| 対象 | 判定・証拠 |
|---|---|
| 保存registryの限定探索 | `MISSING_SOURCE`（今回探索範囲）。5つの既知artifact領域のJSON 8件に対象keyなし。[探索記録](registry_search.json)。全保存領域に存在しないという断定ではない |
| NPZ全体bytes/SHA256 | `VERIFIED`。読取開封1回、再保存なし |
| NPZ member / dtype / shape | `VERIFIED`。下表および[入力記録](result/input_identity.json) |
| constant・one-body・λ・12 DF factor・sector・full/sector state digest | 保存V1の全recordと一致、`VERIFIED` |
| state norm・sector支持・state対応・Hermiticity | 既存 `1e-12` 内、全残差0（固定float演算）。[数値チェック](result/state_model_checks.json)。ground state厳密性・物理準備可能性の認定なし |
| fragment index順 | deterministic `[0,1,2]`、tail `[3,4,5,6,7,8,9,10,11]` は保存値と一致、`VERIFIED`（後続停止への制御フロー到達で確認） |
| partition hash | `INPUT_IDENTITY_MISMATCH`。期待値 `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450` と不一致。実値は未保存 |
| basis再構成 | 既存 `prepare_df_partial_s2` はreturnしたが、直後のhash検査で停止。操作列/registryのserialize・比較は未到達、`BASIS_RECONSTRUCTION_INCONCLUSIVE` |
| 元basisのbyte/hash完全再現 | 元操作列未取得、`MISSING_SOURCE`。数値operator一致も今回は未取得 |
| component比較・label/sign/basis対応 | 未到達、`REANALYSIS_INCONCLUSIVE` |
| exact dyadic λ・p・positivity・和1 | 未到達、`REANALYSIS_INCONCLUSIVE`。保存floatのλを新exact λとして流用しない |
| 元DFモデルとの残差 / rigorous binding | 新残差なし、`MODEL_BINDING_UNCLOSED`。既存保存残差を今回の検証結果として再利用しない |
| T=0.8、q=1、r=4、m=5の候補short-step | 未到達。schedule未採択、全occurrence成立の認定なし |
| G6–DF adapter / native費用 / confidence閉鎖 | `IMPLEMENTATION_REQUIRED` / 未実行 / C1-0の`CONDITIONAL`を継続 |

| NPZ member | dtype | shape |
|---|---|---|
| constant | float64 | scalar |
| one_body | complex128 | 8×8 |
| lambdas | float64 | 12 |
| g_matrices | complex128 | 12×8×8 |
| sector_basis_indices | int64 | 36 |
| state_vector | complex128 | 256 |
| sector_state_vector | complex128 | 36 |
| metadata_json | scalar JSON文字列 | scalar（loaderが`.item()`で読取。dtype文字幅の独立記録なし） |

確認したHamiltonian hashは `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`、combined state hashは `31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd`。NPZ全体SHAと各配列のSOURCE定義digestを区別する。full/sector state、全factorの期待値・実値は入力記録に保存した。

## 失敗と原因の確度

一次失敗はrunnerのpartition hash等値検査（[固定SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d5f962b8b35f3912d45801f4d18998aa309feb0c/docs/research/track_b_c1_1_input_basis_exact_model_20261020/run_reanalysis.py#L139)）。[failure原記録](result/failure.json)、[stdout](result/raw_stdout.log)、[stderr](result/raw_stderr.log)、[構造化評価](failure_assessment.json)を参照。

固定SOURCEの `_partition_hash` はHamiltonian hashとindexだけでなく、各 `RankedDFFragment` のrank/lam/weight/weight_rule、tailのranking proxy和、policyもhash化する。[hash定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py#L210)。固定V1–V3も今回もgeneration順とexplicit ordered policyを使用する。[元経路](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py#L448)、[generation helper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_s0_s1_validation.py#L269)。

したがって派生float metadataやruntime算術の差は原因候補になるが、**原因は`UNRESOLVED`**。実際のhash/payloadを停止前に保存しなかった新runnerの証拠不足があり、どのfieldが違うかは特定できない。入力配列破損、operator不一致、特定ライブラリ版を原因と断定しない。取得のための再実行はしていない。

さらに失敗処理の最終stdout出力で、`say(stage, **state)` とstate内の`stage`が衝突し、`TypeError`が発生した。これは新runnerの不備。`failure.json`、`terminal.json`、当時点の`output_inventory.json`は既に保存されていた。outer processはexit 1。[terminal](result/terminal.json)はmandatory STOPを記録するが、stdoutに最終STOP行はない。原ログを改変して補わず、失敗後のSOURCE修正・再試行もしていない。

NPZ load 1、partition取得1、basis preparation 1、dense DF再構成0、exact model段階0という到達数は、原tracebackと固定SOURCEの直線制御フローからの判定であり、独立profiler計測ではない。内部BogoliubovTransformは承認されたbasis操作抽出に使った。新event/wrapper circuit、compile/synthesis、sampling、SCF、新DF分解、shot budget、native価格比較は実行していない。

## Exact modelの現状と後続の最小情報

実行前契約の予定定義は、保存component順の各binary64係数をexact dyadicとし、λ=Σ|c_i|、p_i=|c_i|/λ、signed involutionを `sign(c_i) B_ideal Z_support B_ideal†` とするものだった。**今回はそのモデルの取得・保証チェックを完了していない**。signed operatorの係数への二重適用、別operatorの同label化、同basis/同係数だけによる集約は認めない。

既存SOURCEはconjugationを `B Z B†` と定義する。[operator規約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py#L907)。後続には保存順label/component/fragment/support/sign、basis ID/hashと操作列、deterministic constant/identity位相 `exp(-it c)`、G6語の適用順・relative phaseの接続が必要。有限float行列を保存してもexact unitary証明や元DF Hamiltonianのrigorous norm bindingにはならない。

後続adapterへ渡せる完成basis registry、exact入力、モデル差certificateはまだない。Gate 1は保存入力側のみ閉じ、実装準備全体は`CONDITIONAL`。今回の失敗から研究価値の否定や実装GOを結論しない。

## 保全・公開・引継ぎ

[保全検証](preservation_verification.json)で固定156ファイルと非公開レビュー原本2件のbytes/hash不変を確認。既存SOURCE/raw/grant/claim/ledger/marker/STOPを変更せず、原本を公開していない。変更は専用namespaceへの追加のみ。他worktreeや共有環境の設定・processに変更なし。

[run output manifest](run_output_manifest.json)は原ログ込みの実行出力11件を収録する。runner自身のoutput inventoryはraw logsを外側でcopyする前の終端時点の記録として保持する。最終公開HEADと全新規ファイルのbytes/SHA256照合は `publication_receipt.json` とその後の最終確認で記録する（自己参照SHAを作らない）。

次の担当・承認は [handoff.md](handoff.md)。本packetは既存STOPを解除しない。**公開・保全確認後mandatory STOP。**
