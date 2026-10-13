# 保存H4 RTE：基底変換費用の小規模内訳診断 — 2026-10-13

## 得られた値

固定した8列の**部分回路を別々にコンパイルした加法的費用**では、基底変換と逆変換がRZ数の **84.8266%**、CX数の **68.9655%**、全nativeゲート数の **85.6919%** を占めた。

これは同じ最終回路の最適化後ゲートを厳密に由来分類した割合ではない。各部分の合計と通常の全体compileには差があり、**「ランダム部分の最適化後の総費用の85%が基底変換」とは主張しない**。T数、shot込みPR総費用、母平均への一般化も未実施。

| metric | 基底変換・逆変換 | 中央作用 | その他 | 加法的合計 | 通常の全体compile | 基底割合（加法的） | 加法的合計の全体compileに対する超過 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RZ | 4,109 | 735 | 0 | 4,844 | 4,465 | 84.8266% | 8.4882% |
| CX | 1,120 | 504 | 0 | 1,624 | 1,572 | 68.9655% | 3.3079% |
| 全nativeゲート | 8,199 | 1,369 | 0 | 9,568 | 8,940 | 85.6919% | 7.0246% |

| 列長 | 保存列数 | 基底RZ／加法的RZ | 基底CX／加法的CX | RZ合計の全体compileに対する超過 |
|---|---:|---:|---:|---:|
| 3 | 4 | 1,592／1,841 = 86.4747% | 432／600 = 72.0000% | 5.4410% |
| 6 | 4 | 2,517／3,003 = 83.8162% | 688／1,024 = 67.1875% | 10.4450% |

全体の割合は8列の費用の**和の比**であり、行別割合の単純平均ではない。列長3／6を4本ずつ選んだ記述的集計で、実験全体のRTE分布やRPE scheduleによる重みは与えていない。実測の各行は[CSV](../../artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_v1/resource_rows.csv)と[rows JSON](../../artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_v1/rows.json)に保存した。

## 対象・入力同一性・手順

- linear H4、1.00 Å、STO-3G、8 system qubit、DF rank 12、`L_D=3`、`delta=0.02`、paired cutoff `K=2`。
- 9-qubit controlled RTE部分のみ。決定論PF、状態準備、readout、shotsは含めない。
- 既存WP06-bのholdout列長3／6それぞれのindex 0–3を、結果前に固定した。新seed・新geometry・新Hamiltonian生成はない。
- 保存NPZは3,589 bytes、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。既存入力の読み取り・基底の再構成のみで、分子計算はしない。
- 同じ旧seed全24本をmetadata復元し、Hamiltonian／partition／preparation／tailの4 identityと旧stream digest `20162aefd750a52ada4638b88965dc2a7615e811494c227b85f2e6a457f8f20f` の一致を確認してから8列をcompileした。新しいtrajectory標本を取得したものではない。
- 既存のdefault full-basis／隣接basis共有を維持。8列の36 eventは全てTaylor order 0。basis pair相殺25、実際の基底開閉22。希少order 2を観測した母平均ではない。
- Qiskit 1.3.0、NumPy 1.26.4、SciPy 1.14.1、Python 3.11.0、`rz,sx,x,cx`、optimization level 1、seed_transpiler 17、coupling mapなし。

診断subclassは既存builderのbasis／central append地点を観測する。既存共有・相殺後の命令をタグ付けし、同じroleが連続する区間をまとめる。隣接する異なる基底間の変換もbasis区間内の実際の順序で評価する。タグ付きraw回路が元builderの回路と完全一致し、区間を順序どおり再結合して同じ回路になることを確認した。通常の全体compileへbarrierや追加gateは入れていない。

各区間は同じcompilerで別compileする。同一run内のQPY bytesが完全一致する区間だけをcache共有し、出現回数を戻して会計した。旧費用の流用はなく、unique native compileは **24回**。上限96回以内で、追加compile・T合成・sampling最適化はない。

## 位相検査での停止と保存値からの照合

初回source `1b60265345ed0772ec7d8949e54889b5146e606f` は24 compile完了後、rawとnativeの厳密なstatevector一致検査で `TECHNICAL_INCONCLUSIVE` に停止した。[原failure](../../artifacts/rte_basis_cost_attribution/2026-10-13/result_v1/failure.json)と原出力・source・設定は変更していない。

その後はcompileを再実行せず、保存したraw／nativeゲート列だけから演算子を検算した。全8列の512×512演算子について、両ancilla枝とoff-diagonal blockを含む**一つの回路全体の共通位相**で比較した結果、最大elementwise残差は `2.392716424309625e-14`。7列は全体の符号−1、1列は符号+1で、枝ごとの位相合わせはしていない。7列のstrict operator差は約2のままで、strict PASSへ書き換えていない。compiler内部のどの処理が符号差を生んだかは特定していない。

既にcontrolledな回路全体に共通のglobal phaseは、最終density operatorやancilla X/Yの期待値を変えない。この限定で費用内訳を利用する。制御**前**のsystem unitaryの位相を捨てたり、このnative回路へさらにcontrolを付けたりする許可・保証ではない。既存controlled relative-phase policyや共有実装を修正していない。

最終保存値診断の分類は `SAVED_COST_ATTRIBUTION_COMPLETE_WITH_COMMON_GLOBAL_PHASE_CAVEAT`。[保存値summary](../../artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_v1/summary.json)を、元runnerのstrict成功結果と混同しない。

## 検証・資源・来歴

登録外observer／相殺／負符号・relative phase／割合集計テスト4 PASS、共通位相とancilla相対位相の区別・保存IR復元テスト3 PASS。最初のphaseテストでorder 0に不正なphaseを置いたfixtureが拒否された記録（3 PASS／1 FAIL）も保存し、合法なorder 2のfixtureへ訂正した。full test suiteは実行しない。

CPU32利用可能、空きRAM約51 GiB確認、CPUのみ。実compileは2 worker・各内部thread 1、保存演算子照合は1 process。same-run component cacheを使用。逐次版との速度比較はなくspeedupを主張しない。

- compileを含む初回outer process：wall 3.68秒、user CPU 5.32秒、system CPU 0.39秒、exit 1。
- 保存値演算子照合：wall 13.13秒、user CPU 12.52秒、system CPU 0.61秒、exit 0。
- 記録された最大worker peak RSSは416,333,824 bytes（約397 MiB）、保存監査peakは402,567,168 bytes（約384 MiB）。同時aggregate peakの実測ではない。
- per-process RSS 768 MiB／AS 2 GiB、最大3 process、native calls 96、出力16 MiB、log 1 MiB、各レビューfile512 KiBの上限を維持。phase／total wall capなし。

[結果前設定](../../artifacts/rte_basis_cost_attribution/2026-10-13/run_plan.json)、[入力identity](../../artifacts/rte_basis_cost_attribution/2026-10-13/result_v1/input_identity.json)、[compile inventory](../../artifacts/rte_basis_cost_attribution/2026-10-13/result_v1/compile_inventory.json)、[元process receipt](../../artifacts/rte_basis_cost_attribution/2026-10-13/original_process_receipt.json)、[保存監査receipt](../../artifacts/rte_basis_cost_attribution/2026-10-13/saved_audit_process_receipt.json)、[証拠manifest](../../artifacts/rte_basis_cost_attribution/2026-10-13/evidence_manifest.json)を参照する。全input／compiled IRは同artifact namespaceの小fileへ保存した。

## この診断から言える範囲

指定した短い保存H4列・default full-basis policyの**加法的RZ/CX会計では基底変換の寄与が大きい**。それ以上の新アルゴリズム採択、基底変更による節約率、H6/H8・他geometry・全Taylor次数・T費用・最終PR総費用の割合は認定しない。

Track Bの新規手法探索の一時休止と旧系列のSTOPは維持し、今回の独立した小規模費用診断から旧runnerを再開しない。追加計算を自動で行わず、ここでSTOPする。
