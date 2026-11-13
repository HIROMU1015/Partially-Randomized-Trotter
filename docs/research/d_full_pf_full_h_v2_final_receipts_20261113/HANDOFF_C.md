# Codex C向け：Full PF保存compiled価格の確定状況と、新しい価格取得の条件

参照：[REPORT](REPORT.md)、[COVERAGE_V2.csv](COVERAGE_V2.csv)（`X_status`、`X_rz_count`、`Y_status`、`Y_rz_count`、`price_kind`、`price_carrier`、`compiler`、`X_previous_status`、`receipt_evidence`、`receipt_native_SOURCE`の列）、[RECEIPTS.json](RECEIPTS.json)。

- **価格の定義**：1回のRPE interrogation（controlled時間発展＋full Hadamard wrapper＋測定、P0）のRZ gate数。cosine（X）とsine（Y）は別々に判定している。
- **作業分担**：Full PF保存価格のSOURCE照合はDが担当した。Cは重複して照合しなくてよい。

## 1. 更新後の件数（136候補）

| 区分 | 件数 | 内容 |
|---|---:|---|
| EXACT（X/Yとも） | **73** | 内訳は下の箇条書き |
| CONDITIONAL | 0 | 前回の17件はすべて解消 |
| MISMATCH | 0 | – |
| SOURCE_UNAVAILABLE | 0 | – |
| 欠測（signalはある） | **63** | 下表 |

EXACT 73件の内訳：
- 長時間研究の決定論direct：49件（`155529e2`のexploration/extension）。
- pf_common T=0.8の再利用：24件。
  - `m5_dataset` receipt：7件。
  - 今回確定：17件。

片軸だけの保存はない。X/Yを互いに補完していない。

## 2. 17件のreceipt照合結果

| 区分 | 件数 | 確認したSOURCE |
|---|---:|---|
| H4 NEW_COMMON_NATIVE（q1/4/8、2nd・4th） | 6 | pf_common H4 `native_results.json.gz`（`0b1886f5`）。native SOURCE `e2179f68` |
| H6 AUDITED_COMPLETE_PAIR（2nd q1/4/8、4th q1/4） | 5 | pf_common H6 `native_results.json.gz`（`90f1cd55`）。SOURCE `b7662487`（inherited） |
| H6 NEW_RECOVERY（4th q8） | 1 | 同上。SOURCE `bdf6772a` |
| pilot q2（H4 2nd/4th/m5_best、H6 2nd/4th） | 5 | M5 pilot run01の`result.json`（`3eca4614`、SOURCE `c7d53f04`）。group IDは元の`pilot_group`で再計算 |

軸ごとに次を確認した。
- **候補**：system、method FULL、prefix 10/19、formula、q、T=0.8（binding）。
- **入力**：H/state/snapshot、compiler `1239304c…`（Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17）、control symmetric_directional。
- **回路・SOURCE**：wrapperは`include_measurement=True`、profile hashは一致、native policyは`M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`。
- **receipt**：result SHA（保存result.jsonのSHA-256）、pilot inventory SHA、fingerprint、gate check（Hadamard・compiled vector）、RZ数（保存値と一致）。

RZ数の再計数・再compileは行っていない。

## 3. X/YのSOURCEとcompiler

- **長時間研究のdirect価格（49件）**：`h46_long_time_native.compose`/`cost_trial`（SOURCE `27ebabdd2`ほか、`scientific_provenance.json`）。
- **pf_common**：`pf_common_run_native.cost_one`（`e2179f68`／`b7662487`／`bdf6772a`）。
- **pilot**：`h46_m5_native`（`c7d53f04`）。

上の3系統で共通している点：
- compiler hashは同一。
- 決定論loweringの呼出し（`build_deterministic_native(formula='2nd', scalar=0., symmetric_directional)`）はSOURCE本文で同一。
- prep hashは長時間研究のFULL trialと一致（H4 `bb736dc3…`、H6 `1de5325d…`）。

## 4. 保存価格を安全に再利用する条件

- **物理候補が一致すること**：system、FULL、全rank、formula、T、q。物理IDはDの`bind()`と同じ。
- **compilerとcost policyが一致すること**：compiler、cost policy `FULL_COSINE_SINE_HADAMARD_WRAPPER_MEASUREMENT_INCLUDED_P0`、control、scalar一回、H/state/snapshot。
- **再利用時に添付するもの**：V2側のidentity（`configuration_id`、`v2_candidate_id`、`identity`、`law_binding`）と、COVERAGE_V2の`receipt_trace`（追跡先、SOURCE、prep hash）。物理ID単独では再利用keyにならない。
- **使わないもの**：D-only・uncontrolled・局所cost、truncated PF（B0/B0I）、MODEL予測（H6 Affine）。

## 5. 価格欠測63件の候補（signalとreferenceはある）

| system | formula | 欠測（T: q） |
|---|---|---|
| H4 | 2nd（7） | 0.8:16、1.2:1/8、1.6:1/4、2.0:1/2 |
| H4 | 4th（9） | 0.8:16、1.2:6/8/16、1.6:1/16、2.0:1/10/16 |
| H4 | m5_best（15） | 0.8:16、1.2:2/4/6/8/16、1.6:2/4/8/16、2.0:2/4/8/10/16 |
| H6 | 2nd（8） | 0.8:16、1.2:1/8、1.6:1/2、2.0:1/2/8 |
| H6 | 4th（9） | 0.8:16、1.2:1/8/16、1.6:1/16、2.0:1/2/16 |
| H6 | m5_best（15） | 0.8:16、1.2:2/4/6/8/16、1.6:2/4/8/16、2.0:2/4/8/10/16 |

## 6. 保存grid外の価格不足

- T∉{0.8,1.2,1.6,2.0}のround、または保存されていないqのroundは、価格・signal・referenceがすべてない。
- 例（人工、h=0.1、δ=0.1）：m0–m2は保存grid外。m4（T1.6 q16）は、2ndだけ価格がある。
- Dの`for_packet()`では`OUTSIDE_SAVED_GRID_ALL_MISSING`と表示される。

## 7. 新compileが必要になった場合の元native SOURCE

- **回路build**：`h46_long_time_native.compose`（`requests=None`の決定論経路）。
  - pieceは`core.schedule(formula, q, 'PER_PIECE', 1, T)`で、loweringは`ax2b_native_df_v5.build_deterministic_native(formula='2nd')`。
  - scalar phaseはTで一回。
- **wrapper**：`build_native_hadamard_wrapper(axis=cosine/sine, include_measurement=True)`。
- **価格取得**：`cost_trial`。trial0で、low/high control、独立occupation vector、Hadamard期待値、compiled vectorを検査する。private `compiled_cache`は照合してから使う。
- **対象と承認**：上記5・6のうち選んだ候補だけを、ユーザーの明示承認後にcompileする。Full PFの公式・幅・M・比較menuは未採択である。
