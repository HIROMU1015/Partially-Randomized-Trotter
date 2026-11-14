# 決定論PF shortlist：m3保存価格12軸のC受入

2026-11-15 JST。**m3の12/12軸を現行C configuration・physical candidate・X/YへSOURCE付きで接続し、既存C guardで受け入れた。保留0。** m4/m5較正24軸・m6独立holdout12軸は元receiptを参照し、直接価格の受入実績は**計48軸**となった。

今回の処理は保存記録のmetadata照合・consumer binding・guard受入だけ。新しいS/Cの計数、平均・残差再解析、fit、m9等の予測、実回路build/compile、MSE/N/G・winnerは0件。D1 Full-Hや他担当の進行を待たず、独立に完了した。

## 正本

- [ADMISSION.json](ADMISSION.json)：12行の判定、現行ID、旧ID/欠測、元SOURCE・trial/receipt selector、各軸fingerprint、scalar、保留理由（全件空）。元ALIGNMENTへのoverlayで、旧364行を書き換えない。
- [ADMISSION.csv](ADMISSION.csv)：12行、完全configuration ID・physical ID・axis・元locator・受入/保留。
- [PRICES.json](PRICES.json)／[SIDECARS.json](SIDECARS.json)／[GUARD_CHECK.json](GUARD_CHECK.json)：現在のC/B価格record形式と12件の個別guard結果。
- [PREVIOUS36_REFERENCES.json](PREVIOUS36_REFERENCES.json)：元24較正＋12holdoutの固定locator。価格値の再集計・モデル再判定はない。
- [HANDOFF.md](HANDOFF.md)、[TESTS.txt](TESTS.txt)：新7＋既存receipt関連5＝**12 test method PASS、skip 0**。

Cのmetadata証拠bundleは **`826f7c9230af7fbfbcc8501d236c44f5eefe2135`**。このcommitのIMPLEMENTATION_EVIDENCE.jsonを実guardが読み、各axisを個別に受入判定した。guardの変更・緩和はしていない。テストは作者自身の技術確認であり、独立科学レビューではない。

## SOURCEと既存成果の再利用

| 証拠 | 固定SOURCEと用途 |
|---|---|
| Dの同等性・coverage | `bcb67b2b2bb78776db1dd8d0625a3facb37cd298:docs/research/d_pf_allround_price_coverage_plan_20261115/{REPORT.md,COVERAGE_DELTA.json}`。6物理点・12axisの元locator、scalar/prep/fingerprint、既存66 metadata checkを再利用 |
| C候補表 | `0b4b2d941cd669aaa45231e1236bc2d3158a9d12:docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json`。Dのalignment_rowsを個別選択し、現行metadata plannerのcanonical/physical contextに照合 |
| C reader/guard | 既存M5受入reader、baseline/optimization planner、price_implementation_guard、paper_nominal_cost._tierを再利用。較正関数・会計関数は呼ばない |
| Stage 1 | `b190bd815c886446291fac8b281c3c7df7343372:docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json`。保存4＋新規20の元24軸を参照 |
| Stage 2 | `a55813373b84b9128cbd10c6812c0d26143a4280:docs/research/d_pf_stage2_holdout_20261115/run01/RESULT.json`。元12receiptと既存報告の12個別PASSを引用。予測/誤差/判定を計算し直さない |

開始時のC branchは`f0fc056464b63551d8f7795766bc0983d5863f95`でremoteと一致、清浄だった。同じm3受入は既存C結果・関連reader/test・直近履歴に未収録だった。Dの物理同等性調査と旧Full receipt監査を再実装せず、Cのcanonical mapping、source locator、受入guardとの差分だけを追加した。

## 12軸の受入結果

configurationは読みやすいprefixを表示し、完全IDはADMISSIONの各行とCSVに保存した。X=cosine、Y=sine。RZは保存値の転記であり、再計数・平均化ではない。全行m3、T=.8。

| route / formula | configuration | physical ID | axis | q / actual幅 | 保存RZ | 元receipt selector | 判定 |
|---|---|---|---|---|---:|---|---|
| Full S2 | 11cf8ca6 | e2b139ad6211e2c63ae6 | X | 4 / .2 | 31439 | PFC[129].prices.cosine | 受入 |
| Full S2 | 11cf8ca6 | e2b139ad6211e2c63ae6 | Y | 4 / .2 | 31439 | PFC[129].prices.sine | 受入 |
| B0I p7 S2 | a6e55777 | 318d7ec57316942b93d0 | X | 4 / .2 | 22359 | PFC[42].prices.cosine | 受入 |
| B0I p7 S2 | a6e55777 | 318d7ec57316942b93d0 | Y | 4 / .2 | 22359 | PFC[42].prices.sine | 受入 |
| Full Y4 | a799697f | 2594f35f92e76d58874c | X | 2 / .4 | 47053 | PILOT.records[0] | 受入 |
| Full Y4 | a799697f | 2594f35f92e76d58874c | Y | 2 / .4 | 47053 | PILOT.records[1] | 受入 |
| B0I p7 Y4 | defc5e88 | f144b792e8889063951f | X | 2 / .4 | 33385 | PFC[40].prices.cosine | 受入 |
| B0I p7 Y4 | defc5e88 | f144b792e8889063951f | Y | 2 / .4 | 33385 | PFC[40].prices.sine | 受入 |
| Full M5 | 70f7b317 | 1b3aa083ff69842d8f31 | X | 1 / .8 | 85928 | M5 CSV legacy_id＋cosine | 受入 |
| Full M5 | 70f7b317 | 1b3aa083ff69842d8f31 | Y | 1 / .8 | 85928 | M5 CSV legacy_id＋sine | 受入 |
| B0I p7 M5 | 424eecbf | 4bf5a2525d402abaad45 | X | 1 / .8 | 60950 | PFC[38].prices.cosine | 受入 |
| B0I p7 M5 | 424eecbf | 4bf5a2525d402abaad45 | Y | 1 / .8 | 60950 | PFC[38].prices.sine | 受入 |

元locatorの凡例：

- **PFC**：`0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c:artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/native_results.json.gz`。生成SOURCE `e2179f68f6979f3ad26392bff877fb422ddfb744`。元candidate tuple、各pricesのcompiler/prep/fingerprint、保存PASSを読む。
- **PILOT**：`3eca46148070ba4d4b0f5815ef45943407cc959e:artifacts/resource_applicability/track_a_h46_m5_tol_pilot_run01/2026-10-19/H4_TOL_PILOT_DET/H4_TOL_FULL_68d5c67e330e9c5f__0/result.json`。生成SOURCE `c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。元trial IDも保持。
- **M5 CSV**：`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb:artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/direct_measured_axes_92.csv`。`candidate_id=M5_MIN_e847f177d7169dd3`とaxisで一意に選択する。生成SOURCE `635ba563766875d688ad6912499b7f90397d0cc5`。元receipt hash、compiled hash、旧private trialのpath/SHAと公開completionとの対応はDの固定照合を再利用。Cはprivate trial/QPY/vectorを再開封していない。

JSON locatorのselectorはJSON key/index、CSVの`csv_selector`はheader名による選択、`csv_data_row`はheaderを除く0始まり行。現行canonical IDを旧recordへ書き込まず、新しいconsumer側mappingとして追加した。

旧trialには現在のRMSE canonical IDが存在しない。`legacy_canonical_id=null`を保ち、元candidate tuple・trial ID・SOURCE・locatorを保持する。旧cohort欄も全12軸で未記録のため、`original_cohort`と価格recordの`cohort`はnullのまま。EXPLORATION等の新しいcohortへ付け替えず、観測・標本数を新しく作らない。決定論receiptのn_paths=1は1固定回路の表現であり、sampling cohortや量子shot数ではない。

## Cで追加確認した条件と、Dから再利用した条件

**Cで確認**：configuration/canonical/physical IDと元recordのmethod・公式・prefix・T/q/actual幅、source/prep/hash/scalar、X/Yと個別fingerprint/RZ/compiler、元bindingのH/state/snapshot。既存price_contextとguardがfull controlled measured wrapper・P0、law/split/profile、scalar量・一回適用、source付きsidecar/固定evidenceを照合した。現在のtier関数も各軸`DETERMINISTIC_DIRECT`を返した。

Fullはone-body＋fragment0..9、p7はone-body＋fragment0..6、元順序のsigned S2 sweep。scalarはFull c_H=`2.29310124732`、p7 B0I c_H＋tail identity=`2.293146549384713`。新しいscalar/fragmentやHamiltonianは生成していない。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional、測定付きfull Hadamard wrapper・P0はDの固定実装証拠と現行profileを保持した。

**Dから再利用**：旧composeと現行composeの機能的同等性、signed weights/piece/fragment順の一致、旧入力containerの科学array一致、旧provider/wrapper/compilerの照合、Full M5 private原trial・config sealと公開receiptの同一性、保存作用gateの受入。Cで新しい作用norm・回路作用を検証したことにはしない。

旧callerの生成commit内には、外部providerのwrapperファイルそのものがない。初期の技術testでlocator差を検出したため、callerの`native_source/generation_SOURCE`と、D確認済みwrapper同等参照`af284d853890155bacd19e61850c196bd1d5c660`を分離した。sidecarの`wrapper_source_role`は同等参照であることを明示し、`wrapper_equivalence_source`からD正本へ辿る。元のnative policy名、生成SOURCE、価格SOURCE、trial・receiptを置換していない。guard条件の緩和や科学条件の変更はない。

Cの`native_policy=H46_LONG_TIME_NATIVE_COMPOSE`はDが同等性を確認したfamilyの識別である。原policy名は`origin.original_native_policy`と`implementation.original_native_policy`へそのまま保持し、原実装をこの名前で生成したと主張しない。

## coverage・役割・ロードマップ

| 既存/追加 | 軸数 | 扱い |
|---|---:|---|
| m3保存直接価格 | **12受入、0保留** | `SAVED_M3_DIRECT_OBSERVATION_NOT_CALIBRATION_OR_HOLDOUT`、model_group=null |
| m4/m5 | **24** | 元Stage1較正・元cohort/receiptをそのまま参照 |
| m6 | **12** | 元Stage2独立holdout・元報告のPASSを参照。新判定なし |
| 受入済み直接価格 | **48** | 6 configurationのm3～m6、各X/Y |

m3のactual幅はS2 .2/Y4 .4/M5 .8で、m4～m6の16h/7・16h/3・16hとは異なる。m3を較正・holdoutへ加えず、同じRZ値があることも同回路の根拠にしない。12 frozen model、CELL_PLAN36、Stage1/2原本、旧STOPは変更していない。

統合担当向けの更新は、Dの「m3は同等性確認済み・C binding待ち」を**「C guardで12軸受入済み、既存36と計48軸の直接価格」**へ進めること。ロードマップ本体は編集しない。短round m0～m2・36軸、長q m7～m12・72軸はDが示した未資格coverageのまま。m3価格受入はsignal/N・平均価格モデルの追加資格や全RPE比較の達成ではない。

**mandatory STOP。m9予測、再fit、追加compile、MSE/N/G・winner、Full-H/Pauliへ進まない。**
