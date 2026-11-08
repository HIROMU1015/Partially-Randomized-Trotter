# Track D（O5）：C6a保存q4・16 pathのcut選択構造調査

2026-11-08 JST。**Case A：全16本が中央cut=2を選び、固定規則による非中央cutの使用機会は観測されなかった。** この保存q4条件では候補Bの追加H4 compileを一旦保留し、候補C（head-tail接口融合）の技術検証を次の優先候補とする案を提示する。候補Cの実装・科学実行は開始していない。

## 対象と来歴

Linear H4、隣接距離1.00 Å、STO-3G、Nα=Nβ=2、元Hamiltonian H、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、各step h=δ=0.1、q=4。元H厳密基底状態を仮定し状態準備費を除外する元条件を維持するが、今回state-vectorやHamiltonianの数値計算は行っていない。保存calibration cohort `track_d_c6a_q2_q4_paired_price_pilot_v1` の `q4_sample00..15` 全16 whole pathsが対象であり、欠損・代替path・追加samplingはない。

- C6a実行SOURCE：`6b6667026ec8b670f610e41141d195883a9eb513`。
- C6a公開索引：`e9d4e2f602d5f9e014507ca7e391719bab017b88`。
- 固定selector：`a54d04059bfc61adef43aa2d40fb9c001b62378f` の `track_d_cut_selection.py`。`track_d_cut_artificial.py`も同commitのbytesを保全したが、人工回路の実行入口は呼んでいない。
- 保存recordの適格性確認：`2f316fb229149117177367720a6eaa5aa8a94b76` の `track_d_cut_h4_saved.py:checked_record`。同commitのsample00 decisionを再利用した。
- 固定特徴：`64072ba81bb70db1df1b34e94adb08e78926f24b` の `track_d_c7_4_models.py:event_features`。module全体SHA-256は `6045e9a1b750494e8863928d119dc5b88448ad259936d0f4870bedec1dbdc33f`。他のモデル・価格関数は呼んでいない。
- 今回の実行SOURCE：`02430298c8bec8fafa6caa32d9d72c6b14803ca4`、tree `b81e0712df72e4e0ea232100c2d44041aa78fc74`。実行前[PLAN](PLAN.md)・runner・人工テスト・固定input索引を公開し、remote/tree/9ファイルのbytes一致を確認後、一回だけ実行した。最終公開commitはこのSOURCEを親とする結果・文書の追補であり、実行コードは変更しない。

原本はSERVER2の本人専用C6a保存領域でread-only使用し、SERVER1への転送は不要だった。各 `paths/q4_sampleXX/circuit_payload.tar.gz` の全体SHA-256・size、20 memberの集合・件数、`events.json`のsize/SHA-256を公開索引と照合した。さらに保存SOURCE・seed・sample ID・cohort・endpoint・packet・law・4×32 event/K0のbindingを既存関数で確認した。全16本が適格、欠損0、不一致0。

[固定input/hash索引](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/technical/fixed_inputs.json)にはC6aの20公開索引、sample00既存4記録、全16 archive/eventsのidentityを記録している。原本の内部絶対path・ordered eventは公開しない。

## 固定規則と再利用方法

各complete stepを0始まりのi=0..3とし、固定特徴のbasis-run数をrᵢ、support workをwᵢとして、lᵢ=1+rᵢ+wᵢ。`event_features`の既存意味論では各stepでbasis履歴をリセットし、application順のbasis変更を数え、supportごとにmax(1,2×support長−1)を加算する。event/application順序・符号・rotation値・既存の入力意味論を変更しない。

cut c=1/2/3は最初のc complete stepsの直後、中央は2。S(c)=max(Σᵢ<c lᵢ, Σᵢ≥c lᵢ)+1を最小化し、同点は中央に近いcut、さらに同点なら小さいcutを選ぶ。係数、候補、tie-breakは固定版のまま。価格・compiled depthを選択器へ渡さず、選択結果を見てfixtureや規則を変えていない。

既存のpayload readerはQPY/NPZのbytesも全て返すため、今回の薄いrunnerではarchiveのidentityを確認してeventsだけをextractし、既存 `checked_record` とselectorへ接続した。QPY/NPZのロード・deserializeは0。sample00はevent/descriptor hash、規則、SOURCE bindingが既存decisionと一致したため、cut選択を再実行していない。旧decisionに別保存されていなかったrᵢ/wᵢのみ一度取得し、負荷117/110/118/117と照合した。残15本は各一回のselector内部の特徴結果をcaptureし、二重の特徴計算を避けた。

## 全16 pathの結果

X/Yは同じwhole-path metadataを共有するため、判断単位はn=16である。Δcut=選択cut−2、表末尾はmin(S(1),S(3))−S(2)。正の値は中央のscoreが小さいことを表す。**これらはdescriptorの無次元proxyであり、回路depthではない。**

| Sample ID | r₀/r₁/r₂/r₃ | w₀/w₁/w₂/w₃ | l₀/l₁/l₂/l₃ | S(1)/S(2)/S(3) | 選択cut | Δcut | 選択tie | 非中央−中央 |
|---|---|---|---|---|---:|---:|---|---:|
| q4_sample00 | 22/15/23/20 | 94/94/94/96 | 117/110/118/117 | 346/236/346 | 2 | 0 | 無 | 110 |
| q4_sample01 | 23/15/23/24 | 94/92/92/94 | 118/108/116/119 | 344/236/343 | 2 | 0 | 無 | 107 |
| q4_sample02 | 23/25/18/23 | 94/94/94/94 | 118/120/113/118 | 352/239/352 | 2 | 0 | 無 | 113 |
| q4_sample03 | 21/24/23/21 | 92/88/94/90 | 114/113/118/112 | 344/231/346 | 2 | 0 | 無 | 113 |
| q4_sample04 | 28/21/15/20 | 92/94/94/90 | 121/116/110/111 | 338/238/348 | 2 | 0 | 無 | 100 |
| q4_sample05 | 21/22/21/22 | 90/94/96/90 | 112/117/118/113 | 349/232/348 | 2 | 0 | 無 | 116 |
| q4_sample06 | 21/19/19/20 | 92/94/92/94 | 114/114/112/115 | 342/229/341 | 2 | 0 | 無 | 112 |
| q4_sample07 | 21/20/22/23 | 92/92/96/96 | 114/113/119/120 | 353/240/347 | 2 | 0 | 無 | 107 |
| q4_sample08 | 24/23/23/17 | 94/94/92/92 | 119/118/116/110 | 345/238/354 | 2 | 0 | 無 | 107 |
| q4_sample09 | 25/21/18/22 | 94/92/96/96 | 120/114/115/119 | 349/235/350 | 2 | 0 | 無 | 114 |
| q4_sample10 | 24/20/20/21 | 92/90/92/92 | 117/111/113/114 | 339/229/342 | 2 | 0 | 無 | 110 |
| q4_sample11 | 18/25/25/23 | 94/94/94/90 | 113/120/120/114 | 355/235/354 | 2 | 0 | 無 | 119 |
| q4_sample12 | 25/21/24/22 | 92/92/96/92 | 118/114/121/115 | 351/237/354 | 2 | 0 | 無 | 114 |
| q4_sample13 | 22/19/16/17 | 96/92/96/96 | 119/112/113/114 | 340/232/345 | 2 | 0 | 無 | 108 |
| q4_sample14 | 21/14/25/17 | 96/92/94/92 | 118/107/120/110 | 338/231/346 | 2 | 0 | 無 | 107 |
| q4_sample15 | 22/18/23/19 | 94/94/90/96 | 117/113/114/116 | 344/231/345 | 2 | 0 | 無 | 113 |

全行の最小scoreは一意にcut2。最小値での選択tieは0本。sample00とsample02ではS(1)=S(3)だが、いずれも非最小候補同士のtieであり、中央へのtie-breakで選ばれた結果ではない。

| 集計量 | 保存16本での結果 |
|---|---|
| cut1 / cut2 / cut3選択数 | 0 / 16 / 0 |
| 非中央選択 | 0/16（0%）、該当IDなし |
| 最良非中央score−中央score | 最小100、最大119、平均110.625 |
| 各path内のstep負荷max−min | 最小3、最大13、平均7.375 |
| 中央の左右負荷差の絶対値 | 最小0、最大16、平均5.25 |
| 中央に対する選択score改善 | 全16本で0 |
| 最小値のtie / 任意の候補間score tie | 0本 / 2本 |

4 stepの負荷は表の範囲で近く、2+2分割に対して1+3分割へ移すと、重い側に3 stepを置く不均衡が増す。今回の固定proxyでは、それを覆して非中央を選ぶ偏りはなかった。人工2-mode q6の不均衡例と条件・event構造・qが異なり、その人工depth削減をこのcohortへ移送する根拠にはならない。

sample00の負荷117/110/118/117、score346/236/346、選択2は[既存報告](../track_d_cut_h4_q4_completion_20261108/REPORT.md)と一致する。既存の「選択−中央の追加depth削減0」は過去報告の結果であり、今回はその価格を再解析していない。他15本の新しいcompiled depth差も測定していない。

[全pathの科学scalar・identity](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/paths.json)／[集計](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/summary.json)。詳細descriptorは専用private領域に保全し、公開は非復元的なstep scalarとhashに限定した。

## 技術チェック・処理数・保全

[人工テスト](../../../tests/check_track_d_o5_q4_metadata.py)は5件PASS。event-only extract、hash不一致停止、固定selectorの人工非中央例/tie、feature captureと価格非依存、ID順の次候補選択を確認した。これは当人のローカル技術チェックであり、独立科学レビュー・immutable CIではない。

| 今回の処理 | 実施数 |
|---|---:|
| 保存event metadataの適格性確認・解析 | 16 paths |
| 既存cut decision再利用 | 1 |
| 新しいcut選択 | 15（各path一回） |
| 固定step特徴関数 | 64（16×4） |
| sampling / H4 build / raw simulation / P0 compile / postcheck | 各0 |
| QPY load / NPZ load / 保存compiled価格再解析 | 各0 |
| model fit / shot再算定 / 全RPE更新 / retry | 各0 |

[実行SOURCE](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/SOURCE.json)、[呼出数](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/CALL_COUNTS.json)、[前後照合](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/PRESERVATION.json)を保存した。C6a archive/source/公開索引/旧STOPとsample00記録、今回の固定コード・input索引は前後一致。原本・既存結果・grant/ledger/markerを変更する処理はない。既知の対象だけを読み、低い資源上限を設けず、小規模metadata処理を完了した。共有環境・venv・OS設定、他ユーザーやq64のprocessには変更・介入していない。SERVER1の結果を変更せず、raw/event/QPY/NPZは追加公開しない。

## 技術判断・未確認事項・次の担当

**Case Aを採る。** この保存q4 cohortでは、固定規則が改善候補となる非中央cutを選ぶ機会は0だった。従って候補Bの追加実H4 compileをここで増やす根拠は弱く、一旦保留する案が妥当である。候補Bの数学的可能性や、別のpath/qでの有効性を否定する結果ではない。

今回の16本は既存calibration cohortであり、新しい独立holdoutでも全sampling lawのsupportでもない。母集団の非中央選択確率や全q一般性の保証は与えない。score差はcompiled depth差ではなく、実機routing・合成精度・T費用も未評価。人工観測を含め、一般的な追加性能・新規性を認定しない。

次の担当は**ユーザー／GPT**。Bの追加H4検証を保留し、候補C（head-tail接口融合）を次の技術検証対象にするかを判断する。Cを採択する場合の最小案は、既存StrongBuilder・whole-path境界処理の未融合接口を限定し、既存対照の合法な融合を維持したまま追加の融合余地と必要な人工correctness条件を固定すること。具体的な実装・人工/実H4科学検証の範囲は別指示で定める。今回はCの設計詳細・実装・回路生成・compileへ進まない。Bを続ける場合も、このcohort内に非中央選択の次候補は存在しないので、別q/path・別規則を無断追加しない。

[STOP記録](../../../artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01/STOP.json)：`METADATA_COMPLETE_MANDATORY_STOP`。専用branch `codex/track-d-o5-q4-cohort-metadata-20261108` へ公開し、remote SHA・Git tree・公開bytesの一致を照合する。**O5は公開後mandatory STOP**。q64の進行中ジョブは別承認の管理下であり、本作業から停止・再起動しない。
