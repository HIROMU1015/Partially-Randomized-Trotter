# SCV最小実装・人工correctness

2026-11-07 JST。**IMPLEMENTED / ARTIFICIAL-ONLY。実H4の科学的妥当性・改善は未検証。**
設計正本は `84e021b6a8decde7fa3ed3fa44d7148cc5428061` の
[層内候補・設計](../intrastratum_cv_design_20261107/REPORT.md)。既存SOURCEを編集せず、新しいmoduleから再利用した。

## 新しいSOURCE

| ファイル | 機能・必要となった理由 |
|---|---|
| [scv_scalar.py](../../../src/trottertracks/algorithm_codesign/scv_scalar.py) | 保存schemaのbyte/hashとlabel→fragment対応、bitを受け取らない群選択、exact条件付き平均・二次moment、層別bank/model、h1のみfit、paired合成。旧bankはglobal平均・3-controlで、条件付き中心化とcell別係数を表現できないため追加。 |
| [scv224_pilot.py](../../../src/trottertracks/algorithm_codesign/scv224_pilot.py) | 注入型224-shot driver、既存groupへのcollector、独立domain、SOURCE照合、paired serializer、物理支出と手法参照費の区別、STOP。旧driverのrank4・全方式・pilot配分は使わない。 |
| [prepare_scv224_pilot.py](../../../scripts/tracks/algorithm_codesign/prepare_scv224_pilot.py) | `--show-plan` は草案表示だけ。`--run` は実入力・RNG・pool生成前に拒否する。科学launcherではない。 |

新しいcontrol/estimatorはscalar rotation-fragment群indicatorだけ。signed-support/product-label特徴、confidence calibration、N選択、割当最適化、管理システムは追加していない。

## 再利用した固定版

全てbase `84e021b6a8decde7fa3ed3fa44d7148cc5428061` に存在する版をそのまま使用。
これらの科学コードは752-shot結果 `3c8e55a2728f86a1eda17a2e0105050a637e7c86` からも不変。
個々のblob/SHA256は [SOURCE.json](../../../artifacts/scv224_preparation/2026-11-07/SOURCE.json)。

| 既存API | 今回の接続 |
|---|---|
| `ConditionalSampler` / `PathChoice` / `validate_batch` | actual 53-bit Q、products→rotationの末尾label、元5層の正質量を保持。群で取得層を再分割しない。 |
| `ControlBank.pattern_mass` / `LocalControl` | ν_hを既存cell indicatorとのjoint mass / p_hでexact計算。h1の群選択はlabel質量の有限和。u²=u、Σ_h=ν_h(1−ν_h)。全実path列挙なし。 |
| `fit_and_freeze` / `FrozenCV` | intercept OLS、binary64 slopeをそのままFractionへ凍結、exact range。新bankがglobal平均をν_hへ置換。rank2とD≤4は新adapterでfail-closed検査。 |
| `production_mean` / `PilotMeanPlan` | 元cell-ID、全5質量、quota、scope、stream分離、加重合成を再利用。model dispatcherだけ追加。 |
| `Domain` / `Streams` | 新purpose/campaignの下で12 domain/24子stream案。既存method tokenは取得規約としてのみ利用。科学PRNGは今回生成0。 |
| `Campaign.group` / worker / `SavedPrices` | h1 trainに取得tag A1、paired評価にB1-OL。新しい`LegacyCollector`は既に用意されたservices/streams/poolを受け取るだけ。実engine・compilerを今回接続/実行していない。 |
| `Runtime` / manager / `durable_new` / `reconcile` / `account_costs` | bit先行、STOP要求先行、排他invocation、worker回収・全tail分類を再利用。旧`run/execute/learn`は呼ばない。 |

A1/train tagの内部規約は直接B1-OL条件付き取得。SCVのtrainはh1だけ32/軸。
評価tagの`prod`はfresh評価を表す既存schemaであり、保証付きproductionではない。
旧reconcileのA1/B1-OL欄は**取得tag別会計**。SCV/pairedの手法結果と混同しないよう、別の`SCV_ACCOUNTING.json`と`*_PAIRED_RESULT.json`を出す。

## Controlと推定器

LAW/catalogのSHA256、H/preparation hash、全label対応、全5正層、a≠bを検査する。
正質量fragment IDを文字列の辞書順でcanonical化し、index iをLSB側bit iとした正整数maskの昇順を同点順とする。
非空proper subsetのν_1∈[1/8,7/8]、|ν_1−1/2|最小を選ぶ。未知bit・旧data・state・信号を引数に取らない。適格群なしはSTOP。
metadataのmutationもfingerprintで検出する。保存schemaをbindする関数はbyte引数のみで、実入力の自動読込機能を持たない。

h1だけ条件付きν_1で学習し、他4層のλは最初から0。X/Yを独立に一回ずつfitし、両軸の凍結後に評価を開始する。
同じfresh評価recordsからraw BとZ=B−λ_h(u−ν_h)を作り、各々Σp_h平均とΣp_h²s²/n_hを計算する。
paired covarianceおよび差のsample-based分散も保存する。旧3-controlは各h内定数なので、条件付き中心化後は全標本で0になる。
これはfixed-n、fresh評価を完了するモデルでの不偏性であり、中断実験を捨てた条件付きcoverageまで保証しない。

## 人工correctnessの結果

- 新規 **34件PASS**：[最終log](ARTIFICIAL_TESTS_FINAL.log)。初回31件もPASSで保存した。SOURCE変化・finalizer失敗・非零paired差の検査を追加したため、最終版を再検証した。
- 再利用 **72件PASS**：[既存テストlog](REUSED_ARTIFICIAL_TESTS.log)。CV、conditional law/facade、worker/journal、pilot合成、managed runtime。合計106の異なるtest case。独立再現/CIではなくSERVER1 local人工検証。
- 人工LAWの全path厳密列挙で全5層のν・二次moment・Σを比較。h0定数/h1変動、range、同点、zero質量、空群、入力identity、正しい条件付き中心化と不偏性を確認。
- rank2/非finite/range停止、inactive補正0、係数凍結、fresh stream、全5層quota、非零paired共分散、旧control補正0を確認。
- toyの224件は**注入した架空bitと価格**。実量子測定224件ではない。train64+eval160を一回課金、cache hitも全RZ課金、baseline eval参照とSCV全費用を区別した。
- identity/stream/価格欠測/重複/rank/worker/finalizer異常で受信dataを保全し、STOPを回収前に保存、再実行を拒否。CLIからscience loader/PRNGが呼ばれないことも確認。

再現コマンドは [TEST_RESULTS.json](../../../artifacts/scv224_preparation/2026-11-07/TEST_RESULTS.json)。
旧SOURCE2913ファイル（README索引2件のみ追記）および旧結果の保全結果は [PRESERVATION.json](../../../artifacts/scv224_preparation/2026-11-07/PRESERVATION.json)。

## 実H4への残る接続

実H4のS、ν_h、Σ_h、相関、rank、λ、分散、RZ、所要時間は全て今回UNKNOWN。
実H4を一度も`metadata_from_bytes/select_fragment_control`へ投入していない。保存752-shotの新規数値再解析も0。

次回は草案採択・明示科学許可後、既存saved-service factory、固定compiler、独立stream、空output/cache、qualified managerへ薄いactivationだけを接続する。
実group/νを取得してmetadata identityをtrain前に凍結し、既存engine/環境/inputのbyte lockを公開SOURCEへ含める。
現在の注入型driverは人工dataで完走済みだが、実bundle/長時間engine/manager経路での新SCV完走は未確認。

古典会計はdriver wall/fit/合成、旧workerの生成/engine/price/cache/compile時間、worker RSSとmanager記録を別々に使う。
旧`TrialCost`のclassical=0 placeholderを測定値として報告しない。metadata setupとlaunch全体wall/RSSはactivationで外側timerへ接続する必要がある。
未記録項はUNKNOWN。exact engine高速化やGPU/float置換は実施していない。
