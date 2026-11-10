# H4 DF10 price provider interface

実装：[`trottertracks.h4_df10_price_provider`](../../../src/trottertracks/h4_df10_price_provider.py)。Bの共通frameworkをimport必須にせず、`pipeline`引数で受け取る。保存artifact readerもBの`SavedArtifacts.read`を再利用する。

## 呼出しと実行境界

| API | 動作 | 今回の検証 |
|---|---|---|
| `H4PriceProvider(read_json, pipeline=None, selections=None)` | 固定6 JSONを読みcatalogを作る。平均・残差を再計算しない | 保存metadata照合 |
| `quote(request)` | 保存直接値／保存Mq値の選択。missはunavailable | 保存値・人工fixture |
| `provider(task, requested_rows)` | Bの`c_provider(task, rows)`。各candidateのX/Yを返す。lookupのみ | Bの作業中APIで静的planまで |
| `predict_path(request, descriptors)` | 既存V1会計本体＋固定bによる明示的な数値entry。lookupから自動起動しない | **人工/mockのみ** |
| `pipeline_record(task, row, axis, quote)` | lookup／数値entryのquoteをBのidentity/measurementで包む | 人工外挿も型を保つことを確認 |

`read_json(ref)`は`{commit: 完全SHA, path: repo相対JSONpath}`に対しJSON objectを返す。catalogはV1 `eec5bdf9…`、S1 `cefc0162…`、独立q `1cf87201…`の固定参照だけを使う。private inputやscientific runnerを読まない。

実H4で`predict_path`を新たに呼ぶこと、descriptorの新抽出・新law推定・新cohort平均は別の科学実行範囲。今回それらを実施していない。値を出せることと科学的に総費用へ採用できることは別であり、全RPE providerの正式採用は行わない。

## request

必須は`candidate, axis, binding, profile, contract`。`q_RPE`は別fieldで保持し、価格式に使うqは`candidate.q`（qPF）だけ。

- `candidate`：元canonical ID付きの`system='H4', method='B0'|'PR', formula='2nd', T, q, delta=.1, prefix, R_outer, K`。PFはp7/None/None、PRはp1/R4/K2。T/q=δ、正の整数qPFを要求。IDは元`h46_long_time_signal.cid`、B接続時はBの`candidate_id`を再利用。
- `binding`：保存H/state/snapshot、DF10/tol-only1e−8、geometry/basis/series/control/compilerを照合。`provider.binding`は固定metadataのcopy。T=.8など入力側reference metadataを新candidateの時刻へ読み替えない。
- `profile`：`PROFILE`のQiskit1.3.0、rz/sx/x/cx、opt1/seed17、元compiler hash、symmetric_directional、full_Hadamard_with_measurement、P=0を照合。別backend/target/couplingは対象外。
- `contract`：Bの`completion='saved_weighted_Hermitian_completion'`、`fragment_order='saved_snapshot_order'`、`state_preparation_cost=0`、`input_source`、`preparation_binding.snapshot/provider_source`を渡す。prep hashを明示した場合は固定PF/PR preparationへ照合。省略時も、固定snapshot/prefix/provider/orderのbindingから解決したhashを返す。新しい物理的同等性の証明ではない。
- `kind`：`direct_compiled`（既定）または`model_prediction`。kindをまたいで代用しない。
- `price_unit`：PFは`deterministic`（既定）。PRは`path`または`cohort_mean`を明示。`law_mean`は未接続としてunavailable。PRの回路族だけでは値を返さない。
- PRは`cohort`を明示。pathには`path_index,event_digest,digest_kind`も必要。q16は`canonical_events_sha256`、保存V1 q8直接pathは`private_events_file_sha256`。この二つを同じhashと扱わない。q16では`ordered_events`を与えれば順序込みhashも照合。q8のfile digestにpayloadを添えただけでは内容同一性を認定しない。

数値entryの`descriptors`は`binding,preparation_hash,source,scalar,blocks`。blocksは元V1 descriptorと同じone_body＋保存順fragment（PF8 block、PR2 block）。PRはさらに`ordered_events`（qPF batch×R4）、canonical digest、`event_source`、非負整数path IDを要求する。K2のorder0/2・application数/index/role/basis metadataを確認し、既存Accountingへその順に渡す。descriptor/eventの生成・物理的資格は上流の責任。providerはmetadataだけからH4作用を証明しない。

## 保存catalogの範囲

| 対象 | direct | Mq保存値 | 役割 |
|---|---|---|---|
| PF q8/q16 | 決定的各軸 | S1の保存値 | q8校正、q16使用済み確認 |
| PF q32 | 決定的各軸 | 取得前固定値 | 有限の独立q検証 |
| PR q8探索9/確認33 | path直接値・cohort平均 | cohort平均 | S1使用済み、cohortを選択して保持 |
| PR q16新33本 | path直接値・cohort平均 | path値・cohort平均 | 有限の独立q検証 |
| その他q | lookupはunavailable | lookupはunavailable | 明示数値entryは計算可能でも未検証外挿 |

q8のpath別Mqは元CSVを証拠参照とし、このJSON reader catalogでは再構成しない。いずれのPR cohort平均も全law期待値ではない。path値をcohort平均へ変換する機能もない。

## quoteとB record

quoteは`kind,value,price_unit,model,conditions,validation,qualified_for_candidate,source,model_provenance,uncertainty`等を返す。未取得値は`None`、不一致理由は`reason`。directでは`model=None`。`conditions`はordered raw payloadをechoせず、hash/path/sourceを保持する。

`validation`は`FINITE_INDEPENDENT_Q`（PF32/PR16）、`S1_KNOWN_DATA`（既知q）、`UNVALIDATED_EXTRAPOLATION`、`OUTSIDE_DOMAIN`を区別。directの資格は保存recordとprice unitの同一性、modelの資格は有限候補域の経験的検証だけを表す。`qualified_for_candidate=True`でもlaw期待値・一様誤差上界・全RPE正式採用を意味せず、`full_RPE_price_adopted=False`。

B recordは既存の`DIRECT_COMPILED_DETERMINISTIC / DIRECT_COMPILED_FINITE_SAMPLE / MODEL_PREDICTION`を使用し、値のないものは`UNAVAILABLE`、数値entryの未検証外挿は`UNVALIDATED_EXTRAPOLATION`。PR cohort平均は`expectation='FINITE_COHORT_ONLY'`、pathは`FINITE_PATH_ONLY`。PFのMqは条件付きmodelとして渡す。raw cohortは`source_cohort`、Bで選択するscenario keyは`cohort`（`DIRECT:`/`MODEL:`とpath ID付き、未取得は`UNAVAILABLE:`）へ分離する。callerは返されたscenario keyを使い、同じcohortを別の独立標本と数えない。

価格単位は一回の測定付きinterrogationのRZ gate countだけ。`normalization_is_price_multiplier=False`。幅・他resourceは未供給で、`peak_qubits=None`。Bのidentity/measurementを直接呼び、NやN×Cを実装しない。

## 不確かさ

`uncertainty.saved_validation`は選択cohortごとのMq保存summary（bias/RMSE/path SD/平均残差SEなど）のcopy。path別残差は元CSVへのselector付き参照。これらのtargetはMqであり、direct価格の誤差と混同しない。

校正係数の不確かさ、長q systematic、全law期待値、joint covariance、total CIは未評価の`None`。Mqにはmethod別の`shared_calibration_group`を返し、qやX/Yが違っても同じbを共有する。directにはこのfit groupを割り当てない。観測maxは上界ではなく、SEは係数/systematicを含むCIではない。αtotal=.05を価格の95%保証へ変換しない。
