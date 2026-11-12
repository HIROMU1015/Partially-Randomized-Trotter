# B/C内部HANDOFF：C1状態監査とStreaming consumer

[REPORT](REPORT.md)／[STATE_AUDIT](STATE_AUDIT.json)／[科学案](SCIENCE_PLAN.json)。実装SOURCE `c1135fdb9a021e85ffabdbf4ef56098ec3ac97e6`、基点`2a882fc9d17ed5e32f5581d0f220514daad0235a`。新9＋旧3＋sampler API1＝13件PASS、実分子S/平均価格/再解析/compile/fit/N/Gは0。

## 確認状況

- **確認済み**：元C1 `structural`とAccountingをそのまま呼ぶforward-only consumer。人工入力で全finish出力・gate列・phase/featureが一致し、同じcounterをpiece/outer間で保持、terminalは一回。
- **未確認**：実C1の実用的な閉状態、平均S・平均compiled価格、実分子用producer/readerの全接続とmemory、長q/R・全lawの資格。
- **成立しない一般変換**：basis/nonzero flagだけの縮約、角度のmodulo周期同一視、chunkごとのtail呼直し/phase再括弧化、途中finish。反例は人工source入力であり、実分子supportでの到達証明ではない。

固定数のfloat registerを保持できることは、小さいMarkov行列の存在を意味しない。逆に、今回の反例は特定のfrozen H4/H6 grammarに有効な別の縮約まで否定しない。物理閉包を確認したと扱わない。

## C内部の追加API

```python
from trottertracks.mean_compiled_price import stream_structure

result = stream_structure(
    original_Accounting_factory, original_signed_blocks, candidate_row,
    scalar, axis, tail_batch_iterator, allocation=original_integer_allocation,
)
# result["accounting"]: 元counterによる一path S/feature/phase
# result["quantity"] == "SINGLE_PATH_STRUCTURAL_PROXY"
# result["mean_structural_price"] is None
# result["mean_compiled_price"] is None
```

これは価格providerではなく、CのS計数内部用consumer。実入力のS計数は別の科学実行許可が必要で、現在は人工fixtureにのみ使用した。array/dictの新fitやSOURCE lock解除、法則/primitiveの生成は行わない。

各legal tailを最後まで一つのiteratorとして渡す。transport chunkは`chain.from_iterable`等で元順に連結し、chunkごとに`tail()`や`finish()`を呼ばない。元の関数がhead/tail/signed piece/outer/scalar/wrapper/terminalを処理する。allocationに対する過不足や未消費は例外で、平均/価格へのfallbackはない。

`stream.consumer_materializes_events=false`はconsumer自身の性質。producerがlist/一括JSONを持っていれば全体memoryは減らない。元`rte.iter_sample_rte_events`と`df_rte_circuit.iter_sample_events`は再利用可能だが、元request/dataclassの検査を維持するS専用leaf、または保存rawのincremental readerは未接続。

新helperはC1のpure `structural`を既存`r_price_phase1_science`からlazy importする。Bがscience backend/旧CLI一式を取り込む必要はない。Bの現provider統合はこのhelperを呼ばず、内部科学leafを使う場合はC側でSOURCE/依存を固定してから連携する。Bへ共通controller/候補管理/集計の差分は要求していない。

## 既存平均法・providerは不変

MeanKernel/MeanPiece/full_wrapper_mean、local_residual_mean/combine_mean、MeanPriceProviderの本体は基点と同一。closed-state人工例や内部局所期待formを、今回の物理閉包の証拠に流用しない。旧connected-clusterの別DF・uncontrolled係数を移植せず、Sへraw κ(C)を加算しない。C−Sの局所modelを選ぶ場合もPF/RTE・piece・outer・wrapperの残差と新holdoutが必要。

新consumerのpath SをBへDIRECT compiledや全law平均として渡さない。C1係数/資格false・有限cohort・共有不確かさ・missing/FAILED・κ schedule use-contextは変更なし。Bの最終RMSE/M/κ controllerには介入していない。

## 次の科学候補と必要条件

S0a案は保存normal path index0を4セル（H4 q8/R8、q16/R8、H6 q1/R32、q2/R32）選び、X/Y計8件の新stream Sを保存baseline S・digestと比較する診断（旧counterの新再計数は0）。新compile/sampling/fit/N/Gは0だが、科学再計数・再解析なので**未承認・未実行**。path reader/検査leaf SOURCEの固定と、保存input/preparation/law/signed time/basis/normalizer/event metadataの検査が実行前に必要。

これは実装parityの最小診断で、平均の正しさや全support閉包を保証しない。既存C1の役割は原本のまま、新mean方式のdevelopmentデータとして位置付ける。残32 pathやfresh cohortへ暗黙に拡張しない。

平均Sの閉包候補を採るなら、frozen event supportの状態/reward同値性・法則とsource-order/精度の根拠を別に用意する。局所期待近似なら新しい仮説として境界/打切りと独立holdoutを固定する。新compiled価格はBの実physical候補domainでC−Sを検証する必要がある場合だけ選び直す。旧40/16/fresh32案を自動実行しない。

次：ユーザーが具体的な科学範囲を判断、Cが必要なleaf/SOURCEを固定、Bが自身の許可scope内で最終統合を担当する。今回は公開・remote照合後mandatory STOP。S0/C2・新平均価格・G評価へ進まない。
