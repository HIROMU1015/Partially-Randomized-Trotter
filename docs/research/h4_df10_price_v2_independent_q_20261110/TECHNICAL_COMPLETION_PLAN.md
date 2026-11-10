# PR停止の技術修正と残66 wrapperの継続案（未実行）

run01のPFは完了、PRは元`cost_trial`が呼ぶ`sample_requests(row, prep, cohort, index)`に対し、新adapterの`sampled(row, cohort, index)`でprep引数を落としたため、代表path0がbuild前にTypeErrorで停止した。モデル不適合や量子回路correctnessの失敗ではない。元SOURCE・停止・生成済み33本・全68軸の固定予測・PF価格を保持する。

今回指示§11は「**新しい科学campaignで停止条件に達した場合、無断で再試行しない。**」と指定している。本案は自動実行しない。元run01を再起動せず、別途明示された継続許可がある場合だけ以下を行う。

- 実装：[complete_h4_df10_price_v2_independent.py](../../../scripts/complete_h4_df10_price_v2_independent.py)。正しい4引数callbackを使う。旧SOURCEのscript/moduleは書き換えない。
- 入力：run01の保存`COHORT_PRIVATE.json.gz`、PREDICTIONS/PROVENANCE、完了PF result。cohort/path/seed/event/application順・予測を固定したまま使う。新sampling、新M0予測、PF再build/compile、Mq再fitは0。
- 既存`h46_same_event.decode_event`でapplication列を復元し、元DFPartialS2StepRequestを直接構成する。`make_rte_config`は同じδ.1/R4/K2の分布構成だけに使い、sample APIを呼ばない。保存eventのroundtripと予測時hash、seed規則、law、preparation、SOURCE/inputを照合する。
- 実行範囲は**同じPR33 trajectoryの未取得66軸wrapperだけ**。prefix1/q16/T1.6/δ.1/R4/K2、元compiler/full wrapperを維持。代表path0の元correctness gate成功後に残32を並列取得。PF2軸は元の成功結果を再利用し、元runとの新compile総数は最大68。top-up・新q・別δなし。
- 新出力はprivate `completion01`、公開は本directory配下`completion01/`。run01のSTOP・結果・SOURCEは上書きしない。完了後は新PR＋既存PFを同じ凍結Mq/M0と比較して公開、remote照合後STOP。再停止しても自動retryしない。
- 人工回帰2 method PASS：元と同じ4引数呼出し、既存人工fixtureの16×4 eventをRNGなしで復元し、順序・全application・seed・law一致を確認。実H4のreplay/build/compileは本案でまだ実行していない。

承認対象はこの残PR66 wrapperの継続と同じ比較・公開まで。新trajectory生成・PF再compile・再fit・全RPE/N/G・詳細科学レビューは含めない。
