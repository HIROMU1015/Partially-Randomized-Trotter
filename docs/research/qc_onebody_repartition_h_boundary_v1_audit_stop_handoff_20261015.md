# 固定H4 h境界v1：独立監査技術停止への引継ぎ

収集4572cells/4560新PF・Schurは完了、独立checker一回はNUMERICAL_ALLOWANCEで停止した。新性能結果はUNVALIDATED_PRODUCER_ONLY。producerとauditのone-shot認可は消費済み、retry0。source5e3341c1と契約/input/旧v1結果を保持し、公開・再取得後Mandatory STOP。

[結果・未検証値・scope](qc_onebody_repartition_h_boundary_v1_execution_results_20261015.md)、[失敗記録](../../artifacts/qc_onebody_repartition_h_boundary_execution_v1/2026-10-15/failure_record.json)、[一次JSON](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/raw.json)、[audit stop](../../artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1/independent_audit.json)。

## 分かっていることと未取得情報

- frozen checker64行でsaved/recomputed allowanceの差<1e-10という照合条件が不成立。handlerはfailed cellや双方の値を返さない。
- producerのspectral/cached UとcheckerのPadé Uからのresidual/cluster幅計算は異なる。この静的観察だけで、全データの誤り・根本原因・安全な修正を確定しない。
- 異常の数値再計算、checkpoint再開、tolerance変更、checker再実行は0。実partial expm回数はUNKNOWN。原データとreason/exit/resourceを保存した。
- 収集側は4条件で登録10%gainなしを保存したが、数値監査完了前に候補終了/成功の科学判断へ使わない。

## 次に別scopeで準備する場合の最小事項

1. oldsource/source manifest/原データ/承認/STOPをbyte保持し、別version・別branchで失敗cell key、saved/recomputed allowance、residual/defect/spread、partial countersを記録する仕様を作る。
2. producerの自己計算allowanceの再現検査と、独立PF再構成の差をどのように扱うかを分け、科学的精度意味論を保つ。合格させるための単純な許容差緩和をしない。
3. 登録外fixtureで新checkerの正誤拒否を検査し、必要なら保存データだけへの限定診断をsource/input/caps・新one-shot認可対象として準備する。分子/PF収集を自動で再実行しない。
4. 数値再監査・診断は別明示実行承認後だけ行う。今回の消費済み認可を流用しない。

これは後続taskの提案であり、このtask中に新checker/数値diagnosticを実装・起動したものではない。精度意味論・primary条件を変える必要があれば、その論点だけGPT科学判断へ戻す。通常の技術修正だけで同じ詳細設計レビューを反復する必要はない。新結果の研究継続/終了、他候補移行、新規性/総費用判断は別GPT開始承認事項。

N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSED、C1–C4/TrackA–B/第二候補、既知global manifest FAIL、旧v1結果/認可/STOPを保持する。
