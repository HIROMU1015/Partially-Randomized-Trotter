# H6 matched-resource supplement v3：準備・認可待ち

Repository `HIROMU1015/Partially-Randomized-Trotter`、branch `track-a-ax2b-h4-post-review-20261010`。
新source `5a903aea2492c2d211c1e15e7737df0d50694c12`。
旧science source `54986a9f54515c8c2289e38fa16fd81e2b6008c2`、再利用結果 `8bd903210cf3fe3ea8d628ad08fff561b8dd5aeb`。

- [新実行契約](../../../../docs/research/track_a_h6_matched_supplement_execution_contract_v3.md)
- [利用者が採用したGPT独立レビュー（原bytes）](../../../../docs/research/track_a_h6_matched_v2_memory_stop_independent_review_2026-10-11.md)
- [sealed manifest](sealed_manifest_v3.json)：runtime224、validation5、親入力33、環境・資源・新output、旧raw5,220のhash、449受理と143未完了のexact task/seedを含む
- [静的準備照合](preparation_verification_v3.json)、[synthetic検証](synthetic_validation_v3.json)、[log](synthetic_pytest_v3.log)、[JUnit](synthetic_pytest_v3.xml)
- [host資源観測](host_resource_observation_v3.json)、[未承認grantテンプレート](authorization_request_template_v3.json)
- [旧STOP・欠測・監査索引](../../../../docs/research/track_a_h6_matched_accuracy_result_v2.md)
- [旧一次signal](../../track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/signal_summary.json)、[原592探索登録](../../track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/exploration_tasks.json)

manifest canonical digest：`630179bfdfd6904f7e94746b5ca86470e0df49b49ee8409cdd66f6e065fec482`。
manifest SHA-256：`7796f92d442207a985fe2594b2d88ce74694d69a4e041e4188a05a16e5200f4c`。
scientific plan digestは旧v2と同じ `801e1eea5d09b6fbe60fe66102bc21edb1bd50a0d9aa94ca05a2b0659d1f8cf3`。

92 signal・84 B3組・449探索group・902 wrapperを静的に受理。新規は143探索group・286 wrapper＋原確認最大512 wrapper。
探索592が完全一致してから原上位2/精度→union最大8、独立32 trajectoryへ進む。
旧grantは消費済み。新grantのapproved_by_userはfalse。新H6科学run、signal再評価、分子sampling、H6 build/compileは0。
189 local synthetic/process検査PASS。実H6 q8の8 GiB完走・新資源map・科学的GOの証拠ではない。

原447を最初の一回取得として単独実行する設計。compiler・AS8 GiB・92候補・4精度・shot会計・原上限1708不変、wall capなし。
MemoryErrorの正確なallocation位置は未保存。新版はsingle-owner wrapperとbounded telemetryを準備した。
実H6 feasibilityは明示認可後に検査する。失敗時もretry/resumeせずraw/partial/auditを公開してSTOPする。

**MANDATORY STOP / H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false**。
H8・追加探索・旧v2再開は未認可。準備のみ完了し、新しい一回実行の認可を待つ。
