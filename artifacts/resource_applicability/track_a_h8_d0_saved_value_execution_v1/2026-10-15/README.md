# H8 D0 one-shot 完了結果

[結果・研究ノートの入口](../../../../docs/research/track_a_h8_d0_saved_value_result_20261015.md)。SOURCE20f89bc32b161ee22e4a1ed709f9e3414c19633a、grant b613db106abd2e1cadb7bd5a29a765f4f257fcc5。旧944precision/3576cost EXACT PASS後に5精度/factor1・10/6metric/Pを一度評価し、native exit0・mandatory STOP。

| 保存資料 | 用途 |
| --- | --- |
| [旧値一致](original_equality_v1.json) | 両scopeの完全一致件数とphase先行gate。 |
| [全accounting](candidate_accounting_v1.json) | 探索920/確認260 records、適格/不適格・N・6metric・paired SE。 |
| [20精度summary](precision_summary_v1.csv) | RZの探索/確認・factor1/10、full PF/B0/取得済detとの比。 |
| [360metric summary](metric_selection_summary_v1.csv) / [selection正本](selection_metrics_v1.json) | metric別選択とRZ選択を区別。 |
| [P感度](preparation_sensitivity_v1.json) | 全120曲線、固定pair交点と全候補envelope、exact fractions。 |
| [P=0 coverage](confirmation_coverage_v1.json) / [P coverage監査](P_confirmation_coverage_audit_v1.json) | 旧確認union外の探索候補を確認32へ混ぜない。 |
| [希少次数](rare_cost_status_v1.json) | 旧観測histogram、未観測費用null、母平均上界なし。 |
| [完了監査](completion_audit_v1.json) / [terminal](terminal_v1.json) / [native wait](native_exit_v1.json) / [launcher wait](launcher_wait_receipt_v1.json) | 完了、監視実績、first STOPなし、退出・cleanup。 |
| [provenance](provenance_v1.json) / [source closure](source_closure_v1.json) | actual SOURCE・旧科学result/profile/input/carry対応。 |
| [contract](contract_v1.json) / [authorization](authorization_v1.json) / [利用者承認](user_approval_evidence_v1.json) / [seal](launch_binding_seal_v1.json) | 今回H8一度だけの固定実行認可。 |
| [入力SHA](input_bytes_receipt_v1.json) / [quota読取](quota_readonly_receipt_v1.json) / [prelaunch](prelaunch_readiness_v1.json) | 起動前bytes/profile/resource gate。別runのfresh証拠へ流用しない。 |
| [公開検証](publication_verification_v1.json) / [manifest](artifact_manifest_v1.json) / [公開receipt](publication_receipt_v1.json) | ファイル対応・内部リンク・Git/remote bytes/SHA。 |

公開outputsはprivate実行結果からbyte-exactでコピーした派生スカラー資料。private input mapping、NPZ、one-shot marker、raw/runtime/checkpoint/cache/credentialは含めない。stdoutは0 bytesである。

条件はlinear H8/1.00 Å/STO-3G/DF28/T=.8、既存native control/compiler、q≤8、保存state/signal、経験的u。確認32のε=.0005/factor1/P=0/RZ PR/取得済B0-B1最良は.761667659。native B0+I未測定、population winner未保証。P感度は共通仮想1-shot overheadで実準備回路ではない。

旧実行環境・source/profile/input/carryは保持、新SCF/DF/state/signal/trajectory/build/compile0、H4/H6 D0再実行0。one-shot markerはprivateに残り消費済み、同output再利用・retry・H6自動起動なし。結果公開後mandatory STOP。
