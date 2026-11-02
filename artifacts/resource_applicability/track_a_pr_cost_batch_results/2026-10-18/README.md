# H4/H6/H8 A_COST・保存B/D解析：完了結果projection

[報告書](../../../../docs/research/track_a_pr_cost_batch_completed_results_20261018.md)。
2026-10-18の完了artifactからの軽量転記。新しい科学計算・費用再会計・順位選択は行わない。

| File | 内容 |
|---|---|
| [completion_status.json](completion_status.json) | 9契約の完了件数と未確定事項。詳細なscope運用記録はprivate保全 |
| [a_candidates.json](a_candidates.json) | 固定48候補、系別binding、探索n8、seed導出SOURCE |
| [a_cost_measured_axes.csv](a_cost_measured_axes.csv) | 保存実測768軸。各RZ/CX/depth等とpaired seed/event digest/fingerprint |
| [native_completion_summary.json](native_completion_summary.json) | 9scopeの正常終了・回収済みの集計。軌道別CPU/RSSログはprivate保全 |
| [d_saved_comparison_summary.json](d_saved_comparison_summary.json) | 既存D結果の120比較。削減率exact fraction・候補・実測／model status |
| [b_diagnostic_scope_summary.json](b_diagnostic_scope_summary.json) | B診断の対象・不足層・未確定事項。無条件平均はnull |
| [verification.json](verification.json) | 保存結果の限定機械検証。独立科学reviewではない |
| [validation_status_entry.json](validation_status_entry.json) | 今回の完了状態。固定commitの旧履歴manifestを保全した独立追補 |
| [files.json](files.json) | 今回の公開対象fileのbytes/SHA。自己参照は除外 |

CSVのseedは元の64-bit整数を十進文字列として保持する。浮動小数点へ変換すると精度を失う場合がある。
RZ/CX等は単一compiled full wrapperの測定値で、shot重み付き総費用ではない。
探索n8を確認n32へ混合しない。Dの比較は新Aのn8を使っていない。
H4は1Åのlegacy rank12、H6 rank19、H8 rank28。異なるDF系列をpoolしない。
詳細なB次数分布、条件付き数値、Dの保存shot／SE等はprivate原本に保持し、公開版は要約に限定する。
詳細native終了ログとprivate証拠一覧も公開対象には含めない。
実行SOURCEは報告書に記載し、既公開の準備／起動報告へ接続する。SOURCE closure原本は保持する。

公開projectionはprivate rawの代替原本ではない。原本を利用できない環境では、公開hash receiptだけで
privateデータのbyte照合を再実施できない。公開測定値・元SOURCE・個別result SHAは追跡可能にしている。
raw NPZ/QPY、state、event列、checkpoint/cache、ledger、有効grant、credentialは含めない。
rootの履歴manifestは既公開bytesを維持する。最新の機械可読状態は上記の専用entryを参照する。
全9scopeは完了済みだが、Bの無条件平均と新Aを反映した総費用比較は未確定／未実施。次stageは未認可。
