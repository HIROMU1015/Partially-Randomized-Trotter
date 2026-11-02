# 2026-10-17：Exact Encoding限定構成案を区切り、④の研究課題選定へ戻る正式方針

Repository：`HIROMU1015/Partially-Randomized-Trotter`。今回のユーザー明示指示による**研究方針の採択記録**。Codexによる新しい科学判断・科学レビュー・候補選定ではない。

## 採択の根拠と対象

ユーザーの今回の指示「Exact Encoding研究の方針確定・状態記録」冒頭：

> ユーザーは、今回のExact Encoding × Controlled Evolutionの限定構成案を区切り、④「PRに限定しないアルゴリズム改善」の研究課題選定段階へ戻る方針を正式に承認しました。

対象は固定commit `830be75335bb9b599488cb365213d555b0dc18bb` の[限定構成報告][result]と[判断用ハンドオフ][handoff]に記録された、一つのfixed-N・spin-free DF平方blockの構成案である。技術分類は `REDUCES_TO_KNOWN_METHOD` のまま保存する。元報告・式・既知法との対応・未検証事項を改訂・再検証しない。

旧ハンドオフの方針判断待ちという表現は当時の記録として残し、今回の正式承認は本書とmachine記録で追跡する。次の具体課題まで採択されたとは扱わない。

## 正式に記録する方針

| 項目 | 今回採択した状態・範囲 |
| --- | --- |
| 今回の限定構成案 | 検討を区切り、追加探索・実装・数値検証を行わない |
| 技術分類 | `REDUCES_TO_KNOWN_METHOD` を維持。この一案の限定分類 |
| ④全体 | **研究課題選定段階へ戻る**。④全体の終了ではない |
| 次の具体的な研究課題 | **未採択**。候補探索・設計・採択は今回開始しない |
| Encoding一般・Hamiltonian simulation一般 | 不可能性・一般NO-GOを認定しない |
| 新しい独立算法・性能・総資源改善 | 今回新たに認定・採択しない |
| 科学レビュー・科学計算・SOURCE実装 | 今回の状態記録の対象外。新規実施0件 |

これは新しいexecution authorizationや科学実行契約ではない。[machine-readable採択記録](../qc_exact_encoding_direction_adoption_20261017/decision.json)の `user_adopted_direction=true` は、上記の方針だけをユーザーが採択したことを示す。新研究テーマ・新算法・新計算の採択や実施許可を意味しない。

## 既存状態と成果の保全

- ①PRリソース評価、②PR内部改善、③M1-Sの研究状態を変更しない。直前の4系列資料を使って別研究の現在地を再判定しない。
- ④のN1現探索STOP／N2 HOLD／N3現系列CLOSED／THRIFT現template CLOSED、H1–H4の現探索PAUSEDを維持する。一体項再配分PFの現構成CLOSED、H-E1現定義CLOSED、M-a/M-b STOP、PF小改変PAUSED、旧D HOLDも維持する。
- SOS-PF現分割生成探索CLOSED、A2 NOT_ADOPTED、K1-F未認可・未実施等、従来の限定状態を変更しない。今回の方針を既存系列再開の承認としない。
- D1・D2、state access、scalar・対象sector・controlled位相の既存条件を変更しない。
- SOURCE、raw結果、科学契約、authorization、grant、ledger、marker、レビュー原本、直前のpreflight・限定構成報告と証拠は保全する。

既存状態の参照先は[2026-10-16状態記録][old-state]と[4系列重複・進捗表][overlap]。今回の変更は上表の限定案と④の段階だけであり、これらの過去記録を新しい科学判断で上書きしない。

## 反映箇所・確認・停止

既存文書の変更は `PROJECT_MAP.md`、`docs/research/研究概要・現状.md` の短い追記と、`docs/research/qc_new_algorithm_research_scope_20261016/research_state_transition.md` の日付付き追記だけに限定する。既存本文を削除・置換しない。数値validation inventoryは変わらないため、`VALIDATION_STATUS.md` と `artifacts/validation_manifest.json` は更新しない。

[静的確認・保全記録](../qc_exact_encoding_direction_adoption_20261017/verification.json)、[GitHub公開照合](../qc_exact_encoding_direction_adoption_20261017/publication_receipt.json)へ接続する。公開・内容一致・保全の確認は科学的再認証ではない。

**状態記録と公開を完了した後、mandatory STOP。** 次の担当はユーザーである。④の課題選定を具体的に依頼する場合は、対象・作業範囲を別途指示する。今回の採択だけで選定作業、新候補、科学レビュー、SOURCE準備、新規検証へ進まない。自動的に開始する後続作業はない。

[result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/830be75335bb9b599488cb365213d555b0dc18bb/docs/research/qc_exact_encoding_controlled_construction_20261017/limited_construction_report.md
[handoff]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/830be75335bb9b599488cb365213d555b0dc18bb/docs/research/qc_exact_encoding_controlled_construction_20261017/decision_handoff.md
[old-state]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/56f1cc55c59ab7bc42c6b24e0953675de9b29b1d/docs/research/qc_new_algorithm_research_scope_20261016/research_state_transition.md
[overlap]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d09ba606a9fceef806a9313e3f7c43cd21d19e3f/docs/research/qc_four_track_strategy_handoff_20261017/overlap_and_progress_map.md
