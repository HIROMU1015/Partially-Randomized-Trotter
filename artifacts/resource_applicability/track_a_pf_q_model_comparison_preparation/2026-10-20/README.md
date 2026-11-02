# PF qモデル/直接実測の比較準備packet

基点2619688b。新規科学計算なし、定量比較未実施、mandatory STOP。

| ファイル | 内容 |
| --- | --- |
| axis_correspondence.csv / .json | 788軸の主表。比較可能0、条件付き450、現状比較不能338 |
| saved_prediction_records.csv / .json | 450保存予測。元の文字列・出典SHA/locator・hash種別 |
| model_registry.json | H6/H8の登録8モデル。係数/公式は転記のみ |
| model_used_points.csv / .json | 学習・holdout・過去直接検証点のroleと来歴 |
| excluded_48_axes.csv | 今回対象外の旧profile照合待ち48軸 |
| direct_system_bindings.json | 元の封印metadataから転記した入力/compiler/profile結合 |
| source_inventory.json / prior_source_inventory.json | 今回読んだ固定Git証拠と、元棚卸しの出典一覧 |
| github_publication_receipt.json | public基点261と承認済みbranchのread-only確認 |
| publication_review.json | 公開承認80パス一致とgzip内を含む情報検査 |
| next_numeric_analysis_scope_DRAFT.json | 450保存値比較と78未保存予測を分けた未承認案 |
| structural_validation.json / packet_manifest.json | primary agentの構造検査と成果物hash |
| publication_scope.json | 今回の新規/更新パス。未公開・公開承認なし |
| MANDATORY_STOP.json | 科学call0、定量比較未承認の境界 |

[報告書](../../../../docs/research/track_a_pf_q_model_comparison_preparation_20261020.md)、[転記runner](../../../../scripts/prepare_pf_q_model_comparison.py)、[12テスト](../../../../tests/test_pf_q_model_comparison_preparation.py)。

runnerは既存棚卸しの固定Git readerを再利用する。`python3 -B scripts/prepare_pf_q_model_comparison.py --check`で表と固定入力を照合できる。初回実行の元metadata seal2件は、このhostの既知の旧worktree内のprivate保存先から読み、公開済みpreservation hashと一致させた。private seal全体は再公開せず、必要な入力/設定metadataだけ転記した。したがってrunnerの完全な再実行にはその保存先が必要であり、公開packetだけで元seal全体の復元を保証しない。

指標は状態準備を除いたfull Hadamard wrapperのcompiled RZ数。shot総費用と混同しない。初回572軸のrun全体最終gate欠測、H2 20軸の旧signal未整合、対象外48軸、モデルruntime同一性未確定を保持する。新規公開も実測/予測の定量比較も行っていない。
