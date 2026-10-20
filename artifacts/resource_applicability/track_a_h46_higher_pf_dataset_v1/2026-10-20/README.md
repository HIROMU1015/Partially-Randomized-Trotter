# H4/H6 higher-PF integrated research dataset v1

2026-10-20、基点`5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb`。**既存の科学値を正確に転記・構造結合した再利用用データセット**。新しい科学的数値解析、会計、順位、公式の主比較採択、最終evidence freezeは行っていない。

[付属実験プロトコル](../../../../docs/research/track_a_h46_higher_pf_supplementary_protocol_v1.md)・[統合報告](../../../../docs/research/track_a_h46_higher_pf_dataset_integration_20261020.md)・[正式共通v2](../../../../docs/research/track_a_pr_resource_evaluation_protocol_v2.md)を併読する。付属文書は実験後の記録であり、遡及的事前登録ではない。

| ファイル | 内容 |
| --- | --- |
| [dataset_index.json](dataset_index.json) | スキーマ、件数、入力/派生索引のSHA、固定参照、null規則 |
| [source_catalog.json](source_catalog.json) | commit/path/Git blob/SHA256/bytes/固定GitHub URL |
| [dataset.json](dataset.json) | 以下の統合metadataをJSONで提供。signal/880適格性/368gateは元表へ参照 |
| [candidates_88.csv](candidates_88.csv) | 候補identity、preparation、signal原結果SHA、両軸適格ε、費用status |
| [candidate_precision_links_440.csv](candidate_precision_links_440.csv) | 88候補×5精度の両軸ラベル・元880行への位置対応 |
| [axis_provenance_176.csv](axis_provenance_176.csv) | 88候補×2軸のsignal/eligibility/実測receipt/SOURCE対応 |
| [measured_gate_costs_96.csv](measured_gate_costs_96.csv) | m5決定論48候補96軸、RZ/CX/depth/size全6metric、新92・保存4を区別 |
| [saved_reuse_metric_projection_4.json](saved_reuse_metric_projection_4.json) | SHA確認済み4原receiptから不足metricをallowlist転記 |
| [conditions.json](conditions.json) | 保存入力hash・DF生成政策・状態・compiler/native条件と未確認範囲 |
| [run_provenance.json](run_provenance.json) | 保存SOURCE/結果/終了SHA、exit/reap、実行時間。PID・private pathは含めない |
| [common_v2_imports.json](common_v2_imports.json) | 最新正式v2と必要な公開済みリンク先27fileの固定snapshot来歴 |
| [private_metadata_verification.json](private_metadata_verification.json) | 使用した原JSONのSHA/転記照合の範囲と限界 |
| [validation.json](validation.json) | 人工・構造検証、保全・公開範囲検査の結果 |

## 関係・値・欠測

候補の主キーは保存`M5_MIN_…` ID。軸キーは`(candidate_id, cosine|sine)`、signalの`real|imag`との対応を明示する。精度キーは`(candidate_id, axis, epsilon)`。row indexは**ヘッダを除く0始まりのrecord位置**であり、CSV物理行番号ではない。元ファイルは固定commit・SHAに結合する。

signal88（82新・6保存）、axis eligibility880、native gate368の数値payloadは複製しない。catalogの`signal`/`eligibility`/`native_gates`から元のJSON/CSVを取得する。統合readerは同じ値をJSONとして取り出せる。未取得PR24群・inactive DET16候補もcandidates/axis表に残す。

CSV空欄はJSON nullに対応し、0費用・0標本・適格を意味しない。n_E、shots、total_RZ、厳密certificateは未確定のまま。saved再利用のcompiled_SHA欠測も空欄を保持する。`SAVED_CT01_SEPARATE_ACCEPTANCE`は元STOPをPASSへ変更した意味ではない。新92軸のgate368行と旧4軸の受入証拠を区別する。

48候補96軸は全て直接実測単価である。既存S2/S4の実測/model混在をこの表へ無条件に輸入していない。本表からshot込み費用や最良qを計算した結果は存在しない。

## 再利用と検査

repository rootで実行する。stdlibとGitだけを使う。scientific module、Qiskit、NumPyのimport、QPY/NPZ復元、grant発行はない。

```bash
/usr/bin/python3.12 -B scripts/h46_higher_pf_dataset.py
/usr/bin/python3.12 -B scripts/h46_higher_pf_dataset.py --table eligibility
/usr/bin/python3.12 -B scripts/h46_higher_pf_dataset.py --table signal
/usr/bin/python3.12 -B scripts/h46_higher_pf_dataset.py --table measured_gate_costs
/usr/bin/python3.12 -B -m unittest discover -s tests -p 'test_h46_higher_pf_dataset.py'
```

初回参照に必要な固定Git objectsは`source_catalog.json`に全て列挙する。shallow clone等でobjectがなければreaderはfail closedする。別版やliveファイルに黙って代替しない。`--write-index`はこのbundleの派生metadata6fileだけを再生成する機能で、科学実行は行わない。

原JSON233fileを今回SHA確認し、signal88件と費用92新軸・4保存軸の数値を比較した。大きなQPY/NPZは読み直さず、元終了監査を参照する。これは作成者による検査で、独立科学レビューではない。private原本・QPY・認証情報・seed値一覧・不要な内部絶対パスは本bundleに含めない。

## 次の担当

ユーザー：PR24群の標本設計経路・n_E、正式shot政策、後続実行を個別に判断する。Codex：その承認後に独立契約を固定し、承認された範囲だけ実行する。今回の整理からsampling・会計・確認・AD03へ自動移行しない。mandatory STOP。
