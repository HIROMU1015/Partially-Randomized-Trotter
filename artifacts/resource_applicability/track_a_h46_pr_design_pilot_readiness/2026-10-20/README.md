# H4/H6 outer32 DESIGN_PILOT readiness

**提案・技術準備のみ、実行未承認。** [報告書](../../../../docs/research/track_a_h46_pr_design_pilot_readiness_20261020.md)と[24群・標本数提案](../../../../references/h46_pr_design_pilot_proposal_v1.json)を参照する。

| 資料 | 内容 |
| --- | --- |
| [summary.json](summary.json) | SOURCE・提案規模・未採択事項・次の担当 |
| [pilot_groups_24.csv](pilot_groups_24.csv) | H4/H6×3公式×q1/2/4/8、保存適格ε・preparation、n_P/n_E未採択 |
| [contracts.json](contracts.json) | 系別の無効契約・private plan digest・資源案 |
| [H4_DESIGN_PILOT_INVALID_DRAFT.json](H4_DESIGN_PILOT_INVALID_DRAFT.json) / [H6_DESIGN_PILOT_INVALID_DRAFT.json](H6_DESIGN_PILOT_INVALID_DRAFT.json) | 公開用projection。実行パスを含まず、CLIへ渡しても起動を拒否 |
| [data_dictionary.json](data_dictionary.json) | paired単価・stratum・piece lawの記録項目、nullの意味 |
| [evaluation_decisions_UNAPPROVED.json](evaluation_decisions_UNAPPROVED.json) | pilot後の数値評価・n_E判断に必要な別採択 |
| [statistics_design_reference.json](statistics_design_reference.json) | 正規IID参照の名目分散精度と、その限界 |
| [reuse_provenance.json](reuse_provenance.json) | 元nativeとの厳密な差分範囲・同law維持 |
| [resource_storage_proposal.json](resource_storage_proposal.json) | 保存出力のbyte数だけから作った未採択scratch容量案 |
| [preparation_resource_observation.json](preparation_resource_observation.json) | 準備時の読取専用観測。起動承認・実行直前gateではない |
| [source_closure.json](source_closure.json) | 実装SOURCEに対応する相対path・byte SHA |
| [validation.json](validation.json) | 人工20＋回帰49、構造・保全・拒否検査 |
| [inventory.json](inventory.json) | bundle資料のSHA。validationとinventory自身を除く |

推奨は各群33本（全792 trajectory、最大1,584 compile軸）。正規IIDを参照した標本分散の相対SE25%という設計目標に対応し、実際のPRの分散精度・希少tail・母平均順位を保証しない。粗い9本案（216 trajectory、432軸）も提案として残す。両案とも未採択で、9→33の自動top-upはない。

現在のscratch空き約457 GBに対し、33本案の保存容量proxyは合計約976 GBで、現保存先では条件未成立。9本案のproxy約266 GBは準備時の空きに収まる。proxyは実PR/rare出力の上界ではない。33本案では十分な保存先・計画を先に判断し、既存rawを削除して空けることはしない。

本探索n_E、独立確認n_C、正式shot政策、AD03は保留。pilotの取得契約に数値評価は含まれない。全群の完全取得・終了監査後にSTOPし、記述統計・n_Eの感度計算は別の明示承認を受ける。本探索へpilot標本を混合しない。

構造検査（stdlibのみ）：

```bash
/usr/bin/python3.12 -B artifacts/resource_applicability/track_a_h46_pr_design_pilot_readiness/2026-10-20/check_structure.py
/usr/bin/python3.12 -B scripts/run_h46_pr_design_pilot.py
```

2つ目はscope表示だけである。新しいH4/H6のseed発行・sampling・compile・signal・shot・費用再解析は0件。native人工検証には既存の封印済みruntimeと小さな合成Hamiltonianを使用した。private原本、QPY/NPZ、認証情報、seed値一覧、内部絶対パスはこのbundleに含まない。

次の担当はユーザー。目的・n_P・資源予算・系別一回実行を判断する。現時点でmandatory STOP。
