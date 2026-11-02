# H4/H6 PR pilot lossless storage planning

**33本/群を優先した保存計画。科学実行・新保存方式の採用は未承認。**

[報告書](../../../../docs/research/track_a_h46_pr_pilot_storage_optimization_20261020.md)と[保存policy DRAFT](storage_policy_DRAFT.json)を参照する。基点は`3e7365a606e18efa585252eef23f7b13264b0f38`。元の科学runner・既存rawは変更していない。

| 資料 | 内容 |
| --- | --- |
| [summary.json](summary.json) | 33本案の推奨・容量proxy・SOURCE・次の担当 |
| [existing_capacity_breakdown.json](existing_capacity_breakdown.json) | 保存3482fileの容量種別、ledger/metadata/NPZ/QPYの区分 |
| [compression_evidence_276.csv](compression_evidence_276.csv) | 別領域の276圧縮copy、元/archive byte SHA・size、原SHAへの復元確認 |
| [compression_measurement.json](compression_measurement.json) | 全level1実測、4最大QPYのlevel6比較、2fileへの明示byte復元 |
| [capacity_cases.json](capacity_cases.json) | n33/9、余裕2/4/8倍、逐次/並列とworker scratchの18案 |
| [storage_policy_DRAFT.json](storage_policy_DRAFT.json) | 証拠保存・原/圧縮SHA区分・新規一時file解放の未承認条件・STOP |
| [source_closure.json](source_closure.json) | standalone prototypeと人工テストの固定SOURCE/SHA |
| [preservation_and_privacy.json](preservation_and_privacy.json) | 既存3482fileと旧pilot封印の保全確認 |
| [validation.json](validation.json) | 人工18件、構造・リンク・公開範囲検査 |
| [inventory.json](inventory.json) | このbundleのSHA。自身とvalidationを除く |

実測はopaque byte圧縮だけで、保存回路・配列の科学的再生や解析ではない。保存QPY/NPZ計10.36 GBは0.635 GBに縮み、276file全て原SHAに復元できた。原本を削除・置換していない。file全体の完全一致共有だけの効果は約28 KBで小さく、主対策は可逆圧縮である。

33本案では、測定最大pairの4倍＋16 MiB/trajectoryの未採択proxyでfinal約132.5 GB、各系1worker並行peak約137.0 GB、合計64workerのH6-heavy envelopeで約264.2 GB。現在空き約455.2 GBに収まる見込み。ただしm5 DET測定からの計画であり、PR/rare出力の上界ではない。既存結果は保持し、その容量を含めて既に減った「現在の空き」へ新規増分を比較する。

逐次化だけでは最終保存を減らせない。新規rawを圧縮・完全復元検証後に解放するか、rawをdiskへ出さない保存契約が必要。prototypeはどのsource fileも削除せず、現在の旧科学runnerにも接続していない。新しい保存hook・archive-aware終了監査・loaderと新SOURCE/config/grantが必要である。今回それらの有効grantは発行していない。

公開物にはprivate原本・QPY/NPZ・圧縮archive本体・認証情報・内部絶対パスを含めない。圧縮copyと明示復元したprivate test fileは別領域に保持する。

```bash
/usr/bin/python3.12 -B artifacts/resource_applicability/track_a_h46_pr_pilot_storage_optimization/2026-10-20/check_structure.py
/usr/bin/python3.12 -B -m unittest discover -s tests -p 'test_h46_pilot_lossless_storage.py'
```

2つ目の標準環境では任意Qiskit testはskipされ得る。既存の固定runtimeでは人工QPY testを含む全18件がPASS。新H4/H6 sampling/compile/signal/shot/数値再解析は0件。次はユーザーの保存方式・標本数判断、Codexの新方式接続、その後ユーザーのH4/H6系別実行承認。mandatory STOP。
