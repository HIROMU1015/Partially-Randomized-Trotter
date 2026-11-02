# H4 B0+I保存値診断：科学判断待ちSTOP

[報告・符号/係数のsource調査・代表表・未承認の次案](../../../../docs/research/track_a_h4_geometry_scalar_diagnostic_20261013.md)が入口です。
追加回路費用ゼロの参考モデルで397/1812点がB2点最小を下回りました。B0+Iの実compiled費用は未測定で、実測winnerではありません。
今回指示の科学的停止条件に従い、主claim変更・原稿本体統合・新compile・H8変更を行いません。

- [診断summary・代表18点・STOPと未実施範囲](diagnostic_summary_v1.json)
- [全1,812点の参考費用・追加RZ/shot交点CSV](scalar_reference_envelope_1812.csv)
- [DF tail scalarと保存energyの軽量metadata](tail_scalar_metadata_v1.json)
- [120保存B0の別管理スカラー位相感度](saved_b0_phase_sensitivity_v1.json)
- [固定source・6NPZ streaming SHA・小memberのprovenance](source_input_provenance_v1.json)
- [認可された限定解析計画](ANALYSIS_PLAN_v1.json)
- [lower-triangle規約確認に伴う初回仮定の修正](INITIAL_ANALYSIS_ASSUMPTION_CORRECTION.json)
- [共有済みGPTレビュー（提供文書として保全）](supplied_gpt_scientific_review_20261013.md)
- [元rawを受領した環境でのbyte/小係数解析script](diagnose_saved_b0_identity.py)
- [公開scalar保存値だけからの照合script](verify_scalar_publication.py)
- [bundle bytes/SHA](artifact_manifest_v1.json)

公開fieldは保存値の派生スカラー・集計・SHAだけです。G/H/state本体、NPZ、runtime/checkpoint/ledger/journal、内部log、credential、SSH情報は収録しません。
source `d0726e946f89f8c30a08d383fb8ff48b363629ec`、結果base `7c4483378047b57f514c47dad98905e3f2a9a756`。
元結果branchの登録比較は不変。formal comparison change/new science authorizationはfalseです。

リポジトリ内の公開JSON/CSVだけの読み取り照合は次のとおりです。install/GPU/NPZ/runtime access不要、1 process/AS256MiB/CPU45s/wall60sです。

```bash
python3 -P -B artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/verify_scalar_publication.py
```
