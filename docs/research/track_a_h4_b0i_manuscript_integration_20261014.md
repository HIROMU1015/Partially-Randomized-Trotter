# H4 B0+I全面実測：原稿v0.4・図表・科学reviewの入口

2026-10-14。固定結果 `98cdb35cd08469714f20d6cc3b982d9c80908f46` の専用branch/worktreeで保存値を統合。
実行SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、data identity SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
**原稿は科学的採否前のdraft。主要claim・新規性・論文化・追加検証はGPT判断待ち。**

- [通し原稿v0.4](../manuscripts/track_a_resource_study_geometry_native_v0_4.md)。
- [GPT科学review handoff](../manuscripts/track_a_h4_b0i_v0_4_scientific_review_handoff.md)。
- [主張候補―証拠―制約](../manuscripts/track_a_h4_b0i_v0_4_claim_evidence.md)、[旧版との差分](../manuscripts/track_a_h4_b0i_v0_4_changes.md)。
- [H4/H6/H8条件対応](track_a_resource_comparison_conditions_20261014_v0_4.md)。
- [図表・再生成・provenance・検証bundle](../../artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/README.md)。

## 整合した実測と追加保存値解析

H4 linear/STO-3G/DF12、6距離、T=.8、prefix3/4/5/6/9、q1/2/4/8、δ=.1〜.8、S2/ordinary/structured/fixed compiler/P=0。
240 coverage・234 new/6 verified reuse・native cleanup/owned14残存0は固定結果receiptの証拠を再利用し、240件の再実行やlive process操作は行わない。
240軸×6 metricsの1,440比較は全差0。旧B0/1/2/3の歴史的全1812点B2は保全し、実測追加後397 B0+I/1415 B2/ties0と分離した。
既存の結果manifest/独立Fraction照合を再利用し、今回新しい因子・SE分類・区間表だけを算術確認する。

保存された同じ302精度点を用い、費用因子1812行、paired SE差表1812行、method区間12、最良template区間19、fixed display216、matched120構成、代表18行を作成した。
374点はB2点推定−2SEも下回り、残るB0+I point minima23点は2SE以内。逆側も1392/23に分かれる。正式confidence intervalへ読み替えない。
.70/1.40 Å・ε=.05は回路費用優位が多いshotを補い、.80 Åはshot減少が重い回路を補う。同一prefix/qの位相効果と再選択されたfamily minimaを分けた。
headroom・bias・axis費用を表に保持し、分子相関や個別DF成分の物理機構を新たに確証したとは記載しない。

## 図表と検証

新主図4＋補助図S1をPNG/SVG/PDFの15 assetsで保存し、5図を目視確認した。元v0.3と旧4図のbytes/SHAは固定Git blobと照合して保全。
算術scriptは公開scalarのみ、図scriptは保存CSVのみを読み、Matplotlibは既存private venv、単一process/thread1、home専用cache/tmp、AS1GiB/CPU120秒/wall180秒/図12MiB上限で動作した。
非正headroom、ceil/normalization、paired covariance、軸別cost不変でもC_eff変化、grid re-entry/隣接bracket、非均一log cellの8限定stdlib検査はPASS。
全生成CSVの再導出、実測240 metrics、元結果/旧図/SOURCE59 closure、profile/input provenance、原稿内部リンク、図とCSVの対応、公開fileのbytes/SHAを[検証結果](../../artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14/verification.json)で追跡する。

## 未取得・未検証

same32でB2の選抜・費用推定、B2細順位の独立確認なし、H4 event/component NOT_SAVED、172 normalization丸めの法則未解決、希少event母平均未保証、1812点は独立標本でない。
εはcomplex signal、P=0。他compiler/H6/H8/energy/QPE/RPE/fault-tolerant総資源への移送保証はない。
H6の固定確認・pilot資料は既存値として継承し、H4へ補正しない。H8 core最終shot-map・対照の最新scalar処理・live状態は今回未取得であり、source/grant/worktreeへ変更しない。
この整備はlocal技術検証で、独立科学review・CI・元科学計算の独立再現ではない。

## 公開とSTOP

成果物は専用branch `track-a-h4-b0i-manuscript-v04-20261014` の軽量文書・表・図・再生成/検証script・限定testだけ。
新科学計算/compile/transpile/sampling/seed replay/NPZ読込/GPU/共有venv・設定/H8変更は0。carry/reset/grant変更なし。
公開状態は公開監査と最終チャット報告を参照し、credential障害時は設定を変えず手動pushを示す。
資料整備後mandatory STOP、追加数値計算や科学判断を自動実行しない。
