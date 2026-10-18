# Track A Mechanism説明側：診断・修正準備の統合報告

**静的原因を確認し技術修正を準備、実科学数値再解析0、有効grant0、mandatory STOP。**

新SOURCE：`b6716396acc5760fde7cf7cc6f2481b183a08f10`。元SOURCE5c256067…と結果2bf5d832…、32/32 core完了・exact費用分解・保存SEはそのまま保持した。

- PR28 side：step bを累積Bとして結合した不具合と、shot用log margin省略をSOURCEで確認。
- PF32 side：nominal log_Bと、保存shot規則が使うlog_B_upperの取り違えを確認。
- 欠測4件：保存accountingから3件を回収。A82_OLD_PRはaccountingだけ部分回収し、原signalのnormalization／signed biasを未回収として保持。
- 新入力案：64 side中63に必要field、1 side未回収。これは数値一致のPASSではない。
- 人工検査：既存64＋追加41＝105件PASS。guardの許容差緩和なし。

[原因と証拠](mechanism_explanation_root_cause_report.md)、[SOURCE trace](mechanism_explanation_source_trace.csv)、[欠測](mechanism_explanation_missing_input_report.md)、[修正仕様](mechanism_explanation_fix_specification.md)、[人工テスト](mechanism_explanation_test_report.md)、[未承認再解析契約](mechanism_explanation_reanalysis_contract_DRAFT.md)。

public projectionには科学上のfield対応・SOURCE code hash・metadata状態だけを保存した。private companionには完全なinput pointer/hash、正本core、旧消費済み記録の保全receipt、新24入力role、Python／68 runtime file、新規output案、無効grant案を保存する。旧科学記録75fileは不変。H4 D/Pや他job・共有設定を変更していない。公開pushは未実施。

現時点の実科学説明完了は0のまま。headroom、saved log bound、整数ceilの新SOURCEでの一致は未実行であり、既知不具合を直したことから推定しない。母平均優位性・新しい科学的保証・論文最終freezeを認定しない。

次担当：**ユーザー**。新SOURCE・private input binding・core SHA・再解析契約を確認し、必要なら説明側だけの新one-shot実行を別途承認する。Codexは停止し、自動再解析・新統計・追加sampling／compile・次stageへ移行しない。
