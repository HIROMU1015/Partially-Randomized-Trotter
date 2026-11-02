# Mechanism論文統合：Figure 2実データ描画・H4 P・A82限定準備

2026-10-19 JST。**今回承認された3作業を完了。A82実数値解析0、公開push0、最終evidence freezeなし。mandatory STOP。**

|成果物|内容|
|---|---|
|[A82契約案](A82_execution_contract_DRAFT.md)／[SOURCE投影](A82_readiness_projection.json)|PR2軸だけのadapter／runner、入力・runtime・output schema・無効grant案|
|[H4 P統合](H4_P_integration_report.md)／[別表14行](data/H4_P_confirmation_separate.csv)|独立P完了と旧cohortの分離、SOURCE／原result記録SHA／公開projection SHAを区別|
|[主表32行](data/main_table.csv)／[manifest](data_manifest.json)|旧全数値を保全しH4最新provenanceを追加、全比較・全説明因子を保持|
|[Figure 2A](figures/figure2A.svg)／[Figure 2B](figures/figure2B.svg)|保存値の実データ図、PNG/PDFも保存|
|[描画対応値](figures/plot_values.csv)／[描画検証](render_validation_report.md)|116点の元token・座標・表示丸め・cohort・PF class|
|[Claim–Evidence更新案](claim_evidence_matrix_update_DRAFT.md)|前8行を全文保全しH4 P／図／A82準備を追記|
|[evidence readiness](paper_evidence_readiness.md)／[技術テスト](technical_test_report.md)|未解決事項と承認境界を保持|
|[次承認](next_approval_list.json)|担当・one-shot承認単位|

A82 SOURCEは`a8961c45d7fd0310c1e34c50ffd004dd58898247`。既存科学式SOURCE b6716396…の3 fileを変更せず再利用し、
新adapterは既存explain_sideをPRへ1回だけ呼ぶ。PF側と他63 side・元core32は再評価しない。
原signalと回収2 field、SOURCE/DF28/T=.8/cohort/byteを再確認した。新contract/binding/runtimeはlocalに固定し、
grant案はfalse/未承認receipt。無効grantの実CLIは入力読込前に拒否され、科学outputは未作成。
元EXPLANATORY_INPUT_MISSINGを保持したまま、別途one-shot承認を待つ。

H4 Pは固定`8a18b277663bf2f8998f79e9b8c0a9e61d2b184c`の独立legacy DF12系列。4,185軌道・8,370 compile、
主比較10/10・診断4/4で事前の点費用方向を再現。S4未観測、母平均優位保証なし。
DのPF代表10/10不変も保持。旧H4 core数値・H4_PRE_D_P_VALIDATION・D_COMPLETE_P_PENDINGは履歴として不変、
P完了を新provenanceへ追記した。Pの新比率・機構・統計は計算していない。

Figure 2描画SOURCEは`3de17acd912a28a1df9905b2f3c3e2e3e15aa0dc`。旧図表コードと保存4表を再利用し、
Figure 2Aに3系の固定18比較、Figure 2BにH8独立n48の固定2例を表示した。
exact tokenからの描画座標変換・表示丸めのみで、新たな科学的比・平均・SE・分解値は0。
H6 modelは中抜き、cohortは色／注記で区別し、H4 Pの値はプロットしていない。
初回Figure 2Bの注記重なりを目視で検出し、初回をlocal監査へ保全して表示用の余白だけ修正した。

主担当による人工・回帰158件PASS。別AIへのdelegation、共有環境／他jobの変更は0。
実際に使う既存SOURCE・入力・結果・grant/ledger/marker/STOPを読取保全し、無関係worktreeを凍結しない。
CPU/RAM/workerの恣意的な恒常上限は追加せず、空き資源を確認して小さい処理に適したown processを使用した。
Matplotlib cacheとlogは今回専用領域へ置き、共有venvは変更しない。

**次担当はユーザー。** A82限定one-shot実行、必要な追加因子／P機構契約、図表・claim採択、公開の要否を判断する。
今回の成果物をlocal commit後、mandatory STOP。sampling／compile／新統計／実A82解析／公開push／最終freeze／次stage自動移行は0。
