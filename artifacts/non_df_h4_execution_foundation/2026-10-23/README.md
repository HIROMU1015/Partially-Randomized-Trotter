# H4非DF/DF PF・PR実行基盤：技術証拠

2026-10-23 JST。**実装・人工検証完了、実H4科学実行は未承認。mandatory STOP。**

- [技術報告](../../../docs/non_df_h4_execution_foundation.md)：実装、同値性、旧SOURCE保全、限界。
- [完成契約候補](H4_execution_contract_DRAFT.json)：未採択。execution_authorized/grant_valid/dispatch_enabledはfalse。
- [一括承認案](H4_batch_approval_proposal.md)：次回に承認する科学条件とcall範囲。
- [開始手順](EXECUTION.md)：承認後にSOURCE/runtime/承認を結合し、新grantを一度発行して実行する手順。
- [SOURCE](SOURCE.json)、[検証集計](validation_summary.json)、[保全・保存証拠監査](verification.json)、[証拠inventory](evidence_inventory.json)、[STOP](STOP.json)。

`technical/final_tests.xml`は429 PASS、FAIL/error/skip0、分子計算3件をdeselect。うち新機能58件を、追加修正・異常系を含む`technical/final_new_tests.xml`の65 PASSへ更新した。重複を除く最終範囲は**既存371＋新規65＝436 PASS**、FAIL/error/skip0、除外3。全repositoryの全testを実行した意味ではない。対象21既存module＋2新moduleは`technical/regression_scope.json`に記録する。guardが持つ追加2件は対象外の`test_rte_h2.py`にあり、今回のdeselect数には含めない。

`technical/eight_qubit_final/`は本物の分子ではなく、供給済みPauli/DF行列と共通人工状態を使う8 system/9 wrapper qubit検査。q8/r4/K4、PR各2 paired trajectoriesで4 caseを完走し、20精度行・6 trajectory join・128実event・12 wrapper compile requestを保存した。最大compile測定確率差5.056e-13、captured trajectory operator差1.175e-14。全20行は既存allowance条件によりN/G nullであり、H4の科学結果や方式の優劣ではない。order0/2/4は別の人工代数検査。入力processと4 case processは全回収済み。

`technical/eight_qubit_run01/`と`*_attempt01`は開発時の人工検査履歴。`final_workers/`・`final_new_workers/`は拒否・中断・部分結果・破損検査の人工証拠を保つ。異常系fixtureのFAILED/STOPや故意のcorrupt fileは、その検査が期待どおり拒否した証拠であり、実分子runではない。絶対symlinkは当時のpytest fixtureとしてそのまま保つ。

技術試験は作業中treeで実行した。記録中のSOURCE=f7da77daは当時のbase HEADで、clean科学SOURCEとして扱わない。最終技術ファイルはSOURCE.jsonのhashとローカルcommitへ固定する。後からテスト記録をclean runへ書き換えない。最終E2E後の変更はsnapshot binding/primary result schema検査/未閾値spin liftの防御と試験追加であり、最終65件で検証した。最終科学条件を使う実分子の検証は未実施。

保存証拠だけの再検査：

```bash
python3 -B scripts/verify_h4_execution_foundation.py
```

旧科学.pyと旧H2/H4-readiness証拠はhash不変。library READMEへの新入口追記だけを保全監査上の許可された文書変更として区別する。旧grant/ledger/marker/STOP、保存図・比較表、旧契約は変更していない。

次の担当：**ユーザー**が完成契約への一括実行承認を判断する。承認後は**Codex**がseal・新grant・入力生成・4case・結果保存を同じ承認範囲で実施する。現在は科学実行・詳細科学レビュー・pushへ進まない。
