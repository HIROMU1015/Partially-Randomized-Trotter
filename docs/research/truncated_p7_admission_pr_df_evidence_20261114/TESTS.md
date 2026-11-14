# 技術テストと既存証拠の再利用

既存専用venv（Python3.12.3、Qiskit1.3.0）、PYTHONDONTWRITEBYTECODE=1、python -B。共有設定・他担当worktreeを変更していない。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B -m unittest check_p7_price_intake check_m5_m4_receipt_admission
```

**15 test method PASS、6.454s**（新p7/coverage8、既存Full PF受入helper回帰7）。P1参照追加後、影響する `check_p7_price_intake.Tests.test_existing_PR_coverage_reused_without_numeric_values` を再確認してPASS（1.074s）。同じtestの再確認を別method数へ加えていない。

- 旧CELL_PLAN不変、2 actionだけ変更、全36行/保存4/新較正20/holdout12。
- 軸欠測・役割変更を拒否。
- Dの既存record emitter＋C guardでp7の人工X/Y価格を受理し、BのDETERMINISTIC_DIRECTを確認。人工値は111/222 RZで、保存60,950はfitに入れていない。
- B0Iをconstant-only policyへ置換、resolved scalar・測定軸の不一致を拒否。
- 人工命令wordでX/Y suffix、scalar一回、control/measurement bit、別scalarを識別。同じ価格を付けてもaxisが違うwordを受理しない。
- D c848のselectは、新CELL_PLAN＋旧T3 actionを拒否。mockで保存actionを反映すると5新較正セル＝10 X/Y wrapperになる。D SOURCEは変更していない。
- PR52軸metadata、LD1のES26軸、同条件finite4軸、P1 n2/P2 n4の分離、資格false/null、C1の元4セル/8軸roleとcohort domainを保持。
- 既存Full PFの受入・guard・人工12 modelの回帰を再利用し、default動作を維持した。

保存p7の実技術照合はADMISSION/SAVED_WRAPPER_CHECKS/GUARD_CHECKに保存。既存QPYのcanonical fingerprintとscalar/suffixを読み取り、固定assertionを実際のGitから読ませた。mockを実receiptの資格証拠にしていない。新しい回路の生成・compile、state action、S/C計数、fit、N/Gを実行していない。

公開用checkpointの再利用入口では、保存QPYのbytesが不変であることを照合してcanonical hashの再計算を省いた。D c848の現record整形との2レコード一致もC_INPUT_COMPATIBILITYに保存。既存Full PF受入資料・旧CELL_PLAN・C guard/validator/modelは2fa88065から不変であることをgit差分でも確認した。

無変更のC628 guard/model/paper経路48 PASSは、固定C `2fa88065:docs/research/m5_m4_receipt_admission_20261114/TESTS.md` の証拠を再利用。C1・ES・P1/P2・law調査の科学実行記録も引用のみで、再実行・再集計しない。保存Full PFの原receiptを再監査していない。

技術準備中のparser key/配列順序に関する作者の不具合は修正済み。初回artifact生成ではC1 profile selectorでKeyErrorとなったが、科学処理はなく、既存の原本・旧run・STOPは変更していない。新しい科学retryという扱いにはしていない。

これらは作者自身の技術確認であり、独立科学レビュー・平均価格の科学的性能保証ではない。
