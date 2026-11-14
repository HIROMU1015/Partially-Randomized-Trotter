# 技術テスト記録

環境：既存プロジェクト専用venv、Python 3.12.3。`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=src:tests`、`python -B`。共有環境・他担当worktreeを変更していない。

実行コマンド（worktree rootから）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B tests/check_m5_m4_receipt_admission.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B -m unittest check_price_implementation_guard check_paper_g_price_routes
```

結果：新規 **7/7 PASS**（3.768s）、既存 **48/48 PASS**（12.676s）、計55 test method PASS。

新規テストは人工receipt値・人工scalar・人工fit値のみを使う。固定実metadataはDの候補/役割/identity照合だけに使う。実保存85,960 RZをtest fitへ入れていない。

- 人工receiptのX/Y順序・cohortを保全。不足軸、scalar/候補/compiled profile/元gate/価格不一致を拒否。
- 13-round metadataから固定m4/m5/m6のみを選択。qは7/14/28、3/6/12、1/2/4。新規34・保存2に対応し、scientific entryは抽出していない。
- 既存C guardへ人工recordを渡し、`DETERMINISTIC_DIRECT`を確認。scalar・SOURCE・軸・physical ID不一致を拒否。G関数を起動しない。
- 12個の人工affine modelで既存fit/validateを再利用。観測値分母2%の境界、不合格後の再validation不可、qualification falseを確認。
- 既存48件はC guard、Pauli bridge、paper会計、price model検証の人工回帰。会計の値は人工入力である。

別に実施した保存receiptの技術照合と固定bundleを用いたguard確認は[ADMISSION.json](ADMISSION.json)・[GUARD_CHECK.json](GUARD_CHECK.json)。既存gateの再実行、実回路compile、実価格fit、実N/Gの評価ではない。

作者自身の技術確認。独立科学レビューではない。
