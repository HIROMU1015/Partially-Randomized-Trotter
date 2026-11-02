# 独立代数チェック

対象は固定source `25d7135f7a0285b6cf415349191b00c00acfb75f` に記載されたA/B/Cのtoy入力です。

必要な環境: Python 3、SymPy。既にSymPyが使える環境で、次を実行します。

```bash
python independent_review_checks.py
```

同じディレクトリに `independent_review_checks.json` を出力します。

元のdecimal入力を有理数として転記した代数チェックであり、元runの浮動小数点データ、RTE全event、compiler gate countを独立再現するscriptではありません。repository module・production runner・分子計算・sampling・量子回路合成を使いません。

Aの占有数保存・残差式・isotropic対照、Bの物理Hamiltonian、Cのt展開を検査します。固定共通射影coreの一般的な不変性はレビュー本文の代数的導出であり、この有限例のscriptを一般証明として用いません。
