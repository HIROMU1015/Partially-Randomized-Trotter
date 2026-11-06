# 同じ承認範囲内の技術修正

- run01 / SOURCE `63796aafb4cfcca18c9fefa9120ebe7a47ad6350`：核scalarの
  1D診断を2D前提のrow-sumへ渡しAxisError。比較結果ファイルは未生成。
  scalar/vectorを`atleast_2d`へ通す変更だけを行った。
- run02 / SOURCE `c6dbfe140452969e4d716d9639c068f1c8522338`：結果JSONは
  `STOP_OPERATOR_VALIDATION_INELIGIBLE`を出力したが、失敗した3件はすべて
  full Fockのbinary64対120桁の照合。静的確認で、MPのOpenFermion MSB行列を
  Qiskit LSB行列と直接比較していたことが判明した。これは入力/モデルの
  不適格を示す診断ではなく、比較driverのbit-order実装誤りである。
  元のrun02 JSON・SOURCEは保存し、PASSへ書換えない。
- 修正版は同じfull行列を既存bit-reversalでOpenFermion順へ戻してから照合する。
  人工複素2-mode例で誤順序の失敗と正順序の一致を両方検査した。
  独立参照の構築、科学入力、target定義、precision、factor、許容値は不変更。
  これはユーザーが承認した第一段階の同一条件内のdebugであり、新条件の採用・
  元H state solve・次段階への進行ではない。

run03を最終の第一段階結果とし、その判定・公開後はmandatory STOPする。
