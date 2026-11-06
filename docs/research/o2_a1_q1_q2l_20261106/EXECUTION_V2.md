# A1 Q1/Q2-L v2：比較payload保存の修正と限定再実行

2026-11-06 JST。ユーザーが修正・再実行を明示許可した範囲に限る。
旧v1のSOURCE `8e002a3815fb4fba709cf090e9c975052f8dd892`、失敗結果、STOPは保存し、新branch・新directoryで作業する。
対象の保存H4/1.00 Å/STO-3G/legacy DF12/4電子/8qubit/L_D3/m0/q1/δ.02/r1/K2/threshold0、a/b、期待hashを変更しない。

最初の修正はhash比較より前にpartitionの全fragment/weight/λ、basis操作とhash、完全component/係数、storage dtype/strideを保存すること。
科学条件を変えずに同じpayloadへ旧/現partition-hash関数を適用し、SOURCE拡張だけが原因かを確認する。
追加診断は、同じtensor値のloaded/complex C-copy/complex F-copy/虚部が厳密0の場合のreal C-copyという4つの事前指定storage表現だけ。
これらはrank/weight丸めを調べる診断で、新しい分子・DF分解・sampling law候補ではない。tensor値の完全一致を記録し、診断のcaseを自動的に科学入力へ採用しない。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B \
 scripts/tracks/algorithm_codesign/run_a1_q1_q2l.py --diagnose \
 --output-dir artifacts/o2_a1_q1_q2l/2026-11-06/diagnostic_v2
```

診断結果に基づく修正は、元opt2のsplit/law/finite targetとの対応を立証できる範囲に限る。
期待hashの書換えや単なるguard削除はしない。非意味的なhash差と科学的な表現差を区別する。
その対応が確認できた後だけ、新出力v2でQ1/Q2-Lを限定再実行する。対応が閉じなければ詳細payloadを残してSTOP。
Q2-T、physical-sector state solve、RNG、Hadamard測定、full interrogation build/compile、variance/N/cost、Q3–Q5は範囲外。
共有core/元WP11/共有環境を変更せず、再利用する元SOURCEと必要な接続差分を別に記録する。
