# 新A：長round移送・全support資格の保存結果

2026-11-06開始、11-07完了。科学実行は `STOP.json` で終了。保存済みrunを再実行しない。
[REPORT](../../../docs/research/pr_reference_transfer_qualification_20261106/REPORT.md)と
[SCOPE](../../../docs/research/pr_reference_transfer_qualification_20261106/SCOPE.md)に条件・限界を記載する。

- `INPUTS.json`：基準79cba2fの入力・旧SOURCE来歴。pickleは基準commitの旧pilot `common.pkl.xz`を再利用し重複公開しない。
- `ENVIRONMENT.json`：SERVER 2、runtime、packages、compiler profile、資源観測。
- `SOURCE.json`、`EXECUTED*.py`：最終SOURCEと実行中の版・呼出順・修正経緯・archive hash。
- `STAGE1_STRUCTURE.json`：旧48価格・native QPYの再利用、source構造、棄却したnaive外挿。
- `STAGE2_PREDECLARED.json`、`LOCAL_CATALOGUE_SCOPE.json`：価格cohortと局所contextの事前固定。
- `ADAPTER_CHECKS.json`：旧r1再構成・r2変更部分だけのcorrectness。
- `EXECUTION.json`：価格worker、固定request、実行SOURCE/preregistration hash。
- `raw_prices.tar.xz`：16 paired path、両軸・両armの64 whole compile、seed/event/trace/metrics。`PAIRED_NATIVE_PRICES.tsv`はその価格欄の転記。
- `native_circuits.tar.xz`：q4/j0/Xの両stratum・両arm、計4 native QPY。
- `local_catalogue_initial.tar.xz` / `local_catalogue_boundary.tar.xz`：12 contextずつ。後者は最終X/CX cancellation前のboundary観測も保存。
- `STAGE2_CX_CERTIFICATE.json` / `RZ_PATH_BOUND_CHECKS.json`：構造上の価格・上界と旧新128軸の照合。
- `STAGE3_LOCAL_INTERVALS.json`：445局所区間上界と解析的RZZ/CRZ検査。192-bit。
- `STAGE3_ALL_SUPPORT.json`：全18round・両armの誤差、phase arithmetic、raw半径・統計予算と算術仮定。
- `STAGE4_RPE.json` / `STAGE4_ROUNDS.tsv`：新しい十分整数shot、finite-bit有理期待価格、総G上界/下界。depth/size/T未資格を保持。
- `STOP.json`：mandatory STOP。旧研究のSTOPを解除しない。

archiveは展開元の全ファイルとbyte単位で照合した。展開元directoryとcacheは `.gitignore` で重複公開を防ぐ。
外向き数値上界と観測価格、理論モデルとhardware保証、著者検証と第三者レビューを区別する。
