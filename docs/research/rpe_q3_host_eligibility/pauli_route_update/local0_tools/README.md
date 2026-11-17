# Pauli remote 2-cell delta dry-check

PR-Pauli の server1 (`qg1`) と local (`RomuTail`) だけを更新します。現行の 10 PASS は再実行せず、SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、job `805877c2ab6c3e03f7f1`、REQUEST `ed04775d03240eee59d639fd039842652f28eaf4472f59cf4f12f9192a35f693`、Pauli route `33e324b43690caeaa353ecdb9e3d42f3139198b6` を使います。

新しい専用 directory に checker と pure profile helper の 2ファイルを配布し、公開済み host profile の Python から既存 adapter の `--mode dry-check` を呼びます。既存 SOURCE の 8ファイルと固定入力は checker が検証します。route view が未配置なら、そのホストの既存 repository の内側に指定 commit の専用 worktree を置きます。既存 HEAD や tracked source が異なる場合は保全して停止します。旧339の入力・runner・spoolは変更しません。

## local0 / Windows PowerShell

公開後の [RUN_PAULI_DELTA.ps1](RUN_PAULI_DELTA.ps1) 全体を PowerShell で実行してください。標準の `py -3`、OpenSSH `scp`/`ssh` と既存 alias `qg2`、`qg1`、`RomuTail` を使います。ZIP SHA256 は `5c5273436d2fafd65c2be5294cd67313a8dc7fcafee3839398b49601d66c1a70` です。

runner はホストごとに開始・返答と最終セル結果を表示します。2件とも `REAL_HOST_DRY_CHECK_PASS` で、qg2 への返送・hash照合まで完了すると終了 code 0 です。失敗時は新しいローカル `results` とホストの専用 directory を残します。ローカルに `RETURN_PAYLOAD.json` がある状態で返送だけ失敗した場合は、次を実行します。dry-check は繰り返しません。

```powershell
$RpePauliHome = Join-Path $env:USERPROFILE 'rpe-pauli-route-33e324b-delta-20261117'
py -3 -B (Join-Path $RpePauliHome 'run_delta.py') --return-only (Join-Path $RpePauliHome 'results')
```

## 返送と検証範囲

qg2 の受領先は `docs/research/rpe_q3_host_eligibility/evidence/pauli_route_33e324b_delta/` です。`server1/` と `local/` の `HOST_FACTS.json`、`RUNTIME.json`、`PROFILE.json`、Pauli の `READINESS.json`、`RESOLVED_REQUEST.json`、adapter log / `DRY_CHECK.json` を返します。root の `TRANSFER_MANIFEST.json` は相対 path → SHA256、`RETURN_RECEIPT.json` は全返送 hash の一致と件数です。同じ bytes の返送は再受領できます。

この bundle の作成・人工検査では実 SSH、実 dry-check、科学計算、queue 操作を一度も行っていません。検査は [ARTIFICIAL_ACCEPTANCE.json](ARTIFICIAL_ACCEPTANCE.json) と `test_run_delta.py` に記録しています。人工 SSH/Git/checker だけで、Pauli 2セル、旧 PASS 再実行0、SOURCE 16ファイル保全、返送19ファイルの hash一致、返送 retry の追加 checker0、Windows `CREATE_NO_WINDOW` を確認しました。実ホストの PASS は返送結果受領後に判定します。

`PAULI_DELTA.zip` の入力・個別 hash は [BUNDLE.json](BUNDLE.json)、pack 元は [PACKAGE_CONFIG.json](PACKAGE_CONFIG.json) です。再pack は `run_delta.py --package PACKAGE_CONFIG.json --output-zip NEW.zip` で行えます。科学実行の入口は実装していません。
