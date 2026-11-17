# Pauli remote 2セルの runtime guard 付き dry-check（V2）

利用する手順は [RUN_PAULI_DELTA_V2.ps1](RUN_PAULI_DELTA_V2.ps1) です。Windows PowerShell に全体をコピーして実行できます。標準 `py -3` / OpenSSH と既存 alias `qg2`、`qg1`、`RomuTail` を使い、server1 と local の Pauli だけを確認します。旧 V1 metadata 版は使いません。

ZIP は [PAULI_DELTA_V2.zip](PAULI_DELTA_V2.zip)、SHA256 は `d095f0750a6e03be26b9e37d005527c3cd238cc3b9a220873aead9c5d55b3641` です。新しい Windows directory `rpe-pauli-route-33e324b-delta-20261117-v2` と、その中の `results` を使います。既存 `results` がある場合は新たに dry-check を開始しません。

科学 SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、job `805877c2ab6c3e03f7f1`、REQUEST `ed04775d03240eee59d639fd039842652f28eaf4472f59cf4f12f9192a35f693`、Pauli route `33e324b43690caeaa353ecdb9e3d42f3139198b6` は維持します。現行10 PASS、他3方式、旧339の入力・runner・spoolは再実行・変更しません。

各ホストに新しい専用 directory を作り、checker と pure profile helper を所定の相対 layout で置きます。server1 Pauli は既存9d公開の専用311 launcher、local Pauli は公開済み `.venv311/bin/python` を使います。新しい Python の探索や環境修正は行いません。SOURCE の8ファイルを先に確認し、未配置の route は repository 内の専用 sparse worktree（scripts/src/configs/schemas）に置きます。既存 view の HEAD または tracked SOURCE が異なる場合は保全して止まります。

helper の `check_runtime_profile(profile)` は同じ interpreter で固定関数 `total_state_dp` の AST hash と既存 accelerator の guard 定数を照合します。関数本体は実行しません。launcher 呼出しと実際の binary を区別し、公開 runtime 要件とも照合します。この確認が FAIL なら adapter dry-check を開始せず、その raw runtime 記録を返します。PASS 後だけ既存 adapter の `--mode dry-check` を使います。Q3実行・信号計算・較正・circuit compile・collect・本番queue の入口はありません。

runner は各ホストの開始・返答と結果を表示します。2件とも `REAL_HOST_DRY_CHECK_PASS` で、qg2 への返送照合まで終わると終了code 0です。qg2 の受領先は `docs/research/rpe_q3_host_eligibility/evidence/pauli_route_33e324b_delta_v2/`。`HOST_FACTS.json`（actual_runtime / runtime_checks）、`RUNTIME.json`、`RUNTIME_CHECKS.json`、`PROFILE.json`、`dry/PR_PAULI/READINESS.json`（runtime_precheck）、`RESOLVED_REQUEST.json`、adapter logs / `DRY_CHECK.json` を返します。root の manifest と receipt で返送 bytes を照合し、既存 derive が採否を判断します。失敗時も新しい各専用 directory を残します。

返送だけ失敗して手元に `results/RETURN_PAYLOAD.json` がある場合は、次で保存結果だけを送り直せます。host dry-check は繰り返しません。

```powershell
$RpePauliHome = Join-Path $env:USERPROFILE 'rpe-pauli-route-33e324b-delta-20261117-v2'
py -3 -B (Join-Path $RpePauliHome 'run_delta.py') --return-only (Join-Path $RpePauliHome 'results')
```

人工検査4件 PASS。Pauli2セルだけの配送、SOURCE16ファイル保全、runtime guard不一致時のchecker開始0、launcher/binary hash保持、checker/runtime記録一致、返送21ファイルの hash、返送retryの追加checker0、Windows `CREATE_NO_WINDOW` を確認しました。[ARTIFICIAL_ACCEPTANCE_V2.json](ARTIFICIAL_ACCEPTANCE_V2.json) は人工測定です。このbundle作成では実SSH・実dry-check・新科学計算0、Windows native実行も未実施です。実ホストのPASSは返送後に判定します。

固定入力とpayload hashは [BUNDLE_V2.json](BUNDLE_V2.json)、pack元は [PACKAGE_CONFIG_V2.json](PACKAGE_CONFIG_V2.json) です。V1 ZIP と metadata 記録は保全しています。
