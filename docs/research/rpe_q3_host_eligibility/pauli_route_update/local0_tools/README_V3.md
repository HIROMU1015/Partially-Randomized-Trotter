# PF / B0I remote 4セルの runtime guard 更新（V3）

[RUN_PF_RUNTIME_DELTA_V3.ps1](RUN_PF_RUNTIME_DELTA_V3.ps1) 全体を local0 の PowerShell にコピーして実行します。新しい専用 home `rpe-pf-runtime-cbc471c91-delta-20261117-v3` と `results` を使い、`qg1` / `RomuTail` の Full PF と B0I の計4セルだけを dry-check します。DF3セルとPauli3セルは再実行しません。

ZIP は [PF_RUNTIME_DELTA_V3.zip](PF_RUNTIME_DELTA_V3.zip)、SHA256 は `bf808423ba8f4f98bf4dacbe4ef9ead1f0e09a7ed56d5620100d2683b7943747` です。SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`、PF/B0I の科学REQUEST / job ID / 入力は339原本とbyte一致です。routeは既配置の `cbc471c913072de45c25a0ed22d51dff8a9b7034` を再利用します。既存 SOURCE、route files、環境、旧input/spoolは変更しません。

PFの実execute経路にも固定acceleratorのAST guardがあるため、旧metadata-only dry-checkのPASSだけではruntime条件を満たしたと判断できませんでした。V3ではFull PF/B0IにもCBC routeの `total_state_dp` AST hash と既存 guard 定数の一致を要求します。関数本体は実行しません。qg1 PF/B0Iは既存9d公開の311専用launcherへ切替え、localは既存 `.venv311/bin/python` を使います。Pythonの再探索・環境修正はしません。

runner / checker / pure profile helper はPauli V2と同じ固定bytesです。manifestが各remote hostの対象2方式を明記し、1つの公開host profileで順に確認します。共通 `HOST_FACTS.json`、`RUNTIME.json`、`RUNTIME_CHECKS.json` は両方式で一致し、方式別 `dry/FULL_PF/`、`dry/TRUNCATED_PF/` に独立したREADINESS / RESOLVED_REQUEST / adapter raw logsを保存します。AST runtime確認がFAILならadapter dry-checkを開始せずruntime記録を返します。PASS後に既存adapterの `--mode dry-check` だけを呼びます。

qg2の受領先は `docs/research/rpe_q3_host_eligibility/evidence/pf_runtime_cbc471c91_delta_v3/` です。manifest / receiptにより返送bytesを照合し、既存deriveが更新した12セルpolicyを採否判定します。4件とも `REAL_HOST_DRY_CHECK_PASS` で、返送まで完了すると終了code 0です。結果がある専用directoryを同じ指令で再dry-checkしません。

返送だけ失敗して手元に `results/RETURN_PAYLOAD.json` がある場合は、保存結果だけを再送できます。

```powershell
$RpePfHome = Join-Path $env:USERPROFILE 'rpe-pf-runtime-cbc471c91-delta-20261117-v3'
py -3 -B (Join-Path $RpePfHome 'run_delta.py') --return-only (Join-Path $RpePfHome 'results')
```

人工SSH検査5件 PASS。V3追加検査では4セル限定、同host2方式の共通runtime/facts一致、SOURCE16ファイル保全、返送31ファイルhash一致、返送retryの追加checker0、DF/Pauli呼出0を確認しました。既存検査はruntime guard不一致時のchecker開始0、launcherと実binaryの区別、Windows `CREATE_NO_WINDOW` も確認します。[ARTIFICIAL_ACCEPTANCE_V3.json](ARTIFICIAL_ACCEPTANCE_V3.json) は人工検査です。bundle作成時の実SSH・実dry-check・科学計算・本番queue操作は0、V3のWindows native実行は未実施です。

個別hashと原REQUESTの対応は [BUNDLE_V3.json](BUNDLE_V3.json)、pack元は [PACKAGE_CONFIG_V3.json](PACKAGE_CONFIG_V3.json) です。旧Pauli V1/V2 ZIP・受領原本、以前の12セル人工チェックポイントは保全しました。最終12セルの人工割当・呼出し検収はPF runtime証拠の受領・policy更新後に再確認します。
