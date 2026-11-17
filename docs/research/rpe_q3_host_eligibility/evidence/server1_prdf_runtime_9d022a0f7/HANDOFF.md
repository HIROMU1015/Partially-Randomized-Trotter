# サーバー2・ローカル0向けPR-DF readiness

指定計画 `33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b` のREQUEST
`48494ae31276a64c46ce`、計算SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774` は
サーバー1で **REAL_HOST_DRY_CHECK_PASS**。実Q3・compile・queue操作は0件。

取り込み入口：

- [HOST_METHOD_READINESS.json](HOST_METHOD_READINESS.json)：host/method/job/SOURCE/route/request hash、
  interpreterの呼出pathと実バイナリpath/hash、version_info、profile・公式READINESSのhash。
- [公式READINESS.json](dry/PR_DF/READINESS.json) / [adapterログ](dry/PR_DF/ADAPTER.log)。
- [PR-DF専用profile](profiles/server1.prdf.python311rc1.json)。固定計画の既存server1.jsonは変更していない。
- [実行環境・保全・範囲](REPORT.md)。

`READINESS.json`最上位の`python`/`python_version`はchecker自身（3.12.3）。
PR-DF adapterの実runtimeは`command[0]`と`adapter_result.runtime`、および別記録の
`HOST_METHOD_READINESS.version_info`/`actual_executable_realpath`で判定する。
PR-DF用profileをFull PF/B0I用の既存profileと一括交換しない。

## サーバー1上での再dry-check

```bash
SERVER1_PRDF_CHECKOUT=/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/server1-prdf-fixed-runtime-readiness-20261117
SERVER1_PRDF_EVIDENCE="$SERVER1_PRDF_CHECKOUT/docs/research/rpe_distributed_h4_plan_20261117/server1_prdf_runtime"
python3 -B "$SERVER1_PRDF_CHECKOUT/scripts/check_rpe_q3_host_readiness.py" \
  --assignment "$SERVER1_PRDF_EVIDENCE/inputs/assignments/PR_DF.json" \
  --profile "$SERVER1_PRDF_EVIDENCE/profiles/server1.prdf.python311rc1.json" \
  --output "$SERVER1_PRDF_EVIDENCE/dry/PR_DF-recheck-NEW"
```

`--output`は毎回未使用directoryにする。checkerは固定計画33928bf80cの原本のまま。
内部では固定SOURCEのvalidate_assignmentとadapterの`--mode dry-check`だけを実行する。
mailbox submit・Worker生成・結果collect・queue投入は行わない。

## ローカル0での取得

既存alias `qg1`から上記`SERVER1_PRDF_EVIDENCE` directoryを`scp -r`で回収できる。
専用branch `codex/server1-prdf-fixed-runtime-readiness-20261117` にも同じ証拠を公開する。
サーバー1→2直接SSHは不要。

`HOST_METHOD_READINESS`のscopeはこの実REQUESTのruntime/count/metadataに限定する。
後続REQUESTはSOURCE・route・入力・profileに照合して判定し、今回は本番queueを起動しない。
