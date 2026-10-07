# RA-D0 v3 one-shot result / saved-value audit / GPT handoff

2026-10-07 JST。最終分類は **`D0_TECHNICAL_INCONCLUSIVE`**。
固定saved-table development one-shotを一回実行し、P1最初のB2 T minimumで
`TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION`となった。mandatory STOP到達、retry=0。
**比較queryは未実行であり、改善の証拠にもnegative resultにも使わない。**

## 固定identityと実行前確認

- source S：`45cffb2aa10f9219b6cad929c3ade49fe7d36ca8`。
- authorization A / 実行開始HEAD：`2daaf3b60a33db58de8fcdbbcce06f8e9ff163d9`。唯一の親はS。
- execution branch：`track-b-ra-d0-one-shot-execution-20261007`。
- 独立worktree：`.worktrees/track-b-ra-d0-one-shot-authorization-20261007`。
- authorization branchのremote SHA=A、実行HEAD clean、許可2 pathのみのS→A差分を確認。
- read-only launch gate PASS。全97 critical pathsとsource manifestはSとbyte-identical。
- contract SHA256：`e9422bab70389a41715d8fe817060b811ed492ea46a0a3a4a54f23d043f4afd1`。
- authorization SHA256：`9419a1ab7cf015afa29f8ed3c17eb6231525ac871403d98e4cdbd5c96fd569fe`。
- 固定runtime：`/home/abe/Project/prt-worktrees/ra-d0-technical-venv/bin/python`。
  Python 3.12.3 / NumPy 2.5.3 / SciPy 1.16.2。version・RECORD identityは実行前後とも固定manifestと一致。
- 実行前markerなし。以下のrunnerを一回だけ呼び出した。終了code=0、stdoutはtechnical分類を報告。

```sh
PYTHONDONTWRITEBYTECODE=1 /home/abe/Project/prt-worktrees/ra-d0-technical-venv/bin/python scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py
```

source、solver設定、denominator、tolerance、candidate、grid、resource capsは変更していない。
branch切替時のsparse checkoutでは、固定Aのauthorization/receiptだけを追加materializeした。
これはGit treeの変更ではなく、runner呼出し前のread-only gateで全identityを再照合した。

## 登録scopeと停止箇所

saved 2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x={1/8,1/4}、
既存3 implementation precisionsのみ。分子geometry・basis・DF rank・split L_D・PF delta窓は適用外。
新しい入力、合成、行列評価、回路構築、trajectoryは取得していない。

保存されたlast taskは`P1_ANCHORS:1/8:767135:minimum:T`。
[full certificate records](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/certificates.jsonl.gz)
には、このB2 minimumのnominal primal、dual certificate、implementation certificateを一件保存している。
HiGHS status=0 / Optimalだったが、implementation certificateは`certified=false`、
`reason=UNCERTIFIED_NUMERICAL_POINT`。membership certificateもfalseで、
保存group内の`ordinary / O0`について`residual_upper > tau`となっている。
この未認証minimumをbudgetや比較へ渡さず停止した。

これはexact-certified infeasible outcomeではない。Farkas取得、正常skip、次のminimum、
second-best探索、設定変更、追加solver callを行っていない。
停止後の監査は保存JSON/gzipのfield・hash・既存閾値のflag整合とprovenance照合だけで、
LP再求解、実装law再構築、certificate再取得、再分類は行っていない。

## 完成範囲と会計

| 項目 | 保存結果 |
|---|---:|
| runner呼出し / retry | 1 / 0 |
| P1 | 開始、最初のT minimumで停止。freeze未完成 |
| P2 conditional coverage | 未開始 |
| completed budget-ready points | 0 |
| B2 certified-infeasible skipped points | 0 |
| completed / certified-comparable queries | 0 / 0 |
| strict witnesses | 0（比較未実行） |
| main / auxiliary / total LP calls | 1 / 0 / 1 |
| wall / CPU seconds | 0.15188044076785445 / 0.151563673 |
| peak RSS | 78,936 KiB |
| runner output bytes、terminal result書込み前 | 7,618 |
| runner output bytes、terminal result書込み後 | 8,896 |

resource usageはcontractどおりterminal serialization前のsample。
後の8,896 bytesは元runner四ファイルの実byte数で、監査資料・本書のbyte数を含めない。
wall/CPU/RSS/output/LP capへの到達はなく、numeric implementation certificate failureである。

| x | anchor strict witness | coverage strict witness | completed comparable queries |
|---|---:|---:|---:|
| 1/8 | 0 | 0 | 0 |
| 1/4 | 0 | 0 | 0 |

両xともcompleted pointsは0。1/4は未到達。prefixをSTRONG/LOCAL/NO_WITNESSへ読み替えない。
certified-infeasible skipはstrict witnessにもnegative evidenceにも数えない契約を維持している。

## 保存資料とprovenance

- [result.json](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/result.json)：runner原分類・資源・input identities。
- [technical_failure.json](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/technical_failure.json)：reasonとlast task。
- [one_shot_consumed.json](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/one_shot_consumed.json)：exclusive marker。
- [saved-value audit](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/saved_value_provenance_audit_v1.json)：保存field整合、元artifact hashes、runtime/source照合。
- [evidence manifest](../../../artifacts/track_b_ra_d0_development_result/2026-10-07/v3/evidence_manifest_v1.json)：元runner出力・監査・本書のidentity。

marker SHA256は`88a471e637d57c9896ffa9d3f6442c86ff1fec9859b50f3e86c4583095f7e735`。
元result/certificate/failure/marker bytesは停止後の監査でも不変。
source manifest・全97 critical files・旧R1/R1.5 protected evidenceは不変。
authorization/receiptもAとbyte-identical。source manifestに固定された概要や索引は変更せず、
本結果の入口として本書を追加する。Track Aや既存共通APIの変更はない。

## GPTへ戻す判断

これはsource-bound local developmentのtechnical incomplete runであり、immutable CI・
独立再現・性能改善・algorithm採択の証拠ではない。
saved-value consistency auditのPASSは、実行成功やscientific GOを意味しない。
研究方針、RQ、新規性、着地点、および将来の追加検証の必要性・範囲はGPTが判断する。
今回の認可は消費済み。source修正や別runは認可されていない。

**mandatory STOP。停止後solver=0、retry=0、new synthesis/science/circuit/matrix/trajectory/DF/molecule/NPZ/GPU=0。
自動でgrid、precision、denominator、tolerance、IS、CTSを追加しない。**
