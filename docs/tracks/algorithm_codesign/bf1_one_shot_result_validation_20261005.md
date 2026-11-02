# BF-1 one-shot incomplete result validation

2026-10-05 JST。status: `BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`。
outcome: **`INCONCLUSIVE`**。科学実行は終了し、mandatory STOPを維持する。

固定sourceと正式authorizationでBF-1を一回実行したが、JSON保存時の`TypeError`で中断した。
F vs Lのdecision-relevantな差は判定できていない。BF-A/B/Cのいずれにも分類しない。
これは実装・保存処理の失敗によるincomplete runであり、中心仮説のnegative resultではない。

## 1. Source、authorization、入力scope

| 項目 | 値 |
|---|---|
| B worktree | `/home/abe/Project/prt-worktrees/track-b-algorithm-codesign` |
| Branch | `track-b-algorithm-codesign` |
| Source S | `e59344a564e70d64dc3ea39d640581c72676df31` |
| Authorization-only A | `cc971e4a2bff9b0c5708003fde7b9519eed27241` |
| Review依頼の公開commit | `e911db6b409f9814386f31b99c6a427cf2771501`、別branch・science chain外 |
| Execution ID | `bf1-20261005-development-v1` |
| Plan fingerprint | `602f6c0fbefbd48499fb0c2c9348753bd72141db7970f064572f214cc0fd63df` |

[authorization receipt](bf1_execution_authorization_v1.md)は利用者提示の最終reviewと、
scope確認への別の明示的応答「はい、進めてください」を記録する。
AはSの直接の子で、追加は正式authorization JSONとreceiptの二pathだけ。
Aをcommit・pushし、全77 sealed source、plan/domain/test、環境、JSONのcommitted bytesを
入力操作前に照合した。実行直後・結果文書更新前にも77 sourceの一致を確認した。

入力はknown/development I2のH4 linear 1.00 Å、STO-3G、8 qubits、DF rank12、
generation-prefix `L_D=3`、one-body込み4 D generators、`T=0.8`。
5-stage symmetric fourth-order family、O/L/F各32評価、四fixed refs、
`q={1,2,4,8}`、`R_bud={5,10,20,40,80}`、固定allocation、K2と事前triggerのみのK4を維持する。
primary `epsilon=0.01, alpha=0.05`、materiality ratio `<=0.95`、bridge `epsilon=0.05`。
根拠は[v1](bf1_preregistration_v1.md)、[v2](bf1_execution_gate_revision_v2.md)、[v3](bf1_assembly_guard_revision_v3.md)。

runnerのinput auditは保存済みdevelopment snapshotのraw SHAとHamiltonian/state/state-vector identityの一致を記録する。
raw SHAは`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
新Hamiltonian/state solveはなく、M2 1.30 Åも使わない。新held-out・blind validationではない。

## 2. 実行結果と保存範囲

| 項目 | 観測 |
|---|---|
| Runner status | `BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY` |
| Outcome | `INCONCLUSIVE` |
| Exception | `TypeError: Object of type int64 is not JSON serializable` |
| Wall / CPU | 41.2755 s / 41.2689 s |
| Process peak RSS | 368,316,416 bytes |
| 保存済みideal / finite records | 200 / 1,069 |
| 記録されたdistinct coefficient identities | 50 |
| 保存済みcross-objective records | 0 |
| Primary decision / cross-score / bridge | resultへ未保存 |
| 新分子生成 / trajectory / circuit / compile / GPU query | 全て0 |

runnerのOS exit codeは0だが、例外を捕捉してincomplete statusを保存したためであり、science成功を意味しない。
cell recordsは重複0、全1,269行が有効JSON。保存済みfinite recordsは全件`valid=true`だが、
これはalgorithm actionの評価が記録されたという意味で、accuracy適格性や研究GOを表さない。
arm別search recordやcoefficient定義もresultへ未保存で、各armの完了範囲を保存結果から認証しない。
メモリ上の未保存primary判定を推定・再構成しない。

raw outputは合計755,463 bytes。固定上限はideal 400、finite 4,000、CPU 8h、wall 4h、
process RSS/AS 4GiB、aggregate 8GiB、output 128MiB。独立のaggregate peak profileは取得していない。
今回の保存理由はserialization例外であり、resource capや数値guardによる不適格とは区別する。

## 3. 保存処理の原因候補と確認の限界

[cross_objectives.py](../../../src/trottertracks/algorithm_codesign/cross_objectives.py)の54行は、
`union_rank = 1 + sum(v < value for v in values)`を用いる。
objective値が`numpy.float64`なら比較は`numpy.bool_`となり、和は`numpy.int64`になる。
標準JSON serializerはこのrankをそのまま保存できない。

STOP後、二つのsynthetic scalar値だけでこの式とserializerを検査し、
`numpy.int64`と保存されたrunと同じ`TypeError`を確認した。
science evaluator、coefficient search/rescore、signal計算は呼んでいない。
これは有力な失敗経路の診断である。runnerはtraceback・phaseを保存していないため、
science runの正確なthrow locationを保存traceから確定したとはしない。
source修正、test suite追加実行、science retryは行っていない。

## 4. Artifact identityとevidence境界

- [result.json](../../../artifacts/track_b_bf1_execution/2026-10-05/v1/result.json)：
  4,834 bytes、SHA256 `99ef73f93f6303dbb1c85ac887dbfd66d02e006305938da0623f880493ea8250`。
- [cells.jsonl](../../../artifacts/track_b_bf1_execution/2026-10-05/v1/cells.jsonl)：
  750,629 bytes、SHA256 `a2974de79d454e062b41c4485e58ef07a0152ba5bf7163912575f0d2af1da0a6`。
- [result validation audit](../../../artifacts/track_b_bf1_execution/2026-10-05/v1/result_validation_audit.json)：
  source/input binding、partial inventory、停止条件、serialization診断を記録する。

raw result/cellsは変更・再生成せず、旧preparation 14ファイルとauthorizationのcommitted bytesも保持した。
source-bound local **incomplete** execution evidenceであり、result commitへ収録する。
auditの未commit表現はaudit作成時点の状態として保持する。commit保存もimmutable CI、外部再現、
independent validation、最終総costを意味しない。

## 5. Mandatory STOP

B one-shot markerはconsumed、retryは禁止のまま保持する。markerを削除・付け替えない。
`mandatory_stop=true`、`automatic_next_stage=null`、`BF2_authorized=false`、retryなし。
既存cellからのpost-hoc rescue classification、source修正後の再実行、grid/geometry/family追加へ自動進行しない。
partial記録からF vs L、materiality、objective attribution、frontierを科学的結論として主張しない。
この結果と実装上の失敗を利用者reviewへ戻し、BのRQ・新規性・着地点の全面再評価はそのreviewで扱う。
