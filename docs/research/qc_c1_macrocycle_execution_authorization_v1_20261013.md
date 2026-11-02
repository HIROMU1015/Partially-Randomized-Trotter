# C1 fixed pilot v1：ユーザー実行承認

2026-10-13 JST。ユーザーが会話で次を明示した。

> C1の固定4入力・940 eigensolver callsの限定pilotを、GitHubに保存された固定契約どおり一回だけ実行することを承認します。独立監査・結果公開後はMandatory STOPしてください。

[認可JSON](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/authorization.json)、[実行前identity監査](../../artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/prelaunch_identity_audit.json)。
[固定scope](qc_c1_macrocycle_execution_approval_scope_v1_20261013.md)のsource `ddaa05d2b51d575c3cfea1194bbfdffa182f9636`、入力・contract・plan・runtimeのSHAを変更しない。
準備公開commit `2d1d00a589a1e350dfd9325552d661eb0ed70c47`、branch `qc-c1-macrocycle-preparation-20261013`、origin HIROMU1015。

固定4件の2qubit非分子toy、39unitary×6h、reference4＋PF936＝940 eigensolver calls。
C2導出72、保存のみ独立監査、結果/失敗記録・GitHub公開・remote取得確認まで。
attempt1/retry0、2spawn workers、各BLAS/OpenMP1thread、既存memory/output/log/call制限を維持し、wall-time capは設けない。
Pauli closed-form/cache/独立case並列を使用し、原sourceを変更して加速しない。測定済みspeedupのclaimはない。

直前の中断ターンはread-only preflightまでで、起動receipt/attempt directory/科学出力はないことを確認した。
IDE metadataは対象・認可に使用しない。新規分子、数値最適化、compile、sampling、C4起動、追加case、科学retryを含めない。
運用selectorをreference前にsealし、真値を選択へ戻さない。reference h frontierは事後診断として保存する。
現認可はGPT科学レビュー開始承認ではなく、新規性/中心研究の採択も含まない。
結果/独立監査/公開後Mandatory STOP。N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template終了と他Trackを保持する。
