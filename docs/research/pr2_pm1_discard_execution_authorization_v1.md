# Track A PM-1結果前authorization draftと最終レビュー待ち

2026-10-04。準備レビューの `APPROVE_PM1_RESULT_PRIOR_AUTHORIZATION_DRAFT` を受け、
PM-1だけの結果前authorization条件を固定します。今回はscience runを開始しません。

## 2026-10-05の追記：最終承認・明示launch待ち

最終review `APPROVE_PM1_EXECUTION`を受領し、
[確定記録](pr2_pm1_execution_finalization_20261005.md)どおり`final_review_approved`だけをtrueへ変更した。
別commit `bf9eaeec868361df0c8e05d06e9570a2bfc5a7a4`、他field/source/planは不変。
science-free実gate PASS、限定201 local tests passed、fail/skip0。本計算は0、利用者の明示launchまでSTOP。
以下のfalse・draft記述は2026-10-04の結果前記録として保持する。旧manifestもdraft時点のblobに対する履歴であり、
確定後の照合は別finalization manifestを使う。固定command/outputを含む実行条件は変更しない。

## draft時点のbarrierと承認後の手順

[authorization JSON](../../artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/execution_authorization_v1.json)は
`final_review_approved=false` です。既存runnerはこの値がtrueでなければ
`final review barrier not passed` として科学データ・registry・outputの前で拒否します。

JSONの `PM1_EXECUTION_AUTHORIZED_ONCE` は予定された一回の条件を表すstatusで、
最終review通過や現在のlaunch許可を意味しません。運用statusは
`PM1_AUTHORIZATION_DRAFT_FROZEN_AWAITING_FINAL_REVIEW` です。
[authorization schema](../../artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/authorization_schema_v1.json)も
科学条件を固定し、最終review booleanだけを後日の記録に対応できるようにしています。

独立reviewが `APPROVE_PM1_EXECUTION` を返した後だけ、承認の記録を別auditへ保存し、
このJSONの `final_review_approved` 一項目のみをtrueへ変えて別commitで固定します。
source、plan、candidate、compiler、environment、caps、permissions、root/outputは変更しません。
その一項目以外の差があればlaunchせずreviewへ戻ります。最終booleanの更新も利用者のlaunch指示ではありません。
現在の承認内容はdraft作成までであり、このturnではfalseのまま停止します。

最終review記録の真正性や利用者の明示launchをrunnerが独立artifactから認証するとは主張しません。
runnerが機械検査するのはcommitted authorizationのbooleanと既存identity gateです。

## 固定identity

- Repository / branch：`HIROMU1015/Partially-Randomized-Trotter` / `pr2-v4-s2-parallelization-20260928`。
- 保存M1/M2 evidence：`b6e65c6123475add5e620ec1064f361378bead95`。
- actual PM-1 source：`fd7552edc0334ccf57ecf501a128c85c8d22822a`。
- 準備review bundle：`d49555f0f1144aa1de6925ace249dc380ae6f4a6`。
- sealed plan SHA-256：`cae692bee2be748ddbf17bace2a5652613537a244fe94238cdf669f5d2ca5624`。
- plan fingerprint：`144824b70264dd3d7d1d22898afbb1f9e1c248f105ecac8096f31a2c8f78ec9e`。
- authorization draft SHA-256：`3916f1e050bf4d4174f0b2b5d4d1d82cdb12396f9eefe3a34a75e496e716377d`。
- authorization schema SHA-256：`675453d1a93019db0857743cbbc626cbadd4f15536bfbb280349e89acbf10941`。

134 source hashはplanと同じで、source commit blobと照合します。candidate fingerprint8件、
axis別wrapper keys16件も同じです。draft commitと最終review bundleの正確なcommitは後続依頼とhandoffで示します。
生成済みPM-0/PM-1 preparation artifactと旧M1/M2 evidenceは変更しません。

## 固定計算と比較範囲

[元契約](pr2_pm1_nearby_discard_contract_v1.md)を変更せず、development H4 linear 1.00 Å、
STO-3G、DF rank12、8 system qubits、同じsnapshot/state、T=0.8への
B0 discard rank4/5 × q=1/2/4/8、r=K=0だけを対象にします。deltaは0.8/0.4/0.2/0.1です。

将来の上限はsignal8、full measured Hadamard wrappers16、development hash/load各1、CPU1、BLAS各1。
random sampling、held-out、quantum shot、GPU、cache reuse、retry/resumeは0です。
full-H targetは保存JSONから使い、exact ground stateやexact truncated-H信号は新しく求めません。
biasはdiscard＋PF総bias、pure componentsはnull。accuracy不適格も記録して置換せず、
baseline completeness用にcompileする一方、matched-workはnullのままです。

新8構成と保存済みdevelopment comparator5件のpoint比較だけです。
既存B2/B3の再sample・再compile、追加10% GO規則、formal CIや厳密winnerの主張は行いません。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17とplanの既存Python/package identityを維持します。

成功は `PM1_DISCARD_MAP_COMPLETE_AWAITING_REVIEW`、失敗は `IMPLEMENTATION_GATE_FAILED`。
両方mandatory STOP、next-stage=false、research_decision=nullです。
PM-2、ε/P sweep、新geometry/分子、strong synthesis、higher-order PF、energy/RPE接続、
Track B統合、追加trajectoryを認可しません。

## 一回制限と将来の固定command

固定rootは `/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928` だけです。
固定outputは `artifacts/resource_applicability/pr2_pm1_discard_execution/2026-10-04`。
root内exclusive registryは消費済みauthorizationをfailure時にも解放せず、no retry/resumeを維持します。
全checkout横断のglobal lockではありません。別root/output/serverや別authorizationへ切り替えて
同じrunを追加しない運用規則も適用します。今回はoutputとregistryを作成しません。

最終review・一項目のcommitted finalization・利用者の明示launch指示を得た後だけ、
上記rootをcwdとして次を一回使用します。**今は実行しないcommandです。**

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
"/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -u \
  scripts/resource_applicability/run_pr2_pm1_discard.py \
  --project-root "/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928" \
  --plan "/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928/artifacts/resource_applicability/pr2_pm1_discard_preparation/2026-10-04/zero_science_plan_v1.json" \
  --authorization "/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928/artifacts/resource_applicability/pr2_pm1_discard_authorization/2026-10-04/execution_authorization_v1.json"
```

ゲート・source・環境・output不一致やfailure/interruptionでは救済せずSTOPします。
成功時にも研究方針reviewへ戻り、追加計算を自動開始しません。

## 科学データを読まない検査

guard付き限定7-file testsを再実行し、201 passed、fail/skip0、禁止データアクセス試行0でした。
小型synthetic compileをH4科学compileやimmutable CIと混同しません。
draftのfinal booleanがfalseで拒否されることと、科学条件・commit/blob・環境の照合を別々に監査します。
過去attemptのNPZ4件stat/hash違反は元access auditへ保存済みで、今回のアクセス0で消しません。
最終reviewでは全repository testsやscience runnerを実行せず、既存guard付きsuiteだけを許可します。
