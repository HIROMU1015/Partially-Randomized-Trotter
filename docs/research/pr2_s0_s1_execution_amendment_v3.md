# PR-2 S0/S1実行許可 amendment v3

日付: 2026-09-28  
状態: `PR2_S0_S1_AUTHORIZED_STOP_AFTER_S1`  
結果前状態: PR-2/PR-3最小pilot結果は既知、S0/S1結果は未見  
S2/S3: 未承認

## 1. 位置付け

本書は、次の文書を上書きせずに保存したまま、**実行許可とS1 sentinelだけ**を追加訂正する結果前
amendmentである。

- v1 preregistration SHA-256:
  `9cff2a38071c3779648ad6179679b425e7a2d63618ee28de8b3784898b91b97b`
- v1 dry-run manifest SHA-256:
  `07d1afa9fa9d2ee805573a9201ac482330b45c67a45f420d713bc28d6ea261f7`
- 外部レビュー SHA-256:
  `748c3546fbacf06046fe5f0ee0457e96f7dfb9bb252ea4b20aa0afdc9e06942b`
- amendment v2 SHA-256:
  `e93eade73723939a727b87d4d43f680532afbea922cc45db15adf2c61e48914a`
- v2 dry-run manifest SHA-256:
  `adcd579eb58c3a20b32fa94f8e038a1a99bb51213952861fb869a413bd23f775`
- Codex検証方針 SHA-256:
  `8079a7b1263f575d1fdd40f9960e8fcb51342fff8eca654218c082fff5a7c21c`
- 親契約 SHA-256:
  `4d325c0cc28dae08e975ab6311f77258f0cfdefa224e5dfcfb93c1b0a6e8a2b0`

estimand、normalization、shot式、baseline、accuracy、compiler、cost scope、engineering interval、
state-preparation感度、S2/S3 decisionはv2を維持する。本書とv2が衝突する場合は、本書の実行許可と
sentinel規則だけを優先する。

## 2. 結果閲覧履歴

`results_seen_before_amendment=false`という曖昧な表現は使わない。

- `pr2_pr3_minimal_pilot_results_seen_before_v3 = true`
- `s0_results_seen_before_v3 = false`
- `s1_results_seen_before_v3 = false`
- `s2_results_seen_before_v3 = false`
- `s3_results_seen_before_v3 = false`

従って、本書は既知のone-step screening結果を隠して作ったものではない。一方、S0/S1のsnapshot、identity、
corrected signal、wrapper compile結果を見る前に固定する。

## 3. 実行許可の置換

v2の`S0 implementation only`を次へ置換する。

1. result-prior文書とmanifestをcommitして固定する。
2. S0 runner、test、schema、non-overwrite guardを実装する。
3. 実装をcommitし、そのcommitをS0/S1 artifactのsource commitとする。
4. S0だけを実行する。
5. S0が`S0_PASS_S1_AUTHORIZED`の場合だけS1 correctnessを実行する。
6. S1 summary JSON/Markdownを生成して必ず停止する。
7. S2/S3は開始しない。

S0実行では、development 1.00 Åとheld-out 1.30 Åの入力生成・freeze、prefix identity、環境・hash・
corrected-estimator test、q=1/q=8 wrapper testを許可する。held-out 1.30 ÅではS3前にsignal、cost、rankingを
計算しない。

## 4. S0 terminal status

S0は次の一つだけを返す。

- `S0_PASS_S1_AUTHORIZED`
- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_INPUT_REPRODUCTION_MISMATCH`
- `STOP_ENVIRONMENT_MISMATCH_NO_EXECUTION`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

最初のstatus以外ではS1を実行しない。code bugはscientific negativeへ数えない。bugを修正する場合は、
Hamiltonian、state、rank、grid、accuracy、compiler、decision ruleを変えず、修正commitとtestを記録して
S0から再実行する。

## 5. S1 sentinelの置換

rank 3/9 wrapper smoke testのsentinelはCodex検証方針に合わせ、v2の$(r,K)=(4,2)$から

$$
(r,K)=(1,2)
$$

へ置換する。これはfixed gridのlexicographic first cellである。resource rankingには使わない。

identity collapse前のfull-wrapper compile上限88は維持する。

- rank-6 random grid: 72
- rank-3/9 random sentinel: 8
- deterministic wrapper: 8

S1では32/128 trajectory expected-cost MC、resource winner、10% materiality、state-preparation break-evenを
実行しない。

## 6. S1 terminal statusと停止

S1は次の一つだけを返す。

- `S1_CORRECTNESS_PASS_AWAITING_EXTERNAL_REVIEW`
- `BLOCKED_IMPLEMENTATION_INVALID`
- `STOP_ESTIMAND_OR_SCOPE_INVALID`

いずれの場合も`automatic_next_stage = null`とし、summary artifact生成後に停止する。

GPTへ渡すpacketは最低限、amendment、dry-run manifest、S0 artifact、S1 summary、prefix identity record、
corrected-estimator tests、test log、source commit、worktree status、deviation一覧を含む。

## 7. 禁止事項

- S2またはS3の開始
- 32/128 trajectory expected-cost MC
- held-out signal/cost/rankingの開封
- rank、precision、grid、threshold、compilerの追加・変更
- H12、別分子、別geometry、長RPE、final total cost
- S1結果を見た後のnormalization規約変更

## 8. 実行順

```text
amendment v3 + authorization manifest commit
  -> S0 implementation + tests commit
  -> S0 execution
  -> only if S0_PASS_S1_AUTHORIZED: S1 correctness execution
  -> S1 summary JSON/MD
  -> mandatory STOP for external review
```

