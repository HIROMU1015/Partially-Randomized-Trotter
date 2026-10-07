# RA-D0 v3 final source review：certified-infeasible minimumを正常処理

2026-10-07 JST。固定v2 `4012cbd167ec10f143fbfddc89feff4dec54bf2b`から独立した
branch `track-b-ra-d0-source-review-v3-20261007`を作り、[利用者指示のbyte-exact原文](inputs/ra_d0_source_review_v3_user_instruction_20261007.md)
に従ってB2 minimumのinfeasibility処理を修正した。

**`READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION`。** ローカルsource確認でblocking issueは残らなかった。
これはGPT最終reviewへ渡す準備statusで、実行authorizationではない。
authorization、one-shot marker、registered optimization／minimum／budget freeze／B3／witnessは未作成・未取得。
この公開後はmandatory STOPする。

## 修正した問題

v2のbudget stageは、certified feasible implementationを返さない全minimum solveをtechnical failureにしていた。
これにはexact Farkas proofを取得済みのinfeasible LPも含まれた。
v3は[engine](../../../src/trottertracks/algorithm_codesign/ra_d0/engine.py)で三つのoutcomeを区別する。

| Minimum outcome | 条件 | 処理 |
|---|---|---|
| `B2_MINIMUM_CERTIFIED_FEASIBLE` | nominal feasible、固定q/y丸め、membership・mean・confidence・workspace・resourceが認証済み | 完全resource vectorをbudget候補にできる |
| `B2_CERTIFIED_INFEASIBLE_AT_N` | main infeasible、最大一Farkas auxiliary、exact rational proofがPASS | 正常な数学的outcome。pointを記録して次のnへ進む |
| Case C / technical | proof取得・検証失敗、nominal feasibleの実装certificate失敗、unexpected status、cap/source失敗 | 従来どおりtechnical STOP。救済・retryなし |

backendのstatus stringやBooleanだけではCase Bにしない。
返却された`full_auxiliary_primal`を、**当該minimum LP**へ戻してexact Farkas verifierで再確認する。
ray欠落・寸法不正・proof不成立はCase C。
minimum inner LPは共通numerical certificateを表す連続集合であり、dyadic B2_numを含む。
そのinfeasibilityのexact proofはB2_numに対しても有効だが、そのn・固定tableの範囲に限定する。
registered tableでそのoutcomeを取得したという主張はしていない。

技術失敗がない限りT/CX/1Qの各solveを一回ずつ行い、各statusとfull proof/vectorのhashを保存する。
先のobjectiveにCase Bがあっても、後続objectiveのCase Cを無視せずtechnical STOPする。
solver settings、denominator、tolerance、second-best探索、retryの変更はない。

## Point / freeze / coverage

三minimumのうち一つでもCase Bならpointは`B2_POINT_CERTIFIED_INFEASIBLE`となる。
**`budget_vectors=[]`、`queries=[]`。** same-n B0_saved profileが存在しても、そのvectorだけでprimary queryを作らない。
B0_savedはreproduction anchorであり、B2_numのsubsetとは仮定しない。
新しいdescriptive B3 feasibility solveも加えない。

[freeze schema](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/budget_freeze_schema_v3.json)に、
point_status、各minimum status、certified_infeasible_objectives、same-n B0 profile IDs、
budget/query arraysとpoint countsを固定した。skipped pointもbatch freezeへ含めてSHA固定する。
Phase Bの[reader](../../../src/trottertracks/algorithm_codesign/ra_d0/freeze.py)はstatus整合・query derivation・countsを検査し、
`BUDGET_READY`だけを比較する。skipped pointにbudget/queryを入れたartifactは拒否する。
technical failureは完成freezeへ混ぜず、既存failure/certificate/terminal result記録へ残して停止する。

P1全18 anchorsのbatch freeze-before-B3と、固定anchor gate後のP2全必要coverageの別freezeを維持した。
infeasible anchorはanchor witnessなしとして扱う。そのxに他のanchor witnessがなければ既存規則でcoverageへ進む。
coverageでは証明済みinfeasible点を正常にskipし、次nを処理する。
同batchの後続`BUDGET_READY`点のbudgetも、batch内B3を呼ぶ前にfreezeする。
全三minimumを取得できない**technical** failureは引き続きbatchを停止し、prefix分類を出さない。

## Classificationと出力

primaryはcertified `U_B3<L_B2_outer`だけである。分類定義を変更していない。

- STRONG：両xの比較可能PRIMARY_ANCHORにstrict witness。infeasible anchorは数えない。
- LOCAL：strict witnessはあるが一方のxだけ、またはcoverage-only。coverageでSTRONGへ昇格しない。
- NO_REGISTERED_WITNESS：完成したcertified-comparable queryにstrict witnessなし。
  skipped pointはnegative evidenceにしない。比較可能queryが0ならその数を明示し、一般不存在とは解釈しない。
- TECHNICAL_INCONCLUSIVE：Case C、incomplete、cap等。証明済みminimum infeasibilityだけではこの分類にしない。

既存paired queryの`B3_ONLY_FEASIBLE_DESCRIPTIVE`もstrict/STRONG/LOCALへ数えない。
[output schema](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/output_schema_v3.json)へ
`certified_comparison`と`point_outcomes.jsonl.gz`を追加した。
skipped pointのx/n/tag、minimum statuses、proof hashes、B0 IDs、B3未実行、primary不適格を保存する。
terminal resultにcertified-comparable query数、skipped point数、completed ready point数、NOの限定scopeを保存する。
未完了のprefixからpositive/negativeを出さない。

## 維持した契約と証拠

[execution contract](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/execution_contract_v3.json)と
[bounded recipe](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/bounded_query_recipe_v3.json)を更新した。
新queryは0。75 main/点、55,275 main、最大一auxiliary/main込み110,550、hard guard111,000を維持する。
skipは後続paired callsを減らすので旧上限は保守的なままである。

process/thread=1、retry=0、per-LP wall=2 s、wall=3600 s、CPU=3300 s、RSS=1536 MiB、
AS=4096 MiB、output=128 MiB、GPU=0を変更していない。
per-LPはHiGHS内蔵time limit、periodic Python signal、post-checkの組合せであり、
長いC呼出しを2 sちょうどで強制killする保証ではないというv2注記も保持した。

B0_saved/ideal分離、B1_num⊂B2_num⊂B3_num、2^60 sampler、membership allowance、mean/confidence、
B2 outer/B3 upper、完全vector pairing、grid、x、precision、resource座標を変更していない。
[policy identities](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/v2_policy_identity_v3.json)は
LP/numerical/exact/backend/guard/table/grid/semanticsのbyte-exact維持を記録する。
launch/runnerはv3 manifest/contract/将来result pathへ参照を更新しただけで、authorization方式は同じである。
現sourceはauthorizationがないためregistered solveへ進めない。

固定R1 `24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`、R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942`、
数学監査`8a04c148a66d23dbc1f045086a95a5e19a6372dc`、v1 `0ddf67756516e08f85fed1b987459a5e862676b7`を継承した。
対象は保存済み2-qubit distinct-basis controlled、finite P3、p=(3/4,1/4)、x={1/8,1/4}と既存三precisionのみ。
分子、DF rank/split、geometry、held-outは対象外。R1/R1.5/旧STOP/Track A/共通APIは不変である。

## 検証・停止

旧80 testsをbyte-exactで保持し、20 regression testsを追加した。
[最終focused verification](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/focused_verification_v3.json)は
**100件PASS**。実solverは人工/off-domain LPだけ。registered solver/minimum/budget/B3/witnessは0。
新synthesis、angle、precision、IS、CTS、DF、molecule/NPZ、science/circuit/matrix/trajectory/GPUも0。
full test suiteは実行していない。これはローカル技術検証で、CI・外部再現・科学結果ではない。

各x21 columns、18 sign controls、保存126 sequence identities、旧R1 result/marker/source/authorization、
v1/v2 artifactsのhashを[provenance audit](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/provenance_audit_v3.json)へ保存する。
native error/signalの再計算や旧分類の変更はない。8 indexはv3案内のprependだけで基点の本文を保持する。

blocking issueは解消したが、registeredの実装certificate取得やcap内完了を保証しない。
τの幅とdouble候補の誤差でCase Cになる可能性は残る。そこを結果後に緩めない。
GPTがこの固定commitを最終確認し、通過後にだけS→authorization-only直接子A→明示one-shot→mandatory STOPへ進むか判断する。
**今回authorizationを作らずmandatory STOP。研究判断はGPT側へ戻す。**
