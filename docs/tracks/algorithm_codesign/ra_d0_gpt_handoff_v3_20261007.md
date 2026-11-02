# GPT final source review：RA-D0 v3

**`READY_FOR_RA_D0_ONE_SHOT_AUTHORIZATION`。実行authorizationは未作成。**
branch `track-b-ra-d0-source-review-v3-20261007`の公開されたfull commitをSとして、
[v3 source report](ra_d0_source_review_v3_20261007.md)、[利用者指示](inputs/ra_d0_source_review_v3_user_instruction_20261007.md)、
[contract](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/execution_contract_v3.json)、
[manifest](../../../artifacts/track_b_ra_d0_source_review_v3/2026-10-07/evidence_manifest_v3.json)を確認してください。
基点はv2 `4012cbd167ec10f143fbfddc89feff4dec54bf2b`。

修正はB2 minimum infeasibilityのblocking issueだけです。

1. exact Farkas proofを当該minimum LPへ再検証できた場合は`B2_CERTIFIED_INFEASIBLE_AT_N`という正常outcome。
2. 一objectiveでも該当すればpointは`B2_POINT_CERTIFIED_INFEASIBLE`。
   B0_savedがあってもbudget/queryは空で、新descriptive B3は加えず次nへ進みます。
3. skipped点もfreezeへ含めhash固定します。Phase Bは`BUDGET_READY`だけ。
4. proof取得・検証失敗や実装certificate失敗は従来どおりtechnical STOP。Case B後のCase Cも無視しません。
5. infeasible anchorはwitnessなしとして既存coverage gateへ渡します。skipはSTRONG/LOCAL/negative evidenceに数えません。
   NO_REGISTERED_WITNESSは比較可能queryだけに限定し、比較可能件数0も明示します。

旧80＋追加20＝100 focused tests PASS。新検証は人工/off-domainのみで、登録最適化0。
21 columns/x、18 sign controls、旧R1保護hashは不変。
core numerical/solver/guard/table/gridはbyte-exact。main55,275 / recipe total110,550 / hard111,000と全資源capも不変。
batch freeze-before-B3、anchor-first、certified U_B3<L_B2_outerというprimary、full-C-call preemption限界注記を維持しました。

新synthesis/science/NPZ/GPU等0、authorization/marker/registered結果なし。
対象は保存2-qubit distinct-basis controlled finite P3、p=(3/4,1/4)、x={1/8,1/4}と既存三precisionだけです。
registered certificate取得・資源cap内完了・研究新規性を保証しません。

問題がなければ次は**Sのauthorization-only直接子A→別の明示実行指示→development one-shot一回→mandatory STOP**です。
この資料公開自体では実行しません。CodexはSTOPし、最終判断をGPT側へ戻します。
