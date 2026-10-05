# SP-0.5 source review受領・one-shot authorization準備

2026-10-06 JST。利用者が返却したGPT reviewを受領し、**source reviewは通過**。
今回のreviewは実行指示ではない。登録target合成・J採点は未実行、one-shot markerは未作成。

## 固定identity

- source S：`65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`。
- [結果前契約・source review packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/65f6fcdb3dc1ad8bfccfaee6e1413336aef91184/docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md)。
- contract：[contract_v1.json](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/contract_v1.json)。
- contract SHA-256：`ee126a0fcde4aa5eff96dfb5835647deec42150c27703e86a3023164b888d09f`。
- [source SHA／34 focused tests／runtimeの記録](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/preparation_manifest_v1.json)。
- [source-bound authorization準備JSON](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/authorization.json)。

作業branch／worktree：`track-b-sp05-authorization-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp05-authorization-20261006`。
Sの直下childを作り、変更は上記authorization JSONと本receiptの二pathに限定する。
source、contract、tool identity、tests、判定閾値、過去STOP、Aの証拠は変更しない。

## 返却reviewの効力

判定は **`APPROVE_SP05_FOR_SEPARATE_ONE_SHOT_AUTHORIZATION`**。
利用者reviewは、Sの結果前契約・実装・launch gate・34 testsについて、SP05の限定目的を止める
blocking issueは確認範囲で見当たらないと判断した。

reviewにある明示的な境界：

> これは実行authorizationそのものではなく、予定どおり
> source固定 → authorization-only child → 明示実行指示 → SP-0.5一回 → mandatory STOP

この境界に従い、JSONは`SOURCE_REVIEW_PASSED_AWAITING_EXPLICIT_EXECUTION_INSTRUCTION`、
`source_commit=S`、contract hash固定、`science_execution_authorized=false`、
`explicit_execution_instruction=null`とする。正式な実行authorizationと明示指示のreceiptを捏造しない。
承認待ちJSONはrunnerが測定前に拒否する。

## 次に明示指示を受けた場合の手順

現sourceは準備済み、source reviewは通過した。次に必要なのは固定契約での一回の実行指示。
研究方針・catalogue・target・thresholdはその際に変更しない。

1. 利用者の明示実行指示を受領したら、**同じSから別の実行用branch／worktree**を作る。
2. 今回公開した準備JSON・本receiptの二pathだけを参照元commitとSHAを記録して引き継ぐ。
   explicit instructionの原文をreceiptへ記録し、JSONを`APPROVED_FOR_ONE_SP05_RUN`、
   `science_execution_authorized=true`、実際の`explicit_execution_instruction`へ固定する。
3. Sの直下に**実行用authorization-only child A**をcommit/pushする。
   今回の承認待ちchildを親とするgrandchildでは実行しない。既存commitをamend／force-pushしない。
4. clean HEAD=A、parents=[S]、二pathだけのdiff、contract/source/runtime identity、
   fresh marker、資源上限を検証して、固定runnerの`run`を一回だけ呼ぶ。
5. 全outcomeでmandatory STOP。partial failureでもretry0、第二catalogue／target追加なし。
   結果・保存値の限定照合・review資料を選択してcommit/pushし、GPT側の次段判断へ戻す。

この実行用childを別branchで作る手順は、source直下childと、commit内の実際の指示receiptを両立させるため。
今回の承認待ちcommitも固定資料として保存し、上書きしない。

## 固定scope・上限

pygridsynth 2.0.0、exact π/4 catalogue一つ、8 target／plan 23 synthesis keys、
ordinary Rzとcontrolled lowering後のsigned native pairのみ。
operator error `10^-6`、canonical independent PAI、interval Jのstrict存在条件を維持する。
T/T†は各1、common exact fast pathを両armへ適用する。
per-key wall／CPU=30／20 s、total wall／CPU=1200／900 s、combined RSS=1024 MiB、
child address=2048 MiB、result=2 MiB、keys<=32、sequence<=8192 chars、retry0。
分子入力・DF・Hamiltonian・trajectory・wrapper build/compile・GPUはscope外。

## 結果後の解釈（利用者reviewを記録）

primary分類は存在条件のまま。結果後にmateriality閾値を作り変えない。

| 結果 | GPTへ返す意味 |
|---|---|
| NO_PRIMITIVE_TRADEOFF_IN_REGISTERED_SET | 登録set内でtrade-offなし。synthesis-placement主線の再考へ |
| INCONCLUSIVE | failure／不確定の原因をreview。target／catalogueを自動拡張しない |
| PRIMITIVE_TRADEOFF_EXISTS | cheap notchと二次モーメントの交換の実装確認。wrapper pilotの必要性をreview |

positiveでも新規性、DF優位、D/R placement優位、16-cell pilot GOを意味しない。
全16 primitive rowsのordinary／controlled別、Jの1からの距離、π-rational／±1/5 rad、
controlled pairのmoment penalty、低deterministic T-count controlを**記述的に確認**する。
primary分類の変更、新しいscientific candidatesの取得、threshold変更は行わない。

現在は必要資料をcommit/pushしてSTOP。次stageへの自動認可なし。
