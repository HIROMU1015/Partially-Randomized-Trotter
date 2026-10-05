# 2026-10-06 Track B SP-1：必須修正と新source review準備

GPT review `APPROVE_SP1_CONTRACT_DIRECTION_WITH_REQUIRED_AMENDMENTS_BEFORE_SOURCE_FREEZE` を受領。
全面再設計をせず、採用されたA/B/C、n=8/16/32/64、4 mask、同じπ/4 catalogue、
accuracy/confidence/materiality/capsを維持してsource preparationを進めた。
[原文](../../tracks/algorithm_codesign/inputs/sp1_contract_gpt_review_20261006.txt)だけを明示copyし、
SHA256 `60481446a066058e2cc10d01e6663031acdb8f688b36b88f042ed6fb2153d520`、10,407 bytesを記録。

独立branch/worktreeは `track-b-sp1-wrapper-source-review-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp1-wrapper-source-review-20261006`。
基点は `f974f8e7caa5a2769255c0483d7e40caae0a3930`。旧未commit資料の一括copyはしない。

必須修正は、joint lowering後・mask前のrole非依存canonical fusionと、primary名の
fusion-normalized additive synthesized-primitive T costへの限定。
全16登録pathを静的に監査し、960 native位置、fusion candidates/actual/cross-roleはすべて0。
D/R lineageとpre/post列を保存した。これはscience resource/signal評価ではない。

実adapter/runnerをB namespaceに追加し、保存係数丸めのL1 bias、signed relative control phase、
同一familywise Bernstein、sufficient integer shots、partial failure保存とone-shot/capsを実装した。
59 focused testsはlocal pass。tiny semantic fixturesを使い、registered SP-1 resource/signal gridは未実行。
tool identityは既存B runtimeから照合し、新synthesis・old SP05 J再採点・trajectory・full suiteは0。

[新結果前契約](../../tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)と
[source manifest](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)を公開する。
source S固定後に別review-only commitでSのfull SHAを付したplan/review依頼を追加する。
review HEADは実行HEADではなく、将来のAは新worktreeでSの直接子でなければならない。
authorizationはpending、science_execution_authorized=false、RUN_READY=false。

B-F/BM closures、SP-0.5一回結果/marker、A/共有APIは保持する。
科学実行0、研究方針の採否判断はGPTへ戻し、source最終reviewへSTOP。
