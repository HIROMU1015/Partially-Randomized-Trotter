# SP-1：固定source 0d01ed9 最終review依頼

2026-10-06 JST。**AWAITING_SP1_FINAL_SOURCE_REVIEW / RUN_READY=false。科学実行は未認可。**
review対象のscience source Sは
**`0d01ed9a332ebc5b66ed08acf56214a9b9c0236d`**。
この依頼を収録するreview-only commit Rは実行HEADに使わない。
Sのコード・契約・test記録は変更せず、Rには本依頼、source-bound static planと索引の案内だけを加える。

## 読む資料（固定S）

1. [採用済み結果前契約・意味論・会計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_v1.md)
2. [機械contract](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/contract_v1.json)
3. [source manifest／23 critical file identities](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/source_manifest_v1.json)
4. [全16 pathの静的fusion監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/static_fusion_audit_v1.json)
5. [runner](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/scripts/tracks/algorithm_codesign/run_sp1_wrapper_pilot.py)、
   [実装namespace](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/src/trottertracks/algorithm_codesign/synthesis_placement)
6. [59 focused testsの記録・scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/focused_verification_v1.json)、
   [test sources](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/tests/tracks/algorithm_codesign)
7. [result shape](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/result_schema_v1.json)、
   [pending authorization](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/artifacts/track_b_sp1_wrapper_source/2026-10-06/authorization.json)

S commit直後に作った[source-bound static plan](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/source_bound_plan_v1.json)は
manifest自体とcritical filesがSのgit blobと一致することを照合した。
信号・coefficient/resource採点0、synthesis0、marker0。
12 wrappers/16 paths/48 mask rows/96 axes、必要な保存keysは13（8 generic native angles＋5 catalogue keys）。
元SP-0.5の23 literal rowsを読み取り専用で検証し、その固定catalogue一つから使う。
新しい13 targetを合成したという意味ではない。

## 前回reviewへの対応

受領判定は `APPROVE_SP1_CONTRACT_DIRECTION_WITH_REQUIRED_AMENDMENTS_BEFORE_SOURCE_FREEZE`。
原文SHA256 `60481446a066058e2cc10d01e6663031acdb8f688b36b88f042ed6fb2153d520`。

| 項目 | 今回Sの対応／確認 |
|---|---|
| 必須① common fusion | logical→joint lowering→role/mask非依存exact fusion→placement→PAI。隣接同一Qだけ、signed exact addition、0削除。commuting reorderなし |
| 全登録path監査 | pre/post各960 native位置、candidate0・actual fusion0・cross-role0。D/R lineage・全ordered列を保存。mixed-role placementは未採択でreject |
| 必須② cost名 | **fusion-normalized additive synthesized-primitive T cost**。保存T/T†数の加法和。primitive境界を越えるsimplification・compiled wrapper費用claimなし |
| Aの呼称 | same-angle, alternating-noncommuting-generator accumulation。identical primitive反復とは呼ばない |
| domain | A/B/C、n={8,16,32,64}、NONE/D/R/DR、2 qubit、同じπ/4 catalogue、new synthesis0を保持 |
| Cの解釈 | wrapper単位toy σ coin。finite RTEではない。gate数・angle・sign・randomnessがconfoundし、一般D/R lawへ外挿しない |
| duplicate | A/BのR=NONE、DR=D。独立positive countへ入れず、placement主要証拠はC |
| 数値adapter | 保存g1/g2 midpoint＋g3=1−g1−g2、canonical p=abs(g)/γをexact rational化。全coefficient displacementをuへ戻す |
| finite confidence | ε=.05、α=.05 familywise/96 axes、α_a=1/1920。common Bernstein sufficient shots、oracle varianceなし、shot cap1e9/axis |
| materiality | 同wrapper NONE比5%。ratio interval分類を保持し、ゼロ/不適格referenceをpositiveにしない |
| caps／STOP | wall1200/CPU900/RSS1GiB/output8MiB、one process/retry0。exclusive marker、partial failure保存、全outcome STOP、research GOなし |

SP-0.5 source `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`、実authorization
`9477cd2fcfca69f3f24b801770a1f02805907eac`、result
`e57c1fdd28589422e9c973e34e53f6c725b921e9`、consumed markerを保持した。
旧SP-1提案 `f974f8e7caa5a2769255c0483d7e40caae0a3930`、B-F/BM closuresとAの証拠境界も保持。
共有APIのimplementation変更、A/root編集、分子NPZ操作、DF生成、trajectory、GPUは0。

## 技術検証と限界

59 tests / 0.242 sのlocal pass。tiny semantic fixturesと人工rational controlsの確認で、
CI・外部再現・registered SP-1 science mapではない。
signed controlled lowering、negative time、U/−U phase、basis/inverse、exact catalogue共通適用、
81枝tiny sumとsigned channel逐次合成の一致、outer correlation、係数biasとBernstein/capsを確認した。
launchのpending/direct-A/path制限とinventory/duplicate/STOPも検証した。
test内pending authorizationはtemporary fixtureだけで、将来の実authorization/markerを消費しない。

既存B runtimeはPython 3.10.12、全28 package versionsとpygridsynth/mpmath source treesが一致。
合成器をimport/invokeせずmetadata/source-file identityだけを照合した。
wrapper signalのmpmath 80 dps値は診断専用。primaryのbias/shots/Gはsaved guardを前提にしたFraction会計。
新しいfinite-time certificateや独立のSP-0.5 guard再検証を行ったとはしない。
JSON Schemaはshape文書、runtime checkerはinventory/STOP等のsubset validationで、両者を区別する。

contract SHA256は `f95931dbde2cfbba23c7585e851a26e3f3217d5820343f88b155b9a3a45093d1`。
source manifest SHA256は `2c691732e2b0d802bb1b10137a0c6dca5a09da8ede97234d964e4123f7ffdd3f`。
全S stage critical bytesとreview原文10,407 bytesをcommit前に照合した。

## GPTに依頼する最終review

研究方針を再び全面改訂する依頼ではなく、固定Sでこの限定one-shotを判断できるかの技術review。
特に次を確認してください。

1. 共通fusionとsigned joint channelが意図したνを全maskで保つか。
2. 保存係数区間→exact g/p→coefficient biasの伝播、saved η/δ guard、target/bias/shotの分離が整合するか。
3. additive費用・outer correlation・duplicate controls・NONE ratio/5%・cap-hit非適格の扱いが公平か。
4. resource watchdog、出力8MiBとfailure receipt、runtime snapshotの範囲、one-shot markerの消費順序にblocking issueがないか。
5. clean **Sの直接子A**しかlaunchできず、source/review HEADやSP-0.5承認を流用できないか。

望む返答は `PASS_FOR_SEPARATE_SP1_AUTHORIZATION` または `REVISE_SP1_SOURCE_BEFORE_AUTHORIZATION` と根拠。
これは既にPASSしたという宣言でも、実行許可を発行する文書でもない。
新規性・論文着地点・実DF/RTEへの追加検証の必要性はこのsource reviewだけで採択しない。

## review後の境界

現在のauthorizationはsource_commit=null / science_execution_authorized=false。
science run・marker・wrapper resource/signal gridは0。
**最終review→別明示実行指示→新worktreeでSの直接子authorization-only A→SP-1一回→全outcome STOP**。
将来Aの変更可能pathは
`artifacts/track_b_sp1_wrapper_source/2026-10-06/authorization.json` と任意receiptだけ。
S・contract・tool・domain・threshold/capsを変えず、Rを実行HEADにしない。
SP-1終了後の研究判断はGPTへ戻す。positiveでもactual compile、DF/RTE、追加catalogue/target/gridを自動認可しない。
今回のCodex作業は必要資料のGitHub公開までで終了し、最終source reviewへSTOPする。
