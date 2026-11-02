# B3採択用の新しい契約候補（未採択・未認可）

[JSON](execution_contract_candidate_refined.json)が機械可読正本。[旧候補](../track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json)は変更していない。
`adopted=false`、`contract_grants_execution=false`、`separate_execution_approval=false`。

## 分類

| category | 内容 |
|---|---|
| Fixed science | input、4arm/A-B/X-Y、旧precision/error/confidence、N/call予算、SOURCE10 SHA、旧受入条件 |
| Technically resolved | math-only scope、F3 lemmaから3つの具体profileの十分性、binding hash、既存scheduleとの一致、mock at-most-once/STOP |
| Proposed values | package R1＝A6(τ=10^-6 rad,Kφ=32,J=32)＋既存seed/setup/cache/保存/二pass一式 |
| Adoption pending | ユーザーによるR1採択、またはA9/A12を選んだ別packageの採択 |
| Execution approval pending | 将来の本番port実装とfreeze、実在authorization、fresh scope/claim、そのSOURCE/正式契約への最後の科学実行承認 |
| Unsupported/unknown | 実G7 correctness、実CPU/memory/output、fresh-IID母費用保証、物理native/C2、電源断耐久性 |
| STOP conditions | 未採択／未承認、不一致、spent scope/claim、precision/coverage/保存失敗、partial failure、retry/resume要求 |

基点e4af6c064db805cbec611bcc8f0ff75a4754d38b、旧driver SHAと科学SOURCE10 SHAを維持。
新準備guard/tests SHAはauxiliary_source_pins。本番port SOURCE SHAは未実装のためnull。
公開payloadはpublication receiptで固定し、self-hash循環を作らない。

## 変えない予算

correctness256、baseline15,855,296、instrumented15,855,296、**総31,710,848**。
constructor24（F3内包分を含む）、Full prepare3、canonical preflight9、保存budget load1、budget再計算0。
B accepted毎にangleと原event再query。Full packetはraw-valid sample＋B angleで元通りに課金。
accepted不足／保存失敗等でも補充sampling、N増減、再試行をしない。
新B1/B2、compile/synthesis、physical C1、P1/P2/P3、C2、科学レビューへ進まない。

## 最小限の採択単位

R1を選ぶなら、ユーザーが選ぶ**実行条件の内容は一括1件**にまとめられる。
τ、seed、cache等を六つの未決質問として繰り返す必要はない。
科学実行承認は採択とは別の1件であり、本番portと正式契約の完全SHA/digestが揃ってから行う。
本番admissionの技術準備承認は採択の文に併記できる。これは実authority発行／scienceを今許可することではない。
R1の代わりに別Angle候補を選ぶ場合も、選んだ値を含む一つのpackageをfreezeしてから最後の承認へ進む。

この候補の編集や準備guardのPASSはauthorityではない。実行は常に拒否するreleaseである。
