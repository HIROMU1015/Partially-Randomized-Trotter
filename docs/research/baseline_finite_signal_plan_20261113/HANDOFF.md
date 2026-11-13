# A/B/Dへのbaseline finite manifest

正本は [BASELINE_FINITE_MANIFEST.json](BASELINE_FINITE_MANIFEST.json)。各candidateの `configuration` はB V2の原形式、`rounds[m]` はphysical recipe・SOURCE・処理区分・reference/price coverageを持つ。`canonical_row_pointer` は同candidate内configurationのidentity/law bindingを参照し、別規則でIDを作らない。

|系列|configuration ID|
|---|---|
|Full PF S2/.1|`f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef`|
|B0I p7 S2/.1|`e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`|
|Full-H K2 qdyadic/R448|`cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`|

## AのPRとの共通条件

A `91349e703` LD1/LD2の全26 signalは再取得しない。元full DF-H/state/snapshot/sector、h/M、prior/decoder/branch、energy budget、P0を共通化した。baseのh/Δはbinary64 exact値を保持。保存E0/proxyで新しいpriorを作っていない。Full-H preparationのcanonical provenance bridgeはmanifestのsharedへ記録し、元bindingと物理値を残した。

PF公式/幅・prefix・K/q/Rを方式間で将来も同じに固定することは公平比較の要件ではない。本3seed/PR2構成だけでmethod内最小Gやglobal最適を認定しない。

## B：signalとrisk/N

承認後の出力は全13roundのraw/corrected/reference、finite logB/B/attenuation、signed bias・局所診断、identity/law、数値uncertainty・SOURCE。原full-H energy targetを保持し、B0I retained-Hのgroundへ替えない。rawだけがX/Y観測meanで、normalizationをN/価格へ二重適用しない。

m6は受入成功時も**旧spectral producer**。他12 Full PF/13 B0Iはdirect-expm、Full-H13は元grouped finite-K。SOURCE/backend別に表示し、新規取得39件や独立検証1件と偽って数えない。全meanのB資格はUNQUALIFIEDで、state/law/numerical boundの不明量はnull。局所bias/attenuationからNを計算しない。

共通RMSE policy/error contractは元A/BのPROPOSED contextをcopyした。baseline固有のcutoff/finite-K等への移送、全loss/claim/solver proxy採択はBの判断待ち。B `23d748551` の固定proxy寄与・nominal値も新Nとして採用しない。whole-shot adapter `b9b9ffa62` の既存入力を使い、CがN資格producerを実装しない。

## D：compiled価格

manifestのphysical candidate、T、q_RPEとq_PF/Full-H grouping q、R/K、signed structure、one-body/head/tail/scalar、binding/source/profileからrequestを作る。Full-H物理rowのlegacy methodがPRでもcomparison_methodはFULL_H_RTE、head0でありPR prefix0/B3ではない。

Full PFのm3/m4 direct receiptは `e7dca5f73` の確定metadataを再利用し、再監査/再compileしない。元receipt trace・SOURCEとconsumer identity/profileが必要。B0I m3は保存price SOURCEあり、truncated domainのconsumer receiptは未資格。Full-H R448は旧light/cohortとは異なる。具体的coverageは [COVERAGE_AND_REUSE](COVERAGE_AND_REUSE.md)。

D `748be6875` のLD1/LD2取得requestはPR向けで、baselineの価格取得完成を示さない。現 `candidate_price_adapter` をFull PF/B0I/Full-Hへ強制適用せず、対象外/未取得を保持する。Dが必要なnative/mean domainを整備し、compiler/lowering/control/full measured wrapper・axis/law/cohort/不確かさを確認する。価格モデル・平均価格の実装/取得はCで追加しない。

本scopeはsignalだけ。新C、law期待値、価格CI、N×C/Gは未実行。実費用比較には別途ユーザー承認と同taskのN/価格資格が必要である。

## 取込み・STOP

plan文書とmanifestを参照し、既存C `configure/prepare/execute`、共有input loaderとreference copy callbackを使う。新numerical moduleや既存科学SOURCEの修正は不要。受入再資格1件と新38件は [FINITE_SCIENCE_PLAN](FINITE_SCIENCE_PLAN.md) の一括scopeとして承認対象にする。旧20セル/旧run/KEEP/grant/ledger/marker/STOPを保全。

今回のCはmetadata計画の公開照合でmandatory STOP。次は**ユーザー**が明示承認、**C**が許可された有限baseline signal、**B**が別途risk/N、**D**が別途価格を担当する。
