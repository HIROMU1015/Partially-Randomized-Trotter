# 最初の有限取得案：p7 S2・1hの短q不足4軸

**提案のみ。未採択・未実行。** 全別幅・全roundを取得するcampaignではない。Bがこの候補をQ3可／費用比較の候補に残し、追加価格の情報が探索判断に必要とした場合だけ、別途明示承認を受けて進める。現行16h/7より細かいという理由だけで優位・劣位を認定しない。BがQ3で候補から外した場合や、ほかの幅を優先した場合は投入しない。未検証Sや移送した価格式で候補をpruneしない。

対象は **B0I p7 / S2 / 要求幅1h=.1**、元configuration `e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`。

| m | T | q | actual幅 | physical ID | axes | 新compile上限 |
|---|---:|---:|---:|---|---|---:|
| 1 | .2 | 2 | .1 | `7d1659503f9d99381006` | X/Y各別 | 2 |
| 2 | .4 | 4 | .1 | `7fb12f906ab2f8cd8083` | X/Y各別 | 2 |

機械可読の[4軸](FIRST_FINITE_PLAN.json)を使う。T2 PLAN_METADATAは1hの13roundをすでに固定し、descriptorは `c5466946ab0ef9b50511210ca1e58fb7147bb0f0:docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json#candidates[1].configuration` に保存されている。Bの既存total-view bridgeは固定alignmentの当候補を参照する。今回そのbridgeを変更・再発行していない。

**再利用6軸**：m0 q1は現在の測定wrapper直接receipt、m3 q8は公開PF common原trial、m4 q16はlong-time原trialと公開carrierがある。旧cohort・SOURCE・axisを保ち、この候補に必要なsidecar/C guardのmetadata bindingだけを行う。再compileしない。全13round signal packetはT2固定 `docs/research/truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/B_SIGNAL_INPUT.json`。新signal計算0、signal intake/資格とQ3可否はB側の契約による。

新規取得の実装は既存m0/m1m2 caller・native compose・Accounting/compiler・fingerprint/cache・作用gate・C guardを再利用し、**このconfigurationと4軸だけ**を選ぶ有限selectorを将来最小拡張する。現在のcallerは現行shortlistに限定されているため、そのままIDだけ変更して実行しない。人工metadata/作用testの後、新しい実行SOURCEをcommitで固定し、新しいnamespaceを使う。元SOURCEや古いone-shot枠を転用しない。

保存H4/linear1Å/STO-3G/tol-only DF10、元H/state/snapshot、P0除外、一体項＋fragment0..6保存順、S2 signed順、scalar c_H＋tail identityをtotal Tに一回、symmetric_directional、X/Y測定付きfull controlled Hadamard wrapperを保持する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、元gate許容差のまま。2 cellのevolutionを各軸で共有し、X/Yは個別compile。資源を観測して並列化し、恣意的なworker恒常上限を設けない。時間/RSSはこの別幅条件で未測定なので固定予算を捏造しない。科学call上限は2 evolution cell・4 wrapper compile、新sampling/新mean/fit/Q3/N/G=0。

## §10.5：実行前に確認する差分

- 今回の[PRICE_COVERAGE.json](PRICE_COVERAGE.json)の当候補26行、特にm1/m2の4行を起点にする。現行B2直接receipt・long-time5 carrier・Full catalog・旧PF common・当D branchの公開study/元trialまでを既知scopeとする。現時点では4軸の同条件直接receiptを確認していない。
- 実行までの同候補・旧ID・別scheduleでの追加価格、元trial・軸別receipt、同familyの実行中jobをmetadataで照合する。matchingのT/q/actual幅・fragment/scalar・signed list・wrapper/axis・compiler/lowering・SOURCEを確認し、解決した既存価格を取得数から除く。
- 元trial/出所・同等性が未解決、同軸が実行中なら該当軸を保留する。未実施とみなして再compileしない。他の確定軸を進める場合は、将来callerも軸subsetを正しく処理できることを人工testで確認してから固定する。片軸の価格を補完しない。
- 今回の所在調査は未知の別host/archiveや別名jobを網羅していない。それを不在証明にせず、取得前の差分照合範囲を報告する。新QPY監査や保存値の再計数をこのpreflightへ混ぜない。

## 得られる判定とSTOP

元low/high-control・独立occupation作用・X/Y Hadamard/compiled-vector、scalar/source/input gateとC guardを通過した軸だけ有効directとして保存する。モデル予測や2%基準を持ち込まず、失敗・出所不足は部分結果と理由を保全してUNKNOWN。科学retry、軸・seed・round代替、guard緩和はしない。

4軸成功なら当幅のdirect証拠は6→10/26軸。q=1/2/4/8/16の同幅系列が保存証拠で揃うため、後にCが必要とする別幅の局所/閉包調査を計画する材料になる。**その場でfit・係数採択・QPY監査・m5〜m12価格の算出を行わない。** 残る16軸の価格と新domainの根拠は未取得である。全law問題を持たない決定論回路でも、compilerの幅依存性は消えない。

全対象判定又は重大な取得/実装異常で新投入を停止し、元結果・partialを保存してcommit/push・remote照合後mandatory STOP。全13round価格、Q3/N/G/winnerへ自動進行しない。Bの優先候補が別幅になった場合は、この案を自動置換せず対象・SOURCE・有限scopeをあらためて提案する。
