# Bへの受渡し：H4 B0I・Full-Hの26 Raw means

数値SOURCE `5a172629db4d32a4230f8bb10a871b42807b1cd5`、scope `6306fb3a74e7864a44a3e6cd022e9a4b062e7d44`。新26評価を一回実行し、全て元数値gate PASS。N/risk/RMSE/compiled C/Gは未評価。結果の詳細は [REPORT](REPORT.md)。

## 読取対象

|系列|configuration ID|B signal packet|全round configuration/physical request|
|---|---|---|---|
|B0I p7/S2/Δ.1|`e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`|[B_SIGNAL_INPUT.json](run01/B0I_P7_S2_D01/B_SIGNAL_INPUT.json)|[PLAN.json](run01/B0I_P7_S2_D01/PLAN.json)|
|Full-H K2/q=2^m/R448|`cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c`|[B_SIGNAL_INPUT.json](run01/FULL_H_K2_QDYADIC_R448/B_SIGNAL_INPUT.json)|[PLAN.json](run01/FULL_H_K2_QDYADIC_R448/PLAN.json)|

packetは既存 `rpe_optimization_pipeline.signal_input(produce_signals(...))` のcopyであり、top-levelは `configuration_id/target/source/means`。PLANの `configuration` は原c546のcandidate1/2と同一。異なるIDへ付け替えない。branch全体のmergeやscience callerの再実行は不要。公開結果の固定commitから必要JSONをread-onlyで受け取る。

各meanはm、identity、law_binding、physical_candidate、raw/corrected/reference、normalization、total_signed_bias、数値診断とsource、provider_bindingを持つ。provider_bindingにはphysical_context・numerical_source・input_source・reference_sourceがある。full H/state/snapshot、canonical T、law、scalar、profileを照合して使用する。binding.T=.8は元input metadataを保持したfieldであり、評価時刻は各round/physical_candidate.T。B0IのqPFは2^m、Full-HはB側qPF=nullで、物理q=2^mがgrouping、R448である。

## B側で保つ条件

- `status=SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED`、`qualification=UNQUALIFIED`。numerical gate PASSをenergy-risk/N資格へ昇格しない。
- Raw座標を既存B signal/measurement規約のまま使い、correctedで置換しない。CSVのreal/imagは元packetのcopy。X/Y wrapper符号の新しい規約は導入していない。
- B0I targetは元full DF-H。tail identityのscalar補正済み。もう一度補正しない。Raw=corrected、B1。
- Full-Hはone-bodyもrandomized、deterministic head0、保存抽出identityと定数scalar一回。finite-K2のideal law。raw/correctedは独立伝播であり、Bをshot/価格へ重複計上しない。
- unknownのstate error、sampler law distance、empirical allowanceはnull。数値診断はrigorous error boundではない。とくにFull-H後半の非常に小さいRawを、数値gate PASSだけで高精度保証付きと扱わない。
- 元target/error contract/decoderは変更していない。baselineに対するrisk producer/allowance/target proxyの採用は別に必要。PR-LD1/LD2用のNや資格を移植しない。
- price profileは元configuration metadataで、今回compiled価格は取得していない。Dの価格law/lowering/scalar/平均資格との照合は別工程。正式G/winnerはまだ出せない。

Bの既知公開基盤 `6106e4d969bb08e95166cbd2b2cc55b9840c7535` のLD1/LD2結果や規約をbaselineへ自動採用していない。Bが後続SOURCEを用いる場合は、packetを受けるbaseline producer/input bindingの差分を確認する。本作業ではBのrisk計算・adapter改造はしていない。

## 正本・保全・次作業

各系列の `round_m.json` と `RESULT.json` が数値/診断正本、[SIGNALS.csv](run01/SIGNALS.csv) は全26件の転記。[COVERAGE.json](run01/COVERAGE.json) と [CHECKS.json](run01/CHECKS.json) に26件取得・source/identity/copy照合を保存。実行環境は [SOURCE.json](run01/SOURCE.json)、入力locator/hashは [INPUT_PROVENANCE.json](run01/INPUT_PROVENANCE.json)。

新reference action/scalar抽出/sampling/compile/Full PF/MC/N/価格/Gは0。旧39round計画、Full PF m6の旧受入案、旧STOP・raw・SOURCEは保全している。本scopeを再実行しない。

mandatory STOP。次の担当はユーザー（Bのbaseline risk有限実行scopeの判断）、その許可後にCodex B（2 packetの受入・所定risk評価）。AのFull PF/H_effとDの価格研究は独立のまま。Cによる追加科学計算は開始しない。
