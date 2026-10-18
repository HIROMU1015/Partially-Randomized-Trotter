# 実行準備状態と採択・承認の境界

[入口](README.md) / [技術指摘](technical_findings.json)。**静的P1は完了、B3は即実行可能とは判定しない。** 実入力correctnessは`NOT_ESTABLISHED`。新driverの既存人工84 PASSは実G7の数値・runtime保証ではない。

## 1. 実行前に必要な技術作業

1. **F01失敗文脈の修正**：前trialのresultを現trialの`inflight_event`へ混入させない。axis/trial_id、event未取得、保存完了prefixと生成traceを区別し、成功→次sample失敗・axis移行・angle/save失敗の人工回帰を新versionで固定する。
2. **SOURCE再固定**：F01修正後のdriver SHAとmanifestを新commitへ固定。今回監査した`e4af6c06...`や旧SOURCEを上書きせず、対応する新候補・テスト記録を公開する。今回この作業は未実施・未承認。
3. **外部admission/launcher binding**：atomic排他・fresh claimの永続消費、ユーザー承認、正式契約digest、完全SOURCE release/pins、one-shot scopeを結ぶ実体が必要。`Boundary`のreceipt検査はその実体ではない。admission後・出力作成前の失敗も、消費状態とともに保全する。

F01は科学条件の変更を要するという指摘ではなく、失敗metadataの正確性を保つ限定技術課題である。今回SOURCE変更禁止のため修正せず、ユーザーへ返す。外部claim/authorizationは今回作成しない。

## 2. ユーザーによる正式採択が必要な候補

| 条件 | 現在の候補／未決 |
|---|---|
| Cost estimand | 当該固定workload・環境の記述的実費。母期待μのcertificateではない |
| Angle | `MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE`、具体τ/Kφ/Jは未定。A ratio／B accepted毎enclosure・再query。物理RZ代表値適用は対象外 |
| Seed | 新namespace `B3_FULL_N_NEW_20261019_V1`、旧G7 SHA-counter、arm/axis別・A/B/phase/pass共通prefix。IID主張なし |
| Schedule | 16 unique cells、correctness16/cell→baseline全N→instrumented全N、arm/layer/phase-pass独立child、X/Y setup共有とidentity検査 |
| Call budget | correctness256、各全N pass15,855,296、合計31,710,848。constructor24、Full F3 prepare3は内包、canonical preflight9。accepted依存angle/requeryは別counter |
| Cache | 新law/packet/angle memo無効。保持stateのみX/Y共有。OS cacheは制御済みcoldとしない |
| 保持・出力 | all-trial compact hash、first16/axis・correctnessの完全文書、failure prefix、stream/chain/file SHA、flush4096、fsync保証なし、stream64GiB/log256MiB候補 |
| SOURCE/契約 | F01修正後の完全release commit、pins、正式契約digestと採択記録 |

上表は[DC][DC]・[DSCHED][DSCHED]・[DANGLE][DANGLE]の**候補の引用**。今回採択した値ではない。新CPU/RAM/worker/wall上限や新費用confidence/重みを設定していない。τ/Kφ/Jは固定H/Kや既存εから一意には定まらず、Codexが便宜的に補わない。

## 3. 別途の科学実行承認

正式条件・修正版SOURCE・admissionの固定後、ユーザーが初回correctnessを含む**一束one-shot**の実G7 sampling/profilingを明示承認する必要がある。候補表のcall数を確認しただけでは承認にならない。

実行する場合はcorrectnessで各cellのaccepted path、event/phase/πW、B preflight/source/state/angle等を確認し、通過後のみ全N二passへ進む。precision不足、accepted未出現、identity不一致、I/O等の技術失敗はprefixを保全してSTOP。不足pathを埋める追加trial、retry/resumeは認めない。新しい再試行には別途指示・記録が必要であり、旧claimは再利用しない。[DCMD][DCMD]

今回これらの科学gateを実行していない。物理C1 providerを選定しなければB3が実行できないという追加条件も課していない。

## 4. 実行後にも残る限界

得られるのは4方式の当該固定workload・保存/共有scopeでの観測費用である。母期待μ、一般計算量、正確bit operation、独立再現、Closed P5/CTSの古典費用、未取得depth、物理native/C2総費用、研究GOは未確定のまま。数値縮約・結果解釈も実行承認scopeを守る。

現在の次の担当は**ユーザー**：F01の限定修正・再固定をCodexへ依頼するか判断する。依頼後の**Codex**はその技術範囲だけを実施する。正式契約の採択、新admission/科学実行、P2/P3/C1/C2へは自動移行しない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[DANGLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/angle_cache_output_contract.md
[DC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json
[DCMD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.md
[DSCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/comparison_schedule.json
