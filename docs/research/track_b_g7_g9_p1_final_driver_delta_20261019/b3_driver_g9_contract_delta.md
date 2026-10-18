# 新B3 driverとG9：比較条件の差分

[入口](README.md) / [D01〜D10](d01_d10_final_crosswalk.md)。旧数学・law・native費用の基本表は[前回PCROSS][PCROSS]を継承し、今回変更された接続・scheduleだけを補う。

## 1. 旧準備から新driverへの変化

| 項目 | 旧`c8cbeb18...` | 新`e4af6c06...` | G9との比較上の意味 |
|---|---|---|---|
| 実engine | callback/interface・fake clockまで | lazy G7/F3、`LegacyEndpoint/run_pair/execute`を追加 | 実G7 correctness未実行。G9 native会計runnerと同じ費用対象ではない |
| Layer A/B | 定義・接続計画 | A=旧generator、B=Full F3＋canonical監査を具体接続 | 異なる層の実費を同じCPU順位へ混ぜない。G9はこれらの全N profileではない |
| Estimand | 固定workload／母期待値が未決 | 固定workload記述費用を具体的第一候補とした | 母期待μ、信号CI、一般計算量の新保証はない。なお正式採択は未了 |
| Workload | 保存N一度/選択layer/axis案 | correctness→baseline全N→instrumented全Nの三phase、16 unique cells | 同traceの再生はobserver診断であり独立科学再現ではない。追加callも一束予算へ明記 |
| Setup | arm/layer一回・X/Y共有案 | arm/layer/phase-passで一回、24 constructors。Full prepare3内包、canonical preflight9 | X/Y setup共有を検査。G9構築/参照CPUをこのsetupへ移植不可 |
| Seed | 旧G7診断seed候補 | 新namespace SHA-counter、arm/axis別・A/B/phase/pass共通prefix | 旧128診断の再利用ではない。fixed SHA streamからIIDを証明しない |
| Cache | cold/warm共有未決 | 新law/packet/angle memo無効、保持stateのみX/Y共有 | G9の19 RZ取得済み資産とは別。OS cacheを操作・cold認定しない |
| Angle | τ/Kφ/Jとallocation未決 | A ratio、B acceptedごとmath-only enclosure＋再query。τ/Kφ/Jはまだnull | G9の取得済み理想atan RZを再合成しない。数学enclosureをnative回路へ物理適用しない |
| 計測 | StageLedger等のinterface | nested inclusive childを一度引くMeter、別fine profiler、独立child、memory samples | coarseとfineを足さない。baselineとの差も純overheadの認証ではない |
| 保存 | streamed summaries/prefix案 | all-trial hash、first16/axis完全文書、chain/file SHA、exclusive JSONL/flush | hashから全eventを復元不可。F01を直し、生成traceと保存完了prefixを区別する必要 |
| Admission | 補助SOURCEは常時実実行拒否 | 外部`admit_once`を要求するscience入口、CLI preflightのみ | authority/atomic claim実体は未提供。bool/candidateだけで実行できる設計ではないが、外部bindingは未完了 |

根拠は[DC][DC] `/decisions,/seed_proposal,/schedule_proposal,/call_budget`、[DSCHED][DSCHED]、[DMEASURE][DMEASURE]、[DANGLE][DANGLE]、[DSRC][DSRC]。前回の補助SOURCEとprovider-price schemaは不変で、C1を実装・選定する変更はない。

## 2. 同一scopeと別scope

**一致を維持する部分**：G7のp/x/m/σ・4方式・保存N、旧H160/K256・η/ρ・primitive error・3ρ/16-axis契約、Full raw/coin zero・digital proposal/weight・phase/order、旧F3 guardとevent/packet再query。保存budgetをloadするだけで再評価しない。

**G9との不一致を維持する部分**：G9は8ρ/22 axes、estimator49/1000＋resource1/1000、明示3-system-qubit/I1 exact Clifford+T provider、direct primary6/helper diagnostic5。G7のα_axis=1/320とG9の49/22000、log/ceil/resource policyは別。G7 NとG9総native価格を同じ行へ合成しない。[PCROSS][PCROSS] [G9MC][G9MC]

新B3が保存Nをworkload長に使っても、固定SHA streamでの新しい量子推定confidenceを実証したことにはならない。physical_RZ_certified=false、population_cost_certificate=falseという結果schemaを維持する。[DRESULT][DRESULT]

## 3. 費用座標の接続

- B3はown process CPU/wall、setup/online/save、R/C/A、observer、RSS/HWM、output bytes等の予定観測。device-level I/O量・正確bit operationや連続RSS peakを取得済みとしない。
- coarse stageとfine function partitionは代替の分解で、互いに加算しない。parent dispatch wallはchild wallを含む。最終自己receipt書込費はnullであり無料ではない。
- G9は固定native T/CX/1Q・workspaceとその期待量/上限、prep/readout affine Kの保存値。実state/backend・depth・全N古典費用ではない。
- W2c-Aの費用式へ将来観測を接続する場合、同じlayer/pass・共有setup schedule・保存scopeを使う。固定観測平均を母期待μと呼ばず、CPUとT/CXを未採択重みで加算しない。

G9旧結果の引用だけなら、新driverにRZ cacheやproviderをbindingする必要も、新compileもない。将来C1へ移す場合だけ前回[R01〜R18再利用条件][PREUSE]を適用する。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[DANGLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/angle_cache_output_contract.md
[DC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json
[DMEASURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/driver_and_measurement.md
[DRESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/schemas/result.schema.json
[DSCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/comparison_schedule.json
[DSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py
[G9MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
[PCROSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/b3_c1_g9_identity_crosswalk.md
[PREUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/native_evidence_reuse_matrix.md
