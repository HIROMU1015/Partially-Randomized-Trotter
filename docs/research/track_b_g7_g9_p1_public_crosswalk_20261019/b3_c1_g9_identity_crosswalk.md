# 公開B3/C1準備とG9 v2のidentity対応

[入口](README.md)。前回の[基本比較表][ACOMP]は固定したまま、新たに[BC][BC]・[B3][B3]・[C1][C1]・[BS][BS]・3 schemasを照合した。以下は新実験条件の採択ではない。

## 1. 数学・sampling・精度：継承先を確定

| 公開B3/C1 field/API | 対応する固定根拠 | 今回確定できること／残る境界 |
|---|---|---|
| BC `/input`、`/arms` | G7 `P5_general_order`、G9MC `/input` | p/x/mと4方式の理想finite P5 first momentは対応。G9のClosed P5/CTS追加をBCが許可するわけではない |
| BC `/original_source_commits`、`/fixed_science_policy` | G7 generator、kernel、F3、前回ACOMP | science SOURCEを変更しない設計。G9の4方式もG7 generatorを使う。Fullのraw/coin zeroは全trial分母、canonicalに擬似rejectを追加しない |
| BC `/reuse`、BREUSE `/rows/*/raw_JSON_pointer` | G7RAW `/rows/4`〜`/rows/7`の保存field | N、digital acceptance、既存係数の転記が原fieldと一致することを管理検査。数値再算定ではない。旧referenceはproductionへ注入しない |
| BC `/fixed_science_policy/normalizer_rule` | G7 `budget`、G9CMP `plan` | Fullはordinary envelope×U、canonicalは各Bの上界、proposalでweight補正。truth/列挙B_newでNを下げない。Closed P5/CTSは各固有normalizerでありFullのBを流用しない |
| BC `alpha_axis=1/320`、16 axes、3ρ bias | G9MCは`alpha_axis=49/22000`、22 axes、8ρ bias＋resource配分 | **総費用の直接混合はNOT_COMPARABLE**。G7の保存NをG9のnative総量へ置換・比例換算しない。G9内部の既存表を引用するだけなら新policy不要 |
| BC H160/K256、η/ρ、ε_axis=1/200、primitive_error=10^-6 | G9MCの同名field | 共通数値条件がある。ただしCTSのQ(√2)係数処理、log/ceil、受理tailが異なる。これだけで同Nとはしない |
| BS `root_witness`/`law_witness`、F3 angle | G7 root/law postcondition、G9 cacheのstrict RZ | 前者はroot/lawの既存条件の検査。angle τ/K_phi/J・G7 error allocationはBCでnull。保存RZと新enclosureの結合は将来driver確認事項 |
| BS `event_witness`、G7 `controlled_event_ir` | G9N `native_ir`/`adjoint` | raw degreeのphase、逆時間word、actual adjoint、Wを保持する条件は対応。CTSはPauli/rotation_sign eventで同じchild/word witnessではない |

G7のbiasは`2*(3*rho+3*(1+rho)*2*primitive_error)`、G9は`2*(8*rho+8*(1+rho)*2*epsilon_Rz)`。G9ではestimation failure49/1000、resource failure1/1000、total1/20。G7の16-axis confidenceをCPU測定CIや6方式へ自動転用しない。[BC][BC] [G9MC][G9MC] [ACOMP][ACOMP]

## 2. B3 Layer A/B・費用scope

| 公開済み定義 | G9側の対応 | 判定 |
|---|---|---|
| Layer A=`A_frozen_G7`：旧4方式generator | 4方式は同SOURCE。Closed P5は`P5Closed`、CTSは`cts_events` | SOURCEの機能は対応付け可能。G9 runnerは全N production費用の取得ではない |
| Layer B=`B_existing_F3_Full_plus_new_canonical_audit` | BS `ARMS`と`canonical_snapshot`は旧canonical3方式だけ | Closed P5/CTSの同等audit入口はない。ARMSだけ増やせば済むとしない |
| SOURCE/hash、state seal、πW/phase、preflight | `AuditParity`、`event_witness`。G9は保存native/reference監査 | 監査目的に共通点はあるが、SOURCE依存のgroup式・Pauli収集・回路意味の証拠は方式ごとに違う。saved PASSをproduction per-trial監査の実測にしない |
| setup/budget load/preflight、online、outputを分離 | G9MAN `/resource`はguard/参照/native会計を含む | G9の約3.094秒をS/μ/Ωのどれかへ無条件移植不可 |
| 一構築/arm/layer、X/Y共有はstate確認後、cold/warm pool候補 | G9は19固定key共有、取得18再利用/1新規 | key資産は再利用候補。arm単独cold費用、共有pool、load/lookup/hash/outputの課金は新scheduleで固定する必要 |
| StageLedger exclusive stage、CPU ns・wall ns | G9 resultには別scopeのCPU/wall | 新CPU観測を旧native集計へ足して単一総費用を作らない。instrumentationとtotalの二重加算も禁止 |
| RSS/allocator/buffer、bit symbolic、output | G9 peak RSSは参照会計時の記録 | 全N peak・正確bit operation・保存費用の代用ではない。peakをN倍しない |
| saved N一度の全workload、固定SHA seed候補 | G9のNは別policy、CTSに旧128 traceなし | 全Nへの延長は新sampling/profiling。今回のP1は承認しない。観測費用と母期待μの保証は別 |

B3の選択未決は`/decisions_required/B3_*`と`/B3_proposal/cost_confidence_policy`、`angle_policy`に明示されている。公開補助SOURCEはcallback/fake clockを受けるinterfaceであり、実driver/schedule/永続failure保存の完成を意味しない。[BC][BC] [B3][B3] [STAGE][STAGE]

## 3. C1 provider/schemaとG9証拠

| 照合項目 | 公開C1 | 既存G9 | 静的な結論 |
|---|---|---|---|
| Q/encoding/basis/sector | BC `/C1_proposal/physical_Q=null`、選択7項目未決 | Q0=Z0、Qi=Vi†ZiVi、system3の明示synthetic model。分子fermion sectorの保証ではない | 未選定provider全般には価格を移植不可。G9 contextをユーザーが選ぶ場合だけ条件付き再利用候補 |
| control/direct/helper | 旧G7 helper、price keyはouter/helper役割 | direct primary6、helper diagnostic5。CTSはPauli direct | 旧G7 helperをG9 directとして扱えない。強い対照にも同じdirect/基底共有機会を許す |
| error | CQ allowance0、strict RZ別 | exact Clifford+T CQ delta0、event bound2ε_RZ | CQ価格の0とrotation/event価格の2εを同じfieldへ混ぜない |
| key | operator_id/label/control_role/direction/compiler_id/gate_set_id/precision_id/phase_policy/optimization_id | ratio-key cache、word/child/ratio/phase・implementation別event/native_ir/cost | 同じ文字列keyではない。opaque IDと具体Q/encoding/event/precisionのregistry bindingが追加で必要 |
| compiler | 未採択、literal primitive poolが第一候補 | CZ→H CX H、adjacent inverse cancellation、同条件全方式、no global optimizer | G9のwhole-event cancellation後の費用はprimitive加算と同一とは限らない |
| RZ | `rz_cache_witness`は保存metadata/checksum | 19 positive ratio keys、18はG7保存列、1はCTS用取得済み | row shapeとfull angle-key/εは対応。新schemaへkey同一性を明示する必要。ratio文字列だけでは不足 |
| price coordinates | T/CX/one_Q/serial_depth/ancilla_peak：整数区間又はnull | event `cost.T/CX/1Q`は整数、期待費用は別rational field、workspaceはrow | `1Q→one_Q`等の結合が必要。期待fractionを整数gate価格へcastしない。depth欠測をnullで保つ |
| phase/actual adjoint | `phase_retained`、forward artifact SHA、verified flag | exact adjoint/逆列・W scalar、診断と既存analytic budget | schema/flagsだけで回路正しさの再証明にはならない。保存SOURCE/契約/証拠hashを結ぶ |
| prep/readout/backend | null、C2除外 | T_prep/readout affine Kとouter1Qはevent費用と別 | stateの実回路/費用は得られていない。共通parameterの既存結論だけ引用可能 |

`price_key`は**9 fieldの厳密一致**を要求する一方、Qの行列・encoding・sector・tool version/seed等を独立fieldとして検証しない。`operator_id`/`compiler_id`/`precision_id`などと外部固定manifestの対応が必要で、同名IDを与えただけではidentity照合が閉じない。JSON schema単独よりruntime guardが強いが、guardも提出証拠のmetadata検査である。[BS][BS] [PRICE][PRICE]

また`event_witness`はword/child/positive ratioを前提にhelperの`provider_calls`を検査する。G9 directの価格へのmappingを検証する機能ではなく、CTSのreal correction/identity eventへもそのまま適用しない。**今回これらを実装変更しない。**

## 4. 完了したこと／まだ完了していないこと

公開固定資料の上記対応、BREUSE原field同一性、schema/API形、RZ資産のmetadata、既存結果の引用scopeは今回で固定した。G7/G9基本表や既存native取得を再調査する必要はない。

未完了は将来のdriver/source identity、正式schedule/angle/call予算、実price/provider binding、採択・実行承認と初回実入力correctnessである。[差分チェックリスト](p1_final_delta_checklist.md)に限定して引き継ぐ。公開準備版から将来の採択・実行済み状態を推測しない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[ACOMP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ff0bed24ba8fdb361a2890646ef84347ae37d457/docs/research/track_b_g7_g9_strong_comparator_audit_20261019/comparison_conditions.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/b3_classical_preparation.md
[BC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/execution_contract_candidate.json
[BS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/src/trottertracks/algorithm_codesign/g6_w2c_cost_preparation.py
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/c1_native_preparation.md
[G9MC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/schemas/provider_price.schema.json
[STAGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8cbeb18f76aef2cb1329d8c810036372fa07cc0/docs/research/track_b_g6_w2c_b3_c1_preparation_20261019/schemas/cost_record.schema.json
