# PR改善研究の概念候補

2026-10-23 JST。**候補の提案であり、研究の正式採択・新規性審査・性能実証・実行契約ではない。** 並び順は優先順位を意味しない。最小検証は将来案で、今回は実施していない。既存のPAUSED/CLOSED/STOPを解除する指示でもない。

- I：既存成果をそのまま、または小変更で転用する候補。
- II：過去に検討した問題に、別条件・具体的手順差があれば再検討する候補。
- III：確認したプロジェクト記録では接続の探索が薄い可能性がある自由度。未探索・世界的新規性の証明ではない。

全候補は、改善対象操作の総費用に占める割合、normalization/moment/range、bias headroom、必要shots、native単価、状態準備/測定と古典取得費を同じtaskで扱う必要がある。全PRで最大の費用項が確定したとはしない。異なるtoy/分子・compiler・precisionの比率で候補を順位付けしない。

## OP-I1 run-level基底共有と合法な境界相殺をPR wrapperへ再利用

| 項目 | 内容 |
|---|---|
| 分類 | I |
| 変更部分 | DF基底変換・隣接event・controlled wrapper |
| 期待する機構 | run-level full/support-union選択とliteral inverse cancellationを同じ合法な回路最適化として両baselineに与える。 |
| 削減資源と全費用との関係 | RZ/CX/depthとone-shot費用。信号mean・normalizationを保持できる場合だけshot数を分離できる。 |
| 既存研究との重複 | basis共有・gate cancellation自体は既存成果。P-Aのinterval DP固有寄与は停止済みで、新名称で復活させない。 |
| 実装等の再利用 | 既存basis-plan builder、support_run_le_1、boundary記録。 |
| 弱点・交絡 | すでにfusion済みの境界、mixed-supportでの悪化、compiler/相対位相への依存。量子費用内のbasis占有率が未取得のcontextもある。 |
| 将来の最小技術検証 | 将来、固定event列で既存と追加共有の差分を限定し、actual-adjoint/ordinary controlを含む同値性とfull wrapper費用を照合する。 |
| 新科学計算の必要性 | 新回路build/compileが必要なら別途実行承認。今回は実施0。 |
| 不足する証拠 | 旧H4 rawへの厳密帰属はGAP-002。別contextへの利益・費用占有率は未検証。 |
| 新規性の留保 | 既存成果の転用。独立新算法claimなし。 |

根拠：[CIR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md), [CIR-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pa_joint_synthesis_v1_formalization.md), [RB-13 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)

## OP-I2 Closed P5特殊化をPR tail候補へ接続

| 項目 | 内容 |
|---|---|
| 分類 | I |
| 変更部分 | 低Taylor次数のtail表現・生成 |
| 期待する機構 | m=5でclosed group normalizerと角度共有を用い、同じ有限meanを生成する。 |
| 削減資源と全費用との関係 | Tとnormalization/十分N、setup/online古典費用。CXはCTS対照が有利になり得る。 |
| 既存研究との重複 | Closed P5はRA内部特殊化で既に評価済み。General Fullとの差を別原理の発見としない。H4 C1移植も既に探索済み。 |
| 実装等の再利用 | g9_p5、matched-native provider、C1 exact packet/adapter。 |
| 弱点・交絡 | ideal ensemble一致とdigital law/全trial law一致は別。H4でのlogical差は小さく、strict native・十分N・deterministic halves未接続。C1は凍結中。 |
| 将来の最小技術検証 | 将来、現在凍結したC1と異なる具体的条件差を先に提示し、同finite target・同情報・P5/CTS強対照の接続不足だけを特定する。 |
| 新科学計算の必要性 | static binding後、native/mean/費用を新取得するなら別承認。既存C1の再開は自動認可されない。 |
| 不足する証拠 | GAP-007は旧失敗の来歴。現在未取得のnative/Nは資料回収と区別する。 |
| 新規性の留保 | 転用候補。現条件での優位を推奨・採択しない。 |

根拠：[RB-06 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md), [RB-09 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md), [RB-10 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md), [RB-17 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)

## OP-I3 Pauli/DFの既存4方式比較基盤を共通PR契約へ利用

| 項目 | 内容 |
|---|---|
| 分類 | I |
| 変更部分 | Hamiltonian表現・prefix単位・tail回路 |
| 期待する機構 | 同じ元H・状態・固定signal taskで、Pauli/DFが作るtail normとnative単価の交換を比較できる。 |
| 削減資源と全費用との関係 | C、analytic N、shot-weighted RZ/CX。未観測次数の費用を0にしない。 |
| 既存研究との重複 | H2固定4方式比較は2026-10-23に完了済み。非DF一般を未探索と呼ばない。H4は既に準備案がある。 |
| 実装等の再利用 | representation identity receipt、finite-RTE law、process分離・監視、既存H2保存結果。 |
| 弱点・交絡 | 異なる表現のL_Dは同じ数え方でない。H2の有限標本・経験的allowanceと別系への移送は限定。 |
| 将来の最小技術検証 | 将来、既存H4準備案との重複を確認し、共通H/sector/scalar・prefix決定規則・rare-orderの扱いを固定する。 |
| 新科学計算の必要性 | H4等の生成/signal/MC/compileには対象限定の別承認。今回は準備・科学実行を追加しない。 |
| 不足する証拠 | H2旧失敗とrun02成功は区別済み。別系の性能は未実施検証で、回収待ちrawではない。 |
| 新規性の留保 | 既存基盤の利用。表現名の追加を新算法としない。 |

根拠：[REP-01 固定資料](https://github.com/HIROMU1015/Evaluation_numGate_highorder_DF/blob/4d90167f0634a67ca1ed6bf2630af075c35a149f/README.md), [REP-08 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)

## OP-I4 rank・split・PF次数・r/Kを同じ費用台帳で扱う

| 項目 | 内容 |
|---|---|
| 分類 | I |
| 変更部分 | 有限候補の協調設計 |
| 期待する機構 | discard/PF/finite-RTE biasとnormalization、単価・shotsを同じ固定taskへ戻して比較する。 |
| 削減資源と全費用との関係 | shot-weighted compiled cost、状態準備感度、取得費。 |
| 既存研究との重複 | PR-2/PMとP-D/Track Aで既存の比較軸。広い最適化だけではR3の停止理由を越えない。 |
| 実装等の再利用 | matched-accuracy accounting、既存PF係数、precision/prep sensitivity。 |
| 弱点・交絡 | cheap surrogateの除外安全性、同情報baseline、rank/係数版/allowance政策の不一致。四次逆転の原因究明は別チャット。 |
| 将来の最小技術検証 | 将来の候補選定前に、既存固定表で既に比較済みの組と未知の接続を文書上で分離する。 |
| 新科学計算の必要性 | 保存rawの新再会計や新候補評価は別途承認。今回は新しい順位を計算しない。 |
| 不足する証拠 | 旧stale DFはGAP-008。次数逆転の因果は未確定として維持。 |
| 新規性の留保 | 再利用可能な評価枠組み。新selector・正式優先順位ではない。 |

根拠：[OPT-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md), [OPT-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_energy_tail_pareto.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md), [EVAL-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md), [EVAL-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/track_a_pf_order_reversal_existing_evidence_audit_20261023.md)

## OP-II1 PR目的を明示した軌道／DF表現の協調設計

| 項目 | 内容 |
|---|---|
| 分類 | II |
| 変更部分 | 軌道、DF fragment、deterministic/random split |
| 期待する機構 | normだけでなくPF bias、tail normalization、basis移動・native単価を同時に小さくする表現を狙う。 |
| 削減資源と全費用との関係 | RZ/CX/Tと必要shots、表現取得/最適化の古典費用。 |
| 既存研究との重複 | orbital-rotated DF、表現selector、UWCは検討済み。UWCのorbital interface予約は実行済み軌道最適化ではない。 |
| 実装等の再利用 | DF/Gaussian builder、UWC sector metadata、N1 exact fallbackとcore/pair強対照。 |
| 弱点・交絡 | norm低減が単価増に負けた既存例、DF/tapering誤差、局所最適、取得oracleを無料化する交絡。 |
| 将来の最小技術検証 | 将来、具体的な軌道生成/更新規則と同情報の既存orbital baselineを一つ定義し、どの費用項を変えるかを先に特定する。 |
| 新科学計算の必要性 | 最適化・分子再生成・PF/signal/compileは別の科学実行承認を要する。今回は規則を実装しない。 |
| 不足する証拠 | UWC成績の根拠はGAP-001。旧rank screeningはGAP-008。 |
| 新規性の留保 | 既存表現研究の再設計候補。軌道最適化そのものの新規性は主張しない。 |

根拠：[REP-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md), [REP-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md), [REP-09 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/representation_attribution_pair_results_20261010.md), [QC-07 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_new_algorithm_research_scope_20261016/research_coverage_map.md), [OPT-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_energy_tail_pareto.md)

## OP-II2 相互作用・回路共有を保つ非prefix grouping/split

| 項目 | 内容 |
|---|---|
| 分類 | II |
| 変更部分 | 決定論/ランダム部分の選び方と演算子順序 |
| 期待する機構 | 単項weightだけでなくcommutator構造と共有basisの価格を考慮し、同じrankでも費用の高い境界を避ける。 |
| 削減資源と全費用との関係 | PF bias、tail normalization、basis/CX/depth、十分shots。 |
| 既存研究との重複 | split最適化・grouping・SPRINT系・compact BCHは既知。prefix追跡とR3/BMの既存停止を名称変更で解除しない。 |
| 実装等の再利用 | 固定DF registry、compact leading評価、full-wrapper builderとmatched accounting。 |
| 弱点・交絡 | score選択の公平性、計算量、候補探索/評価費、同情報known baselineが同じ選択を得る可能性。 |
| 将来の最小技術検証 | 将来、同情報known方法からは得られない具体的な生成/選別差と、差を見分ける固定controlを提示する。 |
| 新科学計算の必要性 | 新splitの数値評価/再ランキングは別承認。今回は新しい最適L_Dやrankを選ばない。 |
| 不足する証拠 | 既存SOURCEは使えるが、独立method deltaと性能は未確立。 |
| 新規性の留保 | 条件を変える理由が必要な再検討候補。一般groupingを新自由度としない。 |

根拠：[OPT-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_geometry_energy_difference_pilot.md), [OPT-05 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/r3_prior_art_and_minimal_contract.md), [RB-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md), [QC-05 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md)

## OP-II3 生成費を支払うcost-aware whole-path sampling

| 項目 | 内容 |
|---|---|
| 分類 | II |
| 変更部分 | sampling law・history・zero/rejection・全path費用 |
| 期待する機構 | 高価なbasis切替やnative eventを避けるproposalを、正しいlikelihood weightと共に生成する。 |
| 削減資源と全費用との関係 | native費用×十分N、range/tail、probability/normalizer取得、online古典費。 |
| 既存研究との重複 | importance sampling、history相関、全path費用最適化自体は既知。現在の候補は独立生成差未成立。 |
| 実装等の再利用 | rte.pyのphase/order metadata、全trial estimator、finite-bit/zero-fill台帳。 |
| 弱点・交絡 | marginal一致だけでmeanが壊れる、重み二次moment/range増、D挿入と相対位相、巨大inverse stack、古典費用。 |
| 将来の最小技術検証 | 将来、一つのcompact proposal生成則についてsupport・全path likelihood・coherent first momentの必要条件を明記し、最も近い既知法との差を先に照合する。 |
| 新科学計算の必要性 | 新証明・sampling/compile/数値比較が必要なら別承認。今回は新lawを生成・実行しない。 |
| 不足する証拠 | 既存文献対応はある。欠けているのは新しい安い具体手順であり、既存raw不足だけではない。 |
| 新規性の留保 | 再検討の条件付き候補。q∝\|a\|/sqrt(g)等の既知式を新規案として数えない。 |

根拠：[QC-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_c4_finite_rte_gpt_handoff_v1_20261014.md), [RB-18 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md), [RB-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md), [RB-14 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md), [RB-08 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md)

## OP-II4 return/Euler回転とnative synthesis placementの協調

| 項目 | 内容 |
|---|---|
| 分類 | II |
| 変更部分 | 回転辞書・parent配分・joint control・合成位置 |
| 期待する機構 | 単独角度価格だけでなく隣接融合、符号付きjoint phase、cheap exact角、十分Nを一体で扱う。 |
| 削減資源と全費用との関係 | T/CX/depth、moment/range、bias、workspace、prep/readout。 |
| 既存研究との重複 | SP・Parent allocation・D2境界相殺・Closed P5に既存結果。単なる再組合せや同toy再探索を新研究としない。 |
| 実装等の再利用 | native signed pair、actual-adjoint、合成sequenceと価格ledger、fixed-target比較。 |
| 弱点・交絡 | 加法単価とwhole compilerの差、TとCXの交換、価格取得後の選択bias、zero-cost eventでの最適law境界。 |
| 将来の最小技術検証 | 将来、既存strong baselineがすでに融合する箇所を除き、具体的に省ける操作と同精度・同情報の対照を指定する。 |
| 新科学計算の必要性 | 新angle/native合成/whole-wrapper比較は別承認。過去SP/S4/DF補正STOPを再開しない。 |
| 不足する証拠 | 全wrapper費用と生成費の接続は未実施。局所PASSは実用GOにしない。 |
| 新規性の留保 | 既存構成の条件付き再設計。独立性は未確認。 |

根拠：[RB-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md), [RB-09 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md), [RB-11 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md), [RB-13 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md), [RB-14 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md), [CIR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)

## OP-III1 接続性を含むPR表現・split・basis移動の協調

| 項目 | 内容 |
|---|---|
| 分類 | III |
| 変更部分 | coupling mapを持つcompiled生成 |
| 期待する機構 | basisを頻繁に跨ぐrandom tailとdeterministic backboneを、実接続上で共有しやすい配置・生成単位へ合わせる。 |
| 削減資源と全費用との関係 | CX/SWAP/depth、RZ/T、状態/制御の追加費。 |
| 既存研究との重複 | routing・mapping・Gaussian共有は既知。今回確認した主compiled比較はroutingなしが中心で、この組合せの研究記録は十分確認できていない。 |
| 実装等の再利用 | 同H/同meanのbuilderとcompiler identity、既存basis plan。 |
| 弱点・交絡 | backend選択による恣意性、両baselineへの同じrouting最適化、単価改善のshotでの相殺。 |
| 将来の最小技術検証 | 将来、抽象接続性と一つの明示生成差を固定し、既知mapping/compilerへ同じ合法情報を与えた比較が必要。 |
| 新科学計算の必要性 | 新build/routing/compileには別実行承認。今回はbackendや配置を選ばない。 |
| 不足する証拠 | 記録不足は未探索の証明でない。発見可能branch以外と近接文献の確認が必要。 |
| 新規性の留保 | プロジェクト内の記録が薄い可能性がある自由度。世界的新規性の認定ではない。 |

根拠：[CIR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md), [REP-01 固定資料](https://github.com/HIROMU1015/Evaluation_numGate_highorder_DF/blob/4d90167f0634a67ca1ed6bf2630af075c35a149f/README.md), [REP-08 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md), [RB-17 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md)

## OP-III2 roundごとの表現・rank・split変更を許すPR契約

| 項目 | 内容 |
|---|---|
| 分類 | III |
| 変更部分 | 複数時刻/RPE round間の協調設計 |
| 期待する機構 | 固定splitで全roundを賄う代わりに、roundごとのbias/visibility/単価条件に応じて構成を変える余地を調べる。 |
| 削減資源と全費用との関係 | round合計のshots×wrapper費、再準備・前処理・compiler取得費。 |
| 既存研究との重複 | round別r/K・beta/alpha配分は既存。round別L_D/表現まで含む同一absolute-energy contractの十分な検討記録は今回確認できていない。 |
| 実装等の再利用 | 既存round schedule・matched-accuracy台帳・branch/phase ledger。 |
| 弱点・交絡 | PF固有位相がroundごとに変わる、同じE0のdecoderへのbridge、state access、探索費。単なる一般動的計画なら新規性は弱い。 |
| 将来の最小技術検証 | 将来、構成を変えても同じ目的energyを読む条件と、固定構成に対して何を省けるかを先に文書化する。 |
| 新科学計算の必要性 | 新しい非自明な保証・signal/RPE/費用再会計はすべて別承認。今回は設計を正式採択しない。 |
| 不足する証拠 | Task Aのsignal比較をTask Bの全RPEへ移す証拠が未取得。これは回収だけで埋まるとは限らない。 |
| 新規性の留保 | 確認資料で薄い接続候補。新しい算法や性能向上を確立したものではない。 |

根拠：[OPT-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_wp11_synthesis.md), [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md), [EVAL-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_pm2_precision_resource_result_validation.md), [RB-15 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md), [PF-05 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/second_study_v2/evidence_integration_20261006/approved_closure.md)

## 提案に留める次段階

ユーザー/GPTが今回の研究地図とGAPを読み、どの具体的な改善レバーを取り上げるか決める。必要な原資料回収はユーザーが別環境Codexへ指示する。候補が選ばれた場合に限り、Codexへ同情報強対照・生成手順・scopeの具体化を別途依頼する。重要科学レビューと、新しい科学計算/再解析/compileの実行承認は別々である。

今回のCodexは候補の正式選定、新算法実装、科学契約/grant、停止研究の再開を開始せずmandatory STOP。
