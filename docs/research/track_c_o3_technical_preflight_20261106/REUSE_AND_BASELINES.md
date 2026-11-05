# Track C：固定SOURCEの再利用と強いbaseline

2026-11-06 JST。[REPORT](REPORT.md)と[CANDIDATE_MATRIX](CANDIDATE_MATRIX.md)を併読する。SOURCEの読解だけを実施し、以下の関数・runner・testは起動していない。過去PASSを新条件の妥当性・新しい科学性能へ移さない。

## 1. 版と確認深度

| 記号 | 固定版・役割 |
|---|---|
| I | `3571477093441ac4852166fe829f8ccc5db5ec4f`の[統合再利用地図][reuse-map]・台帳。PF-ORDER/BF-BCH/ALLOCATION/RPE-COST等を引継ぐ。科学execution SOURCEの一括認定ではない |
| B | `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`。今回branch基点、文献調査/主repository内SOURCEの読取snapshot。**live mainではなくcarrier版** |
| P | `f29eafbf4412193e8e8660d9f051889c2cba684b`。統合表のPF-ORDER/RPE-COST等の実装所在。各元runのexecution SHAを代行しない |
| Y | 別PF repository `HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae`、`20d64c2fea1ed2d2188777df4f3d962ee94df486`。公開係数・processor/費用関数をGitHub connectorで読取 |
| A | `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`。訂正Parent/S4 SOURCE。旧proxy目的版との区別を維持 |
| N | `546bca7ed2cece59b714b986a264d7b0b0871a01`。統合表のnative/再配分/関連testの固定収録版。各過去実行は原manifestへ戻る |

主repositoryの参照は固定Git object本文を確認。function/入力/出力は本文またはASTを静的に読んだ範囲。Yのproduct_formula.pyは先頭260行、processed_cost.pyは本文を直接取得し、これ以外の外部全SOURCEを精読したとはしない。統合の143群/179版を今回全件再監査・再実行していない。

## 2. 再利用単位・入出力・適合範囲

表中のpathは表の固定版に対するもの。リンクのない補助pathも同版のGit treeで実在確認した。新runner/新frameworkは作成していない。

| 資産群・固定SOURCE | 実装機能と入力→出力 | 既存検証・科学的適用条件 | 再利用範囲／最小限の拡張が必要となる場合 | 対応baseline |
|---|---|---|---|---|
| **PF-ORDER / P**：[product_formula.py][pf]、`pf_decomposition.py`、[research_direction_energy_tail_pareto.py][pd-code] | `_get_w_list`、`iter_pf_steps`、`formula_registry`、`composition_tail_weights`。formula label/term数→順序付き係数、tail次数/符号/stage metadata。固定H→数値比較は評価関数側 | S2/4/new4/Yoshida8/旧Morales8のregistry。P-Dの非可換toy次数とH4 LD3/4。new_2の8桁精度・旧m8版を保持 | 既知係数/decompositionを再実装しない。高次finite-RTE full-wrapperはregistryから自動完成しない。科学評価関数の呼出しは別許可 | S2、Yoshida4、同じ既知係数と同じerror/price情報を持つ対照 |
| **内外PF診断**：科学SOURCE `087a984c78e9d0e8389544b1f47a306374917a9a`、[現実化報告][pd-real]。P収録`research_direction_pd_realization.py` | signed tail小行列/control、fragment内部D近似と二block参照の比較。固定snapshot/formula/step→意味論・bias/weight診断 | H4 1Å/STO-3G/DF12、LD3/4/5、δ.2/.4、K2、R64、D内部32step。signed小行列のPASSはcompiled H4全meanではない | 内部Dと外側D/Rの分離・対照を再利用。old gridを再実行しない。新native/signal適合が必要ならその差分だけ | exact-D/Rは診断、実装可能なFull PFと誤差予算を合わせたnear-integrable/既存nested |
| **公開高次・processor / Y**：[product_formula.py][ypf]、[processed_cost.py][ycost] | `morales_2025_y8m10b_list`、`morales_2025_yp8m8_kernel_list`、`morales_2025_yp8m8_processor_sequence`。`ProcessedCostComponents.total(kernel_steps,processor_pair_count)`→kernel反復＋端点pair構造価格 | 旧m8と公開版を分離済み。研究Bにpublished列/actual inverseの照合記録。helper価格はPauli rotations/RZ layerでnative full期待価格ではない | 列とinverse規約・端点会計を再利用。実finite-RTE/processor境界・元H groundのsignal適合を補う必要あり。Yの過去実行SOURCEは各原報告へ | 同kernel単体、公開Y8m10b、既知processed formula、S2/4の同精度調整 |
| **PF phase / P**：[pf_delta_validation.py][phase]、runner`run_pf_delta_validation.py`、test`test_pf_delta_validation.py` | `paper_d6_perturbative_energy_bias`、`validate_pf_delta_grid`。H/sector/LD/δ/q→D6/支配phase/branch weight/survival診断 | D6のsmall-sine conditioning、支配branch・state-actionの元検証窓を保持。D6やC_useは一般厳密energy boundではない | phaseとsurvivalを分ける既存診断を再利用。任意高次の保証をS2検証から移さない。新condition評価は別許可 | exact元H evolutionは診断参照、同stateのPF dominant branch/複素signal |
| **BF-BCH**：[BM auditor / `4301c0f87a16be45b620304c3fa935d5aa2d721a`][bm-code]、[BF1復旧報告][bf-report] | BMはfree wordのFraction演算、degree3 exp/log/comm→形式係数同値。BFは元SOURCE `e59344a564e70d64dc3ea39d640581c72676df31`、復旧SOURCE `f226f8c81b5ba07b6d0c4b248c0eb15bf080a622` | BM既知compact同値、BF元INCONCLUSIVEと保存復旧BF-Aを別に保持。復旧は独立cohortでない | 同値性・符号規約・cache/強対照の記録を引用。auditorも今回は再実行しない。BF runnerのretry/bridge穴埋めはしない | 同じnested列・DF backend・aggregation/cacheのcompact。BFでは同budget L＋fixed/native S2 |
| **RTE-BASE / P**：[rte.py][rte] | `RTEConfig`、`finite_rte_distribution`、`compose_finite_rte_occurrences`、`finite_rte_operator_moments`、`finite_rte_combined_attenuation`。tail/signed time/整数r/even K→finite分布、corrected/raw mean、B/attenuation、truncation | paired Taylorはeven KでK+1次数まで。probabilityとLCU coefficientを分離。heterogeneous occurrenceはtail/time/r/Kと順序を保持 | signed/異種occurrenceを再実装しない。samplerやmatrix関数は起動しない。合成残差辞書を追加する際は同じtarget/normalization/weightの接口に限定 | canonical finite-RTE、実装可能なexact R、通常PR |
| **DF-CONTROL / P**：[df_rte_qiskit.py][control]、`df_partial_s2*.py`、`df_rpe_resource.py` | `QiskitDFRTEEventCircuitBuilder.build_event/build_sequence`。event/spec/basis registry/control→phase保持ordered回路と共有metadata | diagonal-only control、identityのancilla相対phase、隣接equal basis/明示basis plan。元snapshot/compiler/Ordinary control範囲 | control/Givens/既知fusionを共通に流用。任意PFのfull-wrapperやT合成がこのbuilderだけで済むわけではない。adapterは将来必要な差分のみ | 既知共有を全armへ与えたfull/support baseline、無共有は診断 |
| **SP-R1 / `4301c0f…`**：[SP1 auditor][sp-code]、`audit_sp05_saved_result.py`、`analyze_r1p5_saved_attribution.py` | saved interval/key/primitive/price→既存保存関係の監査・帰属。新synthesis/新priceを生成する資産ではない | SP1 synthetic2qubit/additive T、長model loss/cap。R1.5は同情報CTS等との違いを保持 | 有効域と反証の記述、schema/key/実meanと補正meanの対応を参照。今回保存再会計なし。旧SP toyのrunner延長なし | 同primitive catalogue/誤差guardの通常合成NONE、CTS/既知PAI・PR residual |
| **ALLOCATION / A**：[parent_allocation.py][parent]、Nの`rte_reallocation/model.py/native.py/accounting.py` | `digital_proposal`、`reweight_events`、`event_statistics`、`error_accounting`、`native_plan`、`adjoint_plan`。供給済finite table/価格/誤差→proposal、weights、moment、phase保持plan。nativeの`adjoint/simplify/lower_event`→symbolic IR | 訂正S4/KKT、全trial/weightと外部誤差入力を区別。`native_plan`はcallback記述、providerのcorrectness保証ではない。simplifyはexact adjacent inverseのみ | 同mean/実proposal/actual adjointの会計を流用。価格取得もfinite-confidence最適もこのAPIは提供しない。旧承認機構を新研究へ複製しない | canonical/共通角/同情報cost-aware対照。p/√c等の再発見をmethod deltaにしない |
| **RPE-COST / P**：[rpe_resource_accounting.py][account]、[research_direction_full_opt2.py][opt2] | `evaluate_rpe_round_candidate`、`RPERoundCostProvider`、`build_rpe_resource_summary`、`load_and_validate_full_opt2_evidence/build_full_opt2_task_definitions`。preparation/round/β/α/error model/provider→feasibility、N、軸別価格、合計 | accountingはpartial-S2/Cδ²/基準半径1モデル。opt2は保存evidence/固定split入力で、任意processed/weighted保証ではない | provider境界と軸別費用・失敗配分を利用。高次のbias model、実半径、synthesis/weighted測定を受け渡す最小変更が必要。汎用optimizerの新設は不要 | 同じ配分機会を持つoptimized fixed-split Partial/Full、高次Full PF |
| **後続RPE/energy / B**：[V3-R1][r1]、統合`SUP-RPE-V1/V2/V3R1/ENERGY` | 保存fixed plan・較正依存・長q移送・decoder/energyの元接口 | 条件付き18round/energyの既存成果を保持。V3-R1はq≤32直接域と最終q131072までの外挿を区別し総合UNDETERMINED | 較正共分散とsystematicの扱い、branch/aliasの条件を参照。保存値を新rankingへ再集計しない。decoder/enclosureのSOURCEは各原manifestへ | 同target・同alias prior・同保証域のfixed-plan Partial対Full。新round別表現は別課題 |

## 3. テスト・pipelineを再利用する範囲

| 固定版の入口 | 既存の検査対象 | 今回の扱い／将来の差分 |
|---|---|---|
| P：`tests/test_research_direction_energy_tail_pareto.py`、`test_research_direction_pd_realization.py`、`test_research_direction_pd_fair_comparison.py`。対応`run_*.py` | registry symmetry/normalization、toy order、内外PFの区別、signed control、小行列参照、保存artifact | 本文・定義を静的確認。これらには科学評価を呼ぶtestがあるので**起動していない**。新機構を検証する場合だけ適合する人工controlを選ぶ |
| P：`tests/test_df_rte_qiskit_builder.py`、`test_df_partial_s2_repeated.py`、`test_finite_rte_signal_validation.py` | 順序、basis共有、controlled相対phase、反復、finite mean | path実在を確認し、builder/mean接口を読取。単体PASSを元H全RPEの保証にしない |
| A/N：`tests/tracks/algorithm_codesign/test_parent_allocation.py`、`test_parent_allocation_native.py` | corrected mean、実proposal、signed非可換word、外部誤差の欠落拒否、phase落ち、actual adjoint、mock native binding | Aのtest名を静的確認。mock/人工の過去成功とphysical provider資格を分ける。旧testの大量再実行や承認証明testの新設は不要 |
| P：`tests/test_rpe_resource_accounting.py`、`test_rpe_resource_accounting_guards.py`、`test_rpe_allocation_sensitivity_validation.py`。対応runner | 半径/phase/shot/providerとerror配分の元モデル | 適用前提が変わった部分だけ将来検査。unit-radius/S2仮定を高次・weighted測定へ無条件拡張しない |
| PR-2：`e3531956af6a6d359e2f1fb320e9a5179f41eb98`のM2報告・M1 B1結果validator、BのM2 execution/test path | fixed候補のmatched complex accuracy、保存SOURCE/軸別price/paired covariance/transfer | 研究runnerは起動しない。全energy課題を再検証済みとしない。過去実行closureはM2原manifestのまま |

研究Bの元execution SOURCEは一括UNKNOWNへ潰さない。[最終handoff][b-final]はD内部36行 `758a05ed6aac58a5945af5a5330c792a842b174d`、native-v5 `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`、Full Yoshida4 q4 `620448b14235d2d9e0e76d5d0ba7273ce887a628`等を段階別に列挙する。今回直接読んだのはこのhandoffと係数/費用helper等であり、これら全execution closureを再監査したとはしない。

## 4. 強baselineの組み方

| 比較の層 | 必要な強対照と理由 | 禁止する弱い比較 |
|---|---|---|
| PF選択 | S2・Yoshida4、適合する旧/公開Morales・processedを版別に扱い、各方式にstep/精度の最適化機会を与える。near-integrable/SPRINTは適用できる構成要素のみ | energy-only、内部D workを無視したouter stageだけ、同候補探索を禁じたleadingモデルを唯一の対照にする |
| BCH/有限model | 同nested列のcompact再帰、同DF/backend、同集約・cache、P-D B1b/B2/B4、BFのL/fixed対照 | compact側だけflat化/再計算、候補だけexact bias/price情報を使う |
| 合成 | 通常合成、既知residual/PAI/TETRISを同native精度・角度情報・actual adjoint・fusionで対応させる | additive primitive Tのgainをcompiled CX/RZ/全Gのgainと呼ぶ。未知rare priceを0にする |
| processing | 同kernel単体と既知processed、同元H ground、物理P費用と有限時間信号 | 共役で固有値が変わると説明する。候補だけprocessed stateを無料供給する |
| RPE | 同εE/失敗率、同state/alias prior/入力情報、同じβ/α・r/K最適化機会、全X/Y roundとfull wrapper | 短qだけ、最終roundだけ、query数だけ、またはsource_present/test PASSだけで全費用を断定する |

UWCは表現＋split＋RPE費用を結ぶ既知比較、PR原著はrounding-to-residualの直接比較、TE-PAI/continuous-time PAIはchannel/operatorの条件差を持つ比較である。[固定先行調査][survey]のlocatorと本REPORTの限定直接確認を参照する。外部paperのモデル削減率を本SOURCEの改善率へ移さない。

## 5. 並列性・必要な最小実装・次の担当

C-Nは既存PF＋固定合成、C-Sは固定PF＋合成/残差、C-Pは固定kernel＋P有無で個別評価可能。共通のH/state/target・full wrapper・signal/shot/branch会計を使う。C-Nの全価格取得完了や3候補の統合実装を、他候補の静的検討の前提にしない。

将来の最小実装候補は、既存のsigned occurrence/phase保持builderとproviderへ、選ばれた高次またはprocessed構成の**必要差分だけ**を接続すること。unit-radius/S2/weighted測定の違いは明示的なデータとして渡す。新しい承認authority、grant発行、承認証明ledger、汎用研究runnerは不要。元研究で固定済みの条件・SOURCE・STOPは保全する。

**ユーザー**：必要なら候補・元H/精度/成功率/資源/許可範囲を選択し、旧研究再開が含まれるかを明示する。**Codex**：別許可の後に適合する既存SOURCEを再利用し、必要な差分と最小検証を実施する。今回の自動後続作業なし。3成果物の公開照合後mandatory STOP。

[reuse-map]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[pf]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/product_formula.py
[pd-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_energy_tail_pareto.py
[pd-real]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_pd_realization.md
[ypf]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[ycost]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[phase]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/pf_delta_validation.py
[bm-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_bm05_symbolic_equivalence.py
[bf-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d2645a09440f50e5b869ef42a1b73a1b625a1af/docs/tracks/algorithm_codesign/bf1_read_only_recovery_result_validation_20261005.md
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rte.py
[control]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/df_rte_qiskit.py
[sp-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/scripts/tracks/algorithm_codesign/audit_sp1_saved_result.py
[parent]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rpe_resource_accounting.py
[opt2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_full_opt2.py
[r1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_v3_r1_shared_calibration_sensitivity_20261103/REPORT.md
[b-final]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f9adc2ce1c373c7ba311864c733ac6f40b575fbc/docs/research/research_b_final_outcomes_and_pause_handoff_20261026.md
[survey]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
