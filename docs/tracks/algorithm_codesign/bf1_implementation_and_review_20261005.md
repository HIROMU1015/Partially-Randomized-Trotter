# BF-1 preparation / source review packet

2026-10-05 JST。`BF1_PREPARATION_COMPLETE_REVIEW_REQUIRED`。science executionは未認可。

BF-0 commit `15465b0b856d80f9cfde495fd3d22434825f1cc8`のreview回答を受領した。
判定は**`PASS_FOR_BF1_PREREGISTRATION`**。回答者はこのチャットの利用者であり、
別の研究者への依頼・送信・独立signoffを実施したとは記録しない。
reviewされた四文書とそこに固定された三本文hashは変更しない。
後続の[preregistration v1](bf1_preregistration_v1.md)へ五条件と実装上の具体化を記録した。

| review条件 | 反映先と結果前の決定 |
|---|---|
| BF-C primaryを一つに | F対全O/L/fixed finite bestのratio≤.95。q/Pareto/feasibilityはsecondary |
| Oの役割 | 診断ablation。F対Lがnovelty-relevant comparison |
| 五stage scope | 同一五stageのみ探索。四fixed refsを維持し、追加family探索なし |
| numerical domain freeze | 4 chart / 3 component、全端点、arc、Suzuki/Yoshida exact定義、16 starts、eta1/eta3 screen、80-digit lowering |
| F adapter semantic test | synthetic matrix、負時間、phase、exact fusion、F/S差、shared finite RTEとの照合、固定係数のword-series残差 |

入力sourceの読取りから、prefix順序はgeneration-prefixへ、deterministic generator数はone-body込み4へ具体化した。
stateは新規solveをせず、文書に既にidentityがある保存済みstateとする。NPZ実物は触れていない。
arc端点を未採点virtual boundaryとして固定し、各armの同じrefinement budgetで端部を含める。
K4 triggerのideal biasは数値guard込み。これは結果を見た変更ではない。

実装は以下の対応でreviewする。

| 場所 | 内容 |
|---|---|
| `src/trottertracks/algorithm_codesign/domain.py` | exact affine coefficients、全branch、arc、16 starts、published fixed refs |
| `adapter.py` | chronological list、再帰的exact取消・融合、stationarity、固定integer allocation |
| `numerics.py` | CPU spectral actions、residual/orthogonality/roundoff/Taylor guard |
| `pilot.py` | O/L/F objectives、共通32評価policy、common finite rescore、一primary判定、STOP |
| `input_contract.py` / `science_input.py` | literal source identityと将来の保存input loader。preparationからloaderは呼ばない |
| `shared_cpu.py` | 共通sourceを参照するCPU import境界。通常package初期化とGPU moduleのCUDA preloadを回避 |
| `freeze.py` | explicit text-source inventory、environment、source/auth/commit gate |
| `scripts/tracks/algorithm_codesign/prepare_bf1.py` | formula-only enumerationと限定synthetic tests、準備packet作成 |
| `scripts/tracks/algorithm_codesign/run_bf1.py` | 将来のone-shot runner。draft authはinputより前でreject |
| `tests/tracks/algorithm_codesign/` | B固有semantic/fairness/authorization検証のみ。全suiteではない |

`shared_cpu.py`は共有libraryをコピー・編集せず、同じfileをprivate package名でimportする。
`df_gpu_statevector.py`のmodule-level CUDA探索/preloadを実行せず、GPU entry pointsは必ずraiseするstubへ束縛する。
input loaderはpure dense DF reconstructionと既存`exact_df_diagonal_coefficients`を使い、
`dense_extracted_df_tail`のbasis circuit constructionやDF controlled wrapperを呼ばない。
このadapterをDF controlled wrapper検証済みと扱わない。

準備packetのmachine-readable reportを実際のローカル検証記録とする。
`scientific_results_present=false`、`BF1_executed=false`、`source_commit_bound=false`。
synthetic一致とsource hashは実装の部分的検査であり、immutable CI・外部再現・新規性証明ではない。

実行前reviewでは特に次を閉じる。

1. numerical guardのforward-error導出、DF assembly allowance、coefficient loweringとshot/ratio interval。
2. sourceとpreregistrationのO/L/F・refinement・K4・boundary・case定義の一致。
3. source/environmentをcommit固定したidentityと、別の一回実行authorization。

現状のauthorization draftはfalseである。新しいgeometry、算法、係数budget、q/R/K条件を追加して
positiveを探さない。科学実行はこのreview packetから始めず、利用者reviewへ戻る。
