# RA-RTE統合数学監査：GPTへの引継ぎ

2026-10-06。基点R1.5 `af3d014d0a0cfcbbd25bb544f6544652fec92942`。
**DOCS_SYMBOLIC_ONLY_MATHEMATICAL_AUDIT**。science・synthesis・solver・R1資源再採点0。
設計書SHA256=`4d8e448fa7d2386a545fdc7b9204a4738b917605bb4a7c240dd07f0c196b6826`、
checker SHA256=`c6aa6cbc2848921080fae154f85fc766e5e70f7763834e56720492b26305d56e`。

最初に[統合監査報告](ra_rte_mathematical_audit_v1.md)を読む。
原文は[設計snapshot](inputs/ra_rte_mathematical_design_after_r1p5_20261006_user_input.md)にbyte-exactで保持した。
[機械可読readout](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/proposition_readout_v1.json)、
[50 off-domain checks](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/bookkeeping_checks_v1.json)、
[manifest](../../../artifacts/track_b_ra_rte_mathematical_audit/2026-10-06/evidence_manifest_v1.json)から追跡できる。
本資料の固定commit URLはCodexのpush後報告を使用する。

## 結果

主導出Dw=t → Dq=yt、sum q=1、ey-d^Tq>=kappa_nは、明示仮定下で元のBernstein十分条件と同値。
dual、mixed-class normalization下界、residual会計、IS convexity、adjoint pairing、条件付きmulti-block boundsを支持する。
50件の人工的なexact bookkeepingが通ったが、一般定理をその有限件数で証明したとはしていない。

実行contract前の最小修正は四つ：

1. shot-grid factor-r近似は固定total capのfeasibility保存ではない。n=4→8、cost=8→16、cap=8の反例を保存した。
2. ell/kappaのoutward上端、q/yの実装値、samplerのexact lawまたはlaw-error会計を明示する。
3. 二角度formulaはDelta>0。Delta=0は同一columnへ直接配分する。
4. workspaceはpeak容量で扱い、rare high-workspace columnの期待値でcapacityを判定しない。

原設計書は変更していない。既存source/authorization/result/marker、R0/R0.5境界、R1/R1.5分類も保持した。

## GPT側に返す判断

- 修正した最小single-block modelを採用するか。
- known dictionary/IS/LPの適用に対し、有限confidence・degree-local配分・取得費用・実装保証の統合に論文貢献が残るか。
- 次のpilotが、同じ有限表でprecision-only／whole-mixtureを超えるdegree自由化の追加価値を反証するものとして必要か。
- table取得budget、強いIS/CTS対照、native option、caps、certificateとsource/authorizationの範囲。

「理想LPの成立」から性能・新規性・science GOは導かない。
新angle、eta、precisionの取得もしていない。R1のwinning angle付近を救済する探索へ進まない。
**mandatory STOP。Codexは次実装やR2を開始せず、研究方針・追加検証の必要性／範囲をGPTへ返す。**
