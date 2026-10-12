# N3幅比較v2の一次結果：GitHub公開・remote保存照合

2026-10-13 JST。認可された固定6入力pilot一回の結果、監査、集計、引継ぎを公開した。
**結果取得・技術監査・公開は完了。認可消費済み、Mandatory STOP。次はGPT科学レビューの開始承認待ち。**

## 固定資料

repository：`HIROMU1015/Partially-Randomized-Trotter`。
branch：`n3-width-comparison-v2-execution-20261012`。
結果payload commit：`6d8f769b857e50c8d262e7a4e94f8277a8967597`。
数値source commit：`3eba0814e690e95993db60809c4c3c96784b07c0`。
実行認可commit：`fcc75c9ff442df5368c316fd1cc16c475278f3c8`。

- [一次結果・条件・費用・限界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_results_20261013.md)
- [GPTへの新結果引継ぎ](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_results_review_handoff_20261013.md)
- [全係数・seed・候補mode](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)
- [固定execution plan](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_execution_plan_20261012.json)
- [数学・比較契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_contract_20261012.md)、[結果前設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/docs/research/n3_width_comparison_v2_preregistration_20261012.md)
- [raw出力のSHA索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/run_output_inventory.json)、[run結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1/result.json)
- [独立有理証拠監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/independent_saved_audit.json)
- [全集計JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/summary.json)、[全504 rows CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/mode_method_records.csv)
- [厳密な幅比較区間](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6d8f769b857e50c8d262e7a4e94f8277a8967597/artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/saved_metrics/pairwise_saved_intervals.json)
- [固定generator](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_generator_v2.py)、[固定独立checker](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_check_v2.py)

input SHAは`4c59ba790c7a1faefa6e60dbfbde3d13b718e168bb6491fa71594103b3ac8395`、plan SHAは`dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`。
認可SHAは`1ea7cc662478320db5429ced1481711b3be8863ab72df86774ef0d34fc581c96`。

## GitHub取得と保存照合

push後、GitHubから別bare cacheへfetchし、新exportへ対象blobを展開した。local object alternatesなし。
浅い履歴の境界は残るが、対象payload/source commitの必要blobを取得できた。
[取得台帳](../../artifacts/n3_width_comparison_v2/2026-10-12/results_publication_20261013/retrieval_inventory.json)の59 blob、31,864,153 bytesがlocalと一致。
source13 blobは数値source commitとplan SHAにも一致した。

[remote preflight](../../artifacts/n3_width_comparison_v2/2026-10-12/results_publication_20261013/remote_identity_preflight.json)はidentity/schema/環境/settings PASS。
[remote独立保存監査](../../artifacts/n3_width_comparison_v2/2026-10-12/results_publication_20261013/remote_independent_saved_audit.json)は1224 certificateと36反証recordを再取得データから確認し、local監査と一致。
[remote集計照合](../../artifacts/n3_width_comparison_v2/2026-10-12/results_publication_20261013/remote_saved_metrics_match.json)は公開抽出コードとrawデータから全4集計fileを再作成し、byte一致。
科学runner・producer・solverを再実行せず、科学batch総数1/retry0を維持した。
同じhostでの取得・保存照合であり、CIや第三者の独立科学再現ではない。
[公開receipt](../../artifacts/n3_width_comparison_v2/2026-10-12/results_publication_20261013/receipt.json)に状態を保存した。

## 結果と停止地点

3stress family、full5/6、ν2/3、一外部mode、非負density、候補2mode/H、ε_proj=.01（化学geometry/basis/DF/L_D/PF窓なし）。
NORM/OCC/同値SDP/SOSは全6Hと12候補modeを受理し、OCCの追加H縮約0。
OCCの有限H幅はcheap ORより5/6で狭く、同値SDPより全6Hで狭い幅と短いcold medianを取得した。
固定保存幅の差と取得費用の研究価値をGPTへ戻す。N3継続・中心テーマ・新規性・量子総費用優位をCodexで採択しない。

[GPT/Codex共通指示](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/workflow_intake.md)第3節の
「重要な科学的検証が完了し、新しい知見が得られたとき」に該当する新結果が揃ったため、次はGPTの科学的判断。
レビュー開始は別の利用者承認を待つ。消費済み認可を新科学実行・追加scan・新入力へ流用しない。
Mandatory STOP、N1停止、N2保留、旧primary `NO_ADDITIONAL_ACCEPTANCE_OBSERVED`、旧2-base gate、中心テーマnullを保持。
既存global manifest description欠落によるFAILは残し、今回の取得・個別保存監査PASSと区別する。
