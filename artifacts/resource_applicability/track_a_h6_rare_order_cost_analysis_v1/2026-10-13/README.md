# H6希少次数の費用寄与：保存データ解析

[主報告と追加情報の未承認案](../../../../docs/research/track_a_h6_rare_order_cost_reliability_v1.md)、[H4/H6/H8対応](../../../../docs/research/track_a_h4_h6_h8_comparison_readiness_20261013.md)。
linear H6/1Å/STO-3G/tol-only DF19/sector400/T.8、prefix5/10/15/full19・delta.1–.8、P0/経験的u。
探索8と確認32を分離。事前K4 winnerを固定した確認でもPR/PF比約.460/.400/.375/.387。
次数確率は既知だが、未観測full-wrapper費用のpopulation上界は未証明。未知平均はnull。

- [費用層・全8構成・4精度・逆転閾値・仮定付き感度](cost_reliability_analysis_v1.json)
- [選択320本のorder/component/basis variant/full-wrapper費用最小抽出](selected_trajectory_cost_evidence_v1.json)
- [全832本の次数・RZ・seed/source/byte SHA CSV](trajectory_metrics_v1.csv)
- [(n2,n4)層のmultinomial確率（保存binary64 lawの算術）とnative構造bound](order_histogram_probability_v1.csv)
- [凍結component確率/catalog・native構造会計、array復元なし](component_and_structure_v1.json)
- [別q/rのorder4探索回路1本：参考のみ](other_cell_order4_evidence_v1.json)
- [保存探索/確認lawの丸め差](saved_law_roundoff_comparison_v1.json)
- [最小追加情報・call予算の未承認案](additional_information_proposal_v1.json)
- [解析来歴・元結果/source不変・新science0監査](analysis_audit_v1.json)
- [公開抽出からの別算術検査](public_consistency_v1.json)
- [保存JSON限定解析script](analyze_saved_cost.py)、[公開scalar検査script](verify_public_analysis.py)
- [10人工scalar tests](../../../../tests/tracks/resource_applicability/test_h6_rare_order_cost_analysis_v1.py)、[local結果](limited_tests_v1.txt)
- [解析commit82028755・remote ref/tree/27fileの取得照合](publication_receipt_v1.json)
- [固定した全軽量fileのSHA索引](artifact_inventory_v1.json)

主に欠測しているq8/r2/K2のnonzero層は確率1.67%。zero層平均を保存点推定に固定すると、PFに追い付くにはrare条件付き費用約9.19百万RZ/shot（zeroの95.86倍）が必要。
観測rare費用はこの水準を示していないが、上界証明や母平均winner認定ではない。
現sourceのinstruction guardや部品和をcompiler後RZ上界としない。

今回の新trajectory/compile/signal/DF/state/H8/GPU/grant0。科学source/既存結果/venv/H4/共有環境の変更0。
metadata-only source/tests/docsを固定し、公開取得照合後mandatory STOP。最小追加取得8trajectory/16compile等は未承認、allowed_cpus=[]、runtime_authorization=false。
元全raw/NPZ/cacheは追加しない。公開最小抽出から確率と費用感度を再計算できるが、分子・全compileの再現やCIのPASSではない。
