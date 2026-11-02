> 公開用projection。private内部パス、原入力・回路・運用記録、ローカル補助scriptは含まない。本文に記す照合件数は原調査の報告値であり、未公開原資料を再公開した意味ではない。

# E：Claim–Evidence Matrix

A〜Dの完了報告を取り込んだ最終版。これは既存証拠と主張候補の対応表であり、科学的な正式採択・母平均保証・数学的認定ではない。数値は既存報告から引用し、費用・比・shot・順位を新規計算していない。

完全な各列は[CSV](claim_evidence_matrix.csv)／[JSON](claim_evidence_matrix.json)。固定commit・result file・SOURCE・条件・実測/model・探索/確認・未解決・証拠の強さを全行に持つ。SOURCE欄に「報告」としたIDは原報告の来歴引用であり、その全SOURCE closureを今回読了・再現した意味ではない。実取得・使用したfileは[統合hash一覧](../source_inventory_public.csv)と各担当manifestで区別する。

|ID|主張候補|証拠の強さ／限定|
|---|---|---|
|CE1|決定論側の比較対象を強化するとPRの点費用利益が縮小する|対象登録集合内の保存有限標本点比較に直接対応。普遍的なPR非優位や大域最適性は支持しない。|
|CE2|PR側の探索domain拡張で、特にH8両端の点費用の結論が変わる|保存点の符号変更は直接対応。独立再現・期待値優位・境界外の改善を支持しない。|
|CE3|中間精度でのPR点費用削減は強化した比較集合でも残る|保存された中間点削減の記述には対応。厳密rare頑健性・連続精度窓・全域利益・20–35%という一律新A削減率は未認定。|
|CE4|回路費用とshot負担のトレードオフが勝敗を左右する|会計式・実装との構造的対応は確認。新しい定量機構比較や角度別FT費用の証拠ではない。|
|CE5|signed／経験的非相殺の会計区分が結果に影響する|H8の保存会計差は直接対応。H6はsubset限定の引用で、全136候補の非相殺優位という解釈は不成立。|
|CE6|結果は有限時間signal・保存DF・P=0・Qiskit compiled RZ費用に限られる|適用範囲・否定すべき一般化の直接的な来歴対応。論文採否や新しい科学的認定を行わない。|

## CE1：決定論側の比較対象を強化するとPRの点費用利益が縮小する

- **固定commit**：64c85f7a833b5950ef2ccae623708b73081edcd6;fc6bd039dc7171d5674ac04f0b3ad427f1b6495b
- **結果ファイル**：docs/research/track_a_v1_reaccount_results_20261016.md;artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json
- **SOURCE**：V1報告SOURCE2261a3ad1d6c23521474b31204572a6c3a51258b; D実行31ef62a12407010e9c1a07f9b37facedb8dff5fe (H8修正8ff002dd9524b49284fd8fb8e11354ee1ec18a00)
- **評価条件**：H6 DF19・T=.8・P=0・通常bias・factor1・ε=.05/.01/.005/.001/.0005、旧32→S2強化96→S2/S4強化136。H8は旧32/強化184の保存比較を別記。
- **既存証拠の引用**：H6元V1報告のε=.01は旧49.69%→強化30.89%、ε=.005は53.27%→34.97%。ε=.05の元model比較29.35%→−13.04%は後のp7実測反映−13.2901%と時点を分ける。すべて保存表示値の引用。
- **実測／model**：PR paired実測平均。決定論は保存実測とmodelが混在。元V1の選択点と後続Dの直接測定を遡及混同しない。
- **探索／確認**：H6確認union8/n32、H8確認union6/n32。union内のwinner選択は選択後点比較。
- **未解決事項**：全候補直接実測・母平均最良・モデル誤差の全domain保証なし。H4は比較した実測subset16に限定。
- **証拠の強さ**：対象登録集合内の保存有限標本点比較に直接対応。普遍的なPR非優位や大域最適性は支持しない。

技術調査：[B_domain](../B_domain/report.md)、[D_h6_lineage](../D_h6_lineage/report.md)

## CE2：PR側の探索domain拡張で、特にH8両端の点費用の結論が変わる

- **固定commit**：64c85f7a833b5950ef2ccae623708b73081edcd6;434e67a8bb205711d22d5bf1aa5d143fd08b7765
- **結果ファイル**：artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv;artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/new_pr_cost_points.csv;docs/research/track_a_pr_new_a_total_cost_results_20261018.md;docs/research/track_a_h8_new_pr_confirmation_design_20261018.md
- **SOURCE**：新A会計8601b921342fc1cc83780a830e1f65bf64fe9e93; 取得31ef62a12407010e9c1a07f9b37facedb8dff5fe/87f1f0d528478ad212ba98159db06eec3c78433e。確認設計は結果ではない。
- **評価条件**：H8 linear1Å/STO-3G/tol-onlyDF28/T=.8/P=0、通常bias・factor1、強化184対照。新Aは疎な16候補、prefix6/7/8と登録q/r/K。
- **既存証拠の引用**：ε=.05：旧PR−8.0230%、新A p8/q2/r4/K4は+3.1084%。ε=.0005：旧PR−50.8125%、新A p7/q16/r4/K4は+3.1178%。保存報告からの引用であり新計算ではない。
- **実測／model**：新旧PRはcompiled単価の有限標本平均。対応する2点の選択PF単価は直接実測、184集合の他候補にはmodel残存。
- **探索／確認**：新A探索n8対旧確認union6/n32を分離、poolingなし。固定434eのH8確認は設計であって確認結果ではない。
- **未解決事項**：探索選択効果、rare tail、H8原profile bytes未確認。p7q16はq/r schedule端でr固定q走査ではない。母平均で符号が変わることは未認定。
- **証拠の強さ**：保存点の符号変更は直接対応。独立再現・期待値優位・境界外の改善を支持しない。

技術調査：[B_domain](../B_domain/report.md)、[A_rare_event](../A_rare_event/report.md)、[C_nonclifford](../C_nonclifford/report.md)

## CE3：中間精度でのPR点費用削減は強化した比較集合でも残る

- **固定commit**：64c85f7a833b5950ef2ccae623708b73081edcd6;fc6bd039dc7171d5674ac04f0b3ad427f1b6495b;cf8e93eef50333bef41f6d75803f6b819d679343
- **結果ファイル**：artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv;docs/research/track_a_pr_new_a_total_cost_results_20261018.md;docs/research/track_a_h8_rare_sensitivity_selection_run01_results_20261018.md
- **SOURCE**：新A会計8601b921342fc1cc83780a830e1f65bf64fe9e93; D31ef62a12407010e9c1a07f9b37facedb8dff5fe/8ff002dd9524b49284fd8fb8e11354ee1ec18a00; rare/selection3c0c9546701fd953de3b299be561344375c1a78b
- **評価条件**：主にH6 DF19/H8 DF28、linear1Å/STO-3G/T=.8/P=0、通常bias・factor1・ε=.01/.005/.001、強化136/184。
- **既存証拠の引用**：旧PR保存点はH6 30.8905/34.9653/31.1677%、H8 33.2373/31.4549/23.1282%。新Aを含む保存表でも旧候補の値は保持される。新たなunion winnerは選択していない。
- **実測／model**：PR n32 compiled費用平均。H6中間精度の選択PF単価にmodel、H8最新通常factor1の選択PFは直接実測。比較集合全体はmodelを含む。
- **探索／確認**：旧確認n32を新A探索n8と分離。確認union内の点選択を母平均winner認定と呼ばない。
- **未解決事項**：L2全primitive展開/数値TierR未完了、未観測tail、selection-adjusted保証なし。中間winnerにもp/K/次数別prefix境界があり、一括内部判定は不成立。
- **証拠の強さ**：保存された中間点削減の記述には対応。厳密rare頑健性・連続精度窓・全域利益・20–35%という一律新A削減率は未認定。

技術調査：[A_rare_event](../A_rare_event/report.md)、[B_domain](../B_domain/report.md)

## CE4：回路費用とshot負担のトレードオフが勝敗を左右する

- **固定commit**：64c85f7a833b5950ef2ccae623708b73081edcd6;343e0c591d297f99174459d750206b01bf7b9412
- **結果ファイル**：docs/research/track_a_pr_new_a_total_cost_readiness_20261018.md;docs/research/pr_alpha_sensitivity_run01_results_20261018.md;artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/new_pr_cost_points.csv
- **SOURCE**：31ef62a12407010e9c1a07f9b37facedb8dff5fe:src/trotterlib/pr_cost_integration.py;8601b921342fc1cc83780a830e1f65bf64fe9e93:scripts/pr_new_a_accounting.py; α結果報告SOURCE93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb
- **評価条件**：保存G_i=N_real*C_cosine+N_imag*C_sine、headroom ε/√2−bias−allowance、normalizationを含む保存shot会計。α_axisと費用標本SEは別物。
- **既存証拠の引用**：新しい回路比・shot比・B²・headroom比は算出しない。既存の保存G、shot定義と費用結合SOURCEの対応を記録した。レビューの新A回路比推定を実測に昇格しない。
- **実測／model**：回路単価はPR実測標本/PF実測またはmodel。shotは理想fresh-IID等の仮定付き解析会計で、量子測定の実測ではない。
- **探索／確認**：Gの結合式は探索n8と確認n32それぞれへ適用するが、cohortは統合しない。
- **未解決事項**：機構図の比の正式な新算出は別承認。常にPRの1回路が小さい・因果寄与を新たに定量分解したという主張は支持しない。
- **証拠の強さ**：会計式・実装との構造的対応は確認。新しい定量機構比較や角度別FT費用の証拠ではない。

技術調査：[C_nonclifford](../C_nonclifford/report.md)、[D_h6_lineage](../D_h6_lineage/report.md)

## CE5：signed／経験的非相殺の会計区分が結果に影響する

- **固定commit**：fc6bd039dc7171d5674ac04f0b3ad427f1b6495b;343e0c591d297f99174459d750206b01bf7b9412;64c85f7a833b5950ef2ccae623708b73081edcd6
- **結果ファイル**：artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json;docs/research/track_a_pr_cost_batch_completed_results_20261018.md;artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/comparisons.csv;artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B2_input_binding_UNAPPROVED.json
- **SOURCE**：D31ef62a12407010e9c1a07f9b37facedb8dff5fe (H8修正8ff002dd9524b49284fd8fb8e11354ee1ec18a00); 新A8601b921342fc1cc83780a830e1f65bf64fe9e93; α結果報告93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb
- **評価条件**：同じ保存Hamiltonian/stateと比較集合でbias会計を区別。通常bias・非相殺を厳密bias証明書と同一視しない。factor1/10も分離。
- **既存証拠の引用**：保存DのH8 ε=.05/factor1/強化184：通常−8.0230%、非相殺16.6243%。H6の22.2791/32.2240/37.0663/36.2830/20.1071%は利用可能subsetの保存点値に限る。
- **実測／model**：PR有限標本実測平均、PF実測/model混在。H6 subsetの最初の4点はmodel、.0005は直接単価。
- **探索／確認**：Dは旧n32、新Aは別n8。α欠測60入力は12旧B0I候補×5ε、3α180出力。新A倍率10欠測320endpointとは別。
- **未解決事項**：H6全登録非相殺のPR/PF比はnull。旧D unresolved=[]はeligible:nullのtruthinessによる検出漏れで、full-domain完了証拠にはならない。
- **証拠の強さ**：H8の保存会計差は直接対応。H6はsubset限定の引用で、全136候補の非相殺優位という解釈は不成立。

技術調査：[D_h6_lineage](../D_h6_lineage/report.md)、[B_domain](../B_domain/report.md)

## CE6：結果は有限時間signal・保存DF・P=0・Qiskit compiled RZ費用に限られる

- **固定commit**：64c85f7a833b5950ef2ccae623708b73081edcd6;343e0c591d297f99174459d750206b01bf7b9412;1fc58aa3f5568887e0678a9be939551b0ed6523d;434e67a8bb205711d22d5bf1aa5d143fd08b7765;cf8e93eef50333bef41f6d75803f6b819d679343
- **結果ファイル**：docs/research/track_a_pr_new_a_total_cost_results_20261018.md;docs/research/pr_alpha_sensitivity_run01_results_20261018.md;docs/research/track_a_h46_higher_pf_integrated_run01_results_20261018.md;docs/research/track_a_h8_new_pr_confirmation_design_20261018.md
- **SOURCE**：各固定結果の報告SOURCEを上記CE1–5と証拠索引で区別。高次PR統合報告SOURCEa65e4f6e7bcc83f90a054f6aafc4ee2e0ca5d80a、取得d77e439a16d6953cb17e6d1ef91b0d9430144848は別系列。
- **評価条件**：主panelはlinear H4/H6/H8・1Å・STO-3G・T=.8、H4legacyDF12/H6tol-onlyDF19/H8DF28、保存state、P=0、full-wrapper RZ、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。
- **既存証拠の引用**：新A478有限点/162不適格/320欠測は保存status。completeは入力完全性で科学保証ではない。H8の確認設計、高次PRの探索結果は主panelの独立確認完了証拠ではない。
- **実測／model**：RZ命令数はClifford角も含む。非Clifford/FT T資源、state preparation、noise/backendは未評価。PFモデル単価も残る。
- **探索／確認**：H4旧n32はfixed anchor選択/推定。H6/H8確認unionと新A探索n8、高次PR探索n8は別系列。
- **未解決事項**：保存DFから元電子Hamiltonianへの誤差、長時間/新geometry/energy/RPE全round、母平均最適性、非Clifford感度、H8原profile bytes。
- **証拠の強さ**：適用範囲・否定すべき一般化の直接的な来歴対応。論文採否や新しい科学的認定を行わない。

技術調査：[A_rare_event](../A_rare_event/report.md)、[B_domain](../B_domain/report.md)、[C_nonclifford](../C_nonclifford/report.md)、[D_h6_lineage](../D_h6_lineage/report.md)

## 論文へ転記するときの共通条件

各数値に系・geometry・basis・DF政策/rank・prefix・q/r/K/外側次数・T/δ・bias・allowance・cohort・単価来歴を付ける。Bの境界属性は座標または登録scheduleの記述であり、大域最適性や境界外の改善の証明ではない。AのL1を確認しても、未確認L2と未計算Tier Rの数値をPASSに変えない。Cのcompile前QPYを最終compiled回路の角度証拠にしない。Dのsubset値でH6 full-domain欠測を埋めない。

**次の作業：ユーザーが主張の採否・限定と執筆開始を判断する。Codexは統合報告・最終hash検査後mandatory STOPし、新計算・レビュー・次stageを起動しない。**
