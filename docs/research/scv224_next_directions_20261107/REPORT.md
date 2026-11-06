# O2-SCV224結果の整理・次候補比較

2026-11-07 JST。**静的調査と研究判断材料の作成のみ。正式な方針変更・候補採択・科学的GO/NO-GOなし。**

現行Sの追加取得を保留する根拠は、学習接続が成立してもX追加削減を観測せず、Yの既報点改善が約0.024711%に留まり、学習306,990 RZを含むpoint scoreは両軸とも不利だったこと。
低いfeature出現数と少数評価の不確かさも残るため、SCV全般やfragment群族全体が無効とは結論しない。
これは保留を選ぶ材料であり、本作業から研究方針を正式変更したものではない。

## 成果物と判断上の差分

| 文書 | 主な内容 |
|---|---|
| [SCV224_EVIDENCE_AND_LIMITATIONS.md](SCV224_EVIDENCE_AND_LIMITATIONS.md) | 固定REPORT/summaryの数値を引用。学習成立と改善証拠、h1 raw0と母分散、現群と族全体、追加検証で答えられる問いを区別 |
| [SCV_CONTROL_CANDIDATE_COMPARISON.md](SCV_CONTROL_CANDIDATE_COMPARISON.md) | fragment群・signed support・product一致の層/平均/物理的理由/学習/range/API差分を比較。未取得数値で順位付けしない |
| [SCV_NEXT_DIRECTION_OPTIONS.md](SCV_NEXT_DIRECTION_OPTIONS.md) | 現Sの一区切り・同S限定追加・別候補・SCV全体一時停止の4案と、判断を変える証拠。有限confidence/全費用への不足を対応付け |

signed supportはh1の符号/support情報という別の説明変数を使い得る。product一致はh2/h3の同一signed involutionの恒等化を使えるが、希少層の効果が全targetにどう現れるかが課題。
いずれもknown meanを有限和で表せ、feature取得の追加量子測定は不要。ただし独立trainは有料で、実H4での非定数性・相関・range・費用改善は未検証の部分がある。

SCVを続けるなら、最初に明確にすべき疑問は「B1-OL後のpath依存変動を、このmetadataが独立bitから学べるほど説明し、学習費とrange増を回収する余地があるか」。
現群の標本不足を解消したいのか、別の説明情報を問いたいのかを先に分ける。具体shot数・新seed・実行条件は設定していない。
次の担当は、研究方向の選択は**ユーザー**、採択した問いの最小SOURCE差分・有限scopeの具体化は**SERVER1 Codex**。新科学計算には別途明示許可が必要。

## 実際に確認した固定資料

正式共通指示 `research_project_instructions_unified_2026-10-25.md`（特に§8.1/8.2/8.4/10/10.3）、適用AGENTS、PROJECT_MAP、研究概要全文、既存VALIDATION_STATUS/manifestのSCV完了・未保証status、既存科学契約とSTOPを確認した。

| 固定commit | 確認対象・役割 |
|---|---|
| [ba483ab3038693048e58830e39a676493a892053][result] | 完了REPORT、RESULT_SUMMARY、EXECUTION、SOURCE、FINAL_METADATA_AUDIT。数値引用の正本 |
| [996e434e63ce9b1d14c6ae191649c9abe213cf47][science] | 科学SOURCEとactivation、準備driver再利用・停止条件。変更なし |
| [84e021b6a8decde7fa3ed3fa44d7148cc5428061][design] | INTRASTRATUM_CONTROL_CANDIDATES、STRATIFIED_CV_DESIGN、NEXT_SCIENCE_VALIDATION_DRAFT、REPORTの4文書 |
| [af47cf9161241c8710383ff8539a4a7f8f67492c][finite] | REPORT、SCV_FINITE_CONFIDENCE_MAP、SCV_TOTAL_COST_COMPARISON、SCV_REUSE_AND_VALIDATION_MAP。取得前の設計と実行後の状態を区別 |
| [8d8a290fa4d4cf24f968aea27953ad5422e418b3][theory] | A1対B1-OLの層内定数・補正相殺・分散式 |
| [560c6092bfb5e71af8eedc4a4be5416922764f98][prep] | SCV準備実装・API/テスト記録。scv_scalar/scv224_pilot/a1_control_variates/a1_n3_connectionを静的確認 |
| [3c8e55a2728f86a1eda17a2e0105050a637e7c86][pilot] | 752-shot完了REPORTの過去の範囲・保留結果。旧bitを再解析しない |

RTEComponent/DFDiagonalPauliComponentの既存sign/support、ConditionalSamplerの位置制約、ConditionalBank/FragmentControlの平均、OLS/FrozenCVのrange、paired合成と会計をSOURCE上で確認した。
一般signed featureへindicator専用rangeを流用できず、product一致へh1-only driverをそのまま使えない点を候補比較に明記。
既存一次文献への案内はaf47の固定調査を参照した。今回、新しい文献探索・一次文献再読・重要科学レビューを行ったとは主張しない。

## 未実施事項・保全・STOP

実H4計算・新群探索/平均/moment・保存結果の分散/相関/RZ再計算・旧bitへの再fit・RNG/seed/stream生成・Born・実build/compile・新control実装・test実行・cal/production・break-even数値算出・重要科学レビューは**全て0**。
数値は固定REPORT/summaryからの引用だけ。APIは静的読解のみで、project moduleをimport/実行していない。

独立worktree `/tmp/o2-scv-next-options-20261107`、branch `codex/o2-scv-next-options-20261107`、base=ba483ab3。
公開変更は**本directoryの新しい4 Markdownのみ**。旧科学SOURCE・raw・結果・STOP/grant/ledger/marker、元mainとdirty/untracked、C1凍結を維持する。
今回、研究判断・科学的status・validation pathを採択しないため、既存研究概要・manifest・索引・研究ノートを書き換えない。
並行する可能性があるengine高速化作業のSOURCE/worktree/process/cache/成果物や、共有設定・他ユーザーには操作しない。

byte照合では元main 3,087原本とHEAD/dirty/untracked、旧停止記録1,086原本、前pilot29原本、SCV224 raw530原本、旧752 raw698原本の保全を確認した。
検査は固定資料の所在/byte、引用の転記、既存式・APIとの対応、文書リンクと変更範囲、公開SHA/本文/blob/treeの技術照合だけ。
4 Markdownを公開照合した時点で **mandatory STOP**。現224のSTOPを解除せず、追加shot・新control実装・保証付き検証を自動開始しない。

[result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/REPORT.md
[science]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/996e434e63ce9b1d14c6ae191649c9abe213cf47
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107
[finite]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107
[theory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/A1_VS_B1OL_THEORY.md
[prep]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/scv224_preparation_20261107/SCV_IMPLEMENTATION_REPORT.md
[pilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c8e55a2728f86a1eda17a2e0105050a637e7c86/docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md
