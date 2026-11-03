# O1原稿担当向け：H8の主張・証拠・限定の最新追補

2026-11-03 JST。旧O1資料C／H468比較資料Bに重ねる後続証拠メモ。原稿本文・新図表・完成captionは作成していない。正式採択・独立レビューでもない。科学的な点結果は変わらず、G04の匿名終了metadataへのアクセスが更新された。[旧構成][C_OUTLINE]、[旧配置][B_PAPER]、[新6ファイル][A_REPORT]

## 1. 使う比較IDと固定結果

対象はlinear H8、1.00 Å、STO-3G、tol-only1e-8 DF rank28、保存H/state/reference、T=.8、P0、通常signed／allowance1、S2 partial＋無条件finite RTE。以下の2確認だけを主表へ結ぶ。

| 主表比較ID | 元scope | 固定候補と対照 | 保存条件 | aliasと原結果selector |
|---|---|---|---|---|
| H8_DF28_P_1 | H8_CONFIRM_P8_Q2 | B2_p8_q2_2nd_r4_K4 vs B0I_p11_q1_2nd | ε.05、δ.4、n48 | H8_CONFIRM_1。元comparisons.csvのprimary data record1、主表comparison_id、Figure2B contrast_id |
| H8_DF28_P_2 | H8_CONFIRM_P7_Q16 | B2_p7_q16_2nd_r4_K4 vs B0I_p15_q4_4th | ε.0005、δ.05、n48 | H8_CONFIRM_3。元comparisons.csvのprimary data record3、主表comparison_id、Figure2B contrast_id |

固定URL：[元比較表][ORIGINAL_COMPARISONS]、[元軸価格][ORIGINAL_AXES]、[主表JSON][MAIN]、[Mechanism metadata][MECHANISM]、[Figure2B保存値][FIGURE2B]、[最新二条件対応][A_EVIDENCE]。

保存N/G/SD/paired-SE／方向はこれらの原fieldを引用する。再計算・新rankingは不要。p8副評価のPF不適格／null、CI・p値null、未知tailも保持する。主表2行とMechanism／Figure2Bは同じ標本の再引用であり、別追試に数えない。

## 2. 本文・Methods・SI・captionへ渡す要件

| 配置 | 役割・使える証拠 | 必ず同時に明示する限定 |
|---|---|---|
| 本文：H4/H6/H8固定2精度比較 | A08。H8の固定候補は別価格確認で両方POINT_REPRODUCED・保存PR低点費用 | 固定n48・P0・有限価格標本の点方向。母平均優位、外部再現、全候補最良ではない。H4粗の非再現・H6粗のPF低費用方向も残す |
| 本文：強いPFと適用条件 | A07。既存強化184の通常signed／非相殺policy・実測/model・候補域の違い | 固定主表n48と旧union6/n32は別候補・別価格cohort。主表の正方向で強化184の負方向を消さない。新G04は184全候補を実測化しない |
| Methods：input/SOURCE | 固定design、SOURCE登録、保存H/state/snapshot／preparation identity | one-body保持、DF順序、scalar一回、reference不回転。三系の完全共通生成規則は未認定。p8 export flagは未解決 |
| Methods：回路・測定・費用 | Qiskit1.3.0／opt1／seed17／rz,sx,x,cx、ordinary ancilla／symmetric-directional、測定込みfull wrapper | 状態準備を除くcompiled RZ。Nは解析shots、nは古典価格標本。元profile bytes・旧compiler出力完全一致・全law精度保証を意味しない |
| SI：固定確認・終了監査 | 最新COMPLETION_AUDIT、192軸provenance、元価格表 | 作者の匿名metadata監査。private raw receipt全体の第三者再実行ではない。保存hashを再compile一致や完全復元の証拠にしない |
| SI：cohort・resume | p8 replica0再利用、修正SOURCEの残47、p7の48、限定seed/index監査 | 全192軸を新compileとしない。探索／旧確認／条件付きpilotをpoolしない。旧STOP・中断・予約を保持。2026-10-20派生seed監査は保存receiptの引用 |
| SI：希少RTEと残余感度 | A13。正式終了済みS0 n48／S2_SINGLE条件付きn43、残余感度 | 無条件91本に合算しない。S2_MULTI/S4の費用・分散は未知。sampling再開を原稿の必須条件にしない |
| Caption：固定比較の注記 | 比較ID、候補、ε、T、DF28、n、P0、保存点・paired-SE、固定PF直接価格 | Mechanism／Figure2Bと同cohort。技術metadataの追加と科学方向の変更を混同しない。p8制約はMethods／SIへの明示参照を付ける |
| Caption：強化対照・時間 | 184対照はT.8の独立した保存比較域、長時間研究はH4/H6 | H8の高次決定論PFから高次PR完了を推論しない。「三系の長時間検証」などへ広げない |

出典：[終了監査][A_AUDIT]、[192軸provenance][A_AXES]、[固定design][DESIGN]、[回路・入力方法][METHODS]、[入力例外][EXCEPTIONS]、[強化184][STRONG]、[希少層の正式終了][CLOSURE]。

## 3. SOURCE・結果・監査commitの使い分け

- 科学実行SOURCE：`745a674546f2cb6258526fbfde1d9a00601689f8`。[driver][SOURCE_02]、[paired compile実装][SOURCE_05]、[resume受入][SOURCE_06]。SOURCEの存在・hash一致と実行互換性／科学保証は別。
- 元dataset/seed SOURCE：`1549b6f891a15ff48379996c34b344118ba2606c`。p8 replica0のmeasurement_SOURCEはこの旧版で、後続94軸・p7の96軸は修正SOURCE。列値を一つのSOURCEへ正規化しない。
- 元結果公開：`e6eea5387a6e84d1e39a9adb3b61975acc74e200`。[通常・副評価結果][ORIGINAL_COMPARISONS]と[軸価格][ORIGINAL_AXES]を数値の原参照にする。
- 今回利用する監査projection：`37694741ee356fd09b5dfcb56130c55ab381044c`。6ファイル版が監査の最新入口。科学SOURCEや新しい実験のcommitではない。

H/state/snapshot等のidentifier、privateな原result/native/terminal等のhashは[COMPLETION_AUDIT][A_AUDIT]を引用する。LOCALが未公開原本を取得・全hash照合したという書き方は避ける。公開SOURCE11ファイルの固定bytes照合と、private原本の作者照合記録は別の検査範囲である。

## 4. 旧論文化資料との対応

| 旧資料・欄 | 今回加える後続情報 | 変更しないもの |
|---|---|---|
| C CLAIM_EVIDENCE_MAP A08、C STATUS H8詳細audit | 匿名終了metadataと192軸へアクセス可能。G04追補statusと固定commitを併記 | phase3_original_*、USER_FORMAL_ADOPTION=UNKNOWN、同cohort alias、有限点結果 |
| C A07／O1強対照節 | 固定主表の監査入口を補強。184域との区別を維持 | 既存の正負方向・一部model価格・candidate集合 |
| C A13／希少層SI | 元n48確認の終了監査と、層別感度の別scopeを明示 | 正式採択済み限定scope・STOP・未知残余 |
| B REPORT／PLAN／PAPERの「G04未取得」 | 当時の記録として保存し、本追補を現在の入口にする | H8長時間・高次PR未検証、新H8計算は現O1に不要という判断 |
| 旧Phase3 NOT_ACCESSIBLE | 匿名metadataのアクセス不足解消を別欄で追加 | 元coverage行・G04分類／判断・private rawによる再現認定の未保証 |

本ディレクトリは科学的な採用状態や元文書を上書きしない追補である。原稿担当はC／Bの条件を使い、監査の現在値だけ本追補とAの6ファイルへ戻る。[追補CSV](H468_COVERAGE_ADDENDUM.csv)、[残存gap](STATUS_AND_REMAINING_GAPS.md)

## 5. p8の入力制約と次の担当

`saved_exact_export_match=false` は保存比較flagとしてUNRESOLVED。固定design／preparation hash一致でtrueへ変更しない。比較対象・原因・科学的同等性は今回確定していない。p7の保存trueからp8へ同一性を移送することもしない。

**現在のO1に新しいH8科学計算は必須ではない。** 本追補で長時間・高次PRの検証を追加したわけではなく、新計算案も追加していない。次の担当はユーザー：この限定的な証拠配置を正式採用し、原稿作成を別途指示するか判断する。Codexは公開・照合後mandatory STOP。原稿本文・図表・詳細科学レビュー・科学実行へ自動進行しない。

<!-- fixed-reference-definitions -->
[A_AUDIT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/COMPLETION_AUDIT.json>
[A_AXES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/AXIS_PROVENANCE.csv>
[A_EVIDENCE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/H8_CONFIRMATION_EVIDENCE.csv>
[A_REPORT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37694741ee356fd09b5dfcb56130c55ab381044c/docs/research/h8_g04_evidence_reconciliation_20261103/REPORT.md>
[B_PAPER]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/713c45f84c8d7a41ceaef93381bd083e22189085/docs/research/h8_h46_validation_coverage_gap_20261103/PAPER_IMPLICATIONS.md>
[CLOSURE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md>
[C_OUTLINE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/PAPER_STORYLINE_OUTLINE.md>
[DESIGN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json>
[EXCEPTIONS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/input_exceptions.md>
[FIGURE2B]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb33de67475e7d33eed56801b74a2a277147e4be/artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/plot_values.csv>
[MAIN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json>
[MECHANISM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ff6ac0b6ffed6231dac5f655275b620598d0201/artifacts/resource_applicability/track_a_mechanism_results/2026-10-19/comparison_metadata.csv>
[METHODS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_display_methods_20261019.md>
[ORIGINAL_AXES]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/paired_metrics.csv>
[ORIGINAL_COMPARISONS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/comparisons.csv>
[SOURCE_02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation.py>
[SOURCE_05]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation_parallel.py>
[SOURCE_06]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/prepare_h8_pr_confirmation_resume.py>
[STRONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md>
