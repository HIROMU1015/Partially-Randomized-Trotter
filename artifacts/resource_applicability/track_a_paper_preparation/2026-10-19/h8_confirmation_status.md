# H8 independent confirmation status

観測日：2026-10-19 JST。**両固定scopeともCOMPLETED**。最新の公開結果commitは**e6eea5387a6e84d1e39a9adb3b61975acc74e200**。GitHubのbranch head／commit treeとローカルGit objectを直接照合した。branch名：track-a-h8-confirmation-results-public-20261019。

| scope | candidate | ε | n | 最新状態 | 保存点判定 |
|---|---|---:|---:|---|---|
| H8_CONFIRM_P8_Q2 | B2_p8_q2_2nd_r4_K4 | .05 | 48 | COMPLETED | POINT_REPRODUCED |
| H8_CONFIRM_P7_Q16 | B2_p7_q16_2nd_r4_K4 | .0005 | 48 | COMPLETED | POINT_REPRODUCED |

## 固定系譜と旧STOP

| 役割 | commit | 証拠 |
|---|---|---|
| 新A探索結果 |64c85f7a833b5950ef2ccae623708b73081edcd6|n8探索、2候補の選択前結果 |
| 独立確認設計 |434e67a8bb205711d22d5bf1aa5d143fd08b7765|固定2候補・各48・固定PF・主副評価・母平均未確定 |
| 承認済み起動snapshot |d21647d07bc86343b292b1e1154ba05f887b1593|2026-10-18 23:35 JSTのRUNNING記録、完了結果ではない |
| 受入修正／並列resume |be2e5475f4dbb5a07e947b053e8c419109f1eda9|2026-10-19 00:05 JSTのRUNNING記録、元STOP・未完了予約保全 |
| 完了した公開結果 |e6eea5387a6e84d1e39a9adb3b61975acc74e200|完了報告・4比較・192軸の公開表 |

元dataset/seed SOURCEは1549b6f891a15ff48379996c34b344118ba2606c、resume実行SOURCEは745a674546f2cb6258526fbfde1d9a00601689f8。旧p8は正常compile済み代表のidentity list/tuple受入不一致でSTOPし、その元STOPを保全して別受入契約で1trajectoryを再利用。p7の旧未完了試行・予約は保持され、seed/eventの再抽選や未完了費用の返却は行わないという元契約である。今回それらの操作を再実施していない。

## 保存科学値の転記

通常signed、倍率1、P=0、元保存shotを使用。exact値とSEは公開CSVからそのまま転記。

| ε | PR平均G exact | paired SE | 固定PF | PF G exact | 削減率（元報告の表示値） |
|---|---|---|---|---|---|
| .05 |9840910801/8|3175066.353534034|B0I_p11_q1_2nd|1267321444|2.935924%|
| .0005 |546493203592571/6|144211830223.71762|B0I_p15_q4_4th|94193016652860|3.302597%|

固定PFは直接実測単価42,268／668,636 RZ（各軸）。公開comparisons.csvには各PRの保存shotもある。副評価は同じ標本の経験的非相殺で、p8固定PFは不適格のため比較null。p7は保存PF G=110353805016120、元報告の点削減率17.463471%。これらを今回計算し直していない。

元報告のevent観測はp8がTaylor0=384、p7がTaylor0=3072、Taylor2/4はどちらも0。未観測費用を0としない。母平均優位、厳密bias、CI／p値、全184候補実測最良は未確定。n48は固定2点の点費用再現であり、新A全domainや中間点の独立確認ではない。

## 今回確認できた証拠と限界

公開完了報告、固定2候補の4比較、192軸表のscope/candidate/replica/axis/cohort・再利用ラベルの構造を確認した。元契約・起動・resumeとの候補・cohort・SOURCE・固定対照・control方針・旧STOPの対応を静的に照合した。

両scopeの正常終了／回収／最終byte一致は**元公開報告の作者確認**として引用する。詳細native receipt・有効grant・seed・control数値raw・詳細hash台帳は公開されておらず、今回の担当がそれらの原本を再検証したと表記しない。公開報告とユーザー提供の完了通知を根拠にCOMPLETEDとしたもので、RUNNING snapshotから完了を推定したものではない。全入力closure・数学的独立性・外部再現の新認定は行わない。

公開から除外された詳細監査・hash・旧履歴文書は内部保全を維持する。今回追加公開せず、未知のprivate保管先を横断検索しない。既存jobへの問い合わせ操作、停止／再開／retry／追加sampling／候補変更は0。

## 論文への接続

Figure 1の独立固定候補確認insetとMain Tableの確認blockへ接続できる。新A n8・旧n32を置換・poolしない。Figure 2への新しい回路比／shot比はMechanism Contractの別承認後だけ。

H6高次PF独立確認は6e197973d4d4183d4e27b1b8cc95cb02f311bb92、docs/research/track_a_h6_higher_pf_confirmation_run01_results_20261019.mdに所在確認済み。prefix5/K4・S2/Yoshida4の独立13/45/45標本という別系列で、補足掲載の採用候補のみ。今回のmain Track Aへ数値を混ぜない。
