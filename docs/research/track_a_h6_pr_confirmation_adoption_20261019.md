# H6独立確認の正式採用・論文証拠更新

2026-10-19。状態：`FORMALLY_ADOPTED_COMPLETE_FIXED_SCOPE`。ユーザーの明示判断に基づき、H6の固定2条件の独立確認を正式採用し、この実行範囲を完了扱いにする。結果commitは `3cccfcb2051dacf29e91473e2e1152acef1574bd`、科学SOURCEは `bf0c26dae4477de5fe7264a7c91e077661ef9b38`。今回の作業は文書への転記と構造検証だけである。

## 採用した結果

linear H6、隣接1 Å、STO-3G、tol-only 1e-8/rank19、T=.8、P=0。二次PR、元の無条件finite-RTE、r4/K2、通常signed bias、allowance倍率1。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional full Hadamard wrapper。主費用は保存解析shot×compiled RZのcosine/sine合計である。

| ε | PR候補 | n（trajectory） | 固定PF対照 | 保存された点費用順位 | paired SE（保存値） | 判定 | PR_CHEAPER |
|---|---|---:|---|---|---:|---|---|
| .05 | B2_p4_q2_2nd_r4_K2 | 48 | B0I_p7_q1_2nd | PRが78.055%高費用 | 4054009.1059934883 | REPRODUCED | false |
| .0005 | B2_p4_q8_2nd_r4_K2 | 104 | B0I_p11_q4_4th | PRが9.644%低費用 | 65539320043.303085 | REPRODUCED | true |

割合はユーザーの採用表示と固定結果報告からの転記であり、exact fractionの新たな小数化ではない。exact平均・PF費用・削減率はmanifest、元の保存値は[固定結果JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cccfcb2051dacf29e91473e2e1152acef1574bd/artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19/result.json)で追跡できる。両PF対照の単価は直接実測済みであり、今回新たにcompileしたものではない。

判定は固定PF G−PR paired平均の符号が事前の期待方向−1/+1と一致すること。原ラベル `REPRODUCED` を維持し、共通プロトコル上の役割を `POINT_REPRODUCED` として併記する。`PR_CHEAPER`とは別項目で、同費用なら `INCONCLUSIVE` とする固定規則を変えない。

## 論文向けの現行索引

[採用bundle](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/README.md)を、既存の論文準備資料に対する今回の更新版とする。

- [論文manifest](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/paper_data_manifest.csv)：既存1,568行をそのまま保持し、H6確認PR・固定PFの4行を追加した1,572行。
- [Claim–Evidence Matrix](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/claim_evidence_matrix.md)：既存CE1–CE6を保持し、CE_H6P_Q2／CE_H6P_Q8を追加した8行。旧6主張への遡及的な一括認定は行わない。
- [証拠台帳v2](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/evidence_ledger_v2.csv)：旧222行を保持し、H6の保存値・判定・native終了・回収・結果/STOP SHAの34行を追加した256行。
- [完了status](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/completion_status.json)、[採用判断](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/adoption_record.json)、[固定証拠catalog](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/evidence_catalog.json)に、SOURCE・入力binding・compiler/runtime・結果・監査を結合した。

新旧の表は同じ行を含むため、両版を連結して標本や証拠の件数を増やしてはならない。現行版を一つ選んで使用する。H6のcohortは `H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1`。q2/n48とq8/n104も別scopeとして保ち、探索n8、旧H6 union n32、条件付きpilot、高次PR別系列、H4/H8の結果・cohortとはpoolしない。cos/sinは同一trajectoryのpaired軸であり、nを2倍にしない。

## 共通プロトコルv1と監査の対応

[共通プロトコルv1](track_a_pr_resource_evaluation_protocol_v1.md)の固定commit `9036d8a87ad1c38e83b9ff654b412dde7041e9d3` を参照した。プロトコル本文、採択判断、旧88行充足表、H4/H8契約は変更していない。その採用時点のH6 Phase 0記録は履歴として保持し、今回の `completion_status.json` が固定2条件のP進捗を更新する。

P_FIXED／P_ROLE／T_NATIVE／T_CLOSE／S_RAREは、今回の2scopeに限った保存証拠の充足としてCOMPLETEを記録する。S_RAREのCOMPLETEはpaired SEと観測/未解決事項を記録した意味で、希少tail解決ではない。P_DESIGNは最終sealのn48/n104固定を確認した状態であり、標本数・power・visibilityを再解析したという意味ではない。

固定result bundleのcatalog・SHAと、最終seal `f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f` の実行契約・入力bindingをGit objectから読取照合した。SOURCEファイルのSHAは保存された34件の対応と照合する。今回、private raw・有効grant・生ledger・runtime実体を再監査したとは扱わない。元の実行終了監査と、今回の文書構造検証を分ける。

保存監査では両scopeともnative exit0、全worker回収、control／SOURCE・入力・compiler/runtimeの最終照合、親/子ledger照合、独立one-shot消費とscopeごとのSTOPがPASS。固定総数152 trajectory・304 compiled軸・3,712 RTE eventを保存している。今回の実行数は0である。結果・grant・ledger・marker・STOPを再作成・書換えせず、固定commitへの参照で結合した。

## 完了範囲と残る限定

有限標本の点順位が事前方向と一致したという、固定2条件の独立P結果を正式採用する。母平均費用順位、全supportを含む厳密CI、全候補最適性、厳密bias/allowance保証は未解決のままである。CI／p値は保存nullを維持する。

K2の次数supportは0/2。保存event数はq2が0次374・2次10、q8が0次3,327・2次1である。特にq8の2次観測1回から、希少component全体の費用や分散が十分に解像されたとはしない。未観測費用は0補完しない。経験的paired SEと母平均保証を区別する。

新cohortに対する定量的費用分解・normalization/headroomの説明検証は今回未評価。過去n8/n32のM_CORE/M_EXPLを新cohortへ移送しない。H4/H6/H8の全入力生成規則の同一性もNOT_VERIFIEDを保持する。H6のP完了をCOMMON_CORE_READY、共通比較の最終採否、最終evidence freeze、分子サイズの因果効果・scaling保証へ一括昇格しない。

これらの限定は採用記録に残し、H6の追加検証の開始条件とはしない。今回の固定範囲は完了である。

## 検証と停止

[構造checker](../../scripts/check_h6_pr_confirmation_adoption.py)と[検証記録](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/validation_summary.json)で、固定Git/SHA・原表の全cell保持・新規行の保存値転記・scope分離・非混合・既存プロトコル保全を確認する。更新版Matrix CSVのみ旧CRLF改行をLFに統一し、内容は全cell一致を確認する。元のCSVは不変。科学moduleはimportせず、新しい科学計算・sampling・compile・統計解析・shot再会計は0。作者自身による技術検証であり、独立科学レビューではない。

[今回のmetadata検査](../../tests/test_h6_pr_confirmation_adoption.py)13件と、既存プロトコルの構造回帰28件がPASS。固定証拠22file、科学SOURCEの保存34 SHA、旧表・プロトコル保全の照合もPASS。誤ったn48流用、REPRODUCEDとPR_CHEAPERの混同、旧unionとのpooling、K4 support移送、新CI・母平均保証・機構説明・最終freezeへの昇格を拒否した。

成果物を独立local commitに保存して[最終mandatory STOP](../../artifacts/resource_applicability/track_a_h6_pr_confirmation_adoption/2026-10-19/STOP.json)。新規資料の公開範囲について今回確認できる承認はないためpushしない。既存公開結果の再pushも行わない。

次の作業：**ユーザー**が新規採用資料の公開要否を判断する。**Codex**はSTOPを維持する。H6追加検証、H4 A50/P、最終evidence freeze、詳細科学レビュー、H4/H8最終統合へ進まない。
