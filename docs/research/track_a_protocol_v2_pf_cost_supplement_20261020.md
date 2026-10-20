# 共通実験プロトコルv2：PF単価置換100 scopeの採用済み補足証拠

2026-10-20。ユーザー指示により、結果commit **`3983fe4f1ede003953735e757d2c098c12b4644f`** の100 scope（重複除外40条件）を条件付き研究成果として正式採用した。
本資料は旧70件補足を置き換える現行版。旧版は同commitのGit履歴に保全され、旧科学結果・監査receiptを遡及更新しない。
[正式v2](track_a_pr_resource_evaluation_protocol_v2.md)の本文・科学条件・受入閾値はbyte不変。詳細科学レビュー・最終evidence freezeは開始しない。

採用する主要値は、最良PF候補変更2/100、q・PF次数変更0/100、最良PF総費用変更50/100、候補全体の順位変更50/100、PR/PF優劣逆転0/100。
両政策とも保存点費用はPR低費用90件、PF低費用10件。100 scopeはdomain・精度・倍率・cohortが重複し、独立100実験ではない。40 endpointも独立した40 sampling実験とは数えない。

以下の条件・結果・captionは採用対象の保存報告から再利用した。今回の新規科学計算・数値解析・再ランキング・再描画は0。

## 条件とbaselineの意味

Linear H6/H8、隣接1 Å、STO-3G、tol-only DF19/28、T=.8、P=0。候補のprefixを登録split L_Dとして維持し、qとδ=.8/qも変更していない。
ε=.05/.01/.005/.001/.0005、allowance倍率1/10。H6はsigned/noncancellationそれぞれold32（保存scope名old_grid_S2）・fine S2 96・S2+truncated S4 136、H8は両biasでold32/strengthened184の元登録集合を使う。全prefix/qは[固定入力](../../artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20/candidate_domains.json)で追跡できる。

費用は、状態準備を課金しないfull cosine/sine Hadamard wrapperのcompiled RZ数に、保存analytic shot数を掛けて足した指標である。
PRは決定論的prefix・確率的RTE tail・full wrapperを含み、H6 union8/H8 union6各n32の保存平均・SE・cohort・選択をそのまま使う。平均費用推定値と、n本のMC trajectoryを実行する古典取得合計費用を混同しない。独立Pのcohortへ結果を移送しない。

比較する元政策はモデルと既存実測の**混在政策**である。純モデルだけを使う反実仮想ではない。ALL_SAVED458の保存単価置換政策をそのまま使用し、今回30 scopeで実際に使用した240単価参照は新450由来184、過去公開8由来4、元実測保持52。未使用単価を評価・補完していない。

[段階A](track_a_h6_noncancellation_stage_a_20261020.md)の120 endpoint（適格34・不適格86・未解決0）を正式受入したユーザー指示に従い、元のunknown120行だけを接続した。
既存712費用は保存値と完全一致。新34適格endpointの費用は今回構成した新baselineで、旧全domain最良値はnullだった。したがって新30のbaseline選択を「旧保存winnerの再現」とは呼ばない。新34自身の単価は元実測保持だが、他候補を含む同じ登録domainで両政策を比較している。

保存scope名は候補集合の識別子であり、old_grid_S2という名称から全候補のPF次数がS2だけだとは解釈しない。各候補の次数は保存候補ID・結果行で区別する。

## 変更件数

下表は[保存summary](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/summary.json)の転記。scope間の候補集合・倍率・精度は重複し、独立した100反復ではない。

| 項目 | 旧70（転記） | 新30（今回） | 全100 |
| --- | ---: | ---: | ---: |
| 最良PF候補の変更 | 2 | 0 | 2 |
| qの変更 | 0 | 0 | 0 |
| prefixの変更 | 2 | 0 | 2 |
| PF次数の変更 | 0 | 0 | 0 |
| 最良PF総費用の変更 | 30 | 20 | 50 |
| PR/PF方向の逆転 | 0 | 0 | 0 |
| 全候補の並び順が変わるscope | 36 | 14 | 50 |
| 候補pairの厳密な順位逆転 | 146 | 34 | 180 |
| 候補pairの同率関係変化 | 8 | 8 | 16 |
| 最小費用候補集合の変更（単独最良も含む） | 2 | 0 | 2 |
| 直接単価側のPR低費用/PF低費用 | 60/10 | 30/0 | 90/10 |

旧70の候補変更2件は、H8 noncancellation、ε=.0005、strengthened184、倍率1/10の`B0I_p16_q4_4th`→`B0I_p15_q4_4th`。q4/S4を保持し、両側の最良集合はそれぞれ単独候補である。集合変更2を新たな同率最良の発生2件とは読まない。
今回H6 noncancellationは全30件でPR点費用がPF点費用を下回った。全候補順位の変化と最良選択の安定性は別であり、非最良候補の順位が変わったことを無視して全順位不変とは主張しない。

新30の精度別最良PFは、倍率1/10とも以下を維持した。

| ε | old32（保存scope名old_grid_S2） | fine S2 96 | S2+truncated S4 136 |
| --- | --- | --- | --- |
| .05 | B0I p15/q1/S2 | B0I p11/q1/S2 | B0I p11/q1/S2 |
| .01 | B0I p15/q2/S2 | B0I p11/q2/S2 | B0I p11/q2/S2 |
| .005 | B0I p15/q4/S2 | B0I p11/q4/S2 | B0I p11/q4/S2 |
| .001 | B0I p15/q8/S2 | B0I p11/q8/S2 | B0I p11/q2/S4 |
| .0005 | B1 p19/q4/S4 | B0I p11/q8/S2 | B0I p11/q4/S4 |

例として新H6 noncancellation ε=.005、倍率1、fine S2 96は、両政策とも`B0I_p11_q4_2nd`。
PF総費用は元`809083243866/5`から直接単価側`162343640082` RZへ変わり、保存PR/PF比は`0.6293373529209746`から`0.6272944314050776`へ変わる。これは旧70のH6 signedで扱う`B0_p11_q4_2nd`とは異なる条件・候補である。
各条件の総費用・比のexact値、保存SE/cohort、候補順位・同率は下記CSV/JSONに保存した。

## 図表・来歴

成果物は[packet](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/README.md)にまとめた。

- [新30条件 CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/selection_comparison30.csv)・[JSON](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/selection_comparison30.json)。
- [100 scope CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/combined_selection100.csv)・[JSON](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/combined_selection100.json)・[Markdown表](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/conditions100.md)。旧70/new30は`numeric_origin`で区別する。
- [候補費用 CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/candidate_totals.csv)・[JSON](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/candidate_totals.json)、[全候補順位 CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/candidate_ranks.csv)、[pair関係変化 CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/rank_relation_changes.csv)。不適格・旧未保存費用のnullは0補完していない。
- [PF費用変化 PNG](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PF_cost_change100.png)・[SVG](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PF_cost_change100.svg)・[PDF](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PF_cost_change100.pdf)。
- [PR/PF比 PNG](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PR_PF_ratios100.png)・[SVG](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PR_PF_ratios100.svg)・[PDF](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/PR_PF_ratios100.pdf)。

**Figure caption — Selected PF costs under saved direct-price replacement.** The historical mixed measured/model price policy is compared with saved direct compiled RZ counts, using unchanged analytic shots and registered candidate sets for linear H6/H8 at 1 Å, STO-3G, tol-only DF19/28, T=0.8 and P=0. The 100 overlapping scopes represent40 system/bias/accuracy/allowance endpoints. Prior70 results are copied; new30 H6 noncancellation results use stage A shots and a newly completed baseline. Two best candidates and50 minimum totals change. Profile/runtime differences, the S4 hypothesis and missing initial-run final gate remain qualifications.

**Figure caption — Conditional PR/PF point-cost directions.** Ten system/bias/candidate-domain panels show unchanged saved H6 union8 or H8 union6 PR costs, each from32 paired trajectories, divided by selected PF costs under the two price policies. The direction does not reverse in the100 scopes: PR is lower in90 and PF in10 under both policies. These are overlapping saved-cohort comparisons, not population-mean or unseen-event guarantees. Lines connect the five registered accuracy endpoints and do not represent newly evaluated accuracies.

元70の科学SOURCEは`19f2c5a9409db69fd8bd6500eb4228fefa00145c`、Stage Aは`85e250302d15b427731a50ddbe321fc280ec5f1a`、新30は今回SOURCEに対応する。
containerのSOURCEは新しい取りまとめ来歴であり、旧70を新SOURCEで再解析した意味ではない。[証拠manifest](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/result_manifest.json)とsealed contractに元入力・結果・図表SHAを保存した。
旧70の`summary.all_scopes`と新summaryの`previous70`は完全一致。元70行もorigin tagを除き完全一致し、費用・比・順位・PR統計の再計算は0。

## 今回の証拠統合と主張境界

今回GitHubの固定SOURCEと結果を確認し、58対象pathの公開blobと固定ローカルbytesを照合した。保存summary・終了監査・native exit evidenceは直接GETでも一致した。
新30の保存監査は1492費用/30 scope/41578 pair照合、37人工・回帰test、native exit0・同一親wait4回収・残worker0を記録する。今回それらの科学処理を再実行していない。新しい検査は文書・literal・構造・SHA・公開範囲だけで、別担当による独立科学レビューではない。

Stage Aで新しく確定した34適格endpointは新baseline来歴である。旧全domain最良値はnullだったので、旧保存winnerの再現と呼ばない。旧Dのtruthinessによるunresolved空配列も解決証拠としない。Stage A/Bの新結果で欠測を解消し、旧SOURCE・raw・grant・ledger・marker・STOPは保全した。

H6 profile差、H8/model runtime同一性未証明、S4合成仮説、初回572軸の最終gate欠測、旧profile照合待ち48軸を維持する。
この結果はモデル推定と実測の混在政策から保存直接単価へ置換した条件付き診断であり、純モデル反実仮想や、実行環境差を除いたモデル固有誤差だけの検証とは主張しない。
経験的bias/allowance・保存shot政策・登録集合内の点比較に限る。厳密bias certificate、全PF集合の大域最適性、PR母平均・未観測希少event保証、実quantum shots・FT T資源・noise/backend・energy/RPE全round総費用へ拡張しない。
平均費用の推定値とMC trajectory n本の古典取得合計費用を区別し、独立Pや他のcohortへ移送しない。

現行の[論文manifest](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/paper_manifest.json)、[CEM](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/claim_evidence_matrix.csv)、[証拠台帳v4](../../artifacts/resource_applicability/track_a_pf_q_model_evidence_integration/2026-10-20/evidence_ledger_v4.csv)に結果・SOURCE・図表・SHAを対応させた。
旧CEM14行・台帳316行は保持し、旧CE5の欠測記述は歴史的状態として新overlayで限定更新する。論文主表・Figure2・旧図・原データ・正式v2本文と科学条件は不変。
[統合・公開監査報告](track_a_pf_q_model_evidence_integration_20261020.md)に内部パスと公開承認記録の要確認事項を記録した。研究成果の正式採用は新規公開承認・詳細科学レビュー・最終freezeの認定とは別である。

**mandatory STOP。**
