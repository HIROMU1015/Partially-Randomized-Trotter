# H6 noncancellation段階B：保存修正・新契約one-shot完了

2026-10-20。停止commit `c585cef8990ef28f4d43c13dfaf2a0d423fda5a7`から、新SOURCE **`c0d8ca8ecd3fc29528ca8424f8b8d953d06eff91`** で承認済み30 scopeを1回実行した。
専用branchは `codex/h6-noncancellation-stage-b-rerun-20261020`。

**新30 scopeの最良PF候補・q・prefix・PF次数・PR/PF点費用の方向は、元の混在単価から直接実測単価への置換によって変わらなかった。最良PF総費用は20/30件で変化した。**
旧70件を再計算せず参照した100 scope全体では、最良候補変更2、q/PF次数変更0、最良総費用変更50、PR/PF方向逆転0。保存点費用は両政策ともPR低費用90/PF低費用10である。これは重複を含む100 scope、40 system/bias/ε/倍率 endpointの**条件付き比較**である。

## 条件とbaselineの意味

Linear H6/H8、隣接1 Å、STO-3G、tol-only DF19/28、T=.8、P=0。候補のprefixを登録split L_Dとして維持し、qとδ=.8/qも変更していない。
ε=.05/.01/.005/.001/.0005、allowance倍率1/10。H6はsigned/noncancellationそれぞれold S2 32・fine S2 96・S2+truncated S4 136、H8は両biasでold32/strengthened184の元登録集合を使う。全prefix/qは[固定入力](../../artifacts/resource_applicability/track_a_pf_selection_comparison_preparation/2026-10-20/candidate_domains.json)で追跡できる。

費用は、状態準備を課金しないfull cosine/sine Hadamard wrapperのcompiled RZ数に、保存analytic shot数を掛けて足した指標である。
PRは決定論的prefix・確率的RTE tail・full wrapperを含み、H6 union8/H8 union6各n32の保存平均・SE・cohort・選択をそのまま使う。平均費用推定値と、n本のMC trajectoryを実行する古典取得合計費用を混同しない。独立Pのcohortへ結果を移送しない。

比較する元政策はモデルと既存実測の**混在政策**である。純モデルだけを使う反実仮想ではない。ALL_SAVED458の保存単価置換政策をそのまま使用し、今回30 scopeで実際に使用した240単価参照は新450由来184、過去公開8由来4、元実測保持52。未使用単価を評価・補完していない。

[段階A](track_a_h6_noncancellation_stage_a_20261020.md)の120 endpoint（適格34・不適格86・未解決0）を正式受入したユーザー指示に従い、元のunknown120行だけを接続した。
既存712費用は保存値と完全一致。新34適格endpointの費用は今回構成した新baselineで、旧全domain最良値はnullだった。したがって新30のbaseline選択を「旧保存winnerの再現」とは呼ばない。新34自身の単価は元実測保持だが、他候補を含む同じ登録domainで両政策を比較している。

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

| ε | old S2 32 | fine S2 96 | S2+truncated S4 136 |
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

## 修正範囲・契約・検査

停止した旧[報告](track_a_h6_noncancellation_stage_b_20261020.md)・SOURCE224fbba0・部分結果・消費済みgrant・ledger・marker・STOPは変更せず、新runner/checker/plotterを追加した。
修正は全出力mappingのSOURCE一致確認とkey重複のないmerge、および全JSON/CSVの事前シリアライズである。不一致や非finite値があれば保存前に拒否する。
科学core3fileは旧SHAと同一。旧runnerとのAST照合でもmetadataと数値core呼出しは同一である。
全出力保存・旧エラー再現・SOURCE不一致・null・欠落・非finite・空CSV・旧データ非変更を含む新10件、既存Stage B14件、比較回帰13件の**計37件PASS**。実保存値の科学実行はテスト時0。

新契約はSOURCE9file・入力13file・算術runtime86file・旧112証拠fileを封印した。旧契約との入力SHA・30条件・科学call予算・定義・禁止callの完全一致もseal時に確認した。
新grantのみを1回消費し、旧grant再利用・自動retryは0。

| 処理 | 予約 / 実行 |
| --- | ---: |
| one-shot primary pass | 1 / 1 |
| 旧unknown行の接続 | 120 / 120 |
| 既存baseline費用の完全一致検査 | 712 / 712 |
| 新適格baseline費用 | 34 / 34 |
| 直接単価側の費用会計 | 746 / 746 |
| baseline / 置換側PF選択 | 30+30 / 30+30 |
| scope順位比較 | 30 / 30 |
| 候補pair関係比較 | 上限275400 / 41578 |
| 別表現の費用数値検査 | 1492 / 1492 |
| 別表現のscope / pair検査 | 30・上限275400 / 30・41578 |
| 旧70参照＋新30の取りまとめ | 100 / 100 |

primaryはFraction、数値監査は整数cross-product・別比較器で費用/最小値/全同率集合/順位/比/方向/全pair関係を検査した。後処理checkerは保存receipt・literal・CSV/JSON・SHA・call予算・図の100条件coverageを照合し、科学会計を再実行しない。
これらはprimary agent自身の検査であり、別担当による独立レビューではない。

## 終了・保全状態

[終了監査](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/completion_audit.json)はPASS。
親PID532007、子PID532064、native exit0、同じ起動親の`wait4`で回収、owned worker残0、stderr空。
起動・入力gate・回収を含むwall `4.90414447337389`秒、CPU `4.897529`秒、peak RSS `123138048` bytes。子内本体・gate時間は`4.362522399052978`秒。
小規模の有理数演算を1 childでindexし、10 endpointの費用を30 scopeで再利用した。CPU/RAM/workerの独自上限、affinity・共有OS設定・他ユーザー/processへの変更は0。

旧SOURCE・raw・Stage A・旧70・旧停止packet・grant/ledger/marker/STOP、プロトコルv2本文・旧補足・論文主表・Figure2は保全した。新しい研究資料は別来歴として保存し、旧報告の履歴を遡及更新していない。
図2枚をPNG/SVG/PDFの6形式で保存し、coverage/SHAを検査、PNGを目視確認した。

## 残る限界と次の担当

H6のprofile差、H8/model runtime同一性未証明、S4合成モデルの仮説、初回572軸の最終gate欠測、旧profile照合待ち48軸等を維持する。H2の旧signal20軸、残338軸の新モデル評価、未保存予測78軸は対象外のままである。
今回100 scopeは固定された登録集合を対象とする条件付き保存値診断であり、モデル固有誤差と実行環境差を分離した検証、全PF集合の大域最適性、厳密bias certificate、PR母平均優位・未観測希少event保証を主張しない。
新34費用の来歴と旧70のbaseline再現を区別する。初回停止した結果の正常終了を遡及認定したものでもない。

今回の承認範囲を完了するための追加科学実行は不要。新規公開範囲は未承認で、pushは行っていない。
次担当**ユーザー**：新しい成果物の公開範囲を判断する。次担当**Codex**：公開承認を受けた場合に、その範囲だけをpushする。追加解析・詳細科学レビュー・最終evidence freezeへ自動進行しない。

**mandatory STOP。**
