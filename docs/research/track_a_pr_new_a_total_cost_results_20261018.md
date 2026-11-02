# Track A：新A_COSTの保存値総費用比較（完了）

H4/H6/H8の3契約を各1回、独立並列実行し、全件正常終了・回収済み。追加sampling/compile/signal/fit/shot再計算は0。
SOURCE：`8601b921342fc1cc83780a830e1f65bf64fe9e93`。
[固定契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/37c3357bd156f7b91179e14e257be088de40c413/docs/research/track_a_pr_new_a_total_cost_readiness_20261018.md)。
[候補別費用・比較表](../../artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/README.md)。

条件はlinear H-chain、1Å、STO-3G、T=.8、P=0、既存S2 PR・元RTE law。
H4はlegacy固定DF12/p2,3,4、H6はtol-only1e-8 DF19/p3,4,5、H8はDF28/p6,7,8。
各16新候補の探索n8と保存shotを結合した有限標本の点比較。既存結果・科学条件・compilerは変更していない。

## 通常bias・倍率1・強化決定論集合に対するPR削減率（%）

| 系・標本 | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---:|---:|---:|---:|---:|
| H4 新A探索n8 | 16.0220 | 25.6958 | 50.6900 | 46.5074 | 54.9402 |
| H4 旧PR | 19.3462 | 48.0918 | 50.0985 | 43.7395 | 52.6334 |
| H6 新A探索n8 | −76.8195 | 15.4763 | 29.2796 | 30.8980 | 9.9766 |
| H6 旧PR | −13.2901 | 30.8905 | 34.9653 | 31.1677 | 10.3257 |
| H8 新A探索n8 | 3.1084 | 31.8387 | 29.9926 | 17.9519 | 3.1178 |
| H8 旧PR | −8.0230 | 33.2373 | 31.4549 | 23.1282 | −50.8125 |

正はPR点費用が小さい。比較集合はH4 old_plus_direct_subset、H6 strengthened_S2_S4、H8 strengthened184。
H4は厳しい3精度で新Aが旧PRの点費用を下回り、H6の通常5精度では新Aは旧PRより高かった。
H8通常ε=.05の新候補は`B2_p8_q2_2nd_r4_K4`、ε=.0005は`B2_p7_q16_2nd_r4_K4`。
この2点ではPF優位からPR点費用優位へ符号が変わったが、独立確認済みとは扱わない。
対応する選択PF単価は直接実測。比較集合の他候補にはmodel推定が残る。

## 経験的非相殺と欠測

H8非相殺・倍率1の新A対PF削減率は、5精度順に25.2158/33.3166/33.7413/35.3354/17.3057%。
H6非相殺では旧B0+I12候補のsigned成分が未解決で、全登録集合の削減率はnull。
従来Dの取得済みsubset比較を変更せず、欠測候補を除いた値を全登録最良として扱わない。
H6/H8の新A倍率10は保存shotがなく、計320行を欠測/nullのまま保持した。

新Aの960 endpoint枠は有限標本点478・不適格162・欠測320。
120比較枠のうち70で新domainがcomplete、50は未確定。completeは入力欠測がない意味で科学的保証ではない。
全条件の候補ID・点費用・paired SE・実測/model区分はCSVに保存した。旧D7,160行は不変。

新n8と旧n32はpoolingしない。H4旧n32は選択・推定用のfixed anchorで、独立確認とは呼ばない。
両軸の同一trajectory内相関を保ったSEを用い、差の検定・追加信頼区間・rare層の再重み付けは行っていない。
探索選択効果、未観測rare event、model誤差は残り、母平均の優位・全候補実測最良・厳密bias保証は主張しない。
H8 profileは元契約識別子を継承し、原profile file bytes一致とは主張しない。

公開は科学結果表と短い完了要約に限定。詳細運用記録はprivateで保全し、既存の大きな索引は保持した。
**mandatory STOP。次の担当はユーザー/GPT：独立確認の要否を判断する。追加計算は開始していない。**
