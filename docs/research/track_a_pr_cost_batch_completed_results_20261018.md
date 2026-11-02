# Track A：H4/H6/H8 A_COST完了・保存B/D解析の結果

2026-10-18。利用者の一括実行・必要修正の承認と、その後のH8引継ぎ指示に基づく結果を整理する。
**A_COSTは3系とも16候補×探索n8、各128軌道・256軸を完了した。保存D会計3件とB限定診断3件も完了した。**
ただし、新A_COSTを用いた総費用比較・新ランキング、Bの無条件平均費用は未実施／未確定である。
本資料の作成時に新しいsignal、sampling、compile、fit、shot再計算、ランキングは実行していない。

公開データの入口は[結果bundle](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/README.md)。
以下の旧資料は、その時点の状態として保持する。

- [9契約のsealと承認案](track_a_pr_cost_nine_contracts_seal_20261018.md)
- [初回起動とH8 D修正](track_a_pr_cost_batch_execution_20261018.md)
- [H8並列方式・引継ぎの準備](track_a_h8_pr_candidate_parallel_readiness_20261018.md)

## 1. 条件と到達点

全系linear H-chain、隣接距離1.0 Å、STO-3G、T=0.8、状態準備費用P=0、
既存S2 PR／元finite-RTE law、ordinary ancilla branchとsymmetric-directional control。
探索n8は旧確認n32とは別の有限標本である。DF fragmentの固定順序とprefix分割を維持する。
Hamiltonian/state/snapshot/compiler/profileのhashと候補ごとのq/r/K/Rはbundleに記録した。

| 系 | 入力系列 | Aのprefix | Aの取得 | D保存会計 | B限定診断 |
|---|---|---|---|---|---|
| H4 | **legacy固定rank12** | 2/3/4 | 16候補、128軌道、256軸、control16 PASS | 480行・40比較 | 2構成・4条件付き群 |
| H6 | tol-only 1e-8、rank19 | 3/4/5 | 16候補、128軌道、256軸、control16 PASS | 2,880行・40比較 | 2構成・2条件付き群 |
| H8 | tol-only 1e-8、rank28 | 6/7/8 | 16候補、128軌道、256軸、control16 PASS | 3,800行・40比較 | 2構成・4条件付き群 |

合計384 paired軌道・768 measured axes、48代表control。
H4の別tol-only rank10系列、旧geometry map、別の高次PF研究の結果と混合しない。
候補の完全な登録表は[a_candidates.json](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/a_candidates.json)。

## 2. SOURCEとnative終了

| 対象 | 実行SOURCE | 終了 |
|---|---|---|
| H4/H6 A、B3系、H4/H6 D | `31ef62a12407010e9c1a07f9b37facedb8dff5fe` | 各scope native exit0・親のreap確認 |
| H8 D修正後 | `8ff002dd9524b49284fd8fb8e11354ee1ec18a00` | native exit0・reap確認 |
| H8 A引継ぎ後の112軌道 | `87f1f0d528478ad212ba98159db06eec3c78433e` | native exit0・112 worker全回収 |
| H8 A再利用の代表16軌道 | 元SOURCE `31ef62a12407010e9c1a07f9b37facedb8dff5fe` | 元native exit0/reap/result SHAから個別受入れ |

H8のseed導出SOURCEは元31ef62a1…のまま。新SOURCEはscheduler・受入経路の変更であり、
元128 taskのseedと科学kernelを保持した継続である。SOURCEの違いを理由に新しい独立標本とは数えない。
各軌道の実行SOURCE・元result SHA・seed導出SOURCEを測定表に記録した。

新run ID：`h8-pr-candidate-parallel-20261018-resume01`。
15 worker、内部thread1で、112軌道・224軸を3359.055882秒（約56分）で完了。
終了は**2026-10-18 21:14:13.646 JST**。新runの最大単一process RSSは4.673 GiBであり、
全processの同時peak RSSではない。起動時416 GiBはparent・worker・余裕を含む保守的なadmission見積りで、
実際にその量を割り当てた／消費したという意味ではない。
H4/H6は初回各6 worker、H8は初回10→継続15 worker。旧方式との統制された速度比較はしていない。

元H8 Aは意図的引継ぎの`OWN_RUN_SIGNAL_15` STOPを保持する。
元runで開始した追加10軌道は未完了・再利用不可として保存し、trajectory10/native evolution10/
occurrence20/RTE event80のbefore-call予約は返却しない。この10件は成功384軌道に含めない。
新runで同じ固定taskを再実行し、元の結果・grant・ledger・markerは変更していない。

元H8 Dの`ORIGINAL_PR_MEAN_MUST_REPRODUCE` STOPも保持する。
保存通常modeの対応と倍率keyを修正した別SOURCEで完了し、H4/H6/H8それぞれ104/104/88件の
適格な既存PR平均再現検査が実行時に通過している。初回batch全scopeが成功したという報告ではない。

## 3. A_COST：何が取得できたか

[a_cost_measured_axes.csv](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/a_cost_measured_axes.csv)
に全768軸のRZ/CX、depth、RZ_depth、CX_depth、size、compile wall、fingerprint、
paired event digest、seedを転記した。これは既存signal・normalizationに結合したnative実測で、モデル推定単価ではない。
controlではfull-Fock scalarの一回適用、referenceを回転しない規則、独立event oracleを確認した。
代表16件は単にcontrol fileが存在するだけでなく、paired compile・native exit0・reapまで完了している。

**この探索n8を旧確認n32へpoolingしていない。新候補の最良順位・PR削減率はまだ作っていない。**
必要shotの算術、normalization、精度判定、compiler optionsは変更していない。
H4の既存規約による3 prefix preparation復元以外、新しいSCF/DF/state入力生成は行っていない。

## 4. D_ACCOUNTING：既に完了した保存値比較

Dは、新A_COST取得前に固定した保存PR費用と、既存の追加直接実測を使った会計である。
新A_COST探索n8やB条件付き標本は使用していない。
5精度×allowance倍率1/10×通常／経験的非相殺、各系2種類の決定論比較集合を区別する。

- H4：旧固定集合／旧集合＋直接実測subset。別rank10や全geometryの比較ではない。
- H6：old32／strengthened_S2_S4。未実測候補のモデル推定が残る。
- H8：old32／strengthened184。184候補全体を直接実測済みとはしない。

下表は**保存済みD比較の強化集合・倍率1・P=0**の転記。
正はPR有限標本の点費用が小さく、負は比較集合の決定論点費用が小さいことを表す。
厳密bias保証、母平均の優位性、全候補実測最良、energy/QPE/RPE最終総費用の主張ではない。
モデル推定・旧profile不明の来歴は元のstatusのまま残す。

通常signed biasのPR削減率（%）：

| 系 | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---:|---:|---:|---:|---:|
| H4 | 19.3462 | 48.0918 | 50.0985 | 43.7395 | 52.6334 |
| H6 | −13.2901 | 30.8905 | 34.9653 | 31.1677 | 10.3257 |
| H8 | −8.0230 | 33.2373 | 31.4549 | 23.1282 | −50.8125 |

経験的非相殺biasのPR削減率（%）：

| 系 | ε=.05 | .01 | .005 | .001 | .0005 |
|---|---:|---:|---:|---:|---:|
| H4 | 44.2280 | 48.0911 | 50.1002 | 43.7367 | 52.6295 |
| H6 | 22.2791 | 32.2240 | 37.0663 | 36.2830 | 20.1071 |
| H8 | 16.6243 | 34.6849 | 35.1253 | 39.4150 | −28.7267 |

倍率10・旧集合を含む全120比較の削減率exact fraction、選択候補、実測／modelの区別は
[D比較要約](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/d_saved_comparison_summary.json)。
元の詳細な費用・保存shot・SEと全行の結果はprivateで保全する。
今回は既存の選択結果を転記しており、保存値から新たな最小化・再会計はしていない。
過去のH6 B0+I全12構成報告等と比較集合が違うため、それらの数値を上書きしない。

## 5. B_ANALYSIS：完了した限定診断と未確定事項

元lawで条件付けたpilotの各n8費用、記録済みlog mass、元次数分布、保存標本SEを整理した。
H4/H8はq4/q8ごとのs2_multi/s4、H6はK2 s2_multiとK4 s4を扱う。
H6 K4のs2_multi費用を他候補から補完していない。

6構成とも`unconditional_mean_RZ=null`、
`UNRESOLVED_NO_EXACT_EXHAUSTIVE_MASS_RECEIPT`。
完全な元lawの層分割・質量receiptと通常標本の結合が不足しており、
条件付き平均を無条件期待値へ置き換えたり、欠測層を費用0にしたりしていない。
今回のpayloadに結合された通常標本は0件である。
出力`ordinary_counts`の0は「結合された標本がない」という記録で、母集団のevent確率0や未観測性の証明ではない。

診断範囲と不足層は[B診断要約](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/b_diagnostic_scope_summary.json)。
詳細な次数分布・log mass・条件付き平均／SEの数値原本はprivateで保全する。
元n32平均・旧条件付きpilot・元rawは保全した。完全な無条件比較まで終わったと解釈しない。

## 6. 検証・公開範囲と停止

公開時には9 result SHAとterminal、384個別taskの元seed/両軸/予約、代表control48件、native exit0/reap、
3つのSOURCE closureのGit blob/hashを照合した。これらは作者による保存artifactの機械的確認であり、
独立科学レビューや外部再現ではない。既存の人工24＋承認境界18＋無効CLI2等は再実行していない。

公開するのはscalar測定表・保存済み比較のprojection・scope別binding・終了／来歴hash・資料のみ。
最新の機械可読状態は[専用status entry](../../artifacts/resource_applicability/track_a_pr_cost_batch_results/2026-10-18/validation_status_entry.json)。
rootの大きな履歴manifestは既公開bytesで保全し、今回の追補を独立fileとして記録した。
private入力、科学array、raw event列、NPZ/QPY、runtime/checkpoint、ledger本体、
有効grant／認可原本、認証情報、内部SSH情報はstageしない。
旧報告・SOURCE・STOP・既存grant・他研究のworktreeとjobは保全した。
本資料作成中の共有環境・既存venv変更、GPU使用、AI subagent、原稿改訂、新しい実行grant発行は0。

**現在は全契約の実行終了後のmandatory STOP。**
次の担当はユーザー。Aの新費用を使う総費用比較、Bの不足証拠対応など、次に実施するscopeを判断する。
Codexは新たな実行指示の範囲が確定するまで、追加解析・確認sampling・次stageを開始しない。
