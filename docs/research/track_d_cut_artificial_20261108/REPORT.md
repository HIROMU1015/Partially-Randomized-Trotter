# Track D O5候補B：近傍L/R cutの人工検証結果

2026-11-08。**技術検証は成功。事前固定した人工不均衡caseで、同じwhole-path Vと全入力X/Yを保ち、Reference中央cutのtotal depth934→646（約30.8%減）を観測した。** RZ/CX count、size、widthの追加改善は0。均衡caseは中央を選び、追加効果0を保全した。実H4での有効性、新規性、全PR・全RPE優位は未確認。

| 項目 | 結果 |
|---|---|
| 最小実装 | 既存 `prepare_steps(cut=...)` とgadgetを変更せず、C7-4 descriptorだけで中央±1を選ぶ薄いadapter |
| 人工入力 | 2 mode・3 native block、q6/16 event毎step、h=.1/τ=.2、scalar .17/identity .04。geometry/AO basis/分子DF rank/L_D非該当 |
| Correctness | 8 unit/negative tests、110数値比較PASS。全4入力・全reference出力・X/Y observable・actual inverse・χ・signed metadata・P0 full phase |
| P0 | Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/既存MPFR160 |
| Compile | 2case×3構成×X/Y=12結果、10 unique compile、2同IR cache hit |
| 不均衡case | 中央cut3→選択cut2。Strong/中央/選択のdepth=1636/934/646、width=3/5/5 |
| 均衡case | 中央cut3→選択cut3。Strong/中央/選択のdepth=2197/934/934、追加改善なし |
| 実H4/旧価格/モデル | 新input・sampling・build・compile・保存価格再解析・fit・N/13round再評価すべて0 |

## 解釈と限界

Cutはcompile前に決めた。全cut compiled探索やoracle価格の事後minではない。StrongのGaussian/relative/head/tail/step融合を維持し、cutによるstrong弱体化はない。Referenceは両cutともq−2=4内部融合・1境界損失で、whole-path一gadget。A†/Bへの仕事配置とcompiled依存経路が変わったことを確認したが、削減288のcompiler/closure/CSWAP別の因果内訳は未分離。Proxyはheuristicであり価格式ではない。

数学的な全入力ideal同値性と、有限2人工例のbinary64 correctnessと、P0での資源観測を区別する。既知SE-QPE型Referenceのgadgetは再実装しておらず、分割heuristicの独立した新規性は認定しない。今回の2-mode/16 event fixtureから、8-mode実H4/r32へ削減率を外挿しない。全law期待価格、shot数、最終総費用は評価していない。

## 成果物

- [実装とcorrectness](IMPLEMENTATION_AND_CORRECTNESS.md)：再利用SOURCE、選択規則、数理条件、人工/負テスト、技術修正。
- [全7資源比較](ARTIFICIAL_RESOURCE_COMPARISON.md)：中央との差とStrongとの差、負結果、lane/fusion、負荷。
- [未採択の最小H4移送案](NEXT_H4_VALIDATION_DRAFT.md)：保存q4一path、通常新X/Y2本・再利用不能時最大6本という提案。今回実行しない。
- [固定PLAN](PLAN.json)、[SOURCE](../../../artifacts/track_d_cut_artificial/2026-11-08/artificial_run02/SOURCE.json)、[COMPARISONS](../../../artifacts/track_d_cut_artificial/2026-11-08/artificial_run02/COMPARISONS.json)、[STOP](../../../artifacts/track_d_cut_artificial/2026-11-08/artificial_run02/STOP.json)。QPY/全構造は同directoryのpayload archiveとindex。

run01は未使用threadpoolctlの環境metadata取得だけで停止しcompile0。環境記録のみを修正し、同じPLAN・fixture・規則・許容差でrun02完了。失敗/初回SOURCEと修正履歴も保持した。

## 保全と次の判断

専用branch `codex/track-d-cut-artificial-20261108`、独立worktree/output/cacheで実施。元mainと既存dirty/untracked、旧SOURCE/raw/結果/grant/ledger/marker/STOP、C1凍結を保全する。q64別担当のprocess/SOURCE/入力/実行中出力へ介入せず、進捗や未完了結果も判断材料にしていない。新しい共有設定・venv・共有cache変更なし。

公開SHA/tree/bytes照合を完了した時点で**mandatory STOP**。次の担当はユーザー/GPT。候補Bは最小実H4技術移送を検討する価値があるが、固定SOURCE/inputと有限call数を採択して別途実行許可を得る。候補Cの実装、新しいH4計算、qモデル/N/全RPE更新、詳細科学レビューへ自動進行しない。
