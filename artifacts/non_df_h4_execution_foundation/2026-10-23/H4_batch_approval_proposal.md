# H4実分子pilot：次回1回の一括承認案

**未採択・未承認。これは承認依頼の対象を具体化した提案であり、承認記録・有効grantではない。** 現在の3実行flagはfalse。実装追加は不要で、明示承認後のSOURCE/environment sealと新grant発行が残る。

対象は[完成契約候補](H4_execution_contract_DRAFT.json)、[技術SOURCEのファイルhash](SOURCE.json)、実行時の同じファイルを持つclean HEAD、下記の未使用output namespace。実行時にこの3者と承認を正確に結合する。

提案namespace：`artifacts/non_df_h4_pilot/2026-10-23/run01`（この新worktreeからの絶対pathへ固定。今回未作成）。

| 承認対象 | 固定範囲 |
|---|---|
| 分子・入力 | linear H4、隣接1.00 Å、STO-3G、通常canonical real RHF、4 spatial/8 spin、共通未閾値MO積分 |
| 表現・状態 | Pauli-JWとtol-only1e-8 DF（rank未生成、final_rankなし）。非DF Nalpha=Nbeta=2 sector36最低状態を4方式で共有。DF/FCI照合 |
| 4 case | Pauli PF/PR・DF PF/PRを各1構成、S2/T=.8/q8/δ=.1、PR r4/K4 |
| prefix | Pauli abs係数順、DF既存abs(lambda) Frobenius(g)^2順、それぞれ50%以上の最初のprefix。tie-break固定、正の非空tail必須、DF one-body常時保持 |
| PR標本・精度 | 各256 paired trajectories、固定SHA seed/既存法則、ε=.05/.01/.005/.001/.0005。追加sampling・適応変更なし |
| 費用・compiler | ordinary-controlled cosine/sine full Hadamard wrapperのcompiled RZ、P=0、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなし |
| 数値会計 | 既存Hoeffdingとempirical allowance・既存閾値。精度不適格・証拠不足はN/G nullのまま保存 |

科学callは、入力生成のSCF1・共通MO積分1組・Pauli-JW生成1・DF分解1・非DF sector ground1・DF sector crosscheck1（36列）・FCI1・入力full-H reference2と、固定4 caseのみ。case内の既存reference/法則平均/数値evidenceを各1固定評価経路で実施する。決定論2 trajectory＋PR512 trajectory、最大1028 full-wrapper compile requests、16384 production RTE events、Pauli sampler256 batch・DF2048 occurrence batch。cache hitは記録し、乱数標本を間引かない。先頭3 trajectory/PRのcaptured-event operator照合とorder0/2/4代数検査を含む。5精度は同じ評価から射影し、別候補・追加実行へ進まない。

承認後にCodexが、clean SOURCE/runtime/入力条件/compiler/承認/namespaceをseal→新one-shot grant発行→入力生成と封印→4独立processでcase実行→4×5精度統合→技術gate確認と部分結果を含む保存→ローカルcommit→mandatory STOPを進める。工程ごとの再承認は不要。失敗時は科学caseをretryせず、他caseの部分証拠も保全してSTOPする。旧grantを再利用しない。

人工検査は既存371＋新規65件PASS、FAIL/skip0、分子計算3件除外。実H4 rank/prefix/精度適格性・runtime・希少event観測は未確認。8量子ビット人工検査の20行は全N/G nullであり、実H4でもnullになり得る。観測eventと価格の対応記録は、未観測希少eventの母平均費用を保証しない。gate失敗やnullを理由に閾値・q/r/K/prefix/標本数を変更しない。

一括承認の文例（**現在は未承認**）：

> 完成したH4契約候補と固定prefix提案を採択し、提示されたSOURCEと未使用namespaceに結合する。H4入力生成から固定4方式・5精度のone-shot実行、保存とローカルcommitまでを一括承認する。科学retry・追加sampling・条件変更・push・詳細科学レビューは承認しない。

次の担当：**ユーザー**が上記全体の採択・一括実行を判断する。承認が届いた後の担当は**Codex**。現在はmandatory STOP。
