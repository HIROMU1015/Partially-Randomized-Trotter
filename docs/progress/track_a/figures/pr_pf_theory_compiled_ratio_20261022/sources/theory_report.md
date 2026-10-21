# H4/H6：理論総RZ費用とcompiled総RZ費用の図（2026-10-19）

利用者の指示に基づき、S4修正後の保存結果だけを転記・整形・描画した。
[8枚の図・データ表・caption・検査記録](../../artifacts/resource_applicability/track_a_h46_theory_compiled_total_plots/2026-10-19/README.md)。
新規sampling・compile・bootstrap・shot/総費用/比/順位/統計量の数値再解析は行っていない。

基点 `6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1` は修正版**RESULT**である。
実counter評価SOURCEは`f01fa5fbd9c508e468dac3cff8405f57370dbb51`、保存再会計SOURCEは`da8eb64ba8e88c219a6b707e1c72d130f5f37fcb`。
今回これらの科学SOURCEやrunnerをimport・実行していない。

## 使用データと表示の選択

正本は基点commitの以下のファイル。全て固定Git objectとして読み、SHA256・bytesを記録した。

- [shot_weighted_totals.json](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/shot_weighted_totals.json)：420保存行からNATIVE_MIXED/RZの70条件を転記。
- [pf_pr_comparisons.json](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/pf_pr_comparisons.json)：360保存行から同モデル/metricの60固定pair条件を転記。
- [preserved_shots.json](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/preserved_shots.json)：既存shot・n・cohort・適格性の注記用。shotを掛け直さない。
- [direction_audit.json](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/direction_audit.json)：保存方向判定の注記用。新しい順位判定をしない。
- 修正後のterminal・result_summary・Stage2 acceptance・結果報告：S4修正と完了状態を照合。

理論モデルは**NATIVE_MIXED**を用いる。PRではprefix A＋各closed finite-RTE tail B、PFではAと同じ。
全回路B、A/B間の補間、最適なcompilerの理論上界ではない。
モデルA/B・CXは今回のRZ主図から外したが、元データは不変。
`table_c_candidates.json`の代わりに、修正後の正本JSONを直接読む。旧比較ディレクトリや`old_theory`欄は描画に使用しない。

5精度ε=.05/.01/.005/.001/.0005、allowance倍率1/10をどちらも表示した。
**70絶対費用条件のうち50適格・20不適格、60費用比条件のうち32共通適格・28共通適格性なし**。
これは保存行の構造件数で、追加の科学解析結果ではない。
各系・各倍率で絶対費用1枚と費用比1枚、計8枚。PNG/SVG/PDFを各8ファイル保存した。

## 固定候補と比較範囲

|系・DF|候補種別|固定候補ID|L_D / q / δ|保存compiled標本|
|---|---|---|---|---|
|H4 legacy固定DF12|PF|B1_p12_q1_2nd|12 / 1 / .8|固定回路n1|
|H4 legacy固定DF12|PF|B1_p12_q1_4th|12 / 1 / .8|固定回路n1|
|H4 legacy固定DF12|PR|B2_p3_q1_2nd_r4_K2|3 / 1 / .8|fixed_anchor_panel n32|
|H4 legacy固定DF12|PR|B2_p3_q8_2nd_r2_K4|3 / 8 / .1|fixed_anchor_panel n32|
|H6 tol-only1e-8 DF19|PF|B1_p19_q8_4th|19 / 8 / .1|固定回路n1|
|H6 tol-only1e-8 DF19|PR|B2_p10_q8_2nd_r1_K2|10 / 8 / .1|exploration n8|
|H6 tol-only1e-8 DF19|PR|B2_p15_q8_2nd_r2_K4|15 / 8 / .1|exploration n8|

全候補はlinear 1 Å / STO-3G、T=.8、状態準備P=0、symmetric-directional、cos/sin full Hadamard wrapper。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、topologyなしという保存compiler条件を保持する。
H4新tol-only DF10、H6別cohortの独立確認、H8、他geometryや最新v2の新A比較を混ぜない。
H4/H6はDF政策・標本段階が異なるので、別図とし、サイズscalingを主張しない。

絶対費用図は候補ごとのpanelでPF/PRを明記し、横軸を要求複素信号精度、縦軸を対数の総RZ費用とした。
理論lower/upperは紫・白抜きdiamond、compiledは緑青・丸。各図内は同じ縦軸範囲で比較できる。
費用比図は固定6pairを系ごとに分け、縦軸をPR/PF比（対数）、基準線を1とした。
1未満はPR低費用、1超はPF低費用。精度軸は5条件のカテゴリ表示で、補間・fitは行わない。
重なりを避けるため理論/compiledのmarkerを同じεの左右へわずかにずらしている。

## 値の扱いと欠測

`G = Σ_axis N_axis C_axis`という既存定義の**保存済みG**を使う。
候補ごとの必要shotをそのまま継承し、PR/PFを同一shotへ置き換えない。
保存理論期待費用と、保存compiled有限標本平均に基づく費用を比較している。
same-eventの理論/compiled比較ではない。

理論lower/upperの両方を保持し、中点・代表平均を作らない。幅が描画解像度以下のときは重なる。
理論の保存範囲をCIや全不確かさの厳密boundと呼ばない。
compiledの新しいSE/CIは付けない。分数からfloat・小数ラベルへの変換は表示目的だけである。
PR/PF比は保存`theory_PR_PF_lower/upper`・`compiled_PR_PF`を直接使い、総費用を割り直さない。

不適格はexact表でnull、CSVで空欄として保持し、図では空白と灰色×で示す。
×の高さはaxes内の注記位置であり、費用0や測定値ではない。欠測を跨ぐ線は途切れる。
ratioの共通適格性なしを、両候補が同時に不適格であると読み替えない。

## 保存値から読み取れること

H4のPR p3/q8/r2/K4対PF p12/q1/S4、ε=.05、倍率1では、保存PR/PF比が
**理論0.8716321313、compiled1.0393357766**（修正後報告の表示値）である。
この固定条件で、理論とcompiledによる費用の優劣方向が異なることをratio図に明示した。
S4修正後にも残る既存の観測であり、新しい発見や母平均順位の認定ではない。

H6では表示対象の固定PR2候補に対し、保存された費用比の方向は理論・compiledともPR低費用である。
絶対費用が異なっても、相対的なPR/PF比のずれは候補・条件に依存する。全候補や別compilerへ一般化しない。
H4では厳しい精度に不適格が多く、空欄は未完了の科学計算を実行して埋める対象にはしていない。

## 構造検証・保全・公開

[描画script](../../scripts/plot_h46_theory_compiled_totals.py)は前回c5ad7945のJSON/CSV/SHA helperと描画規約を流用した。
科学的な`h46_cost_summary.py`は実行しない。`--check`はstdlibだけで、固定行と転記表の一致、
exact→float座標、null、図形式、SHA、local link、元データ・旧索引・既存manifest項目の保全を照合する。
既存理論counter・SOURCE・結果・grant・ledger・marker・STOP・Figure2・主表に変更なし。
追加の科学test・native回路・compiler・samplingを動かさない。

PNG全8枚を目視確認し、作者であるCodex自身の技術確認として記録した。
別担当の独立科学レビュー、詳細科学レビュー、最終evidence freezeではない。
前回作成したプロジェクト専用Matplotlib環境を再利用し、今回のfont cacheは専用worktree内だけに置く。
共有環境・他process・他worktreeの状態は変更しない。

今回の指示は、保存公開データからの新しい図・表・caption・説明・構造検証コードと索引の専用commit/pushを許可する。
push先は`track-a-h46-theory-compiled-total-plots-20261019`。基点修正結果はremoteで既に公開されていることを確認した。
mainへのmerge、旧論文packetの更新、追加科学stageは行わない。

次の作業なし。Codexは専用commitとpushのremote SHA一致確認後 **mandatory STOP**。
