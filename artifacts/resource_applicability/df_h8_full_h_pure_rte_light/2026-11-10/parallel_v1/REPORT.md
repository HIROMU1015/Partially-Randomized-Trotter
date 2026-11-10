# H8 DF全Hamiltonian純RTE：T=1.6小規模検証

Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit・spin sector4900、T=1.6、ε=.0005、P=0。

Full-H RTE構築PASS。一体項16 Z成分と全28 DF fragmentをsamplingへ含め、決定論blockは0。λ_R=66.0090400399。Hamiltonian再構成最大誤差=2.74e-14。

signal評価は4候補、適格3候補。価格取得前にS2 q32/R704/K2の1候補を固定し、取得済み価格は1候補・9/9 trajectory。広域最適化・境界の反復拡張・新独立確認cohortは行っていない。

固定代表のG=7.57798306e+15 RZ、記述的SE=1.352625e+13。保存B2の112.541倍、保存PFの84.8682倍だった。構築と適格性は確認できたが、この代表の点費用は保存比較より大きい。

## 価格と保存B2/PF比較

|方式|構成|n / cohort|G（RZ）|記述的SE|
|---|---|---|---:|---:|
|FULL_H_PURE_RTE|2nd p0 q32 R704 K2|9 / H8_FULL_H_PURE_RTE_LIGHT_EXPLORATION_20261110|7.57798306e+15|1.352625e+13|
|B2|4th(m5_best) p8 q1 R32 K4|9 / CONFIRMATION|6.73353173e+13|1.031802e+11|
|PF_B0I|4th(m5_best) p15 q1|1 / EXPLORATION|8.92912195e+13|—|

Full-H pure RTE / B2_CONFIRMATION = **112.54099**。

Full-H pure RTE / PF_B0I_SAVED = **84.868177**。

代表のB=1.640710059、Ncos=188,738,782、Nsin=188,738,780。平均compiled RZはcos=20075320.6 / sin=20075320.6、平均CXはcos=5551994.78 / sin=5551994.78。実観測の一体項applicationは60,713件。

対B2：shot合計比=2.178726、平均RZ/回路比=51.6545。今回の両軸価格が同じなので積がG比に一致する。保存log B=0.03531903883、B=1.035950164。大きなG差には1回路価格の増加が強く寄与した。

対PF_B0I：shot合計比=2.591657、平均RZ/回路比=32.74668。今回の両軸価格が同じなので積がG比に一致する。保存log B=0、B=1。大きなG差には1回路価格の増加が強く寄与した。

実観測event次数はorder0=202,750、order2=2。保存lawによる1 trajectory内のorder2出現確率は0.219299766、n9で未観測となる確率は0.107735488。order2は今回は観測されたが、2 eventだけから条件付き母平均価格を保証しない。未観測categoryがあれば価格はnullとし、0補完しない。

固定S2代表と保存m5 B2/PFの点比較であり、同一q/R/Kによる因果比較ではない。cohortをpoolせず、全域最適性・統計的な方式間優位・母平均費用保証を主張しない。

## Signal・normalization・必要shot

|q/R/K|適格性|bias cos / sin|log B|Ncos / Nsin|
|---|---|---:|---:|---:|
|32/576/2|ELIGIBLE|-9.0336e-12 / -6.3017e-12|0.60515388|235194660 / 235194656|
|32/704/2|ELIGIBLE|-5.4645e-12 / -3.8141e-12|0.495129111|188738782 / 188738780|
|32/832/2|ELIGIBLE|-3.0563e-12 / -2.137e-12|0.418956963|162068591 / 162068590|
|32/704/0|INELIGIBLE|0.00074225 / 0.00051738|0.247565084|None / None|

保存T=1.6 referenceと入力bindingを照合し、T=.8等のreferenceは使っていない。保存stateの固有残差を測り、有限Taylor多項式の固有state作用を計算した。代表全22,528 stepのsparse実ベクトルrecurrenceとの差は[signal_vector_validation.json](signal_vector_validation.json)。4900-square密行列・full-space unitaryは作っていない。
共通u floor1e-9、alpha_axis=.025、headroom・shot政策は既存のqualify_recordを再利用。Bは有限K lawに基づき、raw×Bとcorrectedのclosure、native lawとのnormalizationを検証する。必要Nは計算値で、量子shotの実測ではない。Signal biasはcompile前の有限meanの差であり、compiled回路の厳密bias保証とは区別する。

## 回路・correctness・再利用

H8保存loader、Gaussian操作、fragment順0..27を再利用。一体項RTE追加はH4/H6 prepare_full_hへ保存済みbaseを渡す最小拡張。新SCF/DF/state・Gaussian circuit生成は0。scalarは一度だけ適用する。
正負時間・actual adjointとone-body/mixed eventを確認。独立event演算は既存saved-basis algebraと同じ局所行列作用をベクトル化したもので、circuit loweringとは別経路だが独立したorbital分解ではない。
Signal/referenceは保存spin sector4900を使用し、event/native/compiled検査はfull register上で行う。ランダムeventの中間状態をspin sectorへ投影していない。
価格はmeasurement込みfull cosine/sine Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、backend/topology指定なし。H8既存の小gate operator由来のglobal-phase会計を再利用し、statevectorから位相をfitしない。RZ/CX gate数は変更しない。
現行Python3.12.3、保存Stage CはPython3.11.0rc1。NumPy1.26.4・Qiskit1.3.0とcompiler identity・保存入力は一致する。環境全体のbitwise一致や旧価格の再compileは主張しない。実行環境のmetadataはSOURCE_INDEX.json。
作用照合は検証済みAer CPU doubleを使用。parallel_v1では価格用回路を維持し、検証用に限って最大3量子ビットgateの厳密Operatorをcall-local cacheし、実測で選んだ32 threadsを使用する。full-vectorの7検査は代表index0のみ。残り8価格trialでは回路identity・normalization・paired event等を確認した。全9のfull-vector個別検査を行った意味ではない。代表7検査の誤差は[validation_summary.json](validation_summary.json)、位相会計は[price_trials.json.gz](price_trials.json.gz)。主担当による実行・技術確認で、別担当者の科学レビューではない。

旧逐次workerはHIGH_CONTROL検査の途中で、承認された高速化のため同一標本を使うparallel_v1へ移行した。旧SOURCE・未完了出力を保全し、partial標本を集計しない。代表eventファイルのbyte一致は[representative_event_identity.json](representative_event_identity.json)。人工回路の約5.2倍は本番全体の短縮率を保証しない。
固定代表のeventはorder0のみ。K2 productで加わるCZは、元helperの小operator検査を使って追加位相0を確認し、別processの診断許可集合へCZを加えた。固定候補と同じτの条件付き単event full wrapperも両軸compileし、[補助検証](conditional_product_compile_validation.json)でPASS。主実行SOURCE・元helperファイルは変更せず、補助をn9や条件付き母平均費用の標本へ混入していない。

## H4/H6との対応・限界・次段判断

H4/H6でも、λ_R増加でnormalizationを抑えるためのstep数が増え、shot削減とcompiled回路長が競合した。H8でもR576→704→832でlog BとNは減少する。H8はλ_R=66.0090で、H4の21.0279・H6の41.2279より大きい。H4/H6 T=2.0とは時間も異なるため、その数値を同一条件の因果比較へ読み替えない。
価格候補選択はevent長×期待product次数×Nの未較正work proxy。compiled RZ価格でも厳密下界でもなく、未価格候補の劣位は未確定。今回のS2代表についての実装可能性と点費用が成果であり、最適pure RTEの優劣ではない。
Event次数の観測数、保存law確率、未観測条件付き価格unknown/null・母平均tail費用上界nullは[prices.json](prices.json)。追加rare sampling・top-upは0。小さなSEだけで母平均を保証しない。
この固定代表は保存B2/PFを約85～113倍上回った。今回は小規模検証で一区切りとするのが適切と判断する。追加研究を検討するなら、shot削減に加えてGaussian/basis作用を含む1回路価格を大幅に減らせる具体策があるかを先に考える価値がある。未価格候補の優劣・q/R/K全域最適性は未確認であり、広域探索は自動開始せず、次の研究範囲はユーザーへ戻す。
最初の索引tuple形式による技術的IndexErrorは再構成検査前に停止し、SOURCE c6523c737と停止記録を保持した。修正後の別SOURCE・fresh出力でPASSし、価格標本へ混入していない。[technical_history.json](technical_history.json)、[元の技術STOP](../validate_STOP.json)。

[H4/H6 T=2.0保存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md) / [H8 Stage C保存報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md)。
[RESULTS.csv](RESULTS.csv) / [SIGNALS.csv](SIGNALS.csv) / [paired_prices.csv](paired_prices.csv) / [SOURCE_INDEX.json](SOURCE_INDEX.json)。
旧SOURCE・raw/QPY・結果・grant/ledger/marker・mandatory STOPは変更していない。S1 Main Claim・論文構成は変更せず、詳細科学レビュー・新しい広域探索・別分子・T変更は開始していない。公開remote SHA・主要成果物照合後mandatory STOP。
次の担当：ユーザーが次の研究範囲を判断する。Codexによる追加計算は開始しない。
