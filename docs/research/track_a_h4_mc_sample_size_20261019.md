# H4：保存P標本のMC標本数依存性（2026-10-19）

**承認された保存値one-shot解析を完了した。ε=0.05では小標本の順位が変動し、n増大とともに固定PF低費用の結果が安定する。ε=0.0005では保存標本内でPR低費用の順位が維持される。両条件とも変動幅は概ねnの平方根に反比例して縮小する。これは固定経験分布内の評価であり、母平均への収束認定ではない。**

解析・監査・描画完了後にmandatory STOP。新RTE sampling、回路生成・compile、signal、shot再計算、PF再選択、旧cohort混合は全て0。技術監査はprimary agent自身が実施し、別AIによる独立レビューではない。

## 条件と固定来歴

linear H4、隣接1 Å、STO-3G、tol-only 1e-8 DF rank10、T=0.8、P=0、通常signed bias、allowance倍率1。ε=.05のL_D=3/q1/r4/K2/δ=.8、ε=.0005のL_D=2/q8/r2/K4/δ=.1。固定2次PF対照はそれぞれ`B0I_p5_q1_2nd`、`B0_p7_q8_2nd`。保存full cosine/sine Hadamard wrapperのcompiled RZ費用と保存shotを使った。

- 元P完了commit：`821cc0f6ced57168293ea166644024867b064ffb`
- 元P科学SOURCE：`3c664ca4a9f003880b9f6d56f2a0e55ed5b56603`
- 今回の解析SOURCE：`78352b861b3047f6af2d432535e23cc88b439bbe`
- 実行契約・承認receipt固定commit：`4988158f`
- 契約SHA256：`bd4b3491f0d7404961591ea142757e879221ce3c78f840bde55edacfc74d7daa`
- 再標本化seed：`20261019`。これは保存行のbootstrap専用であり、新RTE sampler seedではない。
- 実行：2026-10-19 20:12 JST終了。数値・描画本体9.008秒、保存数値監査込み約9.625秒。dispatch1、retry0。
- 専用branch：`codex/h4-mc-sample-size-20261019`。元PのSOURCE/input・結果・grant・ledger・marker・STOPを変更せず、既存Figure 2と他worktreeにも書き込んでいない。

## 会計復元と再標本化の定義

trajectory iの費用を `G_i = N_cos × RZ_cos,i + N_sin × RZ_sin,i` として整数で復元した。既存`h4_dp_contract.evaluate_pairs`をそのまま使い、保存済み平均と不偏標本分散の有理数表現、SD、paired SE、順位、層別件数を両条件で完全一致させた。別の保存comparison projectionとも照合した。独立した軸としてbootstrapしない。

保存shotはε=.05でcosine12,305/sine8,258、ε=.0005でcosine174,919,953/sine95,102,971。shotをnに応じて変更していない。

| ε | 保存n | 復元PR平均 G_RZ | 保存paired SE | 固定PF G_RZ | 全保存標本の順位 |
|---|---:|---:|---:|---:|---|
|.05|1,108|71,128,344.93|174,917.74|70,108,214.00|PF < PR|
|.0005|78|4,964,066,816,492.41|14,140,828,222.91|7,856,558,002,614.00|PR < PF|

累積平均は保存replica昇順の全prefix（1〜1,108／1〜78）を一度だけ走査し、指定nの点を別表へ転記した。n=1のSEは未定義のままnullとした。この累積列は元の探索n8とは異なる。

bootstrapは各条件の**保存全標本N**から、各nにつきn本を復元抽出する2,000反復。小さいnのprefixのみから再標本化する設計ではない。cosine/sineとstratumラベルは同じ抽出indexを共有する。各endpoint/nの乱数streamは`numpy.PCG64(SeedSequence([20261019, endpoint_index, n]))`で分離し、抽出index列のSHAを保存した。反復数は11＋7群、計36,000。

bootstrap SEは2,000個の平均費用の標本SD（ddof=1）。表の2.5〜97.5分位範囲は経験分布の変動幅であり、母平均の信頼区間ではない。分位点はlinear補間。累積paired SE、全保存標本のSD/√n、経験分布のSD/√n、各bootstrap反復内のpaired SEも別々に保存した。経験分布の分散は保存不偏分散の(N−1)/N倍であり、特にN=78ではこの区別を保持する。

## ε=0.05：小標本の順位変動が大きい

単位は平均・SE・分位範囲すべて `10^6 G_RZ`。固定PFは70.108214。PRがPFより安い割合は各nの2,000反復におけるstrict比較。同率は全nで0件。

| n | 保存順の累積平均 | bootstrap SE | bootstrap 2.5〜97.5分位範囲 | PR < PF | S2を引かない反復 |
|---:|---:|---:|---:|---:|---:|
|4|72.936961|2.812868|66.341379〜77.347853|41.00%|87.10%|
|8|72.731331|2.075113|67.533711〜75.390127|32.70%|76.65%|
|12|70.594493|1.685734|68.062930〜74.645746|28.90%|66.25%|
|16|70.736720|1.461284|68.419109〜74.133824|25.30%|56.35%|
|32|71.364534|0.999307|69.226705〜73.221228|15.00%|34.00%|
|64|72.554939|0.739718|69.759608〜72.715282|8.40%|12.20%|
|128|71.505101|0.515282|70.177587〜72.184873|1.80%|1.20%|
|256|71.567272|0.366785|70.399575〜71.866095|0.25%|0.00%|
|512|71.177820|0.259077|70.640380〜71.648081|0.00%|0.00%|
|1024|71.028016|0.188795|70.759724〜71.494917|0.00%|0.00%|
|1108|71.128345|0.174963|70.789614〜71.468365|0.00%|0.00%|

小標本では平均費用が固定PFの上下へ変動する。n=4では820/2,000（41.0%）、n=32では300/2,000（15.0%）、n=128では36/2,000（1.8%）、n=256では5/2,000（0.25%）がPR低費用。n=512/1,024/1,108は全て0/2,000だった。**0/2,000を確率0や母平均保証へ置き換えない。** 反復割合の刻みは0.05 percentage pointであり、2,000反復由来のMC誤差もある。割合のMC SEはCSVに保存したが、0/1でのplug-in MC SE=0も確実性を意味しない。

全保存平均はPFより約1.02×10^6高く、n=1,108のbootstrap SEは約0.175×10^6。n=128の2.5分位点はPFを上回る一方、分布の低い側にはPR低費用反復が残る。n=128を普遍的な必要標本数や停止基準として採択しない。

## ε=0.0005：保存経験分布内の順位は安定

単位は平均・SE・分位範囲すべて `10^12 G_RZ`。固定PFは7.856558002614。同率は全nで0件。

| n | 保存順の累積平均 | bootstrap SE | bootstrap 2.5〜97.5分位範囲 | PR < PF | S2を引かない反復 |
|---:|---:|---:|---:|---:|---:|
|4|4.934129|0.062015|4.846636〜5.083931|100.00%|67.60%|
|8|4.974025|0.043632|4.879007〜5.048731|100.00%|46.55%|
|12|4.935321|0.035436|4.894982〜5.031274|100.00%|32.50%|
|16|4.944710|0.030474|4.902852〜5.022782|100.00%|22.60%|
|32|4.960650|0.021865|4.920889〜5.005782|100.00%|4.20%|
|64|4.966004|0.015722|4.933090〜4.994910|100.00%|0.50%|
|78|4.964067|0.013939|4.936560〜4.991352|100.00%|0.05%|

全nで2,000/2,000がPR低費用。さらに、保存された単一trajectoryの最大費用5,231,154,106,652自体が固定PFの7,856,558,002,614より小さい。従って**この保存標本の支持だけから作る任意の平均**はPFより小さい。これは経験分布内の決定的性質であり、未知の真の支持や未観測希少層への主張ではない。

bootstrap SEはn=4の約0.0620×10^12からn=78の約0.01394×10^12へ縮小した。全保存標本のpaired SEは0.01414×10^12であり、経験分布と不偏標本分散の違い、有限2,000反復の誤差によってbootstrap SEとは少し異なる。

## 観測S2と費用変動、未観測層

| ε | S0_ONLY | S2_SINGLE | S2_MULTI | S4_PRESENT | S0_ONLYの平均G | S2_SINGLEの平均G |
|---|---:|---:|---:|---:|---:|---:|
|.05|1,071|37|0|0|70,995,591.08|74,971,030.73|
|.0005|71|7|0|0|4,954,882,624,002.25|5,057,220,768,892.57|

いずれも観測S2_SINGLE群の費用平均はS0_ONLY群より大きい。しかしS0_ONLY内にも費用の変動がある。特にε=.05ではS0_ONLY平均自体が固定PFを上回っており、順位の逆転を希少S2だけの作用と結論しない。

bootstrap平均と抽出中S2割合の相関は、ε=.05で約.084〜.174、ε=.0005で約.202〜.266の正の値だった。図と層別表はこの条件付きの関係を示す。因果効果、真のlawの層確率、未観測層への外挿を意味しない。

保存順で初めてS2を含むのはε=.05で80本目（replica79）、ε=.0005で21本目（replica20）。従って小さいprefixのSEや平均だけでは全保存経験分布の希少層を反映できないことがある。bootstrapでS2を引かなかった反復はn=4で87.10%／67.60%、n=32で34.00%／4.20%。この割合も真のlawの無event確率ではない。

元Pの個別RTE order2 eventは37／7個、複数S2 trajectoryは未観測。K2条件のS4は有限cutoff支持外、K4条件のS4は支持内だが未観測。今回の公開projectionからtrajectory層ラベルを再利用し、raw event列・law確率を再構成していない。元のevent件数は元Pの`event_label_audit.json`を保持した。未観測S2_MULTI/S4の平均・分散はunknown/nullのままとし、0費用の層として補完していない。

## 再利用・監査・再現性

- 費用復元：既存`h8_rare_tail_selection.paired_cost`と`h4_dp_contract.evaluate_pairs`を直接使用。既存H8 bootstrapのtrajectory単位・両軸共通indexの意味論を保持し、可変nの抽出・集計だけをNumPyでベクトル化した。既存選択後bootstrapは固定Nと候補再選択を扱うため、その目的外のランキング処理は流用しない。
- 描画：既存manuscriptの色・CSV exporter・書式・PNG/SVG/PDF保存パターンを再利用し、標本数・分布・観測S2を表示する部分を追加した。旧図は変更しない。
- 人工9検査PASS：paired共分散保持、累積順、4種nの再現性とexact順位、復元不一致拒否、overflow拒否、非整数PF拒否、one-shot二重消費拒否。
- mean/分散/SD/SEの保存値完全一致2条件、paired累積恒等式1,186行、18群×2,000のexact総費用・順位・event件数を監査。保存bootstrap出力のSDと分位点は別のscalar式で照合した。監査中の再標本化は0。
- 図は4種×PNG/SVG/PDFの12成果物。PNGの構造と4図の視認性をprimary agentが確認した。
- SOURCE・元P全公開入力・元P SOURCE25ファイル・使用runtimeのSHAは実行前後で不変。監査はこれらに限定し、他worktreeのHEAD/status更新を拒否条件にしない。
- ベクトル化で全反復を約10秒で処理した。新しいworker poolを起動する必要はなかった。CPU/RAM/worker/wallに恣意的capを設けず、既存affinity/quotaを継承し、共有設定・共有依存関係・他ユーザーのprocessを変更していない。図cacheだけは専用worktreeのignored領域を使用する。

## 成果物

- [条件・全標本再現・限界](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/result.json)
- [全18条件の平均分布・SE・分位点・順位割合](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/bootstrap_summary.csv)
- [各paired費用と全累積列](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/trajectory_and_cumulative.csv)
- [36,000個の保存再標本化平均・exact総費用・層件数](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/bootstrap_samples.csv.gz)
- [観測層別費用](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/observed_strata.csv)
- [抽出S2件数別の費用分布](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/bootstrap_S2_groups.csv)
- [実行前固定契約・入力/runtime SHA](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/contract.json)
- [ユーザー承認receipt](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/approval_receipt.json)
- [実call・実行receipt](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/execution_receipt.json)
- [構造・数値監査](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/numerical_audit.json)
- [数値結果と図のSHA](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/output_manifest.json)
- [mandatory STOP](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/terminal.json)

- ε=.05の収束・変動幅・PF順位：[PNG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps005.png)／[SVG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps005.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps005.pdf)
- ε=.0005の収束・変動幅・PF順位：[PNG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps00005.png)／[SVG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps00005.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_convergence_eps00005.pdf)
- 代表nの平均費用分布：[PNG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_bootstrap_distributions.png)／[SVG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_bootstrap_distributions.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_bootstrap_distributions.pdf)
- 観測S2割合と費用：[PNG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_observed_S2_cost.png)／[SVG](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_observed_S2_cost.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19/h4_mc_observed_S2_cost.pdf)

## 残る限界と終了

保存P標本に条件付けた経験的診断であり、未観測希少eventの影響、母平均、厳密CI、同時coverage、実power、必要nの保証を与えない。累積平均と全保存平均は同じ標本を共有する。PF候補とPR候補・shot/bias/compilerを固定した比較であり、候補再選択の不確かさや別条件への一般化は扱わない。bootstrapで同じ保存trajectoryを複数回引いても、新しい物理的eventを観測したことにはならない。

今回の新規成果は専用branchへlocal commitする。公開pushの具体的な対象承認は未確認のためpushは行わない。SOURCE/契約/結果を含むbranch全体の公開には、契約中のローカルruntimeパス情報も含めた範囲確認が必要となる。既存Figure 2や論文主張の正式採択・統合へ自動進行しない。

次の担当：**ユーザー**が必要なら今回の結果の正式採用と公開対象を判断する。**Codex**は追加科学実行・公開pushを行わずmandatory STOPを維持する。
