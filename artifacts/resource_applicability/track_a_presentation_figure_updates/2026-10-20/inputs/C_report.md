# H4：PR単独のMC平均費用変動の整理（2026-10-19）

**主図を、PR平均費用推定値の揺れ、nに対する標準誤差の減少、観測S0/S2と費用の関係に再構成した。数値は既存MC検証commit `defa5f6dbc72910c0a46ae2fef304a584ae408e0` の保存列を転記し、新規sampling・compile・bootstrap・統計解析は全て0。原SOURCE・原結果・原図・grant・ledger・marker・STOPは保全した。**

今回の図表をこの問いに対する表示入口とする。旧MC報告は、当時のPF比較も含む解析来歴としてそのまま保持する。今回、PF順位やPRがPFより安くなる割合を新しい主図・表示用数値表には含めていない。

## 1. 示している費用と誤差

linear H4、隣接1 Å、STO-3G、tol-only 1e-8 DF rank10、T=0.8、P=0、通常signed bias、allowance倍率1。ε=.05は`B2_p3_q1_2nd_r4_K2`（L_D=3/q1/r4/K2/δ=.8）、ε=.0005は`B2_p2_q8_2nd_r2_K4`（L_D=2/q8/r2/K4/δ=.1）。コンパイラは元Pの固定Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、`symmetric_directional`。この表示整理ではcompilerや科学providerを起動していない。

trajectory iの保存費用指標は

`G_i = N_cos × RZ_cos,i + N_sin × RZ_sin,i`。

**決定論的prefix・確率的RTE tail・full cosine/sine Hadamard wrapperを含む、保存shot×compiled RZのPR費用**である。tail単体の費用ではない。コンパイル済みfull wrapperの費用からprefix/tailの加法的な別会計を新設していない。

nはMC平均を推定するpaired trajectoryの本数であり、shot数ではない。cosine/sineの保存shotはε=.05で12,305／8,258、ε=.0005で174,919,953／95,102,971に固定されている。nに応じたshot再計算はない。

図の対象は`平均費用推定値 G_bar_n`とそのSEである。**n本分の合計費用や、計算サーバーのwall time/CPU費用の変動を示しているのではない。** n倍した合計費用の数値や分布は今回算出・表示していない。

## 2. 全保存標本の参照値

次表は元MCの`result.json`内の値の転記。参照平均は真の母平均ではない。丸め表示に加え、表示用CSVには元のexact有理数表記も保持した。

| ε | 保存N | PR平均 G_RZ | 全保存標本のpaired SE |
|---|---:|---:|---:|
|0.05|1,108|71,128,344.93|174,917.74|
|0.0005|78|4,964,066,816,492.41|14,140,828,222.91|

## 3. 標本数nで平均費用がどれだけ揺れるか

[主図1：平均費用と保存分位幅](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_mean_cost_vs_n.png)（[SVG](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_mean_cost_vs_n.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_mean_cost_vs_n.pdf)）。

青い線は保存済みの2,000 bootstrap平均の平均、帯は保存された2.5〜97.5分位点。緑の線は同じP cohortを保存replica順に先頭n本使った累積MC平均。灰色線は全保存N本のPR平均である。全保存経験分布からn本を復元抽出したbootstrapと、先頭n本の累積列を区別する。帯を母平均の信頼区間とは呼ばない。

以下の表は`bootstrap_summary.csv`の保存列からの転記であり、SEや分位点を再計算していない。「変動幅」は保存分位点の下端・上端として示し、幅の差分や相対誤差の新規数値を作っていない。CSVは文字列をそのまま転記し、次表と縦軸だけ単位表示を整えた。

### ε=0.05（平均・SE・分位点の表示単位：10^6 G_RZ）

| n | 累積MC平均 | bootstrap平均の平均 | bootstrap SE | 保存2.5〜97.5分位範囲 |
|---:|---:|---:|---:|---:|
|4|72.936961|71.054812|2.812868|66.341379〜77.347853|
|8|72.731331|71.135240|2.075113|67.533711〜75.390127|
|12|70.594493|71.148360|1.685734|68.062930〜74.645746|
|16|70.736720|71.112620|1.461284|68.419109〜74.133824|
|32|71.364534|71.145618|0.999307|69.226705〜73.221228|
|64|72.554939|71.135160|0.739718|69.759608〜72.715282|
|128|71.505101|71.133743|0.515282|70.177587〜72.184873|
|256|71.567272|71.125870|0.366785|70.399575〜71.866095|
|512|71.177820|71.139840|0.259077|70.640380〜71.648081|
|1024|71.028016|71.131795|0.188795|70.759724〜71.494917|
|1108|71.128345|71.126415|0.174963|70.789614〜71.468365|

小さいnでは平均費用の分位範囲が広く、保存順の累積平均も上下する。保存分位範囲はn=4で66.341379〜77.347853、n=1,108で70.789614〜71.468365（いずれも10^6 G_RZ）。これは全保存平均付近での経験的な変動幅を示す。累積平均がnごとに単調に近づくことは要求されない。

### ε=0.0005（平均・SE・分位点の表示単位：10^12 G_RZ）

| n | 累積MC平均 | bootstrap平均の平均 | bootstrap SE | 保存2.5〜97.5分位範囲 |
|---:|---:|---:|---:|---:|
|4|4.934129|4.964167|0.062015|4.846636〜5.083931|
|8|4.974025|4.963164|0.043632|4.879007〜5.048731|
|12|4.935321|4.964913|0.035436|4.894982〜5.031274|
|16|4.944710|4.963543|0.030474|4.902852〜5.022782|
|32|4.960650|4.964495|0.021865|4.920889〜5.005782|
|64|4.966004|4.964189|0.015722|4.933090〜4.994910|
|78|4.964067|4.964192|0.013939|4.936560〜4.991352|

保存分位範囲はn=4で4.846636〜5.083931、n=78で4.936560〜4.991352（10^12 G_RZ）。この条件でもnを増やすと、PR平均費用推定値の保存bootstrap幅が狭くなる。

## 4. SEはnの増加でどう減るか

[主図2：nとSEの関係](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_standard_error_vs_n.png)（[SVG](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_standard_error_vs_n.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_standard_error_vs_n.pdf)）。左右に2精度、両対数軸で表示する。

- 青線：保存`bootstrap_SE`。既存2,000反復の平均費用のSD。
- 灰色破線：保存`conditional_empirical_SE`。全保存経験分布のSD/√nという既存列であり、新たに理論曲線を計算・fitしていない。
- 緑点線：保存`cumulative_SE`。先頭n本のpaired標本SDから元解析が求めた累積MC平均のSE。

保存bootstrap SEはε=.05でn=4の2,812,868.11からn=1,108の174,963.14へ、ε=.0005でn=4の62,015,416,518.94からn=78の13,938,622,601.99へ小さくなる。元解析で示した「概ね1/√nの縮小」を、PR単独のSE図として見やすくした。指数の新規推定、回帰fit、必要nの設計はしていない。

累積MCのSEは推定に使う標本自体がnごとに変わるため、各点で単調減少するとは限らない。例えばε=.0005の保存累積SEはn=8から12で増える。この挙動を平滑化や補正で消していない。

全保存標本の不偏標本SDに基づくpaired SEと、全保存経験分布を固定したbootstrap SEは定義が異なり、有限2,000反復の揺れもある。前者・後者を同一値として転記しない。`full_sample_plugin_SE`等の保存列も表示用CSVに残した。

## 5. 観測eventとPR費用の関係

[図3：保存S0/S2群の費用](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_observed_strata_cost.png)（[SVG](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_observed_strata_cost.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_observed_strata_cost.pdf)）。点は保存群平均、縦線は**個々の保存trajectory費用のmin〜max**であり、平均のSE・CIではない。

| ε | 群 | 保存本数 | 群平均 G_RZ | 保存trajectory費用のmin〜max |
|---|---|---:|---:|---:|
|.05|S0_ONLY|1,071|70,995,591.08|65,287,525〜87,413,313|
|.05|S2_SINGLE|37|74,971,030.73|66,130,608〜98,805,215|
|.0005|S0_ONLY|71|4,954,882,624,002.25|4,654,655,163,912〜5,231,154,106,652|
|.0005|S2_SINGLE|7|5,057,220,768,892.57|4,957,080,838,792〜5,209,822,295,656|

観測S2_SINGLE群の平均費用はS0_ONLY群より大きい。一方、両群には費用範囲の重なりがあり、S0_ONLY内にも変動がある。**S2件数だけでfull PR費用は決まらない。** 群の値はprefix・tail・wrapperを含むfull PR費用であり、S2 event単独のコストとは解釈しない。

[図4：保存bootstrap内のS2件数別の費用](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_saved_S2_groups.png)（[SVG](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_saved_S2_groups.svg)／[PDF](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_saved_S2_groups.pdf)）。n=4、32、全保存Nについて、`bootstrap_S2_groups.csv`に保存された群平均・min・maxを直接描画する。横軸は復元抽出されたS2_SINGLE trajectoryの件数であり、異なるnの件数を同じ割合とみなさない。各群の反復数も表示用CSVにそのまま保存した。反復数の少ない群の凸凹は新しい因果効果の証拠ではなく、群別min〜maxはCIではない。

複数S2 trajectoryは両精度で未観測。K2のS4は支持外、K4のS4は支持内だが未観測。未知の費用・分散は空欄/nullを保持し、0費用の点を追加していない。元law確率、真の希少層平均・分散、tail単独費用は今回求めていない。

## 6. 転記・構造検証と成果物

- 元MC commit：`defa5f6dbc72910c0a46ae2fef304a584ae408e0`。元MC科学SOURCE `78352b861b3047f6af2d432535e23cc88b439bbe` は不変。
- 新renderer：`scripts/render_h4_pr_cost_variability.py`。元manuscriptの描画helper/色/書式と元H4 MCの線・帯・PNG/SVG/PDF exportパターンを流用した。元MCの実行・監査関数やraw trajectory、個別bootstrap標本ファイルは呼び出していない。
- `bootstrap_summary.csv`の18行、`observed_strata.csv`の8行、`bootstrap_S2_groups.csv`の236行について、選択した各フィールドの**文字列完全一致**を検証。全保存参照値2行も元JSONからの転記一致を確認した。
- 入力4ファイルは指定commitのGit blob SHAと一致。元P/MCのSOURCE・全公開結果・全原図・grant/ledger/STOP等のSHA不変を確認した。既存の科学実行を再開・再監査したものではない。
- 新sampling/compile/bootstrap/統計解析は0。再標本化・平均/SE/分位点/相関/ヒストグラムの再集計・回帰fitの経路を新rendererに設けていない。保存値の数値表現・単位表示・描画以外の数値処理はない。
- 4図×PNG/SVG/PDF、CSV4表、来歴・構造監査を保存。PNG構造と図の視認性をprimary agent自身が確認した。別AIの独立レビューではない。

転記表：[n別PR平均・SE・保存分位点](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_bootstrap_summary.csv)、[全保存参照値](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_full_sample_reference.csv)、[観測層別](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_observed_strata.csv)、[保存S2群別](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/pr_bootstrap_S2_groups.csv)。[表示来歴と原本SHA](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/display_provenance.json)、[構造監査](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/structural_audit.json)、[成果物SHA](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/artifact_manifest.json)、[今回のSTOP](../../artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19/STOP.json)。

## 7. 残る限界と終了

保存P標本に条件付けた経験的表示であり、真の母平均・分散や未観測希少eventを保証しない。保存分位幅を厳密CIと呼ばず、全保存平均との一致を独立した母平均検証と呼ばない。費用の平均推定誤差をn本分の合計費用の揺れと混同しない。図の線の接続は表示であり、新しいnへの推定・fit・外挿ではない。

数値の追加処理を要する事項は発生しなかった。H2/H6/H8への拡張、詳細科学レビュー、正式なclaim採択、最終evidence freezeは行わない。公開は承認済み範囲だけとし、今回の表示整理と元科学解析の来歴を区別する。mandatory STOP。

次の担当：**ユーザー**が必要なら今回のPR単独表示の利用・採用を判断する。**Codex**は追加科学実行へ進まず、mandatory STOPを維持する。
