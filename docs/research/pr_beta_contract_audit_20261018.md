# PR資源評価：β契約・状態仮定・shot接続の技術監査

2026-10-18 JST。利用者承認のA-1〜A-5。主担当Codex自身による静的照合・人工検査であり、別担当の独立科学レビューではない。

**旧H4のβ→座標幅→α付きshot→RZの実装経路と、WP11の比較条件は、宣言されたモデルの下で接続している。既存WP04物理gridには採用PR全18 roundのパラメータ対応があるが、preparation閾値とhashが異なるため、WP11と同一条件の検証として採用できない。保存値の比較が通っても、状態同一性・半径証明書・全round測定成功率の証明とはしない。極小正の座標幅で例外処理が漏れる技術的不具合を人工入力で再現した。既存結果を修正する根拠や、PRの厳密な資源優位とは判定しない。**

科学資料の固定snapshotは `f29eafbf4412193e8e8660d9f051889c2cba684b`。監査branchの起点はα監査を含む `e62ddb59173d100c113be7a33581515d78f152ee`。mainや並行worktreeのHEADを科学入力の代用にしていない。

- [固定79 blob・SHA256・検査記録・所見](../../artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18/technical_validation.json)
- [系／候補／δ／compiler／roundの保存契約表](../../artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18/beta_contracts.json)
- [再検査script](../../scripts/audit_pr_beta_contracts_20261018.py)
- [停止記録と成果物manifest](../../artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18/README.md)

本監査は固定JSONの取得・hash照合、保存予算の和と不等号、人工fixture上のSOURCE関数に限定する。実分子NPZ、private runtime、grant原本は開かない。新しいsignal生成・sampling・二項確率・Monte Carlo・shot再導出・費用再会計・実候補再最適化・native build/compileは0。既存SOURCE、結果、grant、ledger、marker、過去STOPと他チャットの計算は変更していない。

## A-1：SOURCEと保存結果の来歴

共通系はlinear H4、隣接1.00 Å、STO-3G、8 qubit、保存DF rank12。以下の系列を別の条件として扱った。

|系列|固定結果と条件|SOURCE・compiler・費用の扱い|
|---|---|---|
|初期配分感度|2026-09-01の5β×2α、L_D=3、δ=.1、r4/K2、q1/2/4/8|opt1、q1/2/4直接単価とq8 proxy。暫定100倍guardは理論保証ではない。10配分の保存契約を別欄に保持|
|WP04|`wp04_finite_rte_statistical_ablation_v1.json`、42条件、δ=.02、18 round|元8組のdataset/proxy計16 fileのSHAを照合。同一opt1 compiler。schedule目的、β・αのfactorialを保ち、寄与率を独立に足さない|
|WP01-D/C07 v2|`wp01d_c07_full_scope_optimization_compute_v2.json`、L_D3/12×δ.01/.02、計4候補|WP05a/b/bRの保存SHAとopt1 compilerが一致。粗格子・局所格子・境界refinement。旧v1は履歴専用|
|WP11 final|`wp11_all_r_opt2_coherent_analysis_20260925_065827.json`、同4候補|51/51/0は保存監査値。fresh32置換と旧8標本を区別。opt2でβ・α・整数shot・scheduleを共同選択。最終`065827`版を使用|
|4 round物理測定・分枝|2026-09-20、δ=.1、r4/K2、β=(.02,.02,.36)、q≤8|旧物理状態上の二項・fresh-IID・分枝検証。WP11の18 roundの成功率へ転用しない|
|長roundの行列証拠|旧δ候補別scheduleとWP04のδ=.02、108点grid|行列・物理／control状態の保存結果。WP11の測定契約とは分離|

13件のcontent fingerprintと、物理gridのvalidation fingerprintは固定blobからのmetadata hash検査で一致した。WP04 gridのSHA・fingerprint、WP01 providerのSHA、WP11初期／追加aggregateのfingerprint等9件の結合も一致した。これらはraw checkpointや科学計算を再生成した検証ではない。

WP04／WP01 v2／WP11の計50 scenario、904 round行を契約表へ転記した。保存された各roundのPF/RTE使用量と予算の比較、正の座標幅、正の整数shot欄は全50 scenarioで整合する。shot式や総費用を実保存値に再適用していない。

保存provenanceのSOURCE hash31件を固定snapshotの該当fileと個別照合した。27件一致。異なる4件は、旧WP01 v1のrunner/moduleと、WP05aの`df_partial_s2.py`／`research_direction_full_scope.py`である。後続SOURCEを古い実行SOURCEとみなさず、不一致を記録した。最終WP01 v2とWP11が直接保存するSOURCE hashにはこの不一致はない。ただし、その一致だけで未記録の全依存関係・旧runtimeの完全な再現性まで証明しない。

実入力snapshotのpath／SHA宣言は保存metadataとして保持する。NPZ実体を開いて照合しておらず、保存law・raw checkpoint・依存環境の完全closureを検証済みとはしない。過去のdirty-worktree生成・CI未再現という証拠区分も保持する。

## A-2：位相誤差と物理状態への適用可能性

SOURCEの主な経路は次である。

`evaluate_coherent_extension_analysis` → `_grid_search` → `_evaluate_profile` → `evaluate_ablation_scenario` → `_select_schedule` → `_evaluate_pair_round` → `evaluate_rpe_round_candidate`。

|対象|SOURCEと意味|判定|
|---|---|---|
|PF位相|`epsilon_pf=C*delta**2`、`beta_pf=t_m*epsilon_pf`。Cは`same_snapshot_paper_d6_empirical_coefficient`、`is_rigorous_bound=False`|`EMPIRICAL_PROXY`。実行したH4 gridの観測と、全領域の厳密上界を区別|
|RTE位相|finite-RTE合成残差ε_zに対し`asin(ε_z/reference_radius)`。ε_z<reference_radius、正の観測半径が必要|解析式は仮定付きの`ANALYTIC_BOUND`。実Hamiltonian・状態・浮動小数点での数値上界まで自動認証しない|
|状態|会計はeffective partial-S2の厳密な固有状態、減衰前半径1を仮定|`ASSUMED`。full-DF-H基底状態との同一性は未証明|
|branch|`alias_free_target_energy_branch_assumed`と総β<π/3。計算可能域β≤π/2とは別|alias-free性は仮定。円周上の位相差と、非aliasなenergy枝の同定を区別|
|誤差比較の基準|物理signal検証はPF signal対exact signal、finite-RTE対PF signalを別々に比較。複素位相差は`abs(angle(z1*conj(z2)))`|保存された円周上の比較。unwrapした全energy誤差の証明へ読み替えない|

### 既存物理gridで確認できる範囲

WP04の`wp04_ld3_full_schedule_signal_grid_v1.json`は、δ=.02、L_D3、q1〜131072、r1/2/4/8/16/32、K2の108点を収録する。WP11が採用したPRの18行は全てこのgridと(q,r,K,δ)で結合できた。ただし、以下のpreparation差があるため、これはパラメータIDの対応であり、同一SOURCE条件の物理検証を意味しない。

異なるpreparation由来の18行を対応付けた保存値の比較では次を満たす。移送誤差の上界や再会計は計算していない。

- physical DF ground state上のPF位相誤差 ≤ WP11のβ_PF。
- physical radiusを用いたfinite-RTE位相上界 ≤ WP11のβ_RTE。
- 保存された観測raw signal半径 ≥ WP11のモデル半径下界。

最後のq=131072/r32/K2では、保存PF位相誤差は`0.013988443007836207` rad、保存RTE位相上界は`3.0865990630179466e-9` rad。WP11予算はそれぞれ`0.014050012886294481`、`9.899584668737594e-8` radである。これらは再計算値ではない。

grid全体の保存最大physical reference radius偏差は`1.7101176030820398e-10`、control固有状態とphysical状態の保存overlapは`0.9999999999396312`。数値的に近いという有限集合の観測であり、半径1や状態同一性の厳密証明ではない。numerical_atol=`1e-10`という旧検査条件を保持する。半径が1をわずかに超える保存値も、丸めを含む数値観測として扱う。

**B08：preparationの不一致。** gridは`coefficient_atol=1e-12`、`preparation_hash=89145d83…`で、保存metadataには108項を落としたことと`threshold_operator_error_bound=4.8294649659693557e-14`がある。旧WP04のL_D3費用datasetは同じHamiltonian／partition hashだが`preparation_fingerprint=50c77f54…`。benchmark SOURCEではこの欄を`request.preparation.preparation_hash`から取得するため、同種のhashの相違である。WP11会計SOURCEの`prepare_df_partial_s2`呼出しは閾値を省略し、固定SOURCEの既定値は`0.0`。微小だから同じと判断せず、`MODEL_MISMATCH`として保持する。保存state・operatorの移送誤差の大きさや費用影響は未評価。

gridの測定契約はβ=(.015,.005,.38)、一様α。WP11のβ・費用重みαとは異なる。この18行対応は補助資料の所在を特定するが、preparation差を解消しない限り、WP11の物理半径検証の完了とは扱わない。WP11 shotによる厳密二項確率・全round測定・分枝復元も今回確認していない。L_D12側の物理状態適用可能性も、このL_D3 gridの対応だけでは証明しない。

## A-4：β_statからshot・費用への接続

旧RPEはrawなHadamard outcome Y∈{−1,+1}の位相を扱う。宣言した条件の下で

\[
\rho_{\rm lower}=a(1-\epsilon_z),\qquad
\epsilon_{\rm coord}=\rho_{\rm lower}\sin\beta_{\rm stat}/\sqrt2,
\qquad
N_b=\left\lceil 2\epsilon_{\rm coord}^{-2}\log(2/\alpha_b)\right\rceil.
\]

fresh-IID trajectory／各軸内の独立bounded outcomeが条件である。軸間独立性はunion boundに不要。複素円板の半径√2ε_coordが正のsignal半径に対する角度誤差を制御する。位相予算βと失敗確率αは別量で、βの和だけでは成功確率を決めない。

`_weighted_alpha`は軸費用g_bと座標幅から重み`2*g_b/epsilon_coord**2`を作り、総αへ正規化する。各反復でscheduleを再選択し、反復後のscheduleについて再びα整合を確認する。整数shotの大域最適解を証明する規則ではない。

各roundで`N_cos*g_cos+N_sin*g_sin`を作り、全roundを合計する。単価は状態準備なしfull Hadamard wrapperのcompiled RZ。古いq1/q2 affine較正と、独立holdout、後半巨大qへのモデル外挿を区別する。n8/n32のclassical compile標本は量子shot数の倍率ではない。

現行Track Aのnative会計はcorrected outcome `X=B_corr*Y`、headroom hと`B_budget**2`を含む別のshot契約である。旧RPEはraw半径の減衰を座標幅に織り込む。この旧式にTrack AのB²を追加したり、Track Aのεをβ_stat(rad)として扱ったりしない。[既存α監査](pr_alpha_contract_audit_20261018.md)を契約の区別にのみ参照した。経験的u/log_B_margin等の数値条件は旧RPEへ移送しない。

## A-3：境界refinement・反復・例外

粗格子32×32（tail-freeは32×1）、局所格子25×25（同25×1）、費用上位20 seed、各最大8回の境界refinementをSOURCEで確認した。次の予算は現在選択された使用量に

`max(1e-12, abs(beta_used)*1e-10)`

を足して作る。`_evaluate_profile`はβ_statを再計算し、候補・shot・α・scheduleを再評価する。非収束は除外し、粗／局所／境界で得た実行可能候補を費用で比較する。使用量が変わったscheduleを古い数値のまま採択する経路は、確認したSOURCEと人工fixtureでは検出しなかった。格子外の大域最適性は保証しない。

`_select_schedule`のtie-breakは目的費用、compiled費用、(r,K)。grid全体の同費用候補はPythonの安定な最小選択に従う。反復停止の絶対差`1e-13`、α整合許容差`1e-8`はモデル誤差のheadroomではない。

### 人工検査38件

標準libraryだけで固定SOURCEの指定AST定義を取り出し、分子・科学providerをmockへ置き換えた。候補評価の実関数、投影、schedule選択、α反復と最終整合判定、粗格子→局所→境界controllerを検査した。既存のULPテストと同じ保証境界を確認し、更新schedule・反復上限・最終schedule変更による見かけの収束の取消しも検査した。元の依存付き科学テストsuiteを実行したという意味ではない。

|分類|確認した内容|
|---|---|
|通常経路|人工の条件付きcertified、empirical_screening、branch不成立not_certified、非finite/負予算・不正αの拒否、外向き予算加算、整数切上げ|
|制御経路|更新後候補の制約照合、不適格候補除外、tie-break、tail-free RTE=0、α反復／最終整合／非収束除外|
|不具合の再現2件|`epsilon_coord=1e-200`、および人工候補`beta_stat=1e-200`で、二乗が0となり`ZeroDivisionError`|

不具合B03：`_hoeffding_shots`は座標幅自体の正・finite性を検査するが、`epsilon_coordinate*epsilon_coordinate`のunderflowを処理しない。上位候補関数が捕捉するのは`OverflowError`であるため、明示的不適格候補へ分類されず例外が伝播する。偽PASSではなく処理中断の経路である。38件PASSは期待した挙動／不具合の再現に成功したという意味で、不具合解消のPASSではない。

保存された採用行にこの例外の影響があるとは判定していない。科学SOURCEは変更しておらず、修正する場合は新SOURCEで、極小正数を明示的に扱う技術変更として切り離す。保存値のshot・費用を後から変更しない。

### feasibleと保証状態

会計の数値的実行可能性には`1e-15`の比較許容差がある。厳密PFモデルの場合のcertificationは許容差なしで再比較し、総β<π/3を必要とする。経験的PFモデルでは`empirical_screening`に留まる。

WP04/WP01/WP11のround投影は`feasible`で候補を選ぶが、`guarantee_status`／`certification_reasons`／`assumptions`を保存行に引き継がない。このため、`all_rounds_feasible=True`からcertifiedを推論できない。元のWP11 scopeにはfinal total cost／科学的優位を主張しないことが保存されており、それを保持する。既存行へ後付けの証明書欄は追加していない。

50 scenarioの予算和は実行可能性の許容差内で一致する。一方、保存binary64値を厳密な有理数として加え、保存総予算との大小を許容差なしで調べると、βで6/50、αで20/50がわずかに上回る。WP11採用δ=.02ではL_D12のβ、および両候補のαがこの区分に入る。これは厳密予算保証への昇格を妨げる数値上の注意であり、費用順位が変わるとの判定ではない。丸めや予算を変更してPASSにしない。

## A-5：比較公平性と支持できる主張

WP11の両側は総β=.4 rad、総α=.05、Qiskit1.3.0、opt2、basis rz/sx/x/cx、transpiler seed17、topology/backendなし、状態準備P=0。同じ候補δ集合と目的energy精度に由来するround範囲で各々を再最適化する。δ=.02の採用は両側18 round・q_max131072である。

L_D12のtail-free決定論はr=K=0、β_RTE=0。不要なRTE予算を統計へ回す設定は、総予算が共通なら不公平ではない。PRと同じ成分配分を強制する理由はない。WP04の対称予算再配分と、WP01 v2／WP11での個別再最適化を区別した。

|結果|支持する範囲|支持しない範囲|
|---|---|---|
|4 round測定・分枝|旧固定条件の物理signal・二項測定・分枝復元|WP11の最適β／全18 round測定成功率|
|WP04物理gridと今回の18行照合|異なるpreparationを含む既存gridの所在と保存数値比較|WP11と同一preparationの物理検証、厳密下界、WP11のα・shotでの測定成功率|
|WP11費用予測|旧H4の同opt2・P=0・経験的PFモデル条件付き点比較|q>32の直接compile検証、実backend、状態準備込み総資源、頑健な優位|
|現行Track A主表|保存状態上の有限時間complex signal資源比較|旧H4最適βのH6/H8 RPEへの移送、energy精度の成功保証|

保存されたδ=.02点費用はPR `1.263314e12`、決定論 `1.327822e12` RZ、shotは13,588対11,162。点のPR削減4.858%に対して、local・discrepancy・transfer区間は重なるという既存結論を保持する。PRの保存last-three-round費用比率は`0.9139324437497927`。これは後半qのprovider外挿を吟味する理由になるが、β配分の不具合という意味ではない。今回、この数値・区間・順位を再計算していない。

## 所見と必要な後続作業

|ID|判定|影響と対応|
|---|---|---|
|B01|CONDITIONAL / OBSERVED_FIXED_GRID|18行はパラメータID対応。preparation差B08を保持し、厳密固有状態・半径1・alias-freeを仮定として扱う|
|B02|INSUFFICIENT_EVIDENCE|保存round行単独では保証分類を復元できない。引用時にSOURCE仮定・経験的分類を添える。新しい数値解析は不要|
|B03|IMPLEMENTATION_BUG|極小正の統計予算でunderflow例外。新SOURCEでの技術修正候補。保存採用結果の破損や再会計の必要性は認定しない|
|B04|NUMERICALLY_CONSISTENT / CONDITIONAL_PROXY|境界更新・再選択・α再確認は人工検査で整合。微小数値余裕をモデルheadroomと呼ばない|
|B05|CONSISTENT|WP11の総予算・compiler・費用範囲は共通。tail-free側RTE=0を保持|
|B06|CONDITIONAL|binary64予算和の厳密比較超過を記録。厳密証明書としての再利用をしない。科学結果の後付け補正なし|
|B07|INSUFFICIENT_EVIDENCE|歴史SOURCE不一致4件と未確認closureを保持。旧v1・opt1と最終opt2を混合しない|
|B08|MODEL_MISMATCH|物理gridの閾値1e-12／108項dropとWP11会計既定閾値0が異なる。preparation同一性を認定せず、既存行を補助資料としてのみ扱う|

追加数値解析Bを自動実行する必要はない。現状を条件付きH4 benchmarkとして報告して止める選択肢を保持する。

### リポジトリ検査の制約

監査scriptの構文・標準library限定import、成果物の件数とSOURCE hash、保存gridの結合、報告内参照、`git diff --check`は通過した。既存`check_validation_manifest.py`の全体検査は、manifestに多数の既存拡張root keyがあるため失敗する。起点e62ddb59…でも同じerrorとなることを照合した。既存項目やvalidatorを変更して失敗を隠していない。

今回の新規result／artifact entryだけをcore schemaへ射影した限定検査は通過し、既存manifest値の保存も確認した。これは全体manifestのPASSではない。[基点と現在の失敗・限定検査記録](../../artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18/repository_checks.json)に区別して保存する。この既存schemaの整備は本β監査の範囲外で、ここから科学計算や他系列の修正を開始しない。

- **Codex・提案／未実施**：B03を修正する場合、新SOURCEでunderflowの明示的例外処理を実装し、人工検査する。既存結果・SOURCE・停止記録は保持する。
- **ユーザー**：頑健なβ最適化の資源優位を主張するかを判断する。必要な場合のみ、PF係数の上振れ幅・headroom・候補集合・providerを事前固定してB-1を個別承認する。
- **ユーザー**：RPE最終成功率・energy総資源まで主張する場合のみB-3を採択する。今回対応した既存18行signalはpreparation差の調査から始め、同一性や移送条件を確定できた場合に限り入力候補として扱う。新signal生成を最初から必須としない。確率law・状態・丸め契約を固定し、二項／分枝／必要な仮想測定のcallを個別承認する。
- **Codex**：B-1/B-2/B-3/B-4、保存shot再計算、費用再会計、native build/compileへ移行しない。

本監査A-1〜A-5は成果物を保存して**MANDATORY_STOP**。公開commit／remote照合の成否は最終報告で示す。
