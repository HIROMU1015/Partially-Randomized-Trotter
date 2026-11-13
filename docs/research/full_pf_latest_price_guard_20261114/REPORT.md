# 最新Full PF候補の価格接続とS1比較guard

2026-11-14。A71f04be7のshortlist3候補を既存coverage builderへ接続し、39round/78軸のcanonical/physical metadataを照合した。保存direct証拠は**S2 m3、Y4 m3、M5 m3/m4の4round/8軸**。他35round/70軸は確認catalog外。価格値はcopy・再集計していない。

S1について、既存C PriceProvider→compare→B費用計算前に、SOURCE付き価格実装sidecarと明示比較policyを検査するguardを接続した。旧Full-H汎用RZZと新PR ZZを共通新scenarioで混在させる入力を拒否する。**比較policyの科学的採択はしていない。** Bの直接入口を迂回できないようにするための最小取込み手順も [HANDOFF](HANDOFF.md) に記載した。B branch/worktreeは変更していない。

## SOURCE・coverage

|用途|固定SOURCE|
|---|---|
|最新A候補|`71f04be7bf94e7773a32270f36ae29eee1b903d5:docs/research/h4_full_pf_followup_results_20261114/C_HANDOFF.json`、REPORT/HANDOFF/ROUND_SCHEDULE|
|D S1/V1|`3d39f8c0ca68fb1acf86fe65d8ee5e3cc1a15673:docs/research/d_s1_v1_price_alignment_20261114/`|
|C旧候補coverage|`d7c09ef2e2521bb1815d1d11970529fc7030c2e3:docs/research/full_pf_candidate_price_evidence_20261114/`|
|D確定direct receipt|`e7dca5f7301c8633b17a59b6a7161331303aef2d:docs/research/d_full_pf_full_h_v2_final_receipts_20261113/`|
|監査|`51d324ea9c83bc12f1e866da0fa409cef926f74a:docs/research/pr_resource_repo_wide_audit_20261114/`（S1/I3を参照。I1/I2/E1の実装はしない）|
|既存C/B接続|`rpe_mean_price_model`、`rpe_optimization_pipeline`、固定B b9b9ffa6のW.compare/B.cost/select、C.configure/prepareとD.full_pf_spec|

Linear H4/1Å/STO-3G、保存tol-only DF10・weighted-Hermitian completion、sector(2,2)、元H/state/snapshot/target、one-body＋全fragment・元順序、scalar一回、hの元binary64値、M12/P0を維持。Dの完成済みreceipt/sourceチェックを再利用し、private input・raw・回路を開かず、73 receiptの監査も繰り返していない。

旧builderへ任意の固定A locator/label filterを追加しただけで、旧既定3候補と旧COVERAGE/計画は保全した。新 [BUILD_COVERAGE.py](BUILD_COVERAGE.py) はA packet内5配置からshortlistラベルだけを選ぶ。候補外Y4/32h/5やS2/8h/3は今回の比較対象へ混入させない。exact h相対分数から既存B/D生成APIで3 configurationを再生成し、ID/39 physical ID/T/q/actual幅とDの軸別statusを照合した。

|候補|configuration ID|early actual Δ|定常域とm12 qPF|保存direct XY|
|---|---|---|---|---|
|S2 16h/7|`11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d`|m0=h、m1〜3=2h|m≥4で16h/7、q1792|m3 q4、actual2h|
|Y4 16h/3|`a799697fc8ca36b024acec26af43313777948ead432e66804c670649a18afe76`|m0=h、m1=2h、m2〜3=4h|m≥4で16h/3、q768|m3 q2、actual4h|
|M5 16h|`70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd`|m0〜3=h/2h/4h/8h|m≥4で16h、q256|m3 q1/8h、m4 q1/16h|

h=.1の表示小数から幅を再生成していない。全round×X/Yとreceipt/SOURCEは [COVERAGE.json](COVERAGE.json)、metadata確認は [METADATA_CHECKS.txt](METADATA_CHECKS.txt)。S2/Y4のm3は定常要求幅の校正点ではない。Dの旧nativeで同物理回路のdirect証拠として再利用可能だが、consumerでの保存値受入・価格資格付与は未実施。別loweringなら同compilerでも移送しない。35roundのcatalog外を価格0や全世界での不存在にしない。

## 実装した比較拒否

新 [price_implementation_guard.py](../../../src/trottertracks/price_implementation_guard.py) の `check` は価格算術を行わず、選択cohortの値あり価格に対して以下を検査する。

- 比較policy：SOURCE・ID、共通compiler/control/full measured wrapper/P0 profile、method別native policy・RTE lowering・scalar quantity。同policy内のRTE lowering混在は拒否し、別domainを自動的に比較可能とは認めない。
- record/sidecar：canonical identity/law/axis/cohort、元price SOURCE、physical candidate/構成・元H/state、profile_extensionを、固定Git JSONの `implementation_evidence` に束縛。source-free自己申告extensionだけでは通らない。
- SOURCE：native/wrapper/scalarの固定locator、元Accountingとは別の価格単位、scalar総T一回、測定軸/observable/unitsを照合。resolved/expected scalarが供給される場合は値も完全一致を要求し、toleranceを導入しない。
- RTE：実emitterの `_append_central_application` ASTをD S1の既知2 SOURCEと照合。新IDを旧RZZ emitterへ付けたsidecarも拒否する。実行でそのemitterを有効化した証拠はproducer/receipt assertionに残す。
- Full PF/Truncated PF：`NOT_APPLICABLE_NO_RTE` と元deterministic nativeを要求し、PRの新ZZ IDを機械的に要求しない。Full PF/Full-Hの物理照合には既存D metadata生成を再利用し、別のlaw/ID規則を実装しない。

元recordを編集せず、derived metadataをsidecarに置く。guardはproducerのSOURCE監査・科学資格を証明するものではなく、供給されたSOURCE付き実装証拠と比較条件の取り違えを防ぐ。未知・SOURCE矛盾・不一致はValueErrorで拒否し、欠測価格は既存BのUNKNOWNのまま。fit/モデル採択、finite cohortからlaw平均への昇格、価格精度の保証は行わない。

`PriceProvider(comparison_policy=...)` はsidecarを保持して入口で検査する。`I.compare(comparison_policy=...)` はW/Bの算術前に再検査し、科学N packetでpolicy未指定なら費用計算を起動しない。既存SYNTHETIC_ONLY fixtureの旧APIは保つ。`I.select` は全結果の同一比較policyを要求する。canonical configuration/ID/profileは変更・再hashしていない。

**B側の未適用点**：固定BのW._price_metadataはextension自己整合まで、B.costそのものは実loweringを検査しない。C入口を使う経路は保護したが、Bが直接W.compare/B.cost/B.selectを呼ぶ経路には取込みが必要。薄い `G.cost` に実B controllerを渡す人工テストを行い、拒否時にB.costを呼ばないことを確認した。具体的な呼出し差分・family選択の条件をHANDOFFへ示す。Bに適用済みとは記載しない。

## 技術検証と残課題

[TESTS.txt](TESTS.txt)：新guard人工14件＋既存C/B/D接続の回帰37件、**計51 PASS**。実H4/H6入力、実価格値、実N/Gを使用していない。テストでの適格N/C/区間はfixture専用である。中間のFull-Hチェックではphysical formula=Noneと物理candidateの2ndを混同した実装不具合を修正し、既存D physical_scheduleを再利用した。科学実行の失敗やretryではない。旧builderの既定出力は旧COVERAGEと内容一致を確認した。

未知lowering、SOURCE矛盾、同compilerでの旧Full-H/新PR混在、同domain通過、PF/Truncatedの非該当、scalar/wrapper/axis不一致、欠測の非0補完、直接B算術前の拒否を確認する。作者自身の技術確認であり、独立科学レビューではない。今回の新科学計算/保存価格数値再解析/fit/予測/sampling/build/compile/実N/RMSE/Gは0。

[最新候補の有限検証案](NEXT_PRICE_VALIDATION.md)：まずS2/Y4のm4を新directで取得する2round/4 wrapper案。3本の定常幅q-affine識別＋独立holdoutなら、保存M5 m4を校正に使い、新8round/最大16 wrapper案。旧.2/.4/.8向け4round/8 wrapper案を移植しない。early・長qは別の未検証domainで、N/margin/精度はUNKNOWN。C/S観測比は価格上界ではない。

既存SOURCE/raw/旧校正/STOPを保全し、A/B/D/監査の作業領域や共有設定は変更していない。Git prune/gc/ref削除/merge/force pushは行わない。公開・remote照合後mandatory STOP。次はBがguardを自分のconsumerへ取り込み、ユーザーが比較scenarioと必要な有限科学scopeを判断する。C/Dの新科学実行へ自動進行しない。
