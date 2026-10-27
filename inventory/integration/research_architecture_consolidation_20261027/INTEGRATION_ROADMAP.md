# Integration roadmap — research/pipeline units

重点は4科学系列（DF、Pauli-JW、Research B/B2-D、Track B）＋成果横断の計5系列。P0は入口の準備、最初の大規模実装はP1。Research Cは現行研究のため構造上の配置だけを示して保留し、B3は完全除外する。前回の小helper共通化INT-01/02/03/05はP6へ後置する。

|フェーズ|対象|目的|
|---|---|---|
|P0|RS-EVID, ALL_SCIENTIFIC_STREAMS|研究入口・版固定関係の接続|
|P1|RS-DF, PL-CORE|DF端点間パイプライン＋共有計算基盤|
|P2|RS-PAULI, PL-CORE|Pauli-JW分子横断パイプライン|
|P3|RS-B2D|Research B/B2-D評価系の代表入口集約|
|P4|RS-TB|Track B portfolioとgenerator/合成の入口集約|
|P5|RS-EVID, ALL_SCIENTIFIC_STREAMS|成果・図表・論文の版別接続|
|P6|PL-CORE|小規模helper共通化の再評価|

## P0 — 研究入口・版固定関係の接続

研究正本/実装代表/成果入口/現行checkoutを分離し、古いmain案内から現行固定資料へ迷わず辿れるようにする。

**代表/移行先候補（branchと完全tip）**

- `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`
- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`

**機能移行元・比較対象（branchと完全tip）**

- `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`
- `codex/df-pauli-evidence-paper-plan-20261026` @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543`
- `research/pr-algorithm-improvement-landscape-20261023` @ `a9bdcf1d44061daceffe896637e2c92fcb3353b4`

**主な移行/接続差分・統合後の配置**

- 既存PROJECT_MAPと研究READMEへ系列入口リンクを追加する案
- CEM43/50・manifest・図表/script/inputのversion参照を接続

**保持する独自機能・SOURCE/結果/条件**

- 元CEMと科学原本/STOPは版別保持
- private local-fullをpublicへ含めない

**API・数値意味・運用契約の互換条件**

- 入口は新たな科学SOURCE正本にしない
- リンク追加に伴う科学的採択なし

**既存テストと不足検証・実施許可の境界**

- 静的link/schemaとpublic/private範囲
- 新数値・再集計・保存結果の再採点を行わない

**破壊的影響・リスク**

- 古い案内を現在の科学正本へ誤昇格
- claim_idだけで古い行を上書き

**統合完了の受入条件**

- 全系列の5役割が固定branch/commit/既存資料へ辿れる
- 欠損/未承認PLAN/停止を可視化し既存文書を科学的に書換えない

**変更の順序**

- ユーザーが入口候補/保持境界を選ぶ
- 次回Codexが許可されたdocs/indexだけを最小更新

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P1 — DF端点間パイプライン＋共有計算基盤

入力→PF/PR→finite RTE→signal/RPE→compiled price→resultという1研究経路を共通基盤とstudy adapterに分け、後続研究が再利用できる形で接続する。最初の大規模実装候補。

**代表/移行先候補（branchと完全tip）**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`

**機能移行元・比較対象（branchと完全tip）**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/h4-tol-p-seal-20261019` @ `821cc0f6ced57168293ea166644024867b064ffb`
- `codex/h6-pr-confirmation-science-20261019` @ `3cccfcb2051dacf29e91473e2e1152acef1574bd`
- `track-a-native-df-theory-evaluation-20261016` @ `7e41670565d9284da9aded94063be591fdcaa3dd`

**主な移行/接続差分・統合後の配置**

- df_hamiltonian/df_partial_randomized_pf/rte/df_rte_qiskit/rpe_resource_accounting/rte_compiled_costを既存APIで接続
- src/trottertracks/resource_applicability/pipeline.py（新規なら薄いcomposer）を提案し、H4 geometry inputs/circuits/signalの研究固有部分をadapterとして参照
- parallel_validation_executorの機構と科学adapterを境界分けし、新しい非凍結経路だけへ必要部分を取り込む

**保持する独自機能・SOURCE/結果/条件**

- 分子/rank/LD/delta/state/source/cohortの固定packet
- 旧H4 B0/B0I・native symbolicとQiskit実装の別scope
- B3全資産は対象外

**API・数値意味・運用契約の互換条件**

- coreは研究固有packageを逆importしない方針。既存moduleにdomain adapterがある場合は新入口から段階的に分離
- 旧runner/import/SOURCE/hash/call budget/seed/resumeは不変
- branch間コードは同じcheckout内へ選択取り込みした共通commitが必要。別branchの作業pathをimportしない

**既存テストと不足検証・実施許可の境界**

- static import/責務/依存closure
- fake input/typed stage/mock compiler/結果schemaの端点間配線回帰
- 既存test_df_hamiltonian/test_rte_integer_counts/test_parallel_validation_executorから非科学範囲を選定
- 実分子生成・operator/compiled一致・RPE総費用評価が必要なら独立科学許可

**破壊的影響・リスク**

- H4固有準備/許可/ledgerを汎用coreへ混入
- fixed signalとphase RPEの誤差予算混同
- optional Qiskit/ffsim importの副作用
- 古いcache/inputの取り違え

**統合完了の受入条件**

- 1つの非科学fixtureが入力adapterから結果recordまで通る（将来検証）
- 研究固有contractを通らず科学実行できる裏口を作らない
- 全元SOURCE/result/STOPと旧CLIが固定refから参照可能
- 新しい共通code commitとconsumer版を追跡できる

**変更の順序**

- core/donorの依存closureを固定比較
- contextとrepresentation/metric scopeの境界を先に決める
- adapterと薄いcomposerを新規非凍結経路へ実装
- CLIは旧入口を維持し新配線を明示
- 必要な人工regression後、科学検証は別許可で判定

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P2 — Pauli-JW分子横断パイプライン

分子別scriptに分散するsourceロード・law/signal・価格・探索/確認・報告を、固定snapshotを保つ共通の研究composerから使えるようにする。

**代表/移行先候補（branchと完全tip）**

- `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`

**機能移行元・比較対象（branchと完全tip）**

- `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`
- `codex/pauli-rte-event-extension-execution-20261026` @ `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`
- `codex/non-df-h4-pr-mean-cost-scalar-execution-20261025` @ `07d4f6ff7f1df085f292d36c36e460354df8a740`
- `codex/h6-hybrid-search-fairness-audit-20261027` @ `f7757c4320ba34dd36a64de4250c29a69bf38390`

**主な移行/接続差分・統合後の配置**

- 既存pauli_signal_adapter/fixed_signal_evaluation/fixed_signal_accounting/pauli_vector_tight_enclosureを再利用
- 新しいsrc/trottertracks/pauli_jw/pipeline.pyは必要なら薄い配線だけ。対象を設定する層と科学kernelを分離
- run_h4_pauli_b0/run_pauli_hybrid_rte_comparison/run_pauli_rte_event_extensionの固定snapshot読取とdispatch/target/price/reportを分解し、新callerだけで再利用

**保持する独自機能・SOURCE/結果/条件**

- h4_b0_execution_private/hx_study/event_runの元snapshot namespace
- H2/H4/H6・探索/独立確認・Hybrid/pure RTEの独立law/input
- 現行価格/negative/rare-event未観測/STOP

**API・数値意味・運用契約の互換条件**

- cosine/+Im sine、raw norm、constant/global phase、central control、state prep、compiler contextを固定
- H6大r Hybridは新しい提案で未実装・未取得。prefix0 event拡張を無条件転用しない
- 新しい候補・標本設計・科学scopeをこの移行へ含めない

**既存テストと不足検証・実施許可の境界**

- 既存tests/test_pauli_hybrid_rte_preparation.py / test_pauli_rte_event_extension.py等のtest source所在を確認し、後続は純粋adapter/mock範囲だけ選ぶ
- fake frozen loaderでnamespace分離・missing/nullとqualification・cohort混在拒否の配線確認
- mean作用/sampling/compile/shots/Gの再評価は別科学許可

**破壊的影響・リスク**

- scriptのdynamic snapshotをcurrent moduleへ暗黙差替え
- H4/H6・confirmation/selectionのcohort混合
- 同じq/rでも異なるtail法則を共有

**統合完了の受入条件**

- 1つの研究入口からH2/H4/H6の既存source/runner/resultを辿れる
- fake fixtureで旧module namespaceと新composerのAPI対応を確認
- 既存結果の読取と新実行が分離しimportで科学処理を開始しない
- 元法則/価格/context/STOPは不変

**変更の順序**

- P1のcontext境界を採用
- module/runner役割とfrozen loaderを切り分ける
- 薄いcomposerと既存kernelの同checkout依存を準備
- 分子別CLIはcompat wrapperで保持
- 別許可の科学検証を要する差分を限定する

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P3 — Research B/B2-D評価系の代表入口集約

既継承のevaluator/backend/recipeとouter/Exact-R/Full-DF/q4を、一つのstage表と既存packageの入口から使えるようにする。

**代表/移行先候補（branchと完全tip）**

- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`

**機能移行元・比較対象（branchと完全tip）**

- `codex/research-b-b2d-static-cost-outer-20261025` @ `3f5d60122f158b3320fadab86498963950f00734`
- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`

**主な移行/接続差分・統合後の配置**

- 既存b2d_evaluator/b2d_adapterを維持しevaluation/launcher/scope/input_package/backendを層分け
- Exact-R/full_df/full_df_q4 runnerとsource/result/STOPの対応をREADMEの現行入口へ集約する案
- P1/P2で必要性を確認したcompile/実行の機構部分だけ、新規callerで共通coreを利用

**保持する独自機能・SOURCE/結果/条件**

- A0 reference / B0 native / B1 IID / Exact-R / q2/q4独立contract
- B0失敗・recovery・後続修正の各version
- 既存output layoutとevidence serialization

**API・数値意味・運用契約の互換条件**

- whole branch mergeなし。既に後続に存在するevaluation/dense_backend/input_packageは再移植不要
- S2/Y4 signed schedule/adjoint/processor/allowlistとcall budgetを固定
- 既存grant/ledger/STOPを新しい汎用許可機構に拡張しない

**既存テストと不足検証・実施許可の境界**

- 既存tests/management/test_b2d_full_df.py / test_b2d_full_df_q4.py等のmockを選別
- stageごとのfake backend/schema/call-limit/失敗STOPを確認
- 高精度数値・operator/compile・新N/Gは別科学許可

**破壊的影響・リスク**

- final-pauseを勝手に再開
- 未資格u_num/gamma/rare-eventを埋める
- 古い失敗を現成功へ上書き

**統合完了の受入条件**

- 全stageのrunner/input/source/result/STOPが1入口に列挙
- evaluator・backend・orchestrationの責務が説明できる
- mockによる既存stage API互換を必要範囲で確認
- 科学scope/結果/履歴は変わらない

**変更の順序**

- 既継承moduleを確認し再mergeを除外
- stage表とpackage責務を整理
- 必要な非凍結adapterだけ共通coreへ依存
- mockを確認し未資格欄を欠損のまま保持

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P4 — Track B portfolioとgenerator/合成の入口集約

DF correction・allocation・G6・M1-Sを既存umbrellaから追跡可能にし、G6だけのbranchへ別数学を一括統合しない。

**代表/移行先候補（branchと完全tip）**

- `track-b-g6-v1-2-draft-review-integration-20261022` @ `5f750a23f023d75c8c4fbcabc6937d6953ce61f2`
- `track-b-m1s-v1-gamma-formal-adoption-20261017` @ `e7791c9ffe59577528e6095da5b2419bf6fadda4`
- `track-b-m1s-gamma-pi3-formal-adoption-20261017` @ `b17b056deec60820413c0881978f7b77caf0861b`

**機能移行元・比較対象（branchと完全tip）**

- `track-b-g6-construction-paper-evidence-20261018` @ `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`
- `track-b-g6-v1-2-draft-review-integration-20261022` @ `5f750a23f023d75c8c4fbcabc6937d6953ce61f2`
- `track-b-m1s-v1-gamma-formal-adoption-20261017` @ `e7791c9ffe59577528e6095da5b2419bf6fadda4`
- `track-b-m1s-gamma-pi3-formal-adoption-20261017` @ `b17b056deec60820413c0881978f7b77caf0861b`
- `track-b-algorithm-codesign` @ `09c9e89555032213b52a6a60b38f55563d07a34d`

**主な移行/接続差分・統合後の配置**

- 既存algorithm_codesign namespaceとREADMEへG6/M1 γ/M1 π3の3代表入口を接続
- G6 F3 GeneratorSession/prepareとnative inventory/placementのdescriptor関係を明記
- 共通descriptor/storage/read-only記録の適合部分だけを次回に選択移植し、M1-SのKernels/compare.evaluateは独立のまま

**保持する独自機能・SOURCE/結果/条件**

- D/E/M/Gの各数学/interval精度/point/context
- M1-S γとπ3の別SOURCE・法則・gate
- 旧TrackB B3 pilotの歴史。現DF-native B3へ接続しない

**API・数値意味・運用契約の互換条件**

- G6 old→v1.2は祖先・F3追加、M1-S v2はG6 v1.2に不在
- M1 γ→π3にも祖先関係がない
- 異なる保証/normalize/cost scopeを同一API結果として返さない

**既存テストと不足検証・実施許可の境界**

- selected tree/import/descriptor静的確認
- tests/tracks/algorithm_codesign/test_g6_return_aggregation.py / test_g6_f3_preflight_angle.py / test_m1s_v1_v0_static.py等の対象fixtureを将来精査
- 科学interval/primitive/native/operator/実用性比較は別許可

**破壊的影響・リスク**

- 一本の代表を全アルゴリズムの正本とする
- 旧TrackB B3と現DF B3の混同
- read-only helperからold output/archiveへ書き込む

**統合完了の受入条件**

- umbrellaで各別数学の実装/固定source/result/STOPへ辿れる
- G6にないM1-Sは明示した別refで利用し、missingを隠さない
- generic storage等の新共通経路で旧assetを変更しない
- 3入口の独立保証domainと維持担当を明示

**変更の順序**

- 3代表と固定科学正本を分離
- pipeline descriptor/metricの対応表を作成
- 必要な新callerだけの共通実装を採択
- 元SOURCEとfailure/STOPはそのまま維持

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P5 — 成果・図表・論文の版別接続

P1-P4で採択した新code入口を元科学証拠へ結び、branch最新値を科学的採択と誤認しない入口を完成させる。

**代表/移行先候補（branchと完全tip）**

- `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`

**機能移行元・比較対象（branchと完全tip）**

- `codex/pr-boundary-evidence-integration-20261027` @ `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`
- `codex/df-pauli-evidence-paper-plan-20261026` @ `6e3436a16b13e539b0faa2a0a3286f6ab3549543`
- `codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`

**主な移行/接続差分・統合後の配置**

- 既存CEM/manifest schemaでsource/result/study/cohort/conditions/metric/figure/script/inputの関係を接続
- 一般READMEと系列別READMEから同じ固定索引へ戻る導線を整備

**保持する独自機能・SOURCE/結果/条件**

- 旧43行/新50行・public subset/private local-fullを保持
- negative/STOP/未資格を掲載位置の都合で消さない

**API・数値意味・運用契約の互換条件**

- 結合キーはversion+claim/study+source/result+cohort
- 科学的採択/原稿claimの変更を技術的link修正に含めない

**既存テストと不足検証・実施許可の境界**

- schema/linkと公開範囲・欠損の技術検査
- 新集計・数値比較なし

**破壊的影響・リスク**

- 新codeを旧結果のSOURCEとして誤表示
- 図表と別input/cohortの混在

**統合完了の受入条件**

- 新入口→既存source/resultの双方向追跡
- 旧版を上書きせず全重点系列の5役割が明記

**変更の順序**

- P0の索引設計を使う
- 移行後codeと旧結果を別versionで接続
- public/private境界とリンクを確認

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## P6 — 小規模helper共通化の再評価

大きなpipeline境界と入口が定まった後、前回INT-01/02/03/05を必要な順だけ実装する。

**代表/移行先候補（branchと完全tip）**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`

**機能移行元・比較対象（branchと完全tip）**

- `all-r-coherent-opt2-reoptimization` @ `f29eafbf4412193e8e8660d9f051889c2cba684b`
- `codex/research-b-final-pause-handoff-20261026` @ `e87375fa9a998dce80792be0f15597c747f00d83`
- `track-b-g6-construction-paper-evidence-20261018` @ `d4163acd7b577afaa7d6db8c29db64f76ec03ae7`

**主な移行/接続差分・統合後の配置**

- 整数検証・PF係数・canonical JSON/digest・storage cloneの必要部分

**保持する独自機能・SOURCE/結果/条件**

- NumPy整数policy/Unicode hash/format/MAGIC/legacy import/固定SOURCE

**API・数値意味・運用契約の互換条件**

- 前回の具体的互換条件を継承し、utilityを先行目的にしない

**既存テストと不足検証・実施許可の境界**

- 小さなpure unit/人工fixtureだけ。旧archiveを書換えない

**破壊的影響・リスク**

- helperの機械的統一で出力hash/allowlist/STOPを変える

**統合完了の受入条件**

- 複数重点系列で本当に再利用され、元契約が保たれる

**変更の順序**

- P1-P5で実需要があるものを選ぶ
- 最小差分を個別検証

担当: ユーザーが代表/対象系列/互換方針と実装範囲を採択した後、Codexが次回の明示指示内で実装する。科学生成/fit/operator/compile/sampling/数値再解析は別の実行許可範囲を要する。今回このphaseの実装・テスト実行は0。


## 重点5系列の既存入口・不足機能・移行後の姿

### RS-DF DF / PR / PF / finite RTE / RPE

現行の実装入口: src/trotterlib/df_hamiltonian.py; src/trotterlib/df_partial_randomized_pf.py; src/trotterlib/rte.py; src/trotterlib/df_rte_qiskit.py; src/trotterlib/rpe_resource_accounting.py; src/trottertracks/resource_applicability/。成果入口: 既存CEMのDF行・各分子packet・PROJECT_MAP/validation_manifestの版固定導線。

現状/未解決: 分子/比較taskごとの既存局所検証・停止。coreは存在するが後続分子adapterと研究結果が別branchに分散。 同条件での全RPE/final total-cost完了は未確立。旧screening/stale結果は使わず、H12や未実行domainへ外挿しない。

機能分散とAPI・依存・input/outputの境界はRESEARCH_PIPELINE_MAP.mdおよびCROSS_STREAM_CAPABILITIES.csv。移行先/差分/検証/受入条件は上記 P1 を参照。元import/output/科学packetを保持して新規利用から配線を選択移植し、whole branch mergeは行わない提案。

### RS-PAULI Pauli-JW fixed-signal / Hybrid / pure RTE

現行の実装入口: src/trotterlib/pauli_signal_adapter.py; src/trotterlib/fixed_signal_evaluation.py; src/trotterlib/fixed_signal_accounting.py; src/trotterlib/pauli_vector_tight_enclosure.py; src/trotterlib/pauli_rte_event_extension.py; scripts/run_pauli_hybrid_rte_comparison.py; scripts/run_pauli_rte_event_extension.py。成果入口: Pauli summary branchのdocument_pauli_jw_saved_summary.pyによる既存公開導線。root CEM/SYNTHESISは不在なのであると記載しない。。

現状/未解決: 固定有限集合・条件付き費用の既存報告。H6大r Hybridは後続調査で未探索・提案のみと記録。 全family最適性/母平均/実compiled残差の一般保証は未確立。新r候補は既存pure-RTE prefix0 wrapperへ無条件接続できない。

機能分散とAPI・依存・input/outputの境界はRESEARCH_PIPELINE_MAP.mdおよびCROSS_STREAM_CAPABILITIES.csv。移行先/差分/検証/受入条件は上記 P2 を参照。元import/output/科学packetを保持して新規利用から配線を選択移植し、whole branch mergeは行わない提案。

### RS-B2D Research B / B2-D / outer

現行の実装入口: src/trottertracks/b2d_evaluator/; src/trottertracks/b2d_adapter/; scripts/run_b2d_stage1.py; scripts/run_b2d_exact_r.py; scripts/run_b2d_full_df_pf.py。成果入口: src/trottertracks/b2d_evaluator/README.mdとfinal-pause handoffを役割分離して使用。

現状/未解決: final-pause内へ主要evaluator/outer/exact-R/full-DFを既継承。研究全体の一時停止を勝手に解除しない。 u_num/gamma/rare-event/shot-weighted資格に既存留保。A0 high-precision referenceとnative/compiled検証は異なる。

機能分散とAPI・依存・input/outputの境界はRESEARCH_PIPELINE_MAP.mdおよびCROSS_STREAM_CAPABILITIES.csv。移行先/差分/検証/受入条件は上記 P3 を参照。元import/output/科学packetを保持して新規利用から配線を選択移植し、whole branch mergeは行わない提案。

### RS-TB Track B algorithm co-design

現行の実装入口: src/trottertracks/algorithm_codesign/; src/trottertracks/algorithm_codesign/g6_f3_generator.py; src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/; src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1/。成果入口: scripts/tracks/algorithm_codesign/README.mdを既存umbrella入口候補としG6/M1-S γ/π3を別リンク。

現状/未解決: G6 draft/構成・DF correction・M1-S等の既存限定結果/保留/終了が混在。単一global採択なし。 G6後続にM1-S finite-signal v2が存在せず、M1-S γとπ3も祖先でない。保証domain/interval精度/古典費用とcompile費用を混同しない。

機能分散とAPI・依存・input/outputの境界はRESEARCH_PIPELINE_MAP.mdおよびCROSS_STREAM_CAPABILITIES.csv。移行先/差分/検証/受入条件は上記 P4 を参照。元import/output/科学packetを保持して新規利用から配線を選択移植し、whole branch mergeは行わない提案。

### RS-EVID 研究成果 / CEM / 原稿 / 図表

現行の実装入口: PROJECT_MAP.md; VALIDATION_STATUS.md; artifacts/validation_manifest.json; EVIDENCE_MATRIX.csv; RESEARCH_SYNTHESIS.md; FIGURE_TABLE_PLAN.md。成果入口: 既存root RESEARCH_SYNTHESIS.md / EVIDENCE_MATRIX.csv / PAPER_OUTLINE.md / FIGURE_TABLE_PLAN.md。

現状/未解決: 43/50行CEMとpublic boundary統合を既存取得済みとして再利用。新科学的採択なし。 branch tipだけで現行科学結論を決めず、欠損provenanceと図表input/cohortの不明を明示。

機能分散とAPI・依存・input/outputの境界はRESEARCH_PIPELINE_MAP.mdおよびCROSS_STREAM_CAPABILITIES.csv。移行先/差分/検証/受入条件は上記 P0, P5 を参照。元import/output/科学packetを保持して新規利用から配線を選択移植し、whole branch mergeは行わない提案。


## 最初に決める事項

- P1の共通core/DF入口に既存all-rを使うか（主checkoutは切替えない）。H4 adapterから科学条件を切り離す境界を採択する。

- context/metric/representationの意味を既存recordで明示し、どの機能をcore、どれをstudy adapterへ置くか選ぶ。全研究へ新schemaを強制しない。

- Pauli固定snapshot loaderと新composerの共存、Track Bの3数学入口、B2-Dの既継承stage表を認めるか。

- Research Cの現行3入口は維持し、fixed-CAR新準備を勝手に統合しない。N1/N3/R2等の研究再開は今回の移行選択に含めない。

- 次回のcode/API/mock作業範囲と、必要なら別途科学検証の範囲を区別する。共通codeを変更しても旧SOURCEへ適用しない。

rollbackは新規非凍結入口を使わず元固定commit/runnerへ戻る設計とし、今回reset/revertや原本操作は行わない。原本を変えないことを移行の前提にする。

並行更新の境界: 新しいH6 Hybrid readiness/execution branchの追加をremote metadataだけで観測した。新branchの実行状態・実装・新結果は未調査で保護する。本書の未実行/未取得という記述は引用した固定版のscopeであり、現況の不存在を主張しない。次回のP2実装では担当者の引継ぎと固定SOURCEを確認して既存新機能を再利用し、重複実装しない。
