# 研究別SOURCE

[H4 kernel run02：K0 PASS・K1平均構造RZ四値](../../docs/research/h4_structural_kernel_20261113/run02/REPORT.md)（2026-11-13）：SOURCE19b95d8b3で同H4 DF10/S2/p1/δ.1/K2・1prep/2lawを一回実行。K0全9wire PASS、kernel/状態辞書を保存後、m0/m6×XYの実E[S]四値を取得（各mでX=Y、1743.6682557187337/1764659.275564426）。数値区間・元Accounting parityを保存。sampling/full-path S/build/compile/fit/N/G0、E[C]はnull・C1未資格のまま。旧run01/STOP保全、公開照合後mandatory STOP。

[H4 K0入力STOPと技術修正版](../../docs/research/h4_structural_kernel_20261113/REPORT.md)（2026-11-13）：K0→PASS時K1 runnerと数値契約を実装、新8（人工7/固定metadata1）＋旧10技術test PASS。SOURCE0f7d65d5fはB configuration locator誤りで物理K0前にUNKNOWN STOP。実descriptor/kernel/parity/平均S四値は未実施・null。参照chainを修正版19b95d8b3でmetadata照合し、科学再実行なし。指示§4 STOP保全、run02案のみ公開。新sampling/full-path S/compile/fit/N/G0。

[H4 PRの物理閉状態診断](../../docs/research/h4_pr_physical_closure_20261113/REPORT.md)（2026-11-13）：固定H4 DF10/S2/p1/δ.1/K2の1prep・R2/R128を診断。324非identity/9basis、reset/phase/head境界からS observableの有限wire-marginal状態の十分条件を確認（tail enclosure80〜120）。物理kernel/parity/実平均S/Cは未取得、MODEL false。最終10技術test PASS。初回seed設定STOPを保全し同scope修正、成功SOURCE eeccdf18e・縮約解析2b3b0ee7f。新sampling/full-path S/build/compile/fit/N/G0、旧S0a等を保全、公開照合後STOP。

[固定H4 S0の平均compiled価格検証計画](../../docs/research/h4_s0_mean_price_plan_20261113/REPORT.md)（2026-11-13）：SOURCE bdd2b9356。固定A S0のexactκ/M12・H4 m0(q1/R2)/m6(q64/R128)を公開B V2へmetadata接続。新7技術test PASS、13round価格null。両候補の一致direct/modelは公開索引に未確認。閉状態診断1prep/2law・compile0を第一案、別案は有限4path/最大8wrapper。新S/価格再解析/sampling/compile/fit/N/G0、C1未資格・S0a履歴を保全し、公開照合後STOP。

[B V2価格接続と長q平均価格の次段階設計](../../docs/research/bv2_mean_price_validation_design_20261113/REPORT.md)（2026-11-13）：SOURCE58382c54c。固定B6a9 V2 MeanPriceLookupを再利用するC受動入口を追加、新12人工/metadata結合test PASS。値ありrecordに平均compiled quantity明示を要求し、S0a/旧ID付替え/資格昇格を拒否。旧118候補/236軸と4方式の資産/欠測を静的整理。実平均S/C・物理閉包は未取得、選択B domainのclosure診断等4候補を未承認計画化。新S/C再解析/sampling/compile/fit/N/G0、旧結果/STOP保全、公開照合後STOP。

[S0a run02：実保存8件の構造S照合](../../docs/research/s0a_saved_structure_results_20261113/run02/REPORT.md)（2026-11-13）：修正版SOURCE9516351e6で保存normal index0の4path×XY8 Sを一回評価し、元C1保存Sと8/8整数完全一致。各X/Y SはH4 q8/R8=21664、q16/R8=46559、H6 q1/R32=158669、q2/R32=321151。input/preparation/law/phase/allocation/digest検査を通過、元roleとrun01失敗STOPを保全。平均S/C・物理閉包・モデル資格は未取得、qualified=false。新sampling/compile/fit/N/G0、公開照合後STOP。

[S0a run01の設定エラーと技術修正](../../docs/research/s0a_saved_structure_results_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65でrun01を1回起動、context exec scopeのast渡し漏れでS開始前に失敗STOP。新S0/8、保存S照合0、全4path未評価。修正版SOURCE9516351e6と人工14件PASSを公開、run02案はmetadata準備だけで未実行。旧STOP/role/モデル資格を保全、修正版再試行はユーザー判断、公開照合後mandatory STOP。

[S0a保存構造価格の実行準備](../../docs/research/s0a_saved_structure_preparation_20261113/REPORT.md)（2026-11-13）：SOURCE11cb91a65。固定C1の保存4path/XY8計数に向けreader・law/identity検査・S専用runnerを準備。新13＋旧9＝22技術test PASS。raw/inputは所在のみ確認、実S・保存S比較は未実行。元係数/資格/平均provider不変、内容hashと実contextは実行時検査。初回人工importのlowering副作用と修正をREPORTへ記録。明示科学許可待ち、公開照合後STOP。

[C1状態閉包とStreaming計数の技術検証](../../docs/research/c1_physical_state_stream_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE c1135fdb9。C1のpending basis/Z・tail phase・terminalをSOURCEで調査し、元計数wordへforward-only tailを渡す最小consumerを追加。新9＋旧3＋sampler toy1＝13技術test PASS。単純縮約の人工反例は確認、実H4/H6の閉包・event生成なしの平均S/compiled価格は未確認。consumerは全event処理時間が残り、producer全接続は未実装。S0a保存4path/XY8件は未実行案、実科学計算・再解析・compile・fit・N/G0、公開照合後mandatory STOP。

[長q・大Rの平均compiled価格の技術準備](../../docs/research/mean_compiled_price_preparation_20261112/REPORT.md)（2026-11-12）：SOURCE09489eedb。C1 Accounting/旧局所期待formを再利用し、closed-state期待報酬の二乗合成と受動c_providerを実装。新15＋旧pure3＝18技術test PASS、人工全列挙と一致。固定状態ならqR展開を避けられるが、実C1の累積Z/phase閉包・平均S・compiler残差は未資格で実価格missing。C−S残差のみを扱い二重計上を拒否。κ scheduleはB外部contextのcopy、M/N最適化なし。旧40/16案は非実行、将来S0保存再解析等は別許可。実科学計算/compile/価格fit予測/N/G0、公開照合後STOP。

[A P1対応・長q/大R価格検証の準備](../../docs/research/r_price_longq_largeR_preparation_20261111/REPORT.md)（2026-11-11）：公開A71ba49bbの12候補・同policy条件付きNを無変更copy。C1価格との重複はH6 m4/R16,R32の校正2点。第一案はH4 m6/H6 m8のbaseline＋R128、4セル・最大40 wrapper、原因分離R-onlyと短時間各16は別の任意案。metadata SOURCE f0daeb96b、8技術test PASS。構造Sのstream化/局所境界法を比較し、m12やR256を自動compile/外挿しない。新科学計算・価格予測・再解析・N/G0、科学実行未承認、公開照合後STOP。

[R価格Phase 1実行結果](../../docs/research/r_price_phase1_results_20261111/REPORT.md)（2026-11-11）：Linear H4/H6・1Å・STO-3G・保存DF10/19、H4 S2/δ.1/p1/K2とH6 m5_best/δ1.6/p7/K2。SOURCE1a87e5331で通常64＋別rare16 compile、保存C/S59 path再利用。係数を7eeabf0aで検証取得前に凍結し再fitなし。独立n8の相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%。有限cohort観測、採用閾値未設定・model未採用、全law/長q-R/総CI/全Gは未資格。新signal/N/G0、公開照合後mandatory STOP。以下の準備記録・過去STOPは履歴として保持。

[R価格Phase 1の実行準備](../../docs/research/r_price_phase1_preparation_20261111/REPORT.md)（2026-11-11）：保存C/S接続・paired3係数fit・Git凍結後validation・別unit rare診断の専用runnerを準備。公開A/B metadata整合、新14＋旧R回帰9＝23人工/mock PASS。固定SOURCE/PLAN、最大64＋16 wrapperの科学scopeは未実行。実分子sampling/S再計数/compile/fit/価格予測/再解析/N/G0、公開照合後STOP。

[R可変compiled価格の調査・設計](../../docs/research/r_variable_compiled_price_design_20261111/REPORT.md)（2026-11-11）：H4 S2/δ.1/p1/K2・H6 m5_best/δ1.6/p7/K2の固定Rモデルを保全し、構造価格＋少数compiled残差補正を提案。R binding/保存lookup adapter、人工/mock9件PASS。新R Cはnull、実科学計算・価格取得/予測/fit/G0。最小新q×R/希少event計画は未承認・未実行、公開照合後STOP。

[H6 provider統合・H4/H6全13round N/C coverage](../../docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md)（2026-11-11）：固定B cb2 APIへ保存lookup factoryを追加。C provider/テストをbyte一致再利用、24技術test PASS。H4公開89c91c69の整数検証済みNとH6公開f4c0bdaのNを静的転記し52行条件付きPASS。直接CはH4 5候補/H6 8候補、PR全law/長q/共有CI/全RPE採用は未資格。新科学計算・実N×C/G0、B実行領域へ非介入、公開照合後STOP。

[H4 price providerのB接続](../../docs/research/h4_price_pipeline_integration_20261110/REPORT.md)：h4_df10_price_providerと固定V1/Mq依存を取込み。B factory/run_missing/coverageへ接続し、34技術test PASS、実N/C/G0。

共通 `trotterlib` APIと区別した実験別SOURCEの名前空間。
[mainへの収録範囲・原SOURCE・元の制約](../../docs/research/main_collection_20261103.md)を参照する。

- [G6 return aggregation prototype](algorithm_codesign/return_aggregation.py)：元snapshotの内容を保持。native資源評価は未取得、mandatory STOP。

## 他系列のSOURCEと版差

[研究別SOURCEの公開版と配置](../../docs/research/main_collection_versions_20261103.md)を参照する。P1 pipelineとprice-statisticsの旧版は保存原版から元snapshotへ戻る。選定したmain版は下記のSOURCEの組を参照する。
同名SOURCEの全版を保存し、元の入力・契約・STOPを区別している。

## P1・研究C SOURCEの組（2026-11-03）

[選定したSOURCEと依存版・検査範囲](../../docs/integration/main_p1_c_price_cohorts_20261103.md)を参照する。P1は人工・mock検査79件PASS、研究Cはpassive importと静的互換を確認した。
共有coreの変更前版は保存済み。過去の科学SOURCE・入力・結果・契約・STOPは元snapshotを使う。

## サーバー照合からの成果収録（2026-11-03）

[収録ファイル・保存版と確認範囲](../../docs/research/main_server_results_collection_20261103.md)から、過去の公開成果と元SOURCEを辿る。
既存mainの科学ファイルは上書きせず、研究test関連の原版は実行contextと分けて保存する。新科学計算は行っていない。
