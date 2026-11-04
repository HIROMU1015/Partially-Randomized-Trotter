# H8長時間・高次PR：Stage A/B実行

2026-11-03。高速化率のbenchmarkは行わず、承認された実計算を進める。
旧rare-RTE、旧独立確認、H46研究のSTOPは維持する。新しい独立scopeであり、Stage Cは確認n・精度目標が未定のため実行しない。

## 固定条件とSOURCE

Linear H8、隣接距離1.00 Å、STO-3G、保存tol-only1e-8 DF rank28、16 qubit、spin sector4900、元state・fragment順序0..27。
T=1.6、ε=0.0005、状態準備P=0、費用はG=N_cos C_cos+N_sin C_sin。
入力identity・先行計画は[adapter準備報告](../h8_long_time_adapter_preparation_20261103/IMPLEMENTATION_REPORT.md)と[科学実行計画](../h8_long_time_adapter_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)。

Stage AのSOURCEは`d1272016592ff06c0567862f770ed796fd61b366`。
元SOURCE/input/raw/QPY/grant/ledger/marker/STOPを変更せず、専用cloneのprivate run領域に原結果を保存する。
内部path、seed実値、event列、private入力とQPYは公開しない。

## 実行範囲

Stage Aの第一batchは[固定提案一覧](../../../configs/h8_long_time/proposed_scope.json)の57候補。
PRはprefix7/8、S2 q8/16/32とm5 q1/2/4、R_outer32、K4の12候補。
PFはFull28、B0/B0I prefix15/16、S2/Y4/m5、q1/4/16の45候補。
新T full-H sparse reference、signed primitive、finite mean、bias/normalization、axis別headroom/Nを検査する。

Stage Bは探索cohort、n=9。各PR formula/prefixとPF method/formula/prefixで、Stage A評価済みの最小の適格qを初期価格対象とする。
直接価格と固定NによるGまたはheadroomから比較が未解決の場合だけ、許可domain内の次q等を検討する。
未取得価格はunknown。work proxyをcompiled価格・下界へ読み替えず、選択した候補だけの比較とする。
compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/target/topology未指定。
paired cos/sin価格、代表control/actual-adjoint/scalar検査、measurement付きcompiled RZ、希少order診断を保存する。

## 実行状態

Stage A第一batchは57候補PASS、適格30・不適格27・UNDETERMINED0。signed primitiveは855 block/time組×3 probeで最大差2.62e-14。新T full-H referenceはdirect/split-time照合をPASS。
PR S2はq8/16不適格、q32適格。PR m5はq1/2/4で適格。q/Nの差だけで資源優位は判定しない。
強いPF対照の適格性境界を補完するため、同じ計画domain内のY4 q8（5 method/prefix）、S2 q32（5）、B0 S2 q8（2）の12候補を追加実行し、計69候補が完了した（適格40、不適格29、判定保留0）。初期結果は[保存結果](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-03/stage_A_initial.json.gz)。
Stage Bはm5の7候補・23 trajectoryを8 worker、S2/Y4の12候補・28 trajectoryを25 workerで並行実行中。各PR候補はn9、PFは候補ごとに1回。まだcompiled価格・順位の結論はない。既存native APIのinstruction boundは正の整数が必要なため、無限大ではなく既存block_instruction_boundの整数を渡す互換修正をSOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`へ固定した。人工/mock21件PASS。Stage Aのsource mapは別に保持し、科学入力やsignal値を変更しない。

今回の検査は主担当Codexによるもので、別担当者の独立科学レビューではない。
科学性能・全候補の最適性・population cost保証・最終全RPE costを主張しない。

## 18時台の稼働確認と復旧

2026-11-03 18:23 JSTの確認では、元の両起動セッション・計算worker・集計待機プロセスが失われていた。原因は未確定。HOLDや例外logは残っていないため、正常完了とも科学的不一致とも判定しない。
最後の出力は17:41頃。Stage B01は21/23 trajectory・45/46軸、B02は15/28 trajectory・42/56軸まで保存され、計36/51 trajectoryが完了した。

完了済み36件と元のpartial/cacheを保全し、未完了15件だけを新しいprivate復旧先で処理する。科学runtime SOURCE、入力、固定N、cohort、n、candidate/indexと元seed/event列を維持し、新しい標本を追加しない。元のHOLDがある場合は復旧入口が拒否する。
[復旧入口](../../../scripts/resume_h8_long_time_execution.py)は既存Context・price_job・paired会計を使い、保存compiled QPYも再利用する。チャットから独立した自分専用のtransient user serviceで起動する。共有サービス・OS設定は変更しない。
既存テストと人工復旧fixtureは計25件PASS。代表回路の検証は完了するまでPASSにしない。前回の18:30–20:30完了見込みは撤回し、復旧後も数時間を要する可能性がある。

18:30 JSTに独立user serviceで復旧を開始した。PFの未完了9件は稼働中。
PRの6件では、`to_dict()`中のtupleが保存JSONからlistとして復元されるため、Python object同士の直接比較が拒否された。元seedの再構成診断でcanonical JSONは完全一致した。科学的なevent差ではなくメタデータ表現の技術的不一致である。
最初の復旧SOURCEと稼働中のPF workerを変更・停止せず、[追加復旧入口](../../../scripts/repair_h8_long_time_recovery.py)でcanonical JSON一致を要求してPR6件だけを処理する。seed、値、event順序の差は引き続き拒否する。元の技術的HOLDは保持し、異なるHOLD理由は受け入れない。
型差の回帰例を含む人工テストは計27件PASS。新規候補・標本の追加はなく、Stage Cは引き続き未実行。

## 11月4日の状態とcompiler位相診断

48/51標本が完了した。11月4日00:09のユーザーサービス終了で未完了のS2代表2件と診断が中断したため、共有設定を変更せず、自分専用tmuxで再開した。S2代表は元の条件・seedで継続し、検証ベクトルを段階ごとに保存する。
失敗したm5 prefix8代表はHOLDを保全する。診断ではHighLevelSynthesis後の差0に対し、最初のBasisTranslator後に符号反転が現れた。
実際のXX+YYゲートの局所compiler変換で余分な位相iを観測し、対象wrapperの52262ゲートから全体の符号反転を説明できた。
補正を使う実装・再検証は自動承認レビューで却下されており、未実施。
[具体的な修正案・診断根拠と必要な許可範囲](COMPILER_PHASE_REPAIR_PROPOSAL_20261104.md)を参照する。

11月4日17:30 JSTの読取確認では、完了した重複なしの標本は49/51件で、保存判定は全49件PASS。再開したS2 prefix7・q32の代表は11:59 JSTに7検査すべてを通過した（compiled vector差はcosine 4.20e-11、sine 5.24e-11、許容値1e-7、位相整列なし）。一方、S2 prefix8・q32の代表は11:04 JSTに`COMPILED_VECTOR_cosine`でHOLDとなった。この新しいHOLDの原因を、m5代表の診断結果だけで確定しない。
残る2件はm5 prefix8・q1とS2 prefix8・q32の代表で、いずれもHOLDを保全して自動再試行していない。計算workerは終了しており、現在は計算待ちではない。補正実装・同じ標本での必要な再検証は、上記修正案への追加許可待ち。Stage B全体の価格集計・順位は未確定で、Stage Cは未実行。

追加修正として、[位相会計](../../../scripts/h8_compiler_phase_bookkeeping.py)と[残る2代表の再検証・集計入口](../../../scripts/repair_h8_compiler_phase.py)を実装した。元の科学SOURCE・入力はそのまま、実際の局所gate operatorに基づくglobal phaseだけを補正する。人工operator検査、人工3qubit回路から元の7検査を通す検査、補助SOURCEの集計来歴を含む34テストPASS。補助SOURCE公開後に、同じ元標本2件だけを並列実行する。49件の完了結果、元HOLD、元raw/QPYを保全し、Stage Cは実行しない。

補助SOURCE `f04ef60c960fd2a68bcf0506f4c505f0bc872b78`を公開・照合し、17:45頃から元の2代表をtmux内の2 workerで並列再検証している。
ユーザーの中間集計依頼により、[17:50 JST時点の保存結果](INTERIM_RESULTS_20261104_1750JST.md)と全19候補のCSV・匿名価格metadataを保存した。49/51標本、17/19候補の代表検証完了。未完了2候補の8標本平均は記述値として区別し、確定した比較に使わない。稼働中の科学SOURCEは変更していない。

20:27 JSTの確認では、最初の位相補助実行は2件とも技術的HOLDで終了していた。m5は18:20に補正後cosine vector差7.80e-12（許容1e-7、位相整列なし）まで通過したが、sine wrapperの`sdg`が局所operator検査の対象に未登録だった。S2は18:58に、元run以外の復旧先に保存されていたcompiled QPYを再利用対象から漏らしたため、`SAME_ORIGINAL_RAW_COMPILED_CIRCUIT`で停止した。S2の復旧QPYと再compile QPYのhashは一致した。
追加修正の範囲内で、S†も小operator検査を要求する対象に加え、既知の復旧cacheを検索するよう修正した。保存済みcosine wrapperから元evolutionを取り出し、同じnative builderによるcosine再構築fingerprintの一致を要求する。これにより重い同一回路の再構成を避ける。保存vector・局所operator検査は元のSOURCE・標本・preparation・hashを照合して再利用し、補正済みvectorには位相を二重に加えない。人工3qubit回路のS†と完全な保存再利用を含め関連36テストPASS。元の全HOLDと最初の補助SOURCE `f04ef60c960fd2a68bcf0506f4c505f0bc872b78`は保全し、次の補助SOURCEと別のprivate保存先で同じ2標本だけを継続する。科学条件・許容誤差・標本数は変更しない。

## 探索完了（11月4日23:06 JST）

最終補助SOURCE `18ce41942051065fa097644a8a937bd11adba708`で、m5 prefix8代表は21:01、S2 prefix8代表は23:06に全7検査PASS。全51 trajectory・19候補・102軸の探索価格集計を完了した。19代表の計133検査と、51標本からの19候補の固定Nによる平均Gを照合した。
[完全な探索結果](EXPLORATION_RESULTS_20261104.md)を参照。最良の測定済みPFはB0+I/m5/prefix15/q1、G=8.9291219516568e13。PR m5/prefix8/q1/R_outer32/K4の9標本平均G=6.732336259192e13、PF比0.75397517（観測平均で24.60%低い）。母集団保証・全domain最適性・全RPE総費用・確認cohortの結果ではない。元の失敗/HOLD、17:50中間集計と科学SOURCEを保持した。Stage Cは未実行。
