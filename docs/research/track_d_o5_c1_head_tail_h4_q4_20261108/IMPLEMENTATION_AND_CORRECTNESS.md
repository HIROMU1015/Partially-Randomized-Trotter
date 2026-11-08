# 実装・原本再利用・correctness

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2／元H／Number-CDF8（rank8固定）／元frame順L_D2／ordered partial S2/s1/r32/K0／h=δ=.1／q4・complete step2後cut。元H厳密ground利用を仮定し準備費を除外。対象はC6a calibration cohortの保存 `q4_sample00` 一件だけ。

## 固定SOURCEと変更の範囲

実行SOURCEは `0df0134b5f259bbb2603719227e21831d876af9f`、treeは `22c1db4a0a077276742859a5ea91ca49efe175a8`。通常pushはHTTPS認証情報不足で失敗したが、接続済みGitHub機能で同一treeを保存し、取得した公開commitの全10変更file bytesを実行前に照合した。実行時の [SOURCE.json](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/SOURCE.json) はevent decode前に生成し、PLAN内の全file hashと実依存versionを記録している。結果公開commitと実行SOURCEを区別する。

新規は専用 `track_d_head_tail_h4.py` とrunner、人工回帰test。元の2-mode adapterのsignature/same_irを再利用し、8 system wire 0..7／Strong9 wire（ancilla8）／physical8 wireという明示guardを追加した。元C6aの科学code20ファイルが固定C6a SOURCEとbyte一致。旧builder/gadget/serializer/compiler/checkerは変更していない。

再利用先はStrongBuilder._basis/transition/sequence/vacuum_phase、C3 mixed_head/prepare_steps/physical_segment、既存full-frame Provider/control_one/reference_gadget、C3/C6a raw/post checker、MPFR160/P0、既存full-wrapper counter。新しいGaussian合成器、Hamiltonian builder、sampler、Reference gadget、state simulatorはない。

各stepのhead出口block2と、最初のeventのactual full-frame rotationに一致するblockを結ぶ。Basis IDだけで判定しない。保存gate列の隣接span・system wire順・actual basis・analytic vacuum gaugeを照合し、既存G_T†G_Hへ置換する。全体phaseから除去spanのphaseを差し引き、橋のphaseを一回composeする。Strongのordinary tailとuncontrolled Gaussian、Referenceのphysical halfのcontrol modeを変えない。R1のA†は融合後A.inverse()から作る。

今回の実接口は各方式4、拒否0。step0/1/2はhead block2→tail block3、step3は2→6。各spanはnative64命令→32命令。これは局所raw命令数であり、compiled full-wrapper価格差ではない。内部tail異frame、tail出口、step融合規則、global cut、CSWAPは変更していない。

## 保存原本と旧4 wrapper

C6a保存 `q4_sample00` のarchive SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`、events SHA256 `919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`。全20 memberは公開hash索引と一致。H packet、CDF endpoint、sector、signed atoms、lawも固定hashを照合した。元H/CDF入力は既存の保全先からbyte一致でtask専用privateへcopyし、再生成していない。

旧raw全256入力PASS、compiled4/post4、paired X/Y・χ、P0、request/event fingerprint、scalar/identity・signed angle・normalization、readout／qubit配置、保存compiled資源を確認。C3でOFF core／A/B／actual inverseを再構成し、宣言phaseを含むcanonical native IRが保存QPYと完全一致した。OFF wrapperは新規生成せず保存物を返した。S0/R0の4本は再compileしていない。

保存CAR `V` とordinary rawはarchive原本をそのまま用いる。prepare_stepsのOFF IR再構成時だけ必要なstep target placeholderは評価に使わず、返すBundleを保存V/rawへbindingしている。新しいCAR target生成・ground solve・DF fitは0。以前のcut選択を再実行せず、cut2を保持した。

対応：[INPUT_CHECK](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/INPUT_CHECK.json)、[BASELINE_REUSE](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/BASELINE_REUSE.json)、[CANONICAL_IR](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/CANONICAL_IR.json)、[旧証拠hash](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/REUSED_EVIDENCE_INDEX.json)。BASELINE_REUSE内のIR欄は資格付けの段階を示すためpendingと記録され、後続CANONICAL_IRのPASSで解消している。

## 数値経路と実行前テスト

component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fittingなし。既存official Aer doubleのsmall-unitary/exact local cache・BatchedStateActions・nofusion adapterを実行前から選択した。全256列・全131072 reference出力成分を保持し、17-qubit全Operatorは構築していない。P0回路・費用計数はsimulation adapterと別であり変更していない。

[TECHNICAL_CHECKS](TECHNICAL_CHECKS.md)の人工19件PASS。新8-mode guard6件にphysical/ordinary full phase、同frame、π/7 gauge一回、wire/rotation/gauge/mode/register・非隣接拒否を含む。旧原本guard・再利用guard・nofusionの13件を再利用した。人工PASSだけで実H4を資格付けしていない。

実環境で観測したidle CPUと空きRAMからnative numerical budgetを選び、4 compile/postは独立processへ配分した。低い固定CPU/RAM上限は追加せず、共有設定・他jobのprocess/affinity・SOURCE/output/STOPへ操作していない。q64進捗や未完了出力は照会していない。運用情報はprivateに保持する。

## 最終correctnessと技術停止の保全

全256入力raw22項目＋paired2、融合前後等5比較、新compiled4 wrapperの18比較は全PASS。raw最大成分3.154e−14、compiled最大成分4.360e−12、compiled observable Frobenius最大2.263e−12。全出力reference leakage最大成分9.437e−14。保存observableからの最終6集約比較もPASS（最大Frobenius2.378e−12）。

run01の最後のχ要約だけが1D/2D配列の型誤りで例外となった。科学的なFAIL・閾値変更ではなく、旧SOURCE/結果87 file/FAILURE/STOPを保持した。summary専用SOURCE e8a2b14eで同じχの一値を1×1行列として既存component_checkへ渡し、保存PASS4件から未完了集約を完了した。新state-action/build/compile/postcheckは0。[専用PLAN](SUMMARY_COMPLETION_PLAN.md)、[元停止](../../../artifacts/track_d_head_tail_h4/2026-11-08/run01/STOP.json)、[summary SOURCE](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/SOURCE.json)、[保存照合](../../../artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion/FINAL_PRESERVATION.json)。

資源比較には7項目だけを選び、元metric記録のmeasurement_count/global_phaseはそのまま保存する。宣言phaseのheader値の差を資源差・位相破綻と扱わない。全資源と科学call数は[REPORT](REPORT.md)・機械可読summaryを参照する。
