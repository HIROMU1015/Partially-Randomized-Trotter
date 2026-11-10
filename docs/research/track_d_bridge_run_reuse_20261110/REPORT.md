# Native Bridgeのrun内部再走査削減：限定技術結果

2026-11-10 JST。**部分的な削減に成功したが、全unit走査とCX frontier再構築は残る（依頼のケースB）。** 前回高速版に対してnative段階でq4は1.15〜1.18倍、q8は1.32〜1.33倍高速化した。q8のprocess peak RSSは約4.0〜7.5%増えた。長q向けの漸近改善や省メモリ化を達成したとはしない。

[実行前PLAN](PLAN.md)／[実装](IMPLEMENTATION.md)／[正確性](CORRECTNESS.md)／[性能](PERFORMANCE.md)／[次の判断案](NEXT_EXECUTION_PROPOSAL.md)／[STOP](STOP.md)。研究系列はリソース評価／Track D／非full-compile価格検査器。既存branch `codex/resource-eval-track-d-longq-structure-20261109` を継続した。

## 固定範囲とSOURCE

基点 `e6bd6c5dedf27601b0c3b50877653ad43d04cd17`。新engine・PLAN・H4比較runnerの固定SOURCEは `a01dc6620037d9feccd00560e0a6b8d203c06dc0`。保存54人工caseの薄い再利用adapterは `f09908649`（同じ履歴の後続commit）で、計測済みengineは変更していない。各実行の[ SOURCEと結果](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/)が正本。

対象はLinear H4／隣接1.00Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元frame順LD2／ordered partial S2/s1/r32/K0／h=δ=.1。保存q4_sample00/cut2とq8_sample00/cut4、各Strong symmetric-directional／whole-path一gadget Reference、X/Yの8 wrapper。C1 head-tail追加融合OFF、元H厳密ground仮定・状態準備費除外。既存のbasis共有・step融合・scalar/identity・actual adjoint・χ・CSWAP・readoutを含むnative原本を使った。

保存P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=False／topology-free／既存宣言phase保持。新しいH4 raw生成・raw読込/lowering・P0・sampling・state/CAR simulationは行わない。前回固定native gzip8とphase入力8のhashを実行前後に照合した。価格は照合専用で、予測ruleや補正定数へ利用していない。

## 結果と成立範囲

- 人工・技術回帰70 test method PASS（旧63＋新7）。新7内の781短wordと固定100技術列はテスト下位ケースであり、分子標本数に数えない。保存54人工rawの既存差分pipelineもPASS、新P0は0。
- H4旧新8対で、最終opcode／wire／clbit／順序／parameter bitのhash、各論理反復のword hash、反復size・置換数・CX相殺数、binary64 phase、全round phase項のbit、MPFR160 ledger全recordが一致。
- 全8×7＝56資源が旧＝新＝保存P0。q4は10反復、q8は20反復のまま。
- q8 Reference Xでは、1Q分類訪問2,164,443からnative分類721,977＋変更run展開545,681へ減少。ただしunit metadataを1,180,011回訪問し、別の仕事も残る。CX feedは1,825,588→792,992（56.6%減）。1Q合成は2,953→2,957で減っていない。前回の合成cache効果と今回の内部走査省略を区別する。
- 同一16-process poolでのnative decode＋engine＋最終T0比較は全体19.798秒。個々の処理時間／RSS／全走査指標は[PERFORMANCE](PERFORMANCE.md)を参照。raw/loweringやfull P0とのend-to-end速度比ではない。

新しいrewrite規則・近似・閾値・価格補正は0。今回の一致は固定旧bridgeの再現であり、価格一致を新しい量子作用correctnessや一般P0互換へ格上げしない。q4/q8以外・全law期待価格・q1024〜4096・routing／誤り訂正合成費用の保証は得ていない。A/F/S、必要N、全13-round会計は不変。

## 研究目的への含意

有限校正・整合性検査器としての速度改善は保全する価値がある。一方、全epochでのunit走査、可変長frontier、O(L)記憶とraw QC入口が残り、q1024〜4096への実用的な経路が確立したとは言えない。今回はこの一方式で区切り、完全DAG/worklist compilerの開発へ拡大していない。改善版を保存し、既存A/F/Sを条件付きproxyとして使うPR＋RPE共同最適化へ重点を移す案が合理的である。これは技術的な推奨であり、正式な研究方針の採択はユーザー／GPTへ戻す。

## 保全と終了

[原本照合](../../../artifacts/track_d_bridge_run_reuse/2026-11-10/PRESERVATION.json)。新namespaceのコード・結果・scalar性能・hash索引だけを公開する。raw/native角列／QPY／NPZ／個別phase原本／内部保存先は非公開を維持する。旧SOURCE・結果・grant・ledger・marker・STOP・他worktreeは変更しない。公開ref/tree/bytes照合後mandatory STOP。追加q・科学計算・詳細科学レビューへ自動進行しない。
