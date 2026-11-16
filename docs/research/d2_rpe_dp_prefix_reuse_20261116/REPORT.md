# 同じ構成のshot変更に対するtotal-state DP prefix再利用

**m12だけが変わる2配分では、m0〜m11終了時点を正しく再利用できる。** 同じconfiguration・信号・評価契約・h/target・最終round番号・frontier/pair制限のもとで、prefixのcount tableが同一であることが条件になる。状態辞書に加え、累積denominator、transition pairs、残余質量とその損失・momentの会計、最大観測transition超過を一緒に復元する。既存実装は同じ配分の完了結果を再利用するが、異なる配分のDP prefixは再利用していなかったため、別branchにopt-inで追加した。

人工13-round（m0〜m11共通、m12のX/Y shot数4→3）で通常acceleratorと全DP出力がexact一致した。通常の人工Q3 evaluateでもMSE上下界・model判定を含む全recordが一致した。3反復のwall中央値は以下。2配分の初回保存費用を含む比較と、checkpointがある2件目だけの比較を分ける。

|方式|初回DP秒|2件目DP秒|2件合計秒|2件合計速度比|2件目速度比|process peak RSS MiB|cache bytes|
|---|---:|---:|---:|---:|---:|---:|---:|
|ordinary|1.2957|1.2939|2.5915|1.00×|1.00×|46.00|0|
|memory|1.3014|0.0378|1.3392|1.94×|34.29×|46.00|41,327|
|disk|1.3152|0.0385|1.3538|1.91×|33.62×|46.00|41,463|

これは**人工total-state DPの速度**であり、Q3全体や実H4信号での速度ではない。新規実Q3、信号生成、sampling、回路build/compile、価格計算、fitは0件。B2の進行中探索・checkpoint・実行SOURCE・旧結果・旧STOPは変更していない。

実装SOURCE：`f33595ef044a0b84dd543fbc63a1298ffdd94ad6`。branch：`codex/d2-rpe-dp-prefix-reuse-20261116`。起点はD2 scheduler公開 `1eda4ba3f468dfba89823941222843ef5cfe0402`。実装者D2自身の技術検査であり、独立reviewとはしていない。

## 既存実装と重複確認

現行route view `bd42dd1e95eb1d4152216d62792ad3421b803f7c`、元kernel `710aa69b663ef3313439762d25c80a652a03948f`、accelerator `9e294778266c9c867a8b1b98e040c2e94b1cb635`、B2実装 `1e88cd85cc8d2739a4dce43d138b94b153309333`、D2共通schedulerを照合した。関連branchとGit履歴のDP/cache/prefix/reuse範囲でも同等の実装・進行中の改修は確認できなかった。

- `rpe_method_optimizer.Executor`：同じconfiguration＋配分＋評価契約の保存OUTCOMEを再利用する。shotが異なると別jobになる。
- `rpe_total_risk.tables_for/_total`：各配分でcount tables、lemma、total-state DP、valid上下界の組合せを作り直す。元partial evaluatorも自身のcount tables・closure・partial-state refinementを計算する。
- `rpe_total_dp_parallel.Accelerator`：1回のDP内のoutcome並列化・同じbinary64状態の併合・exact和の遅延計算。`_cached` は残余項のFraction換算を1回のDP内で省くもの。`__call__` は毎回状態を初期化する。
- 現行Q3 job adapterは配分ごとに別processを起動するため、単なるprocess内cacheでは次配分へ引き継げない。

今回追加したのはtotal-state DPのround境界checkpointだけ。count table生成、good closure検査、partial/coarse/lemma評価、最終min/maxの組合せ、Q3判定は既存経路で再計算する。同じ配分の旧結果を再評価する機能ではない。

## m11終了時点の依存関係

固定kernel `rpe_total_risk.total_state_dp`（上記710aa69b）を基準とする。

|prefixの量・処理|依存する量|後続shotへの依存と再利用条件|
|---|---|---|
|roundごとのcount table|そのroundのraw、X/Y shot、h、target、phase radius、exact-affine規則、列挙上限|m12変更はm0〜m11のtableを変えない。cache keyはprefixのtable全体を順序付きで照合|
|状態・flag・integer mass|prefix tables、元decoderのbinary64更新・zero/tie規則|後続shotを参照しない。状態はflagとphase hexを保つ|
|frontier pruning|現在のstate massの降順・tie順、maximum_frontier_states、m<M|後続shot数は不使用。ただしMを変えると終端pruning条件が変わるのでM一致が必須|
|pair上限・既切捨て状態|累積pairs、現在のgroup数、maximum_transition_pairs|現在roundまでの計数のみ。pairsを復元し、suffixに同じ残り予算を渡す。配分依存に上限自体を変えるならcache miss|
|denominator|prefixのcount denominatorの積|後続tableのdenominatorをまだ掛けない。m12を変えた後は新denominatorを掛ける|
|dropしたmass・loss・moment上下界|state、completed、最終M、h、target、π enclosure、元allowance|幾何半径 `π_up(2^-completed−2^-M)+(M−completed)allowance` で後続全supportを覆う。後続shot値は使わないがM/h/target/数値SOURCEは一致が必要|
|max observed excess|prefixの全計算transition|保存worstをsuffixの観測と合わせ、元の最終allowance gateを実行|
|最後のMSE・bias/variance・any_zero|保存prefix＋新suffix全体|再計算する。旧MSEや旧判定は使わない|

保存する残余会計は `mass, loss, lo, hi, sq_lo, sq_hi, guard_failed, flagged, events`。raw exact numerator/denominatorを保ち、再導出可能な内部 `_cached` だけを省く。状態とround summaries、accelerator observationsも保存する。復元したobservationsは過去のprefix計算の記録であり、今回再実行したroundではない。今回の実行範囲はHITイベントのreused_rounds欄で識別する。

frontier上限・pair上限・round数が同じだからといって別信号や別LDの状態を流用しない。namespaceにはconfiguration、signal packet、binding/target、評価契約、route SOURCEを含める。さらにprefixの全table、M/h/target、scopeの数値条件、kernel/decoder/acceleratorと今回helperのSOURCE、Python/NumPyのruntimeをkeyへ含める。scopeのトップレベル `source` だけはinvocation provenanceとしてkeyから外し、出力scopeには現在の値を残す。宣言だけが異なるSOURCEを勝手に同一視する入口は追加していない。

## 実装と通常の入口

`src/trottertracks/rpe_dp_prefix_reuse.py` は固定acceleratorの元 `_kernel` 生成処理をそのまま使い、outer round loopの直前に復元、各roundのmass gate後に保存を追加する。固定accelerator methodのbyte SHAと元DPのAST hash・置換数guardを実行する。元のguard期待値を差し替えたり、kernel fileを編集したりしていない。transition、frontier、drop、final moment式の書換えはしない。

既定保存点は最後の1つ前、13 roundならm11。m12以外の変更にも対応するには `prefix_checkpoint_rounds` に変更前の境界を明示する。例 `[5,11]` はm6以降が変わる場合にm5までを再利用できる。より早いtableが変われば一致する境界までしか戻さず、該当がなければ通常計算する。全roundのcheckpointを既定で増やすことはしていない。

通常CLI → concurrent executor → REQUEST.execution_profile → 元Q3 adapter → `prefix_cache_context` → 元supervisorのfork → 既存Observed constructor、という実経路へ接続した。親contextでこのworkerのaccelerator constructorにhookを入れ、fork後も有効であることを、保存Observed＋人工tablesで検査した。Full PF・truncated PF・PR-Pauli・PR-DFが共有する同じfixed acceleratorを対象にする。実科学の4routeで新たなQ3を実行したという意味ではない。Linux/forkと固定SOURCEで検査しており、spawn経路は未検証。

disk modeは各jobが同じ所有cache directoryを開く。exact値はJSONのhex integer、binary64はhex、Mass/Fractionは分子分母を分けて保存し、実行可能なpickleは使わない。checksum付きpayloadを一時fileからatomic replaceする。並列jobの同じprefixへの保存は互いに未完成fileを読まない。cacheがない・不一致・壊れている場合はその理由を記録して通常計算へ戻る。保存失敗はcache失敗として記録し、科学的なPASSを捏造しない。科学結果の保存・gateは元runnerに委ねる。

同時にcold startした2配分がprefixを両方計算する場合はある。prefixを待つために探索順・科学call budget・lookaheadを変更する処理は加えていない。変更した配分のQ3は従来どおり1件の新評価として数え、過去の完了Q3を新規計算扱いにしたり、新配分を0 call扱いにしたりしない。

## 保存資料による確認

[SAVED_INPUT_CHECK.json](SAVED_INPUT_CHECK.json) は公開済みFull PF S2の次の2件をmetadataだけで照合した。

- `b4296b110b129d226dbef2743c14f725a27c3bf6:docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json` cases[1]
- `1b96c832985caf7cbd85a69450258d6ede676faf:docs/research/full_pf_s2_m12_two_results_20261116/INPUTS.json` cases[0]

configuration `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d`、configuration/signal/binding/target、全評価条件が一致し、違いはm12の各軸4→2 shotだけ。frontier=8192、pair上限141559036も同一。したがって同runtime・同じ検証済みcount prefixであれば再利用条件に合う。

**旧保存物にはm11のstates・denominator・res本体がない。** 旧round summaryやMSE値から再構成することはできず、今回も再計算していない。次の承認済み評価でcheckpointを作った後に初めて再利用できる。保存資料による確認を、実H4での新旧数値一致・速度実測とは扱わない。

## 人工検証・資源の読み方

[TESTS.txt](TESTS.txt)：新規prefix検査10、共通scheduler関連15、既存accelerator人工検査7がPASS。全DP出力exact一致、通常evaluateのMSE上下界と判定の一致（人工閾値の両側）、zero/tie、残余mass、frontier=0、pair境界、scope/h/target/M/namespace不一致によるmiss、早い境界までの部分hit、disk再読、巨大整数とsigned zero、破損cache拒否、元guardの有効性、保存Observedをforkした実worker入口でのhitを確認した。H4/P0数値再検査suiteは起動していない。

[BENCHMARK.json](BENCHMARK.json) は固定SOURCEでの3反復9 worker process。13人工round、通常はX/Y各4 shot、変更後はm12だけ各3 shot、frontier256、pair上限10,000,000、workers1。小さい人工表の逐次依存とcache比較を測るためworker1を使い、process並列の同等性は別testでworkers2も確認した。実運用のCPU/worker上限を提案した値ではない。

wallはrestore/save/hashを含むDP呼出し時間。RSSは各modeを新processで実行したprocess peakで、interpreter/fixtureの読込も含む。disk表はcheckpointを再openした次jobを模擬し、別processの共有cache hitそのものはfork worker検査で確認した。実H4の大きな整数frontierでのIOとメモリは未実測。diskは読み込んだ1 snapshot、memory modeは保存stateのcopyを保持する。未使用cacheは消さず、サイズは異なるprefixの数に応じて増えるため、B2は次runの所有出力先として配置し実サイズを記録する。

初回のcheckpoint保存に小さなoverheadがあり、m12以外の早いroundを変え続ける探索や同時cold実行では利益が小さくなる。上表の2件目速度比をQ3全体の速度比に外挿しない。

## 引渡し・ロードマップ

[HANDOFF](HANDOFF.md) にB2用の設定、呼出し例、保存SOURCE・入力locator、未確認範囲を記載した。既存価格、shot探索の選択順、精度契約、Q3判定、科学最安点は変更しない。次の承認済み探索で異なる配分の共通prefixを再利用できる実装を追加したことがロードマップへの差分。実H4による新旧数値一致とend-to-end速度確認は今回の証拠には含めない。
