# 性能対策の再利用索引

## 最適化器の完了Q3出力の重複削減（2026-11-18）

通常のconcurrent探索で `ROUND.json` の `executor.compact_q3_evidence` を `true` にすると、PR-Pauli/PR-DFのcommon-entry workerは `API_RETURN.json` と `PARTIAL_*.json` 3種を個別に書かず、完了まで圧縮した1つのSQLiteチェックポイントに保存する。固定validatorはworkerが書いた値を読み、元のMSE・数値gateを適用する。正常終了後、`RESULT.json` から4種を復元したbytesの完全一致を確かめて、小さい `Q3_COMPACT_EVIDENCE.json` を保存してチェックポイントを解放する。中断、UNKNOWN、照合差がある場合はチェックポイントを保持する。

[Q3 worker adapter](../scripts/run_rpe_optimizer_q3_job.py)の `collect` と通常のC exportはそのまま使える。直接ファイルが必要なら `--mode export-evidence --request RUN/jobs/METHOD/JOB/REQUEST.json --output RUN/jobs/METHOD/JOB --evidence-directory NEW_DIRECTORY` で元のbytesを別ディレクトリに復元できる。分散環境ではcontrollerとworker adapterの対応版をそろえる。旧版や対象外のrouteは元の形式で保存する。

人工の通常入口で有効判定とMSE、元バイト列の復元を比較した。関連ファイルは5→2件、283,862→139,720 B（50.8%減）。既存R08の正常完了111件の照合では4種合計0.97 GiBが同じ `RESULT.json` 内から復元できた。R08への適用、新規実H4 Q3、実H4での性能測定は未実施。

## Q3のDP中間キャッシュの圧縮と容量管理（2026-11-18）

通常入口の `prefix_cache_directory`（分散時はworker側の保存先）に対し、[PrefixStore](../src/trottertracks/rpe_dp_prefix_reuse.py) が中間状態を可逆gzipで保存する。旧V1 JSONも同じ評価条件とruntimeで照合して読める。上限はディレクトリ単位で既定40 GiB。`ROUND.json` の `executor.prefix_cache_max_bytes` で正整数のバイト数を指定できる。新規保存の直前にロックを取り、上限を超える場合だけ、最後の保存・読込が古いキャッシュから解放する。単体で上限を超える状態は保存を省略する。

次の固定版にはcontrollerとQ3 adapterをそろえて配置し、同一キャッシュへの旧版workerからの書込を避ける。保存対象は再計算できるDP中間状態。Q3の原API、RESULT、receipt、再開状態、固定kernelは維持する。 `DP_PREFIX_EVENTS.jsonl` の `serialized_bytes` と `evicted_bytes` で実際の保存量と解放量を確認できる。上限はデータディレクトリ全体には及ばない。

[人工13ラウンドの既存fixture](../scripts/benchmark_rpe_prefix_boundaries.py)を使った各3回の比較では、3ファイルの合計が110,437 Bから37,591 Bへ約66%減り、出力とHITの位置が完全一致した。片方の末尾を変更した評価で新workerから旧形式を読み、12ラウンドを再利用した。中央値は旧2.406秒、新2.422秒。新規実H4評価0件。実H4での削減率と速度は未測定。

## PR探索の計算順とQ3残差積算（2026-11-18）

[Claudeの探索・計算レビュー](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5e9e08ef15d1ff384b45b36caeb3436641350c20/docs/research/a_pr_search_strategy_review_20261118/REPORT.md) の技術改善を、容量対策済みの通常 `scripts/run_rpe_schedule_search.py` へ接続した。新しいroundの設定で必要な機能を有効にできる。

| ROUND.json の設定 | 適用 |
| --- | --- |
| `search_overrides.self_price_floor: true` | 26軸の有効な価格と各軸1 shotの下限を、Q3で適合した方式別費用と比較する。改善不能な構成は新規の通常選択と空き枠補充で保留する。単価や最安が変われば再判定する。 |
| `search_overrides.review_search_order: true` | UPDATEと既存の探索間隔を維持し、新しい幅・分割の最初のEXPLOREを優先する。高価な小κの準備済み点がある場合に、同じfamilyでそれ以上のκの準備順を後ろへ移す。既存の `allocation_exchange.enabled` も設定した場合は、有力な減算候補が尽きた最安構成に費用減の交換を提案する。 |
| `search_overrides.common_pauli_price_floor: true` | 同じ価格規則で比較元の適合費用を再現し、同一物理family内で単調なPauli MODELの下限を証明できた場合に、上位κの新規準備提案を絞る。実stepや監査済み時間、別の価格出所も確認する。PR-DFのκ方向は候補別の価格較正に依存するため、準備順の判断に留める。 |
| `executor.parallel_dp_residual: true` | 固定のDP省略項を分母ごとに厳密加算し、大きなバッチは既存DP worker数の設定内で並列加算する。cacheの有無にかかわらず適用できる。 |
| `candidate_completion.parallel_df_price_axes: true` | 固定軸別workerを使い、同じpathのX/Y wrapperを別プロセスで処理する。seedでイベント列を決定論的に再現し、同じdigestのイベントファイルを共有する。軸別のbuildメモリと時間は要確認。 |

自価格下限の成立範囲は、採用中の価格表で求める条件付き総費用に限る。確定済みの結果は保持し、queueからの取下げは未割当のQ3に限る。PR-DFで複数モデルの出所指定待ちと分かる場合は、その価格較正の追加投入を保留し、競合軸を記録する。

人工テストで通常選択・空き枠補充・再価格付け、並列DPの正確な上下界、圧縮cacheの再開、軸別receiptと共有イベントを検証した。32,768件の人工省略項の厳密加算は3回中央値で、元の積算0.900秒、分母別1 worker 0.300秒、分母別4 worker 0.112秒。測定対象は人工入力の積算部。実H4の速度・探索短縮は未測定。並列残差を有効にしたcacheは残差SOURCEを識別するため、新しいkeyで蓄積する。`prefix_checkpoint_rounds` は既存設定を使う。例えば `[5,8,11]` は明示設定できるが、既定の境界は今回変更していない。

## 同一PR-DF保存信号の再利用（2026-11-18）

[通常のdf-plan](../scripts/pr_candidate_services.py)は、固定validatorを通過した同一物理キーの保存レコードを照合する。固定stageが再結合するBの宣言（候補・構成IDとcanonical identity/law binding）とその再利用履歴を除き、信号、入力・reference、数値gate、allowance等を完全一致で比較する。一致すれば元の保存SOURCEから再利用する。差が残るroundは `saved_signal_conflicts` に両出所と相違項目を記録して保留する。Completionはこの確認を信号・較正workerへの投入前に行い、他候補を通常キューで処理する。

R08で過去に `df-plan` が保留した3件の保存入力を、従来の固定providerとmetadata専用入口で再確認した。元候補キー `b8cba2bd1990188c53f8` と `75a10d084f0948c5f712` は各10 round、`7fc104bbd75cc70800c1` は8 roundを再利用できた。残りは順に3・3・5 roundの計画上の不足。新規信号・compile・Q3は0件。人工回帰では値・数値gate・reference SOURCEの差の保留と、中央準備の投入前に止まることを検査した。稼働中R08への導入は固定SOURCEを切り替える別の手続で行う。

## 探索を止めない共有モデル診断（2026-11-17）

`rpe_price_validation.Service/ModelStore`と既存分散preparation queueを再利用し、凍結モデル＋物理domain単位の初回照合を全配分・run・workerから共有する。固定requestをqueue投入前に保存し、optional compileを通常Q3/準備より後にclaimする。checkpoint再開・同時要求・失敗時の探索継続・通常C同期は人工検査、LD2/PFの独立実測再利用は保存資料で確認。新規実compileの速度・実host配置は未検証。単一controllerの共通namespaceを使用し、別cloneの独立storeを共有とみなさない。[REPORT](research/h4_price_validation_integration_20261117/REPORT.md)／[HANDOFF](research/h4_price_validation_integration_20261117/HANDOFF.md)。

## PR探索の候補分離・後着READY投入（2026-11-17）

`hold_candidate`と既存`ROUND_RESULT.execution_holds`で旧job照合待ちをconfiguration単位に限定し、無検証再利用/retryを避けて他候補を継続する。CompletionのREADY snapshotを元installerへ渡し、準備wave全体を待たず同じqueueでQ3を進める。有限総予算と重大UNKNOWN停止は維持。[REPORT](research/h4_pr_candidate_continuity_20261117/REPORT.md)／[HANDOFF](research/h4_pr_candidate_continuity_20261117/HANDOFF.md)。人工82件・native8件PASS、実H4の速度倍率は未測定。R07 STOPを残す新SOURCE/R08で継続。

## PR複数幅のR06→R07継続（2026-11-17）

候補自身のconfigurationで既存PriceRulesを照会し、保存総額のexact一致で省略26軸を復元する。DFは固定entryのpacket別検査を再利用して、今回のphysical keyだけをmergeする。対象keyのrecord差は従来どおりSTOP。READY入力・完了Q3・prefix cacheをコピー/元SOURCE参照で再利用し、旧STOPへ書き込まない。[REPORT](research/h4_pr_multiwidth_resume_20261117/REPORT.md)／[再開](research/h4_pr_multiwidth_resume_20261117/HANDOFF.md)。人工25件と実入力metadataで確認、科学開始後の速度倍率は未測定。

## PR複数幅：保存候補のQ3 routeとtemplate適合性（2026-11-17）

`Completion.install_saved_routes/common_route`と`Executor.route`は保存候補のSOURCE/bindingを既存common workerへ結び、Reader cacheでtemplate照合を共有する。有効な旧route/job keyは保持し、未接続候補のみ保留して他のqueueを進める。R06ではcatalog-only候補のroute登録漏れと、DFのwildcardに未対応templateを使う失敗があった。人工72件・停止2候補の元worker metadataで確認、実科学call0・実H4速度倍率未測定。[REPORT](research/pr_multiwidth_route_fix_20261117/REPORT.md)／[HANDOFF](research/pr_multiwidth_route_fix_20261117/HANDOFF.md)。旧STOP、数値UNKNOWN、保存値の競合は解除しない。
## 分散Q3の配送・body・回収・採用計時（2026-11-17）

`rpe_q3_timing`と通常transportの小さいstage/eventを再利用し、`summarize_rpe_q3_timing.py RUN --output FILE`でhost/attempt別の秒数・欠測・異なる時計の包絡を出す。SOURCE19c43576c、集計5897d26ab。旧実3host18人工jobはbody欠測、初回約235秒と後続を分離。新模擬3hostは18/18 body/採用を計測、再開0起動。元native人工DP exact一致・cache HIT。実H4速度とrelay並列化効果は未測定。次の固定SOURCE・新runへ導入し、稼働SOURCEは変更しない。[一元HANDOFF](research/rpe_q3_timing_20261117/HANDOFF.md)／[実測と限界](research/rpe_q3_timing_20261117/REPORT.md)。

## 候補ごとの実入力適合ホストへ割当

通常schedule入口の準備で `HOST_ELIGIBILITY.json` を作り、distributed controllerがREQUEST/SOURCE/入力とhostの現在fingerprintをclaim直前に照合する。先頭jobが非対応でも既存優先順位の独立候補へ進む。全host不足は予算予約前に保留し、busyな適合hostは待機する。方式名やjob IDの固定割当はしない。既存attemptの回収・再開は保持。新transportと固定科学adapter SOURCEを分ける。人工queueの検査であり実H4の速度測定はない。[使い方と根拠](research/rpe_q3_host_eligibility/REPORT.md)。

## H4計画の差分取込・小さい出力・早期READY（2026-11-17）

`run_rpe_schedule_search.py --plan-only --prepare-readiness --preparation-cache DIR` は既存method optimizerへ、context再利用と `rpe_plan_history_cache` のexact履歴差分readerを接続する。適合/不適合を候補選択前に取込み、未確定job/queueは毎回観測する。巨大な評価配列を繰返しdecimal化せず原OUTCOME/私物cacheへ参照し、明示dry-check後のREADYをREPORT保存より先に返す。準備実装b5a8a60c5、計算SOURCE28d、旧339の遠隔配布物はbyte不変。

固定H4 metadata/dry-checkの実測は初回READY106.53秒・入力不変6.47秒、REPORT保存107.30秒/6.97秒。これは科学Q3性能や利用者実測50分との倍率ではない。変更したjobは既存identity/route検査へ戻す。保存追加による選択更新を人工検査。失敗例：巨大Fractionのcache復元でもGCD再計算が残り、既存cache互換の正規化済み復元で除いた。cacheは自分生成のlocalだけ、入力/科学条件変更では無条件再利用しない。[実測・限界](research/rpe_h4_readiness_fast_20261117/REPORT.md)／[1コマンド](research/rpe_h4_readiness_fast_20261117/README.md)。

## 複数ホストQ3・SSH mailbox中継（2026-11-17）

次回固定版`28d976b99`の`rpe_optimizer_concurrent.DistributedExecutor`と`rpe_q3_store/worker/coordinator`は通常schedule入口の同じ候補順/予算/後着READYを保ち、独立Q3をhostの空きslotへ割り当てる。local0 relayはWindows PowerShellから既存qg1/qg2/RomuTailを使い、科学選択は中央だけが行う。SOURCE/route/input/評価契約を照合し、DP並列化とcacheは既存adapter/helperをhost内で使用する。選択routeの固定blob hashをprocess内で再利用し、無関係configurationの入力を要求しない。DP worker数はhostごとに解決し、元の科学REQUEST/job IDを変えない。

実2台と実3台の各18人工jobは単一host参照とMSE上下界/判定/方式別最安がexact一致し、完了run再開は追加起動0件。別の実2台fault検収で、relay停止中の完了、controller中断後の同attempt回収、重複完了での予算・採用不変を確認した。最終計算SOURCE`1ffdf610a`の模擬2/3host人工速度は分散1host比2.10/2.89倍だがlegacy参照比0.68/0.94倍で、実H4性能は未測定。人工DPでは逐次exact一致、4 HIT各12round、同host donorと別host cacheを確認。28dの計算8ファイルは1ffとbyte一致。

失敗例のlost-poll枠超過、ACCEPT後公開中断での待機解除漏れ、保存結果読取時の再利用件数漏れ、別processでの長いexact整数JSONの読取を修正・回帰検証した。Windows SSH子processのウィンドウ発生を抑止し、切断時のSSH再試行は最大30秒へbackoffする。結果受領記録は任意の転送計測失敗に隠れないようにした。[適用条件・証拠・運用](research/rpe_distributed_q3/HANDOFF.md)／[REPORT](research/rpe_distributed_q3/REPORT.md)。

## PR複数幅と共通queue/DP・B0Iの通常入口統合（2026-11-17）

`run_rpe_method_optimizer.make_executor`を幅版batchと共用し、Completion後も同じoptimizer/queue/予算/profileを維持する。元prefixのSOURCE・信号・契約・精度照合を再利用し、別構成へのHITや完了jobの二重起動を防ぐ。明示overlapでは準備前snapshotを使い、準備後に同じqueueへ戻す。B0Iは既存自動p選択を同じschedule入口から呼ぶ。

人工8入力で逐次参照とMSE/判定exact一致、4 HIT/各12round、後着READYと同一invocation、再開0起動。実H4はplan-only/保存metadataのみで、実速度や最安幅を認定していない。相対出力制約は所有view内のrunで満たす。[SOURCE・検査・注意点](research/pr_width_throughput_integration_20261117/HANDOFF.md)。

## PRの複数要求幅：実stepごとの準備再利用（2026-11-17）

`run_rpe_schedule_search.WidthDriver` は方式・分割・公式・exact要求幅・Kごとに既存mapperのschedule区間でκを探索し、共通Completionから同じV2/Executorへ戻す。`pr_step_price_registry.py` は固定監査結果をactual step・辞書・law・構造・wrapper時間の範囲だけに登録し、複数候補で共有する。同一物理scheduleの宣言aliasはH/state/辞書/target/law/SOURCEを含む契約で照合し、完了済みQ3を再起動しない。

人工系で両方式の別幅選択・部分価格の取込・同run再開0起動を確認した。保存JSONで旧単一幅とround04較正要求を再現したが、別幅の実H4信号・価格・Q3の性能は未測定。旧価格wrapperで新auditを無条件要求した回帰は、旧経路を維持する分岐と回帰テストで修正した。[SOURCE・適用条件・検収](research/pr_requested_step_search_20261117/HANDOFF.md)を参照。

既存の大規模計算の手法集は固定 `481c4b7abffbf070347f697980f2ec051e6a6538:docs/PERFORMANCE_METHODS.md` を参照する。本C branchへその科学runnerを取り込んだ意味ではない。

## 保存RPE結果の読取

`src/trottertracks/saved_rpe_cost_catalog.py:Reader.document` は既存Git blob readerを使用し、固定commit/pathをprocess内でcacheする。同じSUMMARY/INPUTS/価格表が複数HANDOFF・配分から参照されてもJSONの再読取を省く。seed/計算順/価格値は変更せず、適用対象は保存JSONのmetadata処理だけ。

既存B2の保存summarizer同様、Python process内の整数文字列桁上限を解除し、exact MSEの長い整数を損失なく読み書きする。共有環境設定や科学runtimeを変更しない。

[再利用・検査・実行範囲](research/h4_saved_rpe_cost_catalog_20261116/HANDOFF.md)と[検収](research/h4_saved_rpe_cost_catalog_20261116/TESTS.txt)を参照。今回の固定10入力の実処理時間はCATALOG.jsonに保存。人工＋公開値の読取で確認したが、速度倍率や将来の大入力性能は測定していない。

## RPE最適化器：並列実行・先読み・候補選択・前回roundの再利用（B2、2026-11-16）

- **構成別scopeの並列実行器と先読み**（`1e88cd85c`、`rpe_method_optimizer.Executor`）
  - Q3 jobを `max_concurrent_jobs` 個のslotで同時に走らせる。閾値に近いときだけ、規則の順序で最大 `lookahead` 件を先に投入する。
  - 答えは規則の順に1件ずつ使うので、探索の軌跡は逐次実行と同じになる（人工系の受入で確認）。
  - **実測**：κ round01（18 slot、9構成）では、8時間11分で4,229件、約540件/時、1件約105〜165秒。
  - **失敗例**：先読みのうち895件（21%）は採用されなかった。閾値付近では、slotが先読みで埋まって止まる時間がある。
- **候補選択V2**（`cbe2947ad`、`rpe_priority_search.py`）
  - 方式ごとに1本の優先列で、最安を下げる候補を先にする。不適合点（MSE/τ ≥ 1+1e−4）以下の配分は後回しにする。
  - **保存ログでの再生**：停止時最安に届くQ3は、4,148件（旧の実際）から218件になった。旧が作ったログに条件付けられた比較で、未評価は6件。並列での効率は未測定。
  - **適用条件**：MSE上界は、shotについて厳密には単調ではない（|MSE/τ−1|<1e−7で27組）。後回しにしても、候補は落とさない。
  - 根拠：[kappa_round01/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_round01/REPORT.md) §3・§4。
- **前回roundの再利用**（`cbe2947ad`、`Executor.prior_run_dirs`／`prior_records`）
  - 同じjob key・同じ経路の完了jobを、identity照合の後に知識へ読み込み、再起動しない。実対象で4,229件を読み込んだ。
  - **途中終了したroundの記録**：`--interrupted` で、完了jobだけを使って規則を再生する。先読みとREQUEST-onlyの件数は分けて記録する。

## RPE最適化器：V2の並列先読み（B2、2026-11-16）

- **有限予算での先読み**（`5d9d94e12`）と、**連続不適合に応じた深さ**（`26d84d90a`）
  - 答えはV2の順に1件ずつ使うので、採用される軌跡は逐次実行と同じになる（人工系で確認）。
  - **失敗例1**：round02では、既定の上限40手でbatchが止まっていた。予算で終わるように直した。
  - **失敗例2**：適合が続く1構成だけを下降している間は、先読みの75%が無駄になった（起動166件、採用40件）。深さを適応させると、今度は同時実行が平均2.2件に下がった。
- **連鎖の先読み**（`a58a0f31f`、`rpe_priority_search.chains`）
  - 直前の評価が適合した構成について、次も適合すると仮定し、その次の最安点から出る候補を最大 `lookahead` 手先まで投入する。
  - **人工系**（2秒のworker、予算16）：軌跡は逐次と全長で一致した。実時間は42.1秒から25.8秒、job数は16件から55件。
  - **実runでは未使用**：round03は、実装の前に規則どおり停止した。
- **前回までの評価の再利用**（`prior_run_dirs`）
  - round03では、round01の4,229件とround02の185件を読み込んだ。
  - **失敗例**：経路の登録表key（`configurations_file`）があると、前回の経路と別物と判定されて読み込めなかった。照合でこのkeyを無視するよう直し、回帰検査を加えた。
- 根拠：[kappa_abs_round03/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_abs_round03/REPORT.md) §6、実行器受入 21/21。

## PR-Pauli κ探索：schedule単位の候補・共通入口のbatch・連鎖の先読みの実測（B2、2026-11-16）

- **κの物理schedule区間**（`24b51fbc3`、`rpe_kappa_schedule.py`、`kappa_neighbourhood_plan(..., sched)`）
  - 同じ13 roundのR vectorになるκを1点として扱い、次に異なるscheduleへ厳密に移る。これで、同じscheduleの重複評価を避ける。
  - **実対象で確認**：LD5では、範囲[1/2, 8]の中のscheduleが2つしかない。2点の評価で、この範囲を尽くした（旧規則では、κ=4/3・2・3が同じscheduleだった）。
  - **適用条件**：mapperが R_m=max(J, ceil(ceil(κc_m)/q_m)) であること。生成構成で、毎回の再現を検査している。
- **信号・価格のround単位の再利用**
  - 7点の新規平均値は、R_mが変わった16 roundだけで、全91 roundのうち75 roundを再利用した。
  - 価格は、Cの共通入口（`02e7cc6a1`）のbatchで26軸を再利用し、較正は0件。入口は、同じ較正cellを1件にまとめる。
- **連鎖の先読みの実測**（実runでの初めての使用）
  - Q3は284件で、平均145秒。同時実行は平均10.1件。
  - 採択82件で、V2は72分（逐次なら約198分）。
  - **失敗例**：適合と不適合が交互に出る下降では、先読みの71%（202件）が採択されなかった。
- 根拠：[kappa_abs_round04/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/REPORT.md) §2・§5、検査 `kappa_abs_round04/checks/KAPPA_SCHEDULE_CHECKS.json`（7/7）。

## PR候補準備の再利用と方式間並行（D2、2026-11-16）

- `pr_candidate_completion.py` は既存V2が選んだ不足候補を方式別に並行準備し、元の同一Executor・累積予算へ戻す。候補順/shot規則は再実装しない。
- 信号は既存の物理key、価格は既存PriceRules、κ宣言aliasは元B2登録を再利用する。DFの同一分割descriptor・保存cohort/個別axis receiptを共有し、欠測軸だけ取得する。完了Q3は元job keyで再利用する。
- 実対象でPauli LD4/DF LD6の準備→Q3→C同期と完了runの新規Q3 0件を確認。部分axis再開・後続候補継続は人工検査。速度倍率は測定していない。
- 失敗例：最初のdescriptor envelope不足は科学call前に停止。DF supervisorの終了receipt形式をPauli形式として読んだ誤りは、保存exitcode/identityをcollectして解決し、再Q3していない。
- SOURCE・適用条件・検収：[API](research/d_pr_candidates_auto_20261116/API.md)、[REPORT](research/d_pr_candidates_auto_20261116/REPORT.md)。曖昧な稼働jobや失敗leafをretryせず、既存科学STOPを保持する。

### 固定PR-DF信号workerの読込と候補並行時の照合

- [信号worker](../scripts/run_pr_df_candidate_signal_jobs.py) は、固定`fefd548ec`のprovider群を専用packageで読み込む。運用パス解決で先に読み込んだcontroller側packageと分離し、ローカル・分散とも同じworker入口を使う。viewのHEAD、主要な固定ファイルのbyte列、実際のimport元を照合する。子processの遅延importも専用packageを使う。
- [候補準備](../src/trottertracks/pr_candidate_completion.py) は、別候補を登録できる間も同一物理候補のREADY・旧失敗を同じ候補一覧で照合する。重いGit読込みは一覧取得用のlock外で行う。同一物理候補の準備lockと失敗後の自動retry禁止を維持する。
- R08で旧`DF_SIGNALS`の8候補がprovider import時に停止した。この8候補が要求した延べ39 roundについて、保存設定と固定referenceを照合した。`bc3d1625c731cc58bf79`ではfork先の依存モジュールのimport元も照合した。人工検査は[workerの起動・fork・保存・再開](../tests/test_pr_df_candidate_signal_jobs.py)と[並行登録中の旧失敗照合](../tests/test_pr_candidate_completion.py)を含む。実H4信号の追加計算は0件。実信号からQ3までの完走と速度はこの修正では未測定。

### PR-DFの較正範囲が重なる場合の価格照会

- [価格照会](../src/trottertracks/pr_step_price_registry.py) は元の物理条件検査の後、round・軸ごとに適用可能なモデルを調べる。受入済みの単価が空欄なら、同一物理点の一意な直接観測を優先する。重複登録では較正範囲内のモデルが一つなら使い、複数なら出所選択待ち、外挿しか残らなければ追加根拠待ちとする。較正範囲は保存済みの `q/R` の最小値・最大値であり、適用した値は従来どおり条件付き参照価格として残す。単独登録の既存外挿とその表示は従来のまま。
- [候補準備](../src/trottertracks/pr_candidate_completion.py) はこの2種の保留で較正を再投入せず、照会規則または登録表が変わったときに保存情報だけで再照会する。[軸別照会](../tests/test_pr_step_price_registry.py)と[通常入口・再開](../tests/test_pr_width_completion.py)を人工検証。新しい実H4価格、信号、Q3は取得していない。

### 候補準備と過去Q3の差分再利用

| 対象 | 通常入口での適用 |
| --- | --- |
| Full PF、B0I、PR-Pauli、PR-DF | [共通executor](../scripts/run_rpe_method_optimizer.py) が `prior_run_dirs` のQ3を方式別のローカルSQLiteに索引化する。候補の物理ID・route・評価契約・保存ファイルの変更時は再照合し、保留を引き継ぐ。総費用は現在の単価で求める。幅探索のサイクル末の全完了点照合でも同じ索引を使う。比較用に `executor.prior_history_cache: false` で従来経路を選べる。 |
| PR-DF | [信号準備](../scripts/pr_candidate_services.py) の `df-plan`、`df-stage`、`df-bundle` で、固定SOURCEとdescriptorで検証済みのpacket索引を共有する。packet、構成、rawを完全なcommit参照で固定できるものが対象。追加packetは個別に検証し、同じ物理roundに異なる証拠があれば従来どおり保留する。 |
| PR-DF | [候補準備](../src/trottertracks/pr_candidate_completion.py) では、価格の出所選択待ち、または追加根拠のない外挿保留が先に分かれば、未着手の信号取得を保留する。価格登録の変更後は同じ構成から再開可能。live-ready経路では `candidate_completion.parallel` の既存設定を使って異なる分割を並行準備し、同一分割内は順番を保つ。科学枠・同一較正のlock・着手済みstageの回収は既存経路に従う。 |

PR-Pauliの候補間では共有worktreeへの書込を直列化する。Full PFとB0Iの自動補完は既存の有限候補数を守り、独立した信号・価格の並行処理を使う。新しい索引だけが置かれる `cache/prior-q3` と `cache/df-saved-index` は再生成可能でGit管理外。DP prefix-cacheの境界はこの変更の対象外。

人工・回帰検証で、[4方式の履歴再利用と保留](../tests/test_rpe_plan_history_cache.py)、[packet追加時の再検証と衝突検出](../tests/test_pr_saved_quote_resume.py)、[価格変更時の再開](../tests/test_pr_width_completion.py)、[分割間並行と同一分割の順序](../tests/test_pr_multiwidth_launch_contracts.py) を確認した。実H4の速度倍率は未測定。

## PR-Pauli広域schedule探索：粗い探索の優先・HYBRID並列・価格下界による打ち切り（B2、2026-11-16）

- **価格下界による打ち切り**（`run_rpe_schedule_search.py` の `prune`、`rpe_schedule_search.plan_wide(price_floor=)`）
  - 候補scheduleのMODEL価格で、1 shotずつの配分の費用を計算する。これは、そのscheduleで最も安い配分の費用になる。
  - この値が照合済みの全体最安以上なら、信号・Q3の前に除外する。価格はRとともに増えるので、それより大きいκも同じ理由で除外する。
  - 実対象（round05の初回選択、メタデータのみ）では、37件中7件を除外した。
- **粗い探索の優先**（V2の `coarse_gain_fraction`）：1%未満の削減を後回しにして、別のscheduleの評価を先にする。人工系で確認した。
- **並列枠**（V2の `lookahead_mode`：INDEPENDENT／HYBRID／CHAIN）
  - 保存結果の再生（時刻モデル、32枠、同時に調べるschedule 7つ）の結果：

| 方式 | Q3起動 | 実時間 |
|---|---:|---:|
| 独立候補のみ | 355 | 704分 |
| 連鎖 | 1,898 | 383分 |
| HYBRID | 1,799 | 395分 |

  - **失敗例**：調べるscheduleが少ないと、独立候補だけでは枠が埋まらない。
  - 次roundではHYBRIDを使う。多数のscheduleを同時に評価する実runでの効果は未検証。
- **全完了Q3の照合と再利用**（`build_rpe_q3_ledger.py`）：4,783件を照合し、拒否は0件。2種類の最安を分けて出す。
- 根拠：[kappa_wide_round05/REPORT](research/rpe_method_optimizer_h4_20261116/kappa_wide_round05/REPORT.md) §1・§3。


## PR複数幅の方式別計画・段階準備（2026-11-17）

公開統合5829a7288の通常WidthDriverで、各方式の既存W計画を分離し、元per_decisionの準備waveでfamilyを巡回する。未選択は次cycleへ保持し、初期222候補を一括準備しない。finish後の未採用完了jobも元Executor/価格/exactτ検査で確認済み最安へ含め、規則採用最安とは分ける。実装SOURCEと36人工回帰、実H4稼働範囲は[REPORT](research/h4_pr_multiwidth_optimization_20261117/REPORT.md)／[HANDOFF](research/h4_pr_multiwidth_optimization_20261117/HANDOFF.md)。共通queue・prefix-cache・元数値kernelを維持。実H4速度倍率は未測定。
## 分散Q3の自動枠と優先group（2026-11-17）

[実装・検収・適用条件](research/rpe_q3_auto_allocation/HANDOFF.md)。
固定code e9e63be165ddc8b4d991bca567e0fe20f5007372、rpe_optimizer_concurrent/rpe_q3_resourcesと通常factory。
新distributed runは省略時auto、READY×照合済み空き枠の対応で全体枠を決め、優先3台の余剰をWSLへ渡す。
各hostは資源境界・所有RSS/CPU・有効結果rate/判断待ちを使い1枠ずつ試す。減枠はdrain、旧run上限は保存。
計時81884bd39のhelper/CLIを再利用。実人工process・1OS4模擬hostでpeak5/18結果exact一致、実H4性能は未測定。
旧adapterのPASSを新timed adapterへ転用せず、配置後の実入力dry-checkが必要。
