# 性能対策の再利用索引

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
