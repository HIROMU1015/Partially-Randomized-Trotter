# H4・H6・H8 共通RPE実装レビュー対応記録（2026-11-28）

## 対象・境界

Claudeによる独立技術レビュー review(A)（43所見）への対応。公開共通ブランチ `codex/rpe-common-hchain-fleet-20261126` の確認済みSHA `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` を出発点として、隔離worktree `codex/rpe-common-hchain-review-fixes-20261128` のみを編集した。稼働固定版への上書き、既存RUN/PBS予約/科学SOURCE・計算条件・raw結果の変更、新規探索の葉計算、実PBS接続は行っていない。利用者が個別に許可したR3の数値startupのみを例外として実施した（詳細は下記）。

作業開始時の運用確認では、H4は`local-h4-price-recovery-wsl-20261128`（STOP markerあり、対応するEXITの照合が未完了）、H6は`26b430d71f`系の固定版（停止状態）、H8は`local-operations-512978edc-romutail-only-20261127`（停止状態）を観測した。これらは作業開始時点の観測で、現在のrun状態の保証ではない。公開共通ブランチは作業終盤にもGitHubのcompareで出発SHAと同一であることを確認した。今回の修正はこれらの分子に共通の中央・worker・keeper経路に置き、個別科学provider・既存所有契約を維持した。

「対応」は記載した技術的な問題への修正を意味し、実機での科学性能改善や独立再レビューを意味しない。「一部」は問題の一部に対応した実装であり、レビュー所見全体の検収完了ではない。実装と人工テストはCodexが行った（独立レビューではない）。

## 所見別の対応

最新集計：43所見すべてに技術修正を実装・人工検証。前段38件に加えR10（RomuTail実容量方式選択）、R43（実固定DP kernel）、R36（実queueおよび隔離通常controllerの実機GIL/CPU比較）を検収（41件の記載範囲）。R8・R18はPBS接続不可で実PBS検収保留、利用者も接続不可を了承。R36の稼働科学controllerへの直接GIL attachは権限不足で未実施。全43件の本番実機検収・運用採択とは区別する。詳細は末尾に記載。

| ID | 状態 | 実装済み・残る課題 |
| --- | --- | --- |
| R1 | 対応 | owner/queue差分索引、host別供給計数、READY未受領索引を維持。通知revision付きのadvisory snapshotで正確な空枠・host適合matchingを共有lock外へ移し、通知変化時に破棄。待機述語のlost wakeupを防止し、未起動再配送のTaskIndex遷移も修正。claimの正確な所有・適格性照合を維持。人工競合検証済み。 |
| R2 | 対応 | READY・永続済みinstallの差分化（未永続snapshotは通常ownerで採用）、固定Git locator容量内cache、関連recovery revisionのみ。通常分散幅探索で固定base contextの再構成を幅plannerと並行実行。次cycleも完全一致のbase/record/route/実view HEADだけ既存installを再利用、変更・alias・hold・未永続stateは通常installerへ戻す。人工検証済み。 |
| R3 | 対応 | HASWELL固定H8全体PLANを採択し、1回の全件数値startup PASS（leaf=0）。全2,830 cases、3,450 designs、全schedule一致。新規request/profile/実OpenBLAS dispatch照合、archive metadata確認、startup mismatchの明示分類と同一plan/runtimeのholdを追加。RomuTail採択PLANの技術対応完了。他hostの追加数値startup及び元PLAN・旧62件の変更・再試行は未実施。 |
| R4 | 対応 | 同一host/runtime/規模以上の完了観測でcold全node予約を縮小。明示CPU/RAMまたは同一class形状・同等以上のshot bit長の正のRSS観測から暫定予約し、計測済みclassとの混在packing。真に未計測・比較不能は隔離を維持。実機RSS/稼働率は未計測。 |
| R5 | 対応 | 検証済み`COMPLETED_PENDING_TRANSFER`をCPU/RAM占有から除外。結果の所有・転送証拠は削除しない。UNKNOWNは証明なしに解放しない。 |
| R6 | 対応 | 通常管理型keeperのqsubを既存の一度限りsubmit処理の分離processに移動。nonzero receiptは期限後に利用者とJob_Name限定の履歴照合で未開始generationを終了、応答欠損は同じ限定照合で正の一致だけ回収。不在・不明証拠のみでは再投入しない。人工scheduler検証済み。 |
| R7 | 対応 | 同一attemptと旧SESSION_START/ALLOCATIONの回収後SESSION_END、または正のPBS F/X記録と物理token・job ID・supervisorの一致で資源解放を許可。UNKNOWN所有を保持、自動retry禁止。人工履歴fixture検証済み。 |
| R8 | 技術修正・実機未検収 | RomuTail単独通常入口、固定planの明示contact lease、遠隔未着手区分のOWNER/EXPORT lock・予算/回収照合と永続fence、厳密な元予算のローカル子scope、受理済み子の所有/重複確認とpartition報告への統合。人工で再接続後の再開拒否・二重owner拒否を検証。実PBS・標準入口の実移管は未実施。 |
| R9 | 対応 | 明示的WAITING_FOR_INPUTS/SCOPE_HELD_INPUTSを論理queueイベントrevisionと関連入力の変更で判定し、BLOCK_STATUSへ永続化。worker再起動後もタイマーだけでは再起動せず、入力更新で既存再開経路を使用。人工再起動で確認。 |
| R10 | ローカル方式検収 | 背景sample/cleanup・期限内cache・物理free再照合、終端reclaim差分ledger。RomuTail実46,865 inode、7回すべてduとallocated bytes一致。台帳0.332秒、du0.080秒（中央値）ゆえ速度面から台帳を既定OFF、既存du維持。NFSは既存duのまま（PBS実測未実施）。 |
| R11 | 対応 | 終端資源/prefix cache・永続観測ACK・group差分forecast・観測JSONL（R20）、artifact inventoryは安定shardのみ再利用しrevision ACKで変更時だけ送信。PBS manager再起動・旧版では全一覧を再送。人工wire検証済み。 |
| R12 | 対応 | 回収request限定、WATCH索引照会、archive bundle SHA抽出と終了・adopt済み候補のみ展開。終了runのDB/WAL署名でsnapshotを再利用、実ファイル・他runの参照は引き続き照合。一致/更新時の人工回帰済み。 |
| R13 | 対応 | 物理枠・既存予算不変。満杯時にも各方法のdemandを1件進行し、共有standbyは利用可能な他方法の待機件数に応じて公平化（未照合probe待ちと終了methodは除外）。人工2方式検証。 |
| R14 | 対応 | distributed preparationを用いる通常overlap経路では既存PF/DF producer futuresをwatcherと並行実行しselectorを価格pendingだけでは停止しない。人工同一selector invocationおよびoverlapで検証。旧serial経路と永続selector stateの全面再利用は変更せず、実機効果は未計測。 |
| R15 | 対応 | 非screenの後着READY streamingに加え新規UNLIMITEDの明示opt-in screen waveで除外集合を固定し後着供給、wave終了後screen更新。人工roundで実際の後着候補受領を確認。既定・有限予算維持。 |
| R16 | 対応 | controller_parallelのPR候補laneで既存bridgeのwait=Falseを使い、同一request/attemptの受領待ちはwatcherに移す。受領済みの結果だけを再入し、失敗は既存のNO_RETRY分類。人工overlap・candidate isolation・live refillで検証。実runのlane利用率は未計測。 |
| R17 | 対応 | 候補選定時にCPU/RAM/storage/standbyを仮予約、snapshot一致を再確認して一括claim transaction。LAUNCHと進捗記録を共有queue lock外、worker応答前に耐久化。中断後は同attemptのLAUNCHを再構築。人工回帰済み。 |
| R18 | 技術修正・実機未検収 | 正のworker guardと元owner再確認、新規補充なしの厳格mode（既定）、実時間上限を保証しない見積り補充mode（明示）、元/追加ownerの終了観測と観測延長を記録。補充ありで延長ゼロは保証しない。 |
| R19 | 対応 | 全compatible予測とは別にlive CPU/RAMに収まるready CPUを集計し、現在の空きに不足すると補充する。 |
| R20 | 対応 | owner一致済み観測を差分JSONLで耐久保存、既存snapshotから再生・閾値でcompaction、失敗時ACKなし。forecastを変更workload groupだけ差分更新し不変group cacheを再利用。全再構築との同値性・中断人工検証済み。 |
| R21 | 対応 | advisory readinessはCAS存在のみ、claim前のSHA維持。自分が成功させたhardlinkの場合だけ既検証のinode/size/mtime・nlink増加・target同一性でctime更新をcarry。通常1秒安定則維持、未知変更は再hash、corruption拒否を人工確認。 |
| R22 | 対応 | export済みのキーを先に照合、EXPORT.lock内のmanifest inode/mtime/ctime/sizeに基づくprocess cacheを導入し、他allocationの原子的更新時は再読込。人工export回帰済み。 |
| R23 | 対応 | 新規H-chain partitionのみ固定PLAN stage別unique leaf（共有anchor/holdout含む）比例の有限準備予算を整数配分。reviewの葉数又はbatch数比例という条件を満たし、総額と0-leafを保持。既存partition不変、人工配分検証済み。 |
| R24 | 対応 | H-chain独立blockでselectorが停止中かつ未完了所有・queued jobがないことを論理revision付きで通知し、offline側で同revision・所有を再確認して通常handoff。人工rotation fixtureを検証。reviewの代替案である回転を採用し、同時package資源按分は実施しない。 |
| R25 | 対応 | 選択caseに必要な信号・価格・caseファイル・旧job rawのみ検証。全REQUESTの小さいSHA付きmapping、全manifest/charge照合を維持。legacy SHA stream化、安定statのprocess内・run-local disposable cacheで再起動時にも再利用。変更file再hashを人工確認。 |
| R26 | 対応 | PR候補別HELD、H-chain case・保存modelの局所holdに加え、ready_batch/price_modelsの局所I/O失敗を依存caseへ分離。有限methodのall-ready柵と全体binding異常による停止を維持。人工独立候補継続/有限gateを確認。 |
| R27 | 対応 | schedule/H-chainの明示exact replayを非blocking demandで処理。後着親OUTCOMEは未所有のqueued replayのみcancel、親DBのACCEPTED・request SHA・LAUNCH owner・既存ingest照合一致時のみhold解除。他の不一致はhold保持。人工検証済み。 |
| R28 | 対応 | KAPPA_STATEは最新history+committed head、完全履歴KAPPA_HISTORY.jsonlをfsync付で保持。Completion中間状態は候補単位STATE_DELTAS.jsonlに追記し、終端/globalは完全STATEにcheckpointしてcompact。中断末尾・競合読み出し・固定metadata修復の完全checkpoint要件を確認。 |
| R29 | 対応 | 管理型keeper終了＋元session/worker lock/未起動ASSIGNEDの確認、加えて旧direct PBSの正のF/X・JOB_ID・token・spool lockが一致した未起動だけfence。同じattempt/budget再配送をUNKNOWN後も証拠がそろえば可能に。人工両経路検証済み。証拠欠損はHOLD。 |
| R30 | 対応 | CONTROL/STATE同一書込抑制と時計のみ1秒debounce、GET replay、明示opt-inで正のresponse consumption receiptによる普通RPC応答圧縮。allocation差分journalをfsync、bounded checkpoint・Resume再生・中断末尾を人工検証。終端所有履歴は保存。 |
| R31 | 対応 | 4MiB bounded bulk、直接CAS file streamとSHA一致、明示bulk_parallel_lanesの並列chunk転送をspool/harvest/evidence/offlineに実装。旧1MiB strideを実応答で判定し、順序・size・file identityを維持。人工並列・identity不一致検証済み。実PBS接続なし。 |
| R32 | 対応 | mailbox並列、固定SHA/adapter/runtime metadata reuse。readiness batch内のtracked view全件stat・HEAD・index/config等signature一致時だけGit diff結果を再利用し、編集/削除/stage/不安定inventoryで再計測。旧固定SOURCEと実行直前の完全dry-check/AST guardは不変。人工fixtureでcache hitと失効を検証。 |
| R33 | 対応 | 計測済みクラスについて既存ownerの個別footprintを引いたCPU/RAMに優先順で複数classをpack。中央とworkerでclass別quotaを交換し、中央batch claimで種別上限を守る。互換性が確認できない旧中央へのofferは従来単一class。人工packing/割当検証済み。 |
| R34 | 対応 | 共通H4/H-chain policy関数、適用・非適用のROUND記録、signal phase-cache provider policy。新規固定PLANの明示provider_contractからsystem/rank・準備runtime package版を取得し、packet/routeへexact bindingを引き継ぐ。宣言なしのH6/H8既定・科学条件を維持。新規分子の科学計算は未実施。 |
| R35 | 対応 | UNLIMITEDのreserveで不要な件数計算を省略、有限予算の同一transaction照合保持。 |
| R36 | RomuTail検証controller検収 | 実queue11,282件・5 hostで結果一致、0.275秒→0.257秒。通常controllerを隔離3模擬workerで実行しGIL/CPUを実測、2組とも人工18件の全結果・最良点一致、blocking計測のerror 0。稼働科学controllerへのattachは権限で拒否、実科学負荷のGIL率は未測定。 |
| R37 | 対応 | 明示的な新規UNLIMITED scopeのfeasibility_probe_prefetchを実装。既定OFF、有限予算、step limitは従来probe順。人工selectorで確認。稼働中のscopeに適用せず、実機科学性能は未検証。 |
| R38 | 対応 | 明示managed_afterany_successorで通常centralおよび独立packageの正のready Q3/準備需要が残walltimeを超えた場合に同一資源afterany keeperを自動stage。受領した同一host/job/part・期限内見積りのみ採用し、既存one-shot/retire・STOP・release・重複防止を再利用。人工検証、実PBS予約なし。既定OFF。 |
| R39 | 対応 | keeper一時FS失敗のbackoff、正常受信したapplication error時の接続維持、明示module選択・import元表示。storage samplerの自身のduだけを停止時kill/reapしthreadをjoin。科学childは終了まで待つ。人工子process消滅確認済み。 |
| R40 | 対応 | 通常publicationの既存hardlink走査内でcompletion SHAと全関係dir署名を取得し初回Resumeから再利用。安定dirだけ省略、欠損/入替/別FSコピー/即時変更は従来修復。人工欠損修復・再走査省略を確認。 |
| R41 | 対応 | 人工childを実SIGKILLしsupervisor正のwait receiptからH-chain通常保守Resume入口まで検証。同一PBS/queue owner保持、一度限り・STOP維持。科学worker起動は人工stub。 |
| R42 | 対応 | `owned` snapshotをhost preferenceへ渡す。安定したstat signatureの準備profileを再利用し、変更・新規更新時は読み直す。 |
| R43 | 実kernel検収 | H4保存科学入力、実固定DP kernel、fresh/persistentの全出力JSON・prefix cache内容完全一致。同一2 PIDで4 batch、3回reuse。wall 2.848→2.773秒、RSS増。既定OFF。通常Q3入口は不足SOURCEでpreflight停止、実kernel限定の検収。 |

## 検証方法

実際の科学providerやPBS予約を起動せず、人工queue・fake provider・loopback RPC・PBS scheduler fixtureで検証。テスト範囲は `tests/test_rpe_q3_store.py`、`test_rpe_distributed_controller.py`、`test_rpe_central_maintenance.py`、`test_rpe_pbs_offline.py`、`test_rpe_pbs_worker.py`、`test_rpe_q3_worker.py`、`test_rpe_q3_memory_learning.py`、`test_pr_execution_route_binding.py`、`test_rpe_q3_resources.py`、`test_rpe_hchain_normal_entry.py`、`test_rpe_hchain_preparation_route.py`、`test_rpe_pbs_storage_continuity.py`、`test_rpe_preparation_transport.py`、`test_rpe_scaled_shots.py`、`test_rpe_auto_allocation.py`、`test_rpe_pbs.py`、`test_rpe_pbs_pilot.py`、`test_rpe_preparation_queue.py`、`test_rpe_preparation_round.py`、`test_rpe_search_continuation.py`。

前回の統合回帰：431 passed, 40 subtests passed（99.28秒）。今回追加した実装を含む統合回帰：**445 passed, 40 subtests passed（115.92秒）**（`/tmp/rpe-review-continuation-final.log`）。この統合実行の開始後に追加した終端観測ACK・mailbox・H-chain所有照合の最終差分には、人工重点検証 **10 passed**（`/tmp/rpe-review-continuation-focused-final.log`）を実施した。最終差分を含む関連回帰は **119 passed, 8 subtests passed（44.78秒）**（`/tmp/rpe-review-continuation-latest.log`）。H-chain独立package回転追加後の人工回帰は **34 passed**（`/tmp/rpe-review-continuation-rotation.log`）、offline全体 **35 passed**（`/tmp/rpe-review-continuation-offline-final.log`）。受理済みpublicationの差分照合・修復は **3 passed**（`/tmp/rpe-review-continuation-publication.log`）。

追加で実行した `test_rpe_local0_search.py` 全体のうち10件は、変更中worktreeを固定SOURCEとして扱うfixtureの既存`fixed_implementation`で拒否された（変更されたsourceを検出する正常な柵）。また `test_rpe_preparation_workload.py` の1件では、base SHAにも存在する容量上限一致時の条件とテスト期待の不一致を確認した。実空き容量がない場合に未知の必要容量を持つ仕事を開始するよう柵を緩めず、この不一致を残した。人工fixtureの保護・本番のSOURCE照合は変更していない。

前回からのbulk人工転送検証は、各1MiB超の2つの独立した所有結果を旧版2往復ずつから新版1往復ずつで受け取り、内容一致と所有を確認した。

合成SQL比較（30,000行、ACCEPTED済29,980行、RUNNING20行、1KiBの人工request、同じ列・行集合・順序をassert、メモリ上SQLiteの既存相当索引、40回の中央値）：従来 `state NOT IN (...)` は全表走査で3.3625 ms、変更後 `state IN (...)` は `jobs_state_enqueue` を使用し0.0253 ms。SQLのみの比較であり、実機controller全体や科学計算の高速化倍率ではない。

## 以降

当時の43件の内訳：対応19件、一部（方針・要計測を含む）22件、未対応・方針2件。このcheckoutは全所見の解消を宣言するものではない。

優先して残す課題は、実H8 hash差（R3）、現行PBS所有移転（R8）、受付/容量の追加改善（R10）、selectorとscreen・standbyの残り（R13–R16）、独立package rotation（R24）、差分journal・RPC耐久compaction（R28/R30）、Resume publication（R40）、実機CPU/GIL・RSS/DP計測（R36/R43）。現在の停止・固定SOURCE・予算を変更せずに実施可能な技術項目はCodex担当。既存PBS所有移転、drainでの新旧session並行、科学的probe配分、実H8 startupの新規数値照合等は、それぞれの固定契約・運用方針・必要な証拠に従う。公開共通ブランチへの反映および稼働版切替は未実施。

## 13時台の再開（2026-11-28 JST）

公開共通branchはGitHub compareで `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と同一。再開したのは前回の隔離worktreeのみ。現在確認できた登録runはH4 schedule `search-538c35a68be6425b1b4c`（固定版 `ca27a9ff3a59`、matching EXITなし）、H8 method `search-a7627621581769835f1e`（固定版 `876ece18f366`、STOP/matching EXITあり）およびH8 PBS `search-0ee3f4414b113cb5da4f`（固定版 `7f065218948e`、STOP/matching EXITあり）。H6は固定kit `romutail-h6-pilot-20261125` を確認したが同じregistryから現行runの一意照合はできなかった。どのrunも変更していない。

人工回帰：広い回帰は218成功・11 subtests成功・2失敗（inventoryの同一timestampとrelayのclaim順競合）。両者を修正し、追加回帰は96成功（`/tmp/rpe-review-resume-final-checked.log`）、selector関連54成功・24 subtests成功（`/tmp/rpe-review-resume-selector.log`）、待機枠8成功、legacy/partition6成功、直接/keeper fence4成功、publication3成功、readiness3成功、overlay2成功、差分forecast16成功。最後の追加変更は対象テストで確認した。広い回帰全体の最終再実行は未実施。既存storage境界テストの矛盾は、未知の必要容量と明示0 bytesを区別して安全側の期待へ修正した。旧固定SOURCEをdirty checkoutと照合する10件の拒否は引き続き保護を維持する。

人工履歴ベンチマーク（3000件の512-byte人工metadata、20回checkpoint、同じ3,020履歴を復元一致）：従来1,711,252 bytesのsnapshot全体更新の中央値33.48 ms、差分履歴保存のsnapshot744 bytes、中央値2.04 ms、process CPU中央値0.431 ms。完全journalは2,154,001 bytes（中断分岐も保持）。cProfileではJSON encoderが主要負荷。これは人工I/O比較であり科学計算や中央全体の性能倍率を示さない。

新しい `KAPPA_STATE.json` は従来の最新history要素を保持する。全履歴は同じrunの `KAPPA_HISTORY.jsonl` と `history_archive.head/count` により再生可能であり、公開用metadataの最新決定とは区別する。既存固定版・過去checkpointは変更しない。`--check-startup` は名前に反して数値startupを行うため実機で通常dry-checkから実行しない。R3の個別数値照合の許可は利用者へ確認中。

最終追加検証：固定locator/履歴4件、資源sample/relay3件、overlay/容量2件、固定readiness hash 3件、artifact wire 2件、bulk/容量/PBS 6件、H-chain独立候補2件。環境の空き容量の変動で人工容量テストが2件失敗したため、測定値fixtureと同様にfilesystem freeも明示的に人工値に固定（実際の容量柵は変更せず）し再検証で通過した。

最終worker/PBS/H-chain人工統合回帰：**196 passed, 7 subtests passed（19.99秒）**（`/tmp/rpe-review-resume-final-integration.log`）。広い前回445件のスイートとは範囲が異なる。さらに固定locator、差分forecast、artifact wire、bulk streamおよび容量fixtureの各最終差分を対象回帰で確認した。

## R3 固定H8 numerical startup（利用者が許可した1回）

元の失敗attempt `prep-2f13f96414140ac36f6965c2/f6d56bfcd53146e6a6a61326d1b253a6` の固定PLAN（SHA256 `b5554fba7683f5a62d6aaa26b8e346a99f6436f9d92a7241cf968cb2c6863c25`）、固定provider SOURCE `9aa6d186b45201fda02178edd62e2a5d89cc8d7c`、固定入力4点、SOURCE13点を照合し全一致。保存handoffのruntime Python 262点もそのSHA256SUMSと一致した。元のPython環境（NumPy 1.26.4、Qiskit 1.3.0）・1 CPU・BLAS各1thread・OPENBLAS_CORETYPE未指定で、`--check-startup` のみ1回実行。終了コード1。入力検証成功、29 prefixのbinding、保存された3 prefixの成分等の照合成功後、再構築 scoped hash が3件とも不一致となった。ログは [rpe_h8_startup_identity_check_20261128.log](rpe_h8_startup_identity_check_20261128.log)。

| prefix | expected scoped hash | observed scoped hash |
| --- | --- | --- |
| 7 | 554c9437ccad9a9927f7c8fdb82e220b4c9623775ec56223176209ec48be65e5 | cd61e7287cd03281ec169776f4e20e85b7bb6d0964a296b920f94426e345a6be |
| 14 | 8d0a007ce774777ad8c4bf8a33140ff1ec337bd180aae11e4ab921a8fb503cf0 | 3dd1215fd97460e25df629bfda0758e38c6faacb44ee2ce73c9eb5d81477eef7 |
| 21 | 8094461085982f558984852e0100000d6623a4edebbbe5845d5ade83b456ab4e | e8900f5ef5713a65cd32af9a0c0c62157530521f4346e4fe83797924a8548389 |

3つの保存preparation hash自体は一致。静的調査では再構築hashの構成要素であるpartition hashがBLAS実装依存のFrobenius normに基づくweightを含むことを確認したが、原因の確定には至らない。許可の範囲では信号・価格・Q3・leaf作成なし。既存62件のretry・保存PLANの変更・稼働版切替なし。R3は引き続き一部対応である。

追加静的調査：保存handoffの `PREPARATION_CHECK.json` には、過去のLD0 PBS/server2照合で28 fragment中7個の恒等項係数の丸め末尾がBLAS内積により相違したことが明記されている。`PBS_LD0_IDENTITY_TERMS.json` と `restore_ld0_metadata.py` は保存SHA256SUMSに含まれる。既存の復元関数はLD0の全28 fragmentを1回ずつ読む契約であり、各prefixの部分集合を繰り返す今回のstartupにそのまま使えない。固定保存係数を利用する新しい照合経路を検討できるが、今回のscoped hash差の原因を確定したわけではない。HASWELL/NEHALEM各1回の追加startup許可は確認中であり、未実施。

## 追加修正と回帰（13時台後半）

- R18：新規scopeのみ明示opt-inの soft launcher rollover backfill。元の実行ownerと見積り期限を固定し、既存の残walltime/資源判定を使う。STOP、controller close、元owner終了、新規予約への切替の際は補充しない。元試行が推定前に完了する場合の厳密なdrain時間非延長は保証できない。実PBS検証なし。
- R25：selected blockの検証対象をraw signal/priceからcase記録・旧jobのrawにも拡張。小さいREQUESTは全件について固定SHAでmappingを確認し、他blockの巨大rawを読まずに済む。完全legacy manifestのSHA・全体のcharge inventoryは引き続き確認する。run-localのdisposable cache `LEGACY_HASH_CACHE.json` はmanifest SHA・root・inode/dev/size/mtime/ctime・期待SHAに束縛し、安定した既検証fileだけ再利用する。変更時に再照合。同じrunの人工再起動でcache hit・変更時拒否を確認。
- R28：中間状態の候補record deltaをfsyncして保存。終端/全体保存時に完全snapshotをdurableに置換してからcompact。readiness・進捗・前所有者確認はdeltaを再生し、固定metadata修復に使うSOURCE-bound checkpointはdelta反映前の古い状態を採用しない。人工中断末尾とcheckpoint再生を確認。
- R32：固定Q3 SOURCEのSHAに既存の安定stat cacheを再利用。Git locatorは完全commit SHAに限り、Git blobの存在とobject IDを毎回再確認、実際のblobのSHA-256をcache（短いrefは従来通り）。source hash/固定input hash不一致の拒否は維持。
- R38：管理型keeperの同一resource/afterany後継をexact旧予約に対して1回限りstageするgateway capability。通常central demandが残walltimeを超え、既存scopeのopt-inがある場合のみ呼ぶ。明示release時の未発行target fence・発行中の照合HOLD、元予約の終端後かつ既存ownershipを満たすまでworkerをattachしない。独立package側では中央queueが子の待機task見積りを持たないため自動trigger未接続。実PBS接続・予約投入なし。

検証：Completion/continuation/preparation round **43 passed, 6 subtests passed**、PBS/pilot **83 passed, 3 subtests passed**（追加後継gatewayを含む）、追加のsoft drain・後継fixture **4 passed**。人工広域回帰 **396 passed, 22 subtests passed（57.16秒）**（`/tmp/rpe-review-startup-followup-regression.log`）、最終近傍のworker/PBS/H-chain回帰 **197 passed, 7 subtests passed（21.41秒）**（`/tmp/rpe-review-startup-followup-final.log`）。追加差分では9重点テスト、legacy cache後は6 legacy重点テストに成功。別の既存host eligibilityテストの1件は固定歴史commit `33e324b43690caeaa353ecdb9e3d42f3139198b6` の必要blobをlocalで取得できずSSH DNS接続失敗でUNCHECKEDになった。既存固定SOURCEは変更・緩和していない。人工結果は実PBS性能・科学的結果の保証ではない。

現時点の43所見：**対応21件、一部対応22件、未着手0件**。全43件に実装または調査を進めたが、全件の問題解消・検収完了ではない。R3の固定数値照合は1回実行して不一致を再現した。旧実行契約、固定入力、raw科学結果、実PBS予約、既存停止状態、公開共通branchは変更していない。この計算は利用者の明示許可内のstartupのみであり、信号/価格/Q3は未実行。共通修正は隔離worktreeに未commitで保持。

最終統合回帰（legacy再起動cache、Git blob cache、Completion同一delta抑制を含む）：**399 passed, 22 subtests passed（54.88秒）**（`/tmp/rpe-review-startup-followup-full-final.log`）。`git diff --check` も成功。

## R3 追加固定startup照合（HASWELL・NEHALEM各1回、利用者許可済み）

公開共通ブランチの最新SHAをGitHubで照合し、`512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と一致。元attemptのPLAN、SOURCE 13点、asset 4点、NumPy 1.26.4・Qiskit 1.3.0を再確認。元の1 CPU/1 BLAS threadを維持。OpenBLAS verbose出力で両kernelが選択されたことを確認した。信号・価格・Q3・leafの実行無し。両方とも入力と保存成分照合を通過し、`verified_saved_preparations` で終了コード1。

| prefix | 既存PLANのscoped | HASWELL scoped | NEHALEM scoped |
| --- | --- | --- | --- |
| 7 | 554c9437ccad9a9927f7c8fdb82e220b4c9623775ec56223176209ec48be65e5 | f794c14366f6081237cafd80a91b6659c98450f3eb720c90073ab5ed2263abc4 | d975b5da2f55004a6e8df2a5d956b4afa8727586e7b742878e2772162a8c197a |
| 14 | 8d0a007ce774777ad8c4bf8a33140ff1ec337bd180aae11e4ab921a8fb503cf0 | 45859f28ee7c8c6c8a60c3db796a11cda51132652e337d7d739400b4309fd0a3 | 5f8671ab8f3c7b828619f8d83097ac45ad44d74785b26b8b3fc9ffb6417b02e2 |
| 21 | 8094461085982f558984852e0100000d6623a4edebbbe5845d5ade83b456ab4e | e439150b451ba542d8ed170e0ca05fa751879fc33012fb15ee8f5ac413b0f051 | e439150b451ba542d8ed170e0ca05fa751879fc33012fb15ee8f5ac413b0f051 |

HASWELLの再構築scoped hashは3件とも**保存preparation hashそのもの**に等しい。NEHALEMはprefix21だけ保存hashと一致。固定PLANのscoped hashはどちらのkernelでも全件不一致。保存科学成分の再現性とPLANの完全identity一致は区別する。固定PLANの改変や照合緩和、profileの無断切替、旧62件の再試行は行わない。追加startup許可は消費済み。ログ：[HASWELL](rpe_h8_startup_identity_haswell_20261128.log)、[NEHALEM](rpe_h8_startup_identity_nehalem_20261128.log)。

## 追加技術修正・検証（14時台）

R1: live QUEUED/RUNNING task索引を追加、終端履歴全体の走査をdispatch/standby等から除外。人工30,000件（29,936終端、64待機、300回中央値、同一順序）で全走査0.485ms、索引0.000368ms。局所比較のみ。
R15: 明示opt-inかつUNLIMITEDのscreen付きwaveで除外集合固定の後着READY供給を追加。有限予算・従来scope不変。
R30: 正の応答消費receipt（request/response SHA一致）をgateway経由で照合し、普通RPC応答をtombstoneへ圧縮。ACK後receipt整理。既定OFF、STATE差分保存は残る。
R41: 人工childの実SIGKILL→supervisorのwait receipt→H-chain通常保守Resume入口の同一owner・一度限り検証成功（科学開始は人工stub）。
R43: overlayのmodule/関数attribute実参照の照合追加。
検証: 消費receipt・応答圧縮3件、task索引1件、READY streaming 2件、H-chain SIGKILL 6件、overlay 2件成功。広域219成功・7 subtests・3失敗のうち、新しい固定blob hash APIに合う人工mockへ修正して2件成功。残る旧AST比較1件は公開共通HEADの既存`rpe_method_optimizer.py`自体が古い比較SHAと異なる（同fileのworktree diffなし）。科学選択処理・比較基準は維持。最終人工広域再回帰は空き容量急減で自身のpytestのみ中断、34成功・1除外。ディスク空き0、通常sandboxも起動不可。既存RUN・他作業のデータ削除なし。現時点の内訳は対応22件、一部21件、未着手0件。全43検収未達。上の表と前回集計は過去記録（R41のみ対応へ更新）。R3の追加新規PLAN候補の許可は確認中。

## HASWELL新規診断PLAN候補の照合（14時台、利用者許可済み1回）

空き容量21GBまで回復し、公開共通branch最新SHAは `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と一致。元attemptのSOURCE 13、assets 4、PLANをSHA照合した。元の固定packet SHA `b5554fba7683f5a62d6aaa26b8e346a99f6436f9d92a7241cf968cb2c6863c25` を変更せず、先のHASWELL観測で保存hashと一致した3件のscoped hashだけを差し替えた**分離診断候補** [PLAN](rpe_h8_haswell_startup_candidate_PLAN_20261128.json)（SHA `89d2a4774aad4e908ec0f303a3ef89ba6955efb1f25dbc947527ea5f266c233a`）を作成。元のscience scope・owner・全物理条件・target・compiler・1 compile anchor target `[["794a23e90829ea2fab12",0]]` は不変。候補は診断用batch packetであり、全run用新規採択PLANや運用許可ではない。

NumPy 1.26.4、Qiskit 1.3.0、OPENBLAS_CORETYPE=HASWELL、各BLAS/OMP thread=1、元CPU予約=1にて、`--check-startup` のみ1回実施。**終了コード0、HCHAIN_STARTUP_IDENTITY_PASS、leaf_calls=0。** 29 prefixのbinding、3保存preparation hash、選択anchor schedule、target/compilerの厳密照合に成功。 [ログ](rpe_h8_haswell_startup_candidate_20261128.log)。signal、price、Q3を実行していない。元PLAN・失敗62件・既存RUN/STOP/PBS予約を維持。元PLANの失敗は再現条件が違うためであり、診断PASSを元PLANに転記してretryを許可するものではない。

新規scopeの `hchain.openblas_coretype` をrequest runtime requirementに引き継ぎ、選択worker profileにも同じ値を要求。数値startupを行わないNumPy import probeのOpenBLAS verbose実dispatchを厳密比較する。既存 `preparation_environment` で実childにも同じkernelとthread設定を引き継ぎ、fingerprintに実kernelを記録する。設定なし旧固定requestは不変、実機新runの配置・起動なし。新規候補の診断技術は成功、元固定契約の運用再開および失敗62件の個別処理は未許可・未実施のためR3全体は一部対応とする。

## 今回の残項目への追加修正

R1 owner snapshot event revision再利用（他processからのclaim・同state detail更新・copyの人工検証）。R10監視root内の未指定prefix cache上限を既存reclaim targetから決定（明示上限・除外path不変）。R27後着親の正のACCEPTED receiptに基づく未所有queue cancel・既存ingest再利用、H-chainにも明示exact replayを非blocking接続。R30 heartbeatのみのsnapshot書込を既存1秒worker cadenceでdebounce（所有変更は即時）。R31複数gatewayの並列chunkを既存bounded protocolへ接続（明示設定、旧strideにも対応、identity違い拒否）。R32 stable interpreter metadataをreadiness batch内で再利用、AST guardは共有しない。R34 providerのphase-cache batch policyを実装。R43 DP fork poolの動的batchでexact rational合成同値、PIDごとの実時間集計。DPの固定過去viewはローカルに存在しないため旧固定DP test moduleはimport不能、変更した集約関数の独立合成fixtureのみ実行。

総合回帰の1件失敗はsynthetic harvest fixtureで未設定bulk_gatewayを参照した差分で修正、6項目の追加重点テスト等を通過。前段の統合回帰 **453 passed, 1 deselected, 22 subtests passed（73.35秒）**。除外1件は古い固定SOURCEの厳密AST比較で現在共通baseも元から異なるもの、科学SOURCEも比較基準も変更せず。現在の43件内訳は**技術対応24、一部19、未着手0**。固定旧RUNに触れていない。新規科学startup以外は人工検証。R27/R31を技術的対応完了、R3は新規候補PASSだが旧固定runの再照合・処理は別契約のため部分対応。

追加R21：正規immutable materializeで、自プロセスが作成するhardlinkによるctime変化だけを、事前検証済みidentity、同じdev/inode/size/mtime、nlink+1、target identity一致が揃う時にcacheへ引き継ぐ。通常の安定stat条件を維持し、未知のctimeやデータ変更は再hashする。人工fixtureで再hash省略と書換え拒否を確認。

上記診断packetに加え、保存handoff全体PLAN（SHA `8956f7d73244ee881ebd6256faca50167cd59f4a4b716f72f785ea13eaa874bb`、2,830 cases、3,450 calibration designs）と元packetの共通条件・3旧scoped hashの同一性を静的照合し、3つのscoped hashだけをHASWELL値に置換した全件候補 [PLAN.json](../../artifacts/rpe_h8_haswell_candidate_20261128/PLAN.json)（SHA `c476b62bf859751d4839de0fc36dfa8b5706ad9d76b4e7fca5f5a1aed496c86a`）を保存。候補の全schedule numerical startupは未実行（許可の1回は前記診断packetで消費）。既存PLAN・診断packetで照合した範囲と、全件候補の静的同一性の境界を保持。

全体人工回帰（R21最終差分前）：**456 passed, 1 deselected, 22 subtests passed（71.48秒）**、`/tmp/rpe-review-haswell-final-all.log`。R21追加重点3件PASS。現時点の内訳 **技術的対応25件、一部18件、未着手0件**。元固定H8はHOLDのまま、元62件の再試行・PBS接続や固定条件変更なし。全件検収未達。

最終重点回帰（R21変更後、最新のCAS/H-chain/owner/replay/PBS/DP合成対象）：**199 passed, 12 subtests passed（10.68秒）**（`/tmp/rpe-review-haswell-latest-targeted.log`）。最終 `git diff --check` 通過、空き容量約41GB。

## HASWELL H8 全体PLAN採択・startup照合（利用者許可済み）

空き容量約40 GB、公開共通SHA `512978edcbb2ff884a5d69fa1ca7ad00bceb7048`、固定SOURCE 13点、assets 4点、PLAN SHAを再確認。新規全体PLAN [PLAN.json](../../artifacts/rpe_h8_haswell_candidate_20261128/PLAN.json) SHA `c476b62bf859751d4839de0fc36dfa8b5706ad9d76b4e7fca5f5a1aed496c86a` を採択。元全体PLANおよび旧固定失敗packetは不変。固定NumPy 1.26.4・Qiskit 1.3.0・HASWELL・1 CPU/BLAS threadを維持した。新しい `--check-startup --check-full-startup` 入口を人工8件で検証後、許可されたstartupを1回だけ実行し終了コード0。

- HCHAIN_STARTUP_IDENTITY_PASS、29 prefixes、3 saved partial identities。
- 全2,830 cases、3,450 calibration designs（保存・復元の両件数一致）。signal 17,766、compile 7,484、audit 690 schedule entries一致。
- `leaf_calls=0`。signal・price・Q3の葉計算及び新規探索は実施していない。
- [Full startup log](../../artifacts/rpe_h8_haswell_candidate_20261128/FULL_STARTUP.log)、[採択済みPLAN設定](../../artifacts/rpe_h8_haswell_candidate_20261128/ADOPTED_HCHAIN_PLAN_SETTINGS.json)。既存 `fixed_plan` で設定のSHAと全件数を照合。設定は新規scopeに取り込む断片で既存稼働版の切替ではない。

旧固定PLANのHOLDと失敗62件は別契約のため維持。今回のstartup実行許可は消費済み。

追加技術対応：R15人工screen roundの後着供給、R16人工overlapと同一attempt受領の検証、R30 PBS allocation差分journalと既存readerの接続。R4同一runtime・同一class形状で同等以上のショットbit長の正のRSS観測から暫定予約を設定し、計測済みclassとの混在を可能にした。真に未計測・比較不能の場合の隔離は維持。R10容量sample競合時の期限内cache利用と物理free再照合。R1有効queue数を状態遷移で差分集計し、明らかな空枠時にはready全走査を省いた。R38独立packageの同一host・同一job・期限内の正のready Q3見積りを既存後継予約経路へ連携（人工のみ）。

## 最新統合回帰と残課題（今回）

人工広域回帰（H-chain screen streaming、PBS journal、後継需要、価格watcher、TaskIndex、loopback再起動を含む）：**565 passed, 1 deselected, 35 subtests passed（112.34秒）**、`/tmp/rpe-review-full-adopt-final.log`。最初の広域回帰は529成功・2失敗。容量比較fixtureを物理free固定に改め、人工再起動はログ保存再現でPASS、最終広域回帰で両方PASS。除外1件は前記固定歴史AST比較で共通base自体が不一致のため緩和していない。依存入力解決を既存実行側と独立packageの後継需要見積りで共通化した後、関連回帰は **322 passed, 22 subtests passed（50.07秒）** (`/tmp/rpe-review-full-adopt-last-targeted.log`)。

R4の比較可能なcold class混在、R14の通常分散価格watcher、R24の既存handoff回転、R37の新規無制限scope opt-in probe、R38の通常・独立package後継予約は記載した技術範囲の対応を完了した。共通PBS運用policyの小規模共通関数はH-chain・schedule入口へ接続し、人工試験で保持・buffer・backfill・standbyを確認した。

最終の共通policy差分後の関連回帰：**175 passed, 3 subtests passed（5.15秒）**（`/tmp/rpe-review-full-adopt-policy-final.log`）。`git diff --check` PASS。空き容量約39 GB。

現時点：**技術対応34件、一部対応9件、未着手0件**。残りはR1（共有lock外の曖昧host判定）、R2（全plan再構成の背景化）、R8（既存PBS区分の所有移転）、R10（終端容量の安全な差分台帳）、R18（実行中試行の終端までに確実に収まる保証）、R32（完全view/runtime再照合を安全に減らすこと）、R34（新分子に必要なprovider固有runtime/rank宣言の一般化）、R36（実機中央のGIL/thread別CPU計測）、R43（固定DP pool再利用・実機資源精度）である。対応済みは記載した人工検証済みの技術範囲を意味し、全43所見の検収完了や実環境の速度向上を証明するものではない。R3の既存H8の失敗62件は別固定契約であり、今回採択PLANはそれを過去に遡って変更しない。追加startup、信号、価格、Q3は実施しない。未commitの隔離worktreeで保持。

追加のR1：stop/hold時は待機索引を使用、host別供給数を所有・適格性の既存遷移で更新。人工30,000履歴（待機1,000、host4、300回中央値）で旧active queue走査0.250452ms、新host差分集計0.000466ms、件数一致。これは局所人工比較であり実run全体の速度倍率ではない。終了処理は旧式fixtureの直接状態設定との互換性を保って全状態を正確に再確認する。人工回帰の最初の再実行は人工fixtureの直接RUNNING→DONE変更により93成功後の終了待ちで止まったため、自身のpytestだけを中断し修正。人工fixtureの状態更新を既存transition helperに合わせ、終了処理の全状態確認を維持。中央単独回帰 **52 passed, 4 subtests passed（27.05秒）**、`/tmp/rpe-review-full-adopt-controller-final.log`。

最終統合回帰（共通policy、依存入力解決共有、host別queue差分計数、終了処理互換修正を含む）：**584 passed, 1 deselected, 35 subtests passed（116.20秒）**、`/tmp/rpe-review-full-adopt-final3.log`。`git diff --check` PASS。最終空き容量約37 GB。現時点の表は対応34件・一部対応9件・未着手0件。全43件の検収完了ではない。作業中の全テスト・数値startupは終了、既存RUN・PBS予約は変更なし。

## 残り9項目の継続修正（今回）

公開共通branch `codex/rpe-common-hchain-fleet-20261126` はGitHub compareで `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と完全一致。前回の隔離worktreeを継続使用した。R1通知revision付きlock外のadvisory matchingと正の未起動再配送の索引更新、R2固定base contextを幅planと並行再構成、R8明示contact lease、R10終端cleanupの容量見積り差分ledger、R18元ownerの起動直前再確認（buffer無効）、R32 tracked viewのbatch限定stat cache、R34固定PLAN provider契約、R36現行observer/pollのthread CPU計測、R43reap競合時CPU baseline保持と明示native thread制御を追加。従来の固定契約は変更しない。

局所人工計測：300ファイルの新規人工Git checkout、25回中央値、出力一致。view再照合は従来3.506ms、batch内stat cache 1.693ms。実環境全体の速度倍率ではない。先行人工回帰は既知の固定AST比較1件と再起動fixture1件が失敗。ログ保存した同一人工再起動検証はPASS、追加科学call 0。統合回帰 **593 passed, 1 deselected, 35 subtests passed（116.27秒）** (`/tmp/rpe-review-nine-integrated.log`)。その後のR18配送guardおよびR32 inventory競合対策は追加の重点fixtureで成功、最終回帰中。

現時点の技術内訳：**対応37、一部6、未着手0**。残る一部はR2、R8、R10、R18、R36、R43。固定H8 HASWELL全体PLAN・startup結果、旧HOLD/STOP/62失敗、稼働版、実PBS予約を不変に保持。今回の追加数値startup・科学葉・新規探索・実PBS予約は0件。

追加R2：通常ownerの既存`install`を再利用し、永続済みinstallの差分選択と、snapshotで先行供給した未永続recordの採用を区別。前cycleの完全一致base/全記録と共通route view HEADの正の照合時のみ1回分の初回installを再利用。alias/owner照合が必要なrecord、除外・変更record、未永続/不完全recordは通常installerへ戻す。人工fixtureで同一、変更、消失、view HEAD変化、snapshot未永続時を確認。R2技術対応完了。R18はworkerの正のrollover_guard heartbeatがない旧版では補充しない。

最終人工統合回帰 **610 passed, 1 deselected, 50 subtests passed（128.10秒）** (`/tmp/rpe-review-nine-final-cycle.log`)。中間回帰595成功、追加のREADY差分関連回帰254成功。固定AST比較への緩和無し。人工200件の中央queue observer実測（科学実行0件）：thread CPU 0.006131584秒、wall 0.006137742秒、report CPU 0.003080774秒（1 sample）。この人工計測はGIL占有率や実探索の速度改善率ではない。`git diff --check` PASS、採択済みH8全PLAN SHAは `c476b62bf859751d4839de0fc36dfa8b5706ad9d76b4e7fca5f5a1aed496c86a` のまま。

最新内訳：**技術対応38，一部対応5，未着手0**。R8は既存PBS切断中の正の未着手所有移転（元所有と予算の証拠がない限り不可）、R10はNFS/外部変更・hardlinkを含む正確な容量制約を保つ差分admission台帳、R18は見積りを保証された実行上限と混同せずdrain延長なしを保証する条件、R36は実対象runのGIL/CPU計測、R43は固定DPのfork継承closureを維持したpool再利用と許可済み実kernel同値性検証が残る。今回の修正は未commitの隔離worktreeのみ、既存RUN/STOP/PBS予約/稼働release/固定科学結果を変更していない。H4/H8の既存RUN registryも読み取り確認した（H4最新登録EXITあり、H8 STOPとEXITあり）。H6の現行固定版は前回参照値のみで、中央registryから当該最新状態を再確認できなかったため実運用状態を断定しない。

## 残り5項目の技術修正と実機検収条件（最終段階）

公開共通branchは作業開始時にSHA `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と一致することを確認した。既存の隔離worktreeのみ更新。稼働release（中央registrar上のcurrentは `romutail-pbs-independent-f37e1ebd0963`）を差し替えていない。R3のHASWELL全体PLANとstartup PASSは前節の許可済み結果を再利用し、startupを再実行しない。

| 項目 | 新規実装の主要条件 | 実機検収に残ること |
|---|---|---|
| R8 | 明示lease/fence、元予算のローカル子scope、受理済み子結果の登録 | PBS到達・protocol一致・通常入口の所有照合と実移管、再接続・二重実行なし・結果回収 |
| R10 | ローカルPOSIX限定、差分容量台帳と定期du照合、既定OFF | 代表的実ファイル群の計測精度と速度、NFSは既存du維持 |
| R18 | 厳格mode（新規補充なし）と見積りmodeを区別 | 既存rollover時のCPUとdrain延長、厳格modeの追加ownerゼロ |
| R36 | host別集計の単一SQLite snapshotとthread CPU計測 | 実controllerのGIL・thread CPU・poll遅延、同等負荷での比較 |
| R43 | 固定closureのpersistent DP pool、互換性がない場合はfallback | 固定kernelでの数値同値性、pool reuse、RSS/全CPU/wall時間比較 |

### 実機検収の手順と判定

1. R8は新規opt-in planの全対象workerとgatewayを同一互換版に揃え、lease+grace経過と既存のOWNER/EXPORT排他・checkpoint/claim/exportなしを確認する。中央制御の通常所有照合と重複防止を維持したまま、明示許可された元予算の範囲内でのみ子runを開始する。中央・PBSのjob_id、予算、終端受理を比較して二重計算がないこと、再接続してもPBSがfence区分を開始しないこと、既存reportに子結果が含まれることを合格条件とする。旧固定runの移管や完全通信断中の強制移管は対象外。prior-partition importや回収済みtransferのため正の証拠が欠ける場合は終端採用しない。
2. R10は実workerの同じroot、同じhardlink/除外/容量条件で従来計測と比較し、変化・cleanup・同時作業を含めallocated bytes一致と物理free安全上限を維持する。sample遅延・受付時間の改善が確認できたローカルFSのみopt-inする。既存の安全容量制限を緩めない。
3. R18は実行中ownerの所有を維持し、既存の通常drain中に記録された追加owner、元owner終了観測、全owner終了観測、CPU使用を比較する。厳格modeは追加ownerなし。見積りmodeの実測延長は時刻の標本精度も示す。任意の実行時間の上限を事前保証できないので、補充ありで延長ゼロを保証したとは報告しない。
4. R36は新規実探索を起動せず観測できる対象processが存在する場合に、既存のthread CPUとGIL観測を同時に取得する。旧版・新版の比較は同一相当負荷の別途許可された運用機会に行う。profileが利用できない場合はthread CPUのみ報告し、GIL占有率は不明と明記する。
5. R43は別途許可済みの固定科学入力で従来方式・pool方式を比較する際、数値結果や分類の変化がないことを先に確認する。cache identityとtask-bound seed、各BLAS/OpenMP設定、測定資源、total CPUを記録。同値性が崩れた場合は採択しない。実際のpool再利用が成立しなければ既存方式を維持する。

今回は追加の科学計算・数値startup・PBS予約・実際の所有移転・稼働release切替を行わない。上記は実機合格のための具体条件であり、実測済みの主張ではない。次の実機検収は対象と科学実行許可が揃った際にCodexが担当する。

人工統合回帰：**617 passed, 1 deselected, 50 subtests passed（116.52秒）**（`/tmp/rpe-review-five-final.log`）。除外1件は以前からの固定歴史AST比較（公開baseで既に不一致）で、科学選択SOURCEや基準は変えていない。R36容量集計の局所人工比較（受理320件、8host、15回中央値）は従来1.787365ms、batch 0.507221ms、件数・待機時間合計一致（丸め誤差内）。実中央全体の高速化倍率ではない。

最終のR8受理時の親・兄弟queue重複owner拒否追加後、関連回帰 **77 passed（2.21秒）**。fixtureのcancel API名の誤記は修正し、再試験成功。最終 `git diff --check` PASS。採択済みH8全体PLAN SHA `c476b62bf859751d4839de0fc36dfa8b5706ad9d76b4e7fca5f5a1aed496c86a` 不変。空き容量約34GB。中央registryを読み取り再確認し、H8最新該当RUNはSTOP・EXITあり、H6の現行固定版は再確認できなかった。

## 許可済み実機検収（2026-11-28、今回）

公開共通branchはGitHub compareで `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と同一。通常shellのGitHub SSHはDNS failureだがconnectorで照合できた。RomuTailはローカルext4、32論理CPU、62 GiB RAM、空き容量約50 GB。現在の中央登録は `romutail-pbs-independent-f37e1ebd0963`。H4の実稼働controller PID 665284は `romutail-h4-independent-continuation-ca27a9ff3a59` の別運用overlayであることを実processから確認（過去のEXITファイルの存在だけでは現在状態を判定しない）。H8の旧runのSTOP/EXIT、H6の前回固定版情報は維持し、H6現稼働は確認できていない。稼働中のcontroller/worker、PBS予約、SOURCE、保存結果、予算を変更していない。

### R10：RomuTail実ファイルでの検収

H4の既存実run `search-538c35a68be6425b1b4c` の jobs/transfers roots を読み取り比較。46,865 inode、7,599 directories、allocated size 5,810,044,928 bytes が7回すべてGNU duと一致。中央値は du **0.080308秒**、差分台帳 **0.331714秒**。ledgerは約4.13倍遅いため、当該ファイルシステムでの既定有効化を不採択。既存の背景整理・期限内sample cacheとduを維持。NFSには従来方式を保持する。ローカル容量の精度・方式選択を検収完了（PBS NFSでの新規計測は接続回復後）。[実測](../../artifacts/rpe_common_real_validation_20261128/R10_REAL.json)。

### R43：実固定DP kernelの数値同値性、pool再利用

実際のH4受理job `a651c4cac3335804e4fa` の保存CASE_INPUTS・同一Git configurationとsignal packet、shot vectorを使用。H4 linear 1 Å、STO-3G、DF10（tol-only 1e-8）、PR DF LD=3、2nd、delta_pf=0.6、13 rounds、h=0.1、固定solver proxy target、EXACT_AFFINE、固定frontier8192・transition141,559,036。新たなsignal生成やshot探索ではなく、旧runから完全に分離した許可済みR43の直接kernel検証のみ。

固定view `bd42dd1e95eb1d4152216d62792ad3421b803f7c` をGit archiveから復元。実workerの同viewにあるDP accelerator、risk、decoderのSHA-256と一致。通常Q3 routeの技術preflightは固定max-width source `06c50ce3628d7ce8b98c0cd04a54eb0c749557c8` の不足で停止（SCIENTIFIC_ROUTE_STARTなし）。このguardは緩めず、既存のtables_forを使った**直接固定DP kernel限定の比較**として別ディレクトリで実施した。旧runのretry・科学結果採用・再開ではない。

初回のfresh/persistent比較は全出力一致だがpoolが全回fallbackしたため不合格。原因は固定dropのグローバルにmutable acceleratorのbound methodが入ること。元のstateless grouped accumulationを固定関数bindingとして渡す修正を行い、別の新規比較ディレクトリで同一条件・2 DP workers・BLAS/OMP thread=1を両方式に使用。修正後は全出力JSONのbyte完全一致（SHA `aa2b76ed980937da52f00a6ca06ac7646a7ebb90a1f08d8bc03ed6055d5c493e`）、2 prefix cacheのkeyと解凍内容も一致。残差4回（completed 8,9,10,11）にfreshは4組のPID、persistentは同一2 PIDを使用し3回reuse。正の再利用と結果同値性を検収完了。

| 指標 | fresh | persistent |
|---|---:|---:|
| wall秒 | 2.848132 | 2.773089 |
| parent＋child CPU秒 | 4.458841 | 4.417354 |
| parent peak RSS KiB | 160908 | 166576 |
| 最大child RSS KiB | 128088 | 136668 |

1比較であり実run全般の高速化率ではない。RSSは同時総RSSではない。性能差は小さく、既定有効化は行わない。R43 kernel比較計4回（初回2＋修正後2）、追加Q3 leaf・探索投入0（通常Q3入口は技術preflightのみ）。[照合結果](../../artifacts/rpe_common_real_validation_20261128/R43_COMPARISON.json)・[比較script](../../artifacts/rpe_common_real_validation_20261128/KERNEL_COMPARISON.py)。

### R36：実queueでの改善と稼働controller計測

実稼働H4 controllerの15秒標本で8 threadsのCPU時間合計0.35秒、最大thread 0.23秒。待機負荷の短い標本でありGIL占有率ではない。[CPU計測](../../artifacts/rpe_common_real_validation_20261128/R36_LIVE_CPU.json)。py-spy 0.4.4を一時directoryにのみ取得しnonblocking GIL attachを試したがPermission Denied（ptrace_scope=1、perf_event_paranoid=4）。root/sudo・OS設定変更は行わない。

H4保存実queue `search-78ee25e2f10dcaccee5c` 11,282 wait rows・5 hostを読み取り専用で比較。初期batchは中央値0.540854秒、既存0.284998秒で悪化。EXPLAIN QUERY PLANでwide job recordへの読み出しを特定し、既存host_completion/host_adoptionのselective index、wait用host_state_queue covering indexを生かした単一snapshotへ修正。再度15回比較し件数と待機合計一致（丸め誤差内）、旧 **0.275302秒** → 新 **0.257106秒**。これは実データ局所集計の約6.6%短縮であり、実controller全体の改善率ではない。

さらに自分の読み取り専用診断processをpy-spyから起動できた。修正後旧45回/12.073秒、GIL-filtered trace 7件、新46回/12.240秒、trace 9件（50 Hz、非blocking、各12秒、error 0）。この標本数は全controller GIL占有率に換算しない。実稼働controllerのGIL直接測定は権限上未検収。[実queue比較](../../artifacts/rpe_common_real_validation_20261128/R36_QUEUE_CORRECTED.log)。

### R8・R18：PBS実機条件

既存run設定と現在のrelease設定のPBS SSH先に、BatchMode、ConnectTimeout、厳密host key条件で計2回到達性を確認した。いずれもbanner exchange timeout、終了255。物理予約やownerを変更していない。
R8は遠隔fenceと所有移管、再接続後実worker拒否の確認が残る。完全通信断を根拠に移管しない。

R18はH4既存PBSイベント（`search-78ee25e2f10dcaccee5c`）から過去の同一token DRAIN→IDLEを2組照合、388.533秒（約6.48分）と5,838.178秒（約97.30分）の実baselineを取得。[baseline](../../artifacts/rpe_common_real_validation_20261128/R18_HISTORICAL_BASELINE.json)。この記録は旧版であり、新しい厳格/見積りmodeでの実測比較、CPU利用率、更新時の追加owner観測はPBS接続回復後の検収が必要。

現時点で前段38所見に加えR10のローカル容量方式選択、R43の固定kernel直接検証を完了（40件の記載範囲の検収）。残るR8・R18・R36は技術実装済みだが実機条件を満たさず、全43件の完全検収とはしない。PBSが接続可能となった時にCodexが既存所有・予算を照合し、実移管・rolloverの検証を実施する。R36は現在のOS権限で直接attachが可能な対象または新たに許可された検証controllerが利用可能な時に実GIL測定を行う。科学計算の再承認を重ねて要求しないが、既存の個別STOPや稼働版の無断切替は行わない。

最終修正後の人工統合回帰 **617 passed, 1 deselected, 50 subtests passed（118.96秒）** (`/tmp/rpe-review-real-regression.log`)。固定DP viewを復元した既存回帰にpersistent再利用試験を追加し **20 passed, 5 subtests passed（5.19秒）** (`/tmp/rpe-review-fixed-dp-final.log`)。広域の除外は前述の歴史AST比較1件のみ。最終 `git diff --check` PASS、H8採択PLANは不変。新規証拠は隔離worktreeの `artifacts/rpe_common_real_validation_20261128/` に保持し、稼働版への切替・commit/pushなし。

## R36の検証controllerによる実機GIL/CPU検収（今回）

公開共通branchはGitHub compareで `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` と同一。ディスク約52 GB空き。H4実稼働controller PID 665284は引き続き存在。PBS SSH先prismの到達性は今回もbanner exchange timeout（終了255）。利用者は現在PBSに接続できないことを了承したため、R8/R18の追加接続試行・実PBS操作を中止し、実機検収保留とする。

既存稼働H4 controllerにpy-spyの非停止attachを再試行したがPermission Denied。OSのptrace_scope=1、perf_event_paranoid=4は不変。sudo、OS設定変更、稼働controller切替は行わない。前回の実queue11,282件比較と実H4 thread CPU標本に加えて、今回は許可済みの**隔離通常controllerをpy-spyの子processとして起動**し、通常のschedule controller、worker、RPC、owner受理、reportまで実行してGILとthread CPUを観測した。従来の `check_rpe_distributed_q3.py` fixtureを再利用し、3つの独立spoolの人工workerをRomuTail上のloopbackで動かす。実科学計算0、実物理ホスト1、模擬worker 3。旧個別集計と新版batchのみを切替え、他のcontroller実装と人工入力は同一。新規実探索・稼働版へのhot patchは行わない。

| 検証（100Hz） | 個別集計 | 新batch集計 |
|---|---:|---:|
| 非blocking wall秒 | 4.5115 | 4.2070 |
| 非blocking parent CPU秒 | 1.3258 | 1.2627 |
| 非blocking GIL-filtered traces | 67（error 5） | 78（error 12） |
| blocking wall秒 | 4.1289 | 4.0380 |
| blocking parent CPU秒 | 1.4073 | 1.3217 |
| blocking GIL-filtered traces | 89（error 0） | 68（error 0） |
| blocking 近似 nominal-sample比 | 20.82% | 16.22% |
| blocking queue observer thread CPU秒（各1 sample） | 0.001561 | 0.000867 |

2組とも各18件の人工outcome全体SHA `17afcc63c71955f9cac1a36095551a477d681fea2936da142764fbe26ce6c382` と方法別最良点が完全一致、3workerすべて受理・採用あり。非blockingとblockingでGIL標本の方向が異なるため、短時間標本からGIL負荷の因果的低減や実科学runの高速化倍率を結論しない。比はGIL-filtered trace数 / (設定周波数 × profiler wall) の参考値で、startup/検証も含む。短命threadのCPUは50ms標本から抜ける可能性があるためprocess CPUと別記録。blocking samplingは自分の隔離processのみで、実稼働processは停止していない。前回の実queue局所速度0.275→0.257秒、実controller15秒の8thread CPU標本を併せてR36の**検証環境でのGIL/CPU観測・局所集計改善・通常controller行動一致**を検収完了とする。本番科学controllerのGIL直接計測は未実施であり、実科学負荷のGIL率や本番速度は主張しない。

[比較記録](../../artifacts/rpe_common_real_validation_20261128/R36_CONTROLLER_PROFILE_COMPARISON.json)・[再現script](../../artifacts/rpe_common_real_validation_20261128/R36_CONTROLLER_PROFILE.py)。4本のraw profile、CPU標本、check結果、profilerログは `artifacts/rpe_common_real_validation_20261128/R36_CONTROLLER_PROFILES/` に保存。全人工controller/workerは終了、既存稼働版・raw科学結果・STOP・PBS予約を維持。

最新内訳：**41件の記載範囲の検収、2件（R8・R18）の実PBS検収保留、技術未実装0件**。利用者の最新指示によりPBS接続不可は現在の作業対象から保留。次のPBS復旧時にCodexが許可済み範囲の所有権照合とrollover検証を担当する。今回は追加の科学計算はなく、既存実科学runへの介入もない。

## 共有版統合検証（2026-11-28）

利用者指示により、全43項目の技術修正を `codex/rpe-common-hchain-fleet-20261126` に共有する。公開共通HEAD `512978edcbb2ff884a5d69fa1ca7ad00bceb7048` をGitHub compareとGit ls-remoteの両方で確認し、隔離修正の親と同一。ローカル共有checkoutはこのSHAの祖先にあり、未commit変更なし。既存の全登録release code_rootおよび観測された稼働RPE process（H4 controller・relay・worker等）はローカル共有checkoutを参照していないことを確認。H6の現在固定版は前述の再確認不可の状態を維持し、個別配置は変更しない。

統合前の最終回帰：**665 passed, 1 deselected, 50 subtests passed（141.46秒）**。固定DP viewの別プロセス回帰：**20 passed, 5 subtests passed（4.86秒）**。除外は従来の固定歴史AST比較1件で、科学選択SOURCE・比較条件を緩和していない。[統合回帰](../../artifacts/rpe_common_real_validation_20261128/INTEGRATION_REGRESSION.log)、[固定DP回帰](../../artifacts/rpe_common_real_validation_20261128/INTEGRATION_FIXED_DP.log)。R10/R43の生成cacheの実体は既存ignore規則のままローカル保全、圧縮・解凍後SHA-256とサイズを [R43_CACHE_HASHES.json](../../artifacts/rpe_common_real_validation_20261128/R43_CACHE_HASHES.json) に記録した。H8採択PLAN SHAとstartup証拠は不変、追加科学計算なし。

41所見は記載した範囲で検収済み、R8/R18の実PBS検収は保留。実装は共有するが新機能の既定有効化・新規実PBS操作・稼働版切替を行わない。詳細と既定設定は[共通運用ルール](rpe_distributed_operations.md#共通実装レビュー43項目の統合2026-11-28)も参照。
