# H4 RPE方式別最適化：PBS計算先追加（2026-11-18 JST）

後続の[中央・価格・PBS実統合REPORT](../rpe_pbs_central_integration_20261118/REPORT.md)で、
合流SOURCE925985b4eの実PBS検収・検証済み登録を完了した。以下の単独版22の原結果は保持する。

実装・利用者専用環境配置・保存実入力dry-check・実PBS人工通し検収を完了した。
実H4の信号取得・回路compile・Q3本体は0件であり、科学的性能や優劣を追加認定しない。
人工compileは文字列の回路artifactを作るテストで、量子回路のcompileではない。
通常操作・再開・残る設定は[HANDOFF](HANDOFF.md)。

## 版と既存実装の再利用

専用branchは`codex/rpe-pbs-worker-20261118`、基点はRomuTail統合版`14a6102d4`。
実PBS検収・配置済みcontroller/transport/workerの固定SOURCEは
`22a428bd34cbc3705b90ca6f4407053a6d996f05`。
後続の公開commitは外側の人工障害fixture、release設定生成、文書・証拠の追加である。
科学Q3 SOURCE `28d976b99cdf8329adcc1b79327e329e947ef774`と各provider/routeのSOURCEを保った。

指定された`dea1cb530`、`407fd460`と後続配置`5910058fe`、その実装`a18d5a04d`を確認した。
JobStore、DistributedExecutor、既存Worker、工程別profile、成果CAS、DP cache、task単位予算、
計時・同attempt再開を再利用した。既存PBS実装・同条件の保存PBS検収はなく、新接続が必要だった。
既存のmain checkoutの変更、既存科学run、SOURCE、原結果、STOP、停止中server1は変更していない。

作業中に中央currentは`romutail-6da8554e0-r4`へ進んだ。
8個のqueue/worker/profile/成果/runtime/registry/入口ファイルが基点と同一で、接続method signatureも一致した。
差分と取込み箇所は[中央照合](deployment/CENTRAL_COMPATIBILITY.json)。
新release `pbs-22a428bd3`を既存RELEASESへ追加し、既存currentの選択は保持した。
[登録結果](deployment/REGISTRATION.json)を参照。

## 共有環境と配置

RomuTailのSSH設定で確認したaliasは`prism`。`akeno`経由で`AbeHiromu@192.168.5.26`へ接続し、
hostnameは`CPUcluster-PRISM`である。hostnameをSSH aliasとして仮定していない。
PBS20.0.0、workq/largeを確認。キュー既定memは350GB/1516GBだが、今回の全ジョブはCPU・mem・walltimeを明示した。
可視のserver/queue設定には追加のwalltime/同時実行上限が見えなかった。無制限の利用権を意味しない。

ホームは計算ノードから同じパスで読み書き可能、`TMPDIR=/var/tmp/pbs.<jobid>`はnode一時領域だった。
計算ノードからRomuTailへは到達不可。共有ホームの要求/応答spoolを管理ホスト上の薄いgatewayが中継する。
gatewayは科学workerを管理ホストで起動しない。stdout返送を待たず各成果を回収し、終了後にも共有ホームから読める。

OS affinityは96 CPUを露出し、nodefileは要求2 CPUでも1行だった。
`qstat Resource_List`の実CPU/memを境界にし、さらに狭いprocess/cgroup条件を併用する。
DP並列数×内部thread数、工程profileの最大CPU費用、task memory指定から同一の枠poolを作る。
Q3/compileが別poolで同じCPUを予約する実装にはしていない。
NumPy/BLAS/OpenMP/Numba/Rayon/Qiskitの内部並列環境変数はそのworkerの子processだけに設定する。

system Python3.6.8は使えないため、利用者専用領域にPython3.11.0rc1と既存runtime契約のlibraryを配置した。
Python buildもPBSの8CPU割当内で実施し、compileallの並列数を8へ合わせた。
共有Python、OS設定、他ユーザーのprocess/job/fileは変更していない。
環境の版・公式source・確認記録は[environment](environment/FINDINGS.json)、[runtime](environment/RUNTIME_SOURCE.json)。

## 検収結果

| 検収 | 結果・根拠 |
|---|---|
| 保存実Q3入力4方式 | 4/4 `REAL_HOST_DRY_CHECK_PASS` |
| signal/compile/audit保存入力 | 15/15 `SAVED_INPUT_RUNTIME_INSPECTION_PASS`、将来REQUESTの実行資格は未認定 |
| 通常入口→実PBS→成果採用 | 人工14 task全採用、8人工Q3結果が単機参照と完全一致 |
| 同workerの複数task | job267193でcompile+Q3、障害job267194でcompile+Q3×2 |
| compile→別host audit | PBSのCAS成果をRomuTailのauditへ引き継いで採用 |
| PBS保留中の他host進行 | 同じcontroller時計でH観測後にlocal signal採用を確認 |
| qsub応答欠落 | 受理済みjob267194へ同じtokenで復帰、割当1件・PBS run_count1 |
| 成果転送前切断→controller再起動 | 原compile attemptを採用、ASSIGN/ACCEPT/ADOPT各1回 |
| 完了run再開 | 新規LAUNCH/ASSIGN/ACCEPT/ADOPT/PBS割当すべて0 |
| 割当終了後回収 | 両job F/exit0、worker EXITED、保存成果を再取得 |
| 単機との採用結果 | 8結果、方式別best/rule_adopted_best/confirmed_all_bestの設定・shots・MSE・cost一致 |

原記録は[readiness](readiness/ACCEPTANCE.json)、[通常PBS](acceptance/cluster/COMPARISON.json)、
[障害・再開](acceptance/faults/COMPARISON_AND_RESUME.json)。初回不足Git objectによる4 dry-check失敗も残し、
元blobを補完後、その4件だけを再確認した。科学SOURCEや入力を変更して通してはいない。

PBSジョブは267189（環境）、267190（Python build）、267191/267192（metadata）、
267193/267194（人工worker）。各資源要求・終了状態は個別の記録と[最終job照合](FINAL_JOBS.json)を参照。

人工通し検収で既存の終了処理の不具合を修正した。
方式viewへ複写されたcontroller専用finish closureが中央executor全体を閉じていたため、
viewはclassの方式別finishを使い、任意の価格検証serviceはownerが1回だけ閉じるようにした。
候補順位・MSE計算・科学予算の規則は変更していない。

関連人工回帰73件中72件PASS。1件の古いAST比較は基点14a6102d4でも同じ失敗を再現した。
対象optimizer libraryは基点とbyte一致し、この既存比較を緩めていない。
[TESTS](TESTS.txt)、[基点での確認](BASELINE_AST_CHECK.txt)を参照。PBSと終了処理の追加21件はPASS。
別途、snapshot/remaining-time/cutoff/UNKNOWN/元receipt回収の人工検査を含むが、実H4のwalltime途中終了は実行していない。

## 計時と限界

[計時要約](acceptance/TIMING_SUMMARY.json)と既存の各TIMING.jsonを保存した。
controllerの実行可能compile enqueue→投入intentは1.214/1.308秒、worker内待ちは1.279～1.672秒。
人工本体はcompile 0.243～0.250秒、Q3約2.002秒。
controller受領→Q3採用は約0.027秒、workerのupload ACK往復は約2.016秒。
PBS待機99秒/55秒は検収の意図的H保留を含むため、通常queue latencyや速度倍率に使わない。

固定22aの不足object配置は5 object/58,193 B、RomuTail同時計の全体7.345秒、再利用5.435秒/転送0 B。
保存入力の初回展開・view配置は管理ホスト時計で15.125秒。最初のarchive転送単体は未計測。
異なるhost時計の片道差は算出せず、投入→最初のoffer観測、既存B+C包絡、受領→採用を分けた。
実H4のbody時間、最適slot、task memory、通常PBS待ち、科学的throughputは未測定である。

主担当Codexが接続実装・実PBS障害操作・配置登録・全体照合を担当した。
sub-agent `pbs_environment`が環境probe/runtime、`pbs_readiness`が保存実入力と固定配置、
`pbs_worker`がworker・人工回帰・単機参照・障害成果/計時照合を担当した。
これは技術検収であり、独立の重要科学reviewを実施したという主張ではない。
