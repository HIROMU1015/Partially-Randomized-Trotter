# Approved H4 one-shot execution procedure

利用者の2026-10-09明示指示は [USER_AUTHORIZATION_v1.md](USER_AUTHORIZATION_v1.md) に固定。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a` を変更せず、[plan](plan_authorized_v8.json) / [auth](authorization_v8.json) / [review](review_v8.json) の実digestを独立確認してからartifactをcommitする。

実行順序：

1. 独立review PASSのreportとactual final artifact commit、SOURCE33/全参照SHAの一致を記録。
2. immutable commit後、固定SOURCEのread-only `verify_runtime` / `verify_frozen_receipts` でSOURCE/profile/input/carryを直前照合。既存proofの再受領・13identity再解析・回帰tests・benchmark・extra transpileなし。
3. 固定SOURCEの `host_readonly(...,sample_seconds=3)` / `fresh_gate` を一度行い、CPU/core/load/available affinity、memory/PSI/OOM、FS/block/inode/quotaを採取。unused output/control/one-shot/lockも確認。不合格ならscientific起動0で原因を固定し停止。
4. 合格後、home外部の `ONE_SHOT_EXEC_SUBMITTED_v1.json` をexclusiveに取得し、一度だけ実行ツールの管理sessionで `exec env THREAD_ENV Python -P -B SOURCE-fixed-runner` を実行する。[exact argv](runner_argv_v8.json) を使用。追加wrapper、monkeypatch、runtime supervisor、systemd/共有設定変更なし。
5. SOURCE自身の直前verify/runtime/input/carry/fresh3秒gateも維持する。そのgateがPASSした後だけexclusive one-shotを取得し、control/observer費用をreserve、own-affinity、driver/worker AS/RSS監視、independent observer、12 owned workersを開始する。
6. tool sessionとSOURCE one-shot/observer traceからdriver PID/starttime、observer/registered worker identity、CPU/source/monitorを確認し、起動情報を利用者へ報告。計算はSOURCE監視下で継続する。
7. 正常return/通常例外のcleanup経路ではSOURCEがown子processの停止/wait/reap/FD/pipe終了を行う。独立observerの強制STOP経路はown identity限定signal後にdriverを終了するため、driver側finally/wait/reapが走らない場合がある。終了後のown identity残存と、tool sessionが回収できた実exit statusをread-only確認し、reap等の未記録情報を補完しない。自動retry・worker replacement・入力再生成・次stageなし。

runtime rolesはdriver1＋worker12＋observer1。数値library thread1、Qiskit num_processes1、Python -P -B。SOURCE既存のI/O/監視thread構造は維持する。
driver CPU16、workers [2,4,5,6,8,9,10,11,12,13,14,15]、observer CPU18。driver/worker AS/RSS各8GiB、headroom16GiB、observer AS256MiB/RSS64MiB、admission120.25GiB、monitor5秒。
carry20 actual/165214360 bytes/5466.188392877579秒は固定。累積actualだけ74804、10GiB/72h等は不変。未検証の旧compiler output equivalenceを保持し、旧partial/cacheは利用しない。

科学進行のdurable logはSOURCE control_root/runner.log（8MiB cap）、監視はoutput_root/observer.jsonl（SOURCE予約済みcap）。
preclaim stdout/stderrはtool sessionで回収し、保存する準備/起動diagnosticの合計は64KiB以内のhome内metadataへ限定する。追加の無制限外部log fileやstdout collector processを作らない。
Sourceは8MiB control log＋64KiB metadataと全observer trace費用を起動前にreserveする。raw runtime/checkpoint/cache/log、NPZ、credentials/内部SSH情報はGitへ入れない。

最終artifact固定とfresh gateは利用者承認の実行条件。追加の利用者承認を要求しない。
