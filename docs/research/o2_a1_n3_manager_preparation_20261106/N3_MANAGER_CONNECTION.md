# A1 N3：持続managerの接続と保証範囲

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**既存のユーザーcron経路を人工jobで使用可能と確認した。実N3は未起動。**
固定科学SOURCE03928、中断結果d97e、調査857350e、事前判定a53cを継承する。旧SOURCE・exact kernel・乱数/測定/統計処理のbytesは変更していない。

## 管理経路

既存cron.serviceがactiveで、ユーザーcrontabが利用可能だった。at/Slurm/PBS等のcommandはなく、user managerはLinger=noのまま。
自分の一時crontab一項目だけで人工qualificationを投入し、実jobが`0::/system.slice/cron.service`に属することを確認した。
SSH側の`/user.slice/user-30038.slice/session-212829.scope`から分離され、投入client processはjob開始前に既に終了していた。
実際のSSH・他ユーザー・共有serviceを終了したり、linger/logind/systemd/cgroup設定を変更したりしていない。

```text
既存cron daemon（変更なし）
  └─ own-user consume-once dispatch / 外側recorder
       └─ application manager
            └─ parent driver（private session/PGID）
                 └─ ManagedPool workers
```

`crontab`は標準のユーザーjob投入interfaceを用い、spoolや/etc/crontabへ直接書かない。[crontab(1)](https://manpages.debian.org/bookworm/cron/crontab.1.en.html)
今回、既存の自分のcrontabは無かった。既存tableがあれば保護して拒否し、他の設定を上書きしない。
MAILTOを空にし、出力を自分のjob directoryへ保存した。mail送信は行わない。[crontab(5)](https://manpages.debian.org/bookworm/cron/crontab.5.en.html)

## Restart=noの意味

jobのapplication restart policyは**no**。
dispatchは最初にexclusiveなCONSUMED receiptをfsyncし、一時cron項目を削除して元の「crontabなし」へ戻したことを確認してからapplicationを開始する。
残存triggerや再接続から同じdispatchが呼ばれても、receiptがあればexit72で拒否し、applicationを再起動しない。
receiptは二重起動防止であり、ユーザーの承認を証明するgrant/certificateではない。

共有cron daemon自体の既存systemd設定は`Restart=on-failure`で、**変更していない**。これはapplication jobを再開・再実行する設定ではない。
systemdのjob propertyとしてRestart=noを設定したと偽らず、consume-once/remove-triggerで実装したapplicationの方針と区別する。
消費後・application開始前に障害が起きれば未開始のまま止まる可能性がある。保証はat-most-onceであり、必ず一回完走する保証ではない。

## 追加実装と再利用

- [cron adapter](../../../src/trottertracks/algorithm_codesign/a1_n3_cron_manager.py)：own crontabの確認/投入/撤去、source hash、consume-once、外側recorder。
- [managed runtime](../../../src/trottertracks/algorithm_codesign/a1_n3_managed_runtime.py)：親/worker identity、raw exit/signal、private job groupの回収、STOP先行保存、call trace、未完了分類、旧moduleへの接続context。
- [人工qualification入口](../../../scripts/diagnostics/run_a1_n3_manager_qualification.py)：現在のdispatchは**人工payloadだけ**。実N3を投入するCLIは提供しない。
- 既存のlifecycle observer、process_identity、wait_status、memory_snapshot、traced_call、atomic writerとjournal parserを再利用。旧serializer・bit→価格順序・reconcileはそのまま使う設計。

`connect_legacy(campaign,runtime)`は旧moduleのpool生成をManagedPoolへ接続し、selected build、exact probability、Born、compileの入口/戻り/例外を別journalへ記録する。
数値関数への引数と返り値を変更せず、値そのものをtraceへ保存しない。CVへstate/Born/真信号を追加して渡さない。
旧worker initializerをそのまま呼び、bit保存→価格処理の旧worker関数をpoolのowned taskとして実行する。wrapper自身にsamplingや統計処理はない。
caller側の旧STOP eventも共有し、guardが作動したときに新trialを投入しない。

## 終了・保全の接続

親・workerにPID/PPID/start ticks/boot/PGID/SID/cgroupを保存。実childのgroupとancestorのmemory/max/high/current/peak/eventsを読み取り、hostの/proc/pressureを併記する。各cgroup固有のPSI時系列は今回の取得範囲に含めていない。
共有cron cgroupの値には他jobも含まれ得るため、job単独の物理peakやquotaとして扱わない。設定は変更しない。
親/workerを所有するwait元でraw returncodeを回収し、signal番号と通常exitを区別する。signalの送信元は追加証拠がなければUNKNOWN。

STOP_REQUESTEDをfsyncしてからstop eventを共有し、POOL_DRAIN_STARTより先に記録する。
保存bit/価格を再取得せず、BIT_AND_PRICE_DURABLE、BIT_DURABLE_PRICE_MISSING、ATTEMPT_NO_DURABLE_BITを分ける。duplicate bit/priceは拒否する。
外側recorderはapplication managerのSIGKILLでもwait statusと独立したSTOPを保存できた。
後処理のsignalは記録したidentity・UID・private sessionを照合した自分のjobだけに送る。PIDが再利用されていれば操作しない。

ProcessPoolExecutorのworker registryはPython3.12のprivate `_processes`を参照する。元のwait ownerと別にwaitpidする実装は加えず、既存Process objectのexitcodeを保存する。
この依存は固定Python版の人工testで確認した。別版へ無条件に移送しない。

## 使用可能範囲と残る制約

**このhost・現行cron運用で、会話/投入clientの寿命から分離する管理経路は技術使用可能。** Lingerやtmux/nohup/setsidへの依存で成立させていない。
root cron daemon/host/filesystemの停止、全job group/cgroupの同時kill、外側recorder自体の消失への完全な耐久性は保証しない。
外側recorderも消失した場合は、残存receipt/source/journalを保全してUNKNOWNで停止し、同じjobを自動再開しない。
親wait ownerが失われたworkerのexitcode、manager消失時に未保存の親exitcodeはUNKNOWNのまま。回収signalを送った事実を終了コードに読み替えない。
実SSH logoutの破壊試験、実分子長時間負荷・peak memory、OS全体障害は試験していない。

N3 B用の新seed/identity/outputとsemantic return statusを扱う薄い科学入口・環境lockは、契約採択後に固定する必要がある。
旧`run_a1_n3_campaign.py`は旧output/markerを指し、STOPでも通常process exit0となり得るため、そのまま本番launcherへ接続しない。
manager exit0だけで科学完了と判定せず、N3のSTOP/CAMPAIGN_RESULTと全比較gateを照合する仕様にする。

qualified管理SOURCEは`5cb542ece4d208878a1dfc7011007b99297a5a55`。人工結果は[次文書](N3_LIFECYCLE_TESTS.md)、科学実行は[未採択草案](N3_NEW_CAMPAIGN_CONTRACT_DRAFT.md)を参照。
