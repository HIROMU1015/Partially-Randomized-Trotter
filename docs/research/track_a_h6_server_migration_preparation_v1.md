# H6共用サーバー移行準備 v1 — Phase A認可待ち

2026-10-12 JST。資料入口は[準備bundle](../../artifacts/resource_applicability/track_a_h6_server_preparation_v1/2026-10-11/README.md)。
準備を開始した日付2026-10-11のpathは保持する。新しいH6科学runner起動0、新approved grant0、grant消費0。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` / `next_stage_authorized=false` / mandatory STOP。
今回は移行用source・専用環境・既存結果受領・限定検査・実行計画を完成させた。Phase A実行は別の明示認可を要する。

## 固定identityと取得

Repositoryは`HIROMU1015/Partially-Randomized-Trotter`、originは同repositoryのHTTPS remote。
branchは`track-a-h6-server-migration-preparation-20261011`。
worktreeは`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011`。
起点公開commit `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379`と科学SOURCE
`5a903aea2492c2d211c1e15e7737df0d50694c12`を保全した。対象元branchのorigin SHAは起点と一致する。

移行用SOURCEは`61cadec5b5b2c8f1138828f43f7780d36319ccc3`、Git treeは
`531f62f4c1025c14837af0de3dd98db514f17cca`。remote再取得で同じSOURCE/treeを確認した。
このSOURCEは利用者の「pushはこちらでする」指示より前に公開済み。その指示以降はpushしていない。
REVIEW_BUNDLEはSOURCEの子のローカルcommitへ固定する。REVIEWの公開とremote照合は利用者のpush後に行う。
通常のHTTPS git pushはcredentialがなく失敗し、設定は変更していない。SOURCEの公開には既存GitHub connectorを使用した。

SOURCE追加10 pathsは`source_remote_verification_v1.json`に列挙した。
旧224科学sourceを変更せず、専用contract/resources/quota/execution 4 modules、prepare/audit/runner 3 scriptsを含むruntime closure231 pathsを固定。
環境inspect/reproduce 2 scriptsとtest 1 pathもSOURCEに含む。
既存Git blob由来のraw/inputは専用worktreeへ取得し、新たなNPZ・science raw・cache・credentialをcommitしていない。

## 専用環境

実行Pythonは`/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`。
公式CPython 3.11.0rc1をhome配下の専用prefixへsingle make workerでbuildした。
必要なheader/libraryは専用directoryへdebを展開し、共有package managerへのinstallや設定変更を行っていない。
H4用Python 3.12.3/private venvは変更していない。

| Identity | 検査結果 |
|---|---|
| Python | 3.11.0rc1 完全version一致 |
| Qiskit / NumPy / SciPy | 1.3.0 / 1.26.4 / 1.14.1 |
| OpenFermion / OpenFermion-PySCF / PySCF | 1.6.1 / 0.5 / 2.7.0 |
| Numba / llvmlite / mpmath / h5py | 0.60.0 / 0.43.0 / 1.3.0 / 3.16.0 |
| integral-provider source SHA | `f092eed5b9792bf5ba54b78fe8614a5d261d1eba700dec33f3923f4a81e89add` |
| compiler fingerprint | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` |
| environment profile fingerprint | `cbaf068eb8a0dbb5a6843eeb054e7cb367c6eee2951b04baaed322b746478561` |
| profile file SHA | `2f47110441c6f2bb399ca8241c5cbf11d29ddffcc4fc38a6273a1ecdbe72a985` |

旧environment gateの全項目とCPU側importは一致した。47 distributionsのwheel SHAと33343 environment/native/source filesを保存した。
NumPy同梱bytecodeのRECORD差1件は、専用rc1 interpreterで対応sourceをcompileしたbytesとの一致を検査し、profileに説明を固定した。
Qiskit synthesis plugins/defaults/native lowering sourceを固定した。旧native binary inventoryは提供されていないため、旧hostとのbinary bit同一性と実H6 compiled outputの完全同一性は未検証。
GCC13.3.0/OpenSSL3.0.13での新buildおよびrustworkx0.18.1等のtransitive dependenciesは新profileへ固定し、旧hostと同一と推測しない。

再現資料は`environment_build_inputs_v1.json`、`requirements-server.lock`、`dependency_wheel_inventory_v1.json`。
再現helperは`scripts/resource_applicability/reproduce_track_a_h6_environment_v1.py`。
別のexclusive home directoryにのみ作成でき、再現後には新absolute pathsを含むprofile/sealを作り直す必要がある。
今回、第二の環境再作成は行っていない。

## 受領と再利用

`receipt_server_v1.json`とsealed manifestのregistryに、bytes/SHA/Git blob/terminal/task/seed/compiler/full-wrapper semanticsの照合を保存した。
NPZはstreaming bytes/SHAとheader identityを検査し、準備中に科学arrayをdecodeせず、入力再生成・SCF・DF・state生成を行っていない。

| 既存証拠 | 受理数 |
|---|---:|
| 科学source | 224 paths |
| frozen input | 33 paths |
| 旧raw | 5220 paths / 77980307 bytes |
| signal | 92候補 |
| B3 normalization診断 | 84組 |
| 完了exploration | 449/592 group |
| 保存wrapper | 902件 |
| 未完了exploration | 143 group |

存在確認だけではなく既存reuse契約を適用した。旧449 groupを再計算しない。
未完了indicesは447および450–591。今回のPhase Aは447だけ、残り142 groupは450–591。
旧STOP・旧grant・registry/raw/checkpointは原bytesを保全する。旧abort時の未記録actual呼出数を0へ補完しない。

Phase A固定task：`B1_p19_q8_4th`、exploration index447、replica0、seed`17585593421502146214`、prefix19、q8、四次PF、ordinary=false。
cosine/sineのprimary full wrapper各1、compile最大2、control probe最大4、trajectory/occurrence新生成0。
task digestは`b0d87896948c3b267727d5a7c39794638dcf66b29e52d6a591436a70b41e82b6`。
scientific plan digestは`801e1eea5d09b6fbe60fe66102bc21edb1bd50a0d9aa94ca05a2b0659d1f8cf3`で旧planと一致する。
linear H6/1Å/STO-3G/tol-only rank19/sector400/保存state/T0.8、92候補・4精度・PF/PR/shot/normalization/seed/native lowering/compiler/費用定義を維持する。
H4 Gaussian高速化・H4 cache・proxy費用へは接続しない。

## Phase A資源計画と監視

| 項目 | 固定された提案 |
|---|---|
| cost worker | 1、CPU affinity [20,21]、内部thread1 |
| coordinator/monitor | CPU affinity [22,23]、内部thread1 |
| worker AS soft/hard | 32/32 GiB |
| coordinator AS soft/hard | 2/32 GiB |
| aggregate RSS STOP | 40 GiB（coordinatorとworker descendantsを含む） |
| job RAM予算 | 64 GiB。予約済み資源ではない |
| fresh host / effective memory | 216.25 / 64 GiB以上 |
| running host / effective memory | 168.25 / 16 GiB以上 |
| pressure / OOM | host full PSI avg10 >=5%、cgroup full PSI >0、新OOM/OOM-killでSTOP |
| 観測 | interval1秒、観測/間隔の5秒超・欠測はSTOP、journal通常10秒毎 |
| output / worker log | 2 GiB全体 / 1 MiB |
| worker private / monitor journal | 各32 MiB、全体2 GiBにも算入 |
| filesystem admission | 空き10 GiB以上 / 100000 inode以上、quota既知で余裕あり |
| phase/total wall cap | null（旧H6契約維持） |
| execution | one-shot、retry/resumeなし、成功/失敗どちらもmandatory STOP |

従来18 GiB aggregate RSSはサーバー用resource schemaで40 GiBへ変更する提案。
coordinator hardを32 GiBにすることで子workerが32 GiBを設定でき、親soft2 GiBは維持する。
32 GiBは完走保証ではなく、実H6 q8 peakと所要時間は未測定。
job RAM64 GiBはadmission用予算であり、64 GiBの新cgroupを作成したりscheduler予約を得た値ではない。
実メモリ制御はAS limit、aggregate RSS、host/全可視cgroup headroom、PSI/OOM監視で行う。

coordinatorが重いnative/GIL処理を行うworkerから独立して観測する。新H4 observer roleを流用しない。
第一STOP理由、観測時間、interval/staleness、driver phase、RSS/VmSize/VmHWM/VmPeak、worker退出を保存する。
正常終了時にはworker最終peakもまとめ、transpile traceから両軸の所要時間を保存する。
親死亡時はworkerのPDEATHSIG SIGKILLと親identity再確認を適用する。
own leaderのPID/start/pgrp/sessionを確認し、waitid(WNOWAIT)で未reap identityを保ったままgroupを終了しwait/reapする。
ESRCHだけを既退出扱いし、他error・ownership不一致を成功扱いしない。ログ・pipe/FDはcontext終了で閉じる。
失敗時はpartial outputとSTOP/auditを保全して終了し、AS増量によるauto retryは行わない。

hostはEPYC7532、64 physical/logical CPU、RAM約503.66 GiB、swap16 GiB。
観測したavailable RAMは約465 GiB、home空き約406.7 GiB、available inode約2.20億、quotaはuser/group/projectともdisabled。
自身のinherited RLIMIT_AS hardは無制限、可視cgroup memory.maxはmax。これらはH6への予約・排他的割当を示さない。
SLURM/PBS/LSF clientは検出されず、正式なサーバー利用規則・job予約は未提供。
実行承認時にCPU20–23/RAM64 GiBの利用許可を確認し、schedulerが別途提供されればその運用へ合わせる。
現在の案はH6専用foreground coordinatorを持続する自身のsessionで管理する。sessionは今回作成していない。

H4のread-only観測ではworker CPU2/4/5/6、driver16、observer18で継続中。H6提案CPU20–23と交差しない。
H4のworktree/venv/成果/registry/cache/monitor/processおよび他jobは変更していない。
この静的観測だけで将来の同時利用を予約済みとは判断せず、launch直前にH4の実affinityと資源を再確認する。

容量は`storage_projection_server_v1.json`にpayloadと実allocated block/file数を分けて保存した。
専用environment一式は約2.03 GiB allocated/39130 files、worktree約332 MiB allocated/9910 files。
将来runtimeは全体2 GiB capにcache/tmp/journalを算入し、別途filesystem空き10 GiB/100000 inodeを要求する。
監視journalは32 MiBでSTOPし無限に伸ばさない。atomic pending→hardlink publicationでpayload blocksを二重計上しない。

## 限定検査と限界

`server-tests-release-final.xml` 38件と`compatibility-tests.xml` 18件、計56 distinct local cases PASS、skip0。
38件はmanifest/grant scope、AS継承、CPU/thread、RSS/PSI/OOM、遅延/欠測/退出、own cleanup、quota、one-shot、親死亡とreap、Phase B禁止等のmetadata/mock/最小process検査。
18件は旧小規模人工wrapperの旧/new bytes/意味論/compiler metrics/input lifetime検査。
人工transpileは2 axis×旧/newの4 callsのみ。実H6 input/circuit/transpileは0、大規模stress0。
単一test driver、内部thread1。親死亡caseだけdriver+harness+parent+childの最大4最小process。
AS32 GiBのlimit設定は確認したが32 GiB allocation stressは行っていない。
これはlocal synthetic証拠であり、実H6 feasibility、旧host binary equivalence、CI/外部再現PASSではない。

最終binding auditはSOURCE closure、profile全file、旧input/raw/reuse契約を再照合する。
`final_consistency_review_server_v1.json`に技術結果と未認可事項を分ける。
レビューは担当による限定静的整合性確認であり、別担当の独立reviewと称さない。

## 次の認可対象と未実行command

必要な利用者認可は、固定environment/compiler/profileを使用し、CPU20–23・上記AS/RSS/RAM/監視/output予算で
`B1_p19_q8_4th` index447だけを一度実行すること、および共用サーバーでこのCPU/RAMを使用してよいこと。
正式な利用規則・schedulerがある場合、その条件も必要。
approval後に別fileで新grantを発行し、manifest/profile/source/task/reuse/outputの全digestと結合する。
旧grant・draft fileをそのまま実行認可として使用しない。

現在のdraftは`approved_by_user=false`、`server_usage_approved_by_user=false`。
exclusive output leafと`.server-preparation/h6-server-phase-a-v1/phase-a.lock`は未使用。
未承認のPhase B（最大2 workers/AS32 GiB/RSS72 GiB/job96 GiB候補）、独立32 trajectory確認、H8へ自動移行しない。
Phase A成功時は探索450/592・残り142となる見込みだが、保存済みresultを受理するまでそう報告しない。

以下は認可後の予定commandであり、今回実行していない。
`ACTUAL_APPROVED_GRANT_SHA256`は認可後に作る新grant原bytesの実SHAへ置換する必要がある。
指定する新approved grantも現時点では存在しない。

```bash
env TMPDIR=/home/AbeHiromu/projects/h6-environments/20261011/tmp \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 \
  /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -I -P -B \
  /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011/scripts/resource_applicability/run_track_a_h6_server_phase_a_v1.py \
  --execute \
  --manifest /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011/artifacts/resource_applicability/track_a_h6_server_preparation_v1/2026-10-11/sealed_manifest_server_v1.json \
  --authorization /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011/.server-preparation/h6-server-phase-a-v1/authorization-approved-v1.json \
  --authorization-sha256 ACTUAL_APPROVED_GRANT_SHA256 \
  --output /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011/artifacts/resource_applicability/track_a_h6_server_phase_a_v1/2026-10-11/launch_server_v1
```

launch直前にremote/source/profile/input/reuse一致、H4実affinityとの分離、fresh memory/PSI/OOM/filesystem/inode/quota、
inherited limits、unused leaf/one-shotを検査する。fresh観測不合格ならlaunchせず具体的な原因を報告する。
認可とこのfresh検査は今回の準備testsにより代用しない。

## 固定と公開の手順

SOURCEは公開済み。残るREVIEW_BUNDLEはsourceを変更しないローカルcommitで、metadata JSON/lock/XML/docs/indexだけを明示stageしたもの。
利用者が次のnon-force pushを行う。Codexは今回これを実行しない。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-matched-server-preparation-20261011 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-server-migration-preparation-20261011
```

push後はremoteを再取得してREVIEW SHAとbundle inventory bytes/SHAを照合する。
旧449 groupと今回の準備commitを混同せず、将来のPhase A raw/result/auditは別の結果公開commitへ固定する。
