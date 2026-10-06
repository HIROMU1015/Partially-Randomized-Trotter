# H4 signal／compile：凍結入力に結合した次段草案・容量不足STOP

2026-10-06 JST。利用者の「その方針で進めて」は次段plan・認可草案・容量条件の準備を認可した。
科学計算の開始指示ではない。**H4_SIGNAL_COMPILE_PLAN_PREPARED_STORAGE_BLOCKED_STOP**。
source19件と旧bundle・入力生成plan/auth/review・run01/run02は不変。
signal／trajectory seed／sampling／circuit build／compile／transpile／worker／taskset／GPU query/useは今回0。
shared environment・他job変更0、capacity reservation・削除・移設0。stage reviewは `approved=false`。

## 判定と資料入口

| 項目 | 固定内容・今回観測 |
|---|---|
| 科学scope | H4 linear、neutral singlet、STO-3G、4 spatial/8 system＋ancilla1、DF12、T=0.8、二次DF-prefix PF/canonical finite-RTE |
| 距離 Å | 0.70 / 0.80 / 0.90 / 1.10 / 1.40 / 1.60、run02凍結済み6入力 |
| 登録候補 | 各218、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、r/Kは旧templateをそのまま転記 |
| 将来処理上限 | signal 1,308、random194/templateは32 paired trajectories、baseline24/templateは二軸、logical wrappers / actual invocations最大74,784 |
| CPU提案 | [3,5,6,7,8,9]、6 workers、own-runだけのmask0x3e8、各内部thread1、次段CPU利用許可未取得 |
| 保守的stage必要量 | **3,758,096,384 bytes = 3.5 GiB、560,000 free inodes** |
| 見積り小計 | 3,573,645,312 bytes = 3.328216552734375 GiB、0.25 GiB単位で上方丸めた差分も余裕に含める |
| 利用者向けfilesystem空き | **3,671,527,424 bytes = 3.4193763732910156 GiB** |
| free / available inode | 225,814,677 / 225,814,677 |
| 容量判定 | **不足：86,568,960 bytes = 82.55859375 MiB**。生の見積り小計は空き以下だが、余裕込み必要量には未達 |
| filesystem / mount | ext4、/dev/nvme2n1p1、/home、fragment/block 4 KiB、既存artifacts上位directoryをread-only観測 |
| quota | user/group/project各Q_GETFMTがESRCH、観測時点で非有効。quotaの未確認を「なし」と推定した判定ではない |
| 観測時刻 | 2026-10-06T09:44:16.890435+00:00 / 18:44:16.890453 JST |
| memory / pressure / OOM | effective available約983.119 GiB、6 worker要求72 GiB、host＋全可視非root PSI full avg10=0、3秒sampleのOOM増加0 |
| CPU候補観測 | 異なる6 physical cores、NUMA0、siblings最大busy約0.3345%、online/cpuset/process許容集合に含まれる。予約や専有保証ではない |
| 過去累積charge / 次段見積り / 合計 | 164,698,732 / 4,572,696,448 / 4,737,395,180 bytes。全campaign10 GiB charge cap内 |
| 過去wall / 残りwall | 2.679710050113499 s / 259197.3202899499 s、既存72 h累積上限を引継ぐ |
| 状態 | review=false、別CPU利用許可・最終review・明示launch未取得、起動0 |

容量解決は提案に留める。固定filesystemの利用者向けavailableを3.5 GiB以上・available inodesを560,000以上へ
確保できる状態が必要。現在との差は約82.6 MiBだが、観測後の変動と承認準備file分もあるためlaunch直前に再確認する。
自分の別データの整理、管理者による当該利用者向け容量確保などを利用者が判断する。
この準備では削除・移設・予約・output root変更・mount/quota設定変更を実行しない。
CPU条件や科学条件を変えて不足判定を回避しない。全campaign10 GiB全量の空きを新たな必須条件にはしない。

[plan](signal_compile_plan_v1.json)、[認可草案](authorization_draft_v1.json)、[review草案](stage_review_draft_v1.json)、
[metadata binding監査](binding_audit_v1.json)、[容量内訳](stage_storage_estimate_v1.json)、
[filesystem/quota観測](filesystem_quota_observation_v1.json)、[CPU/memory観測](resource_observation_v1.json)、
[容量判定](storage_verdict_v1.json)、[認可scope](authority_scope_v1.json)を一組で確認する。
読み取り方法・今回実行したmetadata scriptは[READ_ONLY_METHOD_v1.md](READ_ONLY_METHOD_v1.md)と
[preparation_metadata_only_v1.py](preparation_metadata_only_v1.py)を参照する。これらは固定science sourceではなく監査記録であり、
既存bundleを上書きして再実行するrunnerではない。

## source・入力・認可のbinding

science SOURCE_COMMIT：`049e69919af16ad29a67a217dc7a407d6b1754a6`。
science checkout：`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006`。
新資料専用branch：`track-a-h4-signal-compile-plan-review-20261006`。
準備起点commit：`cf72ab5d021d6f316fb37ccb13b96da740b43fbe`。
fixed run ID：`track-a-h4-geometry-v2-20261006-run02`。
fixed output root：`/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02`。

INPUT_BOUND planのSHA-256：`0d3d7fbda75bec6b4629b99fb783ae216006565c0ea3def3506e673ead54d641`。
plan fingerprint：`6d715ebc51a63bb7df1338b208155304e10fbabd65895a2b0005456837ea6db4`。
authorization digest：`e004de5d8a5baa4fb5d853b74be049663e20300b69edbec5a556c536b4f8c093`。
approved=false review digest：`ed420bc45c076ec58ecf34e26d3c002e65c88e2f21a357642cb530031eda2341`。
generation-freeze fingerprint：`0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7`。
generation-freeze file SHA-256：`75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69`。
fingerprintとfile SHAは別値であり、planは前者へ結合する。

run02入力生成planから変えたのはstage/binding/inputs/generation_freeze_digestの4項目のみ。
source19・checkout・run ID/output・218 templates・compiler/environmentは不変。
freeze JSONの6入力record（NPZ byte SHA、input/H/DF/state identities）を機械転記した。
今回はNPZ/NPYをopen/hash/loadしない。NPZの6 bytes SHAは先行完了監査で照合済みで、
launch直前と実runnerで改めて照合する。generation-freeze JSONとbyte-budget.journalだけはhandoff metadataとしてread-only確認した。
旧input authをsignal/compileに使う場合、false review、明示launchなしはproduction authorizeが拒否した。
schemaとsource/blob・contract・45 dependency・compiler metadata照合はPASS。positive実行許可を発行した検査ではない。

## 容量見積りの根拠と仮定

科学arrayを作らず、sourceの出版回数・JSON field数・文字幅と最大72時間から計算した。
全74,784 actual compile、cache reuseなし、全workerログ非空の保守ケースを使う。
毎秒の監視は残り72時間いっぱいの259,200 recordsまで含める。短期完了やログなしを前提にしない。
既存入力約78.5 MiBは既に現在のfilesystem使用量へ含まれ、新規必要量へ再加算しない。

| 新規output | 最大数 | fileあたり余裕 | block丸め後 |
|---|---:|---:|---:|
| wrapper checkpoint JSON | 74,784 | 4 KiB | 306,315,264 bytes |
| ledger delta JSON | 149,569 | 4 KiB | 612,634,624 bytes |
| workerログ | 74,784 | sourceのUTF-8 8 KiB上限 | 612,630,528 bytes |
| signal JSON、302表示点・paired32 metrics | 1,308 | 512 KiB | 685,768,704 bytes |
| 毎秒wall監視 | 259,200 | 128 bytes、4 KiB block丸め | 1,061,683,200 bytes |
| ledger.lock / map-complete | 1 / 1 | 0 / 4 KiB | 0 / 4,096 bytes |
| byte-budget journal増加 | 559,647出版 | sourceの128-byte固定row | 71,634,944 bytes |
| 同時temp/final余裕 | 最大8 publisher | 各512 KiB追加 | 4,194,304 bytes |
| directory entries/index | 559,659相当 | 256 bytes仮定 | 143,273,984 bytes |
| directory block / driver controlログ | 2 / 1 | 4 KiB / 8 MiB | 8,192 / 8,388,608 bytes |
| extent/ACL/filesystem journal等 | — | 64 MiB余裕 | 67,108,864 bytes |

sourceのOutputBudget.writeはpendingをfinalへhard-linkしてからunlinkするため同一data/inodeの二名が短時間存在する。
それでも最大8出版ぶんの追加temp full sizeを余裕として加えた。
workerログ上限・journal row・論理wrapper上限・ledger save回数・監視間隔はsourceに明示される。
JSON4 KiB/512 KiB、directory256 bytes、64 MiB等は容量planning上の保守的余裕で、sourceへ新たに追加した上限ではない。
20桁metric整数、有限float64をceilした最大309桁shot整数、最大幅floatを使ったメタデータだけのsignal specimenは263,198 bytes。
512 KiBはこれを上回る。科学値・実際の性能・実際の将来JSON sizeを測った結果ではない。
metricsは8 GiBのin-memory circuitから得るcounts/depth/sizeのため20桁を十分なplanning幅とする。
仮定を厳密なsource per-file保証と読み替えない。全campaignの実書込みcharge capは従来10 GiBのまま。

ext4 inode tableはformat時の固定record領域で、既にfilesystemのallocated blocksに計上されるため、
新fileごとの4 KiBを追加のtable予約として二重加算しない。
別途559,657個の新inode余裕を計数し560,000へ丸め、directory/extent/ACL等を上の余裕で含めた。
このfilesystem形式の判断は[Linux kernel ext4 inode documentation](https://www.kernel.org/doc/html/latest/filesystems/ext4/inodes.html)に基づく。
前stageで用いた4 KiB/inodeの追加planning余裕は旧監査履歴として保存する。旧判定値は書き換えない。
旧10 GiB全量空き確保案も保存し、今回のstage-specific見積りを新補足として追加する。

circuitは64 MiB IPC frameでメモリ中のowned pipeへ送られ、QPY/QASM/NPZ/SQLiteへ保存しない。
cache/reuse/registryはメモリ内で、永続checkpoint/ledgerは上記に含めた。
journalは旧chargeを累積し、wallもfreezeの既消費時間を累積する。消費予算のreset・予約・旧runのresumeはしない。

## launch直前の必須再確認

準備観測はlaunch許可や資源予約ではない。別認可・review・利用者の明示launchの後も、次が不合格なら起動しない。

1. source19のSOURCE blob/実checkout/source audit、plan/auth/reviewのfingerprint/digest、false草案と別の承認provenanceを照合する。
2. freeze JSONのfile SHA/fingerprint、6 NPZ bytes SHAをstreaming照合し、signal/ledger/mapやpendingが存在しないことを確認する。
   新科学arrayをpreflightでloadせず、record/ledger作成前の完全な入力境界を保つ。
3. CPU [3,5,6,7,8,9]のonline・cpuset・6 physical cores・NUMA・受動SMT sibling loadをfresh確認する。
   共有CPUの予約や他jobのaffinity変更をしない。mask0x3e8は新規own driverだけへ適用し、6 owned workersが継承する。
4. 実launch contextでprocess CPU集合が許可6件と一致することを確認する。
   effective available memory >=72 GiB、per driver/worker AS/RSS各8 GiB、headroom16 GiBを維持する。
   host PSI＋全可視非root PSI full avg10=0、全祖先OOMイベント増加0、namespace/mount visibility不明やinterface欠測はSTOP。
   metadata admission/pollは5秒以内のfreshnessを要求し、起動後も既存observerを使う。
5. fixed outputのfilesystemが同じこと、利用者向けavailable >=3.5 GiB、available inodes >=560,000、
   user/group/project quotaが確認可能で実効残量も十分であることを再検査する。
   root向けreserved free bytesを空きとして使わない。利用者のoutput rootについてwrite/searchとsymlink禁止も再確認する。
6. byte-budget journalが今回のprior chargeから変更されていないこと、10 GiB/72 hの残予算が足りること、
   別signal実行を既に行っていないことを確認する。単一one-shot launch、失敗時retry/resumeなし。
7. 不合格ならoutput/ledger/registryや科学jobを作らずSTOPし、検査理由を新control証跡へ保存する。

scopeは6凍結入力の全218 template mapのみ。map完成後はresearch_decision=null、next_stage_authorized=false、
mandatory_stop=trueの**MAP_COMPLETE_STOP**。H6/Track B/追加trajectory/anchor/長RPE/最終総costへ自動進行しない。
本stage実行に対する別のCPU利用許可・最終review・明示launchは未取得で、現在の容量条件も未成立である。

## 承認対象と未実行command

容量条件が成立した後の承認対象：

> CPU [3,5,6,7,8,9]、6 worker、提示したown-run限定条件で、run02の凍結済みH4の6入力について、
> 固定218 template/距離、32 paired random trajectories、最大74,784 logical wrappers/actual invocationsの
> signal・compiled cost mapを一度実行し、MAP_COMPLETE_STOPで停止する。
> launch直前のfresh CPU・memory・pressure/OOM・capacity検査が不合格なら起動しない。
> 入力再生成・source変更・追加条件・retry/resumeは行わない。

この文は承認案であり承認済みの記録ではない。review草案はfalseのまま保存する。
承認後はreviewをこの草案と別のcontrol fileへ実際の利用者指示・review provenanceとともに記録し、
plan/authのbindingを照合する。下のreview absolute pathは将来の承認記録用で、今はfileを作成していない。
次のcommandはfresh検査の実装・実施と全承認後だけの未実行例。現在は容量不足と未承認のため実行しない。

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
  /usr/bin/taskset 0x3e8 \
  /home/AbeHiromu/venvs/trotter-common/bin/python -P -B \
  '/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006/scripts/resource_applicability/run_h4_geometry_signal_compile.py' \
  --plan '/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-signal-compile-plan-review-20261006/artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/signal_compile_plan_v1.json' \
  --authorization '/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-signal-compile-plan-review-20261006/artifacts/resource_applicability/track_a_h4_signal_compile_plan_review/2026-10-06/authorization_draft_v1.json' \
  --review '/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-signal-compile-run02-review/approved_stage_review_v1.json' \
  --explicit-launch-signal-compile
```

共有環境変更・install/upgrade・GPU query/useを含めない。sourceのfixed PYTHON/THREAD_ENVを使う。
旧run01・旧source・旧bundle・run02入力/全journal/monitor証跡は保持する。
このbundleに科学NPZ・runtime・checkpoint・cacheはcommitしない。

## 準備検査と公開範囲

[preparation_checks_v1.json](preparation_checks_v1.json)と[artifact manifest](artifact_manifest_v1.json)は
ローカルのmetadata/gate/算術/link/不変性検査で、科学検証・immutable CIではない。
補助script作成時の2件の失敗は[preparation_failures_v1.txt](preparation_failures_v1.txt)へ保存した。
いずれも資料作成helper側の表記/JSON転記エラーで、source・科学処理・runtime書込みは伴わない。
commit対象はこの新bundle、関連文書、索引追記だけ。source・科学NPZ/runtime/checkpoint/cacheはこの新commitへ収録しない。
