# H4 run03：人工compileを省き、本計算で問題を診断する準備

利用者の指示に従い、追加の人工回路compile検査を省く。本計算で問題が発生した場合は原因を保存し、
own-runをfail-closed STOPして、その証拠から修正する。元run02の科学compiler/IPC原因は未特定で、完走を保証しない。
今回の資料入口は本書。詳細は[固定bundle](../../artifacts/resource_applicability/track_a_h4_production_run03/2026-10-09/README.md)を参照する。

## 固定sourceと次のrun

branch：`track-a-h4-production-run03-20261009`。
SOURCE：`6e68fd9bcc68e788db6f5d43eaa6a03866e53d3b`、closure44。
起点REVIEW：`00c6ad5fa7c7affc6c6d702af54ac1de2b275d3d`、旧SOURCE：`d3bb388401bfd25d31c53e7f785b533a7753ddb3`。
旧worktree/原本/停止証拠/one-shotは保持する。
次のrun IDは`h4-newhost-signal-compile-20261009-run03`。
outputは`/home/AbeHiromu/projects/h4-handoff-evidence/20261009/h4-production-run03-20261009/h4-newhost-signal-compile-20261009-run03`、controlは`/home/AbeHiromu/projects/h4-handoff-evidence/20261009/h4-production-run03-20261009/control/h4-newhost-signal-compile-20261009-run03`。両方未使用、one-shot未作成。
新source seedへ結合し、旧random/partial/checkpoint/cacheを混合しない。

source変更はlaunch binding、carryを引数に取る静的invocation集計、run02 native STOP validator、認可schemaと純粋tests。
科学関数・signal/build/compile/serialization/worker/observer/ledgerの実装とcompiler optionsは旧SOURCEからbyte-identical。
診断fixの旧39件・旧synthetic28/64・benchmark128、proof受領campaignは再実行しない。

## 環境・入力・STOPとcarry

既存private venv、environment/compiler/library-cache profileは採用済み条件を保持し、install/upgrade/設定変更0。
Python `-P -B`、数値内部thread1、Qiskit num_processes1。
18 version差・45 raw RECORD差（normalized22差/23一致）、旧compiler output完全同一性未検証を維持する。
6凍結NPZとgeneration-freezeは既存受領場所を使用し、streaming byte SHAとmetadataだけを照合。科学array読込/入力再生成0。
旧host/run05のnative13identity proof、run01の14identity STOP、run02の14identity×2回ABSENTを保持し、新carryへ結合。
run02 receipt10034B/SHA、raw7filesのbytes/SHA、observer最後のidentityと2回native audit、first STOP、
journal7rowsの累積sum、ledger chainと`science-000021` RESERVED、消費済みone-shotを照合した。
exact終了時刻/科学compiler開始は不明のまま。post-stop監査待ちを含む記録済みwall upperをそのまま採用する。

carryは21 actual消費/予約、8,692,723,164B charge、5,766.582514658794秒 upper。
前回の失敗費用と未完了reservationを返却・resetしない。現actual上限74,804の残りは74,783。

## 一括の追加承認案

本計算を進める利用者指示を受領済み。以前採用済みのenvironment/compiler・CPU・observerを再承認事項にしない。
今回必要な新しい承認は累積上限2件だけ。

| 条件 | 現承認 | 次の全map必要量 | 追加承認案 |
|---|---:|---:|---:|
| 累積output charge | 13GiB | 17,429,694,796B（16.232668GiB） | 17GiB（18,253,611,008B） |
| 累積actual invocations | 74,804 | carry21＋新map74,784＝74,805 | 74,805 |

17GiBは必要量を満たす最小の整数GiB、余裕823,916,212B。静的な合法cache reuse節約保証0。
output chargeは保存済みpayloadサイズではなく、temp/final/journal/controlとfull72h observer traceの前払いreserveを含む。
observer reserveは1 run当たり4,246,814,848Bで、短時間STOPでも返却しない現契約を維持する。
その他72h、各driver/worker AS/RSS8GiB、headroom16GiB、monitor5秒、科学条件・compiler optionsは変更しない。
17GiB/74,805は未承認proposal。draft `approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`sealed=false`、absolute command=null。
closed gateの実装は、この2件の明示改定と最終bindingなしには起動しない。

## CPU・observer・容量と限定確認

採用済みroleはworker12 CPU `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18。
own-run限定affinity、observer AS256MiB/RSS64MiB、admission120.25GiBを保持する。
この準備ではaffinity/taskset/実worker/runnerを起動していない。
容量はrecord74,784、ledger149,569、各worker log、signal1,308、first STOP8KiB、full72h observer trace、
control runner log8MiB、journal、14同時temp・block丸め・directory/metadata margin・inodeを含む。
新run追加physical必要5GiB/301,000 inodes、過去runと受領入力原本は保持する。
read-only時点でavailable memory 465.842GiB、filesystem 411.110GiB、
inodes 220,452,577、quota KNOWN。
CPU roleは異なる14physical core、受動負荷20%以下・PSI/OOM/namespace/cgroup欠測なしの候補gate合格。
このsnapshotを次のlaunch直前fresh検査へ流用しない。

限定32件PASS、failure/error/skip0、wall 2.401002秒、peak RSS 29,360,128B。
1 test process、内部thread1、AS2GiB/RSS512MiB/wall120秒/output8MiB、実child0。
新carry・reset拒否・旧actual不足・17GiB/74,805の明示認可・draft拒否・quota/capacity/fresh gate・実OutputBudget予約拒否時のcharge保持、
STOP metadata改変拒否を検査した。回路build/人工compile/transpile/科学array/affinity/GPU/本計算0。
最初の32件中1件は旧10GiB pipeline fixtureへ新carryを適用してreserve STOPとなった。fixtureを17GiB案へ合わせ、初回logを保持して再検証32PASS。
独立最終reviewは`TECHNICAL_CANDIDATE_PASS_PENDING_ONLY_TWO_BUDGET_AMENDMENTS`、blocking implementation findingsなし。
結果はbundleの`independent_run03_review_v1.json`、SHA256 `e5de8edcb5db5f89a6a6f4781d6ad974fa8dac1c1dc08549eac95bc147e9ce11`。
過去のcompilerエラーが未特定であることは、利用者が受容した未検証事項として記録し、追加人工compileの条件にはしない。

## 予算承認後に行う一度の起動

利用者の上限2件の承認をplan/authへ反映し、SOURCE/profile/input/carryと新plan/auth/review digest・sealを再結合する。
独立最終整合reviewを反映し、sourceを変更しないlaunch artifactを固定する。
直前にSOURCE44・profile/input/carry・CPU/memory/PSI/OOM・filesystem/block/inode/quota、未使用output/control/one-shotを確認。
全gate合格なら追加の手順承認待ちで止めず、H4 signal/compile mapを一度起動し、run ID/PID/log入口を報告する。
完了またはfail-closed STOP後に終了、自動retry/次stage/入力再生成/旧partial-cache再利用なし。
問題が出ればbounded worker tracebackとphase、first STOP、interval/duration/staleness、own native cleanup証拠を残して修正する。
source変更や再実行が必要なら、そのattemptの累積消費を保持した次のbindingを作る。共有環境/venv/他jobを変更しない。

## 固定・公開

SOURCE commit対象8件、REVIEWは本書・関連索引/概要/note/manifestと軽量bundleだけ。
NPZ、raw runtime/checkpoint/cache/test log、credentials、内部SSH情報、home utilityはcommitしない。
origin non-force pushが認証で失敗した場合、設定を変えず次のcommandを提示する。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-production-run03-20261009 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-production-run03-20261009
```

この固定準備時点で本計算起動0。上限2件の明示改定待ち。
