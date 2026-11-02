# H4本計算前準備・最終承認入口（2026-10-08 JST）

H4_PRELAUNCH_IMPLEMENTATION_FIXED_INPUTS_AND_FINAL_APPROVAL_REQUIRED_STOP。準備実装・profile固定・read-only資源監査・限定人工検証は完了。本計算・実worker・GPU・追加transpileは0。
起点origin `3e0494e1e8a0c5ea72ef0bdc1cfc9b08a542edcc` は指定SHAと一致した。branch `track-a-h4-prelaunch-preparation-20261007`、SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`。
SOURCE32件（31 Python＋schema1）をactual blob/bytesで固定し、別REVIEW_BUNDLEへprofile・草案・監査・資料を固定する。
REVIEWのactual SHA/remote SHAはcommit/push後の外部receiptと最終報告へ記録する。自分自身のSHAをbundle内へ捏造しない。

資料は [bundle入口](../../artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/README.md)、[SOURCE32 closure](../../artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/source_freeze_v2.json)、
[具体的な最終承認案](../../artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/final_approval_proposal_v2.json)、[commit対象](../../artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/commit_inventory_v2.json)から読む。
既存worktree/資料/証拠を保持し、新作業・log・temporary fileは全てhome内。共有環境・既存venv・他job・quota・affinityの変更0。

## 一括承認案

次の範囲だけを提案する：H4 signal/compile mapを一度実行し、MAP_COMPLETE_STOPまたはfail-closed STOP後に停止。自動retry/resume、入力再生成、条件追加、次campaignへ進まない。

| 項目 | 提案と現在の状態 |
|---|---|
| environment/compiler | 既存private venvを変更せず新host固有の本番候補として採用。旧compiler output同一性は未検証 |
| SOURCE | `ad57d1639133f7158cce58d767b8e0aa179bf044`。source/schema32 closure、将来seed再結合、旧random/partial/cacheと混合しない |
| workers/CPU | workers12：`[2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]`、driver `[16]`、observer `[18]`。14 distinct physical cores、提案で利用許可ではない |
| observer | 独立stdlib process1、AS256MiB/RSS64MiB。driver/worker AS/RSS各8GiB、headroom16GiB、monitor5秒は維持。12-worker admission120.25GiB |
| actual cap | 現在74784、carry20、残74764。全74784 logical mapの保証には最小+20、累積74804への別契約変更を提案。未承認・未適用 |
| wall/output | carry5466.188392877579秒/165214360 bytes保持。72h/10GiBは不変。trace/control/journal/tempを計上 |
| storage | 追加output5GiB/301000 inodes。copy前の暫定全量6GiB/305096 inodes（input/control margin1GiB）。実受領byte/inode inventoryで置換 |
| inputs/stop proof | NPZ6/freeze・旧run05 journal/log/controlは未受領。streaming SHA照合0。既存SSH認証copy失敗。metadataだけをSTOP proofとみなさない |
| approval flags | allowed_cpus=[]、approved=false、runtime_authorization=false、sealed=false。command未実行 |

科学条件はH4 linear neutral singlet、STO-3G、DF12、4 spatial/8 system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE。
6距離0.70/0.80/0.90/1.10/1.40/1.60Å、L_D0/3/4/5/6/9/12、q1/2/4/8、delta0.8/0.4/0.2/0.1、各218 templates/32 paired trajectories、1308 signals/74784 logical wrappersを維持する。
compiler15 options（opt1、seed_transpiler17、basis rz/sx/x/cx、num_processes1等）とscience algorithmsは不変。

## 環境候補の採用理由と限界

Python `/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python`。candidate environment `6abf53c37c82ec389d2a5d1f3aa7787a86db379c7e6312ea5f6836c7cc05cb35`、compiler `fbe36b72d4c36f1b9d9dbe15b70ef09ef88376f3500b67694e80b51ac0517476`。
core qiskit1.3.0/numpy1.26.4/scipy1.14.1/openfermion1.6.1/pyscf2.7.0とinstalled11 source hashは旧参照と一致。
18 version差、旧normalized参照対new raw RECORD 45差を保存。
同じread_textの改行正規化方式では22差で、残23件の同hashも完全binary/output equivalenceの証明ではない。
rustworkx0.18.1等の差を新compiler layerとして固定し、旧costと同一環境の結果として混ぜない。version差を理由に環境方針を再質問しない。
install/upgrade/新venv/既存設定変更0。-P -B、process内thread1/Qiskit num_processes1を維持。

## source/schema・起動gate

newhost v2 schemaはsource/profile/inputs/output/plan/auth/reviewとobserver追加role、CPU配置、累積予算を結合する。
false flags/未seal/未承認observer/未承認+20変更はfilesystem/science/affinity/spawn前のpure gateで拒否する。
actual SOURCE32 blobs、loaded module位置、installed source/RECORD・compiler defaults/plugins/options、six inputs/freezeとnative stop proofを確認。
private absolute roots、exclusive one-shot marker、入力と新outputの分離、旧data再生成/partial reuse禁止を守る。
自分のdriver/worker/observerだけに、独立承認されたCPU maskを将来設定する実装があるが、今回の実affinity変更は0（testsはmock）。tasksetは実行しない。
新observerのAS/RSS、traceとdriver first-stop file、control log、起動/終了wall費用を先に予約する。
driver5秒watch、observer独立観測、UID/PID/starttime/parent/pidfdと元driver退出証明後のreparented worker停止を保持。
bounded driver logが満杯でもcleanup/interruptを実行。新host workerはdisk/cache/temp書込禁止、driverは自分のoutput/control外の書込禁止。
TMPDIR/tempfile問い合わせはowned home pathへ向け、unbudgeted package temporary writeはSTOP。OutputBudgetのpending/finalは予約済みの相対dir_fd publish contextで許容する。

## static budget と容量根拠

random194×32×2＋baseline24×2＝12464/点、6点74784 logical wrappers。carry込みzero reuse worst caseは74804 actual calls。
合法cache reuseは同じgeometry/template/axisの数値回路が一致し、既にCOMPLETEのownerがある場合だけ。
random K2/K4の実trajectory/circuitを作らず、少なくとも20件の節約を保証できる静的根拠はない。節約を見込んで上限をすり抜けない。
最小変更は累積actual capだけ74784→74804。plan capsを変え、明示amendment authorityとplan/auth/review digestを再固定する必要がある。
現在cap74784を維持したまま、未承認amendmentをgateで拒否する。失敗予約の返却・carry reset0。

保存formatはrecord/delta各4096B、worker log8192B、signal524288B、map/launch-stop4096Bをhard gate化。
record74784/delta149569/worker log74784/signal1308、72h/1秒observerのsingle append trace（最大8192B/record）を見積もる。
旧one-file-per-tick wall形式を除き、journal128B/row、directory256B/entry、14 concurrent temp publishers×512KiB、block4096B、metadata margin64MiB、control log8MiBを含む。
physical bound 4539042688B、rounded stage5GiB/301000 inodes。
carry/observer/control/全部のtemp+final/journal chargeを含むbound 8902109720B（約8.2907GiB）、10GiB margin 1835308520B。
byte-budget journalは一度読み、その後single-writer cached sum/offsetで追加を計上。foreign size change/short write/negative refund行をSTOPし、全量O(N²) rereadを避ける。
receiptコピーの実サイズは未受領のため未確認。暫定1GiB/4096 inodesの別physical marginを置き、受領後inventoryで更新する（science output capを増やさない）。

read-only観測：scheduler/online0–63、memory available約465.97GiB、filesystem nonroot available約414.51GiB、free inodes 220504177。
kernel quotactl_fd GETFMT（current x86_64 header syscall443）/GETQUOTAとFSGETXATTRのみを使い、user30038/group30038/project0 quotaは現在DISABLED（ESRCH）。setter/on/off/syncなし。
可視cgroup祖先・host/root PSI/OOMを観測し、歴史的OOM値を新しいOOMイベントと混同しない。これらは予約/将来の容量保証ではない。

## 人工検証と未検証

new binding/cleanup25＋監視・serialization回帰32＝57 PASS、fail/error/skip0。初回charge検査でpayload sizeとjournal sizeの混同を検出し修正、失敗log保持。
12 workersはfixture/mock。実動cleanupだけtest/subreaper＋minimal driver＋stdlib observer＋minimal sleep workerの最大4 process、AS/RSS/wall/output planを事前固定。
driver退出後、登録workerの親変更を許容する退出証明、worker/observer停止とtestによるreapを実確認。own test subreaperは復元、他jobに作用しない。
実science/input array load/再生成/production workers/追加transpile/GPU0。旧synthetic28/64・benchmark128保持、再実行0。
人工PASSは旧compiler完全同一性/production成功ではない。real SCF/DF/physical inputs/operators、74784/72h/12-real-worker性能、power-loss durability、uninterruptible kernel I/O＋driver stallのhard realtime停止は未検証。
actual影響を持つCPU affinityはmockだけ、observer production roleは起動しない。schema positive testsはメモリ内模擬承認＋stub scienceで、実行認可を発行しない。

## fresh launch 条件と未実行absolute command

入力/freeze/停止proofのstreaming bytes SHA、source/profile bytes/digests、独立final review、明示user launchが揃った後だけ起動gateを通す。
3秒の受動CPU sampleで選択core busy<=20%、別physical cores、online/scheduler intersection、memory/FS/quota/inodes観測5秒以内、pressure/OOM安定、observer-inclusive120.25GiB admissionを再確認。
source/profile checks後のstartup OOM/cgroup変更とstalenessも拒否。既存output/one-shot markerがあれば再実行しない。
承認後にreceiptとcap/permission変更を反映して再seal・fingerprintを結合する。今の草案へ以下を実行してもfalse flagsで拒否される。今回未実行。

```bash
env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 QISKIT_PARALLEL=false /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -P -B /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-prelaunch-preparation-20261007/scripts/resource_applicability/run_h4_geometry_signal_compile.py --plan /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-prelaunch-preparation-20261007/artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/plan_draft_v2.json --authorization /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-prelaunch-preparation-20261007/artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/authorization_draft_v2.json --review /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-prelaunch-preparation-20261007/artifacts/resource_applicability/track_a_h4_prelaunch_preparation/2026-10-08/review_draft_v2.json --explicit-launch-signal-compile
```

残る承認は candidate environment/compiler採用、12worker＋driver/observer CPU/role、最小actual+20変更、receipt/stop proof完成とplan再seal、独立最終review、利用者の明示launch。
承認範囲は冒頭の一回限りmapだけ。完了/FAIL_CLOSED_STOP後にSTOPし、自動retry/研究判断/条件追加へ進まない。


## 2026-10-08 H4受領監査・独立最終review TECHNICAL FAIL

[統合入口](track_a_h4_receipt_final_review_20261008.md)。origin79827016/SOURCEad57d163照合、source32不変、read-only live profile/資源/quotaを確認。
凍結NPZ6/freeze/native stop/controlは未受領、producer bytes/SHA manifestを含む最小転送手順を具体化。科学array読込/再生成0。
独立reviewはpidfd送信時ESRCH競合で後続cleanupが中断するP1を純mock再現しTECHNICAL FAIL。57人工PASSはこの競合を覆わない。
未受領NOT_EVALUABLE、CPU/env/observer/74804案のUNAPPROVEDと技術FAILを区別。sourceは修正せず再sealなし、flags false、carry20/165214360/5466.188392877579を保持し本計算STOP。
