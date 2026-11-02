# H4 run10 continuation：高速化版への引継ぎ処理を準備

SOURCE `11bba8f0669a264c0e1ef1d71b4d9d14bb394a12`、branch `track-a-h4-resume-preparation-20261011`。
起点はencoding speedup REVIEW `5706ab29e0ede62a079e471d5ebd1ae8e9765328`。
[closure・18回帰・binding草案・差分review・commit一覧](../../artifacts/resource_applicability/track_a_h4_resume_preparation/2026-10-11/README.md)が資料入口。
利用者の「では先に再開処理を先に完成させて」に従い、run10を動かしたままsourceを固定した。

## 完了分の引継ぎ

[continuation.py](../../src/trottertracks/resource_applicability/h4_geometry/continuation.py)はrun10のimmutable plan、連続ledger delta chain、reservation状態、外部completion digest、全file bytes/SHAを照合する。
元STARTED証跡10580 bytes/SHAと、生存中に捕捉したCLOCK証跡586 bytes/SHAを固定。
[receipt runner](../../scripts/resource_applicability/run_h4_resume_receipt.py)の`capture-stop`は停止操作をせず、run10 native6 identitiesの退出を2秒離した2回の監査で確認する。
`receipt`はその証拠が揃った後だけbyte-only inventoryを生成する。
consumer側でもSTARTED/CLOCK、one-shot driver PID/UID/run/SOURCE/review、bootとwall origin、observer traceの最新wall/monotonicへ再結合する。
未記録の過去exit code・正確なexit時刻はnullのまま保持する。

[Ledger](../../src/trottertracks/resource_applicability/h4_geometry/ledger.py)はacknowledged COMPLETEだけを新規outputへコピーし、旧recordのbytes/digestを保持する。
cache reuse ownerも先に検証・取り込み、最終的に全logical streamへ現れることを確認する。
[producer](../../src/trottertracks/resource_applicability/h4_geometry/execution.py)は取り込めるwrapperのevents/build/serialize/compileを省略する。
片方のaxisだけ欠けた場合はそのaxisだけcompileする。
全wrapperが完了し、対応signalのidentity・paired metrics/costが一致したcandidateはsignalも取り込み、SCF/input生成やsignal再計算を行わない。
RESERVED、ledger未完了のorphan record、pending fileは再利用しない。証跡を元directoryへ残し、費用を返却せず未完了分を新しく計算する。

## seedとSOURCEの分離

run10 dataset SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`をcandidate identity・wrapper key・trajectory seedに保持する。
新しい実行SOURCEは`11bba8f0669a264c0e1ef1d71b4d9d14bb394a12`としてsource/profile/plan/auth/reviewへ固定し、新規完成recordには`execution_source_commit`を別に保存する。
新fresh campaignの通常seed規則はその実行SOURCEを使う。continuationは別の明示認可schemaで、同じdatasetの継続として扱う。
科学H4/PF/RTE条件、structured Gaussian gate sequence、compiler options、環境profile、凍結入力、資源capsはrun10のbindingと一致が必須。
旧dense系列や他runのpartialへ範囲を広げない。
前回のencoding人工fixture約2.29倍は引き継ぐ実装根拠であり、実map全体の高速化率は未測定。

## gateと予算

[continuation_binding.py](../../src/trottertracks/resource_applicability/h4_geometry/continuation_binding.py)と[closed schema](../../schemas/h4_newhost_launch_v2.json)はparent receipt・identity SOURCE・import件数・実execution SOURCE・承認authorityを結合する。
新output/controlは互い、旧output/control、input/stop/source/receiptとのancestor/descendant overlapを拒否する。元runはread-only、新runはexclusive one-shot。
CPU候補はworker4の2/4/5/6、driver16、observer18。実行許可とは分ける。

carryは停止後の実ledger reservation全数・byte journal全額・native退出監査までの保守的wallから確定する。
未完了で消費した呼出も引き継ぎ、次のactual IDをcarryの続きから採番する。
保証用の上限は`74784 + carry.actual_invocations - imported_wrappers <= 74805`。保証できなければ起動しない。
17GiB/74805/72h、worker32GiB/driver8GiB、observer AS256MiB/RSS64MiB、内部thread1・5秒監視・pressure1/5/30/152.25は変更しない。
72hはrun10と再開の累積wallとして保持する。容量は旧full-map形式の新規分見積りにcarryを加え、fresh filesystem/inode/quotaで確認する。
SOURCE/profile/input、fresh CPU/memory/pressure/OOM、未使用output/control、one-shotを通過した場合だけ将来の別launchへ接続する。

## 限定検証と現状

[事前計画](../../artifacts/resource_applicability/track_a_h4_resume_preparation/2026-10-11/test_plan_v1.json)は単一test process、thread1、AS256MiB/RSS128MiB、wall45秒、output4MiB。
[test runner](../../scripts/resource_applicability/run_h4_continuation_tests.py)と[test](../../tests/tracks/resource_applicability/test_h4_continuation.py)で18 PASS、failure/error/skip0、wall0.649秒、kernel VmHWM30408704 bytes。
ru_maxrss371695616 bytesも別に保存し、今回processのRSS判定にはkernel VmHWMを用いた。
完了pairはbuild/fingerprint/compile0、partial pairは欠けたaxisのみ1回のmock compile、cache owner、orphan/RESERVED、mutation/missing/ledger gap、seed、native/boot/wall、root交差、ordered publication、first failureと消費保持を検証。
差分reviewのP1/P2/P2を修正し、最終`PASS_CONTINUATION_PREPARATION_DELTA_ONLY`、残blocking0。review担当の追加tests0。
人工compile/実worker/科学array/GPU/affinity/新launch0、private venv・共有環境・他job変更0、作業はhome配下。

SOURCE closure85は旧80の5変更/75不変＋新2module/2runner/1test。
実run10は継続中なので、最終停止proof・carry・receiptは未確定。binding草案のcarryはnullで、0へresetした値ではない。
新branchのallowed_cpus=[]、approved=false、runtime_authorization=false、sealed=false、absolute_launch_command=null。
今回の準備完了を、実receipt受領・technical sealや起動成功とは扱わない。
移行実施時にown run10停止→native proof/receipt→実carryと新SOURCEのbinding/reseal→fresh gateの順で接続する。

全体研究の利用者指定公開起点は`track-a-ax2b-h4-post-review-20261010` / `a87cf25548a3b93262027ce780317872d7c4e883`。
本mapは別途認可されたsix-geometry追加datasetで、AX2b accuracy/u/NG certificationやH6認可へ読み替えない。
