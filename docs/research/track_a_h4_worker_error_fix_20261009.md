# H4 worker失敗の原因保存・cleanup競合修正

利用者の「エラーの原因をまずつぶして」に基づく限定修正。source/testsと軽量監査を固定し、本計算を起動しない。
入口：[review bundle](../../artifacts/resource_applicability/track_a_h4_worker_error_fix/2026-10-09/README.md)。

branchは`track-a-h4-worker-error-fix-20261009`、SOURCEは`d3bb388401bfd25d31c53e7f785b533a7753ddb3`。
起点STOP commitは`ab86c31155c600f24552766ccd9d1505c5fb1de4`、修正前SOURCEは`a7b617600cd7063f7870f2059d5694ef00283f0e`。
独立worktreeと全log・人工一時fileはhome配下。旧run/one-shot/source/bundle/入力・停止証拠を保持する。

## 確認した不具合と修正

旧workerはjobのunpickleとschema検査をdispatchの例外捕捉外で実行し、stderrもDEVNULLへ破棄していた。
dispatch失敗時も、親のpipe threadが元例外をメモリに保持した直後にworker群を停止していた。
driverが次回路のserialization中なら、独立observerが先にprocess退出を観測してdriverをSIGTERMし、
元例外が保存されない経路になる。過去run02には元例外が残っていないため、元のcompiler/IPC原因は特定できない。

- workerのbootstrap/permit復元/checkout/job復元/schema/dispatch/response/GCを同じ例外捕捉に含めた。
  bounded traceback・worker phaseをprivate pipeへ返し、親のSTOP/EOFまで待機する。失敗後の追加dispatch/restartはしない。
- 成功応答はGC完了後に送る。GC失敗後のworkerをavailable queueへ戻す早すぎる成功通知を防いだ。
- 最初の親例外、worker PID、job serial、観測時exit code、phase、logとtracebackを
  `worker-log-first-stop.txt`へexclusive・fsyncで保存する。空log、pickle/EOF/schema失敗も対象、8KiB上限。
- 独立observerにも原因を渡し、first-stop fsync後にown pidfdだけで停止する。
  `pool.failure`を公開するのは保存・report後。driverの`pulse()`が保存前のfailureを読みcleanupする競合も修正した。
- native退出/所有喪失時の観測例外には、対象の登録済みPID/start/parent/UIDを添付する。
  native crash/SIGKILLのPython tracebackは取得できるとは限らない。
- 8KiBのfirst-stop fileを容量見積りへ別枠で追加。保守charge16,512B増、既存8KiB per-file capと13GiB総capは不変。

5秒監視、1秒period、fail-closed、所有確認、ESRCHだけ既退出扱い、wait/reap/FD/pipe終了を保持する。
科学条件・compiler options・signal/builder/identity/ledger/parallel/executionはbyte-identical。

## 限定検証と限界

最新39人工tests PASS、fail/error/skip0。wall0.088208秒、peak RSS98,566,144B。
worker12はmock、実child/worker・transpile・科学array読込・affinity/GPU操作0。
通常は単一test process。publication競合だけmainとEvent制御fixture threadの2threadsを使用し、
内部数値thread1、AS2GiB/RSS512MiB/wall120秒/output8MiB、thread join2秒を事前planへ固定した。

検査はempty-logエラー、pickle/EOF/schema、checkout/GC、出版失敗、first原因保持、次submit拒否、
driver pulseとのpublication競合、observerの保存→signal順、未知PID拒否、ESRCH/非ESRCH、二重cleanupとreap/pipe/FD。
独立担当のreviewは`DIAGNOSTIC_PATH_IMPLEMENTATION_PASS_NO_RUNTIME_AUTHORIZATION`、blocking実装所見なし。
publication競合を別Event検査で確認し、closure/profile/byte-only入力/STOP・carry bindingも照合した。
結果はbundleの`independent_review_v1.json`、SHA256 `b2d177957c5aa0fe5358044724952b2cf66c98b0cb72ae6e9c2f971964aa5dd9`。

最初のB0-rank3-q1と同規模の人工fixture（complex256×256の4 basis、8-system+ancilla、cosine/sine）は、
約11.8MBのpickle IPC前後でnumerical digest一致。cold Qiskit importは既存worker write guard下でPASS。
この二つの限定probeで過去の科学入力を使わず、追加transpileもしていない。
実Gaussian/実compiler成功/元production例外の原因特定/production性能・旧compiler完全同一性は未検証。

最初のprobeのDEVNULL許可漏れ、最初のsuiteのexclusive-directory/observer identity fixture不足を修正し、
失敗原本をprivate evidenceに保持した。これらのfixture失敗をproduction原因と扱わない。
旧benchmark128・旧synthetic28/64の再実行・消費変更0。

## 固定・binding・再起動条件

source closure41、旧closure39から変更6（runtime3・旧test1・新runner/test2）。actual git blob/SHAと差分理由を固定した。
source/testsをSOURCE commitsへ、sourceを変更しないprofile/input/carry参照・監査・reviewを別REVIEW_BUNDLEへ固定する。
新準備bindingは`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`sealed=false`、command=null。
旧attemptの実行認可は当時のsnapshotとして保持し、新SOURCEの再実行許可へ転用しない。

凍結6入力の受領済みbindingと既存environment/compiler/library-cache profileを参照する。再照合campaign・array読込0。
既存venv、共有設定、他jobを変更しない。環境の旧18 version差・45 raw RECORD差、旧compiler output完全同一性未検証は保持する。

carryは21 consumed/reserved invocations /8,692,723,164B /保守的wall upper5,766.582514658794秒。
wall upperはpost-stop監査待ちを含み、正確な終了時刻やcompiler開始を補完しない。失敗分返却/reset0、残actual74,783。
新diagnostic込みfresh全map worstは17,429,694,796B（約16.232668GiB）、actual最小cap74,805案。
現13GiB/74,804は変更しない。次の起動を承認済みとは扱わない。

再起動する場合は、原因が科学compiler内部にあるかの限定確認範囲、累積予算改定、最新SOURCEへのseed/binding、
未使用run/output/control/one-shot、plan/auth/reviewとfresh resource gateを別途確定する必要がある。
旧random/partial/cacheを混合しない。一度のmap後STOP・自動retryなし・次stageなしを保持する。

## 公開

commit対象はsource/tests6 paths、軽量bundleと本報告・索引だけ。NPZ・実runtime/checkpoint/cache・credential・内部SSH情報は対象外。
non-force pushの認証が失敗した場合、設定を変更せず次を利用者へ提示する。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-worker-error-fix-20261009 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-worker-error-fix-20261009
```

本計算再起動0。診断経路の人工合格と元production原因の未特定を区別してSTOPする。
