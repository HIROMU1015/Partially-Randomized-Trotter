# H4 geometry compile並列制御修正・科学未実行

2026-10-06 JST。`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`でSTOPする。
今回の認可はcompile制御修正、分子データを使わない合成検証、二つのcommit、non-force pushのみ。
入力生成専用plan/authorizationの作成、本計算、実worker/production runnerの起動、H6、Track Bは認可されていない。

入口は[新bundle](../../artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/README.md)。
起点は旧REVIEW_BUNDLE_COMMIT `4d5eba454dda06bc2735730cf7c3f132e456db29`、契約baseは
`b662dbd72e49fa713a25c716f323843e547e973b`、旧SOURCE_COMMITは
`d6c7afd02dc0603216982a3cd4ea71b3d047dd8d`。旧source audit SHA-256は
`8c7d68c2f50753e22152079206e6d9a8e6e174a87f4b940b4cfed218c92e5eef`。
旧source/review bundleとsource auditは保存し、新sourceの監査へ流用しない。

## 問題と修正

旧 `execution.py` の `run.wait(run.pool.submit(_compile_worker,circuit,options))` は、
各wrapperが完了してから次を投入するため、compileが実質1件ずつ進んでいた。
新 `parallel.py::compile_wrappers` は、admissionで決まった `run.workers` 件を上限に投入し、
FIRST_COMPLETEDで回収して次を投入する。常に12とは仮定しない。
完了済みで未回収のfutureも保守的に枠へ数えるため、未完了投入が上限を超えない。

`candidate_wrapper_jobs` はtrajectory順に一つのevolutionを生成し、同じevolutionからcosine/sineをyieldする。
driverは枠が空くまでgeneratorを進めない。全74,784 wrapperや全candidate回路を先に作らない。
保持する回路は投入枠と現在のevolution/axis分だけで、完了したfutureから回路を回収・保持しない。
各candidateの最大64件の結果はlogical positionで並べ、従来と同じtrajectory→cosine/sine順に集計する。
sampling・normalization・shot・covariance・metric計算は変更しない。

`workers.py::OwnedPool` はround-robin固定先ではなく、応答を終えたworkerのindexをavailable queueへ戻す。
完了順が変わっても次のjobをbusy workerの後ろへ溜めず、空いたworkerへ渡す。
spawn数、内部numerical thread/process各1、匿名pipe、ownership、8GiB制限は従来どおり。

## 処理中ownerとdurable accounting

同じgeometry・candidate・axis・数値fingerprintの処理中ownerを追跡する。
後続logical requestは独立expected identityをregisterし、full input/H/DF/state/source/compiler/environment等を照合するが、
予約・compile・cache completionを行わずownerの結果を待つ。
ownerが成功し、record/ledgerがCOMPLETEとして永続化され、独立registry・外部digestを用いたread検査を通った後、
既存 `Ledger.complete(...,owner_key=...)` の検査で後続をcache completionへ記録する。

ownerは非cache COMPLETEに限定する。RESERVED、失敗owner、欠測/digest不一致、cache chain、self-link、
cross geometry/cell/axisを拒否する。後続もsample weight1/32を保ち、logical wrapper数は減らさない。
baselineはseed/index=null、weight1の既存wireをそのまま使う。
invocationは実pool.submitの**前**にfsync付きで予約し、logical wrappersとactual invocationsを別に監査する。
ledger更新はdriverのみ。正常監査では登録済みlogical requestとCOMPLETE集合の一致も要求する。

compile例外・worker死亡・monitor failureが観測されたら、追加生成/投入を止め、own childrenだけを停止する。
poolのfailure latchにより後続submitとqueued dispatchも拒否する。
未解決予約、既に成功した部分record、delta ledgerを保持し、正常map completionを作らない。
retry/resume、worker補充、予約払い戻しは行わない。

## 不変条件とsource gate

H4 linear/STO-3G、DF requested/returned fragments12、8 system＋ancilla index8、T=0.8、二次DF-prefix PF/canonical finite-RTE、delta=T/q。
6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、各218 templates（B0=20/B1=4/B2=145/B3=49）、
random194各32 trajectories、signals1,308、logical wrappers/actual science invocation cap74,784を維持する。
master seedとsource/input-bound seed導出規則、compiler identity、Gaussian synthesis、wrapper/serializer意味論は不変。
新SOURCE_COMMITへのbindingで将来のseed identityが変わることを、seed規則の変更や旧結果のコピーとは扱わない。

最大12 workers、AS各8GiB、headroom16GiB、admission8+8w+16GiB、累積wall72h、output10GiB、
fixed run ID/artifact anchor/output root、二段階認可とmandatory STOPは不変。
入力/seed/signal/circuit/resource等の変更対象外Python9件とnamespace parent2件は旧sourceとbyte-identical。
変更は既存Python7件（execution/gates/ledger/workers、専用guard/auditor/tests）＋新parallel.pyだけ。

`gates.SOURCE_AUDIT` は新bundleの `source_freeze_v1.json` を指す。
未来のplanはnew SOURCE_COMMIT、17件のsource＋parent2件、new audit SHA-256、actual loaded checkoutを結合する必要がある。
旧auditの使い回しはsource/hash/path gateで受け付けない。今回はplan/authorizationを作らない。
新source checkoutは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-parallel-source-20261006`。
artifact anchorは契約の `/home/AbeHiromu/projects/partially-randomized-trotter`、future output rootも契約のfixed run01。
本番output/registryはresolve/stat/作成していない。

## 実行済み合成検査

既存94件を維持し、並列回帰17件を加えた111 tests PASS、fail/error/skip0。
今回の実行はattempt01の1回で、開発中test失敗0。旧source段の失敗attempt01/02/04は旧bundleへ保持する。
fake futures/mock workersを使用し、実workerやproduction runnerを起動していない。
既存小型Qiskit fixtureだけのsynthetic transpile3件、旧25＋新3＝28/64。
旧予約台帳を変更せず、旧25件のbyte/hashと残り39件を実行前に照合し、各新invocation前にも合計上限を検査した。
旧benchmark128（比較120＋axis/phase4＋full-operator4）の再実行0。

| 回帰 | 確認した内容 |
|---|---|
| bounded/遅延生成 | admitted workers1/2/5/12で先行job完了前に同数を投入。未完了数は各w以内。初回回収前の生成もw件。 |
| 完了順/axis/weight | 最新jobを先に完了させ、64件のtrajectory/axis順・6metric・weightを元のlogical順へ再構成。 |
| 処理中重複 | 同axisの3同一回路は予約/actual fake compile1件。owner完了前のentriesは空。両axis各32logical samplesはactual2件、weight各1。 |
| owner拒否 | identity不一致、RESERVED、cache chain、cross scope、完了record改変を拒否。独立digest検査後だけ後続cache completion。 |
| failure/accounting | compile/submit/worker/monitor failure、回収batch外のfailed future、生成例外でSTOP。追加投入/払い戻しなし。予約capと未完成logical request、部分ledgerを検査。 |
| serial同値 | 同じ小型合成fixtureで旧直列手順と新schedulerのlogical/actual件数、2×2合成signal、全metric pairing、302表示、P感度・weightが完全一致。 |
| worker/gate | mock available workerを使用し、death latch後のsubmitを拒否。新audit pathをcheckout gateが読むことを確認。 |

実行commandは新bundleのreview依頼とログに記録する。absolute既存Python、process限定環境、thread/process各1を使用した。
一時ledgerはprivate `/tmp/h4-synthetic-*`、合成matrix/circuitは一時メモリのみ。
source closure/AST・依存45件/compiler identityを監査し、二つ目のcommitでsourceを変更しない。

実SCF/分子入力、live worker並列実行・performance・memory/AS/RSS/cgroup、production filesystem障害、
全74,784件のwall/output内完了は未検証。合成PASSから科学成立・実speedup・資源上界を主張しない。
分子アクセス/生成、実signal/sampling/build/compile、GPU、本番起動、actual authorization発行、環境/他job変更は0。
公開後STOPし、入力生成plan/auth作成、本計算、H6/Track Bへ進まない。
