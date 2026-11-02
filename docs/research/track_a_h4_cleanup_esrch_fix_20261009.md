# H4 cleanup ESRCH修正・新SOURCE・独立再review（2026-10-09）

**cleanup競合P1を修正し、限定人工回帰39件と別担当の独立再検査13件がPASS。入力とnative停止証拠は未受領で、本計算はSTOP。**
旧起点branch `track-a-h4-receipt-final-review-20261008` のorigin SHAは `992c09d60c802525e00cf642eb5e1724b1a8c3e8` と一致した。
旧SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044` と旧資料・証拠を保持し、独立worktreeで修正した。
新branchは `track-a-h4-cleanup-esrch-fix-20261009`、SOURCEは `6bd1ba01cd71ec3e2071082963c9f07478dada9a`。
SOURCEと、sourceを変えないREVIEW_BUNDLEを別commitにする。REVIEW/remoteのactual SHAは固定後の外部publication receiptと最終報告で照合する。

本書を一つの資料入口とする。
[source closureと旧/new blob/hash](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/source_freeze_v3.json)、
[限定人工結果](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/focused_test_results_v3.json)、
[独立再review](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/independent_rereview_v3.json)、
[binding状態](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/binding_update_v3.json)、
[最終承認状態](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/final_approval_status_v4.json)、
[commit対象一覧](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/commit_inventory_v3.json)を結合する。
[旧P1不合格報告](track_a_h4_receipt_final_review_20261008.md)と[旧準備・57人工結果](track_a_h4_prelaunch_preparation_20261008.md)は当時の履歴として保存する。

## 修正と検証範囲

[OwnedIdentity](../../src/trottertracks/resource_applicability/h4_geometry/observer.py) の通常terminate経路と元親の退出確認後の経路は、既存のUID/PID/starttime/parent/pidfd検証後、同じ送信helperへ進む。
`pidfd_send_signal()` の `OSError.errno == ESRCH` だけを「既に退出」としてFalseを返し、後続ownerのcleanupを継続する。EPERM/EBADF/EINVALなど他の送信errorはraiseする。
所有不一致での送信拒否、元親のstable pidfdによる退出確認、fail-closed方針を維持した。runtime変更はobserver.pyだけであり、科学処理・compiler・monitor5秒・累積上限は変えていない。

[新14回帰](../../tests/tracks/resource_applicability/test_h4_cleanup_esrch.py) は両経路のfirst/middle/all ESRCH、非ESRCH、所有不一致、二重cleanup、最初のSTOP理由保持を検査する。
呼出側の全owner訪問、process.wait/reap、pidfd・socket・stdin/stdout・executor終了まで確認した。
新14＋関連既存25＝39件、failure/error/skip0。別担当はfileを書かない13件を独立実行し全PASS、P1_SCOPE_TECHNICAL_PASS。
既存の終了検査だけ最大4つの最小人工process（test・driver・stdlib observer・sleep worker）を使用し、own-test subreaperを終了時に復元した。
12科学workersは起動せず、ESRCH caseはpidfd送信をmockした。追加transpile0・科学actual呼出0・本番起動0。

[事前人工計画](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/ARTIFICIAL_TEST_PLAN_v3.json) で、process所有、thread1、180秒/16MiB per attempt、累積600秒、process別AS/RSS上限を固定してから実施した。
測定wall1.4072515107691288秒、driver peak RSS29360128 bytes、人工出力229084 bytes。
raw logはGit外のhomeに保存し、[外部証拠manifest](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/local_artificial_evidence_manifest_v3.json) にbyte数/SHAを固定した。
これはlocal人工実装検査であり、immutable CI・production成功・旧compilerとの完全同一性は証明しない。旧57件全体・旧synthetic28/64・旧benchmark128件は再実行していない。

## SOURCE・環境・binding

source closureは旧32＋新test1＝33件。actual SOURCE blobとcheckout SHAを全件照合し、旧/new blob/hashと変更理由を保存した。
source/tests/schemaのうち変更はobserver.py・人工test runnerの2件と新test1件のみ。schemaと他のruntime sourceは旧SOURCEから不変。
新SOURCEから将来のseed identityを再結合する必要がある。旧random/partial結果・旧cacheとの混合/reuseは行っていない。

既存private venvは変更せず、候補[environment profile](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/environment_profile_v2.json) と
[compiler profile](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/compiler_profile_v2.json)を保持し、live profile一致を独立確認した。
environment fingerprint `6abf53c37c82ec389d2a5d1f3aa7787a86db379c7e6312ea5f6836c7cc05cb35`、compiler fingerprint `fbe36b72d4c36f1b9d9dbe15b70ef09ef88376f3500b67694e80b51ac0517476`。
[環境差分](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/environment_evaluation_v2.json) はversion18差、旧normalized参照対新raw RECORD45差、正規化後22差/23一致を保存する。
主要科学package versionとinstalled11 sourceの一致、既存venvを変更しないことを候補選定理由とする。Rustworkx差、binary/output compiler equivalence未検証という限界は残る。採用承認はまだない。

[plan](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/plan_draft_v3.json)、
[authorization](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/authorization_draft_v3.json)、
[review草案](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/review_draft_v3.json) のSOURCE/closure/source_root/profile参照とplan fingerprint/auth digestを新SOURCEへ更新した。
input_root、run ID、fresh output/control候補、科学/compiler options、carry/capsを保持する。
独立技術合格と実行承認を分離し、`approved=false`・`runtime_authorization=false`・`allowed_cpus=[]`・`sealed=false`を維持する。

## 凍結入力・停止証拠の受領待ち

[入力受領](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/input_receipt_v3.json) は0/6、generation-freezeも未受領。
[停止受領](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/stop_receipt_v3.json) はbyte-budget journal/log・native control未受領。
科学array読込0・入力再生成0・実NPZ使用0。旧handoffの「全owned process終了」metadataだけではnative停止proofを合格扱いしない。
技術P1合格でも入力/native証拠が揃っていないため、technical planを再sealしない。

利用者が行う最小転送は既存の [local-only手順](/home/AbeHiromu/projects/h4-handoff-evidence/20261008/receipt-final-review/USER_TRANSFER_STEPS.md) のpacket生成→2 filesのcopyである。
旧サーバーの既存ログイン済みターミナルで最初のPython here-doc、続いて同書のSCPを実行する。認証できなければ既存SFTPで同じpacket/SHA256SUMSを手元経由でcopyする。
認証設定・鍵・共有環境は変更しない。packetはhome内、originalは変更しない。内部SSH情報・packet・NPZ・native runtimeはGit外に保持する。
[既存の転送要件](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/TRANSFER_REQUIREMENTS_v1.md) に従い、受領時にfile一覧・bytes・streaming SHAを照合しnative終了proofを確認する。
旧handoffには個別byte数がないため、転送元manifestの捕捉値と受領byte数を比較し、既知SHAとも照合する。未受領の値を一致扱いしない。

## 保持する資源・容量・累積budget案

科学条件はH4 linear neutral singlet/STO-3G/DF12、4 spatial/8 system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE。
6距離、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、218 templates/点、32 paired trajectories、1308 signals/74784 logical wrappersを保持する。

CPU提案はworker12 `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18、14 distinct physical cores。使用許可・専有予約はない。
driver/worker各AS/RSS8GiB、headroom16GiB、monitor5秒、observer AS256MiB/RSS64MiB、admission120.25GiB案を保持する。observer本番roleは未承認。

carryは20 actual /165214360 bytes /5466.188392877579秒、現actual cap74784・残74764。失敗分の返却/resetなし。
[静的budget](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/static_budget_v2.json) はcache節約20件を保証せず、全74784新logicalを一度処理するとcarry込み最大74804件。
最小契約変更は累積actual capだけ+20→74804という未承認案で、現planへ適用していない。10GiB/72h等の上限は不変。

[容量内訳](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/storage_projection_v2.json) はrecord/ledger/worker log/signal/1秒append observer log/journal/temporary publish/4KiB blocks/directories/inodesを含む。
追加output必要5GiB/301000 inodes、physical bound4539042688 bytes、carry込みcharge8902109720 bytes、10GiBへの残1835308520 bytes。
受領copy量が未確定の間は別枠1GiB/4096 inodesを加えた6GiB/305096案を保持する。これは物理予約ではなく、受領後に実byte/inode inventoryへ更新する。
前reviewのread-only quota（user/group/project DISABLED）、capacity・memory・pressure/OOM観測は採取時点の証拠として保存する。共有設定を変更せず、launch時に取り直す。

## 次段の承認案とfresh launch条件

技術的な残条件は入力/freeze/native stop受領照合、technical plan再seal、source/profile/input/output/plan/auth/review bindingの最終確認である。
未承認事項は候補環境/compiler、CPU配置、observer追加roleと予算、累積actual cap74804案、利用者の明示launch指示。
承認案のscopeは **H4 signal/compile mapを一度実行し、完了またはfail-closed STOP後に停止、自動retryなし**。

launch直前にSOURCE33/profile/input/receipt/approval一致、fresh output/control（既存partial不使用）、carry/journal、own-run lock/UID/PID/starttime/pidfd、全role上限、capacity/inode/quota、memory/pressure/OOMを検査する。
resource観測5秒以内と3秒CPU低負荷sampleの条件を満たさなければ起動しない。own-run限定STOP、全child wait/reap・FD/pipe終了まで確認して停止する。
[absolute command提案](../../artifacts/resource_applicability/track_a_h4_cleanup_esrch_fix/2026-10-09/unexecuted_launch_proposal_v3.json) は新SOURCEのworktreeとv3草案を指すが未実行で、現在のfalse flags/未sealでは拒否される。
受領・契約変更・利用者承認後はfingerprintを再結合し、新承認に対応したcommandへ更新する。今回の技術PASSはlaunch認可ではない。

本計算・追加transpile・分子アクセス・GPU query/use・CuPy import・taskset・共有環境/venv/他job変更は行っていない。資料を固定・公開し、最終報告後STOPする。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[新受領監査・v6 binding](track_a_h4_byte_receipt_binding_20261009.md)で全2150files/既知9SHA/6入力freeze/carryの照合を完了した。
この報告の「未受領」は以前のsnapshotとして保持する。新SOURCEは不変、P1合格も維持。
run05全owned終了のnative proofが不足しているため再sealせず、実行認可falseで本計算STOPを継続する。
