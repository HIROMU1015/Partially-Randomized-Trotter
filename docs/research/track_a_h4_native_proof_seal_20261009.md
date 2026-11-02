# H4追加native proof・technical再seal・起動承認待ち（2026-10-09）

**追加native停止証拠は現在の旧owned process残存確認としてPASS。凍結入力・SOURCE/profile/carryとのbindingを再固定し、technical planをsealした。実行認可はfalseで、本計算STOP。**
SOURCEは `6bd1ba01cd71ec3e2071082963c9f07478dada9a` のまま変更0。
起点REVIEW `c81144bb6678b603c9f320f2600b7d0a6dba06f7` はorigin branchとの一致を確認した。
旧worktree・packet・proof・監査を保持し、home内の独立worktree、新branch `track-a-h4-native-proof-seal-20261009` で作業した。
今回のREVIEW/remote actual SHAはcommit後の外部publication receiptと最終報告で固定する。

一つの資料入口は本書。
[追加proof監査](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/native_proof_audit_v1.json)、
[stop receipt](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/stop_receipt_v7.json)、
[technical seal](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/technical_seal_v1.json)、
[独立最終整合review](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/independent_final_consistency_review_v3.json)、
[残る起動承認](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/final_approval_status_v6.json)、
[commit対象](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/commit_inventory_v5.json)を結合する。
[前回全2150files受領監査とnative不足](track_a_h4_byte_receipt_binding_20261009.md)と[P1 cleanup修正](track_a_h4_cleanup_esrch_fix_20261009.md)は当時の証拠として保存する。

## 追加native停止proofの照合と証明範囲

incomingのJSONは17521 bytes、SHA-256 `cea6a11dbce913f3ee4d4f45155fbc6c3fb72077fdae3b2a66c3a2909d392f99`、sidecarも一致。
home内の新private exclusive directoryへ通常single-link fileとして受領し、元fileは保持した。
元HANDOFF（5216B）、process record（1313B）、startup（10369B）の3参照は、旧packet manifest/source_path/actual bytes/SHAと全一致。
旧host/UID30038、run05 SOURCE `6d365257770e99022b91d6a38dbee49ee0077503`、準備commit8f77bebf、run ID、session8932、launch/startup記録へ結合した。
新host SOURCE/前REVIEWへの参照も一致した。raw proof/sidecar・旧native runtime/controlはGit外に保持する。

| role | PID | 登録starttime ticks |
|---|---|---|
| driver | 8932 | 307348817 |
| workers | 9090–9092 | 307349254 |
| workers | 9093–9099 | 307349255 |
| workers | 9100–9101 | 307349256 |

driver1＋worker12のPID/starttime/UID/roleが元記録に厳密一致し、時間の重ならない2観測の全26 identityがABSENT。
各own-run scanはsampled21、remaining0、matches/unknown空、scan_complete=true、観測中disappeared0。除外した監査ancestryに旧13PIDはない。
監査中のboot IDはbefore/after一致、proc mountにhidepid設定は記録されていない。
観測範囲は2026-10-09 12:13:36.214947–12:13:36.645144 JST（proofのUTCを換算）。この時点のnative停止条件に以前のgapはない。

**過去のexit code、正確な終了時刻、reap時刻/順序、原run boot IDは未記録のままnull。今回の観測から補完しない。**
旧handoffのhistorical stop claimを正確なexit時刻へ昇格させず、2観測を連続監視とも扱わない。
旧source＋terminal logからのworker/executor/budget cleanup到達は引き続きSUPPORTED_INFERENCEであり、今回の観測で過去のcleanup/reap順を直接証明したとはしない。
旧run02対象のold_owned_run_stopped_v1.jsonもrun05終了証拠へ流用していない。

## 固定SOURCE・profile・input・stop・carry・plan binding

[SOURCE closure33](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/source_freeze_v5.json) はactual commit blob/checkoutと全一致。source/tests/schemaの変更0。
[environment](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/environment_profile_v2.json) と
[compiler](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/compiler_profile_v2.json) はlive private venvと一致し、install/upgrade/設定変更なし。
environment fingerprint `6abf53c37c82ec389d2a5d1f3aa7787a86db379c7e6312ea5f6836c7cc05cb35`、compiler fingerprint `fbe36b72d4c36f1b9d9dbe15b70ef09ef88376f3500b67694e80b51ac0517476`を維持する。
18version差、旧参照対raw RECORD45差、normalized22差/23一致を保持し、旧compiler binary/output equivalenceは未検証。候補採用承認は別途必要。

[入力receipt](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/input_receipt_v5.json) はNPZ6/freezeのstreaming SHAがPASS。
旧generation SOURCE049e699…/INPUTS_FROZEN_STOP/6 identity/freeze fingerprintに一致。科学arrayは読み込んでいない。
旧82 control filesのunique basename・nlink1・sender archive mappingを維持し、追加proof1件をstop_root直下のunique basenameへ正常copyして83件のbyte bindingを固定した。
固定runtime `verify_frozen_receipts()` は実受領filesでPASSし、旧journal charge・freeze lineageまで確認した。新hostの同PID確認やprocessへのsignalは行っていない。

[sealed plan v7](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/plan_sealed_v7.json)、
[authorization草案](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/authorization_draft_v7.json)、
[review草案](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/review_draft_v7.json) を新source_root/stop audit参照と実SHAへ再結合した。
plan fingerprintは `69d48a8298c0441ac2210b944b52fa416eeb6b091acecc9c26dc3e3b3a185e80`、auth digestは `673b43dd996c8903d74f0da87d165ab96324b6f539273a95fc1a37a6d0573d60`。
`sealed=true`、stop `control_complete=true` と現在観測に基づく `all_old_owned_processes_ended=true`。
`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、environment/observer/budget amendment未承認を維持する。
pure permission gateは `unsealed/unapproved newhost launch` のcombined messageで拒否する。今回の拒否理由は未承認flagsであり、technical seal不足ではない。
SOURCEに将来のseedを結合し、旧random/partial/cacheとの混合は行わない。

## 科学条件・累積budget・資源・fresh launch

H4 linear neutral singlet/STO-3G/DF12、8system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE、6距離・218 templates/点・32 paired trajectories・1308 signals/74784 logical条件は不変。
[native carry](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/carry_audit_v2.json) は20 actual /165214360 bytes /5466.188392877579秒を保持する。
run05の8 RESERVEDを返却せず、budget resetなし。現actual cap74784、残74764、今回新science actual0。
[静的budget](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/static_budget_v2.json) は全新74784 logicalを保証するcache節約20件を保証しないため、carry込みactual74804案が必要。現capへ適用していない。

CPU提案はworker12 `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18、14 distinct physical cores。許可・専有予約はない。
driver/worker各AS/RSS8GiB、headroom16GiB、monitor5秒、observer AS256MiB/RSS64MiB・admission120.25GiB案は未承認。
[容量](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/storage_projection_v5.json) はrecord/ledger/log/signal/observer/journal/temp/directory/block/inodeを含む追加output5GiB/301000 inodes、carry込みcharge8902109720 bytesを維持する。
追加proofは17521Bの正常copy2件＋sidecarを実inventoryに記録。既に受領・割当済みのcopiesはfresh outputの追加必要量と区別し、10GiB/72h等の上限を増やさない。
以前のquota user/group/project DISABLED監査は保存するが、launch前にmemory/pressure/OOM/CPU/filesystem/inode/quotaをfreshに確認する。物理予約ではない。

technical sealが成立しても、全mapの実行には明示+20契約改定が必要である。承認後にactual cap/CPU/observer/environment/authority referenceを反映するとplan/auth/reviewのdigestが変わるため、その最終実行artifactを再結合・reviewする。
fresh output/controlとone-shot lockの未使用、SOURCE33/profile/input/receipt/carry一致、5秒以内resource観測・3秒CPU低負荷sample、own UID/PID/starttime/pidfd、全role上限をlaunch直前に検査する。
不合格なら起動せず、own-run STOPと全child wait/reap/FD/pipe終了を維持する。
[absolute command案](../../artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/unexecuted_launch_proposal_v7.json) は未実行で、現在は拒否される。fresh resource条件の成立やproduction成功を今回証明したとはしない。

## 残る承認事項をまとめた次段案

1. 既存private venvを変更せず候補environment/compilerを本番採用する。旧output equivalence未検証という限界を受け入れる。
2. worker12/driver/observerの上記CPU配置をown-runへ許可する。
3. 本番observer追加roleとAS256MiB/RSS64MiB・admission120.25GiB案を承認する。
4. 累積actual capだけ74784→74804へ+20変更し、失敗分返却/resetなし、他の上限を維持する。
5. H4 signal/compile mapを一度実行し、完了またはfail-closed STOP後に停止する。自動retryなし。明示launch指示を与える。

今回は追加transpile・旧benchmark再実行・分子アクセス/SCF/DF生成・科学array読込・本番runner/worker・GPU/CuPy/taskset・共有環境/既存venv/他job変更0。
前P1の39人工/独立13件PASSを保持し、今回のmetadata監査をproduction検証へ読み替えない。
軽量資料だけを固定・non-force pushし、最終報告後STOPする。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[新実行認可入口](track_a_h4_authorized_launch_20261009.md)へ最新利用者指示・v8認可・独立reviewを固定した。
上記未承認は以前のsnapshotとして保持。SOURCE/carry/他capsは不変、actual74804だけ明示改定。
実起動はimmutable commit後のfresh gateに従い一度のみ。旧proof・歴史unknownの扱いは変えない。
