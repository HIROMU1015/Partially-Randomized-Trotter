# H4凍結byte受領・carry照合・最終binding review（2026-10-09）

**凍結入力・全fileのbytes/SHA・freeze lineage・carryはPASS。run05の全owned process終了を確定するnative証拠が不足しており、再seal・本計算はSTOP。**
一つの資料入口は本書。SOURCEは `6bd1ba01cd71ec3e2071082963c9f07478dada9a` のまま変更0。
起点REVIEW `6f8e95ad0f69c9d8d49a61aab20bb4669b826ac5` はorigin branchとの一致を確認した。
既存worktree・packet・旧資料を保持し、新branch `track-a-h4-receipt-reseal-20261009` とhome内の独立worktreeで受領資料を作成した。
今回のREVIEW/remote actual SHAはcommit後の外部publication receiptと最終報告で固定する。

[受領監査](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/receipt_audit_v4.json)、
[carry監査](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/carry_audit_v2.json)、
[native停止評価](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/native_stop_assessment_v2.json)、
[独立最終整合review](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/independent_final_consistency_review_v2.json)、
[binding](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/binding_update_v6.json)、
[残る承認状態](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/final_approval_status_v5.json)、
[commit対象一覧](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/commit_inventory_v4.json)を結合する。
[前回P1修正・39人工/独立13件PASS](track_a_h4_cleanup_esrch_fix_20261009.md)は保存し、再実行していない。

## 受領byteと凍結入力

利用者の訂正指示に従い、incoming直下のtar/SHA256SUMSを使用した。別directory・RECEIVE_INSTRUCTIONSの待機は不要と確認済み。
packetは86691840 bytes、SHA-256 `8312fd1cb8393c46c81e48567161504045b3d4608660a8d3a4a7fc3eeed0326b`。
manifest.jsonは955346 bytes、全2150 filesのsource byte合計83637467、manifestを含む2151 membersのpayload合計84592813 bytes。
全memberはregular/uniqueで、symlink・hardlink・重複・absolute/`..`/非canonical path・sparse/deviceを拒否した。
home内の新exclusive private directoryへ64KiB単位でbyte-only受領し、全fileのmanifest bytes/SHA、書込後SHA、外側tar SHAを照合した。独立担当も全2150 filesを照合してPASS。

旧handoffの既知9SHA（NPZ6・freeze・run05 byte journal/log）も全件PASS。
NPZは各13720266 bytes、6件合計82321596 bytes。generation-freezeは27115 bytes、journal15616 bytes、log4560 bytes。
NPZをopaque bytesとして扱い、科学array読込・入力再生成・科学計算は0。
[入力receipt](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/input_receipt_v4.json) はcomplete=true。
freezeのSOURCE `049e69919af16ad29a67a217dc7a407d6b1754a6`、INPUTS_FROZEN_STOP、mandatory STOP、fingerprintと6入力identityは旧planに一致した。
転送元manifestの捕捉byte数と受領byte数を比較した。旧handoffに個別byte数があると読み替えていない。

最初の受領後RSS判定とread-only再監査の2試行は、実行開始前からのru_maxrss peakを当該executableのRSSと誤認して失敗した。失敗utility/log/計画を保持した。
同じ実行形式の開始時ru peak313724928 bytesに対し、開始時VmHWM15204352 bytesという計測差を確認した。
当該executableの`/proc/self/status` VmHWMへ監査metricを訂正し、最終全file再照合のpeak21495808 bytesは準備監査RSS64MiB/AS256MiB以内だった。
本番observer等の上限を変更したものではない。再展開・受領file上書き・科学retry・carry resetは行っていない。

## 累積carryのnative照合

run01–05のjournalとledger delta chainを検査した。cumulative journalを重複加算せず、後続runの初期carry entryと前run chargeの一致を確認した。

| run | 新actual予約 | 累積actual | 累積charge bytes |
|---|---:|---:|---:|
| run01 | 0 | 0 | 0 |
| run02 | 4 | 4 | 165161600 |
| run03 | 3 | 7 | 165168060 |
| run04 | 5 | 12 | 165184714 |
| run05 | 8 | 20 | 165214360 |

run05には9 ledger deltas、8 RESERVEDが残る。失敗予約は消費扱いで返却しない。今回新science actualは0。
最終wall記録5396.227741413284秒に対し、固定conservative carry5466.188392877579秒を保持し、差69.96065146429464秒を減額しない。
carryは20 actual /165214360 bytes /5466.188392877579秒、現actual cap74784、残74764。
全74784新logical wrappersを一度処理する静的worst caseはcarry込み74804。保証できるcache節約20件はない。
[静的予算](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/static_budget_v2.json) と、累積actualだけ+20→74804という未承認案を維持する。上限は変更していない。

## native停止証拠の合格範囲と不足

82 native control filesを受領した。run05 logはmonitor STOPを記録し、exact旧SOURCE `6d365257770e99022b91d6a38dbee49ee0077503` の末尾failure判定からのtracebackに一致する。
旧closeは `pool.shutdown(wait=True,cancel_futures=True)` → budget.close → monitor failure判定の順であり、worker wait・pipe・executor・budget cleanup経路への到達はsourceとlogから推認できる。
これを直接の全process終了証明や新hostのproduction成功へ読み替えない。

run05 control内の `old_owned_run_stopped_v1.json` は13:11:47 UTCに記録され、run05起動14:24:57 UTCより前である。
対象journalはrun02の `918106…`、ledger7/record2/signal1/wall1837であり、run05の終了証拠には使えない。
HANDOFFのall_owned_processes_ended=trueはmetadataとして保持し、native terminal proofの代替にしない。
driver PID8932/start307348817とworker12件の停止後exit-status/identity/残存0監査はpacketにない。
[最小追加証拠の依頼](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/NATIVE_TERMINAL_PROOF_REQUIRED_v1.md)に旧host担当へ渡せる具体条件を固定した。
これは技術証拠の不足であり、利用者のCPU等の承認不足とは別である。新hostの同PID確認や他jobへのsignalは行わない。

## 新SOURCEへのbindingと整合修正

[source closure33](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/source_freeze_v4.json) はactual SOURCE blob/checkoutに全一致。
既存private venvを変更せず、live environment/compiler profileは固定値と一致した。
version18差、旧参照対raw RECORD45差、正規化後22差/23一致、旧compiler output equivalence未検証を保持する。
[environment](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/environment_profile_v2.json)、
[compiler](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/compiler_profile_v2.json)、
[差分](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/environment_evaluation_v2.json)は候補層であり、採用承認はない。

受領先は元planのinput_root/stop_rootを維持し、SOURCE/source_root/profile/input/output/carryを新worktreeへ結合した。
[plan/auth/review v6](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/plan_draft_v6.json) のplan fingerprintは `b5f4fbd2a294d94faf4b363aed38abc1a0f405b2865f7748555c6ca371492f6d`。
auth digestは `c19a5454cbf3688bb11ac9ffd74a452ffbb88ec57baaa6c041c78a3ee5a084ad`。
科学条件/compiler options/run ID/carry/capsは旧草案と不変。SOURCEに将来のseedを結合し、旧random/partial/cacheを混合しない。

準備utilityの初期hardlink配置は、固定validatorのnlink1条件で拒否された。当taskが作ったaliasだけを確認し、元receiptを保持して通常のexclusive single-link byte copyへ訂正した。
独立reviewでcontrolのnested pathも固定validatorのbasename条件に不一致と判明したため、82件を一意basenameへ追加copyし、元archive pathはmetadataへ保存した。
[control map](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/control_binding_map_v6.json) の全82件を固定receipt_inventoryでPASS確認した。
旧nested copiesと未公開v4草案は証拠として保持し、[訂正履歴](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/draft_corrections_v2.json)へ理由/hashを保存した。SOURCEの所有・basename・nlink条件は緩めていない。

[stop receipt](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/stop_receipt_v6.json) のbyte receiptはcomplete=trueだが、native terminal proof未評価のためcontrol_complete=false。
technical sealは保留し、`sealed=false`、`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`を維持する。
実metadata gateは `native stop/control proof not received`、permission gateは `unsealed/unapproved newhost launch` で拒否する。技術証拠を埋めずにapprovalだけを付けてもlaunchへ進めない。

## CPU・observer・容量・quota・次段承認案

CPU提案はworker12 `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18。14 distinct physical coreの候補であり利用許可・予約はない。
driver/worker各AS/RSS8GiB、headroom16GiB、monitor5秒、observer AS256MiB/RSS64MiB・admission120.25GiB案を保持する。
[容量](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/storage_projection_v4.json) はrecord/ledger/log/signal/observer/journal/temp/directory/block/inodeを含み、追加output5GiB/301000 inodes、charge8902109720 bytesを維持する。
受領tar・private raw receipt・正常配置copiesの実使用は264785920 allocated bytes/2352 unique file・directory inodes。173 regular copies、hardlink0。
受領時available約413.25GiB/free inodes220490120。受領copyは既に割当済みで、旧暫定transfer margin1GiB/4096 inodesを実inventoryへ置き換えた。追加output5GiB要件は不変で物理予約はない。
以前のquota user/group/project DISABLED監査は保存するが、launch前にはmemory/pressure/OOM/CPU/fs/inode/quotaをfreshに取り直す。

残る技術手順は旧host native終了証拠の受領照合 → plan再seal・binding再固定 → 最終整合review。
その後に必要な利用者の承認案は次の5件である。

1. 既存private venvを変更せず候補environment/compilerを本番採用する。
2. 上記worker12/driver/observerのCPU配置をown-runに許可する。
3. 本番observer追加roleとAS256MiB/RSS64MiB・admission120.25GiB案を承認する。
4. 累積actual capだけ74784→74804へ+20変更し、10GiB/72h等は維持する。
5. H4 signal/compile mapを一度実行し、完了またはfail-closed STOP後に停止する。自動retryなし。

H4 linear neutral singlet/STO-3G/DF12、8system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE、6距離・218 templates/点・32 paired trajectories・1308 signals/74784 logical条件は維持する。
[absolute command案](../../artifacts/resource_applicability/track_a_h4_receipt_reseal/2026-10-09/unexecuted_launch_proposal_v6.json) は未実行で、現在は拒否される。
SOURCE/profile/input/plan/auth/review/carry/own PID/start/pidfd/fresh output/control/one-shot lockの最終一致、5秒以内resource観測・3秒CPU低負荷sample、own-run STOPとwait/reap/FD/pipe cleanupがfresh launch条件。
技術証拠・承認が揃った後も不合格なら起動しない。利用者の明示承認・launch指示まで本計算なし。

今回source/tests/schemaの変更0、新科学actual/追加transpile/本番runner・worker/分子アクセス/array生成/GPU/CuPy/taskset/共有環境・venv・他job変更0。
軽量資料だけをcommit/non-force pushし、最終報告後STOPする。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[追加proofと新technical seal](track_a_h4_native_proof_seal_20261009.md)で旧hostの13identity・2回残存0・元証拠hashの照合を完了し、現在の停止条件のgapを解消した。
上記の不足・未sealは以前のsnapshotとして保持する。新planはsealed=trueだがapproval/runtime falseで本計算STOP。
歴史exitcode/正確なexit-reap時刻は依然未記録であり、今回のproofから補完しない。
