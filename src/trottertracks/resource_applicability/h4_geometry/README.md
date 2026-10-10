# Server-native H4 geometry source

Import-safe modules; production execution requires separate reviewed generation and signal authorizations.
See [implementation and limits](../../../../docs/research/track_a_h4_geometry_server_native_source_implementation.md)
and [source/review bundle](../../../../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md).
Synthetic tests run through `scripts/resource_applicability/run_h4_geometry_source_tests.py` only.
No molecular fixture, production launch, environment edit, legacy guard modification, or GPU action is part of this source freeze.


## 2026-10-07 H4新host A案 source/人工検証固定・本計算未認可

[監視修正・32人工tests](../../../../docs/research/track_a_h4_new_server_monitor_fix_a_20261007.md)。既存private venv不変で準備用A案採用。
旧source19は3変更/16不変、新module込み25 closure。256×256人工matrix＋9-qubitの旧/new byte/digest一致、独立observerのGIL/GC観測・I/O delay/EOF/所有・資源境界を検証。
SOURCE `b2a5ad89e8b39d72716f7ddb17d263bd0cdedb45`、production/追加transpile0、旧28/64・benchmark128保持。環境18 version差、旧45 raw-reference RECORD差保持・normalized22差を明記。入力6/freeze/runtime/control未受領。
observer AS256MiB/RSS64MiB/admission120.25GiB、容量5.5625GiB案は未承認。allowed_cpus=[]/approved=false/runtime_authorization=false/launch=null、STOP。科学成果/原稿/Track Bと旧資料を保持。


## 2026-10-08 H4本計算前準備・最終承認待ちSTOP

[統合入口・最終承認案](../../../../docs/research/track_a_h4_prelaunch_preparation_20261008.md)。新host schema/profile/observer/CPU/one-shot/累積budget bindingを整備。
SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、32 closure、57限定人工tests PASS。core quota read-only確認、worker12＋driver/observer各1別coreを提案。
carry20/165214360 bytes/5466.188392877579秒、残74764を保持。全74784 logicalを保証する最小actual cap+20→74804案は未承認。
候補環境はprivate venv不変、18 version/旧参照対raw45・normalized22差。追加output5GiB/301000 inodes、charge約8.29GiB、copy前は暫定6GiB。
入力6/freeze/native stop proof未受領、allowed_cpus=[]/approved=false/runtime_authorization=false/未seal、science/追加transpile/GPU/共有環境・他job変更0。明示承認・final review・launch前にSTOP。

## 2026-10-09 H4 cleanup ESRCH修正・独立再review PASS

[新SOURCE・独立再review・残る承認条件](../../../../docs/research/track_a_h4_cleanup_esrch_fix_20261009.md)。旧992c09d6から独立worktreeでP1を修正し、SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33を固定。
両pidfd送信経路はESRCHだけ既退出扱いで後続cleanupを継続し、他の送信error・所有検証を保持する。
限定人工39件PASS、別担当13純mock件PASSとbinding照合でP1_SCOPE_TECHNICAL_PASS。wait/reap/FD/pipe/first STOP理由保持まで確認。
新test `tests/tracks/resource_applicability/test_h4_cleanup_esrch.py` と既存人工runnerで追跡し、source/profile/plan/auth/reviewを新SHAへ再結合した。
入力0/6・freeze/native停止証拠未受領はNOT_EVALUABLE、未seal。環境/CPU/observer/74804案は未承認、carry20/165214360 bytes/5466.188392877579秒・現actual cap74784不変。
approved=false、runtime_authorization=false、allowed_cpus=[]。追加transpile/科学actual/本番起動0。旧資料を保存して本計算STOP。

## 2026-10-09 H4全byte受領・carry合格・native終端proof待ち

[受領・binding・最終承認案](../../../../docs/research/track_a_h4_byte_receipt_binding_20261009.md)。packet86691840B/全2150files/既知9SHAをbyte-only照合しPASS、NPZ6/freeze受領完了。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変。ledger chain/cumulative journalからcarry20/165214360 bytes/5466.188392877579秒を保持、現cap74784・残74764。
run05 log/exact旧sourceからworker cleanup到達は推認可能だが、driver/12 workers停止後identity/残存0 native proofが不足。古いrun02停止監査をrun05proofへ流用しない。
input/profile/source/output/carryとv6 plan/auth/reviewを結合し、control82件のbasename・単一link・sender manifest mappingを照合。source条件は緩めていない。
sealed=false/approved=false/runtime_authorization=false/allowed_cpus=[]。環境/CPU/observer/累積74804案/一度のmap launchは未承認。科学array読込/新科学actual/追加transpile/共有設定変更0でSTOP。

## 2026-10-09 H4追加native停止proof合格・technical再seal

[再seal・独立最終整合review・一括承認案](../../../../docs/research/track_a_h4_native_proof_seal_20261009.md)。追加JSON17521B/SHA一致、旧host/run05の13identity・2回残存0・元3証拠hashを照合して現在のnative停止条件PASS。
過去のexit code/正確な終了・reap時刻/原boot IDは未記録のままnull。今回の観測で補完せず、連続監視や歴史cleanup順の証明とも扱わない。
SOURCE `6bd1ba01cd71ec3e2071082963c9f07478dada9a`・closure33不変、profile/input/carry/control83件の固定validator合格でplan再seal、sealed=true。
approved=false/runtime_authorization=false/allowed_cpus=[]。carry20/165214360 bytes/5466.188392877579秒、現cap74784・残74764を保持。
候補environment/compiler・CPU・observer・累積actual74804案・一度のmap launchは未承認。承認による最終artifact/digest再結合とfresh resource/CPU/fs/inode/quota gateをlaunch前に確認。
科学array読込/新science actual/追加transpile/source変更/共有設定変更/本計算0でSTOP。

## 2026-10-09 H4一度のmap実行を利用者承認・最終artifact固定

[実行認可・直前gate・起動報告入口](../../../../docs/research/track_a_h4_authorized_launch_20261009.md)。利用者の明示認可で候補environment/compiler採用、worker12 CPUs2/4–6/8–15・driver16・observer18、observerAS256MiB/RSS64MiB/admission120.25GiBを認可。
carry20/165214360 bytes/5466.188392877579秒を保持し、累積actualだけ74804へ+20改定。SOURCE6bd1ba01・science/compiler/options/他caps不変。
sealed/approved/runtime_authorization=true、allowed_cpusはexact14role集合。独立v8 review PASS。artifact commit後fresh CPU/memory/PSI/OOM/FS/inode/quotaとSOURCE/profile/input/carry/unusedrootを確認しPASSなら追加承認なし一度起動。
既存proof/回帰/benchmark再実行0、追加準備campaign/transpile0。oldpartial/cache/GPU/共有環境・venv・他job変更なし、完了またはfail-closed STOP後終了・retry/次stageなし。
これは認可artifact固定時点のsnapshot。実起動/PID/状態は入口への追記・外部runtime receiptで別記録する。

## 2026-10-09 H4 library cache保存先修正・再実行予算不合格

[修正・再実行条件](../../../../docs/research/track_a_h4_library_cache_fix_20261009.md)。前回のOpenFermion→Matplotlib mkdir拒否を、homeの新private library cacheへprocess限定MPLCONFIGDIRを結合して修正。
driver/workerとも既存directoryのEEXIST probe以外のcache writeを拒否、29816BのSHA固定。47限定回帰＋3 import case PASS、科学array/transpile/実worker/affinity/GPU0。
SOURCE `b8b3ce6e8c98f1ec0419a7af79c5d7c5f3a3b9bb`、36 closure、science/compiler/options不変。前回費用を返却せずcarry20/4428938712B/5472.345380863175sへ結合。
累積worst charge13165893832B=12.261694GiB>承認10GiBで未seal/approved=false/runtime_authorization=false、再起動0。13GiBは未承認proposalのみ。
既承認environment/CPU/observer/actual74804を保持。cap改定・新SOURCE/gate binding/review・fresh gate後の一度再実行が残る。
private homeは共有systemと区別し、旧run/one-shot/失敗証拠・全予約課金を保持する。

## 2026-10-09 H4軽量高速化・限定同等性確認

[変更・限定検証・binding](../../../../docs/research/track_a_h4_lightweight_speedup_20261009.md)。driverの距離内共通準備を再利用し、one/DF block呼出を静的13×218→13、全prepareを218→10種類に削減。ledger deltaは変更entry/reservation各最大1だけを参照し、全件走査・saved-historyコピーを除いた。
SOURCE `4d2d1492fc23d0736c305533d78967cc1db8a7c8`、closure38。限定48人工PASS、全218prep/代表8wrapper+dense256case1/代表4signal/ledger13fileの旧new bytes・digest一致。12workersはmock、単一test process内部thread1、science array/transpile/GPU/affinity/production0。
実Gaussian/旧compiler output/実速度・H4本体成功は未検証。monitor/caps/compiler/science/carry不変、旧partial/cacheと混合しない。
carry20/4428938712B/5472.345380863175s、actual74804既承認、worst charge12.261694GiB>承認10GiBは残る。未seal/approved=false/runtime_authorization=false、absolute_launch_command=null、本計算0。

## 2026-10-09 H4利用者が13GiB累積charge・一度の再実行を明示認可

[認可・source・一度の起動入口](../../../../docs/research/track_a_h4_approved_relaunch_20261009.md)。利用者の「これについては問題ないので再実行して」を、既存13GiB cumulative charge案と一度のmap再実行の承認として反映。
SOURCE `a7b617600cd7063f7870f2059d5694ef00283f0e`/closure39、output改定schema/gate/実OutputBudget capとmarginのみ変更。legacy10GiB default・科学/compiler/その他caps・prepare再利用/ledger保存は維持。
限定22pure gate PASS、旧48speedup/library/cleanup/native証拠campaign再実行なし。carry20/4428938712B/5472.345380863175s返却なし、actual74804・新残74784。
worst13165893832B <= 新cap13958643712B、余裕792749880B。sealed/approved/runtime_authorization=trueの認可artifactへ再結合。
独立review/artifact固定後fresh SOURCE/profile/input/carry・CPU/memory/PSI/OOM/fs/block/inode/quota/unusedroot/one-shot合格時にそのまま一度起動。実run状態はruntime証跡へ記録。
既承認12workers CPUs2/4–6/8–15、driver16/observer18、thread1・observerAS256MiB/RSS64MiB/admission120.25GiB保持。自動retry/入力再生成/旧partial/cache/次stage/GPU/共有設定変更なし。


## 2026-10-10 H4 worker AS/RSS32GiBを明示承認・SOURCE固定

利用者の「上限を３２Gで修正して」でnewhost4workersを32GiB、driver8GiB/observer256MiB・64MiB/head16GiBを維持。両fresh/startup gateは152.25GiB。
親soft8でhardをchild準備まで保持し、workerは認可/checkout後に32、全ready後driverhard8。observerはtrusted driver PIDとnative登録workerを8/32で区別。
独立review P2 cleanup signal errorでの後続掃除skipを修正、最終53pure/mock PASS。実child/science/transpile/affinity/本体0。
SOURCE b652fff9d2907015d3b8b7b23ce8bf0fe4fce33a/closure65、旧SOURCE変更7/不変55+new3。科学/compiler/凍結入力/旧pressure/carry0/17GiB/74805/72h不変。
SOURCE/profile/input/carry/capsの準備binding、memory承認true・overalllaunchfalse/allowed[]/未seal/commandnull。fresh run/plan/auth/review/freshgatesは次回起動へ。
旧run07 one-shot/partial/cacheは保持し再利用しない。元native診断2gapsと全map完走は未検証。共有環境/venv/他job/GPU変更0。
[修正資料入口](../../../../docs/research/track_a_h4_worker_memory32_fix_20261010.md)。
