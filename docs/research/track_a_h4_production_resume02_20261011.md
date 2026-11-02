# H4 resume02：完了分を保持し高速化版へ引継ぎ・72h STOP解除

SOURCE `d0726e946f89f8c30a08d383fb8ff48b363629ec`、branch `track-a-h4-production-resume02-20261011`。
[固定artifact・承認・独立review・receipt要約](../../artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/README.md)が資料入口。
利用者の「引き継いでほしい。またその際に可能であればよいので、実行時間の上限をなくして」でown resume01停止・一度のmap引継ぎ・可能ならwall STOP解除を明示認可。
今回は実装・限定検査・容量見積りで条件を満たしたため、wall_seconds=nullを選択する。shared環境・既存venv・他job・GPUには変更/アクセスしない。

## 保持する結果と予算

own driverだけへpidfdの固定PID/start/parent/UID照合後SIGINT。旧driver/4workers/observerのnative残存0を2回・2秒以上間隔で確認。
元Source/Review/全file bytes/streaming SHAとrun10由来import・resume01新規ledgerを照合し、29388 COMPLETEと521 signalを保持する。
receipt 11204422 bytes、SHA `00f50ccf58c682563d79eecd8158d0a941cb611778915400d3e700919347ea43`。
carry 28731 actual reservations / 8788058342 bytes / 104895.53687566705秒を保持。
未完了の旧祖先2・親3 reservationsの費用は保持し、未COMPLETE/orphan/pendingを完了結果として再利用しない。orphan/pending各0。
dataset/seed SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`は固定し、新execution SOURCEを別記録する。
歴史的exit code・正確な終了時刻はnullのまま、今回の観測から補完しない。

残未完了だけを基底inverse・命令/定義encoding memo版で処理し、既検査の回路/旧canonical bytes/digest・compiler/optionsを保つ。
all74784 logical wrappersの保守的actual 74127 <=74805。
累積charge bound 17573199062 bytes <=17GiB、残余 680411946 bytes。
実byte journalは既に過去chargeを含むため二重加算しない。祖先carry込みの容量boundを重ねず、一度のfresh全map bound+実carry+trace差+prior journal128で見積る。
physical bound 4563165688 bytes、追加余裕2GiB、required 7516192768 bytes / 301032 inodes。
停止証拠・raw結果・旧one-shot/control/outputを保持する。

## 時間上限だけの変更

新[wall profile](../../artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/wall_time_profile_v1.json)と
[利用者authority](../../artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/USER_WALL_AUTHORITY_v1.json)に基づきdriver/独立observerの両方で72h累積STOPを解除する。
累積wallのfinite/nonnegative記録を維持し、既定legacy経路の72h capは残す。nullは新scopeと明示承認のある場合だけ。
時間無制限は無期限成功の保証ではない。observer append-only traceのbyte cap2GiB、総charge17GiBは保持し、いずれかの上限でSTOPする。
observer period1秒・hard observation deadline5秒、memory/AS/RSS/pressure/OOM/staleness/missing/identity/own-run停止とcleanupは保持する。
CPU4 workers[2,4,5,6]・driver16・observer18、内部thread1/Qiskit num_processes1、worker AS/RSS32GiB・driver8GiB・observer AS256MiB/RSS64MiB、headroom16GiB/admission152.25GiB不変。
host PSI1〜5%の限定30秒猶予、5%以上/全nonroot PSI/OOM等の停止条件も変えない。
H4 linear/STO-3G/DF12・6 geometry・218 templates・32 paired trajectories、structured Gaussian scientific/cost系列、compiler optionsと既存候補venvは保持する。
旧compilerとのoutput完全同一性は未検証のまま記録する。

## 検査・固定・起動

新wall wiringだけ12限定mock PASS、0.419秒/RSS32MiB、singleprocess/Python -P -B/thread1/AS256MiB/RSS128MiB/wall45s/output4MiB。
初回はfixtureでruntime headroom16GiBとadmission152.25GiBを取り違えたため既存条件に合わせて補正、全試行をprivate保存。
旧24再開campaignと人工compile/旧benchmarkは反復しない。追加transpile/実worker/科学array/affinityは検査で0。
SOURCE closure95のGit/checkout/SHA一致。time変更外の科学/資源/所有cleanupはAST一致をgateで要求する。
独立review `PASS_READY_FOR_FRESH_ONE_SHOT_CONTINUATION`、digest `1ca674085c33ae0b63b925dfc8167e806fa0bac5a3712834e4c3b43d9359fcbc`。
軽量reviewで残った旧storage descriptorを実2GiB trace・物理余裕・残余へ統一し、private launcherのverdict schema名を修正した。SOURCE変更は不要。
private launcher差分も独立PASS。launcherとrunnerで重いreceipt全照合を重複させず、fixed actual runnerでSOURCE/profile/input/carry→fresh→exclusive one-shotを一度確認する。
起動前エラーは所有済み8KiB FDへ保存し、外部console56KiB+diagnostics8KiBを予約64KiB内に保つ。

plan/auth/reviewと最終digestは固定済み。artifact commitをsource無変更で固定後、直前のCPU/memory/pressure/OOM/FS/inode/quota・未使用roots・one-shotを検査し、一度mapを起動する。
map完了またはfail-closed STOPで停止、automatic_retry=false、next_stage_authorized=false。実開始状態・PID・進行はhomeのprivate runtime receipt/observer logに別記録する。

## 公開

軽量source/tests/schema11fileをSOURCEへ、軽量profile/authority/plan/review/receipt要約/docsを別REVIEW_BUNDLEへ固定。
NPZ/実runtime/checkpoint/cache/raw native/log/credential/内部SSH情報はcommitしない。
non-force pushに認証が使えない場合は設定を変えず、以下を利用者の認証済みterminalで実行する。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-production-resume02-20261011 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-production-resume02-20261011
```
