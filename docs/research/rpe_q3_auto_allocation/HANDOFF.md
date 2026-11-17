# 分散Q3：自動割当・自動同時枠と計時

基準は4ホスト統合 `e64e377a8a27e6693231bb4cdf72f605a20965b5`。
実装の固定SHAとfile hashは[SOURCE.json](SOURCE.json)、検査範囲は[CHECKS.json](CHECKS.json)。
計時担当の公開81884bd39d2829294e71be6848c1a3f024d5e75cをmergeし、指定された
19c43576c83ec9de24f6059f4cb39762458c6519、5897d26ab、1026c09b3645c1ef210def1f4678d79db2fda019を再利用した。
今回の実H4 Q3・信号生成・価格較正・回路compile・本番queue投入は0件。
関連57検査を一括PASSし、旧roundの暗黙1/明示2を維持する追加1検査と可変容量metricの追加1検査もPASS（計59の異なる検査）。
既存科学run・原結果・UNKNOWN・STOPと固定科学SOURCE28dは変更していない。

## 通常開始・再開

採択済みの**新しい**科学roundで、従来と同じ入口を使う。

```bash
python3 -B scripts/run_rpe_schedule_search.py \
 --problem /ABS/PROBLEM.json --round /ABS/APPROVED_NEW_ROUND.json \
 --run-dir /ABS/NEW_RUN
```

再開も同じコマンド・同じrun directory。ホスト名や毎回の同時件数を追加指定しない。
`executor.scheduler=concurrent`かつ`executor.distributed`がある新runでは、
max_concurrent_jobs省略時の既定がautoになる。明示する場合も`"max_concurrent_jobs":"auto"`。
通常の`--plan-only --prepare-readiness`が生成するDISTRIBUTED_OPTIONSもauto、
server2のlocal_workerと4ホストのslot_policyを含む。
科学予算・候補範囲・承認条件は従来のroundのまま。autoは新しい科学実行許可を与えない。

明示integer上限は保持する。新runの解決済みmode/上限/優先設定を
`distributed/ALLOCATION_SETTINGS.json`へ保存し、再開時も保持する。
導入前runの暗黙上限1もautoへ変更しない。稼働runへこのSOURCEを途中適用しない。

## 割当

- server2・server1・local（RomuTail）は同じ優先group。local0-wslは次のgroup。
- REQUESTの完全照合と現在のhost fingerprintに合う空き枠だけを使う。
  READY候補と空き枠の対応を求め、現在の未解決attempt数＋今すぐ実行可能なREADY数から全体枠を決める。
  host数4を全体上限4へ読み替えない。候補や枠が少ないときはその範囲になる。
- 優先側に2空き枠、WSLに1空き枠、独立READY3件なら優先側2件＋WSL1件。
  少数の仕事は空いた優先側へ残し、余剰分はWSLへ即時に渡す。
  満杯・Pause・未接続・実行不可・古い空き枠報告は待たない。
- relayは独立hostの状態を取得し、空き枠とreadinessを中央へ通知してからclaimする。
  WSLが先に直接pollした場合も、最初の有界なdiscovery期間後は応答しないhostを待たない。
  既知の優先枠で処理できない余剰は、このdiscovery待ちの対象にしない。
- 新runの中央設定はWSLの参加を許可するが、WSLはrequired_hostsへ含めない。
  Windows参加flagの明示Pauseやrun固有の中央受付OFFを尊重し、OFFを参加待ち・全枠busyへ数えない。
  既存runのHOST_ADMISSIONやWindowsの参加flagは今回変更していない。
  既に所有するjobは元hostで完了・回収する。後から空いた優先hostには次のjobを渡す。
- 元の候補順、条件付きlookahead、科学job key、budget予約、MSE判定、完了採用経路を維持する。

## ホスト別枠

新profileは`slot_policy={"mode":"auto","initial_slots":1}`。WSLも検収済み1枠から始める。
最初の1枠は、実H4の最適値やメモリ使用量を実測済みという意味ではない。
既存fixed profileのslotsや、明示したmax_slotsは上限として保持する。

CPUはprocess affinity、cgroup、scheduler allocation環境変数、明示研究割当の境界を読み取る。
マシン全CPU数や全体の空きCPUを研究専有枠として数えない。
メモリも割当境界と現在のheadroomを分け、所有supervisor/子processのRSS・CPUを観測する。
他ユーザーのprocessを列挙・変更せず、OS・cgroup・scheduler設定を変更しない。
DP並列数は実際のroute profileを使い、数値SOURCEやDP幅を勝手に変更しない。

測定済みjob RSS、CPU/DP幅、READY、中央へ戻ったidentity照合済み結果数、採用数、
選択jobのdependency waitを使い、資源に余裕があれば1枠ずつ試す。
1.5倍のRSS headroomはsamplingによる過小評価に備えた探索用heuristicで、数学的メモリ上界ではない。
実績不足と、throughput/待ち時間を使えた段階を分けて記録する。
追加枠で有用結果のrateまたは判断待ちが改善しない場合は元へ戻す。
待ち時間が欠測なら、その不足を明示してthroughput観測と区別する。

減枠は新規投入だけを抑え、受領済みjobを停止しない。
AUTOではUNKNOWN・完了転送待ちも未解決枠を占める。
`SLOT_STATE.json`、`RESOURCE_OFFER.json`と既存WORKER_EVENTSの
`SLOT_CAPACITY_DECISION`へ、実枠数・段階・根拠を保存する。
中央は`AUTO_CAPACITY`と計時のQUEUE_WAIT_STATEを記録する。
利用率は枠数の変更履歴から利用可能slot秒を積分する。復帰中jobがあり最初の新しい容量報告がまだない区間は、起動時容量が残り得るため利用率の比較対象にしない。
割当制御はこの集計値を使わず、未解決attemptと実際の空き枠を数える。

## 人工検収と実測範囲

[通常入口の結果](checks/entry/ACCEPTANCE.json)と同directoryのTIMINGを参照。
実際の人工adapter subprocessを動かしたが、4つのhost名は**同じ1 OS上の模擬host**。
実4台の新しい科学実行や、実H4での最適枠数の測定ではない。

| 人工4host run | server2 | server1 | local | local0-wsl |
|---|---:|---:|---:|---:|
| 完了・採用件数 | 3 | 6 | 6 | 3 |
| 観測したhost枠 | 1 | 1 | 1 | 1→2（探索的trial） |

WSL途中参加runも各hostの枠1で18結果を完了。auto4の全体枠は1〜5、人工body同時実行peak5。途中参加runは全体枠1〜4・peak4で、
WSLと優先側のbodyが実際に重なった。4hostだから全体4という固定値ではない。
増枠は探索的trialであり、実H4のthroughput最適化が確立したという主張はしない。
減枠・drainと資源制約変化は人工resource unit、実入口では増枠・Pause/Resume・途中参加を確認した。

単機参照・auto4・途中参加で各18結果、exact MSE上下界・判定・方式別最安は一致。
同run再開の新規LAUNCH、二重ASSIGN・ACCEPT・ADOPTは0件。
単機参照は人工sleepを省いているので、壁時間から高速化倍率を算出しない。
既存4方式×4hostの実入力16 PASSは[READINESS_REUSE](READINESS_REUSE.json)で再利用を照合し、再dry-checkしていない。

## 計時

通常run directoryに、そのまま次を実行できる。

```bash
python3 -B scripts/summarize_rpe_q3_timing.py /ABS/RUN_DIR --output /ABS/TIMING.json
```

人工auto4・途中参加の同じrunにこのCLIを実行し、本体18件・探索採用までを取得した。
時計の異なるhost間の絶対時刻差は取らず、計時担当版のsame-clock span・包絡・欠測区別を維持する。
受付OFFのWSLにより参加待ちや全枠busyを作らないことも検査した。
詳細な区間定義・旧保存ログとの違いは[計時HANDOFF](../rpe_q3_timing_20261117/HANDOFF.md)。

## 次run向けの配置とprofile

4hostすべての**transport**と、Windows local0のrelayを今回の固定共通SOURCEへ更新する。
科学checkout28d、route views、内側PR-DF SOURCE27e、入力、Python環境、旧spoolは保全する。
旧workerも固定枠の互換経路では動くが、新しい資源自動調整と新しいbody計時の実装とは区別する。

Q3本体の計時には、実際のadapterも更新する。新しい共通checkoutを
`adapter_root`にして、次の3fileを固定adapter_manifestへ含める。

- scripts/run_rpe_optimizer_q3_job.py
- src/trottertracks/rpe_dp_prefix_reuse.py
- src/trottertracks/rpe_q3_timing.py

transportには同じSOURCEのscripts、src/trottertracksが必要。
特にrpe_q3_worker、rpe_host_profiles、rpe_host_eligibility、rpe_plan_readiness、
rpe_q3_resources、rpe_q3_timingを古いcheckoutへ戻さない。
新しいcomputation rootを使う場合はhelperを含むmanifestになるが、元の28d rootは元の8file manifestを維持する。

[生成profileと全host配置表](next_run_profiles/DEPLOYMENT.json)を同梱した。
各hostのrouteごとの検証済みPython・DP/cacheと元のroute pathsを保持し、
新しいtransport/timed adapterを明示した。27e内側SOURCEのmapは、元DFの実機fingerprintの既存pathを追加した（TIMED_PROFILE_BASE.json）。
科学checkoutを編集したりsymlinkを作ったりしていない。host/Pythonの毎回の手選択は不要。
同じ生成を行うコマンド：

```bash
python3 -B scripts/prepare_rpe_auto_profiles.py \
 --profiles docs/research/rpe_q3_auto_allocation/timed_profile_base \
 --source FIXED_COMMON_SOURCE_FROM_SOURCE_JSON --run-id NEW_APPROVED_RUN_ID \
 --output /ABS/NEW_PROFILE_PACK
```

server2の固定checkout配置も、server1・RomuTail・WSLのremote配置も今回実行していない。
既存許可済み経路（local0のqg2/qg1/RomuTail、WSL実行）で新しい専用directoryへ置く。
Windows relayのworker.rootは新transport、worker.profileは生成profileを指定する。
server2のexecutor.distributed.local_workerにも、生成server2 profileの内容を設定する。
配置表のworker項目を使えば、transport PythonとQ3実行Pythonを取り違えない。
WSLの明示Pauseは保ったまま配置できる。

**新timed adapterへのprofile変更に、旧adapterの16 PASSを転用しない。**
元16セルの科学入力・route/runtime確認は再利用できるが、配置後の新adapter/profileについて
通常のcheck_rpe_q3_host_readiness.pyを実入力dry-checkし、返送READINESSを中央policyへ結合する。
専用Python launcherを使うDFは、新しいprofile/READINESS hashに結び付いたHOST_METHOD_READINESSも更新する。
checker最上位のPythonを実adapterのPythonへ転記せず、command[0]とadapter_result.runtimeを照合する。
そのREQUESTでPASSしたhostから実行可能とし、1台の準備不足で他hostを止めない。
今回は新timed adapterを用いた実H4のremote dry-check・科学Q3は未実施。

## 実H4で今後記録するもの

方式別最安費用の推移、identity照合済みQ3結果/時、採用結果/時、次候補の判断待ち、
host別のslot試行/維持/縮小とCPU・RSS、DP cache HIT、計算・queue・配送・回収・検証・採用時間を残す。
冷始動/入力配置/初回importと定常jobを分け、採用されない先読み完了も別に数える。
同じ問題・候補規則・科学予算・SOURCE/inputで比較できる単機ログがなければ速度倍率を出さない。
実H4の資源上限や最適枠、実4台の総合性能は次の採択済みrunで測る。
