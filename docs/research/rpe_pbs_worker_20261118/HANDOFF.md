# PBS計算先：運用引継ぎ（2026-11-18 JST）

後続：中央・価格branchを実mergeした[合流版HANDOFF](../rpe_pbs_central_integration_20261118/HANDOFF.md)。
最終合流SOURCE925985b4eの実PBS検収・登録はそちらを参照する。以下は旧PBS単独版22の記録である。

実装・配置・保存実入力dry-check・短時間の実PBS人工検収を完了した。
結果は[REPORT](REPORT.md)。実H4信号・回路compile・Q3本体は未実行。
別途採択された探索指示を次の科学実行条件とする。

## 通常入口とrelease

RomuTailの通常入口は`scripts/run_rpe_schedule_search.py`、通常gatewayは
`.rpe-central/SERVICE.json`→既存`.rpe-central/RELEASES.json`。
新releaseは`pbs-22a428bd3`。controller/PBS接続/workerは固定
`22a428bd34cbc3705b90ca6f4407053a6d996f05`、科学Q3 SOURCEは従来の`28d976b99`。
run再開は`CONTROLLER_CONFIG.json`に保存した元release/SOURCE/input/設定を選ぶ。
元のrunへ新releaseを途中適用しない。

作業中にcurrentは別担当の`romutail-6da8554e0-r4`へ更新された。
PBS releaseは追加登録済みで、currentや既存runを切り替えていない。
新しいPBS対象探索は明示的に`-Release pbs-22a428bd3`で選べる。
現在のrelease・接続contractの照合は[REGISTRATION](deployment/REGISTRATION.json)と
[CENTRAL_COMPATIBILITY](deployment/CENTRAL_COMPATIBILITY.json)。

## local0から開始・停止・再開・状態確認

利用者はlocal0のCodexへ範囲・開始・停止・再開を指示する。
既存の移行済み`tools/local0-search/Invoke-RpeSearch.ps1`とRomuTail向けCLIENT設定を使う。
採択済みの探索scopeを保存したREQUESTを使い、通常はプログラムがqsub/qstat/回収を行う。

```powershell
.\Invoke-RpeSearch.ps1 -Action Preview -Release pbs-22a428bd3 -RequestFile .\APPROVED_REQUEST.json
.\Invoke-RpeSearch.ps1 -Action Start -Release pbs-22a428bd3 -RequestFile .\APPROVED_REQUEST.json -Instruction '採択済み探索指示の本文'
.\Invoke-RpeSearch.ps1 -Action Status -RunId '返されたrun_id'
.\Invoke-RpeSearch.ps1 -Action Stop -RunId '返されたrun_id'
.\Invoke-RpeSearch.ps1 -Action Resume -RunId '同じrun_id' -Instruction '同じ採択範囲を再開する指示'
```

これは手順例で、上のStart/Resumeを今回実行したという意味ではない。
既存run再開時に別のReleaseを指定しない。科学的STOP/UNKNOWN/one-shot等の条件は通常入口が保持する。
既存client bootstrapは`release`選択を解決してから固定controllerをimportするため、PBS専用の別入口は不要。

RomuTailから直接操作する場合は、同じschemaのREQUESTに`action`/`run_id`/`release`を設定して次へ渡す。

```bash
python3 .pbs-source-22a428bd3/scripts/run_rpe_local0_search.py \
  --config '/home/abe/Project/Partially Randomized Trotter/.rpe-central/SERVICE.json' \
  --request /ABS/REQUEST.json
```

既存gatewayのStartはcontrollerを独立sessionで起動する。
そのcontrollerがPBS managerを所有し、SSH再接続と要求/成果の配送を継続する。
チャット、Codexの監視、local0のPowerShellが終了しても、開始済みcontroller/PBS jobの継続に依存しない。
停止は新規受付を止め、受領済みtaskをdrainする。待機/保留割当は、所有taskがないことを確認してqdelする。

## 接続と配置

| 項目 | 配置・役割 |
|---|---|
| SSH alias | RomuTailの`prism`→`akeno`→`AbeHiromu@192.168.5.26` |
| PBS管理host | `CPUcluster-PRISM`、PBS20.0.0。workerは管理hostでなく割当計算nodeで起動 |
| 専用root | `/home/AbeHiromu/projects/rpe-pbs-20261118` |
| transport | `releases/22a428bd3`、gatewayはsystem Python3.6互換 |
| worker/工程Python | `runtime/venv311rc1/bin/python`、Python3.11.0rc1と保存契約のlibrary |
| 科学SOURCE/input view | `readiness/views/<完全commit>`、`readiness/objects.git` |
| 保存実入力/runtime補助入力 | `readiness/payload`、`readiness/runtime-inputs` |
| runごとの要求/応答/receipt | `runs/<RomuTail run pathのdigest>/allocations/<submission token>` |
| worker spool/成果/DP cache | 同じrun namespaceの`workers/<host_id>`とprofile指定cache |
| node一時領域 | PBSの`TMPDIR`。必須成果の唯一の保存先にはしない |

実パスは[production profile](readiness/PRODUCTION_PROFILE.json)、[profile layout](readiness/PROFILE_LAYOUT.json)、
[配置](deployment/DEPLOYMENT.json)で解決する。特定計算node名は設定していない。
元receipt/REQUEST中の歴史的絶対pathは書き換えず、host profileのSOURCE/input/path mapでアクセス先を解決する。
管理host↔計算nodeの共有homeは実確認済み。計算node→RomuTailの直接通信は不可だったためspoolを採用した。
spoolを管理するgatewayは標準出力返送時期に依存せず、入力・成果CASは不足hashだけを移送する。
別hostのaudit/Q3も同じ元artifactを取得し、hash照合して引き継げる。

## 工程・資源・終了

Q3、signal、compile、audit/集計は既存共通queue/worker経路を使う。
保存実入力は4方式Q3の4件と準備15件がmetadata PASS。
準備inspectionは将来の科学REQUEST全体の実行資格ではなく、新REQUESTは通常の工程別profile/fingerprint/preflightを通す。
PR-DFもprofileから専用Python、内側SOURCE、入力を解決する。方式専任hostは設けない。

通常設定は`executor.distributed.pbs`、中央の工程profile登録は`preparation_profiles['pbs-0']`。
設定例は[deployment/PBS_SETTINGS.json](deployment/PBS_SETTINGS.json)。
通常queueはworkq、大メモリ用はlarge。キュー、ncpus、mem、walltimeを明示する。
配置済みテンプレートは1node/96CPU/350GB/24時間を記すが、現在の科学予算は全て0、`task_seconds={}`である。
この資源例を実H4で検証済みの最適値と扱わず、採択指示に合わせた新releaseで確定する。
PBSの有限walltimeは1割当の寿命であり、研究全体のwall-time capを追加するものではない。

起動時に固定transport、科学SOURCE、工程runtime、実`qstat Resource_List`を照合する。
nodefileの行数やマシン全96 CPUを割当CPU数と読み替えない。
workerは実CPU/memとprofileのDP並列数×library thread数、task CPU/memory指定を同じpoolで会計する。
auto枠はこの上限内で調整する。Q3とcompileの二重予約はしない。
現時点で実H4のmemory/最適枠は未測定なので、採択された条件に必要な見積りを設定する。

各工程の保守的なtask時間を`task_seconds.q3/signal/compile/audit`へ設定する。
未設定の科学工程はPBSの新規割当/受付対象にしない。
残時間からdrain余裕を除いて入らないtaskを受け付けず、空きtaskがなくなれば短いidle猶予後に解放する。
PBS終了でexit receiptがない元attemptはUNKNOWN、automatic_retry=false。別candidate/namespaceでSTOPを迂回しない。

PBS割当追加は、PBS対応READY task、他hostの実空き枠、既存PBS割当の見込み枠・稼働状態から判断し、
要求資源と理由を`distributed/pbs/EVENTS.jsonl`へ記録する。
Q/H/SUBMIT_PENDINGは実行可能slotに数えない。科学予算は元task単位で、worker起動はQ3件数0。

## 切断・待機・途中終了からの復帰

RomuTailの`RUN_DIR/distributed/pbs/{SETTINGS,STATE}.json`がrun固定設定・job対応を持つ。
PBS job ID、worker host、task/attemptは元JobStoreとcluster側SPEC/receipt/spoolに対応する。

qsub応答欠落時は保存intentの同tokenを照会し、receiptまたはPBS上のjob名と照合する。
未確認の投入を新tokenで再投入しない。確定できない場合はSUBMIT_AMBIGUOUS_HOLDを残す。
controller再開は同じrun directoryで、Q/H/Rの既存jobを追跡して未配送成果を回収する。
結果採用と予算消費は既存JobStoreのidempotent処理で、完了taskを再実行しない。

PBS履歴から消えたjobは、保存job ID、worker STATE、exit receipt、成果を照合する。
履歴がないことだけで再投入しない。worker終了が確定するまでは元taskを実行中/不明のまま保つ。
確定終了後の未完taskはUNKNOWNとして元の科学的retry/resume契約に従う。
標準出力が未返送でも共有領域のtask状態・成果を回収できる。

状態照会例（job IDは当該runのSTATEから取得）：

```bash
ssh prism '/opt/pbs/bin/qstat -xf -F json 267194.CPUcluster-PRISM'
```

PBS20では`-u`と`-F json`を併用するとtext表が混ざるため、実装は既知の自分のjob IDを個別照会する。
識別応答欠落時だけ履歴JSONから自分のjobを抽出する。他ユーザーjobを操作しない。
保存成果の読み取りは、clusterの`rpe_pbs_gateway.py --root <そのrun namespace>`へ
`{"action":"snapshot"}`または`{"action":"harvest","token":"その保存token"}`を1行JSONで渡せる。
harvestは新たなworker/科学taskを起動しない。通常の稼働controllerはこの回収と元結果採用を自動で行う。

## 固定版の配置・中央移行版への取込み

新SOURCEは既存viewを編集せず、専用releaseへ不足Git objectだけを配置する。

```bash
python3 scripts/prepare_rpe_pbs_readiness.py \
  --stage-commit FULL_FIXED_SOURCE \
  --destination /home/AbeHiromu/projects/rpe-pbs-20261118/releases/NEW_SOURCE \
  --output /ABS/OWNED_PLACEMENT_RECORD --ssh-alias prism
```

`prepare_rpe_pbs_release.py`は既存SERVICE、PBS profile/settings、固定source rootから新しいSERVICE/ROUNDを作る。
科学予算・science_approved・既存SOURCEを増やさない。`rpe_release_registry.register`で新IDとして追加する。
run開始後は保存releaseを再利用し、更新したSOURCEをresumeへ押し込まない。

別担当の中央管理branchへ取り込む場合は専用branchをmergeし、
DistributedExecutorのRPC開始後のPBS manager起動、finish/metrics、stage matcher/deadline gate、
source_manifestのPBS補助file追加を保持する。JobStoreや科学leafを別実装へ置き換えない。
current6daでは方式viewのowner finish除去が既にあるため1つだけ残す。
current側のWindows SSH/WSL輸送修正、stop時の`None`返却、runtime状態表示も維持する。
同じ8個のqueue/worker/profile等のprotocolと接続signatureを照合済みだが、
6da branchと本branchをmergeした新SOURCEそのものは今回の実PBS検収対象ではない。
merge後は新固定版として技術検査し、新releaseへ配置する。旧SOURCEの検収結果へmerge版の名称を付け替えない。

## 次に必要な設定と担当

利用者が実H4探索の範囲・入力/SOURCE・科学工程・予算・STOP条件を別途採択する。
Codexがその固定範囲で新release/ROUND、PBS資源、工程別時間/memory、完全REQUESTのhost適格性を設定して通常入口から起動する。
Q3だけの指示から新信号やcompile許可を推定しない。保存dry-check PASSを新候補の実測結果と扱わない。
実H4の科学的優劣、通常queue待ち、最適slot、科学body計時は今後の採択された探索で測定する。
今回の人工fixtureや保留/切断のfault hookを本番探索へ有効化しない。
