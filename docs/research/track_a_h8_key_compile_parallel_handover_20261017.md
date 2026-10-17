# H8主要3構成6軸：並列準備への引継ぎ

利用者の「新方式に引き継いでほしい。引き継いだら元のRUNは止めてよい」に基づく実行記録。
これは稼働snapshotであり、登録6軸の最終完了報告ではない。

- 引継ぎSOURCE：`41e466e13f32e9224bb37790cc16f0ff6621c399`
- 新run：`attempt02-parallel-handover`
- 起動：2026-10-17 21:51:14 JST。新driverの待機・監視を確認してから旧所有launcherのみSIGTERM。
- 旧run：`attempt01`、SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`。
  21:51:45 JSTにnative launcher exit1/reaped。意図した切替STOPと全nested worker回収を確認し、元STOPを保持。
- 新driver PID `2830781`、launcher `2830747`、native-wait witness `2830746`。

## 再利用と残りの計算

B0I p15/q4 S4の制御検証済みcosine/sine QPYを合計33,905,425 bytes再利用。
保存端点・scalar・unrotated reference・preparation・profile・fingerprint・native exit・bytes/SHAを照合した。
切替直前に完了したcosineは **668,636 RZ、固定モデル誤差0.2169021283%でPASS**。
この実測は元SOURCEを保持し、再compileしない。残りは固定順の5軸。
中断したsineのcompile内部状態は再利用できず、新runで再実行する。

旧予約はcompile2、完了model check1を含め非返却で保存。
新actual呼出・再利用logical呼出・旧予約を別々に記録し、失敗分の返却や過去のSTOPのPASS化を行わない。
旧61証拠fileはrelease時および新run開始後の双方で一致した。

残るB0I p11/q1 S2とB0 p15/q4 S2を並行準備し、各構成の2軸workerも並列化する。
実装の準備能力は3構成/6 worker、今回の再利用後に必要な準備workerは最大4。
全3構成の準備が検証済みになってからcompileを1軸ずつ行い、各軸の相対誤差2%以上で次軸を開始せずSTOPする。
追加candidate・refit・費用再会計・次stageへは進まない。

## 固定条件と検証

Linear H8 1Å/STO-3G/DF28/16 modes/sector4900/T=.8。
固定3構成6軸、保存入力11、provider265、runtime34,273、SOURCE closure26を照合。
既存environment/compiler・科学条件・許容差を変更せず、venv install/upgradeは0。
独立した新実行域・SOURCE・sealed config/grant・one-shot lockを使用した。
新driverはREADYを保存して待機し、旧native終了・回収後のbinding済みatomic releaseまでは科学処理を開始しない。

CPUはdriver5、worker[6,8,9,10,11,12]。旧2/3/4とは重ならない別物理core。
fresh gateは実効空き約457.5 GiB、filesystem空き約485.9 GiB、inode約2.19億、quota条件に合格。
192 GiB＋headroom16 GiBはadmission用見積りであり、実測使用量やprocess上限ではない。
独自のCPU/RAM/AS/RSS/wall上限は設けず、OS制御を継承する。
監視は5秒間隔で継続し、一時的なPSIだけで科学workerを停止しない。

引継ぎ・byte破損・native退出・不正STOP・live identity・科学的不一致拒否・再利用によるdispatch抑止・
並列準備・nested cleanupの限定人工回帰 **39件PASS（2.247秒）**。人工compileも実H8 test compileも0。
主担当による差分確認であり、別担当の独立レビューとは扱わない。サブエージェント使用0。
実データ処理では複数の数値workerのCPU増加を確認した。実際の全体高速化率はまだ未確定。

## ログと証拠

private実行入口：
`/home/AbeHiromu/projects/h8-key-candidate-compile-execution/20261017/attempt02-parallel-handover/`

- `output/handover_receipt.json`：受入case/軸・旧予約・SOURCE binding。
- `control/observations.jsonl`：独立launcherによる継続観測。
- `output/parallel_preparation/`：残る2構成と各軸の準備log。
- `output/axis_*_measured.json`、`output/terminal.json`：軸判定と最終状態（未完成）。
- `metadata/launcher_native_exit.json`：終了時のnative wait（現在未作成）。

[軽量稼働snapshot](../../artifacts/resource_applicability/track_a_h8_key_compile_parallel_handover/2026-10-17/handover_running_snapshot.json)、
[SOURCE closure](../../artifacts/resource_applicability/track_a_h8_key_compile_parallel_handover/2026-10-17/source_closure_receipt.json)、
[人工39件log](../../artifacts/resource_applicability/track_a_h8_key_compile_parallel_handover/2026-10-17/artificial_tests.log)。

NPZ/QPY/raw/runtime/有効grantはGit対象外。旧run・他job・既存worktreeのSOURCE/grant/ledger/markerは変更していない。
次はCodexが登録6軸の完了または科学STOP後に結果を監査・報告する。費用再会計は別途利用者承認を必要とする。
