# H8 signal/reference・8ケースcompile事前検証の資料入口

2026-10-13 JST。H8 signal/reference・native16modeの実検証は完了。8case/20compileの実測は進行中。本格H8比較は未認可・未起動。

2026-10-13 13:47 JST受領監査追記：[監査JSON](saved_evidence_audit_v1.json)、[20回路の完了/欠測一覧](compile_inventory_v1.csv)、[stdlib監査script](audit_saved_feasibility.py)。run `heavy01`の保存manifest・SOURCE・one-shot lock・開始ticks・UID・affinityを照合し実行中を確認。8signal/1164probeは再実行しない。SOURCE245件・入力5fileのstreaming SHA・native runtime318件・lowering source一致、監査不一致0。

監査時点で完了wrapper0/20、actual transpile開始記録0、S4/q8のcontrol検証中。evolution892121 instructions、経過3211.45秒（53分31秒、完了wallではない）、保存HWM5654446080 bytes（5.266GiB）/VmPeak6134353920 bytes（5.713GiB）、RSS24GiB guardの21.94%/AS32GiB guardの17.85%。初回hashの記録所要1037.26秒、controlは観測時点で2161秒以上・未終了、nested hashを重複加算しない。RZ/CX/depthは未取得/null、欠測を0や予測費用に置き換えない。残7caseは未dispatch、最大3worker実績なし。これらから本格campaign完了時間やnormal/heavy分類を確定しない。

既存監視は稼働し、STOP理由null・pressure0・新OOMなし。SOURCE/CPU/RSS/AS/compiler/候補を変更せず、新science runner・retry・resumeを起動しない。ユーザーが「処理が開始したらチャット終了可」と指定したため、今回の報告は進行中snapshot。20compile全件終了後の費用/最終peak/exit/reap/残存0/累積counter監査と実測予算更新は未完了。終了時にはprivate `FINAL_RECEIPT.json`または`PIPELINE_STOP.json`とnative terminalを確認する。coordinator正確exit code/時刻が記録されていなければnullを保持し、成功terminalから捏造しない。

本格計画の科学方針は利用者がGPT採択済みresource-map（rank28、prefix7/14/21、q1/2/4/8・r1/2/4・K2/4、full PF S2/S4、finite-time complex-signal/測定込みRZ）を指定。transferは副評価の別枠。20compile完了後に技術的資源見積もりを更新する。本格実行grantは未取得、launchしない。

- [実行契約・科学条件・実装・予算・STOP](../../../../docs/research/track_a_h8_signal_compile_preflight_v1.md)
- SOURCE `5c207fb874083022e5d48d69c879d0bcb40f894c`。旧runtime238件不変、新closure245件。人工15検査PASS、実transpile test0。
- 固定snapshot1,299,688 bytes/SHA `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`、rank28/sector4900。SCF/DF/state再生成0。
- signal scope起動：coordinator PID324393、worker CPU30/co31。旧H4 productionを保持しCPU重複なし。
- [実signal結果](signal_summary.json)、[reference](input_reference.json)、[native全登録検査](primitive_validation.json)、[native終了・cleanup](signal_terminal.json)、[現在の状態とfile bytes/SHA](result_status.json)。signal8/8、388時間点×3probe=1164、native/spin差9.52e-15・spin/CSR差9.75e-15、signal差6.04e-14以下・vector差4.55e-13以下。wall261.436秒、aggregate peak643162112 bytes、exit0/cleanup0/STOPなし。経験的検査でground-stateや数値certificateではない。
- 最初のS4/q8単独実測：coordinator PID327235、worker PID327489、CPU30/co31。最初のnative evolution892121 instructions。AS32GiB/RSS24GiB、内部thread1、hash再利用、.2秒独立監視/5秒fail-closed。実wrapper compile完了と最終peakは未確認。
- continuation PID329282はheavy完了・reap・binding後だけ残7ケースを最大3workerで起動する。20compile/4trajectory/15occurrenceの範囲内、失敗時STOP・retry0・本格H8進行0。private `CONTINUE_PREFLIGHT.py`、`pipeline02.log`、完了時`FINAL_RECEIPT.json`と`wrapper_metrics.csv`。新grant/outputを分け、最後にscope合計を再照合する。
- private run/log入口：`/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-signal-compile-preflight-v1/`。NPZ/runtime/cacheはGitに含めない。

20compileの完了・必要メモリ確定・本格H8 readinessはまだ主張しない。full-map technical seal/grantは未作成、absolute command=null・allowed_cpus=[]。前段階で安全な並列枠を最大3とし、全ケースの実測から本格計算のworker数を提案する。

公開監査：originのREVIEW `806222e34b0ea399ac2f51559966a157bc0acde1`とSOURCE到達可能性を確認し、変更30fileをfetch/Git blobのbyte/SHAで照合済み。[公開証跡](publication_verified_806222e3.json)。Git CLIのHTTPS認証は不足だが、既存GitHub接続にはpush権限がある。認証情報・鍵・helper・Git global configを変更せず、既存接続だけで監査資料をnon-force公開する。人工test結果は[15 tests](synthetic_test_result.txt)、旧runtime238件不変は[監査](prelaunch_review.json)。NPZ・科学array・cache・実runtimeのcopyは公開しない。
