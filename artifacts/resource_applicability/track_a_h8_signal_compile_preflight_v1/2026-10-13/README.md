# H8 signal/reference・8ケースcompile事前検証の資料入口

2026-10-13 JST。H8 signal/reference・native16modeの実検証は完了。8case/20compileの実測は進行中。本格H8比較は未認可・未起動。

- [実行契約・科学条件・実装・予算・STOP](../../../../docs/research/track_a_h8_signal_compile_preflight_v1.md)
- SOURCE `5c207fb874083022e5d48d69c879d0bcb40f894c`。旧runtime238件不変、新closure245件。人工15検査PASS、実transpile test0。
- 固定snapshot1,299,688 bytes/SHA `d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955`、rank28/sector4900。SCF/DF/state再生成0。
- signal scope起動：coordinator PID324393、worker CPU30/co31。旧H4 productionを保持しCPU重複なし。
- [実signal結果](signal_summary.json)、[reference](input_reference.json)、[native全登録検査](primitive_validation.json)、[native終了・cleanup](signal_terminal.json)、[現在の状態とfile bytes/SHA](result_status.json)。signal8/8、388時間点×3probe=1164、native/spin差9.52e-15・spin/CSR差9.75e-15、signal差6.04e-14以下・vector差4.55e-13以下。wall261.436秒、aggregate peak643162112 bytes、exit0/cleanup0/STOPなし。経験的検査でground-stateや数値certificateではない。
- 最初のS4/q8単独実測：coordinator PID327235、worker PID327489、CPU30/co31。最初のnative evolution892121 instructions。AS32GiB/RSS24GiB、内部thread1、hash再利用、.2秒独立監視/5秒fail-closed。実wrapper compile完了と最終peakは未確認。
- continuation PID329282はheavy完了・reap・binding後だけ残7ケースを最大3workerで起動する。20compile/4trajectory/15occurrenceの範囲内、失敗時STOP・retry0・本格H8進行0。private `CONTINUE_PREFLIGHT.py`、`pipeline02.log`、完了時`FINAL_RECEIPT.json`と`wrapper_metrics.csv`。新grant/outputを分け、最後にscope合計を再照合する。
- private run/log入口：`/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-signal-compile-preflight-v1/`。NPZ/runtime/cacheはGitに含めない。

20compileの完了・必要メモリ確定・本格H8 readinessはまだ主張しない。full-map technical seal/grantは未作成、absolute command=null・allowed_cpus=[]。前段階で安全な並列枠を最大3とし、全ケースの実測から本格計算のworker数を提案する。

Git pushはHTTPS認証不足で失敗、設定変更なし。SOURCE/REVIEWはローカル固定し、ユーザーの手動push待ちとする。人工test結果は[15 tests](synthetic_test_result.txt)、旧runtime238件不変は[監査](prelaunch_review.json)。NPZ・科学array・cache・実runtimeのcopyは公開しない。
