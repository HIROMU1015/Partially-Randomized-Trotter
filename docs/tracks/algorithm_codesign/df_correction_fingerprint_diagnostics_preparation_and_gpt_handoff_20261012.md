# DF補正比較：fingerprint診断source準備・GPT差分確認

2026-10-12 JST。利用者の認可は診断保存、登録外control、focused tests、公開まで。
**DIAGNOSTIC_SOURCE_PREPARATION_COMPLETE / mandatory STOP**。
旧one-shotの再実行、registered E/CD評価、T合成、修正algorithm採択は行っていない。

## 正本・今回の変更

- 原因監査の基点：`86e5c43d17f8f4ba2e232776b5e4beb7e5db3a98`。
- 旧science source S'：`41349b73157c7781e0b534cf0fe61e06bf439954`。
- 旧authorization-only A：`a1f537b68f0f22419dff0202c553b7511653caa3`。
- 旧結果・実行監査P：`ce0e286430a2a23869cb9417c82113c98064c72c`。
- 新準備branch：`track-b-df-correction-fingerprint-diagnostics-preparation-20261012`。
- 旧分類：`TECHNICAL_INCONCLUSIVE`、消費済みmarkerは不変。

変更sourceはB専用namespaceの[fingerprint_diagnostics.py](../../../src/trottertracks/algorithm_codesign/df_correction_comparison/fingerprint_diagnostics.py)、
[native_evaluation.py](../../../src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py)、
[worker.py](../../../src/trottertracks/algorithm_codesign/df_correction_comparison/worker.py)。
共通`src/trotterlib`のPython sourceとAPIは変更していない（案内READMEのみ追記）。

新adapterは共通の`_circuit_payload`を一回だけ取得し、共通と同じsorted JSON encodingからhashを作る。
共通のcontext validation、compiler hash、gate/depth helpers、cost dataclass、transpilerを再利用する。
no-backendの既存費用測定経路をB側の小さなadapterに展開し、例外を観測できる位置を追加した。
実装重複の範囲はこのcost assemblyであり、共有moduleのcopyやruntime monkeypatchは行わない。
共通helperの`TypeError`/`ValueError -> actual=None`を保ち、それ以外は記録後に再送出する。
off-domain real compileの共通helper対照で全cost field・transpiled circuitの一致を確認した。

旧/new primitive、commutator、expm、Pauli proxy、sampling、accuracy・confidence・error会計、
compiler settings、expected fingerprint、200000 native instruction guard、22 compile/64 T-key capは変更していない。
旧float DAGへ戻す、名前をhashから除く、semantic guardへ切り替える、期待hashを更新する等は未採用。

## 保存する観測値

whole Aの各controlled状態に専用`*.whole-fingerprint-0/1.json` logical artifactを用意する。
本準備ではregistered workerは起動しておらず、この保存機能は未来の承認済みworker向けである。

1. fingerprint前：frame、Q、T、raw U、input targetのdtype・shape・strides・ownership・flags・C-order SHA256・全raw bytesのhex。
2. fingerprint後：actual（None含む）、expected、policy、同じ一回のrecursive payloadとdefinitionのpath/hash索引、生成例外のtype/message/stack。
3. transpile前／返却後：attempt数、compiler settings hash、instruction数、actual/expected一致状況。
4. worker callback：source、authorization、contract、marker identityを診断資料へ結び付ける。

diagnostic saveは同期的に行い、その後に既存のinstruction／fingerprint guardが判定する。
保存不能ならfail closed。診断のためにfingerprint生成やcompileを再試行しない。
例外時に取得できなかったrecursive payloadは`INCOMPLETE_NOT_REEVALUATED`とする。
取得不能な途中definitionを取得済みと表現しない。

既存lossless chunk storageを使い、各review fileは512 KiB以下。
診断用arrayは登録local最大128×128 complex128のbytesに制限する。
追加出力・一時ファイル・観測CPUは既存supervisorのoutput/log/CPU/RSS/wall会計へ含まれる。
追加観測により既存capへ達する可能性は残る。capの拡大を本準備で認可しない。

## 登録外controlでの観測

事前[準備scope／test plan](../../../artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/preparation_scope_and_test_plan_v1.json)：
固定analytic 2-mode rank1 frame三つ×2 control、theta=1/19,1/23、RNG0。
旧helperの三関数だけを元ASTから実行し、旧main・registered recipesを呼んでいない。
newは現行certified primitiveとframe hex roundtripを使用した。
controlled/uncontrolled入力回路を組み立て、canonical recursive payloadとstrict Operatorを比較した。
これらのwhole回路をnative transpileして費用評価する作業は行っていない。

両controlでQ・T・Uのbytesは一致した。targetの全bytesも一致し、strict operator残差のpoint診断は0。
uncontrolled fingerprintは一致したが、**controlled fingerprintは両controlで不一致**だった。
[保存payload差分](../../../artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/off_domain_payload_difference_audit_v1.json)では、
各controlの差はrecursive definition内の4箇所の`operation.name`だけで、
`circuit-309`対`circuit-379`等の自動生成名だった。
raw matrix・角度・global phase等の他のpayload fieldに差はなかった。

従って、**このoff-domain実装contextでは、同じ行列・演算子でもrecursive自動名によってhashが異なる例を確認した**。
入力fingerprintの不一致だけから物理的誤りを結論することはできない。
ただし、登録E-r3はuncontrolled段階で停止しており、当時のactual/例外が未保存である。
今回の2-mode controlled例を、その停止の直接原因と断定しない。
registered raw Q/T/Uの旧/new一致、None fallbackの発生、historical runtime差は引き続き未確定。
この観測はguardの変更採択でも、A/Bの資源的優劣でもない。

## focused testsと初回失敗の扱い

[新tests](../../../tests/tracks/algorithm_codesign/test_df_correction_fingerprint_diagnostics.py)16件と既存88件を対象にした。
matrix lossless bytes、recursive payload hash、None/例外保存、no retry、failure-before-guard、
保存失敗時STOP、chunk破損検知、real shared-cost対照、strict controlled phaseを検証した。

最終[run03 receipt](../../../artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/focused_tests_v3.json)：
**104 PASS、0 failure/error/skip**。wall 3.085 s、parent/reaped CPU 2.763 s、sampled aggregate peak RSS 410.449 MiB。
32 logical CPUとavailable memory約51.7 GiBをread-only確認し、small matrixには1 worker・numeric threads1を選んだ。
GPU query/use0、speedup比較は行っていない。
固定numeric環境のversion/executable/code・binary tree identityは既存contractと一致した。
本test harnessはCPU90 s、aggregate RSS512 MiB、child448 MiB、output8 MiB、log1 MiB、wall capなし。
これは準備testsの上限であり、凍結science contractの上限は変更していない。

初回は97 PASS/7 FAIL。新16件はすべて通過したが、既存testのmarker-absent仮定と、
外側RLIMIT_FSIZE8 MiBが内側既存guardの64 MiB設定を禁止するharness問題があった。
run02は103 PASS/1 FAILで、metadata childに`PYTHONPATH`が渡っていなかった。
markerのbefore/after bytes不変確認へtestを更新し、外側harnessの継承用RLIMIT_FSIZEだけ64 MiBへ合わせた。
実aggregate出力はperiodic8 MiB guardで維持し、child importsをこのworktreeのsrcへ固定してrun03を通した。
source修正は本番guardではなくtest/harnessの範囲。
両失敗runのreport/XML/log/controlに加え、当時のharness/test sourceを記録hashと完全照合して保存した。
この3回はoff-domain focused-test invocationであり、旧科学one-shotのretryではない。
real off-domain compileは各runで3回（既存control1＋adapter/shared対照2）、本準備合計9回。
registered CD/E compile、registered frame生成、Clifford+T、分子/NPZ/GPU/LP/trajectory/shots/G10/RA-D0は0。
local CI実行結果ではなく、source-bound local focused validationである。

## 次のGPT確認とSTOP

新[source snapshot](../../../artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/source_manifest_diagnostics_v1.json)と
[静的・保存値provenance](../../../artifacts/track_b_df_correction_fingerprint_diagnostics_preparation/2026-10-12/preparation_audit_v1.json)を確認対象にする。
本branchの既存2 module・1 testの変更と新source/test/driverを新manifestで識別し、旧manifestを上書きしていない。
旧実行worktreeのsource・結果・authorization・markerは不変。

**RUN_READY=false、science execution unauthorized**。
旧launchは消費済みresult path・旧manifest・旧authorizationのままで、新sourceからの実行を拒否する。
新manifestを旧authorizationに結び直す、markerを消す、現期待値を更新することはしない。

GPTへの確認事項は、診断adapter差分の妥当性、保存controlの自動名差の解釈、
登録inputで診断付き新one-shotが必要かというscope判断である。
選択する修正は引き続き未採択。
将来実行を選ぶ場合は、新launch/source manifest・新result namespaceの準備と差分review、
その固定sourceの直接子authorization-only commit、別の明示one-shot認可が必要。
本資料の公開はそれを認可しない。

公開・GitHub再取得確認後 **mandatory STOP**。研究方針・追加scope判断をGPT側へ戻す。
