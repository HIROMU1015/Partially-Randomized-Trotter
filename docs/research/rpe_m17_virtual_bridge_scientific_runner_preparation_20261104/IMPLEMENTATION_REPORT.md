# m17科学runnerの統合・SOURCE準備

2026-11-04 JST。**統合runnerを完成し、人工・関連57 testsと実gdb監視下の人工pipelineを検証した。**
状態はPREPARED_NOT_SCIENTIFICALLY_EXECUTED。固定SOURCEに対する別の科学実行許可後、
実行時のruntime検査を通過した場合に実H4の1 passを開始できる。
実H4の新しい数値評価・保存平均再解析・sampling/build/compileは今回0。

## SOURCEと再利用

baseは`e2fd35f1dcc73b697ac85661ef91f21082fd2136`。
728c90a2d…のbridge adapterとmetadata照合、e2fd35f1d…のkernel_bounds、18fffce48…のAcb/global-radiusを再利用する。
新SOURCEはこのfolderの5つの実行module。旧SOURCE・raw・旧--run停止処理は変更していない。

| module | 統合内容 |
|---|---|
| run_scientific_bridge.py | default/--describeはbyte確認。--runtime-checkは人工probeのみ。別許可の--runはfresh出力、SOURCE固定、監視worker、partial結果と停止回収を統合 |
| observe_runtime.py | 自分の新規inferiorだけをgdbで観察。全thread snapshotと各critical kernel/ufunc entryのMXCSRを確認。CBLAS引数/実kernel/出力partitionを同一workerのprobeで確認 |
| runtime_policy.py | stdlib-onlyでgdb側Pythonからも使えるcheckpoint処理。計算環境の観察であり承認証明ではない |
| runtime_checks.py | 実際のmap/binary/core profile、complex128 shape/stride/C layout/有限性、人工dispatch probe |
| bridge_pipeline.py | 固定6入力load、raw ψ対応、numeric context構成、既存bound_context、finite/abs/guard、saved s/εのstrict比較を接続 |

gdbのPythonと評価workerのPythonは異なる環境になり得るため、observer側はstdlibだけで動く。
KernelBounds/Arbをgdb側へ再実装せず、係数と数値評価は既存の評価環境へ置いた。
GDBの[breakpoint API](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Breakpoints-In-Python.html)と
[thread API](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Threads-In-Python.html)で監視し、kernel/libraryを置換しない。
読み取るCBLAS引数は導入済みbinaryの確認済みentry/prologue位置に限り、未知の位置なら停止する。

## Runtime・保証の接続

実H4を読む前に、同じworkerでprofile/dispatch/全thread FP条件を確認する。
checkpointはprobe前後、context前後、pass前後、最終結果前。
加えてcritical entryで呼出threadのMXCSRを読み、新しいthreadも確認する。
途中で不一致があればgdbが計算を止め、親が未判定・STOPを保存する。
runtimeの候補結果は監視成功後にだけ最終判定として保存する。

全support有限性はuniform state-boundary上端からscreenし、complex absの有効rangeと誤差をguardへ渡す。
guardを証明できない場合はU_virtualもnull。平均上界だけから測定完了を推定しない。
clipping後の確率/Bernoulli3uは、完了を確認したscopeで一回だけ加える。
固定PCG64のU_exec=nullと全体UNDETERMINED/falseは維持する。

## 検証結果

新統合13、kernel13、既存bridge16、前段math15、計**57件PASS**。
人工fixtureでload配線、十分条件成立、strict境界、finite screen不明、guard不明、runtime不一致、
late failureの候補PASS抑制、fresh出力保全を確認した。実H4 loaderはmock/禁止する。

実際の--runtime-checkは人工70次元だけで通過し、32 thread、GEMVT18 partitionの出力合計70を確認。
さらに実gdb監視を残して実データentrypointだけを人工2次元へ置換し、
context→96 bit bound→guard→strict比較→最終保存まで通した。
BLAS32 threadに加え、FLINT開始後には63 threadのFP modeをcheckpointで確認できた。
この人工結果は実H4の上界・科学的GO・独立レビューの証拠ではない。
コマンドと検証の範囲は[VALIDATION.txt](VALIDATION.txt)。

## 固定実行scopeと到達点

[SCIENTIFIC_EXECUTION_PLAN.md](SCIENTIFIC_EXECUTION_PLAN.md)へ入力・数値条件・call範囲・判定・出力・STOPを固定した。
最初の新SOURCEは96 bitの1 pass。過去の最大3精度案を改変せず、今回の独立scopeを明示した。
実H4単一セルはlinear H4/1.00 Å/STO-3G/legacy DF12/B state/sector70、L_D3/δ=.02/m17 q131072/r32 K2。
N411/axis、α/β/ε、identity/cutoff、旧ideal categorical結果を保全する。

残る不確定事項は**実データでfinite/guard/strict条件が成立するか**と、将来起動時のruntime条件。
それらを調べるrunnerは完成しているが、今回その科学計算は実施していない。
ptraceまたは必要なruntime読取が許されない環境では正しくSTOPする。

公開branch：`codex/rpe-m17-virtual-bridge-scientific-runner-preparation-20261104`。
commit/pushとremote SHA/tree/成果物照合後にmandatory STOP。
次はユーザーが固定SOURCEを確認し、実行を別途明示許可する段階。
Codexは許可後、固定scopeの単一jobを実行・保存・公開し、その後STOPする。
