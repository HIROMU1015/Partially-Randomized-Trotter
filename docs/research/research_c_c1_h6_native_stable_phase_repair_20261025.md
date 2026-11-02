# C1：P0 declared phase集計の精度修正

ユーザーの継続修正・再実行許可に基づく。旧native SOURCE bf431765…とGaussian修正SOURCE5309006…の各162-call FAIL、[5-call独立診断](research_c_c1_h6_native_state_action_diagnostic_20261025.md)・全STOPを保全する。

診断ではGaussian修正後のraw/reference差4.45e-15に対し、compiled/raw差5.52e-10をAerと独立Qiskit Statevectorの双方で確認した。小さい人工回路では再現せず、長い人工回路でdeclared global_phaseの加算誤差を再現した。

## 方式と公平性

新 `c1_h6_compiler_precision.py` は既存Qiskit1.3.0のP0全pass・opt1・seed17・rz,sx,x,cx・approximation_degree1.0・qubits_initially_zero=falseをそのまま呼ぶ。入力Circuitの**declared global_phaseだけ**に、既存Qiskit ParameterExpression／SymEngine0.13.0の160-bit MPFR定数1とsymbolic markerを加える。compilerのphase加算中はその高精度定数を保ち、最後にmarker=-1をphase式へ直接bindして正確に相殺し、binary64へ一度だけ変換する。

Qiskit1.3のUnrollCustomDefinitionsはglobal-only parameterをCircuit ParameterViewへ登録しないため、Circuit.assign_parametersではなくphase式を直接bindする。marker消失・係数変化・未束縛入力では明示FAILとし、旧方式へfallbackしない。

**raw/reference状態・operator・信号の出力を参照してphaseをfitする処理ではない。** gate本体にはmarkerや追加gateを入れない。相対phase、Gaussian向き、frame順、κ・scalar、sharing/fusion規則、state-action、数値allowanceは変更しない。H2/H4の共有compiler・NativeBuilderやインストール済みpackageも変更しない。

人工短回路と長回路で、旧P0とのgate名・qubit・全angle列が完全一致し、gate countも同一であることを検証した。CDF/XDF双方・kernel/shell全11 compileに同じ方式を適用する。新しい最適化による費用削減は仮定しない。

## 人工・回帰結果

修正後の関連 **97 tests PASS、FAIL/skip0**。新phase保持・body不変・長回路・未束縛STOPに加え、Gaussian精度、state-action、8経路人工end-to-end、phase/向き/frame順/compile/adjoint破損、P0/inverse mutex、H2/H6/PF関連回帰を含む。[log/XML](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/technical/)。先行の185 PASS・旧codec/native mock独立processの証拠も保持する。

人工5-qubit・300 random Gaussian blockのcontrol0 identityで、旧誤差 **4.3527573634e-10** → 修正 **5.3605739426e-13**。旧／修正の出力gate本体は完全一致した。[人工値](../../artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/technical/phase_stress.json)。登録1e-10を超える精度劣化を人工入力で再現し、scientific thresholdを緩めずに修正した。これは実H6の成功や厳密enclosureを意味しない。

## 新固定条件

[新298-call契約](research_c_c1_h6_native_stable_phase_contract_20261025.json)、SOURCE manifest46 files・保存入力25 hash。同じ直鎖H6・1.00 Å・STO-3G、Nα=Nβ3／sector400、保存最終CDF14／selected XDF14、全frame ordered S2 s1/s4、literal/shared8経路、δ0.1・m0..12、CA0.00159／DF reserve CA/8、own-DF ground oracle・状態準備費用除外を維持する。

最大298 logical units、内訳は元の契約と同じ。fresh output `artifacts/research_c_c1_h6_native_stable_phase/2026-10-25/run01`。以前の162＋162＋診断5＝329 unitsは保全し、新実行との累積最大は627。旧残枠の流用やrun内retry/resumeは0。今回の修正依頼中のfresh再実行は2回目であり、旧attemptを隠さない。

SOURCE・契約は実行前にcommit/push・remote照合する。component/norm1e-10・signal2e-10・位相fit禁止は不変。最初のcorrectness FAILで後続を停止し、開始済み作業だけ回収する。新fit・optimizer・rank探索・DF/PF再計算・shots・最終CA／総費用は0。

未知global合成誤差はnull／UNRESOLVED_FINITE_PROBESのまま。費用は1 h-step kernelだけで、全round shots加重総費用へ外挿しない。P0／inverse回路生成だけ既存mutexを共用し、他の重い処理は実際の空き資源に応じて並列化する。共有環境・他jobを変更しない。

結果または部分証拠の公開照合後にmandatory STOPし、追加のglobal bound・CA・shots・総費用計算や重要な科学レビューへ自動進行しない。
