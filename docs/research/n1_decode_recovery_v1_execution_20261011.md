# N1 decode復旧：実行source固定

利用者の新しい実行指示により、準備commit `ba4530678df9d665d2e6005fb08179b0f7a7921f` からN1専用branchを作成した。準備契約・旧failure・旧STOPのbyteは保持する。このbranchはN1一回実行のみ。N3、N2、旧mixed runnerは起動しない。

[固定scope](n1_decode_recovery_contract_v1_20261011.md) の44 factor・178 pair・gauged/ungauged・q1/2/4/8・精度・対照・compiler設定を維持する。元source `bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46` のN1 numerical function/class 48定義はASTで同一と確認した。新moduleはN1部分をコピーし、input load境界だけを交換した。N3/N2定義は含めない。

`ungauged_eta` の実形式real/imag vectorをruntimeのreal vectorへdecodeし、imag非zeroを拒否する。vector/matrixのtyped shape、finite、bool拒否、unknown/oracle field拒否、group partition、ID/order、全context件数を検査する。元input packは書き換えない。raw gをHermitizeせず、既定のanti-Hermitian growth会計をそのまま使う。全保存numeric scalarのbinary64 as_integer_ratioを5428件照合した。preflightはPF、compile、truthを呼ばない。

新N1-only synthetic18 tests PASS。実pack全contextのdecoder回帰を含み、旧N3/N2 tests・科学例は実行していない。Gaussian区間、Pauli/JW平方、独立Pauli交換子、ordinary control/scalar phase、full32列native作用、cache一致、real/imag vectorの新synthetic通し検証を行った。入力形式以外の科学kernel変更は0。

static inspectionと既存timingからcompile/全列作用を主負担と判断し、元の完全発行回路fingerprint cacheとvectorized native actionを再利用する。3 independent contextsを最大2 spawn workersで並列化し、BLAS/OMP/Numba/Rayon1 thread、Qiskit parallel FALSE、transpile num_processes1。CPU32、GPU deviceなし、memory約51GiB。worker AS4GiB、aggregate8GiB、JSON/IR32MiB、output512MiB、log16MiB、native20000 operations、Pauli256、有理数4096 bits。wall/CPU capなし。同条件速度比較はなくspeedup率は主張しない。

source固定後に全context preflightを再確認し、source-bound plan・認可を固定してN1 runnerを一回だけ起動する。contextごとの全selector ledgerを保存してhashを閉じてから、そのcontextのtruth診断へ進む。selectorは真誤差を使わない。RZ/CXを別にsafe integer shot×full XY wrapper費用で選び、exact fallbackを残す。binary64 native全列作用検査は厳密native-error保証ではないため、claimは `ideal_gate_certificate_with_numerical_compile_validation`。

wrapperは候補2848＋元Pauli24＋選択後Pauli最大48の2920要求予定。選択不成立ならselected Pauli targetは生成されない。hard cap3072 wrapper、44 basis、total3116を保持する。exact q全失敗は `UNRESOLVED_PF_CERTIFICATE` と保存し、retry/resume/条件追加は0。科学kernelの別問題が出たらTECHNICALまたはCERTIFICATE_OR_SEMANTICS_STOPで保存し、その場で修正して再実行しない。

[module](../../src/trottertracks/representation_exploration/n1_decode_recovery_v1.py)、[runner](../../scripts/run_n1_decode_recovery_v1.py)、[tests](../../tests/test_n1_decode_recovery_v1.py)、[独立保存checker](../../scripts/verify_n1_decode_recovery_v1.py)、[実行前資料](../../artifacts/n1_decode_recovery_v1/2026-10-11/preexecution/)を参照。終端時はmandatory STOP、中心テーマnull、次段認可false。
