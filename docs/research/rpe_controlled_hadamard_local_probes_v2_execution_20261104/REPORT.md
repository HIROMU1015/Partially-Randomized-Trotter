# Controlled RTE・Hadamard v2：固定12局所probe・24軸が整合

2026-11-04 JST / LOCAL。
**selectorを修正した固定SOURCEをrun02へ1回実行し、12局所probeの両測定軸とcontrol branchが
参照不確かさ込みの既存許容差1e-10以内で一致した。**
結果は `LOCAL_TWELVE_PROBES_CONSISTENT_MANDATORY_STOP`、native exit0。
最大axis mean残差上端は約4.03e-15、最大vector残差上端は約2.66e-15。
全体 `RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、U_exec=nullを維持する。

## 固定SOURCE・条件・修正

科学実行SOURCEは `7279f700a58921209b851c8d9c363d38ff2f1d98`。
[固定範囲](../rpe_controlled_hadamard_runner_v2_20261104/EXECUTION_SCOPE.md)、
[selector](../rpe_controlled_hadamard_runner_v2_20261104/selector.json)、
[人工67 PASS・変更点](../rpe_controlled_hadamard_runner_v2_20261104/PREPARATION_REPORT.md)、
[runner](../rpe_controlled_hadamard_runner_v2_20261104/run_local_probes.py)を実行前commitへ固定してから起動した。
専用branch/worktreeは `codex/rpe-controlled-hadamard-local-probes-v2-20261104`。

linear H4、隣接1.00Å、STO-3G、保存legacy DF12、8 system qubit＋ancilla8、
B保存physical stateのnormalized ray、4電子sector70、Partial L_D3、単一δ=.02、K2、
extract_identity_phase、coefficient_atol=1e-12。
rは1,2,4,8,16,32。親m0,8,10,11,13,17のq/N/α/β/εをmetadataとして保持し、
各rで実際に構築した回路はq_probe1/all0とq_probe2/stressの2本、計12本に限定した。
親q=256〜131072等で回路を直接構築した結果ではない。

run01は旧B由来ID不在で停止していた。今回は保存218 componentのID/sign/basisだけから
c0=`df-fragment-3:Z0Z2`（負、support[0,2]）、
c1=`df-fragment-4:Z0Z1`（正、support[0,1]、異なるbasis）を結果前に固定した。
default静的検査へ実保存payloadのmembershipを追加し、runtimeの係数/確率/basis hashとも照合した。
signalやoperatorを評価して有利なpairを選んでいない。
旧SOURCE76d25ac69、旧停止結果a6d1a2945、原run01はbyte/hashを保持する。

H/state solve、SCF、DF再因子化は行わず、保存HとB stateをloadした。
H hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、
state fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`、
preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`の固定照合を通過した。
保存export stateのhex、sector/endian、cutoff1e-12も対応した。

## 比較した量と方法

比較対象は同じ強制trajectory Tに対する
`z_T=ψ†V_Tψ/(ψ†ψ)`の条件付き平均。
96-bit Acb参照は保存binary64 basis parameterのanalytic RZ/XXPlusYY、
runtime preparationのoccupation energy、signed component2個からF→r event→Rをq_probe回作用させて作った。
production diagonal primitive generatorやcontrolled出力から理論signalを作っていない。
basis descriptorsとDF preparationの意味は共通依存として残る。

同じevent列のuncontrolled raw、ordinary-controlled raw、ordinary-controlled boundary_optimizedを構築した。
controlled経路はsupport_run_le_1と保存training fingerprintを使用。
control0は入力vectorそのもの、control1は参照V_Tψへ、位相alignmentなしで照合した。
Statevectorへ渡す入力はactionごとにcopyし、global phaseによる共有bufferの影響を防いだ。

boundary_optimizedへ測定付きHadamard wrapperを付け、copyの終端測定だけを外してideal Z平均を計算した。
cosine=Re、sine=+Im、bit0=+1、bit1=−1を保持。
実状態準備回路、量子backend、sampling、compileは含めない。
参照MatrixBallの伝播半径とbinary64 target/signal変換誤差を残差の上端へ加えた。
これは実装の局所数値照合であり、float算術全体や全supportへの厳密実装保証ではない。

`z_T`とcategorical分布平均ν、保存attenuated_event_mean_signalのμ0は別の量である。
今回の強制profileはlawからのsampleではなく、νや保存μ0への一致を計算していない。
追加のB除算、β/α/N/ε再配分、成功率・費用の再会計は行っていない。

## 結果

下表は上記H4/1Å/STO-3G/DF12/L_D3/δ=.02/K2/保存state/固定pairだけのlocal numerical evidence。
vector欄はuncontrolled＋control2構成×両branchの最大。
axis欄はcosine/sineの最大。表示は概数で、正式な上端は原RESULTのexact rationalに従う。

| r | profile | q_probe | vector残差上端 | axis mean残差上端 |
|---:|---|---:|---:|---:|
| 1 | all0 | 1 | 9.441e-16 | 3.442e-16 |
| 1 | stress | 2 | 1.797e-15 | 5.349e-16 |
| 2 | all0 | 1 | 9.242e-16 | 2.344e-16 |
| 2 | stress | 2 | 1.515e-15 | 7.533e-16 |
| 4 | all0 | 1 | 1.014e-15 | 8.049e-16 |
| 4 | stress | 2 | 1.345e-15 | 1.031e-15 |
| 8 | all0 | 1 | 1.023e-15 | 1.124e-15 |
| 8 | stress | 2 | 1.139e-15 | 1.085e-15 |
| 16 | all0 | 1 | 1.701e-15 | 2.230e-15 |
| 16 | stress | 2 | 2.665e-15 | 1.755e-15 |
| 32 | all0 | 1 | 2.450e-15 | 4.035e-15 |
| 32 | stress | 2 | 2.309e-15 | 2.861e-15 |

全12probe・24axisが合格。各rの保存signal側τは固定preparation λ_R×(δ/r)とhex完全一致し、
全6群のτ/θ hexを保存した後に参照・回路を構成した。closeなmodel側τへの置換はしていない。
全probeでsupport-restricted applicationが実際に使用された。
scalar/identity、product sign、event phase、rotation符号、F/R・event順序、q_probe境界融合と
Re/+Im wrapperが、指定input/trajectoryで参照と整合した。
異basisというmetadataだけから実H4 pairの非可換性を証明したとはしない。

## 科学callと資源

| top-level操作 | 実施数 |
|---|---:|
| 保存H load／state load／DF preparation／support登録 | 各1 |
| config/distribution parameter replay | 6群 |
| 96-bit参照F/R pair／signed component | 1組／2個 |
| 参照trajectory vector＋signal | 12 |
| uncontrolled raw build／action | 各12 |
| controlled raw build／boundary_optimized build | 各12 |
| controlled branch action | 48 |
| measured wrapper build／ideal mean action | 各24 |
| CDF／sampling／compile／quantum shot／長q回路 | 全て0 |
| 新H/state solve／SCF／DF再因子化／成功率・費用再解析 | 全て0 |
| v2 batch内retry／追加precision／別profile | 全て0 |

v1停止後、今回許可された修正再実行としてrun02を1回だけ実施した。
Python3.11、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.9.0/FLINT3.6.0。
CPU affinity32、利用可能RAM約52 GiB、GPU driver利用不可。
起動前1秒観測のCPU idleは約99.6%、memory pressure avg10=0で、今回は通常priorityを使用した。
6r群を6 spawn worker構成で実行、worker BLAS1 thread、参照の70×70行列はFLINT1 thread。
6 basis、2 signed component、event operatorをcacheし、q_probe≤2で巨大回路展開を避けた。
観測wall timeは約15.25秒。peak RSS・実CPU占有率・speedup factorは測定していない。
共有設定・他job・別Codexの包含評価・SERVER2 H8を変更していない。

## 原本

[run02 directory](../../../artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run02/)。
JSONはsettings/events/参照signal/両branch・axis残差/precision/countsを保持する。
各NPZは2probe分の16vector（参照target、uncontrolled、両control branch、両wrapper）を保存する。
報告時はhash・CRC・array header/shape/dtypeだけを技術照合し、raw vectorから再度signalを評価していない。

| 原本 | Bytes | SHA256 |
|---|---:|---|
| RESULT.json | 3590084 | `5792eb42bcb2bcdd691d8978e5f4711ea71653c2451c1c210d9ba71ae6946077` |
| parameters.json | 1493 | `9b1c302e9ed77c71dd700dc34614a5ba8506f0b2f7e23115706e6f3476b82f85` |
| r1_vectors.npz | 119278 | `437c7d0520c96d4017f8e3bbfa3cd3df390d393b6355ce80828a25169caa62a8` |
| r2_vectors.npz | 119278 | `c571423de2de8ea3d1b5781afb4a698a7f0c5ed52e2affa4520c0c60ce4c8ec9` |
| r4_vectors.npz | 119278 | `da3ff34c999d0c11b636804e886e084d48d62d62f9552523e3ae957cadf18dea` |
| r8_vectors.npz | 119278 | `0e7e1db6e63b186100798c7d7b97aeecaa85c9ce85d7cb229b1934d941c3e19f` |
| r16_vectors.npz | 119310 | `f75eb454843105cc179522d665ed6b619f082715ff6ea807f6b5b197162229ba` |
| r32_vectors.npz | 119310 | `b499e29d1db89fd613944d1212004914016b84f8649c3a64078fd768bc260237` |

## 全18roundへの接続とmandatory STOP

旧Bのr1/指定6requestというscopeから、今回の固定cutoff/preparation・実在する別pair・全6rについて
q_probe1/2の12条件の局所回路整合性まで確認範囲を広げた。
全218 component、任意event列/support run、全入力state、保存q=2^mの大きいqへの保証ではない。
特にboundary scalar再結合・block融合・π64の長q転送、分布平均との全support対応が残る。
別Codexの全18round categorical包含結果を取得・再解析・採用したとはしない。
PCG64/IID、float virtual guard、compiled/noise、実状態準備、RPE成功率と最終総費用も未保証。

旧run01のSTOP、B/V1/V2/V3/R1/m17の既存条件付き判定とraw/grant/ledger/markerを保持する。
主エージェント自身の技術照合であり、独立詳細科学レビュー・immutable CI再現ではない。

次の提案：**Codex**が別指示後に、公開済み全18round categorical SOURCEとの対応と、
長q phase転送・任意trajectoryへの不足を静的に整理する。
**ユーザー**が次に保証するscopeを選び、新しい科学計算・重要レビューはその範囲の別途明示許可後に行う。
今回は報告・commit/push・remote公開照合後にmandatory STOP。追加検証へ自動移行しない。
