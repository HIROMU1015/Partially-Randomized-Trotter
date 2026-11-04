# Controlled RTE・Hadamard：12局所probeの固定実行範囲

2026-11-04 JST。**SOURCE準備のみ完了。実H4の科学実行は未承認・未実行。**
方針採択を受けて、[前段計画](../rpe_controlled_hadamard_preparation_20261104/SCIENTIFIC_VALIDATION_PLAN.md)
のrunnerを実装した。旧計画・SOURCE・結果・grant/ledger/marker/STOPを変更していない。
実行する場合は、このファイルとrunnerがcommitされたSHAを示した後、ユーザーからこの固定範囲の
科学実行の別途明示許可を得る。承認証明書・承認文保存・新grantは要求しない。

## 固定条件と12 probe

linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、8 system qubit、ancilla8、
保存physical state・4電子sector70のnormalized ray、Partial L_D3、δ=.02、K2、
extract_identity_phase、coefficient_atol=1e-12。
保存q/N/α/β/εは親requestのmetadataとして変更せず出力へ保持する。

| 親m | 親q | r | 実際に構築する局所回路 |
|---:|---:|---:|---|
| 0 | 1 | 1 | q_probe1/all0、q_probe2/stress |
| 8 | 256 | 2 | 同上 |
| 10 | 1024 | 4 | 同上 |
| 11 | 2048 | 8 | 同上 |
| 13 | 8192 | 16 | 同上 |
| 17 | 131072 | 32 | 同上 |

q_probeを親qと混同しない。全18roundの直接回路検証ではない。
c0=`df-fragment-10:Z0`（sign−）、c1=`df-fragment-11:Z0`（sign＋）。
全18tupleとsignal/model τの固定値はf490のfixed_inputs.jsonに従う。
selector不成立時に別componentを選ばない。

- Z0/Z1：rotation c0/c1、productsなし、order0。
- T−：rotation c0、products c1,c0、order2、event.phase=−1。
- T+：rotation c1、products c1,c1、order2、event.phase=−1。
- all0：1 occurrenceのr位置をZ0,Z1で交互にする。
- stress/r1：2 occurrenceをT−,T+の順とする。
- stress/r≥2：2r位置をZ0,Z1の交互列にし、step0先頭=T−、step0末尾=T+、step1末尾=T+へ置換。

これは強制条件付きtrajectoryであり、RTE分布のsampleではない。
非可換operator順序の人工テストを通過したが、この実H4 pairの非可換性は今回評価していない。

## SOURCE・入力の対応

production SOURCEはf4905fff592351d33ba07b927892fb9532c99711から変更していない。
前段結果commit bdffe73a6a8af0988e62ff90c9dd344880e8b8b5の
[SOURCE_BINDINGS](../rpe_controlled_hadamard_preparation_20261104/SOURCE_BINDINGS.json)を再利用し、
[source_inputs.json](source_inputs.json)で追加のmetadata validator等を固定する。
実行SOURCE SHAはこのディレクトリを固定した公開commitを指定する。
runnerのdefaultは静的byte/schema照合だけ。`--run`は科学実行用の明示switchであり、許可の証明ではない。

固定入力は前段と同一：

| 入力 | SHA256 |
|---|---|
| H snapshot | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| B physical_state.npz | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| m17 source_replay.json | `ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08` |
| V2 round CSV | `e830057532c0d7b5700b4d723ddfb77e6b0734d714177d18706331996d1b42b0` |

H hash/state fingerprint/preparation hash/tail hash・component/basis・sector/endianは
旧計画の値をそのまま照合する。保存vectorとexport stateのhex配列も照合し、再solveしない。
`signal_config`は保存signal側τと、固定preparationのλ_R×(δ/r)を**hex完全一致**で結ぶ。
closeなmodel側τで代用しない。既存RTEConfig/step validatorは変更せず通す。
新CDFやcategorical平均は作らない。

θはSOURCE固定した`atan(τ/(n+1))`のrecipeで、将来の承認済み6群parameter replay時に生成する。
全群のθ/τ hexをparameters.jsonへ保存してから参照行列・trajectory・回路を構成する。
今回その実H4 θ値を新計算したとはしない。

## 独立参照・精度・比較

[reference_adapter.py](reference_adapter.py)は既存mean_adapter/ball_adapterを再利用する。
96-bit Acb、exact binary64 basis parameterからanalytic RZ/XXPlusYYをsector上で構成する。
runtime preparationのone-body eigenvalue／fragment η,λからoccupation energyを計算し、
productionのdiagonal primitive generator・controlled出力を使わずF/Rを構成する。
保存step scalar位相をFへ1回だけ含める。
選定signed componentは2個だけ構成・cacheし、閉形式eventを所定順で作用させる。
basis descriptorsと準備モデルは共通依存として残り、Hamiltonian定義全体の独立証明ではない。

参照stateは保存raw vectorのnormalized ray。rawを上書きしない。
MatrixBallのnorm/radiusを最後まで伝播し、binary64 target/signalへの変換誤差を含む上端を残す。
Qiskit Statevectorへは**actionごとにcopyした配列**を渡し、global phaseによる入力共有の影響を防ぐ。

uncontrolled raw、controlled raw、controlled boundary_optimizedを同じevent列で比較する。
controlled2経路はsupport_run_le_1と保存training fingerprintを明示。
control0は入力そのもの、control1は参照V_Tψと照合する。位相alignmentはしない。
boundary_optimizedへ両測定wrapperを付け、測定copyの終端測定だけを外してZ平均を得る。
cosine=Re、sine=+Im、bit0=+1、bit1=−1。

vector最大振幅残差、axis平均残差は、参照の不確かさ込みの上端≤1e-10を判定基準とする。
追加precision・tolerance変更・共役・半径projectionでPASSへ変更しない。
これはfloat実装の局所数値整合性であり、全supportや長qの実装誤差の数学的保証ではない。
boundary scalar再結合・block融合の長q転送も新認定しない。

## 科学call・出力・STOP

前段のcall範囲を維持する。以下はtop-level操作の開始数であり、各builder内部のgate追加まで別callと数えない。

| 操作 | 上限 |
|---|---:|
| H/state load、DF preparation、support登録 | 各1 |
| finite config/distribution parameter replay | 6群 |
| 参照F/R pair／signed component | 1組／2個 |
| 参照trajectory vector＋signal | 12 |
| uncontrolled raw build/action | 各12 |
| controlled raw build／boundary_optimized build | 各12 |
| control両branch action | 48 |
| measured wrapper build／ideal mean action | 各24 |
| new H/state solve、SCF/DF再因子化、CDF、sampling、compile、quantum shots | 0 |
| q_probe≥4回路、全18round平均再解析、成功率・総費用・noise・状態準備 | 0 |

fresh outputにparameters.json、RESULT.json、r別vectors.npz最大6件（計最大8ファイル）。
各NPZは参照target、uncontrolled/両control branch/wrapperの数値vectorを保持し、hashをRESULTへ残す。
失敗したprobeのresidual・得られたrawも保存する。worker crash等でcall数が不明ならunknownを明記する。

6r群を独立processで並列化。worker数は6群数とCPU affinityから選び、worker BLASは1 thread。
spawn前に自processの環境を一時設定し、終了後に復元する。referenceの70×70小行列はFLINT1 thread。
CPU/RAM/workerやphase/total wallの人工上限は設けず、実quota・空き資源・共有環境を尊重する。
起動前のCPU/RAM/GPU情報を記録。共有負荷で支障がある場合は自jobだけを調整・停止する。
他ユーザー、別Codexの包含評価、SERVER2 H8、共有環境の設定・job/filesは変更しない。

SOURCE/input/hash/tuple/selector/τ不一致、nonfinite、参照上端不足、1e-10不合格でSTOP。
workerは失敗通知後、新しいbuild/actionを開始しない。既に進行中のactionは終了し、結果を回収する。
途中までのcounts/rawを保持し、profile変更・別namespace・新精度で自動retryしない。
1固定batch完了後もmandatory STOP。既存outputを上書きしない。

**科学実行の別許可後だけ**使う入口：

```bash
<research-python> docs/research/rpe_controlled_hadamard_runner_preparation_20261104/run_local_probes.py \
  --run --output-dir artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run01
```

Qiskit1.3.0／NumPy1.26.4／SciPy1.14.1／python-flint0.9.0・FLINT3.6.0、Python3.11、96 bitを固定。
全体 `RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、U_exec=nullを保持する。
V1/V2/V3/R1/B/m17の既存判定を変更せず、重要な独立科学レビューも開始しない。
