# r16：元Hに対する13-round誤差・十分shot数

固定条件は [PLAN](PLAN.md)。Linear H4／1.00Å／STO-3G／Nα=Nβ2、元H exact ground仮定・準備費除外、number-CDF8／元順LD2／S2/s1/r16/K0／h=.1。13round、εE=.00159Ha、αtotal=.01、各軸α=.01/26、βtotal=π/6、βstat=π/12、prior[-10,10]。

**全13roundでS1_CONDITIONAL_IDEAL_PASS。** 数値operator norm＋保存DF engineering guard、理想finite-RTE IID、exact-nativeに条件付き。実装lawの距離・joint IID・all-law compile誤差は未資格であり、actual RPE成功確率を新評価していない。

## 再利用とr依存項

`c1_stage3_reference.reference_kernels` のPFはr非依存。新kernelを保存PFに照合した上で、13個の保存PF誤差を再利用。DF sector norm＋guard=8.060251994885292e−9Haも同一endpointのr非依存値を再利用した。元H/CDF/保存sector generatorのhashを照合し、J/U一致とgenerator和を確認した。

r16のlaw・angle・normalization・attenuation・corrected RTE kernelとRTE-PF normは新規評価。既存canonical `configure_rte`、`finite_rte_corrected_operator`、B1 `transport_round`、B4 `track_d_original_h_accuracy` を使用。`reference_round` と同じq乗/norm定義を用い、その未使用のown-DF信号・exact exponential枝を呼ばない薄い接続にした。旧r16/s4や別endpointの結果は使用していない。

λR=0.5000442435909978Ha、τ=λR h/16=0.003125276522443736、無符号回転角=atan(τ)、signed係数はcanonical atomに保持。BK=1.0000048836647457、Aq=BK^(−16q)。216 signed atoms、identity=0.041217410629575235はsampling normから分離し、核scalarは一度ずつ含める。law/config/probabilityの全記録は `r16_law.json`。

## 元Hへの十分条件

εm=εPF,m+εRTE,m+min(2,tm dDF)、raw誤差δm=Am εm、半径下限ρm=Am−δm、位相誤差上限asin(εm)。DFをさらにWeyl energy shiftとして二重加算しない。Nは既存B4のρ sin(βstat)/√2とHoeffding算定による各X/Y軸の十分shot数。αは全26軸へ固定配分。ground state-vector、own-DF信号・N、新しいground solve／DF fit／測定samplingは使用していない。

|m|q|A r16|ε to H|ρ lower|phase upper [rad]|phase slack [rad]|N r16 各軸|保存N r32 各軸|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|1|0.999921865|9.32469757e-05|0.999828625|9.32469758e-05|0.261706141|512|512|
|1|2|0.999843735|0.000186063018|0.999657701|0.000186063019|0.261613325|512|512|
|2|4|0.999687495|0.000368718562|0.999318892|0.000368718571|0.261430669|512|512|
|3|8|0.999375088|0.000711507859|0.998664025|0.000711507919|0.26108788|513|512|
|4|16|0.998750566|0.00125214402|0.997499987|0.00125214435|0.260547243|514|513|
|5|32|0.997502693|0.00175593314|0.995751145|0.00175593404|0.260043454|516|514|
|6|64|0.995011623|0.00271716995|0.992308007|0.0027171733|0.259082215|519|516|
|7|128|0.990048130|0.00506059928|0.985037893|0.00506062088|0.256738767|527|521|
|8|256|0.980195300|0.00965233902|0.970734122|0.00965248891|0.252146899|543|530|
|9|512|0.960782825|0.019236117|0.942301094|0.0192373036|0.242562084|576|549|
|10|1024|0.923103637|0.038507127|0.887557568|0.0385166498|0.223282738|649|589|
|11|2048|0.852120325|0.0772176871|0.786321564|0.0772946297|0.184504758|827|679|
|12|4096|0.726109048|0.155538121|0.613171412|0.15617218|0.105627207|1359|910|

q=4096ではεPF=.0868531196299406、εRTE=.06868169950965573、DF evolution=3.301479217105016e−6、ε=.15553812061881345。ρ=.6131714115091349、deterministic phase=.15617218029996552、残余phase slack=.10562720749918392rad。N=1359/軸。r32保存値は910/軸で、新しいr16 Nとは区別する。

horizonはmmax=12/qmax4096が最小で、条件付きenergy誤差上限=.0012783173232380342Ha、統計失敗上限=.01。前のq2048では.0025566346464760683Haとなり目標を満たさない。旧decoder・φ=−hEを維持した。

S1のwall timeは約2.86秒、13個の独立36×36処理を13 worker・小matrix BLAS各1 threadで実行。保存normはrigorous enclosureではなく数値pilotの前提。Strong／Referenceの同一N利用は、同じr16 ordered pathの理想X/Y分布が保存されることに条件付きで、単一compile標本から全law実装保証を主張しない。全round期待価格・総資源の算定は行っていない。

機械可読結果：[S1_result.json](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/S1_result.json)、[round TSV](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/S1_rounds.tsv)、[保存r32比較](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/r32_saved_N_comparison.json)。
