# C1の既存資源観測と限界

以下は固定summary/比較CSVの**引用・表示整形のみ**。raw trajectoryの再解析、改善率、統計推定、資源再ランキング、十分Nの新評価は行っていない。値とJSONの対応は[verbatim引用](quoted_resource_values.json)。

対象：Linear H4 1.00 Å / STO-3G / DF12 / R1 generation-prefix L_D=3 / 8 system qubits / 13 basis・324 labels。T=4/5、q_outer=1、r_RTE=4、m=5、sigma=+1、tail microstep=1/5。H_prob=2048、root=4096、eta/rho/angle tau=10^-12、angle root=256・terms=64、seed=20261021。測定対象は `TAIL_ONLY_PLUS_SCALAR`。[旧実行契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/execution_contract.json)、[P5R契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/execution_contract.json)。

## 保存された3方式比較

全方式1024 whole-path・4096 microsteps。Ordinary/Fullは旧C1-5Aの完走substage、P5は別P5-only実行の完走結果であり、旧C1-5A全体のfailureを変更しない。

|保存指標|Ordinary RTE|General Full Return|Closed P5 (P5R)|
|---|---:|---:|---:|
|completed paths|1024|1024|1024|
|accepted fraction（観測）|1.0|1.0|1.0|
|valid zero paths|0|0|0|
|empirical weight second moment|1.1141049662602474|1.1141049662602474|1.1105279232736538|
|max observed weight（binary64）|1.0555117082535121|1.0555117082535121|1.0538158868007512|
|instructions / classical path|285.125|284.6494140625|284.7666015625|
|depth / classical path|76.673828125|76.619140625|76.6455078125|
|logical_1Q / classical path|65.833984375|65.763671875|65.7626953125|
|logical_2Q / classical path|215.3046875|214.900390625|215.01953125|
|child_rotations / classical path|4.0|4.0|4.0|
|signed_Q / classical path|0.048828125|0.044921875|0.044921875|

根拠：[Ordinary summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/ordinary/summary.json)、[Full summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/full_return/summary.json)、[P5 summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json)。depth/1Q/2Qはraw logical計数でありnative数ではない。経験的momentは測定値±1を伴う認証estimator moment/母分散ではなく、保存weight二乗の記述量。表示maxも一様range上界ではない。

|保存raw degree件数 / 4096 microsteps|Ordinary|Full|P5R|
|---|---:|---:|---:|
|degree 0|4070|4073|4073|
|degree 2|26|23|23|

上表は各summaryの`raw_degree_counts`をそのまま引用した。Ordinaryのreduced lengthは0が4071、2が25、Full/P5Rは0が4073、2が23。degree0はempty parentを意味し、**identity unitary/no-op/無料回路ではない**。各pathに4 child rotationsが残る。ここでの非空parentは少なく、zeroによるquantum call回避は観測されなかった。別method PRNG keyなのでpaired効果・有意差を認定しない。[既存の比較解釈](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.md)。

## 古典費用：層・条件を保持

単位は保存CPU秒。onlineのsampling行にはlocal query・有限bit検査・event Evidence生成/保存が含まれる。算術だけの費用ではない。

|保存計測|Ordinary旧|Full旧|P5R新|
|---|---:|---:|---:|
|normalizer/completion setup CPU|0.026659721|0.203877525|283.993160344|
|sampling/checks/event Evidence CPU|21.573830977|4410.007131644|400.634031349|
|angle query/Evidence CPU|7.104218642|67.692002726|23.539041366|
|logical build CPU|22.452788567|21.202090197|20.856494351|
|全hierarchy CPU（計測境界差あり）|71.138041|4523.42039|998.71687|
|total wall秒（同時負荷差あり）|8.770332224667072|290.6455717757344|731.9986242558807|
|angle cache hits / misses|4059 / 37|4052 / 44|4072 / 24|
|event cache hits / misses|3028 / 1068|3038 / 1058|3951 / 145|

根拠：[旧Ordinary setup](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/ordinary/setup.json)、[旧Full setup](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/full_return/setup.json)、[旧Ordinary terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/ordinary/terminal.json)、[旧Full terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/result/full_return/terminal.json)、[P5R費用報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/RESOURCE_REPORT.md)、[既存比較CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.csv)。

Fullは4096 local parent packet queryを支払った。P5は104,977群のsetupを支払い、P5Rの累積整数prefix初期化は別に0.493558755 CPU秒、P5 level2 sample+event query計数は46だった。これらはSOURCE由来の計数で全有理/bit演算回数ではない。P5R producer online wallは430.542883934秒、logical provider全phase wallは5.506168129秒。旧online wallはOrdinary 3.9195715226233006秒、Full 282.16968657821417秒で、P5Rとは処理分割と境界が異なる。[費用層・計数の正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/RESOURCE_REPORT.md)、[CSV scope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.csv)。

旧2方式は各21 fork worker、3 coordinator同時実行、worker別cache。P5Rはsingle producer終了後32 fresh exec provider、producer-global cache、既知の累積整数selector、job-local BLAS threads1、fsync・Evidence再読込・monitorを採用した。P5Rは巨大stateのfork継承回避の選択で、samplingの本質的逐次性・最速実装を意味しない。全方式に同等の合法cache/selector/前処理を認める強いbaseline比較は未取得。CPU順位・setup償却・general scalingを今回新たに評価しない。[技術修正](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/TECHNICAL_REPAIR.md)。

P5R producer最大RSSは1,772,004 KiB、logical workerの最大個別RSSは346,364 KiB。観測active child合計のPSS最大は6,993,683 KiB（root除外・未観測peakの上界でない）。summary前保存量はOrdinary 11,384,315、Full 29,638,652、P5R 38,887,896 bytesで、schema・monitor・sampling/batch重複の範囲が異なる。上位cgroup peakやRSS和をjob物理memoryへ置換しない。[memory/storage条件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/RESOURCE_REPORT.md)。

## 解釈・未取得資源

この固定入力・短時間・m=5の探索ではlogical差が小さく、zero pathは0だった、という採択判断の根拠を保持する。RA一般の劣位、数学的無効性、他分子/時間/次数の不可能性へ広げない。経験的weight momentが異なることから新しい十分Nや総量子costを作らない。

- `UNKNOWN / NOT_ESTABLISHED`：deterministic halves、native T/CX/1Q/depthのstrict認証、state preparation/readout、保存DF指数taskへのrigorous binding、認証expected native cost、十分N、同精度全task比較。
- 量子shotsは0、coherent signal estimatorは未完成。保存stateアクセスを無料の物理準備としない。Fullの固定P0〜P5 PASSはRNG IID/測定器/provider認証ではなく、canonicalの一様finite-bit certificateも代替しない。
- 1024は承認された探索path数で、必要十分Nではない。tech failure/未完了はzeroにしない。C1追加計算・native保証整備・大規模比較は現在一時凍結。

根拠：[R2の保証範囲](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md)、[C1-4.5 open gates](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/handoff.md)、[P5R handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/handoff.md)。数学保証の適用条件は[v1.1 §§4,9–10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md)へ戻り、H4の完走によって保証を増やさない。
