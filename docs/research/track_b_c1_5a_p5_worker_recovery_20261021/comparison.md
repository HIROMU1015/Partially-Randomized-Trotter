# C1-5A-P5R：保存要約による3方式比較

Linear H4 1.00 Å / STO-3G / DF12 / R1 LD3 / 8 system qubits / 324 labels・13 bases。T=4/5, q=1, r=4, m=5。全方式1024 whole-path・4096 microsteps。`TAIL_ONLY_PLUS_SCALAR` の探索的観測である。

Ordinary/Fullは旧commit `b022f2ea421649015a4c21ea3c904fe619d0c198` の保存summary・setup・terminalだけを引用した。旧rawの新規再解析・旧方式再実行は行っていない。P5は今回の新規one-shot結果。

|指標|Ordinary|General Full|Closed P5 (P5R)|
|---|---:|---:|---:|
|completed|1024|1024|1024|
|accepted_paths|1024|1024|1024|
|valid_zero_paths|0|0|0|
|empirical_weight_second_moment|1.114104966|1.114104966|1.110527923|
|maximum_observed_weight|1.055511708|1.055511708|1.053815887|
|signed_Q|0.048828125|0.044921875|0.044921875|
|logical_1Q|65.83398438|65.76367188|65.76269531|
|logical_2Q|215.3046875|214.9003906|215.0195312|
|instructions|285.125|284.6494141|284.7666016|
|depth|76.67382812|76.61914062|76.64550781|
|angle_cache_misses|37|44|24|
|normalizer_completion_CPU_seconds|0.026659721|0.203877525|283.9931603|
|online_sampling_checks_event_Evidence_CPU_seconds|21.57383098|4410.007132|400.6340313|
|all_hierarchy_CPU_seconds|71.138041|4523.42039|998.71687|
|total_wall_seconds|8.770332225|290.6455718|731.9986243|
|saved_bytes_before_summary|11384315|29638652|38887896|

logical費用はclassical trial当たり。CPU/wall単位は秒。上表のCPU/wall/storageは計測境界と実装条件が異なるため、方式固有の速度優位を示す比較ではない。詳細は[CSV](comparison.csv)のscope欄。

旧Ordinaryの4070/4096、Fullの4073/4096 microstepsがraw degree 0で、非空parentは少ない。両方式ともzero pathは観測されず、各pathに4 child rotationsが残る。Ordinaryではraw degree 2が26回、reduced length 2が25回、Fullでは各23回だった。別method keyの異なる乱数列なので、差をpaired効果や有意な改善と扱えない。保存summaryで見える小さなlogical差と大きなFull古典費用を、この範囲で説明できる。

旧Fullは全4096 microstepsでlocal parent packetを取得し、sampling/checks/Evidence stageに4410.0071 CPU秒を費やした。これには有限bit検査と保存処理が含まれ、Green演算単独のCPU測定ではない。

P5Rは104,977群のnormalizer/completionを既存P5Closedで構築し、全群の構築費用を支払う。既知の累積整数selectorとproducer共通cacheを追加した。Ordinary/Fullにも同等の合法な最適化を認める必要があり、今回のCPU差を手法の新規性・公平な最速baseline比較に変換しない。

旧実行はNumPy/Qiskit初期化後fork・各方式21 worker・3 coordinator同時実行。今回はP5 producer 1＋その終了後32 fresh exec logical providers。小規模行列向けjob-local BLAS threads=1、global angle/event cache、fsyncと監視・Evidence再読込、providerごとのbasis復元が相違点。旧forkのCOWと新provider RSSは同じ物理メモリ尺度でない。

P5の有限bit canonical certificateは依然UNCERTIFIED_EXPLORATORY。人工同値性確認と4096 eventの実proposal×weightチェックは、未知のcanonical一様bias/range定理やIID保証を追加しない。経験的二乗平均・最大値は母分散・厳密range・十分Nではない。native T/CX/1Q、deterministic halves、state preparation/readout、shots、coherent signal、全round RPEは未取得。

研究継続・縮小・終了、資源優位、原著価値、投稿GOはユーザー・GPTの判断に返す。追加比較を検討する場合の最小候補は、同等selector/cacheとCPU計測境界を揃えた比較設計の要否を先に判断すること。実行・証明・native取得は別途承認が必要。

参照：[旧summary群](../track_b_c1_5a_h4_limited_resource_pilot_20261021/result/)、[新P5 summary](result/summary.json)、[新terminal](result/terminal.json)、[技術修正](TECHNICAL_REPAIR.md)。
