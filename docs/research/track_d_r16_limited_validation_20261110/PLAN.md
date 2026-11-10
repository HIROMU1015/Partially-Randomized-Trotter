# Track D：r16限定検証 実行前PLAN

起点：`1f5ea971f8ceac54f2dc00e71824ab8dabf95837`。既存resource interface枝から、科学条件r変更を分離する専用branchを作成する。ユーザー指示によりS1、全PASS時のS2、公開まで許可済み。既存のSTOPは解除・変更しない。

## 固定条件

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元H exact ground仮定、準備費除外。number-CDF8／元frame順LD2／ordered partial S2/s1/**r16**/K0／h=δ=.1。元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。

13round m0..12/q1..4096、εE=.00159Ha、αtotal=.01、各軸α=.01/26、βtotal=π/6、βstat=π/12、prior[-10,10]、既存dyadic decoder/φ=−hE。予算変更・α最適化なし。

## S1

保存済み原sector36の9個のCAR generatorを正本hashで読む。canonical `prepare_arm/configure_rte` によってr16のlaw・signed atom重み・step角・normalizationを構成する。旧r16/s4・別endpointの既存研究は転用しない。

`c1_stage3_reference.reference_kernels`のPF kernelはrを参照せず、head・h・tail・scalar/identityだけで決まる。新生成PF kernelを保存PFに照合し、13roundの保存PF operator誤差を再利用する。DF sector norm＋engineering guardもr非依存。RTE corrected kernel、q乗のRTE-PF norm、attenuationはr16で新評価する。新しいexact exponentials・ground solve・DF fit・state-vector評価は不要。

既存B1 `transport_round`、B4 `track_d_original_h_accuracy` に接続し、元H誤差を一度だけ信号へ含める。各roundのradius・phase余裕・十分N・horizon・failure unionを評価する。保存own-DF信号/Nは使用しない。理想IID/exact-nativeと数値norm前提に条件付きで、rigorous enclosureやactual implemented成功保証にはしない。

13個の36×36独立処理を実CPUに合わせて並列化する。BLASの多重並列を避ける。全round PASS時だけS2へ連続して進む。失敗／未資格ならその結果を公開してSTOP。

## S2（S1 PASS時のみ）

sample ID `r16_q4_sample00`、master `2026111016`、domain `track_d_r16_q4_single_v1`。既存`trajectory_seed((master,domain),'proposed',endpoint,2,'R16_Q4_PRICE_ONLY',0,occurrence)`で4 seedを固定し、既存PCG64 samplerを各step一回だけ呼ぶ。sampling前にseed plan、回路生成前にordered event原本・hashを保存する。4×16 event、whole ordered path一gadget、cut=complete step2、C1追加融合OFF。価格を見たfixture変更・再samplingなし。

既存C3 `prepare_steps`、StrongBuilder、CAR reference、raw/state-action harnessを再利用する。既存関数のr32固定入力部分に限って薄いr16 adapterを設け、r32原本・旧関数の固定条件を変更しない。uncontrolled Gaussian／directional center／head relative／tail同basis共有／合法step境界half融合／scalar集約を両方式で維持する。

AB=V、actual adjoint、vacuum ray/χ、full256入力のcomplex coherence・X/Y測定演算子、ordered CARとの一致を既存精度で検査する。raw PASS後のみStrong/Reference×X/Yの最大4 full-wrapper P0 compile。

P0：Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initially_zero=False、topology-free、既存MPFR160位相保持。postcheckは旧精度を緩和しない。7指標を各wrapperに記録し、同一r16 path内の差を比較する。r32保存q4は別pathの参考値としてのみ表示する。

## 保全・終了

新namespace `artifacts/track_d_r16_limited_validation/2026-11-10/run01/`。実行前SOURCEをcommitし、S1/S2のSOURCE・入力hashと前後一致を保存する。既存SOURCE/raw/result/grant/ledger/marker/STOPは無変更。新ordered event/QPY/NPZはlocal限定、公開は結果・SOURCE・hash索引。技術失敗は初回記録を保持し、科学的再試行・条件変更をしない。

新samplingはS2の1pathだけ、compile最大4、q4以外の回路、fit、ground solve、A/F/S更新、全round価格・総費用、他algorithm、詳細科学レビューを行わない。結果をcommit/pushしremote SHA/tree/公開bytes照合後mandatory STOP。次の研究判断はユーザー／GPTへ返す。
