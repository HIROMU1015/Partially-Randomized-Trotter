# Full-H κ2 m0：事前固定normal pathの参照単価取得

利用者が価格規則FULL_H_KAPPA2_M0_NORMAL_PATH_REFERENCE_V1を今回の費用比較用に採択し、
1 normal path・同一pathの測定付きX/Y各1回・最大2 compileを明示許可した。
今回の別jobだけを固定し、閉じたD7計画・F1・旧STOPを変更しない。

固定参照：C提案4a1cb0d7364fece475437749129fdce740fd8fc4のPROPOSAL.json/HANDOFF.md、
物理正本61f249e6e18d5afab9ba8797bef5e8f9f14e29deのADMISSION.json rows[0]/[1]。
完全path/selectorは[FROZEN_JOB.json](FROZEN_JOB.json)と結果の個別axis recordへ保持する。

対象はFULL_H_2a9a68a11d08af234a8c、H4直線1Å/STO-3G/保存tol-only DF10、one-body込みFull-H pure RTE、
T=.1/q1/R9/finiteK2/κ2/P0。cohort D_FULL_H_KAPPA2_M0_REFERENCE_N1_20261115_V1、index0、seed15612141344395026302。
元trial_seedと一致することをmetadataだけで確認した。96 shotやRTE normalizerを単価へ掛けない。

§10.5はCの提案REPORTの照合を引き継ぎ、提案公開後の関連ref、既知private calibration/holdout rootの名前、
関連jobを確認した。[PREFLIGHT.json](PREFLIGHT.json)。一致する保存価格/receipt0、同一稼働job0。
未知archive/hostを網羅したとは主張しない。旧T=.8/K4、T2、Pauli/DF12-LD3等を転用しない。

既存9cbabc617のP.acquire_pathとnormal sampler/native/wrapper/compiler/gateを再利用する。
同じA.accelerated_pathでX/Yを別processにし、既存exact gate cache・fast fingerprintを流用する。
親observerと協調的progressを再利用し、非同期stack dumpは使わない。共有環境は変更しない。
科学前にSOURCEと入力をcommit固定し、固定checkoutで一度だけ実行する。

sampling前にD7 preparation hash92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100、
H/state/snapshot、fragment生成順0..9、scalar定数/抽出/合計−0.33147781341681615を照合し、component順を保存する。
reference contextのT=.8は既存保存referenceの表示であり、取得の物理T=.1とは区別する。
元scalar総T一回・new ZZ・symmetric-directional・full measured wrapper/P0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持。

normal path drawは1回に限定し、生成orderをそのまま使う。forced/選別/再抽選なし。
index0の所定7数値gate、parity/new lowering/one-body/scalar-event/XY/fingerprintの既存gateを維持。
生成pathが実装gateを満たさない場合も、閾値緩和やorderの差替えで補わず、単価nullでSTOPする。
未compileのlegacy parity参照は既存gateの一部。追加legacy compileなし。

今回の追加人工4件PASS：固定scope/seed、同値X/Yの別receipt、96非乗算、gate/preparation不一致拒否、
normal draw1回/2回目禁止。既存toy/fake compilerとD7のsampling1回・個別X/Y並列・位相検査の証拠を再利用。
実H4を試運転に使わない。これはD1本人の検査であり独立reviewではない。

成功時は各軸の元整数RZをC_0X/C_0Yとして保持。同じ1 pathの経験的参照であり独立path数1、E[C]/CI資格なし。
不一致/欠測/gate失敗は保全しnull/STOP、無断retry/追加path/seed差替えなし。
元F1を書き換えず、m1〜m12との新シナリオ集計・96倍・N/G/winner・fit/Q3は行わない。
commit/push・remoteと公開receipt照合後mandatory STOP。
