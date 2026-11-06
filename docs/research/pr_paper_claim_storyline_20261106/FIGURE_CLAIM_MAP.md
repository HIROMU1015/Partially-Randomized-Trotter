# 既存図表とclaim・構成案の対応

2026-11-06 JST。中心課題Aに沿う研究構成メモ。既存証拠の引用・対応付けのみ。新しい数値解析、詳細科学レビュー、原稿本文・Abstract・題名の執筆、Main Claim/構成の正式採択は行わない。

既存11図表は変更・再生成していない。SVGの固定バイトとmanifest/caption/display provenanceを参照した。今回、pixel上の視認性や最終デザインの再監査は行わず、下記「そのままで足りる」は科学的内容・scopeについての使用候補判断である。完成caption・最終図番号は決めない。[FIG][FM][DISPLAY]

**本文の最小核**：S1はV01/V02/V03、S2はこれらとPauli直接対決定論の既存主表・V05・比較ルールV07、S3はPauli既存主表・V05を核にV10/V11の必要な方を補助とする。11点を全部採用する必要はない。

共通のNは解析上の十分量子shot、nは古典価格trajectory。Gは主にRZ×shot。±SEは元有限価格標本の記述で、quantum shotノイズの実測・母平均tail保証でない。元compiler/profile/control/full-wrapper/P0、H/state/reference/rank、各stageのq/r/R/Kとsplitをmanifestから辿る。SOURCE roleを未転記のstudyは結果carrierへ置換しない。

## V01：DF: strong PF comparison across saved precision conditions

- **内容・claim候補**：Q1/Q3;A10.baseline。強いPF対照を入れると、固定共通候補域でPRの競争力が変わる。
- **使用study / comparison / candidate**：["track_a_h246_pf_common_final","track_a_h246_reconstructed_confirmation"] / ["H246_common588","H246_fixed_high_order_confirmation"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V01.svg)、manifestのsource_refs=["PRI_PF60","SRC_PF588","SRC_HIGHCONF"]。scientific_SOURCE_roles={"track_a_h246_pf_common_final":["common:bdf6772aade7063c65d69f24dfdc5e303a2819a2","H6 metadata:cba2a82021b17cba0eec31b03fb19d80f35225f6"],"track_a_h246_reconstructed_confirmation":[]}。
- **適用条件**：{"DF03":"H2/H4/H6 tol-onlyDF3/10/19（H6 weighted completion）;T=.8;S2/Y4/m5元588域;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["DETERMINISTIC","SAVED_EXPLORATION"] / ["RZ×shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：PRは保存探索。118高次確認をこのPF比較の確認へpoolしない。
- **支えないclaim**：全PF最適性、H2/H4/H6のscaling、PR固定S2への改善を最強PF勝利とすること。
- **S1/S2/S3配置**：S1=本文核、S2=DF主結果の本文、S3=Discussion/SI。
- **既存図のままで足りるか・将来の注意**：現RZ/ALL_FORMULASのscopeを保てば使用候補。本文採用時も原四scopeを混ぜない。

## V02：DF: time and precision, exploration and fixed confirmation

- **内容・claim候補**：Q1/Q2;A11.long_time。精度/時間・取得候補域・探索と固定確認を伴う中心Aの主証拠。
- **使用study / comparison / candidate**：["track_a_h46_long_time"] / ["H46_original_independent_comparisons"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V02.svg)、manifestのsource_refs=["PRI_LONGBEST","DF_LONG_COMPARE","SRC_LONG","G01","G02","DF_LONG_CAND","PRI_LONGPROV","CNG","MATCHED","RK"]。scientific_SOURCE_roles={"track_a_h46_long_time":["provider:c31ef424175c02379eafebe8fc6c7548c1d9aabe","signal/native/extension:PRI_LONGPROV.source_commits"]}。
- **適用条件**：{"DF04":"H4DF10/H6DF19 weighted completion;T=.8/1.2/1.6/2;元5精度;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["DETERMINISTIC","EXPLORATION","FIXED_CONFIRMATION","SAVED_COMPARATOR"] / ["RZ×shot"] / ["DIRECT_SAVED_DETERMINISTIC","DIRECT_SAVED_EXPLORATION","DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：探索と別価格確認を別marker。個別確認SEのないCSVから差SEを誤差棒へ転用しない。
- **支えないclaim**：33/40を独立成功率/普遍精度閾値/外部再現とすること。
- **S1/S2/S3配置**：S1=本文核、S2=DF主結果の本文、S3=SI/Discussion。
- **既存図のままで足りるか・将来の注意**：RZ図として使える。B1/CX維持・反転は別の保存結果参照;図のlabelだけCXへ変更しない。

## V03：DF B3: saved comparisons against B2 and deterministic baselines

- **内容・claim候補**：Q3/Q4;A12.B3_boundary。one-body保持DF B3のRZ負結果・strong baselineの境界。
- **使用study / comparison / candidate**：["track_a_h46_long_time_b3"] / ["B3_fixed_negative"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V03.svg)、manifestのsource_refs=["PRI_B3CSV","B3SUMMARY","SRC_B3","DF_B3_CAND","SRC_B3SRC"]。scientific_SOURCE_roles={"track_a_h46_long_time_b3":[]}。
- **適用条件**：{"DF07":"H4DF10/H6DF19 weighted completion;元T・精度・固定nominee域;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["FIXED_NOMINEE_CONFIRMATION","SOURCE_BASELINE_NOT_RELABELED"] / ["RZ×shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：B3は同じセルをbaselineごと繰返すalias。独立標本でない。
- **支えないclaim**：全Hamiltonian pure RTE不可能性、Pauli pure RTEの否定、CXでも全baselineに不利とすること。
- **S1/S2/S3配置**：S1=本文核、S2=DF境界の本文、S3=Discussion/SI。
- **既存図のままで足りるか・将来の注意**：RZの既存表示は維持。将来captionではB1のFull PF単独CX例外とB2/envelope負結果維持を区別する。

## V04：H8 Stage C: fixed PR confirmation and saved PF comparator

- **内容・claim候補**：Q1/Q2;A14.H8_fixed。固定H8 PRの別価格cohort確認を、別条件の補強として使う。
- **使用study / comparison / candidate**：["track_a_h8_long_time"] / ["H8_M5_P8_Q1_CONFIRMATION_01"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V04.svg)、manifestのsource_refs=["H8_C_RAW","H8_C","H8_AB","H8_C_PRICES","H8_C_PLAN"]。scientific_SOURCE_roles={"track_a_h8_long_time":["signal:d1272016592ff06c0567862f770ed796fd61b366","native:df93fbb88761b61c2691bf01f761ed20a7987d7f","confirmation:82fdbdd098e9d3edae01a930437e3d76c1f8bf36"]}。
- **適用条件**：{"DF09":"H8DF28/T1.6/epsilon=.0005;PR m5/p8/q1/R32/K4 ID7eb0a3d691b94089969b;PF B0I m5/p15/q1 ID1425ae6ed83fd9098b39;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["CONFIRMATION","EXPLORATION"] / ["RZ×shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：PR確認n9、探索PRn9をpoolしない。PFはStage Bのn1でSE null。rare order費用UNKNOWN。
- **支えないclaim**：母平均優位、H2–H8のscaling、旧T.8とT1.6の同一系列扱い。
- **S1/S2/S3配置**：S1=本文任意/SI、S2=本文任意/SI、S3=SI。
- **既存図のままで足りるか・将来の注意**：保存G/paired SEの限定図として足りる。PR平均C未保存をG/Nから逆算して補わない。

## V05：Pauli-JW: saved Hybrid and pure-RTE precision comparisons

- **内容・claim候補**：Q2/Q3/Q5;P01.randomization_endpoint。最新Pauli Hybrid対pure RTEのN/C/G競合と保持端点の境界。
- **使用study / comparison / candidate**：["h6_hybrid_fairness_extension","h2_pauli_rte_validation","non_df_h4_pauli_pr_search","pauli_rte_event_extension","pauli_h2_h6_comparison"] / ["Pauli_latest14of15"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V05.svg)、manifestのsource_refs=["PA_BASE","PRI_PA-E4","PRI_PA-E5","PA_H6_CAND","PAULISYNTH","PAULI_ADDENDUM","PRI_PA-E6compare","PA_H6_SOURCE","PA_H2_SOURCE","PRI_PA-E3"]。scientific_SOURCE_roles={"h6_hybrid_fairness_extension":["887b69e449cb3c378286f09d0d7626599850fb67"],"h2_pauli_rte_validation":["bf5f2339f723f4edb7bf91a8ca88f33251bbef57"],"non_df_h4_pauli_pr_search":[],"pauli_rte_event_extension":["e755ecf581600192fd2745d7066e410cebd933d3"],"pauli_h2_h6_comparison":[]}。
- **適用条件**：{"PA01":"H2 Pauli14/raw state/full-H reference;T=.8/S2/central/P0;hypothetical rho=0","PA02":"H4 Pauli184;保存raw ray/full-H reference;T=.8/S2/central/P0;hypothetical rho=0","PA03":"H6 Pauli918/full-H saved sector state;T=.8/S2/central/P0;Hybrid p78/q8/r32/K2対保存RTE r64;hypothetical rho=0"}
- **cohort・metric・価格種別**：["independent_confirmation","confirmation",""] / ["RZ×shot","analytic quantum shots","compiled RZ/shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：分子内5精度で価格共有。H2逐次/確認後min、H4 primaryとの差、H6r32を維持。
- **支えないclaim**：Pauli対最強決定論の直接比較、DF因果説明、T/order外挿。
- **S1/S2/S3配置**：S1=SI/短いDiscussion、S2=Pauli境界の本文、S3=本文核。
- **既存図のままで足りるか・将来の注意**：RZ図として現H4粗例外を維持。B1/CXは全15でpure低点費用という保存結果を別参照し、RZ値を差替えない。

## V06：H4 P3: conditional matched-partition N / C / G comparison

- **内容・claim候補**：Q6;P05.cross_representation。P3共通Gaussian保持という条件付きcaseのN/C/G。
- **使用study / comparison / candidate**：["df_pauli_matched_partition_validation"] / ["P3_Stage4"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V06.svg)、manifestのsource_refs=["P3_ACCOUNT","VIS_P3_PRICES","P3_COMPARE","P3_STAGE4","P3_STAGE4_BINDINGS","P3_BIT","P3_AXES","P3_BIT_HEADROOM","P3_BIT_BINDINGS"]。scientific_SOURCE_roles={"df_pauli_matched_partition_validation":["Stage4:2f0d31f75ab1ecacc73d9ee5c34ddfc997396102","finite-bit:d22ca967955a664260a3a9857df243b3f151610f"]}。
- **適用条件**：{"XP02":"H4/tol-onlyDF10/Pauli184/DF rotated-Z/ZZ360・10basis;共通raw ray・Gaussian T_eff保持/V_square残差;T=.8/S2/q8/r4/K4;Stage3run04各arm32価格;finite-bit gap=0・compiled rho=0・B engineering coverage仮定"}
- **cohort・metric・価格種別**：["STAGE3_RUN04_PRICES_REUSED","STAGE3_RUN04_PRICES"] / ["RZ×shot","analytic quantum shots","compiled RZ/shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：Stage3/run04各arm32価格再利用。finite-bit gap=0、compiledρ0、B engineering coverageを保持。
- **支えないclaim**：表現だけの因果優位、最新family最良、DF B3対Pauli pureの直接同条件比較。
- **S1/S2/S3配置**：S1=SI/不採用候補、S2=SI、S3=SI/Discussion。
- **既存図のままで足りるか・将来の注意**：既存conditional注記を維持すれば補足に使える。finite-bit追補が全law精度を保証したような更新は不可。

## V07：DF B3 and Pauli pure RTE: definitions and comparison limits

- **内容・claim候補**：Q6;P05.cross_representation。DF B3とPauli pure RTEの保持/split/law/controlが異なるという比較ルール。
- **使用study / comparison / candidate**：["track_a_h46_long_time_b3","h6_hybrid_fairness_extension","h2_pauli_rte_validation","non_df_h4_pauli_pr_search","pauli_rte_event_extension"] / ["B3_fixed_negative","Pauli_latest14of15"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V07.svg)、manifestのsource_refs=["VIS_MATCHING","SRC_B3","PAULISYNTH","PRI_B3CSV","B3SUMMARY","PAULI_ADDENDUM","PRI_PA-E4","PRI_PA-E5","PRI_PA-E6compare","H6_P1","H6_P2","PRI_LONGPROV","DF_B3_CAND","SRC_B3SRC","PA_H6_CAND","PA_H6_SOURCE","PA_H2_SOURCE","PRI_PA-E3"]。scientific_SOURCE_roles={"track_a_h46_long_time_b3":[],"h6_hybrid_fairness_extension":["887b69e449cb3c378286f09d0d7626599850fb67"],"h2_pauli_rte_validation":["bf5f2339f723f4edb7bf91a8ca88f33251bbef57"],"non_df_h4_pauli_pr_search":[],"pauli_rte_event_extension":["e755ecf581600192fd2745d7066e410cebd933d3"]}。
- **適用条件**：{"XP04":"DF10/19長時間one-body保持 vs Pauli14/184/918 T.8全非identityランダム化;H/state/law/cohortは別"}
- **cohort・metric・価格種別**：[] / [] / []
- **不確かさ・共有**：CONTEXTUALの定義表であり、価格SE/nの観測表ではない。
- **支えないclaim**：費用差・優劣の証拠、同一algorithm比較。
- **S1/S2/S3配置**：S1=SI/必要時Discussion、S2=Methods本文候補、S3=Methods本文候補。
- **既存図のままで足りるか・将来の注意**：定義文脈として使用候補。両表現を因果比較していないことを明記。

## V08：Saved higher-order / S2 ratios: N, C_eff and G

- **内容・claim候補**：Q4;A09.order。保存されたhigher/S2のN・C_eff・Gの競合。
- **使用study / comparison / candidate**：["track_a_h46_long_time"] / ["C_N_G_decomposition.csv"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V08.svg)、manifestのsource_refs=["CNG","DF_LONG_CAND","DF_LONG_COMPARE","PRI_LONGBEST","PRI_LONGPROV","MATCHED","RK","SRC_LONG"]。scientific_SOURCE_roles={"track_a_h46_long_time":["provider:c31ef424175c02379eafebe8fc6c7548c1d9aabe","signal/native/extension:PRI_LONGPROV.source_commits"]}。
- **適用条件**：{"ME01":"H4DF10/H6DF19;元長時間探索選択のhigher対S2;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["SAVED_EXPLORATION"] / ["saved higher/S2 ratio"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：元探索選択の比。C_effとcos/sin片軸Cを混同しない。
- **支えないclaim**：qやnormalizationだけによる全Gの因果寄与、最適selector。
- **S1/S2/S3配置**：S1=SI/補助本文候補、S2=SI、S3=SI/不採用候補。
- **既存図のままで足りるか・将来の注意**：既存保存比の図として使用可能。B2-a整合からGの因果説明が完成した図とは呼ばない。

## V09：DF R boundaries: saved N / C / G at fixed T and precision

- **内容・claim候補**：Q3/Q4;A11.R_boundary。保存R境界でshot節約と価格負担が競合する有限例。
- **使用study / comparison / candidate**：["pr_boundary_execution_20261027"] / ["H4_R16_R128_vs_R64","H6_R11_vs_R16"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V09.svg)、manifestのsource_refs=["BOUNDARY_RESULTS","BOUNDARY_SUMMARY","BOUNDARY_REPORT"]。scientific_SOURCE_roles={"pr_boundary_execution_20261027":["execution:ad1286021bdc0e5a0b1760064cefb3c1168b24eb","accounting:0f2d167f9c928ba94bfad1a167bf4d3f5964ad1a"]}。
- **適用条件**：{"ME03":"H4 DF10/19;T2/q1/m5_best;H4 L_D3/K2又はH6 L_D7/K6;epsilon=.0005;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）","ME04":"H6 DF10/19;T2/q1/m5_best;H4 L_D3/K2又はH6 L_D7/K6;epsilon=.0005;linear/隣接1Å/STO-3G;保存H/state/reference/fragment順を各studyのidentity_refsで固定;P=0;元full measured Hadamard wrapper;Qiskit1.3/opt1/seed17（元例外はstudyを優先）"}
- **cohort・metric・価格種別**：["BOUNDARY_20261027_CONFIRMATION","SEVEN_20261027_CONFIRMATION","CONFIRMATION"] / ["RZ×shot","analytic quantum shots","compiled RZ/shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：H4 R16/R128 HIGHER、H6R11 point_lower=True/INCONCLUSIVE。元確認cohortと元2SE感度。
- **支えないclaim**：H4 R64全域最適、H6R11の確定改善。
- **S1/S2/S3配置**：S1=SI、S2=SI、S3=SI/不採用候補。
- **既存図のままで足りるか・将来の注意**：必要ならSI。manifestの複数system連結表現は原studyに従いH4 DF10/H6 DF19と読む。確認なしR32を追加しない。

## V10：H6 P1: saved signal differences and diagnostic resolution

- **内容・claim候補**：Q4;P05.bias_probe。H6 P1のouter split・保持内差・finite/gridとengineering分解能。
- **使用study / comparison / candidate**：["pauli_pure_rte_mechanism_validation/bias"] / ["P1_bias"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V10.svg)、manifestのsource_refs=["P1_DIFFERENCES","H6_P1","H6_P1_BINDINGS"]。scientific_SOURCE_roles={"pauli_pure_rte_mechanism_validation/bias":["a0a3aff915f98456733d6bf09ed641a1be423628"]}。
- **適用条件**：{"ME05":"H6Pauli918/T.8/S2/q8/K2;H prefix78 r16/32/64 vs pureRTE r64;保存full-H ray/reference;engineering floor1e-10"}
- **cohort・metric・価格種別**：["SAVED_SIGNAL_DIAGNOSTIC"] / ["dimensionless signal"] / ["ENGINEERING_DIAGNOSTIC"]
- **不確かさ・共有**：保存信号診断。構造的alias0、浮動残差、未解像を区別。
- **支えないclaim**：全G原因配分、UNRESOLVED=0、全law精度、DFへの移送。
- **S1/S2/S3配置**：S1=SI/不採用候補、S2=SI、S3=補助本文またはSI。
- **既存図のままで足りるか・将来の注意**：診断の図として使用可能。理論新規性や全G定量説明の主図にはしない。

## V11：H6 P2: fixed-program Full / R / E / Empty compiled prices

- **内容・claim候補**：Q4/Q5;F01.price_probe。固定programで保持価格・event価格・非加法性を区別する。
- **使用study / comparison / candidate**：["pauli_pure_rte_mechanism_validation/cost"] / ["P2_parts"]。全candidateは元DISPLAY_DATAのcandidate_id/source_selectorへ。
- **固定SOURCE・元データ**：図carrier `b3b194dbf405285239326fd25b671d30957b801c` の[元SVG](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/figures/V11.svg)、manifestのsource_refs=["P2_INTERACTIONS","H6_P2","H6_P2_BINDINGS"]。scientific_SOURCE_roles={"pauli_pure_rte_mechanism_validation/cost":["11e50e262f95630b14fa4070ad30894ba3f64b74"]}。
- **適用条件**：{"ME06":"H6Pauli918;保存confirmation/index0000;T.8/S2/q8/K2;Full6既存/部品12元compile"}
- **cohort・metric・価格種別**：["CONFIRMATION_INDEX_0000"] / ["compiled RZ/shot"] / ["DIRECT_SAVED_COMPILED_PRICE"]
- **不確かさ・共有**：confirmation/index0000固定。FullをR+Eへ加算復元しない。
- **支えないclaim**：母平均C、N/G因果分解、保持項だけの一般費用則。
- **S1/S2/S3配置**：S1=SI/不採用候補、S2=SI、S3=補助本文またはSI。
- **既存図のままで足りるか・将来の注意**：Full/R/E/Emptyを別棒のまま使う。P1と合わせてもN/G因果分離完成とはしない。

## B1：既存CX感度の保存結果

- **比較内容/claim**：Q5 / A11.P01.CX_sensitivity。元RZ nominee・元Nの固定CX感度。DFのmean-only点比較、Pauli/H8のpaired記述的SEを区別する。
- **固定SOURCE/candidate/cohort**：carrier `29c3bf0ecc687d458df55cf967ff38f990130b81` の[B1結果][B1]・[固定対応][FIXED]・[SOURCE trace][TRACE]。元study/comparison IDはclaim CSVから辿る。
- **支える/支えない**：metricによる維持・反転を支持。metric別再最適化、FT/runtime、Pauli旧primary対強PFのCX優位は未評価。
- **配置**：S1/S2/S3とも既存保存結論の本文参照が可能。V02/V03/V04/V05へ将来CX panelを付す案は別指示で判断する。**今回は新図・数値比較表を作らない**。
- **注意**：V03はRZ negative、CXのFull PF単独例外を保持。V05はRZ14/15であり、CX15条件と同じ表示ではない。DFのSE欠測を0で描かない。

## B2-a：既存normalization整合

- **内容/claim**：Q4 / A02.normalization。固定finite-K式・元整数配分・連続二乗近似の区別。Gの因果説明を支持しない。
- **SOURCE/条件**：carrier `29c3bf0ecc687d458df55cf967ff38f990130b81` の[B2-a結果][B2A]・固定candidate表・[報告][REAN]。H4/H6 DF10/19、B2 prefix3/5、B3 one-body保持/prefix0、T.8/2、q1/8、R16/32/64、K2/4/6、S2/Y4/m5、Linear1Å/STO-3G。価格cohortの新統計ではない。
- **配置**：S1/S2のMethods補足/SI、S3はSIまたは不採用候補。既存の再現結論と適用限界を引用すれば足り、新しい残差図を今回作らない。
- **注意**：APPROXIMATION_CONSISTENTは一律高accuracyの意味でない。Günther A.3の直接先行を明記し、二乗則を新発見の図として構成しない。

## 既存のPauli H4 primary対決定論表

- **内容/claim**：Q1 / P03.strong_deterministic。V05とは別に、Aへ直接答える既存表。
- **固定参照**：`d3fdfad880902b7ff35b710d22e1c9759f7087c2`、[元主表CSV][PRI_PA-E3]と[元report](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md) §3。184項/Linear1Å/STO-3G/T.8/S2/P0/central、保存106決定論対照、固定nominee確認。
- **配置**：S1はSI/Discussion、S2/S3は本文の直接証拠。新しい表へ再集計しない。粗精度Hybrid K2と最新内部比較Hybrid K4、厳精度のpure端点を明記する。

RPE/Track Bの既存表はAの成功例として混入させず、必要な場合だけSI/限界の固定参照に残す。図表の正式採用・最終デザイン・枚数はユーザー判断。全原図とSOURCE、台帳、旧原稿、STOPを保全する。

[SYN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md>
[OLDMAP]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv>
[STATUS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md>
[CAT]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/README.md>
[COMPS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/comparison_catalog.csv>
[MATCH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2779710a86d380bb484accc51d3dccba183370b8/docs/research/o2_data_catalog_20261105/df_pauli_matching.csv>
[FIG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/README.md>
[FM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/FIGURE_MANIFEST.csv>
[DISPLAY]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3b194dbf405285239326fd25b671d30957b801c/docs/research/o2_visualization_20261106/DISPLAY_DATA.csv>
[PRIOR]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/PRIOR_ART_MATRIX.md>
[TRACE]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/788ebad04e620055a80c75b1b3bef4a15712d4c2/docs/research/o2_claude_review_followup_20261106/SOURCE_FORMULA_TRACE.md>
[REAN]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/REPORT.md>
[B1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_CX_SENSITIVITY.csv>
[B2A]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B2A_NORMALIZATION_RESULTS.csv>
[FIXED]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/29c3bf0ecc687d458df55cf967ff38f990130b81/docs/research/o2_b1_b2a_saved_reanalysis_20261106/B1_FIXED_COMPARISONS.json>
[SRC_LONG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md>
[SRC_PF588]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md>
[SRC_B3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md>
[PAULISYNTH]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md>
[PAULI_ADDENDUM]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eb34dad982f8f753a618c609b528fb3c20d142/docs/research/pauli_h2_h4_h6_paper_evidence_addendum_20261102.md>
[PRI_PA-E3]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv>
[H6_P1]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md>
[H6_P2]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md>
[H6_P1_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/source_and_inputs.json>
[H6_P2_BINDINGS]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/source_and_inputs.json>
[P3_STAGE4]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md>
[H8_C]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md>
[G01]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv>
[G02]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv>
[CNG]: <https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/C_N_G_decomposition.csv>
