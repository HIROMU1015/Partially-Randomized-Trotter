# C1 Stage3：compile競合修正後の8候補cost-only結果

2026-10-25 JST。Branch `codex/research-c-c1-stage3-cost-compat-20261025`。
固定公開SOURCE `9ce6703c89874c142c64ad934354bdff0db703d1`。

**限定8候補の費用集計が正常完了した。科学call 936/936、compile・位相同値性・unitarity各133件がPASS、候補合計8・round/axis208行・登録比較12件を保存した。前回のQiskit並列compile例外は今回再発しなかった。結果の公開照合後mandatory STOP。**

[要約](../../artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/execution_summary.json)／[候補8表](../../artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/candidate_costs.csv)／[round/axis208行](../../artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/round_axis_costs.csv)／[固定12比較](../../artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/comparisons.csv)／[原本・QPY・call・STOP](../../artifacts/research_c_c1_stage3_cost_compat_science/2026-10-25/run01/)／[修正・人工再現](research_c_c1_stage3_cost_compat_repair_20261025.md)／[固定契約](research_c_c1_stage3_cost_compat_contract_20261025.json)。

## 固定条件・再利用

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2）、CDF保存rank8 number・元g補正κ／scalar／元frame順、XDF保存rank10。CA=0.00159 Ha・DF reserve CA/8、δ=0.1、m=0..12、各DF sector ground oracle・状態準備除外。各armで決定論ordered S2 s=1/4、PR L_D=2・s=4/r=16/K=0/2の8候補。前段20候補の結果は全保全し、対象外12候補を科学的NO-GOにしない。

保存feasibility・楽観的shots（公開証拠f85ac4c…）、Stage2のD(s1)費用を再利用した。PF/RTE・attenuation・shots式・物理DF normの再計算、新fit／optimizer、実量子shotsは0。既知のStage1物理sector誤差はCDF約7.6660e-5 Ha／XDF約3.3752e-15 Haと異なり、今回もそれぞれを保持する。

P0はQiskit1.3.0・opt1・seed17・rz/sx/x/cx・approximation_degree1.0・qubits_initially_zero=false。既存のfull-frame Gaussian、合法的basis共有・境界融合・角度結合、serial fenceを両armへ適用し、未観測のliteral/shared追加削減を仮定しない。κ／scalar／controlled相対位相・actual adjoint等の既存native接続は変更していない。今回の新検査はraw対P0とunitarityであり、独立分子referenceを再計算したという主張ではない。

旧SOURCE19ファイルを変更せず、compile callbackだけをprocess内mutexで排他化した。人工100 tests PASS、同じ大きい人工8回路で旧3例外→修正後0を確認し、新SOURCEをcommit・push・remote照合後にfresh run01で起動した。科学候補・種・P0条件・許容差・call上限を変えていない。

## 登録8候補の条件付き費用

主指標は保存楽観的Nに条件付けた期待総RZの**点推定**。下表のkernelは1 substep、13 round X/Yの合計はΣ N[m,axis]・(s・2^m・E[kernel]+readout)。全6指標の全桁有理集計は原本JSON／CSVにある。

| 候補 | 保存楽観shots合計 | kernel平均RZ | 条件付き総RZ点推定 | 条件付き総CX点推定 | 費用support |
|---|---:|---:|---:|---:|---|
| CDF_D_s1_r0_K0 | 13606 | 12,118.000 | 1.11798946e+11 | 6.78284786e+10 | 全support |
| CDF_D_s4_r0_K0 | 13314 | 12,118.000 | 4.08641595e+11 | 2.47923144e+11 | 全support |
| CDF_PR_s4_r16_K0 | 58356 | 5,453.969 | 3.79761296e+12 | 1.87187905e+12 | 点推定のみ |
| CDF_PR_s4_r16_K2 | 573646 | 5,512.531 | 4.93496987e+13 | 2.42377255e+13 | 点推定のみ |
| XDF_D_s1_r0_K0 | 13600 | 16,659.000 | 1.53454726e+11 | 8.76936514e+10 | 全support |
| XDF_D_s4_r0_K0 | 13314 | 16,627.000 | 5.60693479e+11 | 3.20492595e+11 | 全support |
| XDF_PR_s4_r16_K0 | 21408 | 6,475.719 | 8.98254794e+11 | 4.05574210e+11 | 点推定のみ |
| XDF_PR_s4_r16_K2 | 46502 | 6,670.562 | 3.41366298e+12 | 1.52664730e+12 | 点推定のみ |

PRの32校正sampleは固定master seed 2026102503を用いたcandidate別独立stream。各量子shot・各coherent occurrenceにはfresh独立trajectoryを使う契約で、校正で得た1sampleを2^m・s回繰り返す量子実装ではない。費用の線形性を使い、反復回数を平均kernel費用へ掛ける。校正の有限sample不確かさとfinite-bit samplerからideal lawへの平均値移送は未資格のまま。

depthはfenceのあるstaged serial実行のdepth和で、長い全回路を新たに一括compileしたdepthではない。状態準備費用は除外され、除外が候補間で相殺されるとの主張もしない。

## 登録比較と読める範囲

比は左／右。1未満は当該点推定の左が小さい意味であり、保証付き同CA優位ではない。比較順序は契約どおりで、後付けの候補選択・最終順位付けは行わない。

| 登録ペア | 総RZ点推定比 | 費用区間の判定 |
|---|---:|---|
| `CDF_D_s1_r0_K0` / `XDF_D_s1_r0_K0` | 0.72854678 | LEFT_LOWER_CONDITIONAL_COST_AT_OPTIMISTIC_SHOTS |
| `CDF_D_s4_r0_K0` / `XDF_D_s4_r0_K0` | 0.7288146 | LEFT_LOWER_CONDITIONAL_COST_AT_OPTIMISTIC_SHOTS |
| `CDF_PR_s4_r16_K0` / `XDF_PR_s4_r16_K0` | 4.2277681 | POINT_ESTIMATES_ONLY |
| `CDF_PR_s4_r16_K2` / `XDF_PR_s4_r16_K2` | 14.456523 | POINT_ESTIMATES_ONLY |
| `CDF_PR_s4_r16_K0` / `CDF_D_s1_r0_K0` | 33.968236 | POINT_ESTIMATES_ONLY |
| `CDF_PR_s4_r16_K0` / `CDF_D_s4_r0_K0` | 9.2932609 | POINT_ESTIMATES_ONLY |
| `CDF_PR_s4_r16_K2` / `CDF_D_s1_r0_K0` | 441.4147 | POINT_ESTIMATES_ONLY |
| `CDF_PR_s4_r16_K2` / `CDF_D_s4_r0_K0` | 120.76524 | POINT_ESTIMATES_ONLY |
| `XDF_PR_s4_r16_K0` / `XDF_D_s1_r0_K0` | 5.8535492 | POINT_ESTIMATES_ONLY |
| `XDF_PR_s4_r16_K0` / `XDF_D_s4_r0_K0` | 1.6020425 | POINT_ESTIMATES_ONLY |
| `XDF_PR_s4_r16_K2` / `XDF_D_s1_r0_K0` | 22.245408 | POINT_ESTIMATES_ONLY |
| `XDF_PR_s4_r16_K2` / `XDF_D_s4_r0_K0` | 6.0882873 | POINT_ESTIMATES_ONLY |

この登録範囲の点推定では、PRのkernel削減は減衰に伴うshots増を補っていない。同じarmのD(s4)に対する総RZ比はK0でCDF9.2933／XDF1.6020、K2でCDF120.765／XDF6.0883だった。一方、同じ設定のCDF/XDF比はD(s1/s4)で約0.729、PR K0で4.2278・K2で14.4565であり、CDFの回路単体削減をPRの総費用削減へそのまま移せない。

これは登録12ペアの校正点推定の記述で、最終protocolの順位付け・採否・研究全体のNO-GO判定ではない。CDF/XDFはDF近似精度が異なり、各自DF ground oracleを仮定する。この観測を無条件の総資源優位へ拡張しない。

決定論4候補は全費用supportを保持する。PR4候補は全word supportが登録32枠を超え、費用区間はnull・POINT_ESTIMATES_ONLY。標本最大値を上界へ昇格せず、未知の合成誤差・実装law誤差もnullを保持する。従って楽観的Nは実際の十分shotsの保証ではなく、CA達成・保証付き総資源優位は未認定。今回の合成残差は実行した有限回路上の数値検査で、全law誤差certificateではない。

## 検査・call・資源

- preflight／実行後SOURCE23・保存input20・元input・依存5版identity照合PASS。
- compile・raw対P0相対位相・unitarity133件、metrics133件が完了、FAIL0。3 readout shell＋D(s4)2＋PR128の計133。D(s1)2はStage2証拠再利用。
- 保存した新kernel130件の最大成分差 `5.8220197878097778e-11`、unitarity最大 `4.7351012000262926e-13`。固定の各1e-10以内、rtol=0・global phase合わせなし。shell3件もcall PASSであるが、旧backendが個別数値残差を保存しないため最大値の母集団は130と明記する。
- QPY 266個、候補合計8、208round/axis行、固定比較12。新fit／optimizer／PF/RTE・shots式再計算／実量子shots0。
- 旧技術的STOP106 callsを保全。修正後fresh attempt1回・936 calls、2attempt累積1042 calls。当attemptのin-place retry／resume0。これはユーザーが許可したSOURCE修正後のfresh再実行で、旧STOPのresumeではない。

| logical kind | 今回 | 最大 |
|---|---:|---:|
| input_load | 1 | 1 |
| fit_export_load | 1 | 1 |
| model_prepare | 2 | 2 |
| saved_shots_load | 1 | 1 |
| saved_stage2_load | 1 | 1 |
| split_native_model | 2 | 2 |
| gaussian_build | 20 | 20 |
| rte_configuration | 4 | 4 |
| cost_support_probe | 4 | 4 |
| cost_kernel_sample | 128 | 128 |
| circuit_build | 133 | 133 |
| compile | 133 | 133 |
| compile_equivalence_check | 133 | 133 |
| metrics | 133 | 133 |
| native_calibration | 8 | 8 |
| cost_interval_assessment | 4 | 4 |
| serial_cost_expansion | 208 | 208 |
| optimistic_total_aggregate | 8 | 8 |
| cost_comparison | 12 | 12 |

8 candidate workers、開始時affinity32 CPU・空きRAM約52 GiB、process内BLAS1でoversubscriptionを抑え、compile以外の独立処理は並列のまま実行した。wall 773.651秒、process CPU 2517.364秒、peak RSS 664,588 KiB。compile排他完了133回、全thread待ち時間の合計27.172秒（wall時間ではない）。人工wall／CPU／RAM cap、共有環境や他ユーザーjobへの変更はない。数値Operator照合が主要な残余処理で、未測定の高速化率は主張しない。

契約SHA256 `4092a5d892eda087e4e3d82c3fe13b0e020c9da9f6ff5af406d6aff6a30a0a42`、SOURCE manifest SHA256 `e8d3cd6ac36763feab5c599ce240335338a75121db690eab41499ac188900a06`。元input SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。実行後は保存JSON・journal・bytesの照合、同じ保存有理和の整合確認、CSV整形だけを実施し、追加Operator／compile／PF/RTE／shots計算へ進めていない。既存SOURCE・旧結果・grant・ledger・marker・STOPを保全した。local committed evidenceであり、CIや外部独立再現ではない。

## 次の担当とmandatory STOP

ユーザー／研究Cが、登録ペアの点推定差とPRの未資格不確かさを確認し、次に費用期待値／合成・law誤差のどの資格を優先するか判断する。Codexはその具体的依頼と、追加科学計算が必要なら別の明示許可を受けてから進める。今回の結果だけで同CA達成、無条件総資源優位、対象外候補のNO-GOを採択しない。

結果・証拠をcommit・push・remote照合後、**mandatory STOP**。追加科学計算や次stageへ自動進行しない。
