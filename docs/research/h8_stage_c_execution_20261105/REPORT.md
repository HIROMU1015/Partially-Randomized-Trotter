# H8長時間・高次PR：Stage C独立確認実験の結果

状態：固定9標本・18価格と結果集計が完了。代表index0は元の7検査すべてPASS。追加標本・自動retry/resumeなし。

実行：2026-11-05T00:31:24.562287+09:00 ～ 2026-11-05T01:22:15.014179+09:00。9 workerで並列実行。
実行SOURCE：`82fdbdd098e9d3edae01a930437e3d76c1f8bf36`。run_tag：`H8_M5_P8_Q1_CONFIRMATION_01`。namespace：`H8_STAGE_C_CONFIRMATION_V1`。

Linear H8、隣接1.00 Å、STO-3G、保存tol-only1e-8 DF28、16 system qubit・spin sector4900、元state・fragment順序0..27。T=1.6、ε=.0005、状態準備P=0。
PR：4th(m5_best)/prefix8/q1/R_outer32/K4、ID `7eb0a3d691b94089969b`、固定N_cos=84916951・N_sin=88339169。
固定PF：B0I/4th(m5_best)/prefix15/q1、ID `1425ae6ed83fd9098b39`、保存G=89291219516568 RZ。PFは保存両軸価格613049 RZ、N_cos=71893270・N_sin=73757762を引用し、新sampling/build/compileなし。元cohort=EXPLORATIONを保持する。
Qiskit1.3.0、rz/sx/x/cx、opt1、transpiler seed17、backend/target/topologyなし。費用はmeasurement付きfull Hadamard wrapperのcompiled RZによるG=N_cos C_cos+N_sin C_sin。

## 確認標本だけの会計と判定

| 項目 | 結果 |
|---|---:|
| 平均G | 67335317264200 RZ |
| 固定PFとの比 | 0.754109056061 |
| 固定PFとの差 PR−PF | -21955902252368 RZ |
| 観測平均の削減率 | 24.58909439% |
| paired G sample variance（n−1） | 95815404731724237421200 RZ² |
| paired G sample SE | 103180211675.02368 RZ |
| 固定h_G | 121655623062.28098 RZ |
| 比較方向 PR平均 < 固定PF | 成立 |
| 記述的SE ≤ h_G | 達成 |

同一trajectoryのcos/sinを固定Nで重み付けしてから分散・SEを求めた。既存paired_Gを再利用し、軸間共分散を含む。探索標本とはpoolしない。95%信頼区間・新しい有意水準は採択していない。

## 回路検査と位相会計

| 元検査名 | error | tolerance |
|---|---:|---:|
| LOW_CONTROL | 2.10155160e-14 | 1e-08 |
| HIGH_CONTROL_LEAKAGE | 0.00000000e+00 | 1e-08 |
| SAVED_BLOCK_ALGEBRAIC_EVENT_VECTOR | 6.03110301e-14 | 1e-08 |
| HADAMARD_cosine | 2.50910404e-14 | 1e-08 |
| COMPILED_VECTOR_cosine | 1.02028738e-11 | 1e-07 |
| HADAMARD_sine | 4.46309656e-14 | 1e-08 |
| COMPILED_VECTOR_sine | 7.82414321e-12 | 1e-07 |

全検査はstatevectorのglobal-phase alignmentなし。固定compiler位相補助は、実際の小gate operatorから導出した余分な位相をfull wrapperのglobal phaseへ会計した。RZ数・gate数・許容値は変更していない。
今回の代表はXX+YY gate 52214個に対しΔφ=π（mod 2π）。局所operator最大誤差3.828435720135541e-14。両軸の証拠hashは公開JSONに保存し、private QPY・局所gate実値は公開しない。

## 希少次数と結論の限界

確認9標本のevent観測はorder0=288、order2=0、order4=0。未観測order2/4の条件付きcompiled費用はunknown/null、population tail cost boundもnull。0で補完していない。
保存sampling lawはUNCHANGED_PCG64_NUMPY_CHOICE_FINITE_PAIRED_RTE。同lawで1 trajectoryにorder2以上が含まれる確率0.017503368117490408、order4以上は1.7365913367114017e-6。9標本で未観測となるlawモデルの確率は0.8530601056102244／0.9999843707865361。これは保存lawの診断で、PRNGの数学的独立性の証明ではない。
独立seed namespaceの新cohortでも、固定PFに対する観測平均の低さと記述的SE目標を確認した。一方、未観測rare群の費用・分散と一様なpost-compile費用上界が未確立なので、母集団平均の優位・母集団精度を保証しない。全候補最適性、全RPE総費用、他分子・別compilerへの一般性も未検証。

## Stage Bとの比較（別cohortのまま）

| cohort | n | 平均G | sample SE |
|---|---:|---:|---:|
| EXPLORATION（保存値） | 9 | 67323362591920 | 121655623062.28098 |
| CONFIRMATION | 9 | 67335317264200 | 103180211675.02368 |

確認平均−探索平均は 11954672280 RZ（探索平均比 0.01775709%）。これは記述比較で、両群を混合した推定や新しい有意差検定ではない。

## 成果物・SOURCE・保全とSTOP

[確認JSON・provenance](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/confirmation.json)、[9標本の価格CSV](../../../artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/paired_prices.csv)、[報告用コード](../../../scripts/report_h8_stage_c.py)。
[固定計画](../h8_stage_c_preparation_20261104/PLAN.md)、[固定scope案](../../../configs/h8_long_time/stage_c_proposal_20261104.json)、[Stage A/B保存結果](../h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)。
Signal SOURCE `d1272016592ff06c0567862f770ed796fd61b366`、native SOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`、位相補助SOURCE `18ce41942051065fa097644a8a937bd11adba708`。Stage A/B圧縮結果hash、入力binding、今回SOURCE hash、原結果hashはJSONから辿る。
元Stage A/BのSOURCE・raw/QPY・結果・旧grant/ledger/marker・失敗/HOLD・STOPを保全した。新cohortのseed一意性と旧seedとの非衝突は実行時に確認済み。seed実値・event列・入力raw・QPY・内部pathはprivateに保持する。
これは主担当Codexによる実行・技術確認であり、別担当者の独立科学レビューや外部再現ではない。
起動記録は本REPORTの直前公開版commit `6439f59d8a24667dfaf207102d738dcff1729c98` に保存されている。

**固定9標本の実行・集計を完了。結果公開とremote照合後はmandatory STOP。希少費用検証・母集団保証・別候補探索・総RPE評価・詳細科学レビューへ自動移行しない。**
次の作業なし（今回の許可範囲を完了）。
