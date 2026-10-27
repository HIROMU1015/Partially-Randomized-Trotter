# 研究C M4：Proposed PR s1/r128・保存PF／RTE・shots・損益分岐価格

2026-10-28 JST。**固定SOURCEのM4は98/98 calls COMPLETE、correctness FAIL・retry/resume0。M3のPF kernel/全13 PF誤差を再利用し、r128の全13roundがPOTENTIALLY_FEASIBLE、登録2仮定scenarioのRPE/energy数値十分条件がPASSした。** Native価格は未知/nullのまま、実sampling/build/compile0。主仮定でM3を下回るRZ条件はμ<17747.772397、同J/U決定論を下回る条件はμ<6122.327980 RZ/kernel。これらは未知価格の条件で、r128の実総費用や優位を測定した結果ではない。

## 固定SOURCE・対象・再利用

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。起点`e61e79f68a29d0ff52a9d0a9448a569fc98e0e86`。
- Branch：`codex/research-c-M4-s1-r128-20261028`。
- 科学SOURCE：`7dbabc7866cc62a5460bf1579ce84c59493d8e1e`。
- [固定契約](research_c_pr_aware_df_M4_s1_r128_contract_20261028.json) SHA256：`0dcd3ffbaad909cab6db31819f8adeeb0b9ce1057390d5e279d4a4db8846bdf3`。
- [SOURCE84](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/technical/SOURCE.json) SHA256：`ff80968d49cc094825e23da94f3740b24844aa6ddd172fb430b458f2488c7634`。
- [準備・再利用・call内訳](research_c_pr_aware_df_M4_s1_r128_preparation_20261028.md)／[63人工・回帰PASS](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/technical/tests_final.log)。
- [原本75件](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/run01/)／[summary](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/execution_summary.json)／[postflight・原本hash](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/execution_postflight.json)。

ユーザーは技術準備・限定M4科学計算・損益分岐会計・次段階技術準備・公開を一括許可した。SOURCE/契約を固定公開後、追加承認なしでfresh run01を一度実行した。承認証明/原文保存/新grantは作らず、科学開始後のSOURCE/条件/閾値変更0。

保存直鎖H4・隣接1.00Å/STO-3G、通常number CDF8 Proposed（Stage A γ=.1終了J/U）、Nα=Nβ2/sector36、8spin modes、元g補正κ・核scalar一回・元frame0..7/LD2・tail identity抽出。Partial ordered symmetric S2、δ=h=.1、s1/**r128/K0**、m0..12/n=2^m、ideal finite-RTE IID reference。CA=.00159 Ha/DF reserveCA/8、βtotalπ/6・βstatπ/12、failure.01/axis.01/26、own-DF ground oracle/状態準備除外を維持。

Input SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、Proposed J/U `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、元P-H4 provenance `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。新fit/rank探索/ground eigensolve/積分再生成0、κ Hermitian化/U polar置換なし。

追加M4 adapter/CLI/testの3 SOURCEだけを用い、旧M3/B1/B2/C1/M2の科学SOURCEは変更しなかった。`prepare_arm/configure`、修正catalog_gate、既存head exponentials/finite-RTE、同じmatrix_power/norm2/attenuation、B1 feasibility、C1 qualification/assess_round/energy/Hoeffdingを再利用した。RTE分岐だけを呼ぶ薄い接続は新しいPF engineではない。

### 保存PFの同一性

M3科学SOURCE `5354edecc6aa995082440dd14be0310bdd9902e1`、manifest `b375bb1f9f979a70de7577729a28f3d7755a66e6643e7b46366d74acc830d857`の81 filesが不変。M3 SOURCE/STOP・同一input/JU/元κ/scalar/sector/LD2/head-tail/schedule/s1/h=.1/n/elapsed t=.1·2^m/NumPy/SciPy/binary64を照合した。rはideal PF branchに入らないため、保存M3 sector NPZのPFと全13 PF誤差をそのまま再利用した。新PF kernel/PF-exact error norm/13 exact evolutionは0。これはM3のraw/compiled回路をr128へ転用したという意味ではない。

Ground/aliasとDF norm+guardは同一Hamiltonianの保存PASSを再利用。新ground eigensolveなし。216atomの係数/符号/順序/basis/probability/identity/λが一致し、JSON tuple/list正規化も維持した。

## A：r128のRTE・attenuationと全13round

τ64=0.002253796696211913、τ128=0.0011268983481059565。数学的rationalとbinary64でτ128=τ64/2がPASS。λ_R=1.4424298855756241 Ha。新AとRTEはr128のconfigurationから評価し、r64値を移植していない。

| m | PF（再利用） | RTE128 | RTE64 | A128 | A64 | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 6.21615922e-05 | 4.67188595e-05 | 9.34398447e-05 | 0.99991873 | 0.999837466 | PASS |
| 1 | 0.000123986862 | 9.33575683e-05 | 0.000186723739 | 0.999837466 | 0.999674959 | PASS |
| 2 | 0.000245310383 | 0.000186068962 | 0.000372172319 | 0.999674959 | 0.999350024 | PASS |
| 3 | 0.000470161325 | 0.000367052608 | 0.000734239499 | 0.999350023 | 0.998700471 | PASS |
| 4 | 0.000800736972 | 0.00069679264 | 0.00139406997 | 0.998700468 | 0.997402631 | PASS |
| 5 | 0.000962476519 | 0.00117604118 | 0.00235346423 | 0.997402626 | 0.994812008 | PASS |
| 6 | 0.00129519545 | 0.00161211698 | 0.0032268316 | 0.994811998 | 0.989650931 | PASS |
| 7 | 0.00245126611 | 0.00212238226 | 0.00424926731 | 0.989650911 | 0.979408964 | PASS |
| 8 | 0.00450460743 | 0.00392982142 | 0.00787508358 | 0.979408925 | 0.959241919 | PASS |
| 9 | 0.00897375395 | 0.00751667111 | 0.0150898413 | 0.959241843 | 0.92014506 | PASS |
| 10 | 0.0179343394 | 0.0150054678 | 0.0302360925 | 0.920144914 | 0.846666931 | PASS |
| 11 | 0.0358354926 | 0.0300247092 | 0.0609508949 | 0.846666663 | 0.716844892 | PASS |
| 12 | 0.0716149979 | 0.0609453085 | 0.125604928 | 0.716844438 | 0.5138666 | PASS |

全13roundで既存十分条件εPF+εRTE≤sin(π/12)、A>0と保存DF reserveを満たし、POTENTIALLY_FEASIBLE。PFは||U_PF(h)^n−exp(−iH_DF t)||₂、RTEは||U_corrected(h)^n−U_PF(h)^n||₂。物理DF誤差Haをoperator/signal誤差へ直接加算しない。これはbinary64 numerical pilotで、厳密enclosure/実装law/全law合成/actual CAの保証ではない。

[全13行CSV](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/rounds13.csv)。

## B：新shots・energy数値十分条件

Evidence laneは未取得の全law synthesis/implemented-law raw-signal誤差をnullにし、全13round未資格・shots/total null。`ideal_exact_native`はexact synthesis/ideal finite-RTE IIDの反実仮定、`assumed_raw_signal_budget`は旧C1同様に各roundのsynthesis/law誤差を各.001 A以内とする未検証感度仮定である。新しい有利な誤差配分は加えない。旧B2価格proxy仮定は保存比較対象だけに適用し、新r128価格には移送しない。

| m | r128 ideal N/axis | M3 r64 N/axis | r128 .001A各項 N/axis | M3 r64 .001A各項 |
|---:|---:|---:|---:|---:|
| 0 | 512 | 512 | 514 | 514 |
| 1 | 512 | 512 | 514 | 514 |
| 2 | 512 | 513 | 514 | 515 |
| 3 | 513 | 514 | 515 | 516 |
| 4 | 514 | 516 | 516 | 518 |
| 5 | 516 | 520 | 518 | 522 |
| 6 | 520 | 527 | 522 | 529 |
| 7 | 527 | 540 | 529 | 543 |
| 8 | 542 | 570 | 544 | 572 |
| 9 | 575 | 634 | 577 | 637 |
| 10 | 646 | 787 | 648 | 791 |
| 11 | 817 | 1219 | 821 | 1225 |
| 12 | 1322 | 3003 | 1328 | 3018 |

| Scenario | r128 shots | r64 shots | W128 | W64 | energy判定 |
|---|---:|---:|---:|---:|---|
| ideal_exact_native | 16056 | 20734 | 16631888 | 32417320 | CONDITIONAL_CA_PASS |
| assumed_raw_signal_budget | 16120 | 20828 | 16705612 | 32578340 | CONDITIONAL_CA_PASS |

両scenarioのenergy数値十分評価値は0.0012783296188530234 Ha、candidate failure upper .01、DF予算は保存norm+guardを一度だけ計上した。実測energy誤差/actual CAではない。Shotsは既存Hoeffdingの整数十分数で、情報理論的最小数や実量子測定ではない。保存M3 shotsを転用せず、新PF/RTE/Aから算定した。

| Scenario/group | 新shots | M3 shots | 新kernel重み | M3 kernel重み |
|---|---:|---:|---:|---:|
| ideal_exact_native/m0_to9 | 10486 | 10716 | 1132624 | 1211944 |
| ideal_exact_native/m10_to12 | 5570 | 10018 | 15499264 | 31205376 |
| ideal_exact_native/m12 | 2644 | 6006 | 10829824 | 24600576 |
| assumed_raw_signal_budget/m0_to9 | 10526 | 10760 | 1136716 | 1217316 |
| assumed_raw_signal_budget/m10_to12 | 5594 | 10068 | 15568896 | 31361024 |
| assumed_raw_signal_budget/m12 | 2656 | 6036 | 10878976 | 24723456 |

[全78axis行](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/axes78.csv)／[高m groups](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/high_m_groups6.csv)。Kernel反復が2^mなので、高mのshots改善がWと価格許容量を大きく変える。

## C：未知価格μの損益分岐

`C_k(μ_k)=W128 μ_k+S128,k`、`W128=Σ_m,axis N_m,axis·2^m`、`S128,k=Σ_m,axis N_m,axis C_shell,axis,k`。Shellは各interrogationで一度だけ計上し、kernel価格へ混ぜない。6指標は独立に会計し、weighted scoreを作らない。

| Scenario | W | S(RZ/RZdepth) | S(CX/CXdepth) | S(totaldepth/size) |
|---|---:|---:|---:|---:|
| ideal_exact_native | 16631888 | 56196 | 0 | 104364 |
| assumed_raw_signal_budget | 16705612 | 56420 | 0 | 104780 |

比較対象総費用Ctargetに対し、同費用となる価格は`μ_BE=(Ctarget−S)/W`。W>0と登録精度適格性・費用のfinite性を確認した。**非負の実平均価格μが閾値未満なら、該当する条件付き点費用は比較対象未満。等値は同費用。** 閾値≤0では非負価格でstrict優位を出さない。今回は36閾値すべて正である。

| Scenario | 対象 | RZ損益分岐 | M3平均からの価格余裕 |
|---|---|---:|---:|
| ideal_exact_native | M3_r64 | 17747.772397021 | 8642.178647021 |
| ideal_exact_native | M2_deterministic | 6122.327979782 | -2983.265770218 |
| ideal_exact_native | old_PR_s4 | 32485.193738257 | 23379.599988257 |
| assumed_raw_signal_budget | M3_r64 | 17757.215094387 | 8651.621344387 |
| assumed_raw_signal_budget | M2_deterministic | 6095.309350834 | -3010.284399166 |
| assumed_raw_signal_budget | old_PR_s4 | 32490.240422500 | 23384.646672500 |

M3の保存32標本平均は9105.59375 RZ/kernel。主仮定で追加64eventsと変更後compile全体に許される会計上の余裕は、M3改善用+8642.178647 RZ、決定論超え用−2983.265770 RZである。これはevent単価の測定でも保証でもない。

| Metric | ideal：M3未満 | ideal：決定論未満 | ideal：旧PR未満 | 感度：M3未満 | 感度：決定論未満 | 感度：旧PR未満 |
|---|---:|---:|---:|---:|---:|---:|
| rz_count | 17747.772397 | 6122.327980 | 32485.193738 | 17757.215094 | 6095.309351 | 32490.240422 |
| rz_depth | 5955.678439 | 3091.703191 | 13194.687447 | 5958.847157 | 3078.059096 | 13196.737287 |
| cx_count | 7625.391668 | 3886.840147 | 16855.629162 | 7629.448754 | 3869.687025 | 16858.247741 |
| cx_depth | 5127.429037 | 3484.829823 | 13304.598053 | 5130.157083 | 3469.450824 | 13306.664967 |
| total_depth | 13081.795154 | 6717.569344 | 28791.874931 | 13088.755321 | 6687.923827 | 28796.347847 |
| circuit_size | 35054.742050 | 11615.543192 | 62888.035974 | 35073.392916 | 11564.282291 | 62897.805843 |

Depthは既存fenced serial段のscheduled和。長回路のglobal再compile、未観測fusion、価格のevent数への線形性を仮定していない。旧PR/M3費用は有限cohort平均と仮定付きshotsの点値、M2決定論は数値synthesis pilotで、資格と融合範囲も異なる。閾値は母平均費用優位やactual CAの認証ではない。

### 高m改善の会計上の寄与

| Scenario/対象 | M3 shotsのままの閾値 | m10..12だけ新shots | 全round新shots |
|---|---:|---:|---:|
| ideal_exact_native/M3_r64 | 9105.593750 | 17663.532280 | 17747.772397 |
| ideal_exact_native/M2_deterministic | 3141.094233 | 6093.268210 | 6122.327980 |
| ideal_exact_native/old_PR_s4 | 16666.710497 | 32331.002230 | 32485.193738 |
| assumed_raw_signal_budget/M3_r64 | 9105.593750 | 17671.952716 | 17757.215094 |
| assumed_raw_signal_budget/M2_deterministic | 3125.569214 | 6066.042310 | 6095.309351 |
| assumed_raw_signal_budget/old_PR_s4 | 16660.435547 | 32334.236543 | 32490.240422 |

これは事前登録したshots会計の置換で、別r/PF候補・物理的な新状態・追加samplingではない。主仮定のM3閾値増分は高mだけで8557.938530、全roundでは8642.178647 RZ/kernel。残る低mの寄与も省略せず原本へ保存した。

[全36閾値・headroom・高m置換CSV](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/breakeven36.csv)。**実r128 kernel価格と実総費用はnullのまま**で、9105.59375やその2倍を測定r128価格に置き換えていない。

## 次段階：128-event native校正の技術準備のみ

M3改善の価格境界約17748 RZ/kernelを実32標本平均が下回るかには情報価値がある。決定論超えの境界約6122は現r64平均9106より低く、今回の価格政策で強決定論を超えるには相当に低い128-event価格が必要。ただしevent数に対する価格単調性や厳密下界を証明していないため、不可能とは断定しない。M3からの改善を確定する測定と、決定論超えの期待は分ける。

既存B2/M3-Bの演算・Gaussian・CAR・compiler・full512 gate・calibrationを再利用する[native専用adapter](../../src/trottertracks/c_pr_aware_df_m4_native.py)と[passive既定runner](../../scripts/run_research_c_pr_aware_df_M4_native.py)を用意した。新しいのはr128/count/seed/保存M4 W/Sへのbindingとorchestrationだけ。旧M3の471-call runnerは変更・再起動しない。

- [未承認native契約](research_c_pr_aware_df_M4_native128_contract_unapproved_20261028.json)：**最大332 units**＝attachment7＋M3-Bの比較以外324＋最終資源接続1。M4のPF/RTE/shots/eigensolve再計算0。
- 固定32非paired PCG64 substreams、新master2026102805/cohort `M4_r128_native_fixed32_v1`、各128 signed K0 events。旧cohort pool・適応追加なし。固定seedはIID証明ではない。
- Whole h=.1/P0 opt1/seed17/MPFR160、head relative/同一basis tail共有、cross-head/tail一般相対融合なし。Full512 raw/compiled/reference/相対位相/unitarity/actual adjoint/X/Y、matrix1e-10/signal2e-10/rtol0。新QPY128、P0compile32、reference NPZ32。
- 保存M4の2scenarioのW/S/6指標閾値へ実有限cohort平均を接続。SEは同じ32標本のW倍で、sqrt13で縮めない。Observed min/maxをall-law supportへ昇格しない。
- [5人工PASS](../../artifacts/research_c_pr_aware_df_M4/2026-10-28/technical/native_preparation_tests.log)：新128-event sampler binding/独立namespace、人工full512/P0/adjoint/XY、保存affine、mock32 dispatch/128QPY/call計上、passive/partial STOP/fresh拒否。人工回路はinvented入力で、保存H4を使っていない。
- 次SOURCE87 manifest SHA256 `093a63379b05864b79acbc9bdb4d3fe734603a812b6119215625256106256d97`、未承認契約SHA256 `af5ad8ca4b8b765d017ba1d9ff6e79b91759f4ec80a20d8d9500c9873805d54e`。Input63を固定し、future fresh `artifacts/research_c_pr_aware_df_M4_native/2026-10-28/run01`は未作成。
- M4の許可ではこのnative科学実行は開始しない。別途ユーザーの明示許可が必要。高価格でも32件を完了して不利な結果を保存し、correctness FAILでは依存処理を停止する。Retry/resume0、公開照合後mandatory STOP。

## Q1〜Q9と次の判断

1. 減衰/RTE：m12 Aは.51386660→.71684444、RTEは.12560493→.06094531へ改善した。
2. Feasibility：13/13 PASS。科学的不適格0、evidence未資格13はそのまま。
3. PF再利用：入力/科学SOURCE/sector/時刻/schedule同一性を照合し、PF kernelと13誤差を再利用。新PF/exact計算0。
4. Shots：主仮定20734→16056、m12各軸3003→1322。高m10..12のshots/反復重み低下が閾値改善の大部分に対応する。
5. M3改善：主17747.772397、感度17757.215094 RZ/kernelが同費用の境界。未満なら当該条件付き点会計で低費用。
6. 決定論超え：主6122.327980、感度6095.309351 RZ/kernel。現r64価格以上のままならこの目標には届かないという条件付き判断で、r128実価格や不可能性の認定ではない。
7. 副指標：6指標別の36閾値を保存した。異なる指標を統合して恣意的順位を出さない。
8. Native測定の価値：M3改善の境界を判別するための限定32校正には価値がある。決定論超えを既定の成果にしない。価格mean/SEが境界付近なら、保証CIや母平均の勝敗とは扱わない。
9. 主要制約：r増加で減衰/RTE/shotsは改善したがPF誤差.071615は固定されて残る。これ以上の資源判断の直接の未確定量は128-eventの実native価格である。値を得ずにr探索/新fit/保証構築へ広げない。

次は**ユーザー／研究C**が未承認332-call native校正を実施するか判断し、**Codex**は別明示許可後だけ固定範囲を実行する。今回その測定・別r/γ/arm/H6/fit/保証/詳細レビューは行っていない。

## Calls・資源・保全・mandatory STOP

| stage | 最大/COMPLETE |
|---|---:|
| A | 39/39 |
| B | 48/48 |
| C | 10/10 |
| program summary | 1/1 |
| 合計 | **98/98** |

人工・回帰はM4 18＋M3 22＋C1 23＝63 PASS（34.75秒）。別のfuture-native技術検証5 PASSを加えて68 unique PASS。M4科学callとは区別する。Correctness FAIL0、科学的INFEASIBLE0、feasibility UNRESOLVED0、evidence未資格13。Retry/resume0。

Runner wall 1.60306996945秒、process CPU差 16.325143089秒、parent peak RSS 377836KiB。開始時affinity32、MemAvailable約56.6GB、memory pressure avg10/60/300=0。既存plannerは13sector36 round workersを選び、BLAS1で協調した。

SOURCE/input読込・小さい準備演算は逐次、13roundは独立thread、B/Cは小さいscalar会計。CPU/RAM/worker/wall人工capなし。Task-local venv/cache/output、既存依存read-only、共有設定/他job変更0。巨大full512分子行列・QPYはM4では作っていない。速度向上率は比較測定していない。

M4原本75files/364901bytes、SOURCE84/input58/旧raw564がhash不変。旧M3run01/run02、M2/B1/B2/C1/F2-K1/F3-K1のSOURCE/raw/契約/STOPを保全した。後段native準備は別3SOURCEで、科学実行済みM4 SOURCE84を変更していない。

結果・技術準備・call・STOPをcommit/pushし、remote SHA/tree/SOURCE/input/旧raw/新原本を照合後**mandatory STOP**。実native128科学実行へ自動進行しない。
