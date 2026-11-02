# 研究C M3：JSON修正後のs1/r64・限定科学実行結果

2026-10-28 JST。**修正SOURCEを固定したfresh run02で471/471 logical calls COMPLETE、correctness FAIL0。Aの全13roundが適格、Bの32trajectory full512検証がPASSし、Cの登録2仮定scenarioまで完了した。** 旧run01の19-call技術STOPは保全し、今回をtechnical relaunch1として区別する。結果は明示仮定付き資源点推定で、actual CA/保証shots/ideal-law母平均優位を認定しない。

## SOURCE・修正・固定条件

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。起点`def3748fc33f923f1a684a5d3ee0608fd7818a44`。
- Branch：`codex/research-c-M3-s1-r64-repair-20261028`。
- 科学SOURCE：`5354edecc6aa995082440dd14be0310bdd9902e1`。実行中のSOURCE/条件変更0。
- [修正契約](research_c_pr_aware_df_M3_s1_r64_repair_contract_20261028.json) SHA256：`8c836c81aad38588ff2a87bc1ed2b89ab143445d96ca4e0ce231db5bc2daa73e`。
- [SOURCE81 manifest](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical_repair/SOURCE.json) SHA256：`b375bb1f9f979a70de7577729a28f3d7755a66e6643e7b46366d74acc830d857`。
- [修正準備・再利用](research_c_pr_aware_df_M3_s1_r64_repair_preparation_20261028.md)／[101人工・回帰PASS](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical_repair/tests_final.log)。
- [run02原本](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02/)／[summary](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_execution_summary.json)／[postflight・raw hash](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_execution_postflight.json)。

ユーザーは修正・再実行と予期しない技術中断の修正・再実行を明示許可した。旧catalog照合のtuple/listを`origins`/`word`に限って正規化し、保存・再読込を経由する人工fixtureを追加した。係数・符号・basis・frame・ID・provenance・順序の内容比較は緩和していない。旧run01/契約/SOURCE/STOPは上書きせず、source-digestを更新してfresh run02を最初から実行した。旧runの成功prefix/cacheをresumeしていない。承認証明・原文保存・新grantは作っていない。

保存直鎖H4・隣接1.00Å・STO-3G、通常number CDF8 Proposed（Stage A γ=.1終了J/U）、Nα=Nβ2/sector36、system256/last-ancilla controlled512、元g補正κ・核scalar一回・元frame0..7/prefix LD2・tail identity抽出を維持した。Partial ordered symmetric S2、δ=.1/**s1/h1/10/r64/K0**、m0..12、n=2^m/64 signed events per kernel。CA=.00159 Ha/DF reserveCA/8、βtotalπ/6・βstatπ/12、candidatefailure.01/axis.01/26、own-DF ground oracle・状態準備除外。新fit/optimizer/rank探索/積分再生成/ground eigensolve/DF norm再計算0。

保存input SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、Proposed J/U `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227`、元P-H4 provenance `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。κをK1用に対称化せずUもpolar Qへ変更しない。F2-K1/F3-K1の保証は移送しない。

## A：τ・減衰・PF/RTE・全13round

SOURCE/input/saved DF予算/9因子/general-K/rawCAR/H/own-ground/physical alias/216atomの係数・符号・basis・probability・identityがPASS。保存groundとsector因子を再利用し、残差と同一性を確認した。独立CAR/reference/PF・finite-RTE・phase-roomは既存B1関数を再利用した。

λ_R=1.4424298855756241 Ha、τ=0.002253796696211913。数学的τは旧s4/r16と一致し、binary64のhexも`0x1.2768de34419a7p-9`で同一、差0.0。全13roundのAの旧新差も0で、総random eventsは双方64·2^m。Aが同じことからPF/RTE同一とは仮定せず、新kernelと全13roundのoperator誤差を実評価した。

| m | 新s1 PF | 旧s4 PF | 新corrected RTE | 旧corrected RTE | A（旧新共通） | 判定 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 6.21615922e-05 | 3.88339621e-06 | 9.34398447e-05 | 9.34141043e-05 | 0.999837466 | PASS |
| 1 | 0.000123986862 | 7.74579093e-06 | 0.000186723739 | 0.000186672434 | 0.999674959 | PASS |
| 2 | 0.000245310383 | 1.53252701e-05 | 0.000372172319 | 0.000372071123 | 0.999350024 | PASS |
| 3 | 0.000470161325 | 2.93729265e-05 | 0.000734239499 | 0.000734048036 | 0.998700471 | PASS |
| 4 | 0.000800736972 | 5.00284489e-05 | 0.00139406997 | 0.00139376183 | 0.997402631 | PASS |
| 5 | 0.000962476519 | 6.0138017e-05 | 0.00235346423 | 0.00235318189 | 0.994812008 | PASS |
| 6 | 0.00129519545 | 8.09348706e-05 | 0.0032268316 | 0.00322686818 | 0.989650931 | PASS |
| 7 | 0.00245126611 | 0.000153183491 | 0.00424926731 | 0.00424935069 | 0.979408964 | PASS |
| 8 | 0.00450460743 | 0.000281496137 | 0.00787508358 | 0.00787575292 | 0.959241919 | PASS |
| 9 | 0.00897375395 | 0.000560779697 | 0.0150898413 | 0.0150903526 | 0.92014506 | PASS |
| 10 | 0.0179343394 | 0.00112074928 | 0.0302360925 | 0.0302377068 | 0.846666931 | PASS |
| 11 | 0.0358354926 | 0.00223951656 | 0.0609508949 | 0.0609542702 | 0.716844892 | PASS |
| 12 | 0.0716149979 | 0.0044762458 | 0.125604928 | 0.125612055 | 0.5138666 | PASS |

PFは||U_PF(h)^n−exp(−iH_DF t)||₂、補正RTEは||U_corrected(h)^n−U_PF(h)^n||₂、t=.1·2^m。全13roundで既存十分条件εPF+εRTE≤sin(π/12)かつA>0と保存DF reserveを満たし、`POTENTIALLY_FEASIBLE`。これは合成誤差・law差・最終CAを含む認定ではない。DF誤差Haとoperator/signal誤差を直接足していない。

[全13行CSV](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_feasibility13.csv)。最大roundではPF誤差が増え、corrected-RTE誤差とAは旧設定に近い。旧値の移植による判定ではない。

## B：32独立trajectory・full512・native価格

Aの全13適格後だけBへ進んだ。M3専用master/cohort/source-keyで32個のPCG64 substream、64K0 events/kernel（計2048events）。旧B2 run02の32cohortと非pairedでpoolしない。固定seedはideal IIDの証明ではない。各occurrenceはfresh独立trajectoryという資源モデルを維持し、校正sample回路をcoherent反復するモデルへ変更していない。

既存PrecisionBuilder/StrongBuilder、head内relative Gaussian・tail隣接同一basis共有、cross-head/tail一般相対融合なし。P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false/MPFR160 declared phaseを維持。Whole h=.1 kernelを一体build/compileし、分割event価格の加算proxyではない。旧s4側は保存h=.025 whole-kernel平均をδあたり4倍する。双方のcompile/fusion範囲差から単純4:1を仮定しない。

全32件でraw/reference、compiled/reference、compiled/raw、unitarity、actual inverse raw/compiledのreference/return、X+iYがPASS。全512入力列・全出力を比較し、sectorへ個々のGaussianを射影して代用していない。位相fitなし、matrix1e-10/complexsignal2e-10/rtol0を維持。単一trajectoryのX/Yにはlaw減衰を掛けない。

| 最大観測残差 | 値 |
|---|---:|
| raw_reference | 1.5596042e-14 |
| compiled_reference | 1.51955601e-12 |
| compiled_raw | 1.5183322e-12 |
| compiled_unitarity | 1.15019105e-13 |
| compiled_adjoint_reference | 1.52412591e-12 |
| compiled_return | 1.06682154e-13 |
| raw_XY | 2.42501943e-14 |
| compiled_XY | 9.29152251e-14 |

これらは固定32trajectoryのbinary64残差で、全law synthesis上界・厳密enclosure・global費用supportではない。共通head/X/Y shell6 QPYはhash/構造で再利用し、新shell build/compile0。新kernel32・P0compile32・actual inverse64、QPY128・reference NPZ32を保存した。

| Metric | 新kernel mean ± 記述的SE | 分散 | 観測範囲 | 旧4 kernels/δ mean | 新/旧δ価格 |
|---|---:|---:|---:|---:|---:|
| rz_count | 9105.59375 ± 185.3774 | 1099672.44 | 7105–11293 | 19063.5 | 0.477645435 |
| rz_depth | 3055.59375 ± 49.52636 | 78491.5393 | 2588–3713 | 7743.125 | 0.394620228 |
| cx_count | 3912.25 ± 56.92289 | 103686.903 | 3310–4606 | 9891.5 | 0.395516352 |
| cx_depth | 2630.65625 ± 26.43821 | 22367.3296 | 2381–2974 | 7807.625 | 0.336934247 |
| total_depth | 6711.6875 ± 108.4328 | 376245.577 | 5686–8144 | 16896.125 | 0.397232354 |
| circuit_size | 17985.0312 ± 373.2627 | 4458401.19 | 13987–22457 | 36905 | 0.487333187 |

[全32標本・seed・metrics](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_kernel_metrics32.csv)／[6指標校正](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_calibration6.csv)。観測min/maxをsupportにしない。非paired旧新δ価格差の記述的SEはsqrt(SE_new²+(4SE_old)²)を別途保存した。これは母平均CI/優位保証ではない。

## C：登録2scenarioのshots・条件付き資源

`evidence`は全13round `UNRESOLVED_SIGNAL_ERROR`のままでshots/total null。`ideal_exact_native`はexact synthesis/ideal finite-RTE IIDと32finite-bit校正平均をideal費用proxyとする反実仮定。`assumed_raw_signal_budget`は各round synthesis/law誤差を各.001 A以内とする旧C1の未検証仮定。新たな有利な仮定は加えず、観測残差を全law上界にしない。

新AのPF/RTE/Aから既存Hoeffding十分shotsを算定した。旧C1 shotsは転用しない。Serial kernel反復はn=2^mで、各interrogationのhead/readoutは一回。Fenced段のscheduled depth加算であり、長回路全体のglobal再compileや未観測融合を仮定しない。

| Scenario | 新合計shots | 旧s4合計shots | 新総RZ | 旧s4総RZ | 新/旧PR RZ | 新/決定論RZ |
|---|---:|---:|---:|---:|---:|---:|
| ideal_exact_native | 20734 | 19574 | 2.95179019e+11 | 5.4029016e+11 | 0.546334249 | 2.89885907 |
| assumed_raw_signal_budget | 20828 | 19658 | 2.96645202e+11 | 5.42769407e+11 | 0.546540019 | 2.91325798 |

| m | ideal N/axis | .001A各項 N/axis |
|---:|---:|---:|
| 0 | 512 | 514 |
| 1 | 512 | 514 |
| 2 | 513 | 515 |
| 3 | 514 | 516 |
| 4 | 516 | 518 |
| 5 | 520 | 522 |
| 6 | 527 | 529 |
| 7 | 540 | 543 |
| 8 | 570 | 572 |
| 9 | 634 | 637 |
| 10 | 787 | 791 |
| 11 | 1219 | 1225 |
| 12 | 3003 | 3018 |

[3lane×13round×X/Yの78行](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/run02_axes78.csv)。Evidence laneは全null。Shotsは算定数で、実量子shots0。

| Metric | 新ideal総費用 | 旧s4 ideal総費用 | 決定論s1費用 | 新/旧PR | 新/決定論 |
|---|---:|---:|---:|---:|---:|
| rz_count | 2.95179019e+11 | 5.4029016e+11 | 1.01825929e+11 | 0.546334249 | 2.89885907 |
| rz_depth | 9.9054233e+10 | 2.1945262e+11 | 5.14209174e+10 | 0.451369562 | 1.92634122 |
| cx_count | 1.2682466e+11 | 2.80340936e+11 | 6.464549e+10 | 0.452394366 | 1.96184854 |
| cx_depth | 8.52788255e+10 | 2.21280585e+11 | 5.79592993e+10 | 0.385387745 | 1.47135708 |
| total_depth | 2.17575056e+11 | 4.78863344e+11 | 1.11725965e+11 | 0.454357301 | 1.94739921 |
| circuit_size | 5.83026648e+11 | 1.04594688e+12 | 1.93188518e+11 | 0.557415163 | 3.01791563 |

全13roundで同じ32標本平均を使うので、総費用の記述的校正SEはW=Σ_m 2N_m·2^m倍。sqrt13で縮めず、32をquantumshots倍率にしない。旧新cohortは非paired。

- ideal_exact_native：W=32417320、総RZ校正SE=6009437045.01。Energy数値十分判定は`CONDITIONAL_CA_PASS`。
- assumed_raw_signal_budget：W=32578340、総RZ校正SE=6039286506.75。Energy数値十分判定は`CONDITIONAL_CA_PASS`。

決定論対照は保存M2の同じProposed J/U、全frame S2/s1・全sweep relative fusion・数値synthesis伝播pilotで、PRの2仮定とは資格と合法融合範囲が異なる。状態準備除外の条件付き比較であり、PF方式だけの純粋ablation、actual CA、厳密enclosure、保証shots、ideal-law母平均資源優位ではない。

## RQ1〜8と次の最小検証

1. τ/attenuation：数学的対応とbinary64一致が全13roundでPASS。
2. PF/RTE：表の通りPF誤差が増加した一方、補正RTE誤差/Aは旧設定に近い。
3. Feasibility：13/13適格。最終CAや合成/law全域保証とは区別する。
4. Native：旧4kernel/δとの6指標比較を実測。単純4:1や未実装fusionを仮定していない。
5. Shots/total：登録2仮定の新shotsと全6指標資源を上表・原本に保存。
6. 決定論との差：上表の新/決定論比で判定する。旧PR/決定論の保存ideal RZ比は5.3060174653。
7. Head反復削減：今回のmacrostep変更は、登録仮定下のδ価格と総資源点推定の改善として支持された。RZのδ価格比は0.477645435、総費用比は0.546334249。PF誤差増加とshots増加を含めても価格減少が上回った。ただし配置/compile範囲/独立cohortも変わるのでheadだけの因果効果とはしない。同J/U決定論よりkernel RZは低い（9105.59375対11026）が、全shotsは20734対13610、m12各軸は3003対591で、登録serial会計では残る主要負担はshotsと高mの反復重みである。PF、補正RTE、Aのどれか一項だけの因果効果には分離していない。
8. 次に一つだけなら、同じProposed J/Uを固定したs1/r128/K0のPF/RTE feasibility・条件付きshotsを先に調べる案。これはmicrostep数を増やして減衰/補正RTEのshots負担を軽くできるかと、後に測るnative価格増との交換条件を調べる。未知の128-event価格低下や総費用改善は仮定しない。s2/r32ではsr64とAが変わらないため、今回見えた減衰側の残差を直接切り分ける候補を優先する。これは次案で、追加数値・sampling・回路・契約は未実施。

ユーザー／研究Cが、今回の価格・shots・総費用を見て次案の採否を判断する。Codexは具体的指示と必要な科学実行許可後にその限定範囲だけ対応する。科学的継続/終了を自動確定しない。

## Call・資源・保全・STOP

| stage | 上限 | COMPLETE |
|---|---:|---:|
| A | 61 | 61 |
| B | 325 | 325 |
| C | 84 | 84 |
| program summary | 1 | 1 |
| 合計 | 471 | 471 |

旧run01の19 attempts/18 COMPLETE/1schema FAILは消去せず、技術relaunch1・run02内retry/resume0として区別する。累積attempt490。新科学的不適格0、feasibility UNRESOLVED0、evidence laneの未資格13roundは保持。過去のcall残枠は再利用していない。

Runner wall 569.608351秒、process CPU差 8937.9265秒、peak RSS 2684272KiB。Stage wall A=1.93447608s/B=567.543s/C=0.126457854s。

開始前にaffinity/quota/MemAvailable/pressureを照合。A13独立sector task、B32独立full512 chainsを既存resource plannerで実行し、BLAS1でworkerと協調。Qiskit build/compile/actual-adjoint-buildだけ共通process mutexを維持。CPU/RAM/worker/wallの人工capなし。Task-local venv/cache/output、既存依存read-only、共有設定/venv/cache/他jobの変更0。Bの8/16/24/32完了時のprocess CPU/RSS/memory pressureを原本に保存。速度向上率は比較測定していない。

原本431files/117064350bytes、QPY128、reference NPZ32＋A sector NPZ1。SOURCE81・入力49・旧raw133（run01含む）のhashが不変、新rawもhash固定。旧契約/manifest/STOP、H2/H4/H6/M2/F2-K1/F3-K1を変更していない。

実行entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_M3.py --contract docs/research/research_c_pr_aware_df_M3_s1_r64_repair_contract_20261028.json --execute`。run02は消費済みで再起動しない。

結果・SOURCE・input・call journal・STOPをcommit/pushし、remote SHA/tree・保存bytesを照合後**mandatory STOP**。別s/r/K/γ、追加arm/H6/fit/sampling/保証/詳細レビューへ自動進行しない。
