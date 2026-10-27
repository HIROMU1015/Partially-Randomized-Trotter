# 研究C：M1保存値の記述的機構分解・20-call実行結果

2026-10-28 JST。**20/20 logical calls COMPLETE、correctness FAIL0、retry/resume0。** 固定SOURCEと契約を変更せず、保存PR Baseline/Proposedのshotsとcomplete serial価格を交差させるM1を一度実行した。両登録シナリオの対称配分では、RZ点推定差のshots側項が価格側項より大きかった。これは会計上の記述的分解で、因果効果・母平均優位・強決定論baselineへの優位の認定ではない。

## 固定SOURCE・条件・実行

- 科学SOURCE：`47e0d406226be365951bd8666d5acae61106f500`。独立branch：`codex/research-c-mechanism-M1-execution-20261028`。
- Scope：`C_PR_AWARE_DF_SAVED_MECHANISM_M1_V1`。
- [固定契約](research_c_pr_aware_df_mechanism_contract_unapproved_20261028.json) SHA256：`bf357a9e145a2179976759e5b3716dd433a85b515e138c4234b03b91e03cb729`。
- [SOURCE7](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/technical/SOURCE.json) SHA256：`9b1adf3659ebd0cba1e940edefd4338046f9ef05e59d76d54dd2f7f86a85e1cf`。保存JSON9入力をhash固定。
- [原本20件](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01/)／[要約](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_execution_summary.json)／[postflight](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_execution_postflight.json)。

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2/sector36、通常number CDF rank8 Baseline/Proposed、元frame順prefix LD2。元g補正κ・核scalar一回。Partial ordered symmetric S2、δ=.1/s4/h1/40/r16/K0/m0..12、n=4·2^m、CA=.00159 Ha/DF reserveCA/8、βtotalπ/6・βstatπ/12・candidate failure.01/axis.01/26。Own-DF ground oracle／状態準備費用除外。保存B2 run02の各32標本のcomplete-kernel平均、共通Gaussian・P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/MPFR160 declared phase、fenced serial政策を維持した。

`ideal_exact_native`はexact native synthesis・ideal finite-RTE IID・finite-bit価格平均をideal価格proxyとする仮定、`assumed_raw_signal_budget`は各round synthesis/law誤差をそれぞれ.001A以内とする旧感度仮定を保持した。未知誤差を実際に0と認定したものではない。未資格evidence laneはコピーしてnullを保持し、今回の費用表へ混ぜない。

実行entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_mechanism.py --contract docs/research/research_c_pr_aware_df_mechanism_contract_unapproved_20261028.json --execute`。チャット許可に基づき、契約の歴史的`UNAPPROVED` metadataは変更せず、承認証明・承認原文保存・新grantは作成していない。

## Gate・call・再利用

| Call | COMPLETE |
|---|---:|
| saved_bundle_load／identity_gate | 各1、計2 |
| scenario_attachment | 2 |
| counterfactual_corner | 8 |
| descriptive_decomposition | 2 |
| tradeoff_table | 2 |
| calibration_uncertainty | 2 |
| reconciliation／result_summary | 各1、計2 |
| **合計** | **20** |

SOURCE/input/runtime/fresh gate、C1/B1/B2のarm/export/SOURCE/STOP、13round×X/Yの価格coverage・反復policyはPASS。元C1のself endpointと共通校正重みWへのreconciliationもPASS。旧C1 runnerは再起動していない。新fit/optimizer、NPZ/QPY decode、DF/PF/RTE/ground、sampling、回路build/compile、新shot式・CA再判定、追加保証はすべて0。53人工/回帰testsは固定SOURCEの準備済み結果を再利用し、重複実行していない。

## 4つのcounterfactualと総RZ

第1文字はshots、第2文字は価格のarm（B=Baseline/P=Proposed）。`C[a,b]=Σ N[a,m](n_m μ[b]+shell_axis)`。BB/PPは元の自己arm点推定に一致し、BP/PBは実在の新候補ではない。両方のtarget_idを保存し、crossed価格をphysical/CA適格性へ移していない。

| 旧仮定scenario | corner | 条件付き総RZ点推定 |
|---|---|---:|
| ideal_exact_native | BB | 3,194,333,274,946 |
| ideal_exact_native | BP | 2,928,602,922,036 |
| ideal_exact_native | PB | 589,314,042,709 |
| ideal_exact_native | PP | 540,290,160,109 |
| assumed_raw_signal_budget | BB | 3,209,642,597,218 |
| assumed_raw_signal_budget | BP | 2,942,638,691,654 |
| assumed_raw_signal_budget | PB | 592,018,246,752 |
| assumed_raw_signal_budget | PP | 542,769,406,705 |

主シナリオの正の差D=CBB−CPPは**2,654,043,114,837 RZ**。価格を先に交換する順序では価格項265,730,352,910、後続shots項2,388,312,761,927。Shotsを先に交換するとshots項2,605,019,232,237、後続価格項49,023,882,600となる。順序による違いを隠さず、交互項I=CPP−CBP−CPB+CBB=**216,706,470,310**も保存した。

対称配分は価格項`P=((CBB−CBP)+(CPB−CPP))/2`、shots項`S=((CBB−CPB)+(CBP−CPP))/2`でP+S=D。価格**157,377,117,755**、shots**2,496,665,997,082**。両既存仮定で保存判定`dominant_positive_net_RZ_component=shots`だった。感度シナリオも差2,666,873,190,513、価格項158,126,372,805.5、shots項2,508,746,817,707.5で同じ記述的方向だった。

新しい寄与率・比率・有意性検定は追加していない。表の数値は20-call原本からの転記・表示整形で、report用に科学関数を再実行していない。

## 主シナリオの6指標・全round・高m群

| 指標 | 保存点推定差D | 対称価格項P | 対称shots項S |
|---|---:|---:|---:|
| RZ count | 2,654,043,114,837 | 157,377,117,755 | 2,496,665,997,082 |
| RZ depth | 1,049,190,127,392 | 46,856,028,700 | 1,002,334,098,692 |
| CX count | 1,320,762,211,810 | 48,289,004,335 | 1,272,473,207,475 |
| CX depth | 1,019,402,488,805 | 24,428,822,730 | 994,973,666,075 |
| total depth | 2,289,606,379,558 | 102,355,402,500 | 2,187,250,977,058 |
| circuit size | 5,164,017,041,923 | 320,099,462,085 | 4,843,917,579,838 |

Depthはscheduled serial sumであり、長い実回路を一体再compileした最小depthではない。[全2scenario×6指標とSE/covariance CSV](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_decomposition12.csv)。

| m | RZ差D | 対称価格項P | 対称shots項S |
|---:|---:|---:|---:|
| 0 | 1,771,264 | 1,771,264 | 0 |
| 1 | 3,542,528 | 3,542,528 | 0 |
| 2 | 7,251,409 | 7,091,975 | 159,434 |
| 3 | 14,890,862 | 14,253,140 | 637,722 |
| 4 | 31,223,196 | 28,672,336 | 2,550,860 |
| 5 | 68,101,576 | 57,898,192 | 10,203,384 |
| 6 | 159,045,296 | 118,231,872 | 40,813,424 |
| 7 | 408,573,536 | 245,320,064 | 163,253,472 |
| 8 | 1,245,612,650 | 531,379,200 | 714,233,450 |
| 9 | 4,595,433,852 | 1,248,741,120 | 3,346,692,732 |
| 10 | 21,940,552,796 | 3,492,932,608 | 18,447,620,188 |
| 11 | 157,584,411,652 | 14,248,047,616 | 143,336,364,036 |
| 12 | 2,467,982,704,220 | 137,379,235,840 | 2,330,603,468,380 |

| 事前固定round群 | RZ差D | 対称価格項P | 対称shots項S |
|---|---:|---:|---:|
| m0_to_9 | 6,535,446,169 | 2,256,901,691 | 4,278,544,478 |
| m10_to_12 | 2,647,507,668,668 | 155,120,216,064 | 2,492,387,452,604 |
| m12 | 2,467,982,704,220 | 137,379,235,840 | 2,330,603,468,380 |

低mの同じNではshots項0となる行を含み、長roundのNとcoherent反復重みに差が集中している。m10..12・m12単独は事前固定した区分で、結果後に選び直していない。[全2scenario×(13round+3群)×6指標192行](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_round_contributions192.csv)に両順序・交互項も保存した。累積値は原本analysis JSONに含む。[全8corner×6指標](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_corners8.csv)。

## 校正不確かさ・tradeoff・資格の限界

主シナリオの保存校正重みはW_B=614494240、W_P=113366400。同じ32標本を全roundへ再利用するため、価格平均のSEをW倍し、sqrt13の独立化や32倍のquantum shots計上はしていない。別law/seedの両cohortをpairedとは扱わない。

RZ差Dの条件付き校正SEは`4.19964e+10`、対称価格項SEは`3.80172e+10`、対称shots項SEは`2.61746e+10`、価格/shots項の共分散は`-1.83363e+20`（表示丸め、原値JSON）。同じ価格平均を使う二項の相関を保存し、SEを独立に足さない。これらは保存N/DF/PF/RTEを固定した校正の記述的SEであり、母平均CI・fit/shot-planの不確かさ・ideal-law転送を保証しない。

[Baseline tradeoff](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01/baseline_tradeoff.json)と[Proposed tradeoff](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01/proposed_tradeoff.json)は保存DF norm/guard、optimizer停止/gradient、PF/RTE/A、両scenarioのNをそのまま対応付けた。DF[Ha]をsignal誤差へ加算せず、Aをraw信号絶対値と同一視しない。新しい物理誤差・N・energyは評価していない。

J/U、sampling law、head/tail・Gaussian価格は同時に変わっている。**M1はshots側の記述的寄与が大きいことを示すが、その差をλ低下だけの因果効果や全条件の一般則とは呼ばない。** 元の約83%は引き続き明示仮定付き点推定。Actual CA、保証shots、全law synthesis、価格support、母平均優位、強決定論baselineへの順位は未資格/nullである。F2-K1/F3-K1の固定trajectory保証を拡張していない。

## 資源・保全・公開・次担当

[Preflight](../../artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1_execution_preflight.json)でSOURCE7/input9・契約hash・Python3.11.1/binary64・fresh outputを確認した。開始時CPU affinity32、MemAvailable55,241,372kB、load1.25/1.24/1.26。Stdlib scalar表と小さいI/Oだけで、process生成のoverheadを追加しない既存順序を維持した。CPU/RAM/worker/wallの人工cap、共有環境・他job操作は0。

Runner wall **0.011114931665360928秒**、parent peak RSS **24320 KiB**、process exit0。速度向上率は測定していない。20原本・40 journal行（20 attempt/20 COMPLETE）・STOPを保存した。実行前後のSOURCE7/input9、旧F2/F3原本24件はhash不変。旧SOURCE/raw/grant/ledger/marker/STOPを変更していない。

SOURCE・結果・call・STOP・CSV・本報告をcommit/pushし、remote SHA/tree・SOURCE/input/成果物を照合する。最終照合はチャットで報告する。

次は**ユーザー／研究C**が、[既存の次期技術設計](research_c_pr_aware_df_performance_mechanism_readiness_20261028.md)を基に、同じ保存J/Uの全frame決定論S2 s1対照の準備や、γ/初期値の最小ablationのどちらを優先するか選ぶ。RQ2はM1では回答していないため、強baselineへ同じ条件で接続する技術準備が次の候補となる。**Codex**は具体的な次依頼と、その範囲で必要な別科学実行許可後だけ対応する。今回のM1で新fit・新回路・移送契約を追加準備／実行したものではない。

本作業は限定実行・証拠整理であり、重要な詳細科学レビューではない。20-callを消費済み、retry/resume0。公開照合後に**mandatory STOP**し、新しい解析、追加保証、別分子、CA/shots/総費用の別条件評価へ自動進行しない。
