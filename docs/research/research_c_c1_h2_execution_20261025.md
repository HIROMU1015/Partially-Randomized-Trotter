# 研究C・C1：H2限定・決定論S2のCA数値pilot実行

2026-10-25 JST。固定公開SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`。
Branch `codex/research-c-c1-h2-execution-20261025`。

**H2を1回実行し、193/196 logical unitsで正常完了した。CDF rank2／XDF rank3がDF予算を満たし、決定論S2のs=1／4・4候補すべてが共通CA=0.00159 Haの数値pilot条件をPASSした。条件付き総RZはCDFがXDFよりs=1で36.4122%、s=4で36.1527%小さい。厳密CAは未資格。公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/execution_summary.json)／[4候補](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/candidate_summary.csv)／[52round](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/round_accuracy.csv)／[rank選定5行](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/rank_selection.csv)／[gate表](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/gate_summary.csv)／[call表](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/calls_summary.csv)／[2比較](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/comparisons.csv)／[原本・input・J/U・QPY・STOP](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H2-run01/)／[固定契約](research_c_c1_h2_h6_contract_unapproved_20261025.json)／[技術準備](research_c_c1_h2_h6_preparation_20261025.md)。

## 実行範囲と固定条件

直前のH2限定科学実行の確認に対するユーザーの進行許可を受け、独立worktree・fresh H2-run01で固定SOURCEを無変更で実行した。H6は今回未実行である。契約の未承認という歴史的ファイル名・statusを変更せず、承認原文保存・承認証明書・新grantを追加していない。

直鎖H2、H原子(0,0,0)/(0,0,1) Å、STO-3G、neutral singlet、Nα=Nβ=1、sector4／full16／controlled32。PySCF2.7.0のRHF(conv_tol1e-12)とMO積分変換で新しい共通h/g/constantを生成し、fit前に保存した。gはchemist ERI、κは元gから補正し、scalarと元frame順を保持する。FCI計算はしていない。

通常number CDFはffsim0.0.83／tol1e-8／L-BFGS-B／Cholesky初期化、rank1..3のfirst-passing。各rankで既存fitを1回だけ呼び、restart・best-of選択・rankの単調性仮定はない。XDFは同じffsimの非最適化modified-Cholesky分解をrank上限3で1回行い、元順prefixを評価する。**このXDF baselineはH4の保存OpenFermion-eigh rank10とは異なる。** H4のrank・入力・PF誤差・shots・費用は転用していない。

決定論ordered symmetric S2、s=1を主対象／s=4を対照、δ=0.1、m=0..12（最大時刻409.6）、全frame、PRのL_D splitは非適用。own-DF sector ground oracle／状態準備費用除外／ideal IID Born、β_total=π/6、β_stat=π/12、各X/Y軸α=0.01/26、候補ごとfailure上限0.01。CA0.00159 Ha、DF reserve CA/8=0.00019875 Ha。

P0は既存Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false。同じ合法的basis共有・融合・角度結合を両armに適用し、serial native kernel反復をfenceで区切る。長い全回路のglobal再compileや未観測の追加削減を仮定しない。

## rank選定と表現同値性

| arm | requested／actual rank | 物理sector norm Ha | engineering guard Ha | CA/8判定 |
|---|---:|---:|---:|---|
| CDF | 1／1 | 0.3935811669709405 | 1.704485499e-12 | 予算超過・次rank |
| CDF | 2／2 | 1.317623158715395e-06 | 2.428993806e-12 | PASS・採用 |
| XDF | 1／1 | 0.4311451239832161 | 1.693684848e-12 | 予算超過・次rank |
| XDF | 2／2 | 0.06910690473058523 | 2.416443923e-12 | 予算超過・次rank |
| XDF | 3／3 | 4.865205950501262e-16 | 3.046595919e-12 | PASS・採用 |

CDF fit2回ともoptimizer success=true（rank1 nit1/nfev3、rank2 nit5/nfev7）。rank1の予算超過はcorrectness FAILではなく、登録済みladder内の選定条件である。rank2で合格したためrank3 fitは実行しなかった。XDFはprefix3で合格した。CDF／XDFの実DF誤差は異なるまま、共通CA予算で比較する。

guardは`sector_operation_scale_guard8_proposal_v1`、`8 eps64 (n^4+R_actual n^3)(d_sector+1) max(1,||Hphys||F+Σ||Hblock||F)`。H4の旧allowanceを書き換えたものではなく、厳密丸めenclosureでもない。

選定後のsparse raw CAR対独立一般Kのmax-entryはCDF6.6613381478e-16、XDF8.8817841970e-16で、固定1e-10を両方PASS。physical／own-DF groundのeigen residual・alias条件も3件PASSした。ground energyはphysical−1.1011503302326189、CDF−1.1011491163047618、XDF−1.1011503302326182 Ha。DF誤差の判定にground energy差を代用せず、別途sector operator normを用いた。

## 4候補・13roundの精度と費用

| 候補 | 13round | 数値energy評価値 Ha | X/Y合計shots | 条件付き総RZ | 条件付き総CX |
|---|---|---:|---:|---:|---:|
| `CDF_D_s1` | PASS | 0.0012796349488257434 | 13,352 | 3,990,125,812 | 2,694,155,960 |
| `CDF_D_s4` | PASS | 0.0012796349488257434 | 13,294 | 15,699,690,593 | 10,600,614,368 |
| `XDF_D_s1` | PASS | 0.0012783173262851165 | 13,364 | 6,274,985,632 | 4,194,706,740 |
| `XDF_D_s4` | PASS | 0.0012783173262851165 | 13,294 | 24,589,446,313 | 16,437,661,520 |

4候補ともraw statusは`CONDITIONAL_CA_PASS`、本報告では未資格事項を明示して`NUMERICAL_PILOT_CA_PASS`と扱う。52roundはすべて`CONDITIONAL_ROUND_PASS`、correctness FAIL0。数値energy評価値はDF norm+guardを一度だけHa単位で数え、登録位相幅π/6を最終時刻409.6で割って接続した値である。実際に得られたenergy推定値・厳密intervalではない。

新分子のPF・合成を含むsignal radiusから既存Hoeffding式でround／X/YごとのNを算定した。H4の保存Nは再利用していない。量子shots・RPE sampling試行は0で、計算した十分shots処方と費用の比較である。

| 比較 | CDF/XDF総RZ比 | CDFの削減率 |
|---|---:|---:|
| s=1 | 0.635878079 | 36.412192% |
| s=4 | 0.638472717 | 36.152728% |

決定論のcost supportは固定回路だけからなるため、登録serial実行の費用区間は点に退化する。両比較は`CDF_LOWER_CONDITIONAL_INTERROGATION_COST`。主指標は条件付き総RZ、副指標RZ depth・CX・CX depth・total depth・circuit sizeをCSV／原本に保存した。depthはstaged serial和であり、global compile後の最小depthではない。

## PF・合成・統計の接続と未資格事項

| 候補 | 最大PF sector operator誤差 | 最大raw complex signal誤差 | kernel距離の数値代用値η |
|---|---:|---:|---:|
| `CDF_D_s1` | 0.0132802154112 | 0.0132802169229 | 1.30484408581e-13 |
| `CDF_D_s4` | 0.00082982930865 | 0.000829834121355 | 1.03854159854e-13 |
| `XDF_D_s1` | 0.0160182315159 | 0.016018233339 | 1.57363840906e-13 |
| `XDF_D_s4` | 0.0010009552105 | 0.00100096246175 | 1.56475972791e-13 |

既存sector S2 referenceとexact DF evolutionを再利用し、mごとの反復を二乗法で評価した。決定論ではA=1、RTE補正誤差0、trajectory lawがない構造上のlaw誤差0であり、未知PR誤差を0にしたのではない。

full controlled kernel4件とhead/X/Y3件の計7 segmentがraw対CAR、raw対P0、raw/P0 unitarityをすべてPASS。nested unitarityは14件。phase fitなし・max-entry1e-10／rtol0で、scalar相対位相・κ・Gaussian向き・元frame順を含む。compiled対raw最大成分残差は5.3002559403e-14。全13roundの合成代用値は既存unitary telescopingとnorm2≤Frobeniusを用い、`n=s 2^m`と両readoutを含めてraw複素signal単位に接続した。

PF・合成を同じsignal単位で合わせ、位相へ変換し、統計位相予算との和を検査した後、最終energy評価でDF誤差を一度だけ加える。異種誤差を直接単純加算していない。

解析的不等式自体と、数値referenceの資格を区別する。SCF／DF／CAR／expm／Operator／norm評価のrounding enclosure、実hardware誤差は未資格。`rigorous_CA_qualification=UNRESOLVED_REFERENCE_ENCLOSURES`、`CA_achieved=null`、保証付きactual shots=null、保証付き総資源優位=falseを維持した。これはown-DF oracle・状態準備除外・理想測定・P0・登録4候補に条件付けた数値pilotであり、厳密CA保証や無条件の総資源優位ではない。候補単位failure配分を4候補同時の保証へ流用しない。

今回のH2結果をH6やPRの結果と扱わず、H4と異なるXDF分解policyの差も残す。新しい研究方針の採否や重要な科学詳細レビューは今回実施していない。

## 科学call・環境・保全

| logical unit | 実績 | 最大 |
|---|---:|---:|
| `scf` | 1 | 1 |
| `spatial_integrals` | 1 | 1 |
| `physical_sector` | 1 | 1 |
| `sector_context` | 1 | 1 |
| `explicit_XDF` | 1 | 1 |
| `CDF_fit` | 2 | 3 |
| `rank_assessment` | 5 | 6 |
| `representation_gate` | 2 | 2 |
| `ground_alias` | 3 | 3 |
| `sector_kernel_reference` | 4 | 4 |
| `PF_round_assessment` | 52 | 52 |
| `full_CAR_context` | 1 | 1 |
| `full_generators` | 2 | 2 |
| `full_kernel_reference` | 4 | 4 |
| `native_model` | 2 | 2 |
| `gaussian_build` | 7 | 8 |
| `circuit_build` | 7 | 7 |
| `compile` | 7 | 7 |
| `operator_decode` | 14 | 14 |
| `segment_assessment` | 7 | 7 |
| `metrics` | 7 | 7 |
| `round_accuracy` | 52 | 52 |
| `energy_assessment` | 4 | 4 |
| `cost_aggregation` | 4 | 4 |
| `cost_comparison` | 2 | 2 |

193 attempt／193 COMPLETE／FAIL0。CDF fit・optimizer2回、非最適化XDF分解1回、実build7／compile7／Operator decode14／QPY14、retry／resume0。上限196から少ない3単位は未実施のCDF rank3 fit・rank assessment・Gaussian1件である。旧H4 grant・STOPや消費済みfit枠を再利用していない。

実行前SOURCE33・契約・環境preflightと人工14 testsがPASS。評価環境は既存専用venvのNumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／PySCF2.7.0、fitは別既存専用venvのffsim0.0.83／JAX0.4.38／Qiskit2.4.2。venvを変更せず絶対パスのPython symlinkを保持した。実行後SOURCE33、契約snapshot、input/export SHA、193callのattempt/COMPLETE/STOP一致をbyte／JSONで確認しPASS。実行後の新operator・norm・PF・fit・compile再計算は0で、報告表は保存JSONからの整形だけである。

開始時affinity32 CPU・空きRAM約52GiB、独立PF候補4 workers／process内BLAS1。wall 6.658695秒、parent CPU 17.773716秒、parent peakRSS 375,976KiB／child peakRSS 374,076KiB。parent CPUはfit subprocessを含まず、journalに各fit時間を保存した。fit rankはfirst-passingのため逐次、P0 compileは競合防止のため逐次。人工CPU/RAM/wall cap・共有環境変更・他ユーザーjob操作はない。未測定のspeedupは主張しない。

契約SHA256 `967173d6cb8a2ac695a55e5fd7338776322cef929cb496a33802c33c65517537`、SOURCE manifest SHA256 `380bb20f69c8c0b92d00069a7e2ed33d42f28ff5d1bf500894a536e571cc2a0b`。新共通input SHA256 `e49a4e65ba26b92f30fe54870c2264b08444464fd56d46cc592867e9b6e903f1`、CDF selected export `bbc6fa98515a4d7834233b75c3d73f789245bbe71f94f590e92570d2132a1464`、XDF selected export `f41898317343f306c74e6572d07de56709c8822454003f04113fa227a48dc559`。原本全ファイルのbyte hashはpostflightに保存した。過去の固定SOURCE／結果／grant／ledger／marker／STOPは保全。これはlocal実行の公開証拠であり、CIや外部独立再現ではない。

## 次の担当とmandatory STOP

ユーザー／研究CがこのH2数値pilot結果とbaseline差を確認し、準備済みH6のsector400・DF/PF限定（最大129 calls、CDF fit最大21回）を別途許可するか判断する。Codexはその明示許可後にH6のSOURCE・資源を再確認して限定実行する。H6 native／CA・shots・総費用はその契約でも対象外であり、自動進行しない。重要な科学レビューも必要なら別許可で行う。

結果・証拠をcommit・push・remote SHA／SOURCE／成果物照合後、**mandatory STOP**。今回の許可でH6・PR・追加科学計算へ進まない。
