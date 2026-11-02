# C1 Stage3：決定論S2・固定shotsの精度pilot限定実行

2026-10-25 JST。固定公開SOURCE `002d0dc2e1dafd8cb71735c43253fa054e675f6d`。
Branch `codex/research-c-c1-stage3-deterministic-execution-20261025`。

**110/110 logical unitsで正常完了。CDF／XDFのs=1／4、4候補すべてが全52roundの数値十分条件をPASSした。保存shotsを増やさず、合成・PF・DFを接続した共通CA=0.00159 Haの数値pilotが成立した。rigorous CA保証は未資格のまま。証拠の公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/execution_summary.json)／[4候補表](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/candidate_summary.csv)／[52round](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/round_accuracy.csv)／[7 gate表](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/gate_summary.csv)／[2比較](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/comparisons.csv)／[call表](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/calls_summary.csv)／[原本・SOURCE・runtime・STOP](../../artifacts/research_c_c1_stage3_deterministic_accuracy_science/2026-10-25/run01/)／[固定契約](research_c_c1_stage3_deterministic_accuracy_contract_unapproved_20261025.json)／[技術準備](research_c_c1_stage3_deterministic_accuracy_preparation_20261025.md)。

## 許可範囲と条件

ユーザーが固定の決定論S2・110-call契約の限定科学実行をチャットで明示的に許可したため、同じ固定SOURCE・契約を独立worktree・fresh run01で1回実行した。契約のunapprovedというファイル名・過去metadataは書き換えず、チャットの許可で実行した。承認証明書・承認原文保存・新grantは作成していない。

保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2／sector36）、CDF保存rank8 number・元g補正κ／scalar／元frame順、XDF保存rank10。決定論ordered symmetric S2で各arm s=1を主対象、s=4を対照、δ=0.1、m=0..12。決定論は全frameを使い、PRのL_D splitは適用しない。各DF ground oracle・状態準備費用除外・理想IID Born測定、CA0.00159 Ha・DF reserve CA/8、β_total=π/6・β_stat=π/12、元の候補単位failure配分（α_total=0.01）を保持した。

P0 Qiskit1.3.0・opt1・seed17・rz/sx/x/cx・approximation1・qubits_initially_zero=false、既存full-frame basis共有・角度結合・serial fenceを保持する。既存回路をdecodeしただけで新規build／compileは0。CDFとXDFの物理DF近似誤差は既知の約7.6660e-5 Ha／3.3752e-15 Haと異なり、同一誤差へ調整していない。

## 固定Nでの結果と費用

| 候補 | 13round | 数値energy評価値 Ha | 保存shots合計 | 条件付き総RZ | 最小統計margin |
|---|---|---:|---:|---:|---:|
| `CDF_D_s1_r0_K0` | PASS | 0.0013549794824010532 | 13606 | 111,798,946,393 | 2.166425905e-06 |
| `CDF_D_s4_r0_K0` | PASS | 0.0013549794824010532 | 13314 | 408,641,594,551 | 2.560869561e-06 |
| `XDF_D_s1_r0_K0` | PASS | 0.0012783188045012185 | 13600 | 153,454,725,962 | 2.183717872e-06 |
| `XDF_D_s4_r0_K0` | PASS | 0.0012783188045012185 | 13314 | 560,693,479,327 | 2.578847139e-06 |

4候補の状態はすべて`NUMERICAL_PILOT_CA_PASS`、52roundはすべて`CONDITIONAL_ROUND_PASS`、INFEASIBLE／UNRESOLVED候補0、correctness FAIL0。上記energy値は保存DF norm＋旧engineering allowanceと、登録RPE位相幅をenergyへ換算した数値条件付き評価値であり、実測energy推定や厳密intervalではない。両armともCA0.00159 Ha以内だった。

保存楽観Nに対して合成誤差を追加した今回の数値十分条件でもmarginは全roundで正だった。新しいshots数を解いたりNを増やしたりせず、保存費用を接続できた。実量子shots・sampling試行は0で、成功確率の経験的検証はしていない。

既存の費用集計と保存総費用の照合は全6指標で一致。登録s=1／4のCDF/XDF比較はいずれも`CDF_LOWER_CONDITIONAL_INTERROGATION_COST`、qualificationは`numerical_pilot_only`。前段の約27%総RZ削減という費用差を、今回の固定N・数値CA条件に接続できた。費用値は前段の保存値を再利用しており、native再計測・追加最適化・新しい候補順位付けはしていない。

決定論の費用supportは全有限supportを既にカバーしており、登録staged実行での費用区間は退化している。depthはserial stageの和で、長い全回路のglobal compile depthではない。状態準備費用・実hardware誤差を含む無条件の総資源優位には拡張しない。

## 合成誤差・PF・統計・DFの接続

| 候補 | kernel距離の数値代用値η | 全round最大の複素合成誤差代用値 |
|---|---:|---:|
| `CDF_D_s1_r0_K0` | 7.6445929217e-10 | 8.8564424767e-06 |
| `CDF_D_s4_r0_K0` | 7.6617771383e-10 | 3.5505403153e-05 |
| `XDF_D_s1_r0_K0` | 1.2812252154e-09 | 1.4843298430e-05 |
| `XDF_D_s4_r0_K0` | 1.2256741249e-09 | 5.6798903366e-05 |

s=1は保存Stage2のcompiled対独立CAR Frobenius reportを再利用した。s=4は未保存だったraw対CAR local referenceを2件だけ評価し、保存raw対P0 residualと合わせた。元のκ／scalar／frame順を保つfull256 CARからfull512 controlled referenceへ接続し、max-entry1e-10・unitarity1e-10・rtol0・global-phase fitなしを維持した。共通head／X／Yの3 segmentも位相・unitarityを含めPASSした。個別値はgate CSVとraw JSONに保存した。

`n=s*2^m`、`d_axis=min(2,n*η_kernel+η_head+η_axis)`、`e_syn_pilot=2*hypot(d_X,d_Y)`により、全13roundのcoherent反復と両readoutを含めた。kernel max-entry toleranceをoperator normへそのまま転用していない。unitary telescopingとnorm2≤Frobeniusは解析的不等式だが、代入値の丸め誤差は厳密に囲んでいない。

決定論でA=1・RTE誤差0を保存条件として確認し、PFと合成だけを同じraw複素signal単位で合わせ、固定Nを逆Hoeffding式へ代入した。`asin(e_PF+e_syn)+β_stat≤β_total`、固定Nの統計margin、全round・horizon・failure配分を確認し、最後にDFを一度だけHa単位で加えた。異なる単位の誤差を根拠なく単純加算していない。

候補単位failure上限は各0.01で、4候補同時や選択後の保証へ流用しない。deterministic law error0はランダムtrajectory lawが存在しない構造上の値で、PRや実hardwareの未知誤差を0にしたものではない。

## 資格の限界

数値pilot PASSは、保存DF／PF norm、CAR／expm、Operator再構成、Frobenius計算の誤差を厳密に囲んだ証明ではない。全候補の`rigorous_qualification=UNRESOLVED_NUMERICAL_REFERENCE_ENCLOSURES`、`CA_achieved=None`、`guaranteed_actual_shots=None`、`guaranteed_total_resource_superiority=False`を保持した。正の統計marginや小さい残差から、未知のreference誤差を0と認定していない。

この結果が支持するのは、保存H4・既定4候補・own-DF oracle／状態準備除外／理想測定／P0／固定Nという数値pilot条件下でのCA適格性と費用比較である。厳密同CA保証、実deviceの到達精度、大きい分子への一般化、PRの資格、研究方向の最終採否は評価していない。

## call・資源・来歴

| logical call | 実績 | 最大 |
|---|---:|---:|
| `saved_bundle_load` | 1 | 1 |
| `saved_operator_load` | 9 | 9 |
| `input_load` | 1 | 1 |
| `car_context` | 1 | 1 |
| `xdf_full_generator` | 11 | 11 |
| `shell_assessment` | 3 | 3 |
| `qpy_load` | 8 | 8 |
| `operator_decode` | 8 | 8 |
| `kernel_assessment` | 4 | 4 |
| `round_accuracy_assessment` | 52 | 52 |
| `local_kernel_reference` | 2 | 2 |
| `energy_assessment` | 4 | 4 |
| `cost_attachment` | 4 | 4 |
| `cost_comparison` | 2 | 2 |

新fit／optimizer／retry／resume／新build／compile／PF round誤差再計算／DF norm再計算／shot-count solving／量子shotsはすべて0。局所reference2、XDF既存GのCAR generator11、保存QPY decode8が今回の必要な新数値処理で、scopeは110 callsから拡張していない。

実行前の公開SOURCE・input・SOURCE27ファイル／保存49ファイル・依存5版のpreflightと人工12 testsがPASS。実行後も同じSOURCE27／保存49／元inputのbyte照合と110 attempt/COMPLETE対応がPASS。旧SOURCE・結果・grant・ledger・marker・STOPは保全した。SOURCEの変更はなく、実行後は保存JSONの整合確認・CSV整形・既存比較結果の記述だけを行った。

開始時affinity32 CPU・空きRAM約52 GiB・CPU idle99.47%、準備10 workers／候補4 workers、process内BLAS1でoversubscriptionを抑えた。wall11.278秒、CPU42.644秒、peak RSS480,520 KiB。人工CPU／RAM／wall cap、共有venv・環境設定・他ユーザーjobへの変更はない。未測定の高速化率は主張しない。

契約SHA256 `59c920cbd08a6b7ad7abda0216ef7d0399d78ea5edbc410a58290aa327b3a957`、SOURCE manifest SHA256 `0aee1202e8bf7150cb0dd2a15805a5fdbaa2b0e2689f2581c310ad4afa284512`。元input SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。元fit SOURCE da2edb46…／保存CAR評価SOURCE bec41ac7…／Stage2 a977a439…／前段cost SOURCE9ce6703c…を保持し、今回の評価SOURCE002d0dc2…と混同しない。固定SOURCEでのlocal committed evidenceであり、外部独立再現やCIでの科学再現ではない。

## 次の担当とmandatory STOP

ユーザー／研究Cが、この数値pilot条件付きの約27%費用差をどの範囲の研究claimに使うか、また厳密reference enclosureの資格付けを優先するか判断する。重要な科学レビュー、新しい数値計算、候補追加はそれぞれ必要な別許可後に行う。Codexは具体的な依頼と該当する許可を受けてから着手する。

結果・証拠をcommit・push・remote照合後、**mandatory STOP**。追加の科学計算・PR資格付け・次stageへ自動進行しない。
