# 研究C：PR対weighted L1・共通r32下流資源比較

2026-11-03 JST。**1231/1231 logical units COMPLETE、correctness FAIL 0、retry/resume 0。**

## 結果の要点

主比較は固定されたPR/ENTRYの有限endpointに対する共通r32/s1/K0評価。Ideal laneのshell込み総RZ点比 PR/ENTRY=0.494181551983（PRが点値で50.5818448017%低い）、事前登録の近似Welch95%差区間による判定は `NOMINAL_MODEL_PR_LOWER`。
PRのkernel平均RZは7,519.09でENTRYの6,681.25より高い。一方でWは12,539,808対28,556,976と小さく、shots/減衰側の差が単価増を上回った。λ順位と条件付き総RZの点順位は今回一致したが、J/U・PF誤差・law・basis価格も違うためλだけの因果効果には分離できない。
小さいλ・RPE条件・価格点推定・名目的区間の判断を分ける。全law synthesis/implemented-law signal誤差、actual CA、validated coverage、一般性、独立新規性、目的関数だけの因果効果は未資格。状態準備を除いた条件付きinterrogation比較である。

## 契約・SOURCE・入力

- 科学SOURCE：`4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、実装SOURCE：`703938aba8cc38080e29e75acde8e0edc8dce868`。
- SOURCE manifest SHA256：`103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739`。
- 契約SHA256：`79830424ae479a21b862178e7b50dfc2d71ef922fe46200153341245cef41b2c`。
- [固定契約](research_c_near_practical_r32_downstream_contract_20261103.json)、[準備・80人工/回帰PASS](research_c_near_practical_r32_downstream_preparation_20261103.md)、[SOURCE.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/SOURCE.json)、[result.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/result.json)、[STOP.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/STOP.json)。
- 入力結果commit：`2ceb0a8825785ebd4f21241d8ef0f20cb321bf69`。前段科学SOURCE `70a98adfb701a209db2c65ea60fb789ff62b1fe7` と新しい下流SOURCEを区別する。旧645/1629の残984を使用しない。
- PR endpoint：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。ENTRY endpoint：`345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f`。
- 元h/g/scalar packet：`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。139 SOURCE files・20保存inputsをbyte固定。新fit/optimizer/U更新/DF norm/量子測定0。

直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、Nα=Nβ=2、sector36/system256/controlled512。元順LD2・partial ordered S2/s1/r32/K0、δ=h=.1、m0..12。CA=.00159 Ha・DF reserve=.00019875 Ha。元g補正κ・scalar一回・signed atom/identity位相を維持。両endpointは16 U更新の有限政策終端、J MET/U NOT_METであり大域最適解ではない。

## 保存DF適格性・ground/alias・tail law

保存DF norm+元engineering guardを、候補hash・同じJ/U・元SOURCE・評価規約を確認して再利用した。新DF normは計算していない。各候補自身のgroundは新しく計算し、同じ候補のPR/Dだけで共有した。元physical groundは共通Hについて一回。GeneralK/保存独立CAR9因子×2、signed tailとidentity、normalizationはPASS。

| Arm | DF sector norm / Ha | guard / Ha | DF予算 | ground / Ha | residual | λ / Ha | identity / Ha |
|---|---:|---:|---|---:|---:|---:|---:|
| PR | 4.84367101908e-09 | 3.2165809758e-09 | PASS | -2.16638744928 | 8.55442918107e-16 | 0.500044243591 | 0.0412174106296 |
| ENTRY | 1.11037670289e-08 | 3.31972010087e-09 | PASS | -2.16638745035 | 2.68207418006e-15 | 0.941312847393 | -0.182350907484 |

Physical ground=-2.16638744863 Ha、residual=1.76422548468e-15。3 groundとも登録window[-10,10] Ha内、alias margin=1.61799387799 rad。全law非identity atom216、確率和1。新しい候補固有値であり005/064の値は転用していない。

## 全26 roundのPF/RTE・RPE

各rowは独立sector36 reference。誤差2種はoperator norm、Aは既存finite-RTE attenuation。Signalはown-DF groundのraw complex mean。表のshotsはideal laneの各X/Y軸あたり。βstat=π/12固定、roomは登録βtotal=π/6からβdetを引いた残りであり、新しい適応配分ではない。mのcoherent反復は2^m。DF誤差[Ha]はenergy評価へ一度だけ入れ、dimensionless signalへ加算しない。

### PR：13/13 POTENTIALLY_FEASIBLE

| m | PF error | corrected RTE error | A | raw Re | raw Im | radius lower | βdet | phase room | shots/axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.30279822399e-05 | 1.51089914516e-05 | 0.999960931398 | 0.976592484692 | 0.214924833721 | 0.999882797477 | 7.8136973771e-05 | 0.523520638625 | 512 |
| 1 | 0.000125681647104 | 3.0189447477e-05 | 0.999921864321 | 0.907540195511 | 0.419787954306 | 0.999766005406 | 0.000155871095212 | 0.523442904503 | 512 |
| 2 | 0.000248409480623 | 6.01511670825e-05 | 0.999843734748 | 0.647407274632 | 0.761948881013 | 0.999535222318 | 0.000308560652602 | 0.523290214946 | 512 |
| 3 | 0.000474442084918 | 0.000118522732356 | 0.999687493915 | -0.161429929401 | 0.986582476689 | 0.999094714403 | 0.000592964852023 | 0.523005810746 | 512 |
| 4 | 0.000803831987064 | 0.00022412462935 | 0.99937508549 | -0.947285323118 | -0.318527941032 | 0.998347771258 | 0.00102795679745 | 0.522570818801 | 513 |
| 5 | 0.00100429220679 | 0.000375737272687 | 0.998750561498 | 0.795889438186 | 0.603473814032 | 0.99737225628 | 0.00138002991751 | 0.522218745681 | 514 |
| 6 | 0.00140916537261 | 0.000653763306658 | 0.997502684092 | 0.269259361938 | 0.960597038208 | 0.995444907197 | 0.00206293014246 | 0.521535845456 | 516 |
| 7 | 0.00276974087235 | 0.00114472338009 | 0.995011604771 | -0.850246072743 | 0.517299541528 | 0.991116667413 | 0.00391447424942 | 0.519684301349 | 521 |
| 8 | 0.00545736014402 | 0.00209519306381 | 0.990048093629 | 0.455319568955 | -0.879663895688 | 0.982570702723 | 0.00755262501061 | 0.516046150588 | 530 |
| 9 | 0.010872795782 | 0.00417275180117 | 0.980195227698 | -0.566492686943 | -0.801056516742 | 0.965447653759 | 0.0150461152807 | 0.508552660318 | 549 |
| 10 | 0.0217325748128 | 0.00835199256154 | 0.960782684402 | -0.320777528959 | 0.90758531783 | 0.931877953001 | 0.0300891073864 | 0.493509668212 | 589 |
| 11 | 0.0434423722772 | 0.0167466216933 | 0.923103366647 | -0.720813055357 | -0.582265999261 | 0.867542703678 | 0.0602253946047 | 0.463373380994 | 679 |
| 12 | 0.0868531196299 | 0.0337706512886 | 0.852119825516 | 0.180537757749 | 0.839409869729 | 0.749333918887 | 0.12091821746 | 0.402680558138 | 910 |

全laneのround/energy/phase/shot根拠：[PR/B/references.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/B/references.json)。各roundのX/Y shotは同じで、合計は2軸を含む。`POTENTIALLY_FEASIBLE`自体はactual CA達成ではない。

### ENTRY：13/13 POTENTIALLY_FEASIBLE

| m | PF error | corrected RTE error | A | raw Re | raw Im | radius lower | βdet | phase room | shots/axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.24081098455e-05 | 3.59657810183e-05 | 0.999861561764 | 0.976497119603 | 0.214903273409 | 0.999753202876 | 0.000108373891076 | 0.523490401707 | 512 |
| 1 | 0.000144504874848 | 7.19022308052e-05 | 0.999723142694 | 0.907363206221 | 0.419704854589 | 0.999506795502 | 0.000216407107342 | 0.523382368491 | 512 |
| 2 | 0.000286565625106 | 0.000143565754414 | 0.999446362037 | 0.64715581786 | 0.76164948224 | 0.999016468795 | 0.000430131392783 | 0.523168644206 | 512 |
| 3 | 0.000555004732375 | 0.000285236219903 | 0.998893030589 | -0.161299293084 | 0.985811768464 | 0.998053719758 | 0.000840241051147 | 0.522758534547 | 513 |
| 4 | 0.00100269412976 | 0.000556233792674 | 0.99778728656 | -0.945807347259 | -0.318021543546 | 0.996231808098 | 0.00155892855387 | 0.522039847044 | 515 |
| 5 | 0.00167754922125 | 0.00101992114178 | 0.99557946922 | 0.793413842241 | 0.601574350677 | 0.992893923108 | 0.00269747363433 | 0.520901301964 | 519 |
| 6 | 0.00261512651698 | 0.00160295094501 | 0.991178479533 | 0.267613857287 | 0.954595001339 | 0.986997611928 | 0.00421808997023 | 0.519380685628 | 525 |
| 7 | 0.00462234769764 | 0.00262677386567 | 0.98243477829 | -0.839634436051 | 0.510925744679 | 0.975312989154 | 0.00724918505475 | 0.516349590544 | 538 |
| 8 | 0.00847369828265 | 0.00506800890083 | 0.965178093593 | 0.443940877874 | -0.857981787995 | 0.95210793447 | 0.0135421210925 | 0.510056654506 | 564 |
| 9 | 0.0168983124227 | 0.00990647613876 | 0.931568752353 | -0.5390492725 | -0.761786498834 | 0.906598248915 | 0.0268079994584 | 0.49679077614 | 622 |
| 10 | 0.0337636117519 | 0.0197894594421 | 0.86782034036 | -0.289744512094 | 0.821280921396 | 0.821345895889 | 0.0535787020421 | 0.470020073556 | 758 |
| 11 | 0.0674495398509 | 0.0398927327027 | 0.753112143143 | -0.590550609591 | -0.475923299202 | 0.67227137421 | 0.107549488235 | 0.416049287363 | 1131 |
| 12 | 0.134780493325 | 0.081372001071 | 0.567177900149 | 0.12224702801 | 0.562113591155 | 0.444580982265 | 0.217872075417 | 0.305726700182 | 2585 |

全laneのround/energy/phase/shot根拠：[ENTRY/B/references.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/ENTRY/B/references.json)。各roundのX/Y shotは同じで、合計は2軸を含む。`POTENTIALLY_FEASIBLE`自体はactual CA達成ではない。

## RPE shots・W・lane資格

αaxis=.01/26、13round×X/Yのfailure union budget=.01。Own-DF ground oracle、状態準備除外。解析的な十分shots算定であり、実quantum shotsは引いていない。

| Arm/lane | energy status | error scale / Ha | shots X+Y | W | shell RZ | conditional total RZ |
|---|---|---:|---:|---:|---:|---:|
| PR/evidence | UNQUALIFIED_ROUNDS | null | null | null | null | null |
| PR/ideal_exact_native | CONDITIONAL_CA_PASS | 0.00127832538349 | 14738 | 12539808 | 51583.0 | 94288043542.0 |
| PR/assumed_raw_signal_budget | CONDITIONAL_CA_PASS | 0.00127832538349 | 14798 | 12601244 | 51793.0 | 94749986795.625 |
| ENTRY/evidence | UNQUALIFIED_ROUNDS | null | null | null | null | null |
| ENTRY/ideal_exact_native | CONDITIONAL_CA_PASS | 0.00127833174673 | 19612 | 28556976 | 68642.0 | 190796364542.0 |
| ENTRY/assumed_raw_signal_budget | CONDITIONAL_CA_PASS | 0.00127833174673 | 19698 | 28703404 | 68943.0 | 191774686918.0 |

Evidence laneは全law synthesis/implemented-law誤差null、shots/費用を計算しない。Idealはexact native＋ideal finite-RTE IID＋finite-bit価格proxy。Assumedはsynthesis≤.001 A、law signal誤差≤.001 Aの未検証仮定。`CONDITIONAL_CA_PASS`の条件付き数値pilotを実CA・厳密CAへ昇格させない。

## Native correctness・価格校正

P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initial-zero false、MPFR160 declared phase、既存合法relative-Gaussian/basis共有とfenced serialを両armへ同様に適用。各32の新cohort、各trajectory32 event。全64固定trajectoryについてfull512 raw/reference・compiled/reference・actual adjoint・complex X/Y・位相/normalization検査PASS。各2強Dとcandidate-independent shellsも検査した。個々の固定回路の全operator数値検査であり、全sampling law保証ではない。

| Arm | samples | mean kernel RZ | SE | mean RZ depth | mean CX | mean CX depth | mean total depth | mean size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PR/PR | 32 | 7519.09375 | 107.321883191 | 2585.75 | 3369 | 2338.71875 | 5661.71875 | 14780.65625 |
| PR/D | fixed1 | 11026 | NA fixed kernel | 5568 | 7000 | 6276 | 12098 | 20919 |
| ENTRY/PR | 32 | 6681.25 | 115.509565307 | 2456.6875 | 3089.1875 | 2246.65625 | 5366.96875 | 13095.9375 |
| ENTRY/D | fixed1 | 11026 | NA fixed kernel | 5568 | 7000 | 6276 | 12098 | 20919 |

| Arm | raw/ref max entry | compiled/ref max entry | actual-adjoint/ref max entry | X/Y complex error |
|---|---:|---:|---:|---:|
| PR | 7.34091687652e-15 | 1.34733969537e-12 | 1.3493629369e-12 | 9.72706532242e-14 |
| ENTRY | 1.05747012911e-14 | 1.09686989622e-12 | 1.09636863979e-12 | 9.40214050507e-14 |

成分allowance1e-10、complex signal2e-10、rtol0、global phase fitなし。上表は固定cohortでの観測残差であって、全law上界でも丸め誤差enclosureでもない。

Master2026110304、arm固有cohort/axis＋新SOURCE/candidate hash、64 seed衝突なし。N32/arm固定、旧cohort混合・adaptive追加・外れ値除去0。Complete kernelがr32 eventを既に含むのでWへrを重複乗算しない。価格校正の32件もquantum shots倍率にしない。未観測の融合効果・supportを仮定しない。

## 主比較：shell込み総RZと不確かさ

- PR点値：94288043542。ENTRY点値：190796364542。点比：0.494181551983。
- PR−ENTRY点差：-96508321000。近似Welch95%差区間：[-103702867354, -89313774645.6]、df=41.0419998454。
- 固定5% halfwidth/ENTRY目標：True、判定 `NOMINAL_MODEL_PR_LOWER`。Skew/delete-one診断は[result.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/result.json)のcomparison.primary.diagnosticsへ保存。
- Difference intervalは近似固定N IIDモデル内であり、validated coverageでもideal-lawへの費用移送保証でもない。比は点値のみで、ランダムなENTRY分母に対する比区間は未評価。診断PASSは未観測tail/IIDを認証しない。

## 各候補の同じJ/U強決定論対照

κ+全8frameのordered S2/s1/h=.1、13round。各候補の新PF/新kernel/P0を評価し、groundだけ同候補PRと共有した。固定kernelなのでsampling価格分布を仮定しない。

| Arm | D kernel RZ | D shots | D W | D ideal total RZ | PR/D ideal point ratio | 個別近似95%比区間 | 個別判定 |
|---|---:|---:|---:|---:|---:|---|---|
| PR | 11026 | 13688 | 9467006 | 104383256064 | 0.903287050982 | [0.8769919462607291, 0.929582155704238] | NOMINAL_MODEL_PR_LOWER |
| ENTRY | 11026 | 13940 | 10188096 | 112333995286 | 1.69847394866 | [1.638585159346598, 1.758362737970442] | UNRESOLVED_PRECISION_TARGET |

PR強Dのnumerical lane：`NUMERICAL_PILOT_CA_PASS`、総RZ=104383256064、energy error尺度=0.00127832538349 Ha。Actual CAはnull。

ENTRY強Dのnumerical lane：`NUMERICAL_PILOT_CA_PASS`、総RZ=112333995286、energy error尺度=0.00127833174673 Ha。Actual CAはnull。

個別PR/D区間は既存一標本t95%・5% halfwidth政策。主PR−ENTRYとの同時優位や副指標の多重優位は認定しない。強Dのnumerical laneはfull512 Frobenius/norm-growthと13roundの合成伝播を含む数値pilot、厳密enclosureはnull。Ideal/idealとAssumed PR/numerical Dの資格を区別する。

### 副指標：ideal条件付きserial総費用（記述的）

| Arm/method | RZ depth | CX count | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|
| PR/PR | 32424860119 | 42246613152 | 29327084091 | 70996961872 | 185346687286 |
| PR/D | 52712337316 | 66269042000 | 59414929656 | 114531927560 | 198040387486 |
| ENTRY/PR | 70155634619 | 88217853297 | 64157708611.5 | 153264525264 | 373980500363 |
| ENTRY/D | 56727367318 | 71316672000 | 63940490496 | 123255676018 | 213124870834 |

全6指標の平均/SE、assumed/numerical結果と比は各armの `C/result.json`、`D/result.json`、`E/statistics.json` に保存。Depthも登録fenced serial合成の指標であり、未実行の跨境界compiler最適化を仮定しない。

## Calls・資源・保全・STOP

1231 logical units、全COMPLETE、correctness FAIL/retry/resume0。最大1,231の候補別・kind別実績は [STOP.json](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/STOP.json) と [science_calls.jsonl](../../artifacts/research_c_near_practical_r32/2026-11-03/run01/science_calls.jsonl)。新fit/optimizer/U更新/r探索/DF norm0。
Wall=930.38188114秒、process CPU=14449.1368835秒、peak RSS=2532900 KiB。CPU affinity32、実行前RAM約50GiB、disk約176GiB。既存resource_planは各arm13round/32trajectory/D4decodeを並列化。Qiskit一時global状態に関わるbuild/compile/adjoint作成は既存mutex、cohort同士は順次。BLASはprocess-local1。worker/wall/CPU/RAMの人工上限追加0、共有venv/他job/OS設定変更0。

旧SOURCE/input/raw/契約/grant/ledger/marker/STOPを保全した。新raw/QPY/reference/Calls/STOPのみを新familyへ追加。Source files/20inputs/契約コピーの一致と旧artifact treeの不変更を確認した。[技術照合・raw inventory](../../artifacts/research_c_near_practical_r32/2026-11-03/technical/verification.json)：新run843ファイル、234,153,810 bytes、QPY264件、最大単体4,194,570 bytes。公開SHA/tree/報告/主要結果との照合はpush後に行う。再利用shell6件は原本のまま。SOURCEは実行時の固定commitであり、後の閲覧・結果収録commitへ付け替えない。

## 研究判断へ渡す事項

小さいexact λという前段結果、今回のr32/RPE適格性、総RZの点順位、名目的価格モデルの順位、同じJ/U強Dとの差をそれぞれ分離する。この2有限endpointだけから、PR目的の大域最適性能・他分子への一般性・独立新規性は判定しない。全law synthesis/law signal誤差とactual CAの資格は残る。

次担当はユーザー／研究C：今回の固定N比較と同J/U強D対照を確認し、残る価格精度・保証・一般化のうち必要な次課題を選ぶ。Codexは今回の公開照合後mandatory STOP。追加標本・r探索・U更新・科学レビューへ自動進行しない。
