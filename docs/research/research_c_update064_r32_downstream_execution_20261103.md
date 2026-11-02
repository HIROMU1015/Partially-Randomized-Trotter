# 研究C・update064/r32：PR下流資源と同一J/U強決定論S2の実行結果

2026-11-03 JST。**条件付き点推定では、update064のPR r32総RZはupdate005/r32比5.27765%減、同じupdate064 J/Uの強D比3.73167%減。** 固定科学SOURCE `928fda0a6df260b66eba1bbea63b51f2734e7b54`・617-call契約を無変更でfresh run01へ一度実行し、**617/617 COMPLETE、correctness FAIL/retry/resume0**。これは有限native32と宣言誤差仮定による資源比較であり、母平均優位・actual CA・一般的なPR優位の保証ではない。公開照合後mandatory STOP。

[全結果](../../artifacts/research_c_update064_r32_downstream/2026-11-03/run01/result.json) ／ [実行集計](../../artifacts/research_c_update064_r32_downstream/2026-11-03/execution_summary.json) ／ [Calls](../../artifacts/research_c_update064_r32_downstream/2026-11-03/run01/science_calls.jsonl) ／ [STOP](../../artifacts/research_c_update064_r32_downstream/2026-11-03/run01/STOP.json) ／ [保全照合](../../artifacts/research_c_update064_r32_downstream/2026-11-03/execution_postflight.json) ／ [固定契約](research_c_update064_r32_downstream_contract_unapproved_20261103.json) ／ [技術準備](research_c_update064_r32_downstream_preparation_20261103.md)。契約名と`UNAPPROVED_SCIENTIFIC_PROPOSAL`は当初の提案metadataを保持し、今回のユーザー明示実行許可に基づき実行した。新grant・承認証明・承認原文保存は追加していない。

## 1. 固定条件・SOURCE・比較単位

直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、signed対称J・実直交U、Nα=Nβ=2/sector36、full256/controlled512。元g補正κ・核scalar一回・元spin/frame順。PRはupdate064/元順LD2・partial ordered symmetric S2、s1、δ=h=.1、r32/K0、m0..12。Dは**同じupdate064 J/U**のκ＋全8 frame ordered symmetric S2/s1、同じδ/m。Dにr/Kは非適用。

CA=.00159 Ha、DF reserve=.00019875 Ha、既存RPE政策（βtotal=π/6、βstat=π/12、failure=.01、各round軸=.01/26）。Own-DF ground-state oracle、状態準備費用除外。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase、global phase fitなし。既存Gaussian/head relative融合、tail exact basis_key共有、D全sweep relative融合、fenced serial会計を維持した。

| 固定対象 | 値 |
|---|---|
| 今回の科学SOURCE | `928fda0a6df260b66eba1bbea63b51f2734e7b54`、tree `4a01cea8364c1e919034b44fce1744e1621e9705` |
| SOURCE manifest | SHA256 `ab9eb265211c83a3836b2322b00b58249e1492b4bd28dd2748e5fe09b4e56434`、91件 |
| 契約 | SHA256 `548bbd4e97778731f26c04824ae1d7f70188904bc1e04610125041619ec5cc5e` |
| update064結果の固定commit | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6`、科学SOURCE `025d495166311a58ec9c24c43f07ed6e40f65f42` |
| update064 terminal SHA256 | `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` |
| update005/r32結果commit | `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5`、科学SOURCE `1cdeaa8bd55eb07a1d5b8cf2080259e929d405d1` |
| update005候補SHA256 | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| 共通h/g/constant snapshot SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 既存update005強D科学SOURCE | `5f27539ae0ab88181d9cfcd45badb54daf855c28` |

科学実行専用branchは`codex/research-c-update064-r32-downstream-execution-20261103`、worktreeは`/tmp/prt-c-update064-r32-downstream-execution-20261103`。ea54 peerは読み取り専用でHEAD/入力8件を固定し、科学branchのmerge・元SOURCE編集はしていない。J/U再最適化・新r探索・rank/LD/γ/ε/PF/compilerの変更0。

## 2. Stage A：候補固有の物理資格・ground・law

保存terminalと物理exportのJ/Uが完全一致し、rank8・J対称・U直交、General-Kと9因子CARがPASS。同じ候補に対する保存DF sector誤差 `5.67341345864e-09` Ha＋engineering guard `3.22444652147e-09` Haは予算`.00019875` Ha内であり、来歴付きで再利用した。**新DF誤差normは0**。これは数値pilotであり厳密enclosureではない。update064は追加64回U受理後もU一次NOT_METの有限endpointである。

新しいcandidate groundは `-2.16638744933` Ha、残差 `1.50749405611e-15`、physical groundは `-2.16638744863` Ha、残差 `1.76422548468e-15`。両alias marginは `1.61799387799` でPASS。Aで求めた新candidate pairを同一H/J/UのDへ再利用し、旧update005 groundの移送とD側の重複eigensolveを避けた。

新signed tail lawは216 atoms、identity `0.0287049550314` Ha、λR `0.542690658752` Ha、確率和1、CAR/law差max-entry `4.99600361081e-16`でPASS。identityはλRから分離し位相へ保持。K0/r32のstep正規化と全13 attenuation恒等式もPASS。旧PF/RTE/shots/価格は新候補の入力へ使っていない。

## 3. Stage B：新PR PF/RTE・全13round・RPE

全13roundがPOTENTIALLY_FEASIBLE。PF operator誤差とcorrected RTE operator誤差を分け、raw complex signalを新candidate groundで評価した。以下のρとβdetはideal反実laneの値。RPE十分shotsは各X/Y軸あたりのNであり、実測shotは0。

| m | PF error | corrected RTE error | A | ρ ideal | βdet ideal | N/axis ideal | N/axis assumed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.30249143437e-05 | 1.8533817977e-05 | 0.999953983445 | 0.999872428466 | 8.15587324111e-05 | 512 | 514 |
| 1 | 0.000125674147767 | 3.70322931639e-05 | 0.999907969008 | 0.999745277541 | 0.000162706441649 | 512 | 514 |
| 2 | 0.000248384708597 | 7.37822718957e-05 | 0.999815946485 | 0.999493838801 | 0.000322166986066 | 512 | 514 |
| 3 | 0.000474347363743 | 0.000145356879089 | 0.999631926846 | 0.9990124507 | 0.000619704282497 | 512 | 514 |
| 4 | 0.000803977892586 | 0.000274671065774 | 0.99926398917 | 0.998186134109 | 0.00107864916753 | 513 | 515 |
| 5 | 0.00104310721039 | 0.000459813340305 | 0.998528520052 | 0.997027811019 | 0.00150292111649 | 514 | 517 |
| 6 | 0.00148001394946 | 0.000799171949343 | 0.997059205357 | 0.994786722076 | 0.00227918787209 | 517 | 519 |
| 7 | 0.00289821138661 | 0.00142865938045 | 0.994127058987 | 0.989825599677 | 0.00432688426832 | 522 | 524 |
| 8 | 0.00571451388376 | 0.00266217707014 | 0.98828860941 | 0.980010021156 | 0.00837678892093 | 532 | 535 |
| 9 | 0.0113867791047 | 0.00530918201133 | 0.97671437549 | 0.960407190256 | 0.0166967368941 | 554 | 557 |
| 10 | 0.0227596012266 | 0.0106365602173 | 0.953970971289 | 0.922112002719 | 0.0334023723714 | 601 | 604 |
| 11 | 0.0454961971123 | 0.0213617208754 | 0.910060614063 | 0.849215856164 | 0.0669078273848 | 709 | 712 |
| 12 | 0.0909586527274 | 0.0431759381054 | 0.828210321268 | 0.717118668701 | 0.134540109652 | 994 | 999 |

| m | raw signal Re | raw signal Im | 同J/U D PF error | D ideal N/axis | D数値 N/axis |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.97658593178 | 0.2149230757 | 6.30322971815e-05 | 511 | 511 |
| 1 | 0.907528152182 | 0.419781703819 | 0.000125688230065 | 512 | 512 |
| 2 | 0.647390662872 | 0.761927424732 | 0.000248407488233 | 512 | 512 |
| 3 | -0.161418741603 | 0.986529380929 | 0.000474352564492 | 512 | 512 |
| 4 | -0.947184171866 | -0.318488723868 | 0.00080375873886 | 512 | 512 |
| 5 | 0.795722789518 | 0.603335080997 | 0.00104084650428 | 512 | 512 |
| 6 | 0.269161549393 | 0.960175116763 | 0.00147780147267 | 513 | 513 |
| 7 | -0.849488331529 | 0.51688448878 | 0.00289373577535 | 514 | 514 |
| 8 | 0.454460847485 | -0.878174763632 | 0.00570538042347 | 517 | 517 |
| 9 | -0.564656286846 | -0.798192236431 | 0.0113681485657 | 523 | 523 |
| 10 | -0.318274075508 | 0.901408521271 | 0.0227223510354 | 535 | 535 |
| 11 | -0.711239094107 | -0.573789973132 | 0.0454217291181 | 561 | 561 |
| 12 | 0.176626108911 | 0.816203725456 | 0.0908098074508 | 619 | 619 |

m12のPRはPF `0.0909586527274`、RTE `0.0431759381054`、A `0.828210321268`、ρ `0.717118668701`、βdet `0.134540109652`。βtotal（統計分を含む）は `0.396339497451` で登録π/6内にある。Aの改善だけから総誤差・費用を推定していない。

| lane | RPE十分shots（X/Y合計） | W | shell込み総RZ | 資格 |
|---|---:|---:|---:|---|
| evidence | null | null | null | 全law synthesis/実装law誤差が未知 |
| ideal_exact_native | 15008 | 13381920 | 101118440083 | exact native/ideal finite-RTE IID＋有限bit価格proxyの反実条件 |
| assumed_raw_signal_budget | 15076 | 13447004 | 101610237397.625 | synthesis≤.001A、law誤差≤.001Aを未検証で仮定 |

両反実laneのenergy十分条件は数値pilotでPASS（bound `0.0012783262211` Ha、statistical failure upper=.01）。Evidence laneはUNRESOLVED_SIGNAL_ERROR/UNQUALIFIED_ROUNDSのまま。未知量を0としていない。Failure=.01は各事前登録candidateの政策であり、旧新比較・探索後選択全体のfamilywise保証ではない。

## 4. Stage C：native32のcorrectnessと価格

Master2026110302、cohort `update064_r32_fixed32_v1`、PCG64、32独立非paired・非adaptive trajectory、各32 events。SOURCE/candidate/axis/sample/occurrenceへseedを束縛し、旧32件とpoolせず、事後追加校正を行っていない。Complete controlled kernelをP0で32件build/compileし、raw/compiled/実生成actual adjoint/complex X-Yを**全512入力列×全512出力**で検査した。新PR QPY128件、D4件。

| 観測correctness指標 | PR32件の最大 | 同J/U D |
|---|---:|---:|
| raw/reference max-entry | 1.09947737173e-14 | 7.52190483984e-15 |
| compiled/reference max-entry | 1.32376711321e-12 | 7.66439640638e-13 |
| compiled actual-adjoint/reference max-entry | 1.3221349831e-12 | 7.7189045235e-13 |
| compiled adjoint return max-entry | 1.06692522923e-13 | 3.70703518837e-13 |
| compiled complex X/Y error | 1.0049358353e-13 | 3.57238721924e-13 |

登録component/adjoint allowance1e-10、signal allowance2e-10、rtol0、phase fitなし。すべてPASS。単一trajectoryのX/Y検査を全law平均signal/CA試験とは同一視しない。

| complete P0 kernel指標 | update005/r32 mean（保存引用） | update064/r32 mean | update064の記述SE | update064 D固定kernel |
|---|---:|---:|---:|---:|
| rz_count | 7636.5 | 7556.34375 | 95.1649629863 | 11026 |
| rz_depth | 2612.1875 | 2608.0625 | 24.7255066772 | 5568 |
| cx_count | 3401.9375 | 3379.8125 | 29.1885361079 | 7000 |
| cx_depth | 2348.875 | 2350.4375 | 13.0999520976 | 6276 |
| total_depth | 5713.6875 | 5709.0625 | 54.1079777952 | 12098 |
| circuit_size | 15006.6875 | 14857.53125 | 192.455053322 | 20919 |

新PR平均RZは7556.34375、記述SE95.1649629863。Observed min/maxはsupportでなく、CI・理想law母平均の保証は未取得。価格差をU構造や目的関数だけの因果効果へ分解しない。

## 5. Stage Dと同J/Uの資源tradeoff

同じupdate064 J/U・Aの新groundから、全frame S2 PF13、full256 CAR/sector36、P0 kernel1、full512/actual adjoint/X-Y、数値合成伝播、RPEを新規に評価した。保存update005 D価格の移送0。D数値laneはNUMERICAL_PILOT_CA_PASS、shots13706、W `9526398`、固定kernel RZ11026、総RZ105038112319。Ideal反実laneも同じ整数shots/費用だが、数値合成laneとの資格は交換しない。

D compiled/reference Frobenius `6.19009655911e-12` はlevel-B数値pilotであり、CAR/expm、Operator再構成、norm丸めの厳密enclosureではない。保存費用JSONのDの退化intervalは固定kernel価格の表現であり、CA・丸め・資源優位の統計保証ではない。

| shell込み総資源 | update064 PR ideal | update064 D ideal/数値 | PR/D ideal比 | PR assumed/D数値比 |
|---|---:|---:|---:|---:|
| rz_count | 101118440083 | 105038112319 | 0.962683333226 | 0.967365417697 |
| rz_depth | 34900936258 | 53043032035 | 0.657974005614 | 0.661174112607 |
| cx_count | 45228380490 | 66684786000 | 0.678241368129 | 0.681540047332 |
| cx_depth | 31453366590 | 59787673848 | 0.526084468012 | 0.528643120396 |
| total_depth | 76398315202 | 115250452093 | 0.662889505547 | 0.666113519501 |
| circuit_size | 198822392137 | 199282808851 | 0.997689631551 | 1.00254197186 |

PRはDよりshots（15008対13706）とW（13381920対9526398）が多い一方、complete kernelが軽い。主RZはideal点推定で3.73167%低く、assumed対D数値でも3.26346%低い。副指標ではideal circuit sizeの差は約0.231%にとどまり、assumed laneではPRのcircuit sizeが**約0.254%高い**。無条件の全指標支配を主張しない。

## 6. update005対update064・会計・不確かさ

| 同r32/ideal | update005（保存） | update064 |
|---|---:|---:|
| shots | 15202 | 15008 |
| W | 13979232 | 13381920 |
| mean kernel RZ | 7636.5 | 7556.34375 |
| shell込み総RZ | 106752458375 | 101118440083 |

新/旧PR総RZ比は `0.947223526486`、差は−5634018292、**5.27765%減の条件付き点推定**。Assumed同士の比は `0.94731909135`。新/旧D ideal総RZ比は `0.995312636248`、約0.469%減。D kernelの6価格は今回実測でも旧と同じだが、PF/RPEを再評価して取得した結果であり価格移送ではない。

`C_total = W × mean(kernel) + S_shell`、`W = Σ_m 2 N_m 2^m`。r32のmicroeventはkernel価格内に含まれ、再びWへ掛けない。32標本は古典価格校正であり量子shotsの倍率ではない。新PR idealのRZ shell寄与は `52528`、各interrogationのhead/X/Y shellを一度ずつ計上した。

共有校正の総RZ記述SEはW×SEkernel＝`1273489921.49`。13round独立校正として縮めない。今回の32件は非paired finite-bit cohortであり、旧cohortとの差やDとの差を母平均優位・保証CI・support保証・統計的有意性へ昇格しない。追加の価格統計設計や大規模校正は実行していない。

追加64回U更新に伴いJ/U、law、PF、価格、shots/Wが変わっている。登録条件では新endpointの下流費用が低い点推定を得たが、目的関数だけの因果的優位、一般的な最適DF、update064でのr32最適性は未判定である。

## 7. Calls・資源・保全

| 部分 | 実行/上限 |
|---|---:|
| A identity/DF再利用/新ground/law | 23/23 |
| B 新PR13round/3 lanes/RPE | 90/90 |
| C 新native32/serial会計 | 357/357 |
| D 同J/U強D | 141/141 |
| 5比較＋最終summary | 6/6 |
| 合計 | **617/617 COMPLETE** |

Fresh run01は一度のみ、first_failure=null、correctness FAIL/retry/resume0、STOP理由`UPDATE064_PR_AND_SAME_JU_D_COMPLETE`。新eigensolve2、PR13＋D13round、RPE assessments65、PR32＋D1 kernel/compile、QPY132、NPZ39、量子shot実測0、新DF norm0。科学STOP後はJSON scalarの報告整理とbytes/Git照合のみで、配列decode・追加科学call0。

内部wall `969.720377127` 秒、parent CPU `9259.98582402` 秒、peak RSS `2478556` KiB。外側`time -v`はwall16:12.51、CPU952%、swap0、exit0。[実行log](../../artifacts/research_c_update064_r32_downstream/2026-11-03/execution.log)。13roundとnative32を既存resource_planで並列化し、nativeは32 workers/worker BLAS1、D decodeは4 workers。既存Qiskit排他とBLAS調整を維持。Compiler/Operator再構成が主な残存コストで、SOURCEを途中変更して高速化していない。速度比の主張はない。

開始時affinity32 CPU、MemAvailable約50 GiB、memory pressure0。GPUはnvidia-smiがdriver通信失敗のため固定CPU/SciPy/Qiskit経路を使用。CPU/RAM/worker/wallの恣意的制限・共有venvへのinstall・他jobの操作0。

新rawは**422 files / 120677678 bytes**、最大file4194570 bytes、QPY132。SOURCE91件・local input14件・peer8件と契約hashを実行前後で照合しすべて一致。元SOURCE/入力/raw/契約/grant/ledger/marker/STOP・ea54 peer HEAD/clean状態は不変。準備時171 unique tests＋29 subtests PASSを固定SOURCEの技術証拠として参照し、今回科学実行時の再テストとは記載しない。

共有索引・Evidence Map・研究概要・validation manifestは、元依頼どおり後続統合担当への引渡し対象として保持し、今回書き換えていない。公表した点推定を統合する際は、本書の条件と未資格事項を一緒に記載する。

## 8. 判断範囲・次段階・mandatory STOP

今回判断できたのは、固定H4/CDF8/LD2/s1/r32/K0/P0/13round/own-DF groundという登録条件で、追加64 U更新後のPRは旧update005より条件付き総RZの点推定が低く、同一新J/U強Dとの主RZの点推定順位もPRが低い、という範囲である。厳密DF enclosure、actual CA、全law synthesis/実装law、理想law価格母平均/support/CI、一般的PR優位、最適r、目的関数だけの因果効果は未資格のまま。

ユーザー／研究Cは、この結果と有限32件の不確かさを踏まえ、Codex-Aの価格統計準備・追加校正の実行価値と対象を判断する。追加校正・新r探索・最適化継続は今回の617-call許可に含まれず、自動開始しない。統合担当は共有索引・Evidence Mapへ本結果を条件付き証拠として接続する。**Codexは公開SHA/tree・本文・科学SOURCE/旧成果物の照合後、mandatory STOP。**
