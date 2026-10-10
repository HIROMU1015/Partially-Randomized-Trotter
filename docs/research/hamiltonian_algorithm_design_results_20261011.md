# N1/N2構成・N3係数下界：限定feasibility結果とGPT判断事項

2026-10-11 JST。提供設計に沿う限定検証を完了した。N1の縮退とframe生成、N2のsigned charge実装、
N3の係数下界取得は小系で検査できた。今回のN1/N2には、元Hamiltonianの誤差と測定数を含めた費用改善は見られなかった。
中心テーマ・新規性・一般的優位を採択しない。結果公開後はGPTの研究判断へ戻す。

## 条件・証拠・来歴

[提供設計](hamiltonian_construction_inputs/hamiltonian_algorithm_design_2026-10-11.md)、
[結果前scopeと技術修正](hamiltonian_algorithm_design_scope.md)、[準備記録](hamiltonian_algorithm_design_preparation.md)を参照。
基点f98050e、独立branch `hamiltonian-construction-feasibility-20261011`。
科学計算sourceは`ed335a007bee6966a37f3026ac188534e526b5b9`、保存復旧sourceは`a08d04c`。
旧A-core/B′の同型探索は一区切り、旧B/C保留。他Trackのsource・契約・結果・STOPは変更しない。

正本は[結果JSON](../../artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/result.json)、
[native IR](../../artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/native_ir.json)、
[実行audit](../../artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/run_audit.json)、
[全比較CSV](../../artifacts/hamiltonian_algorithm_design/2026-10-11/precision_resource_comparison.csv)、
[独立照合](../../artifacts/hamiltonian_algorithm_design/2026-10-11/independent_verification.json)。
85 focused local tests passed、276 warnings。NumPy/SciPy保存verifierが177 source/input blobs、280 native IR、
120 semantic rowsを照合し、最大native作用誤差1.305e−12。位相・workspace次元・shotsの改変を拒否した。
local開発証拠であり、immutable CIや外部による独立科学再現ではない。
追加の保存監査で44 spectral candidatesの解析bound・絶対exterior frame、18の有限選択、
280 IRのcheckpointからの順序・全object保持を照合した。最大bound差4.20e−15。
[構成・選択・保存監査](../../artifacts/hamiltonian_algorithm_design/2026-10-11/construction_selection_and_export_audit.json)。
提供付録も独立に再実行し、JSON最大数値差1.74e−17で一致した。
これは式のreplayで、新しい科学contextや分子・compiler benchmarkではない。

全例syntheticでgeometry・化学basis・分子fittingなし。N1/N2はJW、決定論実装、ordinary control、
準備X0X1とX/Y readoutを含む固定時間complex信号。εcomplex=.04、二軸合計失敗率.05、normalization=1。
各軸shots=ceil[2 log80/(ε/√2−b)²]。RZはlogical Tではなく、RZ/CX等を任意重みで合算しない。
N1のPF誤差は選択後のexact小sector診断に依存する会計で、大系へ適用できる係数のみの精度保証ではない。
RPE/QPE統計・最終total-cost評価・ground-state準備の比較は行っていない。

## N1：回路節約と誤差予算の負担が分かれた

JW 4 modes、ν=2、sector6、signed synthetic DF rank2。
H=H1+.7 dΓ(g0)²−.4 dΓ(g1)²+.11I、H1=diag(.08,−.04,.03,−.02)。
DF-prefix split L_Dは該当せず全fragmentをS2で扱う。
T=.6、q=1/2/4/8、delta=.6/.3/.15/.075、εH予算.02。
近縮退spectraは(1,1.002,−.4,−.397)と(.7,.703,−.2,−.196)。
離れた対照は(1,1.25,−.4,−.1)と(.7,.95,−.2,.1)。frame角度は固定fixtureを正本とする。

constructorはfactor固有分解から群を作り、群projectorによる座標Gram-Schmidtとcolumn assignmentでframeを返した。
全contiguous partitions・平均中心を列挙し、basis-native RZ/CXを小さくする有限候補を選ぶ。
sector平方boundの加算で元Hのmodel errorを計上した。近縮退のcluster化はHamiltonian近似であり、
近似後の同一固有値群内のframe変更だけがexact gaugeである。
選択はfactor・basis費用のみ、全sector truthは後段の診断に使用した。

以下は近縮退2例のq2（delta=.3）のX/Y合計。どちらも今回のq grid内で各候補のRZ/CX仕事量が最小の点である。
「work」はshots_per_axis×X/Y合計gate数。2例ともRZ/CXのbasis selectorは同じ候補を返した。

| context | 候補 | εH bound | shots/axis | XY RZ / CX | RZ work | CX work |
|---|---|---:|---:|---:|---:|---:|
| planted局所gauge | native exact | 0 | 16,201 | 1,250 / 832 | 20,251,250 | 13,479,232 |
| 同上 | N1 cluster/frame | .007007 | 24,131 | 1,126 / 800 | 27,171,506 | 19,304,800 |
| 同上 | 同じ近似・ungauged | .007007 | 24,131 | 1,250 / 832 | 30,163,750 | 20,076,992 |
| 同上 | finite shifted cutoff | .0028028 | 18,824 | 1,198 / 816 | 22,551,152 | 15,360,384 |
| 同上 | original whole-Pauli | 0 | 20,624 | 1,750 / 2,264 | 36,092,000 | 46,692,736 |
| dense群間frame | native exact | 0 | 16,296 | 1,310 / 848 | 21,347,760 | 13,819,008 |
| 同上 | N1 cluster/frame | .0042042 | 20,513 | 1,246 / 832 | 25,559,198 | 17,066,816 |
| 同上 | 同じ近似・ungauged | .0042042 | 20,513 | 1,310 / 848 | 26,872,030 | 17,395,024 |
| 同上 | finite shifted cutoff | .0042042 | 20,513 | 1,246 / 832 | 25,559,198 | 17,066,816 |
| 同上 | original whole-Pauli | 0 | 23,719 | 1,750 / 2,264 | 41,508,250 | 53,699,816 |

同一近似ungaugedとのpairは、frameだけでXY RZ/CXをplanted例で9.92%/3.85%、dense例で4.89%/1.89%削減した。
しかし元Hへの会計b=TεH+PF_biasによるshots増加で、native exactに対するworkは
plantedでRZ34.17%/CX43.22%増、denseでRZ19.73%/CX23.50%増となった。
planted選択は第一factorを2群へcluster化、第二factorはexact。dense選択は第一factorの負の2固有値のみcluster化した。
実測sector model errorはそれぞれ.002113125/.001268925でboundより小さいが、これをboundの代わりに使って勝敗を変えない。
非可換fragmentと相関可能なν2sectorで比較しており、1粒子Gaussian還元の利益ではない。

離れたspectrumではεH=.02内に非自明なclusterが入らずN1はexactへ戻った。
native exactの有限grid最小はRZ q4で34,329,662、CX q2で22,254,064。
zero cutoffは全3例でexactへ戻り、dense例のN1選択は有限shifted cutoffと一致した。
元H whole-Pauliが今回のexact nativeを上回ることはなかった。
なおεH=0のwhole-Pauli再構築でもbinary64係数の微差によりnative RZが変わるため、
originalとreconstructedの小差を近似による構造改善と解釈しない。

この結果はframe機構の実装成立を示すが、basis費用を最小化する目的関数が信号精度込み費用を最小化しないことも示す。
平均中心・有限群探索・今回compiler以外のframe最適化や、精度を織り込んだ構成の可能性は未判断。

## N2：signed chargeを実装できたが加算込みでは高い

JW occupation4、full-ij J二次形式、S∈{−1,0,1}、row d<=2、k<=2、T=.7、係数誤差予算.0021。
DF rank/L_D/delta gridは該当せず密度項は可換、PF誤差0。
constructorへJだけを渡し、40 signed atomsの1/2列820候補をleast squaresでfitした。
負数をtwo's-complementで扱い、compute–controlled phase–uncomputeの全system/ancilla入力とworkspace復帰を検査した。
次表はX/Y合計で、全行shots_per_axis=10,956。workspaceはsystem4/control1とは別のqubits。

| context | 候補 | workspace | XY RZ / CX | originalへのmodel bound |
|---|---|---:|---:|---:|
| signed overlap | direct original / same approximate | 0 | 47 / 64 | 0 / 約1.39e−16 |
| 同上 | J-only generated charge | 6 | 791 / 664 | 約1.39e−16 |
| uniform J=.3 | direct original / same approximate | 0 | 47 / 64 | 0 / 約8.88e−16 |
| 同上 | J-only generated charge | 5 | 543 / 444 | 約8.88e−16 |
| 同上 | known unsigned HWP primitive | 3 | 463 / 380 | 0 |
| perturbed signed | direct original | 0 | 47 / 64 | 0 |
| dense-real rank2 | direct original | 0 | 47 / 64 | 0 |
| sparse disjoint edges | direct original | 0 | 31 / 32 | 0 |

signed overlapは元Sを渡さず等価な2 chargeを回復した。回路費用は直接実装に対しRZ約16.83倍、CX約10.38倍。
uniform生成はproxyにより2 charge（width2+3）を返し、既知unsigned1 chargeにも及ばなかった。
result内の一般説明文字列「one extra sign bit」はこの実際の生成に適用できない。実widths/workspaceを正本とする。
HWP対照も今回の単純increment/phase primitiveであり、既知の最適adder・QROM・low-rank法の包括的性能評価ではない。
小系での負けから大系・fault-tolerant合成での一般的不可能性を導かない。

摂動例、dense-real rank2、疎edge例は820 least-squares候補に予算内のfitがなく、生成器が圧縮を返さなかった。
特に摂動例には、提供されたplanted S/Kを後段で照合すればbound=.002の許容表現が存在する。
しかし同じSへのleast-squares fitはentry-L1 bound=.0036となり予算.0021を外れた。
[post-selector診断](../../artifacts/hamiltonian_algorithm_design/2026-10-11/perturbed_fit_diagnostic.json)を保存した。
これは可圧縮性の否定ではなく、Frobenius fitと採否に使うentry-L1 boundの目的が一致しない生成器の取りこぼしである。
結果を見てfit・予算・候補を変更する追加batchは行わない。

## N3：高い外部エネルギーでは係数下界を取得、小gapでは削減不能

6 modes、ν=2、全sector15、初期active4・sector6、外部はoccupation0で凍結。
one-body e=(−.2,−.1,.1,.25,6,8)または末尾(.4,.6)。density .3n0n1、
hopping12=.2、03=.1、24=.08、35=.04、45=.05。DF rank/L_D/時間発展deltaは該当しないground-energy別課題。
constructorは試行occupation01からU=0、係数norm、first-external classesのsortingだけを使用した。
true ground/gapは選択後の評価にのみ使用した。

| context | 初期下界 | 最終active / sector dimension | coefficient δ | 選択後の実energy shift |
|---|---|---|---:|---:|
| external e=6,8 | gap lower4.86/6.86、β=.08/.04 | 4 modes / 6 | .001549636147 | .000640824981 |
| external e=.4,.6 | 正のgapを取得できず | 6 modes / 15 | 0（full space） | 0 |

active blockには非対角hoppingが残る。gapped例では真のground/gapを使わずにδ<=.004を取得した。
small-gap例は4→5→6 modesへ戻し、最初の2段はUNRESOLVED_LOWER_BOUND、最後はFULL_SPACE。
後者を圧縮成功と数えない。保存verifierも係数下界・U・β・fixed point・選択後のground shiftを照合した。
数学的十分条件をbinary64で評価した結果であり、丸めのinterval認証ではない。
時間発展誤差保証、量子回路資源改善、active solver費用・状態準備・QPE成功率は未評価。

## 既存研究との関係とGPTが決める点

人工的に係数対称性を作り、rotation数とToffoli費用を交換する既知の方向がある。
今回のN1はfactor固有値の近似群とΓ(U) frameの有限探索だが、その実装差分だけで新規性を確定できない。
[Mukhopadhyay–Wiebe–Zhang 2023](https://www.nature.com/articles/s41534-023-00697-6)
の比較が必要である。SCDFもtensor表現とsymmetry shiftを最適化するため、今回の単一平均中心のfinite comparatorを
SCDF全体と呼ばない。[SCDF](https://arxiv.org/html/2403.03502)

N2の集団chargeはHWPや構造化low-rankの費用交換と関係する。
既知HWPはreversibleな加算・位相・逆計算を含むので、その部分を新機構とは扱わない。
[HWP documentation](https://docs.quantinuum.com/guppy/algorithms/examples/hamiltonian_simulation/trotter_hamming_weight_phasing.html)
構造化low-rank法には係数を計算できる構造の仮定があり、今回の指数dictionary探索はそのまま大系constructorにならない。
[Low–Su–Tong–Tran](https://arxiv.org/html/2211.09133)

N3は既知SW/downfolding、active-space選択との関係を精査する必要がある。
今回確認したのは係数から保守的な下界を取る小例で、新しいdownfolding理論や化学精度達成ではない。
[SW theory](https://arxiv.org/abs/1105.0675)、[SQuISH](https://arxiv.org/abs/2211.16522)

GPTへの判断事項は次の三つである。

1. N1を続けるなら、元H精度とfull native/shot費用を目的にした構成へ進むか。既知人工対称性との差分と、
   係数から取得する誤差boundの改善を先に定義する必要がある。basis proxyだけの同型探索を増やす根拠はない。
2. N2でentry-L1予算を直接扱うK fit・候補生成と、より安い算術を研究対象にするか。
   可表現性の回復と実資源改善は別の問いであり、今回のsigned実装成立を資源優位の根拠にしない。
3. N3で真のgapを使わない下界の取れる模型族と、active solverを含むenergy費用を限定して定義するか。
   gapped一例の成立だけで中心テーマへ昇格させない。

## 技術失敗、資源、保存とSTOP

run1 source d0be140：最初のbasis compileでscalar位相差を検出。run2 source b684ba3：
N1 checkpoint保存後、N2の任意入力作用にnon-scalar差を検出。
Qiskit 1.3の`qubits_initially_zero=True`は全入力真空を仮定し、今回の全列契約に合わなかった。
全対照にFalseを適用したrun3 source ed335a0で全計算を完了。
純粋global phaseだけの補正を全列で監査し、280 IR中43本に補正を記録した。
相対位相・workspace漏れは補正不可として拒否する。

run3は最後のpretty-JSON IR集約が64MiB上限を超えTECHNICAL_STOPとなったが、全9 checkpointとresultは保存済み。
stdlib集約器source a08d04cが親hashと全checkpoint/result一致を確認し、数値とgate順序を維持するcompact JSONへ復旧した。
新科学計算0、native IR約37MiB。失敗run・source・auditは保持する。
[復旧audit](../../artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/export_recovery_audit.json)、
[保存integrity](../../artifacts/hamiltonian_algorithm_design/2026-10-11/saved_export_integrity.json)。

spawn2 workers・thread1、run3 wall38.32s、children CPU52.79s、worker peak RSS最大約482MiB。
worker AS4GiB・合計8GiB budget、wall/CPU capなし、per-file64MiB・768 compile上限を保持した。
速度倍率は測定していない。分子load/fitting・量子shots・GPU callsは0。
root baseline1322 pathsのhashとHEAD、および旧manifest85 result setsは不変。
root statusには同時に追加された別Track B review一件が増えたが、変更・復元していない。

global manifest checkerは基点からの`relevant_commits[7]`のdescription欠落で失敗する。
当stageの追加entryとinventory、旧85 entryの保持を別確認し、global checkerをPASSとは報告しない。
公開commitと別Git取得の確認は公開receiptを正本とする。

`FINITE_CONSTRUCTION_FEASIBILITY_COMPLETE_AWAITING_GPT_REVIEW`。
`mandatory_stop=true`、`next_stage_authorized=false`、`central_hypothesis_adopted=null`。
次のscience batch、中心テーマの採択、他Trackの再開は自動実行しない。
