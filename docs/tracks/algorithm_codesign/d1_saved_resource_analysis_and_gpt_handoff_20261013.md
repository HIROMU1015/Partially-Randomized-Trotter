# D1：全6方式の保存資源分析とGPT引き継ぎ

2026-10-13 JST。**D1_SAVED_TECHNICAL_ANALYSIS_COMPLETE / mandatory STOP**。
今回の追加分析により、P5のpartial比T費用低下は、r=1ではshot差が単価増を上回る形、r=2では主に単価低下という異なる内訳になった。
六方式の比較ではCTSのCX、closed P3のr=2 native1Qも別の資源上の位置を持つ。
これは固定保存モデルの数値整理で、新規性・一般化・研究採否の判断ではない。

## 入力と実施範囲

正本はD1結果commit `fc96571881f301435a655f057709df4d6c1678e0`。
[原結果・実行監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/docs/tracks/algorithm_codesign/rte_development_d1_results_and_gpt_handoff_20261013.md)、
[resource map](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc96571881f301435a655f057709df4d6c1678e0/artifacts/track_b_rte_development_d1_result/2026-10-13/v1/resource_map.json)を使用。
source S=`874ca12d347f8f35e5d8d64f65f24abc4a2df523`、authorization A=`f803291638d453873e4ab6f1ecf7bcc607f48e4b`。
一回実行済みの科学分類 `D1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW` を変更しない。

明示3-system-qubit synthetic provider、`p=(1/5,3/10,1/2)`、総時間5/7、state000、符号+、P5、r=1,2。
分子/DFではなく、geometry・basis・DF rank・split L_Dは非適用。既知developmentで独立held-outではない。
六方式はordinary、partial、closed P3、local full、closed P5、matched CTS。
axis error1/200、alpha49/24000、全体failure1/20、primitive strict error1e-6、160-bit proposal / 256-bit rootという元契約を保持。

利用者の追加指示に従い、保存scalarのexact rational演算だけで、全方式順位・Pareto・費用分解・r差・affine感度・未知費用を整理した。
1890 events、36 keys、dyadic law、native phase/source/markerの既済監査を最初から繰り返していない。
登録runner、generator、matrix、native compile、合成、sampling、LP、DF、GPUは実行していない。
新candidate、r、精度、shot policy、threshold、科学分類を追加・再採点していない。

branch `track-b-d1-saved-resource-analysis-20261013`、baseは上記結果commit。
[分析source](../../../scripts/tracks/algorithm_codesign/analyze_d1_saved_resources.py)はstdlibのFraction/Decimalと保存JSONだけを使用する。
[入力identity](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/input_identity_and_scope_v1.json)に原commit・bytes/SHA256を記録。

## 1. 全6方式の期待native費用

下表は `G_k(0)`、すなわちprep/readout費用h_k=0の**加法的native intercept**。
単位は10^6 native gates。表示のみ丸め、正確な有理数は[全12行scalar表](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/all_12_saved_scalar_rows_v1.json)が正本。
actual quantum shots、whole-circuit optimized cost、classicalを含む最終総費用の測定ではない。

| r | 方式 | T | CX | native1Q | N/axis |
|---|---|---:|---:|---:|---:|
| 1 | ordinary | 584.443 | 34.499 | 1569.837 | 2088098 |
| 1 | partial | 433.771 | 26.769 | 1152.161 | 1590438 |
| 1 | closed P3 | 448.434 | 26.923 | 1156.942 | 1564916 |
| 1 | local full | 431.491 | 26.389 | 1091.078 | 1803142 |
| 1 | closed P5 | 428.731 | 26.220 | 1084.099 | 1536780 |
| 1 | CTS | 610.541 | 16.367 | 1526.132 | 2675065 |
| 2 | ordinary | 515.023 | 27.905 | 1344.170 | 926574 |
| 2 | partial | 456.252 | 23.480 | 1163.683 | 778141 |
| 2 | closed P3 | 443.162 | 23.591 | 1081.327 | 776048 |
| 2 | local full | 430.039 | 23.532 | 1108.697 | 847756 |
| 2 | closed P5 | 429.626 | 23.510 | 1107.632 | 773966 |
| 2 | CTS | 643.094 | 15.227 | 1623.188 | 1185751 |

小さい順の全順位・tie群は[rankings r1](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/rankings_r1_v1.json)／[r2](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/rankings_r2_v1.json)。
Tの最小は両rでP5、CXは両rでCTS、native1Qはr=1でP5、r=2でclosed P3。
Tは1Qに含まれるため加算scoreを作らない。acceptanceの昇順は記述用で、単独の優劣指標としない。
per-step/per-whole-trial単価、accepted-tail/hard上界、weight M2、range、normalizer envelope、workspace、期待量子呼出しの全順位も
[secondary rankings](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/secondary_saved_rankings_v1.json)へ記録した。
tail/hardは元の上界であり、期待費用の順位へ置き換えない。

`(G_T(0),G_CX(0),G_1Q(0),workspace)`のquantum-only Pareto frontは、r=1がP5/CTS、r=2がpartial/closed P3/P5/CTS。
全12行を同physical taskの保証付き費用として並べたh=0 frontには、この6点が残る。
workspaceは全行1（systemに追加するouter ancilla）。classical費用が未知なので、full-resource Paretoとは呼ばない。
r間は異なるfinite approximationsであり、同一finite meanによるrepresentationだけの比較ではない。

## 2. 費用差のexact factor分解

保存値の会計は、nonzero step massをZ、per-step全trial平均単価をc_k、nonzero条件付き単価をd_k=c_k/Zとすると

```
native intercept A_k = 2*N*r*Z^r*d_k
prep/readout slope B = 2*N*Z^r
G_k(h_k) = A_k + B*h_k
```

である。zero-fillを除外した平均へ切り替えていない。
積factorはN、r、Z^r、d_k。すべて保存値から読み、単価を再合成・最適化しない。

費用差のinteractionは全factor順序の平均と等価なmultilinear Shapley配分を使った。
factor iの寄与は `Delta f_i * integral_0^1 product_{j!=i}(f_j0+t Delta f_j) dt`。
各interactionを参加factor間で対称に配り、寄与のexact sumが保存された費用差へ一致する。
これは**記述的配分規約**で、一意な因果割合ではない。相殺があるとnet差の寄与率は100%超または負になる。
任意hに対してはintercept差とh×slope差を分離して保存した。

### P5対partial：Tの内訳

両方式は各rでZ=1、同じr、同じTaylor/native bias予算なので、acceptance/r factor寄与は0。

| r | N_P5/N_partial | d_T,P5/d_T,partial | A_T,P5/A_T,partial | Tのnet差 |
|---|---:|---:|---:|---:|
| 1 | 0.966262124019 | 1.02289103826 | 0.988380867273 | -1.161913273% |
| 2 | 0.994634648476 | 0.946721065724 | 0.941641574412 | -5.835842559% |

r=1はNが3.3738%少なく、単価は2.2891%高い。対称配分ではshot差が約-14.802 million T、単価差が約+9.762 million T、net約-5.040 million T。
net改善に対する署名付き寄与はshot293.688%、単価-193.688%であり、「shotが約294%の独立改善を生んだ」という解釈ではない。

r=2はN差0.5365%の低下に対し、単価差は5.3279%の低下。net約-26.626 million Tについてshot約8.949%、単価約91.051%という対称配分になる。
したがって、**P5がpartialより低いTを示す同じ観測でも、r=1とr=2の保存値の内訳は同じでない**。
ここで単価とは保存されたevent混合の条件付き平均で、angle・basis・word等への独立因果分解を新たに行ったものではない。

[主比較factor表](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/primary_P5_vs_partial_factor_summary_v1.json)。
全15 pair×3 metric×2 rの90分解は`within_r1/`・`within_r2/`の小JSONに保存。
pairのbaseline/alternativeとdelta符号を各fileに明記し、都合のよいpairだけを選んでいない。

### N差：moment、range、margin、ceiling

元のBernstein式の連続部分は `L*(2*M/s^2+4*W/(3*s))`、Lは元の固定log上界、Mはbudget用M2上界、Wはrange上界、sは残余幅。
保存Nを更新せず、元式と保存endpointの差を以下へ分解した。

1. M2 term差。
2. range term差。
3. Taylor bias差によるmargin効果。
4. implementation bias差によるmargin効果。
5. 保存整数Nのceiling residue差。

積は対称endpoint配分、非線形marginは保存された二つのs間のsecantによる規約を用いた。
新しい混合candidate、仮想shot policy、丸め後Nを採点・登録していない。
各分解のsumは元のN差にexact一致、ceiling residueは各endpointで[0,1)と確認した。

within-rでは共通sのためTaylor/implementation寄与が0。P5/partialのN差はほぼM2上界差で、rangeとceilingは小さい。
T費用のshot配分へ戻したM2寄与はr=1で約-14.787 million、range約-0.014645 million、ceiling約-68.73 T。
r=2ではM2約-2.379268 million、range約-0.003346 million、ceiling約-127.37 T。
保存された全行の連続式はM2 termが約99.719–99.849%を占める。ただしこれはshot上界式の構造で、量子測定の実測variance割合ではない。
実dyadic weightのM2とbudget上界M2を表で分け、真のsignalを使ったvariance最適化へ置き換えていない。

## 3. r=1→2の違い

同じarmのr=1→2は、representation改善の対照ではなく、同総時間の時間分割に伴う別finite approximation・価格・bias・momentの変化。

| 共通budget量 | r=1 | r=2 |
|---|---:|---:|
| Taylor product upper | 0.0005533762860 | 0.00001729308370 |
| implementation product upper | 0.00001600000800 | 0.00003200054869 |
| axis bias upper | 0.001138752588 | 0.00009858726477 |
| statistical width s | 0.003861247412 | 0.004901412735 |

Taylor上界は小さくなるがnative誤差は反復で増える。N低下をP5等の表現改善へまとめて帰属しない。
六armのN2/N1は約0.443–0.504。量子step数factorが2になるため、Nが半減しても総native費用が必ず下がるわけではない。

| arm | G_T,r2(0)/G_T,r1(0) |
|---|---:|
| ordinary | 0.8812194790 |
| partial | 1.051827802 |
| closed P3 | 0.9882438465 |
| local full | 0.9966353951 |
| closed P5 | 1.002088183 |
| CTS | 1.053318900 |

例えばP5では、N ratio0.503628、r factor2、conditional T price ratio0.994869、path acceptance ratio1で、T interceptは約0.2088%増える。
partialではT interceptが約5.18%増える。ordinaryでは約11.88%下がる。
local fullのwhole acceptanceは約0.857766→0.913837で、zero-fillの量子初期化率変化も分解に含めた。

P5のN差-762814は、対称配分でmoment約-235065、range約-273、Taylor margin約-543705、implementation margin約+16228、ceil約+0.041。
error幅の改善がNを減らす一方、native誤差蓄積は逆方向に働く。
この配分も代数規約であり、rだけを変えて他factorを固定した新実験ではない。
[18 r-contrast files](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/repeat_count/)を参照。

## 4. 全方式のprep/readout感度

各r・metricの全15 pair、全12行の全66 pairについて、nonnegative交点、弱優劣域、tie、各open intervalの全順位をexact rationalで保存した。
partitionにh=0を必ず含める。0がpair交点でない場合もあり、実際の交点は各pairの`crossover`が正本。
これは保存された直線の解析で、新しいparameter gridの科学評価ではない。

| domain | Tのminimum envelope | CXのminimum envelope | native1Qのminimum envelope |
|---|---|---|---|
| r=1の六方式 | P5、全h_T>=0 | CTS→P5、h_CX≈4.327676971 | P5、全h_1Q>=0 |
| r=2の六方式 | P5、全h_T>=0 | CTS→P5、h_CX≈10.05625615 | closed P3→P5、h_1Q≈6317.245619 |
| 全12行 | r1 P5→r2 P5、h_T≈0.5868195982 | r2 CTS→r2 P5、h_CX≈10.05625615 | r2 closed P3→r2 P5、h_1Q≈6317.245619 |

交点そのものでは表中の二線がtie。ほかのpair交点でも全順位は変わり得るが、minimum envelopeの切替と区別する。
**partial/P5のr=2 CX交点3.514949878は、全六方式の最小CX方式が切り替わる点ではない。** そこではCTSがまだ最小CX。

[r1 T](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r1_T_v1.json)／[r1 CX](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r1_CX_v1.json)／[r1 1Q](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r1_1Q_v1.json)、
[r2 T](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r2_T_v1.json)／[r2 CX](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r2_CX_v1.json)／[r2 1Q](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/sensitivity_r2_1Q_v1.json)。
joint_12の3 filesも同directoryにある。

## 5. h依存のquantum-only Pareto

T/CX/1Qのprep/readout費用は別coordinate `h_T,h_CX,h_1Q`であり、同じ数値のhを課した単一scalarへまとめない。
全有向pairについて、各coordinateでの線形不等式の積領域と、全coordinate同値になる点の除外を保存した。
strict dominanceには少なくとも一coordinateのstrict差を要求する。
metadataのworkspaceは全行同じ。classical未知量はPareto判定に無料で補完しない。

各r内での保存モデルfrontは次の条件で表せる。

- r=1：P5は常に残る。CTSはh_CX<4.327676971で残り、以上ではP5がquantum coordinatesでdominateする。他四方式はP5にdominateされる。
- r=2：P5は常に残る。partialはh_CX<3.514949878、CTSはh_CX<10.05625615、closed P3はh_1Q<6317.245619で残る。ordinary/local fullはP5にdominateされる。

境界で一coordinateがtieしても、他coordinateにstrict差があればdominated側はfrontから外れる。
これらは非負hの形式的model域。実際のprep/readout回路の費用vectorは未取得で、Tが1Qに含まれる等の実現可能性制約へ後から限定する必要がある。
未知classicalを含むdominanceや「最良algorithm」の採択はしていない。
[r1領域](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/pareto_regions_r1_v1.json)／[r2領域](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/pareto_regions_r2_v1.json)／[joint12](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/pareto_regions_joint_12_v1.json)。

## 6. 古典費用と情報取得の不足

[不足整理JSON](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/classical_and_information_gaps_v1.json)は、元の12 setup timing、worker CPU、外側receiptを参照として保存した。
保存setup CPU合計0.176721142 sはconstructor/budget/reference enumeration・CTS collection等を含む技術計測。
productionだけの初期費用や、online per-trial生成費用として扱わない。
合成worker CPU7.855002 sと全run CPU631.759466 sも、quantum trialごとの古典費用ではない。

未取得のonline費用は、row別 `C_classical = C_setup + 2*N*c_attempt` と記号的に残す。
zero trialもattemptに含み、無料としない。実際にearly abortするか、random bits/word/angle/cache/rejectionの費用も測定していない。
全方式で同じc_attemptと仮定して順位を確定しない。
classical setup/price acquisitionのamortizationと量子統計trialの費用は別問題。

明示小providerで同じI1情報を全方式へ提供した比較であり、CTSのPauli collectionやfull/P5 normalizer情報がDF規模でどれほど高価かは未取得。
I0 advantage、分子適用、whole-circuit optimization、SP interaction、強いIS最適化の比較へ一般化しない。

## 7. 技術検証・取得性・STOP

新しい[12 scalar checks](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/scalar_validation_v1.json)は人工有理数factor、permutation平均との一致、tie/交点、strict Pareto、workspaceを検証しPASS。
保存D1の1890 event/36 key監査や57 semantic tests、full suiteは繰り返していない。
追加のscientific run/compile/synthesis/matrix/sampling/LP/GPUは0。
分析の各factor差・N差・ceiling residueについて、元保存endpointとのconservationをexactに確認した。
原結果、source、contract、authorization、marker、STOP、科学分類は不変。

[分析manifest](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/analysis_manifest_v1.json)に全90 pair-metric分解、18 r-contrast、感度・Pareto filesを索引化。
解析sourceは一つの共通log/12 scalar rowsをreuseし、約0.28 CPU s、peak RSS17152 KiBのstdlib計算。scientific workloadやGPU backendを起動していない。
すべての新JSONを512 KiB以下に保持し、最大fileは約234 KiB。exact値と[decimal表示用summary](../../../artifacts/track_b_d1_saved_resource_analysis/2026-10-13/v1/display_summary_v1.json)を分けた。

次のGPT科学レビューは、D1原結果とこの追加分析を一束で扱えばよい。
レビューで判断するのは、r依存のshot/単価機構がどこまで一般化できるか、強い対照や未知classical費用のもとでどのclaimに研究価値があるか、追加検証を行う必要・範囲である。
Codexは研究の継続/縮小/採択を決定していない。新しいr、provider、precision、compile、合成、SP、性能修正・再実行へ進まない。
**mandatory STOP、次の科学実行は未認可。**
