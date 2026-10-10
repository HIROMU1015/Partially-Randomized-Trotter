# Track B G7：固定development入力の条件付き費用とGPT引継ぎ

**G7の限定取得は完了し、mandatory STOP。** 固定m=5ではfull returnの条件付き期待T費用は
登録したordinary、partial-return+tail、P3 closed-form+tailを下回った。
m=3ではP3 closed-form対照を上回る。全return方式のhard shot capと古典生成費用は増える。
これだけで新規性、主method採択、一般分子native改善、次stageを認可しない。
継続・縮小・停止の判断をGPT/利用者へ戻す。

## 固定source・取得範囲

- source S：`ab2549f41b3546fb3940342a2162dd9ee93699c4`
- branch：`track-b-g7-budget-control-economics-20261010`
- base G6：`28cfabb1d47e0e1824bce1e154a9c289738fa2b9`
- authorization：利用者の「こんな感じで進めていく」と、採用[GPT G6 review](../../research/track_b_G6_scientific_review_20261010.md) §12。§12.3は意味論を維持する技術詳細の結果前固定をCodexへ委譲している。旧stageのauthorizationを再利用していない。
- [結果前数学/実行契約](g7_mathematical_and_execution_contract_20261010.md)／[machine contract](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/contract_v1.json)／[52 source hashes](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/source_manifest_v1.json)
- [24 key inventory](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/synthesis_key_inventory_v1.json)／[18 focused tests](../../../artifacts/track_b_g7_budget_control_preparation/2026-10-10/focused_tests.json)
- [原結果](../../../artifacts/track_b_g7_budget_control_result/2026-10-10/v1/result_v1.json)／[保存値監査](../../../artifacts/track_b_g7_budget_control_result/2026-10-10/v1/saved_output_audit.json)／[manifest](../../../artifacts/track_b_g7_budget_control_result/2026-10-10/v1/evidence_manifest_v1.json)

最終状態：`G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW`。
run1、retry0、24 synthesis calls、全strict error PASS、8 rows完了。
失敗prefixによる解釈はなく、追加入力・precision/key/backend探索なし。
GPT reviewの同梱selfcheck directoryは未提供。期待値をコピーせず、式・off-domain testsを独立に導出した。

## 共通taskと実装条件

対象は `P_m=sum_(n=0)^m(-i x R)^n/n!`, `R=sum_i p_i Q_i`, `Q_i²=I` の有限平均。
分子、geometry、basis、DF rank、split L_D、delta windowは該当しない。
以下は**G6から既知のdevelopment形式入力**であり、held-out/independent replicationではない。

- P3_control：p=(3/7,4/7)、x=2/5、m=3。
- P5_general_order：p=(1/5,3/10,1/2)、x=5/7、m=5。

精度はRe/Im各1/200、complex<=sqrt(2)/200<1/100。8×2 axesのfamilywise failure<=1/20。
Nは外向きm2/range/log上界と共通s=0.004987999993999988から決める。
fullは未知B_newを全列挙してNを下げず、`m2_plus=kappa*B_ordinary.hi*U.hi`。
canonicalは同bit精度で`kappa*B_arm.hi²`。exact signalをbudgetへ使っていない。
実measurementは0。shotsは独立uniform-bitと測定の確率法則に基づく十分条件である。

controlled-Q providerはexact query-model。物理T/CX/1Q費用はlabel別の非負変数、一般分子実装は未取得。
controlled rotationをhelper compute–rotate–actual-adjoint uncomputeへ落とし、child CQ2 calls、
helper H4、Rz2、CX2を課す。outerとhelperのworkspace2、word順序とouter Z phaseを保持する。
全方式で同じadjacent Q cancellation、同じconditional preparation/readout契約。
positive Rzは固定pygridsynth2.0.0で一回取得し、negativeはW scalarを含む取得sequenceのactual adjoint。
共通strict operator epsilon=10^-6。今回の最大strict Frobenius upperは約1.768×10^-7。
whole-circuit optimization/compileのcostではなく、共通conditional provider IRの加算会計。

## 保存値の費用表

Nは**axisあたり全試行数**。期待quantum callsとTは二つのaxisの合計。
表のT_Rzは取得Rz primitiveのT/T†加算のみで、provider/preparationのTを含まない。
括弧なしのT値は表示用丸め値。原結果の有理数をprimary sign判定に使う。

| 入力 | 方式 | N/axis | digital acceptance | 期待quantum calls/2 axes | T_Rz/2 axes |
|---|---|---:|---:|---:|---:|
| P3 | ordinary | 698,195 | 1 | 1,396,390 | 195,494,600 |
| P3 | partial+tail | 603,994 | 1 | 1,207,988 | 169,118,320 |
| P3 | P3 closed+tail | 602,596 | 1 | 1,205,192 | 168,550,838.340 |
| P3 | full return | 648,863 | 0.928917223 | 1,205,480.032 | 168,591,120.789 |
| P5 | ordinary | 1,174,526 | 1 | 2,349,052 | 321,659,736.670 |
| P5 | partial+tail | 894,664 | 1 | 1,789,328 | 238,505,280.446 |
| P5 | P3 closed+tail | 880,312 | 1 | 1,760,624 | 246,677,716.207 |
| P5 | full return | 1,014,322 | 0.857766138 | 1,740,102.129 | 237,260,558.660 |

各方式のconditional期待Tは `T_Rz+sum_i A_i*T_Q_i+K*T_prep`。
A_iは保存されたexpected controlled-Q呼出し数、Kは期待quantum calls。
T_Q_iとT_prepを都合よく0へ選んだ勝敗にはしない。
full-minus-baselineのexact affine差を原結果に保存した。下表は表示用近似。

| 入力・対照 | ΔT_Rz | ΔA_i | ΔK |
|---|---:|---|---:|
| P3 ordinary | -26,903,479.211 | (-163,245.971,-225,890.061) | -190,909.968 |
| P3 partial+tail | -527,199.211 | (+1,569.607,-7,245.340) | -2,507.968 |
| P3 P3 closed+tail | +40,282.449 | (+258.370,+338.730) | +288.032 |
| P5 ordinary | -84,399,178.010 | (-255,093.762,-392,923.100,-682,297.478) | -608,949.871 |
| P5 partial+tail | -1,244,721.786 | (-13,302.957,-33,675.029,-94,749.834) | -49,225.871 |
| P5 P3 closed+tail | -9,417,157.547 | (-17,579.167,-23,584.025,-32,571.807) | -20,521.871 |

P5では全係数とinterceptが負なので、**この固定provider/error会計の下で**任意の非負provider/preparation
T費用に対し登録3対照より期待Tが小さい。新methodとしての独立価値の判定はしていない。
P3はclosed対照に全項で正。partialとの比較はprovider費用依存で、normalization減少だけから勝敗を決められない。
新たな5%/10% materiality条件やGOラベルは作っていない。

## 予算・取得・古典費用の制約

fullのP3 U.hi≈1.0757035203450838、P5 U.hi≈1.2967558748239574。
fullのm2_plusはP3≈1.2453860046305594、P5≈1.9478479922430536。
保存したexact digital m2 referenceはそれぞれ≈1.2450872826602655、≈1.9353636204663023で、
予算上界以下。これらreference値は**照合専用**でありNを下げていない。

P5 fullのquantum-call hard capは2,028,644、closedのhard capは1,760,624。
期待費用減少はhard cap減少ではない。actual accepted数の別high-probability capも今回取得していない。
providerの誤差を0とするconditional oracle assumptionを、有限Tで一般分子に達成済みと呼ばない。

| 入力・方式 | production 128 trials CPU(s) | 当該方式のunique Rz cold acquisition CPU(s) |
|---|---:|---:|
| P3 ordinary | 0.003799 | 0.096769 |
| P3 partial+tail | 0.004590 | 0.092819 |
| P3 closed+tail | 0.011436 | 0.391252 |
| P3 full | 0.021921 | 0.391252 |
| P5 ordinary | 0.004177 | 0.122731 |
| P5 partial+tail | 0.007372 | 0.126166 |
| P5 closed+tail | 0.017686 | 0.213005 |
| P5 full | 0.030611 | 0.423001 |

固定SHA256 bitstreamのCPUはinterface診断で、quantum trajectoryや頻度に基づくcost推定ではない。
fullの128trial zero数はP3=9、P5=20。exact digital acceptanceは別support referenceから計算した。
support enumerationはP5 full31parents/63eventsに限定し、productionは表を読まずlocal queryを行う。
reference enumeration/kernel構成の時間は原結果に別保存している。
有限native angle cacheの取得は小support列挙を使う。このnative取得経路まで一般的に非列挙だとは主張しない。
CPU0の短い構成時間はtimer分解能以下の記録であり、無料の前計算という意味ではない。
単一の短い実行の時間はruntime advantageの証拠ではない。

## One-shot/provenance

marker SHA256：`881bc4f8bec0a20c1fcfabbb16d30c6f6c25aa17e19b507c49737d3b6aedba2c`。
total wall約1.548s、CPU約1.546s、peak RSS181,176KiB（約176.93MiB）。
最大per-key wall約0.3054s/CPU約0.3052s。全24key取得のCPU合計約1.293s。
wall1200/CPU900/RSS512MiB/AS1536MiB/per-key30s・20s/output16MiBのcap内。
保存値監査11項目PASS。旧873pathにG6資料を加えた893protected paths不変。
52 source hashes、contract、runtime identity、原結果、G7 markerも不変。
append-only indexesの旧prefixを照合した。rootの別branch/Track Aは変更していない。
DF/分子/NPZ/GPU/LP、新入力、actual quantum measurement、trajectory、whole-circuit compileは0。
matrixはsource freeze前のoff-domain 8×8 synthetic semantic testのみ。

## GPTへ戻す未決事項

1. P5の条件付き期待T減少に、既知return/P3対照後の研究情報価値があるか。今回の小さい差を成功閾値へ変換しない。
2. U予算の保守性、hard cap増加、local-query CPU、angle acquisitionを含め、この経路を継続する価値があるか。
3. exact CQの条件付きquery会計から、現実のprovider誤差/取得/compiled costへ進む必要性と範囲。
4. 実Pauli contextでliteral CTS等の強い既知方法をどのtaskに戻すか。今回CTS比較を追加する認可はない。
5. 一般Green式の既知性と今回のfinite mean/local generation/finite bitsの接続を踏まえ、method delta、新規性、論文着地点をどう限定するか。

**G7終了後はmandatory STOP。追加key、入力、backend、provider、confidence精度、CTS/DF等の科学取得は行わない。**
