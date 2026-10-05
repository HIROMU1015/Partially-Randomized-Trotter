# BM-1 小型model pilot提案 v1

2026-10-05 JST。**BM-0における結果前の設計候補。preregistration／実行authorizationではない。**
数値は未実行の入力案。GPT reviewで採否を決め、その後source・manifest・authorizationでfreezeする。
今回はmodel行列生成、state/signal評価、sampling、circuit/compileを一切行わない。

## 1. 一度のpilotで判別すること

問いは、固定native列でfine/coarse比を変えたとき、内部／cut誤差とfusion後countに基づくDF量が、
norm-onlyでは見えない選択へ結び付くか。さらに**同じDF backendを使うgeneral compact対照**以上の差が
あるか。差がなければnew-method候補の縮小判断へ戻す。
最良PF、最良sampling、分子の優位を探すpilotではない。

比較する構成は[数学仕様](bm0_native_sequence_and_error_spec_v1.md)のnested `m={1,2,4}`と
current flat partial S2。A/B割当・native順序は各modelで固定。nested m1とflatを同一視しない。
R、抽出分布、r/Kは固定し、係数探索・全subset探索・全order探索をしない。

## 2. 提案する入力recipe（数値生成前）

`n=4` spin orbitals、N=2 sector（dimension6）。基底は`12,13,14,23,24,34`、
評価state案はその等重み実vector。one-body=0、c=0、lambda_i=1。
`D_i=F(G_i)^2`、固定residual `R=(1/20)F(G_R)^2`、`G_R=diag(1,1,2,2)`。
単一粒子／involutionにより全DF平方がidentityへ退化する構成は使わない。

記号を次のように置く。いずれも4×4 Hermitian小行列の**recipe記述**。

```text
G1 = diag(1,2,0,0)
G2d = diag(2,1,0,0)
G2x = [[2,1/2,0,0], [1/2,1,0,0], [0,0,0,0], [0,0,0,0]]
G3 = diag(0,0,3,1)
O13(theta): modes 1,3の実rotation、他modeはidentity
```

| Family／instances | A（固定順序） | B | 設計した構造／未実証の観測対象 |
|---|---|---|---|
| commuting：1 | F(G1)^2, F(G2d)^2 | F(G3)^2 | 同時対角化control。理想PF誤差は消える。有限tail biasをm効果と誤認しない |
| internal：1 | F(G1)^2, F(G2x)^2 | F(G3)^2 | A内部非可換、B/RはAと可換。fine内部項がmで減る構造 |
| cut：1 | F(G1)^2 | F(G2x)^2 | Aが単一exact block、K_A=0。A反復はfusionされる。mを増やす改善と誤認しない |
| cost/adaptation contrast：3 | F(G1)^2, F(G2x)^2 | F(O13(theta) G3 O13(theta)^T)^2 | theta={0,pi/8,pi/4}。各fragment固有値/normを保持し相対非可換構造を変える |

合計6instances、deterministic最大3DF fragments＋residual1。internal/cutという名称は登録した構造の
役割を指し、finite-Tの大小を測定済みという意味ではない。rotation下の実装cost一定は仮定しない。
全登録controlを保存し、成功したinstanceだけを報告しない。

## 3. Parameter・cost・baselineの有限案

| 項目 | 候補値／扱い |
|---|---|
| T／accuracy／failure probability | T=1/2、epsilon=.02、alpha=.05、alpha_Re=alpha_Im=.025（提案、未承認） |
| q、m | q={1,2,4}、nested m={1,2,4}、flat各q |
| finite RTE | 各macroに一occurrence、r=4、K=2固定案。新K4 triggerなし。canonical source・次数規約・scalar抽出は要freeze |
| random seed | NONE案：BM-1はanalytic mean評価のみ、trajectoryなし。明示recipeのみ、ランダム入力探索なし |
| state | 上記の一state、I2評価専用。design accessに渡さない |
| ordering/fusion | A/B表の順序と数学仕様のexact adjacent fusionを全対照へ同じ適用 |
| primary costの候補 | **synthetic weighted native block proxy**：各A block=1、各B block=8、各R occurrence=4。一つの固定cost modelとして全familyに同じ適用 |
| shot/action metric | 共通b/u/Bの十分shot式によるG_proxyをI2評価で算出する案。I1 selectorのmodel scoreとは分離 |

cost weightは機構検査の明示した人工設定であり、DFから推定したgate costではない。
basis変換・control・phase・gate synthesis・state準備のactual costを表さない。
contrast familyのproxy costが同じでも、actual basis costが同じとのclaimはしない。
cost modelの採否をGPTで決め、結果を見てweightを変えない。
current flatは登録D順序のforward/reverse、nestedはB外側・A内側という別列。
同q/r/Kのtail意味論を共有し、現sourceのboundary fusionをflatにも適用する。

| Baseline／arm | 設計情報・探索機会 | 役割 |
|---|---|---|
| flat native S2 | 各qを共通評価、現在の構成を適用 | nested overheadを含む採用価値 |
| nested m1 | 各qを共通評価 | groupingとrate変更の分離 |
| norm-only grouped BCH selector | 同じ12列domain、同じcost・leading式、fragment normのみ | 一般norm情報との比較 |
| general compact BCH + DF backend selector | 同じ12列domain、G_i/N、同じordered-word backend | 新規性に関係する最重要の同情報対照 |
| DF変更adapter selector | 同じ12列domain、G_i/N、共通floor・内部寄与を再利用 | 特定変更の評価・列生成に差があるか |
| exhaustive oracle | 既存12列のI2共通参照のみ | regret/mechanism評価。候補追加・selectorへの逆流なし |

I1 selector候補は同じleading modelとcostで最小scoreを選ぶ。tie rule案は低cost→flat→低q→低m。
finite-T certified boundにするか、leading heuristicとして使うか、accuracy予測からshot数へどう渡すかは
**未固定**。heuristicならoracle参照で予測missを測るが、I1 accuracy保証と呼ばない。
I2のb/uを見てselector係数・thresholdを調整しない。同情報対照がDF案と同じなら、その一致を主要結果に残す。

SPRINT等との完全比較、full deterministic/discard endpointはBM-1に自動追加しない。
適用可能な既知near-integrable列との対応はBM-0 reviewerがscopeを確認する。
それらを比べないBM-1から「SPRINTより有利」「partial全体が最適」とは言わない。

## 4. 計算budget案とresult-prior freeze

各instance：nested3m×3q=9列、flat3q=3列、計12列。
6instancesで**最大72登録列、ideal72＋finite72＝144 logical mean evaluations**。
selector／cross-scoreは同じ登録列の情報から評価し、別候補を取得しない。
6 exact targetsは一instance一度の評価用cache案。これは許可された生成数ではなく、未承認上限案。
tail polynomial／normalization中間計算・I1小行列取得・numerical guardの回数をsource manifestで別計上し、
144という表示へ隠さない。adaptive grid、予備geometry、追加state、r/K sweepなし。

未固定事項を閉じるまで実行しない。

1. recipe、T/epsilon、cost weights、q/r/K、state、seed NONEの採否とcanonical input manifest。
2. exact列とfusionのsemantic tests、新finite adapter source identity、sector証明。
3. I1 selectorの完全なscore／tie／feasibility／shot予測、同情報compact対照との対応。
4. finite-T remainderを含むcertificateか、leading heuristicかというclaim範囲。
5. numerical u_a・precision・tolerance、materiality／regret／mechanism criteria。
6. wall/CPU/RSS/output上限・中間演算上限を環境に結び付けて結果前固定。
7. atomic保存・標準Python型正規化・round-trip／synthetic end-to-end保存の限定test。
8. science source commit、別authorization、明示実行指示、one-shot registry、全outcome STOP。

旧BFの5%・Aの10%・4000cells/4hは流用しない。新しい判定thresholdはGPT側がscopeとともに決める。
model/typeを増やしてpositive resultを探さない。判定不能は理由付きINCONCLUSIVE、grid自動拡張なし。

## 5. 保存と停止

将来の結果は列identity、group/order/m/q、ideal／finite分離、取得情報・古典cost、
predicted／oracle bias、cost分解、選択、miss/regret、失敗controlを保存する。
科学結果を先にatomic保存し、補助cross-score/表示のserialization失敗でprimaryを失わない。
過去BFのsource/result/markersを新BM identityに流用しない。

BM-1が認可された場合も一度の有限pilot後、全outcomeでSTOPしGPTへ戻す。
BM-2 compile、実分子NPZ、追加geometry、BM-3独立validationは別判断。
**現段階はBM-0文書のみ、BM-1実行は未認可。**
