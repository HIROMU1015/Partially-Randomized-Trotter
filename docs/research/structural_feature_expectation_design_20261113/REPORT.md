# 残差説明変数となる構造量の期待値：調査・人工検証

2026-11-13 JST。**Z/ZZ rotation数・order別個数は解析式で取得可能。C1のtail basis group/shared-pair数も、非identity・同一IID component lawなら閉じた式で扱える。** それ以上のnative相殺やcompiler出力量は、特徴量ごとに状態の十分性を確認する必要がある。

今回はSOURCE調査・数式・最小実装・人工テストのみ。実H4の324確率/13 order分布、F0、K0/K1、E[S]、保存価格/残差の新解析を実行していない。既存26 E[S]・区間・SOURCE・STOPは不変。実特徴量平均・平均compiled C・価格資格は未取得。

## 固定成果と再利用判断

|SOURCE|確認した箇所|再利用と限界|
|---|---|---|
|C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`|全13round E[S] REPORT/HANDOFF、科学SOURCE de2427d4aのh4_structural_kernel/MeanKernel|Sの既存平均・区間を再利用。保存reward gはS用で、新特徴量のrewardではない|
|law `d72dc4e6228bb380f4d1d000e5e030794fce7164`|finite_k_sampler_law、REPORT/MATH、既存cdf_law参照|理想source比とCDF格子・固定seedを分離。実H4のlaw差は未評価のまま|
|D P2 `39ddd59edf54a3a901e5302207b2765ddbd03fbb`|P2 REPORT/HANDOFF、p1_efficient_zz.event_metrics、p2_direct_pilot|P2は直接paired残差pilot。**説明変数や残差fitは採択していない。** 新しい残差統計を算出せず、既存観測を参照|
|D高速化 `5a67cea7705e80ff295306606611d774724ae2e2`|compiled_price_fast.structural_pair/price_accounting、TECHNICAL_HANDOFF/P2_MINIMAL_PLAN|元AccountingのRZ射影とX/Y prefix共有。event全列・回路の走査は残る。Cで取得pipelineを再実装しない|
|元counter `8457c443554af5fadaa3ba36f97f6316ca3c5050`|h4_df10_price_v1.Accounting.tail/basis/flush_basis/_native|featureの定義を直接確認。ローカル版は固定counterと同一blob|
|過去Track D|track_d_c7_4_models.event_features|basis run・lane maxの概念は参考。別H4/CDF8/L_D2/K0/controlの数値係数・lawを移植しない|

対象となる物理条件は既存Cと同じLinear H4/1.00 Å/STO-3G・保存tol-only DF10、L_D1/S2/ΔPF=.1/finite-K2/h=.1/M12/P0、元exactκ/H/state/snapshot・fragment/basis/qubit順・scalar一回、symmetric-directional full Hadamard X/Y。canonical `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、preparation `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`、324非identity/9 basis。Dの新lowering `CONTROLLED_ZZ_CX_CRZ_CX_V1`を旧価格identityへ付け替えない。今回はこれらの実確率を読み込んで評価していない。

## 特徴量別の可否

分類1=解析式、2=既存kernelの最小拡張、3=動的計画法/小状態reward、4=大きな状態又はsamplingが現実的な場合。分類は数値取得済みを意味しない。

|特徴量・厳密な単位|方法|今回の到達点/不足条件|
|---|---|---|
|DのN_Z、N_ZZ|1|tailの非identity **rotation application数**。PF head、product、低レベルRZ、compiled RZを含まない。人工値をDの固定event_metricsと照合|
|order0/order2 event数、component/sign別rotation数|1|有限分布の周辺重み和。希少order2も式に含む。全path IIDは線形な平均の必要条件ではない|
|product application数、productのsupport別CZ呼出し数|1|order2は2 product、各productはsupport長個のCZ。native/compiled CXやRZ数とは別|
|identity application数|1、前提付き|固定preparationは全component非identityという保存証拠を再利用。この前提の外では0を代入せず、今回の実装はidentity supportを拒否|
|tail_basis_groups、tail_shared_basis_pairs|1又は2|sourceのactive basis分岐。IIDなら下記閉形式、異なるlawの順次処理は既存MeanKernelの小kernel。元Accountingと人工全列挙で一致|
|ordered basis pair、first/last basis|1/2|IIDならpairの積・末端周辺。異なる分布やhistory条件にはstateを保持したkernel。basis ID/hashをsupportだけで統合しない|
|PF–RTE entry/close、outer・scalar・X/Y境界の特徴量|1/2/3|回数が固定ならq、q−1、scalar/wrapper一回等。ただし境界の**相殺reward**は元wordと状態が必要。頭/終端を独立にfinishしない|
|basis_transforms_emitted、adjacent_inverse_basis_pairs_cancelled、raw native RZ by role|2/3|active basisだけでなくpending basisのoperations/inverse/roleを追うrewardが必要。group数の定数倍で置き換えない。新物理reward/parityは未実施|
|tail relative-phase gateの有無|1/2、前提付き|固定非identity lawと既存phase証明ではorder2-seen flagで足りる。πの逐次加算値やfold後RZ寄与を非零flagで代替する意味ではない|
|fold後RZ、literal zero/pi-family、wire依存の相殺量|2/3、observable依存|既存Sと同じrewardなら26値をそのまま使う。新rewardならpending-Z hex・native/role/phase/terminal parityを改めて確認。Sの縮約が任意特徴量に十分とは限らない|
|特徴量の積・分散・cross-wire covariance、max lane/depth|3/4|E[f(X)]≠f(E[X])。joint transition/reward又はモーメントを要する。wire-marginalだけではcross-wire相関を復元できない|
|Qiskit最適化後のnode/gate/depth等|4、又は限定モデル|compilerが有限局所状態で閉じる証拠なし。実compile由来のDを解析式だけで確定しない|

## 数式と確率条件

n=qR、p=p2、component確率π_j、support d_j∈{1,2}、ρ_d=Σ_(d_j=d)π_j、basis mass b_l=Σ_(basis_j=l)π_j とする。πとpは**対象lawごと**に定義する。Cのideal source lawでは保存binary64 raw weightの厳密比、uniform-gridモデルなら元CDF helperの実効mass。固定seed PCG64を自動的にIID product lawへ置換しない。

```
E[N_Z]       = n ρ_1              E[N_ZZ]      = n ρ_2
E[N_order2]  = n p                E[N_order0]  = n (1-p)
E[N_product] = 2 n p              E[N_all_app] = n (1+2p)
E[N_product_CZ] = 2 n p (ρ_1+2ρ_2)
E[N_rotation of component j] = n π_j
```

これらは同じ条件付きcomponent周辺が各eventで成り立てば、event間独立性を要さない。product/categoryにorder依存があれば、例えば2n p Σ_j π_(j|2)d_jへ置き換える。SOURCEが同じ関数でも、実PRNGで条件付き周辺が保証されるとは限らない。

非identity・component IID、orderに依存しない同一component lawでは、tailのapplication列長Lの平均はR(1+2p)。長さに条件付けてもbasis列がIID、かつR≥1なら、H=Σ_l b_l²として

```
E[tail_basis_groups over q tails]       = q {1 + [R(1+2p)-1](1-H)}
E[tail_shared_basis_pairs over q tails] = q [R(1+2p)-1]H
E[ordered pair l→k per tail]           = [R(1+2p)-1] b_l b_k
```

sourceのAccounting.tailは毎回active=Noneから始めるため、**group数の式**は各tailで最初のgroupを一つ数える。この局所変数の初期化を、pending basis/Z・counter全体をresetしてよい根拠にしない。queue上のinverse basis相殺はhead/close/outerを跨ぎ得る。basis groupとnative transformを区別する。

小状態kernelでは1 applicationの(A,g)を元MeanKernel.from_eventsで構成し、order2の3 applicationをthen/repeatで合成、eventを `p0(A,g)+p2(A,g)^3` とする。成分選択順はrotation→products、作用順はproducts→rotation。同じIID basis marginalという条件がこの因数分解を許す。新小kernelはbasis group/shareのobservableだけで、物理pending-Z/phase/compilerの閉包証明ではない。

pieceごとに異なるlawなら順序付き合成、entry/close/outerには元境界rewardを使う。確率とrewardの合成則は `(P1P2, g1+P1g2)`。1tailのgroup計数だけなら閉形式の方が軽く、matrix法に固執しない。tail phaseの有無は既存証明の条件下で `q[1-(1-p)^R]`。これはraw P gate呼出し数で、fold後のRZ数でもcompiler残差でもない。

## 計算量・精度・相関

新解析式はcomponent数Jの重み集計とbasis histogram（O(J+B)、bit演算cost別）で、qR個の列を作らない。新kernelはB+1状態、構築O(BJ)、反復は既存二乗法O(B³ log R)、記憶O(B²)。outer合成が必要ならlog qを追加できるが、その境界stateが十分であることが前提。m12のq4096/R8160・33,423,360位置でも人工入力では列生成なし。実H4での速度改善は測定していない。

解析式実装はFractionで供給weightの比を正確に扱う。これは解析的実数weight/libmの誤差保証ではない。小kernelは既存float64 MeanKernelを用い、mass toleranceは変更せず、数値enclosureは未実装。物理利用時は元Band/discrete/mixtureの外向き区間処理を再利用して新rewardを検証する。現在のkernel/SOURCE/数値契約は変更しない。

線形平均が求まっても分散が求まるとは限らない。IID rotationならN_Z+N_ZZ=n、Var(N_Z)=nρ1ρ2、Cov(N_Z,N_ZZ)=−nρ1ρ2だが、phase/product数やbasis rewardとのjoint momentは別に必要。例えば一eventのN_Z N_ZZは常に0であり、平均の積は正になり得る。cross-wire rewardの共分散は独立として足さない。

law誤差はd72dc4e6をそのまま継承する。線形個数は周辺距離の和によるobservable別上界を使え、巨大trajectory TV×巨大rangeという過大な上界を必須にしない。例えばrotation categoryの一event差≤δ_rotなら個数平均差≤nδ_rot。basis pairには隣接joint lawの根拠が必要。一般のfeatureはosc(feature)×TV(path)が候補だが、履歴条件/PRNG/oscillationが未知ならnull。

## Dの残差への利用条件

固定された線形model `D(ω)=θ0+Σ_j θ_j F_j(ω)+ε(ω)` なら、同じlawで `E[D]=θ0+Σ_j θ_j E[F_j]+E[ε]`。E[ε]や係数/共有covarianceを0にしない。非線形modelではE[F]だけでは不足し、採用前にjoint量と計算costを比較する。

一方D P2の直接法 `E[S]+mean(C−S)` は特徴量fitを必要としない。既存pilotが支持する直接法を、本技術成果だけを理由に複雑なmodelへ置換しない。control variateにFを追加するなら、同一lawの既知E[F]・同pathのF/C/S・事前固定係数又は別校正/適切な推定設計が必要。係数を同じ有限cohortで選択した後に自動で不偏・精度保証を主張しない。

N_ZZだけではDを説明できないというDの既存観測を受け、今回の式を `D∝E[N_ZZ]` という新modelとしては提供しない。希少order2のE個数が得られても、条件付きcompiled残差や全lawのtail保証は別。強制rare一例はnormal法則の平均にしない。

## 実装・確認・STOP

[structural_feature_means.py](../../../src/trottertracks/structural_feature_means.py)を追加。既存probability比処理、MeanKernelの検査・合成・二乗法を再利用し、physics reader/runner、全軌道生成、compiler、fit、B controllerは追加しない。既存kernelと全26値も変更しない。

[TESTS.txt](TESTS.txt)：新9件＋既存19件=28技術test PASS。人工10種のK2 event alphabet/長さ2の全100列で元Accountingのbasis group/shareと解析式・kernelが一致。Dの固定event_metricsとも人工一致。piece/次eventへの状態継続、tail active resetの意味、希少確率、巨大位置数、非線形平均の反例、未知/対象外feature拒否、旧数値区間/nominal不修復を確認。作者自身の技術確認であり、独立科学レビューではない。

[HANDOFF](HANDOFF.md)にDへの仕様、[SCIENCE_PLAN](SCIENCE_PLAN.md)に必要になった場合だけの有限scopeを記載。既存C branchを継続、他担当のbranch/worktree/private raw/cache/processへ変更なし。SOURCE/input/raw/結果/KEEP/grant/ledger/marker/旧STOPを保全。実H4科学計算0。

**Codex D**が必要なfeatureの厳密な定義・使用law・model形式を指定し、**ユーザー**が必要な有限科学scopeを判断する。**Codex C**は別途許可された項目だけを将来評価する。今回はcommit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。Dから後続特徴量が届いても実H4評価へ自動進行しない。
