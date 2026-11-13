# D向け：構造特徴量の平均を取得する方法

[REPORT](REPORT.md)／[SOURCE](SOURCE_INDEX.json)／[人工・回帰28 PASS](TESTS.txt)／[未実行の有限案](SCIENCE_PLAN.md)。実H4の新期待値は提供していない。

## D固定成果との接続

- P2 `39ddd59edf54a3a901e5302207b2765ddbd03fbb`のp1_efficient_zz.event_metricsを確認し、人工eventでN_Z/N_ZZ/order/product/identityの定義を照合した。P2の新しい統計、fit、モデル採択は行っていない。
- `5a67cea7705e80ff295306606611d774724ae2e2`のcompiled_price_fastは元Accountingのnative RZ射影/共通X-Y prefixを使う。Cはこの取得pipelineを再実装しない。
- 全26 E[S]は`9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f`のまま。新lowering/control/compiler bindingと旧loweringを分け、S自体を変更しない。

## 必要な特徴量が決まったら

Dから次を指定する。共通B APIの変更や承認証明機構は不要。

|情報|例/理由|
|---|---|
|feature名・SOURCE・厳密な計数定義|rotation_ZZはtail rotation applicationだけ。product CZやPF head、低レベルRZを含めるなら別feature|
|stateと境界|tail内部・head/close・outer・scalar一回・axis suffix・terminal。元演算順/役割を保持|
|target law|ideal source ratio、IID uniform53-bit CDF model、又は実PCG64 seed ensemble。名称だけで同じにしない|
|対象identity|canonical/physical candidate/m/q/R/T/ΔPF/finite-K/preparation/lowering/compiler/axis|
|model形式と用途|固定線形説明変数、非線形、control variate、又は単なる診断。必要なjoint momentが違う|
|必要精度・範囲|特徴量平均の精度をcompiled Cの必要精度と同一視しない。残差model誤差はDの別課題|

今回のlibraryは値だけを返すsupplied-model内部関数で、Bの価格recordを生成しない。物理評価が承認された後の成果では、feature名/定義SOURCE/単位・値又はmissing・law/仮定・数値区間・資格・candidate/source bindingを添える。未知のlaw差/compiled systematic/covarianceはnull。平均FをFULL_WRAPPER_MEAN_COMPILED_PRICEへ変換しない。

## 今すぐ再利用できる技術

`structural_feature_means.py`：

- `k2_linear_counts(raw_component_weights, support_sizes, raw_order_weights, q=..., R_outer=...)`：非identity Z/ZZの解析的first moment。Fractionによる供給重みの比。全event列・乱数なし。
- `k2_basis_counts(..., basis_keys, ...)`：同一IID nonidentity lawのtail_basis_groups/shared pairs。basis keyは元ID/hashを保全。native transformやcompiled相殺の式ではない。
- `k2_basis_kernel(component_probabilities, basis_keys, order_probabilities, feature=...)`：異なるlawの順次合成を既存MeanKernelで扱える小state。Noneはtail active未設定。pending basis/Zやcompiler状態は含まない。

典型的な同一lawは閉形式が最軽量。kernelへ実入力を与える経路、物理rewardの生成、Sの再評価は今回追加していない。既存K0/保存kernelのPやgを別observableに流用する場合、新rewardのsource/parity/precisionが必要。Sの状態同値性を任意特徴量へ拡張してはならない。

技術取込みは新moduleと対応testだけを選ぶ。依存は既存`mean_compiled_price.MeanKernel`と、d72dc4e6の`finite_k_sampler_law._probabilities`。D側に後者がなければ同固定fileを併せて取り込むか、その純粋関数だけを固定SOURCEから再利用する。C branch全体のmerge、B/Dのrunner変更、実確率の評価は不要。取込み先でも人工testだけを実施する。

## Lawとmodelの制約

law調査`d72dc4e6228bb380f4d1d000e5e030794fce7164`のF0は未実行。324確率・13 order分布の新解析は今回も0。数学的source lawと一様格子の積law、固定seed PCG64は区別する。線形個数のfirst momentには適切な周辺lawがあればよいが、basis/相殺には順序やjoint lawが必要。

同じlawの線形modelなら係数とE[F]を線形結合できるが、E[model residual]・fit/移送誤差・共有係数不確かさは消えない。積・max・depth等に平均だけを代入しない。control variateにはDが同pathで特徴量を計数し、係数/検証cohortの扱いを事前に決める必要がある。

固定条件ではorder2も解析式/kernelに含められる。E[N_order2]が得られても、order2条件付き価格の平均・range/CIは未取得。未観測=0としない。直接paired残差法に特徴量が不要なら、今回の追加科学案を実施しない。

## 次の担当

**D**：必要なfeatureだけを上表の粒度で定義。残差研究・compile・fitはDが管理する。
**ユーザー**：必要が生じた場合にSCIENCE_PLANの有限scopeを判断。
**C**：許可後だけ対象law/featureの実平均を評価。現時点では実数値を返さず、公開照合後mandatory STOP。
**B**：今回の成果は方法/診断metadataとして参照し、価格・Gの実数値集計や資格昇格へ進まない。
