# 希少Taylor次数のcompiled費用保証：静的技術調査

今回SOURCE `3c513d1ad52ce99f0d2df7d8a300e4d58d15a553`。調査対象は固定H6 report `eee1343f6e99aa7661892ddfbe1067e974e2b74a`、H8 result `681ec23c93fdd1090d325908f4b155a79115afdf`、native science/source `c31ef424175c02379eafebe8fc6c7548c1d9aabe`のsampler/build/compile。Git blobとsource bytes/SHAは[索引](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/fixed_reference_index_v1.json)。Qiskit1.3.0公開source4fileのbytes/SHAは[別receipt](../../artifacts/resource_applicability/track_a_pre_review_technical_closure_v1/2026-10-15/qiskit_versioned_source_receipt_v1.json)。実probability/費用の新算術、event生成、sampling、compileは0。

## 結論と到達段階

**有限supportに対する粗いbasis-size上界を作る技術的見込みはある。ただし現在の保存SOURCEからcompiled RZ/CXの有用な母平均上界が閉じたとは言えない。** native instruction guard、最大観測費用、未観測確率の小ささ、実行時のcompiled size capは、その証明を代替しない。

|論点|現在確認できた性質|未完了条件・次の技術作業|
|---|---|---|
|finite support|finite RTEはk=0/2/4（K2は0/2）、eventはk個product＋1 rotation。各component supportとR=qrは有限|全componentのbasis operation/definition/arityをbyte固定し、scalar/ancilla/axis/PFを含むsymbolic expansion envelopeへ置換する|
|native count→basis count|native structural plannerは両basis change、Z/ZZ support、relative phaseを含むpre-build上界|operation1個がbasis gate1個とは限らない。XXPlusYY等のdefinition/control/parameter/phaseの最悪展開係数とtranslation pathが未証明|
|compiler保証|固定compilerはbasis rz/sx/x/cx、opt1、seed17、no backend/topology。whole wrapperをtranspileして計測する|全pass/plugin/高水準合成の固定、Qiskit1.3 Rust実装の適用条件、1q run長さとbasis-sizeの保証を監査する。**RZ数が部品のRZ和以下という保証はない**|
|catalog coverage|H6 prefix5の1092component/14tail fragment、basis62–78operations等の既報metadataは保存されている|全component・角度・event組合せをcompiled metricとして測定したcatalogではない。未測定componentのcompiled0補完は禁止|
|期待費用|確率π_layerと一様な費用U_layerが両方保証されればΣπUの上界を使える|現U_layer未取得。πが既知でも未知μ_layerを0としない。degreeだけではcomponent列/rotation angle/basis cancellationを固定しない|
|conditional rare測定|対象層のdescriptive平均を効率よく測る設計は可能|finite sampleで全support最大値を証明しない。whole-trajectory層に対する正しいconditional law、πとの混合、独立seedとsource/input bindingが必要|

## Compiler最適化の非加法性

[Qiskit1.3.0 level1](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/level1.py)はtranslation/optimizationのstage pluginを組み立てる。[同builtin_plugins](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/preset_passmanagers/builtin_plugins.py)ではlevel1にinverse cancellationとOptimize1qGatesDecompositionが入り、depth/sizeのfixed-point loopと必要時再translationを組む。これらは**隣り合う部品の境界を跨ぐ**。

[Optimize1qGatesDecomposition](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py)は1q runをresynthesizeし、global phaseを保持して置換する。Pythonの記述はerror/lengthを判断するが、実`run`はRust accelerateへ委譲する。今回はRustの全保証・plugin全closureを証明していない。total basis-sizeが減る条件が成立しても、RZとSX/Xの比率が変わるのでcomponent RZ和の単調性には直結しない。

DF basis/inverseの隣接reuse、CX/inverse cancellations、同じwireの1q resynthesis、ancilla Pとaxis Sdg/Hの境界はwhole-wrapper metricを変える。保存H6で高次数層の費用が低次数層より低かった例も既報にある。単純な「degree増加なら費用増加」「component実測和が安全なRZ上界」は使わない。

**技術候補**は、全部品を保守的な`rz/sx/x/cx`の総gate数上界へ展開し、no-routing/固定passの総size envelopeを証明し、その後`RZ≤size`、`CX≤size`、各depth≤sizeを使うもの。全trajectory列の列挙は不要で、R・Kとcatalog最大値からsymbolic worst caseを積算できる。しかし各pass/translation/controlの保証が必要で、上界が大き過ぎて観測gapに対する有用な保証を与えない可能性がある。compilerをopt0へ変えて証明を容易にする案は**新cost系列**になるので今回採用しない。

## 既報から引き継ぐ留保

H6 reportのπ0/π2/π4分解、R/degree/basis structural上界、rho感度は既済の証拠として参照し、今回再算術していない。H6のcompiled size cap5,000,000は「成功した実行」に条件付けたguard。全supportでcompile成功することや数学上の量子平均費用を保証しない。H8 cap10,000,000も同様。失敗を避けるSTOP運用を物理cost boundへ読み替えない。

H8確認のorder4観測0と、q4/q8 K4全event0という保存countsを確認した。探索の別q1/r1のorder4費用を選抜q2/q4/q8へ移植しない。cost CSVは有限観測のpaired metricsであり、全support certificateではない。`rare_order_observation`のunobserved mass/probabilityは保存metadataとして参照したが、新確率計算はしていない。

H6 q8/r2/K2 s2のconditional8 trajectory/16compileは**既に完了**（SOURCE `843265b624eac8fbb9acea47ef21cab1207e1880`、result `e0e6473d7047158a7f0f788d026a7a89a01898b8`）。既報local correction +.00734086%はそのcaseだけの診断で、K4 s4/H8/全母平均へ外挿しない。この既済pilotを今回の新実験として再実行しない。

## 科学レビューへ渡す選択肢

1. **留保付き経験的benchmarkを維持**：追加compile0。有限標本/未観測費用/選抜/状態準備P=0の限界を明記する。科学的に十分かはGPT判断。
2. **有限support symbolic上界を先に実装**：新実Hamiltonian計算0でも、operation inventoryとfixed compiler/Rust/pass closureの限定技術監査が必要。粗い上界の有用性はUNKNOWN。現段階REQUIRES_IMPLEMENTATION、保証未成立。
3. **新しい未観測層のconditional pilot**：1case/1層を固定し、correctness最小1 trajectory/2axes（SEなし）、descriptive候補8 trajectory/16axes。layer平均の精度や全support保証に十分な件数とは認定しない。H8 conditioning port/source/予算は新規準備が必要。REQUIRES_SCIENTIFIC_DECISION＋REQUIRES_IMPLEMENTATION＋REQUIRES_NEW_NUMERICAL_RUN。

未観測層の実compileは記述的な費用診断には役立つが、厳密certificateの論理上の必須条件とは限らない。全support enumerationは規模が急増しdefault候補にしない。追加のsampling/conditional event生成、H6既済算術の繰返し、新baseline/claim採用には進まずSTOP。
