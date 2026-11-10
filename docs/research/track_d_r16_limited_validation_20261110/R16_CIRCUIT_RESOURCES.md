# r16 q4：保存資源・測定同値性

Linear H4／1.00Å／STO-3G／Nα=Nβ2、元H exact ground仮定・準備費除外、number-CDF8／元順LD2／ordered partial S2/s1/r16/K0／h=.1。sample `r16_q4_sample00`、4×16 event、whole-path一gadget、cut=complete step2、C1追加融合OFF。

## 固定入力とSOURCE

S2 SOURCE `08f15308ef16429da7e375b1f82e7799510d2578`。seedは [seed_plan.json](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/seed_plan.json) でS1実行前から固定。新しいsamplingは4 occurrenceのみで、event SHA-256 `8abec40ac644c42bddb29c79733aa732cedcbeafa51d89c8002f3b1f4281d6ca`。sampling後、CAR参照・回路生成の前にevent原本とhashを保存した。旧r32 pathの切断・再解釈はしていない。

既存C3 `prepare_steps`、StrongBuilder、C6aのCAR reference helper、B2 `trajectory_reference`、raw/compiled checkerを再利用。r32固定の`prepare_common`は既存のbindingを保持し、末尾のhead/provider構築を`context_for_arm`へ抽出した。shared whole-pathの表示上rはconfig由来にした。新しい回路アルゴリズムは実装していない。

Strongはuncontrolled Gaussian、directional中心、head relative、tail同basis共有、3箇所の合法step境界融合、scalar/identity集約を保持。Referenceは同じ最適化を各A/B内に適用し、内部2箇所を融合。global cutで1箇所の融合が失われる。CSWAPは8組×2 cascade、χ補正、追加8qubit、X/Y readoutをfull-wrapperへ含めた。

## Correctness

raw全22チェックとpaired X/YはPASS。最大component差=2.70894e-14、最大Frobenius差=3.341e-13。AB=V、A actual adjoint、真空ray、scalar/identity、signed角、CAR ordered pathとのdirect対応、全256入力のcomplex coherence・X/Y observableを確認。raw reference leakageは0。

χ=-0.93997799587438358 rad。per-step scalar/identity位相を4回分だけ集約し、χをfitしていない。normalization/attenuationはunitaryへ掛けず、信号会計で別に扱う。

P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、initially_zero=False、topology-free、既存SymEngine MPFR160宣言global-phase保持。raw PASS後にcompileは4回だけ実行し、postcheck4件・最終paired X/Y/χは全PASS。component abs1e−10、observable Frobenius2e−10、rtol0、phase fitなしを維持。

最終paired observable Frobenius差はX=1.228938141332957e−11、Y=5.476837344272538e−12。χの測定照合差はStrong=9.73267375312364e−15、Reference=8.312352556689882e−13。compiled Reference leakageはcomponent最大約2.288e−13、L2最大約2.515e−13。追加閾値緩和・科学retryなし。

## 全7資源：4 wrapper

|方式|軸|RZ count|CX count|RZ depth|CX depth|Total depth|Circuit size|Logical qubits|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|symmetric_directional|X|13,042|5,434|3,114|2,752|7,495|26,861|9|
|symmetric_directional|Y|13,042|5,434|3,114|2,752|7,495|26,861|9|
|reference|X|13,333|4,874|1,594|1,072|3,568|26,784|17|
|reference|Y|13,333|4,874|1,594|1,072|3,568|26,784|17|

同一path内のReference−Strong差（X/Y共通）：

|資源|差|Strong比|
|---|---:|---:|
|rz_count|+291|+2.231%|
|cx_count|-560|-10.305%|
|rz_depth|-1,520|-48.812%|
|cx_depth|-1,680|-61.047%|
|total_depth|-3,927|-52.395%|
|circuit_size|-77|-0.287%|
|qubit_count|+8|+88.889%|

CXは560減（10.305%）、total depthは3927減（52.395%）。RZは291増（2.231%）、qubitは8増。RZをT countへ換算せず、topology-free depthを実機時間と同一視しない。機械可読`resource_comparison.json`のglobal_phase欄は追加診断であり、異なる幅・方式のglobal phase差を資源削減やχ誤差とは解釈しない。

## 保存r32 q4との参考比較

旧C6a SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`、`q4_sample00`、同q4/cut2/C1 OFF。旧compileは再実行していない。

|r|方式|RZ count|CX count|RZ depth|CX depth|Total depth|Circuit size|Qubits|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|32|symmetric_directional|21,020|8,024|5,013|3,874|11,671|42,839|9|
|32|reference|21,239|7,336|2,571|1,590|5,686|42,562|17|
|16|symmetric_directional|13,042|5,434|3,114|2,752|7,495|26,861|9|
|16|reference|13,333|4,874|1,594|1,072|3,568|26,784|17|

**別path間の参考値。** event列・signed係数列・basis runも異なるため、差をrだけの因果効果やlaw母平均の差と認定しない。X/Yは同じ1 pathを共有し、2独立標本と数えない。RPE全round期待価格・総費用の保証は得られていない。

S2 wall timeは272.09秒。4独立compile/postcheck process、実空きCPU観測によりraw state-action 59 threads、各Reference postcheck28 threadsを割り当てた。既存のbatched Aer state-actionを利用し、共有設定・他ユーザー・実行中の別runには変更を加えていない。

ordered events、raw/compiled QPY、参照/observable NPZ、詳細logはlocal専用。公開は[hash索引](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/payload_index.json)、SOURCE、条件要約、correctness・価格のみ。
