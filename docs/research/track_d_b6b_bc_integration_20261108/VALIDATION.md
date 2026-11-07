# B6b：人工・回帰・会計検証

実行SOURCE `13114b2bd9db7dcf1ee7f027ab202f72786e2552`、component1e−10/observable Frobenius2e−10/rtol0/no phase fit。既存gadget、Gaussian emitter、actual inverse、H/Sdg/H wrapper、induced observable、reference-column checker、P0 phase compilerを再利用した。新Gaussian/solver/sampler/compiler/state simulatorは実装していない。

[raw検証](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/raw_theorem_checks.json)／[全16 postcheck](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/postchecks)／[原本hash索引](../../../artifacts/track_d_b6b_bc/2026-11-08/run01/payload_index.json)／[人工・回帰24件](../../../artifacts/track_d_b6b_bc/2026-11-08/technical/synthetic_tests.json)。

人工は2mode。H4 geometry/basis/CDF/lawには非該当。q1/event-cut相当、q2/step1、q4/step2は数理境界の代表例で、実H4のevent16 pathそのものを生成したものではない。四つ目はwhole Vのrayだけを保存する代数的な拡張例で、既存DF構成のeligibilityを緩めない。

|case|DF half-ray条件|χ|final reference非真空Frobenius|
|---|---|---|---|
|q1_event_cut|True|0.021999999999999992|0.0|
|q2_step_cut|True|0.006000000000000005|0.0|
|q4_half_cut|True|-0.034|0.0|
|V_ray_only_not_DF_eligible|False|-0.265|0.4266948566015646|

各caseでAB=V、actual A†、whole V真空ray、complex coherence、全4 system入力のX/Y誘導observableを比較した。DF適格3件は既存branch/zero-leakage checkerもPASS。拡張例では半A/半Bの真空leakageが各約.213347、最終reference非真空Frobenius約.426695でも、正しいwhole V/χにより全入力の信号は一致した。これは数学的必要条件を区別する例であり、既存molecular checkerを通過したと表示しない。

各caseのdirect-controlled比較とReference×XY、合計16 wrapperを同P0（Qiskit1.3,opt1,seed17,rz/sx/x/cx,approx1,initially_zero=false,topology-free/宣言位相保持）でcompileした。**全16 postcheck PASS**、full-wrapper/global-phase/coherenceの最大component差9.473989258092931e−15、X/Y observable最大Frobenius4.1595923209015305e−15。全行列を使い、phase fit/列削減/閾値緩和はない。

このdirect controlは一般unitaryの数学的比較器であり、「強いsymmetricの新価格baseline」ではない。既存StrongBuilder・directional/relative fusion/step境界/scalarのC5回帰8件とC3回帰4件は別にPASS。既存baselineと価格結果を弱体化・再実行していない。

|negative control|coherence target max差|検出|
|---|---|---|
|omit_chi|0.02199955633601752|True|
|wrong_chi_sign|0.04399645075255952|True|
|forward_A_instead_of_adjoint|0.7543208632577311|True|
|swap_AB_order|0.07893440544999865|True|
|V_does_not_preserve_reference_ray|0.10058013630071418|True|

χ欠落/逆符号、forward Aでactual A†を代用、AB順序逆、whole V真空ray非保存を全て検出した。最後の例でも一般式cVは成り立つがc≠1となり、無条件測定同値性への反例として保存した。期待したineligible controlを除外してPASSを作ったものではない。

会計では人工N・人工52価格recordだけを用い、全13roundのN×価格・paired差恒等式、共有round/方式共分散、missing/foreign scope/実H4 total未許可/peak加算拒否を試した。元B4-2 N×実H4価格は一度も呼んでいない。会計テストの値を分子結果として報告しない。

raw/compiled QPY計32とtarget NPZ4、計36人工原本はlocalのみ。公開はSOURCE/PLAN/check/scalar/hash。実H4 event/QPY/NPZを読取展開・公開していない。16独立taskを実CPU/task数に応じて並列化し、小5qubit行列のnested BLASのみ抑えた。内部CPU/RAM/cache/pathは非公開、共有設定/他job変更なし。
