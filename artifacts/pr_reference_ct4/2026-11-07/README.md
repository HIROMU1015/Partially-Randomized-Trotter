# CT-4BのSOURCE・入力・結果

2026-11-07 JST、SERVER 2。基準commit `69e2c0ae4e79febe54df59b6ea7704238467a968`。[REPORT](../../../docs/research/pr_reference_ct4b_20261107/REPORT.md)、[SCOPE](../../../docs/research/pr_reference_ct4b_20261107/SCOPE.md)。H4/1Å/STO-3G/DF12・Partial S2/LD3/h=.02/K2、保存q128/r1のrare/nonrare各j0、B′/Reference×X/Y、一様strict epsilon1e-11だけ。元H ground供給・準備費除外。全support/全18round総Tは未資格/未取得。

- [FIXED_INPUTS](FIXED_INPUTS.json)：native準備前の2保存event配列・seed/hash・8構成・精度・local選択規則。
- [INPUTS](INPUTS.json)、[ENVIRONMENT](ENVIRONMENT.json)、[SOURCE](SOURCE.json)：科学入力・実行環境・SOURCE。新しい承認証明書ではない。
- [PREDECLARED](PREDECLARED.json)：native8構成の依存key299件、再利用111/新規188、物理signed多重度、25 local対象、CT-0 m7 endpoint。新規合成前に固定。
- [NATIVE_INPUT_CHECKS](NATIVE_INPUT_CHECKS.json)：8 compileの保存metric一致、reset/測定、global phase、native file hash、literal signed histogram。
- [KEY_REUSE_AND_SYNTHESIS](KEY_REUSE_AND_SYNTHESIS.json)、[WORD_CORRECTNESS](WORD_CORRECTNESS.json)：実word/spec/hash/guard/T/T†/Wと、全299件のstrict/actual adjoint/±emitter検証。
- [LOCAL_TARGETS](LOCAL_TARGETS.json)、[LOCAL_CORRECTNESS](LOCAL_CORRECTNESS.json)、[TECHNICAL_TESTS](TECHNICAL_TESTS.json)：局所全空間・人工検証。whole state simulationなし。
- [PILOT_T_COUNTS](PILOT_T_COUNTS.json)、[COMPARISONS](COMPARISONS.json)、[ROLE_T_COUNTS](ROLE_T_COUNTS.json)：有限8行の直接T価格と比較、literal histogram一致のrole会計。期待値/最小Tではない。
- [ERROR_AND_SHOTS](ERROR_AND_SHOTS.json)：finite8＋全support一様精度条件式8の16行。保存199 shot/axis条件を維持し、全lawのCT成功保証は未資格。
- [BUNDLES](BUNDLES.json)：native_qpy.tar.xz（8 QPYとmetadata）、strict_records.tar.xz（reused111＋synthesis188のrecord）のbyte hash/size。
- PREPARE_SOURCE.py、SYNTHESIS_SOURCE.py、VALIDATE_SOURCE.py、FINISH_SOURCE.pyと対応log：各段階の実行SOURCEと出力。元SOURCEは変更せず、新adapterの段階差分も保全。
- [STOP](STOP.json)：承認scope完了時のmandatory STOP。科学runnerの再実行や追加探索を許可するものではない。

大きい作業用QPY・recordはarchiveに収録し、common.pklは保存元common.pkl.xzとINPUTSのhashを参照する。旧SOURCE/raw/result/grant/ledger/marker/STOPはread-onlyで保全。再開・追加実行はユーザーの次の判断と別途承認を要する。
