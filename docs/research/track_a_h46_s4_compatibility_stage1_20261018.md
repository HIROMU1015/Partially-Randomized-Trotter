# S4 counter/native互換性：Stage 1完了

COMPLETE_MANDATORY_STOP。SOURCE `f33d917f10c5f92f63b23776387cbd16e85abfd1`。

`pf_schedule`だけを変更し、負係数をnativeと同じ`-a/(2-a)`、stage時間を`T*weight/q/2`へ揃えた。native/producer/zero規則/lowering/scalar/wrapper/compilerは変更しない。

追加9回帰＋既存counter・比較adapterの人工検証、合計68 tests PASS。S4のq1/3/8、正負・signed zero時間、block/mode順、native coefficient/stage hex一致、S2不変、元人工fixtureのRZ/CX差0、strict-zero/support/phase、scalar/wrapper、無効入力拒否を確認した。AST検査で`pf_schedule`以外のcounter定義が不変であることも確認した。

[patch・全テスト・hex・closure・消費済みgrant projection](../../artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage1/)。原grantはprivate保持。native exit0/wait・final source/environment gate PASS。実科学0、compile/trajectory0、retry0。既存SOURCE/結果/STOPは不変。技術自己検査であり独立別担当reviewではない。

Stage 1 STOPを記録済み。今回の利用者の3-stage事前承認とStage 1 PASSに基づき、Codexが別SOURCE/grantでStage 2の固定2構成・8整数行受入へ進む。
