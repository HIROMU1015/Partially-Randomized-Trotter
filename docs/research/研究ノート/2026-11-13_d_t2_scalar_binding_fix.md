# D：tail-only対totalのrequest修正

2026-11-13 JST。[REPORT](../d_t2_scalar_binding_fix_20261113/REPORT.md)／[C HANDOFF](../d_t2_scalar_binding_fix_20261113/HANDOFF.md)／[T2条件](../d_t2_scalar_binding_fix_20261113/T2_RERUN_REQUIREMENTS.md)。

ユーザーが許可した最小技術修正として、Dのld12_price_requestsだけを変更。固定T2 input locatorから保存receipt/snapshotを照合し、constant member一要素だけを読む。元A tail identityとc_H+s_Rのbinary64元順加算でexpected totalを生成し、3 quantityとSOURCEを既存input_source/keyへ保持する。診断値をコードへhard-codeしない。

29人工・metadata項目PASS。元Runtime/guard/compiler/lowering/full wrapperとscalar一回は不変、C380のpinned D748 constructor/既存expected_scalar転送とmetadata互換。C branchを変更していない。選択したLD2/m12 metadataは診断totalとhex一致するが、実context成功や巨大compile性能を認定していない。

新H4 prepare/sampling/event/path/build/compile・新価格/mean/CI/fit/N/RMSE/G・T2再実行0。旧request/coverage/T2/診断rawとSOURCE/input/cache/grant/ledger/marker/KEEP/STOPを保全。次の有限T2はユーザーが別途判断する。公開照合後mandatory STOP。
