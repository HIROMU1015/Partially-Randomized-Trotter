# H8 s2 worker修正・並列化監査

[修正内容・検査・並列化の評価](../../../../docs/research/track_a_h8_conditional_s2_worker_fix_20261018.md)。新SOURCE `03479a7546a994bf9d02561d1ab74cec67201b4f`、科学未実行。pilot28＋既存pool4 PASS。旧STOPをPASSへ変更しない。

- SOURCE_closure / source_change_audit：旧新SOURCEのblob/hashと固定provider/runtime map。
- regression28 / existing_pool4 / test_receipt：限定人工検査。初回fixtureのFAILはinitial_fixture_failureにhashを保全。
- performance_audit：8 trajectory並列・paired回路再利用・streaming cacheの静的監査。実効speedupは未測定。
- preservation_audit：旧run・grant・lock・carryを保持。科学再実行0。
- plan_UNAPPROVED / grant_INVALID / config_UNSEALED / invalid_grant_rejection：新SOURCEへ結合した無効な草案とCLI拒否。
- publication_inventory：軽量bundleのbytes/SHA（自身を除く）。

private原本・NPZ・raw cache・使用済みgrant/lockは公開しない。新しい有効grantなし。mandatory STOP。
