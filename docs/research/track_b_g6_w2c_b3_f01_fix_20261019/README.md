# W2c-B3 F01：失敗provenanceの限定修正

2026-10-19 JST。状態：**`F01_FIXED_ARTIFICIAL_REGRESSION_PASS_INDEPENDENT_SOURCE_NOT_INTEGRATED`**。

**F01を独立SOURCEで修正し、新規26件＋既存36件＋補助48件、計110件の人工テストがPASSした。** FAIL/ERROR/SKIPは各0。実G7 correctness・sampling・CPU profiling、native build/compile/synthesis、新科学計算は0件。修正版をCodex Aへ統合・採択していない。

## 固定基点・修正内容

- 基点：[B3 driver固定版][OLD] `e4af6c064db805cbec611bcc8f0ff75a4754d38b`。
- 技術根拠：[F01原指摘][F01] `6576e606d4edcc6c53f095ee5bdf750defa99a42`。P1/G9監査を再実施していない。
- 原因：各trialの結果を初期化せず、次sample失敗時に前trialの`result.event`を失敗eventとして記録し得た。旧trace digestもstream受理より先に更新された。
- 修正：axis/trial/stageとresult取得状態を明示し、候補hash/counterをcopy上で構築。`emit()`正常return後に、完了prefixを一つの参照として公開する。streamの部分書込み・flush・file hash失敗とtrial完了を分離した。

変更宣言はdriverの`Stream`、`run_pair`、`execute`内の失敗ログ引継ぎ・cleanup保護に限定した。旧G7/ReturnKernel/F3/G9、補助SOURCE、既存36/48 tests、旧候補・SOURCE inventory・publication receipt・共通索引は変更していない。新規testsと人工正常系fixture、専用資料だけを追加した。

## 資料

- [原因・修正](f01_root_cause_and_fix.md)
- [失敗記録・保存境界](failure_provenance_contract.md)
- [人工回帰報告](artificial_regression_report.md) / [machine結果](artificial_test_results.json)
- [SOURCE差分・将来統合](source_diff_and_integration_handoff.md)
- [GPT・ユーザー向けハンドオフ](handoff.md)
- [SOURCE inventory](source_inventory.json) / [未採択binding候補](source_binding_candidate.json)
- [証拠索引](evidence_index.json) / [静的・保全検査](verification.json) / [公開照合](publication_receipt.json)

正常系は**修正前に生成した人工8ケース**で、4方式×A/Bのevent trace・bit消費・counter・mock JSONL bytes/chain SHA・粗stage記録が修正後と一致した。これはfake generator/session/clockだけによる互換性確認で、実G7 lawや実CPU速度を数値検証した主張ではない。SOURCEのsampling/angle path、固定定数・call予算の不変性は静的にも確認した。

## 保存保証と未完了範囲

`emit`正常return、Streamの書込み済みprefix、trial完了、ディスク再読・電源断耐性は別。新記録はreadback/fsyncを未検証と明示する。部分書込み末尾やwrite/flush失敗後のinflight recordを成功trialへ数えない。完全な永続保存不能・OS kill/OOM等では保存できたprefixだけが証拠となる。

正式採択・science実行承認・外部atomic admissionは未実施。旧候補のdriver SHAは旧版専用であり、そのまま修正版を起動できるものではない。新binding候補は未採択・実行権限なしである。

専用branch `track-b-g6-w2c-b3-f01-failure-provenance-fix-20261019`、独立checkoutで作業した。Codex Aのlive worktree/index/lock/process/未commit成果は参照・変更していない。親worktreeと旧SOURCE/raw/契約/consumed ledger/claim/STOPを保全した。共通指示2026-10-17の同一bytes・§8.4を確認し、CPU32・available RAM約49 GiB・/tmp空き約371 GiBで軽量な人工検査を独立実行した。共有設定・venv/cache・他processは変更していない。

**0件**：実G7 science import/実行・保存科学値再解析・実費profiling、native取得、新科学条件/研究方針採択、実authorization/grant/marker/claim発行・消費・再利用。人工FakePortは本番authorityではない。

公開・保全確認後は**mandatory STOP**。次はユーザーがCodex Aの公開成果との統合を別途指示するか判断する。統合・正式契約・angle/seed/schedule採択・科学実行へ自動移行しない。

<!-- FIXED_REFERENCES -->

[F01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6576e606d4edcc6c53f095ee5bdf750defa99a42/docs/research/track_b_g7_g9_p1_final_driver_delta_20261019/technical_findings.json
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py
