# P1採用評価の技術テスト

## 今回実行したもの

今回追加 **19 PASS / 0 FAIL / 0 SKIP**。Python3.11.1 / pytest8.3.5 / NumPy1.26.4の既存task専用venvを読取再利用した。
新規ライブラリ導入/共有環境変更なし。実科学module不足をfake実装で補完してPASSにしていない。
mock providerを用いるケースは人工fixtureとして明記し、実sourceのprobeは独立processで行った。

|区分|ケース数|実施と限界|
|---|---:|---|
|静的API/固定diff|2|413名/順序/target一致、既存数値module/runner/科学artifactの本体差分0|
|人工providerでのfacade比較|8|全413取得identity、star、初期化、reload、未知名、欠損、順序、再入反例。数値providerは実行なし|
|実sourceの隔離probe|7|旧import失敗2環境、P1 no-site、rte代表identity、star不足、失敗値cacheなし、binder不足拒否|
|旧facadeを選んだ一時fixture|2|plan/no-output mock成立、optional outputは依存不足を検出しdirectory未作成|

実identityは`CompilerSettings`（型）、`PROBABILITY_ATOL`（定数）、`make_rte_config`（関数object）を既存rte属性と比較しただけ。
関数を呼んでfinite lawを作ったり、科学的な値を評価していない。
旧rootを使うprobeは固定base initializerとP1内の**変更のない**core submoduleを一時loaderで組み合わせた評価fixtureで、研究codeの移植やruntime推奨実装ではない。

### 確認した差

- 旧eager import：siteなしnumpy不足、task venvでmatplotlib不足。P1 root import：成功しoptional moduleなし。
- P1 star import：`__all__`の順に取得するためopenfermion不足。旧ImportFrom文の順と`__all__`順は同一ではない。
- 取得cache/vars/dirとreload再束縛に差。missing providerの例外は抑止されず、失敗exportはcacheされない。
- 原facadeのままP1 plan/mockは成立。optional保存の旧executor importはmatplotlib不足で停止し、保存先を作らない。

初回評価harnessの3つの想定（存在しないsubmodule fixtureの生成、公開されていない関数名、star importの最初の欠損module）を修正して最終19件を再実行した。
P1本体を修正して期待に合わせたものではない。初回ログ/最終JUnit/fixture sourceは非公開詳細領域に別名保全した。

### 最終19 case名

- `test_all_413_names_order_and_binding_targets_equal`
- `test_numerical_providers_and_old_runners_are_byte_unchanged`
- `test_fake_all_exports_identity_with_attribute_and_from_import`
- `test_fake_star_import_resolves_all_exports_but_loads_all_providers`
- `test_fake_root_import_defers_provider_side_effects`
- `test_fake_cached_value_does_not_track_monkeypatch_and_reload_differs`
- `test_fake_unknown_name_attribute_from_and_hasattr_semantics`
- `test_fake_missing_optional_dependency_exception_moves_to_use`
- `test_fake_import_order_direct_provider_then_root_export`
- `test_fake_reentrant_provider_is_not_made_cycle_safe`
- `test_real_base_import_dependency_failure_is_observed_not_full_pass[False]`
- `test_real_base_import_dependency_failure_is_observed_not_full_pass[True]`
- `test_real_p1_import_with_no_site_has_no_optional_modules`
- `test_real_p1_pure_type_constant_function_identity_and_cache`
- `test_real_p1_star_import_still_requires_optional_stack`
- `test_real_p1_missing_provider_does_not_cache_failed_export`
- `test_real_core_binder_refuses_missing_dependency_without_calls`
- `test_fixture_original_eager_facade_works_for_plan_and_mock_without_output`
- `test_fixture_original_facade_output_dependency_fails_before_directory`

## 静的補足（test件数に加算しない）

47 provider、48旧ImportFrom、65局所core moduleのAST比較を実施。
root export/研究packageへの静的逆importは検出なし。条件付・関数内を含めたgraphのSCCはdf_rte_circuit/df_rte_tailだけで、後者の逆向き参照はTYPE_CHECKING/関数内。
CUDA preload/NUMBA threading/config環境参照の初期化処理はsourceで確認し実行しなかった。
P1変更21 pathすべてを実source/固定差分/資料で判定した。全branch再監査はしていない。
P2主要10 pathの旧summary/最新handoff branchの同blobはmetadata比較でありtest PASS数へ含めない。

## 固定P1の62 PASSとの区別

[固定P1のTEST_MANIFEST](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af54aeb7ebac042a3ca02379d61516ac635feef9/docs/integration/p1_df_pipeline_h4_20261027/TEST_MANIFEST.json) の49新規 + 既存整数6 + executor人工7 = 62 PASSは**過去の別code版の記録**として再利用した。
今回62件は再実行せず、今回19件と足して「81件の統合回帰成功」とは表示しない。
将来の選択統合ではno-site root/既存public export/atomic保存の3ケースを旧facade前提へ再構成する。
残る46ケースの科学境界・配線の論点を維持し、新hook/policy/error差分だけを追加確認する。
既存13件を実施するにはその版の依存環境が必要であり、今回の19件から代表branchでPASS済みとは認定しない。

## 未実施・未確認

実optional stack全体のimport成功、413実providerのidentity、旧研究runnerの実起動、CUDA/NUMBA副作用順序の実比較、
全研究consumer regression、real H4 input/partition/operator/state/control/phase同等性、N/G/cost、sampling、実Qiskit build/compile、性能評価は未実施。
これらは19件のSKIPではなく評価範囲外/依存不足で未確認の項目。
科学実行0、数値同等性未検証。P2実装0、B3アクセス0、A/B接続・転送0。
