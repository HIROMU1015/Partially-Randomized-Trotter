# SOURCE・metadata生成とテストの再現

2026-11-13 JST。D基点`0c6e16342b3c804fa0bd89a4028d6b5bea7b0322`、既存branch `codex/d-compiled-residual-c-minus-s-20261113`を継続する。全科学runは既存STOPのまま。

| SOURCE | 読解・再利用 |
|---|---|
| A `a9abf8de75bf3432fcd714e868e5c14c8884b8f8` | `docs/research/pr_rpe_first_signal_menu_20261113/MENU.json`, `CONFIGURATION_LD1.json`, `CONFIGURATION_LD2.json`, `SCIENCE_PLAN.md`, `HANDOFF.md`, `PREPARE_MENU.py`。最後の旧生成scriptは読解のみで実行していない |
| D `0c6e16342b3c804fa0bd89a4028d6b5bea7b0322` | candidate_price_adapter.from_b/check_request、pending_price_records。NativeRuntime/context・sample・acquireは実H4に呼んでいない。expected_scalarの任意欄とmock照合だけを追加 |
| D `24b0a6eb1d806391a3c6ad454fb0c018148d8545` | 価格探索HANDOFF：partial未知・compiled mean/区間の資格、最終Gの精度・候補選別の前提 |
| B `437ba4acdb82708b6fc98e3cd119a2a734f7cf64` | rpe_shot_evaluation_design HANDOFF、canonical controller/fixed-grid integration。D保存版とのfile byte一致を確認 |
| B `b9b9ffa6233db9dd2b413dc07b27e7039324c666` | rpe_shot_allocation HANDOFF、_price_metadataのprice_extension/price_bindings/interval_contract検査。N/G/decoder数値入口は呼ばない |
| P2 `39ddd59edf54a3a901e5302207b2765ddbd03fbb`、C `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | 既存HANDOFFの参照だけをcoverageに記録。保存C/S/Dや全26 E[S]の数値を読んで新集計しない |

指定B2版の`pr_rpe_rmse_controller.py`、`pr_rpe_rmse_integration.py`はともにD基点版とbyte同一である。既存B._validateがsource-bound A fixed-grid/common exactκからcanonical configurationをmetadata再現し、既存D.from_bがphysical tuple/structure/profileを照合する。Dで候補生成・q/R最適化・N評価規則を再実装しない。

## 出力生成

追加library：[ld12_price_requests.py](../../../src/trottertracks/ld12_price_requests.py)。
追加metadata publisher：[PREPARE_PRICE_REQUESTS.py](PREPARE_PRICE_REQUESTS.py)。

```bash
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python docs/research/d_ld12_price_requests_20261113/PREPARE_PRICE_REQUESTS.py
```

26個のcomplete D requestをmetadataとして生成し、[PRICE_REQUEST_INDEX.json](PRICE_REQUEST_INDEX.json)へpinned configuration selector/ID/keyだけを保存する。52軸の[PRICE_REQUEST_COVERAGE.json](PRICE_REQUEST_COVERAGE.json)に対応・不足・未選択を保存する。巨大event列は作らない。出力に科学値・実preparation hash・cache hit・実価格を補完しない。

技術用execution_sourceはD基点と当該2fileのGit blobを保持する。これは新科学run SOURCE/承認証明書ではない。出版後に異なるmetadataでファイルを上書きせず、科学取得は選択scopeの新SOURCEでrequest_forを呼び直す。

## テスト

新規：[test_ld12_price_requests.py](../../../tests/test_ld12_price_requests.py)。NativeRuntime.context/sample、Adapter.acquireの通常入口を禁止した12項目。scalar検査だけはmock H/preparation descriptorを注入し、実行内容は条件比較と空のhead metadata構築に限る。NumPy/Qiskit/PySCFや実H4入力を使わない。

全52軸のsource/identity/law/physical ID、q/R/pieceとbinary64 time、prefix別λ/scalar/fragment予定、単軸選択、SOURCE/keyの分離、LD取り違え、同event数にした別R候補の拒否、lowering/law mutation、passive importを確認した。Cのpreparation/kernel統合検査を重複実行しない。

```bash
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/test_ld12_price_requests.py
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/test_p2_direct_pilot.py tests/check_pr_rpe_rmse_controller.py tests/test_candidate_price_adapter.py::test_adapter_import_is_passive_and_has_no_scientific_orchestration tests/test_candidate_price_adapter.py::test_saved_LD1_B_configuration_is_accepted_as_metadata_only
```

[TESTS.txt](TESTS.txt)：12 PASS。[REGRESSION.txt](REGRESSION.txt)：26 PASS。合計38項目。
regressionのN/C/G算術は既存人工fixtureだけで、実N/保存価格や科学結果を新しく評価していない。今回の確認は作者自身の技術検証であり、独立科学レビューではない。

新科学call0、実H4preparation/sampling/build/compile0、保存科学値再解析/fit/E[C]/G0。公開後mandatory STOP。
