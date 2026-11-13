# SOURCE・再利用と人工/回帰検証

2026-11-13 JST。基点`24b0a6eb1d806391a3c6ad454fb0c018148d8545`。
既存branch `codex/d-compiled-residual-c-minus-s-20261113`を再利用。関連する指定SOURCEを読解し、他担当branchのmergeやsignal/N/controllerの複製はしない。

| 固定SOURCE | 再利用・確認した内容 |
|---|---|
| D `24b0a6eb1d806391a3c6ad454fb0c018148d8545` | `docs/research/d_rpe_price_search_design_20261113/REPORT.md`, `HANDOFF.md`：candidate contextと価格quantity/未知区間・whole scheduleの仕様 |
| D `5a67cea7705e80ff295306606611d774724ae2e2` | `src/trottertracks/compiled_price_fast.py`：RZ projection/structural_pair、make_backend_class、cache/fingerprint/元compose/wrapper/compiler委譲。新HEADではaxes指定のみ小拡張 |
| D `952ce0113a27b179e4e6b889db124bdf37742fa1` | outer-window資料：長q tail未証明を継承、科学modelとして採用しない |
| A `821c5c1ffd7c95e1690d45d2b6e8ff4ccb15bd1c` | `pr_rpe_signal_plan.py`と探索HANDOFF。実際のmetadata job生成APIを人工fixtureで呼び、from_a_jobと照合 |
| C `37cbc331b6cb9d38e83584f8e5b007da173bc5cd` | 構造特徴量HANDOFF/実装：K2/nonidentity条件・未取得の実feature・law/modelの制約。新feature数値を評価していない |
| B `cdf6a80671b2d0bd58983fbbbaf7334f58dc8422` | 価格検証HANDOFF：identity/law/source・compiled mean/区間資格、最小科学scopeの条件。現V2 assess/_validateを再利用 |
| C1 `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` | p1_efficient_zz.load_c1_backend、Backend.price_requestsとfunctions loader。固定Backend.contextは起動せず、fresh owned Backendへ正しい候補contextを注入 |
| carrier `155529e20a8fac39e00786c40b57830b71203c4f` | gaussian_operations/prepare、weights/allocation/schedule、canonical_tau、trial_seed/sample_requests/composeの元関数bodyをC1と同じ方法で読込む |
| provider `c31ef424175c02379eafebe8fc6c7548c1d9aabe` | 既存prepare/_prepare、DFPartialS2StepRequest、RTE builder/native emitter/full wrapper、CompilerSettings/transpile_and_measure_cost。使用moduleのbytesを実行時に照合 |
| D新lowering・counter・hashの基点保存版 | efficient_controlled_zz、h4_df10_price_v1.Accounting、track_d_native_resources、h46_b3_fast_fingerprint/use_fast_fingerprint、h6_hash_reuse_v1。元展開/加算/representation/canonical bytesを再実装しない |

疎checkoutにない指定SOURCEはgit showで読んだ。保存B configurationはmetadata互換だけに用い、科学価格raw/結果の新しい集計・fitはしていない。元SOURCE/input/raw/結果/KEEP/grant/ledger/marker/STOPを保全する。

## 人工testの内容

追加：[test_candidate_price_adapter.py](../../../tests/test_candidate_price_adapter.py)。20 case（formula/K、軸のparameter展開を含む）。

- 分子ではない8-wire任意行列を作り、pinned prepareでL_D1/2のhead/tail/preparation/λ/scalarを比較。
- S2/m5、K=0/2/4/6、q/R、signed piece/最高order eventを明示構築。実input np.load/normal samplerを禁止。
- 実native compose/full measured wrapper、元Accounting S、canonical hashを使う。新adapter testのcompilerはmockで、返す6値はテスト用のraw circuit指標。
- 固定L_D1/S2/δ=.1では旧C1 contextのhead構築bodyをそのまま別に実行し、S・wrapper fingerprint・6 mock指標を旧price_requestsと照合。
- X/Y両軸・単軸・逆順、ancilla scalar一回、効率的ZZの有効化/例外時復元、S/C quantityとB平均nullを確認。
- context/input SOURCE変更・wrong preparation/K/grouping・mutated request・foreign compiler/loweringを拒否。同event数・別L_Dのcache/contextを誤流用しない。
- Aの実metadata helper、保存B configurationのmetadata互換、passive import（NumPy/Qiskit/pyscf非import）を確認。

既存fast10件は人工回路の実Qiskit compile同値性・mutation/cache/compiler/loweringを含む。既存P2 metadata6件と残差13件も再利用する。新実H4 compile/price mean/fit/Gの検証ではない。

## 再現コマンド

worktree root、既存の専用Python環境で実行。残差suiteはpassive-import検査のため別processにする。

```bash
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/test_candidate_price_adapter.py tests/test_compiled_price_fast.py tests/test_p2_direct_pilot.py
PYTHONPATH=src:tests:scripts /home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -m pytest -q --disable-warnings tests/check_compiled_residual.py
```

[TESTS.txt](TESTS.txt)の36 PASS＋[RESIDUAL_REGRESSION.txt](RESIDUAL_REGRESSION.txt)の13 PASS＝**49項目**。
Qiskit1.3.0/NumPy1.26.4を維持。既存Qiskit deprecation warningはdisplayだけ抑止した。
初回fixture/API接続の修正経緯は[DEVELOPMENT_CHECKS.md](DEVELOPMENT_CHECKS.md)。作者自身の技術確認であり独立科学レビューではない。

この成果の技術的APIは、run SOURCE/finite scopeが固定された将来科学取得に利用できる。今回は実H4のruntime.context/sample/acquireを呼ばず、公開後mandatory STOP。
