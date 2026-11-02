# H4 server preparation bundle

2026-10-06 JSTに準備完了。2026-10-05 handoffのdirectory名を保持する。

入口は [SERVER_PREPARATION_REPORT.md](SERVER_PREPARATION_REPORT.md)。statusは
`SERVER_NATIVE_ENV_REQUIRES_SOURCE_PORT`、science execution未認可、geometry未固定、mandatory STOP。

今回の公開範囲は、このdirectory内の準備契約/schema/zero-compute草案、環境監査、
synthetic fixture/source・結果・構造化ログだけ。監査JSON内の `commit_push=0` は
準備完了時点の履歴であり、その後の利用者の明示依頼による専用branchへのcommit/non-force pushと区別する。
本計算・source portの認可は追加されない。

実行済み **128 transpile** の内訳：

| 用途 | 内訳 | actual transpile |
|---|---|---:|
| worker比較 | 小/中/大各10 task＝30 task × 1/6/12/16 workersの4条件 | 120 |
| 小型wrapperの軸・phase意味論検査 | phase 0/0.173 × cosine/sine | 4 |
| 別途事前定義したcompile前後の全operator照合 | phase 0/0.173 × cosine/sine | 4 |
| 合計 | 比較120件＋正しさ検査8件 | **128** |

残り8件も純synthetic検査。science wrapperは0件で、公開作業では追加transpileを実行しない。
taskごとの実行ログは `synthetic_fixture/benchmark_result_v0.json`、
検査ログは `synthetic_tests_v0.json` と `compiled_operator_checks_v0.json` に保存されている。

| file | 内容 |
|---|---|
| [environment_inventory_v0.json](environment_inventory_v0.json) | 既存CPU/Python/依存/BLAS/compiler情報と資源の観測 |
| [dependency_metadata_closure_v0.json](dependency_metadata_closure_v0.json) | CPU科学依存45件の推移的metadataとversion制約 |
| [static_audit_v0.json](static_audit_v0.json) | 指定保存JSON identity、218 template、source構文とport要件 |
| [zero_compute_plan_draft_v0.json](zero_compute_plan_draft_v0.json) | 新契約・環境・seed/cache/ledger・未固定条件の草案 |
| [zero_compute_plan_draft_schema_v0.json](zero_compute_plan_draft_schema_v0.json) | 未認可・null条件を保持するexact draft schema |
| [future_checkpoint_schema_draft_v0.json](future_checkpoint_schema_draft_v0.json) | future wrapper recordのschema草案 |
| [preparation_summary_v0.json](preparation_summary_v0.json) | worker比較と最終synthetic/science counters |
| [preparation_checks_v0.json](preparation_checks_v0.json) | 静的・契約検査21件の結果 |
| [synthetic_fixture/](synthetic_fixture/) | 科学データを使わないsource/config/task/resultとoperator照合 |
| [final_audit_v0.json](final_audit_v0.json) | 最終不変性・予算・変更scopeの検査 |
| [preparation_artifact_manifest_v0.json](preparation_artifact_manifest_v0.json) | この新規準備bundleだけのfile digest台帳 |

旧result/status/manifest/source/test/原稿/図を変更した科学resultではない。新規science module/runnerの実装は次段階。

再利用できるfixture sourceは [synthetic_fixture.py](synthetic_fixture/synthetic_fixture.py)、task fingerprintは
`2b3fa5ab1b67093be38ebb907f89830ef0f4dd096830a54521bcc189af2fc8d7`。
実行はrepository外の新しいprivate一時directoryを使い、各hostの既存CPU Pythonをabsolute pathで指定する。
共有venv・shell設定・Qiskit設定は変更しない。以下は後日の再現用で、今回ローカルhostの遠隔実行は行っていない。

1. 自分の一時directoryを新規作成し、fixture source/configだけをそこへコピーする。
2. process限定で `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 RAYON_NUM_THREADS=1`、`QISKIT_SETTINGS`はコピーしたprivate configのabsolute pathにする。
3. `<absolute-python> <fixture.py> freeze <new-temporary-directory>` でtask/source/予算を先に固定する。
4. 同じenvで `tests <new-temporary-directory>`、次に `run <new-temporary-directory>`。fixture全体は120 benchmark＋4意味論検査。
5. 必要なら同じenvで `<absolute-python> <verify_compiled_operator.py> <new-temporary-directory>` を一度実行し、全operator4件を加える。全合計128 transpile上限。既存resultがあれば自動retry/resumeしない。

実CPU数・共有load・RAMのguardで危険なworker条件は省略する。fixtureのworker scalingから実H4 ETAやhost間速度倍率を推定しない。
productionのlaunch commandはこのbundleに含めない。
