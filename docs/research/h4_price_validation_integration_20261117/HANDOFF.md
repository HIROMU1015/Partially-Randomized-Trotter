# 通常入口・C同期への引渡し

[REPORT](REPORT.md)の差分を使用する。起動は従来と同じ：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts python3 -B scripts/run_rpe_schedule_search.py \
  --problem /absolute/PROBLEM.json --round /absolute/ROUND.json --run-dir /absolute/OWNED_RUN
```

method/κ-neighbourhood/local0も、共通`make_executor`を経由する。稼働中の固定SOURCEを途中で上書きせず、新しい実行SOURCEとして使用する。既存科学runのSTOP・予算・scopeを変更するものではない。

## 設定・入力

保存独立実測の表示は追加設定不要。既存PriceRules/actual-step registry、PFの価格SOURCEとモデルSOURCEを使用する。直接単価だけなら検証要求は作らない。

新規compileには、そのrunの既存科学実行scopeと`executor.distributed.preparation.budgets.compile`、適合するcompile host/profileを使用する。Q3予算だけからcompile許可を推定しない。`price_validation.compile_approved=false`で今回の取得だけを明示的に保留できる。許可の証明書・新grant・承認原文保存は不要。

`price_validation.store`は省略時に共通Git領域の`rpe-price-validations.sqlite3`。同じ中央controller/共有repositoryの全runから再利用する。独立cloneのcontrollerを併用する場合は同じ共有namespaceへ接続し、各cloneに別々の独立storeを作って二重投入しない。既存別系列の待機job索引は`existing_job_sources`、既存公開metadataは`published_validation_sources`の固定locatorも利用できる。省略時はCの通常公開INDEXを確認する。

Pauli/DFは既存固定normal-path workerの自動組立てを持つ。providerの配置を変える場合は`price_validation.providers.PR_PAULI/PR_DF`に`kind,backend,legacy_view,interpreter`を指定する。これは実装配置の設定でありLD別許可リストではない。Pauli backendは`BUILTIN_PAULI_NORMAL_PATH`、DFは`BUILTIN_DF_NORMAL_PATH`。元configuration、dictionary、law、元packet/数値SOURCE、calibration seed出所を照合する。

任意の既存PF/DF取得CLIは `scripts/run_rpe_price_validation_job.py` のsource-bound command形式でも接続できる。原provider・prepared PLAN・観測JSON selectorを使用する。PF用 `run_pf_preparation_stage.py` には未確認の`MODULE`参照問題があるため、fresh PFを動作確認済みとして扱わず、次の配置確認で必要な小修正・人工検査を行う。今回は実compileを行っていない。

## 出力・同じrunの再開

- `PRICE_MODEL_VALIDATIONS.json`：モデル一意のidentity/status/凍結条件/実測点と、configuration別の適用q/R・26軸単価を持つ公開sidecar。
- 状態：`CALIBRATION_ONLY`（較正のみ）、`PENDING`（検証待ち）、`INDEPENDENT_MEASURED`（独立実測あり）、`INCOMPLETE`（検証未完了）。取得の有無を示し、残差許容差の合否や科学資格ではない。
- 未検証モデルの `price_validation/<model_id>/INPUTS.json` を実測前に固定。PR独立経路数は2、X/Yの4wrapper。
- `ModelStore`に固定requestとjob IDを先に保存。開始・所有者不明をlease切れで再試行しない。同じ通常command・run-dirで既存attemptを照合する。
- 取得不能・未対応・数値/保存異常は検証側の理由として残す。既に適用可能な価格の費用算術・Q3・最安を変更しない。

## 公開と探索マップ

[保存LD2のHTML表示例](index.html)は同じ通常parser・rendererを使用する。保存配分2点、共有モデル1件、issues=0。HTMLを保存してブラウザで開ける。

method/schedule入口はcontroller結果保存時に`SEARCH_PUBLICATION.json`を作成し、sidecarを既存公開索引へ含める。従来と同じproducerのcommit/pushで公開する。Cは更新した既存`sync_saved_rpe_publications.py`→`build_h4_search_map.py`で読む。sidecarだけ更新された場合も検証viewを更新し、元Q3 SOURCEは保全する。

INDEXの`price_validation_models`にモデルを1回だけ置き、配分点は`price_validations`で参照する。詳細は回路の条件・実測m/q/R・適用先m/q/R・事前予測・経路別X/Y・平均・`(予測−実測平均)/実測平均`・元receiptリンク。検証roundの寄与率はその配分と保存26軸単価から求め、欠測ならnull。1点の差を全roundへ適用しない。

C自身の稼働branchへ無断変更・cron再設定は行っていない。取り込むのは本branchのconsumerコード差分であり、各hostから新たに表を作って渡す運用ではない。

## ロードマップへの影響

通常探索の条件付き価格に、独立実測の有無と元根拠を表示する技術経路を追加。モデル共有・待機・再開・通常C同期は人工/保存資料で検収した。科学的な価格資格、実Q3、十分N、正式Gは進めていない。新規科学call・詳細検証の要否は利用者がマップを見て決定する。ロードマップ本体は変更しない。公開後mandatory STOP。

## 次回用配置の追補（2026-11-18）

[配置確認とfresh PF修正](../h4_price_validation_deployment_20261118/REPORT.md)を参照。上記MODULE未確認は本引渡し時点の状態であり、追補では固定providerのMODULEを再利用する最小修正と人工検収を行った。実compileは未実施。
