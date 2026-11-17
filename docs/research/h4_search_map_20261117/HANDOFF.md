# H₄探索マップ：通常同期・各環境の公開手順

[探索マップ](index.html)／[索引](INDEX.json)／[最新収録範囲](REPORT.md)／[固定入力](SOURCES.json)。Cの既存branch `codex/h4-saved-rpe-cost-catalog-20261116`で管理する。ロードマップとは別成果物。HTMLは自己完結で、ファイルを保存してブラウザで開ける（GitHubのblob表示はHTMLソース表示）。CDN・fetch・追加サービスは不要。

## 通常運用

既存の利用者自身の10分cron → `scripts/run_h4_catalog_sync.sh` → `scripts/sync_saved_rpe_publications.py`を使用する。既存比較カタログを生成後、探索マップを同じ同期・commit/pushで更新する。独立runや各hostのprivate実行領域は読まない。公開Gitだけを読む。

- 既存系列は公開ROUND_RESULT・NEEDS・SCOPE_METADATAから初版を作成済み。元run・停止理由・固定commitはREPORTとINDEX.runsに明記。
- 新しい環境・新branchは下記の共通controller公開索引を使う。Cへのcommit/path転送・監視branchの個別追加は不要。全公開branchから `SEARCH_PUBLICATION.json` を検出し、旧式の暗黙読取だけを従来のbranch範囲へ限定する。
- 部分的な取込不能はissuesへ残し、他の保存成果は同期する。マップの作成自体が失敗した場合は直前のHTML/INDEXを保全し、STATUS.jsonをPENDINGにして次回同期で再試行する。科学runのretryではない。
- dirty checkoutはその周期をskipして編集を保全する。C自身の公開branchを入力監視に含めず、同期commitが次の同期commitを呼ぶ循環を防ぐ。
- 費用だけの変化でロードマップ本文を変更しない既存規則は保持。マップの件数やHTML変化だけでもロードマップの科学段階は変更しない。

通常入口を手動で検査する場合も同じshellを使う（定期運用は登録済み）：

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/c-h4-saved-cost-catalog-20261116
bash scripts/run_h4_catalog_sync.sh
```

ログ：`artifacts/local_h4_catalog_sync/cron.log`。同期/公開状態：既存 `h4_saved_rpe_cost_catalog_20261116/PUBLICATIONS.json` と本ディレクトリ `STATUS.json`。追加の科学計算は一切起動しない。

## 各環境・独立runの共通公開手順

controllerが採用した保存結果を用いる。workerごとに同じ評価を登録しない。既存runの生成SOURCE・結果を変更せず、公開sidecarだけを追加する。スクリプトはstdlibのみなので、Cの固定commitからこの1ファイルを取得して各環境で使用できる。

```bash
# 既存の研究checkout内で、保存controller結果に対して実行する。
python3 /path/to/publish_h4_search_run.py \
  --repository "$PWD" \
  --run-dir docs/research/MY_SEARCH/run01 \
  --output docs/research/MY_SEARCH/SEARCH_PUBLICATION.json \
  --run-id MY_CONTROLLER_RUN_ID \
  --execution-host CONTROLLER_HOST --execution-host EXECUTOR_HOST

# 元の公開手順で、この索引と参照先metadataを一緒にcommit/pushする。
# 別hostは別runの保存先へ出力し、他担当のrunやSOURCEを変更しない。
git add docs/research/MY_SEARCH/SEARCH_PUBLICATION.json docs/research/MY_SEARCH/run01
git commit -m 'Publish saved H4 controller search metadata [skip ci]'
git push origin MY_PUBLIC_BRANCH
```

`run-id`は同じcontrollerのcheckpoint/再開で維持する。別runは別ID。同じ配分の再利用は元configuration・N・評価契約・SOURCEを保持する。hostは分散実行の出所であり、独立評価数を増やさない。SOURCEの固定は公開commitとpath/selector・ファイルhashで行う。時刻や現在HEADだけで差分を増やさない。

公開対象は `ROUND_RESULT.json`（又はINTERRUPTED版）、`EVALUATIONS.json`、`NEEDS.json`、`SCOPE_METADATA.json`等、スクリプト内の小さいmetadata一覧。巨大raw・QPY・private job directoryは走査しない。公開するファイルの取捨選択は各producerの既存契約に従う。未公開holdoutを追加開封しない。

## 数え方と価格の意味

- 全run累計：方式・configuration・評価契約・X/Y配分の一意点。job名、host、複数HANDOFF、checkpoint、別run再利用では増えない。
- run表示：そのrunで採用/再利用された点。NEW表示：元`NEW_Q3_JOB`かcontroller自身の取得来歴が確認できる点のみ。SAVED/PRIOR_ROUND_JOBは除外。不明は新規0として、来歴不明と明記。
- Q3構成数は全RPE scheduleのconfiguration数。round×axisの回路件数ではない。κや公式の異なる構成は別点、X/Y shotは同じ配分の一組。
- 要求幅・実T/q・piece/schedule・生成physical IDは固定configurationへ戻る。binary64 0.2と厳密1/5を許容差で統合しない。
- public controller summaryのQ3判定・表示用上下界は原保存分類の転記。exact原本が省略された場合は元result locatorとその制限を表示する。
- 費用最安は既存C consumerが受け入れた条件付き価格だけ。新規のsummary点には保存費用の引用を残しても、価格binding未受入のまま最安へ昇格しない。E[C]・正式G・CI・winnerの資格は変更しない。
- 各HTMLマスは「探索対象・未評価」「評価済み・適合0」「元記録が未取込」を区別。scopeが不明な空白を未探索と断定しない。

## B2に残る最小公開差分

収録した最新multiwidth scopeは実行hostと進捗件数を公開している一方、配分別controller checkpointが未公開の範囲がある。進捗の17件という数字から17配分を捏造しない。

次回の通常公開で `ROUND_RESULT.methods[method].evaluated` に `configuration_id, N_X, N_Y, status, MSE_lower/upper, cost, source=NEW_Q3_JOB/PRIOR_ROUND_JOB/SAVED, job, result_sources` を保持し、対応する `NEEDS.candidates` に固定configuration/signal・価格quoteの出所を含めれば、既存parserで反映できる。精度契約の固定locatorも残す。分散runではcontroller側の採用結果を公開し、host一覧はsidecarへ追加する。新しい科学計算や探索規則は不要。

## 再現・検査

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests python3 -B -m unittest -v \
  check_h4_search_map test_h4_search_run_evidence check_h4_search_map_html check_h4_search_map_sync

# 現在公開している固定headsだけで再生成（ネットワーク不要）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B scripts/build_h4_search_map.py \
  --catalog docs/research/h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json \
  --sources docs/research/h4_search_map_20261117/SOURCES.json \
  --output /tmp/OWNED_FRESH_MAP_OUTPUT
```

保存run同士の新規/再利用、人工の任意環境branch→通常sync→INDEX/HTML、再同期UNCHANGED、元結果locator、source不一致・worker拒否、exact幅の区別を検査。HTMLのJSは構文と人工DOM操作を検査し、実ブラウザ描画そのものの検証とは区別する。詳細はTESTSとSYNC_VERIFICATION。

既存Cの定期同期を継続し、原run05・他担当の稼働run・科学SOURCE・旧STOPは保全。本作業の完了後、新しい科学計算へは進まない。

検査分担は、rootが通常同期・索引会計・公開、別担当sub-agentが保存run parserとHTMLを担当した。これらは実装の技術確認であり、独立した科学レビューではない。初回の技術回帰で、旧版をimportしたprocessが大型scheduleを配分ごとに複製していたため、その自分のtestだけを中断した。元logはlocal_h4_catalog_sync内へ保全し、immutable metadata共有・schedule索引化・phase間cache解放後の固定コードで再検査した。科学runや他担当processは停止していない。

## 初版の公開検収

通常shell入口はUPDATED・終了0で9 controller run／4,877一意配分点を公開した。同じ公開入力での再実行はUNCHANGED・終了0で、INDEX・HTML・REPORT・SOURCES・STATUSのbytesも不変。保存consumer回帰28件＋新規技術検査10件＝38 PASS。[検収記録](SYNC_VERIFICATION.json)／[検査log](TESTS.txt)。最新の件数は自動生成REPORTとHTMLを正本とする。定刻ジョブの設定は既存の10分cronを保全しており、新規の利用者操作は不要。
