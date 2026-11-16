# B2/D2向け：複数幅探索のQ3接続修正

[REPORT](REPORT.md)。公開branchは`codex/pr-multiwidth-route-fix-public-20261117`、公開基点はe0f7838b09bc09412ec764d80686cb9578e44d01。ローカルの実装基点c8bcb3a65と公開基点で既存codeは同一。統合元5829a7288と限定実行修正7f1e26753を保持し、検証した技術差分だけを公開した。元R06はSTOPのままで、科学計算は再開していない。旧runの約118 MBのrawを含むauto-commit履歴は削除・書換えず、元のローカルbranchに保全している。

## 接続

- 通常入口は従来の`python -B scripts/run_rpe_schedule_search.py --problem <fixed> --round <fixed> --run-dir <new>`。別の探索器・queue・Q3 workerは追加していない。
- `Completion.install()`は準備済みrecordに加え、保存catalogの未接続PR候補もSOURCE付きで既存common routeへ結合する。`WidthDriver.batch()`と通常method CLIは既にこの入口を通る。追加の接続スイッチは不要。
- `Executor.route()`はtemplate内に対象configurationが存在するか確認する。`configuration_ids=*`だけでは実行可能としない。Reader cacheを使い、同じtemplateを配分ごとに読み直さない。
- 未接続候補は`execution_needs=[Q3_ROUTE_CONFIGURATION_OR_TEMPLATE]`、保存metadataの矛盾は`execution_binding_hold`、V2への要求は既存`CONFIGURATION_INPUTS_MISSING`。元の`pr_candidate_selection.normalize()`から補完へ渡せる。routeが用意できたら次のoptimizeで再判定する。入力要求が残る幅driverの停止理由は`WAITING_FOR_INPUTS`。
- 入力が使える候補の`ready`は保持し、未接続でも保存適合点を比較から消さない。保留候補の新Q3のみ抑制する。数値UNKNOWN、未解決の旧job、既存STOPは維持。
- Pauliの保存descriptor SOURCE（index/context）と派生SOURCE（commit/path）を区別する。元A descriptor、scalar、signed schedule、decoder、exact τ、compiler/lowering、価格quantityは変更していない。

## 再現

```bash
python3 -B -m unittest discover -s tests -p test_pr_execution_route_binding.py -v
```

回帰72件のmodule一覧と結果は`TEST_GROUP_0.json`と`TEST_GROUP_1.json`。停止2候補の本番worker metadata確認は`SAVED_BINDING_CHECK.json`から各`metadata_preflight/<method>/REQUEST.json`と`check01/DRY_CHECK.json`へ辿れる。科学APIを実行する`execute`モードではない。

## 復旧に引き継ぐ条件

1. 旧run `h4_pr_multiwidth_optimization_20261117/run01`を再起動しても、保存済み`V2_RESULT_REQUIRES_REVIEW`は解除されない。この修正でSTOP・失敗stageを消したり上書きしたりしない。
2. 復旧時は修正SOURCEと新しいround/run/outputを固定し、元の利用者承認範囲・one-shot条件に照合する。旧runの完了Q3・READY入力・価格証拠は既存のidentity/SOURCE条件で再利用し、失敗jobの再試行は自動採択しない。
3. §10.5で公開成果・旧run・同一configuration/配分・現在の稼働jobを差分確認する。既存workerが走っていれば所有runを引き継ぎ、二重起動しない。
4. Pauliの過去8件のsignal準備失敗は旧recordのまま。今回は新しいsignal epochを作って数値計算していない。
5. DFの`Conflicting saved means`は保留が残る。明示した保存inventoryの整合を確認するまで異なる保存値を自動選択しない。route修正を、その競合の解消証拠に使わない。

ロードマップ差分：両PRの保存候補から通常Q3入口への接続欠落を技術的に解消。人工72件、停止2候補のmetadata gate PASSを追加。新しい実H4価格・信号・Q3適合点・達成費用の増分は0。人工結果はCのcatalogに登録していない。実探索の再開・完走と幅別の最終集計は未実施。
