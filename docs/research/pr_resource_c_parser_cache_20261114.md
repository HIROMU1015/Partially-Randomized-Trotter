# C: fixed B namespace candidate parser cache

監査 `51d324ea9c83bc12f1e866da0fa409cef926f74a` のE1に対するC基点の技術修正。
基点は `d7c09ef2e2521bb1815d1d11970529fc7030c2e3`。
元C branch/worktreeには適用せず、`codex/pr-resource-c-parser-cache-20261114` に分離した。

- live pipelineは絶対path＋完全SOURCE bytesでparse済みcid callableを1つ保持する。
- `baseline_signal_pipeline.published_api` のprivate固定B namespaceでは、読み込んだ元関数を
  変更せず、そのnamespace内の `ast.parse` だけにSOURCE内容別cacheを入れる。
  grammar/parserはnamespaceごとに保持し、標準`ast`や他moduleを変更しない。
- 同じ固定Bを読む `check_bv2_mean_price` のfixture loaderにも同じadapterを使う。
  共有moduleの旧byte/AST固定テストは該当歴史commitのblobへ照合する。
- 過去の固定blob、科学SOURCE、入力、結果、契約、STOPを変更しない。新しいadapter層の
  技術SOURCEであり、固定Bの科学codeを変更したり科学実行を再承認したものではない。

`check_candidate_id_cache.py` と `check_published_candidate_cache.py` は、ID完全一致、
元関数code保持、標準ast非変更、namespace別cache、SOURCE変更・入力例外を確認する。
関連するmetadata/人工回帰と測定の統合記録は
`codex/pr-resource-technical-fixes-20261114` の
`docs/research/pr_resource_technical_fixes_20261114/` に保存する。
科学計算・保存数値再解析・sampling・実回路build/compile・正式G評価は行わずmandatory STOP。
