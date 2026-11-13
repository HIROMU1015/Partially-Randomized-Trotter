# PR resource technical fixes: I1 / I2 / E1

監査SOURCEは `51d324ea9c83bc12f1e866da0fa409cef926f74a`。
実装は `codex/pr-resource-technical-fixes-20261114` に分離した。
科学kernel、旧SOURCE、結果、契約、grant、ledger、marker、STOPは変更しない。

実装・回帰・性能・branch適用状況の最終結果は [REPORT](REPORT.md)。

## 配線preflight

専用Python環境を使い、系列ごとに新しいprocessで実行する。
`--root` は対象SOURCEを展開したtree、`--source` はその完全commit SHA。
runnerは自身の実装と対象branchのimport経路を分け、対象treeへ書き込まない。

```bash
PY=/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python
$PY -B scripts/preflight_pr_resource_wiring.py --root /path/to/source/tree --series A --source <40-digit-SHA>
```

`--series B|C|D` で各経路を選ぶ。Dは実際に消費する既存形式の
`--request request.json` と `--plan plan.json` を渡せる。指定しない場合は
既存P1 planとLD1/LD2のm0/m12 requestを受動的に束縛する。
planのsourceは選択したcommitを指す必要がある。

- A: 固定signal kernelの定義を既存loaderで読む。実scriptのimport、Bの全round
  configuration、SignalBatchのmetadata plan、実H4 loaderの保存JSON/hash/bindingまでを確認。
  行列復元直前でreturnするSOURCE sliceを使い、復元・固有値・発展を呼ばない。
- B: 既存S0 passive intakeをそのまま呼ぶ。N/C/Gは取得しない。
- C: 既存configuration照合と固定C1 contextの実exec scopeを検査する。
  既存S-only loaderの最後のcontext呼出しだけを定義のreturnに置き換え、準備はしない。
  実descriptor呼出しの引数名を既存API signatureへ束縛し、literal seedは既存の
  `require_integer_count`だけで検査する。旧SOURCEの`seed=None`も生成前に拒否する。
- D: 固定C1 loader、Backendのmetadata constructor、既存plan-key抽出、保存inputの
  bytes/hash/binding、既存scalar reader/requestを検査。context/sample/build/compileは呼ばない。

過去7 STOPのうち、exec scope・descriptor API・configuration参照・plan key・script import・
scalar意味の6分類を扱う。unitarity gateは数値検証を要するため対象外。
このpreflightは全ての深い実行経路の保証ではなく、科学資格や実行許可でもない。
既存固定runnerの実行契約・SOURCE lockは変更せず、preflightが通ってもSTOPを再開しない。

## I2: historical provenance and live compatibility

S0 pipelineの同一性は、保存SOURCE indexのhashと固定commitのblobで照合する。
現行APIの互換性は全13roundのschedule/identity/law比較、保存configuration・request・
identity map・元N handoffとの完全一致、および不整合の拒否で確認する。
共有module全体への旧AST固定を取り除いた。旧テストも実際の歴史SOURCE blobを使う。
数値kernel等の既存固定チェックは保持する。

## E1: parse cache

module内で1つのparse済みcallableだけを保持する。keyは絶対pathとSOURCEの完全bytes。
fileは毎回読み、同じsize/mtimeでの変更も検出する。候補やconfigurationの結果はcacheしない。
各候補を元の`cid`へ毎回渡すので、入力検査・変更後のID・例外を維持する。

Cのprivate固定B namespaceには別途 `cache_legacy_candidate_parser` を使う。
元の`candidate_id`関数と固定blobを変更せず、namespace内のAST parserだけに
内容別cacheを入れる。標準astと他namespaceには影響しない。
C固有のloader/fixtureへの適用は、C基点の独立branch
`codex/pr-resource-c-parser-cache-20261114` に収録する。
測定scriptの `--legacy` でこの経路も比較できる。

```bash
$PY -B tests/check_candidate_id_cache.py
$PY -B tests/check_pr_resource_wiring.py
$PY -B tests/check_h4_s0_v2_intake.py
$PY -B docs/research/pr_resource_technical_fixes_20261114/bench_candidate_id.py --root "$PWD"
```

固定SOURCEの旧実行をこの新treeから直接再開することはできない。
将来の科学実行では採用する修正のSOURCEを明示し、既存の個別契約と実行許可を適用する。
今回の作業は技術実装・人工検証まででmandatory STOP。
