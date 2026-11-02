# 並列schedulerの設計（非採択）

基点は公開統合版 `564409487a35e15d73f1d0f0c2611824b400ab02`。既存driver/admission/host/readiness/F01/scienceのファイルは変更しない。追加した `g6_w2c_b3_parallel_candidate.py` は **人工portだけを許すprototype** で、本番hostの新entrypointではない。実行時の科学module importも拒否する。

S0は元の `D.execute` / `isolated_pair` / `run_pair` / `Stream` を使う逐次基準。人工adapterで数値依存・saved evidence loader・endpoint・bits・有効Nだけを置換する。実saved結果を数値解析しない。人工endpointをfine observerの対象へ加えるtest用subclassとI/O観測wrapperを使用する。元SOURCEは不変。

Sは新しい管理機構を1 pairずつ固定順序で実行する対照。P1は同一phase内の8 pairを1/2/4/8 fresh workerで実行する。P2は8 fresh workerのまま、global outstanding reservationをcap件に絞り、ENTER前の再確認を加える。SとS0の差が新しい管理・観測費用であり、P対Sの倍率だけで導入利益を表さない。

親が全24 taskを固定identityとoffsetで作る。各phaseの契約順は4 arm × A/B。各fresh childは独立endpointを1個構築し、同一instanceでX→Yを処理する。原 `run_pair` のstate identity、分類、prefix、A/B traceの検査を使う。worker pool、pass overlap、pair分割は採用していない。

```
親host → 1 global artificial admission → correctness (8 pair)
        → 全pair成功・prefix照合 → baseline (8 pair)
        → 全pair成功・trace照合 → instrumented (8 pair)
        → 全join・固定identity集約 → summary → completion receipt → STOP
```

phase境界ではSOURCE/input pin・人工policyの不変性も再確認する。0分類/coverage不合格や未完了pairがあれば次phaseへ進まない。追加trialでcoverageを補わない。実際の完了順は `completion_order`、契約順は `contract_order`、比較用 `pairs` はphase/pair順。未dispatchのpairも `all_pair_states` に残す。

人工bit namespaceは `ARTIFICIAL_R1_PARALLEL/arm/axis`。SHA256 counterのdigest/discard規則を人工実装し、A/B・pass間の同じprefixを照合する。実G7 seedを実行した検証ではない。保存N・law・angleを正式変更する権限はない。人工profile Nはlight128/mixed64/heavy32、correctnessは16/axis。将来の本番Nとは分けて扱う。

workerへの明示的な引数はtask、settings、pipe、停止flag、固定pin、人工factory等。authority/ledger/portをworker APIへ渡さない。fork時に既存の親pipe端を閉じ、親の死亡時はEOFで協調終了する。既存SQLiteの接続は各operation内で閉じてからforkされる。worker独自のadmissionやretry APIはない。

将来の実接続では、固定実Nからoffsetを構成し、既存LegacyEndpointを原constructor/preflight/guard契約のまま接続する技術作業が別途必要。今回のfake host注入を本番の認証迂回手段にしない。
