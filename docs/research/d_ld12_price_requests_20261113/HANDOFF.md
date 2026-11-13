# A・B・C向け：固定PR-LD1/LD2の価格request

2026-11-13 JST。[REPORT](REPORT.md)／[52軸coverage](PRICE_REQUEST_COVERAGE.json)／[26件索引](PRICE_REQUEST_INDEX.json)／[次の有限scope](NEXT_SCIENCE_PLAN.md)。価格取得・平均推定は未実行。

## 選択する入口

```python
from trottertracks import ld12_price_requests as L

# 受動metadata。後で指定されたlabel/m/axesだけを選ぶ。
request = L.request_for(
    repo_root, selected_label, selected_m, axes=selected_axes,
    execution_source=approved_new_scientific_source,
)
```

labelはPR-LD1/PR-LD2、mは0…12、axesはX/Yの明示部分集合。これは取得範囲の承認を生成するAPIではない。実行はユーザーが別途許可した有限範囲だけ。NativeRuntime.context/sample、Adapter.acquireはこのmetadata helperから呼ばない。

A_SOURCE/file/row selectorはmodule定数と索引で固定する。configuration IDはB canonical、physical IDはA/cidの元値。Aのsignal producerやBのN APIを再実装しない。round別Nをこのrequestへ入れたり、event数をcompiled価格とみなさない。

公開索引のexecution_sourceはTECHNICAL_METADATA用であり、後続runは新しい科学SOURCEでrequestを再生成する。request/context keyが変わるので、公開coverageのID/keyを科学runの結果へ単純copyしない。canonical/physical対応と新keyの由来を保存し、過去metadataを保全する。

Runtimeは選択requestのphysical_contextにあるbinding/input_source/execution_sourceで作る。この固定接続ではinput_sourceにprefix別tail selectorを含むため、LD1とLD2は別Runtimeを所有する。同prefixの各roundはそのRuntimeを再利用できる。復元済みHの共有が必要ならread-onlyにし、別候補のcontextを差し替えて使わない。

## A・Cとの境界

- A固定MENU/configは読むだけ。進行中のsignal結果・private raw・process・grant等を読んで新解析したり変更したりしない。
- prefixごとのλ_R/scalarは指定MENU由来metadataで、input/tail出典とselectorを保持する。C又は元loaderが別のsource-bound preparation情報を返す場合、そのhash/representation/sourceのbridgeを別途確認する。
- 実preparation/成分/basisの生成と照合は今回未実行。将来D runtimeで既存prepareを使い、component/tail/preparation hashとexpected scalarを確認する。Cの統合・人工kernelテストをDで繰返す作業は追加していない。
- LD1 E[S]・P2 pilotは参照候補のみ。新LD1 canonicalとの受入bridge、quantity/law/profile/cohort/roleを確認し、LD2に数値を移植しない。

## B437ba/ b9b9の価格入力との対応

Bのcontroller/fixed-grid integrationはD基点版とbyte一致。価格は`FULL_WRAPPER_MEAN_COMPILED_PRICE`、same identity/law、axis/cohort/source、full measured wrapper/P0/RZ gate_count。現在は値付きmean recordを生成せず、必要なら既存D.pending_price_records(request)でmean=nullを渡す。

Bのwhole-shot adapterへ将来meanを供給する場合、lowering extensionは次のように別添する。旧configurationのprofileを歴史上書換えない。

```python
price_extension = dict(
    lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1',
    source=dict(commit=L.D_BASE, path='src/trottertracks/efficient_controlled_zz.py'),
)
# 科学的に供給可能になったmean recordごとに、B仕様のbindingを作る。
# identity/law_binding/axis/cohort、price_source=親meanのsource、
# source=接続証拠、profile_extension=上のextension、mean_interval=根拠付き区間又はNone。
```

今回生成したcoverageのprice_profileはこのlowering/compiler/control/wrapperを要求するmetadataで、対応する価格が取得済みという意味ではない。新旧loweringは同compilerでも混在させない。fingerprint/cacheは将来の実wrapper canonical bytes＋compilerで照合し、hit元を記録する。今回回路fingerprintは未生成。

Bの区間には`FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL`、exact endpoints、domain/source/qualification/evidence_kindと、全使用セルを覆う同時coverage契約が必要である。DのS/イベント数/outer-windowから根拠のない上下界を作らない。現在interval・confidence・N・Gはすべて未取得で、profile bindingだけで資格を埋めない。

## coverageの読み方

status=REQUEST_READYはmetadata互換だけ。preparation=NOT_BUILT、hash/basis/component=null、selected_for_execution=falseを保持する。1 roundのpaired request IDをX/Y cellが共有するが、応答はaxis別。単軸指定では新request IDになり、両軸へ補完しない。

全52軸に取得方法、既存参照とaccepted=false、未指定path/精度、law/rare/長q・計算負荷の障害を記録した。unsupported_conditions=[]はこの固定S2/K2 menuのmetadataで既存APIに非対応欄がなかったことだけを意味し、実行成功・実行可能性・精度保証ではない。

後続はA signalとBの予備分析/shot評価を受け、必要なcandidate/round/axesだけを有限scopeにする。N未資格のまま正式G重みを設定せず、全round compileやm7→m8→m9を既定にしない。詳細はNEXT_SCIENCE_PLANを用いる。

**現在はmandatory STOP。** ユーザーの新しい有限科学承認までDはcontext/sample/acquireへ進まない。
