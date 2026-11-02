# 静的限定moduleと人工runtimeの依存の不一致

現在の判定：**RESOLVED_BY_EXPLICIT_USER_SCOPE_AMENDMENT**。固定Producerを変更していない。

利用者がこのセッションで§3Eを明示修正し、`g6_w2c_b3_driver.py`と`g6_w2c_cost_preparation.py`について、人工runtime・既存回帰・人工negative/mutationに必要なimport/関数実行を許可した。
これは実科学の承認ではない。core科学6 module（G7 generator/provider、return aggregation、F3 contract/precision/generator）は静的読取り・AST・SHA限定を維持する。
親・spawn worker双方でload前guardとload済みmodule一覧を確認し、新たな禁止moduleへの推移的依存があれば実行せずSTOPする。
以下は修正前に観測・保留した経緯であり、過去のSTOPや失敗を消去しない。追加scopeは新たな独立SOURCE commitとして記録する。

対象SOURCE：`b7ad344401b10395c49e4cfa806909282859bab5`。
公開HEAD：`80b7bf61fe0a735d4031bb719c4b3e788611c3ba`。

今回の利用者指示§3Eは8 moduleを列挙し、「静的読取り・AST・bytes/hash照合だけを許可する」と指定している。
この集合には`g6_w2c_cost_preparation.py`と`g6_w2c_b3_driver.py`も含まれる。

人工Bitsの通常importは、固定SOURCEの次のmodule scope依存を辿る。

1. `g6_w2c_b3_real_adapter_v1/bits.py:4` → `g6_w2c_b3_api_binding_v1.ports`
2. `g6_w2c_b3_api_binding_v1/ports.py:7` → `g6_w2c_b3_parallel_driver`
3. `g6_w2c_b3_parallel_driver.py:20` → `g6_w2c_cost_preparation`

独立側の8 module load前guardが、初回subject processの起動時に
`SUBJECT_PRELOAD_DENIAL:trottertracks.algorithm_codesign.g6_w2c_cost_preparation`
で遮断した。対象module本体はloadしていない。試験本体開始前の失敗であり、最終197 PASSへ加算していない。
以後はこの既知経路を再試行せず、Bits/gate 17予定checkとProducer132件・旧107件の再実行を保留した。

Producerの`static.py:16-19`は、core科学6 moduleを`SCIENCE`、これにdriver/helperを加えた8 moduleを`STATIC_MODULES`として区別する。
Producerのguardが今回の静的限定8 module全体を遮断する設計ではないことは、ASTとbytesから確認できる。
旧107回帰の一部は`g6_w2c_b3_driver`を明示的にimportする。

helperのdocstringとimport列は、標準ライブラリによる人工audit支援であることを示す。
この不一致を、実G7/F3 samplingが実行された証拠や科学lawの欠陥とは扱わない。
一方、今回の明示的な静的限定を、agentの判断だけで6 moduleへ縮小することもできない。

期待値は8 moduleの静的限定を守った人工試験、観測値は人工runtimeの起動にそのうち1 helperのloadが必要という状態である。
証拠：[AST依存・固定path・line](static_import_boundary.json)、[未実行17件](blocked_dynamic_checks.json)、[旧回帰のimport列](regression_summary.json)。

次はユーザー・GPTが、helperと旧driverの人工用途だけのimport/実行を明示的に許可するか、
別versioned Producerで依存を分離するかを判断する。後者では`ports.py`と`parallel_driver.py`を含む依存経路の修正範囲を固定する必要がある。
実G7/F3 core6 module・実科学bits・科学処理・実authority・旧STOPについての禁止はどちらの場合も維持する。
承認前にstub差替え、import guardの緩和、固定Producer/科学契約の変更は行わない。

## 修正後の完了記録

利用者の追加許可を適用し、保留17件を含む214独立probe、Producer102unit/30integration、最終SOURCEの旧107回帰、親/spawnの30gate拒否、新人工Bundle2件を完了した。禁止core6 moduleへの新たな推移的依存・実import試行・loadは0。未解決scope衝突はない。旧判断と既存STOPは保持する。
