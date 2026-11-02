# Track B G10 v3：JSON辞書キー互換性の限定修正・S3 source準備

## 1. 目的

G10 v2 one-shot実行は、以下の技術的問題で停止した。

- 結果：`G10_TECHNICAL_INCONCLUSIVE`
- 原因：`TypeError: G10 JSON keys must be strings`
- 発生段階：`before_result_stream / validate_encode_write`
- Peak RSS：237.34 MiB（上限512 MiB）
- 正式な科学result：未保存
- Scientifically usable rows：0
- retry：0
- mandatory STOP：維持

GitHubのsourceと実行監査から、旧JSON serializerとS2の新streaming serializerの間に辞書キーの互換性問題があることが確認されている。

今回の目的は、**旧serializerの意味論と出力内容を維持しながら、S2で導入した低メモリの逐次JSON出力を修正し、新しいsource S3を実行前レビュー可能な状態にすること**である。

新しい科学計算やG10本番再実行は行わない。

## 2. 対象Repository・固定証拠

Repository：
`HIROMU1015/Partially-Randomized-Trotter`

旧G10 science source S：
`05c5ef23fce775a822ab5686f5da2f0d77675864`

G10 v1 technical result：
`e429c99d77b3222c5cca62750b2d111f87e4cb50`

G10 v2 source S2：
`a139b91f119d109430ae3154a045d0fdcf722233`

G10 v2 authorization-only A2：
`1a2cd261ebe0cf097e71026a9150756dc8c9acc3`

G10 v2 technical result R2：
`f9d2665283c707e5a025a2c92c1a051153eaf2e1`

G10 v2 execution branch：
`track-b-g10-v2-one-shot-execution-20261010`

主要報告：
`docs/tracks/algorithm_codesign/g10_v2_results_and_gpt_handoff_20261010.md`

主要source：

- `src/trottertracks/algorithm_codesign/g7_generator.py`
- `src/trottertracks/algorithm_codesign/g10_saved.py`
- `src/trottertracks/algorithm_codesign/g10_io.py`
- `scripts/tracks/algorithm_codesign/g10_degree_matched_native_v2.py`

まずこれらと、v2のfailure receipt、provenance、契約、source manifest、旧serialization testsを確認する。

R2の結果・marker・STOP・authorization、および旧S/S2の固定sourceを変更しない。

修正は専用の新branchで行い、修正sourceをS3として整理する。branch構成と新しい証拠の保存場所は、旧結果の保全とprovenanceを満たす範囲でCodexが決めてよい。

## 3. 確認された互換性問題

旧serializerでは、辞書に対して以下の変換を行っていた。

```python
return {str(k): serial(v) for k, v in value.items()}
```

一方、S2の `g10_io.validate_tree` では、文字列以外のキーを拒否している。

```python
if not isinstance(key, str):
    raise TypeError("G10 JSON keys must be strings")
```

しかし、event生成側の `g7_generator._event` では、

```python
calls = Counter(reduced)
calls[child] += 2
provider_calls = dict(calls)
```

という処理があり、`provider_calls` のkeyには整数labelが入る。

新しいm3/m7のeventがそのままresultに保持されるため、S2のserializerで互換性問題が発生する。

ただし、実行記録は最初の非文字列keyの完全なpathを保存していないため、`provider_calls`以外に同様の問題が存在しないことまでは確定していない。

## 4. 修正内容

### A. 辞書キー互換性の修正

旧serializerが実際に出力していたJSONとの互換性を回復する。

特に以下を満たすこと。

1. 整数labelをkeyとする `provider_calls` を正しく出力できる。
2. 辞書キーの文字列化が旧`serial`と同じ意味になる。
3. 文字列化によるkey衝突、挿入順序、値の上書きについて、旧Python辞書変換の挙動を維持する。
4. 元の科学eventやresultの内容を変更しない。
5. Fractionの文字列化、tuple/list、Unicode、浮動小数点、JSON formatting、末尾改行を維持する。
6. nonfinite値、循環参照、未対応値について、適切な拒否条件を保持する。
7. Streaming出力によるメモリ削減を維持する。

特に、単純な `json.dumps(..., default=str)` への置換や、全resultを再帰的に複製する方式へ戻すことは禁止する。

実装方式はCodexの技術裁量に任せるが、辞書キーの変換を含めても全resultの一括materializationが不要な設計にする。

必要に応じて局所的な辞書の正規化は認める。ただし、全resultを複製してメモリ問題を再発させないこと。

### B. 出力schema全体の確認

`provider_calls`だけを修正して終了しないこと。

旧G10で保存対象となるresultについて、sourceから辞書を生成する経路を確認し、非文字列keyが入り得る箇所を洗い出す。

少なくとも、次を対象とする。

- event情報
- `provider_calls`
- native IR・cost
- synthesis cache
- budget・confidence関連field
- CTS certificate
- production interface trace
- m5の保存anchor
- その他のresult構成要素

実際に出力される型と、計算途中だけで使用する型を区別する。

保存済みJSONは既にkeyが文字列化されているため、それだけでlive typed payloadの互換性が確認できたとは扱わない。

### C. 旧serializerとの同値性検証

旧 `g10_saved.serial` を参照実装として、S3の逐次encoderと比較する。

Synthetic fixtureには少なくとも以下を含める。

- 整数keyを持つ `provider_calls`
- 整数keyと文字列keyの混在
- 文字列化後にkeyが衝突する辞書
- dictの挿入順序が異なるケース
- nested dict/list/tuple
- Fractionを含むevent
- shared subtree
- Unicode
- 非finite・未対応型
- 大きなtyped payload

旧方式と新方式で、同じ入力から生成されるJSONのbytes・SHA256を比較する。

辞書キー衝突に関しては、旧serializerがどの値を保持し、どの位置にkeyを残すかまで確認する。

不一致が見つかった場合、科学的出力として影響するものと、対象外の一般Python objectに限るものを区別し、根拠なく許容しない。

## 5. メモリ・出力保護の維持

S2で実装した以下の機能を保持する。

- 逐次JSON encode
- bounded UTF-8 write
- bytes・SHA256の逐次更新
- 不要参照の早期解放
- guard下の正常I/O
- `.partial` と正式resultの区別
- exclusive final publication
- success STOP・completion token
- 失敗時のbounded receipt
- technical failure時の科学的prefix利用禁止
- one-shot markerの保護

保存済みG10 JSONと新しいtyped fixtureを用いたI/O-only memory testを実施し、修正後も大きなメモリ増幅が復活していないか確認する。

RSS上限512 MiB、AS1536 MiB、その他の既存capsは変更しない。

今回のv2実行で観測された237.34 MiBは、正式payloadの出力前に停止した値である。したがって、S3の本番完了が512 MiB以内になると推定・保証してはいけない。

## 6. 科学的条件の固定

G10の研究目的、比較対象、実行条件は変更しない。

特に以下を維持する。

- `p=(1/5,3/10,1/2)`
- `x=5/7`
- Taylor次数 `m=3,5,7`
- 固定3-qubit synthetic provider
- 各次数内の同一有限演算子比較
- ordinary、partial return、closed P3/P5、full return、matched CTSなどの登録比較集合
- native T/CX/1Q会計
- confidence/error budget
- sampling・precision・seed
- synthesis設定
- fixed-policy lowerの計算方式
- runtime・resource caps
- one-shot・retry=0・mandatory STOP

科学event生成側の処理や、provider呼出し数の意味を変更してJSON問題を解決しないこと。

本修正は、あくまで出力表現の互換性修正である。

## 7. 実装・テストの裁量と禁止事項

Codexは、承認された技術修正の範囲内で、source構成、encoder実装、focused tests、非科学的なmemory検査、監査scriptの構成を自律的に決めてよい。

個々の技術的な修正やテストごとにGPTへ戻す必要はない。

一方、以下は禁止する。

- G10 v1/v2の再実行
- S3による本番科学実行
- 新しいauthorization-only A3の実行認可
- 登録用one-shot markerの作成・消費
- 旧失敗runのmarker解除
- 旧v1/v2結果の上書き・科学的採用
- native synthesisの新規実行
- 新しいsampling、matrix、LP、DF、分子計算
- 既存科学条件・比較指標・capsの変更
- 結果に合わせたprecision/seed/providerの変更
- G11以降への自動進行

保存済みデータのread-only検査、静的解析、人工payloadによるI/O-only検証は認める。

## 8. S3の成果物

次の資料を一括して準備する。

1. 修正済みsource S3
2. S2との差分と修正理由
3. 実際のresult型・辞書キー構造の監査
4. 旧serializerとのJSON同値性検証
5. key衝突・挿入順序のfocused tests
6. Streaming方式のmemory validation
7. 科学的意味論・capsの不変性監査
8. 旧source・result・marker・STOPの保護監査
9. 新しい実行契約の準備資料
10. GPT実行前レビュー向けの引継ぎ報告

新しい実行契約が必要なら、S3を固定sourceとするための準備まで行ってよい。

ただし、実行認可はpendingのままとし、本番用のauthorization-only childやmarkerは作成しない。

すべての主要資料、source、tests、監査記録をGitHubから取得できる状態にする。

必要なファイルだけを明示的にstageしてcommit・pushする。既存のdirty差分、旧証拠、別Trackのsourceを保護する。

push後はremote SHA、worktree clean、主要資料のremote取得可能性を確認する。

## 9. 最終報告とSTOP

最終報告では、以下を明示する。

- S3 branch・commit
- 修正したsource
- 実際に確認された非文字列keyの種類・発生経路
- 旧serializerとの一致範囲
- 辞書キー衝突への対応
- focused testsの件数・結果
- メモリ使用量の比較
- 科学条件・capsの不変性
- protected pathsの監査
- 本番実行前に残るリスク
- 新しい実行前レビューに必要な資料のURL

修正が想定より広い科学的意味論へ影響する場合、無条件にscopeを拡大せず、その問題を報告して停止すること。

**今回の完了条件は、G10の科学的成功ではなく、S2で判明したJSONキー互換性の問題を修正し、S3の実行前レビューが可能な状態にすることである。**

S3の準備と検証が完了した時点でmandatory STOPする。

GPTがS3をレビューして実行可能と判断し、ユーザーが別途one-shot実行を明示承認するまで、本番科学実行へ進まない。