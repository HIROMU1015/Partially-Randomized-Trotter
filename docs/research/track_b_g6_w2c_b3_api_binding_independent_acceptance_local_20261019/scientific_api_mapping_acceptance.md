# 独立API inventory・mapping

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

[独立inventory](scientific_api_independent_inventory.json)は新しいAST NodeVisitorで8 module・201 class/function/nested entryを採取。module/segment SHA、行範囲、name、signature、引数、return annotation、owned return/raise、call sites、assignmentを記録する。Producer inventoryは比較対象のdataとして読むだけで、その解析関数は呼ばない。

|module|entry数|主な境界|
|---|---:|---|
|g7_generator|29|canonical/full dispatch、bits、exact event|
|g7_provider|3|controlled IR記述、cache照会。native実装を実行していない|
|return_aggregation|27|ReturnKernel held state、digital parent、child law|
|g6_f3_contract|23|exact request、Evidence、F3Failure、codec|
|g6_f3_precision|9|固定precision assessment、bind_law、atan enclosure|
|g6_f3_generator|22|session guard、trace、typed stop、requery|
|g6_w2c_b3_driver|59|LegacyEndpoint、run_pair、F01、retention|
|g6_w2c_cost_preparation|29|canonical_snapshot、AuditParity、event_witness、codec|

Source/segment/entry/signatureは一致。Python3.12のPEP701形式でunparseされた `run_pair`/`run_pair.context` のreturn表示2件は3.10で再parseできない。元SOURCEとsegment SHAは一致し、元bytesから独立ASTを採取した。これら表示文字列のcross-version parse比較は未確立として別記する。他のreturn/raise表現はAST正規化で比較する。Producerの包括return集計はnested returnも含むため、独立inventoryにはnested bodyを除いたowned syntaxも別に保持する。

|方式|Layer A|Layer B|X/Y|
|---|---|---|---|
|ordinary|make_generator→CanonicalGenerator.sample|LegacyEndpoint overlay: canonical_snapshot→AuditParity→event_witness、event(group_index,word,child)再照会→enclose_atan|実行IDと独立bits domain|
|partial_return_tail|同canonical・partial groups＋ordinary tail|同canonical監査経路|同上|
|closed_P3_tail|同canonical・collected root/conditional groups＋ordinary tail|同canonical監査経路|同上|
|full_return|make_generator→FullReturnGenerator.sample|F3.prepare→GeneratorSession.sample/angle_for_event|同上|

[静的call site証拠](static_mapping_verification.json)。軸はG7 constructor引数ではなく、sigmaはprovider/angle context。F3 `prepare` はx=0 identity分岐の後、正入力canonicalを `CANONICAL_PREFLIGHT_NOT_IMPLEMENTED` で拒否する。存在しないcanonical facadeを仮定しない。固定正入力P5ではidentity分岐を対応先にしない。

Full：order draw→raw labels→reduce判定→有効rawだけpacket→acceptance draw→acceptedだけchild draw。`sample`のNoneは非reduced/coin rejectを区別できるtrace/packet境界を必要とする。G7 None単体は分類証拠ではない。F3 `_TraceBits` と `sample` はdegreeとdraw数を確認し、RAW_NONREDUCED/COIN_REJECTを別ZERO reasonとして封印する。例外はF3Failure/STOPへ進み、zeroへ変換しない。Aの既存LegacyEndpointではpacket-call観測が分類に使われる。新portのZERO_R/ZERO_Cは人工分類であり実traceの同値ではない。

packetはReturnKernel.digital_parent→parent/word_series/convolution/_mass等へ依存する。setupはfirst_passage/returns等の有理数stateを保持し、sample内のpacket再照会は避ける。Bのangleはevent→packetを再照会してfingerprint一致を確認。成長packet cacheはSOURCE内に見当たらない。F3はinput/law/source/preflight stateをguardし、失敗時sessionがterminalになる。source identityはfile readを伴う。provider cacheは別IR bindingで、実native処理の証拠ではない。

Word/raw/circuit-time wordはtuple[int]、child/phase/provider countはint、ratio/coefficient/proposal/weightはFractionとしてSOURCEが構成する。weight×proposal=coefficient、rawとreducedの区別、reverse circuit orderを維持。AST return構文だけで実行時schemaを完全証明したとはしない。

F3は `$rat/$int/$tuple/$map`、driverは `rat/int` とexact変換を用いる。文書・tuple/mapの表現と封印対象が異なる。codecのbyte互換・変換実装は `PENDING`。実G7/F3 law、bits、angle enclosure、動作同値の実行検証は `NOT_ESTABLISHED`。今回の静的受入は科学的妥当性の新規レビューではない。
