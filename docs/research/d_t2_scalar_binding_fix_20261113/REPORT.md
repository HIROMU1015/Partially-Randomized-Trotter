# T2 scalar quantity接続の最小技術修正

2026-11-13 JST。**Dの専用request selectorを、source-boundなHamiltonian定数＋元A tail identityのtotal scalarを渡すよう修正した。** 元runtimeのscalar定義・binary64完全一致guard・一回適用・wrapper/compilerは変更していない。新17＋既存12＝**29人工・metadata回帰PASS**。実H4の新preparation、sampling/event/path生成、回路build/compile、新価格/mean/CI/fit/N/RMSE/G・T2再実行は0。

[HANDOFF](HANDOFF.md)／[T2再実行の条件](T2_RERUN_REQUIREMENTS.md)／[TESTS](TESTS.txt)／[修正後の選択metadata](SCALAR_BINDING_METADATA.json)。基点は`240a6be56c5096e7ede27a27fa8f7d8dbfee290b`、既存branchを継続。Cの`3802bcc0212325091515325a889fcab20eabd909`はGit objectを読解・metadata関数を試しただけで、C branch/SOURCEを変更していない。

## 修正SOURCE・差分

production変更は [`ld12_price_requests.py`](../../../src/trottertracks/ld12_price_requests.py) のみ。

| 箇所 | 差分と目的 |
|---|---|
| `_constant_from_inputs / _saved_constant` | 固定Git locator→保存receipt/snapshotのSHA・bytes・H/state/rankを照合。NPZの`constant` memberだけを読む。H/stateオブジェクトは復元しない |
| `_request` | `ham_constant + tail_identity`をPython binary64で元順に加算し、既存`D.from_b(expected_scalar=total)`へ渡す。constantにtail-only quantityを与えると拒否 |
| input_source | `scalar_binding`に3 quantity・値・source・加算規則を保存。既存context/request digestに自然に含まれる |
| `request_for / prepare` | public APIは不変。必要な選択時に定数を取得。全coverage生成時はprefixごとに読み、roundごとの重複読取りを避ける。今回全52軸の新coverageは保存しない |

`candidate_price_adapter`、`NativeRuntime.context`、Accounting/compiled_price_fast、native emitter/compose、full measured X/Y、lowering/compiler、fingerprint/cache、seed/samplerのbodyは基点とbyte不変。ID生成関数`D._key`も不変。旧価格・旧requestのIDを新requestへ付け替えない。

## quantityとSOURCE

| quantity | 取得元・意味 |
|---|---|
| `HAMILTONIAN_CONSTANT` c_H | 保存snapshotの`constant`配列。元Hの定数項 |
| `TAIL_EXTRACTED_IDENTITY_COEFFICIENT` s_R | A固定MENUの元`tail_descriptor.scalar`。randomized tailから抽出するidentity。定義・値は変更しない |
| `EXPLICIT_TOTAL_SCALAR` | c_H+s_R。元NativeRuntime.contextが比較・適用する明示scalar。deterministic headに残る項を再抽出・追加しない |

入力locatorは既存T2結果 **752b9747a363eb4f1e510092230ef8dc243187ca** の `docs/research/d_ld2_m12_t2_20261113/run01/PREFLIGHT.json#inputs`。過去runを実行・再解析する入口ではなく、保存input2ファイルのpath/SHA/bytesだけを利用する。

- snapshot SHA：`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。同じ値をA configurationのbinding.snapshotと照合。
- receipt SHA：`7a161c1a3830dbf281d8ef4d05c013a3852955d4869cc41d26e033b7968feb1e`。COMPLETE状態、snapshot SHA/bytes、H/state/rank、constant配列recordを照合。
- constant配列SHA：`874b34c93c04576c14050d91ecace903d7b2693560bf77a05b2178fc48fb232b`。元saved loaderと同じ`np.ascontiguousarray`後のshape/dtype/bytes規則。実NPZの0次元scalarもこの規則ではshape=[1]となる。
- A menu：**a9abf8de75bf3432fcd714e868e5c14c8884b8f8**。tailの元index/selector/context SHAをそのまま保持。

NumPyのpublic `np.load(..., allow_pickle=False)`をlazyに使い、constant以外の配列は展開しない。snapshot全体のbytesはSHA照合のため読む。定数やsourceが欠ける、checksum/binding/arrayが違う、scalarがbinary64一要素でない場合は例外で停止し、診断値や0で補完しない。診断結果の数値はproduction readerから一切参照していない。

## 選択metadataの確認とID

実行しないLD2/m12/X,Yのrequest metadataを1件だけ生成した。保存constantの読取りと期待scalarのmetadata構成は今回の明示された修正範囲であり、新preparationではない。

| 項目 | metadata |
|---|---:|
| snapshot由来 c_H | 2.29310124732 |
| 元A tail identity s_R | 0.2223759094107018 |
| expected total c_H+s_R | 2.515477156730702 |
| expected hex | `0x1.41fb27ccff28cp+1` |

既存診断240a6be56の保存totalとhex一致をmetadata回帰として確認した。値をコードへhard-codeしていない。今回のartifactのexecution_sourceは**TECHNICAL_METADATA_ONLY**で、selector Git blobを含む。科学runのSOURCE・価格証拠ではない。後続の承認済みrunではその実行SOURCEでrequestを作り直す。

context IDは既存canonical JSONのcontent digestで、corrected expected totalと`input_source.scalar_binding`のquantity/provenanceを含む。request IDは従来どおりaxesを含む。同じ物理sourceならX/Yはcontextを共有してaxis別requestを区別でき、sourceが変われば同じ数値でもkeyは変わる。B canonical configuration/physical schedule・lawを変更していない。

## テストと限界

新17項目はscalar-onlyの人工NPZとmock preparation/Accounting、C380の固定metadata関数のみ。非ゼロ/ゼロc_H、0次元/一要素配列、source欠測/不一致、tail-only誤接続、正しいkey、1 ULP差の拒否、scalar一回、X/Y、CのD748 constructorによる再構成・resolved scalar拒否、歴史的記録保全を確認した。元12 metadata項目は人工constantで回帰し、元の誤った「expected=tail」assertionを「expected=constant+tail」へ直した。29項目すべてPASS。作者自身の技術確認で、独立科学レビューではない。

元Sのscalar一回適用は実`structural_pair`のloopへmock Accountingを渡して観測し、native compose/wrapperのsource不変も確認した。量子回路オブジェクト・sampler・実prepare・compilerをテストで起動していない。実runtimeの現在SOURCEでの通過、m12巨大回路のtime/RSS、C/S/D、全law mean/CIは未取得のまま。

旧748 request/52軸coverage、752のT2 run01、240の診断raw/結果、SOURCE/input/cache/grant/ledger/marker/KEEP/STOPは不変。修正metadataのPASSをT2許可にしない。成果をcommit/pushし、remote SHA/tree/主要成果物を照合した後 **mandatory STOP**。
