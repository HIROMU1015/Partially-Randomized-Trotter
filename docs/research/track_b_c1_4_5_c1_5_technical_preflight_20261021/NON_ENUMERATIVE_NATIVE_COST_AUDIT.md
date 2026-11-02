# 非列挙native費用の静的監査

**非列挙のevent生成は既存SOURCEにあるが、実R1のuniform strict native価格・期待費用は取得していない。** source-bound mock guardのPASSでこれを埋めない。

## D1：列挙が残る依存関係

固定[pipeline.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/pipeline.py)を静的追跡した。

| 処理 | 固定行 | 実際の依存 |
|---|---|---|
| `collect_pending` | 51–89、特に75–83 | production generatorとbounds/予算を作った後、`reference_events`を全list化し、`validate_events`とpending保存へ渡す |
| `freeze_inventory` | 22–39 | 全pending eventからratio集合を作り、saved anchorとunion、sort。全key事前inventoryを固定 |
| `acquire` | 92–145 | inventory全keyの固定reuse/新synthesis。価格cacheは全inventoryに依存 |
| `evaluate` | 148–183 | 全eventにnative_ir/costとmatrix評価、mean、全trial pi*price、worstを取得。有限targetと積をmatrix診断 |
| orchestration | 205–211 | collect→freeze→acquire→evaluateを接続 |

したがって`production_reads_reference_table=False`、`all_path_enumeration=False`は、それぞれsamplerへの表入力なし、whole-path直積列挙なしを意味する。**event価格取得全体が非列挙という意味ではない。** pending/ratio/評価を別fileへ動かしてもこの依存は消えない。今回D1を実行・変更していない。

## 回避候補と閉じない部分

| 部品 | 非列挙で得られるもの | 残る条件 |
|---|---|---|
| [G8 on-demand](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_pipeline.py) | actual accepted eventのratio keyだけをcache miss取得。zeroはcache/provider/readout前にreturn。memoは同key再合成を避け、LRU descriptionを再利用 | 観測keyのみ。backend例外/byte超過はfailure、key cap/attempt/accepted capもSTOP。class全体が全例外後terminalを一括保証するわけではなく上位のfailure終端が必要。未観測keyの成功・価格・全supportを保証しない |
| G8 strict error | rowのstrict RZ error、actual inverseと`2*row_error+n_provider*delta` | deltaは物理providerの仮定。G7 helperのCQ呼出回数をdirect DF回路のnative価格へ流用しない。G8自体もquantum_measurements=0 |
| [P5 groups](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py) | root1、ordered pair L(L−1)、length4 first-label群<=L。O(L²)群/angle/normalizer、completion(previous,remaining<=3) O(L)状態・O(L²)遷移仕事。長さ4全word表不要 | 現`dyadic_index`の群scanはO(L²)。word/child conditional O(L)、root/bit/rational/sort/証拠費用別。共有angleだけでword依存basis cancellation/native価格は決まらない |
| Full local Green | 各parentの係数query、棄却、ordinary envelopeとU。global Full normalizer不要 | 現SOURCEの全child masses・Parent.s・dyadic law/sort・再query/Evidence費用は残る。案A/Bを実装済み扱いしない |

O表記は有理演算/状態数であり、bit演算・CPU秒・量子T/CXの主張ではない。pの分母/次数/precisionに依存するexact bit長、integer root、angle区間、DP/cache/Evidenceのbytes・保存終端を別計上する。合法なcache/前処理/selector最適化は対照にも同等に認める。[方法ノート§4–5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L277)

## Uniform controlled DF providerの必要証拠

対象identityはR1の13 basis・324 signed Z/ZZ label。今回catalogの新数値処理をしていない。必要なのは次のsource-bound receiptであり、旧6 eventのfloat controlled一致では代用しない。

| 対象 | 必要なSOURCE/証拠 |
|---|---|
| basis | 全basis ID/hash/operation順・qubit・gate parameter、各実native sequence/誤差と合成profile |
| signed Z/ZZ | label順、sign、support、controlled中央操作、signのcontrol相対位相を一度だけ実装 |
| adjoint | nominal逆角でなく、取得した実sequenceのreverse+actual inverse。誤差伝播と費用を同じ実体に結合 |
| child rotation | ratio→atan enclosure→代表角→binary/native角の誤差、RZ/RZZ controlled loweringのstrict operator norm、precision/profile/source |
| scalar | 抽出identityとHamiltonian scalar、timeの配分、control上P位相、その丸め/合成/費用 |
| uniform coverage | 有限basis/label証拠と、全合法angle-domainまたは全key coverageを結合する既存の合成保証。observed-onlyとは別のscope証拠 |
| native費用 | T/CX/1Q/depth/workspace、tool/runtime/options/native IR・strict証明、費用層・単位。whole-path相殺の採否を全方式共通で固定 |

既存[DF actual inverse](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_qiskit.py#L448)は構造部品で、全native誤差/価格のcertificateではない。up-to-global-phaseはcontrolled相対位相を保証しない。既存[G8 bound](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_bounds.py#L24)はuniform deltaが既に取得できた場合の条件付き適用である。

finite catalogの基底/中央操作と既存uniform rotation guaranteeを結べれば、全word列挙なしの保守的strict上界を得る経路はある。ただし**今回その前提となる価格/certificateを取得していない**。wordごとの最良相殺を含むexact期待native価格を求める一般cost oracleは既存SOURCEからは閉じない。新oracle/非自明証明を要するかは、到達費用層を決めた後に判断する。現時点で`PROOF_REQUIRED`とは認定しない。

## 5費用層と最小guard

1. Logical instruction-size bound：有限basis/操作列からの構造上界。旧DF APIのTaylor次数をreduced parent長へ型偽装しない。
2. Strict native bound：認証tool/profile/誤差/全supportが必要。未知価格に「保守上界」の名前を付けない。
3. Event期待native価格：実全trial piによるsum pi_e*c(e)。uniform価格上界は保守上界でありexact期待価格ではない。observed平均は母期待値certificateではない。
4. 全trial総量子費用：十分N、whole-path law、zero先決定、準備/readout頻度、tail/hard/終端へ結合する。既実行prefixは消さない。
5. Classical setup/online/storage/terminal：raw/coin zeroにも係数/selector/root/監査/保存等の費用がある。CPU秒/有理回数/bit/bytes/native量を無規約で合算しない。

[native_receipt.py](native_receipt.py)はSOURCE/model/basis/labels/angle/precision/profile/control phase/actual inverse/scalar/eventをkeyへ結ぶ。methodをkeyへ入れず、同じ理想Full/P5 eventは同じprovider価格keyにする。strict/up-to-phase、observed/full coverage、unknown価格、event/path費用層を検査し、real provider要求は常に`REAL_PROVIDER_UNSUPPORTED_CERTIFICATE_REQUIRED`で拒否する。

mockは`SYNTHETIC_ONLY`・FAKE_PRICEと明記。coverage文字列のguardは外部coverage証明を新たに作らない。path価格はserial加法、準備/readoutをpathに一回、workspace最大、相殺なしという人工規約だけを確認した。実証明validator、unsupported real provider、期待価格oracleは未実装/UNKNOWN。新native backend・compile・synthesisは0。
