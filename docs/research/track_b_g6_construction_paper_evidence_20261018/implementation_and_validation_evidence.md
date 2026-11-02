# SOURCE・既済検証・有限bitの実装範囲

〔SOURCE静的対応〕科学moduleをimport・実行せず、固定bytes、構文木の宣言位置、処理本文と既存test本文を確認した。[source_inventory.json](source_inventory.json)はqualified name、開始/終了行、固定commit、blob/SHA256を記録する。以下の行番号はその固定SOURCEに対するもの。数学命題の詳細は[Matrix](mathematical_claim_evidence_matrix.md)。

## G6 ReturnKernel：局所packet

固定SOURCE `28cfabb1d47e0e1824bce1e154a9c289738fa2b9`。

| 処理・行 | 対応 | 静的に確認できること／できないこと |
|---|---|---|
| `ReturnKernel.__init__` 138–160、`convolution` 21–28 | M2/M11 | p/x/mのdomain、切断F/G、Cの再利用。量子回路は作らない |
| `_word` 162–168、`word_series` 170–175、`_mass/mass` 177–183 | M1/M3/M4 | 指定reduced語の有限係数。証明は数学監査、コード自体は正値定理の証明ではない |
| `parent` 194–202、`Parent` 114–133 | M5 | child前置、s、d²、tangent、child law。一般odd wordを回転軸にしない |
| `envelope2/order_intervals` 204–212、`digital_parent` 214–236 | M6/M9/M10/M12 | ordinary envelope、root区間、補正前係数とphase/orderをpacket化。RNG/測定なし |
| `root_interval` 45–52、`dyadic_distribution` 55–82、`dyadic_bernoulli` 85–94 | M9/M10 | integer isqrt外向き区間、largest remainders、support・相対誤差の検査、accept下向き丸め。不足時例外 |
| `dyadic_index` 97–110、`rational_event_weight` 239–254 | M9/M11 | bit整数→indexと実πによる有理補正。物理bit sourceではない |

`dyadic_index`はlawを走査し、`dyadic_distribution`はremainderをsortする。従って「O(m)」と記す際はraw語のreduced判定又はlabel draw回数の範囲に限定し、全生成実時間としない。全語履歴を無制限cacheへ蓄積する実装ではない。

## G7：sampler、U、予算、controlled IR

固定SOURCE `ab2549f41b3546fb3940342a2162dd9ee93699c4`。

| 処理・行 | 対応／注意点 |
|---|---|
| `raw_reduced_masses` 19–27、`upper_normalizer` 35–41 | M8。raw語が既にreducedである質量のDP。返す区間はUのenclosureで、B_RAそのものの両側enclosureではない |
| `FullReturnGenerator.__init__` 216–225 | M3/M6。`B`はordinary envelope。Uは別に保持。全B_RA表を必要としない |
| `FullReturnGenerator.sample` 237–251、`event` 230–235、`_event` 79–90 | M6/M9。order→IID labels→非reducedならNone→受理coin→child→補正weight。rejectをreduceして他parentへ流用しない |
| `CanonicalGenerator` 102–212 | ordinary/partial等の比較経路。G6固有の結果と混同しない |
| `budget` 266–284、`log640_upper` 258–263 | M7/M8/M10。digital moment/range/bias、ceil予算。SOURCEのαは1/320固定で任意confidence APIではない |
| `DeterministicBits` 59–72 | SHA counter由来の再現用stream。fresh-IID測定や乱数品質の証明ではない |
| `g7_provider.controlled_event_ir` 2–20、`provider_contract` 23–34、`bind_synthesized_provider_ir` 37–58 | M12。符号、parent phase、順序、helper compute/uncompute、scalarを含む実adjointの保持 |

helper構成は既存GPT G6レビュー§9とG7のIRに対応する。exact controlled-Qを条件とし、rotation部分で2 provider calls、wordで別途|u| calls、helper用H4回、central controlled-Rz（実装は2 Rz＋2 CX）を含む。外側controlに加えてhelperが必要。providerのstrict誤差・T/CX/state preparationを無料にはできない。SOURCEはevent記述・条件付きIRであり、任意Qをnativeへ安価に下ろすcompilerではない。逆回路では位相scalarも反転する。

## G9 P5：特化した別経路

固定SOURCE `0ef2b92738750a3c0d187743dd4b7c9b927db802`。

`P5Closed.__init__` 11–45はroot/長さ2/長さ4の群を作り、`pair_scalars/level2/parent_coefficients` 50–74、`word_law` 76–86、`event/sample` 88–123へ接続する。Green kernelを呼ばずP5を群別生成する。群数と算術setupの既存O(L²)記述は、selector、sort、任意精度整数、native取得までの全費用を意味しない。

`reference_events` 125–135、`independent_raw_coefficients` 138–151、`audit_p5` 154–169は小fixtureのreference側。production `sample`からreference列挙へ進まないことを既存testが確認した。ファイルに`itertools`がある／ないだけでproductionの非列挙性を判定しない。

P5数式経路は汎用Green queryとは異なるが、root/dyadic helperとG7のevent記述を共有する。**完全に独立した全software stackによる再現とは呼ばない。** P5で群normalizerを取得できることを、一般mのB_RAを無料取得できる証拠へ拡張しない。

## 既存test・結果の正確な利用範囲

以下は**当時の保存記録の転記**。今回のtest実行は0件。固定SOURCEとのhash確認は証拠来歴の管理検査であり、数値correctnessの再検証ではない。raw科学配列・係数の再集計は行わない。

| 記録・既済件数 | 検証範囲と境界／negative例 | 利用上の制限 |
|---|---|---|
| G6 `focused_tests.json`：16件、`G6_FORMAL_CHECKS_PASS` | raw独立reduce oracle、一般形式係数、signed mean、m=3、x=0、L=1、invalid domain/query、insertionは等号でない、raw再reduceの二重計上、accepted-only分母の誤り、root、full-support bit map、precision不足、digital L1 | off-domainの有限形式fixture。3965 raw語、3794級数係数、503正値語、178挿入上界、170 parent envelope、12 signed mean、63 digital events、L1 budget1件は既存ログ内のcount。一般定理の代用ではない |
| G7 preparation `focused_tests.json`：18件PASS | raw-reduced DP独立oracle、B_RA≤U、m3 chain、quartic、各armの同じfinite mean、digital moment/range/support、familywise予算、log640、prequantum zero、helper reset、strict phase、実adjoint、launch/marker境界 | local source-bound record。small formal/operator testで、一般providerの実機動作や量子統計samplingの検証ではない |
| G8 preparation：17件PASS | on-demand acquisition/cacheとstub native backend | synthesis0、quantum0。一般Q/native取得性能の実測証拠にはしない |
| G9 original preparation：23件PASS | off-domain2/3-label・L1、Green係数との一致、非列挙sample、各weight、群構築、physical exampleのphase/order、CTS mean、inverse、zero-T、confidence、import guard | P5・指定provider例に限定。source helper共有の限界あり |
| G9 v2 API preparation：19件PASS | epsilon adapter/strict mode、synthetic direct-child・marker・source fixtures | 旧23件はsource不変の既存PASSを継承し、**この19件と同時に再実行していない**。API修正履歴と科学結果を区別 |

個々のtest名・固定pathは[source inventory](source_inventory.json)、記録のbytes/SHA256と公開固定版は[evidence index](evidence_index.json)。G6のtestでは小さいfixtureを全列挙するが、そのoracleをproduction samplerの入力としない。G9 v1のtechnical failure、v2修正、旧STOP/one-shot消費は別履歴として残す。失敗を成功へ上書きしない。

## 有限bitで閉じた部分／閉じない部分

閉じた既存構成は、正の有理p、short-step、十分なbit数という条件の下で、isqrt enclosure、実dyadic law、πW=c̃、係数bias/range、Uを使った保守的budgetまで。固定fixtureのH/K成功を任意入力へ一般化しない。

不足は、一般入力への自動H/K selector、angle enclosureの実装接続、物理bit/measurement interface、任意controlled-Qのstrict error/cost、online合成/cacheまで含む実測である。angleの有限精度処理可能性は既存監査§6にあるが、SOURCEが返すのはsymbolic tangent。数学的existence、bit complexity、logical query model、native gate列、実時間、総費用を別々に記述する。

G7/G8/G9/G10、D1〜D3、S4の保存資源結果は、特定のprovider・toy・精度・契約下の実装可能性と限界を示す補助証拠にとどめる。本Packetで科学値を再転記して一般化しない。G10固定辞書closure、D＋E、Parent限定成果、全旧STOPを維持する。
