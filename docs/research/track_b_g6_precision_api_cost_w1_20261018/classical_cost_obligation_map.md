# W1：setup・online・bit・rejectionを含む費用の所在

2026-10-18 JST。新しい全処理計算量定理、実測値、shot forecast、性能順位は作成しない。以下のO記法は全て[BIT]§2/5、[U]§7.1–7.2、[G7C]からの**既存評価の引用**。その他はloop・演算・データ寿命を静的に記述するだけで、新しい漸近上界を付けない。[主表C01〜C03](precision_api_obligation_map.md)

## 費用モデルに入れる変数

L、次数m、p tableの総bit長H_p、有理xのbit長H_x、η/ρ/angle精度τの記述とlog逆精度、root K・probability H、全trial Nを別にする。mをbinaryで渡す場合、既存のmについての多項式はlog mだけの多項式ではない。η/ρ/τを浮動小数点精度と同一視しない。lawの要素数と各有理数のbit長の両方が必要。

「有理1演算」を単位費用にするモデルと、分子/分母の整数積・division・gcd・比較・allocationを数えるbitモデルを分ける。Python Fractionの演算/正規化、isqrt、sortをCPU秒へ変換する実測やruntime保証は今回ない。input readの費用は除外しない。

## 操作と未取得費用の対応

| ID / 項目 | 既存評価／SOURCEから読める実処理 | Setup/onlineと保存 | 閉じる部分／未取得部分・分類 |
|---|---|---|---|
| K01 input | [BIT]§1のp読込Ω(L)。K139–146、G107–113は全pのFraction化、正値・sum=1、x/domain、χ。m/H/Kの型契約は[精度文書](precision_and_failure_boundaries.md) | setupでp table保存。変換時の一時整数・比較も必要 | 読む対象と入力bit変数は`ROUTINE_SPECIFICATION`。実bit演算/CPUの具体上界は未取得 |
| K02 Green | [BIT]§2：O(Lm²)有理演算、O(Lm)係数。K147–160は全n/i/b、total[n]の再利用。GとFを次数mまで保存。直接j≠i二重label loopとは異なる | fullのsetup。canonicalはG108でGreen前計算をしない。array係数数とbytesを区別 | 既存countは`CLOSED_BY_EXISTING_LEMMA`。Python配列・tuple変換のpeak bytesまでは未取得 |
| K03 coefficient/query | [BIT]§2：word O(length(u)m²)、parent全child O(m³+Lm²)。K21–28畳込み、170–202はparent系列を共有して各childへ1回畳込み。K177–192はmassごとのx powers/factorialと有理和 | valid reduced parentだけonline。child massesを局所packetに保持 | 限定算術count/係数bit長は既存。pow/factorial/正規化・コピー等を含む全bit処理をそのcountと同一視しない |
| K04 B_o/U | [U]§7：raw-already-reduced質量DP O(Lm)有理演算、kernel前計算込みO(Lm²)規模。[G]19–41でDPとempty local parent、tail roots。221–225はorder rootsとB rootsを別呼出ししておりroot計算を一括memoしていない | setupでp/kernel、order/label laws、B/U区間を保持。U計算用のDP/tailは局所一時値 | Uを使うための取得処理は確認。root bit費用を含む全setup上界の具体化は`NOT_VERIFIED` |
| K05 root | K45–52：scale=1<<K、value分子×scale²、整数division、isqrt、exact-square比較、Fraction端点。order、B/U、local dとA²に用いる | setupにもonlineにも発生。同じisqrtでもradicandのbit長が異なる | enclosureの正当性は既存。整数bit長は入力/係数/Kと結ぶ必要。isqrtの単価・一時memoryを無料にしない。全呼出し分の明示bit上界は未取得 |
| K06 distributions | K55–82：interval全read、Σlo/Σhi、midpoint和、再正規化、scaledfloor、**sorted**余り、counts更新、全relative-error検査。G `_law`は一点区間を作る | order/labelはsetup。full childはlocal packetごとonline。canonical conditional/root-childはsample/event中にも作る | 操作列は`ROUTINE_SPECIFICATION`。要素数・各Fraction比較/除算・sort keyの費用が必要。constant-time law setupとはしない |
| K07 selector/bit取得 | K97–110は各callでΣlaw、scaledcountsの全dyadic検査、その後累積**線形走査**。G238でorder一回、239でl label draws、246でaccepted child。G243は別のH-bit coin。G59–72はSHAを積む診断実装 | lawを事前保存していても検査/走査が毎draw発生。fresh-bit供給のモデル/費用は別前提 | `ROUTINE_SPECIFICATION`。label1drawを単位oracleとする算術表現と実SOURCE費用を分ける。新alias/index/cacheを設計しない。SHA診断の処理数から理想IID取得費用を決めない |
| K08 rejection | K11–18/G240はstackによるraw reduced判定。[BIT]§2はraw判定O(m)、word/state O(m log L) bits。nonreducedはpacket前None、coin rejectはpacket後None | 両zero経路で異なる古典prefix費用。量子処理はaccepted後という既存モデル | 既存countは支持。A_u下界を全trial Zやdigital acceptanceと混同しない。「成功まで回す」費用モデルへ変更しない |
| K09 local precision/weight | K121–133のs propertyはsumを再実行し、d²/ratio/child_law等から参照される。K239–254はwordの実label massの積、target/πのFraction除算。G249–251はweightを求め、alphaとalpha/weightを再計算し_event87で再びalpha/proposal | onlineでpacket、child tuple/law、root区間、weight/event dictを保持。再計算は実SOURCEのまま記録 | πW=c̃は既存保証だが処理単価0でない。Green係数のbit上界だけでproposal積/weight/後続集計の全bit費用を保証しない |
| K10 canonical対照 | G115–143はgroup生成とroot検査。153–160はconditional/first law。193–212で作ったlawを162–191 event内でも作り直す経路がある。_event82はwordをreduceしてCounterを作る | canonicalはGreenなし。degree/arm/groupごとのsetup/online分離を維持 | `ROUTINE_SPECIFICATION`。Fullの費用だけ丁寧に数え、対照を全語表baselineにしない。同じ最適化機会は許すが今回SOURCE最適化や新しいO評価なし |
| K11 angle/provider | [BIT]§6はtangent上界/atan enclosureの多項式処理可能性、τからstrict回転誤差。K129/G251はratioだけ。CP2–20はword CQとchild compute/uncompute、H/Rz/CX、parent Z。CP37–58は外部row読込・sequence/逆順tokens | angleは一般にword依存、native全key表を無料にしない。CP cacheは呼出側供給で、G6/G7 generatorが自動取得/増大memoするものではない | angle存在は既存、実装`IMPLEMENTATION_MISSING`。query modelの呼出数は確認、physical/native単価は`PHYSICAL_PROVIDER_UNCERTIFIED`。実証は[B-COST]に委譲 |
| K12 memory/全N | G full sample242はpacket一回、247は第2queryを避ける。kernel/order/labelsは固定保存、過去parentの増大cacheはない。単一packetにもall child arrays・root/rounding/temporary整数がある | setup state、1trial一時領域、出力/保存のN依存、providerが要求する外部cacheを別にする | O(Lm)係数は全peak memory boundではない。accepted record全保存の有無は別仕様で、G単体が全N保存/測定を実装してはいない。実装監査はB参照 |
| K13 budget/postprocess | G258–284は固定log級数/positive remainder、有理κ/moment/range/common bias、remaining検査、ceil。全Nの加重outcome処理はgeneratorの外 | 既存budget計算はsampling前。重みの加算・保存・bit growthは全trial後処理の費用として別取得 | budget式は既存、全試行数を求めることと全N実行費用を得ることは別。accumulatorの実装仕様はB対象外参照。新N・confidence・CPU値は算出しない |

## 既存bit保証の正確な範囲

[BIT]§2は共通分母D=lcm den(p_i)、log D≤H_p、係数分母がD^m den(x)^m m!を割ることに基づき、係数bit長 `O(m(H_p+H_x+log m))` とschoolbook exact arithmeticを含む多項式処理を記述する。§5はsupport下界から精度bit数の多項式的存在、§6はangle処理可能性を記述する。

したがって「bit complexityは何も証明されていない」は誤り。一方、現SOURCEの全callにおけるscaled isqrt、Fraction/gcd、sort/比較、selector検査、canonicalの再構築、proposal/weight、angle取得、全N処理、peak memoryを結び、定数/全費用関数まで明示した証拠は今回の固定資料にない。これは**具体的全処理会計の未完成**である。粗いpolynomial存在を再証明する必要や、新しい非自明理論が不可欠であることまでは示していない。

## 全trialと費用座標

Aの[W1/W2作業案][A-WORK]に既にある会計記法だけを再利用する：

`C_cl = C_setup + N E[c_cl/trial]`

`C_q = N E[1_accept c_q(event)]`

前者はbit/CPU等の古典座標、後者はprovider/T/CX/depth等の量子座標であり、無換算に加算しない。これは費用上界でも新しい期待値評価でもない。最大trial費用を用いる保証と、経路分布を用いる期待費用には別の根拠が要る。N自体も精度/誤差/残余幅の関数であり、入力bit長だけに関する全N polynomial-timeという主張をしない。

fullのraw nonreduced zeroにはdraw/selector/reduced検査費用が残り、coin rejectにはlocal query/root/roundingまで残る。accepted eventにはそこからweight/angle/provider/準備/測定が加わる。provider2回は**回転部分**でありword分は追加。G7 contractの量子hard cap Nと、ideal期待NZを区別する。digitalの実受理率は別で、未知Z・全word費用表・経験varianceを無料で使わない。

実backend/state/weighted accumulatorと分子native/PR費用は[B-COST]へ参照を返しただけで再監査していない。今回の成果は費用の所在と不足の固定であり、総資源改善や実行時間を認証していない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[A-WORK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/de709339e2539fa4d9806a84e738040d52c97be1/docs/research/track_b_g6_theory_deepening_triage_20261018/minimal_next_work_proposals.md
[B-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
