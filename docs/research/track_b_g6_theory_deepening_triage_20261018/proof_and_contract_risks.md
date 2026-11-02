# 証明・実装契約の未閉鎖点

2026-10-18 JST。**今回、既存定理の反証や新しい反例を認証していない。** 下表は固定本文とSOURCEの静的照合。`CONFIRMED_SCOPE_LIMIT`は明示された限定、`CONFIRMED_API_GAP`は呼出仕様の不足、`UNESTABLISHED_EXTENSION`は新しい範囲への未保証、`QUESTION_ONLY`は未解決命題であり不整合確定ではない。

| ID / 確度 | 条件・出典 | 確認したことと影響 | 必要な処理・承認境界 |
|---|---|---|---|
| RK01 / CONFIRMED_SCOPE_LIMIT | L=1,m=1,x=1。MATH §§1,4–8、C01、G7C | 一般証明の域内。L=1のGreenにtransienceは不要。m=1でUとnormalizerは等号。x=1も正値域に含む。境界だから未証明と書かない | 既存記述の参照で足りる。境界testの追加を研究課題にしない |
| RK02 / CONFIRMED_API_GAP | x=0。K 138–146と214–224、G 216–251 | Kのconstructorはx=0を受け、digital_parentはidentity別経路として拒否する。G7 full sampleには明示的なidentity返却分岐がない。主定理の0<x≤1との矛盾ではなく、identityを含む公開APIの契約が別途必要 | 呼出側identity処理の有無・責任を仕様化するだけ。今回は修正・実行しない。identity API整備を新定理としない |
| RK03 / CONFIRMED_SCOPE_LIMIT | domainとsupport。K 162–202、G 103–111 | valid parentはreducedな偶数長≤m−1。非空ならchild i≠first(u)。数学上のpure-parent規約とG7実装domainを区別。CanonicalGeneratorはm≥3を要求し、FullReturnGeneratorと同一domainではない | 全APIに同じdomainを強制せず、各armの用途を記録。s=0処理の記載差だけで域内バグと断定しない |
| RK04 / UNESTABLISHED_EXTENSION | 正値性。MATH §§3–4、BIT §5 | 挿入coverは上界。正値の十分域は0<x≤1、正p。域外で偽だという反例・最大正値域は今回未確認。p_minが小さいことは域内正値性を失わせる根拠ではない | 域外延長を採るなら新しい理論課題。現証明の修正や反例探索は今回行わない |
| RK05 / CONFIRMED_SCOPE_LIMIT | precision。BIT §5、K 45–110,214–236 | 必要bit数の多項式的存在は既存。Kは絶対root幅、Hはdyadic粒度を支配し、相対誤差にはsupport下界が関係する。固定256/160が全入力を覆う保証はない | 既存下界→各局所検査という条件表を作れる。新しい全入力selectorの設計・実装、追加の非自明lemmaは別承認 |
| RK06 / CONFIRMED_API_GAP | fail-closed。K 55–94,222–228、G 216–251 | order/labelは初期化時、parent root/accept/childは試行中にも精度不足で例外になる。十分精度の存在と固定設定の成功は別。例外をX=0として続行する契約はない | 精度不足・入力不正・確率的rejectの3状態を分離する。任意例外をzeroに読み替えたconfidence保証は未成立。retry方針を今回考案しない |
| RK07 / CONFIRMED_SCOPE_LIMIT | meanと誤差。BIT §§3–6、N03 §3 | πW̃=c̃は近似係数への恒等式。proposal丸めの補正はroot係数誤差を消さず、M̃とMは異なる。DF floatのFraction化も物理入力誤差を消さない | 入力→finite M→digital M̃→実装→測定の誤差出所を分ける。Taylor/PF/RPE誤差の統合は別task |
| RK08 / CONFIRMED_API_GAP | confidence。G 266–284、G7C finite-bit節 | α_axis=1/320と16軸の契約。任意α APIではない。η/ρ、primitive_error、残余s>0の条件と共通factor-2 bias規約がある | 任意confidenceへの拡張は仕様・log enclosure等の接続が必要。SOURCEにない一般α実装を達成済みにしない |
| RK09 / CONFIRMED_SCOPE_LIMIT | 算術・bit。BIT §2/5、K/G | 既存算術countと係数bit長は支持される。dyadic_indexはlawの和・dyadic検査・線形走査を含み、roundingはsortする。mについてのpolynomialはlog mについてのpolynomialとは別 | input総bit長、次数m、η/ρ/τ、modelを明示。既存の一般polynomial論証を否定せず、具体的なwhole-sampler上界は追加作業として区別 |
| RK10 / CONFIRMED_SCOPE_LIMIT | moment・受理・budget。MATH §8、G7C、U §7 | A_uはvalid-parent条件付き受理率、Zはideal全trial受理率、digital acceptanceは別。U区間はUのenclosureで、B_RAの両側enclosureではない | N、期待NZ、hard cap N、event価格、reject古典費用を別々に記録。U.loをB_RA下界や実受理率に代入しない |
| RK11 / QUESTION_ONLY | full/partial。G7Cの三次節、P5 11–123 | m=3のnormalizer順序・四次展開は既存。一般mへ同じ順序を延長する証拠はない。P5にはGreenを使わない既存closed生成もある | 一般比較命題の成立／不成立はC2の後続理論課題。m=3再証明、m=5の弱いpartialだけとの勝敗、保存値の再ランキングで代用しない |
| RK12 / CONFIRMED_SCOPE_LIMIT | controlled provider。CP 2–58、BIT §6、N04 | exact CQ、actual adjoint、scalar、順序、helper resetの条件付きIR。Kはtangentを返し、一般native angle enclosure接続はない。2 provider callsは回転部分で、word分は追加 | θ近似、CQ誤差、Rz合成、workspace、prep/readoutを区別。general Qの無料controlled化・provider誤差0の物理的認証は主張しない |
| RK13 / CONFIRMED_SCOPE_LIMIT | RNG/測定。G 59–72,237–251 | uniform independent bitsの法則とSHA-counterの診断traceは別。event/None APIは実測定器・accuracyの実証ではない | 固定bitstream traceからconfidenceを認定しない。後続bit/測定interface実装には明示的指示、科学samplingには別実行承認 |

## 精度仕様で直接引ける条件／追加研究になる条件

既存BIT §5の下界は、order≥t_(m−1)/3、label≥p_min、valid-parent acceptance>1/4、存在するchild≥p_min/2、d_u≥t_(m−1)p_min^(m−1)/2。これらは**既存の保証の引用**である。Kのroot interval、midpoint、largest remainders、acceptance下向き丸めに対し、positive endpoint、relative-error、supportの検査が必要なことまではSOURCEから直接仕様化できる。

H/Kを増やすべき理由や必要な入力（p_min、x、m、η、ρ）は既存証拠にある。具体的な十分精度の定数を全APIの検査へつなぐ作業は未実施。そこで既存lemmaの代入だけで足りるなら通常の技術仕様化であり、「新しい一様精度存在定理」と数えない。新しい誤差伝播・停止・複雑性命題が必要な場合は、その命題を切り出して別承認へ返す。可変精度で試行を続けるsamplerを本調査で設計しない。

angleについてもBIT §6にatan enclosureの有限精度処理可能性とstrict誤差への接続がある。未実装を未証明の存在問題へ読み替えない。一方、任意物理providerの誤差、元Hamiltonianの入力誤差、長時間PRへの一様保証は別の仮定・研究を要する。

## 候補の「負結果」の正確な範囲

固定H/Kの例外、特定providerでの高費用、ある比較不等式の不成立、先行手順への還元は、それぞれ異なる結果である。前二者は既存M1〜M12を反証しない。後二者も、証明対象の保証・入力・費用条件を限定して初めて意味を持つ。今回それらの新しい結果は導出していない。

固定出典：[MATH]、[BIT]、[K]、[G]、[CP]、[G7C]、[U]、[P5]、[C01]、[N03]、[N04]。各表の節・行番号はこれらの固定版を指す。

<!-- FIXED_REFERENCES -->

[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[N03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/pr_rte_connection_and_access.md
[N04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/controlled_native_feasibility.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
