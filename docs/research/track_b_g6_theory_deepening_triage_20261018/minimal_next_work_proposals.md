# 優先候補の最小の次作業案

2026-10-18 JST。**以下は提案であり、新しい実行contract、grant、研究採択、証明・実装・計算の開始ではない。** 今回の調査を公開してSTOPした後、担当へ明示指示された範囲だけを進める。

## W1 / C1：既存lemmaからAPI・費用への接続で残る穴を固定する

**担当候補**：Codex（静的仕様調査）。新しい非自明な保証の採否はGPT、研究範囲の正式採択はユーザー。

**入力・固定資料**：正の有理p tableとその総bit長H_p、p_min、有理xとH_x、奇数m、0<x≤1、η/ρと要求angle誤差τ。実際の入力数値・精度値は次の指示で固定し、今回は選ばない。原数学MATH §§1,3–8、BIT §§2–6、K 45–110/138–254、G 216–284、CP 2–58、既存M9〜M12を使う。m=1、L=1は既存域内、x=0 identityは別項。CQは既存のexact条件付きqueryとし、物理provider・DF入力は持ち込まない。

最小の作業は**差分仕様1冊**にとどめる。候補H/Kの公式、selector、適応的refinement手順をこの調査で新設計しない。

| 対象 | 次に記述する具体的な確認事項 | 既存記述から引ける部分 | 新作業へ進む境界 |
|---|---|---|---|
| 入力/domain | 正規化・零label除去、identity、armごとのm条件、mの符号化と費用変数 | K/G validationとMATH §1 | 入力仕様を変える必要が出たら返す |
| order / labels | root区間からnormalized intervalへの条件、positive support、実dyadic lawの検査 | K 55–82,208–212、BIT §5の下界 | H/Kの十分定数を新たに保証するlemmaが必要なら未解決と記録 |
| parent / child | d midpointのrelative誤差、accept区間、child support、fail-closedの区別 | K 214–236とBIT §5 | 精度不足例外をzero試行へ変換する等の新しいlawは作らない |
| mean / budget | 実π、c̃、weight、係数bias、range、moment、残余s>0、confidence domain | K 239–254、G 266–284。α=1/320の既存規約を明記 | 任意αや異なる誤差規約は別の仕様拡張。元Hamiltonianへの保証を付けない |
| angle / provider | tangentからangle enclosure、strict Rz誤差、actual adjoint、phase、helperの各仮定 | BIT §6、CP。理論的なenclosure可能性は既存 | angle計算、Rz合成、CQ誤差の実取得・実装は別指示。未認証誤差は0にしない |
| setup / online / bit | law検査・線形走査、rounding sort、isqrt、有理算術、packet、reject、word、cache・一時memoryを呼出単位に対応 | 現SOURCEの操作列とBIT §2/5 | whole-samplerの新しい具体上界・下界を要するなら命題だけ返す。新selector/caching法を設計しない |

**想定成果物**：`precision_api_obligation_map.md` と、各義務の出典・入力変数・充足条件・未閉鎖理由の機械索引。作成先は次指示で決める。判定は「既存lemmaの直接適用」「新しい非自明証明が必要」「実装不足」「物理取得不足」の4種。数学存在と実装の不足を混ぜない。

**GO条件**：未閉鎖箇所ごとに、既存lemmaで閉じるか、新しい保証を要するか、どの費用に影響するかを一つずつ特定できること。新証明・実装へ進むGOではない。

**不成立／STOP条件**：全てがroutineな仕様化なら、仕様完成を報告してC1の独立した新理論投資は止める。global辞書・未知normalizer・未取得native情報が必要ならその依存を返す。既存保証の不整合が疑われれば位置・影響・必要確認だけを記録し、修正定理や反例コードを作らない。

**承認**：W1の静的調査は明示指示後のCodex通常作業。H/K selector、angle実装、科学testの追加／実行、sampling、保存値再解析、compileには別の明示許可が必要。新しい非自明な証明に入る場合は対象命題の承認を別途受ける。

## W2 / C2：一般次数の比較を三つの別命題へ分ける

**担当候補**：Codexが既存式・情報条件の差分表を作り、GPTが新しい比較命題の科学的範囲を判断し、ユーザーが取り組む命題を採択する。新たな科学レビューを開催する場合は明示的な開始承認が必要。前回の構成価値レビュー全体をやり直す提案ではない。

**入力・比較対象**：同じ正有理p,x、一般奇数m≥5、同じfinite P_m、coherent X/Y、controlled条件、誤差・confidence規約。対照はG7のordinary、partial-return＋ordinary tail、closed-P3＋ordinary tail。m=5の条件を扱うなら既存P5Closedを必ず含める。比較側にもGreen query、丸め、合法cache、IS、CTS layeringの利用を認める。新しいbaseline samplerを設計せず、該当する既存方式・不足条件を記載する。

必要資料はG7CのU／三次／finite-bit節、G 19–56/102–251/266–284、U §7、P5 11–123、A03のB2/B4/B5、A04のaccess層。三次の一般証明や84セルの文献調査は繰り返さない。P5Closedは別の既存研究SOURCEであり、全く独立したsoftware stackという意味ではない。

| 最小の問い | 何が既知か | 後続に残す具体的な未解決命題 |
|---|---|---|
| W2a normalizer／認証上界 | B_RA≤U≤B_o、三次の順序、L=1/m=1の特殊例 | 一般mでUと指定partial/closed normalizerの順序・等号・比較不能になる条件は何か。単純な全域順序が不成立なら、その限定が何を意味するか。今回は真偽を判定しない |
| W2b 十分全trial数 | fullのm2_plus=κB_o.hi U.hi、canonicalのκB_arm.hi²、各rangeと残余s、ceil N | 同じ認証誤差条件でNの比較が成立する条件は何か。U<B_partialだけではN比較の結論を出さず、range、残余幅、ceil、intervalを保持する |
| W2c 実費・成立限界 | 量子前zero、digital acceptance、word/child provider callsと取得項目 | 全trialの古典費用とevent依存量子価格について、取得済み上界だけで判断できる条件は何か。未知Zや全word価格表を要求するなら未確立とする |

費用の記法は既知の期待値会計として、方式aについて `C_cl,a=C_setup,a+N_a E[c_cl,a/trial]`、`C_q,a=N_a E[1_accept c_q,a(event)]` と**別の成分**にする。CPU時間とT/CXを無換算で足さない。uniform単価の場合だけ量子回数はN_aとdigital acceptanceの積になる。これは比較結果・新上界ではなく、何を取得しなければならないかの記法である。N_aは既存の認証済み十分trial数で、経験varianceで減らさない。

**想定成果物**：`general_order_comparison_question.md`。各問いを、量化入力、同一target、許可access、使用する既存式、未知情報、肯定・否定で変わる判断の順に1ページ程度へ限定する。続いて扱う命題は最大1件を提案し、証明・反例・新アルゴリズムは含めない。

**GO条件**：解決すればnormalizerの小改善を超え、認証予算・費用成立域・明確な限界の少なくとも一つを変える命題を特定できること。新規性や資源優位を保証するGOではない。肯定結果と否定結果の双方を受け入れる。

**不成立／STOP条件**：三次の再記述、P5Closedを落とした弱い比較、既知ISへの直接書換えだけ、小定数しか変わらず判断を変えない場合は独立課題化をやめる。未知B_RA/Z、経験的に有利な保存行、未認証のnative価格が必要なら不足を返す。負結果後に別入力探索や新分割へ移らない。

**承認**：命題案の資料照合はCodexへの次指示後。一般次数の新証明、漸近解析、下界・反例導出は別の理論作業の承認が必要。新SOURCE・test・数値探索・保存値再集計・合成・分子計算はそれぞれ別承認。今回の資料は実行許可書ではない。

## 提案する順序

最初はW1の差分仕様だけ。既存precision保証を再証明せず、全処理へ接続するときの不足を限定する。W2aの問いはC1完了を待たず記述可能だが、W2b/W2cの費用結論はC1で確かめた範囲に依存する。新定理、adapter、native性能取得へ自動的には進まない。

固定出典：[MATH]、[BIT]、[K]、[G]、[CP]、[G7C]、[U]、[P5]、[C01]、[A03]、[A04]。今回、新しいSOURCEや一次文献を生成手順として補完していない。

<!-- FIXED_REFERENCES -->

[A03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/known_method_composition_audit.md
[A04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/access_precision_complexity_comparison.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[C01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/mathematical_claim_evidence_matrix.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
