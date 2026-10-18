# W2a：次に証明・反証する問いを1件へ絞る

2026-10-18 JST。**提案は、m≥7で既存Uが二つの明示closed benchmark以下になるか、という一つの一様比較命題。真偽は今回判定していない。** 理想Fullの改善と、利用可能なUの改善を区別するための問いである。[定義表](baseline_normalizer_definition_matrix.md)の記号・形式P5⊕tailの未実装境界を前提とする。

## 既存結果から直接言える範囲

| 内容 | status・根拠 | 今回の境界 |
|---|---|---|
| 一般奇数m、0<x≤1の正値性、`B_F,m≤U_m≤B_o,m` | `PROVED_IN_EXISTING_SOURCE`。M §§3–8、GC U節、U §7 | 一般Full/partial/closedの順序はこのchainだけでは決まらない |
| m≥3,x>0で`U_m<B_o,m`。L=1で`U_m=B_F,m` | `PROVED_IN_EXISTING_SOURCE`。GC/U §7 | strictなordinary改善を強いclosed後の改善へ移さない |
| m=3,L≥2で`B_3,3=B_F,3<U_3<B_P,3<B_o,3` | `EXACT_FIXED_ORDER_RESULT`。GC三次節、U §§5/7の既存記述 | m=3を再証明・再計算しない |
| L≥2、同じp/xで`B_3,m<B_P,m<B_o,m`（奇数m≥5） | `ROUTINE_DEFINITION_ALIGNMENT`。上の既存P3不等式の各辺へ**同じ**T_(4,m)を加えた直接対応。Gの両armが`_ordinary(4)`を呼ぶ | 新しいFull一般比較ではない。Partialだけを唯一の強い対照にしない |
| m=5で`B_5,5=B_F,5` | `EXACT_FIXED_ORDER_RESULT`。PCの同じ理想ensemble、P5の係数・群定義 | Full固有のideal改善は消える。digital・proposal・取得費用の同一性ではない |
| m=5で`B_5,5≤U_5` | `ROUTINE_DEFINITION_ALIGNMENT`。既存`B_F≤U`への固定次数一致の代入 | `U_5≤B_5,5`を問うなら上界の等号条件を問うことになる。strict gapを今回新たに証明しない |
| m≥7の`B^form_(5⊕o,m)=B_5,5+T_(6,m)` | 数学的有限和は`ROUTINE_DEFINITION_ALIGNMENT`。既存統合sampler／digital契約は`NOT_ESTABLISHED` | 既存の完成済みP5+tail方式として費用・予算を使わない |

上の直接加算・代入以外に、新しい不等式・係数・漸近展開を導出していない。P5の対応は既存独立導出の引用であり、有限testを一般証明の代用にしていない。引用範囲は[GC]、[U]、[PC]、[P5]、[G]、[M]。

## W2a-1〜3の未解決点

| 区分 | 問うこと | 不足とstatus |
|---|---|---|
| W2a-1 理想分解 | 一般m≥5でB_F,mとB_P,m、B_3,mの順序・strict条件・等号・逆転はどうなるか。m≥7の形式P5⊕tailに対してはどうか | `OPEN_COMPARISON_PROPOSITION`。既存の三次式やP5一致を一般mへ延長する証明は未確認。追加の非自明な一般論証に進むなら`REQUIRES_NONTRIVIAL_PROOF`として別承認 |
| W2a-2 認証上界 | 同じ入力でU_mが各対照の理想normalizerを下回るか。Fullの小ささがUに反映される条件は何か | `OPEN_COMPARISON_PROPOSITION`。`B_F≤B_closed`だけでは`U≤B_closed`は出ない。Uが全域最良とも仮定しない |
| W2a-3 取得・認証 | 局所root、r_l、closed群係数と外向き区間だけで比較できるか。全B_F/全辞書を要求しないか | UとP3/P5固定式の取得経路は既存。任意入力の有限精度で比較を決着する条件は`ACCESS_OR_CERTIFICATION_GAP`。H/K・API・bit費用はW1担当。P5⊕tailの統合SOURCEは`NOT_ESTABLISHED` |

B_F,mは一般に全parentの和として定義され、局所queryがあるだけで真の全和が安価に得られるとはしない。U_mとclosed式は有限のroot/群情報から外向き区間を取得する経路を持つが、演算・input bit長・要求精度を無料としない。運用で大小を認証するには、例えば**取得済み**U上側と対照下側の分離等が別途必要で、理想値の一様命題の真偽と同一ではない。新しい精度選択法は今回作らない。

## 選ぶ候補はP_Uの1件だけ

**P_U（未解決・未採択）**：有限L≥2、すべての明示正有理pでΣp=1、すべての有理0<x≤1、すべての奇数m≥7について、

\[
U_m\ \le\ B_{\rm ref,m},\qquad
B_{\rm ref,m}:=\min\{B_{3,m},\ B^{\rm form}_{5\oplus o,m}\}
\]

が成立するか。

ここでB_refは**二つの明示されたscalar式の小さい方という数学的benchmark**であり、新sampler、無料選択oracle、全既知法中の最適値ではない。P5⊕tailは形式和としてのみ含め、既存統合方式が完成していると仮定しない。この区別を失った場合はP_Uの解釈を停止する。strict不等式・最良定数・入力領域の拡張を同時に証明する課題へは広げない。

| 判断項目 | P_Uの具体的内容 |
|---|---|
| 1 量化範囲 | 上記全有理入力、L≥2、奇数m≥7。σ±と一般Qは同一target条件として固定するが、これらscalar式はp/x/mに依存。x=0はidentity、m=3は既知、m=5は固定次数一致の整理だけ |
| 2 比較方式 | Fullの既存U、既存Closed P3＋ordinary tail、形式Closed P5⊕ordinary tail。Partial/ordinaryも定義表に残し、より強いclosedを落とさない |
| 3 同じtarget | Full/Closed P3＋tailは既存P_m。P5⊕tailは有限Taylorの直接恒等式でtargetを合わせた**形式比較**。統合生成・digital保証は未確立のまま |
| 4 使える既存証拠 | Mの正値envelope、GC/Uの`B_F≤U`とraw-reduced DP、GのP3 tail、PC/P5の固定次数係数・normalizer。一からのGreen/84セル調査なし |
| 5 未解決の主張 | P_Uというweak不等式の一様な真偽のみ。厳密な成立域・等号分類・漸近係数は今回の次命題に含めない |
| 6 未取得情報 | P_Uの一般証明または許可域内での認証された否定根拠。形式P5⊕tailを既存方式として解釈するための統合SOURCE/保証も未取得。W1未公開結果は使用しない |
| 7 肯定の意味 | 既存Uが、両closed式の小さい方まで理想係数質量の上側比較を保つという条件付きの数学的意味。B_F≤Uから理想Fullにも同じ比較が伝わる。実装・N・資源優位は別 |
| 8 否定の意味 | このUを使う**一様な**closed後の比較は成立しない。理想B_Fの改善も否定された、Full構成が偽、全入力で費用NO-GO、という結論にはならない |
| 9 研究結論への影響 | 肯定なら一般次数で利用可能なUを強いclosed benchmarkへ結び付けられる。否定ならUによる主張を入力条件付きへ限定し、理想改善と認証の乖離を明確化できる。どちらも単なる小定数改善以上の範囲判断 |
| 10 次作業・承認 | ユーザーがP_Uを採択し、対象限定の新しい証明／反証作業を明示承認した場合だけ着手。科学レビューを開くなら別の開始承認。数値反例探索・SOURCE/test実行は別の科学実行承認。今回は全て0件 |
| 11 投資STOP | 既存定理の直接適用だけで解決済みと判明したら新成果として再証明しない。形式benchmarkを完成済み方式と扱う必要が出たら不足を返す。弱い対照への自動交換、未知B_F/Zの利用、範囲外入力・新samplerが必要なら停止。否定後に有利な別入力を自動探索しない |

P_Uを優先するのは、理想normalizerだけでなく**現在の認証上界が強い比較へ届くか**という結論を変えるからである。取得不能な真のB_Fを比較oracleとして使わず、m=5の固定一致を越えるm≥7に限定できる。これは真であるとの予想、独立新規性、研究方針の正式採択ではない。

## W1・W2b・W2cへの引継ぎ境界

P_Uの理想的な数理定式化・証明／反証の検討自体はW1完了を待つ必要がない。有限bitの上下区間で大小を認証する精度・API、root/selector/bit費用はW1の公開・固定された成果が必要であり、未commit内容を取り込まない。

P_Uの肯定・否定がW2bへ変えるのは「どのnormalizer上界を比較に持ち込めるか」という前提。range・bias・confidence・ceilと全trial Nを保持し、Uの大小だけでNの大小を導かない。W2cでは量子前zero、真のdigital受理率、event価格、全rejectの古典費用が別に必要。未知ZやISの利用機会を省略しない。今回、新しいN、moment上界、期待費用、break-evenを導出していない。

**公開・保全確認後mandatory STOP。** P_Uの証明・反例・漸近解析やW2b/W2cへ進まない。

<!-- FIXED_REFERENCES -->

[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[GC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[PC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
