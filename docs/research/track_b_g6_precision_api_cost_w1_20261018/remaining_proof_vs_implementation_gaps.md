# 未証明・未実装・未認証を分離する

**判定：必須W1条件について、既存lemmaでは足りず新しい非自明定理が不可欠と確認した項目は0件。** 既存の有限bit構成可能性、正値性、support、angle enclosure可能性を新しい未解決問題へ戻す必要はない。W1の差分仕様は完成したが、一般入力APIの実装完成、全処理bit上界の完成、物理認証は未達である。

## 分類した残件

| Gap | 既存保証から直接閉じる部分 | 残る不足・分類 | 解消の性格 |
|---|---|---|---|
| G01 domain/identity | [MATH]§1/5、[BIT]§5のL=1/m=1/x=1とx=0別扱い | K/full/canonicalのdomain・helper前提の記述は`ROUTINE_SPECIFICATION`。G7 identity dispatcherの接続は`IMPLEMENTATION_MISSING` | API仕様・通常実装。新しい境界定理ではない |
| G02 一様精度のAPI接続 | [BIT]§5の一様support下界・十分bit数の構成可能性。Kの各guardの数学的意味 | 具体的なH/Kと同じ全到達lawのguard適合を事前に結ぶ出力/証拠が未接続：`IMPLEMENTATION_MISSING`。固定K/Hで全入力成功は未主張 | 既存lemmaの適用・仕様・実装。今回新しいH/K定数/selectorは作らない |
| G03 停止/zero/完了confidence | [MATH]§8、[G7C]の固定N・既存統計保証 | 例外をzeroにしない原因分類と事前前提の記録：`ROUTINE_SPECIFICATION`。全law精度の接続はG02。途中failureの完全confidenceは未保証 | 新failure budgetや条件付き成功定理を採る必要は現scopeにない。既存前提を満たして使う |
| G04 budget前提 | [U]§7、[G7C]の既存moment/range、固定α・factor規約 | η/ρ/primitive_error/残余幅/入力identityの包括的契約は`ROUTINE_SPECIFICATION`。任意α APIは`IMPLEMENTATION_MISSING`だが今回の必要条件ではない | 固定confidenceを維持。新しい最適Nや一般confidenceの証明を要求しない |
| G05 angle | [BIT]§6の既存enclosure可能性、τ→strict回転誤差 | 有理ratio→angle enclosureのSOURCE接続：`IMPLEMENTATION_MISSING`。native sequence/strict証拠は別 | 通常の有限精度実装の候補。未実装を角度存在の未証明としない |
| G06 全処理bit費用 | [BIT]§2/5/6の算術count・係数bit長・精度/angle多項式性 | 各費用所在の仕様は`ROUTINE_SPECIFICATION`。この固定SOURCE全体の具体的bit費用関数、peak memory、実秒は`NOT_VERIFIED` | 標準算術・integer isqrt・sort等の費用モデルの明示と追加の複雑性解析。**追加解析が未実施であることだけでは非自明な新理論の必要性を確定できない** |
| G07 provider | [CP]のexact CQ条件、phase/actual adjoint/workspace/回転呼出数 | 実物理/nativeの単価・strict errorは`PHYSICAL_PROVIDER_UNCERTIFIED`。実取得は`OUT_OF_SCOPE` | [B-COST]の既存監査を引用。DF bindingや分子回路の再監査は行わない |
| G08 fresh bits/実測 | 数学的fresh IIDを仮定したbit-map/mean/budget | production bit/measurement/全trial accumulator接続は未完成。G6/G7のbits契約は`IMPLEMENTATION_MISSING`、実backend等は`OUT_OF_SCOPE` | [B-COST]の受入条件へ委譲。固定SHA診断をIID証明にしない |

## なぜ「REQUIRES_NONTRIVIAL_PROOF」を機械的に付けないか

1. 全parentに対するsupportの一様下界と有限precisionの取得可能性は既存[BIT]§5にある。有限fixtureから新しい一般保証を外挿する必要はない。
2. root区間、midpoint、normalized interval、dyadic guardの条件はSOURCEから直接読める。下界をこのguardへ適用する仕様・具体値接続が未了だからといって、別の存在lemmaがないとは言えない。
3. angleも既存[BIT]§6にenclosure可能性がある。SOURCEがsymbolic ratioを返すだけなのは実装の不足である。
4. 全費用の具体的な上界はまだ記録されていないが、既存poly構成可能性を超えるどの数学命題が必須かは確定していない。現在のコードのloop、sort、isqrt、Fraction費用の追跡は、まず標準的な仕様化/複雑性解析として扱うべきである。新しい上界やlower boundを本監査で創作しない。

したがって、今回の義務索引には`REQUIRES_NONTRIVIAL_PROOF`の確定行を置かない。単なる「全time未計測」「selectorがない」「精度が不足し得る」を新理論の根拠とすることは避けた。**W1に関する新理論研究は現段階では不要。通常仕様と実装・費用会計を独立の新定理として膨らませずSTOPする。**

## 今回閉じていない強い主張

次は既存lemmaや本監査の結論ではない。採用するなら対象を新たに特定する必要があり、今回は真偽・証明・研究価値を判断しない。

- 任意の正有理pに対しSOURCEの固定default H/Kが無例外で成功するという主張。既存資料はその量化をしていない。
- mをbinaryで与えた入力長のみに対する全処理多項式時間、又は精度のbinary記述長だけに対する全N実行時間の主張。既存countの変数/出力量とは異なる。
- 精度failureをzeroや再試行へ変えた新lawの無条件confidence、完走runへ条件付けても同じconfidenceが保たれるという主張。既存契約の直接適用ではない。
- 同じaccess/精度で強い既知対照より全bit/量子費用が優れるという一般比較。これは未実施のW2/C2等に属し、W1の「poly構成可能」と同義でない。
- 任意のsystem-only Qから、絶対位相を保持したcontrolled providerを無料又は一定のnative費用で得るという主張。既存oracle条件を越える。

これらを現実装へ新しい必須要件として追加していない。将来、特定の数学的主張を求める場合は、量化入力・費用モデル・変更する停止規則・既存lemmaで不足する箇所を限定してユーザーへ返す。本監査には新たな非自明命題の証明、H/K式、sampler、angle/source実装はない。

## 残る不確実性

全費用の追加解析が標準的適用で完結するか、具体化の過程で別の量的lemmaが必要になるかは未検証である。「新定理が絶対に不要」という普遍的判定もしない。現在の証拠で確定できるのは、既存存在保証があること、直接仕様化できる条件、SOURCE未接続箇所、追加会計の対象までである。

独立新規性、論文化、資源優位の科学的受理は扱わない。B/A Packet、D＋E、G6〜G10、Parent、DF/Gaussian等の旧状態を変えず、**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/measurement_and_cost_contract.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[CP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
