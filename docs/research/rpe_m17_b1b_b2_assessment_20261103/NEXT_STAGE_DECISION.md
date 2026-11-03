# B-1b/B-2の必要性と次の担当

2026-11-03 JST。静的調査・数式確認・人工検証までを完了。新しい実H4数値計算は0。
新しい人工11件と再利用した元m17人工13件の計24件がPASS。科学的な上界評価やRPE適合のPASSではない。

1. **保存原本を固定するB-1bの入力は揃わない。** 218成分の係数/確率/basisを完全に保存した本文は関連保存先で未確認。
   H snapshotと固定SOURCEから再構成するvariantは可能だが、新しいreplay入力である。
2. **整合した入力での有限Law/Taylor恒等式は導出できる。** 個々の成分の可換性は不要、独立抽選・involution・
   A/π/θ/Bの整合が必要。回転符号・有限K・normalizationの静的対応に確定した数式変更は見つからなかった。
   保存binary64での一致やSOURCE全体の正しさを証明したものではない。
3. **B-1b/B-2はいずれも原因診断としての価値はあるが、先行必須ではない。** B-2で新しい整合入力を使って一致しても、
   元実装の誤差上界にはならない。B-1bからB-2へ自動的に進めない。
4. **直接の数値誤差上界評価を第一候補に推奨する。** 観測gapの追跡より、採用する測定平均ν_execと保存μ0の
   座標差を上から評価し、固定N/α/εの残余へ接続する方がRPEの判断に直接役立つ。
   名目πとNumPy choiceのCDF/grid、理想gateと仮想/compiled測定を分ける必要がある。
5. **まだ科学実行SOURCEは揃っていない。** 全成分/basis export、sampler→meanのbridge、包含backend/adapterが不足。
   包含の幅・normの増幅・bridge誤差が予算内になることは今回計算しておらず、成功を約束しない。
6. 次の最小科学検証案は、[VALIDATION_STRATEGY.md §4](VALIDATION_STRATEGY.md)の
   m17単一セル・component/basis SOURCE replay1回・CDF評価・最大3精度の限定包含評価。
   理想未compile/categorical scopeの結果と、元の仮想shot・compiled/実機への未保証を明示する。

**次の担当：Codex（次の技術準備を指示された後）**が、専用包含環境と薄いadapter、
人工試験、再生成入力の扱い・sampler/測定scopeを具体化し、実行SOURCEを公開して固定する。
**ユーザー**が、その固定SOURCE・入力replay・包含範囲の新科学実行を別途明示許可する。
現時点で旧B-1a SOURCEを再実行したり、未実装の上界評価を実行可能と案内したりしない。

旧INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持。
旧SOURCE/raw/NPZ/STOP、保存H/state/DF/N/α/β/ε、他研究・他worktreeを保全した。
この提案はm17局所の十分条件を強めるためのもので、全18roundのenergy推定成功・費用の最終評価ではない。

[入力可用性](INPUT_AVAILABILITY.md)・[代数/SOURCE/保存値の対応](ALGEBRAIC_CONSISTENCY.md)・
[検証戦略と最小計画](VALIDATION_STRATEGY.md)・[人工確認](VALIDATION.txt)。
専用branch：`codex/rpe-m17-b1b-b2-necessity-assessment-20261103`。
本作業の公開commitは最終チャットに記載する。commit/pushと公開照合後、mandatory STOP。
