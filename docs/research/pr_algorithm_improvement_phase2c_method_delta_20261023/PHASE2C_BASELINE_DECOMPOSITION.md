# Phase 2Cの既知部分と未解決部分

対象は固定commit `0c94a3d973b0ba0d24b3d277ab8316829739b278` のPhase 2C。下の分類はR1の技術監査であり、原設計の文書や科学的statusを変更しない。

分類は `KNOWN`、`KNOWN_COMBINATION`、`IMPLEMENTATION_GAP`、`POTENTIAL_METHOD_DELTA`、`UNRESOLVED`。実装が見つからないことは新方法の証拠ではない。特定の完全同一selectorを文献で見つけたか、実用費が低いか、独立した量子生成機構があるかも別の問いである。

| # | Phase 2Cの要素 | 主分類 | 確認できることと留保 | 量子生成への作用 |
|---|---|---|---|---|
| 1 | 元fragmentのframe辞書を保持 | KNOWN | 原PR §VII.B、`C-TAIL`、`P2-BUILD`。異軌道基底involutionは既知 | 元のQと制御recipeを保持 |
| 2 | 同fragmentの物理Pauli展開 | KNOWN | 原PRの残差Pauli化、`C-CONSTRUCT`の係数代数。新inputへの厳密bindingは未検証 | 既知Pauli候補へ変更 |
| 3 | fragment別の二辞書選択 | KNOWN_COMBINATION | 両辞書と異種Primitiveを組み合わせた有限選択。特定の二値実装そのものを全先行研究で確認済みとはしない | 既知候補のどれを使うかが変わる |
| 4 | P側の同一wordを全体collect | KNOWN | 係数mapの加算、abs前の相殺。新しいcutoffではない | 同一演算子の係数と出現lawが変わる |
| 5 | λ、canonical finite-RTEを更新 | KNOWN | `C-SAMPLER`の既存law。正しい更新は必須の意味論 | 辞書変更に伴う確率・角度・Bの通常更新 |
| 6 | run共有を含む構造価格 | KNOWN_COMBINATION | 既存support/full policyと有限state moment評価の設計上の組合せ。具体DPの実装・正確性は未検証 | 価格計算だけでは新しい量子操作を生成しない |
| 7 | 二端点・固定一pass反転 | KNOWN_COMBINATION | 有限の局所選択手順。順序・探索範囲・価格を変えただけでは独立機構にならない | 既知候補の選択結果が変わり得る |
| 8 | typed sampler／production router接続 | IMPLEMENTATION_GAP | toy `Primitive`は任意basis回路を持つ。DF production中央部はZ/ZZ制限。新adapterの認証は未了 | 既知作用を正確に運ぶ技術課題 |

## 未解決な点を混ぜない

`UNRESOLVED`として残るのは、特定の一体selectorが近接法と完全に一致するか、取得費を払って実用的か、actual compiler後の順位、mixed adapterのPO-01～09である。これらの未解決性だけから `POTENTIAL_METHOD_DELTA` を割り当てない。

今回の分類では、Phase 2Cの8要素の中に、独立した量子生成機構として擁護できる `POTENTIAL_METHOD_DELTA` は確認できなかった。これは価格DP・selectorの実装価値を否定するものではなく、その分類だけで新しい量子生成原理と呼ばない判断である。

## 価格、選択、生成の切り分け

同じ候補集合を全列挙できる小系でselectorの結果が違っても、独立新規性とはしない。正確な期待価格を得られても、出力が同じ既知Qと同じ合法recipeなら量子生成機構は増えていない。逆に、新しいQを出力した場合でも、そのQの構成と実装が既知なら、新しい量子実行単位と独立新規機構は区別する。

追加案ではこの最後の点を調べた。実際にPauli atomを別のinvolutionへ変える手順を[具体化](METHOD_DELTA_DESIGN.md)したが、反可換対のunitary partitioningとして既出だった。

出典：Phase 2C `ALGORITHM_DESIGN` §5～9、`METHOD_DELTA_AND_PRIOR_ART`、`DESIGN_SPEC`。原PR、`C-SAMPLER`、`C-CONSTRUCT`、`C-TAIL`、`P2-BUILD/REPEAT`、`EV-0027/0028`、cost-aware辞書選択。[固定identity・確認範囲](SOURCE_EVIDENCE_INDEX.json)。
