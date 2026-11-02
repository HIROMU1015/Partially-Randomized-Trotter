# 強い対照との方法差監査

今回の追加案は、固定Pauli辞書の反可換な二項を一つのinvolutionへまとめる手順。[具体式・擬似コード](METHOD_DELTA_DESIGN.md)と、既存のoperator構成・回路recipeを比較した。科学的な性能比較は未実行。

## D1～D5

| 対照 | 実際に確認した構造 | 追加案との関係 |
|---|---|---|
| D1 Phase 2C主案 | fragment別S/P選択、全体collect、canonical RTE、run共有、typed routing案 | 元の二辞書選択規則にAのpair構成規則は含まれない。一般にはP atomから別のA単位への生成差がある。Aが既存frame atomと一致する入力なら新候補ですらなく、その場合も認める |
| D2 原PR | 残差Pauli化、異軌道involution、自己逆性を使う一般RTE | Aを正しく実装する既知部品を与えれば同じsamplerへ接続できる。原PRに本案のpairing手順が逐語的に記載されるとは主張しない |
| D3 cost-aware選択 | 同じ実装可能辞書・係数情報・cache・評価価格での選択 | 既知unitary partitioningから得る同じA候補を対照にも開放。pair poolを候補側だけに独占させない。選択順の差は独立機構ではない |
| D4 既存compiler／回路生成 | 非制御basis、中央制御、同basis相殺、support/full共有、相対Gaussian／Pauli簡約、D境界融合 | すべて共通。W†W相殺は通常のbasis共有。新案だけに実adjointやphase集約を与えない |
| D5 近接するunitary partitioning | 反可換和の正規化とsequence-of-rotationsによるcoherent対角化が明示される | 同じpairなら同じA、ρ、Wが得られる。追加案の生成中核を具体的に再現できる |

D5の出典は[Izmaylov et al., arXiv:1907.09040v2 §II.A](https://arxiv.org/html/1907.09040v2)および[Ralli et al., PRR 3, 033195 §II.A](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.033195)。VQEの費用・統計の定理をPRへ移しているのではなく、明示されたoperatorとcoherent recipeを比較している。

## 同じ情報・同じ実装条件の対応

| 軸 | 追加案が使うもの | 既知構成との対応／残る問題 |
|---|---|---|
| 情報アクセス | Pauli label、実係数、固定task／recipe | 同じbit列と係数で反可換graphと対のρ/φを得る。追加のstate oracleなし |
| 演算子 | A=(aP+bQ)/ρ | 既知unitary partitioningの二項の場合と同じ |
| 生成計画 | stable順disjoint pairing、Wと中央Pによるrecipe | 対の選び方は普通の有限走査。既知構成へ同じ対集合を渡すと同じoperator recipeを得る |
| PR law | Aをcomponentとするcanonical paired law | 一般involution-RTEの既存接続。元のP/Q pathと同一lawではなく、同じ補正finite meanを目標とする |
| 共有・fusion | ordered列の合法な隣接相殺 | 既知W†W簡約と同じ。noncommuting sortはどちらにも許さない |
| 古典計算量 | pair走査、ρ/φ、metadata、sampler更新 | 通常の反可換graph／2-clique処理と同じ。stream/cacheは対照へも許す。新しい漸近計算量差は示していない |
| 保証 | Aの既知自己逆性＋一般RTEの適用条件 | mixed production接続・finite bitは未認証。実装gapを新保証に数えない |
| PR固有性 | Aがchild/productに出現する | pairingとWはnやchild/productに依存しない。有限eventを本質的に使う追加構造なし |

## 確認した還元と、確認していないこと

確認したのは、二項のinvolution構成とそのcoherent回路骨格が、既知unitary partitioningに対応すること。`C-CONSTRUCT.Primitive`の任意basis接口もこのoperatorを表せる。一般RTEに渡す部分は既知条件の接続であり、新しい量子生成中核が残らなかった。

未確認なのは、Phase 2Cの二端点selector全体と特定論文の完全同値性、世界の全方式とのpriority、全inputでの最適性、実compiler費、資源改善率である。今回の限定案の還元を、全RQ-01/03のno-goへ拡張しない。

Mukhopadhyay–Wiebe–ZhangのAlgorithm 1は主に可換fragment lexiconへの割当で、本案の反可換対をそのまま扱う原定理だとはしていない。SPRINT/GRADEも全frameworkを万能対照と仮定しない。両者は「表現と回路費を結ぶ着眼は既知」という境界を確認する資料であり、本案の決定的な対応先は上記D5である。[各出典の版・確認範囲](SOURCE_EVIDENCE_INDEX.json)。
