# PRアルゴリズム改善 Phase 2C：fragment別の残差辞書選択

日付：2026-10-23 JST。起点：`fbb83680247a4565336ca177278255e32843cfb2`。

**設計完了、実行未承認。主案は「残差fragment別のPauli／frame辞書選択」。現時点の位置付けは `KNOWN_METHOD_APPLICATION_CANDIDATE`、独立方法差は `INDEPENDENT_METHOD_DELTA_UNESTABLISHED` である。** 性能改善・独立新規性・正式採択を認定しない。

同じ残差fragmentを、既存の回転基底Z/ZZ辞書で残すか、固定された物理registerでのPauli辞書へ厳密に展開するかを選ぶ。決定論部分、fragmentの帰属、物理Hamiltonian、scalar方針、PRの既存paired finite-RTE samplerは固定する。辞書選択は実行前に一度行い、trajectory内の適応samplingや演算子の並べ替えは行わない。

候補ごとに係数をcollectし、normalizationと既存共有後の構造価格を再評価する。二つの端点から固定順にfragmentの表現を一回ずつ反転検討する、有限の選択手順まで定義した。典型的なorder-0 eventのchild rotationにも作用するが、そこが最大費用だという証拠はない。

PR原論文§VII.Bには残差Pauli化と異なる軌道基底のinvolutionが既にある。既存toy SOURCEの `Primitive` も異種辞書を許す。したがって混在辞書の導入を新原理と呼ばない。今回定義した有限RTE頻度・run共有を反映する選択手順の有用性と独立性は、近接するcost-aware辞書選択との比較で未確定である。

| 資料 | 内容 |
|---|---|
| [ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md) | 単独で読める主設計。入力、辞書構成、価格計算、選択、sampler、回路、fallback |
| [METHOD_DELTA_AND_PRIOR_ART.md](METHOD_DELTA_AND_PRIOR_ART.md) | 確認した既知法との差と、まだ主張できない方法差 |
| [SEMANTICS_AND_PROOF_OBLIGATIONS.md](SEMANTICS_AND_PROOF_OBLIGATIONS.md) | 同一H、有限mean、制御位相、精度階層、未解決条件 |
| [RESOURCE_MECHANISM.md](RESOURCE_MECHANISM.md) | 減らす操作、相殺、古典費、適用条件 |
| [MINIMAL_DISCRIMINATING_VALIDATION.md](MINIMAL_DISCRIMINATING_VALIDATION.md) | F1構成差を先に調べ、F2を条件付きとする将来案 |
| [DESIGN_DECISION_RECORD.md](DESIGN_DECISION_RECORD.md) | 採択方針、変更箇所を一つに絞った理由、留保 |
| [SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json) | 固定公開資料のidentity、実際の確認範囲、文献取得の限界 |
| [DESIGN_SPEC.json](DESIGN_SPEC.json) | 設計・承認境界の機械可読要約 |
| [handoff.md](handoff.md) | 完了範囲、次の担当と必要な承認、mandatory STOP |
| [VERIFICATION.json](VERIFICATION.json) / [PUBLICATION.json](PUBLICATION.json) | 文書管理検査と公開記録。科学的検証ではない |

実施したのは文献・固定SOURCEの静的読解、設計、文書管理検査、公開である。新しい実装、係数生成、科学的数値再解析、sampling、回路build/compile、価格取得、証明実行は0件。Phase 2Aの81資料を全件再監査した意味でもない。ローカルで指定されたPhase 2Bレビュー原本を読み、今回の指示にある採択方針と照合した。原本・非公開path・private fingerprintは公開していない。

新namespaceの追加は、既存の研究status、科学SOURCE、入力、raw、grant、ledger、marker、旧STOP、原稿の変更を意味しない。**公開・管理検証後にmandatory STOP。次段階の実装・科学実行・詳細科学レビューは開始しない。**
